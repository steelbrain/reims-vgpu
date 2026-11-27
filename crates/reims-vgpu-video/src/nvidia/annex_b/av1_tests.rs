//! An AV1 [`AnnexBDecoder`] on the NVIDIA host: our streams' frames (`codec::av1::tests`)
//! against their references, MD5 by MD5, as ffmpeg's `framemd5` hashes NV12 and P010 planes.
//! A stream with film grain is decoded twice: without its grain, each frame against its
//! reference without grain, and as the decoder outputs it, with the grain NVDEC applies
//! (`CurrPicIdx`, `nvdec::av1`), the specification's or NVDEC's own (`judge`).

use std::sync::Arc;

use super::*;
use crate::codec::av1::frame::FilmGrainParams;
use crate::codec::av1::tests::{
    CardFrame, NO_GRAIN, ReadFrame, STREAMS, Stream, WRONG_GRAIN, card_streams, grain_refused,
    judge, samples, without_temporal_delimiters,
};
use crate::nvidia::cuda::GpuSelector;
use crate::nvidia::nvdec::{ChromaFormat, NvdecCodec, decode_capabilities};
use crate::vaapi::gated::GatedRuntime;

/// The largest and smallest frames NVDEC decodes of AV1 at 8 bits, as its capability query lists
/// them.
fn limits(gpu: &Gpu) -> Av1Limits {
    let caps = decode_capabilities(gpu, NvdecCodec::Av1, ChromaFormat::Yuv420, 8)
        .expect("NVDEC's AV1 capabilities");
    assert!(caps.supported, "NVDEC decodes no AV1 here");
    eprintln!(
        "AV1 from {}x{} up to {}x{}",
        caps.min_width, caps.min_height, caps.max_width, caps.max_height
    );
    Av1Limits {
        max_width: caps.max_width,
        max_height: caps.max_height,
        min_width: caps.min_width,
        min_height: caps.min_height,
    }
}

/// `picture` read back: each plane's rows, cropped, with no padding, as `framemd5` hashes them.
fn read_frame(picture: &AnnexBPicture, bit_depth: u8) -> Result<ReadFrame, Error> {
    let (_, _, width, height) = picture.crop();
    let read = picture.read_back((width, height))?;
    let planes = read.planes();
    let [luma, chroma] = planes.as_slice() else {
        panic!("{} planes, not two", planes.len());
    };
    let size = (
        usize::try_from(width).unwrap(),
        usize::try_from(height).unwrap(),
    );
    Ok(ReadFrame::of(luma.bytes, chroma.bytes, size, bit_depth))
}

/// Decodes `stream`, its film grain altered by `alter` where it is given: its frames read
/// back, or reported failed where NVDEC answers a picture's status Error or Error_Concealed
/// ([`Error::DecodeFailed`]), which the decoder delivers as no frame; or what else failed.
fn decode(
    gpu: &Arc<Gpu>,
    stream: &Stream,
    limits: Av1Limits,
    alter: Option<fn(&mut FilmGrainParams)>,
) -> Result<Vec<CardFrame>, Vec<String>> {
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let mut decoder = AnnexBDecoder::new(gpu, AnnexBCodec::Av1(limits), &runtime);
    if let Some(alter) = alter {
        let Codec::Av1(av1) = &mut decoder.codec else {
            unreachable!("an AV1 decoder");
        };
        av1.alter_film_grain(alter);
    }
    let mut read = Vec::new();
    let mut failed = Vec::new();
    for (n, sample) in samples(stream.ivf).into_iter().enumerate() {
        match decoder.decode(&without_temporal_delimiters(sample)) {
            Ok(Some(picture)) => {
                let (_, _, width, height) = picture.crop();
                assert_eq!((width, height), stream.size, "{}", stream.name);
                match read_frame(&picture, stream.bit_depth) {
                    Ok(frame) => read.push(Ok(frame)),
                    Err(Error::DecodeFailed(why)) => read.push(Err(why)),
                    Err(e) => failed.push(format!("{} sample {n}'s frame: {e}", stream.name)),
                }
            }
            Ok(None) => {}
            Err(Error::DecodeFailed(why)) => read.push(Err(why)),
            Err(e) => failed.push(format!("{} sample {n}: {e}", stream.name)),
        }
    }
    if failed.is_empty() {
        Ok(read)
    } else {
        Err(failed)
    }
}

/// Why the decoder refuses `stream` at its first sample, before NVDEC sees a frame, as
/// unsupported; `None` where it does not.
fn refusal(gpu: &Arc<Gpu>, stream: &Stream, limits: Av1Limits) -> Option<String> {
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let mut decoder = AnnexBDecoder::new(gpu, AnnexBCodec::Av1(limits), &runtime);
    match decoder.decode(&without_temporal_delimiters(samples(stream.ivf)[0])) {
        Err(Error::Unsupported(why)) => Some(why),
        _ => None,
    }
}

/// Each stream the decoder takes decodes to its reference, frame by frame, but for frames NVDEC
/// reports failed, which the decoder delivers as no frame: a stream with film grain to its
/// frames before grain, and with the grain NVDEC applies (`judge`). A frame delivered as decoded
/// that is not its reference fails the test, whatever the driver. A stream of a size outside
/// NVDEC's caps is refused at its sequence header.
#[test]
#[ignore = "needs an NVIDIA GPU that decodes AV1: run on the NVIDIA host"]
fn our_av1_streams_decode_to_their_reference_frames() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let limits = limits(&gpu);
    let mut wrong = Vec::new();
    for stream in card_streams() {
        let (width, height) = stream.size;
        if width < limits.min_width
            || height < limits.min_height
            || width > limits.max_width
            || height > limits.max_height
        {
            match refusal(&gpu, stream, limits) {
                Some(why) => eprintln!("{}: {width}x{height}, refused: {why}", stream.name),
                None => wrong.push(format!(
                    "{}: {width}x{height}, outside NVDEC's caps, not refused",
                    stream.name
                )),
            }
            continue;
        }
        let judged = decode(&gpu, stream, limits, None).and_then(|grained| {
            let plain = match stream.md5_without_grain {
                Some(_) => Some(decode(&gpu, stream, limits, Some(NO_GRAIN))?),
                None => None,
            };
            judge(stream, &grained, plain.as_deref())
        });
        match judged {
            Ok(line) => eprintln!("{line}"),
            Err(found) => wrong.extend(found),
        }
    }
    for why in &wrong {
        eprintln!("{why}");
    }
    assert!(wrong.is_empty(), "{} frames wrong", wrong.len());
}

/// The grain check refuses grain that is not the stream's: the grain stream decoded with its
/// grain made half as strong, put on luma alone, or made strong where it is weak, each frame of
/// it refused.
#[test]
#[ignore = "needs an NVIDIA GPU that decodes AV1: run on the NVIDIA host"]
fn grain_of_other_parameters_is_refused_by_the_grain_check() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let limits = limits(&gpu);
    let stream = STREAMS
        .iter()
        .find(|s| s.grain.is_some())
        .expect("a grain stream");
    let plain = decode(&gpu, stream, limits, Some(NO_GRAIN)).expect("the frames before grain");
    for (what, alter) in WRONG_GRAIN {
        let grained = decode(&gpu, stream, limits, Some(alter)).expect("the frames");
        let (refused, judged, fit) = grain_refused(stream, &grained, &plain);
        eprintln!("grain {what}: {refused} of {judged} frames refused, worst {fit:?}");
        assert!(judged > 0, "grain {what}: no frame judged");
        assert_eq!(refused, judged, "grain {what}");
    }
}

/// 4:4:4 (High) and 12-bit (Professional) streams are refused at their sequence header, before
/// NVDEC sees a frame, as unsupported rather than a picture's error.
#[test]
#[ignore = "needs an NVIDIA GPU that decodes AV1: run on the NVIDIA host"]
fn av1_profiles_other_than_main_are_refused() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let limits = limits(&gpu);
    for name in ["aom-444", "aom-main12"] {
        let stream = STREAMS.iter().find(|s| s.name == name).expect("the stream");
        let mut decoder = AnnexBDecoder::new(&gpu, AnnexBCodec::Av1(limits), &runtime);
        match decoder.decode(samples(stream.ivf)[0]) {
            Err(Error::Unsupported(why)) => eprintln!("{name}: {why}"),
            Err(e) => panic!("{name}: {e}"),
            Ok(picture) => panic!("{name}: a picture: {}", picture.is_some()),
        }
    }
}
