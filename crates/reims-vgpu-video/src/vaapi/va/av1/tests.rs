//! The AV1 decoder on the GPU: our streams' frames (`codec::av1::tests`) against their
//! references, MD5 by MD5, as ffmpeg's `framemd5` hashes NV12 and P010 planes. A frame that
//! applies film grain is compared with the reference that has the grain applied, the
//! specification's output, which is what the decoder outputs: the driver applies the grain
//! (`current_display_picture`).

use std::sync::Arc;

use super::*;
use crate::codec::av1::frame::FilmGrainParams;
use crate::codec::av1::tests::{
    CardFrame, NO_GRAIN, ReadFrame, STREAMS, Stream, WRONG_GRAIN, card_streams, grain_refused,
    judge, samples, without_temporal_delimiters,
};
use crate::vaapi::va::gated::GatedRuntime;

fn display() -> Display {
    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    Display::open(node, Arc::new(GatedRuntime)).expect("a VA display")
}

/// `frame` read back: each plane's rows, cropped, with no padding, as `framemd5` hashes them.
pub(crate) fn read_frame(frame: &Frame, bit_depth: u8) -> ReadFrame {
    let sample = if bit_depth > 8 { 2 } else { 1 };
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let (y_row, uv_row, uv_rows) = (w * sample, w.div_ceil(2) * 2 * sample, h.div_ceil(2));
    let mut y = vec![0u8; y_row * h];
    let mut uv = vec![0u8; uv_row * uv_rows];
    let copy = if sample == 2 {
        Frame::copy_p010_to
    } else {
        Frame::copy_nv12_to
    };
    copy(frame, &mut y, y_row, &mut uv, uv_row).expect("a read back");
    ReadFrame::of(&y, &uv, (w, h), bit_depth)
}

/// Decodes `stream`, its film grain altered by `alter` where it is given: its frames read
/// back, or what failed. A sample that fails fails the stream: the driver reports no picture
/// failed apart from the call that decodes it.
fn decode(
    display: &Display,
    stream: &Stream,
    limits: Limits,
    alter: Option<fn(&mut FilmGrainParams)>,
) -> Result<Vec<CardFrame>, Vec<String>> {
    let mut decoder = Av1Decoder::without_record(display, limits);
    if let Some(alter) = alter {
        decoder.alter_film_grain(alter);
    }
    let mut read = Vec::new();
    let mut failed = Vec::new();
    for (n, sample) in samples(stream.ivf).into_iter().enumerate() {
        match decoder.decode(&without_temporal_delimiters(sample), n as u64) {
            Ok(frames) => {
                for frame in frames {
                    assert_eq!(
                        (frame.width(), frame.height()),
                        stream.size,
                        "{}",
                        stream.name
                    );
                    read.push(Ok(read_frame(&frame, stream.bit_depth)));
                }
            }
            Err(e) => failed.push(format!("{} sample {n}: {e}", stream.name)),
        }
    }
    if failed.is_empty() {
        Ok(read)
    } else {
        Err(failed)
    }
}

/// Each stream the decoder takes decodes to its reference, frame by frame: a stream with film
/// grain to its frames before grain, and with the grain the driver applies, the
/// specification's or the driver's own (`judge`).
#[test]
#[ignore = "needs a VA-API GPU that decodes AV1: run on the VA-API host"]
fn our_streams_decode_to_their_reference_frames() {
    let display = display();
    let limits = limits(&display).expect("the driver lists AV1Profile0 decode");
    eprintln!(
        "AV1Profile0 up to {}x{}",
        limits.max_width, limits.max_height
    );
    // What the driver lists of its decoders' sizes, AV1's and HEVC Main's beside it: each
    // decoder's largest picture, and its surfaces' smallest and largest.
    let capabilities = display.capabilities().expect("the driver's capabilities");
    for (name, profile) in [
        ("AV1Profile0", AV1_PROFILE0),
        ("HEVCMain", Profile::HEVC_MAIN),
        ("HEVCMain10", Profile::HEVC_MAIN10),
    ] {
        let Some(config) = capabilities.find(profile, Entrypoint::VLD) else {
            eprintln!("{name}: no decoder");
            continue;
        };
        let surfaces = config.surfaces.as_ref().ok();
        let size = |w: Option<u32>, h: Option<u32>| format!("{w:?}x{h:?}");
        eprintln!(
            "{name}: pictures up to {}, surfaces from {} up to {}",
            size(config.max_width, config.max_height),
            size(
                surfaces.and_then(|s| s.min_width),
                surfaces.and_then(|s| s.min_height)
            ),
            size(
                surfaces.and_then(|s| s.max_width),
                surfaces.and_then(|s| s.max_height)
            ),
        );
    }
    let mut wrong = Vec::new();
    for stream in card_streams() {
        let judged = decode(&display, stream, limits, None).and_then(|grained| {
            let plain = match stream.md5_without_grain {
                Some(_) => Some(decode(&display, stream, limits, Some(NO_GRAIN))?),
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
#[ignore = "needs a VA-API GPU that decodes AV1: run on the VA-API host"]
fn grain_of_other_parameters_is_refused_by_the_grain_check() {
    let display = display();
    let limits = limits(&display).expect("the driver lists AV1Profile0 decode");
    let stream = STREAMS
        .iter()
        .find(|s| s.grain.is_some())
        .expect("a grain stream");
    let plain = decode(&display, stream, limits, Some(NO_GRAIN)).expect("the frames before grain");
    for (what, alter) in WRONG_GRAIN {
        let grained = decode(&display, stream, limits, Some(alter)).expect("the frames");
        let (refused, judged, fit) = grain_refused(stream, &grained, &plain);
        eprintln!("grain {what}: {refused} of {judged} frames refused, worst {fit:?}");
        assert!(judged > 0, "grain {what}: no frame judged");
        assert_eq!(refused, judged, "grain {what}");
    }
}

/// 4:4:4 (High) and 12-bit (Professional) streams are refused at their sequence header,
/// before the driver sees a frame.
#[test]
#[ignore = "needs a VA-API GPU that decodes AV1: run on the VA-API host"]
fn profiles_other_than_main_are_refused() {
    let display = display();
    let limits = limits(&display).expect("the driver lists AV1Profile0 decode");
    for name in ["aom-444", "aom-main12"] {
        let stream = STREAMS.iter().find(|s| s.name == name).expect("the stream");
        let mut decoder = Av1Decoder::without_record(&display, limits);
        let first = samples(stream.ivf)[0];
        match decoder.decode(first, 0) {
            Err(Error::Av1Bitstream(e)) => assert!(!e.is_picture_error(), "{name}: {e}"),
            other => panic!("{name}: {:?}", other.map(|f| f.len())),
        }
        assert_eq!(decoder.frames_decoded(), 0);
    }
}
