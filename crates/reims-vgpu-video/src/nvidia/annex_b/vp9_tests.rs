//! A VP9 [`AnnexBDecoder`] on the NVIDIA host: libvpx's streams
//! (`crate::codec::vp9::tests`) decoded, each frame read back and hashed as
//! libvpx's own decode of it hashes (NV12 and P010 planes, rows packed), MD5
//! by MD5.

use std::sync::Arc;

use reims_vgpu_video_conformance::md5::Md5;

use super::*;
use crate::codec::vp9::tests::{
    PROFILE_0_IVF, PROFILE_0_MD5, PROFILE_2_IVF, PROFILE_2_MD5, ivf_frames,
};
use crate::core::runtime::Runtime;
use crate::nvidia::cuda::GpuSelector;
use crate::vaapi::gated::GatedRuntime;

fn gpu() -> Arc<Gpu> {
    Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"))
}

/// Decodes `file` and hashes each frame read back.
fn frame_md5s(gpu: &Arc<Gpu>, file: &[u8]) -> Vec<String> {
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let mut decoder = AnnexBDecoder::new(gpu, AnnexBCodec::Vp9, &runtime);
    let mut hashes = Vec::new();
    for (n, sample) in ivf_frames(file).into_iter().enumerate() {
        let picture = (decoder.decode(sample))
            .unwrap_or_else(|error| panic!("sample {n}: {error}"))
            .unwrap_or_else(|| panic!("sample {n} shows no frame"));
        let (x, y, width, height) = picture.crop();
        assert_eq!((x, y, width, height), (0, 0, 640, 360), "sample {n}");
        let read = picture.read_back((width, height)).expect("a read back");
        let mut md5 = Md5::new();
        for plane in read.planes() {
            md5.update(plane.bytes);
        }
        hashes.push(md5.hex());
    }
    assert_eq!(
        decoder.bitstreams(),
        Bitstreams {
            in_place: 33,
            gathered: 0
        },
        "every frame, the superframes' hidden ones too, sent in place"
    );
    hashes
}

/// Each frame of profile 0, into NV12, and of profile 2, into P016 (P010's
/// samples for 10 bits), is libvpx's.
#[test]
#[ignore = "needs an NVIDIA GPU that decodes VP9 profiles 0 and 2: run on the NVIDIA host"]
fn libvpx_streams_decode_to_libvpx_frames() {
    let gpu = gpu();
    for (name, file, md5s) in [
        ("profile 0", PROFILE_0_IVF, PROFILE_0_MD5),
        ("profile 2", PROFILE_2_IVF, PROFILE_2_MD5),
    ] {
        let got = frame_md5s(&gpu, file);
        let want: Vec<&str> = md5s.lines().collect();
        assert_eq!(got.len(), want.len(), "{name}: frames");
        for (frame, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got, want, "{name}: frame {frame}");
        }
    }
}

/// A stream that starts after its key frame misses its references, as a
/// picture's error, and decodes from the key frame on.
#[test]
#[ignore = "needs an NVIDIA GPU that decodes VP9 profile 0: run on the NVIDIA host"]
fn a_stream_that_starts_after_its_key_frame_decodes_from_the_key_frame() {
    let gpu = gpu();
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let samples = ivf_frames(PROFILE_0_IVF);
    let mut decoder = AnnexBDecoder::new(&gpu, AnnexBCodec::Vp9, &runtime);
    let error = decoder.decode(samples[1]).err().expect("no reference");
    assert!(error.is_picture_error(), "{error}");
    assert!(decoder.decode(samples[0]).expect("the key frame").is_some());
    assert!(decoder.decode(samples[1]).expect("the next").is_some());
    assert!(decoder.parameter_sets(&[]).is_ok());
    assert!(matches!(
        decoder.parameter_sets(&[0, 0, 1]),
        Err(Error::Bitstream(_))
    ));
}
