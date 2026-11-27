//! The VP9 decoder's parameters, and its decode of libvpx's streams on the
//! GPU: on the display `super::super::teardown`'s `display` opens, the render
//! node `RVG_VAAPI_NODE` names. Each frame read back must hash as libvpx's
//! own decode of it does (`crate::codec::vp9::tests`'s MD5s, over NV12 and
//! P010 planes).

use reims_vgpu_video_conformance::md5::Md5;

use super::super::teardown::display;
use super::*;
use crate::codec::vp9::header::{self, State};
use crate::codec::vp9::quant;
use crate::codec::vp9::tests::{
    FrameSpec, PROFILE_0_IVF, PROFILE_0_MD5, PROFILE_2_IVF, PROFILE_2_MD5, SegmentationSpec,
    ivf_frames,
};

#[test]
fn profile_0_decodes_into_nv12_and_profile_2_into_p010() -> Result<(), Error> {
    let format = |profile, bit_depth| Format {
        profile,
        bit_depth,
        width: 64,
        height: 64,
    };
    assert_eq!(
        profile_for(&format(0, 8))?,
        (Profile::VP9_PROFILE0, SurfaceFormat::Nv12)
    );
    assert_eq!(
        profile_for(&format(2, 10))?,
        (Profile::VP9_PROFILE2, SurfaceFormat::P010)
    );
    for (profile, depth) in [(1, 8), (2, 12), (3, 10)] {
        assert!(matches!(
            profile_for(&format(profile, depth)),
            Err(Error::Unsupported(_))
        ));
    }
    Ok(())
}

#[test]
fn a_segment_carries_its_reference_and_skip_features_where_segmentation_is_on() {
    let mut features = [[None; 4]; 8];
    features[5] = [None, None, Some(2), Some(0)];
    let mut spec = FrameSpec::key(64, 64);
    spec.segmentation = Some(SegmentationSpec {
        update_map: None,
        update_data: Some((true, features)),
    });
    let frame = spec.write();
    let (header, _) = header::parse(&frame, &State::default(), &[None; 8]).unwrap();
    let derived = quant::segment_params(&header);
    let five = segment_parameters(&header, 5, &derived[5]);
    // Enabled (bit 0), the golden frame (2, bits 1 and 2), skipped (bit 3).
    assert_eq!(five.segment_flags, 0b1101);
    assert_eq!(five.luma_ac_quant_scale, derived[5].luma_ac);
    assert_eq!(five.filter_level, derived[5].filter_level);
    assert_eq!(segment_parameters(&header, 0, &derived[0]).segment_flags, 0);
    // With segmentation off, no segment has a feature.
    spec.segmentation = None;
    let frame = spec.write();
    let (off, _) = header::parse(&frame, &State::default(), &[None; 8]).unwrap();
    assert_eq!(segment_parameters(&off, 5, &derived[5]).segment_flags, 0);
}

/// Decodes `file` and hashes each frame read back, NV12 or P010 by its
/// depth.
fn frame_md5s(display: &Display, file: &[u8]) -> Vec<String> {
    let mut decoder = Vp9Decoder::new(display);
    let mut hashes = Vec::new();
    for (id, sample) in (0u64..).zip(ivf_frames(file)) {
        let frames = decoder
            .decode(sample, id)
            .unwrap_or_else(|error| panic!("sample {id}: {error}"));
        assert_eq!(frames.len(), 1, "sample {id}'s frames");
        for frame in frames {
            let (width, height) = (frame.width() as usize, frame.height() as usize);
            let bytes = if frame.bit_depth() == 8 { 1 } else { 2 };
            let mut y = vec![0u8; width * height * bytes];
            let mut uv = vec![0u8; width * height.div_ceil(2) * bytes];
            let back = frame.read_back().expect("a read back");
            if bytes == 1 {
                back.copy_nv12_to(&mut y, width, &mut uv, width)
            } else {
                back.copy_p010_to(&mut y, width * 2, &mut uv, width * 2)
            }
            .expect("a copy");
            let mut md5 = Md5::new();
            md5.update(&y);
            md5.update(&uv);
            hashes.push(md5.hex());
        }
    }
    hashes
}

#[test]
#[ignore = "needs a VA-API driver that decodes VP9 profiles 0 and 2: run on a VA-API host"]
fn libvpx_streams_decode_to_libvpx_frames() {
    let display = display();
    for (name, file, md5s) in [
        ("profile 0", PROFILE_0_IVF, PROFILE_0_MD5),
        ("profile 2", PROFILE_2_IVF, PROFILE_2_MD5),
    ] {
        let got = frame_md5s(&display, file);
        let want: Vec<&str> = md5s.lines().collect();
        assert_eq!(got.len(), want.len(), "{name}: frames");
        for (frame, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got, want, "{name}: frame {frame}");
        }
    }
}

#[test]
#[ignore = "needs a VA-API driver that decodes VP9 profile 0: run on a VA-API host"]
fn a_stream_that_starts_after_its_key_frame_decodes_from_the_next_one() {
    let display = display();
    let samples = ivf_frames(PROFILE_0_IVF);
    let mut decoder = Vp9Decoder::new(&display);
    let error = decoder
        .decode(samples[1], 1)
        .err()
        .expect("an inter frame with no reference");
    assert!(error.is_picture_error(), "{error}");
    assert_eq!(
        decoder.decode(samples[0], 0).expect("the key frame").len(),
        1
    );
    assert_eq!(decoder.profile(), Some(Profile::VP9_PROFILE0));
}
