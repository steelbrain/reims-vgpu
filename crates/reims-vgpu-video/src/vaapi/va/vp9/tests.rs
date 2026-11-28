//! The VP9 decoder's parameters, and its decode of libvpx's streams on the
//! GPU: on the display `super::super::teardown`'s `display` opens, the render
//! node `RVG_VAAPI_NODE` names. Each frame read back must hash as libvpx's
//! own decode of it does (`crate::codec::vp9::tests`'s MD5s, over the two
//! planes of the guest's format of the stream: NV12, P010 and their 4:2:2,
//! 4:4:4 and 12-bit kin).

use reims_vgpu_video_conformance::md5::Md5;

use super::super::frame::PlaneRows;
use super::super::teardown::display;
use super::*;
use crate::codec::vp9::header::{self, State};
use crate::codec::vp9::quant;
use crate::codec::vp9::tests::{
    FORMAT_STREAMS, FrameSpec, PROFILE_0_IVF, PROFILE_0_MD5, PROFILE_2_IVF, PROFILE_2_MD5,
    SegmentationSpec, ivf_frames,
};

#[test]
fn each_profile_decodes_into_the_surfaces_of_its_chroma_format_and_depth() -> Result<(), Error> {
    use Chroma::{Yuv420, Yuv422, Yuv444};
    use SurfaceFormat::{Nv12, P010, P012, Xyuv, Y210, Y212, Y410, Y412, Yuy2};
    let format = |profile, chroma, bit_depth| Format {
        profile,
        bit_depth,
        chroma,
        width: 64,
        height: 64,
    };
    for (profile, chroma, depth, va, surfaces) in [
        (0, Yuv420, 8, Profile::VP9_PROFILE0, Nv12),
        (2, Yuv420, 10, Profile::VP9_PROFILE2, P010),
        (2, Yuv420, 12, Profile::VP9_PROFILE2, P012),
        (1, Yuv422, 8, Profile::VP9_PROFILE1, Yuy2),
        (1, Yuv444, 8, Profile::VP9_PROFILE1, Xyuv),
        (3, Yuv422, 10, Profile::VP9_PROFILE3, Y210),
        (3, Yuv422, 12, Profile::VP9_PROFILE3, Y212),
        (3, Yuv444, 10, Profile::VP9_PROFILE3, Y410),
        (3, Yuv444, 12, Profile::VP9_PROFILE3, Y412),
    ] {
        assert_eq!(
            profile_for(&format(profile, chroma, depth))?,
            (va, surfaces)
        );
        assert_eq!(
            vp9_decoder(profile, chroma, depth),
            Some((va, surfaces.rt_format(), Fourcc(surfaces.fourcc())))
        );
    }
    for (profile, depth) in [(4, 8), (0, 9)] {
        assert!(matches!(
            profile_for(&format(profile, Yuv420, depth)),
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

/// Decodes `file` and hashes each frame read back into the two planes of
/// the guest's format of it: luma, then Cb and Cr interleaved at the
/// chroma's resolution, a sample past 8 bits a 16-bit word.
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
            let (pairs, chroma_rows) = match frame.chroma() {
                Chroma::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
                Chroma::Yuv422 => (width.div_ceil(2), height),
                Chroma::Yuv444 => (width, height),
            };
            let (y_row, uv_row) = (width * bytes, pairs * 2 * bytes);
            let mut y = vec![0u8; y_row * height];
            let mut uv = vec![0u8; uv_row * chroma_rows];
            let planes = |bytes, row, rows| PlaneRows {
                bytes,
                bytes_per_row: row,
                rows,
                row_bytes: row,
            };
            let (y_plane, uv_plane) = (
                planes(&mut y, y_row, height),
                planes(&mut uv, uv_row, chroma_rows),
            );
            let back = frame.read_back().expect("a read back");
            if frame.is_packed() {
                back.copy_packed_into(y_plane, uv_plane)
            } else {
                back.copy_into(y_plane, uv_plane)
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

/// Each frame's hash of `file` against `md5s`, one a line.
fn assert_frames(display: &Display, name: &str, file: &[u8], md5s: &str) {
    let got = frame_md5s(display, file);
    let want: Vec<&str> = md5s.lines().collect();
    assert_eq!(got.len(), want.len(), "{name}: frames");
    for (frame, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(got, want, "{name}: frame {frame}");
    }
}

#[test]
#[ignore = "needs a VA-API driver that decodes VP9 profiles 0 and 2: run on a VA-API host"]
fn libvpx_streams_decode_to_libvpx_frames() {
    let display = display();
    for (name, file, md5s) in [
        ("profile 0", PROFILE_0_IVF, PROFILE_0_MD5),
        ("profile 2", PROFILE_2_IVF, PROFILE_2_MD5),
    ] {
        assert_frames(&display, name, file, md5s);
    }
}

/// The streams of profile 2 at 12 bits and of profiles 1 and 3, each where the
/// driver's decoder of its profile makes its render targets and lists its
/// surfaces (`Vp9Decoder::supports`), each stream's outcome printed: on the
/// Iris Xe's iHD the 12-bit 4:2:0 and every 4:4:4 one (4:2:2's surfaces are
/// not listed, `super` module doc); on radeonsi none, which lists profiles 0
/// and 2 at 10 bits alone.
#[test]
#[ignore = "needs a VA-API driver: run on a VA-API host"]
fn libvpx_streams_of_every_format_the_driver_lists_decode_to_libvpx_frames() {
    let display = display();
    let (mut decoded, mut failed) = (Vec::new(), Vec::new());
    for stream in &FORMAT_STREAMS {
        let size = (352, 288);
        if let Err(why) =
            Vp9Decoder::supports(&display, stream.profile, stream.depth, stream.chroma, size)
        {
            eprintln!("{}: not listed here: {why}", stream.name);
            continue;
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_frames(&display, stream.name, stream.ivf, stream.md5);
        }));
        match outcome {
            Ok(()) => decoded.push(stream.name),
            Err(_) => failed.push(stream.name),
        }
    }
    eprintln!("decoded to libvpx's frames: {decoded:?}");
    assert!(
        failed.is_empty(),
        "not decoded to libvpx's frames: {failed:?}"
    );
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
