use reims_vgpu_video_vaapi::{ImportWrite, SurfaceCapabilities};

use super::*;

/// A VLD config of `profile`, decoding `rt_formats`, up to `max` with surfaces from `min`.
fn config(
    profile: Profile,
    rt_formats: u32,
    max: (u32, u32),
    min: Option<(u32, u32)>,
) -> ConfigCapabilities {
    ConfigCapabilities {
        profile,
        entrypoint: Entrypoint::VLD,
        rt_formats: Some(rt_formats),
        max_width: Some(max.0),
        max_height: Some(max.1),
        dec_slice_mode: None,
        rate_control: None,
        packed_headers: None,
        max_ref_frames: None,
        max_slices: None,
        surfaces: Ok(SurfaceCapabilities {
            min_width: min.map(|m| m.0),
            min_height: min.map(|m| m.1),
            ..SurfaceCapabilities::default()
        }),
        proc_filters: None,
    }
}

fn driver(configs: Vec<ConfigCapabilities>) -> Capabilities {
    Capabilities {
        vendor: "a stand-in".to_owned(),
        api_version: (1, 22),
        configs,
        image_formats: Vec::new(),
    }
}

/// The caps are the driver's answers: each profile it lists, at the least of their sizes, the
/// codecs' top levels and VA's picture-structure bounds, any bitstream place
#[test]
fn the_caps_are_the_driver_s_answers() {
    let caps = caps_of(&driver(vec![
        config(Profile::H264_MAIN, RT_YUV420, (4096, 4096), Some((64, 64))),
        config(Profile::H264_HIGH, RT_YUV420, (4096, 2304), None),
        config(
            Profile::HEVC_MAIN,
            RT_YUV420,
            (8192, 8192),
            Some((128, 128)),
        ),
        config(
            Profile::HEVC_MAIN10,
            RT_YUV420 | RT_YUV420_10,
            (8192, 4352),
            None,
        ),
    ]));
    assert_eq!(caps.len(), 2);
    let h264 = caps[0];
    assert_eq!(
        (h264.codec, h264.direction),
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE)
    );
    assert_eq!(
        h264.profiles,
        RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH
    );
    assert_eq!((h264.min_size, h264.max_size), ([64, 64], [4096, 2304]));
    assert_eq!(
        (h264.max_level, h264.max_dpb_pictures, h264.max_references),
        (62, 17, 16)
    );
    assert_eq!(
        (h264.formats, h264.bitstream_alignment),
        (RGFX_VIDEO_FORMAT_NV12, 1)
    );
    let hevc = caps[1];
    assert_eq!(hevc.codec, RGFX_VIDEO_CODEC_HEVC);
    assert_eq!(
        hevc.profiles,
        RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10
    );
    assert_eq!(
        hevc.formats,
        RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010
    );
    assert_eq!((hevc.min_size, hevc.max_size), ([128, 128], [8192, 4352]));
    assert_eq!(
        (hevc.max_level, hevc.max_dpb_pictures, hevc.max_references),
        (186, 16, 15)
    );
}

/// A profile whose config decodes no format of its own is left out, and a codec of none goes
/// unanswered: Main 10 that decodes only 8-bit, Baseline on ConstrainedBaseline alone
#[test]
fn a_profile_whose_config_decodes_none_of_its_formats_is_left_out() {
    let caps = caps_of(&driver(vec![
        config(
            Profile::H264_CONSTRAINED_BASELINE,
            RT_YUV420,
            (1920, 1088),
            None,
        ),
        config(Profile::HEVC_MAIN10, RT_YUV420, (8192, 8192), None),
    ]));
    assert_eq!(caps.len(), 1);
    assert_eq!(caps[0].profiles, RGFX_VIDEO_H264_BASELINE);
    assert_eq!(caps[0].min_size, [MIN_SIZE; 2]);
    assert!(caps_of(&driver(Vec::new())).is_empty());
}

/// The layout is granularity [2, 2] at the largest alignment the import check found for the
/// codec's formats, and zero where it found none
#[test]
fn the_layout_is_the_import_check_s_alignment() {
    let hevc = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_HEVC,
        direction: RGFX_VIDEO_DECODE,
        formats: RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010,
        ..RgfxVideoCodecCaps::default()
    };
    let h264 = RgfxVideoCodecCaps {
        formats: RGFX_VIDEO_FORMAT_NV12,
        ..hevc
    };
    let writes = ImportWrites::of_answers(Ok(ImportWrite::at(64)), Ok(ImportWrite::at(256)));
    assert_eq!(
        (
            layout(&hevc, &writes).granularity,
            layout(&hevc, &writes).alignment
        ),
        ([2, 2], 256)
    );
    assert_eq!(layout(&h264, &writes).alignment, 64);
    let p010_only = ImportWrites::of_answers(Err("refused".to_owned()), Ok(ImportWrite::at(128)));
    assert_eq!(layout(&hevc, &p010_only).alignment, 128);
    assert_eq!(layout(&h264, &p010_only), RgfxVideoOutputLayout::default());
    // An encode's sources are read by the CPU: no layout is asked of them.
    let encode = RgfxVideoCodecCaps {
        direction: RGFX_VIDEO_ENCODE,
        ..hevc
    };
    assert_eq!(layout(&encode, &writes), RgfxVideoOutputLayout::default());
}

fn place(
    offset: u64,
    bytes_per_row: u64,
    row_bytes: u64,
    width: u64,
    height: u64,
    rows: u64,
) -> Place {
    Place {
        offset,
        bytes_per_row,
        row_bytes,
        width,
        height,
        rows_held: rows,
    }
}

/// A frame lands in place at the planes' own rows and starts, its crop's width and height each
/// rounded up to even as far as the planes hold them, `420v` or `x420` by its depth
#[test]
fn an_in_place_layout_holds_the_crop_s_even_rows_and_columns() {
    // 1080p NV12 in a surface holding the coded 1088 rows (544 chroma), rows of 2048.
    let luma = place(4096, 2048, 1920, 1920, 1080, 1088);
    let chroma = place(4096 + 2048 * 1088, 2048, 1920, 960, 540, 544);
    let layout = in_place_layout(luma, chroma, 8).unwrap();
    assert_eq!(layout.pixel_format, FourCc::new(*b"420v"));
    assert_eq!((layout.width, layout.height), (1920, 1080));
    assert_eq!(
        (layout.planes[0].offset, layout.planes[0].bytes_per_row),
        (4096, 2048)
    );
    assert_eq!(layout.planes[1].offset, 4096 + 2048 * 1088);
    assert_eq!(
        (layout.planes[1].width, layout.planes[1].height),
        (960, 540)
    );
    // 641x361 P010 in planes holding its even 642 columns and 362 rows.
    let luma = place(0, 1536, 1284, 641, 361, 362);
    let chroma = place(1536 * 362, 1536, 1284, 321, 181, 181);
    let layout = in_place_layout(luma, chroma, 10).unwrap();
    assert_eq!(layout.pixel_format, FourCc::new(*b"x420"));
    assert_eq!((layout.width, layout.height), (642, 362));
    // The same crop in planes of its odd size alone: as much as they hold.
    let luma = place(0, 1536, 1282, 641, 361, 361);
    let chroma = place(1536 * 361, 1536, 1284, 321, 181, 181);
    let layout = in_place_layout(luma, chroma, 10).unwrap();
    assert_eq!((layout.width, layout.height), (640, 361));
    assert!(in_place_layout(luma, chroma, 12).is_err());
}
