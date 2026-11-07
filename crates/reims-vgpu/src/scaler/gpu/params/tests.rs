use reims_vgpu_video_core::scaler::format::ScalerFormat;
use reims_vgpu_video_core::scaler::plan::{FormatOf, Orientation, Plan};
use reims_vgpu_video_core::scaler::request::{
    FIXED_ONE, FixedRect, Rect, Transform, TransformFlags,
};
use reims_vgpu_video_core::surfaces::{PlaneLayout, SurfaceLayout};

use super::*;

/// A surface's layout of `format`, `width` by `height`, its rows `pitch` bytes apart and a
/// biplanar format's chroma at `chroma`.
fn layout(
    format: ScalerFormat,
    width: u32,
    height: u32,
    pitch: usize,
    chroma: usize,
) -> SurfaceLayout {
    let planes = match format.packing() {
        Packing::Biplanar420 => vec![
            PlaneLayout {
                offset: 0,
                bytes_per_row: pitch,
                width,
                height,
            },
            PlaneLayout {
                offset: chroma,
                bytes_per_row: pitch,
                width: width.div_ceil(2),
                height: height.div_ceil(2),
            },
        ],
        _ => Vec::new(),
    };
    SurfaceLayout {
        pixel_format: format.fourcc(),
        width,
        height,
        planes,
        bytes_per_row: pitch,
    }
}

/// The plan of a transform into `rect` with `flags`, from the source window of `rect`'s size at
/// the source's origin.
fn plan(source: &SurfaceLayout, destination: &SurfaceLayout, flags: u64, rect: Rect) -> Plan {
    let transform = Transform {
        flags: TransformFlags(flags),
        source_rect: FixedRect {
            x: 0,
            y: 0,
            width: u64::from(rect.width) * FIXED_ONE,
            height: u64::from(rect.height) * FIXED_ONE,
        },
        destination_rect: rect,
        ..Transform::default()
    };
    Plan::new(&transform, source, destination).unwrap()
}

fn bound(plan_of: &FormatOf) -> Vec<BoundPlane> {
    plan_of
        .planes
        .iter()
        .map(|p| BoundPlane::at(p, 0).unwrap())
        .collect()
}

/// The parameters are the shader's `ScalerParams`, word for word: 92 words, each field where
/// the shader reads it
#[test]
fn the_parameters_are_the_shaders_words() {
    let p = Params {
        source_packing: 1,
        source_order: [2, 3, 4, 5],
        luma_map: [-1, 0, 7, 0, 1, 0],
        source_subsample_y: 21,
        chroma_origin_x: 6,
        chroma_last_y: 8,
        rgb: 9,
        linear: 17,
        across_first: 10,
        plane_kind: 11,
        plane_order: [12, 13, 14, 15],
        words: 16,
        code_max: 1023.0,
        lines_first: 18,
        lines_rows: 19,
        chroma_plane_pitch: 20,
        chroma_across_first: 22,
        ycbcr_to_rgb: 23,
        across_points_first: 24,
        adaptive_channels: 25,
        adaptive_scale: 0.5,
        ..Params::default()
    };
    let bytes = p.bytes();
    assert_eq!(bytes.len(), PARAMS_WORDS * 4);
    let word = |at: usize| u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap());
    assert_eq!(word(0), 1);
    assert_eq!([word(2), word(3), word(4), word(5)], [2, 3, 4, 5]);
    assert_eq!(word(14), u32::MAX);
    assert_eq!(word(16), 7);
    assert_eq!(word(23), 21);
    assert_eq!((word(26), word(29)), (6, 8));
    assert_eq!((word(30), word(31)), (9, 17));
    assert_eq!(word(40), 10);
    assert_eq!(word(52), 11);
    assert_eq!([word(53), word(54), word(55), word(56)], [12, 13, 14, 15]);
    assert_eq!(word(63), 16);
    assert_eq!(f32::from_bits(word(68)), 1023.0);
    assert_eq!((word(69), word(72), word(74)), (18, 19, 20));
    assert_eq!((word(75), word(79)), (22, 23));
    assert_eq!((word(80), word(90)), (24, 25));
    assert_eq!(f32::from_bits(word(91)), 0.5);
}

/// A format's codes read and are written as the reference's: 420v's luma from 16, 219 codes to
/// 1, and its chroma about 128.125, 224 codes of 255 to 1 in 256; x420's luma into RGB from 64,
/// 876 codes to 1
#[test]
fn codes_read_and_are_written_as_the_references() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let v = ScalerFormat::Nv12VideoRange;
    let ((low, scale), (middle, chroma_scale)) = decoding(v, Space::YCbCr);
    assert!(
        close(low, 16.0) && close(scale, 1.0 / 219.0),
        "{low} {scale}"
    );
    assert!(
        close(middle, 128.125) && close(chroma_scale, 255.0 / (224.0 * 256.0)),
        "{middle} {chroma_scale}"
    );
    let ((low, span), (middle, chroma_span)) = encoding(v);
    assert!(close(low, 16.0) && close(span, 219.0), "{low} {span}");
    assert!(close(middle, 128.125) && close(chroma_span, 224.0 * 256.0 / 255.0));
    let ((low, scale), _) = decoding(ScalerFormat::P010VideoRange, Space::Rgb);
    assert!(
        close(low, 64.0) && close(scale, 1.0 / 876.0),
        "{low} {scale}"
    );
}

/// Each orientation's map takes the turned picture's pixels to the source's as the reference
/// turns it: `source_point` at each pixel's centre
#[test]
fn a_turn_maps_each_pixel_as_the_reference_turns_it() {
    let (w, h) = (5u32, 3u32);
    for bits in 0..8 {
        let o = Orientation::from_bits(bits);
        let m = turn_map(o, w, h).unwrap();
        let (tw, th) = if o.rotates() { (h, w) } else { (w, h) };
        for y in 0..th {
            for x in 0..tw {
                let (sx, sy) = o.source_point(
                    f64::from(x) + 0.5,
                    f64::from(y) + 0.5,
                    f64::from(w),
                    f64::from(h),
                );
                let (x, y) = (i64::from(x), i64::from(y));
                let mapped = (
                    i64::from(m[0]) * x + i64::from(m[1]) * y + i64::from(m[2]),
                    i64::from(m[3]) * x + i64::from(m[4]) * y + i64::from(m[5]),
                );
                #[expect(clippy::as_conversions, clippy::cast_possible_truncation)]
                let expected = (sx.floor() as i64, sy.floor() as i64);
                assert_eq!(mapped, expected, "orientation {bits} at ({x}, {y})");
            }
        }
    }
}

/// An axis's taps are merged by pixel, without zero weights, divided by their sum, and padded
/// with zero weights to the most a pixel has
#[test]
fn an_axis_s_taps_are_merged_normalized_and_padded() {
    let (count, table) =
        axis(&[vec![(0, 1.0), (0, 1.0), (1, 2.0)], vec![(3, 0.0), (4, 4.0)]]).unwrap();
    assert_eq!(count, 2);
    assert_eq!(table, [(0, 0.5), (4, 1.0), (1, 0.5), (4, 0.0)]);
}

/// A 4:2:0 window is one dispatch for both planes, over the chroma rows and the words of its
/// coverage, the luma words of the same pixels with them. The plan starts it on an even pixel and
/// spans even pixels, so (3, 5) 10x7 is (2, 4) 10x6, its first word in part
#[test]
fn a_biplanar_window_is_a_dispatch_over_its_coverage() {
    let n = ScalerFormat::Nv12VideoRange;
    let source = layout(n, 64, 32, 64, 2048);
    let destination = layout(n, 64, 32, 64, 2048);
    let rect = Rect {
        x: 3,
        y: 5,
        width: 10,
        height: 7,
    };
    let plan = plan(&source, &destination, 0, rect);
    let work = work(
        &plan,
        &bound(&plan.source_format),
        &bound(&plan.destination_format),
    )
    .unwrap();
    let [(Kernel::TransformBothPlanes, both, threads, _)] = work.passes.as_slice() else {
        panic!("one pass: {:?}", work.passes.len());
    };
    assert_eq!(both.plane_kind, PlaneKind::Biplanar8 as u32);
    assert_eq!((both.first_row, both.rows), (2, 3), "chroma rows 2 to 4");
    assert_eq!((both.first_byte, both.end_byte), (2, 12), "pixels 2 to 11");
    assert_eq!(both.words, 3, "bytes 0 to 11");
    assert_eq!(*threads, (3, 3));
    assert_eq!((both.plane_offset, both.chroma_plane_offset), (0, 2048));
    assert_eq!((both.plane_pitch, both.chroma_plane_pitch), (64, 64));
    assert_eq!(work.lines, 0, "a copy takes a pixel a pixel");
    // The taps of a copy: one a pixel, of weight 1; then each pixel's sample point, on a
    // pixel, which the adaptive filter, with no pass to move, does not read.
    assert_eq!(both.across_count, 1);
    assert_eq!(both.down_first, 10);
    assert!(work.taps[..16].iter().all(|&(_, w)| w == 1.0));
    assert_eq!((both.across_points_first, both.down_points_first), (16, 26));
    assert!(work.taps[16..].iter().all(|&(_, f)| f == 0.0));
    assert_eq!((both.adapt_across, both.adapt_down), (0, 0));
}

/// A transform the adaptive filter moves and that is not separable, here a halving across alone,
/// runs `scaler_transform_adaptive`, and one it does not move `scaler_transform`
#[test]
fn a_moved_transform_runs_the_adaptive_kernel() {
    let bgra = ScalerFormat::Bgra;
    let (source, half, whole) = (
        layout(bgra, 64, 32, 256, 0),
        layout(bgra, 32, 32, 128, 0),
        layout(bgra, 64, 32, 256, 0),
    );
    let kernels = |destination: &SurfaceLayout| {
        let transform = Transform {
            flags: TransformFlags(0x1000),
            ..Transform::default()
        };
        let plan = Plan::new(&transform, &source, destination).unwrap();
        let work = work(
            &plan,
            &bound(&plan.source_format),
            &bound(&plan.destination_format),
        )
        .unwrap();
        work.passes
            .iter()
            .map(|(kernel, pass, _, _)| (*kernel, pass.adapt_across, pass.adapt_down))
            .collect::<Vec<_>>()
    };
    assert_eq!(kernels(&half), [(Kernel::TransformAdaptive, 1, 0)]);
    assert_eq!(kernels(&whole), [(Kernel::Transform, 0, 0)]);
}

/// A copy of a 4:2:0 source into BGRA is a dispatch of whole threadgroups of `scaler_quads`, a
/// thread two by two of the window's pixels
#[test]
fn a_copy_into_rgb_is_whole_threadgroups_of_quads() {
    let source = layout(ScalerFormat::Nv12VideoRange, 64, 32, 64, 2048);
    let destination = layout(ScalerFormat::Bgra, 64, 32, 256, 0);
    let rect = Rect {
        x: 3,
        y: 5,
        width: 36,
        height: 10,
    };
    let plan = plan(&source, &destination, 0, rect);
    let work = work(
        &plan,
        &bound(&plan.source_format),
        &bound(&plan.destination_format),
    )
    .unwrap();
    let [(Kernel::Quads, rgb, threads, group)] = work.passes.as_slice() else {
        panic!("one pass of quads: {:?}", work.passes);
    };
    assert_eq!((rgb.first_row, rgb.rows), (5, 10));
    assert_eq!((rgb.first_byte, rgb.end_byte), (12, 156));
    assert_eq!(rgb.words, 36);
    // 18 quads across and 5 down, in threadgroups of 16 by 4.
    assert_eq!((*threads, *group), ((32, 8), GROUP));
    assert_eq!(work.lines, 0);
}

/// A 4:2:0 window of an odd edge, one under two pixels, which the plan keeps as it is, is a
/// dispatch a plane: its luma rows and bytes, and the chroma rows and pairs it shares
#[test]
fn an_odd_biplanar_window_is_a_dispatch_a_plane() {
    let n = ScalerFormat::Nv12VideoRange;
    let source = layout(n, 64, 32, 64, 2048);
    let destination = layout(n, 64, 32, 64, 2048);
    let rect = Rect {
        x: 3,
        y: 5,
        width: 1,
        height: 1,
    };
    let plan = plan(&source, &destination, 0, rect);
    let work = work(
        &plan,
        &bound(&plan.source_format),
        &bound(&plan.destination_format),
    )
    .unwrap();
    let [
        (Kernel::Transform, luma, luma_threads, _),
        (Kernel::Transform, chroma, chroma_threads, _),
    ] = work.passes.as_slice()
    else {
        panic!("two passes: {:?}", work.passes.len());
    };
    assert_eq!(luma.plane_kind, PlaneKind::Luma8 as u32);
    assert_eq!((luma.first_row, luma.rows), (5, 1));
    assert_eq!((luma.first_byte, luma.end_byte), (3, 4));
    assert_eq!(*luma_threads, (1, 1));
    assert_eq!(chroma.plane_kind, PlaneKind::Chroma8 as u32);
    assert_eq!((chroma.first_row, chroma.rows), (2, 1));
    assert_eq!((chroma.first_byte, chroma.end_byte), (2, 4), "pair 1");
    assert_eq!(*chroma_threads, (1, 1));
    assert_eq!(chroma.plane_offset, 2048);
}

/// An offset past 4 GiB, which the kernel's 32-bit offsets do not reach, is too large
#[test]
fn a_plane_past_the_kernels_offsets_is_too_large() {
    let b = ScalerFormat::Bgra;
    let source = layout(b, 64, 32, 256, 0);
    let plan = plan(&source, &source, 0, Rect::default());
    let far = |planes: Vec<BoundPlane>| {
        planes
            .into_iter()
            .map(|p| BoundPlane {
                offset: p.offset + (1 << 32),
                ..p
            })
            .collect::<Vec<_>>()
    };
    let near = bound(&plan.source_format);
    assert!(work(&plan, &near, &near).is_ok());
    assert_eq!(work(&plan, &far(near.clone()), &near), Err(TooLarge));
    assert_eq!(work(&plan, &near, &far(near.clone())), Err(TooLarge));
}

/// A separable transform's bands cover the window's rows once each, in order, each of the rows
/// the lines buffer holds at most, a band past the first from a multiple of the chroma's rows;
/// where it holds fewer, a pair of rows a band
#[test]
fn a_tall_windows_bands_cover_its_rows() {
    for (rows, step) in [(0u32, 2u32), (1, 2), (7, 2), (10, 1), (1000, 2)] {
        let bands = bands((3, 42), rows, step);
        let mut next = 3;
        for &(b0, b1) in &bands {
            assert_eq!(b0, next, "{rows} {step}");
            assert!(b1 > b0);
            assert!(b0 == 3 || b0 % step == 0, "{rows} {step}: {b0}");
            assert!(b1 - b0 <= rows.max(step), "{rows} {step}: {b0}..{b1}");
            next = b1;
        }
        assert_eq!(next, 42, "{rows} {step}");
    }
    assert_eq!(bands((3, 42), 1, 2)[..2], [(3, 4), (4, 6)]);
    assert_eq!(bands((3, 42), 7, 2)[..2], [(3, 8), (8, 14)]);
    assert_eq!(bands((3, 42), 1000, 2), [(3, 42)]);
}

/// A 4:2:0 reduction is separable: its lines' dispatch first, over the columns of the source the
/// window's taps take and the window's rows, then its planes', which read the lines
#[test]
fn a_reduction_runs_its_lines_then_its_planes() {
    let n = ScalerFormat::Nv12VideoRange;
    let source = layout(n, 64, 32, 64, 2048);
    let destination = layout(n, 32, 16, 64, 2048);
    let transform = Transform {
        flags: TransformFlags(0x1000),
        ..Transform::default()
    };
    let plan = Plan::new(&transform, &source, &destination).unwrap();
    let work = work(
        &plan,
        &bound(&plan.source_format),
        &bound(&plan.destination_format),
    )
    .unwrap();
    let [
        (Kernel::Lines, lines, lines_threads, lines_group),
        (Kernel::Shared, both, _, group),
    ] = work.passes.as_slice()
    else {
        panic!("two passes: {:?}", work.passes.len());
    };
    assert_eq!((lines.lines_first, lines.lines_pitch), (0, 64));
    assert_eq!(
        (lines.lines_row, lines.lines_rows),
        (0, 16),
        "every row of the window"
    );
    assert_eq!(*lines_threads, (64, 16));
    assert_eq!(
        work.lines,
        64 * 16 * 12,
        "three floats a line: a Y'CbCr source is opaque"
    );
    assert_eq!(
        (
            both.lines_first,
            both.lines_pitch,
            both.lines_row,
            both.lines_rows
        ),
        (0, 64, 0, 16)
    );
    // The 32 rows of the source 16 rows of the window take fit what a threadgroup of the lines
    // decodes once.
    assert_eq!((*lines_group, *group), ((16, 16), (16, 4)));
    assert_eq!(both.plane_kind, PlaneKind::Biplanar8 as u32);
    assert_eq!((both.first_row, both.rows), (0, 8), "every chroma row");
}

/// Every kernel and threadgroup a transform dispatches is one the making ahead dispatches
/// ([`DISPATCHED`]), and each of those some transform dispatches: over every pair of formats,
/// copies, turns, reductions and enlargements of each axis, by 2 and by 4
#[test]
fn every_dispatch_is_one_the_making_ahead_makes() {
    let mut taken = Vec::new();
    for from in ScalerFormat::ALL {
        for to in ScalerFormat::ALL {
            let source = layout(from, 96, 64, 512, 16384);
            for ((width, height), flags) in [
                ((96, 64), 0x1000),
                ((64, 96), 0x1004),
                ((48, 32), 0x1000),
                ((24, 16), 0x1000),
                ((96, 32), 0x1000),
                ((48, 64), 0x1000),
                ((192, 128), 0x1000),
                ((96, 128), 0x1000),
                ((192, 64), 0x1000),
                ((32, 48), 0x1004),
            ] {
                let destination = layout(to, width, height, 1024, 32768);
                let transform = Transform {
                    flags: TransformFlags(flags),
                    ..Transform::default()
                };
                let Ok(plan) = Plan::new(&transform, &source, &destination) else {
                    continue;
                };
                let work = work(
                    &plan,
                    &bound(&plan.source_format),
                    &bound(&plan.destination_format),
                )
                .unwrap();
                for (kernel, _, _, group) in work.passes {
                    assert!(
                        DISPATCHED.contains(&(kernel, group)),
                        "{kernel:?} at {group:?} is not made ahead"
                    );
                    taken.push((kernel, group));
                }
            }
        }
    }
    for dispatched in DISPATCHED {
        assert!(
            taken.contains(&dispatched),
            "{dispatched:?} is made for none"
        );
    }
}
