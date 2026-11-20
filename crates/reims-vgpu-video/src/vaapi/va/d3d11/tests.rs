//! The plane draws on the default adapter, against `vaGetImage`'s read back of each picture,
//! and the zeros their targets owe.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::core::runtime::{Runtime as _, StdRuntime};

use super::super::d3d12::{Device, SharedTexture};
use super::super::display::Display;
use super::super::frame::{Frame, PlaneRows};
use super::super::h264::H264Decoder;
use super::super::hevc::HevcDecoder;
use super::super::readback::{ReadbackWay, read_back_by};
use super::super::surface::SurfaceFormat;
use super::{PlaneDraw, PlaneTargets, covers};

/// The streams of the directories `RVG_VA_STREAMS` lists (as `PATH` lists them): each H.264
/// (`.264`) and HEVC (`.265`) file, in name order.
fn streams() -> Vec<PathBuf> {
    let dirs =
        std::env::var_os("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let mut streams: Vec<PathBuf> = std::env::split_paths(&dirs)
        .flat_map(|dir| std::fs::read_dir(dir).expect("a streams directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".264") || name.ends_with(".265")
        })
        .collect();
    streams.sort();
    assert!(!streams.is_empty(), "no streams in {dirs:?}");
    streams
}

/// `frame`'s crop as `vaGetImage` reads it back: the luma's rows and the chroma's, each row's
/// bytes with nothing between them.
fn got_image(frame: &Frame) -> [Vec<u8>; 2] {
    let sample = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let rows = [
        (width * sample, height),
        (width.div_ceil(2) * 2 * sample, height.div_ceil(2)),
    ];
    let mut planes = rows.map(|(row, rows)| vec![0u8; row * rows]);
    let [luma, chroma] = &mut planes;
    let back = frame.read_back().expect("a read back");
    back.copy_into(plane(luma, rows[0]), plane(chroma, rows[1]))
        .expect("its copy");
    planes
}

/// `bytes` as a plane of `rows.1` rows of `rows.0` bytes, with nothing between them.
fn plane(bytes: &mut [u8], (row, rows): (usize, usize)) -> PlaneRows<'_> {
    PlaneRows {
        bytes,
        bytes_per_row: row,
        rows,
        row_bytes: row,
    }
}

/// The times of a stream's draws, in milliseconds: the first, and the rest's mean and most.
fn times(draws: &[Duration]) -> String {
    let ms = |time: Duration| time.as_secs_f64() * 1e3;
    let Some((first, rest)) = draws.split_first() else {
        return "none".to_owned();
    };
    let rest_mean = match rest.len() {
        0 => 0.0,
        n => ms(rest.iter().sum::<Duration>()) / n as f64,
    };
    let all_mean = ms(draws.iter().sum::<Duration>()) / draws.len() as f64;
    let most = rest.iter().copied().max().map_or(0.0, ms);
    format!(
        "{} draws: mean {all_mean:.3} ms, the first {:.3} ms, the rest's mean {rest_mean:.3} ms \
         and most {most:.3} ms",
        draws.len(),
        ms(*first)
    )
}

/// Every picture of each H.264 and HEVC stream of `RVG_VA_STREAMS`, 8 and 10 bits, drawn into
/// targets of its crop's size, made before its first picture, holds exactly what `vaGetImage`
/// reads back of its crop, the luma's samples and the chroma's pairs, as the CPU's copy writes
/// them; the draws, made before the first stream, say how long they took to make, and each
/// stream how long its targets took and its draws did, the first apart from the rest. Run
/// holding the GPU, on vaon12 on the default adapter.
#[test]
#[ignore = "needs vaon12 and test streams: run on the Windows host"]
fn each_picture_drawn_into_its_targets_is_what_get_image_reads_back() {
    // The tests' clock: std's, which the library does not read.
    let clock = StdRuntime;
    read_back_by(Some(ReadbackWay::GetImage));
    let display = Display::open_adapter(None, StdRuntime::shared()).expect("a VA display");
    let made = clock.now();
    let draw = PlaneDraw::new(None).expect("the plane draws");
    eprintln!(
        "d3d11: the draws made in {:.3} ms",
        clock.elapsed_since(made).as_secs_f64() * 1e3
    );
    let (mut pictures, mut differ) = (0usize, Vec::new());
    for stream in streams() {
        let name = stream
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let file = std::fs::read(&stream).expect("a stream");
        let hevc = name.ends_with(".265");
        let units = if hevc {
            crate::codec::hevc::nal::access_units(&file)
        } else {
            crate::codec::h264::nal::access_units(&file)
        };
        let (mut h264, mut h265) = if hevc {
            (None, Some(HevcDecoder::new_annex_b(&display)))
        } else {
            (Some(H264Decoder::new_annex_b(&display)), None)
        };
        let mut targets: Option<PlaneTargets> = None;
        let mut draws = Vec::new();
        let mut land = |frame: &Frame| {
            frame.sync().expect("the decode");
            let texture = frame.texture().expect("a picture in a shared texture");
            let size = (frame.width(), frame.height());
            let to = targets.get_or_insert_with(|| {
                let start = clock.now();
                let made = draw
                    .targets(size, texture.is_10_bit())
                    .expect("the targets");
                eprintln!(
                    "d3d11: {name}: {}x{} {}-bit targets made in {:.3} ms",
                    size.0,
                    size.1,
                    frame.bit_depth(),
                    clock.elapsed_since(start).as_secs_f64() * 1e3
                );
                made
            });
            assert_eq!(to.size(), size, "one size a stream");
            let start = clock.now();
            draw.draw(&texture, frame.crop_origin(), to, size)
                .expect("the draw");
            draws.push(clock.elapsed_since(start));
            let drawn = draw.read_back(to).expect("the targets' read back");
            let want = got_image(frame);
            for (plane, (drawn, want)) in ["luma", "chroma"].iter().zip(drawn.iter().zip(&want)) {
                assert_eq!(drawn.len(), want.len(), "{name}: the {plane}'s bytes");
                let wrong = drawn.iter().zip(want).filter(|(a, b)| a != b).count();
                if wrong > 0 {
                    differ.push(format!("{name}: picture {pictures}: {wrong} {plane} bytes"));
                }
            }
            pictures += 1;
        };
        for (id, unit) in units.iter().enumerate() {
            let id = u64::try_from(id).expect("an id");
            let frames = match (&mut h264, &mut h265) {
                (Some(decoder), _) => decoder.decode(unit, id),
                (_, Some(decoder)) => decoder.decode(unit, id),
                _ => unreachable!("a decoder"),
            }
            .expect("a decode");
            frames.iter().for_each(&mut land);
        }
        let flushed = match (&mut h264, &mut h265) {
            (Some(decoder), _) => decoder.flush(),
            (_, Some(decoder)) => decoder.flush(),
            _ => unreachable!("a decoder"),
        }
        .expect("the flush");
        flushed.iter().for_each(&mut land);
        drop(land);
        eprintln!("d3d11: {name}: {}", times(&draws));
    }
    eprintln!(
        "d3d11: {pictures} pictures drawn, {} differ: {differ:?}",
        differ.len()
    );
    assert!(pictures > 0);
    assert!(differ.is_empty(), "{differ:?}");
}

/// A shared texture of the default adapter's, of `ten_bit` samples or 8-bit ones, 64 by 48: a
/// picture's pool texture, its texels as they were made.
fn source(ten_bit: bool) -> Arc<SharedTexture> {
    let format = if ten_bit {
        SurfaceFormat::P010
    } else {
        SurfaceFormat::Nv12
    };
    let device = Device::new(None).expect("a Direct3D 12 device");
    let texture = device
        .shared_texture(format, 64, 48)
        .expect("a shared texture");
    Arc::new(texture)
}

/// The bytes of `targets`' planes past a picture of `extent` luma samples drawn from their first
/// texel, of `sample` bytes a luma sample: the luma's and the chroma's, each a plane of `size`
/// luma samples read back row after row with nothing between them.
fn past_picture(
    planes: &[Vec<u8>; 2],
    size: (usize, usize),
    extent: (usize, usize),
    sample: usize,
) -> [Vec<u8>; 2] {
    let half = |(w, h): (usize, usize)| (w.div_ceil(2), h.div_ceil(2));
    let shapes = [
        (size.0 * sample, extent.0 * sample, extent.1),
        (
            half(size).0 * 2 * sample,
            half(extent).0 * 2 * sample,
            half(extent).1,
        ),
    ];
    std::array::from_fn(|n| {
        let (row, drawn_row, drawn_rows) = shapes[n];
        (planes[n].chunks(row).enumerate())
            .flat_map(|(y, bytes)| {
                if y < drawn_rows {
                    &bytes[drawn_row..]
                } else {
                    bytes
                }
            })
            .copied()
            .collect()
    })
}

/// Targets of 64 by 48 luma samples, of 8-bit samples and of 10-bit ones, cleared to bytes of
/// their own, then drawn a picture of 32 by 16 from a pool texture: where they owe zeros, the
/// draw clears them first, so every byte past the picture reads zero, and they owe none after;
/// where they owe none, every byte past the picture is as it was. Run holding the GPU, on the
/// default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn a_draw_short_of_its_targets_leaves_zeros_past_the_picture_only_where_they_owe_them() {
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let texture = source(ten_bit);
        let sample = if ten_bit { 2 } else { 1 };
        for owed in [true, false] {
            let to = draw.targets((64, 48), ten_bit).expect("the targets");
            draw.clear_to(&to, [255.0; 4]).expect("the targets' clear");
            let before = draw.read_back(&to).expect("the targets' read back");
            let kept = past_picture(&before, (64, 48), (32, 16), sample);
            assert!(
                kept.iter().all(|plane| plane.iter().any(|&b| b != 0)),
                "{ten_bit}: the clear wrote bytes of its own"
            );
            if owed {
                to.owe_zeros();
            }
            draw.draw(&texture, (0, 0), &to, (32, 16))
                .expect("the draw");
            assert!(
                !to.owes_zeros(),
                "{ten_bit}, {owed}: none owed after the draw"
            );
            let after = draw.read_back(&to).expect("the targets' read back");
            let past = past_picture(&after, (64, 48), (32, 16), sample);
            for (plane, (past, kept)) in ["luma", "chroma"].iter().zip(past.iter().zip(&kept)) {
                if owed {
                    assert!(
                        past.iter().all(|&b| b == 0),
                        "{ten_bit}: the {plane}'s bytes past the picture read zero"
                    );
                } else {
                    assert_eq!(
                        past, kept,
                        "{ten_bit}: the {plane}'s bytes past the picture"
                    );
                }
            }
        }
    }
}

/// [`PlaneDraw::zero_owed`] clears targets that owe zeros, and they owe none after; it keeps
/// the bytes of targets that owe none, as of targets another device has since said it wrote
/// whole ([`PlaneTargets::written`]). Run holding the GPU, on the default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn zero_owed_clears_only_targets_that_owe_zeros() {
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let to = draw.targets((64, 48), ten_bit).expect("the targets");
        draw.clear_to(&to, [255.0; 4]).expect("the targets' clear");
        let before = draw.read_back(&to).expect("the targets' read back");
        assert!(before.iter().all(|plane| plane.iter().any(|&b| b != 0)));
        draw.zero_owed(&to).expect("no clear");
        assert_eq!(
            draw.read_back(&to).expect("a read back"),
            before,
            "{ten_bit}: none owed"
        );
        to.owe_zeros();
        to.written();
        draw.zero_owed(&to).expect("no clear");
        assert_eq!(
            draw.read_back(&to).expect("a read back"),
            before,
            "{ten_bit}: written"
        );
        to.owe_zeros();
        draw.zero_owed(&to).expect("the clear");
        assert!(!to.owes_zeros(), "{ten_bit}: none owed after");
        let after = draw.read_back(&to).expect("a read back");
        assert!(
            after.iter().all(|plane| plane.iter().all(|&b| b == 0)),
            "{ten_bit}: every byte reads zero"
        );
    }
}

/// A picture's extent covers its targets where it is their size, or more, each way, and only
/// then.
#[test]
fn a_picture_covers_its_targets_only_at_their_size_each_way() {
    assert!(covers((64, 48), (64, 48)));
    assert!(covers((63, 47), (64, 48)));
    assert!(!covers((64, 48), (64, 47)));
    assert!(!covers((64, 48), (63, 48)));
    assert!(!covers((3840, 2160), (1920, 1080)));
}
