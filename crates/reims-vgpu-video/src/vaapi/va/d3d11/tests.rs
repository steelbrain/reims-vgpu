//! The plane draws on the default adapter, against `vaGetImage`'s read back of each picture.

use std::path::PathBuf;
use std::time::Duration;

use crate::core::runtime::{Runtime as _, StdRuntime};

use super::super::display::Display;
use super::super::frame::{Frame, PlaneRows};
use super::super::h264::H264Decoder;
use super::super::hevc::HevcDecoder;
use super::super::readback::{ReadbackWay, read_back_by};
use super::{PlaneDraw, PlaneTargets};

/// The streams of the directories `RVG_VA_STREAMS` lists (as `PATH` lists them): each H.264
/// (`.264`) and HEVC (`.265`) file, in name order.
fn streams() -> Vec<PathBuf> {
    let dirs = std::env::var_os("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
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
    let rows = [(width * sample, height), (width.div_ceil(2) * 2 * sample, height.div_ceil(2))];
    let mut planes = rows.map(|(row, rows)| vec![0u8; row * rows]);
    let [luma, chroma] = &mut planes;
    let back = frame.read_back().expect("a read back");
    back.copy_into(plane(luma, rows[0]), plane(chroma, rows[1])).expect("its copy");
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
        let name = stream.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
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
                let made = draw.targets(size, texture.is_10_bit()).expect("the targets");
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
            draw.draw(&texture, frame.crop_origin(), to, size).expect("the draw");
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
    eprintln!("d3d11: {pictures} pictures drawn, {} differ: {differ:?}", differ.len());
    assert!(pictures > 0);
    assert!(differ.is_empty(), "{differ:?}");
}
