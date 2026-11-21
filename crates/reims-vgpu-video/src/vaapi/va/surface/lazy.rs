//! Pools that make each surface as it is asked for one (`super::SurfacePool::get`): the decision
//! itself, off the GPU, and decoders' pools on it.
//!
//! Off the GPU, `PoolState::take_or_make`: a pool makes a surface only when one is asked of it
//! and none is free, lends the one freed last first, and makes no more than `MAX_SURFACES`; a
//! decoder's pool makes none with its context.
//!
//! On the GPU, run on a VA-API host holding its GPU: on Windows on the adapter libva-win32 opens
//! first, elsewhere on the render node `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is
//! unset).
//! - A pool's surfaces are made one at a time, each lent again once freed, with how long one
//!   surface's make takes printed for 1080p NV12 and 4K NV12 and P010, beside a pool's count made
//!   at once, and on Linux what the process's GPU memory grew by once they are made and once each
//!   holds an image (`vaPutImage`).
//! - Streams of the driver's own encoder (H.264, HEVC Main and Main 10, at 1080p and 4K, each
//!   where the driver encodes it) decode to the same pictures with the decoder's surfaces made
//!   as pictures need them, its context given no render target, as with them made up front as
//!   the context's render targets (`super::UP_FRONT`: the DPB, the picture and two spare, the
//!   count decoders made before), and make no more of them. Each way's first picture's time,
//!   which holds the decoder's config, context and first surfaces, and its later pictures', are
//!   printed.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

use crate::codec::encode::{EncoderSettings, InputFrame, Plane, Profile as Encoded};
use crate::core::runtime::{Runtime as _, StdRuntime};
use crate::vaapi::va::display::{Display, Error};
use crate::vaapi::va::encoder::VideoEncoder;
use crate::vaapi::va::frame::{Frame, PlaneRows};
use crate::vaapi::va::teardown::display;
use crate::vaapi::va::{H264Decoder, HevcDecoder};

use super::{MAX_SURFACES, PoolState, Surface, SurfaceFormat, SurfacePool, UP_FRONT};

#[test]
fn a_decoders_pool_makes_no_surface_with_its_context() {
    assert_eq!(super::made_with_context(), 0);
}

#[test]
fn a_pool_makes_a_surface_only_when_one_is_asked_for_and_none_is_free() {
    let mut state = PoolState::default();
    let mut made = 0;
    let first = state
        .take_or_make(|before| {
            assert_eq!(before, 0);
            made += 1;
            Ok(10)
        })
        .expect("a surface");
    assert_eq!((first, made), (10, 1));
    assert_eq!(state.all, [10]);
    state.free.push(first);
    let again = state
        .take_or_make(|_| panic!("a surface made while one is free"))
        .expect("a surface");
    assert_eq!(again, 10);
    assert_eq!(state.all, [10]);
}

#[test]
fn the_surface_freed_last_is_lent_first() {
    let mut state = PoolState::default();
    for id in 1..=3 {
        state.take_or_make(|_| Ok(id)).expect("a surface");
    }
    state.free.extend([1, 3, 2]);
    let lent: Vec<_> = (0..3)
        .map(|_| {
            (state.take_or_make(|_| panic!("a surface made while one is free"))).expect("a surface")
        })
        .collect();
    assert_eq!(lent, [2, 3, 1]);
}

#[test]
fn a_pool_makes_no_more_than_max_surfaces() {
    let mut state = PoolState::default();
    for n in 0..MAX_SURFACES {
        let id = u32::try_from(n).expect("an id") + 1;
        state
            .take_or_make(|before| {
                assert_eq!(before, n);
                Ok(id)
            })
            .expect("a surface");
    }
    let past = state.take_or_make(|_| panic!("a surface made past the bound"));
    assert!(matches!(past, Err(Error::OutOfSurfaces)), "{past:?}");
    state.free.push(7);
    let freed = state.take_or_make(|_| panic!("a surface made while one is free"));
    assert_eq!(freed.expect("the one freed"), 7);
}

#[test]
fn a_failed_make_leaves_the_pool_as_it_was() {
    let mut state = PoolState::default();
    let failed = state.take_or_make(|_| Err(Error::Unsupported("no memory".into())));
    assert!(matches!(failed, Err(Error::Unsupported(_))), "{failed:?}");
    assert!(state.all.is_empty() && state.free.is_empty());
}

/// Surfaces lent and freed in any order, as a decode's pictures, its DPB's and the caller's
/// frames take and let go of them: the pool holds as many as were lent at once, at most.
#[test]
fn a_pool_holds_no_more_surfaces_than_were_lent_at_once() {
    let mut state = PoolState::default();
    let mut lent = Vec::new();
    let mut most = 0;
    let mut next = 0;
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..10_000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if lent.is_empty() || (seed % 2 == 0 && lent.len() < MAX_SURFACES) {
            let id = state
                .take_or_make(|_| {
                    next += 1;
                    Ok(next)
                })
                .expect("a surface");
            assert!(!lent.contains(&id), "surface {id} lent twice");
            lent.push(id);
            most = most.max(lent.len());
        } else {
            let at = usize::try_from(seed >> 8).expect("an index") % lent.len();
            state.free.push(lent.swap_remove(at));
        }
        assert_eq!(state.all.len(), most);
    }
}

/// The surfaces each make is timed for.
const COUNT: usize = 8;

/// A make's time in milliseconds.
fn ms(time: Duration) -> f64 {
    time.as_secs_f64() * 1e3
}

/// The GPU memory this process's DRM clients hold, in MiB, each client once: what their objects
/// take (each client's `drm-total-*` sizes in `/proc/self/fdinfo`, or its `drm-memory-*` sizes
/// where it gives none), and what of that is resident (`drm-resident-*`, or the same where it
/// gives none); `None` off Linux, or where no client gives either.
fn gpu_memory_mib() -> Option<[f64; 2]> {
    #[cfg(target_os = "linux")]
    {
        let mut clients = std::collections::HashSet::new();
        let mut kib = [0; 2];
        let mut any = false;
        for entry in std::fs::read_dir("/proc/self/fdinfo").ok()?.flatten() {
            let Ok(text) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let mut lines = text.lines();
            let client = lines.find_map(|line| line.strip_prefix("drm-client-id:"));
            if !client.is_some_and(|id| clients.insert(id.trim().to_owned())) {
                continue;
            }
            let mut total = sizes(&text, "drm-total-");
            if total.is_empty() {
                total = sizes(&text, "drm-memory-");
            }
            let mut resident = sizes(&text, "drm-resident-");
            if resident.is_empty() {
                resident.clone_from(&total);
            }
            any |= !total.is_empty();
            kib[0] += total.iter().sum::<u64>();
            kib[1] += resident.iter().sum::<u64>();
        }
        any.then_some(kib.map(|kib| kib as f64 / 1024.0))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// The sizes, in KiB, of the lines of an fdinfo `text` whose keys start with `prefix`.
#[cfg(target_os = "linux")]
fn sizes(text: &str, prefix: &str) -> Vec<u64> {
    let mut sizes = Vec::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let mut words = value.split_whitespace();
        let Some(Ok(n)) = words.next().map(str::parse::<u64>) else {
            continue;
        };
        if key.starts_with(prefix) {
            sizes.push(match words.next() {
                Some("KiB") => n,
                Some("MiB") => n << 10,
                Some("GiB") => n << 20,
                _ => n >> 10,
            });
        }
    }
    sizes
}

#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn surfaces_are_made_one_at_a_time_and_lent_again_once_freed() {
    let display = display();
    for (width, height, format) in [
        (1920, 1088, SurfaceFormat::Nv12),
        (3840, 2160, SurfaceFormat::Nv12),
        (3840, 2160, SurfaceFormat::P010),
    ] {
        let name = format!("{} {width}x{height}", format.name());
        let pool = SurfacePool::new(&display, format, width, height, 0).expect("a pool");
        assert_eq!(pool.made(), 0, "{name}");
        let image = super::create_image(&display, format, width, height).expect("an image");
        let before = gpu_memory_mib();
        let mut held = Vec::new();
        let mut times = Vec::new();
        for n in 1..=COUNT {
            let start = StdRuntime.now();
            held.push(pool.get().expect("a surface"));
            times.push(StdRuntime.now().saturating_duration_since(start));
            assert_eq!(pool.made(), n, "{name}");
        }
        let made = gpu_memory_mib();
        for surface in &held {
            image
                .put_into(surface.id(), width, height)
                .expect("vaPutImage");
        }
        let used = gpu_memory_mib();
        if let (Some(before), Some(made), Some(used)) = (before, made, used) {
            eprintln!(
                "{name}: the process's GPU memory (DRM fdinfo, total/resident) {:+.1}/{:+.1} MiB \
                 once {COUNT} are made, {:+.1}/{:+.1} MiB once an image is put into each",
                made[0] - before[0],
                made[1] - before[1],
                used[0] - before[0],
                used[1] - before[1],
            );
        }
        drop(image);
        let mut ids: Vec<_> = held.iter().map(Surface::id).collect();
        held.clear();
        let again: Vec<_> = (0..COUNT).map(|_| pool.get().expect("a surface")).collect();
        assert_eq!(pool.made(), COUNT, "{name}: made again once freed");
        let mut lent: Vec<_> = again.iter().map(Surface::id).collect();
        ids.sort_unstable();
        lent.sort_unstable();
        assert_eq!(ids, lent, "{name}");
        drop(again);
        drop(pool);
        let start = StdRuntime.now();
        let batch = SurfacePool::new(&display, format, width, height, COUNT).expect("a pool");
        let at_once = StdRuntime.now().saturating_duration_since(start);
        drop(batch);
        let first = times[0];
        times.sort_unstable();
        eprintln!(
            "{name}: one surface made in {:.3} ms (median), {:.3} to {:.3} ms, the first {:.3} \
             ms; {COUNT} made at once in {:.3} ms",
            ms(times[COUNT / 2]),
            ms(times[0]),
            ms(times[COUNT - 1]),
            ms(first),
            ms(at_once),
        );
    }
}

/// The pictures each stream holds.
const PICTURES: u64 = 30;

/// The rounds each way decodes a stream in.
const ROUNDS: usize = 3;

/// `PICTURES` pictures of a gradient moving a few samples a picture, `size`, by the driver's
/// encoder of `profile`: its configuration record and samples; `None` where the driver encodes
/// no such stream.
fn encoded(
    display: &Display,
    profile: Encoded,
    size: (u32, u32),
) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let mut encoder =
        VideoEncoder::new(display, EncoderSettings::new(profile, size.0, size.1)).ok()?;
    let (w, h) = (size.0 as usize, size.1 as usize);
    let sample = if profile == Encoded::HevcMain10 { 2 } else { 1 };
    let mut samples = Vec::new();
    for n in 0..PICTURES {
        let shift = usize::try_from(n).expect("a shift") * 3;
        let mut bytes = vec![0u8; w * h * 3 / 2 * sample];
        for (i, chunk) in bytes.chunks_exact_mut(sample).enumerate() {
            let (x, y) = (i % w, i / w);
            let value = ((x + shift) * 3 + y * 5 + (x * y) % 7) as u8;
            match chunk {
                [b] => *b = value,
                [low, high] => (*low, *high) = (0, value),
                _ => unreachable!(),
            }
        }
        let (luma, chroma) = bytes.split_at(w * h * sample);
        let frame = InputFrame {
            planes: [
                Plane {
                    data: luma,
                    bytes_per_row: w * sample,
                },
                Plane {
                    data: chroma,
                    bytes_per_row: w * sample,
                },
            ],
            pts: i64::try_from(n).expect("a time") * 3000,
            duration: 3000,
            force_key_frame: false,
            id: n,
        };
        let coded = encoder.encode(&frame).ok()?;
        samples.extend(coded.into_iter().map(|c| c.expect("a sample").data));
    }
    samples.extend(
        encoder
            .flush()
            .into_iter()
            .map(|c| c.expect("a sample").data),
    );
    Some((encoder.configuration().to_vec(), samples))
}

/// A decoder of one of the encoded streams.
enum Decoding {
    H264(H264Decoder),
    Hevc(HevcDecoder),
}

impl Decoding {
    fn new(display: &Display, profile: Encoded, record: &[u8]) -> Self {
        match profile {
            Encoded::HevcMain | Encoded::HevcMain10 => {
                Self::Hevc(HevcDecoder::new(display, record).expect("a decoder"))
            }
            _ => Self::H264(H264Decoder::new(display, record).expect("a decoder")),
        }
    }

    fn decode(&mut self, sample: &[u8], id: u64) -> Vec<Frame> {
        match self {
            Self::H264(decoder) => decoder.decode(sample, id),
            Self::Hevc(decoder) => decoder.decode(sample, id),
        }
        .expect("a decode")
    }

    fn surfaces_made(&self) -> usize {
        match self {
            Self::H264(decoder) => decoder.surfaces_made(),
            Self::Hevc(decoder) => decoder.surfaces_made(),
        }
        .expect("a session")
    }
}

/// The surfaces decoders made up front for `record`'s stream: its first SPS's DPB, the picture
/// and two spare.
fn up_front(profile: Encoded, record: &[u8]) -> usize {
    let dpb = match profile {
        Encoded::HevcMain | Encoded::HevcMain10 => {
            use crate::codec::hevc::{hvcc::DecoderConfiguration, nal, sps::Sps};
            let config = DecoderConfiguration::parse(record).expect("a record");
            let sps = (config.nal_units.iter())
                .find(|unit| {
                    nal::NalUnit::new(unit).is_ok_and(|n| n.unit_type() == nal::nal_type::SPS)
                })
                .expect("an SPS");
            Sps::parse(sps).expect("an SPS").dpb_size()
        }
        _ => {
            use crate::codec::h264::{avcc::DecoderConfiguration, sps::Sps};
            let config = DecoderConfiguration::parse(record).expect("a record");
            let sps = config.sequence_parameter_sets.first().expect("an SPS");
            Sps::parse(sps).expect("an SPS").dpb_frames()
        }
    };
    dpb + 1 + 2
}

/// One decode of a stream: each picture's hash, each access unit's time to decode and wait for
/// its pictures, and the surfaces the decoder made.
struct Run {
    pictures: Vec<u64>,
    times: Vec<Duration>,
    made: usize,
}

/// `samples` decoded by a new decoder, each picture read back as the CPU's copy does once it was
/// waited for, with the pool making `made_up_front` surfaces with the context.
fn run(
    display: &Display,
    profile: Encoded,
    record: &[u8],
    samples: &[Vec<u8>],
    made_up_front: usize,
) -> Run {
    UP_FRONT.with(|count| count.set(made_up_front));
    let mut decoder = Decoding::new(display, profile, record);
    let mut run = Run {
        pictures: Vec::new(),
        times: Vec::new(),
        made: 0,
    };
    for (id, sample) in (0u64..).zip(samples) {
        let start = StdRuntime.now();
        let frames = decoder.decode(sample, id);
        for frame in &frames {
            frame.sync().expect("a picture's wait");
        }
        run.times
            .push(StdRuntime.now().saturating_duration_since(start));
        UP_FRONT.with(|count| count.set(0));
        for frame in frames {
            run.pictures.push(hash(&frame));
        }
    }
    run.made = decoder.surfaces_made();
    UP_FRONT.with(|count| count.set(0));
    run
}

/// A hash of `frame`'s picture, as the CPU's copy reads it back.
fn hash(frame: &Frame) -> u64 {
    let sample = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let row = w * sample;
    let mut luma = vec![0u8; row * h];
    let mut chroma = vec![0u8; row * h.div_ceil(2)];
    let rows = |bytes, rows| PlaneRows {
        bytes,
        bytes_per_row: row,
        rows,
        row_bytes: row,
    };
    (frame.read_back().expect("a read back"))
        .copy_into(rows(&mut luma, h), rows(&mut chroma, h.div_ceil(2)))
        .expect("a copy");
    let mut hasher = DefaultHasher::new();
    luma.hash(&mut hasher);
    chroma.hash(&mut hasher);
    hasher.finish()
}

/// The median of `times`.
fn median(mut times: Vec<Duration>) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn pictures_decode_the_same_into_surfaces_made_as_they_are_needed() {
    let display = display();
    let mut compared = 0;
    for (profile, name) in [
        (Encoded::H264High, "H.264"),
        (Encoded::HevcMain, "HEVC Main"),
        (Encoded::HevcMain10, "HEVC Main 10"),
    ] {
        for size in [(1920, 1080), (3840, 2160)] {
            let name = format!("{name} {}x{}", size.0, size.1);
            let Some((record, samples)) = encoded(&display, profile, size) else {
                eprintln!("{name}: not compared: the driver encodes no such stream");
                continue;
            };
            let count = up_front(profile, &record);
            let mut runs: [Vec<Run>; 2] = Default::default();
            for round in 0..ROUNDS {
                // Each way first in turn.
                for way in [round % 2, 1 - round % 2] {
                    let made = if way == 0 { 0 } else { count };
                    runs[way].push(run(&display, profile, &record, &samples, made));
                }
            }
            let reference = &runs[1][0].pictures;
            assert!(!reference.is_empty(), "{name}: no picture");
            for (way, runs) in ["as needed", "up front"].iter().zip(&runs) {
                for run in runs {
                    assert!(
                        run.pictures == *reference,
                        "{name}: the pictures of surfaces made {way} differ"
                    );
                }
            }
            let [needed, made_up_front] = runs.each_ref().map(|runs| runs[0].made);
            assert!(
                needed <= made_up_front,
                "{name}: {needed} surfaces made as needed, {made_up_front} up front"
            );
            let first = |runs: &[Run]| median(runs.iter().map(|run| run.times[0]).collect());
            let later = |runs: &[Run]| {
                median(
                    runs.iter()
                        .flat_map(|run| run.times[1..].iter().copied())
                        .collect(),
                )
            };
            eprintln!(
                "{name}: {} pictures the same; surfaces made: {needed} as needed, \
                 {made_up_front} up front; the first access unit in {:.3} ms as needed, {:.3} \
                 ms up front; later ones in {:.3} ms, {:.3} ms (medians of {ROUNDS} runs)",
                reference.len(),
                ms(first(runs[0].as_slice())),
                ms(first(runs[1].as_slice())),
                ms(later(runs[0].as_slice())),
                ms(later(runs[1].as_slice())),
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "no stream the driver encodes");
}
