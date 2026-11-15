//! The probes of a decoder's session start ([`va`], [`nvdec`]), as the reims-graphics device's
//! video starts one for each clip a guest's player opens: a new decoder, its first access unit
//! (the parameter sets and an IDR picture) decoded, and the wait for that picture; then the
//! second access unit and its wait, a picture of a session already started, for comparison. Each
//! driver call of a decode is timed by [`super::timeline`], so a session's first decode is split
//! into its calls: on VA-API the config, the surface pool, the context, the picture's buffers and
//! its submission; on NVDEC the capabilities query, the decoder's make and the picture's
//! submission.
//!
//! Each stream's sessions start one after another in one process, on one display or GPU, which
//! is opened once, as the device's video opens them once a device: the process's first session
//! is its coldest, and a later session of a stream decoded before is warm. Where sessions are
//! asked to run beside (`RVG_START_BESIDE`), as many decode another stream over and over on
//! threads of their own meanwhile, each with a decoder of its own, as a guest's other players
//! do.
//!
//! The probes take their inputs from the environment:
//! - `RVG_START_STREAMS`: the Annex B streams, `*.264` (H.264) or `*.265` (HEVC), absolute and
//!   `:`-separated, in the order their sessions start;
//! - `RVG_START_SESSIONS`: the sessions a stream, 3 where unset;
//! - `RVG_START_BESIDE` and `RVG_START_BESIDE_STREAM`: how many sessions decode the latter
//!   meanwhile, none where unset;
//! - `RVG_VAAPI_NODE`: VA-API's render node, `/dev/dri/renderD128` where unset;
//! - `RVG_NVIDIA_GPU`: NVDEC's GPU, as `GpuSelector::parse` reads it, the first where unset.
//!
//! The NVDEC probe asks NVDEC's capabilities once each, as the device's arm does as it opens,
//! before its first session.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use super::timeline::{self, Span, ms};
use crate::codec::{h264, hevc};
use crate::core::runtime::{Runtime, StdRuntime};
use crate::nvidia::{
    AnnexBCodec, AnnexBDecoder, AnnexBPicture, ChromaFormat, Gpu, GpuSelector, NvdecCodec,
    decode_capabilities,
};
use crate::vaapi::gated::GatedRuntime;
use crate::vaapi::{Display, Frame, H264Decoder, HevcDecoder};

/// A stream the probes decode.
struct Stream {
    name: String,
    hevc: bool,
    bytes: Vec<u8>,
}

impl Stream {
    fn read(path: &str) -> Self {
        let path = PathBuf::from(path);
        let hevc = match path.extension().and_then(|e| e.to_str()) {
            Some("265") => true,
            Some("264") => false,
            _ => panic!("{}: not a .264 or .265 stream", path.display()),
        };
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let name =
            (path.file_name()).map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        Self { name, hevc, bytes }
    }

    fn units(&self) -> Vec<&[u8]> {
        if self.hevc {
            hevc::nal::access_units(&self.bytes)
        } else {
            h264::nal::access_units(&self.bytes)
        }
    }
}

/// What the environment asks (module doc).
struct Asked {
    streams: Vec<Stream>,
    sessions: usize,
    beside: usize,
    beside_stream: Option<Stream>,
}

impl Asked {
    fn of_env() -> Self {
        let streams = std::env::var("RVG_START_STREAMS").expect("RVG_START_STREAMS: the streams");
        let streams: Vec<Stream> = (streams.split(':').filter(|s| !s.is_empty()))
            .map(Stream::read)
            .collect();
        assert!(!streams.is_empty(), "no streams in RVG_START_STREAMS");
        let number = |name: &str, default: usize| {
            std::env::var(name).map_or(default, |v| {
                v.parse().unwrap_or_else(|e| panic!("{name}: {e}"))
            })
        };
        let beside = number("RVG_START_BESIDE", 0);
        let beside_stream = (beside > 0).then(|| {
            let path = std::env::var("RVG_START_BESIDE_STREAM")
                .expect("RVG_START_BESIDE_STREAM: the stream the sessions beside decode");
            Stream::read(&path)
        });
        Self {
            streams,
            sessions: number("RVG_START_SESSIONS", 3),
            beside,
            beside_stream,
        }
    }
}

/// The time `work` took, and its answer.
fn timed<T>(work: impl FnOnce() -> T) -> (T, Duration) {
    let started = StdRuntime.now();
    let answer = work();
    (answer, StdRuntime.elapsed_since(started))
}

/// Sessions decoding a stream over and over on threads of their own, until dropped.
struct Beside {
    stop: Arc<AtomicBool>,
    decoded: Arc<AtomicU64>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Beside {
    /// `count` threads, each running `session` until told to stop, counting its pictures.
    fn start(
        count: usize,
        session: impl Fn(&AtomicBool, &AtomicU64) + Send + Sync + 'static,
    ) -> Self {
        let (stop, decoded) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
        );
        let session = Arc::new(session);
        let threads = (0..count)
            .map(|_| {
                let (stop, decoded, session) = (
                    Arc::clone(&stop),
                    Arc::clone(&decoded),
                    Arc::clone(&session),
                );
                StdRuntime::thread("probe-beside", move || session(&stop, &decoded))
                    .expect("a thread")
            })
            .collect();
        let beside = Self {
            stop,
            decoded,
            threads,
        };
        // Each has its decoder and is decoding before the sessions timed start.
        let started = StdRuntime.now();
        while beside.decoded.load(Ordering::Relaxed) < 30 * count as u64
            && StdRuntime.elapsed_since(started) < Duration::from_secs(10)
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        beside
    }

    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let decoded = self.decoded.load(Ordering::Relaxed);
        for thread in self.threads {
            thread.join().expect("a session beside");
        }
        println!("  the sessions beside decoded {decoded} pictures in all");
    }
}

/// A VA-API decoder of either codec.
enum VaCodec {
    H264(H264Decoder),
    Hevc(HevcDecoder),
}

impl VaCodec {
    fn new(display: &Display, hevc: bool) -> Self {
        if hevc {
            Self::Hevc(HevcDecoder::new_annex_b(display))
        } else {
            Self::H264(H264Decoder::new_annex_b(display))
        }
    }

    fn decode(&mut self, unit: &[u8], id: u64) -> Vec<Frame> {
        match self {
            Self::H264(d) => d.decode(unit, id),
            Self::Hevc(d) => d.decode(unit, id),
        }
        .expect("a decode")
    }
}

/// A picture's decode, its frame and the decode's split, and the wait for it.
struct VaPicture {
    decode: Span,
    wait: Duration,
}

/// Decodes `units` from `next` on until an access unit gives a picture, and waits for it; the
/// frame is kept in `held` while the session goes on, as a session holds its pictures until they
/// land.
fn va_picture(
    decoder: &mut VaCodec,
    units: &[&[u8]],
    next: &mut usize,
    held: &mut Vec<Frame>,
) -> VaPicture {
    loop {
        let unit = units.get(*next).expect("an access unit with a picture");
        let id = *next as u64;
        *next += 1;
        let (frames, decode) = timeline::record(|| decoder.decode(unit, id));
        if let Some(frame) = frames.first() {
            let (synced, wait) = timed(|| frame.sync());
            synced.expect("the decode's wait");
            held.extend(frames);
            return VaPicture { decode, wait };
        }
    }
}

/// The VA-API probe (module doc).
pub(crate) fn va() {
    let asked = Asked::of_env();
    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let (display, open) = timed(|| Display::open(&node, Arc::new(GatedRuntime)));
    let display = display.expect("a VA display");
    println!(
        "VA-API at {node}: {}, opened in {} ms",
        display.vendor(),
        ms(open)
    );
    let beside = asked.beside_stream.map(|stream| {
        let display = display.clone_ref();
        Beside::start(asked.beside, move |stop, decoded| {
            let units = stream.units();
            let mut decoder = VaCodec::new(&display, stream.hevc);
            let mut id = 0u64;
            while !stop.load(Ordering::Relaxed) {
                for unit in &units {
                    for frame in decoder.decode(unit, id) {
                        frame.sync().expect("a decode beside");
                        decoded.fetch_add(1, Ordering::Relaxed);
                    }
                    id += 1;
                }
            }
        })
    });
    for stream in &asked.streams {
        let units = stream.units();
        for session in 1..=asked.sessions {
            let mut held = Vec::new();
            let mut next = 0;
            let (mut decoder, made) = timed(|| VaCodec::new(&display, stream.hevc));
            let first = va_picture(&mut decoder, &units, &mut next, &mut held);
            let second = va_picture(&mut decoder, &units, &mut next, &mut held);
            let ((), ended) = timed(|| {
                held.clear();
                drop(decoder);
            });
            let span = &first.decode;
            println!(
                "{} session {session}: made {} ms; first decode {}, its wait {} ms; second \
                 decode {}, its wait {} ms; ended {} ms",
                stream.name,
                ms(made),
                span.describe(),
                ms(first.wait),
                second.decode.describe(),
                ms(second.wait),
                ms(ended),
            );
            println!(
                "  split: config {} ms, pool {} ms, context {} ms, buffers {} ms, submission {} \
                 ms (first vaEndPicture {} ms); to the picture {} ms",
                ms(span.of("vaCreateConfig")),
                ms(span.of("vaCreateSurfaces")),
                ms(span.of("vaCreateContext")),
                ms(span.of("vaCreateBuffer")),
                ms(span.of("vaBeginPicture")
                    + span.of("vaRenderPicture")
                    + span.of("vaEndPicture")),
                ms(span.of("vaEndPicture")),
                ms(span.took + first.wait),
            );
        }
    }
    if let Some(beside) = beside {
        beside.stop();
    }
}

/// An NVDEC picture's decode and the decode's split, and its read back's.
struct NvPicture {
    decode: Span,
    read_back: Span,
}

/// [`va_picture`] on NVDEC: the wait is the read back, which waits for the decode, maps the
/// picture and copies it into the decoder's staging memory.
fn nv_picture(
    decoder: &mut AnnexBDecoder,
    units: &[&[u8]],
    next: &mut usize,
    held: &mut Vec<AnnexBPicture>,
) -> NvPicture {
    loop {
        let unit = units.get(*next).expect("an access unit with a picture");
        *next += 1;
        let (picture, decode) = timeline::record(|| decoder.decode(unit).expect("a decode"));
        if let Some(picture) = picture {
            let (_, _, width, height) = picture.crop();
            let (read, read_back) =
                timeline::record(|| picture.read_back((width, height)).map(drop));
            read.expect("the read back");
            held.push(picture);
            return NvPicture { decode, read_back };
        }
    }
}

/// The NVDEC probe (module doc).
pub(crate) fn nvdec() {
    let asked = Asked::of_env();
    let selector = GpuSelector::parse(&std::env::var("RVG_NVIDIA_GPU").unwrap_or_default());
    let (gpu, open) = timed(|| Gpu::open(&selector));
    let gpu = Arc::new(gpu.expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    // The capabilities the graphics device's NVDEC arm asks as it opens (reims-vgpu's
    // `graphics/nvidia.rs`, `supported`), which load NVDEC's library.
    let ((), caps_took) = timed(|| {
        for (codec, bit_depth) in [
            (NvdecCodec::H264, 8),
            (NvdecCodec::Hevc, 8),
            (NvdecCodec::Hevc, 10),
        ] {
            decode_capabilities(&gpu, codec, ChromaFormat::Yuv420, bit_depth)
                .expect("NVDEC's caps");
        }
    });
    println!(
        "NVDEC on {} ({}), driver {}: opened in {} ms, its caps asked as the arm opens in {} ms",
        gpu.info().name,
        gpu.info().pci_bus_id,
        gpu.driver_version(),
        ms(open),
        ms(caps_took)
    );
    let beside = asked.beside_stream.map(|stream| {
        let (gpu, runtime) = (Arc::clone(&gpu), Arc::clone(&runtime));
        Beside::start(asked.beside, move |stop, decoded| {
            let units = stream.units();
            let codec = if stream.hevc {
                AnnexBCodec::Hevc
            } else {
                AnnexBCodec::H264
            };
            let mut decoder = AnnexBDecoder::new(&gpu, codec, &runtime);
            while !stop.load(Ordering::Relaxed) {
                for unit in &units {
                    if let Some(picture) = decoder.decode(unit).expect("a decode beside") {
                        let (_, _, width, height) = picture.crop();
                        drop(
                            picture
                                .read_back((width, height))
                                .expect("a read back beside"),
                        );
                        decoded.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        })
    });
    for stream in &asked.streams {
        let units = stream.units();
        let codec = if stream.hevc {
            AnnexBCodec::Hevc
        } else {
            AnnexBCodec::H264
        };
        for session in 1..=asked.sessions {
            let mut held = Vec::new();
            let mut next = 0;
            let (mut decoder, made) = timed(|| AnnexBDecoder::new(&gpu, codec, &runtime));
            let first = nv_picture(&mut decoder, &units, &mut next, &mut held);
            let second = nv_picture(&mut decoder, &units, &mut next, &mut held);
            let ((), ended) = timed(|| {
                held.clear();
                drop(decoder);
            });
            let span = &first.decode;
            println!(
                "{} session {session}: made {} ms; first decode {}, its read back {}; second \
                 decode {}, its read back {}; ended {} ms",
                stream.name,
                ms(made),
                span.describe(),
                first.read_back.describe(),
                second.decode.describe(),
                second.read_back.describe(),
                ms(ended),
            );
            println!(
                "  split: capabilities {} ms, stream {} ms, decoder {} ms, submission {} ms; to \
                 the picture {} ms",
                ms(span.of("cuvidGetDecoderCaps")),
                ms(span.of("cuStreamCreate")),
                ms(span.of("cuvidCreateDecoder")),
                ms(span.of("cuvidDecodePicture")),
                ms(span.took + first.read_back.took),
            );
        }
    }
    if let Some(beside) = beside {
        beside.stop();
    }
}
