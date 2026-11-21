//! Decoders let go while others decode on the same display: two threads, each making an H.264
//! decoder, decoding a stream with it, reading each picture back as a CPU copy does, and letting
//! it go (its context, its pool and its config), round after round, so that each decoder's
//! teardown is asked for while the other decodes, as a guest's session ends while its next one
//! decodes (`super::display::Display::release_teardown`). Run on a VA-API host, holding its GPU:
//! on Windows on the adapter libva-win32 opens first, elsewhere on the render node
//! `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is unset). The stream is the Annex B
//! file `RVG_VA_H264` names, or else 4K noise the driver's own encoder makes; vaon12 on the UHD
//! Graphics 630 has no H.264 encoder, so there `RVG_VA_H264` names a stream.
//!
//! The churn test makes short decoders one after another beside a long one, as a guest's short
//! sessions come and go beside a long playback, compares their pictures with the long one's,
//! and on Windows reads the process's memory after each, to check that what the short decoders
//! leave held does not grow with their number (`super::display::Display::context_let_go`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use crate::codec::encode::{EncoderSettings, InputFrame, Plane, Profile};

use super::display::Display;
use super::encoder::VideoEncoder;
use super::h264::H264Decoder;

/// The size the encoder's stream is made at.
pub(super) const SIZE: (u32, u32) = (3840, 2160);

/// The pictures of the encoder's stream.
const FRAMES: u64 = 16;

/// The decoders each thread makes and lets go.
const ROUNDS: usize = 4;

/// The short decoders the churn test makes and lets go beside its long one.
const SHORT: usize = 10;

/// The access units each short decoder decodes, from the stream's first.
const SHORT_SAMPLES: usize = 8;

pub(super) fn display() -> Display {
    #[cfg(windows)]
    let display = Display::open_adapter(None, crate::core::runtime::StdRuntime::shared());
    #[cfg(unix)]
    let display = Display::open(
        std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into()),
        std::sync::Arc::new(super::gated::GatedRuntime),
    );
    display.expect("a VA display")
}

/// A stream the decoders decode: its avcC where it has one (the encoder's), and its samples.
pub(super) struct Stream {
    avcc: Option<Vec<u8>>,
    pub(super) samples: Vec<Vec<u8>>,
}

impl Stream {
    /// `RVG_VA_H264`'s access units, or [`FRAMES`] pictures of noise by the driver's encoder.
    pub(super) fn new(display: &Display) -> Stream {
        if let Some(path) = std::env::var_os("RVG_VA_H264") {
            let file = std::fs::read(&path).expect("RVG_VA_H264: an Annex B file");
            let samples = crate::codec::h264::nal::access_units(&file)
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect();
            return Stream {
                avcc: None,
                samples,
            };
        }
        let mut encoder = [
            Profile::H264High,
            Profile::H264Main,
            Profile::H264ConstrainedBaseline,
        ]
        .into_iter()
        .find_map(|profile| {
            let mut settings = EncoderSettings::new(profile, SIZE.0, SIZE.1);
            settings.allow_frame_reordering = false;
            VideoEncoder::new(display, settings).ok()
        })
        .expect("an H.264 encoder, or RVG_VA_H264");
        let mut samples = Vec::new();
        for n in 0..FRAMES {
            let bytes = noise(n);
            let row = SIZE.0 as usize;
            let (luma, chroma) = bytes.split_at(row * SIZE.1 as usize);
            let input = InputFrame {
                planes: [
                    Plane {
                        data: luma,
                        bytes_per_row: row,
                    },
                    Plane {
                        data: chroma,
                        bytes_per_row: row,
                    },
                ],
                pts: i64::try_from(n).unwrap() * 3000,
                duration: 3000,
                force_key_frame: false,
                id: n,
            };
            let coded = encoder.encode(&input).expect("a frame taken");
            samples.extend(coded.into_iter().map(|c| c.expect("a sample").data));
        }
        samples.extend(
            encoder
                .flush()
                .into_iter()
                .map(|c| c.expect("a sample").data),
        );
        Stream {
            avcc: Some(encoder.configuration().to_vec()),
            samples,
        }
    }

    pub(super) fn decoder(&self, display: &Display) -> H264Decoder {
        match &self.avcc {
            Some(avcc) => H264Decoder::new(display, avcc).expect("a decoder"),
            None => H264Decoder::new_annex_b(display),
        }
    }
}

/// Frame `n` of noise at [`SIZE`], NV12.
fn noise(n: u64) -> Vec<u8> {
    let len = SIZE.0 as usize * SIZE.1 as usize * 3 / 2;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ n.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

/// [`ROUNDS`] decoders, one after another: each decodes `stream`, its pictures read back as they
/// come, and is let go. The pictures it read back.
fn rounds(display: &Display, stream: &Stream) -> usize {
    let mut read = 0;
    for _ in 0..ROUNDS {
        let mut decoder = stream.decoder(display);
        let mut frames = Vec::new();
        for (id, sample) in (0u64..).zip(&stream.samples) {
            frames.extend(decoder.decode(sample, id).expect("a decode"));
            for frame in frames.drain(..) {
                frame.read_back().expect("a read back");
                read += 1;
            }
        }
        frames.extend(decoder.flush().expect("a flush"));
        for frame in frames.drain(..) {
            frame.read_back().expect("a read back");
            read += 1;
        }
        drop(decoder);
    }
    read
}

/// Two threads' decoders, each let go while the other decodes, leave the other's pictures and
/// the process whole: every picture of every round decodes and reads back.
#[test]
#[ignore = "needs a VA-API driver that decodes H.264 at 4K, on its GPU: run on a VA-API host"]
fn a_decoder_let_go_while_another_decodes_leaves_it_whole() {
    let display = display();
    let stream = Stream::new(&display);
    let pictures = stream.samples.len();
    assert!(pictures > 1, "a stream of {pictures} pictures");
    let read = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| read.fetch_add(rounds(&display, &stream), Ordering::Relaxed));
        }
    });
    assert_eq!(read.load(Ordering::Relaxed), 2 * ROUNDS * pictures);
}

/// The process's memory, in bytes: its commit charge (`PrivateUsage`), and what DXGI counts it
/// holding on the first adapter, in the local and the non-local segment groups.
#[cfg(windows)]
#[derive(Clone, Copy, Debug)]
struct Used {
    commit: u64,
    local: u64,
    non_local: u64,
}

#[cfg(windows)]
impl Used {
    fn now() -> Used {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let mut counters = PROCESS_MEMORY_COUNTERS_EX::default();
        let size = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS_EX>()).unwrap();
        counters.cb = size;
        // SAFETY: the process's pseudo-handle, and counters of the size given, which the call
        // writes, `PROCESS_MEMORY_COUNTERS_EX` beginning as `PROCESS_MEMORY_COUNTERS` does.
        let read = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
                size,
            )
        };
        assert_ne!(
            read,
            0,
            "K32GetProcessMemoryInfo: {}",
            std::io::Error::last_os_error()
        );
        let [local, non_local] = super::d3d12::video_memory_used().expect("DXGI's count");
        Used {
            commit: u64::try_from(counters.PrivateUsage).unwrap(),
            local,
            non_local,
        }
    }
}

/// Returns once the releases asked for on `display` so far are made (`Display::release`): a
/// release behind them says it ran.
fn releases_made(display: &Display) {
    let (made, ran) = mpsc::channel();
    display.release(move |_| {
        let _ = made.send(());
    });
    ran.recv().expect("a release made");
}

/// Decodes `samples` with `decoder`, tagged from `id`, and flushes it, its pictures read back as
/// they come. The pictures it read back.
fn decode(decoder: &mut H264Decoder, samples: &[Vec<u8>], id: &mut u64) -> usize {
    let mut read = 0;
    for sample in samples {
        for frame in decoder.decode(sample, *id).expect("a decode") {
            frame.read_back().expect("a read back");
            read += 1;
        }
        *id += 1;
    }
    for frame in decoder.flush().expect("a flush") {
        frame.read_back().expect("a read back");
        read += 1;
    }
    read
}

/// The hashes of the pictures `decoder` decodes of `samples`, tagged from 0, and flushes, each
/// read back as it comes and copied out cropped, as NV12.
fn pictures(decoder: &mut H264Decoder, samples: &[Vec<u8>]) -> Vec<u64> {
    use std::hash::{DefaultHasher, Hasher};

    let hash = |frame: &super::frame::Frame| {
        let (width, height) = (frame.width() as usize, frame.height() as usize);
        let width = width.next_multiple_of(2);
        let mut y = vec![0; width * height];
        let mut uv = vec![0; width * height.div_ceil(2)];
        frame
            .copy_nv12_to(&mut y, width, &mut uv, width)
            .expect("a copy");
        let mut hasher = DefaultHasher::new();
        hasher.write(&y);
        hasher.write(&uv);
        hasher.finish()
    };
    let mut hashes = Vec::new();
    for (id, sample) in (0u64..).zip(samples) {
        hashes.extend(
            decoder
                .decode(sample, id)
                .expect("a decode")
                .iter()
                .map(hash),
        );
    }
    hashes.extend(decoder.flush().expect("a flush").iter().map(hash));
    hashes
}

/// What each short 4K H.264 decoder let go added to DXGI's local count of the process on the
/// UHD Graphics 630 with its context held, its pool's surfaces and its config destroyed at once,
/// beside a long decoder: the mean step over this test's last nine short decoders, **measured**
/// with those teardowns so (vaon12 of Mesa 26.2.7 over Intel's Direct3D 12 video driver,
/// 2026-11-20, potluck job m3-max-61642, 651,022,336 - 336,875,520 bytes over nine). The count is
/// to grow by less than one such context over those nine.
#[cfg(windows)]
const HELD_CONTEXT_LOCAL: u64 = 34_905_201;

/// A decoder's context taken back after it decoded a whole stream starts the next stream with
/// none of the last one's references: beside a decoder that lives throughout (so that, on
/// Windows, a context let go is kept: `super::display::Display::context_let_go`), a decoder
/// decodes the whole stream, its DPB left holding the stream's last pictures, and is let go; the
/// next decoder, on Windows on the same context, decodes the stream's first access units into
/// surfaces of its own, and its pictures are the first decoder's of them, frame 0 among them.
/// **Read from code** (Mesa 26.2.4, `d3d12_video_dec_h264.cpp`, `d3d12_video_dec_hevc.cpp` and
/// `d3d12_video_dec_references_mgr.cpp`): before each picture the driver marks every DPB slot
/// unused, marks those the picture's reference list names, and releases the rest, with their
/// map from the frontend's buffers; an IDR names none.
#[test]
#[ignore = "needs a VA-API driver that decodes H.264 at 4K, on its GPU: run on a VA-API host"]
fn a_context_taken_back_starts_the_next_stream_with_none_of_the_last_ones_references() {
    let display = display();
    let stream = Stream::new(&display);
    let first = &stream.samples[..SHORT_SAMPLES.min(stream.samples.len())];
    let mut living = stream.decoder(&display);
    let reference = pictures(&mut living, first);
    assert!(
        reference.len() > 1,
        "{} pictures of {} access units",
        reference.len(),
        first.len()
    );
    let mut whole = stream.decoder(&display);
    let mut id = 0;
    let read = decode(&mut whole, &stream.samples, &mut id);
    assert!(read > first.len(), "{read} pictures of the whole stream");
    let used = whole.context_id();
    drop(whole);
    let mut next = stream.decoder(&display);
    assert_eq!(
        pictures(&mut next, first),
        reference,
        "the next decoder's pictures"
    );
    #[cfg(windows)]
    assert_eq!(next.context_id(), used, "the next decoder's context");
    #[cfg(unix)]
    let _ = used;
    drop(living);
}

/// Sets its flag when dropped: the long decoder's end, where the test passes or fails.
struct Ends<'a>(&'a AtomicBool);

impl Drop for Ends<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Short decoders made and let go one after another beside a long one, which decodes all the
/// while, decode the pictures the long one decoded of the same access units, and on Windows
/// leave DXGI's count of the process's memory where the first left it.
#[test]
#[ignore = "needs a VA-API driver that decodes H.264 at 4K, on its GPU: run on a VA-API host"]
fn short_decoders_let_go_beside_a_long_one_decode_alike_and_hold_no_more() {
    let display = display();
    let stream = Stream::new(&display);
    let short = &stream.samples[..SHORT_SAMPLES.min(stream.samples.len())];
    let ended = AtomicBool::new(false);
    let (decoding, first) = mpsc::channel();
    std::thread::scope(|s| {
        let long = s.spawn(|| {
            let mut decoder = stream.decoder(&display);
            decoding.send(pictures(&mut decoder, short)).unwrap();
            let mut id = u64::try_from(short.len()).unwrap();
            let mut passes = 0;
            while !ended.load(Ordering::Relaxed) {
                let read = decode(&mut decoder, &stream.samples, &mut id);
                assert!(read > 0, "the long decoder's pass {passes} read nothing");
                passes += 1;
            }
            passes
        });
        let end = Ends(&ended);
        let reference = first.recv().expect("the long decoder's first pictures");
        assert!(
            reference.len() > 1,
            "{} pictures of {} access units",
            reference.len(),
            short.len()
        );
        #[cfg(windows)]
        let mut readings = vec![Used::now()];
        #[cfg(windows)]
        let mut let_go = None;
        // Each short decoder's pictures that are not the long one's, by their place: told after
        // the memory, so that a run says both.
        let mut wrong = Vec::new();
        for n in 0..SHORT {
            let mut decoder = stream.decoder(&display);
            let got = pictures(&mut decoder, short);
            let differ: Vec<usize> = (0..reference.len().max(got.len()))
                .filter(|&at| got.get(at) != reference.get(at))
                .collect();
            eprintln!("short decoder {n}: pictures not the long one's at {differ:?}");
            if !differ.is_empty() {
                wrong.push((n, differ));
            }
            #[cfg(windows)]
            {
                // Each takes back the context the one before it let go.
                if n > 0 {
                    assert_eq!(decoder.context_id(), let_go, "short decoder {n}'s context");
                }
                let_go = decoder.context_id();
            }
            #[cfg(windows)]
            let live = Used::now();
            drop(decoder);
            releases_made(&display);
            #[cfg(windows)]
            {
                let after = Used::now();
                eprintln!("short decoder {n}: live {live:?}, let go {after:?}");
                readings.push(after);
            }
        }
        drop(end);
        let passes = long.join().unwrap();
        eprintln!("the long decoder's passes of the stream beside the short ones: {passes}");
        #[cfg(windows)]
        {
            releases_made(&display);
            eprintln!("every decoder let go: {:?}", Used::now());
            let grown = readings[SHORT].local.saturating_sub(readings[1].local);
            assert!(
                grown < HELD_CONTEXT_LOCAL,
                "DXGI's local count grew {grown} bytes over the short decoders after the first, \
                 a held context's {HELD_CONTEXT_LOCAL}"
            );
        }
        assert!(
            wrong.is_empty(),
            "the short decoders' pictures not the long one's, by decoder and place: {wrong:?}"
        );
    });
}

/// The pictures decoded in
/// [`a_display_let_go_after_its_decodes_leaves_no_reference_on_the_adapters_device`].
#[cfg(windows)]
const LET_GO_PICTURES: usize = 16;

/// A display let go of lets go of the adapter's Direct3D 12 device, and of everything vaon12 made
/// on it: once a decoder decoded the first pictures of the stream into its pool, each read back as
/// a CPU copy does, and the pictures, the decoder and the display are let go of, the Direct3D 12
/// device made on the adapter (one a process, which `D3D12CreateDevice` answers while it lives)
/// has no reference but ours, within 5 s of the display's drop (its `vaTerminate` may be made on
/// its call thread). Run holding the GPU, on vaon12 on the default adapter.
///
/// **Measured** to fail on vaon12 of Mesa 26.2.7 over an Intel UHD Graphics 630, 2026-11-20: one
/// reference beside ours, as the driver leaves one of every decode session (docs/video/bugs.md,
/// U31).
#[cfg(windows)]
#[test]
#[ignore = "vaon12 of Mesa 26.2.7 fails it (docs/video/bugs.md, U31); needs RVG_VA_H264: run on \
            the Windows host"]
fn a_display_let_go_after_its_decodes_leaves_no_reference_on_the_adapters_device() {
    use super::d3d12::{
        default_adapter_device_references, default_adapter_device_references_once_let_go,
    };
    assert_eq!(
        default_adapter_device_references(),
        0,
        "the adapter's Direct3D 12 device, made alone, has no reference but ours"
    );
    let display = display();
    let stream = Stream::new(&display);
    let mut decoder = stream.decoder(&display);
    let mut frames = Vec::new();
    for (id, sample) in (0u64..).zip(stream.samples.iter().take(LET_GO_PICTURES)) {
        frames.extend(decoder.decode(sample, id).expect("a decode"));
    }
    frames.extend(decoder.flush().expect("the flush"));
    assert!(!frames.is_empty(), "pictures decoded");
    for frame in &frames {
        frame.read_back().expect("a read back");
    }
    drop(frames);
    drop(decoder);
    drop(display);
    assert_eq!(
        default_adapter_device_references_once_let_go(),
        0,
        "the adapter's Direct3D 12 device, made once the display is let go of, has no reference \
         but ours"
    );
}
