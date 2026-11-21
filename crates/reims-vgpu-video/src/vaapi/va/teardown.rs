//! Decoders let go while others decode on the same display: two threads, each making an H.264
//! decoder, decoding a stream with it, reading each picture back as a CPU copy does, and letting
//! it go (its context, its pool and its config), round after round, so that each decoder's
//! teardown is asked for while the other decodes, as a guest's session ends while its next one
//! decodes (`super::display::Display::release_teardown`). Run on a VA-API host, holding its GPU:
//! on Windows on the adapter libva-win32 opens first, elsewhere on the render node
//! `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is unset). The stream is the Annex B
//! file `RVG_VA_H264` names, or else 4K noise the driver's own encoder makes; vaon12 on the UHD
//! Graphics 630 has no H.264 encoder, so there `RVG_VA_H264` names a stream.

use std::sync::atomic::{AtomicUsize, Ordering};

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
#[cfg(windows)]
#[test]
#[ignore = "needs vaon12 and RVG_VA_H264: run on the Windows host"]
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
