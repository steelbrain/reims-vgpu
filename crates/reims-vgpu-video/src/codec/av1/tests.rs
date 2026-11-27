//! The decoding process end to end on our own streams, through an accelerator that records
//! what it is asked to decode. Nothing here decodes samples: the streams' frames are checked
//! against their reference MD5s on hardware, by the VA-API and NVIDIA modules' ignored tests.
//!
//! The streams in `tests/streams` were made once from ffmpeg's synthetic sources (`testsrc2`,
//! `mandelbrot`) by aomenc 3.15.3, SvtAv1EncApp 4.2.0 and rav1e 0.8.1, and each `.md5` file is
//! each shown frame's MD5 as ffmpeg's `framemd5` hashes it, NV12 or P010, from ffmpeg n9.0.2's
//! libdav1d 1.5.4 decode: with film grain applied (the specification's output), and, in
//! `.nograin.md5`, without it. `aom-still` is an AVIF still's item, a reduced still picture
//! header and one key frame, as ffmpeg's libaom encoder made it (`-still-picture 1`) and its
//! demux copied it out of the AVIF, byte for byte; its MD5 is libdav1d's decode of the AVIF.

#[cfg(unix)]
mod card;
mod fuzz;

#[cfg(unix)]
pub(crate) use card::{
    CardFrame, NO_GRAIN, ReadFrame, WRONG_GRAIN, card_streams, grain_refused, judge,
};

use super::frame::{FrameHeader, FrameType};
use super::sequence::SequenceHeader;
use super::tile_group::TileGroup;
use super::*;

/// One of our test streams.
pub(crate) struct Stream {
    pub(crate) name: &'static str,
    /// The stream in IVF.
    pub(crate) ivf: &'static [u8],
    /// Each shown frame's MD5, a line each, grain applied.
    pub(crate) md5: &'static str,
    /// Likewise without film grain, where the stream applies it.
    pub(crate) md5_without_grain: Option<&'static str>,
    /// Its film grain residual's statistics, where it applies grain: its `.grain` file, from
    /// libdav1d's decode with grain and without (`ReadFrame`, `Residual`).
    #[cfg_attr(not(unix), expect(dead_code, reason = "read by the card tests alone"))]
    pub(crate) grain: Option<&'static str>,
    /// `UpscaledWidth` and `FrameHeight` of its frames.
    pub(crate) size: (u32, u32),
    pub(crate) bit_depth: u8,
    /// Whether this decoder takes it: Main profile at 8 or 10 bits.
    pub(crate) decodes: bool,
}

macro_rules! stream {
    ($name:literal, $size:expr, $depth:expr, $decodes:expr) => {
        Stream {
            name: $name,
            ivf: include_bytes!(concat!("tests/streams/", $name, ".ivf")),
            md5: include_str!(concat!("tests/streams/", $name, ".md5")),
            md5_without_grain: None,
            grain: None,
            size: $size,
            bit_depth: $depth,
            decodes: $decodes,
        }
    };
}

pub(crate) const STREAMS: &[Stream] = &[
    stream!("svt-main8", (352, 288), 8, true),
    stream!("svt-main10", (352, 288), 10, true),
    stream!("aom-tiles", (640, 360), 8, true),
    Stream {
        md5_without_grain: Some(include_str!("tests/streams/aom-grain.nograin.md5")),
        grain: Some(include_str!("tests/streams/aom-grain.grain")),
        ..stream!("aom-grain", (352, 288), 8, true)
    },
    stream!("aom-superres", (352, 288), 8, true),
    stream!("aom-screen", (352, 288), 8, true),
    stream!("aom-odd", (98, 62), 8, true),
    stream!("rav1e-tiles", (320, 240), 8, true),
    stream!("aom-still", (640, 360), 8, true),
    stream!("aom-444", (64, 64), 8, false),
    stream!("aom-main12", (64, 64), 12, false),
];

/// The stream named `name`.
pub(crate) fn stream(name: &str) -> &'static Stream {
    STREAMS
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no stream {name}"))
}

/// The frames of an IVF file: each a temporal unit.
pub(crate) fn samples(ivf: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&ivf[..4], b"DKIF");
    let header = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut at = header;
    let mut samples = Vec::new();
    while at + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
        samples.push(&ivf[at + 12..at + 12 + size]);
        at += 12 + size;
    }
    samples
}

/// A sample as an MP4 track carries it: its temporal delimiters left out (AV1-ISOBMFF 2.4).
pub(crate) fn without_temporal_delimiters(sample: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = sample;
    while !rest.is_empty() {
        let has_extension = rest[0] & 4 != 0;
        let header = 1 + usize::from(has_extension);
        let (size, taken) = super::bits::leb128(&rest[header..]).unwrap();
        let end = header + taken + size as usize;
        if (rest[0] >> 3) & 0xf != obu::TEMPORAL_DELIMITER {
            out.extend_from_slice(&rest[..end]);
        }
        rest = &rest[end..];
    }
    out
}

/// The frames' MD5s of a reference file.
pub(crate) fn md5s(text: &str) -> Vec<&str> {
    text.lines().filter(|l| !l.is_empty()).collect()
}

/// Limits past every test stream.
pub(crate) const LIMITS: Limits = Limits::up_to(4096, 2304);

/// What the accelerator was asked to decode of one frame.
pub(crate) struct Recorded {
    pub(crate) header: FrameHeader,
    pub(crate) tile_groups: usize,
    pub(crate) tiles: usize,
}

/// An accelerator that records each frame and checks what the hardware's structures take:
/// a frame within the limits, at most 64 tile columns and rows, its tile groups in order and
/// whole, each tile within its group's data, and each slot a frame predicts from held.
pub(crate) struct Recorder {
    pub(crate) limits: Limits,
    pub(crate) configured: Vec<SequenceHeader>,
    pub(crate) frames: Vec<Recorded>,
    next: u32,
}

impl Recorder {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            limits,
            configured: Vec::new(),
            frames: Vec::new(),
            next: 0,
        }
    }
}

/// The checks [`Recorder`] makes, which the fuzzers' accelerator makes too.
pub(crate) fn check_frame(
    limits: &Limits,
    params: &FrameParams<'_, u32>,
    tile_groups: &[TileGroup<'_>],
) -> usize {
    let header = params.header;
    let size = header.size;
    assert!(size.upscaled_width <= limits.max_width && size.frame_height <= limits.max_height);
    assert!(size.upscaled_width <= params.sequence.max_frame_width);
    assert!(size.frame_height <= params.sequence.max_frame_height);
    assert!(size.frame_width >= 1 && size.frame_width <= size.upscaled_width);
    let info = &header.tile_info;
    assert!(info.cols() >= 1 && info.cols() <= 64 && info.rows() >= 1 && info.rows() <= 64);
    let mut next = 0;
    let mut tiles = 0;
    for group in tile_groups {
        assert_eq!(group.start, next, "tile groups out of order");
        assert_eq!(group.tiles.len() as u32, group.end - group.start + 1);
        for tile in &group.tiles {
            assert!(tile.size > 0);
            assert!((tile.offset + tile.size) as usize <= group.data.len());
            assert!(tile.row < info.rows() && tile.column < info.cols());
        }
        tiles += group.tiles.len();
        next = group.end + 1;
    }
    assert_eq!(next, info.num_tiles(), "tiles missing");
    if !header.frame_type.is_intra() {
        for &index in &header.ref_frame_idx {
            assert!(
                params.slots[index as usize].is_some(),
                "a reference slot not held"
            );
        }
    }
    let grain = &header.film_grain;
    assert!(grain.num_y_points <= 14 && grain.num_cb_points <= 10 && grain.num_cr_points <= 10);
    tiles
}

impl Accelerator for Recorder {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sequence: &SequenceHeader) -> Result<(), Error> {
        self.configured.push(sequence.clone());
        Ok(())
    }

    fn new_picture(&mut self, _header: &FrameHeader) -> Result<u32, Error> {
        self.next += 1;
        Ok(self.next)
    }

    fn decode_frame(
        &mut self,
        picture: &u32,
        params: &FrameParams<'_, u32>,
        tile_groups: &[TileGroup<'_>],
    ) -> Result<(), Error> {
        assert!(
            !self.configured.is_empty(),
            "a frame before a configuration"
        );
        let tiles = check_frame(&self.limits, params, tile_groups);
        assert_ne!(*picture, 0);
        self.frames.push(Recorded {
            header: params.header.clone(),
            tile_groups: tile_groups.len(),
            tiles,
        });
        Ok(())
    }
}

/// A stream's decoder after its samples, the frames it output, and each sample's error.
type Decoded = (Decoder<Recorder>, Vec<Output<u32>>, Vec<(usize, Error)>);

/// Decodes every sample of `stream`, each as MP4 carries it: the outputs, and each sample's
/// error.
fn decode(stream: &Stream) -> Decoded {
    let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
    let mut outputs = Vec::new();
    let mut errors = Vec::new();
    for (n, sample) in samples(stream.ivf).into_iter().enumerate() {
        match decoder.decode(&without_temporal_delimiters(sample), n as u64) {
            Ok(shown) => outputs.extend(shown),
            Err(e) => errors.push((n, e)),
        }
    }
    (decoder, outputs, errors)
}

#[test]
fn each_stream_shows_a_frame_a_sample_as_its_reference_does() {
    for stream in STREAMS.iter().filter(|s| s.decodes) {
        let (decoder, outputs, errors) = decode(stream);
        assert!(errors.is_empty(), "{}: {errors:?}", stream.name);
        let samples = samples(stream.ivf).len();
        assert_eq!(outputs.len(), md5s(stream.md5).len(), "{}", stream.name);
        if let Some(without) = stream.md5_without_grain {
            assert_eq!(md5s(without).len(), outputs.len(), "{}", stream.name);
            assert_ne!(md5s(without), md5s(stream.md5), "{}", stream.name);
        }
        assert_eq!(
            outputs.len(),
            samples,
            "{}: one shown frame a sample",
            stream.name
        );
        for (n, output) in outputs.iter().enumerate() {
            assert_eq!(output.id, n as u64, "{}", stream.name);
            assert_eq!(output.size, stream.size, "{}", stream.name);
            assert_eq!(output.bit_depth, stream.bit_depth, "{}", stream.name);
            assert!(!output.full_range, "{}", stream.name);
        }
        assert_eq!(decoder.accelerator().configured.len(), 1, "{}", stream.name);
        let frames = &decoder.accelerator().frames;
        assert!(frames.len() >= samples, "{}", stream.name);
        eprintln!(
            "{}: {} samples, {} frames decoded, {} shown again",
            stream.name,
            samples,
            frames.len(),
            outputs.iter().filter(|o| o.shown_existing).count()
        );
    }
}

#[test]
fn the_streams_reach_the_tools_they_were_made_for() {
    let frames = |name| {
        decode(stream(name))
            .0
            .accelerator()
            .frames
            .iter()
            .map(|f| (f.header.clone(), f.tile_groups, f.tiles))
            .collect::<Vec<_>>()
    };
    // SVT-AV1's random access: hidden alternate references shown later.
    let (_, outputs, _) = decode(stream("svt-main8"));
    assert!(outputs.iter().any(|o| o.shown_existing));
    assert!(
        frames("svt-main8")
            .iter()
            .any(|(h, ..)| !h.show_frame && h.showable_frame)
    );
    // Tiles in more than one group, and tile sizes.
    let tiles = frames("aom-tiles");
    assert!(tiles.iter().all(|(_, _, tiles)| *tiles == 4));
    assert!(tiles.iter().any(|(_, groups, _)| *groups > 1));
    assert!(frames("rav1e-tiles").iter().all(|(_, _, tiles)| *tiles > 1));
    // Film grain, coded and loaded from a reference.
    let grain = frames("aom-grain");
    assert!(
        grain
            .iter()
            .any(|(h, ..)| h.film_grain.apply_grain && h.film_grain.update_grain)
    );
    let (_, outputs, _) = decode(stream("aom-grain"));
    assert!(outputs.iter().any(|o| o.film_grain));
    // Superres: coded narrower than shown.
    assert!(
        frames("aom-superres")
            .iter()
            .any(|(h, ..)| h.size.use_superres && h.size.frame_width < h.size.upscaled_width)
    );
    // Screen content tools.
    assert!(
        frames("aom-screen")
            .iter()
            .any(|(h, ..)| h.allow_screen_content_tools)
    );
    // Error resilience.
    assert!(
        frames("aom-odd")
            .iter()
            .all(|(h, ..)| h.error_resilient_mode)
    );
    // An AVIF still's item: a reduced still picture header, and one key frame.
    let (decoder, outputs, _) = decode(stream("aom-still"));
    let configured = &decoder.accelerator().configured;
    assert!(
        configured
            .iter()
            .all(|s| s.still_picture && s.reduced_still_picture_header)
    );
    assert!(!configured.is_empty());
    assert_eq!(outputs.len(), 1);
}

#[test]
fn profiles_other_than_main_are_refused_at_their_sequence_header() {
    for name in ["aom-444", "aom-main12"] {
        let first = samples(stream(name).ivf)[0];
        let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
        let error = decoder.decode(first, 0).err();
        assert_eq!(
            error,
            Some(Error::Unsupported(
                "a profile other than Main (4:4:4 or 12 bits)"
            )),
            "{name}"
        );
        assert!(!error.is_some_and(|e| e.is_picture_error()));
        assert!(decoder.accelerator().frames.is_empty());
    }
}

#[test]
fn a_sequence_outside_the_limits_is_refused_before_any_frame() {
    let first = samples(stream("svt-main8").ivf)[0];
    let refused = [
        (
            Limits::up_to(352, 287),
            "frames larger than the hardware decodes",
        ),
        (
            Limits::up_to(351, 288),
            "frames larger than the hardware decodes",
        ),
        (
            Limits {
                min_width: 353,
                ..Limits::up_to(4096, 2304)
            },
            "frames smaller than the hardware decodes",
        ),
        (
            Limits {
                min_height: 289,
                ..Limits::up_to(4096, 2304)
            },
            "frames smaller than the hardware decodes",
        ),
    ];
    for (limits, refusal) in refused {
        let mut decoder = Decoder::new(Recorder::new(limits), limits);
        assert_eq!(
            decoder.decode(first, 0).err(),
            Some(Error::Unsupported(refusal)),
            "{limits:?}"
        );
        assert!(decoder.accelerator().frames.is_empty());
    }
    // At the limits themselves it decodes.
    let limits = Limits {
        max_width: 352,
        max_height: 288,
        min_width: 352,
        min_height: 288,
    };
    let mut decoder = Decoder::new(Recorder::new(limits), limits);
    assert_eq!(decoder.decode(first, 0).map(|o| o.len()), Ok(1));
}

#[test]
fn a_record_carries_the_sequence_header_the_first_sample_may_leave_out() {
    let stream = stream("aom-odd");
    let samples = samples(stream.ivf);
    // The sequence header OBU of the first sample, as a record's configOBUs.
    let first = without_temporal_delimiters(samples[0]);
    let size = first[1] as usize;
    let mut record = vec![0x81, 0x00, 0x0c, 0x00];
    record.extend_from_slice(&first[..2 + size]);
    let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
    decoder
        .configure(&DecoderConfiguration::parse(&record).unwrap())
        .unwrap();
    assert!(decoder.sequence().is_some());
    assert_eq!(
        decoder.decode(&first[2 + size..], 0).map(|o| o.len()),
        Ok(1)
    );
    for (n, sample) in samples.iter().enumerate().skip(1) {
        assert_eq!(decoder.decode(sample, n as u64).map(|o| o.len()), Ok(1));
    }
}

#[test]
fn decoding_started_after_a_key_frame_resumes_at_the_next_key_frame() {
    let samples = samples(stream("svt-main8").ivf);
    let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
    assert!(decoder.decode(samples[0], 0).is_ok());
    decoder.reset();
    for (n, sample) in samples.iter().enumerate().skip(1) {
        let result = decoder.decode(sample, n as u64);
        if n < 16 {
            assert_eq!(result.err(), Some(Error::MissingReference), "sample {n}");
        } else {
            let outputs = result.unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].frame_type, FrameType::Key);
        }
    }
}

#[test]
fn a_sample_cut_inside_its_first_frame_empties_every_slot() {
    let samples: Vec<_> = samples(stream("svt-main8").ivf)
        .into_iter()
        .map(without_temporal_delimiters)
        .collect();
    let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
    assert_eq!(decoder.decode(&samples[0], 0).map(|o| o.len()), Ok(1));
    assert_eq!(
        decoder.decode(&samples[1][..8], 1).err(),
        Some(Error::ObuPastEnd)
    );
    // Its frames predict from the key frame, whose slots are gone.
    assert_eq!(
        decoder.decode(&samples[1], 1).err(),
        Some(Error::MissingReference)
    );
}

/// Every truncation of every sample, after the samples before it whole, is refused or decoded,
/// never a panic, and nothing past a tile's bytes reaches the accelerator.
#[test]
fn every_truncation_of_every_sample_is_refused_or_decoded() {
    let mut cuts = 0;
    for stream in STREAMS {
        let samples: Vec<_> = samples(stream.ivf)
            .into_iter()
            .map(without_temporal_delimiters)
            .collect();
        for (n, sample) in samples.iter().enumerate() {
            let lengths: Vec<usize> = if sample.len() <= 96 {
                (0..sample.len()).collect()
            } else {
                (0..64)
                    .chain((64..sample.len()).step_by(sample.len() / 32 + 1))
                    .chain(sample.len() - 4..sample.len())
                    .collect()
            };
            for length in lengths {
                let mut decoder = Decoder::new(Recorder::new(LIMITS), LIMITS);
                for (m, before) in samples[..n].iter().enumerate() {
                    let _ = decoder.decode(before, m as u64);
                }
                let _ = decoder.decode(&sample[..length], n as u64);
                // And the next sample after it.
                if let Some(next) = samples.get(n + 1) {
                    let _ = decoder.decode(next, n as u64 + 1);
                }
                cuts += 1;
            }
        }
    }
    eprintln!("{cuts} truncations");
}

#[test]
fn set_frame_refs_fills_the_references_short_signaling_leaves_out() {
    let seq = SequenceHeader {
        enable_order_hint: true,
        order_hint_bits: 7,
        ..SequenceHeader::default()
    };
    // The current frame is at 8; slots hold 0, 4, 6, 7, 12, 16, 10, 2.
    let hints = [0, 4, 6, 7, 12, 16, 10, 2];
    let refs = frame::set_frame_refs(&seq, &hints, 8, 3, 1).unwrap();
    // LAST = slot 3 (7), GOLDEN = slot 1 (4) as coded; ALTREF the latest backward (slot 5,
    // 16), BWDREF the earliest backward (slot 6, 10), ALTREF2 the next (slot 4, 12); LAST2 and
    // LAST3 the latest forward left (slot 2, 6, then slot 7, 2).
    assert_eq!(refs, [3, 2, 7, 1, 6, 4, 5]);
    // LAST must be before the current frame.
    assert_eq!(
        frame::set_frame_refs(&seq, &hints, 8, 4, 1),
        Err(Error::OutOfRange("last_frame_idx"))
    );
}

#[test]
fn relative_distances_wrap_at_the_order_hint_bits() {
    let seq = SequenceHeader {
        enable_order_hint: true,
        order_hint_bits: 4,
        ..SequenceHeader::default()
    };
    assert_eq!(frame::relative_dist(&seq, 1, 15), 2);
    assert_eq!(frame::relative_dist(&seq, 15, 1), -2);
    assert_eq!(frame::relative_dist(&seq, 3, 3), 0);
    let off = SequenceHeader::default();
    assert_eq!(frame::relative_dist(&off, 1, 15), 0);
}

#[test]
fn the_shear_check_takes_small_warps_and_refuses_large_ones() {
    let one = 1 << 16;
    assert!(frame::warp_valid(&[0, 0, one, 0, 0, one]));
    assert!(frame::warp_valid(&[5, -3, one + 500, 300, -300, one + 500]));
    // An alpha of a quarter: 4 × 2^14 reaches the bound.
    assert!(!frame::warp_valid(&[0, 0, one + (1 << 14), 0, 0, one]));
    assert!(!frame::warp_valid(&[0, 0, 0, 0, 0, one]));
    assert!(!frame::warp_valid(&[0, 0, -one, 0, 0, one]));
}
