//! VP9's superframes, headers and decoding process under seeded fuzzing, up
//! to an accelerator that checks what the hardware is handed. Two kinds of
//! input: streams of the tests' own frames with random fields, some
//! straying from what the specification allows, in superframes and alone,
//! mutated by sample; and libvpx's streams, with samples mutated among
//! whole ones, so that the mutations meet the state real frames leave.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::codec::fuzz_support::{Rng, Target, long, maybe_mutate, quick};

const STREAMS: Target = Target::new("vp9-streams");
const SAMPLES: Target = Target::new("vp9-samples");

/// Frames that reached the accelerator, which says how deep the inputs get.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[test]
fn streams() {
    let report = quick(&STREAMS, 3000, stream_case);
    eprintln!(
        "vp9: {} frames reached the accelerator in {} streams",
        FRAMES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn streams_long() {
    long(&STREAMS, stream_case);
}

#[test]
fn samples() {
    quick(&SAMPLES, 600, sample_case);
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn samples_long() {
    long(&SAMPLES, sample_case);
}

/// An accelerator that checks what the decoder gives it against what the
/// hardware takes: a configuration of a format the decoder supports, each
/// frame of that format, its headers within its bytes, references of its
/// format at scales the hardware scales, and derived values in range. It
/// fails one call in 64, as a driver may.
#[derive(Default)]
struct Checking {
    next: u32,
    configured: Option<Format>,
    failing: u64,
}

impl Checking {
    fn fails(&mut self) -> bool {
        self.failing = self
            .failing
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.failing >> 58 == 0
    }
}

impl Accelerator for Checking {
    /// The picture's number and the format it was made for.
    type Picture = (u32, Format);
    type Error = Error;

    fn configure(&mut self, format: &Format) -> Result<(), Error> {
        assert!(matches!(
            (format.profile, format.bit_depth),
            (0, 8) | (2, 10)
        ));
        assert!((1..=MAX_SIDE).contains(&format.width));
        assert!((1..=MAX_SIDE).contains(&format.height));
        self.configured = None;
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's configuration failure"));
        }
        self.configured = Some(*format);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<(u32, Format), Error> {
        let format = self.configured.expect("a picture before a configuration");
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's picture failure"));
        }
        self.next += 1;
        Ok((self.next, format))
    }

    fn decode_frame(
        &mut self,
        picture: &(u32, Format),
        params: &FrameParams<'_, (u32, Format)>,
    ) -> Result<(), Error> {
        let format = self.configured.expect("a frame before a configuration");
        assert_eq!(picture.1, format, "a picture of another format");
        let header = params.header;
        assert!(header.show_existing_frame.is_none());
        assert_eq!((header.width, header.height), (format.width, format.height));
        assert!(header.header_size_in_bytes > 0);
        assert!(
            usize::from(header.uncompressed_header_size) + usize::from(header.header_size_in_bytes)
                <= params.frame.len(),
            "a compressed header past the frame"
        );
        assert!(header.tile_rows_log2 <= 2 && header.tile_cols_log2 <= 6);
        assert!(header.loop_filter.level <= 63 && header.loop_filter.sharpness <= 7);
        for segment in params.segments {
            assert!(segment.filter_level.iter().flatten().all(|&l| l <= 63));
            for scale in [
                segment.luma_dc,
                segment.luma_ac,
                segment.chroma_dc,
                segment.chroma_ac,
            ] {
                assert!(scale > 0, "a quantizer scale of {scale}");
            }
        }
        if !header.is_intra() {
            for &index in &header.ref_frame_idx {
                let reference = params.references[usize::from(index)]
                    .as_ref()
                    .expect("an inter frame's reference is in its slot");
                assert_eq!(reference.picture.1.bit_depth, format.bit_depth);
                for (frame, of) in [
                    (format.width, reference.width),
                    (format.height, reference.height),
                ] {
                    assert!(2 * u64::from(frame) >= u64::from(of));
                    assert!(u64::from(frame) <= 16 * u64::from(of));
                }
            }
        }
        FRAMES.fetch_add(1, Ordering::Relaxed);
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's decode failure"));
        }
        Ok(())
    }
}

/// A side near `base`: most often `base`, else half or twice it, else
/// anything VP9 codes.
fn side(rng: &mut Rng, base: u32) -> u32 {
    match rng.below(8) {
        0 => (base / 2).max(1),
        1 => (base * 2).min(65536),
        2 => rng.range_u32(1, 65536),
        3 => *rng.pick(&[1, 8, 64, MAX_SIDE, MAX_SIDE + 1, 65536]),
        _ => base,
    }
}

fn some_delta(rng: &mut Rng, magnitude: u32) -> Option<i32> {
    rng.bool()
        .then(|| rng.range(0, u64::from(2 * magnitude + 1)) as i32 - magnitude as i32)
}

/// A frame's header with every field drawn: most often an inter frame of
/// the stream's size, now and then a key frame, an intra-only frame or a
/// frame that shows a slot, and in `stray` frames fields past their
/// ranges.
fn frame_spec(rng: &mut Rng, profile: u32, size: (u32, u32), show: bool) -> FrameSpec {
    let stray = rng.one_in(8);
    let mut spec = if rng.one_in(6) {
        FrameSpec::key(size.0, size.1)
    } else {
        FrameSpec::inter(size.0, size.1)
    };
    spec.profile = if stray { rng.below(4) as u32 } else { profile };
    spec.show = show;
    if rng.one_in(12) {
        spec.show_existing = Some(rng.below(8) as u32);
        return spec;
    }
    spec.error_resilient = rng.one_in(8);
    spec.intra_only = !spec.key && !show && rng.one_in(3);
    spec.reset_frame_context = rng.below(4) as u32;
    spec.twelve_bit = stray && rng.bool();
    spec.color_space = if stray { rng.below(8) as u32 } else { 2 };
    spec.full_range = rng.bool();
    spec.subsampling = (!stray || rng.bool(), !stray || rng.bool());
    if !spec.key {
        spec.refresh = rng.below(256) as u32;
        spec.refs = [0, 1, 2].map(|_| rng.below(8) as u32);
        spec.sign_bias = [rng.bool(), rng.bool(), rng.bool()];
        spec.size_from_ref = (!rng.one_in(4)).then(|| rng.below(3));
    }
    if spec.key || spec.intra_only || spec.size_from_ref.is_none() || stray {
        spec.size = (side(rng, size.0), side(rng, size.1));
    }
    spec.render = rng
        .one_in(8)
        .then(|| (rng.range_u32(1, 65536), rng.range_u32(1, 65536)));
    spec.high_precision_mv = rng.bool();
    spec.filter = (!rng.one_in(3)).then(|| rng.below(4) as u32);
    spec.refresh_context = rng.bool();
    spec.parallel = rng.bool();
    spec.context_idx = rng.below(4) as u32;
    spec.filter_level = rng.below(64) as u32;
    spec.sharpness = rng.below(8) as u32;
    spec.deltas = (!rng.one_in(4)).then(|| {
        (
            [0; 4].map(|_| some_delta(rng, 63)),
            [0; 2].map(|_| some_delta(rng, 63)),
        )
    });
    spec.base_q = rng.below(256) as u32;
    spec.delta_q = [0; 3].map(|_| some_delta(rng, 15).unwrap_or(0));
    if rng.one_in(3) {
        let prob = |rng: &mut Rng| rng.bool().then(|| rng.below(256) as u32);
        spec.segmentation = Some(SegmentationSpec {
            update_map: rng.bool().then(|| {
                (
                    [0; 7].map(|_| prob(rng)),
                    rng.bool().then(|| [0; 3].map(|_| prob(rng))),
                )
            }),
            update_data: rng.bool().then(|| {
                let features = [0; 8].map(|_| {
                    [255, 63, 3, 0].map(|max| {
                        some_delta(rng, max)
                            .map(i32::abs)
                            .map(|v| if max > 3 && rng.bool() { -v } else { v })
                    })
                });
                (rng.bool(), features)
            }),
        });
    }
    spec.tile_cols_increments = rng.below(if stray { 8 } else { 3 }) as u32;
    spec.tile_rows = rng.below(3) as u32;
    spec.compressed = if stray {
        *rng.pick(&[0, 1, 65535])
    } else {
        rng.range_u32(1, 64)
    };
    spec.tile_bytes = rng.below(64);
    spec
}

/// A stream of the tests' own frames: a key frame most often first, then
/// samples of one frame, or of hidden frames and a shown one in a
/// superframe, each mutated now and then.
fn stream_case(rng: &mut Rng) {
    let profile = *rng.pick(&[0, 0, 2]);
    let size = (side(rng, 64), side(rng, 48));
    let size = (size.0.min(4096), size.1.min(4096));
    let mut decoder = Decoder::new(Checking::default());
    let donor = ivf_frames(PROFILE_0_IVF)[1].to_vec();
    let samples = 1 + rng.count(12);
    for id in 0..samples as u64 {
        let hidden = if rng.one_in(4) { 1 + rng.below(3) } else { 0 };
        let mut frames: Vec<Vec<u8>> = (0..hidden)
            .map(|_| frame_spec(rng, profile, size, false).write())
            .collect();
        let mut last = frame_spec(rng, profile, size, true);
        if id == 0 && !rng.one_in(8) {
            last = FrameSpec {
                key: true,
                refresh: 0xff,
                show_existing: None,
                intra_only: false,
                ..last
            };
        }
        frames.push(last.write());
        let mut sample = if frames.len() > 1 {
            superframe(&frames, 1 + rng.below(4))
        } else {
            frames.concat()
        };
        maybe_mutate(rng, &mut sample, &donor, 3);
        match decoder.decode(&sample, id) {
            Ok(out) => assert!(out.len() <= superframe::MAX_FRAMES),
            Err(error) => {
                let _ = error.to_string();
            }
        }
        if rng.one_in(16) {
            decoder.reset();
        }
    }
}

/// libvpx's streams, a few samples mutated, dropped or repeated.
fn sample_case(rng: &mut Rng) {
    let file = if rng.bool() {
        PROFILE_0_IVF
    } else {
        PROFILE_2_IVF
    };
    let samples = ivf_frames(file);
    let mut decoder = Decoder::new(Checking::default());
    for (id, whole) in (0u64..).zip(&samples) {
        let mut sample = whole.to_vec();
        if rng.one_in(4) {
            let donor = rng.pick(&samples);
            maybe_mutate(rng, &mut sample, donor, 8);
        }
        if rng.one_in(16) {
            continue;
        }
        let repeats = if rng.one_in(16) { 2 } else { 1 };
        for _ in 0..repeats {
            if let Ok(out) = decoder.decode(&sample, id) {
                assert!(out.len() <= superframe::MAX_FRAMES);
            }
        }
    }
}
