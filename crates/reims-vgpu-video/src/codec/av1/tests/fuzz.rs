//! AV1's parsers and decoding process under seeded fuzzing, as the host meets bytes the
//! guest wrote: our streams' samples, `av1C` records and sequence headers, mutated by byte,
//! by OBU and by sample, through an accelerator that checks what the hardware's structures
//! take ([`check_frame`]) and fails one call in 64, as a driver may. No input panics, and
//! none reaches the accelerator with a frame past the decoder's limits or a tile past its
//! bytes.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::codec::fuzz_support::{Rng, Target, long, maybe_mutate, mutate_rounds, quick};

const STREAMS_TARGET: Target = Target::new("av1-streams");
const RECORDS_TARGET: Target = Target::new("av1-records");

/// Frames that reached the accelerator, which says how deep the inputs get.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[test]
fn streams() {
    let report = quick(&STREAMS_TARGET, 3000, stream_case);
    eprintln!(
        "av1: {} frames reached the accelerator in {} streams",
        FRAMES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn streams_long() {
    let report = long(&STREAMS_TARGET, stream_case);
    eprintln!(
        "av1: {} frames reached the accelerator in {} streams",
        FRAMES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
fn records() {
    quick(&RECORDS_TARGET, 3000, record_case);
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn records_long() {
    long(&RECORDS_TARGET, record_case);
}

/// The fuzzers' accelerator: [`check_frame`]'s checks, and a failure one call in 64.
struct Checking {
    limits: Limits,
    configured: bool,
    next: u32,
    failing: u64,
}

impl Checking {
    fn new(limits: Limits, seed: u64) -> Self {
        Self {
            limits,
            configured: false,
            next: 0,
            failing: seed,
        }
    }

    fn fails(&mut self) -> bool {
        self.failing = self
            .failing
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.failing >> 58 == 0
    }
}

impl Accelerator for Checking {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sequence: &SequenceHeader) -> Result<(), Error> {
        assert!(sequence.max_frame_width <= self.limits.max_width);
        assert!(sequence.max_frame_height <= self.limits.max_height);
        self.configured = !self.fails();
        if self.configured {
            Ok(())
        } else {
            Err(Error::Unsupported("the stand-in's configuration failure"))
        }
    }

    fn new_picture(&mut self, _header: &FrameHeader) -> Result<u32, Error> {
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's picture failure"));
        }
        self.next += 1;
        Ok(self.next)
    }

    fn decode_frame(
        &mut self,
        _picture: &u32,
        params: &FrameParams<'_, u32>,
        tile_groups: &[TileGroup<'_>],
    ) -> Result<(), Error> {
        assert!(self.configured, "a frame before a configuration");
        check_frame(&self.limits, params, tile_groups);
        FRAMES.fetch_add(1, Ordering::Relaxed);
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's decode failure"));
        }
        Ok(())
    }
}

/// Splits a sample into its OBUs' bytes, where it splits.
fn obus_of(sample: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut rest = sample;
    while let Some(Ok(_)) = obu::obus(rest).next() {
        let has_extension = rest[0] & 4 != 0;
        let header = 1 + usize::from(has_extension);
        let Ok((size, taken)) = super::super::bits::leb128(&rest[header..]) else {
            break;
        };
        let end = header + taken + size as usize;
        out.push(rest[..end].to_vec());
        rest = &rest[end..];
    }
    out
}

/// A stream of ours, its samples mutated: some whole, some by OBU (one dropped, repeated,
/// moved or mutated alone), some by bytes, samples dropped or repeated, and the decoder's
/// limits now and then below the stream's size.
fn stream_case(rng: &mut Rng) {
    let stream = rng.pick(STREAMS);
    let mut samples: Vec<Vec<u8>> = samples(stream.ivf)
        .into_iter()
        .map(|s| {
            if rng.bool() {
                without_temporal_delimiters(s)
            } else {
                s.to_vec()
            }
        })
        .collect();
    let donor = rng.pick(STREAMS).ivf;
    let noisy = rng.below(4);
    for sample in &mut samples {
        if noisy == 0 || !rng.one_in(noisy + 1) {
            continue;
        }
        if rng.bool() {
            let mut obus = obus_of(sample);
            if !obus.is_empty() {
                let at = rng.below(obus.len());
                match rng.below(4) {
                    0 => {
                        obus.remove(at);
                    }
                    1 => {
                        let copy = obus[at].clone();
                        obus.insert(rng.below(obus.len() + 1), copy);
                    }
                    2 => {
                        let moved = obus.remove(at);
                        obus.insert(rng.below(obus.len() + 1), moved);
                    }
                    _ => maybe_mutate(rng, &mut obus[at], donor, 1000),
                }
                *sample = obus.concat();
            }
        } else {
            maybe_mutate(rng, sample, donor, 1000);
        }
    }
    if rng.one_in(8) && samples.len() > 1 {
        let at = rng.below(samples.len());
        if rng.bool() {
            samples.remove(at);
        } else {
            let copy = samples[at].clone();
            samples.insert(at, copy);
        }
    }
    let limits = if rng.one_in(6) {
        Limits {
            min_width: rng.range_u32(0, 200),
            min_height: rng.range_u32(0, 200),
            ..Limits::up_to(rng.range_u32(1, 700), rng.range_u32(1, 400))
        }
    } else {
        LIMITS
    };
    let mut decoder = Decoder::new(Checking::new(limits, rng.next_u64()), limits);
    for (n, sample) in samples.iter().enumerate() {
        if rng.one_in(64) {
            decoder.reset();
        }
        if let Ok(outputs) = decoder.decode(sample, n as u64) {
            for output in outputs {
                assert!(output.size.0 <= limits.max_width && output.size.1 <= limits.max_height);
                assert_eq!(output.id, n as u64);
            }
        }
    }
}

/// An `av1C` record of one of our streams' sequence headers, mutated, and the first frame
/// after it.
fn record_case(rng: &mut Rng) {
    let stream = rng.pick(STREAMS);
    let samples = samples(stream.ivf);
    let first = without_temporal_delimiters(samples[0]);
    let obus = obus_of(&first);
    let mut record = vec![0x81, 0x00, 0x0c, 0x00];
    record.extend(
        obus.iter()
            .filter(|o| (o[0] >> 3) & 0xf == obu::SEQUENCE_HEADER)
            .flatten(),
    );
    let rounds = 1 + rng.count(6);
    mutate_rounds(rng, &mut record, stream.ivf, rounds);
    let Ok(config) = DecoderConfiguration::parse(&record) else {
        return;
    };
    let _ = config.bit_depth();
    let mut decoder = Decoder::new(Checking::new(LIMITS, rng.next_u64()), LIMITS);
    if decoder.configure(&config).is_err() {
        return;
    }
    let frames: Vec<u8> = obus
        .iter()
        .filter(|o| (o[0] >> 3) & 0xf != obu::SEQUENCE_HEADER)
        .flatten()
        .copied()
        .collect();
    let _ = decoder.decode(&frames, 0);
    for (n, sample) in samples.iter().enumerate().skip(1).take(3) {
        let _ = decoder.decode(sample, n as u64);
    }
}
