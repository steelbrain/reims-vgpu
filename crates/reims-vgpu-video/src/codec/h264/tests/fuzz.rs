//! H.264's parsers and decoding process under seeded fuzzing, up to an
//! accelerator that checks what VA-API's is handed. The streams are the
//! tests' own writers' with random specs, some outside the standard's
//! ranges, then mutated by unit or by sample, so that inputs get past the
//! first header.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::codec::fuzz_support::{Rng, Target, long, maybe_mutate, mutate_rounds, quick};

const STREAMS: Target = Target::new("h264-streams");
const RECORDS: Target = Target::new("h264-records");

/// Pictures that reached the accelerator, which says how deep the inputs
/// get.
static PICTURES: AtomicU64 = AtomicU64::new(0);

#[test]
fn streams() {
    let report = quick(&STREAMS, 8000, stream_case);
    eprintln!(
        "h264: {} pictures reached the accelerator in {} streams",
        PICTURES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn streams_long() {
    let report = long(&STREAMS, stream_case);
    eprintln!(
        "h264: {} pictures reached the accelerator in {} streams",
        PICTURES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
fn records() {
    quick(&RECORDS, 3000, record_case);
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn records_long() {
    long(&RECORDS, record_case);
}

/// An accelerator that checks what the decoder gives it against what
/// VA-API's takes: 16 references, lists of 32 that name only those, a
/// configuration before the first picture. It fails one call in 64, as a
/// driver may.
#[derive(Default)]
struct Checking {
    next: u32,
    configured: bool,
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
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sps: &sps::Sps, dpb_frames: usize) -> Result<(), Error> {
        assert!(dpb_frames <= 32, "a DPB of {dpb_frames} frames");
        assert!(sps.width_in_mbs() > 0 && sps.frame_height_in_mbs() > 0);
        self.configured = !self.fails();
        if self.configured {
            Ok(())
        } else {
            Err(Error::Unsupported("the stand-in's configuration failure"))
        }
    }

    fn new_picture(&mut self) -> Result<u32, Error> {
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's picture failure"));
        }
        self.next += 1;
        Ok(self.next)
    }

    fn decode_picture(
        &mut self,
        _picture: &u32,
        params: &PictureParams<'_, u32>,
        slices: &[SliceParams<'_, u32>],
    ) -> Result<(), Error> {
        assert!(self.configured, "a picture before a configuration");
        assert!(!slices.is_empty(), "a picture of no slices");
        assert!(
            params.references.len() <= 16,
            "{} references",
            params.references.len()
        );
        for slice in slices {
            for list in &slice.ref_lists {
                assert!(list.len() <= 32, "a list of {}", list.len());
                for entry in list.iter().flatten() {
                    assert!(
                        params
                            .references
                            .iter()
                            .any(|r| std::ptr::eq(r.picture, entry.picture)),
                        "a list entry outside the references"
                    );
                }
            }
        }
        PICTURES.fetch_add(1, Ordering::Relaxed);
        if self.fails() {
            return Err(Error::Unsupported("the stand-in's decode failure"));
        }
        Ok(())
    }
}

/// How far a stream strays from what the standard allows: 0 not at all,
/// so that it decodes picture after picture; 1 now and then; 2 often.
#[derive(Clone, Copy)]
struct Noise(u8);

impl Noise {
    fn of(rng: &mut Rng) -> Self {
        Self(rng.below(3) as u8)
    }

    /// Whether this element strays.
    fn strays(self, rng: &mut Rng) -> bool {
        match self.0 {
            0 => false,
            1 => rng.one_in(16),
            _ => rng.one_in(4),
        }
    }
}

/// A signed value the writer's `se` takes: anything but `i32::MIN`.
fn signed(rng: &mut Rng) -> i32 {
    if rng.one_in(4) {
        rng.interesting_i32().max(-i32::MAX)
    } else {
        rng.below(9) as i32 - 4
    }
}

fn sps_spec(rng: &mut Rng, noise: Noise) -> SpsSpec {
    let mut spec = SpsSpec {
        profile_idc: *rng.pick(&[66, 77, 100]),
        poc_type: rng.below(3) as u8,
        log2_max_frame_num_minus4: rng.range_u32(0, 12),
        type1: (
            rng.below(5) as i32 - 2,
            (0..rng.range(1, 4)).map(|_| 2).collect(),
        ),
        max_num_ref_frames: rng.range_u32(1, 16),
        gaps: rng.one_in(4),
        fields: None,
        restriction: rng
            .bool()
            .then(|| (rng.range_u32(0, 4), rng.range_u32(4, 16))),
        full_range: rng.bool(),
    };
    if noise.strays(rng) {
        spec.profile_idc = *rng.pick(&[88, 110, 144, 0, 255]);
    }
    if noise.strays(rng) {
        spec.log2_max_frame_num_minus4 = rng.range_u32(13, 28);
    }
    if noise.strays(rng) {
        spec.type1 = (
            signed(rng),
            (0..rng.count(255)).map(|_| signed(rng)).collect(),
        );
    }
    if noise.strays(rng) {
        spec.max_num_ref_frames = if rng.bool() { 0 } else { rng.interesting_u32() };
    }
    if noise.strays(rng) {
        spec.restriction = Some((rng.interesting_u32(), rng.interesting_u32()));
    }
    spec
}

/// Where a stream's frame numbers and order counts stand.
#[derive(Default)]
struct Counts {
    prev_ref_frame_num: u32,
    pictures: u32,
}

fn slice_spec(rng: &mut Rng, noise: Noise, counts: &mut Counts, sps: &SpsSpec) -> SliceSpec {
    let max_frame_num = 1u32
        .checked_shl(sps.log2_max_frame_num_minus4 + 4)
        .unwrap_or(u32::MAX);
    let idr = counts.pictures == 0 || rng.one_in(16);
    let reference = idr || rng.below(4) != 0;
    let frame_num = if idr {
        0
    } else {
        counts.prev_ref_frame_num.wrapping_add(1) % max_frame_num.max(1)
    };
    if reference {
        counts.prev_ref_frame_num = frame_num;
    }
    if idr {
        counts.pictures = 0;
    }
    let mut spec = SliceSpec {
        idr,
        reference,
        slice_type: if idr { 2 } else { rng.range_u32(0, 2) },
        frame_num,
        poc_lsb: (counts.pictures * 2) % 256,
        ..SliceSpec::default()
    };
    counts.pictures += 1;
    if spec.slice_type != 2 && rng.one_in(4) {
        spec.active = Some((rng.range_u32(0, 3), rng.range_u32(0, 3)));
    }
    if rng.one_in(6) {
        // Short-term pictures by difference: back one, ahead one.
        spec.modifications.push((rng.range_u32(0, 1), 0));
    }
    if reference && !idr && rng.one_in(6) {
        // Sliding window's own: unmark the oldest, or mark this long-term.
        spec.mmco
            .push(*rng.pick(&[(1, 0, 0), (6, 0, 0), (4, 1, 0)]));
    }
    spec.temporal_direct = rng.bool();
    if noise.strays(rng) {
        spec.slice_type = rng.range_u32(0, 12);
    }
    if noise.strays(rng) {
        spec.frame_num = rng.next_u32();
    }
    if noise.strays(rng) {
        spec.poc_lsb = rng.range_u32(0, 255);
    }
    if noise.strays(rng) {
        spec.active = Some((rng.range_u32(0, 33), rng.range_u32(0, 33)));
    }
    if noise.strays(rng) {
        for _ in 0..rng.count(5) {
            spec.modifications
                .push((rng.range_u32(0, 6), rng.interesting_u32() % 64));
        }
    }
    if noise.strays(rng) {
        for _ in 0..rng.count(5) {
            let op = rng.range_u32(1, 7);
            spec.mmco
                .push((op, rng.interesting_u32() % 40, rng.interesting_u32() % 20));
        }
    }
    spec.long_term_idr = idr && noise.strays(rng);
    if noise.strays(rng) {
        spec.first_mb = rng.interesting_u32();
    }
    if noise.strays(rng) {
        spec.delta_bottom = Some(signed(rng));
    }
    if noise.strays(rng) {
        spec.slice_qp_delta = signed(rng);
    }
    spec.no_data = noise.strays(rng) && rng.one_in(4);
    spec
}

/// Units in a sample of `format`.
fn frame(units: &[Vec<u8>], format: NalFormat) -> Vec<u8> {
    match format {
        NalFormat::AnnexB => sample(units),
        NalFormat::LengthPrefixed(size) => {
            let size = usize::from(size).min(4);
            let mut out = Vec::new();
            for unit in units {
                let length = (unit.len() as u32).to_be_bytes();
                out.extend_from_slice(&length[4 - size..]);
                out.extend_from_slice(unit);
            }
            out
        }
    }
}

/// A stream of a few pictures, the parameter sets in the first sample
/// and now and then again, with units and samples mutated as its noise
/// says.
fn stream(rng: &mut Rng, format: NalFormat) -> (SpsSpec, Vec<u8>, Vec<Vec<u8>>) {
    let noise = Noise::of(rng);
    let mut spec = sps_spec(rng, noise);
    let first_sps = sps(&spec);
    let pps = |rng: &mut Rng| pps_with(spec.profile_idc == 100 && rng.bool(), rng.bool());
    let first_pps = pps(rng);
    let mut counts = Counts::default();
    let mut samples = Vec::new();
    for picture in 0..rng.range_u32(1, 16) {
        let mut units = Vec::new();
        if picture == 0 {
            units.push(first_sps.clone());
            units.push(first_pps.clone());
        } else if rng.one_in(12) {
            if noise.strays(rng) {
                spec = sps_spec(rng, noise);
            }
            units.push(sps(&spec));
            units.push(pps_with(spec.profile_idc == 100 && rng.bool(), rng.bool()));
        }
        let first = slice_spec(rng, noise, &mut counts, &spec);
        for index in 0..1 + rng.count(3) {
            let mut slice_spec = first.clone();
            if index > 0 {
                slice_spec.first_mb = rng.range_u32(1, 8159);
            }
            if noise.strays(rng) {
                slice_spec.slice_qp_delta = signed(rng);
            }
            units.push(slice(&slice_spec, &spec));
        }
        if rng.one_in(8) {
            // A unit of any type: SEI, AUD, end of sequence and the rest.
            let mut unit = vec![rng.next_u32() as u8 & 0x7f];
            unit.extend(rng.some_bytes(24));
            units.push(unit);
        }
        if noise.strays(rng) {
            let at = rng.below(units.len());
            let donor = units[at].clone();
            maybe_mutate(rng, &mut units[at], &donor, 1);
        }
        let mut bytes = frame(&units, format);
        if noise.strays(rng) && rng.one_in(3) {
            let donor = bytes.clone();
            let rounds = 1 + rng.count(6);
            mutate_rounds(rng, &mut bytes, &donor, rounds);
        }
        samples.push(bytes);
    }
    (spec, first_sps, samples)
}

/// A stream through the decoder, Annex B or length-prefixed, from an
/// `avcC` record or not, in either output order, reset now and then.
fn stream_case(rng: &mut Rng) {
    let format = match rng.below(4) {
        0 | 1 => NalFormat::AnnexB,
        2 => NalFormat::LengthPrefixed(4),
        _ => NalFormat::LengthPrefixed(*rng.pick(&[1, 2])),
    };
    let (_, first_sps, samples) = stream(rng, format);
    let accelerator = Checking {
        failing: rng.next_u64(),
        ..Checking::default()
    };
    let mut decoder = match format {
        NalFormat::LengthPrefixed(size) if rng.one_in(3) => {
            let record = record(rng, &first_sps, size);
            match Decoder::from_avcc(accelerator, &record) {
                Ok(decoder) => decoder,
                Err(_) => return,
            }
        }
        _ => Decoder::new(accelerator, format),
    };
    decoder.set_output_order(if rng.bool() {
        OutputOrder::Display
    } else {
        OutputOrder::Decode
    });
    for (id, sample) in samples.iter().enumerate() {
        let _ = decoder.decode(sample, id as u64);
        if rng.one_in(20) {
            decoder.reset();
        }
    }
    let _ = decoder.flush();
}

/// An `avcC` record for `sps` and a PPS, `size`-byte lengths.
fn record(rng: &mut Rng, sps: &[u8], size: u8) -> Vec<u8> {
    let pps = pps_with(rng.bool(), rng.bool());
    let mut record = vec![1, 77, 0, 40, 0xfc | (size.clamp(1, 4) - 1), 0xe1];
    record.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    record.extend_from_slice(sps);
    record.push(1);
    record.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    record.extend_from_slice(&pps);
    if rng.one_in(3) {
        let donor = record.clone();
        maybe_mutate(rng, &mut record, &donor, 1);
    }
    record
}

/// `avcC` records read or are refused, and so do their parameter sets.
fn record_case(rng: &mut Rng) {
    let noise = Noise::of(rng);
    let spec = sps_spec(rng, noise);
    let size = *rng.pick(&[1, 2, 4]);
    let bytes = record(rng, &sps(&spec), size);
    if let Ok(parsed) = avcc::DecoderConfiguration::parse(&bytes) {
        for unit in parsed.sequence_parameter_sets {
            let _ = Decoder::new(Checking::default(), NalFormat::LengthPrefixed(4))
                .decode(&frame(&[unit.to_vec()], NalFormat::LengthPrefixed(4)), 0);
        }
    }
    let _ = Decoder::from_avcc(Checking::default(), &bytes);
}

/// An accelerator that refuses to configure for High and, like VA-API's,
/// keeps nothing of its last configuration once it refuses one: it has no
/// picture to give until it is configured again.
#[derive(Default)]
struct RefusingHigh {
    configured: bool,
    configurations: usize,
}

impl Accelerator for RefusingHigh {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sps: &sps::Sps, _dpb_frames: usize) -> Result<(), Error> {
        self.configured = sps.profile_idc != 100;
        if !self.configured {
            return Err(Error::Unsupported("High"));
        }
        self.configurations += 1;
        Ok(())
    }

    fn new_picture(&mut self) -> Result<u32, Error> {
        if self.configured {
            Ok(0)
        } else {
            Err(Error::Unsupported("no configuration"))
        }
    }

    fn decode_picture(
        &mut self,
        _picture: &u32,
        _params: &PictureParams<'_, u32>,
        _slices: &[SliceParams<'_, u32>],
    ) -> Result<(), Error> {
        Ok(())
    }
}

/// An IDR picture of the SPS the decoder had before a refused
/// configuration configures the accelerator again.
#[test]
fn a_refused_configuration_is_asked_for_again() -> Result<(), Error> {
    let main = SpsSpec::default();
    let high = SpsSpec {
        profile_idc: 100,
        ..SpsSpec::default()
    };
    let mut d = Decoder::new(RefusingHigh::default(), NalFormat::AnnexB);
    d.set_output_order(OutputOrder::Decode);
    d.decode(&sample(&[sps(&main), pps(false), slice(&idr(), &main)]), 0)?;
    assert!(
        d.decode(&sample(&[sps(&high), slice(&idr(), &high)]), 1)
            .is_err()
    );
    let out = d.decode(&sample(&[sps(&main), slice(&idr(), &main)]), 2)?;
    assert_eq!(out.len(), 1);
    assert_eq!(d.accelerator().configurations, 2);
    Ok(())
}

/// A picture that configures a new SPS without an IDR, after a refused
/// configuration, does not carry the last SPS's frame number into its gap,
/// where it was past the new MaxFrameNum.
#[test]
fn frame_numbers_start_afresh_with_a_new_configuration() -> Result<(), Error> {
    let long = SpsSpec {
        log2_max_frame_num_minus4: 4,
        gaps: true,
        ..SpsSpec::default()
    };
    let short = SpsSpec::default();
    let high = SpsSpec {
        profile_idc: 100,
        ..SpsSpec::default()
    };
    let mut d = Decoder::new(RefusingHigh::default(), NalFormat::AnnexB);
    d.set_output_order(OutputOrder::Decode);
    let intra = |frame_num, poc_lsb| SliceSpec {
        reference: true,
        slice_type: 2,
        frame_num,
        poc_lsb,
        ..SliceSpec::default()
    };
    d.decode(&sample(&[sps(&long), pps(false), slice(&idr(), &long)]), 0)?;
    d.decode(&sample(&[slice(&intra(20, 40), &long)]), 1)?;
    assert!(
        d.decode(&sample(&[sps(&high), slice(&idr(), &high)]), 2)
            .is_err()
    );
    let out = d.decode(&sample(&[sps(&short), slice(&intra(3, 6), &short)]), 3)?;
    assert_eq!(out.len(), 1);
    assert_eq!(d.accelerator().configurations, 2);
    Ok(())
}

/// An SPS whose frames are alike but numbered otherwise takes over at an
/// IDR picture without a new configuration. Where that picture then fails,
/// as one of an SI slice does once its SPS is active, the next picture
/// does not carry the last SPS's frame number into its gap, where it was
/// past the new MaxFrameNum.
#[test]
fn frame_numbers_start_afresh_with_an_sps_of_alike_frames() -> Result<(), Error> {
    let long = SpsSpec {
        log2_max_frame_num_minus4: 4,
        gaps: true,
        ..SpsSpec::default()
    };
    let short = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let intra = |frame_num, poc_lsb| SliceSpec {
        reference: true,
        slice_type: 2,
        frame_num,
        poc_lsb,
        ..SliceSpec::default()
    };
    d.decode(&sample(&[sps(&long), pps(false), slice(&idr(), &long)]), 0)?;
    d.decode(&sample(&[slice(&intra(20, 40), &long)]), 1)?;
    let switching = SliceSpec {
        slice_type: 4,
        ..idr()
    };
    assert!(
        d.decode(&sample(&[sps(&short), slice(&switching, &short)]), 2)
            .is_err()
    );
    assert_eq!(d.active_sps().map(sps::Sps::max_frame_num), Some(16));
    let out = d.decode(&sample(&[slice(&intra(3, 6), &short)]), 3)?;
    assert_eq!(out.len(), 1);
    assert_eq!(d.accelerator().configured, 1);
    Ok(())
}

/// After operation 5 at the ends of 32 bits, the next picture's order
/// count `msb` is worked out without overflow, where
/// `prevPicOrderCntLsb` is the wrapped `TopFieldOrderCnt`, near -2^31.
#[test]
fn an_order_count_after_operation_5_at_the_ends_of_32_bits_does_not_overflow() -> Result<(), Error>
{
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let first = SliceSpec {
        delta_bottom: Some(0),
        ..idr()
    };
    let units = [sps(&spec), pps_with(false, true), slice(&first, &spec)];
    d.decode(&sample(&units), 0)?;
    let far = SliceSpec {
        delta_bottom: Some(i32::MAX),
        mmco: vec![(5, 0, 0)],
        ..p(1, 10)
    };
    d.decode(&sample(&[slice(&far, &spec)]), 1)?;
    let next = SliceSpec {
        delta_bottom: Some(0),
        ..p(1, 12)
    };
    let out = d.decode(&sample(&[slice(&next, &spec)]), 2)?;
    assert_eq!(out.len(), 1);
    Ok(())
}
