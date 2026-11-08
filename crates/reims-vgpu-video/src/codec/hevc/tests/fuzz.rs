//! HEVC's parsers and decoding process under seeded fuzzing, up to an
//! accelerator that checks what VA-API's is handed. The streams are the
//! tests' own writers' with random specs: the tests' hierarchical GOP of
//! four, repeated, and sets, slice segments and parameter sets that stray
//! from it as far as each stream's noise says, then mutated by unit or by
//! sample.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::codec::fuzz_support::{Rng, Target, long, maybe_mutate, mutate_rounds, quick};

const STREAMS: Target = Target::new("hevc-streams");
const RECORDS: Target = Target::new("hevc-records");

/// Pictures that reached the accelerator, which says how deep the inputs
/// get.
static PICTURES: AtomicU64 = AtomicU64::new(0);

#[test]
fn streams() {
    let report = quick(&STREAMS, 3000, stream_case);
    eprintln!(
        "hevc: {} pictures reached the accelerator in {} streams",
        PICTURES.load(Ordering::Relaxed),
        report.inputs
    );
}

#[test]
#[ignore = "long: RVGV_FUZZ_CASES, RVGV_FUZZ_SECONDS, RVGV_FUZZ_SEED"]
fn streams_long() {
    let report = long(&STREAMS, stream_case);
    eprintln!(
        "hevc: {} pictures reached the accelerator in {} streams",
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
/// VA-API's takes: fifteen references, lists of fifteen that index them,
/// slice segments within the picture's CTBs, the last one marked last. It
/// fails one call in 64, as a driver may.
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

    fn configure(&mut self, sps: &sps::Sps, dpb_size: usize) -> Result<(), Error> {
        assert!(dpb_size <= 17, "a DPB of {dpb_size} pictures");
        assert!(sps.pic_size_in_ctbs() > 0);
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
        slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        assert!(self.configured, "a picture before a configuration");
        let references = params.references.len();
        assert!(references <= 15, "{references} references");
        for set in &params.current_sets {
            assert!(
                set.iter().all(|&i| i < references),
                "a set past the references"
            );
        }
        assert!(
            slices.last().is_some_and(|s| s.last),
            "no last slice segment"
        );
        let ctbs = params.sps.pic_size_in_ctbs();
        for slice in slices {
            assert!(
                slice.header.slice_segment_address < ctbs,
                "slice segment address {} of {ctbs} CTBs",
                slice.header.slice_segment_address
            );
            for list in &slice.ref_lists {
                assert!(list.len() <= 15, "a list of {}", list.len());
                assert!(
                    list.iter().all(|&i| i < references),
                    "a list entry past the references"
                );
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
/// 1 now and then, 2 often.
#[derive(Clone, Copy)]
struct Noise(u8);

impl Noise {
    fn of(rng: &mut Rng) -> Self {
        Self(rng.below(3) as u8)
    }

    fn strays(self, rng: &mut Rng) -> bool {
        match self.0 {
            0 => false,
            1 => rng.one_in(16),
            _ => rng.one_in(4),
        }
    }
}

/// A short-term set of any shape, predicted or not.
fn any_set(rng: &mut Rng) -> Set {
    if rng.one_in(3) {
        let delta = loop {
            let delta = rng.below(9) as i32 - 4;
            if delta != 0 {
                break delta;
            }
        };
        let flags = (0..rng.range(1, 6))
            .map(|_| (rng.bool(), rng.bool()))
            .collect();
        Set::Predicted(rng.range_u32(0, 3), delta, flags)
    } else {
        let side = |rng: &mut Rng| {
            (0..rng.count(17))
                .map(|_| (rng.interesting_u32() % 40, rng.bool()))
                .collect()
        };
        Set::Explicit(side(rng), side(rng))
    }
}

fn sps_spec(rng: &mut Rng, noise: Noise) -> SpsSpec {
    let mut spec = SpsSpec {
        profile_idc: *rng.pick(&[1, 1, 2, 3]),
        bit_depth_minus8: if rng.bool() { 0 } else { 2 },
        sets: gop_sets(),
        long_term: rng.one_in(4).then(|| {
            (0..rng.range(1, 3))
                .map(|_| (rng.below(256) as u32, rng.bool()))
                .collect()
        }),
        scaling_list_enabled: rng.one_in(4),
        ..SpsSpec::default()
    };
    if noise.strays(rng) {
        spec.profile_idc = rng.range(0, 31) as u8;
    }
    if noise.strays(rng) {
        spec.chroma_format_idc = rng.range_u32(0, 3);
    }
    if noise.strays(rng) {
        spec.bit_depth_minus8 = rng.range_u32(0, 9);
    }
    if noise.strays(rng) {
        spec.lsb_minus4 = rng.range_u32(0, 12);
    }
    if noise.strays(rng) {
        spec.dpb = rng.range_u32(1, 17);
        spec.reorder = rng.range_u32(0, 17);
    }
    if noise.strays(rng) {
        spec.latency_plus1 = rng.interesting_u32();
    }
    if noise.strays(rng) {
        for _ in 0..rng.count(8) {
            spec.sets.push(any_set(rng));
        }
    }
    if noise.strays(rng) {
        spec.sets = (0..rng.count(64)).map(|_| any_set(rng)).collect();
    }
    spec.range_extension = noise.strays(rng);
    spec
}

fn pps_spec(rng: &mut Rng, noise: Noise) -> PpsSpec {
    let mut spec = PpsSpec {
        dependent_slices: rng.one_in(3),
        output_flag_present: rng.one_in(4),
        weighted: rng.one_in(4),
        lists_modification: rng.one_in(4),
        tiles: rng
            .one_in(4)
            .then(|| (rng.range_u32(1, 2), rng.range_u32(1, 2))),
        wavefronts: rng.one_in(4),
        ..PpsSpec::default()
    };
    if noise.strays(rng) {
        spec.tiles = Some((rng.range_u32(1, 30), rng.range_u32(1, 30)));
    }
    if noise.strays(rng) {
        spec.l0_default_minus1 = rng.range_u32(0, 20);
        spec.l1_default_minus1 = rng.range_u32(0, 20);
    }
    spec
}

/// The GOP of four the SPS's first sets describe, as `(type, slice type,
/// order count past the GOP's start, set)`: P4, B2, B1, B3.
const GOP: [(u8, u32, u32, u32); 4] = [
    (nal_type::TRAIL_R, 1, 4, 0),
    (nal_type::TRAIL_R, 0, 2, 1),
    (nal_type::TRAIL_N, 0, 1, 2),
    (nal_type::TRAIL_N, 0, 3, 3),
];

/// The slice segments of picture `index` of the stream.
fn segments(rng: &mut Rng, noise: Noise, sps: &SpsSpec, pps: &PpsSpec, index: u32) -> Vec<Vec<u8>> {
    let first = if index == 0 {
        // An IRAP picture uses no reference (7.4.7.1), so a CRA one codes
        // an empty set of its own.
        SliceSpec {
            unit_type: *rng.pick(&[nal_type::IDR_W_RADL, nal_type::IDR_N_LP, nal_type::CRA_NUT]),
            set: Err(Set::Explicit(Vec::new(), Vec::new())),
            ..SliceSpec::default()
        }
    } else {
        let (unit_type, slice_type, offset, set) = GOP[((index - 1) % 4) as usize];
        let poc = 4 * ((index - 1) / 4) + offset;
        SliceSpec {
            unit_type,
            slice_type,
            poc_lsb: poc % (1 << (sps.lsb_minus4 + 4)),
            set: Ok(set.min(sps.sets.len().saturating_sub(1) as u32)),
            ..SliceSpec::default()
        }
    };
    let mut first = first;
    if noise.strays(rng) {
        first.unit_type = rng.range(0, 40) as u8;
    }
    if noise.strays(rng) {
        first.slice_type = rng.range_u32(0, 5);
    }
    if noise.strays(rng) {
        first.poc_lsb = rng.next_u32();
    }
    if noise.strays(rng) {
        first.set = if rng.bool() || sps.sets.is_empty() {
            Err(any_set(rng))
        } else {
            Ok(rng.below(sps.sets.len()) as u32)
        };
    }
    if let Some(candidates) = &sps.long_term {
        let strays = noise.strays(rng);
        if strays || rng.one_in(6) {
            // Pictures the stream never coded: within the standard, kept
            // but not used by the current picture, which would miss them.
            let mut from_sps = Vec::new();
            for _ in 0..rng.count(3) {
                let index = rng.below(candidates.len()) as u32;
                let msb = rng.bool().then(|| rng.range_u32(0, 3));
                if strays || !candidates[index as usize].1 {
                    from_sps.push((index, msb));
                }
            }
            let own = (0..rng.count(3))
                .map(|_| {
                    (
                        rng.below(256) as u32,
                        strays && rng.bool(),
                        rng.bool().then(|| rng.interesting_u32() % 64),
                    )
                })
                .collect();
            first.long_term = (from_sps, own);
        }
    }
    if rng.one_in(4) {
        first.active = Some((rng.range_u32(0, 2), rng.range_u32(0, 2)));
    }
    if noise.strays(rng) {
        first.active = Some((rng.range_u32(0, 20), rng.range_u32(0, 20)));
    }
    if pps.lists_modification && rng.bool() {
        // An entry for each place of each list, each naming one of the
        // pictures the current one uses (7.4.7.2), unless it strays.
        let (l0, l1) = first
            .active
            .unwrap_or((pps.l0_default_minus1, pps.l1_default_minus1));
        let total = match &first.set {
            Ok(index) => used(&sps.sets, *index as usize),
            Err(set) => used_by_own(&sps.sets, set),
        } + used_long_term(sps, &first);
        let strays = noise.strays(rng);
        let list = |rng: &mut Rng, places: u32| -> Vec<u32> {
            let places = if strays { 3 } else { places + 1 };
            (0..places)
                .map(|_| {
                    if strays {
                        rng.range_u32(0, 3)
                    } else {
                        rng.below(total.max(1) as usize) as u32
                    }
                })
                .collect()
        };
        let l0 = list(rng, l0);
        let l1 = rng.bool().then(|| list(rng, l1));
        first.list_entries = [Some(l0), l1];
    }
    first.pic_output = !rng.one_in(6);
    if pps.tiles.is_some() || pps.wavefronts {
        // A tile, a CTB row, or a CTB row of each tile column, but the
        // first (7.4.7.1), unless it strays.
        let (columns, rows) = pps.tiles.unwrap_or((1, 1));
        let height = ctbs(sps).1;
        let most = match (pps.tiles.is_some(), pps.wavefronts) {
            (true, false) => columns * rows,
            (false, true) => height,
            _ => columns * height,
        };
        first.entry_points = if noise.strays(rng) {
            rng.range_u32(0, 40)
        } else {
            rng.range_u32(0, most.saturating_sub(1).min(3))
        };
    }
    first.no_data = noise.strays(rng) && rng.one_in(4);
    let mut units = vec![slice(sps, pps, &first)];
    let mut address = 0;
    for _ in 0..rng.count(4) {
        address += rng.range_u32(1, 3);
        let mut segment = first.clone();
        segment.first = false;
        segment.address = if noise.strays(rng) {
            rng.range_u32(0, 15)
        } else {
            address.min(11)
        };
        segment.dependent = pps.dependent_slices && rng.bool();
        if noise.strays(rng) {
            segment.unit_type = rng.range(0, 22) as u8;
        }
        units.push(slice(sps, pps, &segment));
    }
    units
}

/// An SPS and a PPS. Where the PPS has tiles, the pictures are large
/// enough for them, columns of 256 luma samples and rows of 64 (A.3.2),
/// unless the stream strays.
fn parameter_sets(rng: &mut Rng, noise: Noise) -> (SpsSpec, PpsSpec) {
    let mut sps = sps_spec(rng, noise);
    let pps = pps_spec(rng, noise);
    if let Some((columns, rows)) = pps.tiles
        && !noise.strays(rng)
    {
        sps.size = (sps.size.0.max(256 * columns), sps.size.1.max(64 * rows));
    }
    (sps, pps)
}

/// The pictures of `slice`'s long-term part that the current one uses.
fn used_long_term(sps: &SpsSpec, slice: &SliceSpec) -> u32 {
    let (from_sps, own) = &slice.long_term;
    let candidates = sps.long_term.as_deref().unwrap_or_default();
    let from_sps = from_sps
        .iter()
        .filter(|(index, _)| candidates.get(*index as usize).is_some_and(|c| c.1))
        .count();
    (from_sps + own.iter().filter(|o| o.1).count()) as u32
}

/// A stream of a few pictures, the parameter sets in the first sample
/// and now and then again, with units and samples mutated as its noise
/// says.
fn stream(rng: &mut Rng, noise: Noise) -> (SpsSpec, PpsSpec, Vec<Vec<u8>>) {
    let (mut sps_spec_now, mut pps_spec_now) = parameter_sets(rng, noise);
    let first = (sps_spec_now.clone(), pps_spec_now.clone());
    let mut samples = Vec::new();
    for index in 0..rng.range_u32(1, 14) {
        let mut units = Vec::new();
        if index == 0 || rng.one_in(12) {
            if index > 0 && noise.strays(rng) {
                (sps_spec_now, pps_spec_now) = parameter_sets(rng, noise);
            }
            units.push(sps(&sps_spec_now));
            units.push(pps(&pps_spec_now));
        }
        units.extend(segments(rng, noise, &sps_spec_now, &pps_spec_now, index));
        if rng.one_in(8) {
            // A unit of any type: VPS, SEI, AUD, end of sequence.
            let kind = *rng.pick(&[32u8, 35, 36, 37, 39, 40, 48, 63]);
            let mut unit = vec![kind << 1, 1];
            unit.extend(rng.some_bytes(24));
            units.push(unit);
        }
        if noise.strays(rng) {
            let at = rng.below(units.len());
            let donor = units[at].clone();
            maybe_mutate(rng, &mut units[at], &donor, 1);
        }
        let mut bytes = sample(&units);
        if noise.strays(rng) && rng.one_in(3) {
            let donor = bytes.clone();
            let rounds = 1 + rng.count(6);
            mutate_rounds(rng, &mut bytes, &donor, rounds);
        }
        samples.push(bytes);
    }
    (first.0, first.1, samples)
}

/// A stream through the decoder, in either output order, reset now and
/// then.
fn stream_case(rng: &mut Rng) {
    let noise = Noise::of(rng);
    let (_, _, samples) = stream(rng, noise);
    let accelerator = Checking {
        failing: rng.next_u64(),
        ..Checking::default()
    };
    let mut decoder = Decoder::new(accelerator, NalFormat::AnnexB);
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

/// `hvcC` records, as the tests write one, read or are refused, and a
/// decoder made from one decodes a picture or refuses it.
fn record_case(rng: &mut Rng) {
    let noise = Noise::of(rng);
    let (sps_spec, pps_spec) = parameter_sets(rng, noise);
    let mut record = vec![1, 0x01, 0x40, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 93];
    let size = *rng.pick(&[1u8, 2, 4]);
    record.extend_from_slice(&[0xf0, 0, 0xfc, 0xfd, 0xf8, 0xf8, 0, 0, 0x0c | (size - 1), 2]);
    for (kind, unit) in [(33u8, sps(&sps_spec)), (34, pps(&pps_spec))] {
        record.extend_from_slice(&[0x80 | kind, 0, 1]);
        record.extend_from_slice(&(unit.len() as u16).to_be_bytes());
        record.extend_from_slice(&unit);
    }
    let donor = record.clone();
    maybe_mutate(rng, &mut record, &donor, 2);
    let _ = hvcc::DecoderConfiguration::parse(&record);
    let Ok(mut decoder) = Decoder::from_hvcc(Checking::default(), &record) else {
        return;
    };
    let units = segments(rng, noise, &sps_spec, &pps_spec, 0);
    let mut sample = Vec::new();
    let size = usize::from(size);
    for unit in units {
        sample.extend_from_slice(&(unit.len() as u32).to_be_bytes()[4 - size..]);
        sample.extend_from_slice(&unit);
    }
    let _ = decoder.decode(&sample, 0);
    let _ = decoder.flush();
}

/// An accelerator that refuses to configure for 10 bits and, like
/// VA-API's, keeps nothing of its last configuration once it refuses one:
/// it has no picture to give until it is configured again.
#[derive(Default)]
struct RefusingTenBits {
    configured: bool,
    configurations: usize,
}

impl Accelerator for RefusingTenBits {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sps: &sps::Sps, _dpb_size: usize) -> Result<(), Error> {
        self.configured = sps.bit_depth_luma_minus8 == 0;
        if !self.configured {
            return Err(Error::Unsupported("10 bits"));
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
        _slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        Ok(())
    }
}

/// An IDR picture of the SPS the decoder had before a refused
/// configuration configures the accelerator again.
#[test]
fn a_refused_configuration_is_asked_for_again() -> Result<(), Error> {
    let main = SpsSpec::default();
    let ten = SpsSpec {
        profile_idc: 2,
        bit_depth_minus8: 2,
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let idr = SliceSpec {
        unit_type: nal_type::IDR_W_RADL,
        ..SliceSpec::default()
    };
    let mut d = Decoder::new(RefusingTenBits::default(), NalFormat::AnnexB);
    d.decode(
        &sample(&[sps(&main), pps(&pps_spec), slice(&main, &pps_spec, &idr)]),
        0,
    )?;
    let refused = sample(&[sps(&ten), pps(&pps_spec), slice(&ten, &pps_spec, &idr)]);
    assert!(d.decode(&refused, 1).is_err());
    let out = d.decode(
        &sample(&[sps(&main), pps(&pps_spec), slice(&main, &pps_spec, &idr)]),
        2,
    )?;
    assert_eq!(out.len(), 1);
    assert_eq!(d.accelerator().configurations, 2);
    Ok(())
}
