//! GPU-less tests of the H.265 decoder: what each decode names (its setup
//! slot, its references' slots and std information, the current sets as
//! slots) over the codec crate's decoding process, on streams written here
//! from the standard's syntax with a placeholder for the slice data, and
//! the parameter sets as std structures. Each decode's plan is checked
//! against a model of the DPB's slots: every slot a decode names must hold
//! the picture the codec crate names there.

use std::cell::Cell;
use std::slice;

use ash::vk::native::{
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
    StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
    StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
};
use reims_vgpu_video_codec::hevc::scaling::ScalingLists;
use reims_vgpu_video_codec::hevc::{
    self, Accelerator, Decoder, NalFormat, OutputOrder, PictureParams,
};

use super::params::{StdPps, StdSps, StdVps, explicit_set, merged, std_scaling_lists, vps_sources};
use super::*;

// `nal_unit_type`s (Table 7-1).
const TRAIL_R: u8 = 1;
const RADL_N: u8 = 6;
const RASL_N: u8 = 8;
const IDR_W_RADL: u8 = 19;
const CRA_NUT: u8 = 21;
const SPS: u8 = 33;
const PPS: u8 = 34;

/// Writes syntax elements, then an escaped NAL unit.
#[derive(Default)]
struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    fn u(&mut self, n: u32, value: u32) -> &mut Self {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
        self
    }

    fn bit(&mut self, value: bool) -> &mut Self {
        self.bits.push(value);
        self
    }

    fn ue(&mut self, value: u32) -> &mut Self {
        let code = u64::from(value) + 1;
        let len = 64 - code.leading_zeros();
        for _ in 1..len {
            self.bits.push(false);
        }
        for i in (0..len).rev() {
            self.bits.push((code >> i) & 1 == 1);
        }
        self
    }

    fn se(&mut self, value: i32) -> &mut Self {
        let magnitude = value.unsigned_abs();
        self.ue(if value > 0 {
            2 * magnitude - 1
        } else {
            2 * magnitude
        })
    }

    /// A one, then zeros to the byte's end.
    fn align(&mut self) -> &mut Self {
        self.bits.push(true);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        self
    }

    /// The NAL unit: its two header bytes, the bits through
    /// `rbsp_trailing_bits()`, with emulation prevention bytes.
    fn nal(&mut self, unit_type: u8) -> Vec<u8> {
        self.align();
        let mut out = vec![unit_type << 1, 1];
        let mut zeros = 0;
        for chunk in self.bits.chunks(8) {
            let byte = chunk
                .iter()
                .fold(0u8, |acc, &bit| (acc << 1) | u8::from(bit));
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if byte == 0 { zeros + 1 } else { 0 };
            out.push(byte);
        }
        out
    }
}

/// A short-term set as a stream codes it (7.3.7).
#[derive(Clone)]
enum Set {
    /// `delta_poc_s0_minus1` and `delta_poc_s1_minus1`, each with its used
    /// flag.
    Explicit(Vec<(u32, bool)>, Vec<(u32, bool)>),
    /// Predicted from the set before it (in an SPS): deltaRps and each
    /// `(used_by_curr_pic_flag, use_delta_flag)`.
    Predicted(i32, Vec<(bool, bool)>),
}

fn write_set(w: &mut BitWriter, index: usize, set: &Set) {
    match set {
        Set::Explicit(negative, positive) => {
            if index != 0 {
                w.bit(false);
            }
            w.ue(negative.len() as u32).ue(positive.len() as u32);
            for &(minus1, used) in negative.iter().chain(positive) {
                w.ue(minus1).bit(used);
            }
        }
        Set::Predicted(delta_rps, flags) => {
            w.bit(true);
            w.bit(*delta_rps < 0).ue(delta_rps.unsigned_abs() - 1);
            for &(used, use_delta) in flags {
                w.bit(used);
                if !used {
                    w.bit(use_delta);
                }
            }
        }
    }
}

/// What a test stream's SPS says. Its pictures are 64x48 luma samples,
/// cropped by two rows, in CTBs of 16; SAO and AMP are on.
#[derive(Clone)]
struct SpsSpec {
    profile_idc: u8,
    /// `general_profile_compatibility_flag`s, flag 0 the high bit.
    compatibility: u32,
    bit_depth_minus8: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4`.
    lsb_minus4: u32,
    dpb: u32,
    reorder: u32,
    sets: Vec<Set>,
    /// `lt_ref_pic_poc_lsb_sps` with `used_by_curr_pic_lt_sps_flag`; the
    /// SPS has long-term pictures where this is `Some`.
    long_term: Option<Vec<(u32, bool)>>,
    scaling_list_enabled: bool,
}

impl Default for SpsSpec {
    fn default() -> Self {
        Self {
            profile_idc: 1,
            compatibility: 0x6000_0000,
            bit_depth_minus8: 0,
            lsb_minus4: 4,
            dpb: 5,
            reorder: 2,
            sets: Vec::new(),
            long_term: None,
            scaling_list_enabled: false,
        }
    }
}

fn sps(spec: &SpsSpec) -> Vec<u8> {
    let mut w = BitWriter::default();
    // sps_video_parameter_set_id 0, one sub-layer, temporal ID nesting.
    w.u(4, 0).u(3, 0).bit(true);
    // profile_tier_level(1, 0): Main tier, progressive frames, level 3.1.
    w.u(2, 0).bit(false).u(5, u32::from(spec.profile_idc));
    w.u(32, spec.compatibility);
    w.bit(true)
        .bit(false)
        .bit(false)
        .bit(true)
        .u(32, 0)
        .u(12, 0);
    w.u(8, 93);
    w.ue(0).ue(1); // sps_seq_parameter_set_id, chroma_format_idc
    w.ue(64).ue(48);
    w.bit(true).ue(0).ue(0).ue(0).ue(1); // the conformance window
    w.ue(spec.bit_depth_minus8).ue(spec.bit_depth_minus8);
    w.ue(spec.lsb_minus4);
    w.bit(true).ue(spec.dpb - 1).ue(spec.reorder).ue(0);
    // CTBs of 16, coding blocks of 8 to 16, transform blocks of 4 to 16.
    for value in [0, 1, 0, 2, 1, 1] {
        w.ue(value);
    }
    w.bit(spec.scaling_list_enabled);
    if spec.scaling_list_enabled {
        w.bit(false); // sps_scaling_list_data_present_flag
    }
    w.bit(true).bit(true).bit(false); // AMP, SAO, no PCM
    w.ue(spec.sets.len() as u32);
    for (i, set) in spec.sets.iter().enumerate() {
        write_set(&mut w, i, set);
    }
    w.bit(spec.long_term.is_some());
    if let Some(pictures) = &spec.long_term {
        w.ue(pictures.len() as u32);
        for &(lsb, used) in pictures {
            w.u(spec.lsb_minus4 + 4, lsb).bit(used);
        }
    }
    // No temporal MVP; strong intra smoothing; no VUI; no extensions.
    w.bit(false).bit(true).bit(false).bit(false);
    w.nal(SPS)
}

/// What a test stream's PPS says beyond its defaults: no slice header
/// options, one reference a list by default, deblocking on.
#[derive(Clone, Default)]
struct PpsSpec {
    /// Uniformly spaced tile columns and rows.
    tiles: Option<(u32, u32)>,
}

fn pps(spec: &PpsSpec) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.ue(0).ue(0).bit(false).bit(false); // ids, no dependent slices or output flag
    w.u(3, 0).bit(false).bit(true); // no extra bits or sign hiding; cabac_init_present_flag
    w.ue(0).ue(0); // one active reference a list
    w.se(0).bit(false).bit(false).bit(false); // init_qp_minus26; no CIP, TS, CU QP delta
    w.se(0).se(0).bit(false); // chroma QP offsets
    w.bit(false).bit(false).bit(false); // no weighted prediction, no bypass
    w.bit(spec.tiles.is_some()).bit(false);
    if let Some((columns, rows)) = spec.tiles {
        w.ue(columns - 1).ue(rows - 1).bit(true).bit(true);
    }
    // Loop filter across slices; no deblocking control, scaling lists or
    // list modification; merge level 2; no extensions.
    w.bit(true).bit(false).bit(false).bit(false);
    w.ue(0).bit(false).bit(false);
    w.nal(PPS)
}

/// A slice header's own short-term set, coded outright: its pictures'
/// `delta_poc_s0_minus1` and `delta_poc_s1_minus1`, each with its used flag.
type OwnSet = (Vec<(u32, bool)>, Vec<(u32, bool)>);
/// A slice header's long-term pictures: the SPS's candidates by
/// `lt_idx_sps`, then the header's own `(poc_lsb_lt, used)`, each with
/// `delta_poc_msb_cycle_lt`.
type LongTerm = (Vec<(u32, Option<u32>)>, Vec<(u32, bool, Option<u32>)>);

/// What a test picture's one slice segment says.
#[derive(Clone)]
struct SliceSpec {
    unit_type: u8,
    /// 0 B, 1 P, 2 I.
    slice_type: u32,
    poc_lsb: u32,
    /// The SPS's set by index, or one of the header's own, coded outright.
    set: Result<u32, OwnSet>,
    /// The SPS's long-term candidates by `lt_idx_sps`, then the header's
    /// own `(poc_lsb_lt, used)`, each with `delta_poc_msb_cycle_lt`.
    long_term: LongTerm,
}

impl Default for SliceSpec {
    fn default() -> Self {
        Self {
            unit_type: TRAIL_R,
            slice_type: 2,
            poc_lsb: 0,
            set: Ok(0),
            long_term: (Vec::new(), Vec::new()),
        }
    }
}

fn slice(sps: &SpsSpec, spec: &SliceSpec) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.bit(true); // first_slice_segment_in_pic_flag
    if (16..=23).contains(&spec.unit_type) {
        w.bit(false); // no_output_of_prior_pics_flag
    }
    w.ue(0).ue(spec.slice_type);
    if !matches!(spec.unit_type, 19 | 20) {
        w.u(sps.lsb_minus4 + 4, spec.poc_lsb);
        match &spec.set {
            Ok(index) => {
                w.bit(true);
                if sps.sets.len() > 1 {
                    let bits = 32 - (sps.sets.len() as u32 - 1).leading_zeros();
                    w.u(bits, *index);
                }
            }
            Err((negative, positive)) => {
                w.bit(false);
                let set = Set::Explicit(negative.clone(), positive.clone());
                write_set(&mut w, sps.sets.len(), &set);
            }
        }
        if let Some(candidates) = &sps.long_term {
            let (from_sps, own) = &spec.long_term;
            if !candidates.is_empty() {
                w.ue(from_sps.len() as u32);
            }
            w.ue(own.len() as u32);
            for &(index, msb) in from_sps {
                if candidates.len() > 1 {
                    let bits = 32 - (candidates.len() as u32 - 1).leading_zeros();
                    w.u(bits, index);
                }
                w.bit(msb.is_some());
                if let Some(cycle) = msb {
                    w.ue(cycle);
                }
            }
            for &(lsb, used, msb) in own {
                w.u(sps.lsb_minus4 + 4, lsb).bit(used);
                w.bit(msb.is_some());
                if let Some(cycle) = msb {
                    w.ue(cycle);
                }
            }
        }
    }
    w.bit(true).bit(true); // SAO, luma and chroma
    if spec.slice_type != 2 {
        w.bit(false); // num_ref_idx_active_override_flag
        if spec.slice_type == 0 {
            w.bit(false); // mvd_l1_zero_flag
        }
        w.bit(false).ue(0); // cabac_init_flag, five_minus_max_num_merge_cand
    }
    w.se(0).bit(true); // slice_qp_delta, slice_loop_filter_across_slices_enabled_flag
    w.align();
    // A placeholder for slice_data().
    w.u(8, 0xa5).u(8, 0x5a);
    w.nal(spec.unit_type)
}

/// An Annex B sample of `units`.
fn sample(units: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(unit);
    }
    out
}

/// A test stream: its first sample has the SPS and PPS before its picture.
fn stream(spec: &SpsSpec, pictures: &[SliceSpec]) -> Vec<Vec<u8>> {
    pictures
        .iter()
        .enumerate()
        .map(|(i, picture)| {
            let mut units = Vec::new();
            if i == 0 {
                units.push(sps(spec));
                units.push(pps(&PpsSpec::default()));
            }
            units.push(slice(spec, picture));
            sample(&units)
        })
        .collect()
}

/// A picture as the model holds it: the slot its decode set it up in.
#[derive(Default)]
struct Slot(Cell<Option<i32>>);

/// One decode's plan, as order counts where it names slots.
#[derive(Debug, PartialEq, Eq)]
struct Planned {
    poc: i32,
    setup: i32,
    /// `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter` and
    /// `RefPicSetLtCurr`, each slot as the order count it holds.
    sets: [Vec<i32>; 3],
    /// `IrapPicFlag` and `IdrPicFlag`.
    irap: bool,
    idr: bool,
}

/// An accelerator that plans each decode as the Vulkan one does, over a
/// model of the DPB's slots, and checks the plan against the codec
/// crate's references.
#[derive(Default)]
struct Planner {
    /// What each slot holds: the order count of the picture last set up
    /// in it.
    slots: Vec<Option<i32>>,
    planned: Vec<Planned>,
    /// The refusals of `plan` when a picture's references have no slot,
    /// share one, or leave none free, for each picture with two references
    /// or more.
    refusals: Vec<[String; 3]>,
}

impl Accelerator for Planner {
    type Picture = Slot;
    type Error = Error;

    fn configure(&mut self, _sps: &hevc::sps::Sps, dpb_size: usize) -> Result<(), Error> {
        self.slots = vec![None; dpb_size];
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Slot, Error> {
        Ok(Slot::default())
    }

    fn decode_picture(
        &mut self,
        picture: &Slot,
        params: &PictureParams<'_, Slot>,
        _slices: &[hevc::SliceParams<'_>],
    ) -> Result<(), Error> {
        let slots = u32::try_from(self.slots.len()).unwrap();
        let plan = plan(params, |p: &Slot| p.0.get(), slots)?;
        assert_eq!(plan.reference_slots.len(), params.references.len());
        for (i, reference) in params.references.iter().enumerate() {
            let slot = usize::try_from(plan.reference_slots[i]).unwrap();
            assert_eq!(
                self.slots[slot],
                Some(reference.poc),
                "POC {}'s reference {} is not in slot {slot}",
                params.poc,
                reference.poc
            );
            let info = &plan.reference_infos[i];
            assert_eq!(info.PicOrderCntVal, reference.poc);
            assert_eq!(
                info.flags.used_for_long_term_reference(),
                u32::from(reference.long_term)
            );
            assert_eq!(info.flags.unused_for_reference(), 0);
        }
        let least_free = (0..)
            .find(|slot| !plan.reference_slots.contains(slot))
            .unwrap();
        assert_eq!(plan.setup_slot, least_free);
        let info = &plan.picture_info;
        assert_eq!(info.PicOrderCntVal, params.poc);
        assert_eq!(info.flags.IsReference(), 1);
        assert_eq!(
            info.flags.short_term_ref_pic_set_sps_flag(),
            u32::from(params.header.short_term_ref_pic_set_sps_flag)
        );
        assert_eq!(
            u32::from(info.NumBitsForSTRefPicSetInSlice),
            params.header.st_rps_bits
        );
        assert_eq!(plan.setup_info.PicOrderCntVal, params.poc);
        assert_eq!(plan.setup_info.flags.used_for_long_term_reference(), 0);
        let as_pocs = |set: &[u8; CURRENT_SET_SIZE]| -> Vec<i32> {
            let used = set.iter().take_while(|&&s| s != NO_SLOT).count();
            assert!(set[used..].iter().all(|&s| s == NO_SLOT), "{set:?}");
            set[..used]
                .iter()
                .map(|&s| self.slots[usize::from(s)].unwrap())
                .collect()
        };
        let sets = [
            as_pocs(&info.RefPicSetStCurrBefore),
            as_pocs(&info.RefPicSetStCurrAfter),
            as_pocs(&info.RefPicSetLtCurr),
        ];
        for (set, indices) in sets.iter().zip(&params.current_sets) {
            let pocs: Vec<i32> = indices.iter().map(|&i| params.references[i].poc).collect();
            assert_eq!(*set, pocs);
        }
        if params.references.len() >= 2 {
            let refusal = |r: Result<Plan, Error>| r.map(|_| ()).unwrap_err().to_string();
            let next = Cell::new(0);
            let sequential = |_: &Slot| {
                next.set(next.get() + 1);
                Some(next.get() - 1)
            };
            let held = u32::try_from(params.references.len()).unwrap();
            self.refusals.push([
                refusal(super::plan(params, |_: &Slot| None, slots)),
                refusal(super::plan(params, |_: &Slot| Some(0), slots)),
                refusal(super::plan(params, sequential, held)),
            ]);
        }
        self.planned.push(Planned {
            poc: params.poc,
            setup: plan.setup_slot,
            sets,
            irap: info.flags.IrapPicFlag() == 1,
            idr: info.flags.IdrPicFlag() == 1,
        });
        self.slots[usize::try_from(plan.setup_slot).unwrap()] = Some(params.poc);
        picture.0.set(Some(plan.setup_slot));
        Ok(())
    }
}

/// Decodes `samples` in turn, then flushes; returns the order counts
/// output in display order and the planner. A skipped RASL picture
/// outputs nothing.
fn decode(samples: &[Vec<u8>]) -> Result<(Vec<i32>, Planner), Error> {
    let mut decoder = Decoder::new(Planner::default(), NalFormat::AnnexB);
    decoder.set_output_order(OutputOrder::Display);
    let mut out = Vec::new();
    for (id, sample) in samples.iter().enumerate() {
        match decoder.decode(sample, id as u64) {
            Ok(outputs) => out.extend(outputs.iter().map(|o| o.poc)),
            Err(Error::H265Stream(hevc::Error::SkippedLeadingPicture)) => {}
            Err(e) => return Err(e),
        }
    }
    out.extend(decoder.flush().iter().map(|o| o.poc));
    let planner = std::mem::take(decoder.accelerator_mut());
    Ok((out, planner))
}

fn planned(poc: i32, setup: i32, sets: [&[i32]; 3]) -> Planned {
    Planned {
        poc,
        setup,
        sets: sets.map(<[i32]>::to_vec),
        irap: false,
        idr: false,
    }
}

fn picture(unit_type: u8, slice_type: u32, poc_lsb: u32, set: u32) -> SliceSpec {
    SliceSpec {
        unit_type,
        slice_type,
        poc_lsb,
        set: Ok(set),
        ..SliceSpec::default()
    }
}

#[test]
fn a_hierarchical_gop_reuses_the_slots_its_pictures_leave() -> Result<(), Error> {
    // P4 from 0; B2 from 0 and 4; B1 from 0, 2 and 4; B3 from 2 and 4,
    // keeping 0; then the same from 4 to 8.
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![(3, true)], vec![]),
            Set::Explicit(vec![(1, true)], vec![(1, true)]),
            Set::Explicit(vec![(0, true)], vec![(0, true), (1, true)]),
            Set::Explicit(vec![(0, true), (1, false)], vec![(0, true)]),
        ],
        ..SpsSpec::default()
    };
    let mut pictures = vec![picture(IDR_W_RADL, 2, 0, 0)];
    for base in [0, 4] {
        pictures.push(picture(TRAIL_R, 1, base + 4, 0));
        pictures.push(picture(TRAIL_R, 0, base + 2, 1));
        pictures.push(picture(TRAIL_R, 0, base + 1, 2));
        pictures.push(picture(TRAIL_R, 0, base + 3, 3));
    }
    let (out, planner) = decode(&stream(&spec, &pictures))?;
    assert_eq!(out, (0..=8).collect::<Vec<_>>());
    let idr = Planned {
        irap: true,
        idr: true,
        ..planned(0, 0, [&[], &[], &[]])
    };
    assert_eq!(
        planner.planned,
        [
            idr,
            planned(4, 1, [&[0], &[], &[]]),
            planned(2, 2, [&[0], &[4], &[]]),
            planned(1, 3, [&[0], &[2, 4], &[]]),
            // 1 has left the DPB's references, 0 stays in its slot.
            planned(3, 3, [&[2], &[4], &[]]),
            // 0 and 2 have left: 8 takes slot 0.
            planned(8, 0, [&[4], &[], &[]]),
            planned(6, 2, [&[4], &[8], &[]]),
            planned(5, 3, [&[4], &[6, 8], &[]]),
            planned(7, 3, [&[6], &[8], &[]]),
        ]
    );
    assert_eq!(
        planner.refusals[0],
        [
            "unsupported: a reference that was never decoded",
            "unsupported: two references in DPB slot 0",
            "unsupported: no DPB slot is free",
        ]
        .map(String::from)
    );
    Ok(())
}

#[test]
fn a_long_term_picture_goes_in_lt_curr_across_the_order_count_wrap() -> Result<(), Error> {
    // MaxPicOrderCntLsb 16: pictures 1 to 20, each from the one before,
    // the IDR kept as long-term from 2 on, by the SPS's candidate while its
    // LSB alone names it, then by the header's own with its MSB cycle.
    let spec = SpsSpec {
        lsb_minus4: 0,
        dpb: 4,
        reorder: 0,
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        long_term: Some(vec![(0, true)]),
        ..SpsSpec::default()
    };
    let mut pictures = vec![picture(IDR_W_RADL, 2, 0, 0)];
    for poc in 1..=20u32 {
        pictures.push(SliceSpec {
            long_term: match poc {
                1 => (Vec::new(), Vec::new()),
                2..=15 => (vec![(0, None)], Vec::new()),
                _ => (Vec::new(), vec![(0, true, Some(1))]),
            },
            ..picture(TRAIL_R, 1, poc % 16, 0)
        });
    }
    let (out, planner) = decode(&stream(&spec, &pictures))?;
    assert_eq!(out, (0..=20).collect::<Vec<_>>());
    let planned_pocs: Vec<i32> = planner.planned.iter().map(|p| p.poc).collect();
    assert_eq!(planned_pocs, (0..=20).collect::<Vec<_>>());
    assert_eq!(planner.planned[1], planned(1, 1, [&[0], &[], &[]]));
    for (poc, plan) in planner.planned.iter().enumerate().skip(2) {
        let poc = i32::try_from(poc).unwrap();
        // The IDR stays in slot 0; the short-term reference alternates
        // between 1 and 2.
        let setup = if poc % 2 == 0 { 2 } else { 1 };
        assert_eq!(*plan, planned(poc, setup, [&[poc - 1], &[], &[0]]));
    }
    Ok(())
}

#[test]
fn a_cra_skips_its_rasl_pictures_and_names_the_pictures_before_one_mid_stream() -> Result<(), Error>
{
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![], vec![(1, true)]),          // +2
            Set::Explicit(vec![(0, true)], vec![(0, true)]), // -1, +1
            Set::Explicit(vec![(1, true)], vec![]),          // -2
            Set::Explicit(vec![(3, false)], vec![]),         // -4, kept
            Set::Explicit(vec![(1, true)], vec![(1, true)]), // -2, +2
        ],
        ..SpsSpec::default()
    };
    let cra = |poc| Planned {
        irap: true,
        ..planned(poc, 0, [&[], &[], &[]])
    };
    // A leading CRA at 10, as if cut from a longer stream: its RASL
    // picture at 9 is skipped, its RADL picture at 8 decodes from it.
    let pictures = [
        picture(CRA_NUT, 2, 10, 3),
        picture(RASL_N, 0, 9, 1),
        picture(RADL_N, 1, 8, 0),
        picture(TRAIL_R, 1, 12, 2),
    ];
    let (out, planner) = decode(&stream(&spec, &pictures))?;
    assert_eq!(out, [8, 10, 12]);
    assert_eq!(
        planner.planned,
        [
            cra(10),
            planned(8, 1, [&[], &[10], &[]]),
            planned(12, 1, [&[10], &[], &[]]),
        ]
    );
    // A CRA at 16 mid-stream keeps 12 for its RASL picture at 14, which
    // decodes from both.
    let pictures = [
        picture(CRA_NUT, 2, 10, 3),
        picture(TRAIL_R, 1, 12, 2),
        picture(CRA_NUT, 2, 16, 3),
        picture(RASL_N, 0, 14, 4),
        picture(TRAIL_R, 1, 18, 2),
    ];
    let (out, planner) = decode(&stream(&spec, &pictures))?;
    assert_eq!(out, [10, 12, 14, 16, 18]);
    assert_eq!(
        planner.planned,
        [
            cra(10),
            planned(12, 1, [&[10], &[], &[]]),
            cra(16),
            planned(14, 2, [&[12], &[16], &[]]),
            planned(18, 1, [&[16], &[], &[]]),
        ]
    );
    Ok(())
}

#[test]
fn a_slice_headers_own_set_is_planned_from_the_pictures_it_names() -> Result<(), Error> {
    // Each P picture names the pictures before it in its own set, up to
    // four: its bits go to the hardware to skip.
    let spec = SpsSpec {
        dpb: 6,
        reorder: 0,
        ..SpsSpec::default()
    };
    let mut pictures = vec![picture(IDR_W_RADL, 2, 0, 0)];
    for poc in 1..=8u32 {
        let negative = vec![(0, true); poc.min(4) as usize];
        pictures.push(SliceSpec {
            set: Err((negative, Vec::new())),
            ..picture(TRAIL_R, 1, poc, 0)
        });
    }
    let (out, planner) = decode(&stream(&spec, &pictures))?;
    assert_eq!(out, (0..=8).collect::<Vec<_>>());
    let last = &planner.planned[8];
    assert_eq!(last.sets[0], [7, 6, 5, 4]);
    // The four pictures 1 to 3 and 8 replaced leave slot 3 (POC 3) free.
    assert_eq!(last.setup, 3);
    Ok(())
}

#[test]
fn a_current_set_takes_eight_slots_and_fills_the_rest_with_none() {
    let slots = [4, 0, 7, 2, 1, 3, 6, 5, 8];
    assert_eq!(
        current_set(&[2, 0], &slots).unwrap(),
        [7, 4, NO_SLOT, NO_SLOT, NO_SLOT, NO_SLOT, NO_SLOT, NO_SLOT]
    );
    assert_eq!(
        current_set(&(0..8).collect::<Vec<_>>(), &slots).unwrap(),
        [4, 0, 7, 2, 1, 3, 6, 5]
    );
    let nine = current_set(&(0..9).collect::<Vec<_>>(), &slots).unwrap_err();
    assert_eq!(
        nine.to_string(),
        "unsupported: 9 pictures in one of the current reference sets: Vulkan Video takes 8"
    );
    assert!(current_set(&[9], &slots).is_err());
}

#[test]
fn an_explicit_set_is_coded_as_distances_and_used_masks() {
    let set = explicit_set(
        &[(-1, true), (-3, false), (-4, true)],
        &[(2, true), (5, false)],
    )
    .unwrap();
    assert_eq!((set.num_negative_pics, set.num_positive_pics), (3, 2));
    assert_eq!(set.delta_poc_s0_minus1[..3], [0, 1, 0]);
    assert_eq!(set.delta_poc_s1_minus1[..2], [1, 2]);
    assert_eq!(set.used_by_curr_pic_s0_flag, 0b101);
    assert_eq!(set.used_by_curr_pic_s1_flag, 0b01);
    assert_eq!(set.flags.inter_ref_pic_set_prediction_flag(), 0);
    // Out of order, twice the same picture, the wrong side, or too many.
    assert!(explicit_set(&[(-3, true), (-1, true)], &[]).is_err());
    assert!(explicit_set(&[(-1, true), (-1, true)], &[]).is_err());
    assert!(explicit_set(&[(1, true)], &[]).is_err());
    assert!(explicit_set(&[], &[(-1, true)]).is_err());
    let many: Vec<(i32, bool)> = (1..=17).map(|d| (-d, true)).collect();
    assert!(explicit_set(&many, &[]).is_err());
    assert!(explicit_set(&many[..16], &[]).is_ok());
}

#[test]
fn an_sps_goes_with_its_predicted_sets_derived_and_its_long_term_candidates() -> Result<(), Error> {
    // Set 1 is predicted from set 0 (-1 used, -3 not) by deltaRps -1: -2,
    // -4 and -1 itself, all used.
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![(0, true), (1, false)], vec![]),
            Set::Predicted(-1, vec![(true, false), (true, false), (true, false)]),
        ],
        long_term: Some(vec![(3, true), (9, false), (12, true)]),
        scaling_list_enabled: true,
        ..SpsSpec::default()
    };
    let parsed = hevc::sps::Sps::parse(&sps(&spec))?;
    let std = StdSps::new(&parsed, H265Profile::MAIN)?;
    let s = &std.sps;
    assert_eq!(
        (s.pic_width_in_luma_samples, s.pic_height_in_luma_samples),
        (64, 48)
    );
    assert_eq!(s.flags.conformance_window_flag(), 1);
    assert_eq!(s.conf_win_bottom_offset, 1);
    assert_eq!(s.log2_max_pic_order_cnt_lsb_minus4, 4);
    assert_eq!(s.num_short_term_ref_pic_sets, 2);
    // SAFETY: the SPS's sets, which `std` keeps, two of them.
    let sets = unsafe { std::slice::from_raw_parts(s.pShortTermRefPicSet, 2) };
    assert_eq!(
        (sets[0].num_negative_pics, sets[0].num_positive_pics),
        (2, 0)
    );
    assert_eq!(sets[0].delta_poc_s0_minus1[..2], [0, 1]);
    assert_eq!(sets[0].used_by_curr_pic_s0_flag, 0b01);
    assert_eq!(
        (sets[1].num_negative_pics, sets[1].num_positive_pics),
        (3, 0)
    );
    assert_eq!(sets[1].delta_poc_s0_minus1[..3], [0, 0, 1]);
    assert_eq!(sets[1].used_by_curr_pic_s0_flag, 0b111);
    assert!(
        sets.iter()
            .all(|s| s.flags.inter_ref_pic_set_prediction_flag() == 0)
    );
    assert_eq!(s.flags.long_term_ref_pics_present_flag(), 1);
    assert_eq!(s.num_long_term_ref_pics_sps, 3);
    // SAFETY: the candidates `std` keeps.
    let long_term = unsafe { &*s.pLongTermRefPicsSps };
    assert_eq!(long_term.lt_ref_pic_poc_lsb_sps[..3], [3, 9, 12]);
    assert_eq!(long_term.used_by_curr_pic_lt_sps_flag, 0b101);
    // Scaling lists enabled and none coded: the defaults, as present.
    assert_eq!(s.flags.scaling_list_enabled_flag(), 1);
    assert_eq!(s.flags.sps_scaling_list_data_present_flag(), 1);
    // SAFETY: the lists `std` keeps.
    let lists = unsafe { &*s.pScalingLists };
    assert_eq!(
        lists.ScalingList8x8,
        std_scaling_lists(&ScalingLists::default()).ScalingList8x8
    );
    // SAFETY: the profile, tier and level `std` keeps.
    let ptl = unsafe { &*s.pProfileTierLevel };
    // SAFETY: the DPB limits `std` keeps.
    let dpb = unsafe { &*s.pDecPicBufMgr };
    assert_eq!(
        ptl.general_profile_idc,
        StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
    );
    assert_eq!(
        ptl.general_level_idc,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1
    );
    assert_eq!(dpb.max_dec_pic_buffering_minus1[0], 4);
    assert_eq!(dpb.max_num_reorder_pics[0], 2);
    Ok(())
}

#[test]
fn a_profile_outside_main_goes_as_the_profile_that_decodes_it() -> Result<(), Error> {
    // general_profile_idc 0, compatible with Main 10 alone.
    let spec = SpsSpec {
        profile_idc: 0,
        compatibility: 0x2000_0000,
        bit_depth_minus8: 2,
        ..SpsSpec::default()
    };
    let parsed = hevc::sps::Sps::parse(&sps(&spec))?;
    let std = StdSps::new(&parsed, H265Profile::MAIN_10)?;
    let vps = StdVps::new(&parsed, H265Profile::MAIN_10);
    // SAFETY: the profile, tier and level `std` keeps.
    let sps_ptl = unsafe { &*std.sps.pProfileTierLevel };
    // SAFETY: the profile, tier and level `vps` keeps.
    let vps_ptl = unsafe { &*vps.vps.pProfileTierLevel };
    for ptl in [sps_ptl, vps_ptl] {
        assert_eq!(
            ptl.general_profile_idc,
            StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
        );
    }
    assert_eq!(vps.vps.flags.vps_temporal_id_nesting_flag(), 1);
    assert_eq!(vps.vps.vps_max_sub_layers_minus1, 0);
    Ok(())
}

#[test]
fn scaling_lists_go_in_coding_order_with_the_32x32_lists_of_matrix_ids_0_and_3() {
    let mut lists = ScalingLists::default();
    for m in 0..6 {
        let base = u8::try_from(m).unwrap() * 10;
        lists.list_4x4[m] = core::array::from_fn(|i| base + u8::try_from(i).unwrap());
        lists.list_8x8[m] = core::array::from_fn(|i| base + u8::try_from(i).unwrap() + 1);
        lists.list_16x16[m] = core::array::from_fn(|i| base + u8::try_from(i).unwrap() + 2);
        lists.list_32x32[m] = core::array::from_fn(|i| base + u8::try_from(i).unwrap() + 3);
        lists.dc_16x16[m] = base + 4;
        lists.dc_32x32[m] = base + 5;
    }
    let std = std_scaling_lists(&lists);
    assert_eq!(std.ScalingList4x4, lists.list_4x4);
    assert_eq!(std.ScalingList8x8, lists.list_8x8);
    assert_eq!(std.ScalingList16x16, lists.list_16x16);
    assert_eq!(
        std.ScalingList32x32,
        [lists.list_32x32[0], lists.list_32x32[3]]
    );
    assert_eq!(std.ScalingListDCCoef16x16, lists.dc_16x16);
    assert_eq!(std.ScalingListDCCoef32x32, [5, 35]);
}

#[test]
fn uniform_tiles_go_with_each_columns_and_rows_size() -> Result<(), Error> {
    // 64x48 in CTBs of 16 is four by three: three columns of 1, 1 and 2
    // CTBs, two rows of 1 and 2 (6-3, 6-4).
    let parsed_sps = hevc::sps::Sps::parse(&sps(&SpsSpec::default()))?;
    let parsed_pps = hevc::pps::Pps::parse(&pps(&PpsSpec {
        tiles: Some((3, 2)),
    }))?;
    let std = StdPps::new(&parsed_pps, &parsed_sps)?;
    let p = &std.pps;
    assert_eq!(p.flags.tiles_enabled_flag(), 1);
    assert_eq!(p.flags.uniform_spacing_flag(), 1);
    assert_eq!((p.num_tile_columns_minus1, p.num_tile_rows_minus1), (2, 1));
    assert_eq!(p.column_width_minus1[..3], [0, 0, 1]);
    assert_eq!(p.row_height_minus1[..2], [0, 1]);
    Ok(())
}

#[test]
fn a_parameters_object_keeps_the_sets_a_new_one_does_not_replace() -> Result<(), Error> {
    let first = hevc::sps::Sps::parse(&sps(&SpsSpec::default()))?;
    let second = hevc::sps::Sps::parse(&sps(&SpsSpec {
        dpb: 6,
        ..SpsSpec::default()
    }))?;
    let plain = hevc::pps::Pps::parse(&pps(&PpsSpec::default()))?;
    let tiled = hevc::pps::Pps::parse(&pps(&PpsSpec {
        tiles: Some((2, 1)),
    }))?;
    // The same sets again: nothing changes.
    let (sps_list, pps_list) = merged(
        slice::from_ref(&first),
        slice::from_ref(&plain),
        &first,
        &plain,
    );
    assert_eq!(
        (sps_list, pps_list),
        (vec![first.clone()], vec![plain.clone()])
    );
    // A PPS of the same id replaces the one before.
    let (sps_list, pps_list) = merged(
        slice::from_ref(&first),
        slice::from_ref(&plain),
        &first,
        &tiled,
    );
    assert_eq!(
        (sps_list, pps_list),
        (vec![first.clone()], vec![tiled.clone()])
    );
    // An SPS of the same id replaces the one before, with every PPS read
    // against it but the new one.
    let (sps_list, pps_list) = merged(
        slice::from_ref(&first),
        slice::from_ref(&plain),
        &second,
        &tiled,
    );
    assert_eq!((sps_list, pps_list), (vec![second.clone()], vec![tiled]));
    // One VPS for each id, from the last SPS that names it.
    let both = [first, second.clone()];
    assert_eq!(vps_sources(&both), [&second]);
    Ok(())
}
