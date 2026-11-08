//! The slice segment header (H.265 7.3.6), with its reference picture
//! set, list modifications (7.3.6.2) and prediction weights (7.3.6.3).

use std::sync::Arc;

use super::Error;
use super::bits::BitReader;
use super::nal::{NalUnit, is_idr, is_irap, payload_reader};
use super::pps::Pps;
use super::rps::ShortTermRps;
use super::sps::Sps;
use crate::codec::h264::bits::ceil_log2;
use crate::codec::num::ToUsize;

/// `slice_type` (Table 7-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    /// Bi-predictive.
    B,
    /// Predictive.
    P,
    /// Intra.
    I,
}

impl SliceType {
    /// The coded value.
    pub fn raw(self) -> u8 {
        match self {
            Self::B => 0,
            Self::P => 1,
            Self::I => 2,
        }
    }

    /// Whether slices of this type predict from other pictures.
    pub fn is_inter(self) -> bool {
        self != Self::I
    }
}

/// A long-term picture of the slice's reference picture set: an entry of
/// `lt_idx_sps[]` or `poc_lsb_lt[]`, with its MSB cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LongTermPicture {
    /// `PocLsbLt[i]`.
    pub(crate) poc_lsb: u32,
    /// `UsedByCurrPicLt[i]`.
    pub used_by_current: bool,
    /// `DeltaPocMsbCycleLt[i]` (7-52) when `delta_poc_msb_present_flag[i]`.
    pub(crate) delta_poc_msb_cycle: Option<u32>,
}

/// One reference's explicit weights in a `pred_weight_table()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightEntry {
    /// `delta_luma_weight_lX[i]` and `luma_offset_lX[i]`, when
    /// `luma_weight_lX_flag[i]`.
    pub luma: Option<(i8, i8)>,
    /// `delta_chroma_weight_lX[i][j]` and `delta_chroma_offset_lX[i][j]`
    /// for Cb and Cr, when `chroma_weight_lX_flag[i]`.
    pub chroma: Option<[(i8, i16); 2]>,
}

/// `pred_weight_table()` (7.3.6.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_weight_denom: u8,
    /// `delta_chroma_log2_weight_denom` (0 without chroma).
    pub delta_chroma_log2_weight_denom: i8,
    /// List 0's entries and list 1's, one per active reference.
    pub entries: [Vec<WeightEntry>; 2],
}

impl PredWeightTable {
    /// `ChromaLog2WeightDenom`, 0 where the sum leaves a byte (which
    /// parsing refuses).
    pub(crate) fn chroma_log2_weight_denom(&self) -> u8 {
        self.luma_log2_weight_denom
            .checked_add_signed(self.delta_chroma_log2_weight_denom)
            .unwrap_or(0)
    }

    /// `ChromaOffsetLX[i][j]` (7-56), with `WpOffsetHalfRangeC` 128: the
    /// offset VA-API wants beside the coded deltas. 0 where the entry has
    /// no chroma weights, and where a table's `ChromaLog2WeightDenom` is
    /// too large for the offset to be computed in 32 bits; parsing
    /// refuses any past 7.
    pub fn chroma_offset(&self, list: usize, index: usize, component: usize) -> i16 {
        let Some([cb, cr]) = self
            .entries
            .get(list)
            .and_then(|entries| entries.get(index))
            .and_then(|e| e.chroma)
        else {
            return 0;
        };
        let (delta_weight, delta_offset) = if component == 0 { cb } else { cr };
        chroma_offset(self.chroma_log2_weight_denom(), delta_weight, delta_offset).unwrap_or(0)
    }
}

/// 7-56's `ChromaOffsetLX[i][j]` of a `delta_chroma_weight_lX[i][j]` and
/// `delta_chroma_offset_lX[i][j]` under `ChromaLog2WeightDenom` `denom`:
/// Clip3(-128, 127, 128 + delta - ((128 * ChromaWeightLX) >> denom)),
/// where ChromaWeightLX = (1 << denom) + delta_chroma_weight (7-55).
/// `None` where it overflows 32 bits, as no `denom` up to 7 makes it.
fn chroma_offset(denom: u8, delta_weight: i8, delta_offset: i16) -> Option<i16> {
    let shift = u32::from(denom);
    let weight = 1i32
        .checked_shl(shift)?
        .checked_add(i32::from(delta_weight))?;
    let scaled = 128i32.checked_mul(weight)?.checked_shr(shift)?;
    let offset = 128i32
        .checked_sub(scaled)?
        .checked_add(i32::from(delta_offset))?;
    i16::try_from(offset.clamp(-128, 127)).ok()
}

/// A slice segment header, with what it implies (7.4.7.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    /// The NAL unit's `nal_unit_type`.
    pub nal_unit_type: u8,
    /// The NAL unit's `TemporalId`.
    pub(crate) temporal_id: u8,
    /// The active SPS.
    pub(crate) sps: Arc<Sps>,
    /// The slice's PPS.
    pub(crate) pps: Arc<Pps>,
    /// `first_slice_segment_in_pic_flag`.
    pub first_slice_segment_in_pic_flag: bool,
    /// `no_output_of_prior_pics_flag` (IRAP pictures).
    pub(crate) no_output_of_prior_pics_flag: bool,
    /// `dependent_slice_segment_flag`: the fields below
    /// `slice_segment_address` are the independent segment's before it.
    pub dependent_slice_segment_flag: bool,
    /// `slice_segment_address`.
    pub slice_segment_address: u32,
    /// `slice_type`.
    pub slice_type: SliceType,
    /// `pic_output_flag` (1 when absent).
    pub pic_output_flag: bool,
    /// `colour_plane_id`.
    pub colour_plane_id: u8,
    /// `slice_pic_order_cnt_lsb` (0 for an IDR picture).
    pub slice_pic_order_cnt_lsb: u32,
    /// `short_term_ref_pic_set_sps_flag`.
    pub short_term_ref_pic_set_sps_flag: bool,
    /// The short-term set in force: the SPS's
    /// `short_term_ref_pic_set_idx`th, or the header's own. Empty for an
    /// IDR picture.
    pub(crate) short_term_rps: ShortTermRps,
    /// The bits the header's own `st_ref_pic_set()` took, as VA-API's
    /// `st_rps_bits` counts them: 0 when the set is the SPS's.
    pub st_rps_bits: u32,
    /// `NumDeltaPocs[RefRpsIdx]` (7.4.8) of the SPS's set the header's own
    /// set is predicted from: 0 when the header has no set of its own, or
    /// codes it outright.
    pub st_rps_reference_delta_pocs: u32,
    /// The long-term pictures, `num_long_term_sps` from the SPS's list
    /// first, then `num_long_term_pics`.
    pub long_term: Vec<LongTermPicture>,
    /// `slice_temporal_mvp_enabled_flag`.
    pub slice_temporal_mvp_enabled_flag: bool,
    /// `slice_sao_luma_flag`.
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag`.
    pub slice_sao_chroma_flag: bool,
    /// `num_ref_idx_l0_active_minus1`, the PPS's default when not
    /// overridden.
    pub num_ref_idx_l0_active_minus1: u8,
    /// `num_ref_idx_l1_active_minus1`, likewise.
    pub num_ref_idx_l1_active_minus1: u8,
    /// `list_entry_l0[]` and `list_entry_l1[]`, when
    /// `ref_pic_list_modification_flag_lX`.
    pub list_entries: [Option<Vec<u8>>; 2],
    /// `mvd_l1_zero_flag`.
    pub mvd_l1_zero_flag: bool,
    /// `cabac_init_flag`.
    pub cabac_init_flag: bool,
    /// `collocated_from_l0_flag` (1 when absent).
    pub collocated_from_l0_flag: bool,
    /// `collocated_ref_idx` (0 when absent).
    pub collocated_ref_idx: u8,
    /// `pred_weight_table()`, when the PPS weights this slice type.
    pub pred_weight_table: Option<PredWeightTable>,
    /// `five_minus_max_num_merge_cand`.
    pub five_minus_max_num_merge_cand: u8,
    /// `slice_qp_delta`.
    pub slice_qp_delta: i8,
    /// `slice_cb_qp_offset`.
    pub slice_cb_qp_offset: i8,
    /// `slice_cr_qp_offset`.
    pub slice_cr_qp_offset: i8,
    /// `slice_deblocking_filter_disabled_flag` (the PPS's when absent).
    pub slice_deblocking_filter_disabled_flag: bool,
    /// `slice_beta_offset_div2` (the PPS's when absent).
    pub slice_beta_offset_div2: i8,
    /// `slice_tc_offset_div2` (the PPS's when absent).
    pub slice_tc_offset_div2: i8,
    /// `slice_loop_filter_across_slices_enabled_flag` (the PPS's when
    /// absent).
    pub slice_loop_filter_across_slices_enabled_flag: bool,
    /// `num_entry_point_offsets`: tile and wavefront substreams.
    pub num_entry_point_offsets: u32,
    /// `NumPicTotalCurr` (7-55).
    pub num_pic_total_curr: u32,
    /// The bytes from the NAL unit header to `slice_data()`, counted with
    /// the emulation prevention bytes removed: VA-API's
    /// `slice_data_byte_offset`.
    pub header_bytes: u32,
    /// The emulation prevention bytes in those: VA-API's
    /// `slice_data_num_emu_prevn_bytes`.
    pub emulation_prevention_bytes: u32,
}

impl SliceHeader {
    /// Parses a slice segment NAL unit's header. `pps` and `sps` look
    /// parameter sets up by id; `independent` is the picture's last
    /// independent slice segment header, which a dependent segment takes
    /// most of its fields from.
    pub fn parse(
        nal: NalUnit<'_>,
        pps: impl Fn(u8) -> Option<Arc<Pps>>,
        sps: impl Fn(u8) -> Option<Arc<Sps>>,
        independent: Option<&SliceHeader>,
    ) -> Result<Self, Error> {
        let unit_type = nal.unit_type();
        let mut r = payload_reader(&nal)?;
        let first_slice_segment_in_pic_flag = r.bit()?;
        let no_output_of_prior_pics_flag = is_irap(unit_type) && r.bit()?;
        let pps_id = r.ue_u8(63, "slice_pic_parameter_set_id")?;
        let pps = pps(pps_id).ok_or(Error::MissingPps(pps_id))?;
        let sps_id = pps.pps_seq_parameter_set_id;
        let sps = sps(sps_id).ok_or(Error::MissingSps(sps_id))?;
        let mut dependent_slice_segment_flag = false;
        let mut slice_segment_address = 0;
        if !first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                dependent_slice_segment_flag = r.bit()?;
            }
            let size = sps.pic_size_in_ctbs();
            slice_segment_address = r.bits(ceil_log2(size.to_usize()))?;
            if slice_segment_address >= size {
                return Err(Error::OutOfRange("slice_segment_address"));
            }
        }
        let mut header = if dependent_slice_segment_flag {
            let independent = independent.ok_or(Error::NoIndependentSliceSegment)?;
            if !Arc::ptr_eq(&independent.pps, &pps) && *independent.pps != *pps {
                return Err(Error::OutOfRange("slice_pic_parameter_set_id"));
            }
            independent.clone()
        } else {
            Self::parse_independent(&mut r, unit_type, &sps, &pps)?
        };
        header.nal_unit_type = unit_type;
        header.temporal_id = nal.temporal_id();
        header.first_slice_segment_in_pic_flag = first_slice_segment_in_pic_flag;
        header.no_output_of_prior_pics_flag = no_output_of_prior_pics_flag;
        header.dependent_slice_segment_flag = dependent_slice_segment_flag;
        header.slice_segment_address = slice_segment_address;
        header.num_entry_point_offsets = 0;
        if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
            // 7.4.7.1: an entry point a tile, a CTB row, or a CTB row of
            // each tile column, but the first.
            let columns = u32::from(pps.num_tile_columns_minus1) + 1;
            let rows = u32::from(pps.num_tile_rows_minus1) + 1;
            let height = sps.pic_height_in_ctbs();
            // The most entry points less one: the tiles, the CTB rows, or
            // both, each 1 at least as the SPS and PPS hold them.
            let most = match (pps.tiles_enabled_flag, pps.entropy_coding_sync_enabled_flag) {
                (true, false) => columns.checked_mul(rows),
                (false, true) => Some(height),
                _ => columns.checked_mul(height),
            }
            .and_then(|entry_points| entry_points.checked_sub(1))
            .ok_or(Error::Overflow("num_entry_point_offsets"))?;
            let count = r.ue_max(most, "num_entry_point_offsets")?;
            if count > 0 {
                let bits = r
                    .ue_max(31, "offset_len_minus1")?
                    .checked_add(1)
                    .ok_or(Error::Overflow("offset_len_minus1"))?;
                for _ in 0..count {
                    r.skip(bits)?; // entry_point_offset_minus1[i]
                }
            }
            header.num_entry_point_offsets = count;
        }
        if pps.slice_segment_header_extension_present_flag {
            let length = r.ue_max(256, "slice_segment_header_extension_length")?;
            for _ in 0..length {
                r.skip(8)?; // slice_segment_header_extension_data_byte[i]
            }
        }
        // byte_alignment(): a one, then zeros to the byte's end.
        if !r.bit()? {
            return Err(Error::OutOfRange("alignment_bit_equal_to_one"));
        }
        while !r.byte_aligned() {
            if r.bit()? {
                return Err(Error::OutOfRange("alignment_bit_equal_to_zero"));
            }
        }
        header.header_bytes = u32::try_from(r.position() / 8)
            .map_err(|_| Error::Overflow("the slice segment header's length"))?;
        header.emulation_prevention_bytes = u32::try_from(r.emulation_prevention_bytes())
            .map_err(|_| Error::Overflow("the slice segment header's length"))?;
        Ok(header)
    }

    /// The fields of an independent slice segment, from `slice_reserved_flag`
    /// to `slice_loop_filter_across_slices_enabled_flag`.
    fn parse_independent(
        r: &mut BitReader<'_>,
        unit_type: u8,
        sps: &Arc<Sps>,
        pps: &Arc<Pps>,
    ) -> Result<Self, Error> {
        r.skip(u32::from(pps.num_extra_slice_header_bits))?;
        let slice_type = match r.ue_max(2, "slice_type")? {
            0 => SliceType::B,
            1 => SliceType::P,
            _ => SliceType::I,
        };
        if is_irap(unit_type) && slice_type != SliceType::I {
            return Err(Error::OutOfRange("slice_type of an IRAP picture"));
        }
        let pic_output_flag = !pps.output_flag_present_flag || r.bit()?;
        let colour_plane_id = if sps.separate_colour_plane_flag {
            r.u8(2)?
        } else {
            0
        };
        let mut slice_pic_order_cnt_lsb = 0;
        let mut short_term_ref_pic_set_sps_flag = false;
        let mut short_term_rps = ShortTermRps::default();
        let mut st_rps_bits = 0;
        let mut st_rps_reference_delta_pocs = 0;
        let mut long_term = Vec::new();
        let mut slice_temporal_mvp_enabled_flag = false;
        if !is_idr(unit_type) {
            let lsb_bits = u32::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4;
            slice_pic_order_cnt_lsb = r.bits(lsb_bits)?;
            short_term_ref_pic_set_sps_flag = r.bit()?;
            let sets = &sps.short_term_ref_pic_sets;
            if short_term_ref_pic_set_sps_flag {
                if sets.is_empty() {
                    return Err(Error::OutOfRange("short_term_ref_pic_set_sps_flag"));
                }
                let index = if sets.len() > 1 {
                    r.bits(ceil_log2(sets.len()))?.to_usize()
                } else {
                    0
                };
                short_term_rps = sets
                    .get(index)
                    .cloned()
                    .ok_or(Error::OutOfRange("short_term_ref_pic_set_idx"))?;
            } else {
                let start = r.position();
                let max_pictures = u32::from(sps.highest_ordering().max_dec_pic_buffering_minus1);
                let (set, reference) =
                    ShortTermRps::parse_with_reference(r, sets, true, max_pictures)?;
                short_term_rps = set;
                let bits = r
                    .position()
                    .checked_sub(start)
                    .ok_or(Error::Overflow("st_rps_bits"))?;
                st_rps_bits = u32::try_from(bits).unwrap_or(0);
                st_rps_reference_delta_pocs = reference
                    .and_then(|i| sets.get(i))
                    .map_or(0, |set| u32::try_from(set.num_delta_pocs()).unwrap_or(0));
            }
            if sps.long_term_ref_pics_present_flag {
                long_term = parse_long_term(r, sps, lsb_bits)?;
            }
            if sps.sps_temporal_mvp_enabled_flag {
                slice_temporal_mvp_enabled_flag = r.bit()?;
            }
        }
        let mut slice_sao_luma_flag = false;
        let mut slice_sao_chroma_flag = false;
        if sps.sample_adaptive_offset_enabled_flag {
            slice_sao_luma_flag = r.bit()?;
            if sps.chroma_array_type() != 0 {
                slice_sao_chroma_flag = r.bit()?;
            }
        }
        let used = short_term_rps
            .used_by_current()
            .checked_add(long_term.iter().filter(|lt| lt.used_by_current).count())
            .ok_or(Error::Overflow("NumPicTotalCurr"))?;
        let num_pic_total_curr = u32::try_from(used).unwrap_or(u32::MAX);
        // 7.4.7.1: a CRA or BLA picture predicts from none of its set's
        // pictures, which are for the pictures after it.
        if is_irap(unit_type) && num_pic_total_curr != 0 {
            return Err(Error::OutOfRange("NumPicTotalCurr of an IRAP picture"));
        }
        let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        let mut list_entries = [None, None];
        let mut mvd_l1_zero_flag = false;
        let mut cabac_init_flag = false;
        let mut collocated_from_l0_flag = true;
        let mut collocated_ref_idx = 0;
        let mut pred_weight_table = None;
        let mut five_minus_max_num_merge_cand = 0;
        if slice_type.is_inter() {
            let b = slice_type == SliceType::B;
            if r.bit()? {
                num_ref_idx_l0_active_minus1 = r.ue_u8(14, "num_ref_idx_l0_active_minus1")?;
                if b {
                    num_ref_idx_l1_active_minus1 = r.ue_u8(14, "num_ref_idx_l1_active_minus1")?;
                }
            }
            if num_pic_total_curr == 0 {
                return Err(Error::MissingReference);
            }
            if pps.lists_modification_present_flag && num_pic_total_curr > 1 {
                let bits = ceil_log2(num_pic_total_curr.to_usize());
                let counts = [num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1];
                for (list, count) in list_entries
                    .iter_mut()
                    .zip(counts)
                    .take(if b { 2 } else { 1 })
                {
                    if r.bit()? {
                        let mut entries = Vec::new();
                        for _ in 0..=count {
                            let entry = r.bits(bits)?;
                            entries.push(
                                u8::try_from(entry)
                                    .ok()
                                    .filter(|_| entry < num_pic_total_curr)
                                    .ok_or(Error::OutOfRange("list_entry"))?,
                            );
                        }
                        *list = Some(entries);
                    }
                }
            }
            if b {
                mvd_l1_zero_flag = r.bit()?;
            }
            if pps.cabac_init_present_flag {
                cabac_init_flag = r.bit()?;
            }
            if slice_temporal_mvp_enabled_flag {
                if b {
                    collocated_from_l0_flag = r.bit()?;
                }
                let count = if collocated_from_l0_flag {
                    num_ref_idx_l0_active_minus1
                } else {
                    num_ref_idx_l1_active_minus1
                };
                if count > 0 {
                    collocated_ref_idx = r.ue_u8(count, "collocated_ref_idx")?;
                }
            }
            if (pps.weighted_pred_flag && slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && b)
            {
                pred_weight_table = Some(parse_pred_weight_table(
                    r,
                    sps.chroma_array_type() != 0,
                    num_ref_idx_l0_active_minus1,
                    b.then_some(num_ref_idx_l1_active_minus1),
                )?);
            }
            five_minus_max_num_merge_cand = r.ue_u8(4, "five_minus_max_num_merge_cand")?;
        }
        let slice_qp_delta = r.se_i8(-87, 77, "slice_qp_delta")?;
        // 7-54: SliceQpY = 26 + init_qp_minus26 + slice_qp_delta, from
        // -QpBdOffsetY to 51.
        let qp = i32::from(pps.init_qp_minus26)
            .checked_add(26)
            .and_then(|qp| qp.checked_add(i32::from(slice_qp_delta)))
            .ok_or(Error::Overflow("SliceQpY"))?;
        let lowest = sps
            .qp_bd_offset_y()
            .checked_neg()
            .ok_or(Error::Overflow("QpBdOffsetY"))?;
        if !(lowest..=51).contains(&qp) {
            return Err(Error::OutOfRange("SliceQpY"));
        }
        let mut slice_cb_qp_offset = 0;
        let mut slice_cr_qp_offset = 0;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            slice_cb_qp_offset = r.se_i8(-12, 12, "slice_cb_qp_offset")?;
            slice_cr_qp_offset = r.se_i8(-12, 12, "slice_cr_qp_offset")?;
            // 7.4.7.1: the PPS's offset and the slice's, -12 to 12
            // together; a sum past a byte is past them.
            let within = |pps: i8, slice: i8| {
                pps.checked_add(slice)
                    .is_some_and(|sum| (-12..=12).contains(&sum))
            };
            if !within(pps.pps_cb_qp_offset, slice_cb_qp_offset) {
                return Err(Error::OutOfRange("pps_cb_qp_offset + slice_cb_qp_offset"));
            }
            if !within(pps.pps_cr_qp_offset, slice_cr_qp_offset) {
                return Err(Error::OutOfRange("pps_cr_qp_offset + slice_cr_qp_offset"));
            }
        }
        let override_deblocking = pps.deblocking_filter_override_enabled_flag && r.bit()?;
        let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
        let mut slice_beta_offset_div2 = pps.pps_beta_offset_div2;
        let mut slice_tc_offset_div2 = pps.pps_tc_offset_div2;
        if override_deblocking {
            slice_deblocking_filter_disabled_flag = r.bit()?;
            if !slice_deblocking_filter_disabled_flag {
                slice_beta_offset_div2 = r.se_i8(-6, 6, "slice_beta_offset_div2")?;
                slice_tc_offset_div2 = r.se_i8(-6, 6, "slice_tc_offset_div2")?;
            }
        }
        let mut slice_loop_filter_across_slices_enabled_flag =
            pps.pps_loop_filter_across_slices_enabled_flag;
        if pps.pps_loop_filter_across_slices_enabled_flag
            && (slice_sao_luma_flag
                || slice_sao_chroma_flag
                || !slice_deblocking_filter_disabled_flag)
        {
            slice_loop_filter_across_slices_enabled_flag = r.bit()?;
        }
        Ok(Self {
            nal_unit_type: unit_type,
            temporal_id: 0,
            sps: sps.clone(),
            pps: pps.clone(),
            first_slice_segment_in_pic_flag: false,
            no_output_of_prior_pics_flag: false,
            dependent_slice_segment_flag: false,
            slice_segment_address: 0,
            slice_type,
            pic_output_flag,
            colour_plane_id,
            slice_pic_order_cnt_lsb,
            short_term_ref_pic_set_sps_flag,
            short_term_rps,
            st_rps_bits,
            st_rps_reference_delta_pocs,
            long_term,
            slice_temporal_mvp_enabled_flag,
            slice_sao_luma_flag,
            slice_sao_chroma_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            list_entries,
            mvd_l1_zero_flag,
            cabac_init_flag,
            collocated_from_l0_flag,
            collocated_ref_idx,
            pred_weight_table,
            five_minus_max_num_merge_cand,
            slice_qp_delta,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            slice_deblocking_filter_disabled_flag,
            slice_beta_offset_div2,
            slice_tc_offset_div2,
            slice_loop_filter_across_slices_enabled_flag,
            num_entry_point_offsets: 0,
            num_pic_total_curr,
            header_bytes: 0,
            emulation_prevention_bytes: 0,
        })
    }

    /// The number of active references in list `list` for this slice's
    /// type: 0 for a list the type does not use.
    pub(crate) fn active_references(&self, list: usize) -> usize {
        match (self.slice_type, list) {
            (SliceType::I, _) | (SliceType::P, 1) => 0,
            (_, 0) => usize::from(self.num_ref_idx_l0_active_minus1) + 1,
            _ => usize::from(self.num_ref_idx_l1_active_minus1) + 1,
        }
    }
}

/// The long-term half of a slice header's reference picture set.
fn parse_long_term(
    r: &mut BitReader<'_>,
    sps: &Sps,
    lsb_bits: u32,
) -> Result<Vec<LongTermPicture>, Error> {
    let candidates = &sps.long_term_ref_pics;
    let from_sps = if candidates.is_empty() {
        0
    } else {
        r.ue_max(
            u32::try_from(candidates.len()).unwrap_or(0),
            "num_long_term_sps",
        )?
    };
    // 7.4.7.1: 32 long-term pictures at most.
    let left = 32u32
        .checked_sub(from_sps)
        .ok_or(Error::OutOfRange("num_long_term_sps"))?;
    let own = r.ue_max(left, "num_long_term_pics")?;
    let total = from_sps
        .checked_add(own)
        .ok_or(Error::Overflow("num_long_term_pics"))?;
    let mut pictures: Vec<LongTermPicture> = Vec::new();
    // DeltaPocMsbCycleLt[i - 1], with an absent delta_poc_msb_cycle_lt
    // inferred to be 0 (7.4.7.1).
    let mut cycle = 0u32;
    for i in 0..total {
        let (poc_lsb, used_by_current) = if i < from_sps {
            let index = if candidates.len() > 1 {
                r.bits(ceil_log2(candidates.len()))?
            } else {
                0
            };
            *candidates
                .get(index.to_usize())
                .ok_or(Error::OutOfRange("lt_idx_sps"))?
        } else {
            (r.bits(lsb_bits)?, r.bit()?)
        };
        let present = r.bit()?;
        let delta = if present { r.ue()? } else { 0 };
        // 7-52: the cycles accumulate within each of the two groups. The
        // standard's sum is unbounded; past 32 bits it saturates, and the
        // order count it gives is no picture's either way (`Dpb::find`).
        cycle = if i == 0 || i == from_sps {
            delta
        } else {
            delta.saturating_add(cycle)
        };
        let delta_poc_msb_cycle = present.then_some(cycle);
        pictures.push(LongTermPicture {
            poc_lsb,
            used_by_current,
            delta_poc_msb_cycle,
        });
    }
    Ok(pictures)
}

/// `pred_weight_table()` for list 0 of `l0_minus1 + 1` references, and
/// for a B slice list 1 of `l1_minus1 + 1`.
/// Each entry's flags are coded: the condition on a reference with the
/// current picture's order count only matters to multi-layer and screen
/// content streams, which are refused.
fn parse_pred_weight_table(
    r: &mut BitReader<'_>,
    chroma: bool,
    l0_minus1: u8,
    l1_minus1: Option<u8>,
) -> Result<PredWeightTable, Error> {
    let luma_log2_weight_denom = r.ue_u8(7, "luma_log2_weight_denom")?;
    let mut delta_chroma_log2_weight_denom = 0;
    if chroma {
        // 7.4.7.3: ChromaLog2WeightDenom is 0 to 7.
        let denom = luma_log2_weight_denom.cast_signed();
        let overflow = || Error::Overflow("ChromaLog2WeightDenom");
        let lowest = denom.checked_neg().ok_or_else(overflow)?;
        let highest = 7i8.checked_sub(denom).ok_or_else(overflow)?;
        delta_chroma_log2_weight_denom =
            r.se_i8(lowest, highest, "delta_chroma_log2_weight_denom")?;
    }
    let mut table = PredWeightTable {
        luma_log2_weight_denom,
        delta_chroma_log2_weight_denom,
        entries: [Vec::new(), Vec::new()],
    };
    // 7.4.7.3: sumWeightL0Flags, plus sumWeightL1Flags for a B slice, 24
    // at most.
    let mut weight_flags = 0;
    let counts = [Some(l0_minus1), l1_minus1];
    for (entries, count) in table.entries.iter_mut().zip(counts) {
        let Some(count) = count else { continue };
        let n = usize::from(count) + 1;
        let mut luma_flags = Vec::with_capacity(n);
        for _ in 0..n {
            luma_flags.push(r.bit()?);
        }
        let mut chroma_flags = vec![false; n];
        if chroma {
            for flag in &mut chroma_flags {
                *flag = r.bit()?;
            }
        }
        // A sum past usize is past 24.
        let luma_count = luma_flags.iter().filter(|&&flag| flag).count();
        let chroma_count = chroma_flags.iter().filter(|&&flag| flag).count();
        weight_flags = chroma_count
            .checked_mul(2)
            .and_then(|chroma| chroma.checked_add(luma_count))
            .and_then(|flags| flags.checked_add(weight_flags))
            .filter(|&flags| flags <= 24)
            .ok_or(Error::OutOfRange("sumWeightFlags"))?;
        for (&luma, &chroma) in luma_flags.iter().zip(&chroma_flags) {
            let mut entry = WeightEntry::default();
            if luma {
                entry.luma = Some((
                    r.se_i8(-128, 127, "delta_luma_weight")?,
                    r.se_i8(-128, 127, "luma_offset")?,
                ));
            }
            if chroma {
                let mut pair = [(0, 0); 2];
                for component in &mut pair {
                    *component = (
                        r.se_i8(-128, 127, "delta_chroma_weight")?,
                        r.se_i16(-512, 511, "delta_chroma_offset")?,
                    );
                }
                entry.chroma = Some(pair);
            }
            entries.push(entry);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_offsets_follow_7_56() {
        let table = PredWeightTable {
            luma_log2_weight_denom: 6,
            delta_chroma_log2_weight_denom: 0,
            entries: [
                vec![WeightEntry {
                    luma: None,
                    chroma: Some([(0, 0), (-32, 10)]),
                }],
                Vec::new(),
            ],
        };
        // Weight 64: 128 - 128 + 0 = 0. Weight 32: 128 - 64 + 10 = 74.
        assert_eq!(table.chroma_offset(0, 0, 0), 0);
        assert_eq!(table.chroma_offset(0, 0, 1), 74);
        assert_eq!(table.chroma_offset(0, 1, 0), 0);
    }
}
