//! The slice header (H.264 7.3.3) with its reference list modification
//! (7.3.3.1), prediction weight table (7.3.3.2) and reference picture
//! marking (7.3.3.3).

use std::sync::Arc;

use super::Error;
use super::bits::BitReader;
use super::nal::{NalUnit, nal_type};
use super::pps::Pps;
use super::sps::Sps;

/// `slice_type % 5` (Table 7-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    /// P.
    P,
    /// B.
    B,
    /// I.
    I,
    /// SP.
    Sp,
    /// SI.
    Si,
}

impl SliceType {
    fn from_raw(raw: u32) -> Self {
        match raw % 5 {
            0 => Self::P,
            1 => Self::B,
            2 => Self::I,
            3 => Self::Sp,
            _ => Self::Si,
        }
    }

    /// The value VA-API's `slice_type` takes: `slice_type % 5`.
    pub fn raw(self) -> u8 {
        match self {
            Self::P => 0,
            Self::B => 1,
            Self::I => 2,
            Self::Sp => 3,
            Self::Si => 4,
        }
    }

    /// Whether slices of this type use list 0.
    pub fn is_inter(self) -> bool {
        matches!(self, Self::P | Self::Sp | Self::B)
    }
}

/// One `modification_of_pic_nums_idc` operation (7.4.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListModification {
    /// idc 0: `abs_diff_pic_num_minus1`, subtracted.
    ShortTermSubtract(u32),
    /// idc 1: `abs_diff_pic_num_minus1`, added.
    ShortTermAdd(u32),
    /// idc 2: `long_term_pic_num`.
    LongTerm(u32),
}

/// One `memory_management_control_operation` (7.4.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryManagement {
    /// 1: `difference_of_pic_nums_minus1`.
    UnmarkShortTerm(u32),
    /// 2: `long_term_pic_num`.
    UnmarkLongTerm(u32),
    /// 3: `difference_of_pic_nums_minus1`, `long_term_frame_idx`.
    ShortTermToLongTerm(u32, u32),
    /// 4: `max_long_term_frame_idx_plus1`.
    MaxLongTermFrameIdx(u32),
    /// 5: everything unused for reference.
    UnmarkAll,
    /// 6: `long_term_frame_idx`, for the current picture.
    CurrentToLongTerm(u32),
}

/// `dec_ref_pic_marking()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefPicMarking {
    /// `no_output_of_prior_pics_flag` (IDR).
    pub(crate) no_output_of_prior_pics_flag: bool,
    /// `long_term_reference_flag` (IDR).
    pub long_term_reference_flag: bool,
    /// The operations, when `adaptive_ref_pic_marking_mode_flag`.
    pub adaptive: Option<Vec<MemoryManagement>>,
}

/// The weights of one reference: luma weight and offset, then the two
/// chroma weights and offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weights {
    /// Whether `luma_weight_lX_flag` was set.
    pub luma_present: bool,
    /// `luma_weight_lX`, `luma_offset_lX`.
    pub luma: (i16, i16),
    /// Whether `chroma_weight_lX_flag` was set.
    pub chroma_present: bool,
    /// `chroma_weight_lX`, `chroma_offset_lX` for Cb and Cr.
    pub chroma: [(i16, i16); 2],
}

/// `pred_weight_table()`, with the weights 7.4.3.2 infers where a flag is
/// 0 filled in, as VA-API wants them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_weight_denom: u8,
    /// `chroma_log2_weight_denom`.
    pub chroma_log2_weight_denom: u8,
    /// One entry per active reference of list 0, then of list 1.
    pub lists: [Vec<Weights>; 2],
}

/// A parsed slice header, with what a decoder needs from its NAL unit.
#[derive(Clone, Debug)]
pub struct SliceHeader {
    /// `nal_ref_idc`.
    pub nal_ref_idc: u8,
    /// `IdrPicFlag`.
    pub idr: bool,
    /// `first_mb_in_slice`.
    pub first_mb_in_slice: u32,
    /// `slice_type`.
    pub slice_type: SliceType,
    /// `pic_parameter_set_id`.
    pub(crate) pic_parameter_set_id: u8,
    /// `frame_num`.
    pub frame_num: u32,
    /// `field_pic_flag`.
    pub field_pic_flag: bool,
    /// `bottom_field_flag`.
    pub bottom_field_flag: bool,
    /// `idr_pic_id`.
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb`.
    pub(crate) pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom`.
    pub(crate) delta_pic_order_cnt_bottom: i32,
    /// `delta_pic_order_cnt[0..2]`.
    pub(crate) delta_pic_order_cnt: [i32; 2],
    /// `redundant_pic_cnt`.
    pub(crate) redundant_pic_cnt: u32,
    /// `direct_spatial_mv_pred_flag`.
    pub direct_spatial_mv_pred_flag: bool,
    /// `num_ref_idx_l0_active_minus1`, the PPS's default when not
    /// overridden.
    pub num_ref_idx_l0_active_minus1: u8,
    /// `num_ref_idx_l1_active_minus1`, likewise.
    pub num_ref_idx_l1_active_minus1: u8,
    /// The list modifications for lists 0 and 1.
    pub list_modifications: [Vec<ListModification>; 2],
    /// The prediction weights, when the PPS asks for explicit ones.
    pub pred_weight_table: Option<PredWeightTable>,
    /// The marking, for a reference picture.
    pub ref_pic_marking: RefPicMarking,
    /// `cabac_init_idc`.
    pub cabac_init_idc: u8,
    /// `slice_qp_delta`.
    pub slice_qp_delta: i8,
    /// `disable_deblocking_filter_idc`.
    pub disable_deblocking_filter_idc: u8,
    /// `slice_alpha_c0_offset_div2`.
    pub slice_alpha_c0_offset_div2: i8,
    /// `slice_beta_offset_div2`.
    pub slice_beta_offset_div2: i8,
    /// The header's length in RBSP bits, the NAL header byte included:
    /// where `slice_data()` starts.
    pub header_bits: usize,
    /// The parameter sets the slice uses.
    pub(crate) pps: Arc<Pps>,
    /// The SPS its PPS names.
    pub(crate) sps: Arc<Sps>,
}

impl SliceHeader {
    /// Parses the header of a slice NAL unit (type 1 or 5), looking its
    /// parameter sets up with `pps_by_id` and `sps_by_id`.
    pub fn parse(
        nal: NalUnit<'_>,
        pps_by_id: impl Fn(u8) -> Option<Arc<Pps>>,
        sps_by_id: impl Fn(u8) -> Option<Arc<Sps>>,
    ) -> Result<Self, Error> {
        let idr = nal.unit_type() == nal_type::IDR;
        let nal_ref_idc = nal.ref_idc();
        let mut r = BitReader::new(nal.bytes());
        r.skip(8)?;
        let first_mb_in_slice = r.ue()?;
        let slice_type = SliceType::from_raw(r.ue_max(9, "slice_type")?);
        if idr && !matches!(slice_type, SliceType::I | SliceType::Si) {
            return Err(Error::OutOfRange("slice_type of an IDR picture"));
        }
        let pic_parameter_set_id = r.ue_u8(255, "pic_parameter_set_id")?;
        let pps = pps_by_id(pic_parameter_set_id).ok_or(Error::MissingPps(pic_parameter_set_id))?;
        let sps = sps_by_id(pps.seq_parameter_set_id)
            .ok_or(Error::MissingSps(pps.seq_parameter_set_id))?;
        if first_mb_in_slice >= sps.frame_size_in_mbs() {
            return Err(Error::OutOfRange("first_mb_in_slice"));
        }
        if pps.num_slice_groups_minus1 > 0 {
            return Err(Error::Unsupported("slice groups (FMO)"));
        }
        if sps.separate_colour_plane_flag {
            // colour_plane_id: refused with 4:4:4 when the SPS activates,
            // but read so the rest of the header is right.
            r.skip(2)?;
        }
        let frame_num = r.bits(u32::from(sps.log2_max_frame_num_minus4) + 4)?;
        let mut field_pic_flag = false;
        let mut bottom_field_flag = false;
        if !sps.frame_mbs_only_flag {
            field_pic_flag = r.bit()?;
            if field_pic_flag {
                bottom_field_flag = r.bit()?;
            }
        }
        let idr_pic_id = if idr {
            r.ue_max(65535, "idr_pic_id")?
        } else {
            0
        };
        let mut pic_order_cnt_lsb = 0;
        let mut delta_pic_order_cnt_bottom = 0;
        let mut delta_pic_order_cnt = [0; 2];
        if sps.pic_order_cnt_type == 0 {
            pic_order_cnt_lsb = r.bits(u32::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4)?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt_bottom =
                    r.se_range(i32::MIN + 1, i32::MAX, "delta_pic_order_cnt_bottom")?;
            }
        }
        if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
            delta_pic_order_cnt[0] = r.se_range(i32::MIN + 1, i32::MAX, "delta_pic_order_cnt")?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt[1] =
                    r.se_range(i32::MIN + 1, i32::MAX, "delta_pic_order_cnt")?;
            }
        }
        let redundant_pic_cnt = if pps.redundant_pic_cnt_present_flag {
            r.ue_max(127, "redundant_pic_cnt")?
        } else {
            0
        };
        let direct_spatial_mv_pred_flag = slice_type == SliceType::B && r.bit()?;
        let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        if slice_type.is_inter() && r.bit()? {
            let max = if field_pic_flag { 31 } else { 15 };
            num_ref_idx_l0_active_minus1 = r.ue_u8(max, "num_ref_idx_l0_active_minus1")?;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active_minus1 = r.ue_u8(max, "num_ref_idx_l1_active_minus1")?;
            }
        }
        // A frame's lists hold 16 at most (7.4.3); a PPS default above that
        // applies to fields only.
        if !field_pic_flag
            && ((slice_type.is_inter() && num_ref_idx_l0_active_minus1 > 15)
                || (slice_type == SliceType::B && num_ref_idx_l1_active_minus1 > 15))
        {
            return Err(Error::OutOfRange("num_ref_idx_active_minus1"));
        }
        let l0_modifications = if slice_type.is_inter() {
            parse_list_modification(&mut r, num_ref_idx_l0_active_minus1)?
        } else {
            Vec::new()
        };
        let l1_modifications = if slice_type == SliceType::B {
            parse_list_modification(&mut r, num_ref_idx_l1_active_minus1)?
        } else {
            Vec::new()
        };
        let list_modifications = [l0_modifications, l1_modifications];
        let explicit_weights = (pps.weighted_pred_flag
            && matches!(slice_type, SliceType::P | SliceType::Sp))
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
        let pred_weight_table = if explicit_weights {
            let l1 = if slice_type == SliceType::B {
                Some(num_ref_idx_l1_active_minus1)
            } else {
                None
            };
            Some(parse_pred_weight_table(
                &mut r,
                sps.chroma_array_type(),
                num_ref_idx_l0_active_minus1,
                l1,
            )?)
        } else {
            None
        };
        let ref_pic_marking = if nal_ref_idc != 0 {
            parse_ref_pic_marking(&mut r, idr)?
        } else {
            RefPicMarking::default()
        };
        let cabac_init_idc = if pps.entropy_coding_mode_flag
            && !matches!(slice_type, SliceType::I | SliceType::Si)
        {
            r.ue_u8(2, "cabac_init_idc")?
        } else {
            0
        };
        let slice_qp_delta = r.se_i8(-87, 77, "slice_qp_delta")?;
        if matches!(slice_type, SliceType::Sp | SliceType::Si) {
            if slice_type == SliceType::Sp {
                r.skip(1)?; // sp_for_switch_flag
            }
            r.se()?; // slice_qs_delta
        }
        let mut disable_deblocking_filter_idc = 0;
        let mut slice_alpha_c0_offset_div2 = 0;
        let mut slice_beta_offset_div2 = 0;
        if pps.deblocking_filter_control_present_flag {
            disable_deblocking_filter_idc = r.ue_u8(2, "disable_deblocking_filter_idc")?;
            if disable_deblocking_filter_idc != 1 {
                slice_alpha_c0_offset_div2 = r.se_i8(-6, 6, "slice_alpha_c0_offset_div2")?;
                slice_beta_offset_div2 = r.se_i8(-6, 6, "slice_beta_offset_div2")?;
            }
        }
        Ok(Self {
            nal_ref_idc,
            idr,
            first_mb_in_slice,
            slice_type,
            pic_parameter_set_id,
            frame_num,
            field_pic_flag,
            bottom_field_flag,
            idr_pic_id,
            pic_order_cnt_lsb,
            delta_pic_order_cnt_bottom,
            delta_pic_order_cnt,
            redundant_pic_cnt,
            direct_spatial_mv_pred_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            list_modifications,
            pred_weight_table,
            ref_pic_marking,
            cabac_init_idc,
            slice_qp_delta,
            disable_deblocking_filter_idc,
            slice_alpha_c0_offset_div2,
            slice_beta_offset_div2,
            header_bits: r.position(),
            pps,
            sps,
        })
    }

    /// Whether this slice starts a new primary picture after `previous`,
    /// by the comparisons of 7.4.1.2.4.
    pub fn starts_new_picture(&self, previous: &SliceHeader) -> bool {
        let sps = &self.sps;
        self.frame_num != previous.frame_num
            || self.pic_parameter_set_id != previous.pic_parameter_set_id
            || self.field_pic_flag != previous.field_pic_flag
            || self.bottom_field_flag != previous.bottom_field_flag
            || (self.nal_ref_idc == 0) != (previous.nal_ref_idc == 0)
            || (sps.pic_order_cnt_type == 0
                && (self.pic_order_cnt_lsb != previous.pic_order_cnt_lsb
                    || self.delta_pic_order_cnt_bottom != previous.delta_pic_order_cnt_bottom))
            || (sps.pic_order_cnt_type == 1
                && self.delta_pic_order_cnt != previous.delta_pic_order_cnt)
            || self.idr != previous.idr
            || (self.idr && self.idr_pic_id != previous.idr_pic_id)
    }
}

fn parse_list_modification(
    r: &mut BitReader<'_>,
    num_active_minus1: u8,
) -> Result<Vec<ListModification>, Error> {
    let mut ops = Vec::new();
    if !r.bit()? {
        return Ok(ops);
    }
    loop {
        let op = match r.ue_max(5, "modification_of_pic_nums_idc")? {
            0 => ListModification::ShortTermSubtract(r.ue()?),
            1 => ListModification::ShortTermAdd(r.ue()?),
            2 => ListModification::LongTerm(r.ue()?),
            3 => break,
            _ => return Err(Error::Unsupported("multiview list modification")),
        };
        // At most one operation per entry of the list (7.4.3.1).
        if ops.len() > usize::from(num_active_minus1) {
            return Err(Error::OutOfRange("ref_pic_list_modification"));
        }
        ops.push(op);
    }
    Ok(ops)
}

fn parse_pred_weight_table(
    r: &mut BitReader<'_>,
    chroma_array_type: u8,
    l0_minus1: u8,
    l1_minus1: Option<u8>,
) -> Result<PredWeightTable, Error> {
    let luma_log2_weight_denom = r.ue_u8(7, "luma_log2_weight_denom")?;
    let chroma_log2_weight_denom = if chroma_array_type != 0 {
        r.ue_u8(7, "chroma_log2_weight_denom")?
    } else {
        0
    };
    let luma_default = 1i16 << luma_log2_weight_denom;
    let chroma_default = 1i16 << chroma_log2_weight_denom;
    let mut lists = [Vec::new(), Vec::new()];
    for (list, count) in lists.iter_mut().zip([Some(l0_minus1), l1_minus1]) {
        let Some(minus1) = count else { continue };
        for _ in 0..=minus1 {
            let mut weights = Weights {
                luma_present: false,
                luma: (luma_default, 0),
                chroma_present: false,
                chroma: [(chroma_default, 0); 2],
            };
            if r.bit()? {
                weights.luma_present = true;
                weights.luma = (weight(r, "luma_weight")?, weight(r, "luma_offset")?);
            }
            if chroma_array_type != 0 && r.bit()? {
                weights.chroma_present = true;
                for c in &mut weights.chroma {
                    *c = (weight(r, "chroma_weight")?, weight(r, "chroma_offset")?);
                }
            }
            list.push(weights);
        }
    }
    Ok(PredWeightTable {
        luma_log2_weight_denom,
        chroma_log2_weight_denom,
        lists,
    })
}

/// A weight or an offset: -128 to 127 for 8-bit video (7.4.3.2).
fn weight(r: &mut BitReader<'_>, name: &'static str) -> Result<i16, Error> {
    r.se_i16(-128, 127, name)
}

fn parse_ref_pic_marking(r: &mut BitReader<'_>, idr: bool) -> Result<RefPicMarking, Error> {
    if idr {
        return Ok(RefPicMarking {
            no_output_of_prior_pics_flag: r.bit()?,
            long_term_reference_flag: r.bit()?,
            adaptive: None,
        });
    }
    if !r.bit()? {
        return Ok(RefPicMarking::default());
    }
    let mut ops = Vec::new();
    loop {
        let op = match r.ue_max(6, "memory_management_control_operation")? {
            0 => break,
            1 => MemoryManagement::UnmarkShortTerm(r.ue()?),
            2 => MemoryManagement::UnmarkLongTerm(r.ue()?),
            3 => {
                MemoryManagement::ShortTermToLongTerm(r.ue()?, r.ue_max(15, "long_term_frame_idx")?)
            }
            4 => MemoryManagement::MaxLongTermFrameIdx(
                r.ue_max(16, "max_long_term_frame_idx_plus1")?,
            ),
            5 => MemoryManagement::UnmarkAll,
            _ => MemoryManagement::CurrentToLongTerm(r.ue_max(15, "long_term_frame_idx")?),
        };
        // Each frame in a 16-frame DPB can be named a few ways; more than
        // this is not a stream but noise.
        if ops.len() >= 66 {
            return Err(Error::OutOfRange("dec_ref_pic_marking"));
        }
        ops.push(op);
    }
    Ok(RefPicMarking {
        no_output_of_prior_pics_flag: false,
        long_term_reference_flag: false,
        adaptive: Some(ops),
    })
}
