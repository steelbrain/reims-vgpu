//! The picture parameter set (H.264 7.3.2.2).

use std::sync::Arc;

use super::Error;
use super::bits::{BitReader, ceil_log2};
use super::scaling::{ScalingLists, ScalingSyntax};
use super::sps::Sps;

/// A picture parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    /// `pic_parameter_set_id`.
    pub pic_parameter_set_id: u8,
    /// `seq_parameter_set_id`.
    pub seq_parameter_set_id: u8,
    /// `entropy_coding_mode_flag`: CABAC.
    pub entropy_coding_mode_flag: bool,
    /// `bottom_field_pic_order_in_frame_present_flag`.
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    /// `num_slice_groups_minus1`: above 0 is FMO, which is refused at the
    /// first slice that uses this PPS.
    pub num_slice_groups_minus1: u8,
    /// `num_ref_idx_l0_default_active_minus1`.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// `num_ref_idx_l1_default_active_minus1`.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_idc`.
    pub weighted_bipred_idc: u8,
    /// `pic_init_qp_minus26`.
    pub pic_init_qp_minus26: i8,
    /// `pic_init_qs_minus26`.
    pub pic_init_qs_minus26: i8,
    /// `chroma_qp_index_offset`.
    pub chroma_qp_index_offset: i8,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `redundant_pic_cnt_present_flag`.
    pub redundant_pic_cnt_present_flag: bool,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode_flag: bool,
    /// `second_chroma_qp_index_offset` (`chroma_qp_index_offset` when not
    /// coded).
    pub second_chroma_qp_index_offset: i8,
    /// The scaling lists, when `pic_scaling_matrix_present_flag`.
    pub(crate) scaling: Option<ScalingSyntax>,
}

impl Pps {
    /// Parses a PPS NAL unit, header byte included. The SPS it names, if
    /// `sps_by_id` has it, decides how many 8x8 scaling lists the PPS
    /// codes; 4:2:0 is assumed when it does not.
    pub fn parse(nal: &[u8], sps_by_id: impl Fn(u8) -> Option<Arc<Sps>>) -> Result<Self, Error> {
        let mut r = BitReader::new(nal);
        r.skip(8)?;
        let pic_parameter_set_id = r.ue_u8(255, "pic_parameter_set_id")?;
        let seq_parameter_set_id = r.ue_u8(31, "seq_parameter_set_id")?;
        let entropy_coding_mode_flag = r.bit()?;
        let bottom_field_pic_order_in_frame_present_flag = r.bit()?;
        let num_slice_groups_minus1 = r.ue_u8(7, "num_slice_groups_minus1")?;
        if num_slice_groups_minus1 > 0 {
            skip_slice_group_map(&mut r, num_slice_groups_minus1)?;
        }
        let num_ref_idx_l0_default_active_minus1 =
            r.ue_u8(31, "num_ref_idx_l0_default_active_minus1")?;
        let num_ref_idx_l1_default_active_minus1 =
            r.ue_u8(31, "num_ref_idx_l1_default_active_minus1")?;
        let weighted_pred_flag = r.bit()?;
        let weighted_bipred_idc = r.u8(2)?;
        if weighted_bipred_idc == 3 {
            return Err(Error::OutOfRange("weighted_bipred_idc"));
        }
        // The ranges below are for 8-bit video; deeper video is refused
        // when a slice activates its SPS.
        let pic_init_qp_minus26 = r.se_i8(-26 - 36, 25, "pic_init_qp_minus26")?;
        let pic_init_qs_minus26 = r.se_i8(-26, 25, "pic_init_qs_minus26")?;
        let chroma_qp_index_offset = r.se_i8(-12, 12, "chroma_qp_index_offset")?;
        let deblocking_filter_control_present_flag = r.bit()?;
        let constrained_intra_pred_flag = r.bit()?;
        let redundant_pic_cnt_present_flag = r.bit()?;
        let mut transform_8x8_mode_flag = false;
        let mut scaling = None;
        let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
        if r.more_rbsp_data() {
            transform_8x8_mode_flag = r.bit()?;
            if r.bit()? {
                let chroma_444 =
                    sps_by_id(seq_parameter_set_id).is_some_and(|sps| sps.chroma_format_idc == 3);
                let lists_8x8 = if transform_8x8_mode_flag {
                    if chroma_444 { 6 } else { 2 }
                } else {
                    0
                };
                scaling = Some(ScalingSyntax::parse(&mut r, lists_8x8)?);
            }
            second_chroma_qp_index_offset = r.se_i8(-12, 12, "second_chroma_qp_index_offset")?;
        }
        Ok(Self {
            pic_parameter_set_id,
            seq_parameter_set_id,
            entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present_flag,
            num_slice_groups_minus1,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            weighted_pred_flag,
            weighted_bipred_idc,
            pic_init_qp_minus26,
            pic_init_qs_minus26,
            chroma_qp_index_offset,
            deblocking_filter_control_present_flag,
            constrained_intra_pred_flag,
            redundant_pic_cnt_present_flag,
            transform_8x8_mode_flag,
            second_chroma_qp_index_offset,
            scaling,
        })
    }

    /// The scaling lists in force for a picture that uses this PPS with
    /// `sps` (7.4.2.2, `pic_scaling_list_present_flag`): where the PPS codes
    /// lists, those the PPS leaves out fall back by rule A of Table 7-2
    /// when the SPS has no matrix, and by rule B, to the SPS's lists, when
    /// it has. Where the PPS codes none, the SPS's lists hold.
    pub fn scaling_lists(&self, sps: &Sps) -> ScalingLists {
        match &self.scaling {
            Some(syntax) if sps.seq_scaling_matrix_present_flag => {
                syntax.resolve(Some(&sps.scaling_lists))
            }
            Some(syntax) => syntax.resolve(None),
            None => sps.scaling_lists.clone(),
        }
    }

    /// Whether the PPS codes scaling lists of its own.
    pub fn pic_scaling_matrix_present_flag(&self) -> bool {
        self.scaling.is_some()
    }
}

/// The FMO syntax of 7.3.2.2, read past.
fn skip_slice_group_map(r: &mut BitReader<'_>, num_slice_groups_minus1: u8) -> Result<(), Error> {
    let map_type = r.ue_max(6, "slice_group_map_type")?;
    match map_type {
        0 => {
            for _ in 0..=num_slice_groups_minus1 {
                r.ue()?; // run_length_minus1
            }
        }
        2 => {
            for _ in 0..num_slice_groups_minus1 {
                r.ue()?; // top_left
                r.ue()?; // bottom_right
            }
        }
        3..=5 => {
            r.skip(1)?; // slice_group_change_direction_flag
            r.ue()?; // slice_group_change_rate_minus1
        }
        6 => {
            let units = u64::from(r.ue()?) + 1;
            let bits = ceil_log2(usize::from(num_slice_groups_minus1) + 1);
            for _ in 0..units {
                r.skip(bits)?; // slice_group_id
            }
        }
        _ => {}
    }
    Ok(())
}
