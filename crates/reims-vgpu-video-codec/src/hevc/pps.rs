//! The picture parameter set (H.265 7.3.2.3) and its tiles (6.5.1).

use std::num::NonZeroU32;

use super::Error;
use super::nal::{NalUnit, payload_reader};
use super::scaling::ScalingLists;
use super::sps::Sps;

/// The most tile columns and rows any level allows (Table A.8: 20 and
/// 22). VA-API's picture parameters carry as many
/// (`column_width_minus1[19]`, `row_height_minus1[21]`, va_dec_hevc.h).
const MAX_TILE_COLUMNS: u8 = 20;
const MAX_TILE_ROWS: u8 = 22;

/// A picture parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    /// `pps_pic_parameter_set_id`.
    pub pps_pic_parameter_set_id: u8,
    /// `pps_seq_parameter_set_id`.
    pub pps_seq_parameter_set_id: u8,
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments_enabled_flag: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present_flag: bool,
    /// `num_extra_slice_header_bits`.
    pub num_extra_slice_header_bits: u8,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_data_hiding_enabled_flag: bool,
    /// `cabac_init_present_flag`.
    pub cabac_init_present_flag: bool,
    /// `num_ref_idx_l0_default_active_minus1`.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// `num_ref_idx_l1_default_active_minus1`.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// `init_qp_minus26`.
    pub init_qp_minus26: i8,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled_flag: bool,
    /// `diff_cu_qp_delta_depth`.
    pub diff_cu_qp_delta_depth: u8,
    /// `pps_cb_qp_offset`.
    pub pps_cb_qp_offset: i8,
    /// `pps_cr_qp_offset`.
    pub pps_cr_qp_offset: i8,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_flag`.
    pub weighted_bipred_flag: bool,
    /// `transquant_bypass_enabled_flag`.
    pub transquant_bypass_enabled_flag: bool,
    /// `tiles_enabled_flag`.
    pub tiles_enabled_flag: bool,
    /// `entropy_coding_sync_enabled_flag`: wavefront parallel processing.
    pub entropy_coding_sync_enabled_flag: bool,
    /// `num_tile_columns_minus1`.
    pub num_tile_columns_minus1: u8,
    /// `num_tile_rows_minus1`.
    pub num_tile_rows_minus1: u8,
    /// `uniform_spacing_flag` (1 when there are no tiles).
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[]`, when not uniform.
    pub(crate) column_width_minus1: Vec<u32>,
    /// `row_height_minus1[]`, when not uniform.
    pub(crate) row_height_minus1: Vec<u32>,
    /// `loop_filter_across_tiles_enabled_flag` (1 when absent).
    pub loop_filter_across_tiles_enabled_flag: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// `deblocking_filter_override_enabled_flag`.
    pub deblocking_filter_override_enabled_flag: bool,
    /// `pps_deblocking_filter_disabled_flag`.
    pub pps_deblocking_filter_disabled_flag: bool,
    /// `pps_beta_offset_div2`.
    pub pps_beta_offset_div2: i8,
    /// `pps_tc_offset_div2`.
    pub pps_tc_offset_div2: i8,
    /// The PPS's own lists, when `pps_scaling_list_data_present_flag`.
    pub scaling_lists: Option<ScalingLists>,
    /// `lists_modification_present_flag`.
    pub lists_modification_present_flag: bool,
    /// `log2_parallel_merge_level_minus2`.
    pub log2_parallel_merge_level_minus2: u8,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_segment_header_extension_present_flag: bool,
    /// `pps_range_extension_flag`.
    pub(crate) pps_range_extension_flag: bool,
    /// `pps_multilayer_extension_flag`.
    pub(crate) pps_multilayer_extension_flag: bool,
    /// `pps_3d_extension_flag`.
    pub(crate) pps_3d_extension_flag: bool,
    /// `pps_scc_extension_flag`.
    pub(crate) pps_scc_extension_flag: bool,
}

impl Pps {
    /// Parses a PPS NAL unit, header included. Its SPS is not needed to
    /// read it: what depends on the SPS is derived when a picture uses it.
    pub fn parse(unit: &[u8]) -> Result<Self, Error> {
        let nal = NalUnit::new(unit)?;
        let mut r = payload_reader(&nal)?;
        let pps_pic_parameter_set_id = r.ue_u8(63, "pps_pic_parameter_set_id")?;
        let pps_seq_parameter_set_id = r.ue_u8(15, "pps_seq_parameter_set_id")?;
        let dependent_slice_segments_enabled_flag = r.bit()?;
        let output_flag_present_flag = r.bit()?;
        let num_extra_slice_header_bits = r.u8(3)?;
        let sign_data_hiding_enabled_flag = r.bit()?;
        let cabac_init_present_flag = r.bit()?;
        let num_ref_idx_l0_default_active_minus1 =
            r.ue_u8(14, "num_ref_idx_l0_default_active_minus1")?;
        let num_ref_idx_l1_default_active_minus1 =
            r.ue_u8(14, "num_ref_idx_l1_default_active_minus1")?;
        // -(26 + QpBdOffsetY) to 25: QpBdOffsetY is 48 at most (16 bits).
        let init_qp_minus26 = r.se_i8(-74, 25, "init_qp_minus26")?;
        let constrained_intra_pred_flag = r.bit()?;
        let transform_skip_enabled_flag = r.bit()?;
        let cu_qp_delta_enabled_flag = r.bit()?;
        let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
            r.ue_u8(3, "diff_cu_qp_delta_depth")?
        } else {
            0
        };
        let pps_cb_qp_offset = r.se_i8(-12, 12, "pps_cb_qp_offset")?;
        let pps_cr_qp_offset = r.se_i8(-12, 12, "pps_cr_qp_offset")?;
        let pps_slice_chroma_qp_offsets_present_flag = r.bit()?;
        let weighted_pred_flag = r.bit()?;
        let weighted_bipred_flag = r.bit()?;
        let transquant_bypass_enabled_flag = r.bit()?;
        let tiles_enabled_flag = r.bit()?;
        let entropy_coding_sync_enabled_flag = r.bit()?;
        let mut num_tile_columns_minus1 = 0;
        let mut num_tile_rows_minus1 = 0;
        let mut uniform_spacing_flag = true;
        let mut column_width_minus1 = Vec::new();
        let mut row_height_minus1 = Vec::new();
        let mut loop_filter_across_tiles_enabled_flag = true;
        if tiles_enabled_flag {
            num_tile_columns_minus1 = r.ue_u8(MAX_TILE_COLUMNS - 1, "num_tile_columns_minus1")?;
            num_tile_rows_minus1 = r.ue_u8(MAX_TILE_ROWS - 1, "num_tile_rows_minus1")?;
            uniform_spacing_flag = r.bit()?;
            if !uniform_spacing_flag {
                for _ in 0..num_tile_columns_minus1 {
                    column_width_minus1.push(r.ue_max(1 << 16, "column_width_minus1")?);
                }
                for _ in 0..num_tile_rows_minus1 {
                    row_height_minus1.push(r.ue_max(1 << 16, "row_height_minus1")?);
                }
            }
            loop_filter_across_tiles_enabled_flag = r.bit()?;
        }
        let pps_loop_filter_across_slices_enabled_flag = r.bit()?;
        let deblocking_filter_control_present_flag = r.bit()?;
        let mut deblocking_filter_override_enabled_flag = false;
        let mut pps_deblocking_filter_disabled_flag = false;
        let mut pps_beta_offset_div2 = 0;
        let mut pps_tc_offset_div2 = 0;
        if deblocking_filter_control_present_flag {
            deblocking_filter_override_enabled_flag = r.bit()?;
            pps_deblocking_filter_disabled_flag = r.bit()?;
            if !pps_deblocking_filter_disabled_flag {
                pps_beta_offset_div2 = r.se_i8(-6, 6, "pps_beta_offset_div2")?;
                pps_tc_offset_div2 = r.se_i8(-6, 6, "pps_tc_offset_div2")?;
            }
        }
        let scaling_lists = if r.bit()? {
            Some(ScalingLists::parse(&mut r)?)
        } else {
            None
        };
        let lists_modification_present_flag = r.bit()?;
        let log2_parallel_merge_level_minus2 = r.ue_u8(4, "log2_parallel_merge_level_minus2")?;
        let slice_segment_header_extension_present_flag = r.bit()?;
        let mut pps = Self {
            pps_pic_parameter_set_id,
            pps_seq_parameter_set_id,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            init_qp_minus26,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
            cu_qp_delta_enabled_flag,
            diff_cu_qp_delta_depth,
            pps_cb_qp_offset,
            pps_cr_qp_offset,
            pps_slice_chroma_qp_offsets_present_flag,
            weighted_pred_flag,
            weighted_bipred_flag,
            transquant_bypass_enabled_flag,
            tiles_enabled_flag,
            entropy_coding_sync_enabled_flag,
            num_tile_columns_minus1,
            num_tile_rows_minus1,
            uniform_spacing_flag,
            column_width_minus1,
            row_height_minus1,
            loop_filter_across_tiles_enabled_flag,
            pps_loop_filter_across_slices_enabled_flag,
            deblocking_filter_control_present_flag,
            deblocking_filter_override_enabled_flag,
            pps_deblocking_filter_disabled_flag,
            pps_beta_offset_div2,
            pps_tc_offset_div2,
            scaling_lists,
            lists_modification_present_flag,
            log2_parallel_merge_level_minus2,
            slice_segment_header_extension_present_flag,
            pps_range_extension_flag: false,
            pps_multilayer_extension_flag: false,
            pps_3d_extension_flag: false,
            pps_scc_extension_flag: false,
        };
        if r.bit()? {
            pps.pps_range_extension_flag = r.bit()?;
            pps.pps_multilayer_extension_flag = r.bit()?;
            pps.pps_3d_extension_flag = r.bit()?;
            pps.pps_scc_extension_flag = r.bit()?;
        }
        Ok(pps)
    }

    /// `colWidth[]`, each tile column's width in CTBs (6-3), for `sps`'s
    /// picture: uniform, or as coded with the last column taking what is
    /// left. One column when there are no tiles.
    pub fn column_widths(&self, sps: &Sps) -> Result<Vec<u32>, Error> {
        tile_sizes(
            sps.pic_width_in_ctbs(),
            u32::from(self.num_tile_columns_minus1) + 1,
            self.uniform_spacing_flag,
            &self.column_width_minus1,
            "column_width_minus1",
        )
    }

    /// `rowHeight[]`, each tile row's height in CTBs (6-4).
    pub fn row_heights(&self, sps: &Sps) -> Result<Vec<u32>, Error> {
        tile_sizes(
            sps.pic_height_in_ctbs(),
            u32::from(self.num_tile_rows_minus1) + 1,
            self.uniform_spacing_flag,
            &self.row_height_minus1,
            "row_height_minus1",
        )
    }
}

/// 6.5.1's tile sizes along one dimension of `total` CTBs.
fn tile_sizes(
    total: u32,
    count: u32,
    uniform: bool,
    coded_minus1: &[u32],
    name: &'static str,
) -> Result<Vec<u32>, Error> {
    if count > total {
        return Err(Error::OutOfRange(name));
    }
    if uniform {
        // (6-3), (6-4): ((i + 1) * total) / count - (i * total) / count.
        let Some(divisor) = NonZeroU32::new(count) else {
            return Ok(Vec::new());
        };
        let start = |i: u32| i.checked_mul(total).map(|ctbs| ctbs / divisor);
        return (0..count)
            .map(|i| start(i.checked_add(1)?)?.checked_sub(start(i)?))
            .collect::<Option<Vec<u32>>>()
            .ok_or(Error::Overflow("the uniform tile sizes"));
    }
    // The coded sizes, then what they leave, which is at least one CTB:
    // sizes past 32 bits are past the picture.
    let mut sizes = coded_minus1
        .iter()
        .map(|&minus1| minus1.checked_add(1))
        .collect::<Option<Vec<u32>>>()
        .ok_or(Error::OutOfRange(name))?;
    let left = sizes
        .iter()
        .try_fold(0u32, |used, &size| used.checked_add(size))
        .and_then(|used| total.checked_sub(used))
        .filter(|&left| left > 0)
        .ok_or(Error::OutOfRange(name))?;
    sizes.push(left);
    Ok(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_tiles_split_as_6_3_says() -> Result<(), Error> {
        // 30 CTBs in 4 columns: 7, 8, 7, 8.
        assert_eq!(tile_sizes(30, 4, true, &[], "c")?, [7, 8, 7, 8]);
        assert_eq!(tile_sizes(30, 1, true, &[], "c")?, [30]);
        // Coded: 5 and 10, the last takes the 15 left.
        assert_eq!(tile_sizes(30, 3, false, &[4, 9], "c")?, [5, 10, 15]);
        assert!(tile_sizes(30, 3, false, &[19, 9], "c").is_err());
        assert!(tile_sizes(2, 3, true, &[], "c").is_err());
        Ok(())
    }
}
