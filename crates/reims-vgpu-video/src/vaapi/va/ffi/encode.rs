//! libva's encode interface, hand-written for the subset the encoders
//! call, from libva 2.24.1's headers (`va.h`, `va_enc_h264.h`,
//! `va_enc_hevc.h`; VA-API 1.24.0). The test at the end checks every
//! struct's size and offsets, every bit-field's position and every
//! constant's value against those headers with a C compiler, as the parent
//! module's test does for decode.
//!
//! Names keep libva's spelling; struct fields are snake-cased.

#![allow(
    dead_code,
    reason = "the binding declares what the headers have beside what is called"
)]
#![expect(
    non_upper_case_globals,
    reason = "libva's names, kept for grepping the headers"
)]

use core::ffi::{c_int, c_void};

use super::{
    BitField, VA_PADDING_HIGH, VA_PADDING_LOW, VA_PADDING_MEDIUM, VABufferID, VABufferType,
    VAConfigAttribType, VAEntrypoint, VAPictureH264,
};
// What decoding's binding has too.
pub(crate) use super::{
    VA_FOURCC_P010, VA_RT_FORMAT_YUV420_10, VAPictureHEVC, VAProfileHEVCMain, VAProfileHEVCMain10,
};

// VAEntrypoint.
pub(crate) const VAEntrypointEncSlice: VAEntrypoint = 6;
pub(crate) const VAEntrypointEncSliceLP: VAEntrypoint = 8;

// VAConfigAttribType.
pub(crate) const VAConfigAttribEncSliceStructure: VAConfigAttribType = 15;
pub(crate) const VAConfigAttribEncQualityRange: VAConfigAttribType = 21;
pub(crate) const VAConfigAttribEncIntraRefresh: VAConfigAttribType = 23;
pub(crate) const VAConfigAttribEncSkipFrame: VAConfigAttribType = 24;
pub(crate) const VAConfigAttribEncROI: VAConfigAttribType = 25;
pub(crate) const VAConfigAttribEncRateControlExt: VAConfigAttribType = 26;
pub(crate) const VAConfigAttribMaxFrameSize: VAConfigAttribType = 38;
pub(crate) const VAConfigAttribPredictionDirection: VAConfigAttribType = 39;
pub(crate) const VAConfigAttribEncHEVCFeatures: VAConfigAttribType = 50;
pub(crate) const VAConfigAttribEncHEVCBlockSizes: VAConfigAttribType = 51;

// VA_RC_*.
pub(crate) const VA_RC_CBR: u32 = 0x0000_0002;
pub(crate) const VA_RC_VBR: u32 = 0x0000_0004;
pub(crate) const VA_RC_CQP: u32 = 0x0000_0010;
pub(crate) const VA_RC_QVBR: u32 = 0x0000_0400;

// VA_ENC_PACKED_HEADER_*.
pub(crate) const VA_ENC_PACKED_HEADER_SEQUENCE: u32 = 0x0000_0001;
pub(crate) const VA_ENC_PACKED_HEADER_PICTURE: u32 = 0x0000_0002;
pub(crate) const VA_ENC_PACKED_HEADER_SLICE: u32 = 0x0000_0004;
pub(crate) const VA_ENC_PACKED_HEADER_MISC: u32 = 0x0000_0008;
pub(crate) const VA_ENC_PACKED_HEADER_RAW_DATA: u32 = 0x0000_0010;

// VAEncPackedHeaderType.
pub(crate) const VAEncPackedHeaderSequence: u32 = 1;
pub(crate) const VAEncPackedHeaderPicture: u32 = 2;
pub(crate) const VAEncPackedHeaderSlice: u32 = 3;
pub(crate) const VAEncPackedHeaderRawData: u32 = 4;

// VABufferType.
pub(crate) const VAEncCodedBufferType: VABufferType = 21;
pub(crate) const VAEncSequenceParameterBufferType: VABufferType = 22;
pub(crate) const VAEncPictureParameterBufferType: VABufferType = 23;
pub(crate) const VAEncSliceParameterBufferType: VABufferType = 24;
pub(crate) const VAEncPackedHeaderParameterBufferType: VABufferType = 25;
pub(crate) const VAEncPackedHeaderDataBufferType: VABufferType = 26;
pub(crate) const VAEncMiscParameterBufferType: VABufferType = 27;

// VAEncMiscParameterType.
pub(crate) const VAEncMiscParameterTypeFrameRate: c_int = 0;
pub(crate) const VAEncMiscParameterTypeRateControl: c_int = 1;
pub(crate) const VAEncMiscParameterTypeMaxFrameSize: c_int = 4;
pub(crate) const VAEncMiscParameterTypeHRD: c_int = 5;
pub(crate) const VAEncMiscParameterTypeQualityLevel: c_int = 6;

// VA_CODED_BUF_STATUS_*.
pub(crate) const VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW: u32 = 0x1000;
pub(crate) const VA_CODED_BUF_STATUS_BAD_BITSTREAM: u32 = 0x8000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncSequenceParameterBufferH264 {
    pub(crate) seq_parameter_set_id: u8,
    pub(crate) level_idc: u8,
    pub(crate) intra_period: u32,
    pub(crate) intra_idr_period: u32,
    pub(crate) ip_period: u32,
    pub(crate) bits_per_second: u32,
    pub(crate) max_num_ref_frames: u32,
    pub(crate) picture_width_in_mbs: u16,
    pub(crate) picture_height_in_mbs: u16,
    /// `seq_fields.value`: see [`h264_seq_fields`].
    pub(crate) seq_fields: u32,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) num_ref_frames_in_pic_order_cnt_cycle: u8,
    pub(crate) offset_for_non_ref_pic: i32,
    pub(crate) offset_for_top_to_bottom_field: i32,
    pub(crate) offset_for_ref_frame: [i32; 256],
    pub(crate) frame_cropping_flag: u8,
    pub(crate) frame_crop_left_offset: u32,
    pub(crate) frame_crop_right_offset: u32,
    pub(crate) frame_crop_top_offset: u32,
    pub(crate) frame_crop_bottom_offset: u32,
    pub(crate) vui_parameters_present_flag: u8,
    /// `vui_fields.value`: see [`h264_vui_fields`].
    pub(crate) vui_fields: u32,
    pub(crate) aspect_ratio_idc: u8,
    pub(crate) sar_width: u32,
    pub(crate) sar_height: u32,
    pub(crate) num_units_in_tick: u32,
    pub(crate) time_scale: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncPictureParameterBufferH264 {
    pub(crate) curr_pic: VAPictureH264,
    pub(crate) reference_frames: [VAPictureH264; 16],
    pub(crate) coded_buf: VABufferID,
    pub(crate) pic_parameter_set_id: u8,
    pub(crate) seq_parameter_set_id: u8,
    pub(crate) last_picture: u8,
    pub(crate) frame_num: u16,
    pub(crate) pic_init_qp: u8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) chroma_qp_index_offset: i8,
    pub(crate) second_chroma_qp_index_offset: i8,
    /// `pic_fields.value`: see [`h264_pic_fields`].
    pub(crate) pic_fields: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncSliceParameterBufferH264 {
    pub(crate) macroblock_address: u32,
    pub(crate) num_macroblocks: u32,
    pub(crate) macroblock_info: VABufferID,
    pub(crate) slice_type: u8,
    pub(crate) pic_parameter_set_id: u8,
    pub(crate) idr_pic_id: u16,
    pub(crate) pic_order_cnt_lsb: u16,
    pub(crate) delta_pic_order_cnt_bottom: i32,
    pub(crate) delta_pic_order_cnt: [i32; 2],
    pub(crate) direct_spatial_mv_pred_flag: u8,
    pub(crate) num_ref_idx_active_override_flag: u8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) ref_pic_list0: [VAPictureH264; 32],
    pub(crate) ref_pic_list1: [VAPictureH264; 32],
    pub(crate) luma_log2_weight_denom: u8,
    pub(crate) chroma_log2_weight_denom: u8,
    pub(crate) luma_weight_l0_flag: u8,
    pub(crate) luma_weight_l0: [i16; 32],
    pub(crate) luma_offset_l0: [i16; 32],
    pub(crate) chroma_weight_l0_flag: u8,
    pub(crate) chroma_weight_l0: [[i16; 2]; 32],
    pub(crate) chroma_offset_l0: [[i16; 2]; 32],
    pub(crate) luma_weight_l1_flag: u8,
    pub(crate) luma_weight_l1: [i16; 32],
    pub(crate) luma_offset_l1: [i16; 32],
    pub(crate) chroma_weight_l1_flag: u8,
    pub(crate) chroma_weight_l1: [[i16; 2]; 32],
    pub(crate) chroma_offset_l1: [[i16; 2]; 32],
    pub(crate) cabac_init_idc: u8,
    pub(crate) slice_qp_delta: i8,
    pub(crate) disable_deblocking_filter_idc: u8,
    pub(crate) slice_alpha_c0_offset_div2: i8,
    pub(crate) slice_beta_offset_div2: i8,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncSequenceParameterBufferH264::seq_fields.bits`.
pub(crate) mod h264_seq_fields {
    use super::BitField;
    pub(crate) const CHROMA_FORMAT_IDC: BitField = BitField::new(0, 2);
    pub(crate) const FRAME_MBS_ONLY_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const MB_ADAPTIVE_FRAME_FIELD_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const SEQ_SCALING_MATRIX_PRESENT_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const DIRECT_8X8_INFERENCE_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const LOG2_MAX_FRAME_NUM_MINUS4: BitField = BitField::new(6, 4);
    pub(crate) const PIC_ORDER_CNT_TYPE: BitField = BitField::new(10, 2);
    pub(crate) const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: BitField = BitField::new(12, 4);
    pub(crate) const DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG: BitField = BitField::new(16, 1);
}

/// `VAEncSequenceParameterBufferH264::vui_fields.bits`.
pub(crate) mod h264_vui_fields {
    use super::BitField;
    pub(crate) const ASPECT_RATIO_INFO_PRESENT_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const TIMING_INFO_PRESENT_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const BITSTREAM_RESTRICTION_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const LOG2_MAX_MV_LENGTH_HORIZONTAL: BitField = BitField::new(3, 5);
    pub(crate) const LOG2_MAX_MV_LENGTH_VERTICAL: BitField = BitField::new(8, 5);
    pub(crate) const FIXED_FRAME_RATE_FLAG: BitField = BitField::new(13, 1);
    pub(crate) const LOW_DELAY_HRD_FLAG: BitField = BitField::new(14, 1);
    pub(crate) const MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG: BitField = BitField::new(15, 1);
}

/// `VAEncPictureParameterBufferH264::pic_fields.bits`.
pub(crate) mod h264_pic_fields {
    use super::BitField;
    pub(crate) const IDR_PIC_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const REFERENCE_PIC_FLAG: BitField = BitField::new(1, 2);
    pub(crate) const ENTROPY_CODING_MODE_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const WEIGHTED_PRED_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const WEIGHTED_BIPRED_IDC: BitField = BitField::new(5, 2);
    pub(crate) const CONSTRAINED_INTRA_PRED_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const TRANSFORM_8X8_MODE_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const REDUNDANT_PIC_CNT_PRESENT_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const PIC_ORDER_PRESENT_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const PIC_SCALING_MATRIX_PRESENT_FLAG: BitField = BitField::new(12, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncSequenceParameterBufferHEVC {
    pub(crate) general_profile_idc: u8,
    pub(crate) general_level_idc: u8,
    pub(crate) general_tier_flag: u8,
    pub(crate) intra_period: u32,
    pub(crate) intra_idr_period: u32,
    pub(crate) ip_period: u32,
    pub(crate) bits_per_second: u32,
    pub(crate) pic_width_in_luma_samples: u16,
    pub(crate) pic_height_in_luma_samples: u16,
    /// `seq_fields.value`: see [`hevc_seq_fields`].
    pub(crate) seq_fields: u32,
    pub(crate) log2_min_luma_coding_block_size_minus3: u8,
    pub(crate) log2_diff_max_min_luma_coding_block_size: u8,
    pub(crate) log2_min_transform_block_size_minus2: u8,
    pub(crate) log2_diff_max_min_transform_block_size: u8,
    pub(crate) max_transform_hierarchy_depth_inter: u8,
    pub(crate) max_transform_hierarchy_depth_intra: u8,
    pub(crate) pcm_sample_bit_depth_luma_minus1: u32,
    pub(crate) pcm_sample_bit_depth_chroma_minus1: u32,
    pub(crate) log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub(crate) log2_max_pcm_luma_coding_block_size_minus3: u32,
    pub(crate) vui_parameters_present_flag: u8,
    /// `vui_fields.value`: see [`hevc_vui_fields`].
    pub(crate) vui_fields: u32,
    pub(crate) aspect_ratio_idc: u8,
    pub(crate) sar_width: u32,
    pub(crate) sar_height: u32,
    pub(crate) vui_num_units_in_tick: u32,
    pub(crate) vui_time_scale: u32,
    pub(crate) min_spatial_segmentation_idc: u16,
    pub(crate) max_bytes_per_pic_denom: u8,
    pub(crate) max_bits_per_min_cu_denom: u8,
    pub(crate) scc_fields: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM - 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncPictureParameterBufferHEVC {
    pub(crate) decoded_curr_pic: VAPictureHEVC,
    pub(crate) reference_frames: [VAPictureHEVC; 15],
    pub(crate) coded_buf: VABufferID,
    pub(crate) collocated_ref_pic_index: u8,
    pub(crate) last_picture: u8,
    pub(crate) pic_init_qp: u8,
    pub(crate) diff_cu_qp_delta_depth: u8,
    pub(crate) pps_cb_qp_offset: i8,
    pub(crate) pps_cr_qp_offset: i8,
    pub(crate) num_tile_columns_minus1: u8,
    pub(crate) num_tile_rows_minus1: u8,
    pub(crate) column_width_minus1: [u8; 19],
    pub(crate) row_height_minus1: [u8; 21],
    pub(crate) log2_parallel_merge_level_minus2: u8,
    pub(crate) ctu_max_bitsize_allowed: u8,
    pub(crate) num_ref_idx_l0_default_active_minus1: u8,
    pub(crate) num_ref_idx_l1_default_active_minus1: u8,
    pub(crate) slice_pic_parameter_set_id: u8,
    pub(crate) nal_unit_type: u8,
    /// `pic_fields.value`: see [`hevc_pic_fields`].
    pub(crate) pic_fields: u32,
    pub(crate) hierarchical_level_plus1: u8,
    pub(crate) va_byte_reserved: u8,
    pub(crate) scc_fields: u16,
    pub(crate) va_reserved: [u32; VA_PADDING_HIGH - 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncSliceParameterBufferHEVC {
    pub(crate) slice_segment_address: u32,
    pub(crate) num_ctu_in_slice: u32,
    pub(crate) slice_type: u8,
    pub(crate) slice_pic_parameter_set_id: u8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) ref_pic_list0: [VAPictureHEVC; 15],
    pub(crate) ref_pic_list1: [VAPictureHEVC; 15],
    pub(crate) luma_log2_weight_denom: u8,
    pub(crate) delta_chroma_log2_weight_denom: i8,
    pub(crate) delta_luma_weight_l0: [i8; 15],
    pub(crate) luma_offset_l0: [i8; 15],
    pub(crate) delta_chroma_weight_l0: [[i8; 2]; 15],
    pub(crate) chroma_offset_l0: [[i8; 2]; 15],
    pub(crate) delta_luma_weight_l1: [i8; 15],
    pub(crate) luma_offset_l1: [i8; 15],
    pub(crate) delta_chroma_weight_l1: [[i8; 2]; 15],
    pub(crate) chroma_offset_l1: [[i8; 2]; 15],
    pub(crate) max_num_merge_cand: u8,
    pub(crate) slice_qp_delta: i8,
    pub(crate) slice_cb_qp_offset: i8,
    pub(crate) slice_cr_qp_offset: i8,
    pub(crate) slice_beta_offset_div2: i8,
    pub(crate) slice_tc_offset_div2: i8,
    /// `slice_fields.value`: see [`hevc_slice_fields`].
    pub(crate) slice_fields: u32,
    pub(crate) pred_weight_table_bit_offset: u32,
    pub(crate) pred_weight_table_bit_length: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM - 2],
}

/// `VAEncSequenceParameterBufferHEVC::seq_fields.bits`.
pub(crate) mod hevc_seq_fields {
    use super::BitField;
    pub(crate) const CHROMA_FORMAT_IDC: BitField = BitField::new(0, 2);
    pub(crate) const SEPARATE_COLOUR_PLANE_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const BIT_DEPTH_LUMA_MINUS8: BitField = BitField::new(3, 3);
    pub(crate) const BIT_DEPTH_CHROMA_MINUS8: BitField = BitField::new(6, 3);
    pub(crate) const SCALING_LIST_ENABLED_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const STRONG_INTRA_SMOOTHING_ENABLED_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const AMP_ENABLED_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const PCM_ENABLED_FLAG: BitField = BitField::new(13, 1);
    pub(crate) const PCM_LOOP_FILTER_DISABLED_FLAG: BitField = BitField::new(14, 1);
    pub(crate) const SPS_TEMPORAL_MVP_ENABLED_FLAG: BitField = BitField::new(15, 1);
    pub(crate) const LOW_DELAY_SEQ: BitField = BitField::new(16, 1);
    pub(crate) const HIERACHICAL_FLAG: BitField = BitField::new(17, 1);
}

/// `VAEncSequenceParameterBufferHEVC::vui_fields.bits`.
pub(crate) mod hevc_vui_fields {
    use super::BitField;
    pub(crate) const ASPECT_RATIO_INFO_PRESENT_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const NEUTRAL_CHROMA_INDICATION_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const FIELD_SEQ_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const VUI_TIMING_INFO_PRESENT_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const BITSTREAM_RESTRICTION_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const TILES_FIXED_STRUCTURE_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const RESTRICTED_REF_PIC_LISTS_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const LOG2_MAX_MV_LENGTH_HORIZONTAL: BitField = BitField::new(8, 5);
    pub(crate) const LOG2_MAX_MV_LENGTH_VERTICAL: BitField = BitField::new(13, 5);
}

/// `VAEncPictureParameterBufferHEVC::pic_fields.bits`.
pub(crate) mod hevc_pic_fields {
    use super::BitField;
    pub(crate) const IDR_PIC_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const CODING_TYPE: BitField = BitField::new(1, 3);
    pub(crate) const REFERENCE_PIC_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const SIGN_DATA_HIDING_ENABLED_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const CONSTRAINED_INTRA_PRED_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const TRANSFORM_SKIP_ENABLED_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const CU_QP_DELTA_ENABLED_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const WEIGHTED_PRED_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const WEIGHTED_BIPRED_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const TRANSQUANT_BYPASS_ENABLED_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const TILES_ENABLED_FLAG: BitField = BitField::new(13, 1);
    pub(crate) const ENTROPY_CODING_SYNC_ENABLED_FLAG: BitField = BitField::new(14, 1);
    pub(crate) const LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: BitField = BitField::new(15, 1);
    pub(crate) const PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: BitField = BitField::new(16, 1);
    pub(crate) const SCALING_LIST_DATA_PRESENT_FLAG: BitField = BitField::new(17, 1);
    pub(crate) const SCREEN_CONTENT_FLAG: BitField = BitField::new(18, 1);
    pub(crate) const ENABLE_GPU_WEIGHTED_PREDICTION: BitField = BitField::new(19, 1);
    pub(crate) const NO_OUTPUT_OF_PRIOR_PICS_FLAG: BitField = BitField::new(20, 1);
}

/// `VAEncSliceParameterBufferHEVC::slice_fields.bits`.
pub(crate) mod hevc_slice_fields {
    use super::BitField;
    pub(crate) const LAST_SLICE_OF_PIC_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const DEPENDENT_SLICE_SEGMENT_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const COLOUR_PLANE_ID: BitField = BitField::new(2, 2);
    pub(crate) const SLICE_TEMPORAL_MVP_ENABLED_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const SLICE_SAO_LUMA_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const SLICE_SAO_CHROMA_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const MVD_L1_ZERO_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const CABAC_INIT_FLAG: BitField = BitField::new(9, 1);
    /// Two bits wide in va_enc_hevc.h.
    pub(crate) const SLICE_DEBLOCKING_FILTER_DISABLED_FLAG: BitField = BitField::new(10, 2);
    pub(crate) const SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const COLLOCATED_FROM_L0_FLAG: BitField = BitField::new(13, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncPackedHeaderParameterBuffer {
    pub(crate) type_: u32,
    pub(crate) bit_length: u32,
    pub(crate) has_emulation_bytes: u8,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterBuffer` with its payload: va.h's `type` and then the
/// flexible `data[]`, which is `uint32_t`-aligned, so a payload of
/// `uint32_t`s starts at offset 4.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAEncMiscParameterBuffer<T> {
    pub(crate) type_: c_int,
    pub(crate) data: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncMiscParameterRateControl {
    pub(crate) bits_per_second: u32,
    pub(crate) target_percentage: u32,
    pub(crate) window_size: u32,
    pub(crate) initial_qp: u32,
    pub(crate) min_qp: u32,
    pub(crate) basic_unit_size: u32,
    /// `rc_flags.value`: see [`rc_flags`].
    pub(crate) rc_flags: u32,
    pub(crate) icq_quality_factor: u32,
    pub(crate) max_qp: u32,
    pub(crate) quality_factor: u32,
    pub(crate) target_frame_size: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterRateControl::rc_flags.bits`.
pub(crate) mod rc_flags {
    use super::BitField;
    pub(crate) const RESET: BitField = BitField::new(0, 1);
    pub(crate) const DISABLE_FRAME_SKIP: BitField = BitField::new(1, 1);
    pub(crate) const DISABLE_BIT_STUFFING: BitField = BitField::new(2, 1);
    pub(crate) const MB_RATE_CONTROL: BitField = BitField::new(3, 4);
    pub(crate) const TEMPORAL_ID: BitField = BitField::new(7, 8);
    pub(crate) const CFS_I_FRAMES: BitField = BitField::new(15, 1);
    pub(crate) const ENABLE_PARALLEL_BRC: BitField = BitField::new(16, 1);
    pub(crate) const ENABLE_DYNAMIC_SCALING: BitField = BitField::new(17, 1);
    pub(crate) const FRAME_TOLERANCE_MODE: BitField = BitField::new(18, 2);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncMiscParameterFrameRate {
    /// The numerator in the low 16 bits and the denominator in the high,
    /// or a whole number of frames a second when the high bits are 0
    /// (va.h).
    pub(crate) framerate: u32,
    pub(crate) framerate_flags: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncMiscParameterHRD {
    pub(crate) initial_buffer_fullness: u32,
    pub(crate) buffer_size: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncMiscParameterBufferQualityLevel {
    pub(crate) quality_level: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAEncMiscParameterBufferMaxFrameSize {
    /// Deprecated in va.h; set to the misc type, as libva's users do.
    pub(crate) type_: c_int,
    pub(crate) max_frame_size: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VACodedBufferSegment {
    pub(crate) size: u32,
    pub(crate) bit_offset: u32,
    pub(crate) status: u32,
    pub(crate) reserved: u32,
    pub(crate) buf: *mut c_void,
    pub(crate) next: *mut c_void,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

// libva's functions, loaded at run time rather than linked
// (`crate::vaapi::va::libva`): wrappers of the same names and signatures.
pub(crate) use super::super::libva::vaSyncBuffer;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// What the C side prints, line for line.
    const LAYOUT_C: &str = include_str!("encode_layout.c");

    macro_rules! layout {
        ($out:ident, $ty:ty, $c:literal, { $($field:ident => $cfield:literal),* $(,)? }) => {
            $out.push(format!("{} size {}", $c, size_of::<$ty>()));
            $( $out.push(format!("{}.{} {}", $c, $cfield, offset_of!($ty, $field))); )*
        };
    }

    fn fields(out: &mut Vec<String>, union: &str, list: &[(&str, BitField)]) {
        for (name, field) in list {
            out.push(format!("{union}.{name} {:#x}", field.put(u32::MAX)));
        }
    }

    /// `name value` for each constant, in decimal, as the C side's `C`
    /// prints it.
    macro_rules! constants {
        ($out:ident, $($name:ident),* $(,)?) => {
            $( $out.push(format!("{} {}", stringify!($name), $name)); )*
        };
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one list of every field, in the C file's order"
    )]
    fn rust_layout() -> Vec<String> {
        let mut out = Vec::new();
        layout!(out, VAEncSequenceParameterBufferH264, "VAEncSequenceParameterBufferH264", {
            seq_parameter_set_id => "seq_parameter_set_id", level_idc => "level_idc",
            intra_period => "intra_period", intra_idr_period => "intra_idr_period",
            ip_period => "ip_period", bits_per_second => "bits_per_second",
            max_num_ref_frames => "max_num_ref_frames",
            picture_width_in_mbs => "picture_width_in_mbs",
            picture_height_in_mbs => "picture_height_in_mbs", seq_fields => "seq_fields",
            bit_depth_luma_minus8 => "bit_depth_luma_minus8",
            bit_depth_chroma_minus8 => "bit_depth_chroma_minus8",
            num_ref_frames_in_pic_order_cnt_cycle => "num_ref_frames_in_pic_order_cnt_cycle",
            offset_for_non_ref_pic => "offset_for_non_ref_pic",
            offset_for_top_to_bottom_field => "offset_for_top_to_bottom_field",
            offset_for_ref_frame => "offset_for_ref_frame",
            frame_cropping_flag => "frame_cropping_flag",
            frame_crop_left_offset => "frame_crop_left_offset",
            frame_crop_right_offset => "frame_crop_right_offset",
            frame_crop_top_offset => "frame_crop_top_offset",
            frame_crop_bottom_offset => "frame_crop_bottom_offset",
            vui_parameters_present_flag => "vui_parameters_present_flag",
            vui_fields => "vui_fields", aspect_ratio_idc => "aspect_ratio_idc",
            sar_width => "sar_width", sar_height => "sar_height",
            num_units_in_tick => "num_units_in_tick", time_scale => "time_scale",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncPictureParameterBufferH264, "VAEncPictureParameterBufferH264", {
            curr_pic => "CurrPic", reference_frames => "ReferenceFrames",
            coded_buf => "coded_buf", pic_parameter_set_id => "pic_parameter_set_id",
            seq_parameter_set_id => "seq_parameter_set_id", last_picture => "last_picture",
            frame_num => "frame_num", pic_init_qp => "pic_init_qp",
            num_ref_idx_l0_active_minus1 => "num_ref_idx_l0_active_minus1",
            num_ref_idx_l1_active_minus1 => "num_ref_idx_l1_active_minus1",
            chroma_qp_index_offset => "chroma_qp_index_offset",
            second_chroma_qp_index_offset => "second_chroma_qp_index_offset",
            pic_fields => "pic_fields", va_reserved => "va_reserved",
        });
        layout!(out, VAEncSliceParameterBufferH264, "VAEncSliceParameterBufferH264", {
            macroblock_address => "macroblock_address", num_macroblocks => "num_macroblocks",
            macroblock_info => "macroblock_info", slice_type => "slice_type",
            pic_parameter_set_id => "pic_parameter_set_id", idr_pic_id => "idr_pic_id",
            pic_order_cnt_lsb => "pic_order_cnt_lsb",
            delta_pic_order_cnt_bottom => "delta_pic_order_cnt_bottom",
            delta_pic_order_cnt => "delta_pic_order_cnt",
            direct_spatial_mv_pred_flag => "direct_spatial_mv_pred_flag",
            num_ref_idx_active_override_flag => "num_ref_idx_active_override_flag",
            num_ref_idx_l0_active_minus1 => "num_ref_idx_l0_active_minus1",
            num_ref_idx_l1_active_minus1 => "num_ref_idx_l1_active_minus1",
            ref_pic_list0 => "RefPicList0", ref_pic_list1 => "RefPicList1",
            luma_log2_weight_denom => "luma_log2_weight_denom",
            chroma_log2_weight_denom => "chroma_log2_weight_denom",
            luma_weight_l0_flag => "luma_weight_l0_flag", luma_weight_l0 => "luma_weight_l0",
            luma_offset_l0 => "luma_offset_l0", chroma_weight_l0_flag => "chroma_weight_l0_flag",
            chroma_weight_l0 => "chroma_weight_l0", chroma_offset_l0 => "chroma_offset_l0",
            luma_weight_l1_flag => "luma_weight_l1_flag", luma_weight_l1 => "luma_weight_l1",
            luma_offset_l1 => "luma_offset_l1", chroma_weight_l1_flag => "chroma_weight_l1_flag",
            chroma_weight_l1 => "chroma_weight_l1", chroma_offset_l1 => "chroma_offset_l1",
            cabac_init_idc => "cabac_init_idc", slice_qp_delta => "slice_qp_delta",
            disable_deblocking_filter_idc => "disable_deblocking_filter_idc",
            slice_alpha_c0_offset_div2 => "slice_alpha_c0_offset_div2",
            slice_beta_offset_div2 => "slice_beta_offset_div2", va_reserved => "va_reserved",
        });
        layout!(out, VAPictureHEVC, "VAPictureHEVC", {
            picture_id => "picture_id", pic_order_cnt => "pic_order_cnt", flags => "flags",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncSequenceParameterBufferHEVC, "VAEncSequenceParameterBufferHEVC", {
            general_profile_idc => "general_profile_idc", general_level_idc => "general_level_idc",
            general_tier_flag => "general_tier_flag", intra_period => "intra_period",
            intra_idr_period => "intra_idr_period", ip_period => "ip_period",
            bits_per_second => "bits_per_second",
            pic_width_in_luma_samples => "pic_width_in_luma_samples",
            pic_height_in_luma_samples => "pic_height_in_luma_samples",
            seq_fields => "seq_fields",
            log2_min_luma_coding_block_size_minus3 => "log2_min_luma_coding_block_size_minus3",
            log2_diff_max_min_luma_coding_block_size => "log2_diff_max_min_luma_coding_block_size",
            log2_min_transform_block_size_minus2 => "log2_min_transform_block_size_minus2",
            log2_diff_max_min_transform_block_size => "log2_diff_max_min_transform_block_size",
            max_transform_hierarchy_depth_inter => "max_transform_hierarchy_depth_inter",
            max_transform_hierarchy_depth_intra => "max_transform_hierarchy_depth_intra",
            pcm_sample_bit_depth_luma_minus1 => "pcm_sample_bit_depth_luma_minus1",
            pcm_sample_bit_depth_chroma_minus1 => "pcm_sample_bit_depth_chroma_minus1",
            log2_min_pcm_luma_coding_block_size_minus3 =>
                "log2_min_pcm_luma_coding_block_size_minus3",
            log2_max_pcm_luma_coding_block_size_minus3 =>
                "log2_max_pcm_luma_coding_block_size_minus3",
            vui_parameters_present_flag => "vui_parameters_present_flag",
            vui_fields => "vui_fields", aspect_ratio_idc => "aspect_ratio_idc",
            sar_width => "sar_width", sar_height => "sar_height",
            vui_num_units_in_tick => "vui_num_units_in_tick", vui_time_scale => "vui_time_scale",
            min_spatial_segmentation_idc => "min_spatial_segmentation_idc",
            max_bytes_per_pic_denom => "max_bytes_per_pic_denom",
            max_bits_per_min_cu_denom => "max_bits_per_min_cu_denom",
            scc_fields => "scc_fields", va_reserved => "va_reserved",
        });
        layout!(out, VAEncPictureParameterBufferHEVC, "VAEncPictureParameterBufferHEVC", {
            decoded_curr_pic => "decoded_curr_pic", reference_frames => "reference_frames",
            coded_buf => "coded_buf", collocated_ref_pic_index => "collocated_ref_pic_index",
            last_picture => "last_picture", pic_init_qp => "pic_init_qp",
            diff_cu_qp_delta_depth => "diff_cu_qp_delta_depth",
            pps_cb_qp_offset => "pps_cb_qp_offset", pps_cr_qp_offset => "pps_cr_qp_offset",
            num_tile_columns_minus1 => "num_tile_columns_minus1",
            num_tile_rows_minus1 => "num_tile_rows_minus1",
            column_width_minus1 => "column_width_minus1", row_height_minus1 => "row_height_minus1",
            log2_parallel_merge_level_minus2 => "log2_parallel_merge_level_minus2",
            ctu_max_bitsize_allowed => "ctu_max_bitsize_allowed",
            num_ref_idx_l0_default_active_minus1 => "num_ref_idx_l0_default_active_minus1",
            num_ref_idx_l1_default_active_minus1 => "num_ref_idx_l1_default_active_minus1",
            slice_pic_parameter_set_id => "slice_pic_parameter_set_id",
            nal_unit_type => "nal_unit_type", pic_fields => "pic_fields",
            hierarchical_level_plus1 => "hierarchical_level_plus1",
            va_byte_reserved => "va_byte_reserved", scc_fields => "scc_fields",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncSliceParameterBufferHEVC, "VAEncSliceParameterBufferHEVC", {
            slice_segment_address => "slice_segment_address",
            num_ctu_in_slice => "num_ctu_in_slice", slice_type => "slice_type",
            slice_pic_parameter_set_id => "slice_pic_parameter_set_id",
            num_ref_idx_l0_active_minus1 => "num_ref_idx_l0_active_minus1",
            num_ref_idx_l1_active_minus1 => "num_ref_idx_l1_active_minus1",
            ref_pic_list0 => "ref_pic_list0", ref_pic_list1 => "ref_pic_list1",
            luma_log2_weight_denom => "luma_log2_weight_denom",
            delta_chroma_log2_weight_denom => "delta_chroma_log2_weight_denom",
            delta_luma_weight_l0 => "delta_luma_weight_l0", luma_offset_l0 => "luma_offset_l0",
            delta_chroma_weight_l0 => "delta_chroma_weight_l0",
            chroma_offset_l0 => "chroma_offset_l0",
            delta_luma_weight_l1 => "delta_luma_weight_l1", luma_offset_l1 => "luma_offset_l1",
            delta_chroma_weight_l1 => "delta_chroma_weight_l1",
            chroma_offset_l1 => "chroma_offset_l1", max_num_merge_cand => "max_num_merge_cand",
            slice_qp_delta => "slice_qp_delta", slice_cb_qp_offset => "slice_cb_qp_offset",
            slice_cr_qp_offset => "slice_cr_qp_offset",
            slice_beta_offset_div2 => "slice_beta_offset_div2",
            slice_tc_offset_div2 => "slice_tc_offset_div2", slice_fields => "slice_fields",
            pred_weight_table_bit_offset => "pred_weight_table_bit_offset",
            pred_weight_table_bit_length => "pred_weight_table_bit_length",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncPackedHeaderParameterBuffer, "VAEncPackedHeaderParameterBuffer", {
            type_ => "type", bit_length => "bit_length",
            has_emulation_bytes => "has_emulation_bytes", va_reserved => "va_reserved",
        });
        out.push(format!(
            "VAEncMiscParameterBuffer.data {}",
            offset_of!(VAEncMiscParameterBuffer<u32>, data)
        ));
        layout!(out, VAEncMiscParameterRateControl, "VAEncMiscParameterRateControl", {
            bits_per_second => "bits_per_second", target_percentage => "target_percentage",
            window_size => "window_size", initial_qp => "initial_qp", min_qp => "min_qp",
            basic_unit_size => "basic_unit_size", rc_flags => "rc_flags",
            icq_quality_factor => "ICQ_quality_factor", max_qp => "max_qp",
            quality_factor => "quality_factor", target_frame_size => "target_frame_size",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncMiscParameterFrameRate, "VAEncMiscParameterFrameRate", {
            framerate => "framerate", framerate_flags => "framerate_flags",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncMiscParameterHRD, "VAEncMiscParameterHRD", {
            initial_buffer_fullness => "initial_buffer_fullness", buffer_size => "buffer_size",
            va_reserved => "va_reserved",
        });
        layout!(out, VAEncMiscParameterBufferQualityLevel, "VAEncMiscParameterBufferQualityLevel", {
            quality_level => "quality_level", va_reserved => "va_reserved",
        });
        layout!(out, VAEncMiscParameterBufferMaxFrameSize, "VAEncMiscParameterBufferMaxFrameSize", {
            type_ => "type", max_frame_size => "max_frame_size", va_reserved => "va_reserved",
        });
        layout!(out, VACodedBufferSegment, "VACodedBufferSegment", {
            size => "size", bit_offset => "bit_offset", status => "status",
            reserved => "reserved", buf => "buf", next => "next", va_reserved => "va_reserved",
        });
        use {h264_pic_fields as hp, h264_seq_fields as hs, h264_vui_fields as hv};
        fields(
            &mut out,
            "h264.seq_fields",
            &[
                ("chroma_format_idc", hs::CHROMA_FORMAT_IDC),
                ("frame_mbs_only_flag", hs::FRAME_MBS_ONLY_FLAG),
                (
                    "mb_adaptive_frame_field_flag",
                    hs::MB_ADAPTIVE_FRAME_FIELD_FLAG,
                ),
                (
                    "seq_scaling_matrix_present_flag",
                    hs::SEQ_SCALING_MATRIX_PRESENT_FLAG,
                ),
                ("direct_8x8_inference_flag", hs::DIRECT_8X8_INFERENCE_FLAG),
                ("log2_max_frame_num_minus4", hs::LOG2_MAX_FRAME_NUM_MINUS4),
                ("pic_order_cnt_type", hs::PIC_ORDER_CNT_TYPE),
                (
                    "log2_max_pic_order_cnt_lsb_minus4",
                    hs::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
                ),
                (
                    "delta_pic_order_always_zero_flag",
                    hs::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG,
                ),
            ],
        );
        fields(
            &mut out,
            "h264.vui_fields",
            &[
                (
                    "aspect_ratio_info_present_flag",
                    hv::ASPECT_RATIO_INFO_PRESENT_FLAG,
                ),
                ("timing_info_present_flag", hv::TIMING_INFO_PRESENT_FLAG),
                ("bitstream_restriction_flag", hv::BITSTREAM_RESTRICTION_FLAG),
                (
                    "log2_max_mv_length_horizontal",
                    hv::LOG2_MAX_MV_LENGTH_HORIZONTAL,
                ),
                (
                    "log2_max_mv_length_vertical",
                    hv::LOG2_MAX_MV_LENGTH_VERTICAL,
                ),
                ("fixed_frame_rate_flag", hv::FIXED_FRAME_RATE_FLAG),
                ("low_delay_hrd_flag", hv::LOW_DELAY_HRD_FLAG),
                (
                    "motion_vectors_over_pic_boundaries_flag",
                    hv::MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG,
                ),
            ],
        );
        fields(
            &mut out,
            "h264.pic_fields",
            &[
                ("idr_pic_flag", hp::IDR_PIC_FLAG),
                ("reference_pic_flag", hp::REFERENCE_PIC_FLAG),
                ("entropy_coding_mode_flag", hp::ENTROPY_CODING_MODE_FLAG),
                ("weighted_pred_flag", hp::WEIGHTED_PRED_FLAG),
                ("weighted_bipred_idc", hp::WEIGHTED_BIPRED_IDC),
                (
                    "constrained_intra_pred_flag",
                    hp::CONSTRAINED_INTRA_PRED_FLAG,
                ),
                ("transform_8x8_mode_flag", hp::TRANSFORM_8X8_MODE_FLAG),
                (
                    "deblocking_filter_control_present_flag",
                    hp::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
                ),
                (
                    "redundant_pic_cnt_present_flag",
                    hp::REDUNDANT_PIC_CNT_PRESENT_FLAG,
                ),
                ("pic_order_present_flag", hp::PIC_ORDER_PRESENT_FLAG),
                (
                    "pic_scaling_matrix_present_flag",
                    hp::PIC_SCALING_MATRIX_PRESENT_FLAG,
                ),
            ],
        );
        use {
            hevc_pic_fields as ep, hevc_seq_fields as es, hevc_slice_fields as el,
            hevc_vui_fields as ev,
        };
        fields(
            &mut out,
            "hevc.seq_fields",
            &[
                ("chroma_format_idc", es::CHROMA_FORMAT_IDC),
                ("separate_colour_plane_flag", es::SEPARATE_COLOUR_PLANE_FLAG),
                ("bit_depth_luma_minus8", es::BIT_DEPTH_LUMA_MINUS8),
                ("bit_depth_chroma_minus8", es::BIT_DEPTH_CHROMA_MINUS8),
                ("scaling_list_enabled_flag", es::SCALING_LIST_ENABLED_FLAG),
                (
                    "strong_intra_smoothing_enabled_flag",
                    es::STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
                ),
                ("amp_enabled_flag", es::AMP_ENABLED_FLAG),
                (
                    "sample_adaptive_offset_enabled_flag",
                    es::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG,
                ),
                ("pcm_enabled_flag", es::PCM_ENABLED_FLAG),
                (
                    "pcm_loop_filter_disabled_flag",
                    es::PCM_LOOP_FILTER_DISABLED_FLAG,
                ),
                (
                    "sps_temporal_mvp_enabled_flag",
                    es::SPS_TEMPORAL_MVP_ENABLED_FLAG,
                ),
                ("low_delay_seq", es::LOW_DELAY_SEQ),
                ("hierachical_flag", es::HIERACHICAL_FLAG),
            ],
        );
        fields(
            &mut out,
            "hevc.vui_fields",
            &[
                (
                    "aspect_ratio_info_present_flag",
                    ev::ASPECT_RATIO_INFO_PRESENT_FLAG,
                ),
                (
                    "neutral_chroma_indication_flag",
                    ev::NEUTRAL_CHROMA_INDICATION_FLAG,
                ),
                ("field_seq_flag", ev::FIELD_SEQ_FLAG),
                (
                    "vui_timing_info_present_flag",
                    ev::VUI_TIMING_INFO_PRESENT_FLAG,
                ),
                ("bitstream_restriction_flag", ev::BITSTREAM_RESTRICTION_FLAG),
                ("tiles_fixed_structure_flag", ev::TILES_FIXED_STRUCTURE_FLAG),
                (
                    "motion_vectors_over_pic_boundaries_flag",
                    ev::MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG,
                ),
                (
                    "restricted_ref_pic_lists_flag",
                    ev::RESTRICTED_REF_PIC_LISTS_FLAG,
                ),
                (
                    "log2_max_mv_length_horizontal",
                    ev::LOG2_MAX_MV_LENGTH_HORIZONTAL,
                ),
                (
                    "log2_max_mv_length_vertical",
                    ev::LOG2_MAX_MV_LENGTH_VERTICAL,
                ),
            ],
        );
        fields(
            &mut out,
            "hevc.pic_fields",
            &[
                ("idr_pic_flag", ep::IDR_PIC_FLAG),
                ("coding_type", ep::CODING_TYPE),
                ("reference_pic_flag", ep::REFERENCE_PIC_FLAG),
                (
                    "dependent_slice_segments_enabled_flag",
                    ep::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG,
                ),
                (
                    "sign_data_hiding_enabled_flag",
                    ep::SIGN_DATA_HIDING_ENABLED_FLAG,
                ),
                (
                    "constrained_intra_pred_flag",
                    ep::CONSTRAINED_INTRA_PRED_FLAG,
                ),
                (
                    "transform_skip_enabled_flag",
                    ep::TRANSFORM_SKIP_ENABLED_FLAG,
                ),
                ("cu_qp_delta_enabled_flag", ep::CU_QP_DELTA_ENABLED_FLAG),
                ("weighted_pred_flag", ep::WEIGHTED_PRED_FLAG),
                ("weighted_bipred_flag", ep::WEIGHTED_BIPRED_FLAG),
                (
                    "transquant_bypass_enabled_flag",
                    ep::TRANSQUANT_BYPASS_ENABLED_FLAG,
                ),
                ("tiles_enabled_flag", ep::TILES_ENABLED_FLAG),
                (
                    "entropy_coding_sync_enabled_flag",
                    ep::ENTROPY_CODING_SYNC_ENABLED_FLAG,
                ),
                (
                    "loop_filter_across_tiles_enabled_flag",
                    ep::LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG,
                ),
                (
                    "pps_loop_filter_across_slices_enabled_flag",
                    ep::PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
                ),
                (
                    "scaling_list_data_present_flag",
                    ep::SCALING_LIST_DATA_PRESENT_FLAG,
                ),
                ("screen_content_flag", ep::SCREEN_CONTENT_FLAG),
                (
                    "enable_gpu_weighted_prediction",
                    ep::ENABLE_GPU_WEIGHTED_PREDICTION,
                ),
                (
                    "no_output_of_prior_pics_flag",
                    ep::NO_OUTPUT_OF_PRIOR_PICS_FLAG,
                ),
            ],
        );
        fields(
            &mut out,
            "hevc.slice_fields",
            &[
                ("last_slice_of_pic_flag", el::LAST_SLICE_OF_PIC_FLAG),
                (
                    "dependent_slice_segment_flag",
                    el::DEPENDENT_SLICE_SEGMENT_FLAG,
                ),
                ("colour_plane_id", el::COLOUR_PLANE_ID),
                (
                    "slice_temporal_mvp_enabled_flag",
                    el::SLICE_TEMPORAL_MVP_ENABLED_FLAG,
                ),
                ("slice_sao_luma_flag", el::SLICE_SAO_LUMA_FLAG),
                ("slice_sao_chroma_flag", el::SLICE_SAO_CHROMA_FLAG),
                (
                    "num_ref_idx_active_override_flag",
                    el::NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG,
                ),
                ("mvd_l1_zero_flag", el::MVD_L1_ZERO_FLAG),
                ("cabac_init_flag", el::CABAC_INIT_FLAG),
                (
                    "slice_deblocking_filter_disabled_flag",
                    el::SLICE_DEBLOCKING_FILTER_DISABLED_FLAG,
                ),
                (
                    "slice_loop_filter_across_slices_enabled_flag",
                    el::SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
                ),
                ("collocated_from_l0_flag", el::COLLOCATED_FROM_L0_FLAG),
            ],
        );
        fields(
            &mut out,
            "rc_flags",
            &[
                ("reset", rc_flags::RESET),
                ("disable_frame_skip", rc_flags::DISABLE_FRAME_SKIP),
                ("disable_bit_stuffing", rc_flags::DISABLE_BIT_STUFFING),
                ("mb_rate_control", rc_flags::MB_RATE_CONTROL),
                ("temporal_id", rc_flags::TEMPORAL_ID),
                ("cfs_I_frames", rc_flags::CFS_I_FRAMES),
                ("enable_parallel_brc", rc_flags::ENABLE_PARALLEL_BRC),
                ("enable_dynamic_scaling", rc_flags::ENABLE_DYNAMIC_SCALING),
                ("frame_tolerance_mode", rc_flags::FRAME_TOLERANCE_MODE),
            ],
        );
        // Each constant, as the C side prints it.
        constants!(
            out,
            VAEntrypointEncSlice,
            VAEntrypointEncSliceLP,
            VAConfigAttribEncSliceStructure,
            VAConfigAttribEncQualityRange,
            VAConfigAttribEncIntraRefresh,
            VAConfigAttribEncSkipFrame,
            VAConfigAttribEncROI,
            VAConfigAttribEncRateControlExt,
            VAConfigAttribMaxFrameSize,
            VAConfigAttribPredictionDirection,
            VAConfigAttribEncHEVCFeatures,
            VAConfigAttribEncHEVCBlockSizes,
            VA_RC_CBR,
            VA_RC_VBR,
            VA_RC_CQP,
            VA_RC_QVBR,
            VA_ENC_PACKED_HEADER_SEQUENCE,
            VA_ENC_PACKED_HEADER_PICTURE,
            VA_ENC_PACKED_HEADER_SLICE,
            VA_ENC_PACKED_HEADER_MISC,
            VA_ENC_PACKED_HEADER_RAW_DATA,
            VAEncPackedHeaderSequence,
            VAEncPackedHeaderPicture,
            VAEncPackedHeaderSlice,
            VAEncPackedHeaderRawData,
            VAEncCodedBufferType,
            VAEncSequenceParameterBufferType,
            VAEncPictureParameterBufferType,
            VAEncSliceParameterBufferType,
            VAEncPackedHeaderParameterBufferType,
            VAEncPackedHeaderDataBufferType,
            VAEncMiscParameterBufferType,
            VAEncMiscParameterTypeFrameRate,
            VAEncMiscParameterTypeRateControl,
            VAEncMiscParameterTypeMaxFrameSize,
            VAEncMiscParameterTypeHRD,
            VAEncMiscParameterTypeQualityLevel,
            VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW,
            VA_CODED_BUF_STATUS_BAD_BITSTREAM
        );
        out
    }

    /// Compiles `encode_layout.c` against the installed libva headers and
    /// compares its output with this module's layout. `CC` picks the
    /// compiler (`cc` by default).
    #[test]
    #[ignore = "needs libva's headers and a C compiler: run on the test host"]
    fn encode_layouts_match_libva_headers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("rvgv-va-enc-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let binary = dir.join("layout");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
        let mut compile = Command::new(cc)
            .args(["-Wno-overflow", "-x", "c", "-", "-o"])
            .arg(&binary)
            .stdin(Stdio::piped())
            .spawn()?;
        compile
            .stdin
            .take()
            .ok_or("no stdin")?
            .write_all(LAYOUT_C.as_bytes())?;
        if !compile.wait()?.success() {
            return Err("the C compiler failed".into());
        }
        let output = Command::new(&binary).output()?;
        std::fs::remove_dir_all(&dir)?;
        let c_lines: Vec<String> = String::from_utf8(output.stdout)?
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(rust_layout(), c_lines);
        Ok(())
    }
}
