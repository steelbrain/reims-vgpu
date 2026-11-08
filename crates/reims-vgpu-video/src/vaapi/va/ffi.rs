//! libva's C interface, hand-written for the subset this crate calls, from
//! libva 2.24.1's headers (`va.h`, `va_dec_hevc.h`, `va_drm.h`,
//! `va_vpp.h`; VA-API 1.24.0).
//! The test at the end checks every struct's size and offsets, every
//! bit-field's position and every constant's value against those headers
//! (and `drm_fourcc.h` for the DRM formats) with a C compiler. The
//! functions are `super::libva`'s wrappers, which load libva at run time.
//!
//! Names keep libva's spelling, so that each item can be found in the
//! headers; struct fields are snake-cased.

#![expect(
    non_upper_case_globals,
    reason = "libva's names, kept for grepping the headers"
)]

use core::ffi::{c_char, c_int, c_uint, c_void};

pub(crate) mod encode;

pub(crate) type VADisplay = *mut c_void;
pub(crate) type VAStatus = c_int;
pub(crate) type VAGenericID = c_uint;
pub(crate) type VAConfigID = VAGenericID;
pub(crate) type VAContextID = VAGenericID;
pub(crate) type VASurfaceID = VAGenericID;
pub(crate) type VABufferID = VAGenericID;
pub(crate) type VAImageID = VAGenericID;
/// C enums, as `int`.
pub(crate) type VAProfile = c_int;
pub(crate) type VAEntrypoint = c_int;
pub(crate) type VAConfigAttribType = c_int;
pub(crate) type VABufferType = c_int;
pub(crate) type VASurfaceAttribType = c_int;
pub(crate) type VAGenericValueType = c_int;
pub(crate) type VAProcFilterType = c_int;
pub(crate) type VAMessageCallback = Option<unsafe extern "C" fn(*mut c_void, *const c_char)>;

pub(crate) const VA_STATUS_SUCCESS: VAStatus = 0;
/// `VA_STATUS_ERROR_TIMEDOUT`: `vaSyncSurface2` and `vaSyncBuffer`'s
/// answer for work not done within their timeout.
pub(crate) const VA_STATUS_ERROR_TIMEDOUT: VAStatus = 0x26;
/// `VA_STATUS_ERROR_UNIMPLEMENTED`: libva's answer for a function the
/// driver lacks.
pub(crate) const VA_STATUS_ERROR_UNIMPLEMENTED: VAStatus = 0x14;
pub(crate) const VA_INVALID_ID: VAGenericID = 0xffff_ffff;
pub(crate) const VA_INVALID_SURFACE: VASurfaceID = VA_INVALID_ID;
pub(crate) const VA_ATTRIB_NOT_SUPPORTED: u32 = 0x8000_0000;
pub(crate) const VA_PROGRESSIVE: c_int = 0x1;

pub(crate) const VA_RT_FORMAT_YUV420: c_uint = 0x0000_0001;
pub(crate) const VA_RT_FORMAT_YUV420_10: c_uint = 0x0000_0100;

pub(crate) const VA_FOURCC_NV12: u32 = 0x3231_564e;
pub(crate) const VA_FOURCC_P010: u32 = 0x3031_3050;

pub(crate) const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

pub(crate) const VA_PICTURE_H264_INVALID: u32 = 0x01;
pub(crate) const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x08;
pub(crate) const VA_PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x10;

pub(crate) const VA_PICTURE_HEVC_INVALID: u32 = 0x01;
pub(crate) const VA_PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x08;
pub(crate) const VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x10;
pub(crate) const VA_PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x20;
pub(crate) const VA_PICTURE_HEVC_RPS_LT_CURR: u32 = 0x40;

// VAProfile.
pub(crate) const VAProfileNone: VAProfile = -1;
pub(crate) const VAProfileH264Main: VAProfile = 6;
pub(crate) const VAProfileH264High: VAProfile = 7;
pub(crate) const VAProfileH264ConstrainedBaseline: VAProfile = 13;
pub(crate) const VAProfileHEVCMain: VAProfile = 17;
pub(crate) const VAProfileHEVCMain10: VAProfile = 18;

// VAEntrypoint.
pub(crate) const VAEntrypointVLD: VAEntrypoint = 1;
pub(crate) const VAEntrypointVideoProc: VAEntrypoint = 10;

// VAConfigAttribType.
pub(crate) const VAConfigAttribRTFormat: VAConfigAttribType = 0;
pub(crate) const VAConfigAttribRateControl: VAConfigAttribType = 5;
pub(crate) const VAConfigAttribDecSliceMode: VAConfigAttribType = 6;
pub(crate) const VAConfigAttribEncPackedHeaders: VAConfigAttribType = 10;
pub(crate) const VAConfigAttribEncMaxRefFrames: VAConfigAttribType = 13;
pub(crate) const VAConfigAttribEncMaxSlices: VAConfigAttribType = 14;
pub(crate) const VAConfigAttribMaxPictureWidth: VAConfigAttribType = 18;
pub(crate) const VAConfigAttribMaxPictureHeight: VAConfigAttribType = 19;

// VASurfaceAttribType.
pub(crate) const VASurfaceAttribPixelFormat: VASurfaceAttribType = 1;
pub(crate) const VASurfaceAttribMinWidth: VASurfaceAttribType = 2;
pub(crate) const VASurfaceAttribMaxWidth: VASurfaceAttribType = 3;
pub(crate) const VASurfaceAttribMinHeight: VASurfaceAttribType = 4;
pub(crate) const VASurfaceAttribMaxHeight: VASurfaceAttribType = 5;
pub(crate) const VASurfaceAttribMemoryType: VASurfaceAttribType = 6;
pub(crate) const VASurfaceAttribExternalBufferDescriptor: VASurfaceAttribType = 7;

pub(crate) const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x2;

/// `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` (va_drmcommon.h): a surface
/// over dma-bufs, described by a [`VADRMPRIMESurfaceDescriptor`].
pub(crate) const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
/// `vaExportSurfaceHandle`'s flags (va.h).
pub(crate) const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
pub(crate) const VA_EXPORT_SURFACE_COMPOSED_LAYERS: u32 = 0x0008;
/// `DRM_FORMAT_MOD_LINEAR` (drm_fourcc.h).
pub(crate) const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_NV12` and `DRM_FORMAT_P010` (drm_fourcc.h): the same codes
/// as VA's fourccs.
pub(crate) const DRM_FORMAT_NV12: u32 = VA_FOURCC_NV12;
pub(crate) const DRM_FORMAT_P010: u32 = VA_FOURCC_P010;

// VAGenericValueType.
pub(crate) const VAGenericValueTypeInteger: VAGenericValueType = 1;
pub(crate) const VAGenericValueTypePointer: VAGenericValueType = 3;

// VABufferType.
pub(crate) const VAPictureParameterBufferType: VABufferType = 0;
pub(crate) const VAIQMatrixBufferType: VABufferType = 1;
pub(crate) const VASliceParameterBufferType: VABufferType = 4;
pub(crate) const VASliceDataBufferType: VABufferType = 5;

/// `VA_PADDING_LOW`, `_MEDIUM`, `_HIGH` and `_LARGE`, for every binding of
/// the crate's.
pub(crate) const VA_PADDING_LOW: usize = 4;
pub(crate) const VA_PADDING_MEDIUM: usize = 8;
pub(crate) const VA_PADDING_HIGH: usize = 16;
pub(crate) const VA_PADDING_LARGE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAConfigAttrib {
    pub(crate) type_: VAConfigAttribType,
    pub(crate) value: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union VAGenericValueUnion {
    pub(crate) i: i32,
    pub(crate) f: f32,
    pub(crate) p: *mut c_void,
    pub(crate) fn_: Option<unsafe extern "C" fn()>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VAGenericValue {
    pub(crate) type_: VAGenericValueType,
    pub(crate) value: VAGenericValueUnion,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VASurfaceAttrib {
    pub(crate) type_: VASurfaceAttribType,
    pub(crate) flags: u32,
    pub(crate) value: VAGenericValue,
}

impl VASurfaceAttrib {
    pub(crate) fn zeroed() -> Self {
        Self {
            type_: 0,
            flags: 0,
            value: VAGenericValue {
                type_: 0,
                value: VAGenericValueUnion {
                    p: core::ptr::null_mut(),
                },
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAImageFormat {
    pub(crate) fourcc: u32,
    pub(crate) byte_order: u32,
    pub(crate) bits_per_pixel: u32,
    pub(crate) depth: u32,
    pub(crate) red_mask: u32,
    pub(crate) green_mask: u32,
    pub(crate) blue_mask: u32,
    pub(crate) alpha_mask: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAImage {
    pub(crate) image_id: VAImageID,
    pub(crate) format: VAImageFormat,
    pub(crate) buf: VABufferID,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data_size: u32,
    pub(crate) num_planes: u32,
    pub(crate) pitches: [u32; 3],
    pub(crate) offsets: [u32; 3],
    pub(crate) num_palette_entries: i32,
    pub(crate) entry_bytes: i32,
    pub(crate) component_order: [i8; 4],
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// One dma-buf of a [`VADRMPRIMESurfaceDescriptor`] (va_drmcommon.h).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VADRMPRIMESurfaceObject {
    pub(crate) fd: c_int,
    pub(crate) size: u32,
    pub(crate) drm_format_modifier: u64,
}

/// One layer of a [`VADRMPRIMESurfaceDescriptor`]: its planes, each in
/// an object at an offset and pitch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VADRMPRIMESurfaceLayer {
    pub(crate) drm_format: u32,
    pub(crate) num_planes: u32,
    pub(crate) object_index: [u32; 4],
    pub(crate) offset: [u32; 4],
    pub(crate) pitch: [u32; 4],
}

/// `VADRMPRIMESurfaceDescriptor` (va_drmcommon.h): a surface as dma-bufs,
/// for `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` imports and exports.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VADRMPRIMESurfaceDescriptor {
    pub(crate) fourcc: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_objects: u32,
    pub(crate) objects: [VADRMPRIMESurfaceObject; 4],
    pub(crate) num_layers: u32,
    pub(crate) layers: [VADRMPRIMESurfaceLayer; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAPictureH264 {
    pub(crate) picture_id: VASurfaceID,
    pub(crate) frame_idx: u32,
    pub(crate) flags: u32,
    pub(crate) top_field_order_cnt: i32,
    pub(crate) bottom_field_order_cnt: i32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

impl VAPictureH264 {
    /// An unused entry, as libva's users mark one.
    pub(crate) fn invalid() -> Self {
        Self {
            picture_id: VA_INVALID_SURFACE,
            flags: VA_PICTURE_H264_INVALID,
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAPictureParameterBufferH264 {
    pub(crate) curr_pic: VAPictureH264,
    pub(crate) reference_frames: [VAPictureH264; 16],
    pub(crate) picture_width_in_mbs_minus1: u16,
    pub(crate) picture_height_in_mbs_minus1: u16,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) num_ref_frames: u8,
    /// `seq_fields.value`: see [`seq_fields`].
    pub(crate) seq_fields: u32,
    pub(crate) num_slice_groups_minus1: u8,
    pub(crate) slice_group_map_type: u8,
    pub(crate) slice_group_change_rate_minus1: u16,
    pub(crate) pic_init_qp_minus26: i8,
    pub(crate) pic_init_qs_minus26: i8,
    pub(crate) chroma_qp_index_offset: i8,
    pub(crate) second_chroma_qp_index_offset: i8,
    /// `pic_fields.value`: see [`pic_fields`].
    pub(crate) pic_fields: u32,
    pub(crate) frame_num: u16,
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAIQMatrixBufferH264 {
    pub(crate) scaling_list_4x4: [[u8; 16]; 6],
    pub(crate) scaling_list_8x8: [[u8; 64]; 2],
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VASliceParameterBufferH264 {
    pub(crate) slice_data_size: u32,
    pub(crate) slice_data_offset: u32,
    pub(crate) slice_data_flag: u32,
    pub(crate) slice_data_bit_offset: u16,
    pub(crate) first_mb_in_slice: u16,
    pub(crate) slice_type: u8,
    pub(crate) direct_spatial_mv_pred_flag: u8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) cabac_init_idc: u8,
    pub(crate) slice_qp_delta: i8,
    pub(crate) disable_deblocking_filter_idc: u8,
    pub(crate) slice_alpha_c0_offset_div2: i8,
    pub(crate) slice_beta_offset_div2: i8,
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
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAPictureHEVC {
    pub(crate) picture_id: VASurfaceID,
    pub(crate) pic_order_cnt: i32,
    pub(crate) flags: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

impl VAPictureHEVC {
    /// An unused entry, as libva's users mark one.
    pub(crate) fn invalid() -> Self {
        Self {
            picture_id: VA_INVALID_SURFACE,
            flags: VA_PICTURE_HEVC_INVALID,
            ..Self::default()
        }
    }
}

/// `VAPictureParameterBufferHEVC` (va_dec_hevc.h): the base structure,
/// all Main and Main 10 need.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAPictureParameterBufferHEVC {
    pub(crate) curr_pic: VAPictureHEVC,
    pub(crate) reference_frames: [VAPictureHEVC; 15],
    pub(crate) pic_width_in_luma_samples: u16,
    pub(crate) pic_height_in_luma_samples: u16,
    /// `pic_fields.value`: see [`hevc_pic_fields`].
    pub(crate) pic_fields: u32,
    pub(crate) sps_max_dec_pic_buffering_minus1: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) pcm_sample_bit_depth_luma_minus1: u8,
    pub(crate) pcm_sample_bit_depth_chroma_minus1: u8,
    pub(crate) log2_min_luma_coding_block_size_minus3: u8,
    pub(crate) log2_diff_max_min_luma_coding_block_size: u8,
    pub(crate) log2_min_transform_block_size_minus2: u8,
    pub(crate) log2_diff_max_min_transform_block_size: u8,
    pub(crate) log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub(crate) log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub(crate) max_transform_hierarchy_depth_intra: u8,
    pub(crate) max_transform_hierarchy_depth_inter: u8,
    pub(crate) init_qp_minus26: i8,
    pub(crate) diff_cu_qp_delta_depth: u8,
    pub(crate) pps_cb_qp_offset: i8,
    pub(crate) pps_cr_qp_offset: i8,
    pub(crate) log2_parallel_merge_level_minus2: u8,
    pub(crate) num_tile_columns_minus1: u8,
    pub(crate) num_tile_rows_minus1: u8,
    pub(crate) column_width_minus1: [u16; 19],
    pub(crate) row_height_minus1: [u16; 21],
    /// `slice_parsing_fields.value`: see [`hevc_slice_parsing_fields`].
    pub(crate) slice_parsing_fields: u32,
    pub(crate) log2_max_pic_order_cnt_lsb_minus4: u8,
    pub(crate) num_short_term_ref_pic_sets: u8,
    pub(crate) num_long_term_ref_pic_sps: u8,
    pub(crate) num_ref_idx_l0_default_active_minus1: u8,
    pub(crate) num_ref_idx_l1_default_active_minus1: u8,
    pub(crate) pps_beta_offset_div2: i8,
    pub(crate) pps_tc_offset_div2: i8,
    pub(crate) num_extra_slice_header_bits: u8,
    pub(crate) st_rps_bits: u32,
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASliceParameterBufferHEVC` (va_dec_hevc.h), the long format.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VASliceParameterBufferHEVC {
    pub(crate) slice_data_size: u32,
    pub(crate) slice_data_offset: u32,
    pub(crate) slice_data_flag: u32,
    pub(crate) slice_data_byte_offset: u32,
    pub(crate) slice_segment_address: u32,
    pub(crate) ref_pic_list: [[u8; 15]; 2],
    /// `LongSliceFlags.value`: see [`hevc_long_slice_flags`].
    pub(crate) long_slice_flags: u32,
    pub(crate) collocated_ref_idx: u8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) slice_qp_delta: i8,
    pub(crate) slice_cb_qp_offset: i8,
    pub(crate) slice_cr_qp_offset: i8,
    pub(crate) slice_beta_offset_div2: i8,
    pub(crate) slice_tc_offset_div2: i8,
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
    pub(crate) five_minus_max_num_merge_cand: u8,
    pub(crate) num_entry_point_offsets: u16,
    pub(crate) entry_offset_to_subset_array: u16,
    pub(crate) slice_data_num_emu_prevn_bytes: u16,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW - 2],
}

/// `VAIQMatrixBufferHEVC` (va_dec_hevc.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAIQMatrixBufferHEVC {
    pub(crate) scaling_list_4x4: [[u8; 16]; 6],
    pub(crate) scaling_list_8x8: [[u8; 64]; 6],
    pub(crate) scaling_list_16x16: [[u8; 64]; 6],
    pub(crate) scaling_list_32x32: [[u8; 64]; 2],
    pub(crate) scaling_list_dc_16x16: [u8; 6],
    pub(crate) scaling_list_dc_32x32: [u8; 2],
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// A bit-field of a `uint32_t` union in va.h: its lowest bit and width.
/// GCC and Clang lay x86-64 bit-fields out from bit 0 in declaration
/// order; the layout test checks each against the C compiler.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BitField {
    pub(crate) shift: u32,
    pub(crate) width: u32,
}

impl BitField {
    const fn new(shift: u32, width: u32) -> Self {
        Self { shift, width }
    }

    /// `value` placed in the field, with what does not fit dropped.
    pub(crate) fn put(self, value: u32) -> u32 {
        (value & ((1 << self.width) - 1)) << self.shift
    }
}

/// `VAPictureParameterBufferH264::seq_fields.bits`.
pub(crate) mod seq_fields {
    use super::BitField;
    pub(crate) const CHROMA_FORMAT_IDC: BitField = BitField::new(0, 2);
    pub(crate) const RESIDUAL_COLOUR_TRANSFORM_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const GAPS_IN_FRAME_NUM_VALUE_ALLOWED_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const FRAME_MBS_ONLY_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const MB_ADAPTIVE_FRAME_FIELD_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const DIRECT_8X8_INFERENCE_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const MIN_LUMA_BI_PRED_SIZE8X8: BitField = BitField::new(7, 1);
    pub(crate) const LOG2_MAX_FRAME_NUM_MINUS4: BitField = BitField::new(8, 4);
    pub(crate) const PIC_ORDER_CNT_TYPE: BitField = BitField::new(12, 2);
    pub(crate) const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: BitField = BitField::new(14, 4);
    pub(crate) const DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG: BitField = BitField::new(18, 1);
}

/// `VAPictureParameterBufferH264::pic_fields.bits`.
pub(crate) mod pic_fields {
    use super::BitField;
    pub(crate) const ENTROPY_CODING_MODE_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const WEIGHTED_PRED_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const WEIGHTED_BIPRED_IDC: BitField = BitField::new(2, 2);
    pub(crate) const TRANSFORM_8X8_MODE_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const FIELD_PIC_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const CONSTRAINED_INTRA_PRED_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const PIC_ORDER_PRESENT_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const REDUNDANT_PIC_CNT_PRESENT_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const REFERENCE_PIC_FLAG: BitField = BitField::new(10, 1);
}

/// `VAPictureParameterBufferHEVC::pic_fields.bits`.
pub(crate) mod hevc_pic_fields {
    use super::BitField;
    pub(crate) const CHROMA_FORMAT_IDC: BitField = BitField::new(0, 2);
    pub(crate) const SEPARATE_COLOUR_PLANE_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const PCM_ENABLED_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const SCALING_LIST_ENABLED_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const TRANSFORM_SKIP_ENABLED_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const AMP_ENABLED_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const STRONG_INTRA_SMOOTHING_ENABLED_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const SIGN_DATA_HIDING_ENABLED_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const CONSTRAINED_INTRA_PRED_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const CU_QP_DELTA_ENABLED_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const WEIGHTED_PRED_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const WEIGHTED_BIPRED_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const TRANSQUANT_BYPASS_ENABLED_FLAG: BitField = BitField::new(13, 1);
    pub(crate) const TILES_ENABLED_FLAG: BitField = BitField::new(14, 1);
    pub(crate) const ENTROPY_CODING_SYNC_ENABLED_FLAG: BitField = BitField::new(15, 1);
    pub(crate) const PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: BitField = BitField::new(16, 1);
    pub(crate) const LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: BitField = BitField::new(17, 1);
    pub(crate) const PCM_LOOP_FILTER_DISABLED_FLAG: BitField = BitField::new(18, 1);
    pub(crate) const NO_PIC_REORDERING_FLAG: BitField = BitField::new(19, 1);
    pub(crate) const NO_BI_PRED_FLAG: BitField = BitField::new(20, 1);
}

/// `VAPictureParameterBufferHEVC::slice_parsing_fields.bits`.
pub(crate) mod hevc_slice_parsing_fields {
    use super::BitField;
    pub(crate) const LISTS_MODIFICATION_PRESENT_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const LONG_TERM_REF_PICS_PRESENT_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const SPS_TEMPORAL_MVP_ENABLED_FLAG: BitField = BitField::new(2, 1);
    pub(crate) const CABAC_INIT_PRESENT_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const OUTPUT_FLAG_PRESENT_FLAG: BitField = BitField::new(4, 1);
    pub(crate) const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: BitField = BitField::new(5, 1);
    pub(crate) const PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const PPS_DISABLE_DEBLOCKING_FILTER_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const RAP_PIC_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const IDR_PIC_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const INTRA_PIC_FLAG: BitField = BitField::new(13, 1);
}

/// `VASliceParameterBufferHEVC::LongSliceFlags.fields`.
pub(crate) mod hevc_long_slice_flags {
    use super::BitField;
    pub(crate) const LAST_SLICE_OF_PIC: BitField = BitField::new(0, 1);
    pub(crate) const DEPENDENT_SLICE_SEGMENT_FLAG: BitField = BitField::new(1, 1);
    pub(crate) const SLICE_TYPE: BitField = BitField::new(2, 2);
    pub(crate) const COLOR_PLANE_ID: BitField = BitField::new(4, 2);
    pub(crate) const SLICE_SAO_LUMA_FLAG: BitField = BitField::new(6, 1);
    pub(crate) const SLICE_SAO_CHROMA_FLAG: BitField = BitField::new(7, 1);
    pub(crate) const MVD_L1_ZERO_FLAG: BitField = BitField::new(8, 1);
    pub(crate) const CABAC_INIT_FLAG: BitField = BitField::new(9, 1);
    pub(crate) const SLICE_TEMPORAL_MVP_ENABLED_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const SLICE_DEBLOCKING_FILTER_DISABLED_FLAG: BitField = BitField::new(11, 1);
    pub(crate) const COLLOCATED_FROM_L0_FLAG: BitField = BitField::new(12, 1);
    pub(crate) const SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: BitField = BitField::new(13, 1);
}

// libva's functions, loaded at run time rather than linked (`super::libva`):
// wrappers of the same names and signatures.
pub(crate) use super::libva::{
    vaBeginPicture, vaCreateBuffer, vaCreateConfig, vaCreateContext, vaCreateImage,
    vaCreateSurfaces, vaDeriveImage, vaDestroyBuffer, vaDestroyConfig, vaDestroyContext,
    vaDestroyImage, vaDestroySurfaces, vaEndPicture, vaErrorStr, vaExportSurfaceHandle,
    vaGetConfigAttributes, vaGetDisplayDRM, vaGetImage, vaInitialize, vaMapBuffer,
    vaMaxNumEntrypoints, vaMaxNumImageFormats, vaMaxNumProfiles, vaPutImage,
    vaQueryConfigEntrypoints, vaQueryConfigProfiles, vaQueryImageFormats, vaQuerySurfaceAttributes,
    vaQueryVendorString, vaQueryVideoProcFilters, vaRenderPicture, vaSetInfoCallback,
    vaSyncSurface, vaSyncSurface2, vaTerminate, vaUnmapBuffer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// What the C side prints, line for line: `sizeof` and `offsetof` of
    /// each struct and field below, and each bit-field's mask.
    const LAYOUT_C: &str = include_str!("ffi_layout.c");

    macro_rules! layout {
        ($out:ident, $ty:ty, $c:literal, { $($field:ident => $cfield:literal),* $(,)? }) => {
            $out.push(format!("{} size {}", $c, size_of::<$ty>()));
            $( $out.push(format!("{}.{} {}", $c, $cfield, offset_of!($ty, $field))); )*
        };
    }

    /// `name value` for each constant, in decimal, as the C side's `C`
    /// prints it.
    macro_rules! constants {
        ($out:ident, $($name:ident),* $(,)?) => {
            $( $out.push(format!("{} {}", stringify!($name), $name)); )*
        };
    }

    fn rust_layout() -> Vec<String> {
        let mut out = Vec::new();
        layout!(out, VAConfigAttrib, "VAConfigAttrib", { type_ => "type", value => "value" });
        layout!(out, VAGenericValue, "VAGenericValue", { type_ => "type", value => "value" });
        layout!(out, VASurfaceAttrib, "VASurfaceAttrib", {
            type_ => "type", flags => "flags", value => "value",
        });
        layout!(out, VAImageFormat, "VAImageFormat", {
            fourcc => "fourcc", byte_order => "byte_order", bits_per_pixel => "bits_per_pixel",
            depth => "depth", red_mask => "red_mask", green_mask => "green_mask",
            blue_mask => "blue_mask", alpha_mask => "alpha_mask", va_reserved => "va_reserved",
        });
        layout!(out, VAImage, "VAImage", {
            image_id => "image_id", format => "format", buf => "buf", width => "width",
            height => "height", data_size => "data_size", num_planes => "num_planes",
            pitches => "pitches", offsets => "offsets",
            num_palette_entries => "num_palette_entries", entry_bytes => "entry_bytes",
            component_order => "component_order", va_reserved => "va_reserved",
        });
        layout!(out, VADRMPRIMESurfaceDescriptor, "VADRMPRIMESurfaceDescriptor", {
            fourcc => "fourcc", width => "width", height => "height",
            num_objects => "num_objects", objects => "objects", num_layers => "num_layers",
            layers => "layers",
        });
        let object = |n: usize, field: usize| {
            offset_of!(VADRMPRIMESurfaceDescriptor, objects)
                + n * size_of::<VADRMPRIMESurfaceObject>()
                + field
        };
        out.push(format!(
            "VADRMPRIMESurfaceDescriptor.objects[1].drm_format_modifier {}",
            object(1, offset_of!(VADRMPRIMESurfaceObject, drm_format_modifier))
        ));
        let layer = |n: usize, field: usize| {
            offset_of!(VADRMPRIMESurfaceDescriptor, layers)
                + n * size_of::<VADRMPRIMESurfaceLayer>()
                + field
        };
        out.push(format!(
            "VADRMPRIMESurfaceDescriptor.layers[1].pitch {}",
            layer(1, offset_of!(VADRMPRIMESurfaceLayer, pitch))
        ));
        layout!(out, VAPictureH264, "VAPictureH264", {
            picture_id => "picture_id", frame_idx => "frame_idx", flags => "flags",
            top_field_order_cnt => "TopFieldOrderCnt",
            bottom_field_order_cnt => "BottomFieldOrderCnt", va_reserved => "va_reserved",
        });
        layout!(out, VAPictureParameterBufferH264, "VAPictureParameterBufferH264", {
            curr_pic => "CurrPic", reference_frames => "ReferenceFrames",
            picture_width_in_mbs_minus1 => "picture_width_in_mbs_minus1",
            picture_height_in_mbs_minus1 => "picture_height_in_mbs_minus1",
            bit_depth_luma_minus8 => "bit_depth_luma_minus8",
            bit_depth_chroma_minus8 => "bit_depth_chroma_minus8",
            num_ref_frames => "num_ref_frames", seq_fields => "seq_fields",
            num_slice_groups_minus1 => "num_slice_groups_minus1",
            slice_group_map_type => "slice_group_map_type",
            slice_group_change_rate_minus1 => "slice_group_change_rate_minus1",
            pic_init_qp_minus26 => "pic_init_qp_minus26",
            pic_init_qs_minus26 => "pic_init_qs_minus26",
            chroma_qp_index_offset => "chroma_qp_index_offset",
            second_chroma_qp_index_offset => "second_chroma_qp_index_offset",
            pic_fields => "pic_fields", frame_num => "frame_num", va_reserved => "va_reserved",
        });
        layout!(out, VAIQMatrixBufferH264, "VAIQMatrixBufferH264", {
            scaling_list_4x4 => "ScalingList4x4", scaling_list_8x8 => "ScalingList8x8",
            va_reserved => "va_reserved",
        });
        layout!(out, VASliceParameterBufferH264, "VASliceParameterBufferH264", {
            slice_data_size => "slice_data_size", slice_data_offset => "slice_data_offset",
            slice_data_flag => "slice_data_flag",
            slice_data_bit_offset => "slice_data_bit_offset",
            first_mb_in_slice => "first_mb_in_slice", slice_type => "slice_type",
            direct_spatial_mv_pred_flag => "direct_spatial_mv_pred_flag",
            num_ref_idx_l0_active_minus1 => "num_ref_idx_l0_active_minus1",
            num_ref_idx_l1_active_minus1 => "num_ref_idx_l1_active_minus1",
            cabac_init_idc => "cabac_init_idc", slice_qp_delta => "slice_qp_delta",
            disable_deblocking_filter_idc => "disable_deblocking_filter_idc",
            slice_alpha_c0_offset_div2 => "slice_alpha_c0_offset_div2",
            slice_beta_offset_div2 => "slice_beta_offset_div2",
            ref_pic_list0 => "RefPicList0", ref_pic_list1 => "RefPicList1",
            luma_log2_weight_denom => "luma_log2_weight_denom",
            chroma_log2_weight_denom => "chroma_log2_weight_denom",
            luma_weight_l0_flag => "luma_weight_l0_flag", luma_weight_l0 => "luma_weight_l0",
            luma_offset_l0 => "luma_offset_l0", chroma_weight_l0_flag => "chroma_weight_l0_flag",
            chroma_weight_l0 => "chroma_weight_l0", chroma_offset_l0 => "chroma_offset_l0",
            luma_weight_l1_flag => "luma_weight_l1_flag", luma_weight_l1 => "luma_weight_l1",
            luma_offset_l1 => "luma_offset_l1", chroma_weight_l1_flag => "chroma_weight_l1_flag",
            chroma_weight_l1 => "chroma_weight_l1", chroma_offset_l1 => "chroma_offset_l1",
            va_reserved => "va_reserved",
        });
        layout!(out, VAPictureHEVC, "VAPictureHEVC", {
            picture_id => "picture_id", pic_order_cnt => "pic_order_cnt", flags => "flags",
            va_reserved => "va_reserved",
        });
        layout!(out, VAPictureParameterBufferHEVC, "VAPictureParameterBufferHEVC", {
            curr_pic => "CurrPic", reference_frames => "ReferenceFrames",
            pic_width_in_luma_samples => "pic_width_in_luma_samples",
            pic_height_in_luma_samples => "pic_height_in_luma_samples",
            pic_fields => "pic_fields",
            sps_max_dec_pic_buffering_minus1 => "sps_max_dec_pic_buffering_minus1",
            bit_depth_luma_minus8 => "bit_depth_luma_minus8",
            bit_depth_chroma_minus8 => "bit_depth_chroma_minus8",
            pcm_sample_bit_depth_luma_minus1 => "pcm_sample_bit_depth_luma_minus1",
            pcm_sample_bit_depth_chroma_minus1 => "pcm_sample_bit_depth_chroma_minus1",
            log2_min_luma_coding_block_size_minus3 => "log2_min_luma_coding_block_size_minus3",
            log2_diff_max_min_luma_coding_block_size => "log2_diff_max_min_luma_coding_block_size",
            log2_min_transform_block_size_minus2 => "log2_min_transform_block_size_minus2",
            log2_diff_max_min_transform_block_size => "log2_diff_max_min_transform_block_size",
            log2_min_pcm_luma_coding_block_size_minus3 =>
                "log2_min_pcm_luma_coding_block_size_minus3",
            log2_diff_max_min_pcm_luma_coding_block_size =>
                "log2_diff_max_min_pcm_luma_coding_block_size",
            max_transform_hierarchy_depth_intra => "max_transform_hierarchy_depth_intra",
            max_transform_hierarchy_depth_inter => "max_transform_hierarchy_depth_inter",
            init_qp_minus26 => "init_qp_minus26", diff_cu_qp_delta_depth => "diff_cu_qp_delta_depth",
            pps_cb_qp_offset => "pps_cb_qp_offset", pps_cr_qp_offset => "pps_cr_qp_offset",
            log2_parallel_merge_level_minus2 => "log2_parallel_merge_level_minus2",
            num_tile_columns_minus1 => "num_tile_columns_minus1",
            num_tile_rows_minus1 => "num_tile_rows_minus1",
            column_width_minus1 => "column_width_minus1", row_height_minus1 => "row_height_minus1",
            slice_parsing_fields => "slice_parsing_fields",
            log2_max_pic_order_cnt_lsb_minus4 => "log2_max_pic_order_cnt_lsb_minus4",
            num_short_term_ref_pic_sets => "num_short_term_ref_pic_sets",
            num_long_term_ref_pic_sps => "num_long_term_ref_pic_sps",
            num_ref_idx_l0_default_active_minus1 => "num_ref_idx_l0_default_active_minus1",
            num_ref_idx_l1_default_active_minus1 => "num_ref_idx_l1_default_active_minus1",
            pps_beta_offset_div2 => "pps_beta_offset_div2",
            pps_tc_offset_div2 => "pps_tc_offset_div2",
            num_extra_slice_header_bits => "num_extra_slice_header_bits",
            st_rps_bits => "st_rps_bits", va_reserved => "va_reserved",
        });
        layout!(out, VASliceParameterBufferHEVC, "VASliceParameterBufferHEVC", {
            slice_data_size => "slice_data_size", slice_data_offset => "slice_data_offset",
            slice_data_flag => "slice_data_flag",
            slice_data_byte_offset => "slice_data_byte_offset",
            slice_segment_address => "slice_segment_address", ref_pic_list => "RefPicList",
            long_slice_flags => "LongSliceFlags", collocated_ref_idx => "collocated_ref_idx",
            num_ref_idx_l0_active_minus1 => "num_ref_idx_l0_active_minus1",
            num_ref_idx_l1_active_minus1 => "num_ref_idx_l1_active_minus1",
            slice_qp_delta => "slice_qp_delta", slice_cb_qp_offset => "slice_cb_qp_offset",
            slice_cr_qp_offset => "slice_cr_qp_offset",
            slice_beta_offset_div2 => "slice_beta_offset_div2",
            slice_tc_offset_div2 => "slice_tc_offset_div2",
            luma_log2_weight_denom => "luma_log2_weight_denom",
            delta_chroma_log2_weight_denom => "delta_chroma_log2_weight_denom",
            delta_luma_weight_l0 => "delta_luma_weight_l0", luma_offset_l0 => "luma_offset_l0",
            delta_chroma_weight_l0 => "delta_chroma_weight_l0",
            chroma_offset_l0 => "ChromaOffsetL0",
            delta_luma_weight_l1 => "delta_luma_weight_l1", luma_offset_l1 => "luma_offset_l1",
            delta_chroma_weight_l1 => "delta_chroma_weight_l1",
            chroma_offset_l1 => "ChromaOffsetL1",
            five_minus_max_num_merge_cand => "five_minus_max_num_merge_cand",
            num_entry_point_offsets => "num_entry_point_offsets",
            entry_offset_to_subset_array => "entry_offset_to_subset_array",
            slice_data_num_emu_prevn_bytes => "slice_data_num_emu_prevn_bytes",
            va_reserved => "va_reserved",
        });
        layout!(out, VAIQMatrixBufferHEVC, "VAIQMatrixBufferHEVC", {
            scaling_list_4x4 => "ScalingList4x4", scaling_list_8x8 => "ScalingList8x8",
            scaling_list_16x16 => "ScalingList16x16", scaling_list_32x32 => "ScalingList32x32",
            scaling_list_dc_16x16 => "ScalingListDC16x16",
            scaling_list_dc_32x32 => "ScalingListDC32x32", va_reserved => "va_reserved",
        });
        let seq = [
            ("chroma_format_idc", seq_fields::CHROMA_FORMAT_IDC),
            (
                "residual_colour_transform_flag",
                seq_fields::RESIDUAL_COLOUR_TRANSFORM_FLAG,
            ),
            (
                "gaps_in_frame_num_value_allowed_flag",
                seq_fields::GAPS_IN_FRAME_NUM_VALUE_ALLOWED_FLAG,
            ),
            ("frame_mbs_only_flag", seq_fields::FRAME_MBS_ONLY_FLAG),
            (
                "mb_adaptive_frame_field_flag",
                seq_fields::MB_ADAPTIVE_FRAME_FIELD_FLAG,
            ),
            (
                "direct_8x8_inference_flag",
                seq_fields::DIRECT_8X8_INFERENCE_FLAG,
            ),
            ("MinLumaBiPredSize8x8", seq_fields::MIN_LUMA_BI_PRED_SIZE8X8),
            (
                "log2_max_frame_num_minus4",
                seq_fields::LOG2_MAX_FRAME_NUM_MINUS4,
            ),
            ("pic_order_cnt_type", seq_fields::PIC_ORDER_CNT_TYPE),
            (
                "log2_max_pic_order_cnt_lsb_minus4",
                seq_fields::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
            ),
            (
                "delta_pic_order_always_zero_flag",
                seq_fields::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG,
            ),
        ];
        for (name, field) in seq {
            out.push(format!("seq_fields.{name} {:#x}", field.put(u32::MAX)));
        }
        let pic = [
            (
                "entropy_coding_mode_flag",
                pic_fields::ENTROPY_CODING_MODE_FLAG,
            ),
            ("weighted_pred_flag", pic_fields::WEIGHTED_PRED_FLAG),
            ("weighted_bipred_idc", pic_fields::WEIGHTED_BIPRED_IDC),
            (
                "transform_8x8_mode_flag",
                pic_fields::TRANSFORM_8X8_MODE_FLAG,
            ),
            ("field_pic_flag", pic_fields::FIELD_PIC_FLAG),
            (
                "constrained_intra_pred_flag",
                pic_fields::CONSTRAINED_INTRA_PRED_FLAG,
            ),
            ("pic_order_present_flag", pic_fields::PIC_ORDER_PRESENT_FLAG),
            (
                "deblocking_filter_control_present_flag",
                pic_fields::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
            ),
            (
                "redundant_pic_cnt_present_flag",
                pic_fields::REDUNDANT_PIC_CNT_PRESENT_FLAG,
            ),
            ("reference_pic_flag", pic_fields::REFERENCE_PIC_FLAG),
        ];
        for (name, field) in pic {
            out.push(format!("pic_fields.{name} {:#x}", field.put(u32::MAX)));
        }
        {
            use hevc_pic_fields as f;
            let fields = [
                ("chroma_format_idc", f::CHROMA_FORMAT_IDC),
                ("separate_colour_plane_flag", f::SEPARATE_COLOUR_PLANE_FLAG),
                ("pcm_enabled_flag", f::PCM_ENABLED_FLAG),
                ("scaling_list_enabled_flag", f::SCALING_LIST_ENABLED_FLAG),
                (
                    "transform_skip_enabled_flag",
                    f::TRANSFORM_SKIP_ENABLED_FLAG,
                ),
                ("amp_enabled_flag", f::AMP_ENABLED_FLAG),
                (
                    "strong_intra_smoothing_enabled_flag",
                    f::STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
                ),
                (
                    "sign_data_hiding_enabled_flag",
                    f::SIGN_DATA_HIDING_ENABLED_FLAG,
                ),
                (
                    "constrained_intra_pred_flag",
                    f::CONSTRAINED_INTRA_PRED_FLAG,
                ),
                ("cu_qp_delta_enabled_flag", f::CU_QP_DELTA_ENABLED_FLAG),
                ("weighted_pred_flag", f::WEIGHTED_PRED_FLAG),
                ("weighted_bipred_flag", f::WEIGHTED_BIPRED_FLAG),
                (
                    "transquant_bypass_enabled_flag",
                    f::TRANSQUANT_BYPASS_ENABLED_FLAG,
                ),
                ("tiles_enabled_flag", f::TILES_ENABLED_FLAG),
                (
                    "entropy_coding_sync_enabled_flag",
                    f::ENTROPY_CODING_SYNC_ENABLED_FLAG,
                ),
                (
                    "pps_loop_filter_across_slices_enabled_flag",
                    f::PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
                ),
                (
                    "loop_filter_across_tiles_enabled_flag",
                    f::LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG,
                ),
                (
                    "pcm_loop_filter_disabled_flag",
                    f::PCM_LOOP_FILTER_DISABLED_FLAG,
                ),
                ("NoPicReorderingFlag", f::NO_PIC_REORDERING_FLAG),
                ("NoBiPredFlag", f::NO_BI_PRED_FLAG),
            ];
            for (name, field) in fields {
                out.push(format!("hevc.pic_fields.{name} {:#x}", field.put(u32::MAX)));
            }
        }
        {
            use hevc_slice_parsing_fields as f;
            let fields = [
                (
                    "lists_modification_present_flag",
                    f::LISTS_MODIFICATION_PRESENT_FLAG,
                ),
                (
                    "long_term_ref_pics_present_flag",
                    f::LONG_TERM_REF_PICS_PRESENT_FLAG,
                ),
                (
                    "sps_temporal_mvp_enabled_flag",
                    f::SPS_TEMPORAL_MVP_ENABLED_FLAG,
                ),
                ("cabac_init_present_flag", f::CABAC_INIT_PRESENT_FLAG),
                ("output_flag_present_flag", f::OUTPUT_FLAG_PRESENT_FLAG),
                (
                    "dependent_slice_segments_enabled_flag",
                    f::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG,
                ),
                (
                    "pps_slice_chroma_qp_offsets_present_flag",
                    f::PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG,
                ),
                (
                    "sample_adaptive_offset_enabled_flag",
                    f::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG,
                ),
                (
                    "deblocking_filter_override_enabled_flag",
                    f::DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG,
                ),
                (
                    "pps_disable_deblocking_filter_flag",
                    f::PPS_DISABLE_DEBLOCKING_FILTER_FLAG,
                ),
                (
                    "slice_segment_header_extension_present_flag",
                    f::SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG,
                ),
                ("RapPicFlag", f::RAP_PIC_FLAG),
                ("IdrPicFlag", f::IDR_PIC_FLAG),
                ("IntraPicFlag", f::INTRA_PIC_FLAG),
            ];
            for (name, field) in fields {
                out.push(format!(
                    "hevc.slice_parsing_fields.{name} {:#x}",
                    field.put(u32::MAX)
                ));
            }
        }
        {
            use hevc_long_slice_flags as f;
            let fields = [
                ("LastSliceOfPic", f::LAST_SLICE_OF_PIC),
                (
                    "dependent_slice_segment_flag",
                    f::DEPENDENT_SLICE_SEGMENT_FLAG,
                ),
                ("slice_type", f::SLICE_TYPE),
                ("color_plane_id", f::COLOR_PLANE_ID),
                ("slice_sao_luma_flag", f::SLICE_SAO_LUMA_FLAG),
                ("slice_sao_chroma_flag", f::SLICE_SAO_CHROMA_FLAG),
                ("mvd_l1_zero_flag", f::MVD_L1_ZERO_FLAG),
                ("cabac_init_flag", f::CABAC_INIT_FLAG),
                (
                    "slice_temporal_mvp_enabled_flag",
                    f::SLICE_TEMPORAL_MVP_ENABLED_FLAG,
                ),
                (
                    "slice_deblocking_filter_disabled_flag",
                    f::SLICE_DEBLOCKING_FILTER_DISABLED_FLAG,
                ),
                ("collocated_from_l0_flag", f::COLLOCATED_FROM_L0_FLAG),
                (
                    "slice_loop_filter_across_slices_enabled_flag",
                    f::SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
                ),
            ];
            for (name, field) in fields {
                out.push(format!(
                    "hevc.LongSliceFlags.{name} {:#x}",
                    field.put(u32::MAX)
                ));
            }
        }
        // Each constant, as the C side prints it.
        constants!(
            out,
            VA_STATUS_SUCCESS,
            VA_STATUS_ERROR_TIMEDOUT,
            VA_STATUS_ERROR_UNIMPLEMENTED,
            VA_INVALID_ID,
            VA_INVALID_SURFACE,
            VA_ATTRIB_NOT_SUPPORTED,
            VA_PROGRESSIVE,
            VA_RT_FORMAT_YUV420,
            VA_RT_FORMAT_YUV420_10,
            VA_FOURCC_NV12,
            VA_FOURCC_P010,
            VA_SLICE_DATA_FLAG_ALL,
            VA_PICTURE_H264_INVALID,
            VA_PICTURE_H264_SHORT_TERM_REFERENCE,
            VA_PICTURE_H264_LONG_TERM_REFERENCE,
            VA_PICTURE_HEVC_INVALID,
            VA_PICTURE_HEVC_LONG_TERM_REFERENCE,
            VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
            VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
            VA_PICTURE_HEVC_RPS_LT_CURR,
            VAProfileNone,
            VAProfileH264Main,
            VAProfileH264High,
            VAProfileH264ConstrainedBaseline,
            VAProfileHEVCMain,
            VAProfileHEVCMain10,
            VAEntrypointVLD,
            VAEntrypointVideoProc,
            VAConfigAttribRTFormat,
            VAConfigAttribRateControl,
            VAConfigAttribDecSliceMode,
            VAConfigAttribEncPackedHeaders,
            VAConfigAttribEncMaxRefFrames,
            VAConfigAttribEncMaxSlices,
            VAConfigAttribMaxPictureWidth,
            VAConfigAttribMaxPictureHeight,
            VASurfaceAttribPixelFormat,
            VASurfaceAttribMinWidth,
            VASurfaceAttribMaxWidth,
            VASurfaceAttribMinHeight,
            VASurfaceAttribMaxHeight,
            VASurfaceAttribMemoryType,
            VASurfaceAttribExternalBufferDescriptor,
            VA_SURFACE_ATTRIB_SETTABLE,
            VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
            VA_EXPORT_SURFACE_READ_ONLY,
            VA_EXPORT_SURFACE_COMPOSED_LAYERS,
            DRM_FORMAT_MOD_LINEAR,
            DRM_FORMAT_NV12,
            DRM_FORMAT_P010,
            VAGenericValueTypeInteger,
            VAGenericValueTypePointer,
            VAPictureParameterBufferType,
            VAIQMatrixBufferType,
            VASliceParameterBufferType,
            VASliceDataBufferType,
            VA_PADDING_LOW,
            VA_PADDING_MEDIUM,
            VA_PADDING_HIGH,
            VA_PADDING_LARGE
        );
        out
    }

    /// Compiles `ffi_layout.c` against the installed libva headers and
    /// compares its output with this module's layout. `CC` picks the
    /// compiler (`cc` by default).
    #[test]
    #[ignore = "needs libva's headers and a C compiler: run on the test host"]
    fn layouts_match_libva_headers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("rvgv-va-layout-{}", std::process::id()));
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
