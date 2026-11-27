//! libva's AV1 decode interface, hand-written from libva 2.24.1's `va_dec_av1.h` (VA-API
//! 1.24.0). The test at the end checks every struct's size and offsets, every bit-field's
//! position and every constant's value against that header with a C compiler, as the parent
//! module's test does for H.264 and HEVC.
//!
//! Names keep libva's spelling; struct fields are snake-cased.

#![expect(
    non_upper_case_globals,
    reason = "libva's names, kept for grepping the headers"
)]

use core::ffi::c_int;

use super::{BitField, VA_PADDING_LOW, VA_PADDING_MEDIUM, VAProfile, VASurfaceID};

pub(crate) const VAProfileAV1Profile0: VAProfile = 32;

/// `VASegmentationStructAV1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VASegmentationStructAV1 {
    pub(crate) segment_info_fields: u32,
    pub(crate) feature_data: [[i16; 8]; 8],
    pub(crate) feature_mask: [u8; 8],
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAFilmGrainStructAV1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAFilmGrainStructAV1 {
    pub(crate) film_grain_info_fields: u32,
    pub(crate) grain_seed: u16,
    pub(crate) num_y_points: u8,
    pub(crate) point_y_value: [u8; 14],
    pub(crate) point_y_scaling: [u8; 14],
    pub(crate) num_cb_points: u8,
    pub(crate) point_cb_value: [u8; 10],
    pub(crate) point_cb_scaling: [u8; 10],
    pub(crate) num_cr_points: u8,
    pub(crate) point_cr_value: [u8; 10],
    pub(crate) point_cr_scaling: [u8; 10],
    pub(crate) ar_coeffs_y: [i8; 24],
    pub(crate) ar_coeffs_cb: [i8; 25],
    pub(crate) ar_coeffs_cr: [i8; 25],
    pub(crate) cb_mult: u8,
    pub(crate) cb_luma_mult: u8,
    pub(crate) cb_offset: u16,
    pub(crate) cr_mult: u8,
    pub(crate) cr_luma_mult: u8,
    pub(crate) cr_offset: u16,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAWarpedMotionParamsAV1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAWarpedMotionParamsAV1 {
    /// A `VAAV1TransformationType`: `GmType`.
    pub(crate) wmtype: c_int,
    pub(crate) wmmat: [i32; 8],
    pub(crate) invalid: u8,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VADecPictureParameterBufferAV1`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VADecPictureParameterBufferAV1 {
    pub(crate) profile: u8,
    pub(crate) order_hint_bits_minus_1: u8,
    pub(crate) bit_depth_idx: u8,
    pub(crate) matrix_coefficients: u8,
    pub(crate) seq_info_fields: u32,
    pub(crate) current_frame: VASurfaceID,
    pub(crate) current_display_picture: VASurfaceID,
    pub(crate) anchor_frames_num: u8,
    pub(crate) anchor_frames_list: *mut VASurfaceID,
    pub(crate) frame_width_minus1: u16,
    pub(crate) frame_height_minus1: u16,
    pub(crate) output_frame_width_in_tiles_minus_1: u16,
    pub(crate) output_frame_height_in_tiles_minus_1: u16,
    pub(crate) ref_frame_map: [VASurfaceID; 8],
    pub(crate) ref_frame_idx: [u8; 7],
    pub(crate) primary_ref_frame: u8,
    pub(crate) order_hint: u8,
    pub(crate) seg_info: VASegmentationStructAV1,
    pub(crate) film_grain_info: VAFilmGrainStructAV1,
    pub(crate) tile_cols: u8,
    pub(crate) tile_rows: u8,
    pub(crate) width_in_sbs_minus_1: [u16; 63],
    pub(crate) height_in_sbs_minus_1: [u16; 63],
    pub(crate) tile_count_minus_1: u16,
    pub(crate) context_update_tile_id: u16,
    pub(crate) pic_info_fields: u32,
    pub(crate) superres_scale_denominator: u8,
    pub(crate) interp_filter: u8,
    pub(crate) filter_level: [u8; 2],
    pub(crate) filter_level_u: u8,
    pub(crate) filter_level_v: u8,
    pub(crate) loop_filter_info_fields: u8,
    pub(crate) ref_deltas: [i8; 8],
    pub(crate) mode_deltas: [i8; 2],
    pub(crate) base_qindex: u8,
    pub(crate) y_dc_delta_q: i8,
    pub(crate) u_dc_delta_q: i8,
    pub(crate) u_ac_delta_q: i8,
    pub(crate) v_dc_delta_q: i8,
    pub(crate) v_ac_delta_q: i8,
    pub(crate) qmatrix_fields: u16,
    pub(crate) mode_control_fields: u32,
    pub(crate) cdef_damping_minus_3: u8,
    pub(crate) cdef_bits: u8,
    pub(crate) cdef_y_strengths: [u8; 8],
    pub(crate) cdef_uv_strengths: [u8; 8],
    pub(crate) loop_restoration_fields: u16,
    pub(crate) wm: [VAWarpedMotionParamsAV1; 7],
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM],
}

impl VADecPictureParameterBufferAV1 {
    /// Every field zero, the anchor frames' pointer null.
    pub(crate) fn zeroed() -> Self {
        Self {
            profile: 0,
            order_hint_bits_minus_1: 0,
            bit_depth_idx: 0,
            matrix_coefficients: 0,
            seq_info_fields: 0,
            current_frame: 0,
            current_display_picture: 0,
            anchor_frames_num: 0,
            anchor_frames_list: core::ptr::null_mut(),
            frame_width_minus1: 0,
            frame_height_minus1: 0,
            output_frame_width_in_tiles_minus_1: 0,
            output_frame_height_in_tiles_minus_1: 0,
            ref_frame_map: [0; 8],
            ref_frame_idx: [0; 7],
            primary_ref_frame: 0,
            order_hint: 0,
            seg_info: VASegmentationStructAV1::default(),
            film_grain_info: VAFilmGrainStructAV1::default(),
            tile_cols: 0,
            tile_rows: 0,
            width_in_sbs_minus_1: [0; 63],
            height_in_sbs_minus_1: [0; 63],
            tile_count_minus_1: 0,
            context_update_tile_id: 0,
            pic_info_fields: 0,
            superres_scale_denominator: 0,
            interp_filter: 0,
            filter_level: [0; 2],
            filter_level_u: 0,
            filter_level_v: 0,
            loop_filter_info_fields: 0,
            ref_deltas: [0; 8],
            mode_deltas: [0; 2],
            base_qindex: 0,
            y_dc_delta_q: 0,
            u_dc_delta_q: 0,
            u_ac_delta_q: 0,
            v_dc_delta_q: 0,
            v_ac_delta_q: 0,
            qmatrix_fields: 0,
            mode_control_fields: 0,
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            cdef_y_strengths: [0; 8],
            cdef_uv_strengths: [0; 8],
            loop_restoration_fields: 0,
            wm: [VAWarpedMotionParamsAV1::default(); 7],
            va_reserved: [0; VA_PADDING_MEDIUM],
        }
    }
}

/// `VASliceParameterBufferAV1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VASliceParameterBufferAV1 {
    pub(crate) slice_data_size: u32,
    pub(crate) slice_data_offset: u32,
    pub(crate) slice_data_flag: u32,
    pub(crate) tile_row: u16,
    pub(crate) tile_column: u16,
    pub(crate) tg_start: u16,
    pub(crate) tg_end: u16,
    pub(crate) anchor_frame_idx: u8,
    pub(crate) tile_idx_in_tile_list: u16,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VADecPictureParameterBufferAV1::seq_info_fields.fields`.
pub(crate) mod seq_info_fields {
    use super::BitField;
    pub(crate) const STILL_PICTURE: BitField = BitField::new(0, 1);
    pub(crate) const USE_128X128_SUPERBLOCK: BitField = BitField::new(1, 1);
    pub(crate) const ENABLE_FILTER_INTRA: BitField = BitField::new(2, 1);
    pub(crate) const ENABLE_INTRA_EDGE_FILTER: BitField = BitField::new(3, 1);
    pub(crate) const ENABLE_INTERINTRA_COMPOUND: BitField = BitField::new(4, 1);
    pub(crate) const ENABLE_MASKED_COMPOUND: BitField = BitField::new(5, 1);
    pub(crate) const ENABLE_DUAL_FILTER: BitField = BitField::new(6, 1);
    pub(crate) const ENABLE_ORDER_HINT: BitField = BitField::new(7, 1);
    pub(crate) const ENABLE_JNT_COMP: BitField = BitField::new(8, 1);
    pub(crate) const ENABLE_CDEF: BitField = BitField::new(9, 1);
    pub(crate) const MONO_CHROME: BitField = BitField::new(10, 1);
    pub(crate) const COLOR_RANGE: BitField = BitField::new(11, 1);
    pub(crate) const SUBSAMPLING_X: BitField = BitField::new(12, 1);
    pub(crate) const SUBSAMPLING_Y: BitField = BitField::new(13, 1);
    pub(crate) const CHROMA_SAMPLE_POSITION: BitField = BitField::new(14, 1);
    pub(crate) const FILM_GRAIN_PARAMS_PRESENT: BitField = BitField::new(15, 1);
}

/// `VADecPictureParameterBufferAV1::pic_info_fields.bits`.
pub(crate) mod pic_info_fields {
    use super::BitField;
    pub(crate) const FRAME_TYPE: BitField = BitField::new(0, 2);
    pub(crate) const SHOW_FRAME: BitField = BitField::new(2, 1);
    pub(crate) const SHOWABLE_FRAME: BitField = BitField::new(3, 1);
    pub(crate) const ERROR_RESILIENT_MODE: BitField = BitField::new(4, 1);
    pub(crate) const DISABLE_CDF_UPDATE: BitField = BitField::new(5, 1);
    pub(crate) const ALLOW_SCREEN_CONTENT_TOOLS: BitField = BitField::new(6, 1);
    pub(crate) const FORCE_INTEGER_MV: BitField = BitField::new(7, 1);
    pub(crate) const ALLOW_INTRABC: BitField = BitField::new(8, 1);
    pub(crate) const USE_SUPERRES: BitField = BitField::new(9, 1);
    pub(crate) const ALLOW_HIGH_PRECISION_MV: BitField = BitField::new(10, 1);
    pub(crate) const IS_MOTION_MODE_SWITCHABLE: BitField = BitField::new(11, 1);
    pub(crate) const USE_REF_FRAME_MVS: BitField = BitField::new(12, 1);
    pub(crate) const DISABLE_FRAME_END_UPDATE_CDF: BitField = BitField::new(13, 1);
    pub(crate) const UNIFORM_TILE_SPACING_FLAG: BitField = BitField::new(14, 1);
    pub(crate) const ALLOW_WARPED_MOTION: BitField = BitField::new(15, 1);
    /// Left 0, as large-scale tile decoding is no stream's here; the layout test checks it.
    #[cfg_attr(not(test), expect(dead_code, reason = "only the layout test reads it"))]
    pub(crate) const LARGE_SCALE_TILE: BitField = BitField::new(16, 1);
}

/// `VASegmentationStructAV1::segment_info_fields.bits`.
pub(crate) mod segment_info_fields {
    use super::BitField;
    pub(crate) const ENABLED: BitField = BitField::new(0, 1);
    pub(crate) const UPDATE_MAP: BitField = BitField::new(1, 1);
    pub(crate) const TEMPORAL_UPDATE: BitField = BitField::new(2, 1);
    pub(crate) const UPDATE_DATA: BitField = BitField::new(3, 1);
}

/// `VAFilmGrainStructAV1::film_grain_info_fields.bits`.
pub(crate) mod film_grain_info_fields {
    use super::BitField;
    pub(crate) const APPLY_GRAIN: BitField = BitField::new(0, 1);
    pub(crate) const CHROMA_SCALING_FROM_LUMA: BitField = BitField::new(1, 1);
    pub(crate) const GRAIN_SCALING_MINUS_8: BitField = BitField::new(2, 2);
    pub(crate) const AR_COEFF_LAG: BitField = BitField::new(4, 2);
    pub(crate) const AR_COEFF_SHIFT_MINUS_6: BitField = BitField::new(6, 2);
    pub(crate) const GRAIN_SCALE_SHIFT: BitField = BitField::new(8, 2);
    pub(crate) const OVERLAP_FLAG: BitField = BitField::new(10, 1);
    pub(crate) const CLIP_TO_RESTRICTED_RANGE: BitField = BitField::new(11, 1);
}

/// `VADecPictureParameterBufferAV1::loop_filter_info_fields.bits`, a `uint8_t`.
pub(crate) mod loop_filter_info_fields {
    use super::BitField;
    pub(crate) const SHARPNESS_LEVEL: BitField = BitField::new(0, 3);
    pub(crate) const MODE_REF_DELTA_ENABLED: BitField = BitField::new(3, 1);
    pub(crate) const MODE_REF_DELTA_UPDATE: BitField = BitField::new(4, 1);
}

/// `VADecPictureParameterBufferAV1::qmatrix_fields.bits`, a `uint16_t`.
pub(crate) mod qmatrix_fields {
    use super::BitField;
    pub(crate) const USING_QMATRIX: BitField = BitField::new(0, 1);
    pub(crate) const QM_Y: BitField = BitField::new(1, 4);
    pub(crate) const QM_U: BitField = BitField::new(5, 4);
    pub(crate) const QM_V: BitField = BitField::new(9, 4);
}

/// `VADecPictureParameterBufferAV1::mode_control_fields.bits`.
pub(crate) mod mode_control_fields {
    use super::BitField;
    pub(crate) const DELTA_Q_PRESENT_FLAG: BitField = BitField::new(0, 1);
    pub(crate) const LOG2_DELTA_Q_RES: BitField = BitField::new(1, 2);
    pub(crate) const DELTA_LF_PRESENT_FLAG: BitField = BitField::new(3, 1);
    pub(crate) const LOG2_DELTA_LF_RES: BitField = BitField::new(4, 2);
    pub(crate) const DELTA_LF_MULTI: BitField = BitField::new(6, 1);
    pub(crate) const TX_MODE: BitField = BitField::new(7, 2);
    pub(crate) const REFERENCE_SELECT: BitField = BitField::new(9, 1);
    pub(crate) const REDUCED_TX_SET_USED: BitField = BitField::new(10, 1);
    pub(crate) const SKIP_MODE_PRESENT: BitField = BitField::new(11, 1);
}

/// `VADecPictureParameterBufferAV1::loop_restoration_fields.bits`, a `uint16_t`.
pub(crate) mod loop_restoration_fields {
    use super::BitField;
    pub(crate) const YFRAME_RESTORATION_TYPE: BitField = BitField::new(0, 2);
    pub(crate) const CBFRAME_RESTORATION_TYPE: BitField = BitField::new(2, 2);
    pub(crate) const CRFRAME_RESTORATION_TYPE: BitField = BitField::new(4, 2);
    pub(crate) const LR_UNIT_SHIFT: BitField = BitField::new(6, 2);
    pub(crate) const LR_UV_SHIFT: BitField = BitField::new(8, 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// What the C side prints, line for line.
    const LAYOUT_C: &str = include_str!("av1_layout.c");

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

    #[expect(
        clippy::too_many_lines,
        reason = "one list of every field, in the C file's order"
    )]
    fn rust_layout() -> Vec<String> {
        let mut out = Vec::new();
        layout!(out, VASegmentationStructAV1, "VASegmentationStructAV1", {
            segment_info_fields => "segment_info_fields", feature_data => "feature_data",
            feature_mask => "feature_mask", va_reserved => "va_reserved",
        });
        layout!(out, VAFilmGrainStructAV1, "VAFilmGrainStructAV1", {
            film_grain_info_fields => "film_grain_info_fields", grain_seed => "grain_seed",
            num_y_points => "num_y_points", point_y_value => "point_y_value",
            point_y_scaling => "point_y_scaling", num_cb_points => "num_cb_points",
            point_cb_value => "point_cb_value", point_cb_scaling => "point_cb_scaling",
            num_cr_points => "num_cr_points", point_cr_value => "point_cr_value",
            point_cr_scaling => "point_cr_scaling", ar_coeffs_y => "ar_coeffs_y",
            ar_coeffs_cb => "ar_coeffs_cb", ar_coeffs_cr => "ar_coeffs_cr", cb_mult => "cb_mult",
            cb_luma_mult => "cb_luma_mult", cb_offset => "cb_offset", cr_mult => "cr_mult",
            cr_luma_mult => "cr_luma_mult", cr_offset => "cr_offset",
            va_reserved => "va_reserved",
        });
        layout!(out, VAWarpedMotionParamsAV1, "VAWarpedMotionParamsAV1", {
            wmtype => "wmtype", wmmat => "wmmat", invalid => "invalid",
            va_reserved => "va_reserved",
        });
        layout!(out, VADecPictureParameterBufferAV1, "VADecPictureParameterBufferAV1", {
            profile => "profile", order_hint_bits_minus_1 => "order_hint_bits_minus_1",
            bit_depth_idx => "bit_depth_idx", matrix_coefficients => "matrix_coefficients",
            seq_info_fields => "seq_info_fields", current_frame => "current_frame",
            current_display_picture => "current_display_picture",
            anchor_frames_num => "anchor_frames_num", anchor_frames_list => "anchor_frames_list",
            frame_width_minus1 => "frame_width_minus1",
            frame_height_minus1 => "frame_height_minus1",
            output_frame_width_in_tiles_minus_1 => "output_frame_width_in_tiles_minus_1",
            output_frame_height_in_tiles_minus_1 => "output_frame_height_in_tiles_minus_1",
            ref_frame_map => "ref_frame_map", ref_frame_idx => "ref_frame_idx",
            primary_ref_frame => "primary_ref_frame", order_hint => "order_hint",
            seg_info => "seg_info", film_grain_info => "film_grain_info",
            tile_cols => "tile_cols", tile_rows => "tile_rows",
            width_in_sbs_minus_1 => "width_in_sbs_minus_1",
            height_in_sbs_minus_1 => "height_in_sbs_minus_1",
            tile_count_minus_1 => "tile_count_minus_1",
            context_update_tile_id => "context_update_tile_id",
            pic_info_fields => "pic_info_fields",
            superres_scale_denominator => "superres_scale_denominator",
            interp_filter => "interp_filter", filter_level => "filter_level",
            filter_level_u => "filter_level_u", filter_level_v => "filter_level_v",
            loop_filter_info_fields => "loop_filter_info_fields", ref_deltas => "ref_deltas",
            mode_deltas => "mode_deltas", base_qindex => "base_qindex",
            y_dc_delta_q => "y_dc_delta_q", u_dc_delta_q => "u_dc_delta_q",
            u_ac_delta_q => "u_ac_delta_q", v_dc_delta_q => "v_dc_delta_q",
            v_ac_delta_q => "v_ac_delta_q", qmatrix_fields => "qmatrix_fields",
            mode_control_fields => "mode_control_fields",
            cdef_damping_minus_3 => "cdef_damping_minus_3", cdef_bits => "cdef_bits",
            cdef_y_strengths => "cdef_y_strengths", cdef_uv_strengths => "cdef_uv_strengths",
            loop_restoration_fields => "loop_restoration_fields", wm => "wm",
            va_reserved => "va_reserved",
        });
        layout!(out, VASliceParameterBufferAV1, "VASliceParameterBufferAV1", {
            slice_data_size => "slice_data_size", slice_data_offset => "slice_data_offset",
            slice_data_flag => "slice_data_flag", tile_row => "tile_row",
            tile_column => "tile_column", tg_start => "tg_start", tg_end => "tg_end",
            anchor_frame_idx => "anchor_frame_idx",
            tile_idx_in_tile_list => "tile_idx_in_tile_list", va_reserved => "va_reserved",
        });
        {
            use seq_info_fields as f;
            fields(
                &mut out,
                "seq_info_fields",
                &[
                    ("still_picture", f::STILL_PICTURE),
                    ("use_128x128_superblock", f::USE_128X128_SUPERBLOCK),
                    ("enable_filter_intra", f::ENABLE_FILTER_INTRA),
                    ("enable_intra_edge_filter", f::ENABLE_INTRA_EDGE_FILTER),
                    ("enable_interintra_compound", f::ENABLE_INTERINTRA_COMPOUND),
                    ("enable_masked_compound", f::ENABLE_MASKED_COMPOUND),
                    ("enable_dual_filter", f::ENABLE_DUAL_FILTER),
                    ("enable_order_hint", f::ENABLE_ORDER_HINT),
                    ("enable_jnt_comp", f::ENABLE_JNT_COMP),
                    ("enable_cdef", f::ENABLE_CDEF),
                    ("mono_chrome", f::MONO_CHROME),
                    ("color_range", f::COLOR_RANGE),
                    ("subsampling_x", f::SUBSAMPLING_X),
                    ("subsampling_y", f::SUBSAMPLING_Y),
                    ("chroma_sample_position", f::CHROMA_SAMPLE_POSITION),
                    ("film_grain_params_present", f::FILM_GRAIN_PARAMS_PRESENT),
                ],
            );
        }
        {
            use pic_info_fields as f;
            fields(
                &mut out,
                "pic_info_fields",
                &[
                    ("frame_type", f::FRAME_TYPE),
                    ("show_frame", f::SHOW_FRAME),
                    ("showable_frame", f::SHOWABLE_FRAME),
                    ("error_resilient_mode", f::ERROR_RESILIENT_MODE),
                    ("disable_cdf_update", f::DISABLE_CDF_UPDATE),
                    ("allow_screen_content_tools", f::ALLOW_SCREEN_CONTENT_TOOLS),
                    ("force_integer_mv", f::FORCE_INTEGER_MV),
                    ("allow_intrabc", f::ALLOW_INTRABC),
                    ("use_superres", f::USE_SUPERRES),
                    ("allow_high_precision_mv", f::ALLOW_HIGH_PRECISION_MV),
                    ("is_motion_mode_switchable", f::IS_MOTION_MODE_SWITCHABLE),
                    ("use_ref_frame_mvs", f::USE_REF_FRAME_MVS),
                    (
                        "disable_frame_end_update_cdf",
                        f::DISABLE_FRAME_END_UPDATE_CDF,
                    ),
                    ("uniform_tile_spacing_flag", f::UNIFORM_TILE_SPACING_FLAG),
                    ("allow_warped_motion", f::ALLOW_WARPED_MOTION),
                    ("large_scale_tile", f::LARGE_SCALE_TILE),
                ],
            );
        }
        {
            use segment_info_fields as f;
            fields(
                &mut out,
                "segment_info_fields",
                &[
                    ("enabled", f::ENABLED),
                    ("update_map", f::UPDATE_MAP),
                    ("temporal_update", f::TEMPORAL_UPDATE),
                    ("update_data", f::UPDATE_DATA),
                ],
            );
        }
        {
            use film_grain_info_fields as f;
            fields(
                &mut out,
                "film_grain_info_fields",
                &[
                    ("apply_grain", f::APPLY_GRAIN),
                    ("chroma_scaling_from_luma", f::CHROMA_SCALING_FROM_LUMA),
                    ("grain_scaling_minus_8", f::GRAIN_SCALING_MINUS_8),
                    ("ar_coeff_lag", f::AR_COEFF_LAG),
                    ("ar_coeff_shift_minus_6", f::AR_COEFF_SHIFT_MINUS_6),
                    ("grain_scale_shift", f::GRAIN_SCALE_SHIFT),
                    ("overlap_flag", f::OVERLAP_FLAG),
                    ("clip_to_restricted_range", f::CLIP_TO_RESTRICTED_RANGE),
                ],
            );
        }
        {
            use loop_filter_info_fields as f;
            fields(
                &mut out,
                "loop_filter_info_fields",
                &[
                    ("sharpness_level", f::SHARPNESS_LEVEL),
                    ("mode_ref_delta_enabled", f::MODE_REF_DELTA_ENABLED),
                    ("mode_ref_delta_update", f::MODE_REF_DELTA_UPDATE),
                ],
            );
        }
        {
            use qmatrix_fields as f;
            fields(
                &mut out,
                "qmatrix_fields",
                &[
                    ("using_qmatrix", f::USING_QMATRIX),
                    ("qm_y", f::QM_Y),
                    ("qm_u", f::QM_U),
                    ("qm_v", f::QM_V),
                ],
            );
        }
        {
            use mode_control_fields as f;
            fields(
                &mut out,
                "mode_control_fields",
                &[
                    ("delta_q_present_flag", f::DELTA_Q_PRESENT_FLAG),
                    ("log2_delta_q_res", f::LOG2_DELTA_Q_RES),
                    ("delta_lf_present_flag", f::DELTA_LF_PRESENT_FLAG),
                    ("log2_delta_lf_res", f::LOG2_DELTA_LF_RES),
                    ("delta_lf_multi", f::DELTA_LF_MULTI),
                    ("tx_mode", f::TX_MODE),
                    ("reference_select", f::REFERENCE_SELECT),
                    ("reduced_tx_set_used", f::REDUCED_TX_SET_USED),
                    ("skip_mode_present", f::SKIP_MODE_PRESENT),
                ],
            );
        }
        {
            use loop_restoration_fields as f;
            fields(
                &mut out,
                "loop_restoration_fields",
                &[
                    ("yframe_restoration_type", f::YFRAME_RESTORATION_TYPE),
                    ("cbframe_restoration_type", f::CBFRAME_RESTORATION_TYPE),
                    ("crframe_restoration_type", f::CRFRAME_RESTORATION_TYPE),
                    ("lr_unit_shift", f::LR_UNIT_SHIFT),
                    ("lr_uv_shift", f::LR_UV_SHIFT),
                ],
            );
        }
        out.push(format!("VAProfileAV1Profile0 {VAProfileAV1Profile0}"));
        out
    }

    /// Compiles `av1_layout.c` against the installed libva headers and compares its output
    /// with this module's layout. `CC` picks the compiler (`cc` by default).
    #[test]
    #[ignore = "needs libva's headers and a C compiler: run on the test host"]
    fn av1_layouts_match_libva_headers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("rvgv-va-av1-layout-{}", std::process::id()));
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
