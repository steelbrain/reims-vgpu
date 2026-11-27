//! `CUVIDAV1PICPARAMS` (ffnvcodec's `dynlink_cuviddec.h`, NVDECAPI 13.1): one AV1 frame's
//! parameters, `CUVIDPICPARAMS::CodecSpecific.av1`. The header's bit-fields are held in
//! words named for what they hold, each marked as ours, with their fields' positions in
//! [`bits`]; `crate::nvidia::layout`'s test checks them against the header.

use std::ffi::{c_char, c_int, c_short, c_uint, c_ushort};

use super::Zeroed;

/// `CUVIDAV1PICPARAMS::ref_frame[]`. Name is ours: an anonymous struct in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidAv1RefFrame {
    pub(crate) width: c_uint,
    pub(crate) height: c_uint,
    pub(crate) index: u8,
    pub(crate) reserved24Bits: [u8; 3],
}

/// `CUVIDAV1PICPARAMS::global_motion[]`. Name is ours: an anonymous struct in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidAv1GlobalMotion {
    /// `invalid`, `wmtype` and `reserved5Bits`. Name is ours.
    pub(crate) motion_word: u8,
    pub(crate) reserved24Bits: [c_char; 3],
    pub(crate) wmmat: [c_int; 6],
}

/// `CUVIDAV1PICPARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDAV1PICPARAMS {
    pub(crate) width: c_uint,
    pub(crate) height: c_uint,
    pub(crate) frame_offset: c_uint,
    pub(crate) decodePicIdx: c_int,
    /// The sequence header's bit-fields, `profile` to `reserved0_7bits`. Name is ours.
    pub(crate) sequence_word: c_uint,
    /// The frame header's bit-fields, `frame_type` to `reserved1_3bits`. Name is ours.
    pub(crate) frame_word: c_uint,
    /// `num_tile_cols`, `num_tile_rows` and `context_update_tile_id`. Name is ours.
    pub(crate) tiling_word: c_uint,
    pub(crate) tile_widths: [c_ushort; 64],
    pub(crate) tile_heights: [c_ushort; 64],
    /// `cdef_damping_minus_3`, `cdef_bits` and `reserved2_4bits`. Name is ours.
    pub(crate) cdef_word: u8,
    pub(crate) cdef_y_strength: [u8; 8],
    pub(crate) cdef_uv_strength: [u8; 8],
    /// `SkipModeFrame0` and `SkipModeFrame1`. Name is ours.
    pub(crate) skip_mode_word: u8,
    pub(crate) base_qindex: u8,
    pub(crate) qp_y_dc_delta_q: c_char,
    pub(crate) qp_u_dc_delta_q: c_char,
    pub(crate) qp_v_dc_delta_q: c_char,
    pub(crate) qp_u_ac_delta_q: c_char,
    pub(crate) qp_v_ac_delta_q: c_char,
    pub(crate) qm_y: u8,
    pub(crate) qm_u: u8,
    pub(crate) qm_v: u8,
    /// `segmentation_enabled` to `reserved3_4bits`. Name is ours.
    pub(crate) segmentation_word: u8,
    pub(crate) segmentation_feature_data: [[c_short; 8]; 8],
    pub(crate) segmentation_feature_mask: [u8; 8],
    pub(crate) loop_filter_level: [u8; 2],
    pub(crate) loop_filter_level_u: u8,
    pub(crate) loop_filter_level_v: u8,
    pub(crate) loop_filter_sharpness: u8,
    pub(crate) loop_filter_ref_deltas: [c_char; 8],
    pub(crate) loop_filter_mode_deltas: [c_char; 2],
    /// `loop_filter_delta_enabled` to `reserved4_2bits`. Name is ours.
    pub(crate) loop_filter_word: u8,
    pub(crate) lr_unit_size: [u8; 3],
    pub(crate) lr_type: [u8; 3],
    pub(crate) primary_ref_frame: u8,
    pub(crate) ref_frame_map: [u8; 8],
    /// `temporal_layer_id` and `spatial_layer_id`. Name is ours.
    pub(crate) layer_word: u8,
    pub(crate) reserved5_32bits: [u8; 4],
    pub(crate) ref_frame: [CuvidAv1RefFrame; 7],
    pub(crate) global_motion: [CuvidAv1GlobalMotion; 7],
    /// `apply_grain` to `reserved6_4bits`. Name is ours.
    pub(crate) film_grain_word: c_ushort,
    pub(crate) num_y_points: u8,
    pub(crate) scaling_points_y: [[u8; 2]; 14],
    pub(crate) num_cb_points: u8,
    pub(crate) scaling_points_cb: [[u8; 2]; 10],
    pub(crate) num_cr_points: u8,
    pub(crate) scaling_points_cr: [[u8; 2]; 10],
    pub(crate) reserved7_8bits: u8,
    pub(crate) random_seed: c_ushort,
    pub(crate) ar_coeffs_y: [c_short; 24],
    pub(crate) ar_coeffs_cb: [c_short; 25],
    pub(crate) ar_coeffs_cr: [c_short; 25],
    pub(crate) cb_mult: u8,
    pub(crate) cb_luma_mult: u8,
    pub(crate) cb_offset: c_short,
    pub(crate) cr_mult: u8,
    pub(crate) cr_luma_mult: u8,
    pub(crate) cr_offset: c_short,
    pub(crate) reserved: [c_int; 7],
}

// SAFETY: integers and arrays of them, and structs of those: all-zero bytes are a value.
unsafe impl Zeroed for CUVIDAV1PICPARAMS {}

/// The bit-fields' positions in their words.
pub(crate) mod bits {
    use super::super::bits::Bits;

    const fn b(shift: u32, width: u32) -> Bits {
        Bits { shift, width }
    }

    /// `CUVIDAV1PICPARAMS::sequence_word`.
    pub(crate) mod sequence {
        use super::{Bits, b};
        pub(crate) const profile: Bits = b(0, 3);
        pub(crate) const use_128x128_superblock: Bits = b(3, 1);
        pub(crate) const subsampling_x: Bits = b(4, 1);
        pub(crate) const subsampling_y: Bits = b(5, 1);
        pub(crate) const mono_chrome: Bits = b(6, 1);
        pub(crate) const bit_depth_minus8: Bits = b(7, 4);
        pub(crate) const enable_filter_intra: Bits = b(11, 1);
        pub(crate) const enable_intra_edge_filter: Bits = b(12, 1);
        pub(crate) const enable_interintra_compound: Bits = b(13, 1);
        pub(crate) const enable_masked_compound: Bits = b(14, 1);
        pub(crate) const enable_dual_filter: Bits = b(15, 1);
        pub(crate) const enable_order_hint: Bits = b(16, 1);
        pub(crate) const order_hint_bits_minus1: Bits = b(17, 3);
        pub(crate) const enable_jnt_comp: Bits = b(20, 1);
        pub(crate) const enable_superres: Bits = b(21, 1);
        pub(crate) const enable_cdef: Bits = b(22, 1);
        pub(crate) const enable_restoration: Bits = b(23, 1);
        pub(crate) const enable_fgs: Bits = b(24, 1);
    }

    /// `CUVIDAV1PICPARAMS::frame_word`.
    pub(crate) mod frame {
        use super::{Bits, b};
        pub(crate) const frame_type: Bits = b(0, 2);
        pub(crate) const show_frame: Bits = b(2, 1);
        pub(crate) const disable_cdf_update: Bits = b(3, 1);
        pub(crate) const allow_screen_content_tools: Bits = b(4, 1);
        pub(crate) const force_integer_mv: Bits = b(5, 1);
        pub(crate) const coded_denom: Bits = b(6, 3);
        pub(crate) const allow_intrabc: Bits = b(9, 1);
        pub(crate) const allow_high_precision_mv: Bits = b(10, 1);
        pub(crate) const interp_filter: Bits = b(11, 3);
        pub(crate) const switchable_motion_mode: Bits = b(14, 1);
        pub(crate) const use_ref_frame_mvs: Bits = b(15, 1);
        pub(crate) const disable_frame_end_update_cdf: Bits = b(16, 1);
        pub(crate) const delta_q_present: Bits = b(17, 1);
        pub(crate) const delta_q_res: Bits = b(18, 2);
        pub(crate) const using_qmatrix: Bits = b(20, 1);
        pub(crate) const coded_lossless: Bits = b(21, 1);
        pub(crate) const use_superres: Bits = b(22, 1);
        pub(crate) const tx_mode: Bits = b(23, 2);
        pub(crate) const reference_mode: Bits = b(25, 1);
        pub(crate) const allow_warped_motion: Bits = b(26, 1);
        pub(crate) const reduced_tx_set: Bits = b(27, 1);
        pub(crate) const skip_mode: Bits = b(28, 1);
    }

    /// `CUVIDAV1PICPARAMS::tiling_word`.
    pub(crate) mod tiling {
        use super::{Bits, b};
        pub(crate) const num_tile_cols: Bits = b(0, 8);
        pub(crate) const num_tile_rows: Bits = b(8, 8);
        pub(crate) const context_update_tile_id: Bits = b(16, 16);
    }

    /// `CUVIDAV1PICPARAMS::cdef_word`.
    pub(crate) mod cdef {
        use super::{Bits, b};
        pub(crate) const cdef_damping_minus_3: Bits = b(0, 2);
        pub(crate) const cdef_bits: Bits = b(2, 2);
    }

    /// `CUVIDAV1PICPARAMS::skip_mode_word`.
    pub(crate) mod skip_mode {
        use super::{Bits, b};
        pub(crate) const SkipModeFrame0: Bits = b(0, 4);
        pub(crate) const SkipModeFrame1: Bits = b(4, 4);
    }

    /// `CUVIDAV1PICPARAMS::segmentation_word`.
    pub(crate) mod segmentation {
        use super::{Bits, b};
        pub(crate) const segmentation_enabled: Bits = b(0, 1);
        pub(crate) const segmentation_update_map: Bits = b(1, 1);
        pub(crate) const segmentation_update_data: Bits = b(2, 1);
        pub(crate) const segmentation_temporal_update: Bits = b(3, 1);
    }

    /// `CUVIDAV1PICPARAMS::loop_filter_word`.
    pub(crate) mod loop_filter {
        use super::{Bits, b};
        pub(crate) const loop_filter_delta_enabled: Bits = b(0, 1);
        pub(crate) const loop_filter_delta_update: Bits = b(1, 1);
        pub(crate) const delta_lf_present: Bits = b(2, 1);
        pub(crate) const delta_lf_res: Bits = b(3, 2);
        pub(crate) const delta_lf_multi: Bits = b(5, 1);
    }

    /// `CUVIDAV1PICPARAMS::layer_word`.
    pub(crate) mod layer {
        use super::{Bits, b};
        pub(crate) const temporal_layer_id: Bits = b(0, 4);
        pub(crate) const spatial_layer_id: Bits = b(4, 4);
    }

    /// `CuvidAv1GlobalMotion::motion_word`.
    pub(crate) mod motion {
        use super::{Bits, b};
        pub(crate) const invalid: Bits = b(0, 1);
        pub(crate) const wmtype: Bits = b(1, 2);
    }

    /// `CUVIDAV1PICPARAMS::film_grain_word`.
    pub(crate) mod film_grain {
        use super::{Bits, b};
        pub(crate) const apply_grain: Bits = b(0, 1);
        pub(crate) const overlap_flag: Bits = b(1, 1);
        pub(crate) const scaling_shift_minus8: Bits = b(2, 2);
        pub(crate) const chroma_scaling_from_luma: Bits = b(4, 1);
        pub(crate) const ar_coeff_lag: Bits = b(5, 2);
        pub(crate) const ar_coeff_shift_minus6: Bits = b(7, 2);
        pub(crate) const grain_scale_shift: Bits = b(9, 2);
        pub(crate) const clip_to_restricted_range: Bits = b(11, 1);
    }
}
