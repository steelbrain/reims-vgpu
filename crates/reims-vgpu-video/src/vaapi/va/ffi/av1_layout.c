/*
 * The layout of the libva AV1 decode structs ffi/av1.rs declares, from the installed headers:
 * sizeof and offsetof of each, then the mask of each bit-field, then the value of each
 * constant. av1.rs's av1_layouts_match_libva_headers test compiles this and compares its
 * output with Rust's, line for line.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <va/va.h>
#include <va/va_dec_av1.h>

#define S(T) printf(#T " size %zu\n", sizeof(T))
#define F(T, f) printf(#T "." #f " %zu\n", offsetof(T, f))
#define C(c) printf(#c " %lld\n", (long long)(c))
#define BITS(T, member, sub, label, f)                                 \
    do {                                                               \
        T p;                                                           \
        memset(&p, 0, sizeof p);                                       \
        p.member.sub.f = ~0u;                                          \
        printf(label "." #f " %#x\n", (unsigned)p.member.value);       \
    } while (0)

int main(void)
{
    S(VASegmentationStructAV1);
    F(VASegmentationStructAV1, segment_info_fields);
    F(VASegmentationStructAV1, feature_data);
    F(VASegmentationStructAV1, feature_mask);
    F(VASegmentationStructAV1, va_reserved);

    S(VAFilmGrainStructAV1);
    F(VAFilmGrainStructAV1, film_grain_info_fields);
    F(VAFilmGrainStructAV1, grain_seed);
    F(VAFilmGrainStructAV1, num_y_points);
    F(VAFilmGrainStructAV1, point_y_value);
    F(VAFilmGrainStructAV1, point_y_scaling);
    F(VAFilmGrainStructAV1, num_cb_points);
    F(VAFilmGrainStructAV1, point_cb_value);
    F(VAFilmGrainStructAV1, point_cb_scaling);
    F(VAFilmGrainStructAV1, num_cr_points);
    F(VAFilmGrainStructAV1, point_cr_value);
    F(VAFilmGrainStructAV1, point_cr_scaling);
    F(VAFilmGrainStructAV1, ar_coeffs_y);
    F(VAFilmGrainStructAV1, ar_coeffs_cb);
    F(VAFilmGrainStructAV1, ar_coeffs_cr);
    F(VAFilmGrainStructAV1, cb_mult);
    F(VAFilmGrainStructAV1, cb_luma_mult);
    F(VAFilmGrainStructAV1, cb_offset);
    F(VAFilmGrainStructAV1, cr_mult);
    F(VAFilmGrainStructAV1, cr_luma_mult);
    F(VAFilmGrainStructAV1, cr_offset);
    F(VAFilmGrainStructAV1, va_reserved);

    S(VAWarpedMotionParamsAV1);
    F(VAWarpedMotionParamsAV1, wmtype);
    F(VAWarpedMotionParamsAV1, wmmat);
    F(VAWarpedMotionParamsAV1, invalid);
    F(VAWarpedMotionParamsAV1, va_reserved);

    S(VADecPictureParameterBufferAV1);
    F(VADecPictureParameterBufferAV1, profile);
    F(VADecPictureParameterBufferAV1, order_hint_bits_minus_1);
    F(VADecPictureParameterBufferAV1, bit_depth_idx);
    F(VADecPictureParameterBufferAV1, matrix_coefficients);
    F(VADecPictureParameterBufferAV1, seq_info_fields);
    F(VADecPictureParameterBufferAV1, current_frame);
    F(VADecPictureParameterBufferAV1, current_display_picture);
    F(VADecPictureParameterBufferAV1, anchor_frames_num);
    F(VADecPictureParameterBufferAV1, anchor_frames_list);
    F(VADecPictureParameterBufferAV1, frame_width_minus1);
    F(VADecPictureParameterBufferAV1, frame_height_minus1);
    F(VADecPictureParameterBufferAV1, output_frame_width_in_tiles_minus_1);
    F(VADecPictureParameterBufferAV1, output_frame_height_in_tiles_minus_1);
    F(VADecPictureParameterBufferAV1, ref_frame_map);
    F(VADecPictureParameterBufferAV1, ref_frame_idx);
    F(VADecPictureParameterBufferAV1, primary_ref_frame);
    F(VADecPictureParameterBufferAV1, order_hint);
    F(VADecPictureParameterBufferAV1, seg_info);
    F(VADecPictureParameterBufferAV1, film_grain_info);
    F(VADecPictureParameterBufferAV1, tile_cols);
    F(VADecPictureParameterBufferAV1, tile_rows);
    F(VADecPictureParameterBufferAV1, width_in_sbs_minus_1);
    F(VADecPictureParameterBufferAV1, height_in_sbs_minus_1);
    F(VADecPictureParameterBufferAV1, tile_count_minus_1);
    F(VADecPictureParameterBufferAV1, context_update_tile_id);
    F(VADecPictureParameterBufferAV1, pic_info_fields);
    F(VADecPictureParameterBufferAV1, superres_scale_denominator);
    F(VADecPictureParameterBufferAV1, interp_filter);
    F(VADecPictureParameterBufferAV1, filter_level);
    F(VADecPictureParameterBufferAV1, filter_level_u);
    F(VADecPictureParameterBufferAV1, filter_level_v);
    F(VADecPictureParameterBufferAV1, loop_filter_info_fields);
    F(VADecPictureParameterBufferAV1, ref_deltas);
    F(VADecPictureParameterBufferAV1, mode_deltas);
    F(VADecPictureParameterBufferAV1, base_qindex);
    F(VADecPictureParameterBufferAV1, y_dc_delta_q);
    F(VADecPictureParameterBufferAV1, u_dc_delta_q);
    F(VADecPictureParameterBufferAV1, u_ac_delta_q);
    F(VADecPictureParameterBufferAV1, v_dc_delta_q);
    F(VADecPictureParameterBufferAV1, v_ac_delta_q);
    F(VADecPictureParameterBufferAV1, qmatrix_fields);
    F(VADecPictureParameterBufferAV1, mode_control_fields);
    F(VADecPictureParameterBufferAV1, cdef_damping_minus_3);
    F(VADecPictureParameterBufferAV1, cdef_bits);
    F(VADecPictureParameterBufferAV1, cdef_y_strengths);
    F(VADecPictureParameterBufferAV1, cdef_uv_strengths);
    F(VADecPictureParameterBufferAV1, loop_restoration_fields);
    F(VADecPictureParameterBufferAV1, wm);
    F(VADecPictureParameterBufferAV1, va_reserved);

    S(VASliceParameterBufferAV1);
    F(VASliceParameterBufferAV1, slice_data_size);
    F(VASliceParameterBufferAV1, slice_data_offset);
    F(VASliceParameterBufferAV1, slice_data_flag);
    F(VASliceParameterBufferAV1, tile_row);
    F(VASliceParameterBufferAV1, tile_column);
    F(VASliceParameterBufferAV1, tg_start);
    F(VASliceParameterBufferAV1, tg_end);
    F(VASliceParameterBufferAV1, anchor_frame_idx);
    F(VASliceParameterBufferAV1, tile_idx_in_tile_list);
    F(VASliceParameterBufferAV1, va_reserved);

    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", still_picture);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", use_128x128_superblock);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_filter_intra);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_intra_edge_filter);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_interintra_compound);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_masked_compound);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_dual_filter);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_order_hint);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_jnt_comp);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", enable_cdef);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", mono_chrome);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", color_range);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", subsampling_x);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", subsampling_y);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", chroma_sample_position);
    BITS(VADecPictureParameterBufferAV1, seq_info_fields, fields, "seq_info_fields", film_grain_params_present);

    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", frame_type);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", show_frame);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", showable_frame);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", error_resilient_mode);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", disable_cdf_update);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", allow_screen_content_tools);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", force_integer_mv);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", allow_intrabc);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", use_superres);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", allow_high_precision_mv);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", is_motion_mode_switchable);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", use_ref_frame_mvs);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", disable_frame_end_update_cdf);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", uniform_tile_spacing_flag);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", allow_warped_motion);
    BITS(VADecPictureParameterBufferAV1, pic_info_fields, bits, "pic_info_fields", large_scale_tile);

    BITS(VASegmentationStructAV1, segment_info_fields, bits, "segment_info_fields", enabled);
    BITS(VASegmentationStructAV1, segment_info_fields, bits, "segment_info_fields", update_map);
    BITS(VASegmentationStructAV1, segment_info_fields, bits, "segment_info_fields", temporal_update);
    BITS(VASegmentationStructAV1, segment_info_fields, bits, "segment_info_fields", update_data);

    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", apply_grain);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", chroma_scaling_from_luma);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", grain_scaling_minus_8);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", ar_coeff_lag);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", ar_coeff_shift_minus_6);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", grain_scale_shift);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", overlap_flag);
    BITS(VAFilmGrainStructAV1, film_grain_info_fields, bits, "film_grain_info_fields", clip_to_restricted_range);

    BITS(VADecPictureParameterBufferAV1, loop_filter_info_fields, bits, "loop_filter_info_fields", sharpness_level);
    BITS(VADecPictureParameterBufferAV1, loop_filter_info_fields, bits, "loop_filter_info_fields", mode_ref_delta_enabled);
    BITS(VADecPictureParameterBufferAV1, loop_filter_info_fields, bits, "loop_filter_info_fields", mode_ref_delta_update);

    BITS(VADecPictureParameterBufferAV1, qmatrix_fields, bits, "qmatrix_fields", using_qmatrix);
    BITS(VADecPictureParameterBufferAV1, qmatrix_fields, bits, "qmatrix_fields", qm_y);
    BITS(VADecPictureParameterBufferAV1, qmatrix_fields, bits, "qmatrix_fields", qm_u);
    BITS(VADecPictureParameterBufferAV1, qmatrix_fields, bits, "qmatrix_fields", qm_v);

    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", delta_q_present_flag);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", log2_delta_q_res);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", delta_lf_present_flag);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", log2_delta_lf_res);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", delta_lf_multi);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", tx_mode);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", reference_select);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", reduced_tx_set_used);
    BITS(VADecPictureParameterBufferAV1, mode_control_fields, bits, "mode_control_fields", skip_mode_present);

    BITS(VADecPictureParameterBufferAV1, loop_restoration_fields, bits, "loop_restoration_fields", yframe_restoration_type);
    BITS(VADecPictureParameterBufferAV1, loop_restoration_fields, bits, "loop_restoration_fields", cbframe_restoration_type);
    BITS(VADecPictureParameterBufferAV1, loop_restoration_fields, bits, "loop_restoration_fields", crframe_restoration_type);
    BITS(VADecPictureParameterBufferAV1, loop_restoration_fields, bits, "loop_restoration_fields", lr_unit_shift);
    BITS(VADecPictureParameterBufferAV1, loop_restoration_fields, bits, "loop_restoration_fields", lr_uv_shift);

    C(VAProfileAV1Profile0);
    return 0;
}
