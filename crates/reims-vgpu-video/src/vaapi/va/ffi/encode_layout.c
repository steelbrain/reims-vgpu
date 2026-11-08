/*
 * The layout of the libva encode structs ffi/encode.rs declares, from the
 * installed headers: sizeof and offsetof of each, then the mask of each
 * bit-field, then the value of each constant. encode.rs's encode_layouts_match_libva_headers test compiles
 * this and compares its output with Rust's, line for line.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <va/va.h>
#include <va/va_enc_h264.h>
#include <va/va_enc_hevc.h>

#define S(T) printf(#T " size %zu\n", sizeof(T))
#define F(T, f) printf(#T "." #f " %zu\n", offsetof(T, f))
#define C(c) printf(#c " %lld\n", (long long)(c))
#define U(c) printf(#c " %llu\n", (unsigned long long)(c))
#define BITS(T, member, label, f)                                      \
    do {                                                               \
        T p;                                                           \
        memset(&p, 0, sizeof p);                                       \
        p.member.bits.f = ~0u;                                         \
        printf(label "." #f " %#x\n", (unsigned)p.member.value);       \
    } while (0)

int main(void)
{
    S(VAEncSequenceParameterBufferH264);
    F(VAEncSequenceParameterBufferH264, seq_parameter_set_id);
    F(VAEncSequenceParameterBufferH264, level_idc);
    F(VAEncSequenceParameterBufferH264, intra_period);
    F(VAEncSequenceParameterBufferH264, intra_idr_period);
    F(VAEncSequenceParameterBufferH264, ip_period);
    F(VAEncSequenceParameterBufferH264, bits_per_second);
    F(VAEncSequenceParameterBufferH264, max_num_ref_frames);
    F(VAEncSequenceParameterBufferH264, picture_width_in_mbs);
    F(VAEncSequenceParameterBufferH264, picture_height_in_mbs);
    F(VAEncSequenceParameterBufferH264, seq_fields);
    F(VAEncSequenceParameterBufferH264, bit_depth_luma_minus8);
    F(VAEncSequenceParameterBufferH264, bit_depth_chroma_minus8);
    F(VAEncSequenceParameterBufferH264, num_ref_frames_in_pic_order_cnt_cycle);
    F(VAEncSequenceParameterBufferH264, offset_for_non_ref_pic);
    F(VAEncSequenceParameterBufferH264, offset_for_top_to_bottom_field);
    F(VAEncSequenceParameterBufferH264, offset_for_ref_frame);
    F(VAEncSequenceParameterBufferH264, frame_cropping_flag);
    F(VAEncSequenceParameterBufferH264, frame_crop_left_offset);
    F(VAEncSequenceParameterBufferH264, frame_crop_right_offset);
    F(VAEncSequenceParameterBufferH264, frame_crop_top_offset);
    F(VAEncSequenceParameterBufferH264, frame_crop_bottom_offset);
    F(VAEncSequenceParameterBufferH264, vui_parameters_present_flag);
    F(VAEncSequenceParameterBufferH264, vui_fields);
    F(VAEncSequenceParameterBufferH264, aspect_ratio_idc);
    F(VAEncSequenceParameterBufferH264, sar_width);
    F(VAEncSequenceParameterBufferH264, sar_height);
    F(VAEncSequenceParameterBufferH264, num_units_in_tick);
    F(VAEncSequenceParameterBufferH264, time_scale);
    F(VAEncSequenceParameterBufferH264, va_reserved);

    S(VAEncPictureParameterBufferH264);
    F(VAEncPictureParameterBufferH264, CurrPic);
    F(VAEncPictureParameterBufferH264, ReferenceFrames);
    F(VAEncPictureParameterBufferH264, coded_buf);
    F(VAEncPictureParameterBufferH264, pic_parameter_set_id);
    F(VAEncPictureParameterBufferH264, seq_parameter_set_id);
    F(VAEncPictureParameterBufferH264, last_picture);
    F(VAEncPictureParameterBufferH264, frame_num);
    F(VAEncPictureParameterBufferH264, pic_init_qp);
    F(VAEncPictureParameterBufferH264, num_ref_idx_l0_active_minus1);
    F(VAEncPictureParameterBufferH264, num_ref_idx_l1_active_minus1);
    F(VAEncPictureParameterBufferH264, chroma_qp_index_offset);
    F(VAEncPictureParameterBufferH264, second_chroma_qp_index_offset);
    F(VAEncPictureParameterBufferH264, pic_fields);
    F(VAEncPictureParameterBufferH264, va_reserved);

    S(VAEncSliceParameterBufferH264);
    F(VAEncSliceParameterBufferH264, macroblock_address);
    F(VAEncSliceParameterBufferH264, num_macroblocks);
    F(VAEncSliceParameterBufferH264, macroblock_info);
    F(VAEncSliceParameterBufferH264, slice_type);
    F(VAEncSliceParameterBufferH264, pic_parameter_set_id);
    F(VAEncSliceParameterBufferH264, idr_pic_id);
    F(VAEncSliceParameterBufferH264, pic_order_cnt_lsb);
    F(VAEncSliceParameterBufferH264, delta_pic_order_cnt_bottom);
    F(VAEncSliceParameterBufferH264, delta_pic_order_cnt);
    F(VAEncSliceParameterBufferH264, direct_spatial_mv_pred_flag);
    F(VAEncSliceParameterBufferH264, num_ref_idx_active_override_flag);
    F(VAEncSliceParameterBufferH264, num_ref_idx_l0_active_minus1);
    F(VAEncSliceParameterBufferH264, num_ref_idx_l1_active_minus1);
    F(VAEncSliceParameterBufferH264, RefPicList0);
    F(VAEncSliceParameterBufferH264, RefPicList1);
    F(VAEncSliceParameterBufferH264, luma_log2_weight_denom);
    F(VAEncSliceParameterBufferH264, chroma_log2_weight_denom);
    F(VAEncSliceParameterBufferH264, luma_weight_l0_flag);
    F(VAEncSliceParameterBufferH264, luma_weight_l0);
    F(VAEncSliceParameterBufferH264, luma_offset_l0);
    F(VAEncSliceParameterBufferH264, chroma_weight_l0_flag);
    F(VAEncSliceParameterBufferH264, chroma_weight_l0);
    F(VAEncSliceParameterBufferH264, chroma_offset_l0);
    F(VAEncSliceParameterBufferH264, luma_weight_l1_flag);
    F(VAEncSliceParameterBufferH264, luma_weight_l1);
    F(VAEncSliceParameterBufferH264, luma_offset_l1);
    F(VAEncSliceParameterBufferH264, chroma_weight_l1_flag);
    F(VAEncSliceParameterBufferH264, chroma_weight_l1);
    F(VAEncSliceParameterBufferH264, chroma_offset_l1);
    F(VAEncSliceParameterBufferH264, cabac_init_idc);
    F(VAEncSliceParameterBufferH264, slice_qp_delta);
    F(VAEncSliceParameterBufferH264, disable_deblocking_filter_idc);
    F(VAEncSliceParameterBufferH264, slice_alpha_c0_offset_div2);
    F(VAEncSliceParameterBufferH264, slice_beta_offset_div2);
    F(VAEncSliceParameterBufferH264, va_reserved);

    S(VAPictureHEVC);
    F(VAPictureHEVC, picture_id);
    F(VAPictureHEVC, pic_order_cnt);
    F(VAPictureHEVC, flags);
    F(VAPictureHEVC, va_reserved);

    S(VAEncSequenceParameterBufferHEVC);
    F(VAEncSequenceParameterBufferHEVC, general_profile_idc);
    F(VAEncSequenceParameterBufferHEVC, general_level_idc);
    F(VAEncSequenceParameterBufferHEVC, general_tier_flag);
    F(VAEncSequenceParameterBufferHEVC, intra_period);
    F(VAEncSequenceParameterBufferHEVC, intra_idr_period);
    F(VAEncSequenceParameterBufferHEVC, ip_period);
    F(VAEncSequenceParameterBufferHEVC, bits_per_second);
    F(VAEncSequenceParameterBufferHEVC, pic_width_in_luma_samples);
    F(VAEncSequenceParameterBufferHEVC, pic_height_in_luma_samples);
    F(VAEncSequenceParameterBufferHEVC, seq_fields);
    F(VAEncSequenceParameterBufferHEVC, log2_min_luma_coding_block_size_minus3);
    F(VAEncSequenceParameterBufferHEVC, log2_diff_max_min_luma_coding_block_size);
    F(VAEncSequenceParameterBufferHEVC, log2_min_transform_block_size_minus2);
    F(VAEncSequenceParameterBufferHEVC, log2_diff_max_min_transform_block_size);
    F(VAEncSequenceParameterBufferHEVC, max_transform_hierarchy_depth_inter);
    F(VAEncSequenceParameterBufferHEVC, max_transform_hierarchy_depth_intra);
    F(VAEncSequenceParameterBufferHEVC, pcm_sample_bit_depth_luma_minus1);
    F(VAEncSequenceParameterBufferHEVC, pcm_sample_bit_depth_chroma_minus1);
    F(VAEncSequenceParameterBufferHEVC, log2_min_pcm_luma_coding_block_size_minus3);
    F(VAEncSequenceParameterBufferHEVC, log2_max_pcm_luma_coding_block_size_minus3);
    F(VAEncSequenceParameterBufferHEVC, vui_parameters_present_flag);
    F(VAEncSequenceParameterBufferHEVC, vui_fields);
    F(VAEncSequenceParameterBufferHEVC, aspect_ratio_idc);
    F(VAEncSequenceParameterBufferHEVC, sar_width);
    F(VAEncSequenceParameterBufferHEVC, sar_height);
    F(VAEncSequenceParameterBufferHEVC, vui_num_units_in_tick);
    F(VAEncSequenceParameterBufferHEVC, vui_time_scale);
    F(VAEncSequenceParameterBufferHEVC, min_spatial_segmentation_idc);
    F(VAEncSequenceParameterBufferHEVC, max_bytes_per_pic_denom);
    F(VAEncSequenceParameterBufferHEVC, max_bits_per_min_cu_denom);
    F(VAEncSequenceParameterBufferHEVC, scc_fields);
    F(VAEncSequenceParameterBufferHEVC, va_reserved);

    S(VAEncPictureParameterBufferHEVC);
    F(VAEncPictureParameterBufferHEVC, decoded_curr_pic);
    F(VAEncPictureParameterBufferHEVC, reference_frames);
    F(VAEncPictureParameterBufferHEVC, coded_buf);
    F(VAEncPictureParameterBufferHEVC, collocated_ref_pic_index);
    F(VAEncPictureParameterBufferHEVC, last_picture);
    F(VAEncPictureParameterBufferHEVC, pic_init_qp);
    F(VAEncPictureParameterBufferHEVC, diff_cu_qp_delta_depth);
    F(VAEncPictureParameterBufferHEVC, pps_cb_qp_offset);
    F(VAEncPictureParameterBufferHEVC, pps_cr_qp_offset);
    F(VAEncPictureParameterBufferHEVC, num_tile_columns_minus1);
    F(VAEncPictureParameterBufferHEVC, num_tile_rows_minus1);
    F(VAEncPictureParameterBufferHEVC, column_width_minus1);
    F(VAEncPictureParameterBufferHEVC, row_height_minus1);
    F(VAEncPictureParameterBufferHEVC, log2_parallel_merge_level_minus2);
    F(VAEncPictureParameterBufferHEVC, ctu_max_bitsize_allowed);
    F(VAEncPictureParameterBufferHEVC, num_ref_idx_l0_default_active_minus1);
    F(VAEncPictureParameterBufferHEVC, num_ref_idx_l1_default_active_minus1);
    F(VAEncPictureParameterBufferHEVC, slice_pic_parameter_set_id);
    F(VAEncPictureParameterBufferHEVC, nal_unit_type);
    F(VAEncPictureParameterBufferHEVC, pic_fields);
    F(VAEncPictureParameterBufferHEVC, hierarchical_level_plus1);
    F(VAEncPictureParameterBufferHEVC, va_byte_reserved);
    F(VAEncPictureParameterBufferHEVC, scc_fields);
    F(VAEncPictureParameterBufferHEVC, va_reserved);

    S(VAEncSliceParameterBufferHEVC);
    F(VAEncSliceParameterBufferHEVC, slice_segment_address);
    F(VAEncSliceParameterBufferHEVC, num_ctu_in_slice);
    F(VAEncSliceParameterBufferHEVC, slice_type);
    F(VAEncSliceParameterBufferHEVC, slice_pic_parameter_set_id);
    F(VAEncSliceParameterBufferHEVC, num_ref_idx_l0_active_minus1);
    F(VAEncSliceParameterBufferHEVC, num_ref_idx_l1_active_minus1);
    F(VAEncSliceParameterBufferHEVC, ref_pic_list0);
    F(VAEncSliceParameterBufferHEVC, ref_pic_list1);
    F(VAEncSliceParameterBufferHEVC, luma_log2_weight_denom);
    F(VAEncSliceParameterBufferHEVC, delta_chroma_log2_weight_denom);
    F(VAEncSliceParameterBufferHEVC, delta_luma_weight_l0);
    F(VAEncSliceParameterBufferHEVC, luma_offset_l0);
    F(VAEncSliceParameterBufferHEVC, delta_chroma_weight_l0);
    F(VAEncSliceParameterBufferHEVC, chroma_offset_l0);
    F(VAEncSliceParameterBufferHEVC, delta_luma_weight_l1);
    F(VAEncSliceParameterBufferHEVC, luma_offset_l1);
    F(VAEncSliceParameterBufferHEVC, delta_chroma_weight_l1);
    F(VAEncSliceParameterBufferHEVC, chroma_offset_l1);
    F(VAEncSliceParameterBufferHEVC, max_num_merge_cand);
    F(VAEncSliceParameterBufferHEVC, slice_qp_delta);
    F(VAEncSliceParameterBufferHEVC, slice_cb_qp_offset);
    F(VAEncSliceParameterBufferHEVC, slice_cr_qp_offset);
    F(VAEncSliceParameterBufferHEVC, slice_beta_offset_div2);
    F(VAEncSliceParameterBufferHEVC, slice_tc_offset_div2);
    F(VAEncSliceParameterBufferHEVC, slice_fields);
    F(VAEncSliceParameterBufferHEVC, pred_weight_table_bit_offset);
    F(VAEncSliceParameterBufferHEVC, pred_weight_table_bit_length);
    F(VAEncSliceParameterBufferHEVC, va_reserved);

    S(VAEncPackedHeaderParameterBuffer);
    F(VAEncPackedHeaderParameterBuffer, type);
    F(VAEncPackedHeaderParameterBuffer, bit_length);
    F(VAEncPackedHeaderParameterBuffer, has_emulation_bytes);
    F(VAEncPackedHeaderParameterBuffer, va_reserved);

    F(VAEncMiscParameterBuffer, data);

    S(VAEncMiscParameterRateControl);
    F(VAEncMiscParameterRateControl, bits_per_second);
    F(VAEncMiscParameterRateControl, target_percentage);
    F(VAEncMiscParameterRateControl, window_size);
    F(VAEncMiscParameterRateControl, initial_qp);
    F(VAEncMiscParameterRateControl, min_qp);
    F(VAEncMiscParameterRateControl, basic_unit_size);
    F(VAEncMiscParameterRateControl, rc_flags);
    F(VAEncMiscParameterRateControl, ICQ_quality_factor);
    F(VAEncMiscParameterRateControl, max_qp);
    F(VAEncMiscParameterRateControl, quality_factor);
    F(VAEncMiscParameterRateControl, target_frame_size);
    F(VAEncMiscParameterRateControl, va_reserved);

    S(VAEncMiscParameterFrameRate);
    F(VAEncMiscParameterFrameRate, framerate);
    F(VAEncMiscParameterFrameRate, framerate_flags);
    F(VAEncMiscParameterFrameRate, va_reserved);

    S(VAEncMiscParameterHRD);
    F(VAEncMiscParameterHRD, initial_buffer_fullness);
    F(VAEncMiscParameterHRD, buffer_size);
    F(VAEncMiscParameterHRD, va_reserved);

    S(VAEncMiscParameterBufferQualityLevel);
    F(VAEncMiscParameterBufferQualityLevel, quality_level);
    F(VAEncMiscParameterBufferQualityLevel, va_reserved);

    S(VAEncMiscParameterBufferMaxFrameSize);
    F(VAEncMiscParameterBufferMaxFrameSize, type);
    F(VAEncMiscParameterBufferMaxFrameSize, max_frame_size);
    F(VAEncMiscParameterBufferMaxFrameSize, va_reserved);

    S(VACodedBufferSegment);
    F(VACodedBufferSegment, size);
    F(VACodedBufferSegment, bit_offset);
    F(VACodedBufferSegment, status);
    F(VACodedBufferSegment, reserved);
    F(VACodedBufferSegment, buf);
    F(VACodedBufferSegment, next);
    F(VACodedBufferSegment, va_reserved);

#define HS(f) BITS(VAEncSequenceParameterBufferH264, seq_fields, "h264.seq_fields", f)
    HS(chroma_format_idc);
    HS(frame_mbs_only_flag);
    HS(mb_adaptive_frame_field_flag);
    HS(seq_scaling_matrix_present_flag);
    HS(direct_8x8_inference_flag);
    HS(log2_max_frame_num_minus4);
    HS(pic_order_cnt_type);
    HS(log2_max_pic_order_cnt_lsb_minus4);
    HS(delta_pic_order_always_zero_flag);
#define HV(f) BITS(VAEncSequenceParameterBufferH264, vui_fields, "h264.vui_fields", f)
    HV(aspect_ratio_info_present_flag);
    HV(timing_info_present_flag);
    HV(bitstream_restriction_flag);
    HV(log2_max_mv_length_horizontal);
    HV(log2_max_mv_length_vertical);
    HV(fixed_frame_rate_flag);
    HV(low_delay_hrd_flag);
    HV(motion_vectors_over_pic_boundaries_flag);
#define HP(f) BITS(VAEncPictureParameterBufferH264, pic_fields, "h264.pic_fields", f)
    HP(idr_pic_flag);
    HP(reference_pic_flag);
    HP(entropy_coding_mode_flag);
    HP(weighted_pred_flag);
    HP(weighted_bipred_idc);
    HP(constrained_intra_pred_flag);
    HP(transform_8x8_mode_flag);
    HP(deblocking_filter_control_present_flag);
    HP(redundant_pic_cnt_present_flag);
    HP(pic_order_present_flag);
    HP(pic_scaling_matrix_present_flag);
#define ES(f) BITS(VAEncSequenceParameterBufferHEVC, seq_fields, "hevc.seq_fields", f)
    ES(chroma_format_idc);
    ES(separate_colour_plane_flag);
    ES(bit_depth_luma_minus8);
    ES(bit_depth_chroma_minus8);
    ES(scaling_list_enabled_flag);
    ES(strong_intra_smoothing_enabled_flag);
    ES(amp_enabled_flag);
    ES(sample_adaptive_offset_enabled_flag);
    ES(pcm_enabled_flag);
    ES(pcm_loop_filter_disabled_flag);
    ES(sps_temporal_mvp_enabled_flag);
    ES(low_delay_seq);
    ES(hierachical_flag);
#define EV(f) BITS(VAEncSequenceParameterBufferHEVC, vui_fields, "hevc.vui_fields", f)
    EV(aspect_ratio_info_present_flag);
    EV(neutral_chroma_indication_flag);
    EV(field_seq_flag);
    EV(vui_timing_info_present_flag);
    EV(bitstream_restriction_flag);
    EV(tiles_fixed_structure_flag);
    EV(motion_vectors_over_pic_boundaries_flag);
    EV(restricted_ref_pic_lists_flag);
    EV(log2_max_mv_length_horizontal);
    EV(log2_max_mv_length_vertical);
#define EP(f) BITS(VAEncPictureParameterBufferHEVC, pic_fields, "hevc.pic_fields", f)
    EP(idr_pic_flag);
    EP(coding_type);
    EP(reference_pic_flag);
    EP(dependent_slice_segments_enabled_flag);
    EP(sign_data_hiding_enabled_flag);
    EP(constrained_intra_pred_flag);
    EP(transform_skip_enabled_flag);
    EP(cu_qp_delta_enabled_flag);
    EP(weighted_pred_flag);
    EP(weighted_bipred_flag);
    EP(transquant_bypass_enabled_flag);
    EP(tiles_enabled_flag);
    EP(entropy_coding_sync_enabled_flag);
    EP(loop_filter_across_tiles_enabled_flag);
    EP(pps_loop_filter_across_slices_enabled_flag);
    EP(scaling_list_data_present_flag);
    EP(screen_content_flag);
    EP(enable_gpu_weighted_prediction);
    EP(no_output_of_prior_pics_flag);
#define EL(f) BITS(VAEncSliceParameterBufferHEVC, slice_fields, "hevc.slice_fields", f)
    EL(last_slice_of_pic_flag);
    EL(dependent_slice_segment_flag);
    EL(colour_plane_id);
    EL(slice_temporal_mvp_enabled_flag);
    EL(slice_sao_luma_flag);
    EL(slice_sao_chroma_flag);
    EL(num_ref_idx_active_override_flag);
    EL(mvd_l1_zero_flag);
    EL(cabac_init_flag);
    EL(slice_deblocking_filter_disabled_flag);
    EL(slice_loop_filter_across_slices_enabled_flag);
    EL(collocated_from_l0_flag);
#define RC(f) BITS(VAEncMiscParameterRateControl, rc_flags, "rc_flags", f)
    RC(reset);
    RC(disable_frame_skip);
    RC(disable_bit_stuffing);
    RC(mb_rate_control);
    RC(temporal_id);
    RC(cfs_I_frames);
    RC(enable_parallel_brc);
    RC(enable_dynamic_scaling);
    RC(frame_tolerance_mode);
    C(VAEntrypointEncSlice);
    C(VAEntrypointEncSliceLP);
    C(VAConfigAttribEncSliceStructure);
    C(VAConfigAttribEncQualityRange);
    C(VAConfigAttribEncIntraRefresh);
    C(VAConfigAttribEncSkipFrame);
    C(VAConfigAttribEncROI);
    C(VAConfigAttribEncRateControlExt);
    C(VAConfigAttribMaxFrameSize);
    C(VAConfigAttribPredictionDirection);
    C(VAConfigAttribEncHEVCFeatures);
    C(VAConfigAttribEncHEVCBlockSizes);
    C(VA_RC_CBR);
    C(VA_RC_VBR);
    C(VA_RC_CQP);
    C(VA_RC_QVBR);
    C(VA_ENC_PACKED_HEADER_SEQUENCE);
    C(VA_ENC_PACKED_HEADER_PICTURE);
    C(VA_ENC_PACKED_HEADER_SLICE);
    C(VA_ENC_PACKED_HEADER_MISC);
    C(VA_ENC_PACKED_HEADER_RAW_DATA);
    C(VAEncPackedHeaderSequence);
    C(VAEncPackedHeaderPicture);
    C(VAEncPackedHeaderSlice);
    C(VAEncPackedHeaderRawData);
    C(VAEncCodedBufferType);
    C(VAEncSequenceParameterBufferType);
    C(VAEncPictureParameterBufferType);
    C(VAEncSliceParameterBufferType);
    C(VAEncPackedHeaderParameterBufferType);
    C(VAEncPackedHeaderDataBufferType);
    C(VAEncMiscParameterBufferType);
    C(VAEncMiscParameterTypeFrameRate);
    C(VAEncMiscParameterTypeRateControl);
    C(VAEncMiscParameterTypeMaxFrameSize);
    C(VAEncMiscParameterTypeHRD);
    C(VAEncMiscParameterTypeQualityLevel);
    C(VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW);
    C(VA_CODED_BUF_STATUS_BAD_BITSTREAM);
    return 0;
}
