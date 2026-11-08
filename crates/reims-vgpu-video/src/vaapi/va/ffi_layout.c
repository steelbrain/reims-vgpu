/*
 * The layout of the libva structs ffi.rs declares, from the installed
 * headers: sizeof and offsetof of each, then the mask of each bit-field,
 * then the value of each constant.
 * ffi.rs's layouts_match_libva_headers test compiles this and compares its
 * output with Rust's, line for line.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <va/va.h>
#include <va/va_drmcommon.h>
#include <drm/drm_fourcc.h>

#define S(T) printf(#T " size %zu\n", sizeof(T))
#define F(T, f) printf(#T "." #f " %zu\n", offsetof(T, f))
#define C(c) printf(#c " %lld\n", (long long)(c))
#define U(c) printf(#c " %llu\n", (unsigned long long)(c))
#define SEQ(f)                                                         \
    do {                                                               \
        VAPictureParameterBufferH264 p;                                \
        memset(&p, 0, sizeof p);                                       \
        p.seq_fields.bits.f = ~0u;                                     \
        printf("seq_fields." #f " %#x\n", p.seq_fields.value);         \
    } while (0)
#define PIC(f)                                                         \
    do {                                                               \
        VAPictureParameterBufferH264 p;                                \
        memset(&p, 0, sizeof p);                                       \
        p.pic_fields.bits.f = ~0u;                                     \
        printf("pic_fields." #f " %#x\n", p.pic_fields.value);         \
    } while (0)
/* An HEVC bit-field: union u's bits.f (or fields.f) within T. */
#define HEVC(T, u, bits, f)                                            \
    do {                                                               \
        T p;                                                           \
        memset(&p, 0, sizeof p);                                       \
        p.u.bits.f = ~0u;                                              \
        printf("hevc." #u "." #f " %#x\n", p.u.value);                 \
    } while (0)
#define HPIC(f) HEVC(VAPictureParameterBufferHEVC, pic_fields, bits, f)
#define HSLICEPARSE(f) HEVC(VAPictureParameterBufferHEVC, slice_parsing_fields, bits, f)
#define HLONG(f) HEVC(VASliceParameterBufferHEVC, LongSliceFlags, fields, f)

int main(void)
{
    S(VAConfigAttrib);
    F(VAConfigAttrib, type);
    F(VAConfigAttrib, value);

    S(VAGenericValue);
    F(VAGenericValue, type);
    F(VAGenericValue, value);

    S(VASurfaceAttrib);
    F(VASurfaceAttrib, type);
    F(VASurfaceAttrib, flags);
    F(VASurfaceAttrib, value);

    S(VAImageFormat);
    F(VAImageFormat, fourcc);
    F(VAImageFormat, byte_order);
    F(VAImageFormat, bits_per_pixel);
    F(VAImageFormat, depth);
    F(VAImageFormat, red_mask);
    F(VAImageFormat, green_mask);
    F(VAImageFormat, blue_mask);
    F(VAImageFormat, alpha_mask);
    F(VAImageFormat, va_reserved);

    S(VAImage);
    F(VAImage, image_id);
    F(VAImage, format);
    F(VAImage, buf);
    F(VAImage, width);
    F(VAImage, height);
    F(VAImage, data_size);
    F(VAImage, num_planes);
    F(VAImage, pitches);
    F(VAImage, offsets);
    F(VAImage, num_palette_entries);
    F(VAImage, entry_bytes);
    F(VAImage, component_order);
    F(VAImage, va_reserved);

    S(VADRMPRIMESurfaceDescriptor);
    F(VADRMPRIMESurfaceDescriptor, fourcc);
    F(VADRMPRIMESurfaceDescriptor, width);
    F(VADRMPRIMESurfaceDescriptor, height);
    F(VADRMPRIMESurfaceDescriptor, num_objects);
    F(VADRMPRIMESurfaceDescriptor, objects);
    F(VADRMPRIMESurfaceDescriptor, num_layers);
    F(VADRMPRIMESurfaceDescriptor, layers);
    F(VADRMPRIMESurfaceDescriptor, objects[1].drm_format_modifier);
    F(VADRMPRIMESurfaceDescriptor, layers[1].pitch);

    S(VAPictureH264);
    F(VAPictureH264, picture_id);
    F(VAPictureH264, frame_idx);
    F(VAPictureH264, flags);
    F(VAPictureH264, TopFieldOrderCnt);
    F(VAPictureH264, BottomFieldOrderCnt);
    F(VAPictureH264, va_reserved);

    S(VAPictureParameterBufferH264);
    F(VAPictureParameterBufferH264, CurrPic);
    F(VAPictureParameterBufferH264, ReferenceFrames);
    F(VAPictureParameterBufferH264, picture_width_in_mbs_minus1);
    F(VAPictureParameterBufferH264, picture_height_in_mbs_minus1);
    F(VAPictureParameterBufferH264, bit_depth_luma_minus8);
    F(VAPictureParameterBufferH264, bit_depth_chroma_minus8);
    F(VAPictureParameterBufferH264, num_ref_frames);
    F(VAPictureParameterBufferH264, seq_fields);
    F(VAPictureParameterBufferH264, num_slice_groups_minus1);
    F(VAPictureParameterBufferH264, slice_group_map_type);
    F(VAPictureParameterBufferH264, slice_group_change_rate_minus1);
    F(VAPictureParameterBufferH264, pic_init_qp_minus26);
    F(VAPictureParameterBufferH264, pic_init_qs_minus26);
    F(VAPictureParameterBufferH264, chroma_qp_index_offset);
    F(VAPictureParameterBufferH264, second_chroma_qp_index_offset);
    F(VAPictureParameterBufferH264, pic_fields);
    F(VAPictureParameterBufferH264, frame_num);
    F(VAPictureParameterBufferH264, va_reserved);

    S(VAIQMatrixBufferH264);
    F(VAIQMatrixBufferH264, ScalingList4x4);
    F(VAIQMatrixBufferH264, ScalingList8x8);
    F(VAIQMatrixBufferH264, va_reserved);

    S(VASliceParameterBufferH264);
    F(VASliceParameterBufferH264, slice_data_size);
    F(VASliceParameterBufferH264, slice_data_offset);
    F(VASliceParameterBufferH264, slice_data_flag);
    F(VASliceParameterBufferH264, slice_data_bit_offset);
    F(VASliceParameterBufferH264, first_mb_in_slice);
    F(VASliceParameterBufferH264, slice_type);
    F(VASliceParameterBufferH264, direct_spatial_mv_pred_flag);
    F(VASliceParameterBufferH264, num_ref_idx_l0_active_minus1);
    F(VASliceParameterBufferH264, num_ref_idx_l1_active_minus1);
    F(VASliceParameterBufferH264, cabac_init_idc);
    F(VASliceParameterBufferH264, slice_qp_delta);
    F(VASliceParameterBufferH264, disable_deblocking_filter_idc);
    F(VASliceParameterBufferH264, slice_alpha_c0_offset_div2);
    F(VASliceParameterBufferH264, slice_beta_offset_div2);
    F(VASliceParameterBufferH264, RefPicList0);
    F(VASliceParameterBufferH264, RefPicList1);
    F(VASliceParameterBufferH264, luma_log2_weight_denom);
    F(VASliceParameterBufferH264, chroma_log2_weight_denom);
    F(VASliceParameterBufferH264, luma_weight_l0_flag);
    F(VASliceParameterBufferH264, luma_weight_l0);
    F(VASliceParameterBufferH264, luma_offset_l0);
    F(VASliceParameterBufferH264, chroma_weight_l0_flag);
    F(VASliceParameterBufferH264, chroma_weight_l0);
    F(VASliceParameterBufferH264, chroma_offset_l0);
    F(VASliceParameterBufferH264, luma_weight_l1_flag);
    F(VASliceParameterBufferH264, luma_weight_l1);
    F(VASliceParameterBufferH264, luma_offset_l1);
    F(VASliceParameterBufferH264, chroma_weight_l1_flag);
    F(VASliceParameterBufferH264, chroma_weight_l1);
    F(VASliceParameterBufferH264, chroma_offset_l1);
    F(VASliceParameterBufferH264, va_reserved);

    S(VAPictureHEVC);
    F(VAPictureHEVC, picture_id);
    F(VAPictureHEVC, pic_order_cnt);
    F(VAPictureHEVC, flags);
    F(VAPictureHEVC, va_reserved);

    S(VAPictureParameterBufferHEVC);
    F(VAPictureParameterBufferHEVC, CurrPic);
    F(VAPictureParameterBufferHEVC, ReferenceFrames);
    F(VAPictureParameterBufferHEVC, pic_width_in_luma_samples);
    F(VAPictureParameterBufferHEVC, pic_height_in_luma_samples);
    F(VAPictureParameterBufferHEVC, pic_fields);
    F(VAPictureParameterBufferHEVC, sps_max_dec_pic_buffering_minus1);
    F(VAPictureParameterBufferHEVC, bit_depth_luma_minus8);
    F(VAPictureParameterBufferHEVC, bit_depth_chroma_minus8);
    F(VAPictureParameterBufferHEVC, pcm_sample_bit_depth_luma_minus1);
    F(VAPictureParameterBufferHEVC, pcm_sample_bit_depth_chroma_minus1);
    F(VAPictureParameterBufferHEVC, log2_min_luma_coding_block_size_minus3);
    F(VAPictureParameterBufferHEVC, log2_diff_max_min_luma_coding_block_size);
    F(VAPictureParameterBufferHEVC, log2_min_transform_block_size_minus2);
    F(VAPictureParameterBufferHEVC, log2_diff_max_min_transform_block_size);
    F(VAPictureParameterBufferHEVC, log2_min_pcm_luma_coding_block_size_minus3);
    F(VAPictureParameterBufferHEVC, log2_diff_max_min_pcm_luma_coding_block_size);
    F(VAPictureParameterBufferHEVC, max_transform_hierarchy_depth_intra);
    F(VAPictureParameterBufferHEVC, max_transform_hierarchy_depth_inter);
    F(VAPictureParameterBufferHEVC, init_qp_minus26);
    F(VAPictureParameterBufferHEVC, diff_cu_qp_delta_depth);
    F(VAPictureParameterBufferHEVC, pps_cb_qp_offset);
    F(VAPictureParameterBufferHEVC, pps_cr_qp_offset);
    F(VAPictureParameterBufferHEVC, log2_parallel_merge_level_minus2);
    F(VAPictureParameterBufferHEVC, num_tile_columns_minus1);
    F(VAPictureParameterBufferHEVC, num_tile_rows_minus1);
    F(VAPictureParameterBufferHEVC, column_width_minus1);
    F(VAPictureParameterBufferHEVC, row_height_minus1);
    F(VAPictureParameterBufferHEVC, slice_parsing_fields);
    F(VAPictureParameterBufferHEVC, log2_max_pic_order_cnt_lsb_minus4);
    F(VAPictureParameterBufferHEVC, num_short_term_ref_pic_sets);
    F(VAPictureParameterBufferHEVC, num_long_term_ref_pic_sps);
    F(VAPictureParameterBufferHEVC, num_ref_idx_l0_default_active_minus1);
    F(VAPictureParameterBufferHEVC, num_ref_idx_l1_default_active_minus1);
    F(VAPictureParameterBufferHEVC, pps_beta_offset_div2);
    F(VAPictureParameterBufferHEVC, pps_tc_offset_div2);
    F(VAPictureParameterBufferHEVC, num_extra_slice_header_bits);
    F(VAPictureParameterBufferHEVC, st_rps_bits);
    F(VAPictureParameterBufferHEVC, va_reserved);

    S(VASliceParameterBufferHEVC);
    F(VASliceParameterBufferHEVC, slice_data_size);
    F(VASliceParameterBufferHEVC, slice_data_offset);
    F(VASliceParameterBufferHEVC, slice_data_flag);
    F(VASliceParameterBufferHEVC, slice_data_byte_offset);
    F(VASliceParameterBufferHEVC, slice_segment_address);
    F(VASliceParameterBufferHEVC, RefPicList);
    F(VASliceParameterBufferHEVC, LongSliceFlags);
    F(VASliceParameterBufferHEVC, collocated_ref_idx);
    F(VASliceParameterBufferHEVC, num_ref_idx_l0_active_minus1);
    F(VASliceParameterBufferHEVC, num_ref_idx_l1_active_minus1);
    F(VASliceParameterBufferHEVC, slice_qp_delta);
    F(VASliceParameterBufferHEVC, slice_cb_qp_offset);
    F(VASliceParameterBufferHEVC, slice_cr_qp_offset);
    F(VASliceParameterBufferHEVC, slice_beta_offset_div2);
    F(VASliceParameterBufferHEVC, slice_tc_offset_div2);
    F(VASliceParameterBufferHEVC, luma_log2_weight_denom);
    F(VASliceParameterBufferHEVC, delta_chroma_log2_weight_denom);
    F(VASliceParameterBufferHEVC, delta_luma_weight_l0);
    F(VASliceParameterBufferHEVC, luma_offset_l0);
    F(VASliceParameterBufferHEVC, delta_chroma_weight_l0);
    F(VASliceParameterBufferHEVC, ChromaOffsetL0);
    F(VASliceParameterBufferHEVC, delta_luma_weight_l1);
    F(VASliceParameterBufferHEVC, luma_offset_l1);
    F(VASliceParameterBufferHEVC, delta_chroma_weight_l1);
    F(VASliceParameterBufferHEVC, ChromaOffsetL1);
    F(VASliceParameterBufferHEVC, five_minus_max_num_merge_cand);
    F(VASliceParameterBufferHEVC, num_entry_point_offsets);
    F(VASliceParameterBufferHEVC, entry_offset_to_subset_array);
    F(VASliceParameterBufferHEVC, slice_data_num_emu_prevn_bytes);
    F(VASliceParameterBufferHEVC, va_reserved);

    S(VAIQMatrixBufferHEVC);
    F(VAIQMatrixBufferHEVC, ScalingList4x4);
    F(VAIQMatrixBufferHEVC, ScalingList8x8);
    F(VAIQMatrixBufferHEVC, ScalingList16x16);
    F(VAIQMatrixBufferHEVC, ScalingList32x32);
    F(VAIQMatrixBufferHEVC, ScalingListDC16x16);
    F(VAIQMatrixBufferHEVC, ScalingListDC32x32);
    F(VAIQMatrixBufferHEVC, va_reserved);

    SEQ(chroma_format_idc);
    SEQ(residual_colour_transform_flag);
    SEQ(gaps_in_frame_num_value_allowed_flag);
    SEQ(frame_mbs_only_flag);
    SEQ(mb_adaptive_frame_field_flag);
    SEQ(direct_8x8_inference_flag);
    SEQ(MinLumaBiPredSize8x8);
    SEQ(log2_max_frame_num_minus4);
    SEQ(pic_order_cnt_type);
    SEQ(log2_max_pic_order_cnt_lsb_minus4);
    SEQ(delta_pic_order_always_zero_flag);

    PIC(entropy_coding_mode_flag);
    PIC(weighted_pred_flag);
    PIC(weighted_bipred_idc);
    PIC(transform_8x8_mode_flag);
    PIC(field_pic_flag);
    PIC(constrained_intra_pred_flag);
    PIC(pic_order_present_flag);
    PIC(deblocking_filter_control_present_flag);
    PIC(redundant_pic_cnt_present_flag);
    PIC(reference_pic_flag);

    HPIC(chroma_format_idc);
    HPIC(separate_colour_plane_flag);
    HPIC(pcm_enabled_flag);
    HPIC(scaling_list_enabled_flag);
    HPIC(transform_skip_enabled_flag);
    HPIC(amp_enabled_flag);
    HPIC(strong_intra_smoothing_enabled_flag);
    HPIC(sign_data_hiding_enabled_flag);
    HPIC(constrained_intra_pred_flag);
    HPIC(cu_qp_delta_enabled_flag);
    HPIC(weighted_pred_flag);
    HPIC(weighted_bipred_flag);
    HPIC(transquant_bypass_enabled_flag);
    HPIC(tiles_enabled_flag);
    HPIC(entropy_coding_sync_enabled_flag);
    HPIC(pps_loop_filter_across_slices_enabled_flag);
    HPIC(loop_filter_across_tiles_enabled_flag);
    HPIC(pcm_loop_filter_disabled_flag);
    HPIC(NoPicReorderingFlag);
    HPIC(NoBiPredFlag);

    HSLICEPARSE(lists_modification_present_flag);
    HSLICEPARSE(long_term_ref_pics_present_flag);
    HSLICEPARSE(sps_temporal_mvp_enabled_flag);
    HSLICEPARSE(cabac_init_present_flag);
    HSLICEPARSE(output_flag_present_flag);
    HSLICEPARSE(dependent_slice_segments_enabled_flag);
    HSLICEPARSE(pps_slice_chroma_qp_offsets_present_flag);
    HSLICEPARSE(sample_adaptive_offset_enabled_flag);
    HSLICEPARSE(deblocking_filter_override_enabled_flag);
    HSLICEPARSE(pps_disable_deblocking_filter_flag);
    HSLICEPARSE(slice_segment_header_extension_present_flag);
    HSLICEPARSE(RapPicFlag);
    HSLICEPARSE(IdrPicFlag);
    HSLICEPARSE(IntraPicFlag);

    HLONG(LastSliceOfPic);
    HLONG(dependent_slice_segment_flag);
    HLONG(slice_type);
    HLONG(color_plane_id);
    HLONG(slice_sao_luma_flag);
    HLONG(slice_sao_chroma_flag);
    HLONG(mvd_l1_zero_flag);
    HLONG(cabac_init_flag);
    HLONG(slice_temporal_mvp_enabled_flag);
    HLONG(slice_deblocking_filter_disabled_flag);
    HLONG(collocated_from_l0_flag);
    HLONG(slice_loop_filter_across_slices_enabled_flag);
    C(VA_STATUS_SUCCESS);
    C(VA_STATUS_ERROR_TIMEDOUT);
    C(VA_STATUS_ERROR_UNIMPLEMENTED);
    C(VA_INVALID_ID);
    C(VA_INVALID_SURFACE);
    C(VA_ATTRIB_NOT_SUPPORTED);
    C(VA_PROGRESSIVE);
    C(VA_RT_FORMAT_YUV420);
    C(VA_RT_FORMAT_YUV420_10);
    C(VA_FOURCC_NV12);
    C(VA_FOURCC_P010);
    C(VA_SLICE_DATA_FLAG_ALL);
    C(VA_PICTURE_H264_INVALID);
    C(VA_PICTURE_H264_SHORT_TERM_REFERENCE);
    C(VA_PICTURE_H264_LONG_TERM_REFERENCE);
    C(VA_PICTURE_HEVC_INVALID);
    C(VA_PICTURE_HEVC_LONG_TERM_REFERENCE);
    C(VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE);
    C(VA_PICTURE_HEVC_RPS_ST_CURR_AFTER);
    C(VA_PICTURE_HEVC_RPS_LT_CURR);
    C(VAProfileNone);
    C(VAProfileH264Main);
    C(VAProfileH264High);
    C(VAProfileH264ConstrainedBaseline);
    C(VAProfileHEVCMain);
    C(VAProfileHEVCMain10);
    C(VAEntrypointVLD);
    C(VAEntrypointVideoProc);
    C(VAConfigAttribRTFormat);
    C(VAConfigAttribRateControl);
    C(VAConfigAttribDecSliceMode);
    C(VAConfigAttribEncPackedHeaders);
    C(VAConfigAttribEncMaxRefFrames);
    C(VAConfigAttribEncMaxSlices);
    C(VAConfigAttribMaxPictureWidth);
    C(VAConfigAttribMaxPictureHeight);
    C(VASurfaceAttribPixelFormat);
    C(VASurfaceAttribMinWidth);
    C(VASurfaceAttribMaxWidth);
    C(VASurfaceAttribMinHeight);
    C(VASurfaceAttribMaxHeight);
    C(VASurfaceAttribMemoryType);
    C(VASurfaceAttribExternalBufferDescriptor);
    C(VA_SURFACE_ATTRIB_SETTABLE);
    C(VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2);
    C(VA_EXPORT_SURFACE_READ_ONLY);
    C(VA_EXPORT_SURFACE_SEPARATE_LAYERS);
    C(VA_EXPORT_SURFACE_COMPOSED_LAYERS);
    C(DRM_FORMAT_MOD_LINEAR);
    C(DRM_FORMAT_NV12);
    C(DRM_FORMAT_P010);
    C(VAGenericValueTypeInteger);
    C(VAGenericValueTypePointer);
    C(VAPictureParameterBufferType);
    C(VAIQMatrixBufferType);
    C(VASliceParameterBufferType);
    C(VASliceDataBufferType);
    C(VA_PADDING_LOW);
    C(VA_PADDING_MEDIUM);
    C(VA_PADDING_HIGH);
    C(VA_PADDING_LARGE);
    return 0;
}
