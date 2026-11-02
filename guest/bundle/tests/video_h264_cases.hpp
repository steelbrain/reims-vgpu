/*
 * video_h264_cases: the parameter sets of seven streams x264 encoded from ffmpeg's testsrc2,
 * each as the fields AppleGVA's parsed form holds and the NAL units x264 wrote, with the bit at
 * which the SPS's vui_parameters_present_flag sits (the NAL header's first bit is bit 0). The
 * fields were read with ffmpeg's trace_headers. The streams:
 *   high       1920x1080 High, x264's defaults (cropped to 1080)
 *   baseline   640x360 Constrained Baseline (POC type 2, CAVLC, no 8x8 transform)
 *   main       1280x720 Main, 2 B-frames, 3 references, weighted P prediction
 *   cqm        640x480 High, --cqm jvt (the PPS's scaling matrix flag, every list left out)
 *   customcqm  640x480 High, a CQM file of its own (PPS scaling lists written out, a run cut short)
 *   mbaff      720x480 High, interlaced (MBAFF, top field first)
 *   qpoffs     352x288 High, --chroma-qp-offset -3, 5 references, no deblocking
 * The lists are in their coded (scan) order.
 */

#ifndef VIDEO_H264_CASES_HPP_
#define VIDEO_H264_CASES_HPP_

Case case_high() {
	Case c = {};
	c.name = "high";
	c.sps.profile_idc = 100;
	c.sps.level_idc = 40;
	c.sps.seq_parameter_set_id = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.bit_depth_luma_minus8 = 0;
	c.sps.bit_depth_chroma_minus8 = 0;
	c.sps.qpprime_y_zero_transform_bypass_flag = 0;
	c.sps.seq_scaling_matrix_present_flag = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 4;
	c.sps.pic_width_in_mbs_minus1 = 119;
	c.sps.pic_height_in_map_units_minus1 = 67;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 1;
	c.sps.frame_crop_left_offset = 0;
	c.sps.frame_crop_right_offset = 0;
	c.sps.frame_crop_top_offset = 0;
	c.sps.frame_crop_bottom_offset = 4;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 0;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.transform_8x8_mode_flag = 1;
	c.pps.pic_scaling_matrix_present_flag = 0;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x64,0x0,0x28,0xac,0xd9,0x40,0x78,0x2,0x27,0xe5,0xc0,0x44,0x0,0x0,0x3,0x0,0x4,0x0,0x0,0x3,0x0,0xf0,0x3c,0x60,0xc6,0x58};
	c.pps_nal = {0x68,0xeb,0xe3,0xcb,0x22,0xc0};
	c.sps_vui_bit = 88;
	return c;
}

Case case_baseline() {
	Case c = {};
	c.name = "baseline";
	c.sps.profile_idc = 66;
	c.sps.level_idc = 30;
	c.sps.seq_parameter_set_id = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 2;
	c.sps.max_num_ref_frames = 3;
	c.sps.pic_width_in_mbs_minus1 = 39;
	c.sps.pic_height_in_map_units_minus1 = 22;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 1;
	c.sps.frame_crop_left_offset = 0;
	c.sps.frame_crop_right_offset = 0;
	c.sps.frame_crop_top_offset = 0;
	c.sps.frame_crop_bottom_offset = 4;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.constraint_set_flag[0] = 1;
	c.sps.constraint_set_flag[1] = 1;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 0;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 0;
	c.pps.weighted_bipred_idc = 0;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x42,0xc0,0x1e,0xd9,0x0,0xa0,0x2f,0xf9,0x70,0x11,0x0,0x0,0x3,0x0,0x1,0x0,0x0,0x3,0x0,0x3c,0xf,0x16,0x2e,0x48};
	c.pps_nal = {0x68,0xcb,0x83,0xcb,0x20};
	c.sps_vui_bit = 74;
	return c;
}

Case case_main() {
	Case c = {};
	c.name = "main";
	c.sps.profile_idc = 77;
	c.sps.level_idc = 31;
	c.sps.seq_parameter_set_id = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 4;
	c.sps.pic_width_in_mbs_minus1 = 79;
	c.sps.pic_height_in_map_units_minus1 = 44;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 0;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 1;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x4d,0x40,0x1f,0xec,0xa0,0x28,0x2,0xdd,0x80,0x88,0x0,0x0,0x3,0x0,0x8,0x0,0x0,0x3,0x1,0xe0,0x78,0xc1,0x8c,0xb0};
	c.pps_nal = {0x68,0xeb,0xe3,0xcb,0x20};
	c.sps_vui_bit = 71;
	return c;
}

Case case_cqm() {
	Case c = {};
	c.name = "cqm";
	c.sps.profile_idc = 100;
	c.sps.level_idc = 30;
	c.sps.seq_parameter_set_id = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.bit_depth_luma_minus8 = 0;
	c.sps.bit_depth_chroma_minus8 = 0;
	c.sps.qpprime_y_zero_transform_bypass_flag = 0;
	c.sps.seq_scaling_matrix_present_flag = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 4;
	c.sps.pic_width_in_mbs_minus1 = 39;
	c.sps.pic_height_in_map_units_minus1 = 29;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 0;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 0;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_scaling_list_present_flag[0] = 0;
	c.pps.pic_scaling_list_present_flag[1] = 0;
	c.pps.pic_scaling_list_present_flag[2] = 0;
	c.pps.pic_scaling_list_present_flag[3] = 0;
	c.pps.pic_scaling_list_present_flag[4] = 0;
	c.pps.pic_scaling_list_present_flag[5] = 0;
	c.pps.pic_scaling_list_present_flag[6] = 0;
	c.pps.pic_scaling_list_present_flag[7] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.transform_8x8_mode_flag = 1;
	c.pps.pic_scaling_matrix_present_flag = 1;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x64,0x0,0x1e,0xac,0xd9,0x40,0xa0,0x3d,0xb0,0x11,0x0,0x0,0x3,0x0,0x1,0x0,0x0,0x3,0x0,0x3c,0xf,0x16,0x2d,0x96};
	c.pps_nal = {0x68,0xeb,0xe3,0xcb,0x30,0x2,0xc0};
	c.sps_vui_bit = 74;
	return c;
}

Case case_customcqm() {
	Case c = {};
	c.name = "customcqm";
	c.sps.profile_idc = 100;
	c.sps.level_idc = 30;
	c.sps.seq_parameter_set_id = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.bit_depth_luma_minus8 = 0;
	c.sps.bit_depth_chroma_minus8 = 0;
	c.sps.qpprime_y_zero_transform_bypass_flag = 0;
	c.sps.seq_scaling_matrix_present_flag = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 4;
	c.sps.pic_width_in_mbs_minus1 = 39;
	c.sps.pic_height_in_map_units_minus1 = 29;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 0;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 0;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_scaling_list_present_flag[0] = 0;
	c.pps.pic_scaling_list_present_flag[1] = 1;
	c.pps.pic_scaling_list_present_flag[2] = 0;
	c.pps.pic_scaling_list_present_flag[3] = 0;
	c.pps.pic_scaling_list_present_flag[4] = 1;
	c.pps.pic_scaling_list_present_flag[5] = 0;
	c.pps.pic_scaling_list_present_flag[6] = 0;
	c.pps.pic_scaling_list_present_flag[7] = 0;
	fill(c.pps.scaling_list_4x4[1], {7,13,13,21,21,21,28,28,28,28,33,33,33,37,37,43});
	fill(c.pps.scaling_list_4x4[4], {11,15,15,21,21,21,25,25,25,25,28,28,28,31,31,35});
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.transform_8x8_mode_flag = 1;
	c.pps.pic_scaling_matrix_present_flag = 1;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x64,0x0,0x1e,0xac,0xd9,0x40,0xa0,0x3d,0xb0,0x11,0x0,0x0,0x3,0x0,0x1,0x0,0x0,0x3,0x0,0x3c,0xf,0x16,0x2d,0x96};
	c.pps_nal = {0x68,0xeb,0xe3,0xcb,0x35,0x8c,0x84,0x31,0xdc,0x56,0x22,0x30,0x98,0x44,0x66,0x23,0x9b,0x34,0x40,0x2c};
	c.sps_vui_bit = 74;
	return c;
}

Case case_mbaff() {
	Case c = {};
	c.name = "mbaff";
	c.sps.profile_idc = 100;
	c.sps.level_idc = 30;
	c.sps.seq_parameter_set_id = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.bit_depth_luma_minus8 = 0;
	c.sps.bit_depth_chroma_minus8 = 0;
	c.sps.qpprime_y_zero_transform_bypass_flag = 0;
	c.sps.seq_scaling_matrix_present_flag = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 4;
	c.sps.pic_width_in_mbs_minus1 = 44;
	c.sps.pic_height_in_map_units_minus1 = 14;
	c.sps.frame_mbs_only_flag = 0;
	c.sps.mb_adaptive_frame_field_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 0;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 0;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 1;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 2;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 0;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -2;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.transform_8x8_mode_flag = 1;
	c.pps.pic_scaling_matrix_present_flag = 0;
	c.pps.second_chroma_qp_index_offset = -2;
	c.sps_nal = {0x67,0x64,0x0,0x1e,0xac,0xd9,0x40,0xb4,0x7b,0x60,0x22,0x0,0x0,0x3,0x0,0x2,0x0,0x0,0x3,0x0,0x78,0x3e,0x28,0x53,0x2c};
	c.pps_nal = {0x68,0xfb,0xa3,0xcb,0x22,0xc0};
	c.sps_vui_bit = 73;
	return c;
}

Case case_qpoffs() {
	Case c = {};
	c.name = "qpoffs";
	c.sps.profile_idc = 100;
	c.sps.level_idc = 13;
	c.sps.seq_parameter_set_id = 0;
	c.sps.chroma_format_idc = 1;
	c.sps.bit_depth_luma_minus8 = 0;
	c.sps.bit_depth_chroma_minus8 = 0;
	c.sps.qpprime_y_zero_transform_bypass_flag = 0;
	c.sps.seq_scaling_matrix_present_flag = 0;
	c.sps.log2_max_frame_num_minus4 = 0;
	c.sps.pic_order_cnt_type = 0;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	c.sps.max_num_ref_frames = 5;
	c.sps.pic_width_in_mbs_minus1 = 21;
	c.sps.pic_height_in_map_units_minus1 = 17;
	c.sps.frame_mbs_only_flag = 1;
	c.sps.direct_8x8_inference_flag = 1;
	c.sps.frame_cropping_flag = 0;
	c.sps.gaps_in_frame_num_value_allowed_flag = 0;
	c.sps.constraint_set_flag[0] = 0;
	c.sps.constraint_set_flag[1] = 0;
	c.sps.constraint_set_flag[2] = 0;
	c.sps.constraint_set_flag[3] = 0;
	c.sps.constraint_set_flag[4] = 0;
	c.sps.constraint_set_flag[5] = 0;
	c.pps.pic_parameter_set_id = 0;
	c.pps.seq_parameter_set_id = 0;
	c.pps.entropy_coding_mode_flag = 1;
	c.pps.bottom_field_pic_order_in_frame_present_flag = 0;
	c.pps.num_slice_groups_minus1 = 0;
	c.pps.num_ref_idx_l0_default_active_minus1 = 4;
	c.pps.num_ref_idx_l1_default_active_minus1 = 0;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_idc = 2;
	c.pps.pic_init_qp_minus26 = -3;
	c.pps.pic_init_qs_minus26 = 0;
	c.pps.chroma_qp_index_offset = -5;
	c.pps.deblocking_filter_control_present_flag = 1;
	c.pps.constrained_intra_pred_flag = 0;
	c.pps.redundant_pic_cnt_present_flag = 0;
	c.pps.transform_8x8_mode_flag = 1;
	c.pps.pic_scaling_matrix_present_flag = 0;
	c.pps.second_chroma_qp_index_offset = -5;
	c.sps_nal = {0x67,0x64,0x0,0xd,0xac,0xd9,0x81,0x60,0x96,0xc0,0x44,0x0,0x0,0x3,0x0,0x4,0x0,0x0,0x3,0x0,0xf0,0x3c,0x50,0xa6,0x68};
	c.pps_nal = {0x68,0xe9,0x78,0xf1,0x72,0x17};
	c.sps_vui_bit = 72;
	return c;
}

const std::vector<Case (*)()> kCases = {case_high, case_baseline, case_main, case_cqm, case_customcqm, case_mbaff, case_qpoffs};

#endif
