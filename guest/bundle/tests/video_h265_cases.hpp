/*
 * video_h265_cases: the parameter sets of eleven HEVC streams x265 4.3 encoded from ffmpeg's
 * test sources, three of them rewritten by hevc-craft (crates/reims-vgpu-video/examples),
 * each as the fields AppleGVA's parsed form holds, each scaling list and short-term set derived,
 * and the NAL units the encoder wrote, with the bits the test compares to (the NAL header's first
 * bit is bit 0). The fields were read with ffmpeg's trace_headers. The streams:
 *   hevc-main-720p           1280x720 Main, 4 B-frames in a pyramid, 4 references, open GOP
 *   hevc-main-1080p          1920x1080 Main, 4 slices
 *   hevc-main10-1080p        1920x1080 Main 10, 3 slices, the default scaling lists, weighted B
 *   hevc-main-1366x766       1366x766 Main, 16x16 CTUs (a conformance window), AMP, transform skip
 *   hevc-main-720p-cqm       1280x720 Main, scaling lists of its own in the SPS (some copied)
 *   hevc-main-720p-sets-sps  hevc-main-720p with its short-term sets in the SPS, most predicted
 *   hevc-main-720p-ref1-lt   1280x720 Main, one reference, made long-term from the SPS's list
 *   hevc-main-720p-lists     hevc-main-720p with reference list modification in its PPS
 *   hevc-main-720p-temporal  1280x720 Main, 3 temporal sub-layers
 *   hevc-main-360p-lossless  640x360 Main, lossless (transquant bypass)
 *   hevc-main-720p-fade      1280x720 Main, weighted P and B prediction
 * Each list is in raster order, a row after another, as the parsed form holds it.
 */

#ifndef VIDEO_H265_CASES_HPP_
#define VIDEO_H265_CASES_HPP_

Case case_hevc_main_720p()
{
	Case c = {};
	c.name = "hevc-main-720p";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 5;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0xa4,0x93,0x2b,0xc0,0x40,0x40,0x0,0x0,0x3,0x0,0x40,0x0,0x0,0x7,0x82};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x62,0x40};
	c.sps_vui_bit = 208;
	c.sps_sets_bit = 205;
	c.sps_sets_end_bit = 205;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_1080p()
{
	Case c = {};
	c.name = "hevc-main-1080p";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 120;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1920;
	c.sps.pic_height_in_luma_samples = 1080;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 4;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x78,0xa0,0x3,0xc0,0x80,0x10,0xe5,0x96,0x56,0x59,0x24,0xca,0xf0,0x10,0x10,0x0,0x0,0x3,0x0,0x10,0x0,0x0,0x3,0x1,0xe0,0x80};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x42,0x40};
	c.sps_vui_bit = 210;
	c.sps_sets_bit = 207;
	c.sps_sets_end_bit = 207;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main10_1080p()
{
	Case c = {};
	c.name = "hevc-main10-1080p";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 2;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 120;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1920;
	c.sps.pic_height_in_luma_samples = 1080;
	c.sps.bit_depth_luma_minus8 = 2;
	c.sps.bit_depth_chroma_minus8 = 2;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 4;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.scaling_list_enabled_flag = 1;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x2,0x20,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x78,0xa0,0x3,0xc0,0x80,0x10,0xe4,0xd9,0x65,0x65,0x92,0x4e,0x57,0x80,0x80,0x80,0x0,0x0,0x3,0x0,0x80,0x0,0x0,0xf,0x4};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb6,0x42,0x40};
	c.sps_vui_bit = 215;
	c.sps_sets_bit = 212;
	c.sps_sets_end_bit = 212;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_1366x766()
{
	Case c = {};
	c.name = "hevc-main-1366x766";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 120;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1368;
	c.sps.pic_height_in_luma_samples = 768;
	c.sps.conformance_window_flag = 1;
	c.sps.conf_win_right_offset = 1;
	c.sps.conf_win_bottom_offset = 1;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 3;
	c.sps.log2_diff_max_min_luma_coding_block_size = 1;
	c.sps.log2_diff_max_min_luma_transform_block_size = 2;
	c.sps.amp_enabled_flag = 1;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x78,0xa0,0x2,0xac,0x80,0x30,0x1d,0x56,0x59,0x59,0x2a,0xf6,0xbc,0x4,0x4,0x0,0x0,0x3,0x0,0x4,0x0,0x0,0x3,0x0,0x78,0x20};
	c.pps_nal = {0x44,0x1,0xc1,0x73,0xd1,0x89};
	c.sps_vui_bit = 212;
	c.sps_sets_bit = 209;
	c.sps_sets_end_bit = 209;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_720p_cqm()
{
	Case c = {};
	c.name = "hevc-main-720p-cqm";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 4;
	c.sps.log2_diff_max_min_luma_coding_block_size = 2;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.scaling_list_enabled_flag = 1;
	c.sps.sps_scaling_list_data_present_flag = 1;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	fill(c.sps.scaling_lists.list[0][0], {34,37,40,43,41,44,6,9,7,10,13,16,14,17,20,23});
	fill(c.sps.scaling_lists.list[0][1], {8,11,14,17,15,18,21,24,22,25,28,31,29,32,35,38});
	fill(c.sps.scaling_lists.list[0][2], {8,11,14,17,15,18,21,24,22,25,28,31,29,32,35,38});
	fill(c.sps.scaling_lists.list[0][3], {45,7,10,13,11,14,17,20,18,21,24,27,25,28,31,34});
	fill(c.sps.scaling_lists.list[0][4], {19,22,25,28,26,29,32,35,33,36,39,42,40,43,46,8});
	fill(c.sps.scaling_lists.list[0][5], {19,22,25,28,26,29,32,35,33,36,39,42,40,43,46,8});
	fill(c.sps.scaling_lists.list[1][0], {38,41,44,6,9,12,15,18,45,7,10,13,16,19,22,25,11,14,17,20,23,26,29,32,18,21,24,27,30,33,36,39,25,28,31,34,37,40,43,46,32,35,38,41,44,6,9,12,39,42,45,7,10,13,16,19,46,8,11,14,17,20,23,26});
	fill(c.sps.scaling_lists.list[1][1], {12,15,18,21,24,27,30,33,19,22,25,28,31,34,37,40,26,29,32,35,38,41,44,6,33,36,39,42,45,7,10,13,40,43,46,8,11,14,17,20,6,9,12,15,18,21,24,27,13,16,19,22,25,28,31,34,20,23,26,29,32,35,38,41});
	fill(c.sps.scaling_lists.list[1][2], {12,15,18,21,24,27,30,33,19,22,25,28,31,34,37,40,26,29,32,35,38,41,44,6,33,36,39,42,45,7,10,13,40,43,46,8,11,14,17,20,6,9,12,15,18,21,24,27,13,16,19,22,25,28,31,34,20,23,26,29,32,35,38,41});
	fill(c.sps.scaling_lists.list[1][3], {8,11,14,17,20,23,26,29,15,18,21,24,27,30,33,36,22,25,28,31,34,37,40,43,29,32,35,38,41,44,6,9,36,39,42,45,7,10,13,16,43,46,8,11,14,17,20,23,9,12,15,18,21,24,27,30,16,19,22,25,28,31,34,37});
	fill(c.sps.scaling_lists.list[1][4], {23,26,29,32,35,38,41,44,30,33,36,39,42,45,7,10,37,40,43,46,8,11,14,17,44,6,9,12,15,18,21,24,10,13,16,19,22,25,28,31,17,20,23,26,29,32,35,38,24,27,30,33,36,39,42,45,31,34,37,40,43,46,8,11});
	fill(c.sps.scaling_lists.list[1][5], {23,26,29,32,35,38,41,44,30,33,36,39,42,45,7,10,37,40,43,46,8,11,14,17,44,6,9,12,15,18,21,24,10,13,16,19,22,25,28,31,17,20,23,26,29,32,35,38,24,27,30,33,36,39,42,45,31,34,37,40,43,46,8,11});
	fill(c.sps.scaling_lists.list[2][0], {15,18,21,24,27,30,33,36,22,25,28,31,34,37,40,43,29,32,35,38,41,44,6,9,36,39,42,45,7,10,13,16,43,46,8,11,14,17,20,23,9,12,15,18,21,24,27,30,16,19,22,25,28,31,34,37,23,26,29,32,35,38,41,44});
	c.sps.scaling_lists.dc[0][0] = 24;
	fill(c.sps.scaling_lists.list[2][1], {30,33,36,39,42,45,7,10,37,40,43,46,8,11,14,17,44,6,9,12,15,18,21,24,10,13,16,19,22,25,28,31,17,20,23,26,29,32,35,38,24,27,30,33,36,39,42,45,31,34,37,40,43,46,8,11,38,41,44,6,9,12,15,18});
	c.sps.scaling_lists.dc[0][1] = 10;
	fill(c.sps.scaling_lists.list[2][2], {30,33,36,39,42,45,7,10,37,40,43,46,8,11,14,17,44,6,9,12,15,18,21,24,10,13,16,19,22,25,28,31,17,20,23,26,29,32,35,38,24,27,30,33,36,39,42,45,31,34,37,40,43,46,8,11,38,41,44,6,9,12,15,18});
	c.sps.scaling_lists.dc[0][2] = 10;
	fill(c.sps.scaling_lists.list[2][3], {26,29,32,35,38,41,44,6,33,36,39,42,45,7,10,13,40,43,46,8,11,14,17,20,6,9,12,15,18,21,24,27,13,16,19,22,25,28,31,34,20,23,26,29,32,35,38,41,27,30,33,36,39,42,45,7,34,37,40,43,46,8,11,14});
	c.sps.scaling_lists.dc[0][3] = 29;
	fill(c.sps.scaling_lists.list[2][4], {41,44,6,9,12,15,18,21,7,10,13,16,19,22,25,28,14,17,20,23,26,29,32,35,21,24,27,30,33,36,39,42,28,31,34,37,40,43,46,8,35,38,41,44,6,9,12,15,42,45,7,10,13,16,19,22,8,11,14,17,20,23,26,29});
	c.sps.scaling_lists.dc[0][4] = 15;
	fill(c.sps.scaling_lists.list[2][5], {41,44,6,9,12,15,18,21,7,10,13,16,19,22,25,28,14,17,20,23,26,29,32,35,21,24,27,30,33,36,39,42,28,31,34,37,40,43,46,8,35,38,41,44,6,9,12,15,42,45,7,10,13,16,19,22,8,11,14,17,20,23,26,29});
	c.sps.scaling_lists.dc[0][5] = 15;
	fill(c.sps.scaling_lists.list[3][0], {31,34,37,40,43,46,8,11,38,41,44,6,9,12,15,18,45,7,10,13,16,19,22,25,11,14,17,20,23,26,29,32,18,21,24,27,30,33,36,39,25,28,31,34,37,40,43,46,32,35,38,41,44,6,9,12,39,42,45,7,10,13,16,19});
	c.sps.scaling_lists.dc[1][0] = 17;
	fill(c.sps.scaling_lists.list[3][1], {42,45,7,10,13,16,19,22,8,11,14,17,20,23,26,29,15,18,21,24,27,30,33,36,22,25,28,31,34,37,40,43,29,32,35,38,41,44,6,9,36,39,42,45,7,10,13,16,43,46,8,11,14,17,20,23,9,12,15,18,21,24,27,30});
	c.sps.scaling_lists.dc[1][1] = 22;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0x6e,0x4f,0x83,0x41,0xc2,0x41,0xe8,0x12,0x84,0x83,0x51,0x22,0x40,0x94,0xd,0x44,0x89,0xb,0x9,0x1d,0x8e,0x12,0x16,0x12,0x24,0x3c,0x24,0x48,0x90,0xf0,0x91,0x21,0x61,0x23,0x8a,0x4,0xa0,0x22,0x89,0xb,0x9,0x12,0x1e,0x12,0x24,0x48,0x78,0x48,0x90,0xb0,0x91,0xd0,0xb0,0xe1,0x21,0x61,0x22,0x43,0xc2,0x44,0x89,0xf,0x9,0x12,0x16,0x12,0x4,0x52,0x83,0xc1,0xc2,0x41,0xe8,0x90,0x25,0x3,0x51,0x22,0x44,0x82,0x61,0x22,0x44,0x89,0x5,0xc2,0x44,0x89,0x12,0x24,0x1b,0x9,0x12,0x24,0x48,0x91,0x20,0xf8,0x48,0x91,0x22,0x44,0x89,0x12,0x15,0x2,0x50,0x91,0x22,0x44,0x89,0xe,0x89,0x2,0x50,0x91,0x22,0x41,0x28,0x91,0x20,0x4a,0x12,0xb,0x44,0x89,0x2,0x50,0x35,0x12,0x24,0x2c,0x24,0x74,0x40,0xe1,0x21,0x61,0x22,0x43,0xc2,0x44,0x89,0x4,0xc2,0x44,0x89,0x12,0x9,0x40,0x94,0x24,0x48,0x91,0x21,0xd1,0x20,0x4a,0x12,0x24,0x48,0x90,0xa8,0x91,0x22,0x40,0x94,0x24,0x48,0x90,0xa8,0x91,0x22,0x44,0x81,0x28,0x48,0x74,0x48,0x91,0x22,0x44,0x82,0xe1,0x22,0x44,0x89,0x4,0xc2,0x44,0x89,0xf,0x9,0x12,0x16,0x12,0x38,0xb1,0xc2,0x42,0xc2,0x44,0x87,0x84,0x89,0x12,0x9,0x84,0x89,0x12,0x24,0x17,0x9,0x12,0x24,0x48,0x90,0xe8,0x12,0x84,0x89,0x12,0x24,0x48,0x54,0x48,0x90,0x25,0x9,0x12,0x24,0x48,0x54,0x48,0x91,0x20,0x4a,0x12,0x24,0x3a,0x24,0x48,0x91,0x20,0x4a,0x4,0xa2,0x44,0x89,0x12,0x9,0x84,0x89,0x12,0x1e,0x12,0x24,0x2c,0x24,0x74,0x3c,0x38,0x48,0x58,0x48,0x90,0xf0,0x91,0x22,0x41,0x68,0x90,0x25,0x9,0x12,0x9,0x44,0x89,0x2,0x50,0x91,0x21,0xd1,0x22,0x44,0x89,0x2,0x50,0x90,0xa8,0x91,0x22,0x44,0x89,0x12,0x4,0xa0,0xa8,0x91,0x22,0x44,0x89,0x12,0xd,0x84,0x89,0x12,0x24,0x48,0x2e,0x12,0x24,0x48,0x90,0x4c,0x24,0x48,0x90,0xf0,0x91,0x20,0xf4,0x9,0x40,0x45,0x28,0x20,0x9,0x8e,0x12,0x16,0x12,0x24,0x3c,0x24,0x48,0x90,0x4c,0x24,0x48,0x91,0x20,0x94,0x9,0x42,0x44,0x89,0x12,0x1d,0x12,0x24,0x9,0x42,0x44,0x89,0xa,0x89,0x12,0x24,0x48,0x12,0x84,0x89,0xa,0x89,0x12,0x24,0x48,0x90,0x25,0x7,0x44,0x89,0x12,0x24,0x48,0x2e,0x12,0x24,0x48,0x90,0x4c,0x24,0x48,0x90,0xf0,0x91,0x21,0x61,0x23,0xa4,0x5,0x3,0x84,0x85,0x84,0x89,0x6,0xa2,0x40,0x94,0x24,0x16,0x89,0x12,0x4,0xa1,0x20,0x94,0x48,0x91,0x22,0x40,0x94,0x1d,0x12,0x24,0x48,0x91,0x22,0x41,0xf0,0x91,0x22,0x44,0x89,0x12,0x24,0x1f,0x9,0x12,0x24,0x48,0x91,0x20,0xd8,0x48,0x91,0x22,0x44,0x82,0x50,0x25,0x9,0x12,0x24,0x16,0x81,0x28,0x48,0x90,0x6a,0x24,0x9,0x40,0xf4,0x48,0xe2,0x82,0xa3,0x8e,0x12,0x16,0x12,0x24,0x1a,0x81,0x28,0x48,0x90,0x5a,0x24,0x9,0x42,0x44,0x82,0x51,0x22,0x44,0x81,0x28,0x48,0x74,0x48,0x91,0x22,0x44,0x81,0x28,0x2a,0x24,0x48,0x91,0x22,0x44,0x89,0x7,0xc2,0x44,0x89,0x12,0x24,0x48,0x36,0x12,0x24,0x48,0x91,0x20,0xb8,0x48,0x91,0x22,0x41,0x30,0x91,0x22,0x41,0xa8,0x12,0x84,0x83,0xd1,0x23,0xa3,0x81,0xa0,0x11,0x40,0x94,0xf,0x44,0x89,0xf,0x9,0x12,0x24,0x13,0x9,0x12,0x24,0x48,0x2e,0x12,0x24,0x48,0x91,0x20,0xd8,0x48,0x91,0x22,0x44,0x89,0xa,0x81,0x28,0x48,0x91,0x22,0x44,0x89,0xa,0x89,0x2,0x50,0x91,0x22,0x44,0x87,0x44,0x89,0x2,0x50,0x91,0x20,0x94,0x48,0x90,0x25,0x9,0x5,0xa2,0x44,0x89,0xf,0x9,0x12,0x16,0x12,0x38,0xa1,0x20,0xe0,0xe1,0x21,0x61,0x22,0x41,0xa8,0x90,0x25,0x9,0x5,0xa2,0x44,0x89,0x2,0x50,0x25,0x12,0x24,0x48,0x90,0x25,0x7,0x44,0x89,0x12,0x24,0x48,0x90,0x7c,0x24,0x48,0x91,0x22,0x44,0x89,0x7,0xc2,0x44,0x89,0x12,0x24,0x48,0x36,0x12,0x24,0x48,0x91,0x20,0x94,0x9,0x42,0x44,0x89,0x5,0xa2,0x40,0x94,0x24,0x1a,0x89,0x2,0x50,0x3d,0x12,0x3a,0x1c,0x5,0x0,0x45,0x2,0x50,0x3d,0x12,0x24,0x3c,0x24,0x48,0x90,0x4c,0x24,0x48,0x91,0x20,0xb8,0x48,0x91,0x22,0x44,0x83,0x61,0x22,0x44,0x89,0x12,0x24,0x2a,0x4,0xa1,0x22,0x44,0x89,0x12,0x24,0x2a,0x24,0x9,0x42,0x44,0x89,0x12,0x1d,0x12,0x24,0x9,0x42,0x44,0x82,0x51,0x22,0x44,0x81,0x28,0x16,0x89,0x12,0x24,0x3c,0x24,0x48,0x58,0x48,0xe5,0x78,0x8,0x8,0x0,0x0,0x3,0x0,0x8,0x0,0x0,0x3,0x0,0xf0,0x40};
	c.pps_nal = {0x44,0x1,0xc1,0x73,0xd1,0x89};
	c.sps_vui_bit = 5995;
	c.sps_sets_bit = 5992;
	c.sps_sets_end_bit = 5992;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_720p_sets_sps()
{
	Case c = {};
	c.name = "hevc-main-720p-sets-sps";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 5;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.num_short_term_ref_pic_sets = 25;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	rps(c.sps.st_ref_pic_set[0], {}, {}, {}, {});
	rps(c.sps.st_ref_pic_set[1], {-5}, {1}, {}, {});
	rps(c.sps.st_ref_pic_set[2], {-3}, {1}, {2}, {1});
	rps(c.sps.st_ref_pic_set[3], {-1}, {1}, {2,4}, {1,1});
	rps(c.sps.st_ref_pic_set[4], {-2}, {1}, {1,3}, {1,1});
	rps(c.sps.st_ref_pic_set[5], {-1,-4}, {1,1}, {1}, {1});
	rps(c.sps.st_ref_pic_set[6], {-5,-7,-10}, {1,1,1}, {}, {});
	rps(c.sps.st_ref_pic_set[7], {-3,-5,-8}, {1,1,1}, {2}, {1});
	rps(c.sps.st_ref_pic_set[8], {-1,-3}, {1,1}, {2,4}, {1,1});
	rps(c.sps.st_ref_pic_set[9], {-2,-4}, {1,1}, {1,3}, {1,1});
	rps(c.sps.st_ref_pic_set[10], {-1,-4,-6}, {1,1,1}, {1}, {1});
	rps(c.sps.st_ref_pic_set[11], {-5,-7,-10,-12}, {1,1,1,1}, {}, {});
	rps(c.sps.st_ref_pic_set[12], {-3,-5,-10}, {1,1,1}, {2}, {1});
	rps(c.sps.st_ref_pic_set[13], {-3,-5,-8,-10}, {1,1,1,1}, {}, {});
	rps(c.sps.st_ref_pic_set[14], {-2,-4,-9}, {1,1,1}, {1}, {1});
	rps(c.sps.st_ref_pic_set[15], {-1,-3}, {1,1}, {1,2}, {1,1});
	rps(c.sps.st_ref_pic_set[16], {-5,-6,-8,-10}, {0,0,0,0}, {}, {});
	rps(c.sps.st_ref_pic_set[17], {-3,-4,-8}, {1,1,1}, {2}, {1});
	rps(c.sps.st_ref_pic_set[18], {-1,-2}, {1,1}, {2,4}, {1,1});
	rps(c.sps.st_ref_pic_set[19], {-2,-3}, {1,1}, {1,3}, {1,1});
	rps(c.sps.st_ref_pic_set[20], {-1,-4,-5}, {1,1,1}, {1}, {1});
	rps(c.sps.st_ref_pic_set[21], {-5,-6,-8,-10}, {1,1,1,1}, {}, {});
	rps(c.sps.st_ref_pic_set[22], {-5,-7,-10,-11}, {1,1,1,1}, {}, {});
	rps(c.sps.st_ref_pic_set[23], {-3,-5,-9}, {1,1,1}, {2}, {1});
	rps(c.sps.st_ref_pic_set[24], {-5,-7,-10,-12}, {0,0,0,0}, {}, {});
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0xa4,0x93,0x20,0xd7,0x97,0x2e,0x5f,0xf3,0x5c,0xcd,0xc9,0x7c,0xb3,0xfe,0x6b,0xcc,0xde,0x4b,0x3c,0xb9,0xdc,0xee,0x79,0xd5,0x49,0x67,0x2c,0xff,0x9a,0xf0,0x59,0x7a,0xa8,0xb2,0xd5,0xf9,0x67,0x9d,0x45,0x78,0x8,0x8,0x0,0x0,0x3,0x0,0x8,0x0,0x0,0x3,0x0,0xf0,0x40};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x62,0x40};
	c.sps_vui_bit = 499;
	c.sps_sets_bit = 213;
	c.sps_sets_end_bit = 496;
	c.sps_sets_predicted = true;
	return c;
}

Case case_hevc_main_720p_ref1_lt()
{
	Case c = {};
	c.name = "hevc-main-720p-ref1-lt";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 1;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.num_short_term_ref_pic_sets = 1;
	c.sps.long_term_ref_pics_present_flag = 1;
	c.sps.num_long_term_ref_pics_sps = 16;
	c.sps.used_by_curr_pic_lt_sps_flag[0] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[1] = 1;
	c.sps.used_by_curr_pic_lt_sps_flag[1] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[2] = 2;
	c.sps.used_by_curr_pic_lt_sps_flag[2] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[3] = 3;
	c.sps.used_by_curr_pic_lt_sps_flag[3] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[4] = 4;
	c.sps.used_by_curr_pic_lt_sps_flag[4] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[5] = 5;
	c.sps.used_by_curr_pic_lt_sps_flag[5] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[6] = 6;
	c.sps.used_by_curr_pic_lt_sps_flag[6] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[7] = 7;
	c.sps.used_by_curr_pic_lt_sps_flag[7] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[8] = 8;
	c.sps.used_by_curr_pic_lt_sps_flag[8] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[9] = 9;
	c.sps.used_by_curr_pic_lt_sps_flag[9] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[10] = 10;
	c.sps.used_by_curr_pic_lt_sps_flag[10] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[11] = 11;
	c.sps.used_by_curr_pic_lt_sps_flag[11] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[12] = 12;
	c.sps.used_by_curr_pic_lt_sps_flag[12] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[13] = 13;
	c.sps.used_by_curr_pic_lt_sps_flag[13] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[14] = 14;
	c.sps.used_by_curr_pic_lt_sps_flag[14] = 1;
	c.sps.lt_ref_pic_poc_lsb_sps[15] = 15;
	c.sps.used_by_curr_pic_lt_sps_flag[15] = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	rps(c.sps.st_ref_pic_set[0], {}, {}, {}, {});
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.pps.lists_modification_present_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x5b,0xa9,0x24,0xc9,0x70,0x88,0x4,0x6,0x5,0x3,0x82,0x41,0x60,0xd0,0x78,0x44,0x26,0x15,0xb,0x86,0x43,0x61,0xd0,0xfb,0x80,0x80,0x80,0x0,0x0,0x3,0x0,0x80,0x0,0x0,0xf,0x4};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x66,0x40};
	c.sps_vui_bit = 359;
	c.sps_sets_bit = 201;
	c.sps_sets_end_bit = 203;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_720p_lists()
{
	Case c = {};
	c.name = "hevc-main-720p-lists";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 5;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.pps.lists_modification_present_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0xa4,0x93,0x2b,0xc0,0x40,0x40,0x0,0x0,0x3,0x0,0x40,0x0,0x0,0x7,0x82};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x66,0x40};
	c.sps_vui_bit = 208;
	c.sps_sets_bit = 205;
	c.sps_sets_end_bit = 205;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_720p_temporal()
{
	Case c = {};
	c.name = "hevc-main-720p-temporal";
	c.sps.sps_max_sub_layers_minus1 = 2;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 4;
	c.sps.sps_max_dec_pic_buffering_minus1[1] = 4;
	c.sps.sps_max_num_reorder_pics[1] = 2;
	c.sps.sps_max_latency_increase_plus1[1] = 4;
	c.sps.sps_max_dec_pic_buffering_minus1[2] = 4;
	c.sps.sps_max_num_reorder_pics[2] = 2;
	c.sps.sps_max_latency_increase_plus1[2] = 4;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x4,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0x0,0x0,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0x4a,0xca,0x56,0x59,0x24,0xca,0xf0,0x10,0x10,0x0,0x0,0x3,0x0,0x10,0x0,0x0,0x3,0x1,0xe0,0x80};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb4,0x62,0x40};
	c.sps_vui_bit = 250;
	c.sps_sets_bit = 247;
	c.sps_sets_end_bit = 247;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_360p_lossless()
{
	Case c = {};
	c.name = "hevc-main-360p-lossless";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 255;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 640;
	c.sps.pic_height_in_luma_samples = 360;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 3;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.transquant_bypass_enabled_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0xff,0xa0,0x5,0x2,0x1,0x69,0x65,0x95,0x92,0x49,0x32,0xbc,0x4,0x4,0x0,0x0,0x3,0x0,0x4,0x0,0x0,0x3,0x0,0x78,0x20};
	c.pps_nal = {0x44,0x1,0xc1,0x71,0xab,0x12};
	c.sps_vui_bit = 204;
	c.sps_sets_bit = 201;
	c.sps_sets_end_bit = 201;
	c.sps_sets_predicted = false;
	return c;
}

Case case_hevc_main_720p_fade()
{
	Case c = {};
	c.name = "hevc-main-720p-fade";
	c.sps.sps_temporal_id_nesting_flag = 1;
	c.sps.general_profile_idc = 1;
	c.sps.general_profile_compatibility_flag[1] = 1;
	c.sps.general_profile_compatibility_flag[2] = 1;
	c.sps.general_progressive_source_flag = 1;
	c.sps.general_frame_only_constraint_flag = 1;
	c.sps.general_level_idc = 93;
	c.sps.chroma_format_idc = 1;
	c.sps.pic_width_in_luma_samples = 1280;
	c.sps.pic_height_in_luma_samples = 720;
	c.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	c.sps.sps_sub_layer_ordering_info_present_flag = 1;
	c.sps.sps_max_dec_pic_buffering_minus1[0] = 4;
	c.sps.sps_max_num_reorder_pics[0] = 2;
	c.sps.sps_max_latency_increase_plus1[0] = 4;
	c.sps.log2_diff_max_min_luma_coding_block_size = 3;
	c.sps.log2_diff_max_min_luma_transform_block_size = 3;
	c.sps.sample_adaptive_offset_enabled_flag = 1;
	c.sps.sps_temporal_mvp_enabled_flag = 1;
	c.sps.strong_intra_smoothing_enabled_flag = 1;
	c.pps.sign_data_hiding_enabled_flag = 1;
	c.pps.cu_qp_delta_enabled_flag = 1;
	c.pps.diff_cu_qp_delta_depth = 1;
	c.pps.weighted_pred_flag = 1;
	c.pps.weighted_bipred_flag = 1;
	c.pps.entropy_coding_sync_enabled_flag = 1;
	c.pps.pps_loop_filter_across_slices_enabled_flag = 1;
	c.sps_nal = {0x42,0x1,0x1,0x1,0x60,0x0,0x0,0x3,0x0,0x90,0x0,0x0,0x3,0x0,0x0,0x3,0x0,0x5d,0xa0,0x2,0x80,0x80,0x2d,0x16,0x59,0x59,0x64,0x93,0x2b,0xc0,0x40,0x40,0x0,0x0,0x3,0x0,0x40,0x0,0x0,0x7,0x82};
	c.pps_nal = {0x44,0x1,0xc1,0x72,0xb6,0x62,0x40};
	c.sps_vui_bit = 208;
	c.sps_sets_bit = 205;
	c.sps_sets_end_bit = 205;
	c.sps_sets_predicted = false;
	return c;
}

const std::vector<Case (*)()> kCases = {case_hevc_main_720p, case_hevc_main_1080p, case_hevc_main10_1080p, case_hevc_main_1366x766, case_hevc_main_720p_cqm, case_hevc_main_720p_sets_sps, case_hevc_main_720p_ref1_lt, case_hevc_main_720p_lists, case_hevc_main_720p_temporal, case_hevc_main_360p_lossless, case_hevc_main_720p_fade};

#endif
