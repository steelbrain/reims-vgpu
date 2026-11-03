/*
 * RgfxVideoH265: HEVC parameter sets for the host's decoder, made from AppleGVA's.
 *
 * As for H.264 (RgfxVideoH264.hpp), AppleGVA parses a stream's SPS and PPS itself and hands the
 * video driver each picture's slice segments as Annex-B NAL units, with the parameter sets only
 * in its own parsed form: a field per syntax element in the syntax's order, flags a byte each and
 * the rest 32-bit words, with each array at its largest size and the lists the syntax derives
 * (each scaling list, each short-term reference picture set's pictures) as the derivation leaves
 * them. The driver writes the SPS and PPS back out as NAL units from those fields: the SPS
 * without its VUI, which holds nothing a decoder needs, and the PPS whole. The host makes the VPS
 * from the SPS, so none is sent.
 *
 * What a set derives is written in the syntax's own explicit form, which decodes the same: each
 * short-term reference picture set by its pictures, each scaling list as the default where it
 * equals it, else as a copy of the nearest earlier list it equals, else coefficient by coefficient.
 *
 * Every field below is placed by HEVC's syntax order between fields whose meaning the calls' use
 * states; a field whose meaning nothing states is field_xxx, xxx its byte offset.
 */

#ifndef RGFX_VIDEO_H265_HPP_
#define RGFX_VIDEO_H265_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgfx::video::h265 {

// A scaling list set (7.3.4), as its derivation leaves it (7.4.5): the lists of each size (4x4,
// 8x8, 16x16, 32x32), each a square of 4x4 (16) or 8x8 (64) coefficients in raster order, a row
// after another, and the DC coefficients of the 16x16 and 32x32 lists. The 32x32 lists are the
// luma ones, intra and inter, at matrices 0 and 1.
struct ScalingLists {
	uint8_t field_000[0xac];
	uint32_t list[4][6][64];
	uint32_t dc[2][6];
};
static_assert(offsetof(ScalingLists, list) == 0xac, "scaling list layout");
static_assert(offsetof(ScalingLists, dc) == 0x18ac, "scaling list layout");
static_assert(sizeof(ScalingLists) == 0x18dc, "scaling list layout");

// A short-term reference picture set (7.3.7), as its derivation leaves it (7.4.8): the pictures
// before the current one (S0) and after it (S1), each its order count's distance from the
// current one's, nearest first, and whether the current picture uses it. AppleGVA holds a set
// the SPS predicts from another (inter_ref_pic_set_prediction_flag) by its pictures too, derived,
// after the prediction's syntax (measured on the guest with hevc-main-720p-sets-sps: each of its
// 25 sets, 22 of them predicted, holds the pictures 7-61 and 7-62 give), so the writer writes
// every set by its pictures.
struct ShortTermRps {
	uint8_t field_000[0xdc];
	uint32_t num_negative_pics;
	uint32_t num_positive_pics;
	uint32_t num_delta_pocs;
	uint8_t used_by_curr_pic_s0[16];
	uint8_t used_by_curr_pic_s1[16];
	int32_t delta_poc_s0[16];
	int32_t delta_poc_s1[16];
};
static_assert(offsetof(ShortTermRps, num_negative_pics) == 0xdc, "RPS layout");
static_assert(offsetof(ShortTermRps, num_delta_pocs) == 0xe4, "RPS layout");
static_assert(offsetof(ShortTermRps, used_by_curr_pic_s0) == 0xe8, "RPS layout");
static_assert(offsetof(ShortTermRps, used_by_curr_pic_s1) == 0xf8, "RPS layout");
static_assert(offsetof(ShortTermRps, delta_poc_s0) == 0x108, "RPS layout");
static_assert(offsetof(ShortTermRps, delta_poc_s1) == 0x148, "RPS layout");
static_assert(sizeof(ShortTermRps) == 0x188, "RPS layout");

// AppleGVA's parsed SPS, as far as the decoder reads it (its VUI follows).
struct Sps {
	uint32_t sps_video_parameter_set_id;
	uint32_t sps_max_sub_layers_minus1;
	uint8_t sps_temporal_id_nesting_flag;
	// profile_tier_level(1, sps_max_sub_layers_minus1)
	uint32_t general_profile_space;
	uint8_t general_tier_flag;
	uint32_t general_profile_idc;
	uint8_t general_profile_compatibility_flag[32];
	uint8_t general_progressive_source_flag;
	uint8_t general_interlaced_source_flag;
	uint8_t general_non_packed_constraint_flag;
	uint8_t general_frame_only_constraint_flag;
	uint8_t field_03c[0x6c - 0x3c];
	uint32_t general_level_idc;
	uint8_t field_070[0xe04 - 0x70];
	uint32_t sps_seq_parameter_set_id;
	uint32_t chroma_format_idc;
	uint8_t separate_colour_plane_flag;
	uint32_t pic_width_in_luma_samples;
	uint32_t pic_height_in_luma_samples;
	uint8_t conformance_window_flag;
	uint32_t conf_win_left_offset;
	uint32_t conf_win_right_offset;
	uint32_t conf_win_top_offset;
	uint32_t conf_win_bottom_offset;
	uint32_t bit_depth_luma_minus8;
	uint32_t bit_depth_chroma_minus8;
	uint32_t log2_max_pic_order_cnt_lsb_minus4;
	uint8_t sps_sub_layer_ordering_info_present_flag;
	uint32_t sps_max_dec_pic_buffering_minus1[64];
	uint32_t sps_max_num_reorder_pics[64];
	uint32_t sps_max_latency_increase_plus1[64];
	uint32_t log2_min_luma_coding_block_size_minus3;
	uint32_t log2_diff_max_min_luma_coding_block_size;
	uint32_t log2_min_luma_transform_block_size_minus2;
	uint32_t log2_diff_max_min_luma_transform_block_size;
	uint32_t max_transform_hierarchy_depth_inter;
	uint32_t max_transform_hierarchy_depth_intra;
	uint8_t scaling_list_enabled_flag;
	uint8_t sps_scaling_list_data_present_flag;
	ScalingLists scaling_lists;
	uint8_t amp_enabled_flag;
	uint8_t sample_adaptive_offset_enabled_flag;
	uint8_t pcm_enabled_flag;
	uint32_t pcm_sample_bit_depth_luma_minus1;
	uint32_t pcm_sample_bit_depth_chroma_minus1;
	uint32_t log2_min_pcm_luma_coding_block_size_minus3;
	uint32_t log2_diff_max_min_pcm_luma_coding_block_size;
	uint8_t pcm_loop_filter_disabled_flag;
	uint32_t num_short_term_ref_pic_sets;
	// The SPS's 64, then a slice header's own.
	ShortTermRps st_ref_pic_set[65];
	uint8_t long_term_ref_pics_present_flag;
	uint32_t num_long_term_ref_pics_sps;
	uint32_t lt_ref_pic_poc_lsb_sps[32];
	uint8_t used_by_curr_pic_lt_sps_flag[32];
	uint8_t sps_temporal_mvp_enabled_flag;
	uint8_t strong_intra_smoothing_enabled_flag;
};
static_assert(offsetof(Sps, sps_max_sub_layers_minus1) == 0x4, "SPS layout");
static_assert(offsetof(Sps, general_profile_idc) == 0x14, "SPS layout");
static_assert(offsetof(Sps, general_profile_compatibility_flag) == 0x18, "SPS layout");
static_assert(offsetof(Sps, chroma_format_idc) == 0xe08, "SPS layout");
static_assert(offsetof(Sps, separate_colour_plane_flag) == 0xe0c, "SPS layout");
static_assert(offsetof(Sps, pic_width_in_luma_samples) == 0xe10, "SPS layout");
static_assert(offsetof(Sps, bit_depth_luma_minus8) == 0xe2c, "SPS layout");
static_assert(offsetof(Sps, log2_max_pic_order_cnt_lsb_minus4) == 0xe34, "SPS layout");
static_assert(offsetof(Sps, sps_max_dec_pic_buffering_minus1) == 0xe3c, "SPS layout");
static_assert(offsetof(Sps, log2_min_luma_coding_block_size_minus3) == 0x113c, "SPS layout");
static_assert(offsetof(Sps, max_transform_hierarchy_depth_intra) == 0x1150, "SPS layout");
static_assert(offsetof(Sps, scaling_list_enabled_flag) == 0x1154, "SPS layout");
static_assert(offsetof(Sps, scaling_lists) == 0x1158, "SPS layout");
static_assert(offsetof(Sps, amp_enabled_flag) == 0x2a34, "SPS layout");
static_assert(offsetof(Sps, pcm_sample_bit_depth_luma_minus1) == 0x2a38, "SPS layout");
static_assert(offsetof(Sps, pcm_loop_filter_disabled_flag) == 0x2a48, "SPS layout");
static_assert(offsetof(Sps, num_short_term_ref_pic_sets) == 0x2a4c, "SPS layout");
static_assert(offsetof(Sps, st_ref_pic_set) == 0x2a50, "SPS layout");
static_assert(offsetof(Sps, long_term_ref_pics_present_flag) == 0x8dd8, "SPS layout");
static_assert(offsetof(Sps, num_long_term_ref_pics_sps) == 0x8ddc, "SPS layout");
static_assert(offsetof(Sps, sps_temporal_mvp_enabled_flag) == 0x8e80, "SPS layout");
static_assert(offsetof(Sps, strong_intra_smoothing_enabled_flag) == 0x8e81, "SPS layout");

// AppleGVA's parsed PPS.
struct Pps {
	uint32_t pps_pic_parameter_set_id;
	uint32_t pps_seq_parameter_set_id;
	uint8_t dependent_slice_segments_enabled_flag;
	uint8_t output_flag_present_flag;
	uint32_t num_extra_slice_header_bits;
	uint8_t sign_data_hiding_enabled_flag;
	uint8_t cabac_init_present_flag;
	uint32_t num_ref_idx_l0_default_active_minus1;
	uint32_t num_ref_idx_l1_default_active_minus1;
	int32_t init_qp_minus26;
	uint8_t constrained_intra_pred_flag;
	uint8_t transform_skip_enabled_flag;
	uint8_t cu_qp_delta_enabled_flag;
	uint32_t diff_cu_qp_delta_depth;
	int32_t pps_cb_qp_offset;
	int32_t pps_cr_qp_offset;
	uint8_t pps_slice_chroma_qp_offsets_present_flag;
	uint8_t weighted_pred_flag;
	uint8_t weighted_bipred_flag;
	uint8_t transquant_bypass_enabled_flag;
	uint8_t tiles_enabled_flag;
	uint8_t entropy_coding_sync_enabled_flag;
	uint32_t num_tile_columns_minus1;
	uint32_t num_tile_rows_minus1;
	uint8_t uniform_spacing_flag;
	uint32_t column_width_minus1[20];
	uint32_t row_height_minus1[22];
	uint8_t loop_filter_across_tiles_enabled_flag;
	uint8_t pps_loop_filter_across_slices_enabled_flag;
	uint8_t deblocking_filter_control_present_flag;
	uint8_t deblocking_filter_override_enabled_flag;
	uint8_t pps_deblocking_filter_disabled_flag;
	int32_t pps_beta_offset_div2;
	int32_t pps_tc_offset_div2;
	uint8_t pps_scaling_list_data_present_flag;
	ScalingLists scaling_lists;
	uint8_t lists_modification_present_flag;
	uint32_t log2_parallel_merge_level_minus2;
	uint8_t slice_segment_header_extension_present_flag;
};
static_assert(offsetof(Pps, num_extra_slice_header_bits) == 0xc, "PPS layout");
static_assert(offsetof(Pps, init_qp_minus26) == 0x1c, "PPS layout");
static_assert(offsetof(Pps, diff_cu_qp_delta_depth) == 0x24, "PPS layout");
static_assert(offsetof(Pps, pps_slice_chroma_qp_offsets_present_flag) == 0x30, "PPS layout");
static_assert(offsetof(Pps, num_tile_columns_minus1) == 0x38, "PPS layout");
static_assert(offsetof(Pps, uniform_spacing_flag) == 0x40, "PPS layout");
static_assert(offsetof(Pps, column_width_minus1) == 0x44, "PPS layout");
static_assert(offsetof(Pps, row_height_minus1) == 0x94, "PPS layout");
static_assert(offsetof(Pps, loop_filter_across_tiles_enabled_flag) == 0xec, "PPS layout");
static_assert(offsetof(Pps, deblocking_filter_override_enabled_flag) == 0xef, "PPS layout");
static_assert(offsetof(Pps, pps_beta_offset_div2) == 0xf4, "PPS layout");
static_assert(offsetof(Pps, pps_scaling_list_data_present_flag) == 0xfc, "PPS layout");
static_assert(offsetof(Pps, scaling_lists) == 0x100, "PPS layout");
static_assert(offsetof(Pps, lists_modification_present_flag) == 0x19dc, "PPS layout");
static_assert(offsetof(Pps, log2_parallel_merge_level_minus2) == 0x19e0, "PPS layout");
static_assert(offsetof(Pps, slice_segment_header_extension_present_flag) == 0x19e4, "PPS layout");

// Appends the SPS (without its VUI) and the PPS, each as an Annex-B NAL unit with a four-byte
// start code, to out. False, and out as it was, where a field is out of HEVC's range or the
// stream is of a format the decoder does not take (not 4:2:0, more than 10 bits).
bool writeSps(const Sps &sps, std::vector<uint8_t> &out);
bool writePps(const Pps &pps, std::vector<uint8_t> &out);

} // namespace rgfx::video::h265

#endif
