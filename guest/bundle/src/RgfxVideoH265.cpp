/*
 * RgfxVideoH265: HEVC parameter sets for the host's decoder, made from AppleGVA's
 * (RgfxVideoH265.hpp).
 */

#include "RgfxVideoH265.hpp"

#include "RgfxVideoBits.hpp"

namespace rgfx::video::h265 {
namespace {

// The default 8x8 lists (Table 7-6), intra then inter, in coded (up-right diagonal) order; every
// 4x4 default is 16 throughout.
constexpr uint8_t kDefaultIntra[64] = {
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21,
	19, 20, 21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29,
	31, 35, 35, 31, 29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
};
constexpr uint8_t kDefaultInter[64] = {
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20,
	20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28,
	28, 28, 28, 28, 28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
};

// The raster index of each position of the up-right diagonal scan of a square of `n` (6.5.3).
template <uint32_t N> struct DiagonalScan {
	uint8_t raster[N * N];
	constexpr DiagonalScan() : raster()
	{
		uint32_t i = 0;
		int32_t x = 0, y = 0;
		while (i < N * N) {
			while (y >= 0) {
				if (x < (int32_t)N && y < (int32_t)N)
					raster[i++] = (uint8_t)(y * N + x);
				y--;
				x++;
			}
			y = x;
			x = 0;
		}
	}
};
constexpr DiagonalScan<4> kScan4;
constexpr DiagonalScan<8> kScan8;

// The matrices of size `sizeId`: six, but two of the 32x32 lists at 4:2:0.
constexpr uint32_t matrices(uint32_t sizeId) { return sizeId == 3 ? 2 : 6; }

// The coefficients of `lists`' matrix `matrixId` of `sizeId`, in coded order, `count` of them.
void coded(const ScalingLists &lists, uint32_t sizeId, uint32_t matrixId, uint32_t *out,
           uint32_t *count)
{
	const uint32_t *list = lists.list[sizeId][matrixId];
	*count = sizeId == 0 ? 16 : 64;
	for (uint32_t i = 0; i < *count; i++)
		out[i] = list[sizeId == 0 ? kScan4.raster[i] : kScan8.raster[i]];
}

// The default coefficient of a matrix of `sizeId`, `matrixId` at coded position `i`; the 32x32
// inter list is matrix 1 at 4:2:0.
uint32_t defaultCoefficient(uint32_t sizeId, uint32_t matrixId, uint32_t i)
{
	if (sizeId == 0)
		return 16;
	const bool inter = sizeId == 3 ? matrixId == 1 : matrixId >= 3;
	return inter ? kDefaultInter[i] : kDefaultIntra[i];
}

// Whether matrix `matrixId` of `sizeId` equals `refMatrixId`'s, its DC included, or the default
// where they are the same matrix.
bool predicted(const ScalingLists &lists, uint32_t sizeId, uint32_t matrixId,
               uint32_t refMatrixId)
{
	uint32_t a[64], b[64], n = 0;
	coded(lists, sizeId, matrixId, a, &n);
	if (refMatrixId == matrixId) {
		for (uint32_t i = 0; i < n; i++)
			if (a[i] != defaultCoefficient(sizeId, matrixId, i))
				return false;
		return sizeId < 2 || lists.dc[sizeId - 2][matrixId] == 16;
	}
	coded(lists, sizeId, refMatrixId, b, &n);
	for (uint32_t i = 0; i < n; i++)
		if (a[i] != b[i])
			return false;
	return sizeId < 2 || lists.dc[sizeId - 2][matrixId] == lists.dc[sizeId - 2][refMatrixId];
}

// scaling_list_data() (7.3.4): each matrix as the default, or a copy of the nearest earlier one,
// where it equals it, else coefficient by coefficient; false where a coefficient is out of
// range.
bool scalingListData(BitWriter &w, const ScalingLists &lists)
{
	for (uint32_t sizeId = 0; sizeId < 4; sizeId++)
		for (uint32_t matrixId = 0; matrixId < matrices(sizeId); matrixId++) {
			uint32_t ref = matrixId + 1;
			for (uint32_t r = matrixId + 1; r-- > 0;)
				if (predicted(lists, sizeId, matrixId, r)) {
					ref = r;
					break;
				}
			if (ref <= matrixId) {
				w.flag(0); // scaling_list_pred_mode_flag
				w.ue(matrixId - ref);
				continue;
			}
			w.flag(1);
			uint32_t list[64], n = 0;
			coded(lists, sizeId, matrixId, list, &n);
			int32_t next = 8;
			if (sizeId > 1) {
				const uint32_t dc = lists.dc[sizeId - 2][matrixId];
				if (dc < 1 || dc > 255)
					return false;
				w.se((int32_t)dc - 8);
				next = (int32_t)dc;
			}
			for (uint32_t i = 0; i < n; i++) {
				if (list[i] < 1 || list[i] > 255)
					return false;
				int32_t delta = (int32_t)list[i] - next;
				if (delta > 127)
					delta -= 256;
				if (delta < -128)
					delta += 256;
				w.se(delta);
				next = (int32_t)list[i];
			}
		}
	return true;
}

// st_ref_pic_set(idx) (7.3.7) of `rps`, written by its pictures: without prediction from another
// set.
bool shortTermRps(BitWriter &w, const ShortTermRps &rps, uint32_t idx)
{
	if (rps.num_negative_pics > 16 || rps.num_positive_pics > 16 ||
	    rps.num_negative_pics + rps.num_positive_pics > 16)
		return false;
	if (idx != 0)
		w.flag(0); // inter_ref_pic_set_prediction_flag
	w.ue(rps.num_negative_pics);
	w.ue(rps.num_positive_pics);
	int32_t last = 0;
	for (uint32_t i = 0; i < rps.num_negative_pics; i++) {
		const int32_t delta = last - rps.delta_poc_s0[i];
		if (delta < 1)
			return false;
		w.ue((uint32_t)(delta - 1));
		w.flag(rps.used_by_curr_pic_s0[i]);
		last = rps.delta_poc_s0[i];
	}
	last = 0;
	for (uint32_t i = 0; i < rps.num_positive_pics; i++) {
		const int32_t delta = rps.delta_poc_s1[i] - last;
		if (delta < 1)
			return false;
		w.ue((uint32_t)(delta - 1));
		w.flag(rps.used_by_curr_pic_s1[i]);
		last = rps.delta_poc_s1[i];
	}
	return true;
}


// profile_tier_level(1, maxNumSubLayersMinus1) (7.3.3): the general profile, tier and level,
// and no sub-layer's.
void profileTierLevel(BitWriter &w, const Sps &s)
{
	w.u(2, s.general_profile_space);
	w.flag(s.general_tier_flag);
	w.u(5, s.general_profile_idc);
	for (uint8_t flag : s.general_profile_compatibility_flag)
		w.flag(flag);
	w.flag(s.general_progressive_source_flag);
	w.flag(s.general_interlaced_source_flag);
	w.flag(s.general_non_packed_constraint_flag);
	w.flag(s.general_frame_only_constraint_flag);
	w.u(32, 0); // the reserved and RExt constraint flags, 43 bits, of Main and Main 10
	w.u(11, 0);
	w.flag(0); // general_inbld_flag
	w.u(8, s.general_level_idc);
	for (uint32_t i = 0; i < s.sps_max_sub_layers_minus1; i++) {
		w.flag(0); // sub_layer_profile_present_flag
		w.flag(0); // sub_layer_level_present_flag
	}
	if (s.sps_max_sub_layers_minus1 > 0)
		for (uint32_t i = s.sps_max_sub_layers_minus1; i < 8; i++)
			w.u(2, 0); // reserved_zero_2bits
}

} // namespace

bool writeSps(const Sps &s, std::vector<uint8_t> &out)
{
	if (s.sps_video_parameter_set_id > 15 || s.sps_max_sub_layers_minus1 > 6 ||
	    s.general_profile_space > 3 || s.general_profile_idc > 31 || s.general_level_idc > 255 ||
	    s.sps_seq_parameter_set_id > 15 || s.chroma_format_idc != 1 ||
	    s.bit_depth_luma_minus8 > 2 || s.bit_depth_chroma_minus8 > 2 ||
	    s.log2_max_pic_order_cnt_lsb_minus4 > 12 || s.num_short_term_ref_pic_sets > 64 ||
	    s.num_long_term_ref_pics_sps > 32 || s.pcm_sample_bit_depth_luma_minus1 > 15 ||
	    s.pcm_sample_bit_depth_chroma_minus1 > 15)
		return false;
	BitWriter w;
	w.u(4, s.sps_video_parameter_set_id);
	w.u(3, s.sps_max_sub_layers_minus1);
	w.flag(s.sps_temporal_id_nesting_flag);
	profileTierLevel(w, s);
	w.ue(s.sps_seq_parameter_set_id);
	w.ue(s.chroma_format_idc);
	w.ue(s.pic_width_in_luma_samples);
	w.ue(s.pic_height_in_luma_samples);
	w.flag(s.conformance_window_flag);
	if (s.conformance_window_flag) {
		w.ue(s.conf_win_left_offset);
		w.ue(s.conf_win_right_offset);
		w.ue(s.conf_win_top_offset);
		w.ue(s.conf_win_bottom_offset);
	}
	w.ue(s.bit_depth_luma_minus8);
	w.ue(s.bit_depth_chroma_minus8);
	w.ue(s.log2_max_pic_order_cnt_lsb_minus4);
	w.flag(s.sps_sub_layer_ordering_info_present_flag);
	for (uint32_t i = s.sps_sub_layer_ordering_info_present_flag ? 0 : s.sps_max_sub_layers_minus1;
	     i <= s.sps_max_sub_layers_minus1; i++) {
		w.ue(s.sps_max_dec_pic_buffering_minus1[i]);
		w.ue(s.sps_max_num_reorder_pics[i]);
		w.ue(s.sps_max_latency_increase_plus1[i]);
	}
	w.ue(s.log2_min_luma_coding_block_size_minus3);
	w.ue(s.log2_diff_max_min_luma_coding_block_size);
	w.ue(s.log2_min_luma_transform_block_size_minus2);
	w.ue(s.log2_diff_max_min_luma_transform_block_size);
	w.ue(s.max_transform_hierarchy_depth_inter);
	w.ue(s.max_transform_hierarchy_depth_intra);
	w.flag(s.scaling_list_enabled_flag);
	if (s.scaling_list_enabled_flag) {
		w.flag(s.sps_scaling_list_data_present_flag);
		if (s.sps_scaling_list_data_present_flag && !scalingListData(w, s.scaling_lists))
			return false;
	}
	w.flag(s.amp_enabled_flag);
	w.flag(s.sample_adaptive_offset_enabled_flag);
	w.flag(s.pcm_enabled_flag);
	if (s.pcm_enabled_flag) {
		w.u(4, s.pcm_sample_bit_depth_luma_minus1);
		w.u(4, s.pcm_sample_bit_depth_chroma_minus1);
		w.ue(s.log2_min_pcm_luma_coding_block_size_minus3);
		w.ue(s.log2_diff_max_min_pcm_luma_coding_block_size);
		w.flag(s.pcm_loop_filter_disabled_flag);
	}
	w.ue(s.num_short_term_ref_pic_sets);
	for (uint32_t i = 0; i < s.num_short_term_ref_pic_sets; i++)
		if (!shortTermRps(w, s.st_ref_pic_set[i], i))
			return false;
	w.flag(s.long_term_ref_pics_present_flag);
	if (s.long_term_ref_pics_present_flag) {
		w.ue(s.num_long_term_ref_pics_sps);
		for (uint32_t i = 0; i < s.num_long_term_ref_pics_sps; i++) {
			w.u(s.log2_max_pic_order_cnt_lsb_minus4 + 4, s.lt_ref_pic_poc_lsb_sps[i]);
			w.flag(s.used_by_curr_pic_lt_sps_flag[i]);
		}
	}
	w.flag(s.sps_temporal_mvp_enabled_flag);
	w.flag(s.strong_intra_smoothing_enabled_flag);
	w.flag(0); // vui_parameters_present_flag
	w.flag(0); // sps_extension_present_flag
	w.appendNal({0x42, 0x01}, out);
	return true;
}

bool writePps(const Pps &p, std::vector<uint8_t> &out)
{
	if (p.pps_pic_parameter_set_id > 63 || p.pps_seq_parameter_set_id > 15 ||
	    p.num_extra_slice_header_bits > 7 || p.num_ref_idx_l0_default_active_minus1 > 14 ||
	    p.num_ref_idx_l1_default_active_minus1 > 14 || p.num_tile_columns_minus1 > 19 ||
	    p.num_tile_rows_minus1 > 21)
		return false;
	BitWriter w;
	w.ue(p.pps_pic_parameter_set_id);
	w.ue(p.pps_seq_parameter_set_id);
	w.flag(p.dependent_slice_segments_enabled_flag);
	w.flag(p.output_flag_present_flag);
	w.u(3, p.num_extra_slice_header_bits);
	w.flag(p.sign_data_hiding_enabled_flag);
	w.flag(p.cabac_init_present_flag);
	w.ue(p.num_ref_idx_l0_default_active_minus1);
	w.ue(p.num_ref_idx_l1_default_active_minus1);
	w.se(p.init_qp_minus26);
	w.flag(p.constrained_intra_pred_flag);
	w.flag(p.transform_skip_enabled_flag);
	w.flag(p.cu_qp_delta_enabled_flag);
	if (p.cu_qp_delta_enabled_flag)
		w.ue(p.diff_cu_qp_delta_depth);
	w.se(p.pps_cb_qp_offset);
	w.se(p.pps_cr_qp_offset);
	w.flag(p.pps_slice_chroma_qp_offsets_present_flag);
	w.flag(p.weighted_pred_flag);
	w.flag(p.weighted_bipred_flag);
	w.flag(p.transquant_bypass_enabled_flag);
	w.flag(p.tiles_enabled_flag);
	w.flag(p.entropy_coding_sync_enabled_flag);
	if (p.tiles_enabled_flag) {
		w.ue(p.num_tile_columns_minus1);
		w.ue(p.num_tile_rows_minus1);
		w.flag(p.uniform_spacing_flag);
		if (!p.uniform_spacing_flag) {
			for (uint32_t i = 0; i < p.num_tile_columns_minus1; i++)
				w.ue(p.column_width_minus1[i]);
			for (uint32_t i = 0; i < p.num_tile_rows_minus1; i++)
				w.ue(p.row_height_minus1[i]);
		}
		w.flag(p.loop_filter_across_tiles_enabled_flag);
	}
	w.flag(p.pps_loop_filter_across_slices_enabled_flag);
	w.flag(p.deblocking_filter_control_present_flag);
	if (p.deblocking_filter_control_present_flag) {
		w.flag(p.deblocking_filter_override_enabled_flag);
		w.flag(p.pps_deblocking_filter_disabled_flag);
		if (!p.pps_deblocking_filter_disabled_flag) {
			w.se(p.pps_beta_offset_div2);
			w.se(p.pps_tc_offset_div2);
		}
	}
	w.flag(p.pps_scaling_list_data_present_flag);
	if (p.pps_scaling_list_data_present_flag && !scalingListData(w, p.scaling_lists))
		return false;
	w.flag(p.lists_modification_present_flag);
	w.ue(p.log2_parallel_merge_level_minus2);
	w.flag(p.slice_segment_header_extension_present_flag);
	w.flag(0); // pps_extension_present_flag
	w.appendNal({0x44, 0x01}, out);
	return true;
}

} // namespace rgfx::video::h265
