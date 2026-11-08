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

namespace {

// Ceil(Log2(n)), the bits of a syntax element that indexes n things; 0 for n of 1 or less.
uint32_t ceilLog2(uint32_t n)
{
	uint32_t bits = 0;
	while (bits < 32 && ((uint64_t)1 << bits) < n)
		bits++;
	return bits;
}

// The pictures the current one uses of the short-term set a slice header codes itself,
// st_ref_pic_set(num_short_term_ref_pic_sets) (7.3.7), read from `r`: a set of its own pictures,
// or one predicted from the SPS's set RefRpsIdx (7-61, 7-62). False where it is out of range.
bool ownShortTermUsed(BitReader &r, const Sps &sps, uint32_t *used)
{
	const uint32_t idx = sps.num_short_term_ref_pic_sets;
	const bool predicted = idx != 0 && r.flag(); // inter_ref_pic_set_prediction_flag
	*used = 0;
	if (!predicted) {
		const uint32_t negative = r.ue(), positive = r.ue();
		if (r.failed() || negative > 16 || positive > 16 || negative + positive > 16)
			return false;
		for (uint32_t i = 0; i < negative + positive; i++) {
			r.ue(); // delta_poc_s0_minus1, delta_poc_s1_minus1
			*used += r.flag();
		}
		return !r.failed();
	}
	const uint32_t deltaIdx = r.ue() + 1; // delta_idx_minus1 + 1
	if (r.failed() || deltaIdx > idx)
		return false;
	const ShortTermRps &ref = sps.st_ref_pic_set[idx - deltaIdx];
	const uint32_t negative = ref.num_negative_pics, positive = ref.num_positive_pics;
	if (negative > 16 || positive > 16 || negative + positive > 16)
		return false;
	const bool sign = r.flag();     // delta_rps_sign
	const uint32_t abs = r.ue() + 1; // abs_delta_rps_minus1 + 1
	if (r.failed() || abs > 0x8000)
		return false;
	const int32_t deltaRps = sign ? -(int32_t)abs : (int32_t)abs;
	// used_by_curr_pic_flag[j] and use_delta_flag[j], j to NumDeltaPocs[RefRpsIdx]: the SPS
	// set's S0 pictures, its S1 pictures, then its own picture.
	bool usedBy[17], useDelta[17];
	for (uint32_t j = 0; j <= negative + positive; j++) {
		usedBy[j] = r.flag();
		useDelta[j] = usedBy[j] || r.flag();
	}
	if (r.failed())
		return false;
	// 7-61 and 7-62: a picture of the set at dPoc from the current one is kept where its
	// use_delta_flag is, in S0 where dPoc < 0 and S1 where dPoc > 0; the current picture uses it
	// where its used_by_curr_pic_flag is.
	const auto take = [&](int32_t dPoc, uint32_t j) {
		if (dPoc != 0 && useDelta[j] && usedBy[j])
			(*used)++;
	};
	for (uint32_t j = 0; j < negative; j++)
		take(ref.delta_poc_s0[j] + deltaRps, j);
	for (uint32_t j = 0; j < positive; j++)
		take(ref.delta_poc_s1[j] + deltaRps, negative + j);
	take(deltaRps, negative + positive);
	return true;
}

// The pictures of the SPS's short-term set `rps` the current picture uses.
uint32_t shortTermUsed(const ShortTermRps &rps)
{
	uint32_t used = 0;
	for (uint32_t i = 0; i < rps.num_negative_pics && i < 16; i++)
		used += rps.used_by_curr_pic_s0[i] != 0;
	for (uint32_t i = 0; i < rps.num_positive_pics && i < 16; i++)
		used += rps.used_by_curr_pic_s1[i] != 0;
	return used;
}

} // namespace

bool currentReferences(const Sps &sps, const Pps &pps, const uint8_t *au, size_t size,
                       uint32_t *count)
{
	// The first VCL NAL unit: from after its start code to the next start code or the end.
	const uint8_t *nal = nullptr;
	size_t length = 0;
	for (size_t i = 0; i + 3 <= size && !nal; i++) {
		if (au[i] != 0 || au[i + 1] != 0 || au[i + 2] != 1)
			continue;
		const size_t start = i + 3;
		size_t end = start;
		while (end + 3 <= size && !(au[end] == 0 && au[end + 1] == 0 && au[end + 2] <= 1))
			end++;
		if (end + 3 > size)
			end = size;
		if (end - start > 2 && ((au[start] >> 1) & 0x3f) < 32) {
			nal = au + start;
			length = end - start;
		}
		i = end > i + 2 ? end - 1 : i + 2;
	}
	if (!nal)
		return false;
	const uint32_t type = (nal[0] >> 1) & 0x3f;
	BitReader r(nal + 2, length - 2);
	if (!r.flag()) // first_slice_segment_in_pic_flag
		return false;
	if (type >= 16 && type <= 23)
		r.flag(); // no_output_of_prior_pics_flag
	if (r.ue() != pps.pps_pic_parameter_set_id || r.failed()) // slice_pic_parameter_set_id
		return false;
	r.u(pps.num_extra_slice_header_bits);             // slice_reserved_flag[]
	r.ue();                                           // slice_type
	if (pps.output_flag_present_flag)
		r.flag(); // pic_output_flag
	if (sps.separate_colour_plane_flag)
		r.u(2); // colour_plane_id
	*count = 0;
	if (type == 19 || type == 20) // IDR_W_RADL, IDR_N_LP
		return !r.failed();
	if (sps.log2_max_pic_order_cnt_lsb_minus4 > 12)
		return false;
	r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4); // slice_pic_order_cnt_lsb
	uint32_t used = 0;
	if (!r.flag()) { // short_term_ref_pic_set_sps_flag
		if (sps.num_short_term_ref_pic_sets > 64 || !ownShortTermUsed(r, sps, &used))
			return false;
	} else {
		const uint32_t sets = sps.num_short_term_ref_pic_sets;
		const uint32_t idx = r.u(ceilLog2(sets)); // short_term_ref_pic_set_idx
		if (r.failed() || idx >= sets || sets > 64)
			return false;
		used = shortTermUsed(sps.st_ref_pic_set[idx]);
	}
	if (sps.long_term_ref_pics_present_flag) {
		const uint32_t inSps = sps.num_long_term_ref_pics_sps;
		if (inSps > 32)
			return false;
		const uint32_t fromSps = inSps > 0 ? r.ue() : 0; // num_long_term_sps
		const uint32_t own = r.ue();                     // num_long_term_pics
		if (r.failed() || fromSps > inSps || fromSps + own > 32)
			return false;
		for (uint32_t i = 0; i < fromSps + own; i++) {
			if (i < fromSps) {
				const uint32_t lt = r.u(ceilLog2(inSps)); // lt_idx_sps[i]
				if (lt >= inSps)
					return false;
				used += sps.used_by_curr_pic_lt_sps_flag[lt] != 0;
			} else {
				r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4); // poc_lsb_lt[i]
				used += r.flag();                                // used_by_curr_pic_lt_flag[i]
			}
			if (r.flag()) // delta_poc_msb_present_flag[i]
				r.ue(); // delta_poc_msb_cycle_lt[i]
		}
	}
	if (r.failed())
		return false;
	*count = used;
	return true;
}

bool narrowestTileColumn(const Sps &sps, const Pps &pps, uint32_t *narrowest)
{
	*narrowest = 0;
	if (!pps.tiles_enabled_flag)
		return true;
	const uint32_t ctbLog2 = sps.log2_min_luma_coding_block_size_minus3 + 3 +
	                         sps.log2_diff_max_min_luma_coding_block_size;
	const uint32_t columns = pps.num_tile_columns_minus1 + 1;
	if (ctbLog2 > 6 || columns > 20 || sps.pic_width_in_luma_samples == 0)
		return false;
	const uint32_t ctb = 1u << ctbLog2;
	const uint32_t across = (sps.pic_width_in_luma_samples + ctb - 1) >> ctbLog2; // PicWidthInCtbsY
	if (columns > across)
		return false;
	uint32_t least = UINT32_MAX, used = 0;
	for (uint32_t i = 0; i < columns; i++) {
		// 6-3, or the PPS's widths and the rest of the picture for the last column.
		uint32_t width;
		if (pps.uniform_spacing_flag)
			width = (i + 1) * across / columns - i * across / columns;
		else if (i + 1 < columns)
			width = pps.column_width_minus1[i] + 1;
		else
			width = across > used ? across - used : 0;
		if (width == 0 || width > across - used)
			return false;
		used += width;
		least = width < least ? width : least;
	}
	*narrowest = least << ctbLog2;
	return true;
}

} // namespace rgfx::video::h265
