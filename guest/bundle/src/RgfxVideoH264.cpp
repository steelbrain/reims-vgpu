/*
 * RgfxVideoH264: H.264 parameter sets for the host's decoder, made from AppleGVA's
 * (RgfxVideoH264.hpp).
 */

#include "RgfxVideoH264.hpp"

#include "RgfxVideoBits.hpp"

namespace rgfx::video::h264 {
namespace {

// The bits se(v) takes.
uint32_t seBits(int32_t value)
{
	uint64_t v = value > 0 ? 2 * (uint64_t)value : (uint64_t)(-2 * (int64_t)value) + 1;
	uint32_t length = 0;
	while ((v >> length) > 1)
		length++;
	return 2 * length + 1;
}

// A scaling list as H.264 codes it, in its coded (scan) order: the default matrix flagged by a
// first delta that makes the next scale 0, else each entry as its delta from the last. Where the
// list ends in a run of its last value, the run is cut short by a delta to 0 when that is no
// longer than the run's one-bit zero deltas, as encoders write it.
void scalingList(BitWriter &w, const uint32_t *list, uint32_t size, bool useDefault)
{
	if (useDefault) {
		w.se(-8);
		return;
	}
	uint32_t end = size;
	while (end > 1 && list[end - 1] == list[end - 2])
		end--;
	if (end < size && size - end < seBits(-(int32_t)list[end]))
		end = size;
	int32_t last = 8;
	for (uint32_t j = 0; j < size; j++) {
		int32_t next = j < end ? (int32_t)list[j] : 0;
		int32_t delta = next - last;
		if (delta > 127)
			delta -= 256;
		if (delta < -128)
			delta += 256;
		w.se(delta);
		if (next == 0)
			return;
		last = next;
	}
}

bool highProfile(uint32_t profile)
{
	switch (profile) {
	case 100: case 110: case 122: case 244: case 44: case 83: case 86: case 118: case 128:
	case 138: case 139: case 134: case 135:
		return true;
	default:
		return false;
	}
}

} // namespace

bool writeSps(const Sps &s, std::vector<uint8_t> &out)
{
	if (s.profile_idc > 255 || s.level_idc > 255 || s.seq_parameter_set_id > 31 ||
	    s.chroma_format_idc > 2 || s.log2_max_frame_num_minus4 > 12 || s.pic_order_cnt_type > 2 ||
	    s.log2_max_pic_order_cnt_lsb_minus4 > 12 || s.num_ref_frames_in_pic_order_cnt_cycle > 255)
		return false;
	BitWriter w;
	w.u(8, s.profile_idc);
	for (uint32_t flag : s.constraint_set_flag)
		w.flag(flag);
	w.u(2, 0);
	w.u(8, s.level_idc);
	w.ue(s.seq_parameter_set_id);
	if (highProfile(s.profile_idc)) {
		w.ue(s.chroma_format_idc);
		w.ue(s.bit_depth_luma_minus8);
		w.ue(s.bit_depth_chroma_minus8);
		w.flag(s.qpprime_y_zero_transform_bypass_flag);
		w.flag(s.seq_scaling_matrix_present_flag);
		if (s.seq_scaling_matrix_present_flag)
			for (int i = 0; i < 8; i++) {
				w.flag(s.seq_scaling_list_present_flag[i]);
				if (!s.seq_scaling_list_present_flag[i])
					continue;
				if (i < 6)
					scalingList(w, s.scaling_list_4x4[i], 16, s.use_default_scaling_matrix_4x4_flag[i]);
				else
					scalingList(w, s.scaling_list_8x8[i - 6], 64,
						    s.use_default_scaling_matrix_8x8_flag[i - 6]);
			}
	}
	w.ue(s.log2_max_frame_num_minus4);
	w.ue(s.pic_order_cnt_type);
	if (s.pic_order_cnt_type == 0) {
		w.ue(s.log2_max_pic_order_cnt_lsb_minus4);
	} else if (s.pic_order_cnt_type == 1) {
		w.flag(s.delta_pic_order_always_zero_flag);
		w.se(s.offset_for_non_ref_pic);
		w.se(s.offset_for_top_to_bottom_field);
		w.ue(s.num_ref_frames_in_pic_order_cnt_cycle);
		for (uint32_t i = 0; i < s.num_ref_frames_in_pic_order_cnt_cycle; i++)
			w.se(s.offset_for_ref_frame[i]);
	}
	w.ue(s.max_num_ref_frames);
	w.flag(s.gaps_in_frame_num_value_allowed_flag);
	w.ue(s.pic_width_in_mbs_minus1);
	w.ue(s.pic_height_in_map_units_minus1);
	w.flag(s.frame_mbs_only_flag);
	if (!s.frame_mbs_only_flag)
		w.flag(s.mb_adaptive_frame_field_flag);
	w.flag(s.direct_8x8_inference_flag);
	w.flag(s.frame_cropping_flag);
	if (s.frame_cropping_flag) {
		w.ue(s.frame_crop_left_offset);
		w.ue(s.frame_crop_right_offset);
		w.ue(s.frame_crop_top_offset);
		w.ue(s.frame_crop_bottom_offset);
	}
	w.flag(0); // vui_parameters_present_flag
	w.appendNal({0x67}, out);
	return true;
}

bool writePps(const Pps &p, const Sps &s, std::vector<uint8_t> &out)
{
	if (p.pic_parameter_set_id > 255 || p.seq_parameter_set_id > 31 || p.num_slice_groups_minus1 != 0 ||
	    p.num_ref_idx_l0_default_active_minus1 > 31 || p.num_ref_idx_l1_default_active_minus1 > 31 ||
	    p.weighted_bipred_idc > 2)
		return false;
	BitWriter w;
	w.ue(p.pic_parameter_set_id);
	w.ue(p.seq_parameter_set_id);
	w.flag(p.entropy_coding_mode_flag);
	w.flag(p.bottom_field_pic_order_in_frame_present_flag);
	w.ue(p.num_slice_groups_minus1);
	w.ue(p.num_ref_idx_l0_default_active_minus1);
	w.ue(p.num_ref_idx_l1_default_active_minus1);
	w.flag(p.weighted_pred_flag);
	w.u(2, p.weighted_bipred_idc);
	w.se(p.pic_init_qp_minus26);
	w.se(p.pic_init_qs_minus26);
	w.se(p.chroma_qp_index_offset);
	w.flag(p.deblocking_filter_control_present_flag);
	w.flag(p.constrained_intra_pred_flag);
	w.flag(p.redundant_pic_cnt_present_flag);
	// The PPS's extension: written where it says more than its absence would (a transform or
	// scaling flag set, or a second chroma offset of its own).
	if (p.transform_8x8_mode_flag || p.pic_scaling_matrix_present_flag ||
	    p.second_chroma_qp_index_offset != p.chroma_qp_index_offset) {
		w.flag(p.transform_8x8_mode_flag);
		w.flag(p.pic_scaling_matrix_present_flag);
		if (p.pic_scaling_matrix_present_flag) {
			int lists = 6 + (s.chroma_format_idc == 3 ? 6 : 2) * (p.transform_8x8_mode_flag ? 1 : 0);
			if (lists > 8)
				return false;
			for (int i = 0; i < lists; i++) {
				w.flag(p.pic_scaling_list_present_flag[i]);
				if (!p.pic_scaling_list_present_flag[i])
					continue;
				if (i < 6)
					scalingList(w, p.scaling_list_4x4[i], 16, p.use_default_scaling_matrix_4x4_flag[i]);
				else
					scalingList(w, p.scaling_list_8x8[i - 6], 64,
						    p.use_default_scaling_matrix_8x8_flag[i - 6]);
			}
		}
		w.se(p.second_chroma_qp_index_offset);
	}
	w.appendNal({0x68}, out);
	return true;
}

} // namespace rgfx::video::h264
