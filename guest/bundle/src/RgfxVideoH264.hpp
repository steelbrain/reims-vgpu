/*
 * RgfxVideoH264: H.264 parameter sets for the host's decoder, made from AppleGVA's.
 *
 * AppleGVA parses a stream's SPS and PPS itself and hands the video driver each picture's slices
 * as Annex-B NAL units, with the parameter sets only in its own parsed form: one 32-bit field per
 * syntax element, in the syntax's order, with the scaling lists and the POC cycle's offsets at
 * their full sizes. The host decodes access units as a stream gives them, so the driver writes
 * the parameter sets back out as NAL units from those fields: the SPS without its VUI, which
 * holds nothing a decoder needs (the host decodes each picture as it comes, and AppleGVA orders
 * the output), and the PPS whole.
 *
 * Every field below is placed by H.264's syntax order between fields whose meaning the calls'
 * use states (the profile and level, the scaling lists, the reference count, the size, the PPS's
 * last field). Both records start with a word that is not a syntax element (1 in every stream
 * seen). On the guest (macOS 26.6.1), the parameter sets written from AppleGVA's records of
 * six x264 streams equal the streams' own, the SPS up to its VUI, for Constrained Baseline
 * with POC type 2, Main with B-frames and weighted prediction, High with cropping, the default
 * and custom scaling matrices (the lists in their coded order) and chroma QP offsets; AppleGVA
 * takes no interlaced stream. Fields those streams leave at 0 or unused keep their syntax-order
 * place unchecked: the two ids (pic_parameter_set_id first, then seq_parameter_set_id), POC
 * type 1's cycle, frame_mbs_only_flag 0 and MBAFF, and the SPS's own scaling lists.
 */

#ifndef RGFX_VIDEO_H264_HPP_
#define RGFX_VIDEO_H264_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rgfx::video::h264 {

// AppleGVA's parsed SPS, as far as the decoder reads it (its VUI follows).
struct Sps {
	uint32_t field_000;
	uint32_t profile_idc;
	uint32_t constraint_set_flag[6];
	uint32_t reserved_zero_2bits;
	uint32_t level_idc;
	uint32_t seq_parameter_set_id;
	uint32_t chroma_format_idc;
	uint32_t separate_colour_plane_flag;
	uint32_t bit_depth_luma_minus8;
	uint32_t bit_depth_chroma_minus8;
	uint32_t qpprime_y_zero_transform_bypass_flag;
	uint32_t seq_scaling_matrix_present_flag;
	uint32_t seq_scaling_list_present_flag[8];
	uint32_t scaling_list_4x4[6][16];
	uint32_t scaling_list_8x8[2][64];
	uint32_t use_default_scaling_matrix_4x4_flag[6];
	uint32_t use_default_scaling_matrix_8x8_flag[2];
	uint32_t log2_max_frame_num_minus4;
	uint32_t pic_order_cnt_type;
	uint32_t log2_max_pic_order_cnt_lsb_minus4;
	uint32_t delta_pic_order_always_zero_flag;
	int32_t offset_for_non_ref_pic;
	int32_t offset_for_top_to_bottom_field;
	uint32_t num_ref_frames_in_pic_order_cnt_cycle;
	int32_t offset_for_ref_frame[256];
	uint32_t max_num_ref_frames;
	uint32_t gaps_in_frame_num_value_allowed_flag;
	uint32_t pic_width_in_mbs_minus1;
	uint32_t pic_height_in_map_units_minus1;
	uint32_t frame_mbs_only_flag;
	uint32_t mb_adaptive_frame_field_flag;
	uint32_t direct_8x8_inference_flag;
	uint32_t frame_cropping_flag;
	uint32_t frame_crop_left_offset;
	uint32_t frame_crop_right_offset;
	uint32_t frame_crop_top_offset;
	uint32_t frame_crop_bottom_offset;
	uint32_t vui_parameters_present_flag;
};
static_assert(offsetof(Sps, level_idc) == 0x24, "SPS layout");
static_assert(offsetof(Sps, scaling_list_4x4) == 0x64, "SPS layout");
static_assert(offsetof(Sps, log2_max_frame_num_minus4) == 0x404, "SPS layout");
static_assert(offsetof(Sps, max_num_ref_frames) == 0x820, "SPS layout");
static_assert(offsetof(Sps, direct_8x8_inference_flag) == 0x838, "SPS layout");

// AppleGVA's parsed PPS.
struct Pps {
	uint32_t field_000;
	uint32_t pic_parameter_set_id;
	uint32_t seq_parameter_set_id;
	uint32_t entropy_coding_mode_flag;
	uint32_t bottom_field_pic_order_in_frame_present_flag;
	uint32_t num_slice_groups_minus1;
	uint32_t slice_group_map_type;
	uint32_t run_length_minus1[8];
	uint32_t top_left[8];
	uint32_t bottom_right[8];
	uint32_t slice_group_change_direction_flag;
	uint32_t slice_group_change_rate_minus1;
	uint32_t field_084;
	uint64_t field_088;
	uint32_t num_ref_idx_l0_default_active_minus1;
	uint32_t num_ref_idx_l1_default_active_minus1;
	uint32_t weighted_pred_flag;
	uint32_t weighted_bipred_idc;
	int32_t pic_init_qp_minus26;
	int32_t pic_init_qs_minus26;
	int32_t chroma_qp_index_offset;
	uint32_t deblocking_filter_control_present_flag;
	uint32_t constrained_intra_pred_flag;
	uint32_t redundant_pic_cnt_present_flag;
	uint32_t transform_8x8_mode_flag;
	uint32_t pic_scaling_matrix_present_flag;
	uint32_t pic_scaling_list_present_flag[8];
	uint32_t scaling_list_4x4[6][16];
	uint32_t scaling_list_8x8[2][64];
	uint32_t use_default_scaling_matrix_4x4_flag[6];
	uint32_t use_default_scaling_matrix_8x8_flag[2];
	int32_t second_chroma_qp_index_offset;
};
static_assert(offsetof(Pps, entropy_coding_mode_flag) == 0xc, "PPS layout");
static_assert(offsetof(Pps, slice_group_change_rate_minus1) == 0x80, "PPS layout");
static_assert(offsetof(Pps, num_ref_idx_l0_default_active_minus1) == 0x90, "PPS layout");
static_assert(offsetof(Pps, pic_scaling_matrix_present_flag) == 0xbc, "PPS layout");
static_assert(offsetof(Pps, second_chroma_qp_index_offset) == 0x480, "PPS layout");

// The layouts AppleGVA's parsed SPS comes in. Every release measured from 12 on lays it out as
// Sps (kSixConstraintFlags). macOS 11's AppleGVA (108.7.2) parses the header of H.264's older
// syntax: four constraint flags, then one field for the four bits after them, where later
// releases hold constraint_set4_flag, constraint_set5_flag and reserved_zero_2bits. So from
// level_idc on, every field of 11's record sits 8 bytes before Sps's (kFourConstraintFlags):
// AppleGVA 108.7.2 reads level_idc at 0x1c, chroma_format_idc at 0x24, max_num_ref_frames at
// 0x818 and frame_mbs_only_flag at 0x828 (read from its code), and on the guest (11.7.11, the
// 780M host, 2026-11-08) a 640x480 x264 stream's record holds its level, size and reference
// count there, and 12.7.6's, 13.7.8's and 26.7.2's where Sps holds them. A record does not say
// its layout, so the driver takes the one layout in which the record describes the picture it
// decodes (chooseSpsLayout). The parsed PPS is laid out as Pps on all four.
enum class SpsLayout : uint8_t { kSixConstraintFlags = 1, kFourConstraintFlags = 2 };

// The SPS a record of AppleGVA's holds in `layout`, as Sps. In kFourConstraintFlags the field
// after the four flags holds the four header bits that follow them, constraint_set4_flag first;
// they come back as constraint_set_flag[4] and [5] and reserved_zero_2bits, which writeSps
// writes as the same bits.
Sps spsIn(const uint8_t *record, SpsLayout layout);

// Whether `sps` describes the picture AppleGVA decodes into a `width` x `height` surface with a
// PPS of `ppsSps`, its seq_parameter_set_id: a profile_idc and a level_idc of H.264's, its own
// seq_parameter_set_id that PPS's, and a coded size the surface's rounds up to (its width to
// whole macroblocks, its height to whole macroblocks of a frame, or of a field pair where
// frame_mbs_only_flag is 0).
bool spsFits(const Sps &sps, uint32_t ppsSps, uint32_t width, uint32_t height);

// Why chooseSpsLayout chose none.
enum class NoLayout : uint8_t { kNone, kMany };

// The one layout in which `record` fits (spsFits) the picture of a `width` x `height` surface with
// a PPS of `ppsSps`, its SPS in that layout in `sps`; false, with `why`, where it fits in none or
// in more than one.
bool chooseSpsLayout(const uint8_t *record, uint32_t ppsSps, uint32_t width, uint32_t height,
                     SpsLayout *layout, Sps *sps, NoLayout *why);

// Appends the SPS (without its VUI) and the PPS, each as an Annex-B NAL unit with a four-byte
// start code, to out. False, and out as it was, where a field is out of H.264's range or the
// stream uses what the decoder does not take (slice groups, 4:4:4's twelve scaling lists).
bool writeSps(const Sps &sps, std::vector<uint8_t> &out);
bool writePps(const Pps &pps, const Sps &sps, std::vector<uint8_t> &out);

} // namespace rgfx::video::h264

#endif
