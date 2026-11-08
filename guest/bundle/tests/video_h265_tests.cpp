/*
 * video_h265_tests: the video driver's HEVC parameter-set writer (src/RgfxVideoH265.*), run on
 * the Mac (guest/bundle/Makefile `test`). For each stream of video_h265_cases.hpp it writes the
 * SPS and PPS from the fields AppleGVA's parsed form holds and checks them against the NAL units
 * the encoder wrote: the SPS bit for bit up to its VUI flag, which ours clears, and the PPS byte
 * for byte. Where the encoder predicted a short-term set from another, which ours writes by its
 * pictures, the SPS's sets are read back instead and the bits after them compared. It also checks
 * the emulation prevention of a large field, the refusals, and the count of a picture's current
 * references read from its first slice segment's header (currentReferences).
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "RgfxVideoBits.hpp"
#include "RgfxVideoH265.hpp"
#include "RgfxVideoStatus.hpp"

using namespace rgfx::video::h265;

namespace {

int g_checks = 0;

#define CHECK(cond, ...)                                                                  \
	do {                                                                              \
		g_checks++;                                                               \
		if (!(cond)) {                                                            \
			std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
			std::fprintf(stderr, __VA_ARGS__);                                \
			std::fprintf(stderr, "\n");                                       \
			std::abort();                                                     \
		}                                                                         \
	} while (0)

struct Case {
	const char *name;
	Sps sps;
	Pps pps;
	std::vector<uint8_t> sps_nal; // header bytes first, no start code, as the encoder escaped it
	std::vector<uint8_t> pps_nal;
	uint32_t sps_vui_bit;      // the SPS's vui_parameters_present_flag
	uint32_t sps_sets_bit;     // its first st_ref_pic_set()
	uint32_t sps_sets_end_bit; // long_term_ref_pics_present_flag, after its last
	bool sps_sets_predicted;   // whether the encoder predicted a set from another
};

template <size_t N> void fill(uint32_t (&list)[N], std::initializer_list<uint32_t> values)
{
	size_t i = 0;
	for (uint32_t v : values)
		list[i++] = v;
}

// A short-term set by its pictures: the order count distances before and after the current
// picture, nearest first, and whether it uses each.
void rps(ShortTermRps &r, std::initializer_list<int32_t> s0, std::initializer_list<uint8_t> used0,
         std::initializer_list<int32_t> s1, std::initializer_list<uint8_t> used1)
{
	uint32_t i = 0;
	for (int32_t d : s0)
		r.delta_poc_s0[i++] = d;
	r.num_negative_pics = i;
	i = 0;
	for (uint8_t u : used0)
		r.used_by_curr_pic_s0[i++] = u;
	i = 0;
	for (int32_t d : s1)
		r.delta_poc_s1[i++] = d;
	r.num_positive_pics = i;
	i = 0;
	for (uint8_t u : used1)
		r.used_by_curr_pic_s1[i++] = u;
	r.num_delta_pocs = r.num_negative_pics + r.num_positive_pics;
}

#include "video_h265_cases.hpp"

// A NAL unit's RBSP: its emulation prevention bytes taken out.
std::vector<uint8_t> unescape(const uint8_t *nal, size_t size)
{
	std::vector<uint8_t> out;
	uint32_t zeros = 0;
	for (size_t i = 0; i < size; i++) {
		if (zeros >= 2 && nal[i] == 3) {
			zeros = 0;
			continue;
		}
		out.push_back(nal[i]);
		zeros = nal[i] == 0 ? zeros + 1 : 0;
	}
	return out;
}

uint32_t bitAt(const std::vector<uint8_t> &bytes, uint32_t bit)
{
	return bit / 8 < bytes.size() ? (bytes[bit / 8] >> (7 - bit % 8)) & 1 : 0;
}

// Reads an RBSP from a bit on.
struct Reader {
	const std::vector<uint8_t> &bytes;
	uint32_t bit;

	uint32_t u(uint32_t n)
	{
		uint32_t v = 0;
		while (n--)
			v = (v << 1) | bitAt(bytes, bit++);
		return v;
	}
	uint32_t ue()
	{
		uint32_t zeros = 0;
		while (bitAt(bytes, bit) == 0 && zeros < 32) {
			zeros++;
			bit++;
		}
		bit++;
		return (uint32_t)(((uint64_t)1 << zeros) - 1 + u(zeros));
	}
};

// The written NAL unit after its four-byte start code.
std::vector<uint8_t> payload(const std::vector<uint8_t> &written)
{
	CHECK(written.size() > 6 && written[0] == 0 && written[1] == 0 && written[2] == 0 && written[3] == 1,
	      "no start code");
	return std::vector<uint8_t>(written.begin() + 4, written.end());
}

// No 0x000000, 0x000001 or 0x000002 left (0x000003 is the escape itself).
void checkEscaped(const std::vector<uint8_t> &nal, const char *name)
{
	for (size_t i = 2; i < nal.size(); i++)
		CHECK(!(nal[i - 2] == 0 && nal[i - 1] == 0 && nal[i] <= 2), "%s: unescaped run at %zu", name, i);
}

// Reads ours's sets from `bit` on and checks each is the case's, written by its pictures; answers
// the bit after them.
uint32_t checkSets(const Case &c, const std::vector<uint8_t> &ours, uint32_t bit)
{
	Reader r{ours, bit};
	for (uint32_t idx = 0; idx < c.sps.num_short_term_ref_pic_sets; idx++) {
		const ShortTermRps &want = c.sps.st_ref_pic_set[idx];
		if (idx != 0)
			CHECK(r.u(1) == 0, "%s: set %u predicted", c.name, idx);
		const uint32_t negative = r.ue(), positive = r.ue();
		CHECK(negative == want.num_negative_pics && positive == want.num_positive_pics,
		      "%s: set %u holds %u and %u pictures", c.name, idx, negative, positive);
		int32_t poc = 0;
		for (uint32_t i = 0; i < negative; i++) {
			poc -= (int32_t)r.ue() + 1;
			CHECK(poc == want.delta_poc_s0[i] && r.u(1) == want.used_by_curr_pic_s0[i],
			      "%s: set %u picture S0[%u]", c.name, idx, i);
		}
		poc = 0;
		for (uint32_t i = 0; i < positive; i++) {
			poc += (int32_t)r.ue() + 1;
			CHECK(poc == want.delta_poc_s1[i] && r.u(1) == want.used_by_curr_pic_s1[i],
			      "%s: set %u picture S1[%u]", c.name, idx, i);
		}
	}
	return r.bit;
}

void checkCase(const Case &c)
{
	std::vector<uint8_t> written;
	CHECK(writeSps(c.sps, written), "%s: SPS refused", c.name);
	std::vector<uint8_t> nal = payload(written);
	checkEscaped(nal, c.name);
	std::vector<uint8_t> ours = unescape(nal.data(), nal.size());
	std::vector<uint8_t> theirs = unescape(c.sps_nal.data(), c.sps_nal.size());
	CHECK(bitAt(theirs, c.sps_vui_bit) == 1, "%s: the stream's SPS has no VUI", c.name);
	// Where the encoder predicted sets, ours are read back and the rest compared after them.
	const uint32_t same = c.sps_sets_predicted ? c.sps_sets_bit : c.sps_vui_bit;
	for (uint32_t bit = 0; bit < same; bit++)
		CHECK(bitAt(ours, bit) == bitAt(theirs, bit), "%s: SPS bit %u differs", c.name, bit);
	uint32_t vui = c.sps_vui_bit;
	if (c.sps_sets_predicted) {
		const uint32_t after = checkSets(c, ours, c.sps_sets_bit);
		for (uint32_t bit = c.sps_sets_end_bit; bit < c.sps_vui_bit; bit++)
			CHECK(bitAt(ours, after + bit - c.sps_sets_end_bit) == bitAt(theirs, bit),
			      "%s: SPS bit %u after the sets differs", c.name, bit);
		vui = after + c.sps_vui_bit - c.sps_sets_end_bit;
	}
	// Ours: the VUI and extension flags clear, the stop bit, and zeros to the byte's end.
	CHECK(bitAt(ours, vui) == 0 && bitAt(ours, vui + 1) == 0, "%s: VUI or extension flag set",
	      c.name);
	CHECK(bitAt(ours, vui + 2) == 1, "%s: no stop bit", c.name);
	CHECK(ours.size() == (vui + 3 + 7) / 8, "%s: SPS of %zu bytes", c.name, ours.size());
	for (uint32_t bit = vui + 3; bit < ours.size() * 8; bit++)
		CHECK(bitAt(ours, bit) == 0, "%s: SPS trailing bit %u set", c.name, bit);

	written.clear();
	CHECK(writePps(c.pps, written), "%s: PPS refused", c.name);
	nal = payload(written);
	CHECK(nal == c.pps_nal, "%s: PPS of %zu bytes differs from the stream's %zu", c.name, nal.size(),
	      c.pps_nal.size());

	// Both appended after what out held.
	std::vector<uint8_t> spsWritten, ppsWritten, both = {0xaa};
	CHECK(writeSps(c.sps, spsWritten) && writePps(c.pps, ppsWritten), "%s: refused", c.name);
	CHECK(writeSps(c.sps, both) && writePps(c.pps, both), "%s: refused", c.name);
	CHECK(both.size() == 1 + spsWritten.size() + ppsWritten.size() && both[0] == 0xaa &&
		      std::memcmp(both.data() + 1, spsWritten.data(), spsWritten.size()) == 0,
	      "%s: not appended", c.name);
}

void checkEscapes()
{
	// A width of 2^30 - 1 is a ue(v) of 30 zeros, a one and 30 zeros: runs of zero bytes, each
	// escaped.
	Case c = kCases[0]();
	Sps s = c.sps;
	s.pic_width_in_luma_samples = 0x3fffffff;
	std::vector<uint8_t> written;
	CHECK(writeSps(s, written), "large width refused");
	std::vector<uint8_t> nal = payload(written);
	checkEscaped(nal, "large width");
	std::vector<uint8_t> rbsp = unescape(nal.data(), nal.size());
	CHECK(rbsp.size() < nal.size(), "nothing escaped");
	// After the header (16), the ids and nesting flag (8), the profile, tier and level (96, and
	// no sub-layer's), the SPS id ue(0) and the chroma format ue(1).
	CHECK(c.sps.sps_max_sub_layers_minus1 == 0 && c.sps.sps_seq_parameter_set_id == 0,
	      "the first case has sub-layers or another SPS id");
	Reader r{rbsp, 16 + 8 + 96 + 1 + 3};
	CHECK(r.ue() == 0x3fffffff, "width read back");
}

void checkRefusals()
{
	Case c = kCases[0]();
	std::vector<uint8_t> out = {1, 2};
	Sps s = c.sps;
	s.chroma_format_idc = 2;
	CHECK(!writeSps(s, out) && out.size() == 2, "4:2:2 taken");
	s = c.sps;
	s.bit_depth_luma_minus8 = 4;
	CHECK(!writeSps(s, out) && out.size() == 2, "12 bits taken");
	s = c.sps;
	s.num_short_term_ref_pic_sets = 65;
	CHECK(!writeSps(s, out) && out.size() == 2, "65 short-term sets taken");
	s = c.sps;
	s.num_short_term_ref_pic_sets = 1;
	s.st_ref_pic_set[0].num_negative_pics = 17;
	CHECK(!writeSps(s, out) && out.size() == 2, "a set of 17 pictures taken");
	s = c.sps;
	s.num_short_term_ref_pic_sets = 1;
	rps(s.st_ref_pic_set[0], {-1, -1}, {1, 1}, {}, {});
	CHECK(!writeSps(s, out) && out.size() == 2, "a set of one picture twice taken");
	Pps p = c.pps;
	p.num_tile_columns_minus1 = 20;
	CHECK(!writePps(p, out) && out.size() == 2, "21 tile columns taken");
	p = c.pps;
	p.num_ref_idx_l0_default_active_minus1 = 15;
	CHECK(!writePps(p, out) && out.size() == 2, "16 default references taken");
}

// The first slice segments of pictures 0 (IDR_N_LP), 1, 8, 9 and 15 of kvazaar 2.3.2's `--gop 0
// --ref 15` at 1280x720 (hevc-check.sh's kvz-720p-ref15), each its start code and first 24 bytes:
// low-delay P pictures that code their own short-term set, picture n using the n before it, up to
// 15. Its SPS: POC LSBs of 4 bits, no short-term set of its own, no long-term pictures; its PPS:
// id 0, no extra header bits, no output flag.
const std::vector<uint8_t> kRef15Pictures[] = {
	{0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x8c, 0x1f, 0x0d, 0x21, 0xf2, 0x03, 0xf9, 0x07, 0xb0,
	 0x0f, 0x37, 0xdf, 0x02, 0x38, 0xc4, 0xa4, 0x0f, 0x4c, 0xae, 0x73, 0x4b},
	{0x00, 0x00, 0x01, 0x02, 0x01, 0xe2, 0x5f, 0xf7, 0x18, 0x3d, 0xc6, 0x5b, 0x48, 0x16, 0x61,
	 0x0c, 0xaa, 0x59, 0x25, 0x32, 0xa9, 0x5f, 0x9d, 0x20, 0xf2, 0x50, 0x44},
	{0x00, 0x00, 0x01, 0x02, 0x01, 0xf0, 0x13, 0xff, 0xff, 0xf1, 0x02, 0x1e, 0x30, 0x7a, 0xf8,
	 0xa5, 0xd4, 0xeb, 0xb5, 0x57, 0x30, 0xae, 0x20, 0x5b, 0xda, 0xb7, 0xfe},
	{0x00, 0x00, 0x01, 0x02, 0x01, 0xf2, 0x15, 0xff, 0xff, 0xfc, 0x48, 0x97, 0x8c, 0x1e, 0xbf,
	 0x91, 0x79, 0xba, 0xee, 0xc5, 0xbf, 0xab, 0x93, 0x56, 0xf9, 0x2d, 0xc4},
	{0x00, 0x00, 0x01, 0x02, 0x01, 0xfe, 0x08, 0x7f, 0xff, 0xff, 0xff, 0xf1, 0xe3, 0xde, 0x30,
	 0x7a, 0xf3, 0x55, 0xad, 0x8b, 0x8c, 0x57, 0x4a, 0x2e, 0x22, 0x5b, 0xee},
};

// An access unit of one slice segment whose header BitWriter `w` wrote, of NAL unit type `type`:
// zeros before it, its start code, its header, then 16 bytes of slice data.
std::vector<uint8_t> sliceUnit(rgfx::video::BitWriter &w, uint32_t type)
{
	for (int i = 0; i < 16; i++)
		w.u(8, 0xa5);
	std::vector<uint8_t> out = {0, 0};
	w.appendNal({(uint8_t)(type << 1), 1}, out);
	return out;
}

// A PPS's narrowest tile column, as 6.5.1 lays the columns out over the SPS's CTBs.
void checkNarrowestTileColumn()
{
	Sps sps = {};
	Pps pps = {};
	sps.pic_width_in_luma_samples = 1280;
	sps.pic_height_in_luma_samples = 720;
	sps.log2_diff_max_min_luma_coding_block_size = 3; // CTBs of 64: 20 across
	uint32_t narrowest = 99;
	CHECK(narrowestTileColumn(sps, pps, &narrowest) && narrowest == 0, "no tiles: %u", narrowest);
	pps.tiles_enabled_flag = 1;
	struct Columns {
		uint8_t uniform;
		std::vector<uint32_t> minus1; // the PPS's widths, the last column's left out
		uint32_t want;                // 0: refused
	};
	for (const Columns &c : {
	         // kvz-720p-tiles-narrow's: 256, 192, 512 and the 320 left.
	         Columns{0, {3, 2, 7}, 192},
	         // kvz-720p-tiles-col64's: 64, 64, 1088 and 64.
	         Columns{0, {0, 0, 16}, 64},
	         // The last column the rest: 1216 and 64.
	         Columns{0, {18}, 64},
	         // Uniform: 10 of 2 CTBs; 3 of 6, 7 and 7 (6-3); 20 of one.
	         Columns{1, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 128},
	         Columns{1, {0, 0}, 384},
	         Columns{1, std::vector<uint32_t>(19), 64},
	         // Columns that leave the last none, or pass the picture; more columns than CTBs.
	         Columns{0, {9, 9}, 0},
	         Columns{0, {19}, 0},
	         Columns{0, {20}, 0},
	         Columns{1, std::vector<uint32_t>(20), 0},
	     }) {
		pps.uniform_spacing_flag = c.uniform;
		pps.num_tile_columns_minus1 = (uint32_t)c.minus1.size();
		for (size_t i = 0; i < c.minus1.size(); i++)
			pps.column_width_minus1[i] = c.minus1[i];
		const bool read = narrowestTileColumn(sps, pps, &narrowest);
		CHECK(c.want ? read && narrowest == c.want : !read,
		      "%u columns (uniform %u): %s, %u, not %u", pps.num_tile_columns_minus1 + 1, c.uniform,
		      read ? "read" : "refused", narrowest, c.want);
	}
	// A picture of no whole number of CTBs: its last, part, column counts whole, as
	// ColumnWidthInLumaSamples does. 1288 samples are 21 CTBs; 20 and 1.
	sps.pic_width_in_luma_samples = 1288;
	pps.uniform_spacing_flag = 0;
	pps.num_tile_columns_minus1 = 1;
	pps.column_width_minus1[0] = 19;
	CHECK(narrowestTileColumn(sps, pps, &narrowest) && narrowest == 64, "a part CTB: %u", narrowest);
	// CTBs of 16.
	sps.pic_width_in_luma_samples = 1280;
	sps.log2_diff_max_min_luma_coding_block_size = 1;
	pps.column_width_minus1[0] = 3;
	CHECK(narrowestTileColumn(sps, pps, &narrowest) && narrowest == 64, "CTBs of 16: %u", narrowest);
}

void checkCurrentReferences()
{
	using rgfx::video::BitWriter;
	Sps sps = {};
	Pps pps = {};
	sps.pic_width_in_luma_samples = 1280;
	sps.pic_height_in_luma_samples = 720;
	sps.log2_diff_max_min_luma_coding_block_size = 3;
	uint32_t count = 99;

	// kvazaar's: picture n uses the n before it.
	const uint32_t want[] = {0, 1, 8, 9, 15};
	for (size_t i = 0; i < 5; i++) {
		const std::vector<uint8_t> &au = kRef15Pictures[i];
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == want[i],
		      "ref15 picture %u: %u current references", want[i], count);
	}
	// Its header cut short of its set.
	CHECK(!currentReferences(sps, pps, kRef15Pictures[4].data(), 9, &count), "a cut header read");
	// Another PPS than its own.
	pps.pps_pic_parameter_set_id = 1;
	CHECK(!currentReferences(sps, pps, kRef15Pictures[3].data(), kRef15Pictures[3].size(), &count),
	      "a header of another PPS read");
	pps.pps_pic_parameter_set_id = 0;
	// No VCL unit, or none that starts a picture.
	const uint8_t sei[] = {0, 0, 1, 0x4e, 0x01, 0x05, 0x01, 0x00, 0x80};
	CHECK(!currentReferences(sps, pps, sei, sizeof sei, &count), "an SEI read as a slice");
	std::vector<uint8_t> second = kRef15Pictures[3];
	second[5] &= 0x7f; // first_slice_segment_in_pic_flag 0
	CHECK(!currentReferences(sps, pps, second.data(), second.size(), &count), "a later segment read");

	// A CRA picture (its no_output_of_prior_pics_flag), with extra header bits and an output flag,
	// using the SPS's second set by its index: its 3 used pictures of 4.
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	sps.num_short_term_ref_pic_sets = 3;
	rps(sps.st_ref_pic_set[0], {-1}, {1}, {}, {});
	rps(sps.st_ref_pic_set[1], {-1, -3}, {1, 0}, {2, 4}, {1, 1});
	rps(sps.st_ref_pic_set[2], {-2}, {1}, {}, {});
	pps.num_extra_slice_header_bits = 2;
	pps.output_flag_present_flag = 1;
	{
		BitWriter w;
		w.flag(1);   // first_slice_segment_in_pic_flag
		w.flag(0);   // no_output_of_prior_pics_flag
		w.ue(0);     // slice_pic_parameter_set_id
		w.u(2, 3);   // slice_reserved_flag[2]
		w.ue(1);     // slice_type P
		w.flag(1);   // pic_output_flag
		w.u(8, 200); // slice_pic_order_cnt_lsb
		w.flag(1);   // short_term_ref_pic_set_sps_flag
		w.u(2, 1);   // short_term_ref_pic_set_idx
		const std::vector<uint8_t> au = sliceUnit(w, 21);
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == 3,
		      "the SPS's set: %u", count);
	}
	// A set of its own predicted from the SPS's last, {-2 used}, by -1: the set's picture at -3,
	// unused, and the set's own at -1, used, so one.
	{
		BitWriter w;
		w.flag(1);
		w.ue(0);
		w.u(2, 0);
		w.ue(0);     // slice_type B
		w.flag(1);
		w.u(8, 7);
		w.flag(0);   // short_term_ref_pic_set_sps_flag
		w.flag(1);   // inter_ref_pic_set_prediction_flag
		w.ue(0);     // delta_idx_minus1: set 2
		w.flag(1);   // delta_rps_sign
		w.ue(0);     // abs_delta_rps_minus1: deltaRps -1
		w.flag(0);   // used_by_curr_pic_flag[0], -3
		w.flag(1);   // use_delta_flag[0]
		w.flag(1);   // used_by_curr_pic_flag[1], -1
		const std::vector<uint8_t> au = sliceUnit(w, 1);
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == 1,
		      "a predicted set: %u", count);
	}
	// Predicted from the SPS's second, {-1, -3 | 2, 4}, by +1: pictures at 0 (none), -2, 3, 5 and
	// the set's own at 1; all used but -2, so 3.
	{
		BitWriter w;
		w.flag(1);
		w.ue(0);
		w.u(2, 0);
		w.ue(0);
		w.flag(1);
		w.u(8, 7);
		w.flag(0);
		w.flag(1);
		w.ue(1); // delta_idx_minus1: set 1
		w.flag(0);
		w.ue(0); // deltaRps +1
		for (int used : {1, 0, 1, 1, 1}) {
			w.flag(used);
			if (!used)
				w.flag(1);
		}
		const std::vector<uint8_t> au = sliceUnit(w, 1);
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == 3,
		      "a predicted set with a picture at 0: %u", count);
	}
	// A set of its own of 2 and 1 pictures, 2 used, and long-term pictures: two of the SPS's,
	// the first used, and one of its own, used, with an MSB cycle; 4 in all.
	sps.long_term_ref_pics_present_flag = 1;
	sps.num_long_term_ref_pics_sps = 2;
	sps.used_by_curr_pic_lt_sps_flag[0] = 1;
	sps.used_by_curr_pic_lt_sps_flag[1] = 0;
	{
		BitWriter w;
		w.flag(1);
		w.ue(0);
		w.u(2, 0);
		w.ue(1);
		w.flag(1);
		w.u(8, 9);
		w.flag(0); // short_term_ref_pic_set_sps_flag
		w.flag(0); // inter_ref_pic_set_prediction_flag
		w.ue(2);   // num_negative_pics
		w.ue(1);   // num_positive_pics
		w.ue(0), w.flag(1), w.ue(1), w.flag(0), w.ue(0), w.flag(1);
		w.ue(2);    // num_long_term_sps
		w.ue(1);    // num_long_term_pics
		w.u(1, 0);  // lt_idx_sps[0]
		w.flag(0);  // delta_poc_msb_present_flag[0]
		w.u(1, 1);  // lt_idx_sps[1]
		w.flag(0);
		w.u(8, 50); // poc_lsb_lt[2]
		w.flag(1);  // used_by_curr_pic_lt_flag[2]
		w.flag(1);
		w.ue(3);    // delta_poc_msb_cycle_lt[2]
		const std::vector<uint8_t> au = sliceUnit(w, 1);
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == 4,
		      "long-term pictures: %u", count);
	}
	// An IDR picture's: none, before any set.
	{
		BitWriter w;
		w.flag(1);
		w.flag(0);
		w.ue(0);
		w.u(2, 0);
		w.ue(2);
		w.flag(1);
		const std::vector<uint8_t> au = sliceUnit(w, 19);
		CHECK(currentReferences(sps, pps, au.data(), au.size(), &count) && count == 0,
		      "an IDR picture: %u", count);
	}
	// A set index past the SPS's sets, and a prediction from past them.
	{
		BitWriter w;
		w.flag(1);
		w.ue(0);
		w.u(2, 0);
		w.ue(1);
		w.flag(1);
		w.u(8, 9);
		w.flag(1);
		w.u(2, 3);
		const std::vector<uint8_t> au = sliceUnit(w, 1);
		CHECK(!currentReferences(sps, pps, au.data(), au.size(), &count), "set 3 of 3 read");
	}
	{
		BitWriter w;
		w.flag(1);
		w.ue(0);
		w.u(2, 0);
		w.ue(1);
		w.flag(1);
		w.u(8, 9);
		w.flag(0);
		w.flag(1);
		w.ue(3); // delta_idx_minus1: set -1
		const std::vector<uint8_t> au = sliceUnit(w, 1);
		CHECK(!currentReferences(sps, pps, au.data(), au.size(), &count), "a prediction from set -1 read");
	}
}

// What the driver answers AppleGVA for a decode it could not complete (RgfxVideoStatus.hpp): the
// schedule of a command it did not send fails, and so does the schedule of an HEVC picture the
// host failed, which waits for the host's verdict, the one answer AppleGVA's HEVC decoder
// reports; an H.264 schedule waits for none, as AppleGVA's H.264 decoder reports none; a query
// answers a command running or done as it is.
void checkStatus()
{
	using rgfx::video::queryStatus;
	using rgfx::video::scheduleFails;

	CHECK(queryStatus(0) == rgfx::video::kStatusRunning, "a running command");
	CHECK(queryStatus(1) == rgfx::video::kStatusDone, "a decoded command");
	CHECK(queryStatus(-1) == rgfx::video::kStatusDone, "a failure on the host");

	CHECK(!scheduleFails(true, false), "a sent command fails its schedule");
	CHECK(scheduleFails(false, false), "a command not sent is scheduled");
	CHECK(scheduleFails(true, true), "a picture the host failed is scheduled");
	CHECK(scheduleFails(false, true), "a command not sent beside a failed one is scheduled");
	CHECK(rgfx::video::waitsForHost(true), "an HEVC schedule waits for no verdict");
	CHECK(!rgfx::video::waitsForHost(false), "an H.264 schedule waits for the host");
	CHECK(rgfx::video::kNotSent != 0, "the schedule's failure is no success");
}

} // namespace

int main()
{
	for (auto make : kCases)
		checkCase(make());
	checkEscapes();
	checkRefusals();
	checkCurrentReferences();
	checkNarrowestTileColumn();
	checkStatus();
	std::printf("video_h265_tests: %d checks passed over %zu streams\n", g_checks, kCases.size());
	return 0;
}
