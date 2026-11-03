/*
 * video_h265_tests: the video driver's HEVC parameter-set writer (src/RgfxVideoH265.*), run on
 * the Mac (guest/bundle/Makefile `test`). For each stream of video_h265_cases.hpp it writes the
 * SPS and PPS from the fields AppleGVA's parsed form holds and checks them against the NAL units
 * the encoder wrote: the SPS bit for bit up to its VUI flag, which ours clears, and the PPS byte
 * for byte. Where the encoder predicted a short-term set from another, which ours writes by its
 * pictures, the SPS's sets are read back instead and the bits after them compared. It also checks
 * the emulation prevention of a large field, and the refusals.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

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

// What the driver answers AppleGVA for a decode it could not complete (RgfxVideoStatus.hpp): the
// schedule of a command it did not send fails, and so does a session's next schedule after a
// failure on the host, the one answer AppleGVA reports; a decoded or still-running command is
// answered as it always was.
void checkStatus()
{
	using rgfx::video::queryReport;
	using rgfx::video::QueryReport;
	using rgfx::video::scheduleFails;

	QueryReport r = queryReport(0);
	CHECK(r.status == rgfx::video::kStatusRunning && !r.hostFailed, "a running command");
	r = queryReport(1);
	CHECK(r.status == rgfx::video::kStatusDone && !r.hostFailed, "a decoded command");
	r = queryReport(-1);
	CHECK(r.status == rgfx::video::kStatusDone && r.hostFailed, "a failure on the host");

	CHECK(!scheduleFails(true, false), "a sent command fails its schedule");
	CHECK(scheduleFails(false, false), "a command not sent is scheduled");
	CHECK(scheduleFails(true, true), "a schedule after a failure on the host succeeds");
	CHECK(scheduleFails(false, true), "a command not sent after a failure is scheduled");
	CHECK(rgfx::video::kNotSent != 0, "the schedule's failure is no success");
}

} // namespace

int main()
{
	for (auto make : kCases)
		checkCase(make());
	checkEscapes();
	checkRefusals();
	checkStatus();
	std::printf("video_h265_tests: %d checks passed over %zu streams\n", g_checks, kCases.size());
	return 0;
}
