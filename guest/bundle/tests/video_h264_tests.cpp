/*
 * video_h264_tests: the video driver's H.264 parameter-set writer (src/RgfxVideoH264.*), run on
 * the Mac (guest/bundle/Makefile `test`). For each stream of video_h264_cases.hpp it writes the
 * SPS and PPS from the fields AppleGVA's parsed form holds and checks them against the NAL units
 * the encoder wrote: the SPS bit for bit up to its VUI flag, which ours clears, and the PPS byte
 * for byte. It also checks the emulation prevention of a large field, and the refusals.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "RgfxVideoH264.hpp"

using namespace rgfx::video::h264;

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
	std::vector<uint8_t> sps_nal; // header byte first, no start code, as the encoder escaped it
	std::vector<uint8_t> pps_nal;
	uint32_t sps_vui_bit;
};

template <size_t N> void fill(uint32_t (&list)[N], std::initializer_list<uint32_t> values)
{
	size_t i = 0;
	for (uint32_t v : values)
		list[i++] = v;
}

#include "video_h264_cases.hpp"

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
	return (bytes[bit / 8] >> (7 - bit % 8)) & 1;
}

// The written NAL unit after its four-byte start code.
std::vector<uint8_t> payload(const std::vector<uint8_t> &written)
{
	CHECK(written.size() > 5 && written[0] == 0 && written[1] == 0 && written[2] == 0 && written[3] == 1,
	      "no start code");
	return std::vector<uint8_t>(written.begin() + 4, written.end());
}

// No 0x000000, 0x000001 or 0x000002 left (0x000003 is the escape itself).
void checkEscaped(const std::vector<uint8_t> &nal, const char *name)
{
	for (size_t i = 2; i < nal.size(); i++)
		CHECK(!(nal[i - 2] == 0 && nal[i - 1] == 0 && nal[i] <= 2), "%s: unescaped run at %zu", name, i);
}

void checkCase(const Case &c)
{
	std::vector<uint8_t> written;
	CHECK(writeSps(c.sps, written), "%s: SPS refused", c.name);
	std::vector<uint8_t> nal = payload(written);
	checkEscaped(nal, c.name);
	std::vector<uint8_t> ours = unescape(nal.data(), nal.size());
	std::vector<uint8_t> theirs = unescape(c.sps_nal.data(), c.sps_nal.size());
	for (uint32_t bit = 0; bit < c.sps_vui_bit; bit++)
		CHECK(bitAt(ours, bit) == bitAt(theirs, bit), "%s: SPS bit %u differs", c.name, bit);
	CHECK(bitAt(theirs, c.sps_vui_bit) == 1, "%s: the stream's SPS has no VUI", c.name);
	// Ours: the VUI flag clear, the stop bit, and zeros to the byte's end.
	CHECK(bitAt(ours, c.sps_vui_bit) == 0, "%s: VUI flag set", c.name);
	CHECK(bitAt(ours, c.sps_vui_bit + 1) == 1, "%s: no stop bit", c.name);
	CHECK(ours.size() == (c.sps_vui_bit + 2 + 7) / 8, "%s: SPS of %zu bytes", c.name, ours.size());
	for (uint32_t bit = c.sps_vui_bit + 2; bit < ours.size() * 8; bit++)
		CHECK(bitAt(ours, bit) == 0, "%s: SPS trailing bit %u set", c.name, bit);

	written.clear();
	CHECK(writePps(c.pps, c.sps, written), "%s: PPS refused", c.name);
	nal = payload(written);
	CHECK(nal == c.pps_nal, "%s: PPS of %zu bytes differs from the stream's %zu", c.name, nal.size(),
	      c.pps_nal.size());

	// Both appended after what out held.
	std::vector<uint8_t> spsWritten, ppsWritten, both = {0xaa};
	CHECK(writeSps(c.sps, spsWritten) && writePps(c.pps, c.sps, ppsWritten), "%s: refused", c.name);
	CHECK(writeSps(c.sps, both) && writePps(c.pps, c.sps, both), "%s: refused", c.name);
	CHECK(both.size() == 1 + spsWritten.size() + ppsWritten.size() && both[0] == 0xaa &&
		      std::memcmp(both.data() + 1, spsWritten.data(), spsWritten.size()) == 0,
	      "%s: not appended", c.name);
}

void checkEscapes()
{
	// Profile 66 with no flags and level 0 makes the second and third bytes zero; then come five
	// ue(0) (the SPS id, frame-num length, POC type 0, its lsb length, the references) and the
	// gaps flag, and the width's ue with 24 leading zeros, which runs from bit 30 to bit 53: two
	// zero bytes, then 0x02 or 0x03.
	Sps s = {};
	s.profile_idc = 66;
	s.pic_width_in_mbs_minus1 = 0xffffff;
	s.frame_mbs_only_flag = 1;
	std::vector<uint8_t> written;
	CHECK(writeSps(s, written), "large width refused");
	std::vector<uint8_t> nal = payload(written);
	checkEscaped(nal, "large width");
	std::vector<uint8_t> rbsp = unescape(nal.data(), nal.size());
	CHECK(rbsp.size() < nal.size(), "nothing escaped");
	CHECK(rbsp[0] == 0x67 && rbsp[1] == 66 && rbsp[2] == 0 && rbsp[3] == 0, "SPS head");
	uint32_t bit = 8 + 24 + 5 + 1;
	uint32_t zeros = 0;
	while (bitAt(rbsp, bit) == 0) {
		zeros++;
		bit++;
	}
	uint32_t value = 0;
	for (uint32_t i = 0; i <= zeros; i++)
		value = (value << 1) | bitAt(rbsp, bit++);
	CHECK(zeros == 24 && value - 1 == 0xffffff, "width read back as %u zeros, %#x", zeros, value - 1);
}

void checkRefusals()
{
	Case c = case_high();
	std::vector<uint8_t> out = {1, 2};
	Sps s = c.sps;
	s.chroma_format_idc = 3;
	CHECK(!writeSps(s, out) && out.size() == 2, "4:4:4 taken");
	s = c.sps;
	s.pic_order_cnt_type = 3;
	CHECK(!writeSps(s, out) && out.size() == 2, "POC type 3 taken");
	Pps p = c.pps;
	p.num_slice_groups_minus1 = 1;
	CHECK(!writePps(p, c.sps, out) && out.size() == 2, "slice groups taken");
	p = c.pps;
	p.weighted_bipred_idc = 3;
	CHECK(!writePps(p, c.sps, out) && out.size() == 2, "bipred 3 taken");
}

} // namespace

int main()
{
	for (auto make : kCases)
		checkCase(make());
	checkEscapes();
	checkRefusals();
	std::printf("video_h264_tests: %d checks passed over %zu streams\n", g_checks, kCases.size());
	return 0;
}
