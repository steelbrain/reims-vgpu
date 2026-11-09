// The native test of RgfxLegacySwapInfo.hpp and the swap-info columns of RgfxLegacySwapLayout.hpp:
// each release's offsets as its AppleParavirtGPU's notify_swap_done_gated, do_swap_info_notify_gated
// and set_notify_tag_mask_gated and its IOMobileGraphicsFamily's note_vbl_ISR, swap_submit,
// abort_swap_gated and dropSwap_gated use them; the record the paravirtual display makes; the
// async result's values; and the records kept while the class holds them.
#include "RgfxLegacySwapInfo.hpp"
#include "RgfxLegacySwapLayout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int checks = 0;

void check(bool ok, const char *what, const char *about)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "legacy_swap_info_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

// What each release's kexts read and write, by byte offset: in the class, the time note_vbl_ISR
// writes and the halfword notify_swap_done_gated tests; in a swap, the doubleword swap_submit
// copies from its record and the word abort_swap_gated sets bit 0 of, both of which
// notify_swap_done_gated reads, with the state word it tests; and in a notify request, the
// async reference, tag mask and last time do_swap_info_notify_gated and set_notify_tag_mask_gated
// use. Read from 14.6.1, 15.6.1, 26.6.2 and 27.0.1.
struct Release {
	const char *name;
	uint32_t classSize, vblankTime, notifyHeld;
	uint32_t swapState, swapRecordField00, swapFlags;
	uint32_t requestReference, requestMask, requestLast;
};

constexpr Release kReleases[] = {
    {"14", 0x18a0, 0x108, 0x189e, 0x32c, 0x2d0, 0x310, 0x18, 0x78, 0x80},
    {"15", 0x19c0, 0x108, 0x19be, 0x32c, 0x2d0, 0x310, 0x18, 0x78, 0x80},
    {"26", 0x1af8, 0x108, 0x1af6, 0x34c, 0x2f0, 0x330, 0x18, 0x78, 0x80},
    {"27", 0x1dd8, 0x108, 0x1dd2, 0x3ec, 0x390, 0x3d0, 0x18, 0x78, 0x80},
};

void layouts()
{
	for (const Release &r : kReleases) {
		const RgfxLegacySwapLayout *l = rgfxLegacySwapLayoutOf(r.classSize);
		check(l != nullptr, "the release has a layout", r.name);
		check(l->vblankTime == r.vblankTime, "the class's vblank time", r.name);
		check(l->notifyHeld == r.notifyHeld, "the class's held halfword", r.name);
		check(l->state == r.swapState, "the swap's state word", r.name);
		check(l->swapRecordField00 == r.swapRecordField00, "the swap's copy of field_00", r.name);
		check(l->swapFlags == r.swapFlags, "the swap's flags word", r.name);
		check(kRgfxNotifyRequestReference == r.requestReference &&
		          kRgfxNotifyRequestMask == r.requestMask &&
		          kRgfxNotifyRequestLast == r.requestLast,
		      "the notify request's offsets", r.name);
		// Where they lie: aligned, in the class, and none of them the queue or the size.
		check(l->vblankTime % 8 == 0 && l->swapRecordField00 % 8 == 0, "doublewords are aligned",
		      r.name);
		check(l->swapFlags % 4 == 0 && l->notifyHeld % 2 == 0, "words are aligned", r.name);
		check(l->notifyHeld + 2 <= l->classSize, "the halfword lies in the class", r.name);
		check(l->notifyHeld + 2 <= l->current || l->notifyHeld >= l->current + 24,
		      "the halfword is not the queue", r.name);
		check(l->notifyHeld + 2 <= l->displaySize || l->notifyHeld >= l->displaySize + 8,
		      "the halfword is not the display's size", r.name);
		check(l->swapRecordField00 + 8 <= l->swapFlags && l->swapFlags + 4 <= l->state,
		      "the swap's fields come before its state word", r.name);
	}
}

void record()
{
	const RgfxSwapInfoRecord r = rgfxSwapInfoRecord(77, 0x1122334455667788ull, 0xabcdef, 0x80000001u);
	check(r.field_00 == 77 && r.field_04 == 0, "the swap's ID first", "record");
	check(r.field_08 == 0x1122334455667788ull && r.field_10 == 0, "the vblank time, then 0", "record");
	check(r.field_18 == 0xabcdef && r.field_20 == 0x80000001u && r.field_28 == 0,
	      "the swap's field_00 copy, its flags zero-extended, then 0", "record");
	uint8_t bytes[sizeof r];
	std::memcpy(bytes, &r, sizeof r);
	uint64_t at18;
	std::memcpy(&at18, bytes + 0x18, 8);
	check(at18 == 0xabcdef, "field_18 is at 0x18", "record");
}

void args()
{
	const RgfxSwapInfoRecord r = rgfxSwapInfoRecord(9, 0x0000000500000010ull, 0x0000000700000020ull, 3);
	uint64_t a[kRgfxSwapInfoMaxArgs + 1];

	uint64_t last = 0x0000000500000000ull;
	uint32_t n = rgfxSwapInfoArgs(0x1f, &last, r, a);
	check(n == 12, "five values in two words each, after the mask and the ID", "args");
	check(a[0] == 0x1f && a[1] == 9, "the mask, then the swap's ID", "args");
	check(a[2] == 0x10 && a[3] == 5, "field_08, low word first", "args");
	check(a[4] == 0x10 && a[5] == 0, "the second value is field_08 less the last", "args");
	check(a[6] == 0x20 && a[7] == 7, "field_18", "args");
	check(a[8] == 3 && a[9] == 0 && a[10] == 0 && a[11] == 0, "field_20, then field_28", "args");
	check(last == r.field_08, "the request's last becomes field_08", "args");

	n = rgfxSwapInfoArgs(0x80000000u | 0x1f, &last, r, a);
	check(n == 7, "with bit 31, one argument a value", "args");
	check(a[0] == (0x80000000u | 0x1f) && a[2] == r.field_08 && a[3] == 0 &&
	          a[4] == r.field_18 && a[5] == 3 && a[6] == 0,
	      "the values whole, the second 0 against itself", "args");

	last = 0x10;
	n = rgfxSwapInfoArgs(0x2, &last, r, a);
	check(n == 4 && a[2] == static_cast<uint32_t>(r.field_08 - 0x10) && a[3] == (r.field_08 - 0x10) >> 32,
	      "a mask's bits pick the values", "args");
	n = rgfxSwapInfoArgs(0, &last, r, a);
	check(n == 2 && a[0] == 0 && a[1] == 9, "no bit, the mask and the ID alone", "args");
	n = rgfxSwapInfoArgs(0x60, &last, r, a);
	check(n == 2, "bits past 4 pick nothing", "args");
}

void kept()
{
	RgfxSwapInfoKept k = {};
	std::vector<uint32_t> seen;
	auto collect = [&seen](RgfxSwapInfoRecord &r) { seen.push_back(r.field_00); };
	k.flush(collect);
	check(seen.empty(), "none kept, none dispatched", "kept");
	for (uint32_t id = 1; id <= 3; id++)
		k.keep(rgfxSwapInfoRecord(id, 0, 0, 0));
	k.flush(collect);
	check(seen == std::vector<uint32_t>({1, 2, 3}), "the kept records, oldest first", "kept");
	seen.clear();
	k.flush(collect);
	check(seen.empty(), "a flush keeps none", "kept");
	for (uint32_t id = 1; id <= kRgfxSwapInfoKept; id++)
		k.keep(rgfxSwapInfoRecord(id, 0, 0, 0));
	k.flush(collect);
	check(seen.empty(), "six kept read as none, as the paravirtual display's ring does", "kept");
	for (uint32_t id = 1; id <= kRgfxSwapInfoKept + 2; id++)
		k.keep(rgfxSwapInfoRecord(id, 0, 0, 0));
	k.flush(collect);
	check(seen == std::vector<uint32_t>({7, 8}), "past six, the slots written last", "kept");
}

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), "the file can be read", path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The dispatch is the class's, called by the name each release exports.
void imports()
{
	const std::string source = readFile("src/ReimsGraphicsMobile.cpp");
	check(source.find("\"__ZN25IOMobileFramebufferLegacy31swap_info_notify_dispatch_gatedEPNS_"
	                  "12SwapInfoBlobE\"") != std::string::npos,
	      "the source calls the class's dispatch", "imports");
	check(source.find("sendAsyncResult64WithOptions") != std::string::npos &&
	          source.find("kIOUserNotifyOptionCanDrop") != std::string::npos,
	      "the result is sent as one the client may drop", "imports");
}

} // namespace

int main()
{
	layouts();
	record();
	args();
	kept();
	imports();
	std::printf("legacy_swap_info_test: %d checks passed (%zu releases)\n", checks,
	            sizeof(kReleases) / sizeof(kReleases[0]));
	return 0;
}
