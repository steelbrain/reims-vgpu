// The native test of RgfxLegacySwapInfo.hpp and the swap-info columns of RgfxLegacySwapLayout.hpp:
// each release's offsets as its AppleParavirtGPU's notify_swap_done_gated, do_swap_info_notify_gated
// and set_notify_tag_mask_gated and its IOMobileGraphicsFamily's note_vbl_ISR, swap_submit,
// abort_swap_gated and dropSwap_gated use them; the record the paravirtual display makes; its
// words as IOMobileFramebuffer.framework reads them from the client's shared data queue; the frame
// info tags the device answers and its presentation time; and the records kept while the class
// holds them.
#include "RgfxLegacySwapInfo.hpp"
#include "RgfxLegacySwapLayout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
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
// notify_swap_done_gated reads, with the state word it tests; and in a notify request, the user
// client add_notification_gated stores, and the tag mask and last time do_swap_info_notify_gated
// and set_notify_tag_mask_gated use. Read from 14.6.1, 15.6.1, 26.6.2 and 27.0.1.
struct Release {
	const char *name;
	uint32_t classSize, vblankTime, notifyHeld;
	uint32_t swapState, swapRecordField00, swapFlags;
	uint32_t requestClient, requestMask, requestLast;
};

constexpr Release kReleases[] = {
    {"14", 0x18a0, 0x108, 0x189e, 0x32c, 0x2d0, 0x310, 0x10, 0x78, 0x80},
    {"15", 0x19c0, 0x108, 0x19be, 0x32c, 0x2d0, 0x310, 0x10, 0x78, 0x80},
    {"26", 0x1af8, 0x108, 0x1af6, 0x34c, 0x2f0, 0x330, 0x10, 0x78, 0x80},
    {"27", 0x1dd8, 0x108, 0x1dd2, 0x3ec, 0x390, 0x3d0, 0x10, 0x78, 0x80},
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
		check(kRgfxNotifyRequestClient == r.requestClient &&
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
	check(r.field_08 == 0x1122334455667788ull && r.field_10 == 0, "the presentation time, then 0",
	      "record");
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

	// As FrameInfoNotifyFunc reads them: with the mask's bit 31 clear, one argument a value.
	uint64_t last = 0x0000000500000000ull;
	uint32_t n = rgfxSwapInfoArgs(0x1f, &last, r, a);
	check(n == 7, "five values in one argument each, after the mask and the ID", "args");
	check(a[0] == 0x1f && a[1] == 9, "the mask, then the swap's ID", "args");
	check(a[2] == r.field_08, "field_08 whole", "args");
	check(a[3] == 0x10, "the second value is field_08 less the last", "args");
	check(a[4] == r.field_18 && a[5] == 3 && a[6] == 0, "field_18, field_20, then field_28", "args");
	check(last == r.field_08, "the request's last becomes field_08", "args");

	// With bit 31, two arguments a value, its low and then its high word.
	last = 0x0000000500000000ull;
	n = rgfxSwapInfoArgs(0x80000000u | 0x1f, &last, r, a);
	check(n == 12, "with bit 31, two arguments a value, the fifth left out", "args");
	check(a[0] == (0x80000000u | 0x1f) && a[1] == 9, "the mask as asked, then the ID", "args");
	check(a[2] == 0x10 && a[3] == 5, "field_08, low word first", "args");
	check(a[4] == 0x10 && a[5] == 0, "the difference, low word first", "args");
	check(a[6] == 0x20 && a[7] == 7, "field_18", "args");
	check(a[8] == 3 && a[9] == 0 && a[10] == 0 && a[11] == 0, "field_20, then field_28", "args");

	last = 0x10;
	n = rgfxSwapInfoArgs(0x2, &last, r, a);
	check(n == 3 && a[2] == r.field_08 - 0x10, "a mask's bits pick the values", "args");
	n = rgfxSwapInfoArgs(0, &last, r, a);
	check(n == 2 && a[0] == 0 && a[1] == 9, "no bit, the mask and the ID alone", "args");
	n = rgfxSwapInfoArgs(0x60, &last, r, a);
	check(n == 2, "bits past 4 pick nothing", "args");
}

// The frame info the device answers: two tags, by index, whose bits pick the record's
// presentation time and its difference from the one sent before, one argument each, as
// IOMobileFramebuffer.framework builds the client's dictionary from them.
void frameInfo()
{
	check(kRgfxFrameInfoTagCount == 2, "two tags", "frameInfo");
	check(std::strcmp(rgfxFrameInfoTag(0), "Presentation_time") == 0,
	      "tag 0, QuartzCore's presentation time", "frameInfo");
	check(std::strcmp(rgfxFrameInfoTag(1), "Presentation_delta") == 0, "tag 1, its difference",
	      "frameInfo");
	check(rgfxFrameInfoTag(2) == nullptr && rgfxFrameInfoTag(31) == nullptr, "none past them",
	      "frameInfo");
	for (uint32_t i = 0; i < kRgfxFrameInfoTagCount; i++)
		check(std::strlen(rgfxFrameInfoTag(i)) < 64, "a name fits the 64 bytes asked", "frameInfo");

	// A swap shown at 1000 absolute, the continuous clock 40 ahead: 1040; a swap that showed
	// nothing, 0.
	check(rgfxFrameInfoTime(1000, 5000, 5040) == 1040, "the shown time in the continuous clock",
	      "frameInfo");
	check(rgfxFrameInfoTime(1000, 5000, 5000) == 1000, "the clocks equal, the time as it is",
	      "frameInfo");
	check(rgfxFrameInfoTime(0, 5000, 5040) == 0, "nothing shown, 0", "frameInfo");

	// The clocks read together: the continuous one 40 ahead of the absolute one, which ticks
	// at the reads of it `ticks` says (from 1). Reads that agree give the lead exactly, the reads
	// before them taken again where a tick fell between; where none agree, the last try's later
	// read.
	struct Clocks {
		std::vector<uint64_t> ticks;
		uint64_t now = 5000, reads = 0, continuousReads = 0;
		uint64_t absolute()
		{
			reads++;
			for (uint64_t t : ticks)
				now += t == reads ? 1 : 0;
			return now;
		}
		uint64_t continuous()
		{
			continuousReads++;
			return now + 40;
		}
	};
	const std::vector<std::vector<uint64_t>> agreeing = {{}, {3}, {2}, {2, 4}, {1, 2, 4, 6}};
	for (const std::vector<uint64_t> &ticks : agreeing) {
		Clocks k{ticks};
		uint64_t a = 0, c = 0;
		rgfxClocksTogether([&] { return k.absolute(); }, [&] { return k.continuous(); }, &a, &c);
		check(c - a == 40 && rgfxFrameInfoTime(1000, a, c) == 1040,
		      "reads that agree give the lead exactly", "frameInfo");
	}
	{
		Clocks k{};
		for (uint64_t r = 2; r <= 2 * kRgfxClockReads; r += 2)
			k.ticks.push_back(r); // a tick between the two reads of every try
		uint64_t a = 0, c = 0;
		rgfxClocksTogether([&] { return k.absolute(); }, [&] { return k.continuous(); }, &a, &c);
		check(k.continuousReads == kRgfxClockReads && c - a == 39,
		      "none agree: every try made, the last one's later read", "frameInfo");
	}

	// QuartzCore's mask for both tags: bits 0 and 1, the presentation time, then its difference.
	const uint64_t first = rgfxFrameInfoTime(1000, 5000, 5040);
	const uint64_t second = rgfxFrameInfoTime(1400, 5400, 5440);
	uint64_t a[kRgfxSwapInfoMaxArgs + 1];
	uint64_t last = 0;
	rgfxSwapInfoArgs(0x3, &last, rgfxSwapInfoRecord(11, first, 0, 0), a);
	const uint32_t n = rgfxSwapInfoArgs(0x3, &last, rgfxSwapInfoRecord(12, second, 0, 0), a);
	check(n == 4 && a[0] == 0x3 && a[1] == 12, "the mask and the swap's ID", "frameInfo");
	check(a[2] == 1440 && a[3] == 400, "Presentation_time, then Presentation_delta", "frameInfo");
}

// What IOMobileFramebuffer.framework makes of an entry of the client's shared data queue
// (FrameInfoNotifyFuncIOShq, PackAndDispatchFrameInfoNotifictionData; read from 14.6.1's and
// 27.0.1's): the entry's bytes copied into a buffer of 0x100 bytes; word 0 the mask, whose bit 31
// makes a value two words, low first; word 1's low half the swap's ID; then a value for each bit
// of the mask, in the bits' order, under the name of the tag of the bit's index.
struct Parsed {
	uint32_t swapId = 0;
	std::map<std::string, uint64_t> values;
};

Parsed readEntry(const uint64_t *words, uint32_t bytes, const char *about)
{
	check(bytes >= 16 && bytes % 8 == 0 && bytes <= 0x100, "an entry of whole words that fits",
	      about);
	uint64_t buffer[0x100 / 8] = {};
	std::memcpy(buffer, words, bytes);
	Parsed p;
	const uint64_t first = buffer[0];
	const uint32_t mask = static_cast<uint32_t>(first) & 0x7fffffffu;
	p.swapId = static_cast<uint32_t>(buffer[1]);
	uint32_t at = 2;
	for (uint32_t i = 0; i < 31; i++) {
		if ((mask & (1u << i)) == 0)
			continue;
		uint64_t v = buffer[at++];
		if (first & 0x80000000u)
			v |= buffer[at++] << 32;
		const char *tag = rgfxFrameInfoTag(i);
		p.values[tag ? tag : "none"] = v;
	}
	return p;
}

// Each release's notify request, as do_swap_info_notify_gated reads it, gives the record
// IOMobileFramebuffer.framework reads QuartzCore's two tags from: the swap's ID, its presentation
// time and the time since the last record.
void queueRecord()
{
	check(kRgfxSwapInfoMaxArgs * sizeof(uint64_t) <= 0x100, "the longest record fits the copy",
	      "queueRecord");
	for (const Release &r : kReleases) {
		alignas(8) uint8_t request[0x88] = {};
		const uint32_t mask = 0x3;
		std::memcpy(request + r.requestMask, &mask, sizeof(mask));
		uint64_t words[kRgfxSwapInfoMaxArgs + 1];
		const uint64_t first = rgfxFrameInfoTime(1000, 5000, 5040);
		const uint64_t second = rgfxFrameInfoTime(1400, 5400, 5440);
		const uint32_t n1 = rgfxFrameInfoRecord(request, rgfxSwapInfoRecord(11, first, 7, 1), words);
		Parsed p = readEntry(words, n1 * 8, r.name);
		check(n1 == 4 && p.swapId == 11, "the first swap's record", r.name);
		check(p.values.size() == 2 && p.values["Presentation_time"] == 1040 &&
		          p.values["Presentation_delta"] == 1040,
		      "its time, and its time since none", r.name);
		uint64_t last = 0;
		std::memcpy(&last, request + r.requestLast, sizeof(last));
		check(last == 1040, "the request keeps the time sent", r.name);
		const uint32_t n2 =
		    rgfxFrameInfoRecord(request, rgfxSwapInfoRecord(12, second, 7, 1), words);
		p = readEntry(words, n2 * 8, r.name);
		check(n2 == 4 && p.swapId == 12, "the next swap's record", r.name);
		check(p.values["Presentation_time"] == 1440 && p.values["Presentation_delta"] == 400,
		      "Presentation_time, then Presentation_delta, in the tags' order", r.name);
		const uint32_t n3 = rgfxFrameInfoRecord(request, rgfxSwapInfoRecord(13, 0, 7, 1), words);
		p = readEntry(words, n3 * 8, r.name);
		check(p.swapId == 13 && p.values["Presentation_time"] == 0,
		      "a swap that showed nothing reads 0", r.name);
	}
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
	check(source.find("\"__ZN29IOMobileFramebufferUserClient26enqueueInIOSharedDataQueueEPyj\"") !=
	          std::string::npos,
	      "the record goes in the user client's shared data queue", "imports");
	check(source.find("*dataQueue = 1;") != std::string::npos,
	      "supported_frame_info answers the records come in a data queue", "imports");
	check(source.find("sendAsyncResult") == std::string::npos, "no async result", "imports");
	const std::string graft = readFile("src/graft-arm64e.json");
	const size_t policy =
	    graft.find("\"__ZN29IOMobileFramebufferUserClient26enqueueInIOSharedDataQueueEPyj\": {");
	check(policy != std::string::npos &&
	          graft.find("\"library\": 1", policy) == graft.find("\"library\"", policy),
	      "Graft binds the enqueue from IOMobileGraphicsFamily", "imports");
	check(source.find("rgfxFrameInfoTag(index)") != std::string::npos,
	      "supported_frame_info answers the tags", "imports");
	check(source.find("rgfxClocksTogether([] { return mach_absolute_time(); }, [] { return "
	                  "mach_continuous_time(); },") != std::string::npos &&
	          source.find("rgfxFrameInfoTime(shown, absolute, continuous)") != std::string::npos,
	      "the presentation time goes in mach_continuous_time's clock, the clocks read together",
	      "imports");
}

} // namespace

int main()
{
	layouts();
	record();
	args();
	frameInfo();
	queueRecord();
	kept();
	imports();
	std::printf("legacy_swap_info_test: %d checks passed (%zu releases)\n", checks,
	            sizeof(kReleases) / sizeof(kReleases[0]));
	return 0;
}
