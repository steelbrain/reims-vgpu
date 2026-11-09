// The native test of RgfxLegacySwapInfo.hpp and the swap-info columns of RgfxLegacySwapLayout.hpp:
// each release's offsets as its AppleParavirtGPU's notify_swap_done_gated, do_swap_info_notify_gated
// and set_notify_tag_mask_gated and its IOMobileGraphicsFamily's note_vbl_ISR, swap_submit,
// abort_swap_gated and dropSwap_gated use them; the record the paravirtual display makes; its
// words as IOMobileFramebuffer.framework reads them from the client's shared data queue; the frame
// info tags the device answers and its presentation times, over a long run of refreshes as the
// display's clocks give them; and the records kept while the class holds them.
#include "RgfxLegacySwapInfo.hpp"
#include "RgfxLegacySwapLayout.hpp"
#include "RgfxVblankClock.hpp"

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
	check(rgfxFrameInfoTime(1000, 40) == 1040, "the shown time in the continuous clock",
	      "frameInfo");
	check(rgfxFrameInfoTime(1000, 0) == 1000, "no lead, the time as it is", "frameInfo");
	check(rgfxFrameInfoTime(0, 40) == 0, "nothing shown, 0", "frameInfo");

	// The clocks read: the continuous one 40 ahead of the absolute one, which ticks at the reads
	// of it `ticks` says (from 1). Reads that agree give the lead exactly; where none agree, the
	// lead is at least the last continuous read less the later absolute read.
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
		RgfxFrameInfoClock clock;
		rgfxReadClocks([&] { return k.absolute(); }, [&] { return k.continuous(); }, clock);
		check(clock.lead() == 40 && clock.time(1000, 0) == 1040,
		      "reads that agree give the lead exactly", "frameInfo");
	}
	{
		Clocks k{};
		for (uint64_t r = 2; r <= 2 * kRgfxClockReads; r += 2)
			k.ticks.push_back(r); // a tick between the continuous read and the later absolute one
		RgfxFrameInfoClock clock;
		rgfxReadClocks([&] { return k.absolute(); }, [&] { return k.continuous(); }, clock);
		check(k.continuousReads == kRgfxClockReads && clock.lead() == 39,
		      "none agree: every try made, the lead at least the continuous less the later read",
		      "frameInfo");
		check(clock.time(1000, 0) == 1039, "a time short by the tick", "frameInfo");
		// A later reading that agrees: the lead exact, the time with it.
		Clocks again{};
		again.now = 9000;
		rgfxReadClocks([&] { return again.absolute(); }, [&] { return again.continuous(); },
		               clock);
		check(clock.lead() == 40 && clock.time(1000, 0) == 1040, "the lead exact once read so",
		      "frameInfo");
		// A reading short again leaves the lead: it never shrinks.
		clock.read(20000, 20039, 20001);
		check(clock.lead() == 40, "a lower bound under the lead leaves it", "frameInfo");
		// The system slept 500 ticks: the lead grows with the next reading.
		clock.read(30000, 30540, 30000);
		check(clock.lead() == 540 && clock.time(2000, 0) == 2540, "a longer lead after a sleep",
		      "frameInfo");
	}
	{
		// Each time is a refresh, 400 ticks, after the one given before: a swap whose refresh is
		// not after the last one's is given the refresh after that, up to two periods past its
		// own, then the last one's; a swap that showed nothing reads 0 and changes nothing.
		RgfxFrameInfoClock clock;
		clock.read(5000, 5040, 5000);
		check(clock.time(1001, 400) == 1041, "the first time", "frameInfo");
		check(clock.time(1000, 400) == 1441, "one before it: the next refresh", "frameInfo");
		check(clock.time(0, 400) == 0, "nothing shown, 0", "frameInfo");
		check(clock.time(1401, 400) == 1841, "the last one's refresh: the next", "frameInfo");
		check(clock.time(1402, 400) == 2241, "two periods past its own at most", "frameInfo");
		check(clock.time(1403, 400) == 2241, "past that, the last time", "frameInfo");
		check(clock.time(2201, 400) == 2641, "its own refresh, the last one's: the next",
		      "frameInfo");
		check(clock.time(3001, 400) == 3041, "a later refresh as it is", "frameInfo");
		check(clock.time(3441, 400) == 3481, "a period on, as it is", "frameInfo");
		// A sleep of 500 ticks between two swaps: the second is a period after the first in the
		// absolute clock, its time with the longer lead.
		clock.read(6000, 6540, 6000);
		check(clock.time(3442, 400) == 3841 + 540, "a period on after a sleep, with its lead",
		      "frameInfo");
		// No period (no refresh seen): never before the last one.
		RgfxFrameInfoClock still;
		still.read(5000, 5040, 5000);
		check(still.time(1001, 0) == 1041 && still.time(1000, 0) == 1041,
		      "no period: a time before the last given as it", "frameInfo");
	}

	// QuartzCore's mask for both tags: bits 0 and 1, the presentation time, then its difference.
	const uint64_t first = rgfxFrameInfoTime(1000, 40);
	const uint64_t second = rgfxFrameInfoTime(1400, 40);
	uint64_t a[kRgfxSwapInfoMaxArgs + 1];
	uint64_t last = 0;
	rgfxSwapInfoArgs(0x3, &last, rgfxSwapInfoRecord(11, first, 0, 0), a);
	const uint32_t n = rgfxSwapInfoArgs(0x3, &last, rgfxSwapInfoRecord(12, second, 0, 0), a);
	check(n == 4 && a[0] == 0x3 && a[1] == 12, "the mask and the swap's ID", "frameInfo");
	check(a[2] == 1440 && a[3] == 400, "Presentation_time, then Presentation_delta", "frameInfo");
}

// A long run of refreshes as the display gives its swaps' times: a 60 Hz display whose VBLANKs
// reach the guest 0.1 to 3.1 ms late, a host that answers each present at once (no window), one to
// three swaps a refresh, the guest's clock at 24 MHz with nanoseconds_to_absolutetime's conversion
// (3/125 of a nanosecond, rounded down), and the clocks
// read with 0 to 2 ticks of mach time passing at each read, so many readings agree in none of
// their tries; midway the system sleeps for a second, which grows the continuous clock's lead.
// QuartzCore takes each time back by the exact lead (mach_get_times): what it traces never goes
// back, is never more than 2 ticks before the shown refresh nor more than two periods after it,
// and is never before it once a reading gave the lead exactly; each is a period after the one
// before, but where swaps come faster than the refreshes and the one before is two periods ahead
// already, which gives its time again. The run also gives each swap the time its own last try's
// reading gave, as the display did before the clock kept its lead: that time goes back in the
// run, as it did on Ambrosia's boots.
void longRun()
{
	uint64_t x = 0x2545f4914f6cdd1dull;
	auto next = [&x](uint64_t n) {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		return x % n;
	};
	auto toTicks = [](uint64_t ns) { return ns * 3 / 125; };
	const uint64_t periodNs = 16666667;
	RgfxVblankClock vblank;
	RgfxFrameInfoClock clock;
	uint64_t lead = 0, traced = 0, lastShown = 0, lastGiven = 0, swaps = 0, shared = 0;
	bool lastExact = false;
	uint64_t mach = 0;                    // the absolute clock's ticks, as the reads advance it
	uint64_t lastAfter = 0, lastC = 0;    // the last try's later absolute read and continuous read
	uint64_t oldTraced = 0, oldBack = 0;  // the time from each swap's own last try
	uint64_t again = 0;                   // swaps given the time before theirs
	for (uint64_t n = 1; n <= 100000; n++) {
		if (n == 50000)
			lead += 24000000; // a second asleep: the absolute clock stops, the continuous one not
		const uint64_t hostNs = 5000000000ull + n * periodNs;
		const uint64_t receiptNs = hostNs + 100000 + next(3000000);
		vblank.vblank(n, hostNs, receiptNs);
		uint64_t nowNs = receiptNs;
		const uint64_t answers = 1 + next(3);
		for (uint64_t k = 0; k < answers; k++) {
			nowNs += 1000 + next(periodNs / 4);
			const uint64_t shown = toTicks(vblank.shownAt(nowNs));
			// The swap completes, and the clocks are read, a little after the answer.
			mach = toTicks(nowNs) + 10 + next(100);
			auto absolute = [&] {
				const uint64_t t = mach;
				mach += next(3);
				lastAfter = t;
				return t;
			};
			auto continuous = [&] {
				const uint64_t t = mach + lead;
				mach += next(3);
				lastC = t;
				return t;
			};
			rgfxReadClocks(absolute, continuous, clock);
			const uint64_t periodTicks = toTicks(vblank.period());
			const uint64_t ahead = RgfxFrameInfoClock::kAheadPeriods * periodTicks;
			const uint64_t time = clock.time(shown, periodTicks);
			// What QuartzCore traces, by the lead itself, and the refresh the clock gave, by the
			// lead it read: the two agree where it read the lead exactly.
			const uint64_t seen = time - lead;
			const uint64_t given = time - static_cast<uint64_t>(clock.lead());
			const bool exact = clock.lead() == static_cast<int64_t>(lead);
			check(clock.lead() <= static_cast<int64_t>(lead), "the lead never read past itself",
			      "longRun");
			check(given >= shown && given <= shown + ahead,
			      "a given refresh is its own or at most two periods after", "longRun");
			check(swaps == 0 || given - lastGiven >= periodTicks ||
			          (given == lastGiven && lastGiven + periodTicks > shown + ahead),
			      "a period after the one before, or its refresh where that is two periods ahead",
			      "longRun");
			check(seen + 2 >= shown && seen <= shown + ahead,
			      "a traced time is at most 2 ticks before its refresh, two periods after",
			      "longRun");
			if (swaps && exact && lastExact)
				check(seen == traced || seen - traced >= periodTicks,
				      "with the lead read exactly, a traced time is a period after the one before "
				      "or that one",
				      "longRun");
			if (exact)
				check(seen >= shown, "never before its refresh once the lead was read exactly",
				      "longRun");
			again += swaps && given == lastGiven ? 1 : 0;
			const uint64_t oldSeen = shown + (lastC - lastAfter) - lead;
			oldBack += swaps && oldSeen < oldTraced ? 1 : 0;
			oldTraced = oldSeen;
			shared += swaps && shown == lastShown ? 1 : 0;
			traced = seen;
			lastGiven = given;
			lastExact = exact;
			lastShown = shown;
			swaps++;
		}
	}
	check(shared > 1000, "the run has swaps sharing a refresh", "longRun");
	check(again > 0, "the run has swaps faster than the refreshes", "longRun");
	check(oldBack > 0, "each swap's own last reading gives times that go back", "longRun");
}

// A swap a refresh, each answered at, or just before or after, its refresh's VBLANK is taken (a
// host with no window answers as it shows the surface; the VBLANK reaches the guest 0.1 to 3.1 ms
// after its refresh, the answer between the refresh and a millisecond past the VBLANK's receipt).
// An answer taken before its refresh's VBLANK maps to the refresh before (RgfxVblankClock::shownAt).
// Each swap's time is a period after the one before, and its own refresh's: no earlier than the
// refresh, no later than the latest the VBLANKs' delivery maps it to. The rule before,
// each time never before the one given before (a period of 0), gives two swaps a refresh apart
// one time.
void refreshEdges()
{
	uint64_t x = 0x9e3779b97f4a7c15ull;
	auto next = [&x](uint64_t n) {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		return x % n;
	};
	auto toTicks = [](uint64_t ns) { return ns * 3 / 125; };
	const uint64_t periodNs = 16666667, lead = 1000;
	RgfxVblankClock vblank;
	RgfxFrameInfoClock clock, before;
	clock.read(0, lead, 0);
	before.read(0, lead, 0);
	uint64_t last = 0, lastBefore = 0, early = 0, onTime = 0, pushed = 0, shared = 0;
	for (uint64_t n = 1; n <= 20000; n++) {
		const uint64_t hostNs = 5000000000ull + n * periodNs;
		const int64_t delay = 100000 + static_cast<int64_t>(next(3000000));
		const uint64_t receiptNs = hostNs + static_cast<uint64_t>(delay);
		// Where the answer comes, from the VBLANK's receipt: at it, a nanosecond or a microsecond
		// to either side, half a millisecond or a millisecond after, or a microsecond after the
		// refresh itself, or midway between the refresh and the receipt.
		const int64_t edges[] = {0, -1, 1, -1000, 1000, 500000, 1000000, 1000 - delay, -delay / 2};
		const int64_t at = edges[next(sizeof(edges) / sizeof(edges[0]))];
		const uint64_t answerNs = receiptNs + static_cast<uint64_t>(at);
		uint64_t shownNs = 0;
		if (at < 0) {
			shownNs = vblank.shownAt(answerNs);
			vblank.vblank(n, hostNs, receiptNs);
			early++;
		} else {
			vblank.vblank(n, hostNs, receiptNs);
			shownNs = vblank.shownAt(answerNs);
			onTime++;
		}
		const uint64_t periodTicks = toTicks(vblank.period());
		const uint64_t shown = toTicks(shownNs);
		const uint64_t time = clock.time(shown, periodTicks) - lead;
		const uint64_t old = before.time(shown, 0) - lead;
		if (n > 1) {
			check(time - last >= periodTicks, "a period after the one before", "refreshEdges");
			shared += old == lastBefore ? 1 : 0;
		}
		// The guest's clock is the host's here: a refresh maps to its time plus the least delivery
		// the VBLANKs showed, 0.1 to 3.1 ms on, a period before the next.
		check(time >= toTicks(hostNs) && time <= toTicks(hostNs + 3100000) + 2,
		      "on its own refresh", "refreshEdges");
		pushed += time != shown ? 1 : 0;
		last = time;
		lastBefore = old;
	}
	check(early > 1000 && onTime > 1000, "answers on both sides of the VBLANKs", "refreshEdges");
	check(pushed > 1000, "answers before their VBLANK given the refresh after the last",
	      "refreshEdges");
	check(shared > 0, "the rule before gives two swaps a refresh apart one time", "refreshEdges");
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
		const uint64_t first = rgfxFrameInfoTime(1000, 40);
		const uint64_t second = rgfxFrameInfoTime(1400, 40);
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
	check(source.find("rgfxReadClocks([] { return mach_absolute_time(); }, [] { return "
	                  "mach_continuous_time(); },") != std::string::npos &&
	          source.find("mobilePresentedAt = mobileFrameClock.time(shown, period);") !=
	              std::string::npos &&
	          source.find("nanoseconds_to_absolutetime(self->vblankClock.period(), &period);") !=
	              std::string::npos,
	      "the presentation time goes in mach_continuous_time's clock by the display's frame clock",
	      "imports");
}

} // namespace

int main()
{
	layouts();
	record();
	args();
	frameInfo();
	longRun();
	refreshEdges();
	queueRecord();
	kept();
	imports();
	std::printf("legacy_swap_info_test: %d checks passed (%zu releases)\n", checks,
	            sizeof(kReleases) / sizeof(kReleases[0]));
	return 0;
}
