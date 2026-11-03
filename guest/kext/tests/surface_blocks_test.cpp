// The native test of RgfxSurfaceBlocks.hpp: a plane's block is found by its pages and its mode
// alone, each client holds it once however often it asks, the hold ends with the client's last
// let-go, the block with its last holder's, and only a holder's span is named. A dormant block is
// held again only by its own device cache, live, and the dormant blocks past their caps go
// oldest first.
#include "RgfxSurfaceBlocks.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_blocks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

const char kA = 0, kB = 0, kC = 0;
const void *const a = &kA;
const void *const b = &kB;
const void *const c = &kC;
char pagesA, pagesB, pagesC;
char cacheA, cacheB;

// Makes entry `i` of `t` a cached block in place of `length` bytes at `page`, held by no one.
RgfxSurfaceEntry &cachedEntry(RgfxSurfaceTable &t, uint32_t i, RgfxPageRun *run, uint64_t page,
                              uint64_t length, void *cache)
{
	*run = {page, length};
	RgfxSurfaceEntry &e = t.entries[i];
	e = RgfxSurfaceEntry{};
	e.used = true;
	e.offset = 0x100000 + i * 0x2000ull;
	e.size = 0x1000;
	e.inPlace = true;
	e.length = length;
	e.runs = run;
	e.runCount = 1;
	e.cache = cache;
	e.cached = true;
	return e;
}

void dormancy()
{
	RgfxSurfaceEntry entries[3] = {};
	RgfxSurfaceTable t = {entries, 3};
	RgfxPageRun runs[3];
	RgfxSurfaceEntry &e = cachedEntry(t, 0, &runs[0], 0x10000, 0x4000, &cacheA);
	RgfxSurfaceEntry &f = cachedEntry(t, 1, &runs[1], 0x20000, 0x8000, &cacheB);
	check(rgfxSurfaceOldestDormant(t) == -1, "no dormant block yet");

	// The last let-go keeps each dormant, in order, counted with its pages' bytes.
	rgfxSurfaceSleep(t, e);
	rgfxSurfaceSleep(t, f);
	rgfxSurfaceSleep(t, e);
	check(t.dormantCount == 2 && t.dormantBytes == 0xc000, "two dormant, counted once each");
	check(rgfxSurfaceOldestDormant(t) == 0, "the first slept is the oldest");
	bool differs = false;
	RgfxPageRun same = {0x10000, 0x4000};
	check(rgfxSurfaceFind(t, &same, 1, 0x4000, 0x1000, true, &differs) == 0,
	      "a dormant block is found by its pages");

	// Revival: only the block's own device cache, and only while nothing asked it to go.
	check(rgfxSurfaceRevives(e, &cacheA, true), "its own cache holds it again");
	check(!rgfxSurfaceRevives(e, &cacheA, false),
	      "a block asked to go never revives, its own cache's ask included");
	check(!rgfxSurfaceRevives(e, &cacheB, true), "another surface's cache does not");
	check(!rgfxSurfaceRevives(e, nullptr, true), "nor an ask with no cache");
	rgfxSurfaceWake(t, e);
	check(!rgfxSurfaceRevives(e, &cacheA, true), "nor one of a block held");
	check(t.dormantCount == 1 && t.dormantBytes == 0x8000 && rgfxSurfaceOldestDormant(t) == 1,
	      "held again: the other is the oldest");
	rgfxSurfaceSleep(t, e);
	check(rgfxSurfaceOldestDormant(t) == 1 && e.dormantSeq > f.dormantSeq,
	      "dormant again, it is the newest");

	// The ask of another mode, or of another cache, finds the dormant block in its way.
	RgfxPageRun inside = {0x10200, 0x3e00};
	check(rgfxSurfaceFind(t, &inside, 1, 0x3e00, 0x1000, false, &differs) == -1 && differs,
	      "a copy of its plane differs");
	check(rgfxSurfaceDormantMeeting(t, &inside, 1) == 0, "and meets the dormant block");
	RgfxPageRun elsewhere = {0x30000, 0x1000};
	check(rgfxSurfaceDormantMeeting(t, &elsewhere, 1) == -1, "other pages meet none");

	// Teardown: a dormant entry cleared leaves the counts.
	rgfxSurfaceClear(t, f);
	check(t.dormantCount == 1 && t.dormantBytes == 0x4000 && rgfxSurfaceOldestDormant(t) == 0,
	      "the one let go leaves the counts");
	rgfxSurfaceClear(t, e);
	check(t.dormantCount == 0 && t.dormantBytes == 0 && rgfxSurfaceOldestDormant(t) == -1,
	      "none dormant");

	// The caps let the oldest go first: the bytes past the caller's cap, the count past
	// kRgfxDormantMaxBlocks.
	static RgfxSurfaceEntry many[kRgfxDormantMaxBlocks + 2];
	static RgfxPageRun manyRuns[kRgfxDormantMaxBlocks + 2];
	RgfxSurfaceTable m = {many, kRgfxDormantMaxBlocks + 2};
	RgfxDormantCause why = kRgfxDormantCauses;
	for (uint32_t i = 0; i < 3; i++)
		rgfxSurfaceSleep(m, cachedEntry(m, i, &manyRuns[i], 0x100000 * (i + 1), 0x1000, &cacheA));
	check(rgfxSurfaceOverCaps(m, 0x3000, &why) == -1, "within the caps");
	check(rgfxSurfaceOverCaps(m, 0x2fff, &why) == 0 && why == kRgfxDormantBytes,
	      "past the bytes: the oldest");
	rgfxSurfaceClear(m, many[0]);
	check(rgfxSurfaceOverCaps(m, 0x1fff, &why) == 1 && why == kRgfxDormantBytes,
	      "then the next oldest");
	for (uint32_t i = 3; i < kRgfxDormantMaxBlocks + 2; i++)
		rgfxSurfaceSleep(m, cachedEntry(m, i, &manyRuns[i], 0x100000 * (i + 1), 0x1000, &cacheA));
	check(m.dormantCount == kRgfxDormantMaxBlocks + 1, "one past the count");
	check(rgfxSurfaceOverCaps(m, ~0ull, &why) == 1 && why == kRgfxDormantCount,
	      "past the count: the oldest");
	rgfxSurfaceClear(m, many[1]);
	check(rgfxSurfaceOverCaps(m, ~0ull, &why) == -1, "at the count: none");

	// A dormant block whose surface is volatile goes first, the oldest of those first; one held
	// again is no longer marked.
	many[700].purgeable = true;
	many[9].purgeable = true;
	check(rgfxSurfaceOldestDormant(m) == 9, "the oldest volatile one goes first");
	check(rgfxSurfaceOverCaps(m, 0x1000 * 4, &why) == 9 && why == kRgfxDormantBytes,
	      "past a cap too");
	rgfxSurfaceWake(m, many[9]);
	check(!many[9].purgeable && rgfxSurfaceOldestDormant(m) == 700,
	      "held again, it is not; the next volatile one");
	rgfxSurfaceClear(m, many[700]);
	check(rgfxSurfaceOldestDormant(m) == 2, "none volatile: the oldest");
	many[2].pending = true;
	check(rgfxSurfaceOldestDormant(m) == 3,
	      "one whose BLOCK_UNSHARE is not yet posted is passed over");
	many[2].pending = false;
}
} // namespace

int main()
{
	dormancy();
	RgfxSurfaceEntry entries[4] = {};
	RgfxSurfaceTable t = {entries, 4};
	RgfxPageRun plane[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	bool differs = false;
	check(rgfxSurfaceFind(t, plane, 2, 0x4000, 0x5000, false, &differs) == -1 && !differs,
	      "an empty table holds no plane");

	// The first ask makes the entry.
	int32_t at = rgfxSurfaceFreeEntry(t);
	check(at == 0, "a free entry");
	RgfxSurfaceEntry &e = entries[at];
	e.used = true;
	e.offset = 0x100000;
	e.size = 0x5000;
	e.serial = 7;
	e.length = 0x4000;
	e.runs = plane;
	e.runCount = 2;
	check(rgfxSurfaceHold(e, a, 1, &pagesA) == kRgfxSurfaceHoldNew, "a's first hold");
	check(rgfxSurfaceFreeEntry(t) == 1, "the next free entry");

	// Another process's same pages find it; other pages, or a part of them, do not.
	RgfxPageRun same[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x5000, false, &differs) == 0, "the same pages");
	RgfxPageRun other[2] = {{0x10000, 0x3000}, {0x41000, 0x1000}};
	check(rgfxSurfaceFind(t, other, 2, 0x4000, 0x5000, false, &differs) == -1 && !differs,
	      "other pages are another plane");
	check(rgfxSurfaceFind(t, same, 1, 0x3000, 0x5000, false, &differs) == -1 && !differs,
	      "a part of the pages is no plane's");
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x6000, false, &differs) == -1 && differs,
	      "the same pages with another block size get no block");

	// A surface's blocks are of one mode: its whole allocation's pages, asked for in place, find
	// no copy block, nor its plane's pages a block in place.
	RgfxPageRun whole[2] = {{0x10000, 0x3000}, {0x40000, 0x3000}};
	check(rgfxSurfaceFind(t, whole, 2, 0x6000, 0x1000, true, &differs) == -1 && differs,
	      "an ask in place finds no copy block of the surface");
	check(rgfxSurfaceFind(t, other + 1, 1, 0x1000, 0x1000, true, &differs) == -1 && !differs,
	      "nor does it differ from another surface's");
	check(!rgfxSurfacePending(t, whole, 2), "a copy block waits for nothing");
	RgfxSurfaceEntry &in = entries[1];
	in.used = true;
	in.inPlace = true;
	in.pending = true;
	in.offset = 0x200000;
	in.size = 0x1000;
	in.length = 0x6000;
	RgfxPageRun inRuns[2] = {{0x80000, 0x2000}, {0x90000, 0x4000}};
	in.runs = inRuns;
	in.runCount = 2;
	RgfxPageRun inSame[2] = {{0x80000, 0x2000}, {0x90000, 0x4000}};
	check(rgfxSurfacePending(t, inSame, 2), "a block in place waits for the device's answer");
	check(rgfxSurfacePending(t, inSame, 1), "for every ask of its surface's pages");
	in.pending = false;
	check(!rgfxSurfacePending(t, inSame, 2), "until it has it");
	check(rgfxSurfaceFind(t, inSame, 2, 0x6000, 0x1000, true, &differs) == 1 && !differs,
	      "then an ask in place of the same pages finds it");
	check(rgfxSurfaceFind(t, inSame, 1, 0x2000, 0x3000, false, &differs) == -1 && differs,
	      "and an ask for a copy of its plane none");
	check(rgfxSurfaceFind(t, inSame, 2, 0x6000, 0x2000, true, &differs) == -1 && differs,
	      "nor one in place with another block size");
	// A plane that starts past its surface's first page's start (CoreVideo's, 512 bytes in) meets
	// the surface's block in place by that page.
	RgfxPageRun inPlane[1] = {{0x80200, 0x1e00}};
	check(rgfxSurfaceFind(t, inPlane, 1, 0x1e00, 0x3000, false, &differs) == -1 && differs,
	      "a copy of a plane starting inside the first page finds the block in place's mode");
	in.pending = true;
	check(rgfxSurfacePending(t, inPlane, 1), "and waits for its answer");
	in.pending = false;
	check(rgfxSurfaceHeldBy(t, a, 0x100000, 0x10) && !rgfxSurfaceHeldBy(t, a, 0x200000, 0x10),
	      "a holds only the copy block");
	rgfxSurfaceClear(t, in);
	check(rgfxSurfaceFreeEntry(t) == 1, "the block in place cleared");

	// Each client holds it once, however often it asks.
	check(rgfxSurfaceHold(e, b, 2, &pagesB) == kRgfxSurfaceHoldNew, "b's first hold");
	check(rgfxSurfaceHold(e, b, 2, &pagesB) == kRgfxSurfaceHoldAgain, "b asks again");
	check(e.holdCount == 2, "two holders");

	// Only a holder's span is named, and only inside the block.
	check(rgfxSurfaceHeldBy(t, a, 0x100000, 0x5000), "a holder names the block");
	check(rgfxSurfaceHeldBy(t, b, 0x101000, 0x1000), "and a span in it");
	check(!rgfxSurfaceHeldBy(t, c, 0x100000, 0x10), "no one else names it");
	check(!rgfxSurfaceHeldBy(t, a, 0x104000, 0x1001), "nor a span past it");
	check(!rgfxSurfaceHeldBy(t, a, 0xff000, 0x2000), "nor one before it");

	// A let-go ends a hold with the client's last ask, and the block with its last holder.
	void *pages = nullptr;
	uint32_t context = 0;
	check(rgfxSurfaceLetGo(e, c, false, &pages, &context) == kRgfxSurfaceLetGoNotHeld,
	      "c holds nothing");
	check(rgfxSurfaceLetGo(e, b, false, &pages, &context) == kRgfxSurfaceLetGoRef,
	      "b lets one ask go");
	check(rgfxSurfaceLetGo(e, b, false, &pages, &context) == kRgfxSurfaceLetGoHold &&
	          pages == &pagesB && context == 2,
	      "b's hold ends; a holds it still");
	check(!rgfxSurfaceHeldBy(t, b, 0x100000, 0x10), "b no longer names it");
	check(rgfxSurfaceHold(e, c, 3, &pagesC) == kRgfxSurfaceHoldNew, "c holds it");
	check(rgfxSurfaceHold(e, c, 3, &pagesC) == kRgfxSurfaceHoldAgain, "c asks again");
	check(rgfxSurfaceLetGo(e, c, true, &pages, &context) == kRgfxSurfaceLetGoHold &&
	          pages == &pagesC && context == 3,
	      "c's close lets every ask go");
	check(rgfxSurfaceLetGo(e, a, false, &pages, &context) == kRgfxSurfaceLetGoLast &&
	          pages == &pagesA && context == 1,
	      "a was its last holder");
	check(rgfxSurfaceAt(t, 0x100000) == 0, "found by its offset until cleared");
	rgfxSurfaceClear(t, e);
	check(rgfxSurfaceAt(t, 0x100000) == -1 && rgfxSurfaceFreeEntry(t) == 0, "cleared");

	// The holders are bounded.
	RgfxSurfaceEntry full = {};
	static char clients[kRgfxSurfaceHolders + 1];
	for (uint32_t i = 0; i < kRgfxSurfaceHolders; i++)
		check(rgfxSurfaceHold(full, &clients[i], i, nullptr) == kRgfxSurfaceHoldNew, "a holder");
	check(rgfxSurfaceHold(full, &clients[kRgfxSurfaceHolders], 99, nullptr) ==
	          kRgfxSurfaceHoldFull,
	      "past the most holders");

	// Runs: adjacent segments merge, gaps start a run, and too many runs do not fit.
	RgfxPageRun runs[2];
	uint32_t count = 0;
	check(rgfxSurfaceAppendRun(runs, &count, 2, 0x1000, 0x800) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x1800, 0x800) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x5000, 0x0) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x9000, 0x1000) && count == 2 &&
	          runs[0].address == 0x1000 && runs[0].length == 0x1000 && runs[1].address == 0x9000,
	      "runs merge where they meet");
	check(!rgfxSurfaceAppendRun(runs, &count, 2, 0x20000, 0x1000), "a third run does not fit");

	// No room for a new block: the first time it lets dormant blocks go and waits, or gives up
	// with none; after that it waits on to its deadline, whatever wakes it, and gives up past it.
	check(rgfxSurfaceRoomStep(0, 3, 50) == kRgfxSurfaceRoomLetGoAndWait,
	      "no room: dormant blocks go, and the ask waits");
	check(rgfxSurfaceRoomStep(0, 0, 50) == kRgfxSurfaceRoomGiveUp,
	      "no room and none dormant: the ask gives up");
	check(rgfxSurfaceRoomStep(150, 3, 60) == kRgfxSurfaceRoomWait,
	      "woken with no room before the deadline: it waits on");
	check(rgfxSurfaceRoomStep(150, 0, 149) == kRgfxSurfaceRoomWait,
	      "it waits on once none is dormant: their releases are in flight");
	check(rgfxSurfaceRoomStep(150, 3, 150) == kRgfxSurfaceRoomGiveUp,
	      "at the deadline it gives up");

	std::printf("surface_blocks_test: OK (%d checks)\n", checks);
	return 0;
}
