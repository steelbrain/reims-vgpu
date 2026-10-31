// The native test of RgfxSurfaceBlocks.hpp: a plane's block is found by its pages alone, each
// client holds it once however often it asks, the hold ends with the client's last let-go, the
// block with its last holder's, and only a holder's span is named.
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
} // namespace

int main()
{
	RgfxSurfaceEntry entries[4] = {};
	RgfxSurfaceTable t = {entries, 4};
	RgfxPageRun plane[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	bool differs = false;
	check(rgfxSurfaceFind(t, plane, 2, 0x4000, 0x5000, &differs) == -1 && !differs,
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
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x5000, &differs) == 0, "the same pages");
	RgfxPageRun other[2] = {{0x10000, 0x3000}, {0x41000, 0x1000}};
	check(rgfxSurfaceFind(t, other, 2, 0x4000, 0x5000, &differs) == -1 && !differs,
	      "other pages are another plane");
	check(rgfxSurfaceFind(t, same, 1, 0x3000, 0x5000, &differs) == -1 && !differs,
	      "a part of the pages is no plane's");
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x6000, &differs) == -1 && differs,
	      "the same pages with another block size get no block");

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
	rgfxSurfaceClear(e);
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
	          runs[0].addr == 0x1000 && runs[0].length == 0x1000 && runs[1].addr == 0x9000,
	      "runs merge where they meet");
	check(!rgfxSurfaceAppendRun(runs, &count, 2, 0x20000, 0x1000), "a third run does not fit");

	std::printf("surface_blocks_test: OK (%d checks)\n", checks);
	return 0;
}
