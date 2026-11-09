// The native test of RgfxDeferCarry.hpp: the bundle lists a surface's resident pages in its order,
// merged, kRgfxCarryPagesMost at most; the kext takes a list of whole pages of the surface, apart
// and in order, and none other. A deferral reads a surface fresh with no page resident, or fresh
// but for exactly the listed pages, none compressed or paged out either way, and carries them only
// where their wire took no fault and the counts after it are the same; every other read, a lock
// that moved the seed, pages not counted, a page compressed or paged out, a count other than the
// list's, a wire refused, a page faulted in, or runs that cannot be made, names its pages at once.
// The runs carry each listed page at its guest address, merged where they follow in the surface
// and in guest RAM, every other byte of the surface zeros, a kernel page of 16 KiB as four of the
// device's.
#include "RgfxDeferCarry.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "defer_carry_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kPage = 0x1000, kBigPage = 0x4000;
// The bytes of a 4K NV12 decoder output, not whole pages.
constexpr uint64_t kLength = 3840ull * 2160 * 3 / 2;

void lists()
{
	RgfxSurfaceResident r = {};
	check(rgfxResidentAdd(r, 0, kPage) && r.count == 1 && r.runs[0].offset == 0 &&
	          r.runs[0].length == kPage,
	      "page 0 resident, one run");
	check(rgfxResidentAdd(r, kPage, kPage) && r.count == 1 && r.runs[0].length == 2 * kPage,
	      "page 1 follows it, merged");
	check(rgfxResidentAdd(r, 5 * kPage, kPage) && r.count == 2 && r.runs[1].offset == 5 * kPage,
	      "page 5, a run of its own");
	check(!rgfxResidentAdd(r, 3 * kPage, kPage) && r.count == 2,
	      "a page before the last listed is refused, the list unchanged");
	check(!rgfxResidentAdd(r, 6 * kPage, 0) && r.count == 2, "a page of no bytes is refused");
	check(rgfxResidentPages(r, kLength, kPage) == 3, "three pages listed");

	RgfxSurfaceResident full = {};
	for (uint64_t p = 0; p < kRgfxCarryPagesMost; p++)
		check(rgfxResidentAdd(full, 2 * p * kPage, kPage), "a page apart from the last");
	check(full.count == kRgfxCarryPagesMost &&
	          rgfxResidentPages(full, kLength, kPage) == kRgfxCarryPagesMost,
	      "sixteen pages, in sixteen runs");
	check(!rgfxResidentAdd(full, 40 * kPage, kPage) && full.count == kRgfxCarryPagesMost,
	      "a seventeenth page is refused: the surface is not deferred");
	RgfxSurfaceResident merged = {};
	for (uint64_t p = 0; p < kRgfxCarryPagesMost; p++)
		rgfxResidentAdd(merged, p * kPage, kPage);
	check(merged.count == 1 && !rgfxResidentAdd(merged, kRgfxCarryPagesMost * kPage, kPage),
	      "so is the seventeenth of one run");
}

void listsTaken()
{
	RgfxSurfaceResident r = {};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "an empty list carries nothing");
	r.count = 1;
	r.runs[0] = {0, kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 1, "page 0");
	check(rgfxResidentPages(r, kLength, 0) == 0, "no page size, nothing");
	r.runs[0] = {0x800, kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a run not at a page's start is refused");
	r.runs[0] = {0, 0x1800};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a run not of whole pages is refused");
	r.runs[0] = {0, 0};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a run of no bytes is refused");
	r.runs[0] = {kLength / kPage * kPage, 2 * kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a run past the surface is refused");
	r.runs[0] = {~0ull & ~(kPage - 1), kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "so is one whose end wraps");
	r.count = 2;
	r.runs[0] = {4 * kPage, kPage};
	r.runs[1] = {2 * kPage, kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "runs out of order are refused");
	r.runs[1] = {4 * kPage, kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "so are runs that overlap");
	r.runs[1] = {5 * kPage, kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 2, "adjacent runs are taken");
	r.count = kRgfxCarryPagesMost + 1;
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a count past the runs is refused");
	r.count = 1;
	r.runs[0] = {0, (kRgfxCarryPagesMost + 1) * kPage};
	check(rgfxResidentPages(r, kLength, kPage) == 0, "a run of seventeen pages is refused");
	check(rgfxResidentPages(r, kLength, kPage, ~0ull) == kRgfxCarryPagesMost + 1,
	      "and counted where nothing bounds it");
	// A list of the kernel's 16 KiB pages, an arm64 guest's: whole device pages too.
	r.runs[0] = {kBigPage, 2 * kBigPage};
	check(rgfxResidentPages(r, kLength, kBigPage) == 2 && rgfxResidentPages(r, kLength, kPage) == 8,
	      "two kernel pages of 16 KiB are eight device pages");
}

// Page counts as rgfxSurfaceLocksPageCounts reads them: counted, `resident` resident and
// `swapped` compressed or paged out.
RgfxSurfacePageCounts at(uint64_t resident, uint64_t swapped = 0)
{
	return {true, resident, swapped};
}

// Counts not taken: the surface's memory is not a buffer the kernel allocated, or the regions of
// its mapping are not exactly it.
constexpr RgfxSurfacePageCounts kUncounted = {false, ~0ull, ~0ull};

void reads()
{
	check(rgfxDeferRead(false, 0, at(0), 0) == kRgfxDeferReadStale, "no lock, not deferred");
	check(rgfxDeferRead(true, 1, at(0), 0) == kRgfxDeferReadStale, "a seed moved, not deferred");
	check(rgfxDeferRead(true, 0, at(0), 0) == kRgfxDeferReadFresh, "no page resident: fresh");
	check(rgfxDeferRead(true, 0, at(0), 1) == kRgfxDeferReadFresh,
	      "none resident though one was listed: fresh, carrying nothing");
	check(rgfxDeferRead(true, 0, at(1), 1) == kRgfxDeferReadCarry,
	      "as many resident as listed: carried, once wired");
	check(rgfxDeferRead(true, 0, at(1), 0) == kRgfxDeferReadStale,
	      "a page resident with none listed: not deferred");
	check(rgfxDeferRead(true, 0, at(2), 1) == kRgfxDeferReadStale,
	      "more resident than listed: not deferred");
	check(rgfxDeferRead(true, 0, at(1), 2) == kRgfxDeferReadStale,
	      "fewer resident than listed: not deferred");
	check(rgfxDeferRead(true, 3, at(1), 1) == kRgfxDeferReadStale,
	      "the listed count with the seed moved: not deferred");
	// A page compressed or paged out holds bytes though it is not resident.
	check(rgfxDeferRead(true, 0, at(0, 1), 0) == kRgfxDeferReadStale,
	      "none resident, one compressed or paged out: not deferred");
	check(rgfxDeferRead(true, 0, at(0, 1), 1) == kRgfxDeferReadStale,
	      "the listed page compressed or paged out since the ask: not deferred, nor fresh");
	check(rgfxDeferRead(true, 0, at(1, 1), 1) == kRgfxDeferReadStale,
	      "the listed page resident, another compressed or paged out: not deferred");
	check(rgfxDeferRead(true, 0, {false, 0, 0}, 0) == kRgfxDeferReadStale,
	      "pages not counted, though they read as none: not deferred");
	check(rgfxDeferRead(true, 0, kUncounted, 1) == kRgfxDeferReadStale,
	      "pages not counted, one listed: not deferred");

	check(rgfxDeferCarried(true, 0, at(1), 1), "wired with no fault, the count unchanged: carried");
	check(!rgfxDeferCarried(false, 0, at(1), 1),
	      "a wire refused (a listed page not resident): not");
	check(!rgfxDeferCarried(true, 0, at(2), 1), "a page faulted in by the wire: not");
	check(!rgfxDeferCarried(true, 0, at(1, 1), 1),
	      "a page compressed or paged out between the reads: not");
	check(!rgfxDeferCarried(true, 0, kUncounted, 1), "the count after the wire not taken: not");
	check(!rgfxDeferCarried(true, 0, at(0), 0), "nothing listed: carries nothing");
}

void outcomes()
{
	check(rgfxCarryOutcome(true, 0, at(1), 1, true, at(1), 2) == kRgfxCarryTaken,
	      "listed, wired, counted again and its runs made: carried");
	check(rgfxCarryOutcome(true, 0, at(0), 1, false, kUncounted, 0) == kRgfxCarryNone,
	      "none resident: deferred carrying nothing");
	check(rgfxCarryOutcome(false, 0, at(1), 1, true, at(1), 2) == kRgfxCarryTouched,
	      "the read-only lock failed: touched");
	check(rgfxCarryOutcome(true, 1, at(1), 1, true, at(1), 2) == kRgfxCarryTouched,
	      "the seed moved: touched");
	check(rgfxCarryOutcome(true, 1, kUncounted, 1, false, kUncounted, 0) == kRgfxCarryTouched,
	      "the seed moved, its pages not counted: touched");
	check(rgfxCarryOutcome(true, 0, kUncounted, 1, false, kUncounted, 0) == kRgfxCarryUncounted,
	      "its pages not counted: uncounted");
	check(rgfxCarryOutcome(true, 0, {false, 0, 0}, 1, false, kUncounted, 0) ==
	          kRgfxCarryUncounted,
	      "not counted, though they read as none: uncounted, not fresh");
	check(rgfxCarryOutcome(true, 0, at(17), 16, true, at(17), 2) == kRgfxCarryCounted,
	      "more pages resident than listed: counted");
	check(rgfxCarryOutcome(true, 0, at(1), 2, true, at(1), 2) == kRgfxCarryCounted,
	      "fewer: counted");
	check(rgfxCarryOutcome(true, 0, at(1), 0, false, kUncounted, 0) == kRgfxCarryCounted,
	      "one resident, nothing listed (over 16, or an invalid list): counted");
	check(rgfxCarryOutcome(true, 0, at(0, 1), 1, false, kUncounted, 0) == kRgfxCarryCounted,
	      "the listed page compressed or paged out since the ask: counted");
	check(rgfxCarryOutcome(true, 0, at(1, 1), 1, false, kUncounted, 0) == kRgfxCarryCounted,
	      "the listed page resident, another compressed or paged out: counted");
	check(rgfxCarryOutcome(true, 0, at(1), 1, false, kUncounted, 0) == kRgfxCarryUnwired,
	      "the NoFault wire refused: unwired");
	check(rgfxCarryOutcome(true, 0, at(1), 1, true, at(2), 2) == kRgfxCarryFaulted,
	      "a page faulted in by the wire: faulted");
	check(rgfxCarryOutcome(true, 0, at(1), 1, true, at(1, 1), 2) == kRgfxCarryFaulted,
	      "a page compressed or paged out between the reads: faulted");
	check(rgfxCarryOutcome(true, 0, at(1), 1, true, kUncounted, 2) == kRgfxCarryFaulted,
	      "the count after the wire not taken: faulted");
	check(rgfxCarryOutcome(true, 0, at(1), 1, true, at(1), 0) == kRgfxCarryNoRuns,
	      "no runs made: not carried");
	static_assert(kRgfxCarryOutcomes == 8, "eight outcomes");
}

void runs()
{
	RgfxPageRun out[2 * kRgfxCarryPagesMost + 1] = {};
	const uint32_t room = sizeof(out) / sizeof(out[0]);
	RgfxSurfaceResident r = {};
	rgfxResidentAdd(r, 0, kPage);
	const uint64_t page0[] = {0x7000};
	check(rgfxCarryRuns(r, page0, 3 * kPage, kPage, out, room) == 2 &&
	          out[0].address == 0x7000 && out[0].length == kPage &&
	          out[1].address == RGFX_PAGE_RUN_ZEROS && out[1].length == 2 * kPage,
	      "page 0 at its guest address, the rest of the surface zeros");
	check(rgfxCarryRuns(r, page0, 2 * kPage + 1, kPage, out, room) == 2 &&
	          out[1].length == 2 * kPage,
	      "the surface's bytes rounded up to whole pages");
	check(rgfxCarryRuns(r, page0, kPage, kPage, out, room) == 1 && out[0].length == kPage,
	      "a one-page surface: no zeros");

	RgfxSurfaceResident mid = {};
	rgfxResidentAdd(mid, kPage, kPage);
	check(rgfxCarryRuns(mid, page0, 3 * kPage, kPage, out, room) == 3 &&
	          out[0].address == RGFX_PAGE_RUN_ZEROS && out[0].length == kPage &&
	          out[1].address == 0x7000 && out[2].address == RGFX_PAGE_RUN_ZEROS &&
	          out[2].length == kPage,
	      "a page amid the surface: zeros before and after it");

	RgfxSurfaceResident two = {};
	rgfxResidentAdd(two, 0, kPage);
	rgfxResidentAdd(two, kPage, kPage);
	const uint64_t following[] = {0x7000, 0x8000}, apart[] = {0x7000, 0x9000};
	check(rgfxCarryRuns(two, following, 4 * kPage, kPage, out, room) == 2 &&
	          out[0].length == 2 * kPage && out[1].length == 2 * kPage,
	      "pages that follow in guest RAM too, one run");
	check(rgfxCarryRuns(two, apart, 4 * kPage, kPage, out, room) == 3 &&
	          out[0].address == 0x7000 && out[1].address == 0x9000,
	      "pages apart in guest RAM, a run each");
	const uint64_t unaligned[] = {0x7000, 0x9800};
	check(rgfxCarryRuns(two, unaligned, 4 * kPage, kPage, out, room) == 0,
	      "an address not a page's: no runs");
	check(rgfxCarryRuns(two, apart, 4 * kPage, kPage, out, 2) == 0, "runs past the room: none");
	RgfxSurfaceResident none = {};
	check(rgfxCarryRuns(none, page0, 4 * kPage, kPage, out, room) == 0, "nothing listed: none");

	// The most a list holds, sixteen pages apart: sixteen runs and seventeen of zeros.
	RgfxSurfaceResident full = {};
	uint64_t addresses[kRgfxCarryPagesMost] = {};
	for (uint64_t p = 0; p < kRgfxCarryPagesMost; p++) {
		rgfxResidentAdd(full, (2 * p + 1) * kPage, kPage);
		addresses[p] = 0x100000 + 2 * p * kPage;
	}
	check(rgfxCarryRuns(full, addresses, kLength, kPage, out, room) == room,
	      "sixteen pages apart fill the room a deferral keeps");
	uint64_t bytes = 0;
	for (uint32_t i = 0; i < room; i++)
		bytes += out[i].length;
	check(bytes == (kLength + kPage - 1) / kPage * kPage, "its runs are the surface's bytes");

	// A kernel page of 16 KiB, listed, is four device pages at their addresses.
	RgfxSurfaceResident big = {};
	rgfxResidentAdd(big, 0, kBigPage);
	const uint64_t fours[] = {0x20000, 0x21000, 0x22000, 0x23000};
	check(rgfxCarryRuns(big, fours, 2 * kBigPage, kPage, out, room) == 2 &&
	          out[0].address == 0x20000 && out[0].length == kBigPage &&
	          out[1].address == RGFX_PAGE_RUN_ZEROS && out[1].length == kBigPage,
	      "a 16 KiB page's four device pages, one run");
}
} // namespace

int main()
{
	lists();
	listsTaken();
	reads();
	outcomes();
	runs();
	std::printf("defer_carry_test: %d checks passed\n", checks);
	return 0;
}
