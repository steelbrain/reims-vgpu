// The native test of RgfxPagesBlock.hpp: a pages block's range rounded out to whole pages with
// the call's offset in the first, its runs counted and then written as one per stretch of
// contiguous segments, and its size in whole pages; and a span's runs and length taken to the
// end of its last 16 KiB page, as PAGES_16K names them; and how a block's pages are wired, and
// marked, for memory the process maps writable or read-only; and the physical alignment of the
// kext's chunks.
#include "RgfxPagesBlock.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pages_block_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kPage = 0x1000;

// A wired range's segments: `pages` physical pages in order, each answered as a segment of its
// own from the byte asked to its page's end, as getPhysicalSegment answers across pages that are
// not contiguous.
struct Pages {
	std::vector<uint64_t> pages;
	uint64_t operator()(uint64_t at, uint64_t *bytes) const
	{
		const uint64_t i = at / kPage;
		if (i >= pages.size() || pages[i] == 0) {
			*bytes = 0;
			return 0;
		}
		*bytes = kPage - at % kPage;
		return pages[i] + at % kPage;
	}
};
} // namespace

int main()
{
	RgfxPagesRange r = {};
	check(rgfxPagesRange(0x7000'0123, 0x2000, kPage, &r), "a range across three pages");
	check(r.base == 0x7000'0000 && r.length == 0x3000 && r.offset == 0x123,
	      "rounded out, the first byte's offset kept");
	check(rgfxPagesRange(0x7000'0000, 0x1000, kPage, &r) && r.base == 0x7000'0000 &&
	          r.length == 0x1000 && r.offset == 0,
	      "a page is itself");
	check(rgfxPagesRange(0x7000'0fff, 1, kPage, &r) && r.length == 0x1000 && r.offset == 0xfff,
	      "a page's last byte");
	check(!rgfxPagesRange(0x7000'0000, 0, kPage, &r), "an empty range");
	check(!rgfxPagesRange(UINT64_MAX - 0x10, 0x20, kPage, &r), "a range that wraps");
	check(!rgfxPagesRange(UINT64_MAX - 0x1800, 0x1000, kPage, &r), "one whose last page wraps");
	check(!rgfxPagesRange(0x1000, kRgfxPagesBlockMaxBytes + 1, kPage, &r), "one past the most");
	check(rgfxPagesRange(0, kRgfxPagesBlockMaxBytes, kPage, &r), "the most");
	check(!rgfxPagesRange(0x1000, 0x1000, 0x1800, &r), "a page that is no power of two");

	// Pages 0x10 and 0x11 meet, and 0x40, 0x41 and 0x42.
	Pages pages = {{0x10'000, 0x11'000, 0x40'000, 0x41'000, 0x42'000}};
	const uint64_t length = 5 * kPage;
	check(rgfxPagesRuns(length, pages, nullptr, 0) == 2, "counted: two stretches");
	pages.pages = {0x10'000, 0x11'000, 0x40'000, 0x30'000, 0x31'000};
	check(rgfxPagesRuns(length, pages, nullptr, 0) == 3, "three where the third page meets none");
	RgfxPageRun runs[3] = {};
	check(rgfxPagesRuns(length, pages, runs, 3) == 3, "written: as many as counted");
	check(runs[0].address == 0x10'000 && runs[0].length == 0x2000, "the first stretch");
	check(runs[1].address == 0x40'000 && runs[1].length == 0x1000, "the lone page");
	check(runs[2].address == 0x30'000 && runs[2].length == 0x2000, "the last stretch");
	check(rgfxPagesRuns(length, pages, runs, 2) == 0, "more runs than room: none");
	// A range that ends inside a segment takes its bytes alone.
	check(rgfxPagesRuns(kPage + 0x10, pages, runs, 3) == 1 && runs[0].length == 0x1010,
	      "a range ending inside a page");
	pages.pages = {0x10'000, 0};
	check(rgfxPagesRuns(2 * kPage, pages, nullptr, 0) == 0, "a page with no segment: none");
	check(rgfxPagesRuns(0, pages, nullptr, 0) == 0, "no bytes, no runs");

	check(rgfxPagesBlockSize(1, kPage) == kPage, "one run takes a page");
	check(rgfxPagesBlockSize(256, kPage) == kPage, "256 runs fill one");
	check(rgfxPagesBlockSize(257, kPage) == 2 * kPage, "257 take two");
	check(rgfxPagesBlockSize(1u << 24, kPage) == (1ull << 28), "a 64 GiB range of lone pages");

	// PAGES_16K: a span named in whole 16 KiB pages, as the guest's IOSurfaces' lengths are not
	// (0xfd2000 for 1920 x 1080 of 8 bytes, in 1013 runs whose last is 8 KiB).
	check(rgfxToPageEnd(0xfd2000, kRgfxPage16K) == 0xfd4000, "a 1080p surface to its page's end");
	check(rgfxToPageEnd(0x1000, kRgfxPage16K) == kRgfxPage16K, "a 4 KiB surface takes its page");
	check(rgfxToPageEnd(0x8000, kRgfxPage16K) == 0x8000, "whole pages are themselves");
	check(rgfxToPageEnd(0x1000, kPage) == 0x1000, "4 KiB pages leave 4 KiB lengths");
	RgfxPageRun span[3] = {{0x8000'0000, 0x8000}, {0x9000'0000, 0x4000}, {0xa000'0000, 0x2000}};
	rgfxRunsToPageEnd(span, 3, kRgfxPage16K);
	check(span[0].address == 0x8000'0000 && span[0].length == 0x8000 &&
	          span[1].address == 0x9000'0000 && span[1].length == 0x4000,
	      "the runs before the last stay as they are");
	check(span[2].address == 0xa000'0000 && span[2].length == 0x4000,
	      "the last run ends at its page's end");
	check(span[0].length + span[1].length + span[2].length ==
	          rgfxToPageEnd(0x8000 + 0x4000 + 0x2000, kRgfxPage16K),
	      "the runs cover the span's length taken to its page's end");
	RgfxPageRun lone = {0x8000'2000, 0x1000};
	rgfxRunsToPageEnd(&lone, 1, kRgfxPage16K);
	check(lone.address == 0x8000'2000 && lone.length == 0x2000,
	      "a run inside a page ends at that page's end, its start kept");
	RgfxPageRun whole = {0x8000'0000, 0x4000};
	rgfxRunsToPageEnd(&whole, 1, kRgfxPage16K);
	check(whole.length == 0x4000, "a whole page stays whole");
	rgfxRunsToPageEnd(nullptr, 0, kRgfxPage16K);

	// Writable memory: wired for reading and writing, a block of the process's, whatever the
	// word.
	const uint64_t pagesWord = RGFX_FEATURE_PAGES_BUFFERS;
	const uint64_t readOnlyWord = RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_READ_ONLY_PAGES;
	for (uint64_t word : {pagesWord, readOnlyWord}) {
		const RgfxPagesWiring w = rgfxPagesWiring(true, word);
		check(w.access == RgfxPagesAccess::kReadWrite && w.flags == RGFX_SURFACE_PAGES_PRIVATE,
		      "writable memory is wired for reading and writing, unmarked");
	}
	// Read-only memory: wired for reading alone and marked read-only where the host took
	// READ_ONLY_PAGES, which holds every object over the block to leave its pages as they are;
	// else no block, as a host that does not know the mark would not hold it.
	const RgfxPagesWiring ro = rgfxPagesWiring(false, readOnlyWord);
	check(ro.access == RgfxPagesAccess::kRead &&
	          ro.flags == (RGFX_SURFACE_PAGES_PRIVATE | RGFX_SURFACE_PAGES_READ_ONLY),
	      "read-only memory is wired for reading alone, marked read-only");
	check(rgfxPagesWiring(false, pagesWord).access == RgfxPagesAccess::kRefused,
	      "read-only memory makes no block where the host did not take READ_ONLY_PAGES");
	check(rgfxPagesWiring(false, 0).access == RgfxPagesAccess::kRefused,
	      "nor with no feature at all");

	// The chunks' physical mask: each chunk at a multiple of its own 64 KiB, on 4 KiB and on
	// 16 KiB pages; a mask that clears only the page's bits gives the page alone.
	check(rgfxPhysicalMaskAlignment(kRgfxPagesChunkMask, kPage) == kRgfxPagesChunk,
	      "the chunks' mask: 64 KiB");
	check(rgfxPhysicalMaskAlignment(kRgfxPagesChunkMask, kRgfxPage16K) == kRgfxPagesChunk,
	      "the chunks' mask on 16 KiB pages: 64 KiB");
	check(rgfxPhysicalMaskAlignment(0x0000'FFFF'FFFF'F000ull, kPage) == kPage,
	      "a mask clear of the page's bits alone: the page");
	check(rgfxPhysicalMaskAlignment(0x0000'FFFF'FFFF'F000ull, kRgfxPage16K) == kRgfxPage16K,
	      "raised to the page");
	check(rgfxPhysicalMaskAlignment(0x0000'0000'FFFF'FFFFull, kPage) == kPage,
	      "a 32-bit mask asks no more than the page");
	check(rgfxPhysicalMaskAlignment(0, kPage) == kPage, "no mask: the page");
	check(kRgfxPagesChunkMask == (((1ull << 48) - 1) & ~uint64_t{kRgfxPagesChunk - 1}),
	      "any multiple of 64 KiB of guest RAM below 2^48");
	// A block of four such chunks, two of them meeting: its runs start and end at multiples of
	// the chunk, so every boundary between two lies on the host's 64 KiB granule.
	Pages chunks = {};
	for (uint64_t chunk : {0x4'0000ull, 0x5'0000ull, 0x9'0000ull, 0x2'0000ull})
		for (uint64_t at = 0; at < kRgfxPagesChunk; at += kPage)
			chunks.pages.push_back(chunk + at);
	RgfxPageRun chunkRuns[4] = {};
	check(rgfxPagesRuns(4 * kRgfxPagesChunk, chunks, chunkRuns, 4) == 3,
	      "four chunks, two meeting: three runs");
	for (const RgfxPageRun &run : {chunkRuns[0], chunkRuns[1], chunkRuns[2]})
		check(run.address % kRgfxPagesChunk == 0 && run.length % kRgfxPagesChunk == 0,
		      "each run of chunks starts and ends at a multiple of 64 KiB");
	check(chunkRuns[0].length == 2 * kRgfxPagesChunk, "the two chunks that meet are one run");

	std::printf("pages_block_test: OK (%d checks)\n", checks);
	return 0;
}
