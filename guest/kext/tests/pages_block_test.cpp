// The native test of RgfxPagesBlock.hpp: a pages block's range rounded out to whole pages with
// the call's offset in the first, its runs counted and then written as one per stretch of
// contiguous segments, and its size in whole pages; and a span's runs and length taken to the
// end of its last 16 KiB page, as PAGES_16K names them.
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

	std::printf("pages_block_test: OK (%d checks)\n", checks);
	return 0;
}
