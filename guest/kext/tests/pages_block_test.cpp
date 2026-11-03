// The native test of RgfxPagesBlock.hpp: a pages block's range rounded out to whole pages with
// the call's offset in the first, its runs counted and then written as one per stretch of
// contiguous segments, and its size in whole pages.
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

	std::printf("pages_block_test: OK (%d checks)\n", checks);
	return 0;
}
