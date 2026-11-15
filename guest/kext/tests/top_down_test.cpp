// The native test of RgfxTopDown.hpp: a range's pages visited from the page of its last byte down
// to the page of its first, each page's first byte once, the pages a wire rounds the range out
// to; the visit stopped where it answers false, the pages taken counted; and nothing visited for
// an empty range, one that wraps or ends in the address space's last page, or a page size that is
// not a power of two.
#include "RgfxTopDown.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "top_down_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kPage = 0x1000;

// The pages rgfxTopDown visits for `length` bytes at `address`, in its order, every one taken.
std::vector<uint64_t> visited(uint64_t address, uint64_t length, uint64_t page = kPage)
{
	std::vector<uint64_t> pages;
	const uint64_t taken = rgfxTopDown(address, length, page, [&](uint64_t at) {
		pages.push_back(at);
		return true;
	});
	check(taken == pages.size(), "every page visited is counted");
	return pages;
}
} // namespace

int main()
{
	// Whole pages: each, from the last down.
	check(visited(0x10000, 3 * kPage) == std::vector<uint64_t>{0x12000, 0x11000, 0x10000},
	      "three whole pages from the last down");
	// A range into its first page and part of its last: every page it touches.
	check(visited(0x10800, 2 * kPage) == std::vector<uint64_t>{0x12000, 0x11000, 0x10000},
	      "rounded out to the pages it touches");
	check(visited(0x10fff, 2) == std::vector<uint64_t>{0x11000, 0x10000},
	      "two bytes over a page's end");
	check(visited(0x10001, 1) == std::vector<uint64_t>{0x10000}, "one byte inside a page");
	// 16 KiB pages, as the wire would take them.
	check(visited(0x8000, 0x8001, 0x4000) == std::vector<uint64_t>{0x10000, 0xc000, 0x8000},
	      "16 KiB pages");
	// Stopped at the first refusal: the pages above it taken, it and those below not.
	{
		std::vector<uint64_t> pages;
		const uint64_t taken = rgfxTopDown(0x20000, 4 * kPage, kPage, [&](uint64_t at) {
			pages.push_back(at);
			return at != 0x21000;
		});
		check(taken == 2, "the pages above the refused one taken");
		check(pages == std::vector<uint64_t>{0x23000, 0x22000, 0x21000}, "none visited past it");
	}
	// Nothing for an empty range, one that wraps, one that ends in the last page, or a page size
	// that is not a power of two.
	check(visited(0x10000, 0).empty(), "an empty range");
	check(visited(UINT64_MAX - kPage, 2 * kPage).empty(), "a range that wraps");
	check(visited(UINT64_MAX - kPage + 1, 1).empty(), "a range in the last page");
	check(visited(0x10000, kPage, 0).empty(), "no page size");
	check(visited(0x10000, kPage, 0x3000).empty(), "a page size not a power of two");
	// The last page but one is visited.
	const uint64_t last = UINT64_MAX - 2 * kPage + 1;
	check(visited(last, kPage) == std::vector<uint64_t>{last}, "the last page but one");
	std::printf("top_down_test: %d checks passed\n", checks);
	return 0;
}
