// The native test of RgfxRegionMapping.hpp as the arm64e kext builds it (RGFX_GUEST_ARM64, the
// Makefile's rule for this test): a process's page is 16 KiB, so a window's span and a block are
// whole 16 KiB pages, and no write-combined part is offered, so every span is mapped write-back.
#include "RgfxBlockTable.hpp"
#include "RgfxWindowSpans.hpp"

#include <cstdio>
#include <cstdlib>

#if !RGFX_GUEST_ARM64
#error "arm64_region_test is built with RGFX_GUEST_ARM64=1, as the arm64e kext is"
#endif

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "arm64_region_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

const char kA = 0, kB = 0;
const void *const a = &kA;
const void *const b = &kB;
} // namespace

int main()
{
	check(kRgfxProcessPage == 0x4000, "an arm64 process's page is 16 KiB");
	check(kRgfxWindowPage == kRgfxProcessPage && kRgfxBlockPage == kRgfxProcessPage,
	      "the window and the blocks are in the process's pages");
	check(!kRgfxRegionWriteCombines && kRgfxKextFeatureRegionWriteCombined == 0,
	      "no write-combined part is offered");
	check(RGFX_ALLOCATIONS_OFFSET % kRgfxProcessPage == 0 &&
	          RGFX_CONTEXT_ALLOCATIONS_OFFSET % kRgfxProcessPage == 0,
	      "the region's reserved head and the scanout end on a 16 KiB page");

	// A span is whole 16 KiB pages: a 4 KiB one, or one off a 16 KiB page, would map or take back
	// the rest of its page with it.
	const uint64_t window = 0x10000000; // 256 MiB
	const uint64_t head = RGFX_ALLOCATIONS_OFFSET;
	check(rgfxWindowSpan(window, head, 0x4000), "a 16 KiB page past the head");
	check(rgfxWindowSpan(window, head + 0x4000, 0x8000), "two pages");
	check(!rgfxWindowSpan(window, head, 0x1000), "no 4 KiB span");
	check(!rgfxWindowSpan(window, head + 0x1000, 0x4000), "no span off a 16 KiB page");
	check(!rgfxWindowSpan(window, head + 0x4000, 0x5000), "no span that ends off one");

	// A block is whole 16 KiB pages at a 16 KiB offset, whatever it asks for, so two processes'
	// blocks never share a page.
	const uint64_t start = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
	RgfxBlock blocks[4] = {};
	RgfxBlockTable t = {blocks, 0, 4, start, start + 0x100000, 0, 0, 0, 0, 0, 0, 0};
	uint64_t first = 0, second = 0;
	check(rgfxBlockAlloc(t, a, 1, 0x100, 0x1000, &first) == kRgfxBlockOk &&
	          first % 0x4000 == 0 && t.blocks[0].size == 0x4000,
	      "a 256-byte block takes a 16 KiB page");
	check(rgfxBlockAlloc(t, b, 2, 0x5000, 0x1000, &second) == kRgfxBlockOk &&
	          second == first + 0x4000 && t.blocks[1].size == 0x8000,
	      "the next process's block starts on the next page, rounded to whole pages");
	check(rgfxWindowSpan(start + 0x1000000, second, t.blocks[1].size),
	      "a block is a span a window maps");

	std::printf("arm64_region_test: %d checks passed\n", checks);
	return 0;
}
