// The native test of RgfxDoorbellPages.hpp: the doorbell pages of BAR 4 taken for contexts up to
// every page, round-robin, none for a context the bind cannot name, given back, and the bind's
// value and each page's place in the BAR as the device reads them.
#include "RgfxDoorbellPages.hpp"

#include <cstdio>
#include <cstdlib>

// The wire's layout as the kext reads it (design.md; wire.rs asserts the same).
static_assert(RGFX_PCI_BAR_DOORBELLS == 4, "BAR 4");
static_assert(RGFX_DOORBELL_PAGE_SIZE == 0x4000, "16 KiB pages");
static_assert(RGFX_DOORBELL_PAGES == 256, "256 pages");
static_assert(RGFX_DOORBELLS_SIZE == 0x40'0000, "4 MiB");
static_assert(RGFX_REG_DOORBELL_PAGE_BIND == 0x040, "DOORBELL_PAGE_BIND");
static_assert(RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT == 8, "the bind's context from bit 8");
static_assert(RGFX_FEATURE_DOORBELL_PAGES == (1ull << 46), "DOORBELL_PAGES is bit 46");

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "doorbell_pages_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// Every page, each once, in order, then none.
	{
		RgfxDoorbellPages pages;
		rgfxDoorbellPagesInit(pages);
		check(rgfxDoorbellPagesHeld(pages) == 0, "none held");
		for (uint32_t i = 0; i < RGFX_DOORBELL_PAGES; i++) {
			uint32_t page = 0;
			check(rgfxDoorbellPageTake(pages, i + 1, &page) && page == i, "the next page");
		}
		check(rgfxDoorbellPagesHeld(pages) == RGFX_DOORBELL_PAGES, "every page held");
		uint32_t page = 0;
		check(!rgfxDoorbellPageTake(pages, 1000, &page) && page == kRgfxNoDoorbellPage,
		      "none past them");

		// A page given back is taken again; nothing else is free.
		rgfxDoorbellPageFree(pages, 17);
		check(rgfxDoorbellPagesHeld(pages) == RGFX_DOORBELL_PAGES - 1, "one free");
		check(rgfxDoorbellPageTake(pages, 1001, &page) && page == 17, "the one given back");
		check(!rgfxDoorbellPageTake(pages, 1002, &page), "then none again");

		// Giving back a page past them, or no page, changes nothing.
		rgfxDoorbellPageFree(pages, RGFX_DOORBELL_PAGES);
		rgfxDoorbellPageFree(pages, kRgfxNoDoorbellPage);
		check(rgfxDoorbellPagesHeld(pages) == RGFX_DOORBELL_PAGES, "still every page held");
	}

	// Round-robin: a page given back is the last the next contexts take.
	{
		RgfxDoorbellPages pages;
		rgfxDoorbellPagesInit(pages);
		uint32_t a = 0, b = 0, c = 0;
		check(rgfxDoorbellPageTake(pages, 1, &a) && a == 0, "page 0");
		check(rgfxDoorbellPageTake(pages, 2, &b) && b == 1, "page 1");
		rgfxDoorbellPageFree(pages, a);
		check(rgfxDoorbellPageTake(pages, 3, &c) && c == 2, "past the last taken, not page 0");
		// Every page from 3 on, and page 0 last.
		uint32_t page = 0;
		for (uint32_t i = 3; i < RGFX_DOORBELL_PAGES; i++)
			check(rgfxDoorbellPageTake(pages, 10 + i, &page) && page == i, "on to the end");
		check(rgfxDoorbellPageTake(pages, 7, &page) && page == 0, "page 0 after them");
		check(!rgfxDoorbellPageTake(pages, 8, &page), "then none");
	}

	// No page for context 0, which unbinds, nor for a context of 2^24 or more, which the bind's
	// value cannot name; the last one below gets one.
	{
		RgfxDoorbellPages pages;
		rgfxDoorbellPagesInit(pages);
		uint32_t page = 0;
		check(!rgfxDoorbellPageTake(pages, 0, &page) && page == kRgfxNoDoorbellPage, "context 0");
		check(!rgfxDoorbellPageTake(pages, 1u << 24, &page), "context 2^24");
		check(!rgfxDoorbellPageTake(pages, UINT32_MAX, &page), "the last context");
		check(rgfxDoorbellPagesHeld(pages) == 0, "nothing taken");
		check(rgfxDoorbellPageTake(pages, (1u << 24) - 1, &page) && page == 0,
		      "context 2^24 - 1");
	}

	// The bind's value: the page in bits 0 to 7, the context from bit 8; the unbind's, context 0.
	check(rgfxDoorbellPageBind(5, 9) == (5u | (9u << 8)), "page 5 to context 9");
	check(rgfxDoorbellPageBind(255, (1u << 24) - 1) == UINT32_MAX, "the last page and context");
	check(rgfxDoorbellPageBind(7, 0) == 7, "an unbind is the page alone");
	// Each page's doorbell at its first byte, the last ending at the BAR's end.
	check(rgfxDoorbellPageOffset(0) == 0, "page 0 at 0");
	check(rgfxDoorbellPageOffset(1) == RGFX_DOORBELL_PAGE_SIZE, "page 1 a page on");
	check(rgfxDoorbellPageOffset(RGFX_DOORBELL_PAGES - 1) + RGFX_DOORBELL_PAGE_SIZE ==
	          RGFX_DOORBELLS_SIZE,
	      "the last page ends the BAR");

	std::printf("doorbell_pages_test: OK (%d checks)\n", checks);
	return 0;
}
