// RgfxDoorbellPages: the doorbell pages of the device's BAR 4 the kext hands the processes'
// contexts (RGFX_FEATURE_DOORBELL_PAGES, docs/graphics-kext/design.md, "Doorbell pages").
//
// BAR 4 holds RGFX_DOORBELL_PAGES pages of RGFX_DOORBELL_PAGE_SIZE bytes. The kext binds a free
// page to a context as it opens (RGFX_REG_DOORBELL_PAGE_BIND) and maps it into the context's
// process, whose bundle then writes its queues' ring ids there with no call into the kext; the
// host wakes only the bound context's rings at a page. A context past the pages, or numbered
// 2^24 or more (which the bind's value cannot name), gets none and rings through the kext's
// doorbell call, as every context did before the pages.
//
// The kext unmaps the page from the process at the context's close, then unbinds it, then gives
// it back. Pages are taken round-robin: the search starts past the last page taken, so a page
// given back is the last one the next contexts take.
//
// The caller holds the contexts' lock. Header-only and free of the kernel's headers, so its
// native test (tests/doorbell_pages_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DOORBELL_PAGES_HPP_
#define RGFX_DOORBELL_PAGES_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

static_assert(RGFX_DOORBELL_PAGES % 64 == 0, "the held pages fill whole words");
static_assert(RGFX_DOORBELL_PAGES <= (1u << RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT),
              "a bind's value names every page");

// No page: a context that has none.
static constexpr uint32_t kRgfxNoDoorbellPage = UINT32_MAX;

// The pages held, a bit each, and the page the next search starts at.
struct RgfxDoorbellPages {
	uint64_t held[RGFX_DOORBELL_PAGES / 64];
	uint32_t next;
};

// Every page free.
inline void rgfxDoorbellPagesInit(RgfxDoorbellPages &pages)
{
	for (uint64_t &word : pages.held)
		word = 0;
	pages.next = 0;
}

// Whether `context` can be bound to a page: not 0, which unbinds, and below 2^24.
inline bool rgfxDoorbellPageContext(uint32_t context)
{
	return context != 0 && context < (1u << (32 - RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT));
}

// Takes a free page for `context`, the first free one from the last taken on: false, with
// kRgfxNoDoorbellPage in *page, where every page is held or the context can be bound to none.
inline bool rgfxDoorbellPageTake(RgfxDoorbellPages &pages, uint32_t context, uint32_t *page)
{
	*page = kRgfxNoDoorbellPage;
	if (!rgfxDoorbellPageContext(context))
		return false;
	for (uint32_t i = 0; i < RGFX_DOORBELL_PAGES; i++) {
		const uint32_t at = (pages.next + i) % RGFX_DOORBELL_PAGES;
		uint64_t &word = pages.held[at / 64];
		const uint64_t bit = uint64_t(1) << (at % 64);
		if (word & bit)
			continue;
		word |= bit;
		pages.next = (at + 1) % RGFX_DOORBELL_PAGES;
		*page = at;
		return true;
	}
	return false;
}

// Gives `page` back; a page not held, or kRgfxNoDoorbellPage, is left as it is.
inline void rgfxDoorbellPageFree(RgfxDoorbellPages &pages, uint32_t page)
{
	if (page >= RGFX_DOORBELL_PAGES)
		return;
	pages.held[page / 64] &= ~(uint64_t(1) << (page % 64));
}

// How many pages are held.
inline uint32_t rgfxDoorbellPagesHeld(const RgfxDoorbellPages &pages)
{
	uint32_t n = 0;
	for (uint64_t word : pages.held)
		n += static_cast<uint32_t>(__builtin_popcountll(word));
	return n;
}

// The RGFX_REG_DOORBELL_PAGE_BIND value that binds `page` to `context`, or, for context 0,
// unbinds it.
inline uint32_t rgfxDoorbellPageBind(uint32_t page, uint32_t context)
{
	return page | (context << RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT);
}

// Where `page`'s doorbell is in BAR 4: its first byte.
inline uint64_t rgfxDoorbellPageOffset(uint32_t page)
{
	return static_cast<uint64_t>(page) * RGFX_DOORBELL_PAGE_SIZE;
}

#endif // RGFX_DOORBELL_PAGES_HPP_
