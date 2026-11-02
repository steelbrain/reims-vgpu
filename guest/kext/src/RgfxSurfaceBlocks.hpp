// RgfxSurfaceBlocks: the accelerator kext's shared blocks, one per IOSurface plane the processes
// holding the surface ask by, which they draw and sample on the GPU (the bundle asks by each
// surface's plane 0, the block holding every plane: guest/bundle/src/RgfxIOSurface.h).
//
// A process asks for a plane's block with its own CPU mapping of the plane. The kext wires that
// range in the process's task and reads its physical pages, and the pages are the block's key:
// a process that holds the surface maps the surface's own pages, and one that does not cannot
// present them, so a surface's block goes only to the processes that hold the surface, whatever
// id they claim. Each holder's wired pages stay wired while it holds the block, so the pages
// cannot be reused for another surface under its key.
//
// A block is made in one of two modes, which the asker picks and its entry keeps. A copy block
// holds a copy of the plane, keyed by the plane's pages. An in-place block holds only what the
// holders share past the pixels (the bundle's header and writers), keyed by the pages of the
// surface's whole allocation, which the device's GPU uses in place (SURFACE_PAGES): past the
// part the processes map, the block holds the pages' runs for the device. One surface's blocks
// are all of one mode: an ask in the other mode finds no block while one of the first lives.
//
// The block is the table's (its RgfxBlockTable owner is its entry), and each holding client is
// counted: one hold per client, with a count of its asks. The device learns of each holder's
// context before the process learns of the block (BLOCK_SHARE), and of a holder letting it go
// (BLOCK_UNSHARE). The last holder's let-go frees the block (its BLOCK_RELEASE, which the device
// refuses while an object of any context lies in it). The caller holds the accelerator's lock
// around every call. Header-only and free of the kernel's headers, so its native test
// (tests/surface_blocks_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_SURFACE_BLOCKS_HPP_
#define RGFX_SURFACE_BLOCKS_HPP_

#include <stdint.h>

// RgfxPageRun, a run of physically contiguous bytes of a plane's pages, is the wire's: an
// in-place block hands its runs to the device as they are.
#include "reims_graphics_wire.h"

// The most clients that hold one plane's block.
constexpr uint32_t kRgfxSurfaceHolders = 16;

// One client's hold of a plane's block.
struct RgfxSurfaceHold {
	const void *client;
	uint32_t context;
	// How many of the client's asks it holds: the hold ends with its last let-go.
	uint32_t refs;
	// The client's wired pages of the plane (a kernel memory descriptor, opaque here), which the
	// caller unwires once the hold ends.
	void *pages;
};

// A plane's shared block, where `used`.
struct RgfxSurfaceEntry {
	bool used;
	// The block: its region offset, the bytes its holders map and its RgfxBlockTable serial.
	uint64_t offset;
	uint64_t size;
	uint64_t serial;
	// Whether the device's GPU uses the surface's pages in place, the block holding no copy.
	bool inPlace;
	// An in-place block the device has not yet answered for its pages (SURFACE_PAGES): no other
	// client holds it until the answer, which keeps the block or lets it go.
	bool pending;
	// The key: the plane's bytes and their runs (the caller's memory, which it frees once the
	// entry goes).
	uint64_t length;
	RgfxPageRun *runs;
	uint32_t runCount;
	RgfxSurfaceHold holds[kRgfxSurfaceHolders];
	uint32_t holdCount;
};

struct RgfxSurfaceTable {
	RgfxSurfaceEntry *entries;
	uint32_t capacity;
};

// The guest's page, by which the two modes' keys of one surface meet: a copy's starts at plane
// 0's first byte, which may lie past the start of the surface's first page (CoreVideo places a
// plane 512 bytes in), and one in place at the page's start.
constexpr uint64_t kRgfxSurfacePage = 0x1000;

// Whether a run starting at `a` and one starting at `b` start in the same guest page.
inline bool rgfxSurfaceSamePage(uint64_t a, uint64_t b)
{
	return a / kRgfxSurfacePage == b / kRgfxSurfacePage;
}

// The entry of mode `inPlace` whose key is the `count` runs of `runs` over `length` bytes, its
// block `size` bytes, or -1. A plane whose pages match but whose block size does not is no
// entry's, and nor is one whose first byte lies in the first page of an entry of the other mode:
// the asker gets no block (`*differs`), rather than one laid out apart from the other holders'.
inline int32_t rgfxSurfaceFind(const RgfxSurfaceTable &t, const RgfxPageRun *runs, uint32_t count,
                               uint64_t length, uint64_t size, bool inPlace, bool *differs)
{
	*differs = false;
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used)
			continue;
		if (e.inPlace != inPlace) {
			if (count && e.runCount && rgfxSurfaceSamePage(e.runs[0].address, runs[0].address)) {
				*differs = true;
				return -1;
			}
			continue;
		}
		if (e.length != length || e.runCount != count)
			continue;
		bool same = true;
		for (uint32_t r = 0; r < count && same; r++)
			same = e.runs[r].address == runs[r].address && e.runs[r].length == runs[r].length;
		if (!same)
			continue;
		if (e.size != size) {
			*differs = true;
			return -1;
		}
		return static_cast<int32_t>(i);
	}
	return -1;
}

// Whether an entry whose pages start in `runs[0]`'s page, of either mode, waits for the device's
// answer (`pending`): the asker waits for it before it looks for the plane's block.
inline bool rgfxSurfacePending(const RgfxSurfaceTable &t, const RgfxPageRun *runs, uint32_t count)
{
	for (uint32_t i = 0; i < t.capacity && count; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (e.used && e.pending && e.runCount &&
		    rgfxSurfaceSamePage(e.runs[0].address, runs[0].address))
			return true;
	}
	return false;
}

// The entry of the block at `offset`, or -1.
inline int32_t rgfxSurfaceAt(const RgfxSurfaceTable &t, uint64_t offset)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (t.entries[i].used && t.entries[i].offset == offset)
			return static_cast<int32_t>(i);
	}
	return -1;
}

// An unused entry, or -1 where every one is used.
inline int32_t rgfxSurfaceFreeEntry(const RgfxSurfaceTable &t)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (!t.entries[i].used)
			return static_cast<int32_t>(i);
	}
	return -1;
}

enum RgfxSurfaceHoldResult {
	// The client's first hold: the caller tells the device (BLOCK_SHARE), and keeps `pages`.
	kRgfxSurfaceHoldNew = 0,
	// The client holds it already, one ask more: the caller unwires `pages`.
	kRgfxSurfaceHoldAgain,
	// kRgfxSurfaceHolders clients hold it: no hold; the caller unwires `pages`.
	kRgfxSurfaceHoldFull,
};

// `client` of `context` holds the entry's block, with its wired `pages`.
inline RgfxSurfaceHoldResult rgfxSurfaceHold(RgfxSurfaceEntry &e, const void *client,
                                             uint32_t context, void *pages)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		if (e.holds[i].client == client) {
			e.holds[i].refs++;
			return kRgfxSurfaceHoldAgain;
		}
	}
	if (e.holdCount == kRgfxSurfaceHolders)
		return kRgfxSurfaceHoldFull;
	e.holds[e.holdCount++] = {client, context, 1, pages};
	return kRgfxSurfaceHoldNew;
}

enum RgfxSurfaceLetGo {
	// The client holds no such block: nothing.
	kRgfxSurfaceLetGoNotHeld = 0,
	// One ask fewer; the client holds it still.
	kRgfxSurfaceLetGoRef,
	// The client's hold ended, and other clients hold it: the caller tells the device
	// (BLOCK_UNSHARE) and unwires the pages.
	kRgfxSurfaceLetGoHold,
	// The client was its last holder: the caller frees the block (its BLOCK_RELEASE names
	// `context`), unwires the pages and lets the entry go (rgfxSurfaceClear).
	kRgfxSurfaceLetGoLast,
};

// `client` lets one ask of the entry's block go, or every one where `all` (its close). Where the
// hold ends, `*pages` is its wired pages and `*context` its context.
inline RgfxSurfaceLetGo rgfxSurfaceLetGo(RgfxSurfaceEntry &e, const void *client, bool all,
                                         void **pages, uint32_t *context)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		RgfxSurfaceHold &h = e.holds[i];
		if (h.client != client)
			continue;
		if (!all && h.refs > 1) {
			h.refs--;
			return kRgfxSurfaceLetGoRef;
		}
		*pages = h.pages;
		*context = h.context;
		e.holds[i] = e.holds[--e.holdCount];
		e.holds[e.holdCount] = {};
		return e.holdCount ? kRgfxSurfaceLetGoHold : kRgfxSurfaceLetGoLast;
	}
	return kRgfxSurfaceLetGoNotHeld;
}

// Whether `client` holds an entry's block that holds the `length` bytes at `offset`.
inline bool rgfxSurfaceHeldBy(const RgfxSurfaceTable &t, const void *client, uint64_t offset,
                              uint64_t length)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used || offset < e.offset || offset - e.offset > e.size ||
		    length > e.size - (offset - e.offset))
			continue;
		for (uint32_t h = 0; h < e.holdCount; h++) {
			if (e.holds[h].client == client)
				return true;
		}
	}
	return false;
}

// Lets the entry go once its block has none holding it; its runs are the caller's to free.
inline void rgfxSurfaceClear(RgfxSurfaceEntry &e)
{
	e = RgfxSurfaceEntry{};
}

// Appends the physical segment of `length` bytes at `addr`, the next of a plane's in its order,
// to the `*count` runs of `runs` (of `capacity`), merged with the last where they meet. False
// where it does not fit.
inline bool rgfxSurfaceAppendRun(RgfxPageRun *runs, uint32_t *count, uint32_t capacity,
                                 uint64_t addr, uint64_t length)
{
	if (length == 0)
		return true;
	if (*count && runs[*count - 1].address + runs[*count - 1].length == addr) {
		runs[*count - 1].length += length;
		return true;
	}
	if (*count == capacity)
		return false;
	runs[(*count)++] = {addr, length};
	return true;
}

#endif // RGFX_SURFACE_BLOCKS_HPP_
