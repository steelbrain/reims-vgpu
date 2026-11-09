// RgfxDeferCarry: a deferral that carries its surface's resident pages
// (RGFX_FEATURE_SURFACE_DEFERRED_CARRY; design.md, "Deferred pages").
//
// A surface no CPU lock has touched (its seed 0) may still have a few resident pages, written by a
// process without a lock: measured, VTDecoderXPCService writes page 0 of each new decoder output
// before its first decode, the output's other pages never resident before it. Deferred as a fresh
// surface is, its cache of zeros would lose those bytes. So the bundle lists, at its ask, the
// surface's resident pages it finds in its own mapping (mach_vm_page_range_query), at most
// kRgfxCarryPagesMost of them, and the kext proves the list under its read of the surface
// (rgfxSurfaceLocksDefer): the surface's memory holds exactly as many resident pages as listed and
// none compressed or paged out (rgfxSurfaceHoldsOnlyResident, the fresh test's own verdict, over
// the kernel's counts: where it counts no compressed or paged-out pages, nothing is deferred), the
// listed runs wire with no fault, and its counts are the same after the wire. The deferral then
// names those pages, every other run of the surface's bytes RGFX_PAGE_RUN_ZEROS, and the host
// fills its cache from them as a new cache's first fill. The kext holds them wired until the
// block's pages are named, or its release is answered. Anything else, a list that is not proven
// among it, is not deferred: its pages are named at once, as a stale surface's are.
//
// Header-only and free of the kernel's headers, so its native test (tests/defer_carry_test.cpp)
// builds it as the kext and the bundle do.
//
// Our own code; it depends only on the committed wire header and RgfxSurfaceCaches.hpp.
#ifndef RGFX_DEFER_CARRY_HPP_
#define RGFX_DEFER_CARRY_HPP_

#include <stdint.h>

#include "RgfxSurfaceCaches.hpp"
#include "reims_graphics_wire.h"

// The most pages of its surface a deferral carries: a surface with more resident is not deferred.
constexpr uint32_t kRgfxCarryPagesMost = 16;

// A run of a surface's resident pages: its first byte, from the surface's, and its bytes.
struct RgfxResidentRun {
	uint64_t offset;
	uint64_t length;
};

// A surface's resident pages, as the bundle found them in its mapping at its ask: `count` runs,
// in the surface's order. The SurfaceBlock call's struct input holds it after the layout
// (RgfxUserClientABI.hpp).
struct RgfxSurfaceResident {
	uint32_t count;
	uint32_t reserved_04;
	RgfxResidentRun runs[kRgfxCarryPagesMost];
};

// Adds the resident page at `offset` of `page` bytes to `r`, past the pages before it, merged
// with the run it follows: false, `r` unchanged, where it would take `r` past kRgfxCarryPagesMost
// pages, or it does not follow them.
inline bool rgfxResidentAdd(RgfxSurfaceResident &r, uint64_t offset, uint64_t page)
{
	uint64_t pages = 0, end = 0;
	for (uint32_t i = 0; i < r.count; i++) {
		pages += r.runs[i].length / page;
		end = r.runs[i].offset + r.runs[i].length;
	}
	if (page == 0 || pages >= kRgfxCarryPagesMost || (r.count != 0 && offset < end))
		return false;
	if (r.count != 0 && offset == end) {
		r.runs[r.count - 1].length += page;
		return true;
	}
	r.runs[r.count++] = {offset, page};
	return true;
}

// The `page`s `r` lists of a surface of `length` bytes: its runs whole `page`s of the surface, in
// its order and apart, `most` at most; 0, which carries nothing, otherwise. The bundle lists the
// pages of its task's size, which are the kernel's, and the kext counts them in those
// (rgfxSurfaceLocksPageCounts).
inline uint64_t rgfxResidentPages(const RgfxSurfaceResident &r, uint64_t length, uint64_t page,
                                  uint64_t most = kRgfxCarryPagesMost)
{
	if (page == 0 || r.count == 0 || r.count > kRgfxCarryPagesMost)
		return 0;
	uint64_t pages = 0, end = 0;
	for (uint32_t i = 0; i < r.count; i++) {
		const RgfxResidentRun &run = r.runs[i];
		if (run.length == 0 || run.offset % page || run.length % page || run.offset < end ||
		    run.length > length || run.offset > length - run.length)
			return 0;
		pages += run.length / page;
		end = run.offset + run.length;
	}
	return pages <= most ? pages : 0;
}

// What a deferral's read of its surface finds (rgfxSurfaceLocksDefer): its read-only lock taken
// (`locked`), the seed and the page counts it read (rgfxSurfaceLocksPageCounts), and the pages
// the ask listed (`listed`, rgfxResidentPages). Each verdict is rgfxSurfaceHoldsOnlyResident's,
// the one the fresh test of a deferral that lists nothing makes, so a surface whose pages the
// kernel does not count, or with a page compressed or paged out, is never deferred.
enum RgfxDeferRead {
	// Touched: a lock moved its seed, its pages were not counted, a page is compressed or paged
	// out, or more pages are resident than the ask listed, or other ones. Its pages are named at
	// once.
	kRgfxDeferReadStale = 0,
	// Fresh: no page resident, compressed or paged out. Deferred, carrying nothing.
	kRgfxDeferReadFresh,
	// Fresh but for the pages the ask listed, as many resident as it lists, none compressed or
	// paged out: deferred carrying them, once their wire proves them (rgfxDeferCarried).
	kRgfxDeferReadCarry,
};

inline RgfxDeferRead rgfxDeferRead(bool locked, uint32_t seed,
                                   const RgfxSurfacePageCounts &counts, uint64_t listed)
{
	if (rgfxSurfaceHoldsOnlyResident(locked, seed, counts, 0))
		return kRgfxDeferReadFresh;
	return listed != 0 && rgfxSurfaceHoldsOnlyResident(locked, seed, counts, listed)
	           ? kRgfxDeferReadCarry
	           : kRgfxDeferReadStale;
}

// Whether a deferral that read kRgfxDeferReadCarry, with `seed`, carries the `listed` pages:
// their runs wired with no fault (`wired`), which fails at a page that is not resident, and the
// counts read after the wire, under the same lock (`after`), as the read before found them, the
// listed pages resident, no other, none compressed or paged out: a listed page faulted in would
// raise the resident ones, and a page a writer that takes no lock dirtied meanwhile would count
// resident, or once compressed or paged out, swapped. Both together: the listed pages are the
// surface's resident ones, and no other page of it holds bytes. **Measured** on a macOS 26.7.2
// guest (x86-64), a prepare with kIODirectionPrepareNoFault of a page of a decoder output that
// is not resident fails with kIOReturnNoResources, faulting nothing in: the resident pages
// counted the same before it, after it and after its complete, on each of 14 surfaces.
inline bool rgfxDeferCarried(bool wired, uint32_t seed, const RgfxSurfacePageCounts &after,
                             uint64_t listed)
{
	return wired && listed != 0 && rgfxSurfaceHoldsOnlyResident(true, seed, after, listed);
}

// How a deferral of a surface whose ask listed resident pages ended, as the kext counts them.
enum RgfxCarryOutcome : uint32_t {
	// Carried: the listed pages are the surface's resident pages, and their runs were made.
	kRgfxCarryTaken = 0,
	// Fresh with no page resident, compressed or paged out: deferred carrying nothing.
	kRgfxCarryNone,
	// Not deferred: the surface's seed moved, or the read-only lock failed.
	kRgfxCarryTouched,
	// Not deferred: the surface's pages were not counted (rgfxSurfaceLocksPageCounts): the kernel
	// counts no compressed or paged-out pages, or its memory is not a buffer the kernel allocated.
	kRgfxCarryUncounted,
	// Not deferred: more pages resident than listed, or fewer, or a page compressed or paged out.
	kRgfxCarryCounted,
	// Not deferred: a listed run did not wire with no fault, a page of it not resident.
	kRgfxCarryUnwired,
	// Not deferred: the counts after the wire were not the ones before: a listed page faulted
	// in, or another page was written, or compressed or paged out, meanwhile.
	kRgfxCarryFaulted,
	// Not deferred: the runs did not fit, or a listed page's address was not a page's.
	kRgfxCarryNoRuns,
	kRgfxCarryOutcomes,
};

// How a deferral whose ask listed `listed` pages (rgfxResidentPages) ended: its read-only lock
// taken (`locked`), the seed and the page counts it read, the listed runs wired with no fault
// (`wired`, tried only for a read of kRgfxDeferReadCarry), the counts after the wire, and the
// runs made of them (`runs`, rgfxCarryRuns). It defers the block where it answers
// kRgfxCarryTaken or kRgfxCarryNone; every other names its pages at once.
inline RgfxCarryOutcome rgfxCarryOutcome(bool locked, uint32_t seed,
                                         const RgfxSurfacePageCounts &counts, uint64_t listed,
                                         bool wired, const RgfxSurfacePageCounts &after,
                                         uint32_t runs)
{
	switch (rgfxDeferRead(locked, seed, counts, listed)) {
	case kRgfxDeferReadFresh:
		return kRgfxCarryNone;
	case kRgfxDeferReadStale:
		if (!locked || seed != 0)
			return kRgfxCarryTouched;
		return counts.counted ? kRgfxCarryCounted : kRgfxCarryUncounted;
	case kRgfxDeferReadCarry:
		break;
	}
	if (!wired)
		return kRgfxCarryUnwired;
	if (!rgfxDeferCarried(wired, seed, after, listed))
		return kRgfxCarryFaulted;
	return runs != 0 ? kRgfxCarryTaken : kRgfxCarryNoRuns;
}

// The runs of a carrying deferral's SURFACE_PAGES (RGFX_FEATURE_SURFACE_DEFERRED_CARRY): the bytes
// of a surface of `length` bytes, whole `page`s (the device's), in its order, each `page` of `r`'s
// runs at the guest physical address `addresses` gives for it (one a page, in `r`'s order),
// merged where a page follows the one before it both in the surface and in guest RAM, every other
// run RGFX_PAGE_RUN_ZEROS. How many it wrote to `out`, `room` at most; 0 where they do not fit,
// `r` lists no whole `page`s (rgfxResidentPages), or an address is not a page's.
inline uint32_t rgfxCarryRuns(const RgfxSurfaceResident &r, const uint64_t *addresses,
                              uint64_t length, uint64_t page, RgfxPageRun *out, uint32_t room)
{
	if (rgfxResidentPages(r, length, page, UINT64_MAX) == 0)
		return 0;
	const uint64_t whole = (length + page - 1) / page * page;
	uint32_t n = 0;
	uint64_t at = 0, next = 0;
	auto add = [&](uint64_t address, uint64_t bytes) {
		const bool zeros = address == RGFX_PAGE_RUN_ZEROS;
		RgfxPageRun *last = n ? &out[n - 1] : nullptr;
		if (last && (zeros ? last->address == RGFX_PAGE_RUN_ZEROS
		                   : last->address != RGFX_PAGE_RUN_ZEROS &&
		                         last->address + last->length == address)) {
			last->length += bytes;
			return true;
		}
		if (n == room)
			return false;
		out[n++] = {address, bytes};
		return true;
	};
	for (uint32_t i = 0; i < r.count; i++) {
		const RgfxResidentRun &run = r.runs[i];
		if (run.offset > at && !add(RGFX_PAGE_RUN_ZEROS, run.offset - at))
			return 0;
		for (uint64_t p = 0; p < run.length; p += page) {
			const uint64_t address = addresses[next++];
			if (address % page || !add(address, page))
				return 0;
		}
		at = run.offset + run.length;
	}
	if (whole > at && !add(RGFX_PAGE_RUN_ZEROS, whole - at))
		return 0;
	return n;
}

#endif // RGFX_DEFER_CARRY_HPP_
