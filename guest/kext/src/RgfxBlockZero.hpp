// RgfxBlockZero: what of a region block the kext zeroes as it hands the block to a process, so that
// no block a process maps holds bytes another context left there. A block's offset is given out
// again once its last holder freed it and the device answered its release, or its holder closed,
// in any process; and the region's pages keep their bytes across the device's reset and the
// guest's reboot. So each new block, before any process's window maps it, has zeros written over
// exactly the spans that window maps (rgfxBlockZeroSpans, which allocBlock maps from too), each
// through a kernel mapping of the cache mode the window maps it with: the zeros and the process's
// stores never meet through mappings of two modes. Those kernel mappings are of the whole blocks'
// range, one a mode, made at the kext's start and kept, so a block's zeros cost only their stores;
// a span is mapped alone only where the kernel would not make them. A surface's new block is
// zeroed over the part its holders map (rgfxSurfaceZeroSpan): the whole copy for a copy block, the
// header for one in place; the runs past that part are the kext's own, never mapped into a
// process. A block held again (a surface's, by a process that shares it, or revived) holds its own
// surface's bytes, and a pages block is never in a window: neither is zeroed.
//
// Header-only and free of the kernel's headers, so its native test (tests/block_zero_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_BLOCK_ZERO_HPP_
#define RGFX_BLOCK_ZERO_HPP_

#include <stdint.h>

#include "RgfxLatency.hpp"
#include "RgfxWindowSpans.hpp"

// The blocks the kext zeroes, by what the asker's mapping makes them, as its log line names them.
enum RgfxZeroKind : uint32_t {
	// Write-combined from the first byte (writeCombinedFrom 0): command buffers' bodies.
	kRgfxZeroBodies = 0,
	// Write-back up to a page's offset, write-combined past it: a submission ring, a library's
	// staging chunks.
	kRgfxZeroSplit,
	// Write-back whole: buffers' slabs, heaps, copies and the rest.
	kRgfxZeroWriteBack,
	// A surface's copy block, the whole copy.
	kRgfxZeroSurfaceCopy,
	// The part of a surface's block in place that its holders map.
	kRgfxZeroSurfaceHeader,
	kRgfxZeroKinds,
};

// A span of the region a window maps, at its offset, and whether write-combined.
struct RgfxZeroSpan {
	uint64_t offset;
	uint64_t length;
	bool writeCombined;
};

// The kind of a block asked as `size` bytes, write-combined from `writeCombinedFrom`.
inline RgfxZeroKind rgfxBlockZeroKind(uint64_t size, uint64_t writeCombinedFrom)
{
	if (writeCombinedFrom == 0)
		return kRgfxZeroBodies;
	return writeCombinedFrom < size ? kRgfxZeroSplit : kRgfxZeroWriteBack;
}

// The spans a window maps of a block of `bytes` (whole pages, its size as allocated) at region
// offset `offset`, asked as `size` bytes write-combined from `writeCombinedFrom`, in offset order:
// its write-back part (rgfxWindowWriteBackPart), then the write-combined rest. Where the kext has
// no write-combined descriptor of the region (`combines` false, as on arm64), a write-combined
// span is mapped write-back, and the whole block is one write-back span. How many spans (1 or 2;
// 0 for no bytes).
inline uint32_t rgfxBlockZeroSpans(uint64_t offset, uint64_t size, uint64_t bytes,
                                   uint64_t writeCombinedFrom, bool combines, RgfxZeroSpan out[2])
{
	if (!bytes)
		return 0;
	const uint64_t writeBack = combines ? rgfxWindowWriteBackPart(size, bytes, writeCombinedFrom)
	                                    : bytes;
	uint32_t n = 0;
	if (writeBack)
		out[n++] = {offset, writeBack, false};
	if (writeBack < bytes)
		out[n++] = {offset + writeBack, bytes - writeBack, true};
	return n;
}

// The span a window maps of a surface's new block at region offset `offset`, whose holders use
// `size` bytes: their whole pages, write-back.
inline RgfxZeroSpan rgfxSurfaceZeroSpan(uint64_t offset, uint64_t size)
{
	return {offset, rgfxWindowSpanLength(size), false};
}

// What the zeroing took, for the kext's log line: by kind, each block's time from its first
// span's zeros begun to its last's done, and its bytes; of every kind, the time in making and
// taking down a span's own mappings (none where the kept ones hold it) and the time in the
// stores; and the grants the kernel would not map a span for, which failed.
struct RgfxZeroStats {
	RgfxLatency took[kRgfxZeroKinds];
	uint64_t ns[kRgfxZeroKinds];
	uint64_t bytes[kRgfxZeroKinds];
	uint64_t mapNs;
	uint64_t storeNs;
	uint64_t failed;
	uint64_t loggedCount;
	uint64_t loggedAt;
};

// One block of `kind` zeroed, `bytes` of it in `ns` in all, `mapNs` of that in its mappings and
// `storeNs` in its stores; or, where not `zeroed`, one whose grant failed for a span not mapped.
inline void rgfxZeroNote(RgfxZeroStats &s, RgfxZeroKind kind, uint64_t bytes, uint64_t ns,
                         uint64_t mapNs, uint64_t storeNs, bool zeroed)
{
	if (!zeroed) {
		s.failed++;
		return;
	}
	if (kind >= kRgfxZeroKinds)
		return;
	rgfxLatencyAdd(s.took[kind], ns);
	s.ns[kind] += ns;
	s.bytes[kind] += bytes;
	s.mapNs += mapNs;
	s.storeNs += storeNs;
}

// Every block counted, of every kind.
inline uint64_t rgfxZeroCount(const RgfxZeroStats &s)
{
	uint64_t n = 0;
	for (uint32_t k = 0; k < kRgfxZeroKinds; k++)
		n += s.took[k].count;
	return n;
}

#endif // RGFX_BLOCK_ZERO_HPP_
