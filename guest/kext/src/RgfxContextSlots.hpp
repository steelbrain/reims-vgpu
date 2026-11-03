// RgfxContextSlots: the ranges of GPU addresses the kext hands the processes' contexts.
//
// The guest's GPU addresses, [RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_END), are
// RGFX_CONTEXT_VA_UNITS units of RGFX_CONTEXT_VA_UNIT bytes (1024 of 64 GiB). A context holds an
// aligned run of units, a power of two of them, from its open to its close: its bundle picks
// every GPU address of its objects there, and the host pins them as given, so no two processes'
// picks meet. The host knows none of this: it only keeps its own ranges below the guest's.
//
// How many units a context holds (rgfxContextUnitsFor): one, unless the device's maxBufferLength
// is more than half a unit; then the least power of two that holds twice it, so a buffer of
// maxBufferLength fits beside as much again of the process's other objects. A bundle that takes
// no span from the kext (OpenContext's five values) picks through RGFX_CONTEXT_VA_SPAN from its
// base, so it gets that span: an aligned run of kRgfxContextOldSpanUnits units.
//
// Sized from what a busy desktop's processes hold (docs/graphics-kext/design.md, "Contexts",
// measured on the RTX 5080 host): 74 processes held a context with 42 apps and 7 WebGL and WebGPU
// pages open, where 32 slots of 2 TiB refused the 33rd process's and every later one's, and the
// most any one held of its range at once was WindowServer's 204 MiB, so a unit holds it over 300
// times.
//
// Runs are taken round-robin: the search starts past the last run taken, so a run given back is
// the last one the next contexts take. The last kRgfxContextReservedUnits units free go to
// WindowServer alone, so a WindowServer started again finds a range however many processes hold
// the others (as RgfxBlockTable's reserve does for blocks).
//
// The caller holds the contexts' lock. Header-only and free of the kernel's headers, so its
// native test (tests/context_slots_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_CONTEXT_SLOTS_HPP_
#define RGFX_CONTEXT_SLOTS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The units of the guest's GPU addresses.
static constexpr uint32_t kRgfxContextUnits = RGFX_CONTEXT_VA_UNITS;
// The units of RGFX_CONTEXT_VA_SPAN, the span of a context whose bundle takes none from the kext,
// and the most a context holds.
static constexpr uint32_t kRgfxContextOldSpanUnits =
    static_cast<uint32_t>(RGFX_CONTEXT_VA_SPAN / RGFX_CONTEXT_VA_UNIT);
// The units only WindowServer's contexts take: one context of the old span, or as many of fewer
// units.
static constexpr uint32_t kRgfxContextReservedUnits = kRgfxContextOldSpanUnits;

static_assert(kRgfxContextUnits % 64 == 0, "the held units fill whole words");
static_assert(kRgfxContextOldSpanUnits * RGFX_CONTEXT_VA_UNIT == RGFX_CONTEXT_VA_SPAN,
              "the old span is whole units");
static_assert((kRgfxContextOldSpanUnits & (kRgfxContextOldSpanUnits - 1)) == 0,
              "the old span is a power of two of units");

// The units held, a bit each, the unit the next search starts at, and how many are free.
struct RgfxContextSlots {
	uint64_t held[kRgfxContextUnits / 64];
	uint32_t next;
	uint32_t freeUnits;
};

// Every unit free.
static constexpr RgfxContextSlots kRgfxContextSlotsEmpty = {{}, 0, kRgfxContextUnits};

// The units a context holds where the device's maxBufferLength is `maxBufferLength` (0 where the
// device names none): the least power of two, at most kRgfxContextOldSpanUnits, whose span holds
// twice it.
inline uint32_t rgfxContextUnitsFor(uint64_t maxBufferLength)
{
	uint32_t units = 1;
	while (units < kRgfxContextOldSpanUnits &&
	       units * RGFX_CONTEXT_VA_UNIT / 2 < maxBufferLength)
		units *= 2;
	return units;
}

// The GPU address unit `first` starts at.
inline uint64_t rgfxContextBase(uint32_t first)
{
	return RGFX_CONTEXT_VA_BASE + uint64_t{first} * RGFX_CONTEXT_VA_UNIT;
}

inline bool rgfxContextUnitHeld(const RgfxContextSlots &s, uint32_t unit)
{
	return ((s.held[unit / 64] >> (unit % 64)) & 1) != 0;
}

// Whether every unit of the run of `units` from `first` is held (`held`) or every one free.
inline bool rgfxContextRunIs(const RgfxContextSlots &s, uint32_t first, uint32_t units, bool held)
{
	for (uint32_t u = first; u < first + units; u++)
		if (rgfxContextUnitHeld(s, u) != held)
			return false;
	return true;
}

inline void rgfxContextRunMark(RgfxContextSlots &s, uint32_t first, uint32_t units, bool held)
{
	for (uint32_t u = first; u < first + units; u++) {
		const uint64_t bit = uint64_t{1} << (u % 64);
		s.held[u / 64] = held ? s.held[u / 64] | bit : s.held[u / 64] & ~bit;
	}
}

// Takes a free run of `units` units (a power of two, at most kRgfxContextUnits) whose first is a
// multiple of `units`, the first such past the last run taken, and answers its first unit in
// `*first`. False, taking nothing, where none is free, or where it would leave fewer than
// kRgfxContextReservedUnits free and the context is not WindowServer's (`mayReserve`).
inline bool rgfxContextSlotsTake(RgfxContextSlots &s, uint32_t units, bool mayReserve,
                                 uint32_t *first)
{
	if (units == 0 || units > kRgfxContextUnits || (units & (units - 1)) != 0)
		return false;
	if (s.freeUnits < units || (!mayReserve && s.freeUnits - units < kRgfxContextReservedUnits))
		return false;
	const uint32_t runs = kRgfxContextUnits / units;
	const uint32_t start = ((s.next + units - 1) / units) % runs;
	for (uint32_t i = 0; i < runs; i++) {
		const uint32_t at = ((start + i) % runs) * units;
		if (!rgfxContextRunIs(s, at, units, false))
			continue;
		rgfxContextRunMark(s, at, units, true);
		s.freeUnits -= units;
		s.next = (at + units) % kRgfxContextUnits;
		*first = at;
		return true;
	}
	return false;
}

// Gives back the run of `units` units from `first` a take answered. A run not held whole, or
// past the units' end, is left as it is.
inline void rgfxContextSlotsGive(RgfxContextSlots &s, uint32_t first, uint32_t units)
{
	if (units == 0 || first >= kRgfxContextUnits || units > kRgfxContextUnits - first ||
	    !rgfxContextRunIs(s, first, units, true))
		return;
	rgfxContextRunMark(s, first, units, false);
	s.freeUnits += units;
}

#endif // RGFX_CONTEXT_SLOTS_HPP_
