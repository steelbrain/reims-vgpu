// The native test of RgfxContextSlots.hpp: the contexts' runs of GPU-address units, taken past
// the 32 slots the kext once had and up to every unit, round-robin, aligned, kept apart, with
// WindowServer's reserve, and sized from the device's maxBufferLength.
#include "RgfxContextSlots.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "context_slots_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

uint32_t heldUnits(const RgfxContextSlots &s)
{
	uint32_t n = 0;
	for (uint32_t u = 0; u < kRgfxContextUnits; u++)
		n += rgfxContextUnitHeld(s, u) ? 1 : 0;
	return n;
}
} // namespace

int main()
{
	// The layout: 1024 units of 64 GiB fill [2^46, 2^47); the old span is 32 of them.
	check(kRgfxContextUnits == 1024, "1024 units");
	check(RGFX_CONTEXT_VA_UNIT == uint64_t{64} << 30, "of 64 GiB");
	check(rgfxContextBase(0) == uint64_t{1} << 46, "the first from 2^46");
	check(rgfxContextBase(kRgfxContextUnits - 1) + RGFX_CONTEXT_VA_UNIT == uint64_t{1} << 47,
	      "the last to 2^47");
	check(kRgfxContextOldSpanUnits == 32, "the old span is 32 units");
	check(kRgfxContextReservedUnits == kRgfxContextOldSpanUnits,
	      "WindowServer's reserve holds one context of the old span");

	// Units a context holds: one unit holds twice maxBufferLength up to 32 GiB, as the RTX 5080
	// laptop's 12 GiB (three quarters of its 17,094,934,528-byte working set, 16 KiB aligned);
	// more for a larger device, never more than the old span.
	check(rgfxContextUnitsFor(0) == 1, "one unit where the device names no maxBufferLength");
	check(rgfxContextUnitsFor(12821200896) == 1, "one on the 5080");
	check(rgfxContextUnitsFor(uint64_t{32} << 30) == 1, "one up to 32 GiB");
	check(rgfxContextUnitsFor((uint64_t{32} << 30) + 1) == 2, "two past it");
	check(rgfxContextUnitsFor(uint64_t{77} << 30) == 4, "four for 77 GiB");
	check(rgfxContextUnitsFor(uint64_t{1} << 40) == 32, "32 for 1 TiB");
	check(rgfxContextUnitsFor(~uint64_t{0}) == 32, "never more than the old span");
	for (uint64_t length = 1; length < (uint64_t{1} << 41); length *= 3) {
		const uint32_t units = rgfxContextUnitsFor(length);
		check((units & (units - 1)) == 0, "a power of two of units");
		check(units == kRgfxContextOldSpanUnits || units * RGFX_CONTEXT_VA_UNIT >= 2 * length,
		      "whose span holds twice maxBufferLength");
	}

	// One unit a context: the 33rd context, which 32 slots refused, and every one up to the
	// reserve is taken, in order, each its own unit; the next is refused, but WindowServer's is
	// not, up to the last unit.
	{
		RgfxContextSlots s = kRgfxContextSlotsEmpty;
		uint32_t taken = 0, first = 0;
		while (rgfxContextSlotsTake(s, 1, false, &first)) {
			check(first == taken, "round-robin from unit 0");
			taken++;
		}
		check(taken > 32, "past the 32 slots");
		check(taken == kRgfxContextUnits - kRgfxContextReservedUnits,
		      "every unit but WindowServer's reserve");
		check(heldUnits(s) == taken && s.freeUnits == kRgfxContextReservedUnits,
		      "each unit held once");
		check(!rgfxContextSlotsTake(s, 1, false, &first), "the next refused");
		for (uint32_t i = 0; i < kRgfxContextReservedUnits; i++) {
			check(rgfxContextSlotsTake(s, 1, true, &first), "WindowServer's in the reserve");
			check(first == taken + i, "in order");
		}
		check(s.freeUnits == 0 && heldUnits(s) == kRgfxContextUnits, "every unit held");
		check(!rgfxContextSlotsTake(s, 1, true, &first), "and none past the last");

		// A unit given back is the last the next contexts take: the search goes on past the
		// last run taken, and comes round to it.
		rgfxContextSlotsGive(s, 100, 1);
		rgfxContextSlotsGive(s, 900, 1);
		check(s.freeUnits == 2, "two units back");
		check(rgfxContextSlotsTake(s, 1, true, &first) && first == 100,
		      "the search comes round to the first given back");
		check(rgfxContextSlotsTake(s, 1, true, &first) && first == 900, "then the next");
		// A run given back twice, or one not held, changes nothing.
		rgfxContextSlotsGive(s, 900, 1);
		rgfxContextSlotsGive(s, 900, 1);
		check(s.freeUnits == 1 && !rgfxContextUnitHeld(s, 900), "a second give is ignored");
		rgfxContextSlotsGive(s, kRgfxContextUnits, 1);
		rgfxContextSlotsGive(s, kRgfxContextUnits - 1, 2);
		rgfxContextSlotsGive(s, 0, 0);
		check(s.freeUnits == 1, "a run past the end, or of no units, is ignored");
		check(rgfxContextSlotsTake(s, 1, true, &first) && first == 900, "and taken again");
	}

	// Freed in the meantime: a unit freed behind the search is taken only once it comes round.
	{
		RgfxContextSlots s = kRgfxContextSlotsEmpty;
		uint32_t first = 0;
		for (uint32_t i = 0; i < 40; i++)
			check(rgfxContextSlotsTake(s, 1, false, &first), "40 contexts");
		rgfxContextSlotsGive(s, 3, 1);
		check(rgfxContextSlotsTake(s, 1, false, &first) && first == 40,
		      "the next past the last taken, not the unit given back");
	}

	// The old span: 32 units, aligned, beside contexts of one unit, which never meet it.
	{
		RgfxContextSlots s = kRgfxContextSlotsEmpty;
		uint32_t first = 0, old = 0;
		check(rgfxContextSlotsTake(s, 1, false, &first) && first == 0, "a context of one unit");
		check(rgfxContextSlotsTake(s, kRgfxContextOldSpanUnits, false, &old), "one of the old span");
		check(old == 32, "at the next aligned run");
		check(rgfxContextBase(old) % RGFX_CONTEXT_VA_SPAN == 0, "its base 2 TiB aligned");
		check(rgfxContextRunIs(s, old, kRgfxContextOldSpanUnits, true), "all of it held");
		check(rgfxContextSlotsTake(s, 1, false, &first) && first == 64, "the next one past it");
		rgfxContextSlotsGive(s, old, kRgfxContextOldSpanUnits);
		check(rgfxContextRunIs(s, old, kRgfxContextOldSpanUnits, false), "given back whole");
		check(s.freeUnits == kRgfxContextUnits - 2, "two units held");
		// Fill all but the reserve with old spans' contexts, which need whole aligned runs.
		uint32_t olds = 0;
		while (rgfxContextSlotsTake(s, kRgfxContextOldSpanUnits, false, &old)) {
			check(old % kRgfxContextOldSpanUnits == 0, "each aligned");
			olds++;
		}
		// Units 0 and 64 split two runs, whose 62 free units are WindowServer's reserve.
		check(olds == kRgfxContextUnits / kRgfxContextOldSpanUnits - 2, "every whole run");
		check(s.freeUnits == 2 * kRgfxContextOldSpanUnits - 2, "the split runs' units free");
		// The search went round to the run given back (32), so it goes on past it, at 65.
		check(rgfxContextSlotsTake(s, 1, true, &first) && first == 65,
		      "and WindowServer's context of one unit takes one");
	}

	// The runs taken never meet: random takes and gives of 1, 2, 4 and 32 units.
	{
		RgfxContextSlots s = kRgfxContextSlotsEmpty;
		struct Run {
			uint32_t first, units;
		} runs[kRgfxContextUnits];
		uint32_t count = 0;
		uint32_t seed = 12345;
		const uint32_t sizes[] = {1, 1, 1, 2, 4, 32};
		for (int step = 0; step < 200000; step++) {
			seed = seed * 1103515245 + 12345;
			if (count > 0 && (seed >> 16) % 3 == 0) {
				const uint32_t i = (seed >> 8) % count;
				rgfxContextSlotsGive(s, runs[i].first, runs[i].units);
				runs[i] = runs[--count];
				continue;
			}
			const uint32_t units = sizes[(seed >> 20) % 6];
			uint32_t first = 0;
			if (!rgfxContextSlotsTake(s, units, (seed >> 4) % 2 == 0, &first))
				continue;
			check(first % units == 0, "aligned");
			for (uint32_t i = 0; i < count; i++)
				check(first + units <= runs[i].first || runs[i].first + runs[i].units <= first,
				      "no two runs meet");
			runs[count++] = {first, units};
		}
		uint32_t held = 0;
		for (uint32_t i = 0; i < count; i++)
			held += runs[i].units;
		check(held == heldUnits(s) && s.freeUnits == kRgfxContextUnits - held,
		      "the free count is the units no run holds");
	}

	// A take of no units, of more than every unit, or of a count that is no power of two, takes
	// nothing.
	{
		RgfxContextSlots s = kRgfxContextSlotsEmpty;
		uint32_t first = 7;
		check(!rgfxContextSlotsTake(s, 0, true, &first), "no units");
		check(!rgfxContextSlotsTake(s, 3, true, &first), "three units");
		check(!rgfxContextSlotsTake(s, 2 * kRgfxContextUnits, true, &first), "more than all");
		check(first == 7 && s.freeUnits == kRgfxContextUnits && heldUnits(s) == 0, "nothing held");
		check(rgfxContextSlotsTake(s, kRgfxContextUnits, true, &first) && first == 0,
		      "every unit, WindowServer's");
	}

	std::printf("context_slots_test: OK (%d checks)\n", checks);
	return 0;
}
