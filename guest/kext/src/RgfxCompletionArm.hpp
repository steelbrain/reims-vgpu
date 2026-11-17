// RgfxCompletionArm: how the kext takes ring 0's completion ring and arms its interrupt
// (ReimsGraphics::takeCompletions, at each interrupt of the control vector, which every command
// queue's completion shares, and at a waiter's look for its answer). Header-only and free of the
// kernel's headers, over the kext's accesses to the ring's header (`Ring`), so the native test
// (tests/completion_arm_test.cpp) checks the rule as the kext keeps it.
//
// The header is uncached memory to the guest on an AMD host under xnu's PAT, and write-back once
// the kext sets the PAT's entry 2 at its start (RgfxPat.hpp; docs/graphics-kext/design.md, "The
// region"); the kext touches it only where something changed, as the bundle's completion thread
// arms its queues' rings (RgfxReplyReader::armAndCheck): a take that finds nothing leaves the
// consumer as it was, and a ring whose last arm found it empty, from which nothing has been taken
// since, is not armed again while `producer` reads as it did then, as the host clears `armed` only
// in a post, after its publish. Where the host published since, the records are taken and the ring
// armed again, the arm fenced before `producer` is read once more, so either the host's post finds
// the arm and raises the vector, or the read finds the record and takes it.
//
// Our own code.
#ifndef RGFX_COMPLETION_ARM_HPP_
#define RGFX_COMPLETION_ARM_HPP_

#include <stdint.h>

// The arm's mark where the ring is not known to be armed.
constexpr uint64_t kRgfxNotArmed = ~0ull;

// Takes the ring and arms it, `consumed` the count taken (which `Ring::take` moves) and `armedAt`
// the count at which the last arm found the ring empty (kRgfxNotArmed where it did not, or none
// was made). `Ring`:
//   uint64_t producer()   the producer, read with acquire;
//   void take(uint64_t)   takes the records up to that producer, moving `consumed`, and stores the
//                         consumer once at the end, only where it moved;
//   void arm()            stores 1 in `armed` and fences fully.
template <class Ring>
inline void rgfxTakeAndArm(Ring &ring, const uint64_t &consumed, uint64_t &armedAt)
{
	uint64_t producer = ring.producer();
	for (;;) {
		if (producer != consumed)
			ring.take(producer);
		// Armed by a look that found the ring empty, nothing taken since, nothing published since.
		if (armedAt == consumed && producer == consumed)
			return;
		ring.arm();
		producer = ring.producer();
		armedAt = producer == consumed ? consumed : kRgfxNotArmed;
		if (producer == consumed)
			return;
	}
}

#endif // RGFX_COMPLETION_ARM_HPP_
