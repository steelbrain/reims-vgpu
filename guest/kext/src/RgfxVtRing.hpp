/*
 * RgfxVtRing.hpp — the arm64e kext's take of the VideoToolbox ring (RGFX_FEATURE_VT_RING,
 * design.md, "Decided", 17): the host's messages to the guest's VideoToolbox, each a VT_MESSAGE
 * record as the event of that name is, which display 0's framebuffer takes through the count each
 * VT_RING event carries, as it takes that event from the control completion ring
 * (ReimsGraphics::vtRing). The ring lies in the region's room before the guest's allocations,
 * which the framebuffer maps with the control rings: it is the device's memory, and the kext wires
 * nothing of the guest's RAM for it.
 *
 * Header-only and free of kernel types, so tests/vt_ring_test.cpp checks it on the Mac.
 */

#ifndef RGFX_VT_RING_HPP_
#define RGFX_VT_RING_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// What a take found wrong with the ring, which takes nothing more of it: none, a count out of the
// ring's reach, or a record whose size the protocol does not allow.
enum class RgfxVtRingFault : uint32_t {
	None = 0,
	Count = 1,
	Size = 2,
};

// What a take of the ring did: the count it took through, and what it found wrong.
struct RgfxVtRingTaken {
	uint64_t consumed;
	RgfxVtRingFault fault;
};

// Takes the VideoToolbox ring's records from count `consumed` through `produced`, the count a
// VT_RING carries, over the ring's byte ring `bytes` (RGFX_VT_RING_DATA_SIZE bytes, after its
// header): each VT_MESSAGE record goes to `message` as the bytes after its message header and
// their size, in order, as a VT_MESSAGE event's payload goes to ReimsGraphics::vtMessage; a pad
// (RGFX_MSG_PAD) skips to the ring's end, and a record of another type is passed. A `produced` the
// take has reached already takes nothing. One ahead of `consumed` by more than the ring holds, or
// a record whose size is not a multiple of 8, is under its header or reaches past the ring's end
// or `produced`, takes nothing more: the count moves to `produced`, as the control completion
// ring's take does (ReimsGraphics::drainEvents), and the fault is answered. The caller stores the
// count answered to the ring's `consumer`.
template <typename Message>
inline RgfxVtRingTaken rgfxTakeVtRing(const volatile uint8_t *bytes, uint64_t consumed,
                                      uint64_t produced, Message &&message)
{
	constexpr uint64_t kSize = RGFX_VT_RING_DATA_SIZE;
	constexpr uint32_t kHeader = sizeof(RgfxMsgHeader);
	static_assert((kSize & (kSize - 1)) == 0, "the ring's size is a power of two");
	if (static_cast<int64_t>(produced - consumed) <= 0)
		return {consumed, RgfxVtRingFault::None};
	if (produced - consumed > kSize)
		return {produced, RgfxVtRingFault::Count};
	while (consumed != produced) {
		const uint32_t offset = static_cast<uint32_t>(consumed & (kSize - 1));
		const uint32_t tail = static_cast<uint32_t>(kSize) - offset;
		const volatile uint8_t *m = bytes + offset;
		const uint32_t type = *reinterpret_cast<const volatile uint32_t *>(m);
		const uint32_t size = *reinterpret_cast<const volatile uint32_t *>(m + 4);
		const uint64_t left = produced - consumed;
		if (type == RGFX_MSG_PAD) {
			if (tail > left)
				return {produced, RgfxVtRingFault::Size};
			consumed += tail;
			continue;
		}
		if (size < kHeader || size % 8 || size > tail || size > left)
			return {produced, RgfxVtRingFault::Size};
		if (type == RGFX_EVT_VT_MESSAGE)
			message(m + kHeader, size - kHeader);
		consumed += size;
	}
	return {consumed, RgfxVtRingFault::None};
}

#endif // RGFX_VT_RING_HPP_
