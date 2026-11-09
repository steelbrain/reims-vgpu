// RgfxLegacySwapLayout: where IOMobileGraphicsFamily's IOMobileFramebufferLegacy keeps its swap
// queue, its display's size and what a swap's swap-info record is made of (RgfxLegacySwapInfo.hpp)
// on each arm64 release the arm64e kext runs on, the queue as a
// display subclass's swap_begin_gated takes its swaps from it, and the size as its
// set_digital_out_mode sets it (ReimsGraphicsMobile.cpp). The class declares no accessor for them.
// Apple's paravirtual display, the class's subclass in a VM (AppleParavirtDisplay), reads and
// writes the queue at these offsets in its swap_begin_gated, clean_queue_gated and
// program_hardware_gated, and the size in its set_digital_out_mode, and the class's own
// swap_complete_gated and dropSwap_gated do the same with the queue, and its get_display_size with
// the size (read from each release's AppleParavirtGPU and IOMobileGraphicsFamily). Each release's class is told by its size, the size its metaclass gives
// it. Header-only and free of the kernel's headers, for the native tests.
//
// Our own code.
#ifndef RGFX_LEGACY_SWAP_LAYOUT_HPP_
#define RGFX_LEGACY_SWAP_LAYOUT_HPP_

#include <stddef.h>
#include <stdint.h>

// Byte offsets in the class and in a swap (IOMFBSwapIORequest).
struct RgfxLegacySwapLayout {
	// The release, for the log.
	const char *release;
	// The class's size, which tells the release.
	uint32_t classSize;
	// In the class: the swap being shown (null where none), then the queue of swaps waiting to
	// begin, a TAILQ head: its first swap, then the address of the last swap's next pointer (of
	// the first pointer where the queue is empty).
	uint32_t current;
	// In a swap: its TAILQ entry in that queue, the next swap and then the address of the pointer
	// that points at the swap.
	uint32_t link;
	// In a swap: a word the paravirtual display begins the swap at where it holds 0 or 8. At any
	// other value it drops the swap instead.
	uint32_t state;
	// In a swap: a word whose bit N says layer N has a surface.
	uint32_t layers;
	// In a swap: each layer's IOSurface, layer 0's first.
	uint32_t surfaces;
	// In the class: the display's width, then its height, in pixels, two words. The class's
	// get_display_size, display_width and display_height answer them, and its swap_submit and
	// default framebuffer surface read them; the class never writes them, and the paravirtual
	// display's set_digital_out_mode writes both, in its gate.
	uint32_t displaySize;
	// What the paravirtual display's notify_swap_done_gated reads to make a swap's swap-info
	// record (RgfxLegacySwapInfo.hpp), with the class's own uses of each:
	// - in the class, the time of the last vblank, a doubleword the class's note_vbl_ISR writes;
	// - in a swap, the doubleword the class's swap_submit copies from offset 0 of the swap's
	//   IOMFBSwapRec, whose meaning nothing states (field_00), adding the time where the record
	//   asks;
	// - in a swap, a word whose bit 0 the class's abort_swap_gated and dropSwap_gated set;
	// - in the class, a halfword: where it is not 0, the paravirtual display keeps the record
	//   instead of dispatching it.
	uint32_t vblankTime;
	uint32_t swapRecordField00;
	uint32_t swapFlags;
	uint32_t notifyHeld;
};

// macOS 14.6 (Darwin 23.6), 15.6 (24.6), 26.6 (25.6), and 27.0 and 27.0.1 (27.0): the arm64
// releases Ambrosia boots from 14 on, where the arm64e kext loads (its MinKernel is 23).
constexpr RgfxLegacySwapLayout kRgfxLegacySwapLayouts[] = {
    {"14", 0x18a0, 0x1560, 0xb90, 0x32c, 0x318, 0xa40, 0x16b0, 0x108, 0x2d0, 0x310, 0x189e},
    {"15", 0x19c0, 0x1668, 0xc00, 0x32c, 0x318, 0xa48, 0x17c8, 0x108, 0x2d0, 0x310, 0x19be},
    {"26", 0x1af8, 0x1790, 0xc78, 0x34c, 0x338, 0xac0, 0x1900, 0x108, 0x2f0, 0x330, 0x1af6},
    {"27", 0x1dd8, 0x1a40, 0xd38, 0x3ec, 0x3d8, 0xb80, 0x1bf0, 0x108, 0x390, 0x3d0, 0x1dd2},
};

// The layout of the release whose IOMobileFramebufferLegacy is `classSize` bytes, or null for a
// size that no release here has.
inline const RgfxLegacySwapLayout *rgfxLegacySwapLayoutOf(uint32_t classSize)
{
	for (const RgfxLegacySwapLayout &l : kRgfxLegacySwapLayouts)
		if (l.classSize == classSize)
			return &l;
	return nullptr;
}

// Sets the display's size in the class's own fields at a release's offsets: `framebuffer` is the
// IOMobileFramebufferLegacy. Call in the class's gate, as the paravirtual display writes them.
inline void rgfxLegacySetDisplaySize(void *framebuffer, const RgfxLegacySwapLayout &layout,
                                     uint32_t width, uint32_t height)
{
	uint32_t *size = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(framebuffer) +
	                                              layout.displaySize);
	size[0] = width;
	size[1] = height;
}

// One framebuffer's swap queue at a release's offsets: `framebuffer` is the
// IOMobileFramebufferLegacy, and each swap is the IOMFBSwapIORequest's address.
class RgfxLegacySwapQueue {
public:
	RgfxLegacySwapQueue(void *framebuffer, const RgfxLegacySwapLayout &layout)
	    : base(static_cast<uint8_t *>(framebuffer)), l(layout)
	{
	}

	void *current() const { return *pointer(base + l.current); }
	void setCurrent(void *swap) { *pointer(base + l.current) = swap; }
	void *first() const { return *pointer(base + l.current + 8); }

	// Whether the first swap is linked as the head's first entry is, both ways, so that taking it
	// off the queue (removeFirst) writes only the queue's own pointers. The paravirtual display and
	// the class's dropSwap_gated check the same two links and stop the kernel where either fails.
	// This check reads no pointer it has not checked first, except the next swap's back link.
	bool firstLinked() const
	{
		uint8_t *swap = static_cast<uint8_t *>(first());
		if (!swap || *pointer(swap + l.link + 8) != static_cast<void *>(base + l.current + 8))
			return false;
		uint8_t *next = static_cast<uint8_t *>(*pointer(swap + l.link));
		return !next || *pointer(next + l.link + 8) == static_cast<void *>(swap + l.link);
	}

	// Takes the first swap off the queue, as TAILQ_REMOVE does, where firstLinked holds.
	void removeFirst()
	{
		uint8_t *swap = static_cast<uint8_t *>(first());
		uint8_t *next = static_cast<uint8_t *>(*pointer(swap + l.link));
		void **back = static_cast<void **>(*pointer(swap + l.link + 8));
		if (next)
			*pointer(next + l.link + 8) = back;
		else
			*pointer(base + l.current + 16) = back;
		*back = next;
	}

	// Whether the paravirtual display would begin `swap` rather than drop it.
	bool begins(const void *swap) const { return (word(swap, l.state) & ~8u) == 0; }

	// Layer `layer`'s surface in `swap`, or null where the swap gives the layer none.
	void *surface(const void *swap, unsigned layer) const
	{
		if ((word(swap, l.layers) & (1u << layer)) == 0)
			return nullptr;
		return *pointer(static_cast<const uint8_t *>(swap) + l.surfaces + 8 * layer);
	}

private:
	static void **pointer(const uint8_t *at)
	{
		return reinterpret_cast<void **>(const_cast<uint8_t *>(at));
	}
	static uint32_t word(const void *swap, uint32_t offset)
	{
		return *reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(swap) + offset);
	}

	uint8_t *base;
	const RgfxLegacySwapLayout &l;
};

#endif // RGFX_LEGACY_SWAP_LAYOUT_HPP_
