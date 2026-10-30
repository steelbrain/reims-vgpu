/*
 * RgfxRegion: the guest bundle's allocator over the reims-graphics device's BAR 2
 * region (the committed region layout in reims_graphics_wire.h). The device maps
 * the region write-back and hands this allocator its base and size; this hands out
 * aligned blocks past the host-reserved head of the region:
 *   - per-queue submission + completion ring pairs (a ring header then a power-of-
 *     two byte ring, placed for RgfxSerializer::createQueue), and
 *   - resource blocks for Shared / no-copy buffers and Shared heaps (placed for
 *     createBufferOverHostMemory / createHeapOverHostMemory).
 *
 * THE RESERVED HEAD. The region's first RGFX_ALLOCATIONS_OFFSET bytes are the host's:
 * the control submission ring at RGFX_CONTROL_SUBMIT_OFFSET, the control completion ring
 * at RGFX_CONTROL_COMPLETE_OFFSET, and the device-info caps page at
 * RGFX_DEVICE_INFO_OFFSET in the control completion ring's tail. (The MSI-X table lives
 * in BAR 0, not here.) The framebuffer's scanout follows (RGFX_SCANOUT_SPAN), and the
 * accelerator kext's blocks start at RGFX_CONTEXT_ALLOCATIONS_OFFSET.
 *
 * THE POLICY. In the bundle, every block comes from the accelerator kext (setBlockSource):
 * the kext owns the region's blocks for every process, hands each out to one process and
 * frees the process's at its end, so processes never place blocks over each other.
 * release() hands a block back. Without a block source (the unit tests, over a plain
 * buffer) it is deterministic bump allocation from the floor, and release() does nothing.
 * Allocation is thread-safe (Metal creates resources and queues on the app's threads). Rings are
 * aligned to a cache line so the ring header's 64-byte lines never share one; resource
 * blocks default to a page so a block can later back a mapping.
 *
 * Our own code; depends only on the committed wire header. In a unit test the region
 * is a plain buffer; in the bundle it is the BAR-mapped region.
 */

#ifndef RGFX_REGION_HPP_
#define RGFX_REGION_HPP_

#include <cstdint>
#include <mutex>

#include "reims_graphics_wire.h"

class RgfxRegion {
public:
	/// A block handed out of the region. `offset` is its place in the region (always
	/// at or above the floor), `ptr` the CPU pointer base + offset, `size` the bytes
	/// reserved. `ok` is false when the region had no room (the other fields are 0).
	struct Block {
		uint64_t offset;
		volatile uint8_t *ptr;
		uint64_t size;
		bool ok;
	};

	/// A submission + completion ring pair for a queue. The offsets and data sizes
	/// are what RgfxSerializer::createQueue carries; each ring's header is zeroed.
	/// `complete_data_size` 0 allocates no completion ring (the single-ring model
	/// routes replies on the control completion ring); `complete_offset` is then 0.
	/// `ok` is false when a size is not a power of two or the region had no room.
	struct RingPair {
		uint64_t submit_offset;
		uint64_t complete_offset;
		uint32_t submit_data_size;
		uint32_t complete_data_size;
		bool ok;
	};

	/// The allocator over the region whose CPU base is `base`, `size` bytes long.
	/// Allocation starts at the floor (RGFX_ALLOCATIONS_OFFSET).
	RgfxRegion(volatile uint8_t *base, uint64_t size);

	/// The region's CPU base and byte size.
	volatile uint8_t *base() const { return m_base; }
	uint64_t size() const { return m_size; }

	/// Where guest allocations begin (RGFX_ALLOCATIONS_OFFSET); nothing is placed
	/// below it.
	uint64_t floor() const { return m_floor; }

	/// Bytes handed out past the floor (diagnostics / the next free offset − floor).
	uint64_t used() const;

	/// A block of `size` bytes aligned up to `align` (a power of two, at least 1), from the
	/// block source where one is set, else bump-allocated; never overlaps another live block.
	/// `ok` is false when `align` is not a power of two or the region has no room.
	Block allocate(uint64_t size, uint64_t align = kDefaultBlockAlign);

	/// Hands back the block at region offset `offset` (one allocate() returned) to the block
	/// source; nothing without one.
	void release(uint64_t offset);

	/// Where blocks come from: `alloc` places `size` bytes aligned to `align` and writes the
	/// region offset (false where it has no room), `release` frees one. Both are called with
	/// `context`, from any thread.
	typedef bool (*AllocFn)(void *context, uint64_t size, uint64_t align, uint64_t *offset);
	typedef void (*ReleaseFn)(void *context, uint64_t offset);
	void setBlockSource(AllocFn alloc, ReleaseFn release, void *context);

	/// A submission + completion ring pair (see RingPair). Each data size must be a
	/// power of two; `complete_data_size` may be 0 (no completion ring placed).
	RingPair allocateRingPair(uint32_t submit_data_size, uint32_t complete_data_size);

	/// Whether `[p, p + len)` lies wholly inside the region at or above the floor:
	/// what a no-copy buffer needs to be wrappable in place (createBufferOverHostMemory).
	bool contains(const void *p, uint64_t len) const;

	/// The offset of `p` in the region; valid only when contains(p, …) is true, 0
	/// otherwise.
	uint64_t offsetOf(const void *p) const;

	/// Whether `v` is a power of two (and non-zero).
	static bool isPowerOfTwo(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

	/// The default block alignment: a page, so a block can back a mapping.
	static const uint64_t kDefaultBlockAlign = 0x1000;
	/// A ring's alignment: a cache line, so the ring header's 64-byte lines (producer,
	/// consumer, consumer_idle, armed) never share one across rings.
	static const uint64_t kRingAlign = 64;

private:
	/// Rounds `value` up to a multiple of `align` (a power of two).
	static uint64_t alignUp(uint64_t value, uint64_t align);

	volatile uint8_t *m_base;
	uint64_t m_size;
	uint64_t m_floor;   // RGFX_ALLOCATIONS_OFFSET
	uint64_t m_cursor;  // the next free offset, at or above the floor
	AllocFn m_alloc = nullptr;     // the block source, where set (the kext)
	ReleaseFn m_release = nullptr;
	void *m_sourceContext = nullptr;

	std::mutex m_lock;
};

#endif // RGFX_REGION_HPP_
