/*
 * RgfxRegion: the guest bundle's allocator over the reims-graphics device's BAR 2
 * region (the committed region layout in reims_graphics_wire.h), as the process sees it in
 * its window: region offset o at the window's base plus o. The kext maps in only what is the
 * process's (its blocks, the shared blocks of the surface planes it holds, the scanout it is
 * granted), write-back, each before the process learns of it; the rest of the window, the
 * reserved head with it, reads zeros. This hands out aligned blocks past the head:
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
 * WRITE-COMBINED PARTS. The region is no RAM of the guest's, so the kernel maps it uncached
 * whatever the kext asks for write-back (xnu's pmap sets PCD on every page it does not manage),
 * and an AMD host applies the guest's memory type: plain stores there cost about 0.3 us per
 * 4 bytes. Where the kext maps part of a block write-combined (setWriteCombinedSource,
 * kRgfxKextFeatureWriteCombined), the bytes the process only writes take that part: a submission
 * ring's data (allocateRingPair, its header alone in the page before it), command buffers' bodies
 * and a library's staging chunks (allocate's `writeCombinedFrom`). Write-combined stores are
 * weakly ordered: the ring writer fences them (sfence) before it publishes a message that names
 * them, and a lock or a fence orders them for another thread, as for any store. The process never
 * reads them back, nor takes an atomic there: loads there are uncached, and a locked operation on
 * write-combined memory is not atomic.
 *
 * Our own code; depends only on the committed wire header. In a unit test the region
 * is a plain buffer; in the bundle it is the context's window.
 */

#ifndef RGFX_REGION_HPP_
#define RGFX_REGION_HPP_

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

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
		/// The blocks the rings lie in, which release() takes: a ring's own offset, but for a
		/// submission ring whose data is write-combined, which starts its block's second page.
		uint64_t submit_block;
		uint64_t complete_block;
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

	/// The bytes no block can take yet, where the region places blocks itself (no block source):
	/// true, with `*bytes` the room past the last block. False where a block source places them,
	/// which keeps its own count (the kext). A diagnostic, read without the lock, as used().
	bool freeBytes(uint64_t *bytes) const;

	/// A block of `size` bytes aligned up to `align` (a power of two, at least 1), from the
	/// block source where one is set, else bump-allocated; never overlaps another live block.
	/// `ok` is false when `align` is not a power of two or the region has no room.
	Block allocate(uint64_t size, uint64_t align = kDefaultBlockAlign);

	/// As allocate, with the block's bytes from `writeCombinedFrom` (a multiple of
	/// kDefaultBlockAlign) mapped write-combined where a write-combined source is set
	/// (WRITE-COMBINED PARTS); the whole block write-back otherwise.
	Block allocate(uint64_t size, uint64_t align, uint64_t writeCombinedFrom);

	/// Hands back the block at region offset `offset` (one allocate() or acquirePages() returned)
	/// to the block source; nothing without one. A pages block given more than once
	/// (acquirePages' `join`) goes back at the release for the last time it was given.
	void release(uint64_t offset);

	/// Where blocks come from: `alloc` places `size` bytes aligned to `align` and writes the
	/// region offset (false where it has no room), `release` frees one. Both are called with
	/// `context`, from any thread.
	typedef bool (*AllocFn)(void *context, uint64_t size, uint64_t align, uint64_t *offset);
	typedef void (*ReleaseFn)(void *context, uint64_t offset);
	void setBlockSource(AllocFn alloc, ReleaseFn release, void *context);

	/// Where blocks part of which is write-combined come from (WRITE-COMBINED PARTS): as the
	/// block source's `alloc`, with the bytes from `writeCombinedFrom` mapped write-combined.
	/// Called with the block source's context; its blocks go back through its `release`.
	typedef bool (*WriteCombinedAllocFn)(void *context, uint64_t size, uint64_t align,
	                                     uint64_t writeCombinedFrom, uint64_t *offset);
	void setWriteCombinedSource(WriteCombinedAllocFn alloc);

	/// Whether a write-combined source is set.
	bool writeCombines() const { return m_writeCombinedAlloc != nullptr; }

	/// What the process keeps for reuse and can give back where the kext has no room: `reclaim`
	/// lets go of such blocks (the buffers' empty and cached slabs, RgfxSlabs.hpp, and the
	/// IOSurface views the keep holds, RgfxIOSurface.h) and answers whether it let any go, and
	/// allocate() then asks the block source once more, so a block is refused only where the
	/// process holds nothing it could give back; so does acquireSurfacePages where the kext had
	/// no room (kPagesNoRoom). The surface sources' other refusals are not asked again: theirs
	/// also answer how the kext shares a surface, and their callers fall back to allocate().
	/// Called with `context` from the thread that asks, holding no lock of the region's.
	typedef bool (*ReclaimFn)(void *context);
	void setReclaim(ReclaimFn reclaim, void *context);

	/// The shared block of an IOSurface plane (acquireSurface): the block, its serial (unique to
	/// the block's allocation) and whether this ask made it, so its bytes are not yet the
	/// plane's. A block in place may be `cached` (acquireSurfacePages).
	struct SurfaceBlock {
		Block block;
		uint64_t serial;
		bool fresh;
		bool cached;
	};

	/// Where surface planes' shared blocks come from: `acquire` answers the block of `size`
	/// bytes of the plane of `length` bytes this process maps at `plane` (false where it gives
	/// none), `release` lets go of one. Both are called with `context`, from any thread.
	typedef bool (*SurfaceAcquireFn)(void *context, const void *plane, uint64_t length,
	                                 uint64_t size, uint64_t *offset, uint64_t *serial,
	                                 bool *fresh);
	void setSurfaceSource(SurfaceAcquireFn acquire, ReleaseFn release, void *context);

	/// The shared block of `size` bytes for the plane of `length` bytes at `plane`, the one every
	/// process that holds its surface gets (the kext's, RgfxSurfaceBlocks.hpp). `block.ok` is
	/// false without a surface source, or where it gives none.
	SurfaceBlock acquireSurface(const void *plane, uint64_t length, uint64_t size);

	/// Lets go of one acquireSurface or acquireSurfacePages of the block at `offset`.
	void releaseSurface(uint64_t offset);

	/// How an ask for a surface's shared block in place ended (acquireSurfacePages).
	enum PagesAnswer {
		kPagesGiven = 0,
		// The device does not use surfaces' pages in place, or refused this surface's.
		kPagesUnsupported,
		// No block: another holder of the surface copies it.
		kPagesRefused,
		// No block: the kext had no room for it, in the region or its table, even once the
		// process gave back what it keeps (setReclaim).
		kPagesNoRoom,
	};

	/// Where surfaces' shared blocks in place come from (RgfxIOSurface.h): `acquire` answers the
	/// block whose part the processes map is `size` bytes, for the surface of IOSurfaceID
	/// `surfaceId` whose whole allocation, `length` bytes, this process maps at `base`, the
	/// device's GPU using those pages in place. Where `cacheable`, the block may be made cached,
	/// its GPU copy kept in a device cache on the host; `*cached` says whether the block is,
	/// whoever made it. `layout`, where not null, is the surface's planes as its memory holds
	/// them, by which a host whose GPU shares the CPU's memory decides whether the block needs a
	/// device cache (RGFX_FEATURE_SURFACE_CACHE_REFUSED). Its blocks go back through the surface
	/// source's release (releaseSurface). `alignment` is what the host's textures over the pages
	/// ask of each plane's offset and rows (0: none known). Called with `context`, from any
	/// thread.
	typedef PagesAnswer (*SurfacePagesFn)(void *context, const void *base, uint64_t length,
	                                      uint64_t size, uint32_t surfaceId, bool cacheable,
	                                      const RgfxSurfacePagesLayout *layout, uint64_t *offset,
	                                      uint64_t *serial, bool *fresh, bool *cached);
	void setSurfacePagesSource(SurfacePagesFn acquire, uint64_t alignment, void *context);

	/// Where the kext learns of command buffers' writes of surfaces' shared blocks, so a CPU lock
	/// of a surface waits for them as Metal's does (the kext's RgfxSurfaceLocks.hpp): `note` names
	/// queue `queue`'s buffer at `seqno` and the `count` blocks it writes, two words each at
	/// `writes` (the block's region offset and its surface's IOSurfaceID); `retire` says the
	/// queue's buffers up to `seqno` have their results in the surfaces' pages. Both are called
	/// with `context`, from any thread.
	typedef void (*SurfaceWritesFn)(void *context, uint64_t queue, uint64_t seqno,
	                                const uint64_t *writes, uint32_t count);
	typedef void (*SurfaceRetireFn)(void *context, uint64_t queue, uint64_t seqno);
	void setSurfaceWriteSource(SurfaceWritesFn note, SurfaceRetireFn retire, void *context);

	/// Whether a surface write source is set.
	bool hasSurfaceWrites() const { return m_writesNote != nullptr; }

	/// Names queue `queue`'s buffer at `seqno` and its `count` writes (two words each) to the
	/// kext, where a surface write source is set.
	void noteSurfaceWrites(uint64_t queue, uint64_t seqno, const uint64_t *writes, uint32_t count);

	/// Queue `queue`'s buffers up to `seqno` have their results in the surfaces' pages.
	void retireSurfaceWrites(uint64_t queue, uint64_t seqno);

	/// Whether a surface pages source is set, and the alignment it gave.
	bool hasSurfacePages() const { return m_pagesAcquire != nullptr; }
	uint64_t surfacePagesAlignment() const { return m_pagesAlignment; }

	/// The shared block in place of `size` bytes for the surface `surfaceId` of `length` bytes
	/// at `base`, laid out as `layout` says, the one every process that holds the surface gets,
	/// asked cached where `cacheable` (SurfacePagesFn); `*answer` says how the ask ended, and
	/// `block.ok` is true only where it is kPagesGiven.
	SurfaceBlock acquireSurfacePages(const void *base, uint64_t length, uint64_t size,
	                                 uint32_t surfaceId, bool cacheable,
	                                 const RgfxSurfacePagesLayout *layout, PagesAnswer *answer);

	/// Where pages blocks come from (the kext's PagesBlock, RGFX_FEATURE_PAGES_BUFFERS): `acquire`
	/// answers the block of the runs of the pages that hold the `length` bytes at `address` in the
	/// process, which the kext wires for the block's life and the device maps in place from the
	/// first of them, where the memory is `writable`: its region offset and size, or false where
	/// it gives none. Its blocks go back through the block source's release (release()). Called
	/// with `context`, from any thread.
	typedef bool (*PagesBlockFn)(void *context, const void *address, uint64_t length,
	                             bool writable, uint64_t *offset, uint64_t *size);
	void setPagesBlockSource(PagesBlockFn acquire, void *context);

	/// Whether a pages block source is set.
	bool hasPagesBlocks() const { return m_pagesBlock != nullptr; }

#if defined(__arm64__)
	/// The page the kext rounds a pages block's range out to, which the device's view of the block
	/// starts at: 16 KiB where the kext took RGFX_FEATURE_PAGES_16K (its RgfxPagesBlock.hpp),
	/// else 4 KiB. Set with the pages block source, before any block is asked.
	void setPagesPage(uint64_t page) { m_pagesPage = page; }
	uint64_t pagesPage() const { return m_pagesPage; }
#endif

	/// The pages block of the `length` bytes at `address` (PagesBlockFn): `ok` false without a
	/// pages block source, or where it gives none. Its `ptr` is null: the process never maps a
	/// pages block, which holds the device's runs alone.
	///
	/// With a `join` other than kPagesOwn, a block given for the same bytes, writability and
	/// `join` that is still joinable is given again rather than a new one, so the device makes one
	/// view of the pages, and one copy of them where it copies them: two no-copy buffers over the
	/// same memory lie in one host memory, as on a Mac they lie in the same pages. A block stays
	/// joinable until leavePages() was called for each time it was given: the caller calls it as
	/// the object over the block goes for the app, while the app still keeps the memory, so a
	/// block is never joined over memory the app freed and took again at the same address. It goes
	/// back through the block source at the release() for the last time it was given.
	Block acquirePages(const void *address, uint64_t length, bool writable,
	                   uint32_t join = kPagesOwn);

	/// acquirePages' `join` for a block of its own, given to no one else.
	static const uint32_t kPagesOwn = 0;

	/// The pages block at `offset` is no longer joinable for one of the times acquirePages gave
	/// it (its object went for the app); nothing for a block given with kPagesOwn.
	void leavePages(uint64_t offset);

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
	/// Whether the reclaim, where set, let a block go (setReclaim).
	bool reclaimed();

	volatile uint8_t *m_base;
	uint64_t m_size;
	uint64_t m_floor;   // RGFX_ALLOCATIONS_OFFSET
	uint64_t m_cursor;  // the next free offset, at or above the floor
	AllocFn m_alloc = nullptr;     // the block source, where set (the kext)
	WriteCombinedAllocFn m_writeCombinedAlloc = nullptr; // the write-combined source (the kext)
	ReleaseFn m_release = nullptr;
	void *m_sourceContext = nullptr;
	ReclaimFn m_reclaim = nullptr; // the process's blocks kept for reuse, where set (the slabs)
	void *m_reclaimContext = nullptr;
	SurfaceAcquireFn m_surfaceAcquire = nullptr; // the surface source, where set (the kext)
	ReleaseFn m_surfaceRelease = nullptr;
	void *m_surfaceContext = nullptr;
	SurfacePagesFn m_pagesAcquire = nullptr; // the surface pages source, where set (the kext)
	uint64_t m_pagesAlignment = 0;
	void *m_pagesContext = nullptr;
	PagesBlockFn m_pagesBlock = nullptr; // the pages block source, where set (the kext)
	void *m_pagesBlockContext = nullptr;
#if defined(__arm64__)
	uint64_t m_pagesPage = 0x1000; // pagesPage()
#endif
	// The pages blocks given with a join (acquirePages): the joinable ones by their bytes,
	// writability and join, and each such block's counts by its offset, under m_pagesLock.
	struct PagesRange {
		uintptr_t address;
		uint64_t length;
		bool writable;
		uint32_t join;
		bool operator<(const PagesRange &o) const
		{
			return std::tie(address, length, writable, join) <
			       std::tie(o.address, o.length, o.writable, o.join);
		}
	};
	struct JoinedPages {
		PagesRange range;
		uint64_t size;
		uint32_t joinable; // the times given whose leavePages has not come
		uint32_t held;     // the times given whose release has not come
	};
	std::map<PagesRange, uint64_t> m_joinablePages;
	std::map<uint64_t, JoinedPages> m_joinedPages;
	std::mutex m_pagesLock;
	SurfaceWritesFn m_writesNote = nullptr; // the surface write source, where set (the kext)
	SurfaceRetireFn m_writesRetire = nullptr;
	void *m_writesContext = nullptr;

	std::mutex m_lock;
};

#endif // RGFX_REGION_HPP_
