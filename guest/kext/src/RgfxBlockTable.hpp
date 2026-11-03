// RgfxBlockTable: the accelerator kext's blocks of the device's region, past
// RGFX_CONTEXT_ALLOCATIONS_OFFSET (a process's rings, Shared buffers and heaps, surfaces), each
// owned by the client that asked for it. A first-fit allocator over a table sorted by offset.
//
// A block is given out again only once the device has said nothing of it is in use: a free marks
// it freed, the kext posts its BLOCK_RELEASE, and the block leaves the table when the device
// answers OK. Until then it is no one's to name and no one's to take. A block a create in flight
// names is pinned, and is not freed while it is. The caller holds the accelerator's lock around
// every call. Header-only and free of the kernel's headers, so its native test
// (tests/blocks_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_BLOCK_TABLE_HPP_
#define RGFX_BLOCK_TABLE_HPP_

#include <stdint.h>

// Where a block is in its life.
enum RgfxBlockState : uint8_t {
	// Its owner's, to name in its messages.
	kRgfxBlockLive = 0,
	// Freed by its owner; its BLOCK_RELEASE is still to be posted.
	kRgfxBlockFreed,
	// Its BLOCK_RELEASE is posted, and not yet answered.
	kRgfxBlockReleasing,
	// The device refused its release, or did not answer it, or its owner's close was not
	// answered: kept out of use until its owner's close is answered, or for good.
	kRgfxBlockKept,
};

// A block of the region.
struct RgfxBlock {
	uint64_t offset;
	uint64_t size;
	const void *owner;
	// Unique to this block's allocation, so an answer or an unpin for an earlier block at the
	// same offset finds nothing.
	uint64_t serial;
	// The owner's context, for the block's BLOCK_RELEASE.
	uint32_t context;
	// Creates in flight that name the block.
	uint32_t pins;
	RgfxBlockState state;
	// The device refused this block's release (an object of some context lies in it). It is
	// kept, but posted again (rgfxBlockRetryRefused) after a context close, which may have
	// dropped that object. A block kept for another reason (a close not answered) is not
	// `refused`, and is never posted again.
	bool refused;
	// For an in-place surface block or a pages block: the wired pages (an opaque kernel
	// descriptor) the host's GPU maps in place, kept while the block is in the table so the guest
	// cannot recycle the pages under the host's view. The caller unwires it once the block leaves
	// the table (its release answered OK, rgfxBlockReleased; a pages block's owner's close
	// answered, rgfxBlockTakeWired). Null for any other block.
	void *wired;
	// Made for a client that may take the reserve (rgfxBlockAlloc's `mayReserve`): counted in
	// the table's reservedCount and reservedBytes while it is in the table.
	bool reserved;
	// When it was freed for its release (the caller's clock, 0 where not given), for the time a
	// release takes.
	uint64_t freedAt;
};

// A pin of one block (rgfxBlockPin), to give back with rgfxBlockUnpin.
struct RgfxBlockPin {
	uint64_t offset;
	uint64_t serial;
};

// The table: `count` blocks of `capacity`, sorted by offset, between `start` and `end`.
//
// The reserve: the last `reserveBytes` of the range and the last `reserveEntries` of the
// capacity are given only to an allocation that may take them (`mayReserve`, WindowServer's), so
// no other process, however much it holds, leaves the compositor without room or an entry. Such
// an allocation takes the first gap from `start` like any other, and reaches into the reserve only
// where the rest has no room. The blocks made for it are counted while they are in the table
// (reservedCount, reservedBytes), with the most they came to (the peaks), for the reserve's size.
struct RgfxBlockTable {
	RgfxBlock *blocks;
	uint32_t count;
	uint32_t capacity;
	uint64_t start;
	uint64_t end;
	uint64_t nextSerial;
	uint64_t reserveBytes;
	uint32_t reserveEntries;
	uint32_t reservedCount;
	uint64_t reservedBytes;
	uint32_t reservedPeakCount;
	uint64_t reservedPeakBytes;
};

enum RgfxBlockResult {
	kRgfxBlockOk = 0,
	kRgfxBlockBadArgument,
	kRgfxBlockNoSpace,
	kRgfxBlockTableFull,
	kRgfxBlockBusy,
};

// Every block is page-aligned and a whole number of pages, so a block can back a mapping.
constexpr uint64_t kRgfxBlockPage = 0x1000;

// The owner of the blocks of a client whose close the device did not answer: no client, so they
// are never named or given out again.
inline const void *rgfxBlockNoOwner()
{
	return reinterpret_cast<const void *>(static_cast<uintptr_t>(1));
}

inline uint64_t rgfxBlockAlignUp(uint64_t value, uint64_t align)
{
	return (value + (align - 1)) & ~(align - 1);
}

// The index of the block that starts at `offset`, or -1.
inline int64_t rgfxBlockIndex(const RgfxBlockTable &t, uint64_t offset)
{
	uint32_t lo = 0, hi = t.count;
	while (lo < hi) {
		uint32_t mid = (lo + hi) / 2;
		if (t.blocks[mid].offset < offset)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo < t.count && t.blocks[lo].offset == offset ? static_cast<int64_t>(lo) : -1;
}

inline void rgfxBlockRemove(RgfxBlockTable &t, uint32_t at)
{
	if (t.blocks[at].reserved) {
		t.reservedCount--;
		t.reservedBytes -= t.blocks[at].size;
	}
	__builtin_memmove(&t.blocks[at], &t.blocks[at + 1], sizeof(RgfxBlock) * (t.count - at - 1));
	t.count--;
}

// A block of `size` bytes (rounded up to pages), aligned to `align` (a power of two, at least a
// page), for `owner` of context `context`: the first gap, in offset order, that holds it, inside
// the table's reserve too where `mayReserve` (RgfxBlockTable). Where `pin` is not null the block
// starts pinned, so it is not freed until the caller unpins it (once the device has its grant).
inline RgfxBlockResult rgfxBlockAlloc(RgfxBlockTable &t, const void *owner, uint32_t context,
                                      uint64_t size, uint64_t align, uint64_t *offset,
                                      RgfxBlockPin *pin = nullptr, bool mayReserve = false)
{
	if (size == 0 || align == 0 || (align & (align - 1)) != 0)
		return kRgfxBlockBadArgument;
	if (align < kRgfxBlockPage)
		align = kRgfxBlockPage;
	const uint64_t span = t.end - t.start;
	const uint64_t end =
	    mayReserve ? t.end : t.end - (t.reserveBytes < span ? t.reserveBytes : span);
	const uint32_t capacity =
	    mayReserve ? t.capacity
	               : t.capacity - (t.reserveEntries < t.capacity ? t.reserveEntries : t.capacity);
	if (size > end - t.start)
		return kRgfxBlockNoSpace;
	size = rgfxBlockAlignUp(size, kRgfxBlockPage);
	if (t.count >= capacity)
		return kRgfxBlockTableFull;
	uint64_t cursor = t.start;
	for (uint32_t at = 0; at <= t.count && cursor < end; at++) {
		uint64_t start = rgfxBlockAlignUp(cursor, align);
		uint64_t gapEnd = at < t.count && t.blocks[at].offset < end ? t.blocks[at].offset : end;
		if (start <= gapEnd && size <= gapEnd - start) {
			__builtin_memmove(&t.blocks[at + 1], &t.blocks[at],
			                  sizeof(RgfxBlock) * (t.count - at));
			RgfxBlock &b = t.blocks[at];
			b.offset = start;
			b.size = size;
			b.owner = owner;
			b.serial = ++t.nextSerial;
			b.context = context;
			b.pins = pin ? 1 : 0;
			b.state = kRgfxBlockLive;
			b.refused = false;
			b.wired = nullptr;
			b.reserved = mayReserve;
			b.freedAt = 0;
			t.count++;
			if (mayReserve) {
				t.reservedCount++;
				t.reservedBytes += size;
				if (t.reservedCount > t.reservedPeakCount)
					t.reservedPeakCount = t.reservedCount;
				if (t.reservedBytes > t.reservedPeakBytes)
					t.reservedPeakBytes = t.reservedBytes;
			}
			*offset = start;
			if (pin)
				*pin = {b.offset, b.serial};
			return kRgfxBlockOk;
		}
		if (at < t.count)
			cursor = t.blocks[at].offset + t.blocks[at].size;
	}
	return kRgfxBlockNoSpace;
}

// Pins the live block of `owner` that holds the `length` bytes at `offset`, for a create that
// names them: false where no such block does.
inline bool rgfxBlockPin(RgfxBlockTable &t, const void *owner, uint64_t offset, uint64_t length,
                         RgfxBlockPin *pin)
{
	for (uint32_t i = 0; i < t.count; i++) {
		RgfxBlock &b = t.blocks[i];
		if (b.offset > offset)
			return false;
		if (offset - b.offset <= b.size && length <= b.size - (offset - b.offset)) {
			if (b.owner != owner || b.state != kRgfxBlockLive)
				return false;
			b.pins++;
			*pin = {b.offset, b.serial};
			return true;
		}
	}
	return false;
}

// Gives back a pin; nothing where its block has gone since.
// Pins the live block that holds the `length` bytes at `offset`, whoever owns it: for a span the
// caller has found the client may name another way (a shared block it holds,
// RgfxSurfaceBlocks.hpp).
inline bool rgfxBlockPinAny(RgfxBlockTable &t, uint64_t offset, uint64_t length,
                            RgfxBlockPin *pin)
{
	for (uint32_t i = 0; i < t.count; i++) {
		RgfxBlock &b = t.blocks[i];
		if (b.offset > offset)
			return false;
		if (offset - b.offset <= b.size && length <= b.size - (offset - b.offset)) {
			if (b.state != kRgfxBlockLive)
				return false;
			b.pins++;
			*pin = {b.offset, b.serial};
			return true;
		}
	}
	return false;
}

// The context the BLOCK_RELEASE of the block at `offset` names: a shared block's last holder's.
inline void rgfxBlockSetContext(RgfxBlockTable &t, uint64_t offset, uint32_t context)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at >= 0)
		t.blocks[at].context = context;
}

// Keeps the block at `offset` from use for good: no client owns it, and the device may still use
// it (a shared block whose last holder's close was not answered).
inline void rgfxBlockKeepAt(RgfxBlockTable &t, uint64_t offset)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0)
		return;
	t.blocks[at].owner = rgfxBlockNoOwner();
	// Kept for good: never posted again (not `refused`).
	t.blocks[at].refused = false;
	if (t.blocks[at].state != kRgfxBlockReleasing)
		t.blocks[at].state = kRgfxBlockKept;
}

// Attaches the wired pages of an in-place surface block to the block at `offset`, so the host's
// in-place view cannot outlive their wiring: the caller unwires them only once the block leaves
// the table (rgfxBlockReleased OK). True where the block is there to hold it.
inline bool rgfxBlockSetWired(RgfxBlockTable &t, uint64_t offset, void *wired)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0)
		return false;
	t.blocks[at].wired = wired;
	return true;
}

// The block at `offset`, which stays in the table, holds `wire`, a wiring of pages the device may
// map in place there, where it holds none yet: it keeps it until it leaves the table
// (rgfxBlockReleased OK), and never where it is kept for good. True where it holds `wire` now;
// false where the caller lets `wire` go (the block's own wiring holds the same pages, or no block
// is there).
inline bool rgfxBlockHoldWire(RgfxBlockTable &t, uint64_t offset, void *wire)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || wire == nullptr || t.blocks[at].wired != nullptr)
		return false;
	t.blocks[at].wired = wire;
	return true;
}

inline void rgfxBlockUnpin(RgfxBlockTable &t, const RgfxBlockPin &pin)
{
	int64_t at = rgfxBlockIndex(t, pin.offset);
	if (at >= 0 && t.blocks[at].serial == pin.serial && t.blocks[at].pins)
		t.blocks[at].pins--;
}

// Takes back the block `pin` names, as if never given out (the device was not told of it).
inline void rgfxBlockDrop(RgfxBlockTable &t, const RgfxBlockPin &pin)
{
	int64_t at = rgfxBlockIndex(t, pin.offset);
	if (at >= 0 && t.blocks[at].serial == pin.serial)
		rgfxBlockRemove(t, static_cast<uint32_t>(at));
}

// Whether `owner` may free its block at `offset` now, as rgfxBlockFree would answer, with the
// block's bytes in `*size` where it may: the caller takes the block from the owner's sight first.
inline RgfxBlockResult rgfxBlockFreeable(const RgfxBlockTable &t, const void *owner,
                                         uint64_t offset, uint64_t *size)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || t.blocks[at].owner != owner || t.blocks[at].state != kRgfxBlockLive)
		return kRgfxBlockBadArgument;
	if (t.blocks[at].pins)
		return kRgfxBlockBusy;
	*size = t.blocks[at].size;
	return kRgfxBlockOk;
}

// `owner` frees its live block at `offset`, at `now`: it is freed, for its release to be posted.
// Busy while a create naming it is in flight.
inline RgfxBlockResult rgfxBlockFree(RgfxBlockTable &t, const void *owner, uint64_t offset,
                                     uint64_t now = 0)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || t.blocks[at].owner != owner || t.blocks[at].state != kRgfxBlockLive)
		return kRgfxBlockBadArgument;
	if (t.blocks[at].pins)
		return kRgfxBlockBusy;
	t.blocks[at].state = kRgfxBlockFreed;
	t.blocks[at].freedAt = now;
	return kRgfxBlockOk;
}

// The first freed block, now releasing: its release is the caller's to post. False where none is.
inline bool rgfxBlockNextRelease(RgfxBlockTable &t, RgfxBlock *block)
{
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.blocks[i].state == kRgfxBlockFreed) {
			t.blocks[i].state = kRgfxBlockReleasing;
			*block = t.blocks[i];
			return true;
		}
	}
	return false;
}

// Whether the block at `offset` of `serial` is in the table, its release posted and not yet
// answered: false where it has gone since (its owner's close) and the offset may be another's.
inline bool rgfxBlockReleasingAt(const RgfxBlockTable &t, uint64_t offset, uint64_t serial)
{
	int64_t at = rgfxBlockIndex(t, offset);
	return at >= 0 && t.blocks[at].serial == serial && t.blocks[at].state == kRgfxBlockReleasing;
}

// The device answered the release of the block at `offset` of `serial`: OK gives it out again,
// and hands back its wired pages in `*wired` (null where it had none) for the caller to unwire;
// anything else keeps it, to post again after a context close (rgfxBlockRetryRefused). Nothing
// where that block has gone since (its owner's close).
inline void rgfxBlockReleased(RgfxBlockTable &t, uint64_t offset, uint64_t serial, bool ok,
                              void **wired = nullptr)
{
	if (wired)
		*wired = nullptr;
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || t.blocks[at].serial != serial || t.blocks[at].state != kRgfxBlockReleasing)
		return;
	if (ok) {
		if (wired)
			*wired = t.blocks[at].wired;
		rgfxBlockRemove(t, static_cast<uint32_t>(at));
	} else {
		t.blocks[at].state = kRgfxBlockKept;
		t.blocks[at].refused = true;
	}
}

// Posts again the release of every block kept after the device refused it (an object of some
// context lay in it): each goes back to freed, for rgfxBlockNextRelease, since a context close
// may have dropped that object. A block kept for good (a close not answered) is not `refused`,
// and is left. How many are posted again, freed at `now`.
inline uint32_t rgfxBlockRetryRefused(RgfxBlockTable &t, uint64_t now = 0)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.blocks[i].state == kRgfxBlockKept && t.blocks[i].refused) {
			t.blocks[i].state = kRgfxBlockFreed;
			t.blocks[i].refused = false;
			t.blocks[i].freedAt = now;
			n++;
		}
	}
	return n;
}

// `owner`'s close was answered: one of its blocks' wired pages (its pages blocks', the only ones it
// owns that hold any), taken from the block for the caller to unwire before the blocks go
// (rgfxBlockFreeOwner); null where none is left.
inline void *rgfxBlockTakeWired(RgfxBlockTable &t, const void *owner)
{
	for (uint32_t i = 0; i < t.count; i++) {
		RgfxBlock &b = t.blocks[i];
		if (b.owner == owner && b.wired) {
			void *wired = b.wired;
			b.wired = nullptr;
			return wired;
		}
	}
	return nullptr;
}

// `owner`'s close was answered: the device holds nothing of its blocks, and they all go. How
// many. The caller takes their wired pages first (rgfxBlockTakeWired).
inline uint32_t rgfxBlockFreeOwner(RgfxBlockTable &t, const void *owner)
{
	uint32_t kept = 0, freed = 0;
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.blocks[i].owner == owner) {
			freed++;
			continue;
		}
		t.blocks[kept++] = t.blocks[i];
	}
	t.count = kept;
	return freed;
}

// `owner`'s close was not answered in time: its blocks, in whatever state, are `to`'s, a late
// close's token (RgfxCloseTable.hpp), until the answer lets them go as an answer in time would
// (rgfxBlockTakeWired, rgfxBlockFreeOwner). None is given out again meanwhile. None is released
// either, a freed one or a refused one: the device would take a release posted behind the close
// with the context gone, so its answer would not wait for the context's buffers. A releasing one,
// posted before the close, goes at its answer as before. How many.
inline uint32_t rgfxBlockHand(RgfxBlockTable &t, const void *from, const void *to)
{
	uint32_t n = 0;
	for (uint32_t i = 0; from != nullptr && i < t.count; i++) {
		RgfxBlock &b = t.blocks[i];
		if (b.owner != from)
			continue;
		b.owner = to;
		if (b.state == kRgfxBlockFreed)
			b.state = kRgfxBlockKept;
		b.refused = false;
		n++;
	}
	return n;
}

// `owner`'s close was not answered: the device may still use its blocks, so none is given out
// again, save a releasing one the device still answers OK. How many are kept.
inline uint32_t rgfxBlockKeepOwner(RgfxBlockTable &t, const void *owner)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < t.count; i++) {
		RgfxBlock &b = t.blocks[i];
		if (b.owner != owner)
			continue;
		b.owner = rgfxBlockNoOwner();
		// Kept for good: never posted again (not `refused`).
		b.refused = false;
		if (b.state != kRgfxBlockReleasing)
			b.state = kRgfxBlockKept;
		n++;
	}
	return n;
}

#endif // RGFX_BLOCK_TABLE_HPP_
