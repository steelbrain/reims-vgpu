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
};

// A pin of one block (rgfxBlockPin), to give back with rgfxBlockUnpin.
struct RgfxBlockPin {
	uint64_t offset;
	uint64_t serial;
};

// The table: `count` blocks of `capacity`, sorted by offset, between `start` and `end`.
struct RgfxBlockTable {
	RgfxBlock *blocks;
	uint32_t count;
	uint32_t capacity;
	uint64_t start;
	uint64_t end;
	uint64_t nextSerial;
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
	__builtin_memmove(&t.blocks[at], &t.blocks[at + 1], sizeof(RgfxBlock) * (t.count - at - 1));
	t.count--;
}

// A block of `size` bytes (rounded up to pages), aligned to `align` (a power of two, at least a
// page), for `owner` of context `context`: the first gap, in offset order, that holds it. Where
// `pin` is not null the block starts pinned, so it is not freed until the caller unpins it (once
// the device has its grant).
inline RgfxBlockResult rgfxBlockAlloc(RgfxBlockTable &t, const void *owner, uint32_t context,
                                      uint64_t size, uint64_t align, uint64_t *offset,
                                      RgfxBlockPin *pin = nullptr)
{
	if (size == 0 || align == 0 || (align & (align - 1)) != 0)
		return kRgfxBlockBadArgument;
	if (align < kRgfxBlockPage)
		align = kRgfxBlockPage;
	if (size > t.end - t.start)
		return kRgfxBlockNoSpace;
	size = rgfxBlockAlignUp(size, kRgfxBlockPage);
	if (t.count == t.capacity)
		return kRgfxBlockTableFull;
	uint64_t cursor = t.start;
	for (uint32_t at = 0; at <= t.count; at++) {
		uint64_t start = rgfxBlockAlignUp(cursor, align);
		uint64_t gapEnd = at < t.count ? t.blocks[at].offset : t.end;
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
			t.count++;
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

// `owner` frees its live block at `offset`: it is freed, for its release to be posted. Busy while
// a create naming it is in flight.
inline RgfxBlockResult rgfxBlockFree(RgfxBlockTable &t, const void *owner, uint64_t offset)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || t.blocks[at].owner != owner || t.blocks[at].state != kRgfxBlockLive)
		return kRgfxBlockBadArgument;
	if (t.blocks[at].pins)
		return kRgfxBlockBusy;
	t.blocks[at].state = kRgfxBlockFreed;
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

// The device answered the release of the block at `offset` of `serial`: OK gives it out again;
// anything else keeps it. Nothing where that block has gone since (its owner's close).
inline void rgfxBlockReleased(RgfxBlockTable &t, uint64_t offset, uint64_t serial, bool ok)
{
	int64_t at = rgfxBlockIndex(t, offset);
	if (at < 0 || t.blocks[at].serial != serial || t.blocks[at].state != kRgfxBlockReleasing)
		return;
	if (ok)
		rgfxBlockRemove(t, static_cast<uint32_t>(at));
	else
		t.blocks[at].state = kRgfxBlockKept;
}

// `owner`'s close was answered: the device holds nothing of its blocks, and they all go. How
// many.
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
		if (b.state != kRgfxBlockReleasing)
			b.state = kRgfxBlockKept;
		n++;
	}
	return n;
}

#endif // RGFX_BLOCK_TABLE_HPP_
