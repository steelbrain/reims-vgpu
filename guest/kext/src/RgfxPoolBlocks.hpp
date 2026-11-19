// RgfxPoolBlocks: the blocks of the device's Shared pool a process holds (RGFX_FEATURE_POOL,
// docs/graphics-kext/design.md, "The Shared pool"), and the check of the host's answer to a take.
//
// The pool is a range of guest physical addresses past the guest's RAM that the host fills with
// its own memory. A process asks for a block of it (kRgfxAccelMethodPoolTake): the kext posts its
// context's POOL_TAKE, the host carves the block, zeroes it and answers with its offset in the
// pool and its length, and the kext maps those bytes into the process, read and write and
// write-back, and lists the mapping in the client's list here. The process names the block by
// its offset (HEAP_OVER_POOL), and gives it back through kRgfxAccelMethodPoolGive: the kext takes
// the mapping from the list and out of the process, then posts the POOL_GIVE, so the host never
// takes back a block a process still maps. At the client's close the kext takes every mapping out
// of the process before it posts the CONTEXT_CLOSE, which gives every block back.
//
// The list grows as the process takes blocks: the caller makes it a larger array where it is full
// (rgfxPoolBlocksGrow), up to kRgfxPoolBlocksMax blocks.
//
// A take waits for the host's answer with no deadline, THREAD_UNINT like each of the kext's other
// asks of the host (theirs end at 5 to 60 s): the host answers each take of an open context, after
// it adds memory to the pool where none is free and zeroes the block, and the context's close
// waits for the takes in flight, so no answer comes for a take the kext gave up on. Where the host
// stops answering (the library stalled), each taking thread waits until it answers, holding one
// of the kext's 32 waiters for answers (ReimsGraphics::kLifecycleWaiters), and a kill of its
// process ends it only then; the process's other threads, and other processes, go on, and their
// asks end at their deadlines. A paused VM runs no guest thread, and the answer is there
// when it resumes. The device's reset ends a take unanswered, but the kext resets the device only
// at its start, before a process can open it, and at its stop, once every process closed it; a
// VM's reset restarts the guest.
//
// The caller holds the client's list lock. Header-only and free of the kernel's headers, so its
// native test (tests/pool_blocks_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_POOL_BLOCKS_HPP_
#define RGFX_POOL_BLOCKS_HPP_

#include <stddef.h>
#include <stdint.h>

#include "RgfxUserClientABI.hpp"
#include "reims_graphics_wire.h"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// The kext's pool feature bit: the x86-64 kext's alone. The device offers an arm64 guest no pool
// until a boot shows its kernel maps a range of the pool's kind (past the guest's RAM, above
// 2^36) write-back, and the arm64e kext neither takes the device's POOL nor sets this bit:
// kRgfxAccelMethodPoolTake and kRgfxAccelMethodPoolGive answer kIOReturnUnsupported there, and
// its processes keep the region's blocks and the kext's chunked pages, as before the pool.
constexpr uint64_t kRgfxKextFeaturePoolCarried = RGFX_GUEST_ARM64 ? 0 : kRgfxKextFeaturePool;

// The kext's bit for the GPU addresses the host gives the pool's blocks (PoolTake's fourth
// output), carried with the pool's: the x86-64 kext's alone.
constexpr uint64_t kRgfxKextFeaturePoolAddressesCarried =
    RGFX_GUEST_ARM64 ? 0 : kRgfxKextFeaturePoolAddresses;

// Whether a POOL_BLOCK of `answered` bytes holds a whole answer: every field up to its length,
// an answer of a host before RGFX_FEATURE_POOL_ADDRESSES' included, whose GPU address then reads
// 0 (the answer's buffer zeroed); 0 where the feature word lacks the feature, as the host answers.
inline bool rgfxPoolAnswerWhole(uint32_t answered)
{
	return answered >= offsetof(RgfxPoolBlock, gpu_address);
}

// A block a process holds: its offset in the pool, its length, and the kext's mapping of it in
// the process (an IOMemoryMap), which holds it.
struct RgfxHeldPoolBlock {
	uint64_t offset;
	uint64_t length;
	void *map;
};

// A client's blocks, `count` of the `capacity` entries at `blocks`, in no order.
struct RgfxPoolBlockList {
	RgfxHeldPoolBlock *blocks;
	uint32_t count;
	uint32_t capacity;
};

// The list's first array, and the most blocks one client holds: 16,384, a GiB at least.
constexpr uint32_t kRgfxPoolBlocksFirst = 64;
constexpr uint32_t kRgfxPoolBlocksMax = 16384;

// The length the host answers a take of `asked` bytes with: `asked` rounded up to
// RGFX_POOL_BLOCK_ALIGN; 0 for 0 bytes or past RGFX_POOL_TAKE_MAX, which the host refuses.
inline uint64_t rgfxPoolTakeLength(uint64_t asked)
{
	if (asked == 0 || asked > RGFX_POOL_TAKE_MAX)
		return 0;
	return (asked + RGFX_POOL_BLOCK_ALIGN - 1) & ~uint64_t(RGFX_POOL_BLOCK_ALIGN - 1);
}

// Whether the host's OK answer to a take of `asked` bytes, a block of `length` bytes at `offset`,
// is one the kext maps: the take's rounded length, at a multiple of RGFX_POOL_BLOCK_ALIGN, and
// within the pool's `poolLength` bytes.
inline bool rgfxPoolAnswerFits(uint64_t poolLength, uint64_t asked, uint64_t offset,
                               uint64_t length)
{
	const uint64_t want = rgfxPoolTakeLength(asked);
	return want != 0 && length == want && offset % RGFX_POOL_BLOCK_ALIGN == 0 &&
	       offset <= poolLength && length <= poolLength - offset;
}

// A Private block (RGFX_FEATURE_PRIVATE_BLOCKS, design.md, "Private blocks") is a range of GPU
// addresses the host gives a context for its Private objects, with no memory a process maps: the
// kext posts the context's PRIVATE_TAKE and answers the block's address and length, and lists
// nothing, as the host knows each context's blocks and gives them back at its close.
//
// The length the host answers a Private take of `asked` bytes with at least: `asked` rounded up to
// RGFX_PRIVATE_BLOCK_ALIGN; 0 for 0 bytes or past RGFX_PRIVATE_TAKE_MAX, which the host refuses.
inline uint64_t rgfxPrivateTakeLength(uint64_t asked)
{
	if (asked == 0 || asked > RGFX_PRIVATE_TAKE_MAX)
		return 0;
	return (asked + RGFX_PRIVATE_BLOCK_ALIGN - 1) & ~uint64_t(RGFX_PRIVATE_BLOCK_ALIGN - 1);
}

// Whether the host's OK answer to a Private take of `asked` bytes, a block of `length` bytes at
// GPU address `gpuAddress`, is one the kext hands out: at least the take's rounded length, a
// multiple of RGFX_PRIVATE_BLOCK_ALIGN, at an address other than 0 whose block does not wrap.
inline bool rgfxPrivateAnswerFits(uint64_t asked, uint64_t gpuAddress, uint64_t length)
{
	const uint64_t want = rgfxPrivateTakeLength(asked);
	return want != 0 && length >= want && length % RGFX_PRIVATE_BLOCK_ALIGN == 0 &&
	       gpuAddress != 0 && gpuAddress <= UINT64_MAX - length;
}

// The capacity the list grows to where it is full: kRgfxPoolBlocksFirst at first, then twice as
// many, up to kRgfxPoolBlocksMax; 0 where it holds that many already.
inline uint32_t rgfxPoolBlocksGrowTo(const RgfxPoolBlockList &list)
{
	if (list.count < list.capacity)
		return list.capacity;
	if (list.capacity >= kRgfxPoolBlocksMax)
		return 0;
	const uint32_t next = list.capacity ? list.capacity * 2 : kRgfxPoolBlocksFirst;
	return next < kRgfxPoolBlocksMax ? next : kRgfxPoolBlocksMax;
}

// Moves the list's blocks into `blocks`, an array of `capacity` entries (rgfxPoolBlocksGrowTo's),
// and returns the array it had, for the caller to free (null where it had none).
inline RgfxHeldPoolBlock *rgfxPoolBlocksGrow(RgfxPoolBlockList &list, RgfxHeldPoolBlock *blocks,
                                             uint32_t capacity)
{
	for (uint32_t i = 0; i < list.count; i++)
		blocks[i] = list.blocks[i];
	RgfxHeldPoolBlock *old = list.blocks;
	list.blocks = blocks;
	list.capacity = capacity;
	return old;
}

// Lists a block the process now maps: false where the list is full, or the block overlaps one it
// lists (a host's answer of a block the client holds already), and nothing is listed.
inline bool rgfxPoolBlocksAdd(RgfxPoolBlockList &list, uint64_t offset, uint64_t length, void *map)
{
	if (list.count >= list.capacity || length == 0)
		return false;
	for (uint32_t i = 0; i < list.count; i++) {
		const RgfxHeldPoolBlock &b = list.blocks[i];
		if (offset < b.offset + b.length && b.offset < offset + length)
			return false;
	}
	list.blocks[list.count++] = {offset, length, map};
	return true;
}

// Takes the block at pool offset `offset` off the list into `*block`: false where none is there.
inline bool rgfxPoolBlocksTake(RgfxPoolBlockList &list, uint64_t offset, RgfxHeldPoolBlock *block)
{
	for (uint32_t i = 0; i < list.count; i++) {
		if (list.blocks[i].offset != offset)
			continue;
		*block = list.blocks[i];
		list.blocks[i] = list.blocks[--list.count];
		return true;
	}
	return false;
}

// Takes any block off the list into `*block`, the client's close's: false where none is left.
inline bool rgfxPoolBlocksTakeAny(RgfxPoolBlockList &list, RgfxHeldPoolBlock *block)
{
	if (!list.count)
		return false;
	*block = list.blocks[--list.count];
	return true;
}

#endif // RGFX_POOL_BLOCKS_HPP_
