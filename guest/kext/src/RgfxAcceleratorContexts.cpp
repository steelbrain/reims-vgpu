/*
 * RgfxAccelerator's contexts: what each Metal process that opens our user client holds of the
 * device, and what the accelerator tells the device of it.
 *
 *   - A context number, handed out once from 1: the process's object ids are
 *     c << RGFX_CONTEXT_OBJ_ID_SHIFT and up, so no two processes' ids meet in the host's one
 *     handle table, and the host drops the process's at CONTEXT_CLOSE.
 *   - A slot of RGFX_CONTEXT_SLOTS, held until the close: the process's GPU addresses come from
 *     the slot's range. Slots are handed out round-robin, so a slot freed by one process is the
 *     last one the next process takes.
 *   - Chunks of the texture resource ids, on demand, from one pool for the guest: every
 *     process's textures share the host device's one space of ids, which ends below
 *     RGFX_RESOURCE_ID_LIMIT, so ids are not split by slot. Each chunk is posted to the device
 *     (RESOURCE_ID_CHUNK) before the process learns of it, and the client's are freed at its
 *     close, after its CONTEXT_CLOSE is answered.
 *   - Region blocks past RGFX_CONTEXT_ALLOCATIONS_OFFSET (rings, Shared buffers and heaps,
 *     surfaces), from a first-fit allocator over a table sorted by offset, each owned by the
 *     client that asked; the client's are freed at its close.
 *   - The object lifecycle, on the framebuffer's control ring: creates and destroys, a
 *     QUEUE_CREATE answered once the host has the queue's ring pair (so the process writes it at
 *     once), and a CONTEXT_CLOSE answered once the host has dropped the context's objects.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAccelerator.hpp"
#include "ReimsGraphics.hpp"

#include <IOKit/IOLib.h>
#include <libkern/libkern.h>

namespace {
// The most blocks handed out at once.
constexpr UInt32 kMaxBlocks = 16384;
// How long a QUEUE_CREATE and a CONTEXT_CLOSE wait for their answers: the host takes the
// control ring in order, behind whatever pipeline compiles were posted before them.
constexpr UInt32 kQueueAnswerMs = 30000;
constexpr UInt32 kCloseAnswerMs = 10000;
// Every block is page-aligned and a whole number of pages, so a block can back a mapping.
constexpr UInt64 kPage = 0x1000;

UInt64 alignUp(UInt64 value, UInt64 align)
{
	return (value + (align - 1)) & ~(align - 1);
}
} // namespace

// The device's contexts and blocks: one accelerator per device, one device per guest. Made at
// the first context's open (startContexts) and kept for the kernel's life.
namespace {
// One initialised object (it lands in __DATA,__data, a section the kext has already, so the
// image's layout before its code stays as the loader's sites were taken from).
struct ContextState {
	// Under contextLock: the slots held (a bit each), the next slot to try, the next context
	// number, and the blocks handed out, sorted by offset.
	IOLock *contextLock;
	UInt32 slotsHeld;
	UInt32 nextSlot;
	UInt32 nextContext;
	RgfxBlock *blocks;
	UInt32 blockCount;
	UInt64 blocksStart;
	UInt64 blocksEnd;
	// Under contextLock: the client holding each chunk of the texture resource ids, null where
	// none does.
	const void *idChunkOwners[RGFX_RESOURCE_ID_CHUNKS];
};
ContextState state = {nullptr, 0, 0, 1, nullptr, 0, 0, 0, {}};
IOLock *&contextLock = state.contextLock;
UInt32 &slotsHeld = state.slotsHeld;
UInt32 &nextSlot = state.nextSlot;
UInt32 &nextContext = state.nextContext;
RgfxBlock *&blocks = state.blocks;
UInt32 &blockCount = state.blockCount;
UInt64 &blocksStart = state.blocksStart;
UInt64 &blocksEnd = state.blocksEnd;
const void *(&idChunkOwners)[RGFX_RESOURCE_ID_CHUNKS] = state.idChunkOwners;
} // namespace

bool RgfxAccelerator::startContexts()
{
	if (__atomic_load_n(&contextLock, __ATOMIC_ACQUIRE))
		return true;
	if (regionSize <= RGFX_CONTEXT_ALLOCATIONS_OFFSET + RGFX_CURSOR_SPAN) {
		IOLog("RgfxAccelerator: no contexts (region 0x%llx bytes)\n", regionSize);
		return false;
	}
	// The first opener makes the state; one that loses the race frees what it made.
	IOLock *lock = IOLockAlloc();
	RgfxBlock *table = static_cast<RgfxBlock *>(IOMalloc(sizeof(RgfxBlock) * kMaxBlocks));
	RgfxBlock *expected = nullptr;
	if (lock && table && __atomic_compare_exchange_n(&blocks, &expected, table, false,
	                                                 __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
		bzero(table, sizeof(RgfxBlock) * kMaxBlocks);
		blocksStart = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
		blocksEnd = (regionSize - RGFX_CURSOR_SPAN) & ~(kPage - 1);
		__atomic_store_n(&contextLock, lock, __ATOMIC_RELEASE);
		IOLog("RgfxAccelerator: blocks 0x%llx-0x%llx, %u context slots\n", blocksStart, blocksEnd,
		      RGFX_CONTEXT_SLOTS);
		return true;
	}
	if (lock)
		IOLockFree(lock);
	if (table)
		IOFree(table, sizeof(RgfxBlock) * kMaxBlocks);
	// Another opener is making it: wait for it to publish.
	while (expected && !__atomic_load_n(&contextLock, __ATOMIC_ACQUIRE))
		IOSleep(1);
	return __atomic_load_n(&contextLock, __ATOMIC_ACQUIRE) != nullptr;
}

// MARK: - Contexts

IOReturn RgfxAccelerator::openContext(UInt32 *context, UInt32 *slot)
{
	if (!startContexts())
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	UInt32 found = RGFX_CONTEXT_SLOTS;
	for (UInt32 i = 0; i < RGFX_CONTEXT_SLOTS; i++) {
		UInt32 s = (nextSlot + i) % RGFX_CONTEXT_SLOTS;
		if (!(slotsHeld & (1u << s))) {
			found = s;
			break;
		}
	}
	if (found == RGFX_CONTEXT_SLOTS) {
		IOLockUnlock(contextLock);
		IOLog("RgfxAccelerator: every context slot is held\n");
		return kIOReturnNoResources;
	}
	slotsHeld |= 1u << found;
	nextSlot = (found + 1) % RGFX_CONTEXT_SLOTS;
	*slot = found;
	*context = nextContext++;
	IOLockUnlock(contextLock);
	IOLog("RgfxAccelerator: context %u opened in slot %u\n", *context, *slot);
	return kIOReturnSuccess;
}

void RgfxAccelerator::closeContext(UInt32 context, UInt32 slot)
{
	// The device ends the context's queues and drops its objects before it answers; the
	// caller has closed its queues already, and frees its blocks after this.
	RgfxContextClose close = {};
	close.context = context;
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 length = 0;
	IOReturn r = postLifecycle(RGFX_CMD_CONTEXT_CLOSE, &close, sizeof(close), RGFX_EVT_COMPLETE,
	                           reply, sizeof(reply), &length, nullptr, kCloseAnswerMs);
	if (r != kIOReturnSuccess)
		IOLog("RgfxAccelerator: context %u: the device did not answer its close (0x%x)\n",
		      context, r);
	if (contextLock && slot < RGFX_CONTEXT_SLOTS) {
		IOLockLock(contextLock);
		slotsHeld &= ~(1u << slot);
		IOLockUnlock(contextLock);
	}
	IOLog("RgfxAccelerator: context %u closed (slot %u)\n", context, slot);
}

// MARK: - Resource ids

IOReturn RgfxAccelerator::takeIdChunk(const void *owner, UInt32 context, UInt64 *firstId)
{
	if (!contextLock)
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	UInt32 chunk = 0;
	while (chunk < RGFX_RESOURCE_ID_CHUNKS && idChunkOwners[chunk])
		chunk++;
	if (chunk == RGFX_RESOURCE_ID_CHUNKS) {
		IOLockUnlock(contextLock);
		IOLog("RgfxAccelerator: context %u: every resource-id chunk is held\n", context);
		return kIOReturnNoResources;
	}
	idChunkOwners[chunk] = owner;
	IOLockUnlock(contextLock);
	// The device learns of the chunk before the process does, so the grant is on the control
	// ring before any create that names one of its ids.
	RgfxResourceIdChunk grant = {};
	grant.context = context;
	grant.chunk = chunk;
	IOReturn r = postLifecycle(RGFX_CMD_RESOURCE_ID_CHUNK, &grant, sizeof(grant), 0, nullptr, 0,
	                           nullptr, nullptr, 0);
	if (r != kIOReturnSuccess) {
		IOLockLock(contextLock);
		idChunkOwners[chunk] = nullptr;
		IOLockUnlock(contextLock);
		return r;
	}
	*firstId = 1 + static_cast<UInt64>(chunk) * RGFX_RESOURCE_ID_CHUNK;
	return kIOReturnSuccess;
}

void RgfxAccelerator::freeIdChunks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	UInt32 freed = 0;
	for (UInt32 i = 0; i < RGFX_RESOURCE_ID_CHUNKS; i++) {
		if (idChunkOwners[i] == owner) {
			idChunkOwners[i] = nullptr;
			freed++;
		}
	}
	IOLockUnlock(contextLock);
	if (freed)
		IOLog("RgfxAccelerator: %u resource-id chunks freed\n", freed);
}

// MARK: - Blocks

IOReturn RgfxAccelerator::allocBlock(const void *owner, UInt64 size, UInt64 align, UInt64 *offset)
{
	if (!contextLock || size == 0 || align == 0 || (align & (align - 1)) != 0)
		return kIOReturnBadArgument;
	if (align < kPage)
		align = kPage;
	if (size > blocksEnd - blocksStart)
		return kIOReturnNoSpace;
	size = alignUp(size, kPage);
	IOLockLock(contextLock);
	if (blockCount == kMaxBlocks) {
		IOLockUnlock(contextLock);
		return kIOReturnNoResources;
	}
	// First fit: the first gap, in offset order, that holds the aligned block.
	UInt64 cursor = blocksStart;
	UInt32 at = 0;
	for (; at <= blockCount; at++) {
		UInt64 start = alignUp(cursor, align);
		UInt64 gapEnd = at < blockCount ? blocks[at].offset : blocksEnd;
		if (start <= gapEnd && size <= gapEnd - start) {
			memmove(&blocks[at + 1], &blocks[at], sizeof(RgfxBlock) * (blockCount - at));
			blocks[at].offset = start;
			blocks[at].size = size;
			blocks[at].owner = owner;
			blockCount++;
			*offset = start;
			IOLockUnlock(contextLock);
			return kIOReturnSuccess;
		}
		if (at < blockCount)
			cursor = blocks[at].offset + blocks[at].size;
	}
	IOLockUnlock(contextLock);
	return kIOReturnNoSpace;
}

IOReturn RgfxAccelerator::freeBlock(const void *owner, UInt64 offset)
{
	if (!contextLock)
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	// Binary search: the table is sorted by offset.
	UInt32 lo = 0, hi = blockCount;
	while (lo < hi) {
		UInt32 mid = (lo + hi) / 2;
		if (blocks[mid].offset < offset)
			lo = mid + 1;
		else
			hi = mid;
	}
	if (lo == blockCount || blocks[lo].offset != offset || blocks[lo].owner != owner) {
		IOLockUnlock(contextLock);
		return kIOReturnBadArgument;
	}
	memmove(&blocks[lo], &blocks[lo + 1], sizeof(RgfxBlock) * (blockCount - lo - 1));
	blockCount--;
	IOLockUnlock(contextLock);
	return kIOReturnSuccess;
}

void RgfxAccelerator::freeBlocks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	UInt32 kept = 0, freed = 0;
	for (UInt32 i = 0; i < blockCount; i++) {
		if (blocks[i].owner == owner) {
			freed++;
			continue;
		}
		blocks[kept++] = blocks[i];
	}
	blockCount = kept;
	IOLockUnlock(contextLock);
	if (freed)
		IOLog("RgfxAccelerator: %u blocks freed\n", freed);
}

bool RgfxAccelerator::ownsRange(const void *owner, UInt64 offset, UInt64 length)
{
	if (!contextLock)
		return false;
	IOLockLock(contextLock);
	bool owned = false;
	for (UInt32 i = 0; i < blockCount; i++) {
		const RgfxBlock &b = blocks[i];
		if (b.offset > offset)
			break;
		if (offset - b.offset <= b.size && length <= b.size - (offset - b.offset)) {
			owned = b.owner == owner;
			break;
		}
	}
	IOLockUnlock(contextLock);
	return owned;
}

// MARK: - The object lifecycle

IOReturn RgfxAccelerator::postLifecycle(UInt32 type, const void *message, UInt32 size,
                                        UInt32 replyType, void *reply, UInt32 replyCapacity,
                                        UInt32 *replyLength, UInt64 *seqno, UInt32 timeoutMs)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return kIOReturnNotReady;
	IOReturn r = fb->postLifecycle(type, message, size, replyType, reply, replyCapacity,
	                               replyLength, seqno, timeoutMs);
	fb->release();
	return r;
}

UInt32 RgfxAccelerator::openQueue(const RgfxQueueCreate &create)
{
	RgfxComplete answer = {};
	UInt32 length = 0;
	IOReturn r = postLifecycle(RGFX_CMD_QUEUE_CREATE, &create, sizeof(create), RGFX_EVT_COMPLETE,
	                           &answer, sizeof(answer), &length, nullptr, kQueueAnswerMs);
	if (r != kIOReturnSuccess || length < sizeof(answer)) {
		IOLog("RgfxAccelerator: queue 0x%llx: no answer to its create (0x%x)\n", create.obj_id, r);
		return RGFX_STATUS_INVALID_ARGUMENT;
	}
	return answer.status;
}

void RgfxAccelerator::closeQueue(UInt64 objId)
{
	RgfxObjectDestroy destroy = {};
	destroy.obj_id = objId;
	postLifecycle(RGFX_CMD_OBJECT_DESTROY, &destroy, sizeof(destroy), 0, nullptr, 0, nullptr,
	              nullptr, 0);
}
