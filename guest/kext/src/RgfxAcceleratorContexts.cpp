/*
 * RgfxAccelerator's contexts: what each Metal process that opens our user client holds of the
 * device, and what the accelerator tells the device of it.
 *
 *   - A context number, handed out once from 1: the process's object ids are
 *     c << RGFX_CONTEXT_OBJ_ID_SHIFT and up, so no two processes' ids meet in the host's one
 *     handle table, and the host drops the process's at CONTEXT_CLOSE.
 *   - A slot of RGFX_CONTEXT_SLOTS, held until the close: the process's GPU addresses and
 *     texture resource ids come from the slot's range. Slots are handed out round-robin, so a
 *     slot freed by one process is the last one the next process takes.
 *   - Region blocks past RGFX_CONTEXT_ALLOCATIONS_OFFSET (rings, Shared buffers and heaps,
 *     surfaces), from a first-fit allocator over a table sorted by offset, each owned by the
 *     client that asked; the client's are freed at its close.
 *   - Command queues, opened and closed through the device's mailbox on the vCPU, so the host
 *     has a queue's ring pair before openQueue returns and the process writes it at once.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAccelerator.hpp"

#include <IOKit/IOLib.h>
#include <libkern/libkern.h>

namespace {
// The most blocks handed out at once.
constexpr UInt32 kMaxBlocks = 16384;
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
	// Serialises the mailbox and its register writes.
	IOLock *mailboxLock;
};
ContextState state = {nullptr, 0, 0, 1, nullptr, 0, 0, 0, nullptr};
IOLock *&contextLock = state.contextLock;
UInt32 &slotsHeld = state.slotsHeld;
UInt32 &nextSlot = state.nextSlot;
UInt32 &nextContext = state.nextContext;
RgfxBlock *&blocks = state.blocks;
UInt32 &blockCount = state.blockCount;
UInt64 &blocksStart = state.blocksStart;
UInt64 &blocksEnd = state.blocksEnd;
IOLock *&mailboxLock = state.mailboxLock;
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
	IOLock *mbox = IOLockAlloc();
	RgfxBlock *table = static_cast<RgfxBlock *>(IOMalloc(sizeof(RgfxBlock) * kMaxBlocks));
	IOLock *expected = nullptr;
	if (lock && mbox && table && __atomic_compare_exchange_n(&mailboxLock, &expected, mbox, false,
	                                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
		bzero(table, sizeof(RgfxBlock) * kMaxBlocks);
		blocks = table;
		blocksStart = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
		blocksEnd = (regionSize - RGFX_CURSOR_SPAN) & ~(kPage - 1);
		__atomic_store_n(&contextLock, lock, __ATOMIC_RELEASE);
		IOLog("RgfxAccelerator: blocks 0x%llx-0x%llx, %u context slots\n", blocksStart, blocksEnd,
		      RGFX_CONTEXT_SLOTS);
		return true;
	}
	if (lock)
		IOLockFree(lock);
	if (mbox)
		IOLockFree(mbox);
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
	// The device ends the context's queues and drops its objects; the caller has closed its
	// queues already, and frees its blocks after this.
	if (registers && mailboxLock) {
		IOLockLock(mailboxLock);
		writeRegister(RGFX_REG_CONTEXT_CLOSE, context);
		IOLockUnlock(mailboxLock);
	}
	if (contextLock && slot < RGFX_CONTEXT_SLOTS) {
		IOLockLock(contextLock);
		slotsHeld &= ~(1u << slot);
		IOLockUnlock(contextLock);
	}
	IOLog("RgfxAccelerator: context %u closed (slot %u)\n", context, slot);
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

// MARK: - Command queues

UInt32 RgfxAccelerator::openQueue(const RgfxQueueOpen &open)
{
	if (!registers || !mailboxLock)
		return RGFX_STATUS_INVALID_ARGUMENT;
	static_assert(sizeof(RgfxQueueOpen) <= RGFX_MAILBOX_WORDS * 4, "the mailbox holds the open");
	UInt32 words[RGFX_MAILBOX_WORDS] = {};
	memcpy(words, &open, sizeof(open));
	IOLockLock(mailboxLock);
	for (UInt32 i = 0; i < (sizeof(open) + 3) / 4; i++)
		writeRegister(RGFX_REG_MAILBOX + i * 4, words[i]);
	writeRegister(RGFX_REG_QUEUE_OPEN, 1);
	UInt32 status = readRegister(RGFX_REG_MAILBOX_STATUS);
	IOLockUnlock(mailboxLock);
	return status;
}

void RgfxAccelerator::closeQueue(UInt64 objId)
{
	if (!registers || !mailboxLock)
		return;
	IOLockLock(mailboxLock);
	writeRegister(RGFX_REG_MAILBOX, static_cast<UInt32>(objId));
	writeRegister(RGFX_REG_MAILBOX + 4, static_cast<UInt32>(objId >> 32));
	writeRegister(RGFX_REG_QUEUE_CLOSE, 1);
	IOLockUnlock(mailboxLock);
}
