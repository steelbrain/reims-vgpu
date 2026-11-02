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
 *     surfaces), from a first-fit allocator over a table sorted by offset (RgfxBlockTable.hpp),
 *     each owned by the client that asked. A freed block is given out again once the device
 *     answers its BLOCK_RELEASE OK, which a thread call of the kext's own posts, so a free
 *     returns at once; the client's are freed after its close is answered, and kept from use
 *     for good where it is not.
 *   - The shared blocks of IOSurface planes (RgfxSurfaceBlocks.hpp): one block per plane asked
 *     for by every process that holds the surface (the bundle asks by each surface's plane 0, its
 *     block holding every plane), found by the plane's physical pages, which the kext wires in
 *     each asking process's task while it holds the block. The device learns of each
 *     holder (BLOCK_SHARE) before the process learns of the block, and of each but the last
 *     letting go (BLOCK_UNSHARE); the last frees the block as a client's free does.
 *   - The completion rings of the clients' queues, watched: the device raises its one vector
 *     for a queue's completion as for the control ring's, and the framebuffer's work loop then
 *     calls queueInterrupt, which wakes the client of each ring whose producer moved and
 *     holds records the client has not taken. Each ring's header is mapped into the kernel
 *     at its queue's create; the scan takes only the watch table's lock.
 *   - The object lifecycle, on the framebuffer's control ring: creates and destroys, a
 *     QUEUE_CREATE answered once the host has the queue's ring pair (so the process writes it at
 *     once), and a CONTEXT_CLOSE answered once the host has dropped the context's objects.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "ReimsGraphics.hpp"
#include "RgfxSurfaceBlocks.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/thread_call.h>
#include <libkern/libkern.h>

namespace {
// The most blocks handed out at once.
constexpr UInt32 kMaxBlocks = 16384;
// The most command queues watched at once, over every client.
constexpr UInt32 kMaxWatched = 2048;
// A ring header's fields (RgfxRingHeader).
constexpr UInt32 kRingProducer = 0x00;
constexpr UInt32 kRingConsumer = 0x40;
// How long a QUEUE_CREATE and a CONTEXT_CLOSE wait for their answers: the host takes the
// control ring in order, behind whatever pipeline compiles were posted before them.
constexpr UInt32 kQueueAnswerMs = 30000;
constexpr UInt32 kCloseAnswerMs = 10000;
// How long a BLOCK_RELEASE waits for its answer, which comes once the context's command buffers
// committed before it have finished. One not answered in time keeps its block.
constexpr UInt32 kReleaseAnswerMs = 10000;
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
	RgfxBlockTable blockTable;
	// Under contextLock: the client holding each chunk of the texture resource ids, null where
	// none does.
	const void *idChunkOwners[RGFX_RESOURCE_ID_CHUNKS];
	// Under watchLock: the watched completion rings, and whether the framebuffer calls
	// queueInterrupt yet.
	IOLock *watchLock;
	RgfxWatchedRing *watched;
	UInt32 watchedCount;
	bool hooked;
	// The thread call that posts freed blocks' releases (releaseFreedBlocks).
	thread_call_t releaseCall;
};
ContextState state = {
    nullptr, 0, 0, 1, {nullptr, 0, kMaxBlocks, 0, 0, 0}, {}, nullptr, nullptr, 0, false, nullptr};
IOLock *&contextLock = state.contextLock;
UInt32 &slotsHeld = state.slotsHeld;
UInt32 &nextSlot = state.nextSlot;
UInt32 &nextContext = state.nextContext;
RgfxBlockTable &blockTable = state.blockTable;
RgfxBlock *&blocks = state.blockTable.blocks;
UInt64 &blocksStart = state.blockTable.start;
UInt64 &blocksEnd = state.blockTable.end;
thread_call_t &releaseCall = state.releaseCall;
const void *(&idChunkOwners)[RGFX_RESOURCE_ID_CHUNKS] = state.idChunkOwners;
IOLock *&watchLock = state.watchLock;
RgfxWatchedRing *&watched = state.watched;
UInt32 &watchedCount = state.watchedCount;
bool &hooked = state.hooked;
// Under contextLock: the shared blocks of surface planes, made at the first ask.
RgfxSurfaceTable surfaceTable = {nullptr, 0};

UInt64 loadHeader64(const volatile UInt8 *header, UInt32 field)
{
	return __atomic_load_n(reinterpret_cast<const volatile UInt64 *>(header + field),
	                       __ATOMIC_ACQUIRE);
}

void queueInterruptHook(OSObject *target)
{
	if (RgfxAccelerator *accel = OSDynamicCast(RgfxAccelerator, target))
		accel->queueInterrupt();
}

// The release thread call: posts each freed block's BLOCK_RELEASE in turn and waits for its
// answer here, never on the freeing thread. OK gives the block out again; a refusal, or no answer
// in time, keeps it until its owner's close is answered.
void releaseFreedBlocks(thread_call_param_t accelerator, thread_call_param_t)
{
	RgfxAccelerator *accel = static_cast<RgfxAccelerator *>(accelerator);
	for (;;) {
		RgfxBlock block;
		IOLockLock(contextLock);
		bool any = rgfxBlockNextRelease(blockTable, &block);
		IOLockUnlock(contextLock);
		if (!any)
			return;
		RgfxBlockRelease release = {};
		release.context = block.context;
		release.offset = block.offset;
		release.size = block.size;
		RgfxComplete answer = {};
		UInt32 length = 0;
		IOReturn r = accel->postLifecycle(RGFX_CMD_BLOCK_RELEASE, &release, sizeof(release),
		                                  RGFX_EVT_COMPLETE, &answer, sizeof(answer), &length,
		                                  nullptr, kReleaseAnswerMs);
		bool ok = r == kIOReturnSuccess && length >= sizeof(answer) &&
		          answer.status == RGFX_STATUS_OK;
		IOLockLock(contextLock);
		rgfxBlockReleased(blockTable, block.offset, block.serial, ok);
		IOLockUnlock(contextLock);
		if (!ok)
			IOLog("RgfxAccelerator: context %u's block 0x%llx (0x%llx bytes) is kept: %s\n",
			      block.context, block.offset, block.size,
			      r == kIOReturnSuccess ? "the device refused its release"
			                            : "its release was not answered");
	}
}
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
	IOLock *wlock = IOLockAlloc();
	RgfxBlock *table = static_cast<RgfxBlock *>(IOMalloc(sizeof(RgfxBlock) * kMaxBlocks));
	RgfxWatchedRing *rings =
	    static_cast<RgfxWatchedRing *>(IOMalloc(sizeof(RgfxWatchedRing) * kMaxWatched));
	// The accelerator is the device's one, for the kernel's life, as the state is.
	thread_call_t call = thread_call_allocate(releaseFreedBlocks, this);
	RgfxBlock *expected = nullptr;
	if (lock && wlock && table && rings && call &&
	    __atomic_compare_exchange_n(&blocks, &expected, table, false, __ATOMIC_ACQ_REL,
	                                __ATOMIC_ACQUIRE)) {
		bzero(table, sizeof(RgfxBlock) * kMaxBlocks);
		bzero(rings, sizeof(RgfxWatchedRing) * kMaxWatched);
		watched = rings;
		watchLock = wlock;
		releaseCall = call;
		blocksStart = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
		blocksEnd = (regionSize - RGFX_CURSOR_SPAN) & ~(kPage - 1);
		__atomic_store_n(&contextLock, lock, __ATOMIC_RELEASE);
		IOLog("RgfxAccelerator: blocks 0x%llx-0x%llx, %u context slots\n", blocksStart, blocksEnd,
		      RGFX_CONTEXT_SLOTS);
		return true;
	}
	if (lock)
		IOLockFree(lock);
	if (wlock)
		IOLockFree(wlock);
	if (table)
		IOFree(table, sizeof(RgfxBlock) * kMaxBlocks);
	if (rings)
		IOFree(rings, sizeof(RgfxWatchedRing) * kMaxWatched);
	if (call)
		thread_call_free(call);
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

bool RgfxAccelerator::closeContext(UInt32 context, UInt32 slot)
{
	// The device ends the context's queues and drops its objects, and answers once the
	// context's command buffers have finished; the caller has closed its queues already, and
	// frees its blocks after this only where the close was answered OK.
	RgfxContextClose close = {};
	close.context = context;
	RgfxComplete answer = {};
	UInt32 length = 0;
	IOReturn r = postLifecycle(RGFX_CMD_CONTEXT_CLOSE, &close, sizeof(close), RGFX_EVT_COMPLETE,
	                           &answer, sizeof(answer), &length, nullptr, kCloseAnswerMs);
	bool closed = r == kIOReturnSuccess && length >= sizeof(answer) &&
	              answer.status == RGFX_STATUS_OK;
	if (!closed)
		IOLog("RgfxAccelerator: context %u: the device did not answer its close (0x%x, "
		      "status %u)\n",
		      context, r, length >= sizeof(answer) ? answer.status : 0);
	if (contextLock && slot < RGFX_CONTEXT_SLOTS) {
		IOLockLock(contextLock);
		slotsHeld &= ~(1u << slot);
		IOLockUnlock(contextLock);
	}
	IOLog("RgfxAccelerator: context %u closed (slot %u)\n", context, slot);
	return closed;
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

IOReturn RgfxAccelerator::allocBlock(RgfxAcceleratorUserClient *owner, UInt32 context,
                                    UInt64 size, UInt64 align, UInt64 *offset)
{
	if (!contextLock)
		return kIOReturnBadArgument;
	IOLockLock(contextLock);
	RgfxBlockPin pin = {};
	RgfxBlockResult r = rgfxBlockAlloc(blockTable, owner, context, size, align, offset, &pin);
	IOLockUnlock(contextLock);
	switch (r) {
	case kRgfxBlockOk: {
		const UInt64 bytes = rgfxBlockAlignUp(size, kRgfxBlockPage); // the block's, as allocated
		// The block is in the process's window, and the device learns of it, before the
		// process does: the grant is on the control ring before any message naming it, on that
		// ring or a queue's. The block is pinned until then, so no free overtakes either.
		IOReturn posted = owner->mapSpan(pin.offset, bytes);
		if (posted == kIOReturnSuccess) {
			RgfxBlockGrant grant = {};
			grant.context = context;
			grant.offset = pin.offset;
			grant.size = bytes;
			posted = postLifecycle(RGFX_CMD_BLOCK_GRANT, &grant, sizeof(grant), 0, nullptr, 0,
			                       nullptr, nullptr, 0);
		}
		IOLockLock(contextLock);
		if (posted == kIOReturnSuccess) {
			rgfxBlockUnpin(blockTable, pin);
		} else {
			// Out of the window again before the block goes back.
			owner->unmapSpan(pin.offset, bytes);
			rgfxBlockDrop(blockTable, pin);
		}
		IOLockUnlock(contextLock);
		return posted;
	}
	case kRgfxBlockNoSpace:
		return kIOReturnNoSpace;
	case kRgfxBlockTableFull:
		return kIOReturnNoResources;
	default:
		return kIOReturnBadArgument;
	}
}

IOReturn RgfxAccelerator::freeBlock(RgfxAcceleratorUserClient *owner, UInt64 offset)
{
	if (!contextLock)
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	// Out of the process's window before the block is freed, under the lock, so no other free
	// of it and no new grant of its span comes between.
	UInt64 size = 0;
	RgfxBlockResult r = rgfxBlockFreeable(blockTable, owner, offset, &size);
	if (r == kRgfxBlockOk) {
		owner->unmapSpan(offset, size);
		r = rgfxBlockFree(blockTable, owner, offset);
	}
	IOLockUnlock(contextLock);
	if (r == kRgfxBlockBusy)
		return kIOReturnBusy;
	if (r != kRgfxBlockOk)
		return kIOReturnBadArgument;
	// The release goes from the thread call, after every message the client posted before
	// this free; the block is no one's to name meanwhile.
	thread_call_enter(releaseCall);
	return kIOReturnSuccess;
}

void RgfxAccelerator::freeBlocks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	UInt32 freed = rgfxBlockFreeOwner(blockTable, owner);
	IOLockUnlock(contextLock);
	if (freed)
		IOLog("RgfxAccelerator: %u blocks freed\n", freed);
}

void RgfxAccelerator::keepBlocks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	UInt32 kept = rgfxBlockKeepOwner(blockTable, owner);
	IOLockUnlock(contextLock);
	if (kept)
		IOLog("RgfxAccelerator: %u blocks kept from use: their context's close was not "
		      "answered\n",
		      kept);
}

bool RgfxAccelerator::pinRange(const void *owner, UInt64 offset, UInt64 length, RgfxBlockPin *pin)
{
	if (!contextLock)
		return false;
	IOLockLock(contextLock);
	bool pinned = rgfxBlockPin(blockTable, owner, offset, length, pin);
	// Or a shared block of a surface plane the owner holds.
	RgfxSurfaceTable &table = surfaceTable;
	if (!pinned && table.entries && rgfxSurfaceHeldBy(table, owner, offset, length))
		pinned = rgfxBlockPinAny(blockTable, offset, length, pin);
	IOLockUnlock(contextLock);
	// Or a scanout span a surface of its process is drawn into (RgfxAcceleratorSurfaces.cpp):
	// no block holds it and nothing frees it, so its pin names no block (serial 0).
	if (!pinned && grantedRange(owner, offset, length)) {
		*pin = {offset, 0};
		pinned = true;
	}
	return pinned;
}

void RgfxAccelerator::unpinBlock(const RgfxBlockPin &pin)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	rgfxBlockUnpin(blockTable, pin);
	IOLockUnlock(contextLock);
}

// MARK: - Shared blocks of surface planes

namespace {
// The most planes with a shared block at once, and the most bytes of one plane.
constexpr UInt32 kMaxSurfaceBlocks = 512;
constexpr UInt64 kMaxPlaneBytes = 256ull << 20;

// The plane of `length` bytes at `address` in `task`, wired: its descriptor, prepared, and its
// pages' runs (`*runs`, IOMalloc'd, `*count` of them, the caller's to free with freeRuns), or
// null where the range is not mapped there or its runs do not fit.
IOMemoryDescriptor *wirePlane(task_t task, UInt64 address, UInt64 length, RgfxPageRun **runs,
                              UInt32 *count)
{
	IOMemoryDescriptor *md =
	    IOMemoryDescriptor::withAddressRange(address, length, kIODirectionOutIn, task);
	if (!md)
		return nullptr;
	if (md->prepare(kIODirectionOutIn) != kIOReturnSuccess) {
		md->release();
		return nullptr;
	}
	// One run a page at most.
	UInt32 capacity = static_cast<UInt32>(((address & (kPage - 1)) + length + kPage - 1) / kPage);
	RgfxPageRun *all = static_cast<RgfxPageRun *>(IOMalloc(sizeof(RgfxPageRun) * capacity));
	UInt32 n = 0;
	bool fits = all != nullptr;
	for (UInt64 at = 0; fits && at < length;) {
		IOByteCount segment = 0;
		addr64_t phys = md->getPhysicalSegment(at, &segment, kIOMemoryMapperNone);
		if (!phys || !segment) {
			fits = false;
			break;
		}
		if (segment > length - at)
			segment = length - at;
		fits = rgfxSurfaceAppendRun(all, &n, capacity, phys, segment);
		at += segment;
	}
	RgfxPageRun *kept = fits && n ? static_cast<RgfxPageRun *>(IOMalloc(sizeof(RgfxPageRun) * n))
	                              : nullptr;
	if (kept)
		bcopy(all, kept, sizeof(RgfxPageRun) * n);
	if (all)
		IOFree(all, sizeof(RgfxPageRun) * capacity);
	if (!kept) {
		md->complete(kIODirectionOutIn);
		md->release();
		return nullptr;
	}
	*runs = kept;
	*count = n;
	return md;
}

void freeRuns(RgfxPageRun *runs, UInt32 count)
{
	if (runs)
		IOFree(runs, sizeof(RgfxPageRun) * count);
}

void unwire(void *pages)
{
	if (IOMemoryDescriptor *md = static_cast<IOMemoryDescriptor *>(pages)) {
		md->complete(kIODirectionOutIn);
		md->release();
	}
}

// The last holder of entry `e` let it go: its block is freed (its BLOCK_RELEASE names
// `context`), or kept for good where the device may still use it (`keep`, a close not answered,
// or a create naming it in flight), and the entry goes. Under contextLock; whether a release is
// to be posted.
bool letGoBlock(RgfxSurfaceEntry &e, UInt32 context, bool keep)
{
	bool release = false;
	if (!keep) {
		rgfxBlockSetContext(blockTable, e.offset, context);
		release = rgfxBlockFree(blockTable, &e, e.offset) == kRgfxBlockOk;
	}
	if (!release)
		rgfxBlockKeepAt(blockTable, e.offset);
	freeRuns(e.runs, e.runCount);
	rgfxSurfaceClear(e);
	return release;
}
} // namespace

IOReturn RgfxAccelerator::acquireSurfaceBlock(RgfxAcceleratorUserClient *client, UInt32 context,
                                             task_t task, UInt64 address, UInt64 length,
                                             UInt64 size, SurfaceBlock *block)
{
	if (!contextLock)
		return kIOReturnNotReady;
	if (!length || length > kMaxPlaneBytes || !size || size % kPage || size < length ||
	    size > 2 * kMaxPlaneBytes)
		return kIOReturnBadArgument;
	RgfxPageRun *runs = nullptr;
	UInt32 count = 0;
	IOMemoryDescriptor *pages = wirePlane(task, address, length, &runs, &count);
	if (!pages)
		return kIOReturnBadArgument;
	RgfxSurfaceTable &table = surfaceTable;
	IOLockLock(contextLock);
	if (!table.entries) {
		auto *made = static_cast<RgfxSurfaceEntry *>(
		    IOMalloc(sizeof(RgfxSurfaceEntry) * kMaxSurfaceBlocks));
		if (made) {
			bzero(made, sizeof(RgfxSurfaceEntry) * kMaxSurfaceBlocks);
			table = {made, kMaxSurfaceBlocks};
		}
	}
	bool sizeDiffers = false;
	int32_t at = table.entries ? rgfxSurfaceFind(table, runs, count, length, size, &sizeDiffers)
	                           : -1;
	if (at >= 0) {
		// Another process's plane: this client holds its block too, in its window from its
		// first hold, mapped under the lock so no let-go of the hold comes between.
		RgfxSurfaceEntry &e = table.entries[at];
		RgfxSurfaceHoldResult held = rgfxSurfaceHold(e, client, context, pages);
		*block = {e.offset, e.size, e.serial, false};
		IOReturn mapped = kIOReturnSuccess;
		if (held == kRgfxSurfaceHoldNew) {
			mapped = client->mapSpan(e.offset, e.size);
			if (mapped != kIOReturnSuccess) {
				// No hold the device was not told of, nor one not in the window.
				void *mine = nullptr;
				UInt32 c = 0;
				rgfxSurfaceLetGo(e, client, true, &mine, &c);
				client->unmapSpan(e.offset, e.size);
				held = kRgfxSurfaceHoldFull;
			}
		}
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		if (held != kRgfxSurfaceHoldNew) {
			unwire(pages);
			if (mapped != kIOReturnSuccess)
				return mapped;
			return held == kRgfxSurfaceHoldAgain ? kIOReturnSuccess : kIOReturnNoResources;
		}
		RgfxBlockGrant share = {};
		share.context = context;
		share.offset = block->offset;
		share.size = block->size;
		IOReturn posted = postLifecycle(RGFX_CMD_BLOCK_SHARE, &share, sizeof(share), 0, nullptr,
		                                0, nullptr, nullptr, 0);
		if (posted != kIOReturnSuccess) {
			// No hold the device was not told of: let it go again.
			void *mine = nullptr;
			UInt32 c = 0;
			IOLockLock(contextLock);
			bool release = false;
			int32_t again = rgfxSurfaceAt(table, block->offset);
			RgfxSurfaceLetGo gone = again >= 0 ? rgfxSurfaceLetGo(table.entries[again], client,
			                                                      true, &mine, &c)
			                                   : kRgfxSurfaceLetGoNotHeld;
			if (gone != kRgfxSurfaceLetGoNotHeld)
				client->unmapSpan(block->offset, block->size);
			if (gone == kRgfxSurfaceLetGoLast)
				release = letGoBlock(table.entries[again], c, false);
			IOLockUnlock(contextLock);
			unwire(mine);
			if (release)
				thread_call_enter(releaseCall);
		}
		return posted;
	}
	int32_t slot = !sizeDiffers && table.entries ? rgfxSurfaceFreeEntry(table) : -1;
	RgfxBlockPin pin = {};
	UInt64 offset = 0;
	RgfxBlockResult made = slot < 0 ? kRgfxBlockTableFull
	                                : rgfxBlockAlloc(blockTable, &table.entries[slot], context,
	                                                 size, kPage, &offset, &pin);
	if (made != kRgfxBlockOk) {
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		unwire(pages);
		if (sizeDiffers)
			return kIOReturnBadArgument;
		return made == kRgfxBlockNoSpace ? kIOReturnNoSpace : kIOReturnNoResources;
	}
	RgfxSurfaceEntry &e = table.entries[slot];
	e.used = true;
	e.offset = offset;
	e.size = size;
	e.serial = pin.serial;
	e.length = length;
	e.runs = runs;
	e.runCount = count;
	rgfxSurfaceHold(e, client, context, pages);
	*block = {offset, size, pin.serial, true};
	// In the client's window, under the lock, so no let-go of the hold comes between. A block
	// that is not goes back as if never made: the device was not told of it.
	IOReturn mapped = client->mapSpan(offset, size);
	if (mapped != kIOReturnSuccess) {
		client->unmapSpan(offset, size);
		rgfxBlockDrop(blockTable, pin);
		rgfxSurfaceClear(e);
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		unwire(pages);
		return mapped;
	}
	IOLockUnlock(contextLock);
	// The device learns of the block before any process does; it is pinned until then, so no
	// release overtakes the share.
	RgfxBlockGrant share = {};
	share.context = context;
	share.offset = offset;
	share.size = size;
	IOReturn posted = postLifecycle(RGFX_CMD_BLOCK_SHARE, &share, sizeof(share), 0, nullptr, 0,
	                                nullptr, nullptr, 0);
	void *mine = nullptr;
	bool release = false;
	IOLockLock(contextLock);
	rgfxBlockUnpin(blockTable, pin);
	if (posted != kIOReturnSuccess) {
		UInt32 c = 0;
		RgfxSurfaceLetGo gone = rgfxSurfaceLetGo(e, client, true, &mine, &c);
		if (gone != kRgfxSurfaceLetGoNotHeld)
			client->unmapSpan(offset, size);
		if (gone == kRgfxSurfaceLetGoLast)
			release = letGoBlock(e, c, false);
	}
	IOLockUnlock(contextLock);
	unwire(mine);
	if (release)
		thread_call_enter(releaseCall);
	return posted;
}

IOReturn RgfxAccelerator::releaseSurfaceBlock(RgfxAcceleratorUserClient *client, UInt64 offset)
{
	if (!contextLock)
		return kIOReturnNotReady;
	RgfxSurfaceTable &table = surfaceTable;
	IOLockLock(contextLock);
	int32_t at = table.entries ? rgfxSurfaceAt(table, offset) : -1;
	void *pages = nullptr;
	UInt32 context = 0;
	RgfxSurfaceLetGo gone = at < 0 ? kRgfxSurfaceLetGoNotHeld
	                               : rgfxSurfaceLetGo(table.entries[at], client, false, &pages,
	                                                  &context);
	UInt64 size = at < 0 ? 0 : table.entries[at].size;
	// Out of the client's window once its hold ends, before the block can be anyone else's.
	if (gone == kRgfxSurfaceLetGoHold || gone == kRgfxSurfaceLetGoLast)
		client->unmapSpan(offset, size);
	bool release = false;
	if (gone == kRgfxSurfaceLetGoLast)
		release = letGoBlock(table.entries[at], context, false);
	IOLockUnlock(contextLock);
	if (gone == kRgfxSurfaceLetGoHold) {
		// The other holders keep it; the device no longer lets this context name it.
		RgfxBlockGrant unshare = {};
		unshare.context = context;
		unshare.offset = offset;
		unshare.size = size;
		postLifecycle(RGFX_CMD_BLOCK_UNSHARE, &unshare, sizeof(unshare), 0, nullptr, 0, nullptr,
		              nullptr, 0);
	}
	unwire(pages);
	if (release)
		thread_call_enter(releaseCall);
	return gone == kRgfxSurfaceLetGoNotHeld ? kIOReturnBadArgument : kIOReturnSuccess;
}

void RgfxAccelerator::dropSurfaceBlocks(const void *client, bool closed)
{
	if (!contextLock)
		return;
	RgfxSurfaceTable &table = surfaceTable;
	bool release = false;
	UInt32 dropped = 0;
	for (UInt32 i = 0; i < table.capacity; i++) {
		void *pages = nullptr;
		UInt32 context = 0;
		IOLockLock(contextLock);
		RgfxSurfaceLetGo gone =
		    table.entries[i].used
		        ? rgfxSurfaceLetGo(table.entries[i], client, true, &pages, &context)
		        : kRgfxSurfaceLetGoNotHeld;
		// The device ended the context's holds at its close; the last holder's block goes as
		// its blocks go, or is kept where the close was not answered.
		if (gone == kRgfxSurfaceLetGoLast)
			release |= letGoBlock(table.entries[i], context, !closed);
		IOLockUnlock(contextLock);
		if (gone != kRgfxSurfaceLetGoNotHeld)
			dropped++;
		unwire(pages);
	}
	if (release)
		thread_call_enter(releaseCall);
	if (dropped)
		IOLog("RgfxAccelerator: %u shared surface blocks let go at a close\n", dropped);
}

// MARK: - The queues' completion rings

IOReturn RgfxAccelerator::watchQueue(RgfxAcceleratorUserClient *client, UInt64 objId,
                                     UInt64 completeOffset)
{
	if (!contextLock || !region || completeOffset % 8 ||
	    completeOffset + RGFX_RING_HEADER_SIZE > regionSize)
		return kIOReturnBadArgument;
	// The header's page or pages, write-back as the bundle maps the region.
	UInt64 page = completeOffset & ~(kPage - 1);
	UInt64 length = alignUp(completeOffset + RGFX_RING_HEADER_SIZE, kPage) - page;
	IOMemoryMap *map = region->createMappingInTask(kernel_task, 0,
	                                               kIOMapAnywhere | kIOMapCopybackCache, page,
	                                               length);
	if (!map)
		return kIOReturnNoMemory;
	volatile UInt8 *header =
	    reinterpret_cast<volatile UInt8 *>(map->getVirtualAddress()) + (completeOffset - page);
	IOLockLock(watchLock);
	if (watchedCount == kMaxWatched) {
		IOLockUnlock(watchLock);
		map->release();
		IOLog("RgfxAccelerator: queue 0x%llx: every watched ring is taken\n", objId);
		return kIOReturnNoResources;
	}
	RgfxWatchedRing &w = watched[watchedCount++];
	w.client = client;
	w.objId = objId;
	w.map = map;
	w.header = header;
	w.lastProducer = loadHeader64(header, kRingProducer);
	bool hook = !hooked;
	hooked = true;
	IOLockUnlock(watchLock);
	// The first watched ring has the framebuffer call queueInterrupt from now on.
	if (hook) {
		ReimsGraphics *fb = copyFramebuffer();
		if (fb) {
			fb->setQueueInterrupt(this, &queueInterruptHook);
			fb->release();
		} else {
			IOLockLock(watchLock);
			hooked = false;
			IOLockUnlock(watchLock);
		}
	}
	return kIOReturnSuccess;
}

void RgfxAccelerator::unwatchQueue(RgfxAcceleratorUserClient *client, UInt64 objId)
{
	if (!watchLock)
		return;
	IOMemoryMap *map = nullptr;
	IOLockLock(watchLock);
	for (UInt32 i = 0; i < watchedCount; i++) {
		if (watched[i].client == client && watched[i].objId == objId) {
			map = watched[i].map;
			watched[i] = watched[--watchedCount];
			break;
		}
	}
	IOLockUnlock(watchLock);
	if (map)
		map->release();
}

void RgfxAccelerator::unwatchClient(RgfxAcceleratorUserClient *client)
{
	if (!watchLock)
		return;
	IOLockLock(watchLock);
	for (UInt32 i = 0; i < watchedCount;) {
		if (watched[i].client == client) {
			// Released under the lock: the scan never meets a ring whose map is gone.
			watched[i].map->release();
			watched[i] = watched[--watchedCount];
		} else {
			i++;
		}
	}
	IOLockUnlock(watchLock);
}

// On the framebuffer's work loop, once per pass over the control ring: the device raised its
// vector, for a completion on some queue's ring or the control ring's. A ring whose producer
// moved since the last look and that holds records its client has not taken wakes the client.
void RgfxAccelerator::queueInterrupt()
{
	if (!watchLock)
		return;
	IOLockLock(watchLock);
	for (UInt32 i = 0; i < watchedCount; i++) {
		RgfxWatchedRing &w = watched[i];
		UInt64 producer = loadHeader64(w.header, kRingProducer);
		if (producer == w.lastProducer)
			continue;
		w.lastProducer = producer;
		if (producer != loadHeader64(w.header, kRingConsumer))
			w.client->wakeCompletions();
	}
	IOLockUnlock(watchLock);
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
