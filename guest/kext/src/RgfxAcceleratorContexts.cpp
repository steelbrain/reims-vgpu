/*
 * RgfxAccelerator's contexts: what each Metal process that opens our user client holds of the
 * device, and what the accelerator tells the device of it.
 *
 *   - A context number, handed out once from 1: the process's object ids are
 *     c << RGFX_CONTEXT_OBJ_ID_SHIFT and up, so no two processes' ids meet in the host's one
 *     handle table, and the host drops the process's at CONTEXT_CLOSE.
 *   - A run of units of the guest's GPU addresses (RgfxContextSlots.hpp), held until the close:
 *     the process's GPU addresses come from the run's range. Runs are handed out round-robin, so
 *     a run freed by one process is the last one the next process takes, and the last units
 *     free go to WindowServer alone.
 *   - Chunks of the texture resource ids, on demand, from one pool for the guest: every
 *     process's textures share the host device's one space of ids, which ends below
 *     RGFX_RESOURCE_ID_LIMIT, so ids are not split by context. Each chunk is posted to the device
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
 *     letting go (BLOCK_UNSHARE); the last frees the block as a client's free does. Where the
 *     device has RGFX_FEATURE_SURFACE_PAGES, a process may ask for a surface's block in place:
 *     the kext wires the surface's whole allocation, writes its runs into the block past the
 *     part the processes map, and hands them to the device (SURFACE_PAGES) before the first
 *     share. A block whose pages the device refuses goes again, and the process copies.
 *     Where the device also has RGFX_FEATURE_SURFACE_CACHE, an ask in place that names its
 *     surface and may have it cached makes a cached block (RgfxSurfaceCaches.hpp): the kext makes
 *     the surface's device cache first, then marks the block cached in its SURFACE_PAGES, and a
 *     cached block's release copies the host's cache into the pages (its locks waiting for it).
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
#include "RgfxCloseTable.hpp"
#include "RgfxContextSlots.hpp"
#include "RgfxIOSurfaceKPI.hpp"
#include "ReimsGraphics.hpp"
#include "RgfxSurfaceBlocks.hpp"
#include "RgfxSurfaceCaches.hpp"
#include "RgfxLatency.hpp"
#include "RgfxPagesBlock.hpp"
#include "RgfxIdChunks.hpp"
#include "RgfxReleaseTable.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/clock.h>
#include <kern/thread_call.h>
#include <libkern/libkern.h>
#include <sys/sysctl.h>

namespace {
// The most blocks handed out at once.
constexpr UInt32 kMaxBlocks = 16384;
// The blocks' reserve for WindowServer (RgfxBlockTable): the last bytes of the region and the last
// entries of the table, which no other process is given. Sized from what WindowServer held on an
// RTX 5080 host at the display's 3840x2400 mode, measured: 134 MiB at most (at login), 118 to
// 126 MiB as five apps opened, in 110 to 366 blocks; so the reserve holds a WindowServer started
// again with a fifth to spare, and three times its blocks.
constexpr UInt64 kWindowServerReserveBytes = 160ull << 20;
constexpr UInt32 kWindowServerReserveEntries = 1024;
// The most command queues watched at once, over every client (RgfxQueueLimits.hpp).
constexpr UInt32 kMaxWatched = kRgfxMaxWatchedQueues;
// A ring header's fields (RgfxRingHeader).
constexpr UInt32 kRingProducer = 0x00;
constexpr UInt32 kRingCompletedSeqno = 0x08;
constexpr UInt32 kRingConsumer = 0x40;
// How long a QUEUE_CREATE and a CONTEXT_CLOSE wait for their answers: the host takes the
// control ring in order, behind whatever pipeline compiles were posted before them.
constexpr UInt32 kQueueAnswerMs = 30000;
constexpr UInt32 kCloseAnswerMs = 10000;
// The most closes in flight, late ones included (RgfxCloseTable.hpp).
constexpr UInt32 kMaxCloses = 32;
// How long a BLOCK_RELEASE waits for its answer, which comes once the context's command buffers
// committed before it have finished, before it is logged as late. A late one stays in flight,
// its block, pages and surface the device's, until its answer.
constexpr UInt32 kReleaseAnswerMs = 10000;
// Every block is page-aligned and a whole number of pages, so a block can back a mapping.
constexpr UInt64 kPage = 0x1000;
// The guest's memory where hw.memsize gives none, for the dormant blocks' cap of bytes.
constexpr UInt64 kDormantMemoryUnknown = 2ull << 30;
// How often the dormant blocks' line is logged at most, and how long a new block waits for
// dormant blocks' releases to give the region room for it.
constexpr UInt32 kDormantLogMs = 10000;
constexpr UInt32 kDormantRoomMs = 100;
// How many dormant blocks a new block that finds no room in the region lets go.
constexpr UInt32 kDormantRoomBlocks = 16;
// How often the releases' line is logged at most.
constexpr UInt32 kReleaseLogMs = 10000;
// The most BLOCK_RELEASEs in flight at once (RgfxReleaseTable.hpp): 40 bytes each on ring 0,
// under a third of its 32 KiB.
constexpr UInt32 kMaxReleases = 256;
// The releases posted under one hold of the ring's lock, with one doorbell at most.
constexpr UInt32 kReleaseBatch = 16;

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
	// Under contextLock: the units of GPU addresses held, the units a context holds (set at the
	// state's start, from the device's maxBufferLength), the next context number, and the
	// blocks handed out, sorted by offset.
	IOLock *contextLock;
	RgfxContextSlots slots;
	UInt32 contextUnits;
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
ContextState state = {nullptr,
                      kRgfxContextSlotsEmpty,
                      1,
                      1,
                      {nullptr, 0, kMaxBlocks, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {},
                      nullptr,
                      nullptr,
                      0,
                      false,
                      nullptr};
IOLock *&contextLock = state.contextLock;
RgfxContextSlots &slots = state.slots;
UInt32 &contextUnits = state.contextUnits;
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
// Whether the device uses surfaces' own pages in place (RGFX_FEATURE_SURFACE_PAGES), read from
// FEATURES_LO at the first context's open.
bool surfacePages = false;
// Whether the device keeps cached surfaces' GPU copies in its own memory
// (RGFX_FEATURE_SURFACE_CACHE), read with it.
bool surfaceCache = false;
// Under contextLock: the shared blocks of surface planes, made at the first ask.
RgfxSurfaceTable surfaceTable = {nullptr, 0};
// The bytes of surface pages the dormant blocks hold at most (RgfxSurfaceBlocks.hpp): an eighth
// of the guest's memory, read at the first context's open.
UInt64 dormantMaxBytes = 0;
// What the dormant blocks did, under contextLock, logged once something changed and
// kDormantLogMs passed since the last line (noteDormant).
struct DormantStats {
	UInt64 slept;
	UInt64 revived;
	UInt64 letGo[kRgfxDormantCauses];
	// Let go as their surface's cache was purged (op 5, a client of the surface going) with a
	// cache the GPU wrote since its last copy: the host copies it into the pages, whether or not
	// a client of the surface is left as the release is posted (`unread`).
	UInt64 purgedWritten;
	// Let go past a cap, a table entry or the region's room, of a surface made volatile.
	UInt64 volatileFirst;
	// Blocks of cached surfaces released with no client of the surface left
	// (RGFX_BLOCK_RELEASE_NO_CLIENT), and of them those whose cache a buffer wrote since its
	// last copy, which the host copies into the pages all the same.
	UInt64 unread;
	UInt64 unreadWritten;
	UInt64 changes;
	UInt64 loggedChanges;
	// The surfaces left uncached for want of room (rgfxSurfaceCacheNoRoom) at the last line.
	UInt64 loggedNoRoom;
	UInt64 loggedAt;
};
DormantStats dormantStats = {};

// What the releases took, under contextLock, logged once more were answered and kReleaseLogMs
// passed since the last line (noteReleases): from a block's free to its release's answer
// (`latency`), and from a dormant block's ask to go to the worker's reap of it (`reapWait`, while
// the worker is busy with releases); the releases kept, the ones late, the most in flight at
// once, and the new surface blocks that waited for the region's room and the ones that found
// none after.
struct ReleaseStats {
	RgfxLatency latency;
	RgfxLatency reapWait;
	UInt64 kept;
	// The releases not answered in kReleaseAnswerMs, each still waiting then.
	UInt64 late;
	UInt32 inFlightMost;
	UInt64 roomWaits;
	UInt64 roomWaitsFailed;
	UInt64 reapAskedAt;
	UInt64 loggedCount;
	UInt64 loggedAt;
};
ReleaseStats releaseStats = {};

UInt64 nanosSince(UInt64 then, UInt64 now)
{
	UInt64 ns = 0;
	if (then && now > then)
		absolutetime_to_nanoseconds(now - then, &ns);
	return ns;
}

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

bool reapDormant();
void noteDormant();
void noteReserve();
void noteReleases();
void unwire(void *pages);
bool releaseAnswered(OSObject *, UInt64 seqno, UInt32 status);

// The releases in flight (RgfxReleaseTable.hpp), made at the first context's open, under
// releaseLock, which the work loop takes to hand an answer over; and the timer that runs the
// worker once the oldest release waited kReleaseAnswerMs.
IOLock *releaseLock = nullptr;
RgfxReleaseTable releaseTable = {nullptr, 0, 0};
thread_call_t releaseTimer = nullptr;
bool releaseHooked = false;

// The closes in flight (RgfxCloseTable.hpp), under releaseLock: their answers come through the
// same hook as the releases'. Its entries' addresses are their late holdings' owners, so it is
// never moved.
RgfxCloseEntry closeEntries[kMaxCloses] = {};
RgfxCloseTable closeTable = {closeEntries, kMaxCloses};

// The framebuffer's work loop hands this worker's hook each COMPLETE no lifecycle waiter takes:
// named once, before the first release or close is posted.
void hookAnswers(RgfxAccelerator *accel)
{
	if (__atomic_load_n(&releaseHooked, __ATOMIC_ACQUIRE))
		return;
	if (ReimsGraphics *fb = accel->copyFramebuffer()) {
		fb->setReleaseAnswer(accel, &releaseAnswered);
		fb->release();
		__atomic_store_n(&releaseHooked, true, __ATOMIC_RELEASE);
	}
}

// The framebuffer's work loop hands each COMPLETE no lifecycle waiter takes here first: an
// answer to a release in flight is recorded, and the worker runs to retire it; an answer to a
// close wakes its waiting thread, or, late, runs the worker to let go of what it held.
bool releaseAnswered(OSObject *, UInt64 seqno, UInt32 status)
{
	IOLock *lock = __atomic_load_n(&releaseLock, __ATOMIC_ACQUIRE);
	if (!lock)
		return false;
	IOLockLock(lock);
	bool mine = rgfxReleaseAnswer(releaseTable, seqno, status);
	const bool close = !mine && rgfxCloseAnswer(closeTable, seqno, status);
	if (close)
		IOLockWakeup(lock, &closeTable, false);
	IOLockUnlock(lock);
	if (mine || close)
		thread_call_enter(releaseCall);
	return mine || close;
}

void releaseTimerFired(thread_call_param_t, thread_call_param_t)
{
	thread_call_enter(releaseCall);
}

// A release done with: answered OK, the block goes out again; refused, or never posted, it is
// kept until a context close posts it again (rgfxBlockReleased). A release not answered in time
// never comes here before its answer (retireReleases).
void finishRelease(const RgfxReleaseEntry &e, bool answered)
{
	const bool ok = answered && e.status == RGFX_STATUS_OK;
	IOLockLock(contextLock);
	// Answered OK, a cached block's pages hold its cache: its surface's locks go on, and its
	// surface and the surface's wire are let go, before the block can be anyone else's. A kept
	// block's surface keeps them, its locks waiting, for the release posted again: the device
	// copies the cache into the pages at that release's answer. Only while the block is still
	// this release's: a pages block its owner's answered close took may have given its offset to
	// another block since, whose own release holds its surface. What it held goes out of the
	// lock.
	RgfxReleasedSurface gone = {};
	if (ok && rgfxBlockReleasingAt(blockTable, e.offset, e.serial))
		rgfxSurfaceCacheReleased(e.offset, &gone);
	if (e.flags & RGFX_BLOCK_RELEASE_NO_CLIENT) {
		dormantStats.unread++;
		dormantStats.unreadWritten += e.written;
		dormantStats.changes++;
	}
	void *wired = nullptr;
	rgfxBlockReleased(blockTable, e.offset, e.serial, ok, &wired);
	rgfxLatencyAdd(releaseStats.latency, nanosSince(e.freedAt, mach_absolute_time()));
	releaseStats.kept += !ok;
	// A new block waiting for the region's room looks again.
	IOLockWakeup(contextLock, &surfaceTable, false);
	IOLockUnlock(contextLock);
	// The device no longer maps an in-place block's pages once it has answered OK: unwire them,
	// and let go of its surface, out of the lock.
	rgfxSurfaceCacheLetGo(gone);
	unwire(wired);
	if (!ok)
		IOLog("RgfxAccelerator: context %u's block 0x%llx (0x%llx bytes) is kept: %s\n",
		      e.context, e.offset, e.size,
		      answered ? "the device refused its release" : "its release was not posted");
}

// Retires every release answered. One waiting past kReleaseAnswerMs is logged, once, and stays in
// flight: the device may still run a buffer over its block or copy its surface's cache into the
// surface's pages, so its block, its wired pages, its surface and the surface's wire stay the
// device's until the answer, which retires it as any other.
void retireReleases()
{
	uint64_t timeout = 0;
	nanoseconds_to_absolutetime(static_cast<uint64_t>(kReleaseAnswerMs) * 1000000ull, &timeout);
	for (;;) {
		RgfxReleaseEntry e;
		bool answered = false;
		IOLockLock(releaseLock);
		const bool any =
		    rgfxReleaseTakeDone(releaseTable, mach_absolute_time(), timeout, &e, &answered);
		IOLockUnlock(releaseLock);
		if (!any)
			return;
		if (answered) {
			finishRelease(e, true);
			continue;
		}
		IOLockLock(contextLock);
		releaseStats.late++;
		IOLockUnlock(contextLock);
		IOLog("RgfxAccelerator: context %u's block 0x%llx (0x%llx bytes): its release is not "
		      "answered after %u ms: the block, its pages and its surface stay the device's until "
		      "it is\n",
		      e.context, e.offset, e.size, kReleaseAnswerMs);
	}
}

// Lets go of what each late close answered held, its token's, as the closing thread lets go of a
// close answered in time (RgfxAcceleratorUserClient::releaseContext): answered OK, the device
// holds nothing of the context, so its resource-id chunks, its blocks (their wired pages
// unwired) and its holds of shared blocks go, its units are given back, and the releases it
// refused are posted again; refused, they are kept from use for good.
void retireCloses(RgfxAccelerator *accel)
{
	for (;;) {
		uint32_t at = 0;
		IOLockLock(releaseLock);
		const bool any = rgfxCloseTakeLate(closeTable, &at);
		const RgfxCloseEntry c = any ? closeTable.entries[at] : RgfxCloseEntry{};
		IOLockUnlock(releaseLock);
		if (!any)
			return;
		const void *token = rgfxCloseToken(closeTable, at);
		const bool ok = c.status == RGFX_STATUS_OK;
		if (ok) {
			accel->freeIdChunks(token);
			accel->freeBlocks(token);
			accel->dropSurfaceBlocks(token, true);
			IOLockLock(contextLock);
			rgfxContextSlotsGive(slots, c.firstUnit, c.units);
			IOLockUnlock(contextLock);
			accel->retryRefusedBlocks();
		} else {
			accel->keepIdChunks(token);
			accel->keepBlocks(token);
			accel->dropSurfaceBlocks(token, false);
		}
		IOLockLock(releaseLock);
		rgfxCloseRetired(closeTable, at);
		IOLockUnlock(releaseLock);
		IOLog("RgfxAccelerator: context %u's late close was answered (status %u): what it held is "
		      "%s\n",
		      c.context, c.status, ok ? "let go" : "kept from use for good");
	}
}

// The release thread call, the guest's one (THREAD_CALL_OPTIONS_ONCE): lets go of the dormant
// blocks their surfaces asked to go (reapDormant); retires the releases answered; then posts
// each freed block's BLOCK_RELEASE on ring 0, kReleaseBatch at a time with one doorbell at most,
// as many as the release table holds, and returns without waiting for any. The work loop hands
// each answer to its entry, in whatever order the contexts' command buffers finish, and runs the
// worker again to retire it; so releases posted together wait for their contexts' drains at
// once, and a slow context holds back only its own blocks. Never on the freeing thread.
void releaseFreedBlocks(thread_call_param_t accelerator, thread_call_param_t)
{
	RgfxAccelerator *accel = static_cast<RgfxAccelerator *>(accelerator);
	IOLockLock(contextLock);
	if (releaseStats.reapAskedAt) {
		rgfxLatencyAdd(releaseStats.reapWait,
		               nanosSince(releaseStats.reapAskedAt, mach_absolute_time()));
		releaseStats.reapAskedAt = 0;
	}
	reapDormant();
	IOLockUnlock(contextLock);
	noteDormant();
	// The answers come through the framebuffer's work loop: name this worker's hook there
	// before the first release is posted.
	hookAnswers(accel);
	// The late closes answered let go of what they held, whose releases this run posts.
	retireCloses(accel);
	retireReleases();
	for (bool more = true; more;) {
		// A batch of freed blocks, each in an entry of its own, posted at once.
		RgfxBlockRelease batch[kReleaseBatch];
		UInt64 *seqnos[kReleaseBatch];
		int32_t at[kReleaseBatch];
		UInt32 n = 0;
		while (n < kReleaseBatch) {
			IOLockLock(releaseLock);
			at[n] = rgfxReleaseReserve(releaseTable);
			IOLockUnlock(releaseLock);
			// A full table: the answers that free its entries run the worker again.
			if (at[n] < 0) {
				more = false;
				break;
			}
			RgfxReleaseEntry &e = releaseTable.entries[at[n]];
			RgfxBlock block;
			IOLockLock(contextLock);
			const bool any = rgfxBlockNextRelease(blockTable, &block);
			IOLockUnlock(contextLock);
			if (!any) {
				IOLockLock(releaseLock);
				rgfxReleaseUnposted(releaseTable, static_cast<uint32_t>(at[n]));
				IOLockUnlock(releaseLock);
				more = false;
				break;
			}
			RgfxBlockRelease &release = batch[n];
			release = {};
			release.context = block.context;
			release.offset = block.offset;
			release.size = block.size;
			// A cached surface with no client left is flagged so (RGFX_BLOCK_RELEASE_NO_CLIENT),
			// which the host only counts: it copies the cache into the pages all the same, as a
			// mach port or an XPC object holds a surface with no client, and the process that
			// looks it up from one reads its pages. Its clients are read here, out of every
			// lock, as at a block's sleep.
			bool written = false;
			if (OSObject *surface = rgfxSurfaceCacheReleasingSurface(block.offset, &written)) {
				release.flags = rgfxCacheReleaseFlags(rgfxIOSurfaceClientCount(surface));
				surface->release();
			}
			IOLockLock(releaseLock);
			e.offset = block.offset;
			e.serial = block.serial;
			e.size = block.size;
			e.context = block.context;
			e.freedAt = block.freedAt;
			e.flags = release.flags;
			e.written = written;
			IOLockUnlock(releaseLock);
			seqnos[n] = &e.seqno;
			n++;
		}
		if (!n)
			break;
		// Each post writes its entry's seqno before the message is published, so an answer that
		// comes before the entry is marked posted finds it.
		UInt32 posted = 0;
		ReimsGraphics *fb = accel->copyFramebuffer();
		if (fb) {
			fb->postReleases(batch, n, seqnos, &posted);
			fb->release();
		}
		RgfxReleaseEntry unposted[kReleaseBatch];
		IOLockLock(releaseLock);
		const uint64_t now = mach_absolute_time();
		for (UInt32 i = 0; i < n; i++) {
			if (i < posted) {
				rgfxReleasePosted(releaseTable, static_cast<uint32_t>(at[i]), now);
			} else {
				unposted[i] = releaseTable.entries[at[i]];
				rgfxReleaseUnposted(releaseTable, static_cast<uint32_t>(at[i]));
			}
		}
		const UInt32 inFlight = releaseTable.inFlight;
		IOLockUnlock(releaseLock);
		IOLockLock(contextLock);
		if (inFlight > releaseStats.inFlightMost)
			releaseStats.inFlightMost = inFlight;
		IOLockUnlock(contextLock);
		// Not on the ring: kept, as an unanswered release is.
		for (UInt32 i = posted; i < n; i++)
			finishRelease(unposted[i], false);
		if (posted < n)
			more = false;
	}
	// What was answered while this run posted, and a timer for the oldest still waiting.
	retireReleases();
	IOLockLock(releaseLock);
	const uint64_t oldest = rgfxReleaseOldest(releaseTable);
	IOLockUnlock(releaseLock);
	if (oldest) {
		uint64_t timeout = 0;
		nanoseconds_to_absolutetime(static_cast<uint64_t>(kReleaseAnswerMs) * 1000000ull,
		                            &timeout);
		thread_call_enter_delayed(releaseTimer, oldest + timeout);
	}
	noteReleases();
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
	// One release worker at a time: THREAD_CALL_OPTIONS_ONCE coalesces an enter made while it
	// runs into a single re-run after it returns, so a storm of frees cannot fan it out into a
	// thread per free (each of which blocks in postLifecycle for its answer, until the 'high'
	// group's 500-thread cap panics). It runs at kernel priority, below the framebuffer work
	// loop that delivers its answers, so the one worker never starves that loop.
	thread_call_t call = thread_call_allocate_with_options(
	    releaseFreedBlocks, this, THREAD_CALL_PRIORITY_KERNEL, THREAD_CALL_OPTIONS_ONCE);
	// The releases in flight, and the timer for the oldest's answer.
	IOLock *rlock = IOLockAlloc();
	RgfxReleaseEntry *releases =
	    static_cast<RgfxReleaseEntry *>(IOMalloc(sizeof(RgfxReleaseEntry) * kMaxReleases));
	thread_call_t timer = thread_call_allocate(releaseTimerFired, nullptr);
	RgfxBlock *expected = nullptr;
	if (lock && wlock && table && rings && call && rlock && releases && timer &&
	    __atomic_compare_exchange_n(&blocks, &expected, table, false, __ATOMIC_ACQ_REL,
	                                __ATOMIC_ACQUIRE)) {
		bzero(table, sizeof(RgfxBlock) * kMaxBlocks);
		bzero(rings, sizeof(RgfxWatchedRing) * kMaxWatched);
		bzero(releases, sizeof(RgfxReleaseEntry) * kMaxReleases);
		watched = rings;
		watchLock = wlock;
		releaseCall = call;
		releaseTable = {releases, kMaxReleases, 0};
		releaseTimer = timer;
		__atomic_store_n(&releaseLock, rlock, __ATOMIC_RELEASE);
		const UInt32 features = readRegister(RGFX_REG_FEATURES_LO);
		surfacePages = (features & RGFX_FEATURE_SURFACE_PAGES) != 0;
		surfaceCache = surfacePages && (features & RGFX_FEATURE_SURFACE_CACHE) != 0;
		blocksStart = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
		blocksEnd = (regionSize - RGFX_CURSOR_SPAN) & ~(kPage - 1);
		// A reserve of no more than a quarter of the blocks' room, on a region too small for it.
		const UInt64 quarter = ((blocksEnd - blocksStart) / 4) & ~(kPage - 1);
		blockTable.reserveBytes =
		    kWindowServerReserveBytes < quarter ? kWindowServerReserveBytes : quarter;
		blockTable.reserveEntries = kWindowServerReserveEntries;
		uint64_t memory = 0;
		size_t length = sizeof(memory);
		if (sysctlbyname("hw.memsize", &memory, &length, nullptr, 0) != 0 || !memory)
			memory = kDormantMemoryUnknown;
		dormantMaxBytes = memory / 8;
		// A context's units hold twice the device's largest buffer (RgfxContextSlots.hpp).
		UInt64 maxBufferLength = 0;
		const void *info = nullptr;
		if (IOMemoryMap *map = mapDeviceInfo(sizeof(RgfxDeviceInfo), &info)) {
			const RgfxDeviceInfo *d = static_cast<const RgfxDeviceInfo *>(info);
			if (d->version == RGFX_DEVICE_INFO_VERSION)
				maxBufferLength = d->sizes[RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH];
			map->release();
		}
		contextUnits = rgfxContextUnitsFor(maxBufferLength);
		__atomic_store_n(&contextLock, lock, __ATOMIC_RELEASE);
		IOLog("RgfxAccelerator: blocks 0x%llx-0x%llx, %u contexts of %u GiB of GPU addresses "
		      "(maxBufferLength %llu MiB), surfaces %s; WindowServer's reserve: %u of %u "
		      "entries, %llu MiB\n",
		      blocksStart, blocksEnd, kRgfxContextUnits / contextUnits,
		      static_cast<UInt32>((contextUnits * RGFX_CONTEXT_VA_UNIT) >> 30),
		      maxBufferLength >> 20, surfacePages ? "in place" : "copied",
		      blockTable.reserveEntries, kMaxBlocks, blockTable.reserveBytes >> 20);
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
	if (rlock)
		IOLockFree(rlock);
	if (releases)
		IOFree(releases, sizeof(RgfxReleaseEntry) * kMaxReleases);
	if (timer)
		thread_call_free(timer);
	// Another opener is making it: wait for it to publish.
	while (expected && !__atomic_load_n(&contextLock, __ATOMIC_ACQUIRE))
		IOSleep(1);
	return __atomic_load_n(&contextLock, __ATOMIC_ACQUIRE) != nullptr;
}

// MARK: - Contexts

IOReturn RgfxAccelerator::openContext(UInt32 *context, bool oldSpan, bool windowServer,
                                     UInt32 *first, UInt32 *units)
{
	if (!startContexts())
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	const UInt32 want = oldSpan ? kRgfxContextOldSpanUnits : contextUnits;
	UInt32 at = 0;
	if (!rgfxContextSlotsTake(slots, want, windowServer, &at)) {
		const UInt32 free = slots.freeUnits;
		IOLockUnlock(contextLock);
		IOLog("RgfxAccelerator: every context slot is held (%u of %u units free, %u asked)\n",
		      free, kRgfxContextUnits, want);
		return kIOReturnNoResources;
	}
	*first = at;
	*units = want;
	*context = nextContext++;
	IOLockUnlock(contextLock);
	IOLog("RgfxAccelerator: context %u opened at unit %u (%u units)\n", *context, at, want);
	return kIOReturnSuccess;
}

RgfxAccelerator::CloseAnswer RgfxAccelerator::closeContext(UInt32 context, UInt32 first,
                                                           UInt32 units, int32_t *late)
{
	// The device ends the context's queues and drops its objects, and answers once the
	// context's command buffers have finished; the caller has closed its queues already, and
	// lets go of what the context held only once the close is answered OK.
	*late = -1;
	if (!startContexts())
		return kCloseRefused;
	RgfxContextClose close = {};
	close.context = context;
	// The close's entry is the answer's to find, its seqno written before the device sees the
	// close; with every entry taken, the close waits as any lifecycle message does, and one not
	// answered then keeps what the context held for good.
	hookAnswers(this);
	IOLockLock(releaseLock);
	const int32_t at = rgfxCloseReserve(closeTable, context, first, units);
	IOLockUnlock(releaseLock);
	IOReturn r = kIOReturnNotReady;
	uint32_t status = RGFX_STATUS_OK;
	bool inTime = false;
	if (at < 0) {
		RgfxComplete answer = {};
		UInt32 length = 0;
		r = postLifecycle(RGFX_CMD_CONTEXT_CLOSE, &close, sizeof(close), RGFX_EVT_COMPLETE,
		                  &answer, sizeof(answer), &length, nullptr, kCloseAnswerMs);
		inTime = r == kIOReturnSuccess && length >= sizeof(answer);
		status = inTime ? answer.status : 0;
	} else {
		ReimsGraphics *fb = copyFramebuffer();
		if (fb) {
			r = fb->postAnswered(RGFX_CMD_CONTEXT_CLOSE, &close, sizeof(close),
			                     &closeTable.entries[at].seqno);
			fb->release();
		}
		uint64_t deadline = 0;
		clock_interval_to_deadline(kCloseAnswerMs, kMillisecondScale, &deadline);
		IOLockLock(releaseLock);
		if (r != kIOReturnSuccess) {
			rgfxCloseUnposted(closeTable, static_cast<uint32_t>(at));
		} else {
			while (!rgfxCloseAnswered(closeTable, static_cast<uint32_t>(at))) {
				if (IOLockSleepDeadline(releaseLock, &closeTable, deadline, THREAD_UNINT) ==
				    THREAD_TIMED_OUT)
					break;
			}
			inTime = rgfxCloseInTime(closeTable, static_cast<uint32_t>(at), &status);
		}
		IOLockUnlock(releaseLock);
		if (r == kIOReturnSuccess && !inTime) {
			*late = at;
			IOLog("RgfxAccelerator: context %u: its close is not answered after %u ms: what it "
			      "held, its %u GPU address units from %u among it, stays the device's until it "
			      "is\n",
			      context, kCloseAnswerMs, units, first);
			return kCloseLate;
		}
	}
	const bool closed = inTime && status == RGFX_STATUS_OK;
	if (!closed) {
		IOLog("RgfxAccelerator: context %u: the device did not answer its close (0x%x, "
		      "status %u): what it held, its %u GPU address units from %u among it, is kept "
		      "from use for good\n",
		      context, r, status, units, first);
		return kCloseRefused;
	}
	IOLockLock(contextLock);
	rgfxContextSlotsGive(slots, first, units);
	IOLockUnlock(contextLock);
	IOLog("RgfxAccelerator: context %u closed (unit %u, %u units)\n", context, first, units);
	return kCloseAnswered;
}

void RgfxAccelerator::parkClose(const void *client, int32_t late)
{
	if (late < 0 || static_cast<uint32_t>(late) >= kMaxCloses || !contextLock)
		return;
	const void *token = rgfxCloseToken(closeTable, static_cast<uint32_t>(late));
	IOLockLock(contextLock);
	const uint32_t ids = rgfxIdChunksHand(idChunkOwners, RGFX_RESOURCE_ID_CHUNKS, client, token);
	const uint32_t held = rgfxBlockHand(blockTable, client, token);
	const uint32_t holds = rgfxSurfaceHand(surfaceTable, client, token);
	IOLockUnlock(contextLock);
	IOLockLock(releaseLock);
	const bool answered = rgfxClosePark(closeTable, static_cast<uint32_t>(late));
	const uint32_t pending = rgfxCloseLateCount(closeTable);
	IOLockUnlock(releaseLock);
	IOLog("RgfxAccelerator: context %u's late close holds %u blocks, %u holds of shared blocks "
	      "and %u resource-id chunks for its answer (%u closes late)\n",
	      closeTable.entries[late].context, held, holds, ids, pending);
	if (answered)
		thread_call_enter(releaseCall);
}

// MARK: - Resource ids

IOReturn RgfxAccelerator::takeIdChunk(const void *owner, UInt32 context, UInt64 *firstId)
{
	if (!contextLock)
		return kIOReturnNotReady;
	IOLockLock(contextLock);
	const int32_t taken = rgfxIdChunkTake(idChunkOwners, RGFX_RESOURCE_ID_CHUNKS, owner);
	const UInt32 held = rgfxIdChunksHeld(idChunkOwners, RGFX_RESOURCE_ID_CHUNKS);
	IOLockUnlock(contextLock);
	if (taken < 0) {
		IOLog("RgfxAccelerator: context %u: every resource-id chunk is held\n", context);
		return kIOReturnNoResources;
	}
	const UInt32 chunk = static_cast<UInt32>(taken);
	IOLog("RgfxAccelerator: context %u: resource-id chunk %u (%u of %u held)\n", context, chunk,
	      held, RGFX_RESOURCE_ID_CHUNKS);
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
	const UInt32 freed = rgfxIdChunksHand(idChunkOwners, RGFX_RESOURCE_ID_CHUNKS, owner, nullptr);
	IOLockUnlock(contextLock);
	if (freed)
		IOLog("RgfxAccelerator: %u resource-id chunks freed\n", freed);
}

void RgfxAccelerator::keepIdChunks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	const UInt32 kept =
	    rgfxIdChunksHand(idChunkOwners, RGFX_RESOURCE_ID_CHUNKS, owner, rgfxBlockNoOwner());
	IOLockUnlock(contextLock);
	if (kept)
		IOLog("RgfxAccelerator: %u resource-id chunks kept from use: their context's close was "
		      "refused\n",
		      kept);
}

// MARK: - Blocks

IOReturn RgfxAccelerator::allocBlock(RgfxAcceleratorUserClient *owner, UInt32 context,
                                    UInt64 size, UInt64 align, UInt64 *offset)
{
	if (!contextLock)
		return kIOReturnBadArgument;
	const bool mayReserve = owner->isWindowServer();
	IOLockLock(contextLock);
	RgfxBlockPin pin = {};
	RgfxBlockResult r =
	    rgfxBlockAlloc(blockTable, owner, context, size, align, offset, &pin, mayReserve);
	IOLockUnlock(contextLock);
	if (r == kRgfxBlockOk && mayReserve)
		noteReserve();
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
		// A pages block, the one kind of a client's own that holds wired pages, was never in
		// the window.
		const int64_t at = rgfxBlockIndex(blockTable, offset);
		if (at < 0 || !blockTable.blocks[at].wired)
			owner->unmapSpan(offset, size);
		r = rgfxBlockFree(blockTable, owner, offset, mach_absolute_time());
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
	// The device holds nothing of the owner's now: its pages blocks' pages are unwired, out of
	// the lock, each taken from its block first so no release's answer unwires it again.
	for (;;) {
		IOLockLock(contextLock);
		void *wired = rgfxBlockTakeWired(blockTable, owner);
		IOLockUnlock(contextLock);
		if (!wired)
			break;
		unwire(wired);
	}
	IOLockLock(contextLock);
	UInt32 freed = rgfxBlockFreeOwner(blockTable, owner);
	IOLockUnlock(contextLock);
	if (freed)
		IOLog("RgfxAccelerator: %u blocks freed\n", freed);
}

void RgfxAccelerator::holdBlocks(const void *owner)
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	rgfxBlockHand(blockTable, owner, owner);
	IOLockUnlock(contextLock);
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
// The most planes with a shared block at once, held or dormant, and the most bytes of one plane.
constexpr UInt32 kMaxSurfaceBlocks = kRgfxSurfaceMaxBlocks;
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

// Writes the `count` runs of `runs` at region offset `at` (page aligned) for the device, through a
// kernel mapping of their pages, write-back as the processes map the region: false where none
// could be made.
bool writeRuns(IODeviceMemory *region, UInt64 at, const RgfxPageRun *runs, UInt32 count)
{
	UInt64 bytes = sizeof(RgfxPageRun) * static_cast<UInt64>(count);
	IOMemoryMap *map = region->createMappingInTask(
	    kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache, at, alignUp(bytes, kPage));
	if (!map)
		return false;
	bcopy(runs, reinterpret_cast<void *>(map->getVirtualAddress()), bytes);
	map->release();
	return true;
}

// The last holder of entry `e` let it go, or the dormant `e` goes: its block is freed (its
// BLOCK_RELEASE names `context`, or a dormant block's last holder's), or kept for good where the
// device may still use it (`keep`, a close not answered, or a create naming it in flight), and
// the entry goes. Under contextLock; whether a release is to be posted.
//
// For an in-place block the host maps the surface's own pages, so those pages stay wired as long
// as the block is in the table, lest the guest recycle them under the host's view. `wired` is the
// last holder's wired descriptor of them (null for a copy block, or where the caller has none):
// it is attached to the block and unwired only once the block leaves the table (its release
// answered OK, releaseFreedBlocks). `*kept` is then true and the caller must not unwire `wired`.
bool letGoBlock(RgfxSurfaceEntry &e, UInt32 context, bool keep, void *wired = nullptr,
                bool *kept = nullptr)
{
	bool release = false;
	if (!keep) {
		if (!e.dormant)
			rgfxBlockSetContext(blockTable, e.offset, context);
		release = rgfxBlockFree(blockTable, &e, e.offset, mach_absolute_time()) == kRgfxBlockOk;
	}
	if (!release)
		rgfxBlockKeepAt(blockTable, e.offset);
	// The block stays in the table (freed for its release, or kept): its surface's pages stay
	// wired until it leaves. A copy block holds no such wire.
	bool ownedWire = e.inPlace && wired && rgfxBlockSetWired(blockTable, e.offset, wired);
	if (kept)
		*kept = ownedWire;
	freeRuns(e.runs, e.runCount);
	// Its surface's locks no longer wait for the writes named of it; a cached one's wait for
	// its release's answer, which copies the cache into the pages. A cached block kept instead
	// is cached no more, but the device may still map its pages in place: the block holds their
	// wiring, the last holder's, or where it gave none (a dormant block's, kept as a create in
	// flight names it), the cache's, so they never go back to macOS while it is in the table.
	void *cacheWire = nullptr;
	rgfxSurfaceLocksForget(e.cache, e.surface, e.cached && release, &cacheWire);
	if (cacheWire && !rgfxBlockHoldWire(blockTable, e.offset, cacheWire))
		unwire(cacheWire);
	rgfxSurfaceClear(surfaceTable, e);
	return release;
}

// The dormant entry `i` goes, for `why`. Under contextLock; whether a release is to be posted.
bool letGoDormant(UInt32 i, RgfxDormantCause why, bool written = false)
{
	dormantStats.letGo[why]++;
	if (why == kRgfxDormantPurged && written)
		dormantStats.purgedWritten++;
	if (why >= kRgfxDormantCount && why <= kRgfxDormantNoSpace &&
	    surfaceTable.entries[i].purgeable)
		dormantStats.volatileFirst++;
	dormantStats.changes++;
	return letGoBlock(surfaceTable.entries[i], 0, false);
}

// The dormant blocks past their caps go, the oldest first. Under contextLock; whether a release
// is to be posted.
bool trimDormant()
{
	bool release = false;
	RgfxDormantCause why = kRgfxDormantCauses;
	for (int32_t i; (i = rgfxSurfaceOverCaps(surfaceTable, dormantMaxBytes, &why)) >= 0;)
		release |= letGoDormant(static_cast<UInt32>(i), why);
	return release;
}

// The dormant blocks their surfaces asked to go (rgfxSurfaceCacheLetGoOf) go, and the others
// take whether their surfaces are volatile. Under contextLock; whether a release is to be
// posted.
bool reapDormant()
{
	bool release = false;
	for (UInt32 i = 0; surfaceTable.dormantCount && i < surfaceTable.capacity; i++) {
		RgfxSurfaceEntry &e = surfaceTable.entries[i];
		if (!e.used || !e.dormant || e.pending)
			continue;
		bool written = false, purgeable = false;
		const uint32_t why = rgfxSurfaceCacheLetGoOf(e.offset, &written, &purgeable);
		e.purgeable = purgeable;
		if (why < kRgfxDormantCauses)
			release |= letGoDormant(i, static_cast<RgfxDormantCause>(why), written);
	}
	return release;
}

// Logs what the releases took (ReleaseStats), where more were answered and kReleaseLogMs passed
// since the last line. Not under contextLock.
void noteReleases()
{
	if (!contextLock)
		return;
	const uint64_t now = mach_absolute_time();
	uint64_t interval = 0;
	nanoseconds_to_absolutetime(static_cast<uint64_t>(kReleaseLogMs) * 1000000ull, &interval);
	IOLockLock(contextLock);
	const bool log = releaseStats.latency.count != releaseStats.loggedCount &&
	                 now - releaseStats.loggedAt >= interval;
	UInt64 count = 0, p50 = 0, p99 = 0, most = 0, reaps = 0, r50 = 0, r99 = 0, rmost = 0;
	UInt64 kept = 0, late = 0, waits = 0, failed = 0;
	UInt32 inFlight = 0;
	if (log) {
		const ReleaseStats &s = releaseStats;
		count = s.latency.count;
		p50 = rgfxLatencyPercentile(s.latency, 500);
		p99 = rgfxLatencyPercentile(s.latency, 990);
		most = s.latency.maxNs;
		reaps = s.reapWait.count;
		r50 = rgfxLatencyPercentile(s.reapWait, 500);
		r99 = rgfxLatencyPercentile(s.reapWait, 990);
		rmost = s.reapWait.maxNs;
		kept = s.kept;
		late = s.late;
		inFlight = s.inFlightMost;
		waits = s.roomWaits;
		failed = s.roomWaitsFailed;
		releaseStats.loggedCount = count;
		releaseStats.loggedAt = now;
	}
	IOLockUnlock(contextLock);
	if (log)
		IOLog("RgfxAccelerator: releases: %llu answered (%llu kept, %llu late), %u in flight at "
		      "most; freed to answered p50 %llu us, p99 %llu us, max %llu us; dormant reaps %llu, "
		      "waited p50 %llu us, p99 %llu us, max %llu us; room waits %llu (%llu found none)\n",
		      count, kept, late, inFlight, p50 / 1000, p99 / 1000, most / 1000, reaps,
		      r50 / 1000, r99 / 1000, rmost / 1000, waits, failed);
}

// The most WindowServer's blocks came to at the last line of noteReserve. Under contextLock.
UInt32 reserveLoggedCount = 0;
UInt64 reserveLoggedBytes = 0;

// Logs what WindowServer's blocks came to at most (RgfxBlockTable's peaks), as that grows by
// 64 blocks or a MiB: what its reserve is sized from. Not under contextLock.
void noteReserve()
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	const RgfxBlockTable t = blockTable;
	const bool log = t.reservedPeakCount >= reserveLoggedCount + 64 ||
	                 t.reservedPeakBytes >= reserveLoggedBytes + (1ull << 20);
	if (log) {
		reserveLoggedCount = t.reservedPeakCount;
		reserveLoggedBytes = t.reservedPeakBytes;
	}
	IOLockUnlock(contextLock);
	if (!log)
		return;
	// And its queues, of the watched rings (RgfxQueueLimits.hpp's reserve).
	UInt32 queues = 0, all = 0;
	if (watchLock) {
		IOLockLock(watchLock);
		for (UInt32 i = 0; i < watchedCount; i++)
			queues += watched[i].client && watched[i].client->isWindowServer();
		all = watchedCount;
		IOLockUnlock(watchLock);
	}
	IOLog("RgfxAccelerator: WindowServer's blocks: %u (%llu KiB), at most %u (%llu KiB); "
	      "its reserve: %u entries, %llu MiB; the table holds %u; its queues: %u of %u "
	      "watched (reserve %u)\n",
	      t.reservedCount, t.reservedBytes >> 10, t.reservedPeakCount, t.reservedPeakBytes >> 10,
	      t.reserveEntries, t.reserveBytes >> 20, t.count, queues, all,
	      kRgfxWindowServerQueues);
}

// Logs what the dormant blocks did, where it changed and kDormantLogMs passed since the last
// line. Not under contextLock.
void noteDormant()
{
	if (!contextLock)
		return;
	const uint64_t now = mach_absolute_time();
	uint64_t interval = 0;
	nanoseconds_to_absolutetime(static_cast<uint64_t>(kDormantLogMs) * 1000000ull, &interval);
	// Read first, holding no other lock.
	const UInt64 noRoom = rgfxSurfaceCacheNoRoom();
	IOLockLock(contextLock);
	const bool log = (dormantStats.changes != dormantStats.loggedChanges ||
	                  noRoom != dormantStats.loggedNoRoom) &&
	                 now - dormantStats.loggedAt >= interval;
	const DormantStats d = dormantStats;
	const UInt32 count = surfaceTable.dormantCount;
	const UInt64 bytes = surfaceTable.dormantBytes;
	if (log) {
		dormantStats.loggedChanges = dormantStats.changes;
		dormantStats.loggedNoRoom = noRoom;
		dormantStats.loggedAt = now;
	}
	IOLockUnlock(contextLock);
	if (log)
		IOLog("RgfxAccelerator: dormant surface blocks: %u held (%llu of %llu MiB); %llu slept, "
		      "%llu revived; let go: %llu purged (%llu with a cache written), %llu made "
		      "empty, %llu with no client at sleep, %llu past the count, %llu past the "
		      "bytes, %llu for a table entry, %llu for the region's room (%llu of these "
		      "volatile), %llu asked otherwise; %llu released with no client left (%llu of them "
		      "written, not copied); %llu not cached: no room\n",
		      count, bytes >> 20, dormantMaxBytes >> 20, d.slept, d.revived,
		      d.letGo[kRgfxDormantPurged], d.purgedWritten, d.letGo[kRgfxDormantEmptied],
		      d.letGo[kRgfxDormantNoClients],
		      d.letGo[kRgfxDormantCount], d.letGo[kRgfxDormantBytes],
		      d.letGo[kRgfxDormantTableFull], d.letGo[kRgfxDormantNoSpace], d.volatileFirst,
		      d.letGo[kRgfxDormantStale], d.unread, d.unreadWritten, noRoom);
}

// The last holder, of context `context`, of the cached entry `e` let it go: the block sleeps
// rather than going, where its surface stays cached (rgfxSurfaceCacheSleep), the entry keeping
// its references of the surface and its device cache. Under contextLock. The entry stays
// pending until the caller has posted the hold's BLOCK_UNSHARE and called wakeSlept with
// `*surface`, the surface retained once more, so no ask's BLOCK_SHARE overtakes it.
bool sleepBlock(RgfxSurfaceEntry &e, UInt32 context, OSObject **surface)
{
	*surface = nullptr;
	bool purgeable = false;
	if (!e.cached || !e.inPlace || e.pending || !e.cache || !e.surface ||
	    !rgfxSurfaceCacheSleep(e.cache, &purgeable))
		return false;
	rgfxBlockSetContext(blockTable, e.offset, context);
	rgfxSurfaceSleep(surfaceTable, e);
	e.purgeable = purgeable;
	e.pending = true;
	*surface = static_cast<OSObject *>(e.surface);
	(*surface)->retain();
	dormantStats.slept++;
	dormantStats.changes++;
	return true;
}

// The BLOCK_UNSHARE of the block at `offset` (serial `serial`) that slept is posted: the entry
// is no longer pending, and the dormant blocks past their caps, or asked to go meanwhile, go.
// Its surface, `surface` (sleepBlock), is read for its clients once, out of the contexts' lock,
// then released: one with none left, its last gone while the block was held, its op 5 then
// letting nothing go, asks the block to go. Takes contextLock; whether a release is to be
// posted.
bool wakeSlept(UInt64 offset, UInt64 serial, OSObject *surface)
{
	if (surface && rgfxIOSurfaceClientCount(surface) == 0)
		rgfxSurfaceCacheUnclaimed(offset);
	OSSafeReleaseNULL(surface);
	IOLockLock(contextLock);
	const int32_t at = rgfxSurfaceAt(surfaceTable, offset);
	if (at >= 0 && surfaceTable.entries[at].serial == serial)
		surfaceTable.entries[at].pending = false;
	IOLockWakeup(contextLock, &surfaceTable, false);
	bool release = reapDormant();
	release |= trimDormant();
	IOLockUnlock(contextLock);
	return release;
}
} // namespace

IOReturn RgfxAccelerator::acquireSurfaceBlock(RgfxAcceleratorUserClient *client, UInt32 context,
                                             task_t task, UInt64 address, UInt64 length,
                                             UInt64 size, bool inPlace, uint32_t surfaceId,
                                             bool cacheable, SurfaceBlock *block)
{
	if (!contextLock)
		return kIOReturnNotReady;
	if (inPlace && (!surfacePages || !region))
		return kIOReturnUnsupported;
	if (!length || length > kMaxPlaneBytes || !size || size % kPage || size > 2 * kMaxPlaneBytes)
		return kIOReturnBadArgument;
	// A copy block holds the plane's bytes; a block in place holds none of the surface's whole
	// pages, which the device maps as they are.
	if (inPlace ? address % kPage || length % kPage : size < length)
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
	// A block in place is no other ask's until the device has answered for its pages: the ask
	// waits for the answer, which keeps the block or lets it go. A dormant block of the plane is
	// held again by an ask of the same surface (reviveSurfaceBlock), and goes where the ask is
	// another's, or of the other mode: the ask then makes its own.
	bool differs = false, revived = false, release = false, waitedForRoom = false;
	// Until when a new block waits for the region's room, once dormant blocks went for it.
	uint64_t roomDeadline = 0;
	int32_t at = -1, slot = -1;
	RgfxBlockPin pin = {};
	UInt64 offset = 0;
	// A block in place holds the runs past the part the processes map, whole pages of them.
	UInt64 runBytes = sizeof(RgfxPageRun) * static_cast<UInt64>(count);
	UInt64 total = inPlace ? size + alignUp(runBytes, kPage) : size;
	RgfxBlockResult made = kRgfxBlockTableFull;
	for (;;) {
		while (table.entries && rgfxSurfacePending(table, runs, count))
			IOLockSleep(contextLock, &surfaceTable, THREAD_UNINT);
		differs = false;
		at = table.entries ? rgfxSurfaceFind(table, runs, count, length, size, inPlace, &differs)
		                   : -1;
		if (at >= 0 && table.entries[at].dormant) {
			revived = reviveSurfaceBlock(table.entries[at], task, surfaceId);
			if (revived)
				break;
			release |= letGoDormant(static_cast<UInt32>(at), kRgfxDormantStale);
			continue;
		}
		const int32_t meeting =
		    at < 0 && differs ? rgfxSurfaceDormantMeeting(table, runs, count) : -1;
		if (meeting >= 0) {
			release |= letGoDormant(static_cast<UInt32>(meeting), kRgfxDormantStale);
			continue;
		}
		if (at >= 0 || differs || !table.entries)
			break;
		// A new block: a dormant one gives its entry where the table is full, and the region's
		// room where it has none, its release waited for a while, once.
		slot = rgfxSurfaceFreeEntry(table);
		if (slot < 0) {
			const int32_t oldest = rgfxSurfaceOldestDormant(table);
			if (oldest >= 0) {
				release |= letGoDormant(static_cast<UInt32>(oldest), kRgfxDormantTableFull);
				slot = rgfxSurfaceFreeEntry(table);
			}
		}
		made = slot < 0 ? kRgfxBlockTableFull
		                : rgfxBlockAlloc(blockTable, &table.entries[slot], context, total, kPage,
		                                 &offset, &pin, client->isWindowServer());
		if (made != kRgfxBlockNoSpace)
			break;
		// No room: dormant blocks go once, and the ask waits for room until its deadline,
		// whatever else wakes it meanwhile (rgfxSurfaceRoomStep).
		const RgfxSurfaceRoomStep step =
		    rgfxSurfaceRoomStep(roomDeadline, table.dormantCount, mach_absolute_time());
		if (step == kRgfxSurfaceRoomGiveUp)
			break;
		if (step == kRgfxSurfaceRoomLetGoAndWait) {
			waitedForRoom = true;
			releaseStats.roomWaits++;
			for (UInt32 n = 0; n < kDormantRoomBlocks; n++) {
				const int32_t oldest = rgfxSurfaceOldestDormant(table);
				if (oldest < 0)
					break;
				release |= letGoDormant(static_cast<UInt32>(oldest), kRgfxDormantNoSpace);
			}
			if (release)
				thread_call_enter(releaseCall);
			release = false;
			clock_interval_to_deadline(kDormantRoomMs, kMillisecondScale, &roomDeadline);
		}
		IOLockSleepDeadline(contextLock, &surfaceTable, roomDeadline, THREAD_UNINT);
	}
	if (waitedForRoom && made == kRgfxBlockNoSpace)
		releaseStats.roomWaitsFailed++;
	if (release)
		thread_call_enter(releaseCall);
	if (at >= 0) {
		// Another process's plane, or a dormant one held again: this client holds its block
		// too, in its window from its first hold, mapped under the lock so no let-go of the
		// hold comes between. A block held again is fresh to its holders.
		RgfxSurfaceEntry &e = table.entries[at];
		RgfxSurfaceHoldResult held = rgfxSurfaceHold(e, client, context, pages);
		*block = {e.offset, e.size, e.serial, revived, e.cached};
		IOReturn mapped = kIOReturnSuccess;
		bool keptWire = false;
		if (held == kRgfxSurfaceHoldNew) {
			mapped = client->mapSpan(e.offset, e.size);
			if (mapped != kIOReturnSuccess) {
				// No hold the device was not told of, nor one not in the window; a block held
				// again goes, holding none.
				void *mine = nullptr;
				UInt32 c = 0;
				rgfxSurfaceLetGo(e, client, true, &mine, &c);
				client->unmapSpan(e.offset, e.size);
				held = kRgfxSurfaceHoldFull;
				if (!e.holdCount && letGoBlock(e, context, false, mine, &keptWire))
					thread_call_enter(releaseCall);
			}
		}
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		if (held != kRgfxSurfaceHoldNew) {
			if (!keptWire)
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
			release = false;
			int32_t again = rgfxSurfaceAt(table, block->offset);
			RgfxSurfaceLetGo gone = again >= 0 ? rgfxSurfaceLetGo(table.entries[again], client,
			                                                      true, &mine, &c)
			                                   : kRgfxSurfaceLetGoNotHeld;
			if (gone != kRgfxSurfaceLetGoNotHeld)
				client->unmapSpan(block->offset, block->size);
			bool shareKeptWire = false;
			if (gone == kRgfxSurfaceLetGoLast)
				release = letGoBlock(table.entries[again], c, false, mine, &shareKeptWire);
			IOLockUnlock(contextLock);
			if (!shareKeptWire)
				unwire(mine);
			if (release)
				thread_call_enter(releaseCall);
		}
		if (revived)
			noteDormant();
		// The asker holds the surface: the processes see its clients counted anew.
		if (posted == kIOReturnSuccess && block->cached)
			rgfxSurfaceCacheNoteClients(block->offset);
		return posted;
	}
	if (made != kRgfxBlockOk) {
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		unwire(pages);
		if (differs)
			return kIOReturnBadArgument;
		return made == kRgfxBlockNoSpace ? kIOReturnNoSpace : kIOReturnNoResources;
	}
	RgfxSurfaceEntry &e = table.entries[slot];
	e.used = true;
	e.offset = offset;
	e.size = size;
	e.serial = pin.serial;
	e.inPlace = inPlace;
	e.pending = inPlace;
	e.length = length;
	e.runs = runs;
	e.runCount = count;
	rgfxSurfaceHold(e, client, context, pages);
	*block = {offset, size, pin.serial, true, false};
	// In the client's window, under the lock, so no let-go of the hold comes between: the part
	// the processes map, never the runs. A block that is not goes back as if never made: the
	// device was not told of it.
	IOReturn mapped = client->mapSpan(offset, size);
	if (mapped != kIOReturnSuccess) {
		client->unmapSpan(offset, size);
		rgfxBlockDrop(blockTable, pin);
		rgfxSurfaceClear(table, e);
		if (inPlace)
			IOLockWakeup(contextLock, &surfaceTable, false);
		IOLockUnlock(contextLock);
		freeRuns(runs, count);
		unwire(pages);
		return mapped;
	}
	IOLockUnlock(contextLock);
	// The device learns of the block before any process does; it is pinned until then, so no
	// release overtakes the share. A block in place first hands the device its surface's pages,
	// and goes as a share not posted goes where the device maps none of them: the process makes
	// a copy instead.
	// A block in place may be cached: its surface's device cache is made before the device
	// learns of the block, so its CPU generation counts from the start.
	bool cached = inPlace && surfaceCache && cacheable && surfaceId &&
	              cacheSurface(e, task, surfaceId, offset);
	block->cached = cached;
	IOReturn posted = kIOReturnSuccess;
	if (inPlace && !postSurfacePages(context, offset, total, size, length, runs, count, cached))
		posted = kIOReturnUnsupported;
	if (posted == kIOReturnSuccess) {
		RgfxBlockGrant share = {};
		share.context = context;
		share.offset = offset;
		share.size = size;
		posted = postLifecycle(RGFX_CMD_BLOCK_SHARE, &share, sizeof(share), 0, nullptr, 0,
		                       nullptr, nullptr, 0);
	}
	void *mine = nullptr;
	release = false;
	bool keptWire = false;
	IOLockLock(contextLock);
	rgfxBlockUnpin(blockTable, pin);
	if (posted != kIOReturnSuccess) {
		UInt32 c = 0;
		RgfxSurfaceLetGo gone = rgfxSurfaceLetGo(e, client, true, &mine, &c);
		if (gone != kRgfxSurfaceLetGoNotHeld)
			client->unmapSpan(offset, size);
		if (gone == kRgfxSurfaceLetGoLast)
			release = letGoBlock(e, c, false, mine, &keptWire);
	}
	if (inPlace) {
		// The askers waiting for the answer find the block, or none.
		if (e.used && e.offset == offset && e.serial == pin.serial)
			e.pending = false;
		IOLockWakeup(contextLock, &surfaceTable, false);
	}
	IOLockUnlock(contextLock);
	if (!keptWire)
		unwire(mine);
	if (release)
		thread_call_enter(releaseCall);
	if (client->isWindowServer())
		noteReserve();
	return posted;
}

void RgfxAccelerator::wakeReleases()
{
	if (!__atomic_load_n(&contextLock, __ATOMIC_ACQUIRE))
		return;
	// When the first dormant block asked to go, for the worker's reap of it (ReleaseStats).
	IOLockLock(contextLock);
	if (!releaseStats.reapAskedAt)
		releaseStats.reapAskedAt = mach_absolute_time();
	IOLockUnlock(contextLock);
	thread_call_enter(releaseCall);
}

void RgfxAccelerator::retryRefusedBlocks()
{
	if (!contextLock)
		return;
	IOLockLock(contextLock);
	UInt32 n = rgfxBlockRetryRefused(blockTable, mach_absolute_time());
	IOLockUnlock(contextLock);
	if (n) {
		thread_call_enter(releaseCall);
		IOLog("RgfxAccelerator: %u refused block release(s) posted again after a context "
		      "close\n",
		      n);
	}
}

bool RgfxAccelerator::reviveSurfaceBlock(RgfxSurfaceEntry &e, task_t task, uint32_t surfaceId)
{
	void *surface = nullptr, *cache = nullptr;
	if (!surfaceCache || !surfaceId || !e.runCount ||
	    !rgfxSurfaceLocksAttach(this, task, surfaceId, e.runs[0].address, &surface, &cache))
		return false;
	// The asker's surface is the dormant block's only where IOSurface gives it the same device
	// cache, which the block holds: no other surface has it, and no new cache takes its address
	// while it is held.
	const bool same = cache == e.cache && surface == e.surface &&
	                  rgfxSurfaceRevives(e, cache, rgfxSurfaceCacheWake(cache));
	// The attach's references go: the entry holds its own of both.
	static_cast<OSObject *>(cache)->release();
	static_cast<OSObject *>(surface)->release();
	if (!same)
		return false;
	rgfxSurfaceWake(surfaceTable, e);
	e.surfaceId = surfaceId;
	dormantStats.revived++;
	dormantStats.changes++;
	return true;
}

bool RgfxAccelerator::cacheSurface(RgfxSurfaceEntry &e, task_t task, uint32_t surfaceId,
                                   UInt64 offset)
{
	void *surface = nullptr, *cache = nullptr;
	if (!rgfxSurfaceLocksAttach(this, task, surfaceId, e.runs[0].address, &surface, &cache))
		return false;
	// The header's page, where the kext writes the surface's CPU generation for the processes.
	IOMemoryMap *header = region->createMappingInTask(
	    kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache, offset, kPage);
	if (!header || !rgfxSurfaceCacheAdd(cache, surface, offset, surfaceId, header,
	                                    kRgfxSurfaceCpuGenOffset)) {
		if (header)
			header->release();
		// Not cached: the cache's reference and the surface's go.
		static_cast<OSObject *>(cache)->release();
		static_cast<OSObject *>(surface)->release();
		return false;
	}
	// The entry is pending until the device answers its pages, so no other asker reads these.
	IOLockLock(contextLock);
	e.surface = surface;
	e.cache = cache;
	e.surfaceId = surfaceId;
	e.cached = true;
	IOLockUnlock(contextLock);
	return true;
}

bool RgfxAccelerator::downloadSurfaceCache(UInt64 offset, UInt32 timeoutMs)
{
	RgfxSurfaceDownload message = {};
	message.block_offset = offset;
	RgfxComplete answer = {};
	UInt32 answered = 0;
	IOReturn r = postLifecycle(RGFX_CMD_SURFACE_DOWNLOAD, &message, sizeof(message),
	                           RGFX_EVT_COMPLETE, &answer, sizeof(answer), &answered, nullptr,
	                           timeoutMs);
	return r == kIOReturnSuccess && answered >= sizeof(answer) && answer.status == RGFX_STATUS_OK;
}

bool RgfxAccelerator::postSurfacePages(UInt32 context, UInt64 offset, UInt64 total, UInt64 size,
                                       UInt64 length, const RgfxPageRun *runs, UInt32 count,
                                       bool cached)
{
	if (!writeRuns(region, offset + size, runs, count)) {
		IOLog("RgfxAccelerator: the block 0x%llx's runs could not be written: its surface is "
		      "copied\n",
		      offset);
		return false;
	}
	IOReturn r = postPages(context, offset, total, offset + size, length, count,
	                       cached ? RGFX_SURFACE_PAGES_CACHED : 0);
	if (r != kIOReturnSuccess && r != kIOReturnUnsupported)
		IOLog("RgfxAccelerator: the block 0x%llx's pages were not answered (0x%x): its surface "
		      "is copied\n",
		      offset, r);
	return r == kIOReturnSuccess;
}

IOReturn RgfxAccelerator::postPages(UInt32 context, UInt64 offset, UInt64 total,
                                    UInt64 runsOffset, UInt64 length, UInt32 count, UInt32 flags)
{
	// The message, then where it has flags, the part that holds them.
	struct {
		RgfxSurfacePages pages;
		RgfxSurfacePagesCache cache;
	} message = {};
	static_assert(sizeof(RgfxSurfacePages) == RGFX_SURFACE_PAGES_CACHE_OFFSET,
	              "the cache's part follows the message");
	message.pages.context = context;
	message.pages.run_count = count;
	message.pages.block_offset = offset;
	message.pages.block_size = total;
	message.pages.runs_offset = runsOffset;
	message.pages.length = length;
	message.cache.flags = flags;
	RgfxComplete answer = {};
	UInt32 answered = 0;
	IOReturn r = postLifecycle(RGFX_CMD_SURFACE_PAGES, &message,
	                           flags ? sizeof(message) : sizeof(message.pages), RGFX_EVT_COMPLETE,
	                           &answer, sizeof(answer), &answered, nullptr, kQueueAnswerMs);
	if (r == kIOReturnSuccess && answered < sizeof(answer))
		r = kIOReturnTimeout;
	if (r != kIOReturnSuccess)
		return r;
	return answer.status == RGFX_STATUS_OK ? kIOReturnSuccess : kIOReturnUnsupported;
}

IOReturn RgfxAccelerator::pagesBlock(RgfxAcceleratorUserClient *client, UInt32 context,
                                     task_t task, UInt64 address, UInt64 length, bool writable,
                                     UInt64 *offset, UInt64 *size)
{
	if (!contextLock)
		return kIOReturnNotReady;
	// A device that maps no guest pages in place, or one that did not take the feature, makes
	// no pages block: the process copies the memory instead.
	if (!surfacePages || !region || !(featureWord() & RGFX_FEATURE_PAGES_BUFFERS))
		return kIOReturnUnsupported;
	// Read-only memory is not mapped in place until such pages have a user: the GPU's writes
	// would land in pages the process cannot write.
	if (!writable)
		return kIOReturnUnsupported;
	RgfxPagesRange range = {};
	if (!rgfxPagesRange(address, length, kPage, &range))
		return kIOReturnBadArgument;
	// Wired for the block's life, which resolves copy-on-write first, so a private mapping's
	// pages are its own before the device maps them.
	IOMemoryDescriptor *md =
	    IOMemoryDescriptor::withAddressRange(range.base, range.length, kIODirectionOutIn, task);
	if (!md)
		return kIOReturnNoMemory;
	if (md->prepare(kIODirectionOutIn) != kIOReturnSuccess) {
		md->release();
		return kIOReturnBadArgument;
	}
	auto segment = [md](uint64_t at, uint64_t *bytes) -> uint64_t {
		IOByteCount n = 0;
		const addr64_t phys = md->getPhysicalSegment(at, &n, kIOMemoryMapperNone);
		*bytes = n;
		return phys;
	};
	const uint32_t count = rgfxPagesRuns(range.length, segment, nullptr, 0);
	if (!count) {
		unwire(md);
		return kIOReturnBadArgument;
	}
	// The client's block, holding the wired pages from here on, and pinned until the device has
	// its grant. It is never in the client's window: the process never sees a guest physical
	// address.
	const UInt64 bytes = rgfxPagesBlockSize(count, kPage);
	RgfxBlockPin pin = {};
	UInt64 at = 0;
	IOLockLock(contextLock);
	const RgfxBlockResult made = rgfxBlockAlloc(blockTable, client, context, bytes, kPage, &at,
	                                            &pin, client->isWindowServer());
	if (made == kRgfxBlockOk)
		rgfxBlockSetWired(blockTable, at, md);
	IOLockUnlock(contextLock);
	if (made != kRgfxBlockOk) {
		unwire(md);
		if (made == kRgfxBlockNoSpace)
			return kIOReturnNoSpace;
		return made == kRgfxBlockTableFull ? kIOReturnNoResources : kIOReturnBadArgument;
	}
	// The runs, written straight into the block through a kernel mapping of it, write-back as
	// the device reads the region.
	bool written = false;
	if (IOMemoryMap *map = region->createMappingInTask(
	        kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache, at, bytes)) {
		auto *runs = reinterpret_cast<RgfxPageRun *>(map->getVirtualAddress());
		written = rgfxPagesRuns(range.length, segment, runs, count) == count;
		map->release();
	}
	IOReturn r = written ? postPages(context, at, bytes, at, range.length, count,
	                                 RGFX_SURFACE_PAGES_PRIVATE)
	                     : kIOReturnUnsupported;
	// The device has no view of a block it never heard of, or refused: the block goes back as if
	// never made, and its pages are unwired.
	if (!written || r == kIOReturnUnsupported) {
		void *wired = nullptr;
		IOLockLock(contextLock);
		const int64_t i = rgfxBlockIndex(blockTable, at);
		if (i >= 0 && blockTable.blocks[i].serial == pin.serial) {
			wired = blockTable.blocks[i].wired;
			rgfxBlockDrop(blockTable, pin);
		}
		IOLockUnlock(contextLock);
		unwire(wired);
		return kIOReturnUnsupported;
	}
	if (r == kIOReturnSuccess) {
		RgfxBlockGrant grant = {};
		grant.context = context;
		grant.offset = at;
		grant.size = bytes;
		r = postLifecycle(RGFX_CMD_BLOCK_GRANT, &grant, sizeof(grant), 0, nullptr, 0, nullptr,
		                  nullptr, 0);
	}
	IOLockLock(contextLock);
	rgfxBlockUnpin(blockTable, pin);
	// Pages not answered, or a grant not posted: the device may have a view of the pages, so the
	// block goes through its release, which drops the view, and is unwired at its answer.
	const bool release =
	    r != kIOReturnSuccess &&
	    rgfxBlockFree(blockTable, client, at, mach_absolute_time()) == kRgfxBlockOk;
	IOLockUnlock(contextLock);
	if (release)
		thread_call_enter(releaseCall);
	if (r != kIOReturnSuccess) {
		IOLog("RgfxAccelerator: context %u's pages block 0x%llx (%u runs of 0x%llx bytes) was "
		      "not answered or granted (0x%x): it is released, and the memory copied\n",
		      context, at, count, range.length, r);
		return kIOReturnUnsupported;
	}
	*offset = at;
	*size = bytes;
	return kIOReturnSuccess;
}

IOReturn RgfxAccelerator::releaseSurfaceBlock(RgfxAcceleratorUserClient *client, UInt64 offset)
{
	if (!contextLock)
		return kIOReturnNotReady;
	RgfxSurfaceTable &table = surfaceTable;
	IOLockLock(contextLock);
	int32_t at = table.entries ? rgfxSurfaceAt(table, offset) : -1;
	// A block in place waiting for the device's answer is no process's to let go yet: its
	// maker reads its runs until then, and no process has learnt of it.
	if (at >= 0 && table.entries[at].pending)
		at = -1;
	void *pages = nullptr;
	UInt32 context = 0;
	RgfxSurfaceLetGo gone = at < 0 ? kRgfxSurfaceLetGoNotHeld
	                               : rgfxSurfaceLetGo(table.entries[at], client, false, &pages,
	                                                  &context);
	UInt64 size = at < 0 ? 0 : table.entries[at].size;
	UInt64 serial = at < 0 ? 0 : table.entries[at].serial;
	// Out of the client's window once its hold ends, before the block can be anyone else's.
	if (gone == kRgfxSurfaceLetGoHold || gone == kRgfxSurfaceLetGoLast)
		client->unmapSpan(offset, size);
	bool release = false, slept = false, keptWire = false;
	OSObject *sleeper = nullptr;
	if (gone == kRgfxSurfaceLetGoLast) {
		slept = sleepBlock(table.entries[at], context, &sleeper);
		if (!slept)
			release = letGoBlock(table.entries[at], context, false, pages, &keptWire);
	}
	IOLockUnlock(contextLock);
	if (gone == kRgfxSurfaceLetGoHold || slept) {
		// Other holders keep it, or it sleeps: the device no longer lets this context name it.
		RgfxBlockGrant unshare = {};
		unshare.context = context;
		unshare.offset = offset;
		unshare.size = size;
		postLifecycle(RGFX_CMD_BLOCK_UNSHARE, &unshare, sizeof(unshare), 0, nullptr, 0, nullptr,
		              nullptr, 0);
	}
	if (slept)
		release |= wakeSlept(offset, serial, sleeper);
	// An in-place block's pages stay wired until its release is answered (the block owns the
	// wire now); else unwire the last holder's here.
	if (!keptWire)
		unwire(pages);
	if (release)
		thread_call_enter(releaseCall);
	if (slept)
		noteDormant();
	return gone == kRgfxSurfaceLetGoNotHeld ? kIOReturnBadArgument : kIOReturnSuccess;
}

RgfxAccelerator::SurfaceCacheAnswer
RgfxAccelerator::surfaceDeviceCache(RgfxAcceleratorUserClient *client, task_t task,
                                    UInt64 offset, uint32_t surfaceId, void **cache,
                                    OSObject **surface, bool *inPlace)
{
	if (!contextLock)
		return kSurfaceCacheNotHeld;
	RgfxSurfaceTable &table = surfaceTable;
	IOLockLock(contextLock);
	int32_t at = table.entries ? rgfxSurfaceAt(table, offset) : -1;
	RgfxSurfaceEntry *e = at >= 0 ? &table.entries[at] : nullptr;
	SurfaceCacheAnswer answer = kSurfaceCacheMade;
	if (!e || !e->runCount || !rgfxSurfaceHeldBy(table, client, offset, 1))
		answer = kSurfaceCacheNotHeld;
	else if (e->pending)
		answer = kSurfaceCachePending;
	if (answer == kSurfaceCacheMade && !e->cache && !e->noCache) {
		e->noCache = !rgfxSurfaceLocksAttach(this, task, surfaceId, e->runs[0].address,
		                                     &e->surface, &e->cache);
		if (!e->noCache)
			e->surfaceId = surfaceId;
	}
	if (answer == kSurfaceCacheMade && !e->cache)
		answer = kSurfaceCacheNone;
	else if (answer == kSurfaceCacheMade && e->surfaceId != surfaceId)
		answer = kSurfaceCacheOtherSurface;
	if (answer == kSurfaceCacheMade) {
		*cache = e->cache;
		static_cast<OSObject *>(e->cache)->retain();
		*surface = static_cast<OSObject *>(e->surface);
		(*surface)->retain();
		*inPlace = e->inPlace;
	}
	IOLockUnlock(contextLock);
	return answer;
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
		// its blocks go, or sleeps where cached, or is kept where the close was not answered.
		const UInt64 offset = table.entries[i].offset, serial = table.entries[i].serial;
		bool slept = false, keptWire = false;
		OSObject *sleeper = nullptr;
		if (gone == kRgfxSurfaceLetGoLast) {
			slept = closed && sleepBlock(table.entries[i], context, &sleeper);
			if (!slept)
				release |= letGoBlock(table.entries[i], context, !closed, pages, &keptWire);
		}
		IOLockUnlock(contextLock);
		if (gone != kRgfxSurfaceLetGoNotHeld)
			dropped++;
		// The device ended the hold already: no BLOCK_UNSHARE.
		if (slept)
			release |= wakeSlept(offset, serial, sleeper);
		// An in-place block kept for its release holds the wire now; else unwire it here.
		if (!keptWire)
			unwire(pages);
	}
	if (release)
		thread_call_enter(releaseCall);
	if (dropped)
		IOLog("RgfxAccelerator: %u shared surface blocks let go at a close\n", dropped);
	noteDormant();
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
	if (!rgfxWatchAdmitted(watchedCount, client->isWindowServer())) {
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
	if (map) {
		// The queue's writes go with it: none of its buffers holds a lock any more.
		forgetQueueWrites(objId);
		map->release();
	}
}

void RgfxAccelerator::unwatchClient(RgfxAcceleratorUserClient *client)
{
	if (!watchLock)
		return;
	IOLockLock(watchLock);
	for (UInt32 i = 0; i < watchedCount;) {
		if (watched[i].client == client) {
			// Released under the lock: the scan never meets a ring whose map is gone.
			forgetQueueWrites(watched[i].objId);
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
	// A lock waiting for a surface drawn in place looks at the completions again.
	rgfxSurfaceLocksWake();
}

UInt64 RgfxAccelerator::queueCompleted(UInt64 objId)
{
	if (!watchLock)
		return 0;
	UInt64 completed = 0;
	IOLockLock(watchLock);
	for (UInt32 i = 0; i < watchedCount; i++) {
		if (watched[i].objId == objId) {
			completed = loadHeader64(watched[i].header, kRingCompletedSeqno);
			break;
		}
	}
	IOLockUnlock(watchLock);
	return completed;
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

IOReturn RgfxAccelerator::postDeferred(UInt32 context, UInt32 type, const void *message,
                                       UInt32 size, UInt64 *seqno)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return kIOReturnNotReady;
	IOReturn r = fb->postDeferred(context, type, message, size, seqno);
	fb->release();
	return r;
}

IOReturn RgfxAccelerator::takeDeferred(UInt32 context, UInt64 waitFor, UInt32 timeoutMs,
                                       UInt8 *out, UInt32 capacity, UInt32 *length,
                                       UInt32 *count)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return kIOReturnNotReady;
	IOReturn r = fb->takeDeferred(context, waitFor, timeoutMs, out, capacity, length, count);
	fb->release();
	return r;
}

void RgfxAccelerator::dropDeferred(UInt32 context)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return;
	fb->dropDeferred(context);
	fb->release();
}

void RgfxAccelerator::forgetDeferred(UInt64 seqno)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return;
	fb->forgetDeferred(seqno);
	fb->release();
}

UInt32 RgfxAccelerator::openQueue(const RgfxQueueCreate &create, UInt32 flags)
{
	struct {
		RgfxQueueCreate create;
		RgfxQueueCreateFlags tail;
	} message = {};
	static_assert(sizeof(message) == RGFX_QUEUE_CREATE_FLAGS_OFFSET + sizeof(RgfxQueueCreateFlags),
	              "the tail follows the create");
	message.create = create;
	message.tail.flags = flags;
	const UInt32 size = flags ? sizeof(message) : sizeof(create);
	RgfxComplete answer = {};
	UInt32 length = 0;
	IOReturn r = postLifecycle(RGFX_CMD_QUEUE_CREATE, &message, size, RGFX_EVT_COMPLETE, &answer,
	                           sizeof(answer), &length, nullptr, kQueueAnswerMs);
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
