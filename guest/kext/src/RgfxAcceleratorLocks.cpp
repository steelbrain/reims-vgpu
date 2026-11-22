/*
 * RgfxAccelerator's CPU locks of the IOSurfaces the GPU writes: a lock of a surface, in any
 * process, waits for the GPU's pending writes to it, as Metal's does (RgfxSurfaceLocks.hpp has
 * the rule and how a write retires).
 *
 * The kext drives IOSurface's device caches as a GPU driver does (RgfxIOSurfaceKPI.hpp). The
 * first write a process names of a surface's shared block (kRgfxAccelMethodSurfaceWrites) looks
 * the surface up by the IOSurfaceID the process gives, in the process's task, and takes it only
 * where its memory holds the block's first page: a process names only a surface it holds, and
 * only the surface of the block. The kext then makes the surface's device cache for this
 * accelerator, with its own control function, and keeps both with the block's entry, each
 * retained, until the block goes, which releases both. Each write named notes the write in the
 * lock table, then marks the cache current, which has IOSurface call the control function inside
 * every CPU lock of the surface from then on and moves the surface's seed.
 *
 * The control function waits at op 2 while a write of the surface is pending, whatever the lock's
 * mode (a pending read holds no lock), and does nothing at ops 0 and 3: once a write retires, the
 * surface's pages hold the GPU's results, the device's own for a surface drawn in place, else the
 * process's write-back. It looks at the device's completed seqno of each queue with a pending
 * write in place when the device raises its vector or a process retires writes (the bundle
 * retires a buffer that writes only surfaces in place as it takes the buffer's completion), and
 * at most every kBackstopMs where nothing wakes it. A wake that comes while it looks has it look
 * again, so it sleeps through none. A lock waits for the device's completion of the writes'
 * buffers with no deadline, as Metal's lock waits for the GPU's writes: a buffer behind a
 * pipeline's compile on the host holds the lock as long as the compile takes, and a lock that
 * went on before its buffer would read whatever bytes the pages held. Once the device has
 * completed every pending write's buffer, what is left is the bundle's retire, the process's own,
 * and a lock still waiting kRetireWaitMs after that goes on, with an "RGFX: WARN" line naming the
 * surface and the task: a process that locks a surface its own later buffer writes from the
 * thread that would retire it holds no lock for good (rgfxLockGivesUp, RgfxSurfaceLocks.hpp). A
 * lock waiting for the host's answers about a cached surface's pages (a download's, its own or
 * another lock's, an attach's, a deferral's, or its block's release's) waits with no deadline too
 * (RgfxSurfaceCaches.hpp). Every such wait ends at the device's stop, which wakes the locks
 * (rgfxSurfaceLocksStopped), and at the locking thread's abort (its process ending: the sleeps
 * are THREAD_INTERRUPTIBLE, which no caught signal breaks), the lock then going on with its
 * warning; a GPU fault on the host completes the buffers. A download it stopped waiting for stays
 * the host's: the host answers a download or a release asked after it once its copy lands.
 *
 * A surface whose block has a device cache on the host (RgfxSurfaceCaches.hpp) has its cache made
 * with the block, and the control function does more at op 2, once no write is pending: where
 * the GPU wrote the surface since the pages last took the cache, it asks the host to copy the
 * cache into the pages (SURFACE_DOWNLOAD) and waits for the answer; a read-write lock then moves
 * the surface's CPU generation, which the kext writes into the block's header; and a lock waits
 * for the answer to the block's BLOCK_RELEASE, which copies the cache into the pages a last time.
 * The kext holds such a cache until that answer, so the surface's locks go on calling the
 * control function until then.
 *
 * A deferred block's surface (RgfxSurfaceBlocks.hpp) has its cache made at its ask, before the
 * block is posted, and the ask reads the surface's seed and its pages' counts (resident, and
 * compressed or paged out: rgfxSurfaceLocksPageCounts) only once the control function is set,
 * under a read-only lockSurface, which waits out any lock calling the
 * control functions (RgfxIOSurfaceKPI.hpp): a read-write lock begun before it has moved the seed,
 * and one after it calls the control function, which waits for the host's answer to the
 * deferral, then names the block's pages before it goes on (SURFACE_PAGES with ATTACH), one lock
 * at a time and under no lock of the contexts'. Its release names them first where the GPU wrote
 * it and its surface lives on (rgfxSurfaceCacheReleaseAttach), on the naming worker, never the
 * release thread, which goes on with the other releases meanwhile (RgfxReleaseTable.hpp).
 *
 * A count of a surface's pages maps its memory into the kernel, walks the mapping's regions and
 * removes the mapping (rgfxSurfaceLocksPageCounts). Each ask that may defer a block counts
 * (rgfxSurfaceLocksFresh), out of the contexts' lock, and so does the deferral's read under the
 * surface's lock. Measured with every count's time logged (a scratch build of this code, the log
 * after the timed span), on a Radeon 780M Linux host, over a cold boot, Safari's scroll and
 * Mission Control:
 *  - 26.7.2 (25G332): 3695 counts, median 9.8 us, 99th percentile 68 us, 70.7 ms in all;
 *  - 11.7.11 (20G1443): 1516 counts, median 13.3 us, 99th percentile 65 us, 29.3 ms in all.
 * In medians, the mapping's making takes 4-8 us and its removal about 1 us, whatever the size;
 * the walk grows with it: about 1 us at 64 KiB or less, 6-8 us for Safari's 7296 KiB tiles (the
 * whole count 14-18 us), 21-27 us at 16-32 MiB, 46-53 us at 53-63 MiB. A walk over resident
 * pages is slower: 46 us for a 7296 KiB tile holding some. Every mapping lay in one of IOKit's
 * pageable maps, in one region, and every count counted; no page was compressed or paged out in
 * these runs. The longest counts, 1-9 ms, 0.2% of them, waited mostly in the mapping's making:
 * on 26.7.2 the longest, 9.1 ms, was for a 4 KiB surface. A later cold boot of 26.7.2 on a Radeon
 * 780M (Hawk Point) Linux host, its counts' times summed and their longest kept, met one of 81.8 ms
 * before the desktop settled, the longest of its 4649 counts over the boot, Safari's scroll and
 * Mission Control, 8 of which took 1 ms or more. Under the contexts' lock, which every process's
 * block allocations, surface asks and releases, and context opens and closes take, such a count
 * held all of them that long: an ask counts out of it (RgfxAcceleratorContexts.cpp,
 * acquireSurfaceBlock).
 *
 * Our own code; it names SDK-public IOKit calls and IOSurface's interface (RgfxIOSurfaceKPI.hpp).
 */

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxAnswerWait.hpp"
#include "RgfxIOSurfaceKPI.hpp"
#include "RgfxSurfaceCaches.hpp"
#include "RgfxSurfaceLocks.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOLib.h>
#include <kern/clock.h>
#include <libkern/OSAtomic.h>
#include <libkern/OSKextLib.h>
#include <libkern/libkern.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#include <sys/proc.h>

namespace {
// The most pending writes kept at once (a surface's per queue), and the most queues with any.
constexpr UInt32 kMaxUses = 4096;
constexpr UInt32 kMaxQueues = kRgfxMaxWatchedQueues;
// How long a lock waits for the bundle's retire once the device completed the writes' buffers
// before it goes on (rgfxLockGivesUp), and how long it sleeps at most between looks at the
// device's completions where no wake comes.
constexpr UInt32 kRetireWaitMs = 2000;
constexpr UInt32 kBackstopMs = 20;
// The most queues one surface's lock looks at the completions of.
constexpr UInt32 kMaxLookedAt = 16;
// The most cached surfaces at once (kRgfxCachedMaxSurfaces: every shared block in place may be
// one, held, dormant or released and not yet answered). A lock waits for a download's answer
// with no deadline, as for the host's other answers: the host copies once the buffers that used
// the cache before have finished.
constexpr UInt32 kMaxCached = kRgfxCachedMaxSurfaces;
// How many device caches made are logged, with what IOSurface answered, and how many writes
// named of surfaces the kext cannot follow.
constexpr UInt32 kCachesLogged = 4;
constexpr UInt32 kSkipsLogged = 1;
// How many surfaces left uncached for want of room are logged one by one: one a boot, as for
// every degradation (docs/conventions.md); the dormancy line counts them all.
constexpr UInt32 kNoRoomLogged = 1;

// One initialised object (it lands in __DATA,__data, as the contexts' state does), made at the
// first write named and kept for the kernel's life.
struct LockState {
	// Under `lock`: the pending writes. The waiting locks sleep on `table`. The control function
	// takes `lock` holding IOSurface's own locks, and a thread holding the contexts' lock may take
	// it: so nothing calls into IOSurface under it but IOSurface's getters, which take no lock,
	// and nothing takes the contexts' lock under it (the lock order, ContextState in
	// RgfxAcceleratorContexts.cpp).
	IOLock *lock;
	RgfxLockTable table;
	// Set once (startRoot), under no lock: IOSurfaceRoot, retained, and the accelerator ID it
	// gave this accelerator. Atomic: the device caches made, the first kCachesLogged logged.
	OSObject *root;
	uint32_t acceleratorId;
	uint32_t cachesMade;
	// Writes the table had no room for, logged at the first.
	uint32_t dropped;
	uint32_t retireWaitMs;
	// Writes named of surfaces the kext cannot follow, logged up to kSkipsLogged.
	uint32_t skipsLogged;
	// Under `lock`: the surfaces whose blocks have a device cache on the host. The waiting locks
	// sleep on `table` for these too.
	RgfxCacheTable caches;
	// Under `lock`: surfaces left uncached as the table had no room, the first kNoRoomLogged
	// logged one by one.
	uint64_t noRoom;
	// Under `lock`: the releases posted unheld (RGFX_BLOCK_RELEASE_UNHELD) whose surface a
	// process took again before their answer (rgfxCacheRetaken), counted at the answer.
	uint64_t retaken;
	// Under `lock`: the waiting locks' wakes so far (wakeLocked).
	uint64_t wakes;
	// Under `lock`: how the deferrals whose asks listed resident pages ended (RgfxCarryOutcome).
	uint64_t carries[kRgfxCarryOutcomes];
	// Set once, at the first context's open (rgfxSurfaceCacheSetGrows): where the device caches
	// every surface, `caches` grows, twice as large each time it is full (growCaches), rather than
	// leave a surface uncached; under `lock`, how many times it grew.
	bool grows;
	uint64_t grown;
	// Under `lock`: the CPU locks that slept for their block's release, and the longest wait
	// (rgfxSurfaceLocksReleaseWaits).
	RgfxReleaseLockWaits releaseWaits;
};
LockState state = {nullptr, {nullptr, 0, kMaxUses, nullptr, 0, kMaxQueues}, nullptr, 0, 0, 0,
                   kRetireWaitMs, 0, {nullptr, 0, kMaxCached}, 0, 0, 0, {}, false, 0, {0, 0}};

// Under state.lock: where the table of cached surfaces grows and is full, it doubles
// (rgfxCacheGrow). Where the memory is not there, it stays as it is, and the add finds no room,
// as before.
void growCaches()
{
	RgfxCacheTable &t = state.caches;
	if (!__atomic_load_n(&state.grows, __ATOMIC_RELAXED) || t.count < t.capacity || !t.entries)
		return;
	const uint32_t was = t.capacity, capacity = 2 * t.capacity;
	auto *grown = static_cast<RgfxCachedSurface *>(IOMalloc(sizeof(RgfxCachedSurface) * capacity));
	RgfxCachedSurface *old = rgfxCacheGrow(t, grown, capacity);
	if (old && old != grown) {
		IOFree(old, sizeof(RgfxCachedSurface) * was);
		state.grown++;
	} else if (grown) {
		IOFree(grown, sizeof(RgfxCachedSurface) * capacity);
	}
}

bool startLocks()
{
	if (__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return true;
	IOLock *lock = IOLockAlloc();
	auto *uses = static_cast<RgfxLockUse *>(IOMalloc(sizeof(RgfxLockUse) * kMaxUses));
	auto *queues = static_cast<RgfxLockQueue *>(IOMalloc(sizeof(RgfxLockQueue) * kMaxQueues));
	auto *cached =
	    static_cast<RgfxCachedSurface *>(IOMalloc(sizeof(RgfxCachedSurface) * kMaxCached));
	RgfxLockUse *expected = nullptr;
	if (lock && uses && queues && cached &&
	    __atomic_compare_exchange_n(&state.table.uses, &expected, uses, false, __ATOMIC_ACQ_REL,
	                                __ATOMIC_ACQUIRE)) {
		bzero(uses, sizeof(RgfxLockUse) * kMaxUses);
		bzero(queues, sizeof(RgfxLockQueue) * kMaxQueues);
		bzero(cached, sizeof(RgfxCachedSurface) * kMaxCached);
		state.table.queues = queues;
		state.caches.entries = cached;
		__atomic_store_n(&state.lock, lock, __ATOMIC_RELEASE);
		return true;
	}
	if (lock)
		IOLockFree(lock);
	if (uses)
		IOFree(uses, sizeof(RgfxLockUse) * kMaxUses);
	if (queues)
		IOFree(queues, sizeof(RgfxLockQueue) * kMaxQueues);
	if (cached)
		IOFree(cached, sizeof(RgfxCachedSurface) * kMaxCached);
	// Another namer is making it: wait for it to publish.
	while (expected && !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		IOSleep(1);
	return __atomic_load_n(&state.lock, __ATOMIC_ACQUIRE) != nullptr;
}

// Whether `surface`'s memory holds the physical byte `phys`.
bool surfaceHolds(OSObject *surface, UInt64 phys)
{
	IOMemoryDescriptor *md = OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	if (!md || md->prepare(kIODirectionOutIn) != kIOReturnSuccess)
		return false;
	bool found = false;
	for (IOByteCount at = 0; at < md->getLength() && !found;) {
		IOByteCount segment = 0;
		addr64_t seg = md->getPhysicalSegment(at, &segment, kIOMemoryMapperNone);
		if (!seg || !segment)
			break;
		found = phys >= seg && phys - seg < segment;
		at += segment;
	}
	md->complete(kIODirectionOutIn);
	return found;
}

void wakeLocked();

// The time `ms` milliseconds past `start` (mach_absolute_time's units).
uint64_t deadlineAfter(uint64_t start, uint64_t ms)
{
	uint64_t interval = 0;
	nanoseconds_to_absolutetime(ms * 1000000ull, &interval);
	return start + interval;
}

// The milliseconds since `start` (mach_absolute_time's units).
uint64_t msSince(uint64_t start)
{
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(mach_absolute_time() - start, &ns);
	return ns / 1000000ull;
}

// Whether `surface` is an IOSurface, as IOSurfaceRoot hands out.
bool isSurface(const OSObject *surface)
{
	const OSMetaClass *meta = surface->getMetaClass();
	return meta && strcmp(meta->getClassName(), "IOSurface") == 0;
}

// Whether a caller of startRoot is finding IOSurfaceRoot, or found it.
bool rootStarting = false;

// IOSurfaceRoot, retained, and the accelerator ID it gives `accel`, found at the first call and
// kept. Under no lock of the kext's, as the ID is IOSurface's to give (the lock order,
// ContextState in RgfxAcceleratorContexts.cpp): one caller finds them, and the others wait for
// it to publish them, or to find no root, which the next call looks for again.
bool startRoot(RgfxAccelerator *accel)
{
	if (__atomic_load_n(&state.root, __ATOMIC_ACQUIRE))
		return true;
	bool expected = false;
	if (!__atomic_compare_exchange_n(&rootStarting, &expected, true, false, __ATOMIC_ACQ_REL,
	                                 __ATOMIC_ACQUIRE)) {
		while (!__atomic_load_n(&state.root, __ATOMIC_ACQUIRE) &&
		       __atomic_load_n(&rootStarting, __ATOMIC_ACQUIRE))
			IOSleep(1);
		return __atomic_load_n(&state.root, __ATOMIC_ACQUIRE) != nullptr;
	}
	OSDictionary *matching = IOService::serviceMatching("IOSurfaceRoot");
	IOService *root = matching ? IOService::copyMatchingService(matching) : nullptr;
	OSSafeReleaseNULL(matching);
	if (!root) {
		IOLog("RgfxAccelerator: no IOSurfaceRoot: CPU locks of surfaces wait for no GPU write\n");
		__atomic_store_n(&rootStarting, false, __ATOMIC_RELEASE);
		return false;
	}
	state.acceleratorId = rgfxIOSurfaceRootAcceleratorID(root, accel);
	__atomic_store_n(&state.root, root, __ATOMIC_RELEASE);
	IOLog("RgfxAccelerator: IOSurface accelerator ID %u\n", state.acceleratorId);
	return true;
}

// Lets go of a cached surface's memory: unwired where it was wired (rgfxCacheWired), else only
// released.
void letGoMemory(void *memory, bool wired)
{
	IOMemoryDescriptor *md = static_cast<IOMemoryDescriptor *>(memory);
	if (!md)
		return;
	if (wired)
		md->complete(kIODirectionOutIn);
	md->release();
}

// Names the pages of surface `key`'s deferred block, whose attach rgfxCacheNext or
// rgfxCacheReleaseNext asked (kRgfxCacheAttaching): under state.lock, which it lets go around
// the SURFACE_PAGES, and takes again. Whether the host took them; one not taken is logged where
// `logged` (a caller's first naming of the block).
bool attachLocked(RgfxAccelerator *accel, uint64_t key, bool logged)
{
	const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
	if (c == nullptr)
		return false;
	const UInt32 context = c->context;
	const UInt64 offset = c->offset, size = c->size, length = c->length;
	IOMemoryDescriptor *memory = static_cast<IOMemoryDescriptor *>(c->wire);
	const uint32_t surfaceId = c->surfaceId;
	IOLockUnlock(state.lock);
	const RgfxPagesNaming naming =
	    memory ? accel->attachSurfacePages(context, offset, size, length, memory,
	                                       RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_ATTACH,
	                                       nullptr, nullptr)
	           : kRgfxPagesNotNamed;
	const bool ok = naming == kRgfxPagesNamed;
	IOLockLock(state.lock);
	void *map = nullptr, *wire = nullptr, *surface = nullptr, *carried = nullptr;
	bool keepWired = false;
	OSObject *cache = reinterpret_cast<OSObject *>(static_cast<uintptr_t>(rgfxCacheAttached(
	    state.caches, key, naming, &map, &wire, &surface, &keepWired, &carried)));
	// Its pages named, the host reads them from now on, not those its deferral carried.
	if (!cache && ok)
		carried = rgfxCacheTakeCarried(state.caches, key);
	wakeLocked();
	if (carried) {
		IOLockUnlock(state.lock);
		letGoMemory(carried, true);
		IOLockLock(state.lock);
	}
	if (cache) {
		// Its block went meanwhile (rgfxCacheLetGo): the let-go is this attach's, out of the
		// lock. IOSurface holds the cache for the lock calling this, and the lock's process the
		// surface.
		IOLockUnlock(state.lock);
		if (map)
			static_cast<IOMemoryMap *>(map)->release();
		cache->release();
		if (surface)
			static_cast<OSObject *>(surface)->release();
		// A block kept for good with no wire of its own: the host may map the pages through it
		// for good, so the memory stays wired, and held, as an unanswered naming's does.
		if (keepWired)
			IOLog("RgfxAccelerator: IOSurface %u's deferred block 0x%llx's pages stay wired for "
			      "good: its block is kept, and named them as it went\n",
			      surfaceId, offset);
		else
			letGoMemory(wire, rgfxPagesNamingWired(naming));
		IOLockLock(state.lock);
	}
	if (!ok && logged)
		IOLog("RgfxAccelerator: IOSurface %u's deferred block 0x%llx: its pages were not "
		      "named: named again until the wait ends\n",
		      surfaceId, offset);
	return ok;
}

// How a sleep for the host's answer ended (sleepForHost).
enum class Slept { kWoken, kStopped, kAborted };

// Sleeps on the table for the host's next answer, under state.lock, with no deadline: a wake
// ends it, and so do the device's stop, which wakes the table (rgfxSurfaceLocksStopped), and the
// thread's abort, its process ending (THREAD_INTERRUPTIBLE: a caught signal does not break it).
// Whether the device still answers is looked at first, out of state.lock (the lock order, above),
// and a wake that came meanwhile has it return at once, so it sleeps through none.
Slept sleepForHost(RgfxAccelerator *accel)
{
	const uint64_t woken = state.wakes;
	IOLockUnlock(state.lock);
	const bool up = accel->deviceAnswers();
	IOLockLock(state.lock);
	if (!up)
		return Slept::kStopped;
	if (state.wakes != woken)
		return Slept::kWoken;
	return IOLockSleep(state.lock, &state.table, THREAD_INTERRUPTIBLE) == THREAD_INTERRUPTED
	           ? Slept::kAborted
	           : Slept::kWoken;
}

// Sleeps, out of state.lock, until the next naming of a deferred block's pages after `tries`
// not taken (rgfxAttachRetryMs), or `deadline` where sooner; false, sleeping not, where
// `deadline` has passed. Under state.lock. No wake ends the sleep early: the next naming backs
// off whatever moves meanwhile, and what moves is read again at the wake. The sleeps are bounded
// by `deadline`, kRgfxAttachRetryWaitMs from the first refusal: a wait on the host's refusals,
// not on its work, each naming's answer itself waited for in full.
bool attachBackoff(uint32_t tries, uint64_t deadline)
{
	if (mach_absolute_time() >= deadline)
		return false;
	uint64_t until = 0;
	clock_interval_to_deadline(rgfxAttachRetryMs(tries), kMillisecondScale, &until);
	if (until > deadline)
		until = deadline;
	// An event no one wakes: the sleep ends at `until`.
	IOLockSleepDeadline(state.lock, &tries, until, THREAD_UNINT);
	return true;
}

static_assert(kRgfxPurgeableNonVolatile == kIOMemoryPurgeableNonVolatile, "op 7's nonvolatile");
static_assert(kRgfxPurgeableVolatile == kIOMemoryPurgeableVolatile, "op 7's volatile");
static_assert(kRgfxPurgeableEmpty == kIOMemoryPurgeableEmpty, "op 7's empty");

// The unlock seed of cached surface `c` into `seed`, where IOSurface exports it. IOSurface's
// getter loads a field and takes no lock (rgfxSurfaceLocksSeeds), so it is read under state.lock,
// inside a CPU lock's op 2 too.
bool unlockSeedOf(const RgfxCachedSurface &c, uint32_t *seed)
{
	if (c.object == nullptr || !rgfxSurfaceUnlockSeedExported())
		return false;
	*seed = rgfxIOSurfaceUnlockSeed(static_cast<const OSObject *>(c.object));
	return true;
}

// A block just cached: its refill record at generation 0 (rgfxCacheNoteFirstRefill), by the
// surface's seeds now. Under state.lock.
void noteFirstRefill(const RgfxCachedSurface &c)
{
	uint32_t unlockSeed = 0;
	const bool seeds = unlockSeedOf(c, &unlockSeed);
	const uint32_t lockSeed =
	    seeds ? rgfxIOSurfaceLockSeed(static_cast<const OSObject *>(c.object)) : 0;
	rgfxCacheNoteFirstRefill(c, seeds, lockSeed, unlockSeed);
}

// The control function every device cache the kext makes calls (RgfxIOSurfaceKPI.hpp), from
// inside a CPU lock of its surface: op 2 waits while a write of the surface is pending, op 3
// releases the reference IOSurface took for the call it ends, op 6 answers in its byte, and
// ops 5, 6 and 7 mark a dormant block of the surface: to go, or the first to go, or neither
// (RgfxSurfaceCaches.hpp).
void surfaceControl(void *accelerator, RgfxDeviceCache *cache, uint32_t op, uint64_t a,
                    uint64_t b)
{
	rgfxControlAnswer(op, b);
	if (rgfxControlReleases(op)) {
		// Under no lock of the kext's: the release may free the cache, its op 4 called from
		// inside it.
		reinterpret_cast<OSObject *>(cache)->release();
		return;
	}
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	RgfxAccelerator *accel = static_cast<RgfxAccelerator *>(accelerator);
	const uint64_t key = reinterpret_cast<uintptr_t>(cache);
	if (op == 5 || op == 6 || op == 7) {
		// The cache was purged (a client of the surface went), or the surface made empty: a
		// dormant block of it goes, on the release thread, never in this call. Made volatile,
		// or nonvolatile again, the release thread marks it. A client gone, the processes see
		// the surface's clients left (the bundle's keep): IOSurface counts the client out before
		// it calls op 5 (RgfxIOSurfaceKPI.hpp).
		IOLockLock(state.lock);
		// A dormant deferred block stays while clients are left (rgfxCachePurged).
		int32_t clients = 0;
		if (op == 5) {
			const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
			if (c != nullptr && c->object != nullptr) {
				const auto *surface = static_cast<const OSObject *>(c->object);
				// IOSurface's getter under state.lock, kept there on purpose: it loads a field and
				// takes no lock (read from IOSurface's x86_64 code in macOS 26.6's KDK), and op 5
				// calls it holding IOSurface's own lock at each client's release, which a getter
				// taking that lock could not do (the lock order, ContextState in
				// RgfxAcceleratorContexts.cpp).
				clients = rgfxIOSurfaceClientCount(surface);
				rgfxCacheNoteClients(state.caches, key, clients);
			}
		}
		const bool look = op == 5   ? rgfxCachePurged(state.caches, key, clients)
		                  : op == 6 ? rgfxCacheUnpurged(state.caches, key)
		                            : rgfxCachePurgeable(state.caches, key, a);
		IOLockUnlock(state.lock);
		if (look)
			accel->wakeReleases();
		return;
	}
	if (op != 2)
		return;
	const uint64_t start = mach_absolute_time();
	// The bound on the bundle's retire, from the device's completion of the writes' buffers
	// (rgfxLockGivesUp), in mach_absolute_time's units.
	uint64_t retireBound = 0, deviceDoneAt = 0;
	nanoseconds_to_absolutetime(state.retireWaitMs * 1000000ull, &retireBound);
	uint32_t surfaceId = 0;
	bool gaveUp = false;
	const char *waitedFor = nullptr;
	IOLockLock(state.lock);
	while (rgfxLockWaits(state.table, key)) {
		// The wakes so far: one that comes while the completions are read, out of the lock, has
		// the lock look again rather than sleep through it.
		const uint64_t woken = state.wakes;
		// The device's completions of the queues writing the surface: they retire the writes in
		// place, and once they pass every write, the bundle's retire is all that is left.
		uint64_t queues[kMaxLookedAt];
		uint32_t n = rgfxLockPendingQueues(state.table, key, false, queues, kMaxLookedAt,
		                                   &surfaceId);
		IOLockUnlock(state.lock);
		uint64_t done[kMaxLookedAt];
		for (uint32_t i = 0; i < n; i++)
			done[i] = accel->queueCompleted(queues[i]);
		const bool up = accel->deviceAnswers();
		IOLockLock(state.lock);
		for (uint32_t i = 0; i < n; i++)
			rgfxLockCompleted(state.table, queues[i], done[i]);
		if (!rgfxLockWaits(state.table, key))
			break;
		if (!up) {
			gaveUp = true;
			waitedFor = "the GPU's writes to it pending, the device stopped";
			break;
		}
		if (rgfxLockGivesUp(state.table, key, mach_absolute_time(), retireBound,
		                    &deviceDoneAt)) {
			gaveUp = true;
			waitedFor = "the GPU's writes to it done on the device, and not retired";
			break;
		}
		if (state.wakes != woken)
			continue;
		uint64_t backstop = 0;
		clock_interval_to_deadline(kBackstopMs, kMillisecondScale, &backstop);
		if (IOLockSleepDeadline(state.lock, &state.table, backstop, THREAD_INTERRUPTIBLE) ==
		    THREAD_INTERRUPTED) {
			gaveUp = true;
			waitedFor = "the GPU's writes to it pending, its thread aborted";
			break;
		}
	}
	// A cached surface: the pages take what the GPU wrote, one lock asking at a time, and a
	// read-write lock moves its CPU generation.
	uint32_t attachTries = 0;
	uint64_t attachUntil = 0, releaseSince = 0;
	if (const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key)) {
		if (!surfaceId)
			surfaceId = c->surfaceId;
	}
	while (!gaveUp) {
		uint64_t offset = 0, mark = 0;
		RgfxCacheStep step = rgfxCacheNext(state.caches, key, &offset, &mark, a == 0);
		if (step == kRgfxCacheGoOn)
			break;
		if (step == kRgfxCacheAttach) {
			// A deferred block's pages, named first: then a download, where the GPU wrote it. A
			// naming not taken is made again for kRgfxAttachRetryWaitMs from the first refusal:
			// the lock never goes on over pages that hold neither the GPU's writes nor a place
			// for the CPU's but there. One asked and not answered (kRgfxCacheAttachLate: the
			// device stopped, or the thread aborted) goes on as for pages named.
			if (attachLocked(accel, key, attachTries == 0))
				continue;
			const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
			if (c == nullptr || c->deferral != kRgfxCacheDeferred)
				continue;
			if (!attachUntil)
				attachUntil = deadlineAfter(mach_absolute_time(), kRgfxAttachRetryWaitMs);
			if (!attachBackoff(++attachTries, attachUntil)) {
				gaveUp = true;
				waitedFor = "its deferred block's pages not named";
				break;
			}
			continue;
		}
		if (step == kRgfxCacheDownload) {
			// Its answer, with no deadline: the pages hold the GPU's writes only once the host's
			// copy lands, behind the buffers that used the cache.
			IOLockUnlock(state.lock);
			const IOReturn r = accel->downloadSurfaceCache(offset, kRgfxWaitUntilAnswered);
			IOLockLock(state.lock);
			const bool ok = r == kIOReturnSuccess;
			rgfxCacheDownloaded(state.caches, key, mark, ok);
			wakeLocked();
			if (!ok) {
				waitedFor = r == kIOReturnAborted
				                ? "its device cache's copy into its pages unanswered, its thread "
				                  "aborted"
				            : r == kIOReturnUnsupported
				                ? "its device cache's copy into its pages refused"
				                : "its device cache's copy into its pages unanswered, the device "
				                  "stopped";
				break;
			}
			continue;
		}
		// Another lock's download, attach or deferral, or the block's release, with no deadline.
		rgfxReleaseWaitSleeps(state.caches, key, mach_absolute_time(), &releaseSince);
		const Slept slept = sleepForHost(accel);
		if (slept != Slept::kWoken) {
			gaveUp = true;
			const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
			const bool releasing = c != nullptr && c->releasing;
			waitedFor = slept == Slept::kAborted
			                ? (releasing ? "its block's release unanswered, its thread aborted"
			                             : "another lock's copy of its device cache, or its "
			                               "pages' naming, unanswered, its thread aborted")
			                : (releasing ? "its block's release unanswered, the device stopped"
			                             : "another lock's copy of its device cache, or its "
			                               "pages' naming, unanswered, the device stopped");
			break;
		}
	}
	rgfxReleaseWaitEnds(state.releaseWaits, releaseSince, mach_absolute_time());
	if (a == 0) {
		if (RgfxCachedSurface *c = rgfxCacheFind(state.caches, key)) {
			// The unlock seed before this lock's: the generation's refill record is owed until
			// the lock's unlock moves it (RgfxSurfaceRefill.hpp).
			uint32_t unlockSeed = 0;
			const bool seeds = unlockSeedOf(*c, &unlockSeed);
			const uint64_t gen = rgfxCacheReadWriteLock(state.caches, key, seeds, unlockSeed);
			if (c->genAt)
				__atomic_store_n(c->genAt, gen, __ATOMIC_RELEASE);
		}
	}
	IOLockUnlock(state.lock);
	// How long the lock waited, whatever it waited for: once a boot per what it waited for
	// (docs/conventions.md).
	static char wentOn;
	if (waitedFor && rgfxFirstCalls(&wentOn, rgfxKeyOf(waitedFor), 1)) {
		const uint64_t waitedMs = msSince(start);
		char name[64] = {};
		proc_selfname(name, sizeof(name));
		IOLog("RGFX: WARN: a CPU lock of IOSurface %u by %s[%d] goes on after %llu ms with %s\n",
		      surfaceId, name, proc_selfpid(), waitedMs, waitedFor);
	}
}

// Wakes the waiting locks, and counts the wake for a lock looking meanwhile. Under state.lock.
void wakeLocked()
{
	state.wakes++;
	IOLockWakeup(state.lock, &state.table, false);
}
} // namespace

bool rgfxSurfaceLocksAttach(RgfxAccelerator *accel, task_t task, uint32_t surfaceId, UInt64 phys,
                            void **surfaceOut, void **cacheOut)
{
	if (!startRoot(accel))
		return false;
	OSObject *surface = rgfxIOSurfaceRootLookupSurface(state.root, surfaceId, task);
	// Each refusal below is logged once a boot (docs/conventions.md).
	static char refused;
	if (!surface) {
		if (rgfxFirstCalls(&refused, 0, 1))
			IOLog("RgfxAccelerator: IOSurface %u is not the naming process's: its locks wait for "
			      "no GPU write\n",
			      surfaceId);
		return false;
	}
	if (!isSurface(surface) || !surfaceHolds(surface, phys)) {
		if (rgfxFirstCalls(&refused, 1, 1))
			IOLog("RgfxAccelerator: IOSurface %u is not its shared block's surface: its locks "
			      "wait for no GPU write\n",
			      surfaceId);
		surface->release();
		return false;
	}
	RgfxDeviceCache *cache = rgfxIOSurfaceDeviceCache(surface, accel, state.acceleratorId, 0);
	if (!cache) {
		if (rgfxFirstCalls(&refused, 2, 1))
			IOLog("RgfxAccelerator: IOSurface %u gave no device cache: its locks wait for no "
			      "GPU write\n",
			      surfaceId);
		surface->release();
		return false;
	}
	rgfxDeviceCacheSetControl(cache, accel, &surfaceControl);
	if (__atomic_fetch_add(&state.cachesMade, 1, __ATOMIC_RELAXED) < kCachesLogged)
		IOLog("RgfxAccelerator: IOSurface %u's device cache %p, retain count %d\n", surfaceId,
		      cache, reinterpret_cast<OSObject *>(cache)->getRetainCount());
	*surfaceOut = surface;
	*cacheOut = cache;
	return true;
}

OSObject *rgfxSurfaceLocksLookup(RgfxAccelerator *accel, task_t task, uint32_t surfaceId)
{
	if (!startRoot(accel))
		return nullptr;
	OSObject *surface = rgfxIOSurfaceRootLookupSurface(state.root, surfaceId, task);
	if (surface && !isSurface(surface))
		OSSafeReleaseNULL(surface);
	return surface;
}

bool rgfxSurfaceUnlockSeedExported()
{
	return &rgfxIOSurfaceUnlockSeed != nullptr && OSKextSymbolIsResolved(rgfxIOSurfaceUnlockSeed);
}

IOReturn rgfxSurfaceLocksSeeds(RgfxAccelerator *accel, task_t task, uint32_t surfaceId,
                               uint32_t *lockSeed, uint32_t *unlockSeed)
{
	if (!rgfxSurfaceUnlockSeedExported())
		return kIOReturnUnsupported;
	OSObject *surface = rgfxSurfaceLocksLookup(accel, task, surfaceId);
	if (!surface)
		return kIOReturnNotFound;
	// IOSurface's getters load a field each and take no lock (read from its x86_64 code in macOS
	// 26.7.1's KDK): a lock or an unlock between the two loads leaves them a step apart, which
	// the next ask reads again.
	*lockSeed = rgfxIOSurfaceLockSeed(surface);
	*unlockSeed = rgfxIOSurfaceUnlockSeed(surface);
	surface->release();
	return kIOReturnSuccess;
}

bool rgfxSurfaceLocksMapsAt(OSObject *surface, task_t task, UInt64 address, UInt64 length)
{
	IOMemoryDescriptor *md = OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	if (!md || !length || length > md->getLength())
		return false;
	// IOSurfaceClient::mapInTask maps the surface's memory descriptor whole into the client's
	// task, anywhere, read-only or not (read from IOSurface 393.5.8's code). kIOMapReference only
	// finds a mapping that exists, never makes one (IOMemoryDescriptor.h), and with no
	// kIOMapAnywhere only one at `address`, of the same read-only-ness, holding the descriptor's
	// first `length` bytes (IOMemoryMap::copyCompatible, read from xnu-12377.121.6): the
	// reference found is released at once, which unmaps nothing.
	// A surface IOSurface maps otherwise (one made over its client's own memory) is found by
	// none: its ask wires its range.
	const IOOptionBits modes[] = {0, kIOMapReadOnly};
	for (IOOptionBits mode : modes) {
		if (IOMemoryMap *map = md->createMappingInTask(task, address, kIOMapReference | mode, 0,
		                                               length)) {
			map->release();
			return true;
		}
	}
	return false;
}

bool rgfxSurfaceLocksStartsAt(OSObject *surface, const RgfxPageRun *runs, UInt32 count,
                              UInt64 length)
{
	IOMemoryDescriptor *md = OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	if (!md || length > md->getLength() || md->prepare(kIODirectionOutIn) != kIOReturnSuccess)
		return false;
	RgfxRunsWalk walk = rgfxSurfaceRunsWalk(runs, count);
	for (IOByteCount at = 0; walk.same && at < length;) {
		IOByteCount segment = 0;
		const addr64_t phys = md->getPhysicalSegment(at, &segment, kIOMemoryMapperNone);
		if (!phys || !segment) {
			walk.same = false;
			break;
		}
		if (segment > length - at)
			segment = length - at;
		rgfxSurfaceRunsNext(walk, phys, segment);
		at += segment;
	}
	md->complete(kIODirectionOutIn);
	return rgfxSurfaceRunsSame(walk);
}

// The kernel's map, exported in com.apple.kpi.unsupported, and IOKit's walk of its pageable maps,
// exported in com.apple.kpi.iokit, which no SDK header declares.
extern "C" vm_map_t kernel_map;
using RgfxPageableMapsCallback = kern_return_t (*)(vm_map_t map, void *ref);
extern "C" kern_return_t IOIteratePageableMaps(vm_size_t size, RgfxPageableMapsCallback callback,
                                               void *ref);

namespace {
// The most regions of a surface's mapping a count walks: more, and it is not counted.
constexpr uint32_t kMaxCountRegions = 8;
// Surfaces of this many bytes or more have each count's time logged, the first few of them.
constexpr uint64_t kCountLoggedBytes = 16ull << 20;
constexpr uint32_t kCountsLogged = 16;
} // namespace

// The regions of `map` from `start` on, as many as `room`, up to `start + length`: how many it
// found. A failed call ends the walk short of the end, which rgfxRegionPageCounts refuses.
static uint32_t walkRegions(vm_map_t map, uint64_t start, uint64_t length, RgfxPageRegion *regions,
                            uint32_t room)
{
	uint32_t found = 0;
	mach_vm_address_t at = start;
	while (found < room && at < start + length) {
		mach_vm_address_t address = at;
		mach_vm_size_t size = 0;
		vm_region_extended_info_data_t info = {};
		mach_msg_type_number_t infoCount = VM_REGION_EXTENDED_INFO_COUNT;
		mach_port_t object = MACH_PORT_NULL;
		if (mach_vm_region(map, &address, &size, VM_REGION_EXTENDED_INFO,
		                   reinterpret_cast<vm_region_info_t>(&info), &infoCount,
		                   &object) != KERN_SUCCESS)
			break;
		regions[found++] = {address, size, info.pages_resident, info.pages_swapped_out,
		                    info.external_pager != 0, info.share_mode == SM_EMPTY};
		if (address + size <= address)
			break;
		at = address + size;
	}
	return found;
}

namespace {
// The pageable map that holds an address, as IOIteratePageableMaps' callback finds it.
struct PageableMapFind {
	uint64_t address;
	vm_map_t map;
};
} // namespace

// IOIteratePageableMaps' callback: success, which ends the walk, for the map with a region that
// holds the address; failure otherwise, which neither ends the walk nor, at its end, has IOKit
// make a map (it makes one only where the last answer is KERN_NO_SPACE).
static kern_return_t pageableMapHolds(vm_map_t map, void *ref)
{
	PageableMapFind *find = static_cast<PageableMapFind *>(ref);
	mach_vm_address_t address = find->address;
	mach_vm_size_t size = 0;
	vm_region_basic_info_data_64_t info = {};
	mach_msg_type_number_t infoCount = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t object = MACH_PORT_NULL;
	if (map && mach_vm_region(map, &address, &size, VM_REGION_BASIC_INFO_64,
	                          reinterpret_cast<vm_region_info_t>(&info), &infoCount,
	                          &object) == KERN_SUCCESS &&
	    address <= find->address && find->address - address < size) {
		find->map = map;
		return KERN_SUCCESS;
	}
	return KERN_FAILURE;
}

// What a count of a surface's pages took, logged where it is the longest yet by a quarter or
// more, or where the surface is large, for the first kCountsLogged of those.
static void noteCountTime(uint64_t bytes, uint64_t ns, const RgfxSurfacePageCounts &counts)
{
	static uint64_t longest = 0;
	static uint32_t largeLogged = 0;
	uint64_t was = __atomic_load_n(&longest, __ATOMIC_RELAXED);
	bool longer = false;
	while (ns > was + was / 4 && !longer)
		longer = __atomic_compare_exchange_n(&longest, &was, ns, false, __ATOMIC_RELAXED,
		                                     __ATOMIC_RELAXED);
	const bool large = bytes >= kCountLoggedBytes &&
	                   __atomic_fetch_add(&largeLogged, 1, __ATOMIC_RELAXED) < kCountsLogged;
	if (longer || large)
		IOLog("RgfxAccelerator: counting an IOSurface's %llu KiB of pages took %llu us%s: %s, "
		      "%llu resident, %llu compressed or paged out\n",
		      bytes >> 10, ns / 1000, longer ? ", the longest yet" : "",
		      counts.counted ? "counted" : "not counted", counts.counted ? counts.resident : 0,
		      counts.counted ? counts.swapped : 0);
}

// The counts come from the regions of a mapping of the surface's memory in the kernel's own map,
// made for the count (read-only, unique, faulting nothing in) and gone after it: mach_vm_region's
// VM_REGION_EXTENDED_INFO walks the memory's object page by page and counts each page resident,
// or compressed or paged out (pages_swapped_out), measured on 11.7.11 (20G1443) for an IOSurface
// whose pages a child's allocations pushed into the compressor and on to swap: its counts equal
// each page's disposition, resident, compressed or mixed. mach_vm_region and kernel_map are in
// com.apple.kpi.unsupported, kernel_task in com.apple.kpi.mach and IOIteratePageableMaps in
// com.apple.kpi.iokit, in the symbol sets of every release kernel checked (x86_64 11.7.10 to
// 26.7.2, arm64 13 to 26; each kernel's __LINKINFO,__symbolsets): strong imports, bound by every
// linker, OpenCore's included. A pageable buffer's mapping lies inside one of IOKit's pageable
// maps, where the walk finds its regions (walkRegions, pageableMapHolds). The
// walk costs about a microsecond per 1-16 pages (measured from user space on 11.7.11: 0.13 ms
// for 16384 resident pages, 1.3 ms for the same compressed), plus the mapping's making and
// removal; what whole counts take in the kernel is at the top of this file.
RgfxSurfacePageCounts rgfxSurfaceLocksPageCounts(OSObject *surface)
{
	RgfxSurfacePageCounts counts = {false, ~0ull, ~0ull};
	static bool memoryNoted = false, mapNoted = false;
	// An IOBufferMemoryDescriptor allocates its memory itself, pageable or wired, from the
	// kernel's anonymous allocators, never over a file (xnu's IOBufferMemoryDescriptor.cpp; not
	// re-read for this change): a page of it neither resident nor compressed nor paged out was
	// never written. A descriptor over other memory, a client's own, may be a file's, whose
	// pages not resident hold the file's bytes, which no count shows: it is not counted. The
	// walk says so too (external_pager), and a region of a pager's is not counted either.
	IOMemoryDescriptor *memory = OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	if (!memory)
		return counts;
	if (!OSDynamicCast(IOBufferMemoryDescriptor, memory)) {
		if (!__atomic_exchange_n(&memoryNoted, true, __ATOMIC_RELAXED))
			IOLog("RgfxAccelerator: an IOSurface's memory is a %s, not a buffer the kernel "
			      "allocated: its block, and any such, is not deferred\n",
			      memory->getMetaClass()->getClassName());
		return counts;
	}
	const uint64_t began = mach_absolute_time();
	IOMemoryMap *map =
	    memory->createMappingInTask(kernel_task, 0, kIOMapAnywhere | kIOMapReadOnly | kIOMapUnique);
	if (!map) {
		if (!__atomic_exchange_n(&mapNoted, true, __ATOMIC_RELAXED))
			IOLog("RgfxAccelerator: an IOSurface's memory could not be mapped to count its "
			      "pages: its block, and any such, is not deferred\n");
		return counts;
	}
	const uint64_t start = map->getAddress(), length = map->getLength();
	RgfxPageRegion regions[kMaxCountRegions] = {};
	uint32_t found = walkRegions(kernel_map, start, length, regions, kMaxCountRegions);
	// IOKit maps a pageable buffer into the kernel inside one of its pageable maps, a submap of
	// the kernel's, whose one region in the kernel's map has nothing to count (SM_EMPTY) and
	// reaches past the mapping: measured on 11.7.11 (20G1443), a 512 MiB region in the kernel's
	// map, and in the pageable map one region that is the mapping exactly. The walk then goes on
	// in that pageable map, found by IOKit's own walk of them.
	if (found == 1 && regions[0].empty && regions[0].start <= start &&
	    start + length - regions[0].start <= regions[0].size) {
		PageableMapFind find = {start, nullptr};
		if (IOIteratePageableMaps(PAGE_SIZE, &pageableMapHolds, &find) == KERN_SUCCESS &&
		    find.map)
			found = walkRegions(find.map, start, length, regions, kMaxCountRegions);
	}
	counts = rgfxRegionPageCounts(regions, found, start, length);
	map->release();
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(mach_absolute_time() - began, &ns);
	noteCountTime(length, ns, counts);
	return counts;
}

bool rgfxSurfaceLocksFresh(OSObject *surface, UInt64 length, uint64_t listed)
{
	// Called out of the contexts' lock (acquireSurfaceBlock), whose holder would hold every
	// process's block allocations and surface asks for the count, which took up to 81.8 ms
	// (above). It takes no lock of the kext's: IOSurface's getters here load a field and
	// take no lock (read from IOSurface's x86_64 code in macOS 26.6's KDK), and the count maps and
	// walks the surface's memory through IOKit and the VM, never IOSurface.
	const uint32_t seed = rgfxIOSurfaceLockSeed(surface);
	IOMemoryDescriptor *md = OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	return md && length <= md->getLength() &&
	       rgfxDeferRead(true, seed, rgfxSurfaceLocksPageCounts(surface), listed) !=
	           kRgfxDeferReadStale;
}

namespace {
// The device's page, of which SURFACE_PAGES' runs are whole ones: a kernel page is whole ones too.
constexpr UInt64 kPage = 0x1000;
// A kernel's page, 4 KiB on x86-64 and 16 KiB on arm64 (PAGE_SIZE, which an arm64 kext reads at
// run time), is whole device pages, PAGE_MAX_SIZE at most.
static_assert(PAGE_MIN_SIZE % kPage == 0, "a kernel page is whole device pages");

// A wire of the `resident` pages the ask listed of the surface `task` maps at `address`, made
// but not prepared: one descriptor over their runs, in the surface's order. Null where it is not
// made.
IOMemoryDescriptor *carryWire(task_t task, UInt64 address, const RgfxSurfaceResident &resident)
{
	IOAddressRange ranges[kRgfxCarryPagesMost] = {};
	for (uint32_t i = 0; i < resident.count; i++)
		ranges[i] = {address + resident.runs[i].offset, resident.runs[i].length};
	return IOMemoryDescriptor::withAddressRanges(ranges, resident.count, kIODirectionOutIn, task);
}

// The guest physical address of each page of `wire`, prepared, in its order, `pages` of them:
// false where one is not a page's.
bool carryAddresses(IOMemoryDescriptor *wire, uint64_t pages, uint64_t *addresses)
{
	for (uint64_t p = 0; p < pages; p++) {
		IOByteCount segment = 0;
		addresses[p] = wire->getPhysicalSegment(p * kPage, &segment, kIOMemoryMapperNone);
		if (!addresses[p] || addresses[p] % kPage || segment < kPage)
			return false;
	}
	return true;
}
} // namespace

RgfxDeferAnswer rgfxSurfaceLocksDefer(RgfxAccelerator *accel, OSObject *surface,
                                      uint32_t surfaceId, UInt64 offset, UInt64 size,
                                      UInt64 length, UInt32 context, IOMemoryMap *header,
                                      task_t task, UInt64 address,
                                      const RgfxSurfaceResident *resident, RgfxPageRun *runs,
                                      UInt32 room, UInt32 *runCount, void **cacheOut)
{
	*cacheOut = nullptr;
	*runCount = 0;
	IOMemoryDescriptor *memory =
	    OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface));
	if (!memory || !startLocks())
		return kRgfxDeferNoCache;
	RgfxDeviceCache *cache = rgfxIOSurfaceDeviceCache(surface, accel, state.acceleratorId, 0);
	static char notDeferred;
	if (!cache) {
		if (rgfxFirstCalls(&notDeferred, 0, 1))
			IOLog("RgfxAccelerator: IOSurface %u gave no device cache: its block is not "
			      "deferred, nor cached\n",
			      surfaceId);
		return kRgfxDeferNoCache;
	}
	// The surface's memory, retained, not wired: its pages are named only where a lock needs them.
	memory->retain();
	const uint64_t key = reinterpret_cast<uintptr_t>(cache);
	IOLockLock(state.lock);
	// As rgfxSurfaceCacheAdd: the surface's previous block's attach in flight is answered first,
	// with no deadline, and its release unanswered was the ask's to wait for.
	for (const RgfxCachedSurface *previous;
	     (previous = rgfxCacheFind(state.caches, key)) != nullptr && !previous->releasing &&
	     sleepForHost(accel) == Slept::kWoken;)
		;
	const bool previous = rgfxCacheFind(state.caches, key) != nullptr;
	if (!previous)
		growCaches();
	RgfxCachedSurface *c = previous ? nullptr : rgfxCacheAdd(state.caches, key, offset, surfaceId);
	if (c) {
		c->genAt = reinterpret_cast<volatile uint64_t *>(header->getVirtualAddress() +
		                                                 kRgfxSurfaceCpuGenOffset);
		__atomic_store_n(c->genAt, 0, __ATOMIC_RELEASE);
		c->map = header;
		c->wire = memory;
		c->clientsAt = reinterpret_cast<volatile uint32_t *>(header->getVirtualAddress() +
		                                                     kRgfxSurfaceClientsOffset);
		c->refillAt = reinterpret_cast<volatile uint64_t *>(header->getVirtualAddress() +
		                                                    kRgfxSurfaceRefillOffset);
		c->object = surface;
		noteFirstRefill(*c);
		c->deferral = kRgfxCacheGating;
		c->context = context;
		c->size = size;
		c->length = length;
		// IOSurface's getter, as in op 5, under state.lock, kept there on purpose: it loads a field
		// and takes no lock (read from IOSurface's x86_64 code in macOS 26.6's KDK), and op 5 calls
		// it holding IOSurface's own lock at each client's release, which a getter taking that lock
		// could not do (the lock order, ContextState in RgfxAcceleratorContexts.cpp).
		rgfxCacheNoteClients(state.caches, key, rgfxIOSurfaceClientCount(surface));
	} else if (!previous) {
		state.noRoom++;
	}
	IOLockUnlock(state.lock);
	if (!c) {
		if (rgfxFirstCalls(&notDeferred, previous ? 1 : 2, 1))
			IOLog("RgfxAccelerator: IOSurface %u: %s: its block is not deferred, nor cached\n",
			      surfaceId,
			      previous ? "its previous block is cached still, its release unanswered"
			               : "no room for its device cache's state");
		memory->release();
		reinterpret_cast<OSObject *>(cache)->release();
		return kRgfxDeferNoCache;
	}
	// The pages the ask listed resident, the kernel's, as rgfxSurfaceLocksPageCounts counts them,
	// where it listed any of the surface the asker maps where it asks: their wire, made now, is
	// prepared only under the read.
	const uint64_t listed = resident ? rgfxResidentPages(*resident, length, PAGE_SIZE) : 0;
	IOMemoryDescriptor *carry = listed && rgfxSurfaceLocksMapsAt(surface, task, address, length)
	                                ? carryWire(task, address, *resident)
	                                : nullptr;
	// The control function first, then the read: a read-write lock either moved the seed before
	// the read-only lock below returned, or calls the control function. A surface fresh but for
	// the listed pages carries them where their wire, with no fault, proves them its resident
	// pages (RgfxDeferCarry.hpp).
	rgfxDeviceCacheSetControl(cache, accel, &surfaceControl);
	uint32_t seed = ~0u;
	RgfxSurfacePageCounts counts = {false, ~0ull, ~0ull}, after = counts;
	bool wired = false;
	const IOReturn locked = rgfxIOSurfaceLock(surface, kRgfxIOSurfaceLockReadOnly, &seed);
	if (locked == kIOReturnSuccess) {
		counts = rgfxSurfaceLocksPageCounts(surface);
		if (rgfxDeferRead(true, seed, counts, carry ? listed : 0) == kRgfxDeferReadCarry) {
			wired = carry->prepare(kIODirectionOutIn | kIODirectionPrepareNoFault) ==
			        kIOReturnSuccess;
			if (wired)
				after = rgfxSurfaceLocksPageCounts(surface);
		}
		rgfxIOSurfaceUnlock(surface, kRgfxIOSurfaceLockReadOnly, nullptr);
	}
	// The deferral's runs: each device page of the listed pages at its guest address, the rest
	// zeros.
	if (rgfxDeferCarried(wired, seed, after, listed)) {
		uint64_t addresses[kRgfxCarryPagesMost * (PAGE_MAX_SIZE / kPage)] = {};
		if (carryAddresses(carry, listed * (PAGE_SIZE / kPage), addresses))
			*runCount = rgfxCarryRuns(*resident, addresses, length, kPage, runs, room);
	}
	const bool isLocked = locked == kIOReturnSuccess;
	const RgfxCarryOutcome outcome =
	    rgfxCarryOutcome(isLocked, seed, counts, carry ? listed : 0, wired, after, *runCount);
	const bool fresh = outcome == kRgfxCarryTaken || outcome == kRgfxCarryNone;
	IOLockLock(state.lock);
	if (carry)
		state.carries[outcome]++;
	if (fresh) {
		rgfxCacheGated(state.caches, key);
		RgfxCachedSurface *gated = rgfxCacheFind(state.caches, key);
		if (gated != nullptr && outcome == kRgfxCarryTaken)
			gated->carried = carry;
	}
	IOLockUnlock(state.lock);
	if (outcome != kRgfxCarryTaken) {
		letGoMemory(carry, wired);
		*runCount = 0;
	}
	*cacheOut = cache;
	return fresh ? kRgfxDeferFresh : kRgfxDeferStale;
}

void rgfxSurfaceLocksRefusedCarry(void *cache)
{
	if (!cache || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	void *carried = rgfxCacheRefusedCarried(state.caches, reinterpret_cast<uintptr_t>(cache));
	IOLockUnlock(state.lock);
	letGoMemory(carried, true);
}

void rgfxSurfaceCacheSettle(void *cache, bool deferred)
{
	if (!cache || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	rgfxCacheSettled(state.caches, reinterpret_cast<uintptr_t>(cache), deferred);
	wakeLocked();
	IOLockUnlock(state.lock);
}

void rgfxSurfaceLocksUngate(void *cache)
{
	if (!cache || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	void *map = nullptr, *wire = nullptr, *carried = nullptr;
	IOLockLock(state.lock);
	const bool gone = rgfxCacheUngated(state.caches, reinterpret_cast<uintptr_t>(cache), &map,
	                                   &wire, &carried);
	wakeLocked();
	IOLockUnlock(state.lock);
	if (!gone)
		return;
	letGoMemory(carried, true);
	// The entry's references, as rgfxSurfaceLocksDefer took them: its header's mapping, the
	// surface's memory retained and never prepared for it, and the cache.
	if (map)
		static_cast<IOMemoryMap *>(map)->release();
	if (wire)
		static_cast<IOMemoryDescriptor *>(wire)->release();
	reinterpret_cast<OSObject *>(cache)->release();
}

RgfxReleaseAttach rgfxSurfaceCacheReleaseAttach(RgfxAccelerator *accel, UInt64 offset,
                                                OSObject *surface, bool name)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return kRgfxReleaseAsIs;
	// Read out of every lock, as at a block's sleep.
	const int32_t clients = rgfxIOSurfaceClientCount(surface);
	const int32_t retains = surface->getRetainCount();
	RgfxReleaseAttach did = kRgfxReleaseAsIs;
	if (!name) {
		IOLockLock(state.lock);
		const RgfxCacheReleaseStep step =
		    rgfxCacheReleasePeek(state.caches, offset, clients, retains);
		const RgfxCachedSurface *c = rgfxCacheAtBlock(state.caches, offset);
		if (step != kRgfxCacheReleaseAsIs)
			did = kRgfxReleaseNameFirst;
		else if (c != nullptr && c->deferral == kRgfxCacheDeferred)
			did = kRgfxReleaseUnattached;
		IOLockUnlock(state.lock);
		return did;
	}
	// A naming not taken is made again, as a lock's is, for kRgfxAttachRetryWaitMs from the first
	// refusal: past it the release goes with the pages unnamed, what the GPU wrote not copied into
	// them. Each naming waits for its answer with no deadline, on the naming worker.
	uint64_t retryUntil = 0;
	uint32_t tries = 0;
	IOLockLock(state.lock);
	for (;;) {
		const RgfxCacheReleaseStep step =
		    rgfxCacheReleaseNext(state.caches, offset, clients, retains);
		if (step == kRgfxCacheReleaseAttach) {
			const uint64_t key = rgfxCacheAtBlock(state.caches, offset)->key;
			const bool named = attachLocked(accel, key, tries == 0);
			// Named, or asked and not answered (kRgfxCacheAttachLate): the host may map them.
			const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
			const bool hostPages = c != nullptr && rgfxCacheHostPages(*c);
			if (!named && !hostPages && c != nullptr && !retryUntil)
				retryUntil = deadlineAfter(mach_absolute_time(), kRgfxAttachRetryWaitMs);
			if (!named && !hostPages && c != nullptr && attachBackoff(++tries, retryUntil))
				continue;
			did = hostPages ? kRgfxReleaseAttached : kRgfxReleaseUnattached;
			static char unnamed;
			if (!named && !hostPages && rgfxFirstCalls(&unnamed, 0, 1))
				IOLog("RGFX: WARN: the release of IOSurface %u's deferred block 0x%llx goes on "
				      "after %u ms with its pages not named: what the GPU wrote is not copied "
				      "into them\n",
				      c != nullptr ? c->surfaceId : 0, offset, kRgfxAttachRetryWaitMs);
			break;
		}
		// A lock's attach in flight ends at its SURFACE_PAGES's answer, or at the device's stop
		// or its thread's abort, and none starts once the block is releasing (rgfxCacheNext): the
		// release goes after it, never over it, under no lock but this one, which the sleep lets
		// go.
		if (step == kRgfxCacheReleaseWait) {
			IOLockSleep(state.lock, &state.table, THREAD_UNINT);
			continue;
		}
		const RgfxCachedSurface *c = rgfxCacheAtBlock(state.caches, offset);
		if (c != nullptr && c->deferral == kRgfxCacheDeferred)
			did = kRgfxReleaseUnattached;
		break;
	}
	IOLockUnlock(state.lock);
	return did;
}

IOMemoryDescriptor *rgfxSurfaceCacheHostPages(void *cache, UInt64 offset)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return nullptr;
	IOLockLock(state.lock);
	const RgfxCachedSurface *c = cache ? rgfxCacheFind(state.caches, reinterpret_cast<uintptr_t>(cache))
	                                   : rgfxCacheAtBlock(state.caches, offset);
	IOMemoryDescriptor *memory = c != nullptr && rgfxCacheHostPages(*c)
	                                 ? static_cast<IOMemoryDescriptor *>(c->wire)
	                                 : nullptr;
	if (memory)
		memory->retain();
	IOLockUnlock(state.lock);
	if (memory && memory->prepare(kIODirectionOutIn) != kIOReturnSuccess) {
		static char unwired;
		if (rgfxFirstCalls(&unwired, 0, 1))
			IOLog("RgfxAccelerator: the deferred block 0x%llx's pages could not be wired for "
			      "its block\n",
			      offset);
		OSSafeReleaseNULL(memory);
	}
	return memory;
}

namespace {
// Lets go of a cached surface's memory, wired while its block lived (rgfxSurfaceCacheAdd).
void unwireSurface(void *wire)
{
	letGoMemory(wire, true);
}
} // namespace

bool rgfxSurfaceLocksForget(void *cache, void *surface, UInt64 offset, bool releasing,
                            bool blockWired, void **wireOut)
{
	if (wireOut)
		*wireOut = nullptr;
	void *map = nullptr, *wire = nullptr, *carried = nullptr;
	bool kept = false, wired = true;
	if (cache && __atomic_load_n(&state.lock, __ATOMIC_ACQUIRE)) {
		IOLockLock(state.lock);
		const uint64_t key = reinterpret_cast<uintptr_t>(cache);
		// The writes named through this block go; those of the surface's other blocks, by the
		// same device cache, stay pending, their locks waiting for them.
		bool wake = rgfxLockForgetBlock(state.table, key, offset);
		// A cached surface's locks wait for its block's release, the cache and the surface held
		// until the answer (rgfxSurfaceCacheReleased), or it is cached no more. A lock naming a
		// deferred block's pages uses its memory: that attach lets the surface go once answered,
		// and nothing here waits for it.
		const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key);
		wake |= c != nullptr;
		wired = c == nullptr || rgfxCacheWired(*c);
		// The cached entry is the cached block's alone: another block of the surface leaves it.
		kept = !rgfxCacheBlockLetGo(state.caches, key, offset, releasing, surface, &map, &wire,
		                            blockWired, &carried);
		if (wake)
			wakeLocked();
		IOLockUnlock(state.lock);
	}
	if (map)
		static_cast<IOMemoryMap *>(map)->release();
	letGoMemory(carried, true);
	if (kept)
		return false;
	// The references rgfxSurfaceLocksAttach took are the caller's to release, out of every lock,
	// the cache's first: IOSurface frees a cache released to none (RgfxIOSurfaceKPI.hpp), while
	// its surface lives. The surface's memory is unwired, or its wiring handed to the caller, who
	// keeps it while the device may map the memory: the wiring holds the memory, whatever becomes
	// of the surface. A deferred block's memory never wired, whose pages the device never had, is
	// only released.
	if (wireOut && wired)
		*wireOut = wire;
	else
		letGoMemory(wire, wired);
	return true;
}

bool rgfxSurfaceCacheAdd(RgfxAccelerator *accel, void *cache, void *surface, UInt64 offset,
                         uint32_t surfaceId, IOMemoryMap *header, UInt32 genOffset)
{
	if (!cache || !surface || !header || !startLocks())
		return false;
	// The surface's memory, wired while the block lives, held or dormant: the pages the host
	// maps stay the surface's, past its last holder and its own end, until the block's release
	// is answered.
	IOMemoryDescriptor *wire =
	    OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(static_cast<OSObject *>(surface)));
	if (!wire || wire->prepare(kIODirectionOutIn) != kIOReturnSuccess) {
		static char unwired;
		if (rgfxFirstCalls(&unwired, 0, 1))
			IOLog("RgfxAccelerator: IOSurface %u: its memory could not be wired: it is not "
			      "cached\n",
			      surfaceId);
		return false;
	}
	wire->retain();
	auto *genAt = reinterpret_cast<volatile uint64_t *>(header->getVirtualAddress() + genOffset);
	__atomic_store_n(genAt, 0, __ATOMIC_RELEASE);
	const uint64_t key = reinterpret_cast<uintptr_t>(cache);
	IOLockLock(state.lock);
	// IOSurface gives a surface one device cache, so the surface's previous block may still be
	// in the table: a let-go's attach of its pages in flight, whose answer lets it go, is waited
	// for, with no deadline (sleepForHost). One whose release is unanswered was the ask's to wait
	// for, before it made this block (rgfxSurfaceCacheAwaitWriteBack): the device stopped, or the
	// asker's process is ending, and this block goes on uncached.
	for (const RgfxCachedSurface *previous;
	     (previous = rgfxCacheFind(state.caches, key)) != nullptr && !previous->releasing &&
	     sleepForHost(accel) == Slept::kWoken;)
		;
	const char *why = nullptr;
	RgfxCachedSurface *c = nullptr;
	bool logged = true;
	static char previousCached;
	if (rgfxCacheFind(state.caches, key)) {
		why = "its previous block is cached still, its release unanswered";
		logged = rgfxFirstCalls(&previousCached, 0, 1);
	} else if (growCaches(), !(c = rgfxCacheAdd(state.caches, key, offset, surfaceId))) {
		why = "no room for its device cache's state";
		logged = ++state.noRoom <= kNoRoomLogged;
	}
	if (c) {
		c->genAt = genAt;
		c->map = header;
		c->wire = wire;
		c->clientsAt = reinterpret_cast<volatile uint32_t *>(header->getVirtualAddress() +
		                                                     kRgfxSurfaceClientsOffset);
		c->refillAt = reinterpret_cast<volatile uint64_t *>(header->getVirtualAddress() +
		                                                    kRgfxSurfaceRefillOffset);
		c->object = surface;
		noteFirstRefill(*c);
		// IOSurface's getter, as in op 5, under state.lock, kept there on purpose: it loads a field
		// and takes no lock (read from IOSurface's x86_64 code in macOS 26.6's KDK), and op 5 calls
		// it holding IOSurface's own lock at each client's release, which a getter taking that lock
		// could not do (the lock order, ContextState in RgfxAcceleratorContexts.cpp).
		rgfxCacheNoteClients(state.caches, key,
		                     rgfxIOSurfaceClientCount(static_cast<const OSObject *>(surface)));
	}
	IOLockUnlock(state.lock);
	if (why) {
		if (logged)
			IOLog("RgfxAccelerator: IOSurface %u: %s: it is not cached\n", surfaceId, why);
		unwireSurface(wire);
	}
	return c != nullptr;
}

bool rgfxSurfaceCacheWritingBack(OSObject *surface)
{
	if (!surface || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return false;
	IOLockLock(state.lock);
	const bool writing = rgfxCacheWritingBack(state.caches, surface);
	IOLockUnlock(state.lock);
	return writing;
}

void rgfxSurfaceLocksReleaseWaits(uint64_t *count, uint64_t *mostUs)
{
	*count = 0;
	*mostUs = 0;
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	const RgfxReleaseLockWaits waits = state.releaseWaits;
	IOLockUnlock(state.lock);
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(waits.most, &ns);
	*count = waits.count;
	*mostUs = ns / 1000ull;
}

bool rgfxSurfaceCacheAwaitWriteBack(RgfxAccelerator *accel, OSObject *surface)
{
	if (!surface || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return true;
	// The release's answer takes the cached surface out and wakes the locks' sleepers
	// (rgfxSurfaceCacheReleased).
	IOLockLock(state.lock);
	bool done;
	while (!(done = !rgfxCacheWritingBack(state.caches, surface)) &&
	       sleepForHost(accel) == Slept::kWoken)
		;
	IOLockUnlock(state.lock);
	return done;
}

void rgfxSurfaceCacheNoteClients(UInt64 offset)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	RgfxCachedSurface *c = rgfxCacheAtBlock(state.caches, offset);
	// IOSurface's getter, as in op 5, under state.lock, kept there on purpose: it loads a field and
	// takes no lock (read from IOSurface's x86_64 code in macOS 26.6's KDK), and op 5 calls it
	// holding IOSurface's own lock at each client's release, which a getter taking that lock could
	// not do (the lock order, ContextState in RgfxAcceleratorContexts.cpp).
	if (c != nullptr && c->object != nullptr)
		rgfxCacheNoteClients(state.caches, c->key,
		                     rgfxIOSurfaceClientCount(static_cast<const OSObject *>(c->object)));
	IOLockUnlock(state.lock);
}

UInt64 rgfxSurfaceCacheNoRoom()
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return 0;
	IOLockLock(state.lock);
	const UInt64 n = state.noRoom;
	IOLockUnlock(state.lock);
	return n;
}

void rgfxSurfaceCacheSetGrows(bool grows)
{
	__atomic_store_n(&state.grows, grows, __ATOMIC_RELAXED);
}

UInt64 rgfxSurfaceCacheGrown()
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return 0;
	IOLockLock(state.lock);
	const UInt64 n = state.grown;
	IOLockUnlock(state.lock);
	return n;
}

void rgfxSurfaceCacheCarries(UInt64 counts[kRgfxCarryOutcomes])
{
	for (uint32_t i = 0; i < kRgfxCarryOutcomes; i++)
		counts[i] = 0;
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	for (uint32_t i = 0; i < kRgfxCarryOutcomes; i++)
		counts[i] = state.carries[i];
	IOLockUnlock(state.lock);
}

UInt64 rgfxSurfaceCacheRetaken()
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return 0;
	IOLockLock(state.lock);
	const UInt64 n = state.retaken;
	IOLockUnlock(state.lock);
	return n;
}

uint32_t rgfxSurfaceCacheReleaseFlags(UInt64 offset, OSObject *surface)
{
	// Read out of every lock, as at a block's sleep and at a deferred block's release.
	const int32_t clients = rgfxIOSurfaceClientCount(surface);
	const int32_t retains = surface->getRetainCount();
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return clients == 0 ? RGFX_BLOCK_RELEASE_NO_CLIENT : 0;
	IOLockLock(state.lock);
	// The references read again under the lock a lock's or a new block's taking of the surface
	// marks its entry under (rgfxCacheRetaken), the higher of the two reads deciding: a process
	// that took the surface between the reads holds one more.
	const int32_t again = surface->getRetainCount();
	const uint32_t flags =
	    rgfxCacheReleaseFlags(state.caches, offset, clients, again > retains ? again : retains);
	IOLockUnlock(state.lock);
	return flags;
}

void rgfxSurfaceCacheReleased(UInt64 offset, RgfxReleasedSurface *gone)
{
	*gone = {};
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	const RgfxCachedSurface *c = rgfxCacheAtBlock(state.caches, offset);
	gone->wired = c == nullptr || rgfxCacheWired(*c);
	if (c != nullptr && c->releasing && c->unheld && c->retaken)
		state.retaken++;
	// One cached surface at most has the block: its offset is reused only past this release.
	gone->cache = reinterpret_cast<void *>(static_cast<uintptr_t>(rgfxCacheReleased(
	    state.caches, offset, &gone->map, &gone->wire, &gone->surface, &gone->carried)));
	if (gone->cache)
		wakeLocked();
	IOLockUnlock(state.lock);
}

void rgfxSurfaceCacheLetGo(const RgfxReleasedSurface &gone)
{
	if (gone.map)
		static_cast<IOMemoryMap *>(gone.map)->release();
	// The cache's reference and the surface's, held for the answer (rgfxSurfaceLocksForget), in
	// that order; then the surface's memory is unwired, or only released where it was never wired.
	if (gone.cache)
		static_cast<OSObject *>(gone.cache)->release();
	if (gone.surface)
		static_cast<OSObject *>(gone.surface)->release();
	letGoMemory(gone.wire, gone.wired);
	letGoMemory(gone.carried, true);
}

OSObject *rgfxSurfaceCacheReleasingSurface(UInt64 offset, bool *written)
{
	*written = false;
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return nullptr;
	IOLockLock(state.lock);
	OSObject *surface =
	    static_cast<OSObject *>(rgfxCacheReleasingSurface(state.caches, offset, written));
	if (surface)
		surface->retain();
	IOLockUnlock(state.lock);
	return surface;
}

bool rgfxSurfaceCacheSleep(void *cache, bool *purgeable)
{
	*purgeable = false;
	if (!cache || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return false;
	IOLockLock(state.lock);
	const bool slept =
	    rgfxCacheSleep(state.caches, reinterpret_cast<uintptr_t>(cache), true, purgeable);
	IOLockUnlock(state.lock);
	return slept;
}

bool rgfxSurfaceCacheWake(void *cache)
{
	if (!cache || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return false;
	bool purgeable = false;
	IOLockLock(state.lock);
	const bool woke =
	    rgfxCacheSleep(state.caches, reinterpret_cast<uintptr_t>(cache), false, &purgeable);
	IOLockUnlock(state.lock);
	return woke;
}

void rgfxSurfaceCacheUnclaimed(UInt64 offset)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	rgfxCacheUnclaimed(state.caches, offset);
	IOLockUnlock(state.lock);
}

uint32_t rgfxSurfaceCacheLetGoOf(UInt64 offset, bool *written, bool *purgeable)
{
	*written = *purgeable = false;
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return kRgfxCacheKept;
	IOLockLock(state.lock);
	const uint32_t why = rgfxCacheLetGoOf(state.caches, offset, purgeable);
	if (const RgfxCachedSurface *c = rgfxCacheAtBlock(state.caches, offset))
		*written = c->writes != c->copied;
	IOLockUnlock(state.lock);
	return why;
}

bool rgfxSurfaceCacheGeneration(uint32_t surfaceId, uint64_t *gen)
{
	if (!surfaceId || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return false;
	IOLockLock(state.lock);
	const RgfxCachedSurface *c = rgfxCacheOfSurface(state.caches, surfaceId);
	if (c)
		*gen = c->cpuGen;
	IOLockUnlock(state.lock);
	return c != nullptr;
}

void rgfxSurfaceLocksWake()
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	if (state.table.useCount)
		wakeLocked();
	IOLockUnlock(state.lock);
}

void rgfxSurfaceLocksStopped()
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	wakeLocked();
	IOLockUnlock(state.lock);
}

void RgfxAccelerator::noteSurfaceWrites(RgfxAcceleratorUserClient *client, task_t task,
                                        UInt64 queue, UInt64 seqno, const UInt64 *pairs,
                                        UInt32 count)
{
	if (!startLocks()) {
		if (__atomic_fetch_add(&state.skipsLogged, 1, __ATOMIC_RELAXED) < kSkipsLogged)
			IOLog("RgfxAccelerator: no memory for the GPU's pending writes: CPU locks of "
			      "surfaces wait for no GPU write\n");
		return;
	}
	for (UInt32 i = 0; i < count; i++) {
		const UInt64 offset = pairs[2 * i];
		const UInt64 id = pairs[2 * i + 1];
		RgfxDeviceCache *cache = nullptr;
		OSObject *surface = nullptr;
		bool inPlace = false;
		SurfaceCacheAnswer answer =
		    id == 0 || id > UINT32_MAX
		        ? kSurfaceCacheOtherSurface
		        : surfaceDeviceCache(client, task, offset, static_cast<uint32_t>(id),
		                             reinterpret_cast<void **>(&cache), &surface, &inPlace);
		if (answer != kSurfaceCacheMade) {
			// kSurfaceCacheNone was logged where the cache was tried.
			if (answer != kSurfaceCacheNone &&
			    __atomic_fetch_add(&state.skipsLogged, 1, __ATOMIC_RELAXED) < kSkipsLogged) {
				static const char *const why[] = {"", "is not a shared block it holds",
				                                  "has its pages not yet the device's",
				                                  "is another surface's"};
				char name[32] = {};
				proc_name(proc_selfpid(), name, sizeof(name));
				IOLog("RgfxAccelerator: %s[%d] names a GPU write of IOSurface %llu at block "
				      "0x%llx, which %s: its locks do not wait for it\n",
				      name, proc_selfpid(), id, offset, why[answer]);
			}
			continue;
		}
		// Noted before the cache is current, so a lock that finds it current finds the write.
		IOLockLock(state.lock);
		bool kept = rgfxLockNote(state.table, reinterpret_cast<uintptr_t>(cache), queue, seqno,
		                         static_cast<uint32_t>(id), inPlace, offset);
		// A cached surface's pages take the write at the next lock.
		rgfxCacheNoteWrite(state.caches, reinterpret_cast<uintptr_t>(cache));
		bool first = !kept && state.dropped++ == 0;
		IOLockUnlock(state.lock);
		if (first)
			IOLog("RGFX: WARN: no room for the GPU's writes to IOSurface %llu: CPU locks of the "
			      "surfaces past %u writes pending do not wait for them\n",
			      id, kMaxUses);
		rgfxDeviceCacheSetCurrent(cache);
		reinterpret_cast<OSObject *>(cache)->release();
		surface->release();
	}
}

void RgfxAccelerator::retireSurfaceWrites(UInt64 queue, UInt64 seqno)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	if (rgfxLockRetire(state.table, queue, seqno))
		wakeLocked();
	IOLockUnlock(state.lock);
}

void RgfxAccelerator::forgetQueueWrites(UInt64 queue)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	IOLockLock(state.lock);
	if (rgfxLockForgetQueue(state.table, queue))
		wakeLocked();
	IOLockUnlock(state.lock);
}
