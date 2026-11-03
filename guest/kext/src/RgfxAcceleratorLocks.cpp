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
 * again, so it sleeps through none. A lock still waiting after kWaitMs goes on, with an
 * "RGFX: WARN" line naming the surface and the task: a write that never retires (a host that
 * never completes the buffer, or a process that locks a surface its own later buffer writes from
 * the thread that would retire it) holds no lock for good.
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
 * Our own code; it names SDK-public IOKit calls and IOSurface's interface (RgfxIOSurfaceKPI.hpp).
 */

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxIOSurfaceKPI.hpp"
#include "RgfxSurfaceCaches.hpp"
#include "RgfxSurfaceLocks.hpp"

#include <IOKit/IOLib.h>
#include <kern/clock.h>
#include <libkern/OSAtomic.h>
#include <libkern/libkern.h>
#include <sys/proc.h>

namespace {
// The most pending writes kept at once (a surface's per queue), and the most queues with any.
constexpr UInt32 kMaxUses = 4096;
constexpr UInt32 kMaxQueues = kRgfxMaxWatchedQueues;
// How long a lock waits before it goes on, and how long it sleeps at most between looks at the
// device's completions where no wake comes.
constexpr UInt32 kWaitMs = 2000;
constexpr UInt32 kBackstopMs = 20;
// The most queues one surface's lock looks at the completions of.
constexpr UInt32 kMaxLookedAt = 16;
// The most cached surfaces at once (kRgfxCachedMaxSurfaces: every shared block in place may be
// one, held, dormant or released and not yet answered), and how long a lock waits for a
// download's answer: the host copies once the buffers that used the cache before have finished.
constexpr UInt32 kMaxCached = kRgfxCachedMaxSurfaces;
constexpr UInt32 kDownloadAnswerMs = 2000;
// How many device caches made are logged, with what IOSurface answered, and how many writes
// named of surfaces the kext cannot follow.
constexpr UInt32 kCachesLogged = 4;
constexpr UInt32 kSkipsLogged = 8;
// How many surfaces left uncached for want of room are logged one by one; the dormancy line
// counts them all.
constexpr UInt32 kNoRoomLogged = 4;

// One initialised object (it lands in __DATA,__data, as the contexts' state does), made at the
// first write named and kept for the kernel's life.
struct LockState {
	// Under `lock`: the pending writes. The waiting locks sleep on `table`.
	IOLock *lock;
	RgfxLockTable table;
	// Under the contexts' lock (RgfxAccelerator::surfaceDeviceCache): IOSurfaceRoot, retained,
	// and the accelerator ID it gave this accelerator.
	OSObject *root;
	uint32_t acceleratorId;
	uint32_t cachesMade;
	// Writes the table had no room for, logged at the first.
	uint32_t dropped;
	uint32_t waitMs;
	// Writes named of surfaces the kext cannot follow, logged up to kSkipsLogged.
	uint32_t skipsLogged;
	// Under `lock`: the surfaces whose blocks have a device cache on the host. The waiting locks
	// sleep on `table` for these too.
	RgfxCacheTable caches;
	// Under `lock`: surfaces left uncached as the table had no room, the first kNoRoomLogged
	// logged.
	uint64_t noRoom;
	// Under `lock`: the waiting locks' wakes so far (wakeLocked).
	uint64_t wakes;
};
LockState state = {nullptr, {nullptr, 0, kMaxUses, nullptr, 0, kMaxQueues}, nullptr, 0, 0, 0,
                   kWaitMs, 0, {nullptr, 0, kMaxCached}, 0, 0};

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

static_assert(kRgfxPurgeableNonVolatile == kIOMemoryPurgeableNonVolatile, "op 7's nonvolatile");
static_assert(kRgfxPurgeableVolatile == kIOMemoryPurgeableVolatile, "op 7's volatile");
static_assert(kRgfxPurgeableEmpty == kIOMemoryPurgeableEmpty, "op 7's empty");

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
		// or nonvolatile again, the release thread marks it.
		IOLockLock(state.lock);
		const bool look = op == 5   ? rgfxCachePurged(state.caches, key)
		                  : op == 6 ? rgfxCacheUnpurged(state.caches, key)
		                            : rgfxCachePurgeable(state.caches, key, a);
		IOLockUnlock(state.lock);
		if (look)
			accel->wakeReleases();
		return;
	}
	if (op != 2)
		return;
	uint64_t deadline = 0;
	clock_interval_to_deadline(state.waitMs, kMillisecondScale, &deadline);
	uint32_t surfaceId = 0;
	bool gaveUp = false;
	IOLockLock(state.lock);
	while (rgfxLockWaits(state.table, key)) {
		// The wakes so far: one that comes while the completions are read, out of the lock, has
		// the lock look again rather than sleep through it.
		const uint64_t woken = state.wakes;
		// The device's completions of the queues writing the surface in place.
		uint64_t queues[kMaxLookedAt];
		uint32_t n = rgfxLockPendingQueues(state.table, key, true, queues, kMaxLookedAt,
		                                   &surfaceId);
		if (n) {
			IOLockUnlock(state.lock);
			uint64_t done[kMaxLookedAt];
			for (uint32_t i = 0; i < n; i++)
				done[i] = accel->queueCompleted(queues[i]);
			IOLockLock(state.lock);
			for (uint32_t i = 0; i < n; i++)
				rgfxLockCompleted(state.table, queues[i], done[i]);
			if (!rgfxLockWaits(state.table, key))
				break;
		}
		uint64_t now = mach_absolute_time();
		if (now >= deadline) {
			gaveUp = true;
			break;
		}
		if (state.wakes != woken)
			continue;
		uint64_t backstop = 0;
		clock_interval_to_deadline(kBackstopMs, kMillisecondScale, &backstop);
		IOLockSleepDeadline(state.lock, &state.table, backstop < deadline ? backstop : deadline,
		                    THREAD_UNINT);
	}
	// A cached surface: the pages take what the GPU wrote, one lock asking at a time, and a
	// read-write lock moves its CPU generation.
	const char *waitedFor = gaveUp ? "the GPU's writes to it pending" : nullptr;
	if (const RgfxCachedSurface *c = rgfxCacheFind(state.caches, key)) {
		if (!surfaceId)
			surfaceId = c->surfaceId;
	}
	while (!gaveUp) {
		uint64_t offset = 0, mark = 0;
		RgfxCacheStep step = rgfxCacheNext(state.caches, key, &offset, &mark);
		if (step == kRgfxCacheGoOn)
			break;
		if (step == kRgfxCacheDownload) {
			IOLockUnlock(state.lock);
			bool ok = accel->downloadSurfaceCache(offset, kDownloadAnswerMs);
			IOLockLock(state.lock);
			rgfxCacheDownloaded(state.caches, key, mark, ok);
			wakeLocked();
			if (!ok) {
				waitedFor = "its device cache not copied into its pages";
				break;
			}
			continue;
		}
		if (mach_absolute_time() >= deadline) {
			gaveUp = true;
			waitedFor = "its device cache's copy or its block's release unanswered";
			break;
		}
		IOLockSleepDeadline(state.lock, &state.table, deadline, THREAD_UNINT);
	}
	if (a == 0) {
		if (RgfxCachedSurface *c = rgfxCacheFind(state.caches, key)) {
			const uint64_t gen = rgfxCacheReadWriteLock(state.caches, key);
			if (c->genAt)
				__atomic_store_n(c->genAt, gen, __ATOMIC_RELEASE);
		}
	}
	IOLockUnlock(state.lock);
	if (waitedFor) {
		char name[64] = {};
		proc_selfname(name, sizeof(name));
		IOLog("RGFX: WARN: a CPU lock of IOSurface %u by %s[%d] goes on after %u ms with %s\n",
		      surfaceId, name, proc_selfpid(), state.waitMs, waitedFor);
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
	if (!state.root) {
		OSDictionary *matching = IOService::serviceMatching("IOSurfaceRoot");
		IOService *root = matching ? IOService::copyMatchingService(matching) : nullptr;
		OSSafeReleaseNULL(matching);
		if (!root) {
			IOLog("RgfxAccelerator: no IOSurfaceRoot: CPU locks of surfaces wait for no GPU "
			      "write\n");
			return false;
		}
		state.root = root;
		state.acceleratorId = rgfxIOSurfaceRootAcceleratorID(root, accel);
		IOLog("RgfxAccelerator: IOSurface accelerator ID %u\n", state.acceleratorId);
	}
	OSObject *surface = rgfxIOSurfaceRootLookupSurface(state.root, surfaceId, task);
	if (!surface) {
		IOLog("RgfxAccelerator: IOSurface %u is not the naming process's: its locks wait for no "
		      "GPU write\n",
		      surfaceId);
		return false;
	}
	const OSMetaClass *meta = surface->getMetaClass();
	if (!meta || strcmp(meta->getClassName(), "IOSurface") != 0 || !surfaceHolds(surface, phys)) {
		IOLog("RgfxAccelerator: IOSurface %u is not its shared block's surface: its locks wait "
		      "for no GPU write\n",
		      surfaceId);
		surface->release();
		return false;
	}
	RgfxDeviceCache *cache = rgfxIOSurfaceDeviceCache(surface, accel, state.acceleratorId, 0);
	if (!cache) {
		IOLog("RgfxAccelerator: IOSurface %u gave no device cache: its locks wait for no GPU "
		      "write\n",
		      surfaceId);
		surface->release();
		return false;
	}
	rgfxDeviceCacheSetControl(cache, accel, &surfaceControl);
	if (state.cachesMade++ < kCachesLogged)
		IOLog("RgfxAccelerator: IOSurface %u's device cache %p, retain count %d\n", surfaceId,
		      cache, reinterpret_cast<OSObject *>(cache)->getRetainCount());
	*surfaceOut = surface;
	*cacheOut = cache;
	return true;
}

namespace {
// Lets go of a cached surface's memory, wired while its block lived (rgfxSurfaceCacheAdd).
void unwireSurface(void *wire)
{
	if (IOMemoryDescriptor *md = static_cast<IOMemoryDescriptor *>(wire)) {
		md->complete(kIODirectionOutIn);
		md->release();
	}
}
} // namespace

void rgfxSurfaceLocksForget(void *cache, void *surface, bool releasing)
{
	void *map = nullptr, *wire = nullptr;
	bool kept = false;
	if (cache && __atomic_load_n(&state.lock, __ATOMIC_ACQUIRE)) {
		IOLockLock(state.lock);
		const uint64_t key = reinterpret_cast<uintptr_t>(cache);
		bool wake = rgfxLockForgetKey(state.table, key);
		// A cached surface's locks wait for its block's release, the cache and the surface held
		// until the answer (rgfxSurfaceCacheReleased), or it is cached no more.
		wake |= rgfxCacheFind(state.caches, key) != nullptr;
		kept = !rgfxCacheLetGo(state.caches, key, releasing, surface, &map, &wire);
		if (wake)
			wakeLocked();
		IOLockUnlock(state.lock);
	}
	if (map)
		static_cast<IOMemoryMap *>(map)->release();
	if (kept)
		return;
	// The references rgfxSurfaceLocksAttach took, out of every lock, the cache's first: IOSurface
	// frees a cache released to none (RgfxIOSurfaceKPI.hpp), while its surface lives. Then the
	// surface's memory is unwired.
	if (cache)
		reinterpret_cast<OSObject *>(cache)->release();
	if (surface)
		static_cast<OSObject *>(surface)->release();
	unwireSurface(wire);
}

bool rgfxSurfaceCacheAdd(void *cache, void *surface, UInt64 offset, uint32_t surfaceId,
                         IOMemoryMap *header, UInt32 genOffset)
{
	if (!cache || !surface || !header || !startLocks())
		return false;
	// The surface's memory, wired while the block lives, held or dormant: the pages the host
	// maps stay the surface's, past its last holder and its own end, until the block's release
	// is answered.
	IOMemoryDescriptor *wire =
	    OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(static_cast<OSObject *>(surface)));
	if (!wire || wire->prepare(kIODirectionOutIn) != kIOReturnSuccess) {
		IOLog("RgfxAccelerator: IOSurface %u: its memory could not be wired: it is not cached\n",
		      surfaceId);
		return false;
	}
	wire->retain();
	auto *genAt = reinterpret_cast<volatile uint64_t *>(header->getVirtualAddress() + genOffset);
	__atomic_store_n(genAt, 0, __ATOMIC_RELEASE);
	const uint64_t key = reinterpret_cast<uintptr_t>(cache);
	uint64_t deadline = 0;
	clock_interval_to_deadline(state.waitMs, kMillisecondScale, &deadline);
	IOLockLock(state.lock);
	// IOSurface gives a surface one device cache, so the surface's previous block may still be
	// in the table, its release unanswered: the copy into the pages the answer waits for comes
	// before this block's locks.
	while (rgfxCacheFind(state.caches, key) && mach_absolute_time() < deadline)
		IOLockSleepDeadline(state.lock, &state.table, deadline, THREAD_UNINT);
	const char *why = nullptr;
	RgfxCachedSurface *c = nullptr;
	bool logged = true;
	if (rgfxCacheFind(state.caches, key)) {
		why = "its previous block is cached still, its release unanswered";
	} else if (!(c = rgfxCacheAdd(state.caches, key, offset, surfaceId))) {
		why = "no room for its device cache's state";
		logged = ++state.noRoom <= kNoRoomLogged;
	}
	if (c) {
		c->genAt = genAt;
		c->map = header;
		c->wire = wire;
	}
	IOLockUnlock(state.lock);
	if (why) {
		if (logged)
			IOLog("RgfxAccelerator: IOSurface %u: %s: it is not cached\n", surfaceId, why);
		unwireSurface(wire);
	}
	return c != nullptr;
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

void rgfxSurfaceCacheReleased(UInt64 offset)
{
	if (!__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	void *map = nullptr, *wire = nullptr, *surface = nullptr;
	IOLockLock(state.lock);
	// One cached surface at most has the block: its offset is reused only past this release.
	OSObject *cache = reinterpret_cast<OSObject *>(
	    static_cast<uintptr_t>(rgfxCacheReleased(state.caches, offset, &map, &wire, &surface)));
	if (cache)
		wakeLocked();
	IOLockUnlock(state.lock);
	if (map)
		static_cast<IOMemoryMap *>(map)->release();
	// The cache's reference and the surface's, held for the answer (rgfxSurfaceLocksForget), in
	// that order; then the surface's memory is unwired.
	if (cache)
		cache->release();
	if (surface)
		static_cast<OSObject *>(surface)->release();
	unwireSurface(wire);
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
		                         static_cast<uint32_t>(id), inPlace);
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
