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
 * accelerator, with its own control function, and keeps both with the block's entry, the surface
 * retained, until the block goes. Each write named notes the write in the lock table, then marks
 * the cache current, which has IOSurface call the control function inside every CPU lock of the
 * surface from then on and moves the surface's seed.
 *
 * The control function waits at op 2 while a write of the surface is pending, whatever the lock's
 * mode (a pending read holds no lock), and does nothing at ops 0 and 3: once a write retires, the
 * surface's pages hold the GPU's results, the device's own for a surface drawn in place, else the
 * process's write-back. It looks at the device's completed seqno of each queue with a pending
 * write in place every millisecond, and at once when the device raises its vector or a process
 * retires writes. A lock still waiting after kWaitMs goes on, with an "RGFX: WARN" line naming
 * the surface and the task: a write that never retires (a host that never completes the buffer,
 * or a process that locks a surface its own later buffer writes from the thread that would
 * retire it) holds no lock for good.
 *
 * Our own code; it names SDK-public IOKit calls and IOSurface's interface (RgfxIOSurfaceKPI.hpp).
 */

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxIOSurfaceKPI.hpp"
#include "RgfxSurfaceLocks.hpp"

#include <IOKit/IOLib.h>
#include <kern/clock.h>
#include <libkern/OSAtomic.h>
#include <libkern/libkern.h>
#include <sys/proc.h>

namespace {
// The most pending writes kept at once (a surface's per queue), and the most queues with any.
constexpr UInt32 kMaxUses = 4096;
constexpr UInt32 kMaxQueues = 2048;
// How long a lock waits before it goes on, and how often it looks at the device's completions.
constexpr UInt32 kWaitMs = 2000;
constexpr UInt32 kPollUs = 1000;
// The most queues one surface's lock looks at the completions of.
constexpr UInt32 kMaxLookedAt = 16;
// How many device caches made are logged, with what IOSurface answered.
constexpr UInt32 kCachesLogged = 4;

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
};
LockState state = {nullptr, {nullptr, 0, kMaxUses, nullptr, 0, kMaxQueues}, nullptr, 0, 0, 0,
                   kWaitMs};

bool startLocks()
{
	if (__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return true;
	IOLock *lock = IOLockAlloc();
	auto *uses = static_cast<RgfxLockUse *>(IOMalloc(sizeof(RgfxLockUse) * kMaxUses));
	auto *queues = static_cast<RgfxLockQueue *>(IOMalloc(sizeof(RgfxLockQueue) * kMaxQueues));
	RgfxLockUse *expected = nullptr;
	if (lock && uses && queues &&
	    __atomic_compare_exchange_n(&state.table.uses, &expected, uses, false, __ATOMIC_ACQ_REL,
	                                __ATOMIC_ACQUIRE)) {
		bzero(uses, sizeof(RgfxLockUse) * kMaxUses);
		bzero(queues, sizeof(RgfxLockQueue) * kMaxQueues);
		state.table.queues = queues;
		__atomic_store_n(&state.lock, lock, __ATOMIC_RELEASE);
		return true;
	}
	if (lock)
		IOLockFree(lock);
	if (uses)
		IOFree(uses, sizeof(RgfxLockUse) * kMaxUses);
	if (queues)
		IOFree(queues, sizeof(RgfxLockQueue) * kMaxQueues);
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

// The control function every device cache the kext makes calls (RgfxIOSurfaceKPI.hpp), from
// inside a CPU lock of its surface: op 2 waits while a write of the surface is pending.
void surfaceControl(void *accelerator, RgfxDeviceCache *cache, uint32_t op, uint64_t a,
                    uint64_t b)
{
	if (op != 2 || !__atomic_load_n(&state.lock, __ATOMIC_ACQUIRE))
		return;
	RgfxAccelerator *accel = static_cast<RgfxAccelerator *>(accelerator);
	const uint64_t key = reinterpret_cast<uintptr_t>(cache);
	uint64_t deadline = 0;
	clock_interval_to_deadline(state.waitMs, kMillisecondScale, &deadline);
	uint32_t surfaceId = 0;
	bool gaveUp = false;
	IOLockLock(state.lock);
	while (rgfxLockWaits(state.table, key)) {
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
		uint64_t tick = 0;
		clock_interval_to_deadline(kPollUs, kMicrosecondScale, &tick);
		IOLockSleepDeadline(state.lock, &state.table, tick < deadline ? tick : deadline,
		                    THREAD_UNINT);
	}
	IOLockUnlock(state.lock);
	if (gaveUp) {
		char name[64] = {};
		proc_selfname(name, sizeof(name));
		IOLog("RGFX: WARN: a CPU lock of IOSurface %u by %s[%d] goes on after %u ms with the "
		      "GPU's writes to it pending\n",
		      surfaceId, name, proc_selfpid(), state.waitMs);
	}
}

// Wakes the waiting locks. Under state.lock.
void wakeLocked()
{
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

void rgfxSurfaceLocksForget(void *cache, void *surface)
{
	if (cache && __atomic_load_n(&state.lock, __ATOMIC_ACQUIRE)) {
		IOLockLock(state.lock);
		if (rgfxLockForgetKey(state.table, reinterpret_cast<uintptr_t>(cache)))
			wakeLocked();
		IOLockUnlock(state.lock);
	}
	if (surface)
		static_cast<OSObject *>(surface)->release();
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
	if (!startLocks())
		return;
	for (UInt32 i = 0; i < count; i++) {
		const UInt64 offset = pairs[2 * i];
		const UInt64 id = pairs[2 * i + 1];
		RgfxDeviceCache *cache = nullptr;
		OSObject *surface = nullptr;
		bool inPlace = false;
		if (id == 0 || id > UINT32_MAX ||
		    !surfaceDeviceCache(client, task, offset, static_cast<uint32_t>(id),
		                        reinterpret_cast<void **>(&cache), &surface, &inPlace))
			continue;
		// Noted before the cache is current, so a lock that finds it current finds the write.
		IOLockLock(state.lock);
		bool kept = rgfxLockNote(state.table, reinterpret_cast<uintptr_t>(cache), queue, seqno,
		                         static_cast<uint32_t>(id), inPlace);
		bool first = !kept && state.dropped++ == 0;
		IOLockUnlock(state.lock);
		if (first)
			IOLog("RGFX: WARN: no room for the GPU's writes to IOSurface %llu: CPU locks of the "
			      "surfaces past %u writes pending do not wait for them\n",
			      id, kMaxUses);
		rgfxDeviceCacheSetCurrent(cache);
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
