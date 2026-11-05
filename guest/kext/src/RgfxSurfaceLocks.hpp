// RgfxSurfaceLocks: the GPU's pending writes of IOSurfaces, and whether a CPU lock of a surface
// waits for them.
//
// A CPU lock of a surface waits for the GPU's pending writes to it, in any process, the one that
// committed them included, and a read-write lock waits for nothing more: a pending buffer that
// only reads the surface holds no lock (measured on a MacBook Pro's AMD Radeon Pro 5300M and Intel
// UHD 630 under macOS 26.7, guest/bundle/tools/lock_probe.m; the M3 Max differs, its read-write
// lock waiting for pending reads too). The bundle names each command buffer's written surfaces to
// the kext as it commits it, with the buffer's queue and seqno. A queue's command buffers complete
// in order, so once its buffer at seqno n has finished with its results in the surface's pages,
// so have all of the queue's before it, and a write is kept per surface and queue as the queue's
// last seqno that writes the surface.
//
// A write retires one of two ways. The bundle's retire call, once the buffer's completion is
// delivered and a copy of the surface written back to its pages, retires the queue's writes up to
// that seqno. And a surface the device's GPU draws in place (`inPlace`) has its pixels in its
// pages once the device has completed the buffer: the device's completed seqno, which the kext
// reads from the queue's completion ring, retires its writes as well.
//
// A surface is named by its key, the device cache the kext made on it, which IOSurface's lock
// calls the kext's control function with, and a write by the key and the block it was named
// through: one surface's blocks share its key (a copy, and one in place; a cached block in place
// and a block of the surface uncached), and each block's let-go takes only its own writes with it,
// the others' pending still. The caller holds one lock around every call.
// Header-only and free of the kernel's headers, so its native test (tests/surface_locks_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_SURFACE_LOCKS_HPP_
#define RGFX_SURFACE_LOCKS_HPP_

#include <stdint.h>

// One queue's pending writes of one surface.
struct RgfxLockUse {
	uint64_t key;
	// The block the writes were named through: its region offset.
	uint64_t block;
	uint64_t queue;
	// The queue's last seqno that writes the surface.
	uint64_t lastWrite;
	// The surface's IOSurfaceID, for the log.
	uint32_t surfaceId;
	// Whether the device draws the surface in place, so the device's completion retires it.
	bool inPlace;
};

// A queue's marks: the seqno its writes up to which the bundle retired, and the one the device
// completed up to.
struct RgfxLockQueue {
	uint64_t queue;
	uint64_t retired;
	uint64_t completed;
};

// The uses and the queues, each kept dense: the first `useCount` and `queueCount` are live.
struct RgfxLockTable {
	RgfxLockUse *uses;
	uint32_t useCount;
	uint32_t useCapacity;
	RgfxLockQueue *queues;
	uint32_t queueCount;
	uint32_t queueCapacity;
};

// The queue's marks, made at its first use where `make`, or null.
inline RgfxLockQueue *rgfxLockQueue(RgfxLockTable &t, uint64_t queue, bool make)
{
	for (uint32_t i = 0; i < t.queueCount; i++) {
		if (t.queues[i].queue == queue)
			return &t.queues[i];
	}
	if (!make || t.queueCount == t.queueCapacity)
		return nullptr;
	t.queues[t.queueCount] = {queue, 0, 0};
	return &t.queues[t.queueCount++];
}

// Whether use `u` is done by queue `q`'s marks.
inline bool rgfxLockDone(const RgfxLockUse &u, const RgfxLockQueue &q)
{
	return u.lastWrite <= q.retired || (u.inPlace && u.lastWrite <= q.completed);
}

// Drops the uses of `queue` its marks `q` have done. True where any went.
inline bool rgfxLockSweep(RgfxLockTable &t, const RgfxLockQueue &q)
{
	bool any = false;
	for (uint32_t i = 0; i < t.useCount;) {
		if (t.uses[i].queue == q.queue && rgfxLockDone(t.uses[i], q)) {
			t.uses[i] = t.uses[--t.useCount];
			any = true;
		} else {
			i++;
		}
	}
	return any;
}

// Notes that queue `queue`'s command buffer at `seqno` writes the surface `key` (IOSurfaceID
// `surfaceId`) through the block at `block`, drawn in place where `inPlace`. A buffer already done
// (its completion came first) leaves nothing. False where the table has no room, and the write is
// not kept.
inline bool rgfxLockNote(RgfxLockTable &t, uint64_t key, uint64_t queue, uint64_t seqno,
                         uint32_t surfaceId, bool inPlace, uint64_t block)
{
	RgfxLockQueue *q = rgfxLockQueue(t, queue, true);
	if (q == nullptr)
		return false;
	if (rgfxLockDone(RgfxLockUse{key, block, queue, seqno, surfaceId, inPlace}, *q))
		return true;
	for (uint32_t i = 0; i < t.useCount; i++) {
		RgfxLockUse &u = t.uses[i];
		if (u.key == key && u.block == block && u.queue == queue) {
			if (seqno > u.lastWrite)
				u.lastWrite = seqno;
			u.inPlace = inPlace;
			return true;
		}
	}
	if (t.useCount == t.useCapacity)
		return false;
	t.uses[t.useCount++] = {key, block, queue, seqno, surfaceId, inPlace};
	return true;
}

// The bundle retired queue `queue`'s command buffers up to `seqno`: their completions delivered and
// their copies written back. True where any write retired, so the caller wakes the waiting locks.
inline bool rgfxLockRetire(RgfxLockTable &t, uint64_t queue, uint64_t seqno)
{
	RgfxLockQueue *q = rgfxLockQueue(t, queue, false);
	if (q == nullptr || seqno <= q->retired)
		return false;
	q->retired = seqno;
	return rgfxLockSweep(t, *q);
}

// The device completed queue `queue`'s command buffers up to `seqno`, which retires the writes of
// the surfaces it draws in place. True where any retired.
inline bool rgfxLockCompleted(RgfxLockTable &t, uint64_t queue, uint64_t seqno)
{
	RgfxLockQueue *q = rgfxLockQueue(t, queue, false);
	if (q == nullptr || seqno <= q->completed)
		return false;
	q->completed = seqno;
	return rgfxLockSweep(t, *q);
}

// The queue is gone, every buffer of it finished or never to finish: its uses and its marks go.
// True where any use went.
inline bool rgfxLockForgetQueue(RgfxLockTable &t, uint64_t queue)
{
	bool any = false;
	for (uint32_t i = 0; i < t.useCount;) {
		if (t.uses[i].queue == queue) {
			t.uses[i] = t.uses[--t.useCount];
			any = true;
		} else {
			i++;
		}
	}
	for (uint32_t i = 0; i < t.queueCount; i++) {
		if (t.queues[i].queue == queue) {
			t.queues[i] = t.queues[--t.queueCount];
			break;
		}
	}
	return any;
}

// The block at `block` of surface `key` goes (its device cache's reference is no longer the
// block's): the writes named through it go, and the writes named through the surface's other
// blocks, which hold the same key, stay pending, their locks waiting for them. True where any
// went.
inline bool rgfxLockForgetBlock(RgfxLockTable &t, uint64_t key, uint64_t block)
{
	bool any = false;
	for (uint32_t i = 0; i < t.useCount;) {
		if (t.uses[i].key == key && t.uses[i].block == block) {
			t.uses[i] = t.uses[--t.useCount];
			any = true;
		} else {
			i++;
		}
	}
	return any;
}

// Whether a CPU lock of surface `key` waits: a write of it is pending. A lock's mode does not
// matter, nor does the task that takes it.
inline bool rgfxLockWaits(const RgfxLockTable &t, uint64_t key)
{
	for (uint32_t i = 0; i < t.useCount; i++) {
		if (t.uses[i].key == key)
			return true;
	}
	return false;
}

// The queues whose writes of surface `key` are pending, up to `capacity` of them in `queues`, of
// those drawn in place only where `inPlace` (the ones the device's completion can retire): how
// many. `*surfaceId` is the surface's IOSurfaceID where any is pending.
inline uint32_t rgfxLockPendingQueues(const RgfxLockTable &t, uint64_t key, bool inPlace,
                                      uint64_t *queues, uint32_t capacity, uint32_t *surfaceId)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < t.useCount; i++) {
		const RgfxLockUse &u = t.uses[i];
		if (u.key != key)
			continue;
		*surfaceId = u.surfaceId;
		if ((inPlace && !u.inPlace) || n == capacity)
			continue;
		bool seen = false;
		for (uint32_t q = 0; q < n && !seen; q++)
			seen = queues[q] == u.queue;
		if (!seen)
			queues[n++] = u.queue;
	}
	return n;
}

// The control function's op 6, which IOSurfaceDeviceCache::unpurge calls as a surface turns
// nonvolatile again (RgfxIOSurfaceKPI.hpp), its second argument `b` the address of a byte. The
// kext sets the byte to 0xff, as the AMD's driver does both after volatile and after empty, and
// IOSurface then answers the surface's old state volatile, its contents kept: they are, in the
// host's cache and the pages. Left 0, the byte has IOSurface answer empty after volatile too,
// the contents lost (measured in the guest). After empty, the AMD's answer is empty, its VRAM
// copy dropped; the kext's is volatile, the contents being kept, which IOSurfaceRef.h's old
// states allow. No other op's argument is written.
constexpr uint32_t kRgfxControlUnpurge = 6;

inline void rgfxControlAnswer(uint32_t op, uint64_t b)
{
	if (op == kRgfxControlUnpurge && b)
		*reinterpret_cast<volatile uint8_t *>(static_cast<uintptr_t>(b)) = 0xff;
}

// The control function's op 3, which IOSurfaceDeviceCache::accelRelease calls to end each call
// IOSurface retained the device cache for: a CPU lock's ops 0 and 2, op 6 and op 7
// (RgfxIOSurfaceKPI.hpp). The reference is the accelerator's to release, as the AMD's driver
// does at each op 3; one left unreleased keeps the cache, and its surface, from ever being
// freed. The release comes last, out of every lock: it may free the cache, whose op 4 then
// comes from inside it.
constexpr uint32_t kRgfxControlDone = 3;

inline bool rgfxControlReleases(uint32_t op)
{
	return op == kRgfxControlDone;
}

#endif // RGFX_SURFACE_LOCKS_HPP_
