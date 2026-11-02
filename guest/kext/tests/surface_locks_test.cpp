// The native test of RgfxSurfaceLocks.hpp: a lock of a surface waits for its pending writes, from
// any task and in either mode, and for nothing a buffer only reads; a queue's writes retire in its
// order, by the bundle's retire call, or for a surface drawn in place by the device's completion;
// they go with their queue or the surface's key; and a surface's pending queues are found.
#include "RgfxSurfaceLocks.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_locks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kSurface = 0x1000, kOther = 0x2000;
constexpr uint64_t kQueue = 7, kQueue2 = 9;
constexpr bool kInPlace = true, kCopy = false;
constexpr uint32_t kId = 42;
} // namespace

int main()
{
	RgfxLockUse uses[4] = {};
	RgfxLockQueue queues[2] = {};
	RgfxLockTable t = {uses, 0, 4, queues, 0, 2};

	check(!rgfxLockWaits(t, kSurface), "nothing pending, nothing waits");

	// WindowServer's buffer 1 writes a surface it draws in place.
	check(rgfxLockNote(t, kSurface, kQueue, 1, kId, kInPlace), "a write is noted");
	check(rgfxLockWaits(t, kSurface), "a lock waits for it");
	check(!rgfxLockWaits(t, kOther), "another surface's lock does not wait");

	// Buffer 2 of the same queue writes it again.
	check(rgfxLockNote(t, kSurface, kQueue, 2, kId, kInPlace), "a second write is noted");
	check(t.useCount == 1, "one use per surface and queue");

	// The device completes buffer 1: buffer 2's write holds the lock still.
	check(!rgfxLockCompleted(t, kQueue, 1), "the use stays while a write of it is pending");
	check(rgfxLockWaits(t, kSurface), "the lock still waits");
	check(!rgfxLockCompleted(t, kQueue, 1), "the same mark again changes nothing");

	// The device completes buffer 2, before the bundle's retire call: the pages hold its pixels.
	check(rgfxLockCompleted(t, kQueue, 2), "the device's completion retires a write in place");
	check(t.useCount == 0, "a retired use goes");
	check(!rgfxLockWaits(t, kSurface), "nothing waits once the device is past it");
	check(!rgfxLockRetire(t, kQueue, 2), "the bundle's later retire finds nothing");

	// A surface the bundle copies: the device's completion is not enough, the write-back is.
	check(rgfxLockNote(t, kOther, kQueue, 3, kId, kCopy), "a copied surface's write is noted");
	check(!rgfxLockCompleted(t, kQueue, 3), "the device's completion leaves a copy's write");
	check(rgfxLockWaits(t, kOther), "a lock of the copy waits for its write-back");
	check(rgfxLockRetire(t, kQueue, 3), "the bundle's retire retires it");
	check(!rgfxLockWaits(t, kOther), "and the lock goes on");

	// A completion the commit's note comes after: the write is not kept.
	check(rgfxLockCompleted(t, kQueue, 5) == false, "a completion ahead of its note");
	check(rgfxLockNote(t, kSurface, kQueue, 4, kId, kInPlace) && t.useCount == 0,
	      "a buffer the device already completed leaves no write in place");
	check(rgfxLockNote(t, kOther, kQueue, 4, kId, kCopy) && t.useCount == 1,
	      "but a copy's write is kept until its retire");
	check(rgfxLockRetire(t, kQueue, 4) && t.useCount == 0, "which retires it");
	check(rgfxLockNote(t, kOther, kQueue, 4, kId, kCopy) && t.useCount == 0,
	      "a write the bundle already retired leaves nothing");

	// Two queues: each retires in its own order.
	check(rgfxLockNote(t, kSurface, kQueue, 6, kId, kInPlace), "WindowServer writes");
	check(rgfxLockNote(t, kSurface, kQueue2, 1, kId, kInPlace), "an app writes too");
	check(t.useCount == 2, "one use per queue");
	{
		uint64_t qs[2] = {};
		uint32_t id = 0;
		check(rgfxLockPendingQueues(t, kSurface, true, qs, 2, &id) == 2 && id == kId &&
		          ((qs[0] == kQueue && qs[1] == kQueue2) || (qs[0] == kQueue2 && qs[1] == kQueue)),
		      "the surface's pending queues, once each");
		check(rgfxLockPendingQueues(t, kSurface, true, qs, 1, &id) == 1,
		      "no more queues than the room given");
		check(rgfxLockNote(t, kOther, kQueue, 6, kId, kCopy), "a copy's write on the same queue");
		check(rgfxLockPendingQueues(t, kOther, true, qs, 2, &id) == 0 && id == kId,
		      "a copy's queue is not one the device's completion retires");
		check(rgfxLockPendingQueues(t, kOther, false, qs, 2, &id) == 1 && qs[0] == kQueue,
		      "but it is pending");
		check(rgfxLockForgetKey(t, kOther), "and goes with its key");
	}
	check(rgfxLockRetire(t, kQueue, 6), "WindowServer's write retires");
	check(rgfxLockWaits(t, kSurface), "the app's write holds the lock still");

	// The app's queue goes with its write pending: it never holds a lock again.
	check(rgfxLockForgetQueue(t, kQueue2), "the gone queue's use goes");
	check(!rgfxLockWaits(t, kSurface), "nothing waits for a gone queue");
	check(t.queueCount == 1, "the gone queue's marks go");
	check(!rgfxLockRetire(t, kQueue2, 1) && !rgfxLockCompleted(t, kQueue2, 1),
	      "a gone queue's marks are not made again by a late retire");

	// A surface whose key goes takes its uses with it.
	check(rgfxLockNote(t, kOther, kQueue, 7, kId, kCopy), "a write of the other surface");
	check(rgfxLockForgetKey(t, kOther), "the key's use goes");
	check(!rgfxLockWaits(t, kOther), "nothing waits for a gone key");

	// A full table keeps what it can and says so.
	check(rgfxLockNote(t, 0x10, kQueue, 8, kId, kCopy) && rgfxLockNote(t, 0x20, kQueue, 8, kId, kCopy) &&
	          rgfxLockNote(t, 0x30, kQueue, 8, kId, kCopy) && rgfxLockNote(t, 0x40, kQueue, 8, kId, kCopy),
	      "four uses fill the table");
	check(!rgfxLockNote(t, 0x50, kQueue, 8, kId, kCopy), "a fifth has no room");
	check(rgfxLockNote(t, 0x40, kQueue, 9, kId, kCopy), "a kept use still takes later buffers");
	check(rgfxLockQueue(t, 0x99, true) != nullptr && rgfxLockQueue(t, 0x98, true) == nullptr,
	      "the queues fill too");
	check(!rgfxLockNote(t, 0x10, 0x98, 1, kId, kCopy), "a write of a queue with no marks is not kept");
	check(rgfxLockRetire(t, kQueue, 8) && t.useCount == 1, "buffer 8's retire leaves buffer 9's");
	check(rgfxLockWaits(t, 0x40) && !rgfxLockWaits(t, 0x10), "only the later write holds a lock");
	check(rgfxLockRetire(t, kQueue, 9) && t.useCount == 0, "the queue's retire empties the table");

	std::printf("surface_locks_test: OK (%d checks)\n", checks);
	return 0;
}
