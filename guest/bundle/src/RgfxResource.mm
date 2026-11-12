/*
 * RgfxResource: see RgfxResource.h for the wrapper convention this base
 * establishes. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxResource.h"

#import <Metal/Metal.h>

#import "RgfxHeap.h"
#import "RgfxInterop.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mach/mach_time.h>
#include <mutex>
#include <unistd.h>

#include "RgfxLog.hpp"
#include "RgfxSerializer.hpp"

namespace {
// The last allocation id taken (rgfxAllocationId): every object takes the next as it is made.
std::atomic<uint64_t> g_lastAllocationId{0};

// The threads in RgfxResourceWaitUntilComplete, which sleep on `g_landedWake` under
// `g_landedLock` until the count they wait on is 0: RgfxInFlightEnd wakes them where it makes a
// count 0 and any waits. A waiter is counted before it reads its count, and a count is lowered
// before the waiters are read, both sequentially consistent, so one of the two sees the other.
std::atomic<uint32_t> g_landedWaiters{0};
std::mutex g_landedLock;
std::condition_variable g_landedWake;
// Set by RgfxResourceWaitsStop: the waits end, and no later one waits.
std::atomic<bool> g_waitsStopped{false};

// The resource SPI's answers so far in the process, a diagnostic: waitUntilComplete's calls,
// those that waited, the time they waited and the longest, and the NO answers of isComplete and
// isWriteComplete. An os_log line (debug) says them at most every kWaitsReportNs while waits or
// NO answers come.
std::atomic<uint64_t> g_waitCalls{0};
std::atomic<uint64_t> g_waited{0};
std::atomic<uint64_t> g_waitedNs{0};
std::atomic<uint64_t> g_longestNs{0};
std::atomic<uint64_t> g_notComplete{0};
std::atomic<uint64_t> g_waitsReported{0};
constexpr uint64_t kWaitsReportNs = 10ull * NSEC_PER_SEC;

uint64_t nowNs()
{
	static const mach_timebase_info_data_t base = [] {
		mach_timebase_info_data_t b;
		mach_timebase_info(&b);
		return b;
	}();
	return mach_absolute_time() * base.numer / base.denom;
}

void reportWaits(uint64_t now)
{
	uint64_t last = g_waitsReported.load(std::memory_order_relaxed);
	if (now - last < kWaitsReportNs ||
	    !g_waitsReported.compare_exchange_strong(last, now, std::memory_order_relaxed))
		return;
	os_log_debug(rgfxLog(),
	             "RGFX: resource waits so far: %llu waitUntilComplete calls, %llu waited, %.2f ms "
	             "in all, longest %.2f ms; %llu isComplete or isWriteComplete answered NO",
	             (unsigned long long)g_waitCalls.load(), (unsigned long long)g_waited.load(),
	             g_waitedNs.load() / 1e6, g_longestNs.load() / 1e6,
	             (unsigned long long)g_notComplete.load());
}
} // namespace

void RgfxInFlightEnd(__unsafe_unretained RgfxResource *r)
{
	if (__atomic_sub_fetch(r->_rgfxInFlightAt, 1u, __ATOMIC_SEQ_CST) == 0 &&
	    g_landedWaiters.load(std::memory_order_seq_cst) != 0) {
		std::lock_guard<std::mutex> hold(g_landedLock);
		g_landedWake.notify_all();
	}
}

void RgfxResourceWaitsStop(void)
{
	g_waitsStopped.store(true);
	std::lock_guard<std::mutex> hold(g_landedLock);
	g_landedWake.notify_all();
}

BOOL RgfxResourceIsComplete(__unsafe_unretained RgfxResource *r)
{
	if (__atomic_load_n(r->_rgfxInFlightAt, __ATOMIC_SEQ_CST) == 0)
		return YES;
	g_notComplete.fetch_add(1, std::memory_order_relaxed);
	reportWaits(nowNs());
	return NO;
}

void RgfxResourceWaitUntilComplete(__unsafe_unretained RgfxResource *r)
{
	g_waitCalls.fetch_add(1, std::memory_order_relaxed);
	const uint32_t *count = r->_rgfxInFlightAt;
	if (__atomic_load_n(count, __ATOMIC_SEQ_CST) == 0 || g_waitsStopped.load())
		return;
	const uint64_t start = nowNs();
	g_landedWaiters.fetch_add(1, std::memory_order_seq_cst);
	{
		std::unique_lock<std::mutex> hold(g_landedLock);
		// Woken where any count reaches 0, so the count is read again each time.
		while (__atomic_load_n(count, __ATOMIC_SEQ_CST) != 0 && !g_waitsStopped.load())
			g_landedWake.wait_for(hold, std::chrono::seconds(1));
	}
	g_landedWaiters.fetch_sub(1, std::memory_order_seq_cst);
	const uint64_t end = nowNs(), waited = end - start;
	g_waited.fetch_add(1, std::memory_order_relaxed);
	g_waitedNs.fetch_add(waited, std::memory_order_relaxed);
	uint64_t longest = g_longestNs.load(std::memory_order_relaxed);
	while (waited > longest &&
	       !g_longestNs.compare_exchange_weak(longest, waited, std::memory_order_relaxed)) {
	}
	reportWaits(end);
}

BOOL RgfxResourcesAlias(RgfxResource *a, id b)
{
	if (![b isKindOfClass:[RgfxResource class]]) {
		return NO;
	}
	uint64_t aOffset = 0, aSize = 0, bOffset = 0, bSize = 0;
	id<MTLHeap> aHeap = [a rgfxSpanOffset:&aOffset size:&aSize];
	id<MTLHeap> bHeap = [static_cast<RgfxResource *>(b) rgfxSpanOffset:&bOffset size:&bSize];
	return aHeap != nil && aHeap == bHeap && aOffset < bOffset + bSize && bOffset < aOffset + aSize;
}

@implementation RgfxResource {
	__weak id<MTLDevice> _device;
	id<MTLHeap> _heap; // the heap the resource lies in, kept alive
	// The process the resource is used for (setResponsibleProcess:): this one until told.
	int _responsibleProcess;
	// The purgeable state the client set (setPurgeableState:), NonVolatile until set.
	std::atomic<MTLPurgeableState> _purgeableState;
	// The resource's offset in its heap (rgfxSetHeapOffset:), and whether it was made
	// aliasable (makeAliasable).
	NSUInteger _heapOffset;
	std::atomic<bool> _aliasable;
	// The bytes of its Automatic heap the resource lies in (rgfxHoldHeapRange:size:), and
	// whether it holds them still.
	uint64_t _rangeOffset;
	uint64_t _rangeSize;
	std::atomic<bool> _rangeHeld;
	uint64_t _allocationId; // taken as it was made (rgfxAllocationId)
}

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super init];
	if (self) {
		_objId = objId;
		_device = device;
		_serializer = serializer;
		_responsibleProcess = getpid();
		_purgeableState.store(MTLPurgeableStateNonVolatile);
		_allocationId = g_lastAllocationId.fetch_add(1) + 1;
		_rgfxInFlightAt = &_rgfxInFlight;
	}
	return self;
}

RGFX_LOG_UNRECOGNIZED

- (uint64_t)rgfxObjId
{
	return _objId;
}

std::atomic<uintptr_t> gRgfxResourceClasses[32];

bool RgfxIsResourceClassSlow(__unsafe_unretained id obj, uintptr_t cls)
{
	if (![obj isKindOfClass:[RgfxResource class]])
		return false;
	// The slot's last class: two of ours that share a slot take turns, and stay right.
	gRgfxResourceClasses[(cls >> 4) & 31].store(cls, std::memory_order_relaxed);
	return true;
}

- (id<MTLDevice>)device
{
	return _device;
}

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (id<MTLHeap>)heap
{
	return _heap;
}

- (void)rgfxSetHeap:(id<MTLHeap>)heap
{
	_heap = heap;
}

- (void)rgfxRetireThen:(std::function<void()>)after
{
	// The bytes of its heap the resource still holds go back last, once its destroy has gone,
	// so a resource placed there later is made after it.
	RgfxHeap *rangeHeap = _rangeHeld.exchange(false) ? static_cast<RgfxHeap *>(_heap) : nil;
	const uint64_t rangeOffset = _rangeOffset, rangeSize = _rangeSize;
	RgfxSerializer *serializer = _serializer;
	if (serializer == nullptr) {
		if (after)
			after(); // no host object to wait for
		[rangeHeap rgfxGiveRangeAt:rangeOffset size:rangeSize];
		return;
	}
	// Once the destroy is on the ring, which may be later where the kext refuses it now
	// (RgfxSerializer::destroy): the host holds the object until then.
	serializer->destroy(_objId, [after = std::move(after), rangeHeap, rangeOffset, rangeSize] {
		if (after)
			after();
		[rangeHeap rgfxGiveRangeAt:rangeOffset size:rangeSize];
	});
}

- (void)rgfxHoldHeapRange:(uint64_t)offset size:(uint64_t)size
{
	_rangeOffset = offset;
	_rangeSize = size;
	_rangeHeld.store(true);
}

- (uint64_t)rgfxHeapRangeSize
{
	return _rangeSize;
}

- (uint64_t)rgfxHeapPlacement
{
	return _rangeSize != 0 ? _rangeOffset : _heapOffset;
}

// --- the resource SPI's shared answers (RGFX_RESOURCE_SPI) --------------------------

// A heap's resources lie in the heap's allocation, and share its id as Metal's do (measured on the
// M3 Max under macOS 27); any other is an allocation of its own. Metal suballocates small buffers
// from pools whose buffers share an id too, and the device makes no pools.
- (uint64_t)rgfxAllocationId
{
	RgfxResource *heap = [_heap isKindOfClass:[RgfxResource class]] ? (RgfxResource *)_heap : nil;
	return heap != nil ? heap->_allocationId : _allocationId;
}

// Its bytes in its heap, from where the heap placed it, as many as it takes there.
- (id<MTLHeap>)rgfxSpanOffset:(uint64_t *)offset size:(uint64_t *)size
{
	id<MTLHeap> heap = self.heap;
	if (heap == nil || ![self respondsToSelector:@selector(allocatedSize)]) {
		return nil;
	}
	*offset = self.rgfxHeapPlacement;
	*size = [(id<MTLResource>)self allocatedSize];
	return heap;
}

- (BOOL)rgfxIsPurgeable
{
	const MTLPurgeableState state = _purgeableState.load();
	return state == MTLPurgeableStateVolatile || state == MTLPurgeableStateEmpty;
}

// The resource SPI WindowServer calls as it prepares a window's surface for its compositor:
// the process the resource is used for. The device accounts memory to no process, so it is
// kept and answered.
- (void)setResponsibleProcess:(int)pid
{
	_responsibleProcess = pid;
}

- (int)responsibleProcess
{
	return _responsibleProcess;
}

// The resource SPI WindowServer reads as it composites into a texture: the content protection
// the resource was made with. The device protects no content and no call sets it, so none.
- (uint64_t)protectionOptions
{
	return 0;
}

- (void)rgfxSetHeapOffset:(NSUInteger)offset
{
	_heapOffset = offset;
}

// --- MTLResource's placement, aliasing and ownership --------------------------

- (NSUInteger)heapOffset
{
	return _heapOffset;
}

// Only a resource made in a heap can be made aliasable, as in Metal, and one of a Placement
// heap is aliasable from its making (measured on the M3 Max under macOS 27.0.1: YES for a
// buffer of a Placement heap, NO for one of an Automatic heap until made so). The guest places
// every heap resource itself, so aliasing changes nothing on the device. A resource of an
// Automatic heap gives its bytes back to the heap then, which places what it makes next there,
// as Metal's does (measured on the M3 Max under macOS 27: usedSize drops by its size at once,
// and not again when it is freed).
- (void)makeAliasable
{
	if (_heap != nil) {
		_aliasable.store(true);
		if (_rangeHeld.exchange(false)) {
			[static_cast<RgfxHeap *>(_heap) rgfxGiveRangeAt:_rangeOffset size:_rangeSize];
		}
	}
}

- (BOOL)isAliasable
{
	return _aliasable.load() || (_heap != nil && _heap.type == MTLHeapTypePlacement);
}

// The task the resource's memory is accounted to. The device accounts memory to no task, so
// the change is taken and nothing else follows.
- (kern_return_t)setOwnerWithIdentity:(task_id_token_t)identity
{
	(void)identity;
	return KERN_SUCCESS;
}

// The state is kept as told and the previous one answered (KeepCurrent only answers it). The
// device never purges a resource, so a Volatile one keeps its contents and stays Volatile, and
// an Empty one keeps them too, which no client may read. A heap's resource stays NonVolatile,
// whatever it is told, as Metal's does (measured on the M3 Max under macOS 27: it answers
// NonVolatile to Volatile and then to KeepCurrent, and isPurgeable NO, whatever the heap's state).
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	if (_heap != nil) {
		return MTLPurgeableStateNonVolatile;
	}
	if (state == MTLPurgeableStateKeepCurrent) {
		return _purgeableState.load();
	}
	return _purgeableState.exchange(state);
}

@end
