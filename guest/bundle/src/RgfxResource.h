/*
 * RgfxResource: the base of every guest Metal object wrapper the reims-graphics
 * bundle vends (buffers, textures, samplers, heaps, queues, pipelines, events…).
 *
 * THE WRAPPER CONVENTION (established here; the resource/pipeline/command-buffer/
 * sync slices replicate it):
 *  - A wrapper is an `NSObject` subclass that conforms to the matching Metal
 *    *protocol* (`RgfxBuffer : RgfxResource <MTLBuffer>`, and so on). It is NOT a
 *    subclass of any Apple-private class, so — unlike the device shell — it
 *    instantiates and is unit-tested on the Mac in user space.
 *  - Every wrapper carries the guest-picked `objId` (RgfxObjId) that names its
 *    host handle, a weak back-pointer to the owning `id<MTLDevice>` (our device
 *    shell at runtime; nil in a unit test), and a raw pointer to that device's
 *    `RgfxSerializer`, which the wrapper routes its mutating Metal calls through.
 *    The serializer and the device own their lifetimes; the wrapper only borrows
 *    them, so the pointers are not retained/freed here.
 *  - A wrapper's getters answer from local state (length, gpuAddress, options,
 *    label…). A wrapper's mutating calls (create, modify, encode) serialise a wire
 *    message through the serializer. A create is done by the device category that
 *    makes the wrapper (it calls the serializer, then hands the RgfxCreated to the
 *    wrapper's initialiser), so a wrapper is never half-created.
 *
 * RgfxResource itself holds the three common fields and the MTLResource `label`.
 * Concrete wrappers add their protocol's state and methods. This is our own
 * ObjC++; it depends only on the serialiser core and the committed wire header,
 * never on Apple's private SPI.
 */

#ifndef RGFX_RESOURCE_H_
#define RGFX_RESOURCE_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "RgfxObjectModel.hpp" // RgfxObjId

#ifdef __cplusplus
#include <atomic>
#include <cstdint>
#include <functional>
#include <objc/runtime.h>

class RgfxSerializer;
#endif

@protocol MTLDevice;
@protocol MTLHeap;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxResource : NSObject {
@package
	// The guest-picked id naming this object's host handle (RgfxObjId == uint64_t). Read by
	// the encoders with no message (RgfxResourceId).
	uint64_t _objId;
	// Borrowed: the owning device's serialiser, the route for mutating calls.
	// Null in a unit test that drives a wrapper without a device.
	RgfxSerializer *_serializer;
	// The keep stamp of the command buffer that last kept the object (RgfxEncoder::keepStamp),
	// 0 before any: its encoders keep it once, at its first naming. Any thread's encoder reads
	// and writes it, relaxed: a lost write only keeps it once more.
	uint64_t _rgfxKeptBy;
	// The texture over an IOSurface the object is, or views through any views (a view holds
	// its texture, so this outlives it); nil for every other.
	__unsafe_unretained id _Nullable _rgfxSurfaceRoot;
	// The command buffers in flight that name the object's memory: submitted and not yet
	// complete (RgfxInFlightBegin, RgfxInFlightEnd), counted where `_rgfxInFlightAt` points: at
	// `_rgfxInFlight`, or at the count of the object whose memory it lies over (a view's texture,
	// a texture's buffer), which it holds. Read and written by atomic builtins.
	uint32_t _rgfxInFlight;
	uint32_t *_Nonnull _rgfxInFlightAt;
}

/// The object's guest-picked id (0 names no object).
@property(nonatomic, readonly) uint64_t rgfxObjId;

/// The owning device (our shell at runtime). Weak: the device owns its wrappers'
/// lifetimes, not the other way round.
@property(nonatomic, readonly, weak, nullable) id<MTLDevice> device;

/// MTLResource's debug label, kept locally (diagnostics only, not on the wire).
@property(nonatomic, copy, nullable) NSString *label;

/// Designated initialiser: binds the wrapper to its host handle `objId`, owning
/// `device` (may be nil in a test), and that device's `serializer` (may be null).
- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/// The borrowed serialiser (null when the wrapper was made without one).
- (nullable RgfxSerializer *)serializer;

/// The heap the resource lies in (MTLResource's `heap`), nil for one that lies in none. The
/// resource keeps it alive, as Metal's heap-placed resources do.
@property(nonatomic, readonly, nullable) id<MTLHeap> heap;

/// Places the resource in `heap` (the heap's sub-allocation sets it).
- (void)rgfxSetHeap:(nullable id<MTLHeap>)heap;

/// Records the resource's offset in its heap (MTLResource's `heapOffset` answers it).
- (void)rgfxSetHeapOffset:(NSUInteger)offset;

/// Holds the `size` bytes at `offset` of its Automatic heap (`heap`, an RgfxHeap) that the heap
/// placed it at. They go back to the heap once, when the resource is made aliasable or else once
/// its host object is dropped (rgfxRetireThen:), as Metal's go when its resource is made
/// aliasable or freed.
- (void)rgfxHoldHeapRange:(uint64_t)offset size:(uint64_t)size;

/// The bytes rgfxHoldHeapRange: held, given back or not; 0 for a resource it never held.
@property(nonatomic, readonly) uint64_t rgfxHeapRangeSize;
/// Where in its heap the resource lies: the offset its Automatic heap placed it at, else its
/// heapOffset. 0 for a resource in no heap.
@property(nonatomic, readonly) uint64_t rgfxHeapPlacement;

#ifdef __cplusplus
/// For a wrapper's dealloc: drops the object's host handle (OBJECT_DESTROY), then runs `after`
/// (when set), which gives back what the host object lay over: its region block, its resource
/// id; then the bytes of its Automatic heap it still holds go back. Where the destroy waits (the
/// kext refused it), they go back once a later destroy or commit posts it
/// (RgfxSerializer::destroy). A command buffer that names the object keeps its wrapper until it
/// completes, so no buffer in flight names a dropped object. Without a serialiser there is no
/// host object.
- (void)rgfxRetireThen:(std::function<void()>)after;
#endif

/// The MTLResource members every wrapper shares (RgfxResource.mm), declared so a subclass's
/// protocol conformance sees them.
@property(nonatomic, readonly) NSUInteger heapOffset;
- (void)makeAliasable;
- (BOOL)isAliasable;
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state;
- (kern_return_t)setOwnerWithIdentity:(task_id_token_t)identity;

/// The resource SPI's content protection (RgfxMetalPrivate.h): none.
- (uint64_t)protectionOptions;

/// What the resource SPI members a buffer and a texture share answer (RGFX_RESOURCE_SPI):
/// the id of the allocation the resource lies in, its heap's, else one of its own;
@property(nonatomic, readonly) uint64_t rgfxAllocationId;
/// the heap its memory lies in, with the offset and the bytes it takes there, nil for none;
- (nullable id<MTLHeap>)rgfxSpanOffset:(uint64_t *)offset size:(uint64_t *)size;
/// and whether the purgeable state it keeps is Volatile or Empty.
@property(nonatomic, readonly) BOOL rgfxIsPurgeable;
@end

#ifdef __cplusplus
/// Whether `obj` is one of ours (an RgfxResource), with no message: its class is looked up in a
/// small table of classes found to be ours, and only a class not yet there asks isKindOfClass:.
/// A class never leaves the table: classes are not unloaded.
bool RgfxIsResourceClassSlow(__unsafe_unretained id _Nonnull obj, uintptr_t cls);
extern std::atomic<uintptr_t> gRgfxResourceClasses[32];
inline bool RgfxIsResource(__unsafe_unretained id _Nullable obj)
{
	if (obj == nil)
		return false;
	const uintptr_t cls = reinterpret_cast<uintptr_t>((__bridge void *)object_getClass(obj));
	if (gRgfxResourceClasses[(cls >> 4) & 31].load(std::memory_order_relaxed) == cls)
		return true;
	return RgfxIsResourceClassSlow(obj, cls);
}

/// `obj`'s rgfxObjId where it is one of ours, else 0: nil, and an object no device of ours made,
/// name no object. Read with no message and no retain, for the encoders, which name an object
/// for each op that takes one.
inline uint64_t RgfxResourceId(__unsafe_unretained id _Nullable obj)
{
	if (!RgfxIsResource(obj))
		return 0;
	__unsafe_unretained RgfxResource *r = (RgfxResource *)obj;
	return r->_objId;
}

/// Whether `a`'s memory meets `b`'s: both lie in one heap over byte ranges that meet.
BOOL RgfxResourcesAlias(RgfxResource *a, id _Nullable b);

/// Counts `r`'s memory in flight in one more command buffer, as the buffer is submitted.
inline void RgfxInFlightBegin(__unsafe_unretained RgfxResource *r)
{
	__atomic_fetch_add(r->_rgfxInFlightAt, 1u, __ATOMIC_SEQ_CST);
}

/// Counts `r`'s memory out of one command buffer, as the buffer's results land where the CPU
/// sees them, and wakes the threads waiting for it (RgfxResourceWaitUntilComplete) where none
/// is left.
void RgfxInFlightEnd(__unsafe_unretained RgfxResource *r);

/// The resource SPI's isComplete and isWriteComplete (RGFX_RESOURCE_SPI): whether no command
/// buffer that names `r`'s memory is in flight.
BOOL RgfxResourceIsComplete(__unsafe_unretained RgfxResource *r);

/// The resource SPI's waitUntilComplete (RGFX_RESOURCE_SPI): returns once no command buffer that
/// names `r`'s memory is in flight, or once RgfxResourceWaitsStop was called.
void RgfxResourceWaitUntilComplete(__unsafe_unretained RgfxResource *r);

/// Ends RgfxResourceWaitUntilComplete's waits, those under way and every later one: no thread
/// takes the process's completions as they come any more (its completion thread ended, or never
/// started), so a count in flight may not fall until a waiter of a command buffer takes them. A
/// wait then returns at once, as it did before any wait was kept.
void RgfxResourceWaitsStop(void);
#endif

/// The resource SPI members (RgfxMetalPrivate.h) a buffer and a texture answer alike, as Metal's
/// (measured on the M3 Max under macOS 27):
///  - allocationID: the id of the allocation it lies in, which a heap's resources share;
///  - doesAliasResource: YES only where both lie in one heap over byte ranges that meet, so a
///    resource in no heap aliases nothing, itself included; doesAliasAllResources:count: and
///    doesAliasAnyResources:count: ask it of each of a list (YES and NO for an empty one);
///  - isComplete and isWriteComplete NO, and waitUntilComplete waits, while a command buffer
///    that names it, or a view of its memory, is submitted and its results are yet to land where
///    the CPU sees them: from its commitAndWaitUntilSubmitted's return, or its scheduling, to its
///    completion's take (RgfxInFlightBegin, RgfxInFlightEnd), for any use. That is the Intel
///    630's answer (macOS 26.7.1); the M3 Max answers isComplete NO for any use and
///    isWriteComplete NO for a write alone, and waits too; the AMD 5300M answers YES and returns
///    at once. All three answer YES and return at once for a buffer not yet scheduled, or not
///    committed. Apple's GL renderer reads a texture back with getBytes: after its
///    waitUntilComplete and a synchronize on a queue of its own, which does not wait for the
///    draw's queue: an answer at once read the texture as it was before the draw;
///  - isPurgeable YES once made Volatile or Empty, NO before and once NonVolatile again.
/// The macro goes in the class's @implementation.
#define RGFX_RESOURCE_SPI                                                                          \
	-(uint64_t)allocationID                                                                        \
	{                                                                                              \
		return self.rgfxAllocationId;                                                              \
	}                                                                                              \
	-(BOOL)doesAliasResource : (id<MTLResource>)resource                                           \
	{                                                                                              \
		return RgfxResourcesAlias(self, resource);                                                 \
	}                                                                                              \
	-(BOOL)doesAliasAllResources : (const id<MTLResource> _Nonnull[_Nullable])resources count      \
	    : (NSUInteger)count                                                                        \
	{                                                                                              \
		for (NSUInteger i = 0; i < count; i++) {                                                   \
			if (!RgfxResourcesAlias(self, resources[i]))                                           \
				return NO;                                                                         \
		}                                                                                          \
		return YES;                                                                                \
	}                                                                                              \
	-(BOOL)doesAliasAnyResources : (const id<MTLResource> _Nonnull[_Nullable])resources count      \
	    : (NSUInteger)count                                                                        \
	{                                                                                              \
		for (NSUInteger i = 0; i < count; i++) {                                                   \
			if (RgfxResourcesAlias(self, resources[i]))                                            \
				return YES;                                                                        \
		}                                                                                          \
		return NO;                                                                                 \
	}                                                                                              \
	-(BOOL)isComplete                                                                              \
	{                                                                                              \
		return RgfxResourceIsComplete(self);                                                       \
	}                                                                                              \
	-(BOOL)isWriteComplete                                                                         \
	{                                                                                              \
		return RgfxResourceIsComplete(self);                                                       \
	}                                                                                              \
	-(void)waitUntilComplete                                                                       \
	{                                                                                              \
		RgfxResourceWaitUntilComplete(self);                                                       \
	}                                                                                              \
	-(BOOL)isPurgeable                                                                             \
	{                                                                                              \
		return self.rgfxIsPurgeable;                                                               \
	}

/// The options Metal answers for an object made with `options`: a Default hazard tracking mode is
/// `fallback`, as Metal resolves it as it makes the object (Untracked for a heap, Tracked for a
/// resource in no heap; measured on the M3 Max under macOS 27).
static inline MTLResourceOptions RgfxResolveHazardTracking(MTLResourceOptions options,
                                                           MTLResourceOptions fallback)
{
	return (options & MTLResourceHazardTrackingModeMask) != 0 ? options : (options | fallback);
}

NS_ASSUME_NONNULL_END

#endif // RGFX_RESOURCE_H_
