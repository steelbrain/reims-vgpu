/*
 * RgfxHeap: the reims-graphics bundle's MTLHeap wrapper (resource slice; the
 * wrapper convention is RgfxResource.h).
 *
 * Created by `+heapWithDescriptor:device:serializer:`, which maps the
 * MTLHeapDescriptor (size, resource options, heap type) into the serialiser's
 * createHeap args and routes an `RGFX_CMD_HEAP_CREATE` (picking the objId and the
 * heap's GPU address). The device category `newHeapWithDescriptor:` is a thin
 * forwarder.
 *
 * The heap sub-allocates its own resources: a Placement heap's
 * `newBufferWithLength:options:offset:` routes an `RGFX_CMD_BUFFER_AT_OFFSET` and
 * `newTextureWithDescriptor:offset:` an `RGFX_CMD_TEXTURE_CREATE` into this heap,
 * each wrapped as an RgfxBuffer / RgfxTexture whose `heap` is this one. A placed
 * buffer's GPU address is the heap's plus its offset, and a Shared heap's placed
 * buffer's contents are the heap's bytes there, so placed resources alias the
 * heap's memory as Metal's do.
 *
 * An Automatic heap places what `newBufferWithLength:options:` and
 * `newTextureWithDescriptor:` make itself, as Metal's does (RgfxHeapRanges.hpp): a
 * buffer takes its length at the device's heap buffer alignment, a texture the size
 * and alignment the host's device gives it (HEAP_TEXTURE_SIZE_AND_ALIGN), and each
 * goes in at the offset picked, as in a Placement heap, which the host makes the heap.
 * Its `heapOffset` answers 0, as Metal's does. Its bytes go back to the heap when it
 * is made aliasable or freed. `usedSize` and `maxAvailableSizeWithAlignment:` answer
 * from those placements, and a Placement heap answers 0 for both, as Metal's does.
 *
 * The getters answer from the stored descriptor (type, storage/cache/hazard modes),
 * and the size is the descriptor's rounded up to Metal's 16 KiB page, as Metal's is
 * (measured: M3 Max, macOS 27: a heap of 100 bytes is 16384, of 1000000 1015808, of
 * either type). Our own ObjC++: conforms to MTLHeap but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_HEAP_H_
#define RGFX_HEAP_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxRegion;
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxHeap : RgfxResource <MTLHeap>

/// Creates a heap of shape `descriptor`, serialising an `RGFX_CMD_HEAP_CREATE`
/// through `serializer` (which picks the objId and GPU address). `device` is the
/// owning shell (nil in a unit test). Returns nil if `serializer` is null or
/// `descriptor` is nil.
+ (nullable instancetype)heapWithDescriptor:(MTLHeapDescriptor *)descriptor
                                     device:(nullable id<MTLDevice>)device
                                 serializer:(nullable RgfxSerializer *)serializer;

#ifdef __cplusplus
/// The heap `newHeapWithDescriptor:` makes: the host's own (heapWithDescriptor:), of any mode
/// but Shared. A Shared one ends the process with Metal's words, sending nothing, as the heap
/// descriptor validation of the AMD 5300M and the Intel 630 under macOS 26.7.1 does (oracle_bundle
/// heapbuffers; the M3 Max makes one); `region` is unused.
///
/// Nil where Metal makes none (measured on the M3 Max, macOS 27.0.1, plain and under
/// MTL_DEBUG_LAYER, Private, Automatic and Placement): for a size of 0, and for one past
/// `maxLength` (the device's maxBufferLength, at which Metal makes one; 0 holds none back).
+ (nullable instancetype)heapForDeviceWithDescriptor:(nullable MTLHeapDescriptor *)descriptor
                                           maxLength:(NSUInteger)maxLength
                                              region:(nullable RgfxRegion *)region
                                              device:(nullable id<MTLDevice>)device
                                          serializer:(nullable RgfxSerializer *)serializer;
#endif

/// Creates a Shared heap over a block of the device region at `regionOffset`, whose bytes
/// are at `contents` here, serialising an `RGFX_CMD_HEAP_OVER_HOST_MEMORY` through
/// `serializer` (which picks the objId and GPU address). The heap's memory is the block, in
/// place on the host too, so a buffer placed in it is the block's bytes at its offset, as
/// Metal's is. The block holds the heap's size rounded up to Metal's 16 KiB page.
/// `deallocator` (may be nil) gives the block back once the host has dropped the heap.
/// Returns nil if `serializer` is null or `descriptor` is nil.
+ (nullable instancetype)heapOverHostMemoryWithDescriptor:(MTLHeapDescriptor *)descriptor
                                             regionOffset:(uint64_t)regionOffset
                                                 contents:(nullable void *)contents
                                              deallocator:(nullable void (^)(void))deallocator
                                                   device:(nullable id<MTLDevice>)device
                                               serializer:(nullable RgfxSerializer *)serializer;

/// A Shared heap's bytes here (heapOverHostMemoryWithDescriptor:'s `contents`), where its
/// resources lie at their offsets; NULL for a heap of any other storage mode.
@property(nonatomic, readonly, nullable) void *rgfxContents;

/// Gives back the `size` bytes at `offset` that one of the heap's Automatic placements held
/// (RgfxResource's rgfxHoldHeapRange:size:). Thread-safe.
- (void)rgfxGiveRangeAt:(uint64_t)offset size:(uint64_t)size;

@end

#ifdef __cplusplus
/// heapTextureSizeAndAlignWithDescriptor:'s answer for `descriptor`: the host device's, through
/// `serializer` (HEAP_TEXTURE_SIZE_AND_ALIGN), else, where none came back, the bound of Metal's
/// that RgfxHeapSizes.hpp computes, logged the first time `cls` is asked `cmd`. 0 bytes for a
/// nil descriptor.
MTLSizeAndAlign rgfxHeapTextureSizeAndAlign(RgfxSerializer *_Nullable serializer,
                                            MTLTextureDescriptor *_Nullable descriptor, Class cls,
                                            SEL cmd);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_HEAP_H_
