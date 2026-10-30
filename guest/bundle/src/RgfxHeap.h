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
 * each wrapped as an RgfxBuffer / RgfxTexture whose `heap` is this one. The
 * offset-less variants (an Automatic heap, where Metal picks the placement) need
 * the host's per-resource size and alignment, which arrives with the region/block
 * transport; until then they are documented nil stubs.
 *
 * The getters answer from the stored descriptor (size, type, storage/cache/hazard
 * modes). Our own ObjC++: conforms to MTLHeap but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_HEAP_H_
#define RGFX_HEAP_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
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

/// Creates a Shared heap over a block of the device region at `regionOffset`,
/// serialising an `RGFX_CMD_HEAP_OVER_HOST_MEMORY` through `serializer` (which picks the
/// objId and GPU address). The host sees the heap's memory in place. Returns nil if
/// `serializer` is null or `descriptor` is nil.
+ (nullable instancetype)heapOverHostMemoryWithDescriptor:(MTLHeapDescriptor *)descriptor
                                             regionOffset:(uint64_t)regionOffset
                                                   device:(nullable id<MTLDevice>)device
                                               serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_HEAP_H_
