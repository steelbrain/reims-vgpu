/*
 * RgfxBuffer: the reims-graphics bundle's MTLBuffer wrapper, and the worked
 * example of the wrapper convention (RgfxResource.h). The resource slice adds
 * RgfxTexture/RgfxSampler/RgfxHeap/… the same way; this is the one concrete
 * wrapper slice 0 ships so the pattern is pinned and unit-tested.
 *
 * A buffer is created by `+bufferWithLength:options:device:serializer:`, which
 * routes an `RGFX_CMD_BUFFER_CREATE` through the serialiser (picking the objId and
 * GPU address) and wraps the result. The device category `newBufferWithLength:
 * options:` (resource slice) is a thin forwarder to this factory. The getters
 * answer from local state: `contents` hands back the buffer's CPU-visible backing,
 * `length`/`gpuAddress`/`resourceOptions` the stored values.
 *
 * Our own ObjC++: it conforms to the real `MTLBuffer` protocol but subclasses only
 * NSObject (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_BUFFER_H_
#define RGFX_BUFFER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxBuffer : RgfxResource <MTLBuffer>

/// Creates a device buffer of `length` bytes with `options`, serialising an
/// `RGFX_CMD_BUFFER_CREATE` through `serializer` (which picks the objId and GPU
/// address). `device` is the owning shell (nil in a unit test). Returns nil if
/// `serializer` is null (a buffer must have a route) or the backing cannot be
/// allocated.
+ (nullable instancetype)bufferWithLength:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                                   device:(nullable id<MTLDevice>)device
                               serializer:(nullable RgfxSerializer *)serializer;

/// Designated initialiser: wraps an already-created host buffer. `objId` and
/// `gpuAddress` are the guest-picked values the create returned; `length` and
/// `options` are the buffer's. Allocates the CPU-visible backing `contents` vends.
- (nullable instancetype)initWithObjId:(uint64_t)objId
                            gpuAddress:(uint64_t)gpuAddress
                                length:(NSUInteger)length
                               options:(MTLResourceOptions)options
                                device:(nullable id<MTLDevice>)device
                            serializer:(nullable RgfxSerializer *)serializer
    NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_BUFFER_H_
