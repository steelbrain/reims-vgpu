/*
 * RgfxTexture: the reims-graphics bundle's MTLTexture wrapper (resource slice,
 * following the RgfxBuffer worked example in RgfxBuffer.h / the wrapper convention
 * in RgfxResource.h).
 *
 * A device-placed texture is created by `+textureWithDescriptor:heap:heapOffset:
 * device:serializer:` with `heap` == RGFX_OBJ_NONE, which routes an
 * `RGFX_CMD_TEXTURE_CREATE` through the serialiser (picking the objId, the GPU
 * address and the resource id) and wraps the result. A heap-placed texture passes
 * the owning heap's objId and the placement offset; the host derives its address
 * from the heap it pinned, so no device address is picked. The device category
 * `newTextureWithDescriptor:` and the heap's `newTextureWithDescriptor:offset:`
 * are thin forwarders to this factory.
 *
 * The getters answer from the stored shape (the `RgfxTextureDescriptor` the create
 * carried) and the picked resource id. Our own ObjC++: it conforms to the real
 * `MTLTexture` protocol but subclasses only NSObject (via RgfxResource), so it
 * instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_TEXTURE_H_
#define RGFX_TEXTURE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#include "RgfxObjectModel.hpp"   // RgfxObjId
#include "reims_graphics_wire.h" // RgfxTextureDescriptor (a typedef, not a tag)

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxTexture : RgfxResource <MTLTexture>

/// Creates a texture of shape `descriptor`. When `heap` is RGFX_OBJ_NONE the
/// texture is device-placed (a GPU address is picked); otherwise it lies at
/// `heapOffset` in the heap `heap`. A resource id is picked either way, and given
/// back as the texture goes. Serialises an `RGFX_CMD_TEXTURE_CREATE` through
/// `serializer`; `device` is the owning shell (nil in a unit test). Returns nil if
/// `serializer` is null or `descriptor` is nil, or where no resource id is left
/// (every chunk of the guest's pool is held), as Metal answers a texture it cannot
/// make.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          heap:(RgfxObjId)heap
                                    heapOffset:(uint64_t)heapOffset
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

#ifdef __cplusplus
/// Designated initialiser: wraps an already-created host texture. `objId`,
/// `gpuAddress` (0 when heap-placed) and `resourceId` are the guest-picked values
/// the create returned; `desc` is the mapped shape the getters answer from. The
/// texture gives `resourceId` back to `serializer` as it goes.
- (instancetype)initWithObjId:(uint64_t)objId
                   gpuAddress:(uint64_t)gpuAddress
                   resourceId:(uint64_t)resourceId
                   descriptor:(const RgfxTextureDescriptor &)desc
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;
#endif

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_TEXTURE_H_
