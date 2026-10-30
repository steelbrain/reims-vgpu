/*
 * RgfxDepthStencilState: the reims-graphics bundle's MTLDepthStencilState wrapper
 * (resource slice; the wrapper convention is RgfxResource.h).
 *
 * Created by `+depthStencilStateWithDescriptor:device:serializer:`, which maps the
 * MTLDepthStencilDescriptor (the depth test and write, and the two faces' stencil
 * state) into the serialiser's fields and routes an `RGFX_CMD_DEPTH_STENCIL_CREATE`
 * (picking the objId). The device category `newDepthStencilStateWithDescriptor:`
 * is a thin forwarder.
 *
 * MTLDepthStencilState exposes only `label` and `device` (and `gpuResourceID` on
 * macOS 26+); its state lives on the wire, so the wrapper holds only its objId.
 *
 * Our own ObjC++: conforms to MTLDepthStencilState but subclasses only NSObject
 * (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_DEPTH_STENCIL_STATE_H_
#define RGFX_DEPTH_STENCIL_STATE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxDepthStencilState : RgfxResource <MTLDepthStencilState>

/// Creates a depth/stencil state of shape `descriptor`, serialising an
/// `RGFX_CMD_DEPTH_STENCIL_CREATE` through `serializer` (which picks the objId).
/// `device` is the owning shell (nil in a unit test). Returns nil if `serializer`
/// is null or `descriptor` is nil.
+ (nullable instancetype)depthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
                                                  device:(nullable id<MTLDevice>)device
                                              serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_DEPTH_STENCIL_STATE_H_
