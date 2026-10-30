/*
 * RgfxFence: the reims-graphics bundle's MTLFence wrapper (sync slice). A fence is
 * an opaque intra-command-buffer ordering handle; like an event it carries nothing
 * guest-side but its `rgfxObjId` (the command-buffer/encoders slice names it on
 * the wire via updateFence:/waitForFence:) and the base `label`.
 *
 * Created by `+fenceWithDevice:serializer:`, which routes an RGFX_CMD_FENCE_CREATE
 * through the serialiser (picking the objId) and wraps the result. The device
 * category `newFence` (RgfxMetalDevice+Sync) is a thin forwarder to this factory.
 *
 * Our own ObjC++: it conforms to the real MTLFence protocol but subclasses only
 * NSObject (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_FENCE_H_
#define RGFX_FENCE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxFence : RgfxResource <MTLFence>

/// Creates a fence, serialising an RGFX_CMD_FENCE_CREATE through `serializer`
/// (which picks the objId). `device` is the owning shell (nil in a unit test).
/// Returns nil if `serializer` is null (a fence must have a route to the host).
+ (nullable instancetype)fenceWithDevice:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_FENCE_H_
