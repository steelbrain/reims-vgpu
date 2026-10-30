/*
 * RgfxSampler: the reims-graphics bundle's MTLSamplerState wrapper (resource
 * slice; the wrapper convention is RgfxResource.h, the worked example RgfxBuffer.h).
 *
 * Created by `+samplerWithDescriptor:device:serializer:`, which maps the
 * MTLSamplerDescriptor into an RgfxSamplerParams and routes an
 * `RGFX_CMD_SAMPLER_CREATE` through the serialiser (picking the objId). The device
 * category `newSamplerStateWithDescriptor:` is a thin forwarder.
 *
 * MTLSamplerState exposes only `label`, `device` and `gpuResourceID`; the sampler's
 * state lives on the wire, not in gettable protocol properties, so the wrapper
 * holds only its objId. The sampler create carries no resource id yet, so
 * `gpuResourceID` answers 0 (a follow-up: argument-buffer samplers need one).
 *
 * Our own ObjC++: conforms to MTLSamplerState but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_SAMPLER_H_
#define RGFX_SAMPLER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxSampler : RgfxResource <MTLSamplerState>

/// Creates a sampler state of shape `descriptor`, serialising an
/// `RGFX_CMD_SAMPLER_CREATE` through `serializer` (which picks the objId).
/// `device` is the owning shell (nil in a unit test). Returns nil if `serializer`
/// is null or `descriptor` is nil.
+ (nullable instancetype)samplerWithDescriptor:(MTLSamplerDescriptor *)descriptor
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_SAMPLER_H_
