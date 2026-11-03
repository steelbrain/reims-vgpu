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
 * holds its objId and its resource id. The host's device numbers its samplers
 * itself, in one space with its own and its pipelines', and answers one sampler
 * for equal descriptors (measured: M3 Max, macOS 27, as metal2vulkan's do), so
 * the guest picks none: the create waits for the host's answer (SAMPLER_STATE),
 * whose id `gpuResourceID` answers and an argument buffer holds. A sampler the
 * host refuses, or whose create is not answered, is nil, as Metal answers one it
 * cannot make.
 *
 * Our own ObjC++: conforms to MTLSamplerState but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_SAMPLER_H_
#define RGFX_SAMPLER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include "RgfxSerializer.hpp" // RgfxSamplerParams
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxSampler : RgfxResource <MTLSamplerState>

/// Creates a sampler state of shape `descriptor`, serialising an
/// `RGFX_CMD_SAMPLER_CREATE` through `serializer` (which picks the objId), labelled with the
/// descriptor's label (nil for an empty one, as Metal's). A new object each call: the device
/// answers equal descriptors with one through its RgfxStateCache. `device` is the owning shell
/// (nil in a unit test). Returns nil if `serializer` is null or `descriptor` is nil.
+ (nullable instancetype)samplerWithDescriptor:(MTLSamplerDescriptor *)descriptor
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

@end

#ifdef __cplusplus
/// `descriptor`'s state as SAMPLER_CREATE carries it: each enum its Metal raw value, the LOD
/// clamps as floats, the three booleans in `flags`.
RgfxSamplerParams RgfxSamplerParamsOf(MTLSamplerDescriptor *descriptor);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_SAMPLER_H_
