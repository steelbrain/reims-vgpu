/*
 * RgfxSampler: see RgfxSampler.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSampler.h"

#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxSamplerParams

namespace {

/// Maps an MTLSamplerDescriptor into the serialiser's RgfxSamplerParams. Each
/// enum is carried as its Metal raw value; the LOD clamps travel as floats (the
/// serialiser stores their bits); the three booleans pack into `flags`.
RgfxSamplerParams rgfxSamplerParamsFromMTL(MTLSamplerDescriptor *d)
{
	RgfxSamplerParams out = {};
	out.min_filter = static_cast<uint32_t>(d.minFilter);
	out.mag_filter = static_cast<uint32_t>(d.magFilter);
	out.mip_filter = static_cast<uint32_t>(d.mipFilter);
	out.s_address_mode = static_cast<uint32_t>(d.sAddressMode);
	out.t_address_mode = static_cast<uint32_t>(d.tAddressMode);
	out.r_address_mode = static_cast<uint32_t>(d.rAddressMode);
	out.border_color = static_cast<uint32_t>(d.borderColor);
	out.compare_function = static_cast<uint32_t>(d.compareFunction);
	out.max_anisotropy = static_cast<uint32_t>(d.maxAnisotropy);
	out.lod_min_clamp = d.lodMinClamp;
	out.lod_max_clamp = d.lodMaxClamp;
	out.flags = (d.normalizedCoordinates ? 0x1u : 0u) | (d.lodAverage ? 0x2u : 0u) |
	            (d.supportArgumentBuffers ? 0x4u : 0u);
	return out;
}

} // namespace

@implementation RgfxSampler

+ (instancetype)samplerWithDescriptor:(MTLSamplerDescriptor *)descriptor
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a sampler must have a route and a shape
	}
	RgfxSamplerParams params = rgfxSamplerParamsFromMTL(descriptor);
	RgfxCreated created = serializer->createSampler(params);
	return [[RgfxSampler alloc] initWithObjId:created.obj_id device:device serializer:serializer];
}

// --- MTLSamplerState ---------------------------------------------------------

- (MTLResourceID)gpuResourceID
{
	// The sampler create carries no resource id yet; a sampler used from an
	// argument buffer needs one (a wire follow-up). Answer 0 until then.
	MTLResourceID rid;
	rid._impl = 0;
	return rid;
}

@end
