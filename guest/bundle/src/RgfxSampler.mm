/*
 * RgfxSampler: see RgfxSampler.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSampler.h"
#include "RgfxLogOnce.hpp"

#import "RgfxMetalPrivate.h"

#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxSamplerParams

// Each enum is carried as its Metal raw value; the LOD clamps and bias travel as floats (the
// serialiser stores their bits); the three booleans pack into `flags`. lodBias and reductionMode
// are Metal's defaults, 0 and weighted average, where the descriptor has neither (before macOS 26).
RgfxSamplerParams RgfxSamplerParamsOf(MTLSamplerDescriptor *d)
{
	static const BOOL hasLodBias = [d respondsToSelector:@selector(lodBias)];
	static const BOOL hasReduction = [d respondsToSelector:@selector(reductionMode)];
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
	out.reduction_mode = hasReduction ? static_cast<uint32_t>(d.reductionMode) : 0;
	out.lod_bias = hasLodBias ? d.lodBias : 0.0f;
	return out;
}

@interface RgfxSampler () <RgfxMetalSamplerInternals>
@end

@implementation RgfxSampler {
	uint64_t _resourceId; // the host's answer; 0 where none came (a create on a test's ring)
}

// The host drops the sampler as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)samplerWithDescriptor:(MTLSamplerDescriptor *)descriptor
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a sampler must have a route and a shape
	}
	RgfxSamplerParams params = RgfxSamplerParamsOf(descriptor);
	RgfxCreated created = serializer->createSampler(params);
	// A create the device never got, one the host refused, or one not answered
	// (kRgfxNoAnswer), made no sampler: nil, as Metal answers a sampler it cannot make. One the
	// kext timed out on is followed by its destroy there, so nothing is left to let go.
	if (serializer->refused(created) || created.status != RGFX_STATUS_OK) {
		static char site;
		if (rgfxLogFirst(&site, created.status))
			NSLog(@"RGFX: newSamplerStateWithDescriptor: no sampler (status 0x%x)",
			      created.status);
		return nil;
	}
	RgfxSampler *sampler = [[RgfxSampler alloc] initWithObjId:created.obj_id
	                                                    device:device
	                                                serializer:serializer];
	if (sampler) {
		sampler->_resourceId = created.resource_id;
		// The descriptor's label, nil for an empty one (measured on the M3 Max under macOS 27).
		if (descriptor.label.length != 0)
			sampler.label = descriptor.label;
	}
	return sampler;
}

// --- MTLSamplerState ---------------------------------------------------------

// The id the host's device numbered the sampler by (RgfxSampler.h).
- (MTLResourceID)gpuResourceID
{
	MTLResourceID rid;
	rid._impl = _resourceId;
	return rid;
}

// --- what Metal's sampler answers past its protocol (RgfxMetalPrivate.h) ----------------
// Each as Metal's (measured on the M3 Max under macOS 27): no GPU address and no pixel format,
// whatever the descriptor, one that supports argument buffers among them; and an identifier that
// is its gpuResourceID's value, so the samplers of equal descriptors, which the host answers with
// one id, answer one.

- (uint64_t)gpuAddress
{
	return 0;
}

- (MTLPixelFormat)pixelFormat
{
	return MTLPixelFormatInvalid;
}

- (uint64_t)uniqueIdentifier
{
	return _resourceId;
}

// Metal's base sampler answers gpuResourceID with gpuHandle and gpuHandle with uniqueIdentifier
// (read from code, _MTLSamplerState, macOS 27.0.1), and both x86 Macs' fresh sampler answers its
// identifier as its resource index too (measured: macOS 26.7.1, oracle_spi_members answers): each
// is the id the host numbered the sampler by.
- (uint64_t)gpuHandle
{
	return [self uniqueIdentifier];
}

- (uint64_t)resourceIndex
{
	return [self uniqueIdentifier];
}

@end
