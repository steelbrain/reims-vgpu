/*
 * RgfxDepthStencilState: see RgfxDepthStencilState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxDepthStencilState.h"
#import "RgfxUnsupported.h"

#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxStencilDescriptor (wire)

RgfxStencilDescriptor RgfxStencilOf(MTLStencilDescriptor *s)
{
	RgfxStencilDescriptor out = {};
	if (s == nil) {
		return out;
	}
	out.stencil_compare = static_cast<uint32_t>(s.stencilCompareFunction);
	out.stencil_fail_op = static_cast<uint32_t>(s.stencilFailureOperation);
	out.depth_fail_op = static_cast<uint32_t>(s.depthFailureOperation);
	out.pass_op = static_cast<uint32_t>(s.depthStencilPassOperation);
	out.read_mask = s.readMask;
	out.write_mask = s.writeMask;
	return out;
}

@implementation RgfxDepthStencilState

// The host drops the state as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)depthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
                                         device:(id<MTLDevice>)device
                                     serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a state must have a route and a shape
	}
	RgfxStencilDescriptor front = RgfxStencilOf(descriptor.frontFaceStencil);
	RgfxStencilDescriptor back = RgfxStencilOf(descriptor.backFaceStencil);
	RgfxCreated created = serializer->createDepthStencil(
	    static_cast<uint32_t>(descriptor.depthCompareFunction),
	    descriptor.depthWriteEnabled ? true : false, front, back);
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxDepthStencilState *state = [[RgfxDepthStencilState alloc] initWithObjId:created.obj_id
	                                                                     device:device
	                                                                 serializer:serializer];
	// The descriptor's label, nil for an empty one (measured on the M3 Max under macOS 27).
	if (state != nil && descriptor.label.length != 0)
		state.label = descriptor.label;
	return state;
}

// The state's id in an argument buffer or an indirect command buffer: the depth/stencil create
// carries no resource id on the wire, and the device supports neither use. 0, logged.
- (MTLResourceID)gpuResourceID
{
	RGFX_UNSUPPORTED("depth/stencil resource ids (none on the wire)");
	MTLResourceID rid;
	rid._impl = 0;
	return rid;
}

@end
