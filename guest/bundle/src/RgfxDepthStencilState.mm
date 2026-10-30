/*
 * RgfxDepthStencilState: see RgfxDepthStencilState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxDepthStencilState.h"

#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxStencilDescriptor (wire)

namespace {

/// Maps one face of an MTLStencilDescriptor into the wire's RgfxStencilDescriptor.
/// Each enum is carried as its Metal raw value. A nil face (Metal's default) maps
/// to the zero-initialised descriptor, which is MTLCompareFunctionNever and the
/// keep operations.
RgfxStencilDescriptor rgfxStencilFromMTL(MTLStencilDescriptor *s)
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

} // namespace

@implementation RgfxDepthStencilState

+ (instancetype)depthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
                                         device:(id<MTLDevice>)device
                                     serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a state must have a route and a shape
	}
	RgfxStencilDescriptor front = rgfxStencilFromMTL(descriptor.frontFaceStencil);
	RgfxStencilDescriptor back = rgfxStencilFromMTL(descriptor.backFaceStencil);
	RgfxCreated created = serializer->createDepthStencil(
	    static_cast<uint32_t>(descriptor.depthCompareFunction),
	    descriptor.depthWriteEnabled ? true : false, front, back);
	return [[RgfxDepthStencilState alloc] initWithObjId:created.obj_id
	                                             device:device
	                                         serializer:serializer];
}

@end
