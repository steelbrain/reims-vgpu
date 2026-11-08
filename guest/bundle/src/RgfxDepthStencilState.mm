/*
 * RgfxDepthStencilState: see RgfxDepthStencilState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxDepthStencilState.h"

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

// The GL renderer's queries of the state it binds (GLDContextRec::updateRenderDepthStencil sends
// them unguarded, read from its code). The AMD 5300M's states (GFX10_MtlDepthStencilState) and
// the Intel 630's (MTLIGDepthStencilState) answer YES to each for every descriptor asked: the
// default one, depth tests with and without writes, stencil tests, stencil replaces, and a
// replace with a write mask of 0 (measured: macOS 26.7.1). So does this.
- (BOOL)readsDepth
{
	return YES;
}

- (BOOL)writesDepth
{
	return YES;
}

- (BOOL)readsStencil
{
	return YES;
}

- (BOOL)writesStencil
{
	return YES;
}

// The state's id in an argument buffer or an indirect command buffer, which no x86 Mac's state
// has: both the AMD 5300M's and the Intel 630's answer respondsToSelector: YES for it and raise an
// unrecognized selector, an NSInvalidArgumentException that ends the process where nothing
// catches it (macOS 26.7.1: oracle_bundle argbuf.ds.id), and so does this, sending nothing.
- (MTLResourceID)gpuResourceID
{
	[self doesNotRecognizeSelector:_cmd];
	MTLResourceID rid;
	rid._impl = 0;
	return rid;
}

// Nor an index or an identifier: Metal's base state raises an unrecognized selector for each (it
// sends itself doesNotRecognizeSelector:; read from code, _MTLDepthStencilState, macOS 27.0.1),
// which ends the process on both x86 Macs (measured: macOS 26.7.1, oracle_spi_members answers),
// and so does this.
- (uint64_t)resourceIndex
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (uint64_t)uniqueIdentifier
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

@end
