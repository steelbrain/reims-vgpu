/*
 * RgfxDepthStencilState: see RgfxDepthStencilState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxDepthStencilState.h"
#import "RgfxTexture.h" // rgfxAppleGPU

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

namespace {

// Whether a face of a descriptor writes the stencil, and whether it reads it, as the M3 Max's
// states answer (below).
bool faceWritesStencil(MTLStencilDescriptor *s)
{
	return s.writeMask != 0 && (s.stencilFailureOperation != MTLStencilOperationKeep ||
	                            s.depthFailureOperation != MTLStencilOperationKeep ||
	                            s.depthStencilPassOperation != MTLStencilOperationKeep);
}

bool faceReadsStencil(MTLStencilDescriptor *s)
{
	return faceWritesStencil(s) ||
	       (s.readMask != 0 && s.stencilCompareFunction != MTLCompareFunctionAlways &&
	        s.stencilCompareFunction != MTLCompareFunctionNever);
}

} // namespace

@implementation RgfxDepthStencilState {
	// What the descriptor reads and writes, as an Apple GPU's state answers (below).
	BOOL _readsDepth, _writesDepth, _readsStencil, _writesStencil;
}

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
	if (state != nil) {
		const MTLCompareFunction depth = descriptor.depthCompareFunction;
		MTLStencilDescriptor *f = descriptor.frontFaceStencil, *b = descriptor.backFaceStencil;
		state->_writesDepth = descriptor.depthWriteEnabled;
		state->_readsDepth = descriptor.depthWriteEnabled ||
		                     (depth != MTLCompareFunctionAlways && depth != MTLCompareFunctionNever);
		state->_writesStencil = faceWritesStencil(f) || faceWritesStencil(b);
		state->_readsStencil = faceReadsStencil(f) || faceReadsStencil(b);
	}
	return state;
}

// The GL renderer's queries of the state it binds (GLDContextRec::updateRenderDepthStencil sends
// them unguarded, read from its code). The AMD 5300M's states (GFX10_MtlDepthStencilState) and
// the Intel 630's (MTLIGDepthStencilState) answer YES to each for every descriptor asked: the
// default one, depth tests with and without writes, stencil tests, stencil replaces, and a
// replace with a write mask of 0 (measured: macOS 26.7.1). So does this, but on an Apple GPU
// (rgfxAppleGPU), whose states answer what the descriptor reads and writes (measured: the M3 Max
// under macOS 27.0.1, oracle_bundle Apple ds.queries and ds.grid, 717 descriptors: every depth
// compare with and without writes, every stencil compare of either face with each operation set
// and read and write masks of 0, 1 and 0xff, under the depth tests Always, Never and Less):
//  - writesDepth: the depth write is enabled;
//  - readsDepth: it is, or the depth compare is neither Always nor Never;
//  - writesStencil: a face's write mask is not 0 and one of its three operations is not Keep;
//  - readsStencil: a face writes it, or its read mask is not 0 and its compare is neither Always
//    nor Never.
- (BOOL)readsDepth
{
	return rgfxAppleGPU(self.device) ? _readsDepth : YES;
}

- (BOOL)writesDepth
{
	return rgfxAppleGPU(self.device) ? _writesDepth : YES;
}

- (BOOL)readsStencil
{
	return rgfxAppleGPU(self.device) ? _readsStencil : YES;
}

- (BOOL)writesStencil
{
	return rgfxAppleGPU(self.device) ? _writesStencil : YES;
}

// The state's id in an argument buffer or an indirect command buffer, which no x86 Mac's state
// has: both the AMD 5300M's and the Intel 630's answer respondsToSelector: YES for it and raise an
// unrecognized selector, an NSInvalidArgumentException that ends the process where nothing
// catches it (macOS 26.7.1: oracle_bundle argbuf.ds.id), and so does this, sending nothing. On an
// Apple GPU (rgfxAppleGPU) it answers 0, as the M3 Max's state answers every time under macOS
// 27.0.1 (oracle_bundle Apple argbuf.ds.id, 11-13).
- (MTLResourceID)gpuResourceID
{
	MTLResourceID rid;
	rid._impl = 0;
	if (!rgfxAppleGPU(self.device))
		[self doesNotRecognizeSelector:_cmd];
	return rid;
}

// Nor an index or an identifier: Metal's base state raises an unrecognized selector for each (it
// sends itself doesNotRecognizeSelector:; read from code, _MTLDepthStencilState, macOS 27.0.1),
// which ends the process on both x86 Macs (measured: macOS 26.7.1, oracle_spi_members answers) and
// on the M3 Max, whose state answers respondsToSelector: YES too (measured: macOS 27.0.1,
// oracle_bundle Apple ds.resourceIndex and ds.uniqueIdentifier), and so does this, on every guest.
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
