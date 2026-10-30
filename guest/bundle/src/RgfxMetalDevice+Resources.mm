/*
 * RgfxMetalDevice (Resources): see RgfxMetalDevice+Resources.h. The device's
 * MTLDevice resource creators, each a thin forwarder to a wrapper factory, routing
 * through `-serializer`. Our own ObjC++.
 */

#import "RgfxMetalDevice+Resources.h"

#import "RgfxBuffer.h"
#import "RgfxDepthStencilState.h"
#import "RgfxHeap.h"
#import "RgfxSampler.h"
#import "RgfxTexture.h"

#include <cstring>

#include "RgfxObjectModel.hpp" // RGFX_OBJ_NONE
#include "RgfxSerializer.hpp"

@implementation RgfxMetalDevice (Resources)

// --- buffers -----------------------------------------------------------------

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	return [RgfxBuffer bufferWithLength:length
	                           options:options
	                            device:self
	                        serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer
                             length:(NSUInteger)length
                            options:(MTLResourceOptions)options
{
	RgfxBuffer *buf = [RgfxBuffer bufferWithLength:length
	                                       options:options
	                                        device:self
	                                    serializer:[self serializer]];
	if (buf && pointer && length > 0) {
		std::memcpy([buf contents], pointer, length);
	}
	return buf;
}

- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer
                                   length:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                              deallocator:(void (^)(void *, NSUInteger))deallocator
{
	// No-copy wraps the caller's pages as the buffer's backing, which the host
	// sees only once those pages are mapped into the device region. That mapping
	// is createBufferOverHostMemory, which the region transport lands; until then
	// there is no way to honour no-copy (RgfxBuffer allocates its own backing), so
	// this is a documented nil stub (we do not silently copy, which would break
	// the caller's aliasing contract).
	(void)pointer;
	(void)length;
	(void)options;
	(void)deallocator;
	return nil;
}

// --- textures ----------------------------------------------------------------

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	return [RgfxTexture textureWithDescriptor:descriptor
	                                     heap:RGFX_OBJ_NONE
	                               heapOffset:0
	                                   device:self
	                               serializer:[self serializer]];
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                 iosurface:(IOSurfaceRef)iosurface
                                     plane:(NSUInteger)plane
{
	// Texture-over-IOSurface import (TEXTURE_FROM_IOSURFACE) has no wire struct
	// yet and needs the IOSurface shared into the device region: a later transport
	// piece. Documented nil stub until then.
	(void)descriptor;
	(void)iosurface;
	(void)plane;
	return nil;
}

// --- samplers ----------------------------------------------------------------

- (id<MTLSamplerState>)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)descriptor
{
	return [RgfxSampler samplerWithDescriptor:descriptor
	                                   device:self
	                               serializer:[self serializer]];
}

// --- depth/stencil state -----------------------------------------------------

- (id<MTLDepthStencilState>)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
	return [RgfxDepthStencilState depthStencilStateWithDescriptor:descriptor
	                                                      device:self
	                                                  serializer:[self serializer]];
}

// --- heaps -------------------------------------------------------------------

- (id<MTLHeap>)newHeapWithDescriptor:(MTLHeapDescriptor *)descriptor
{
	return [RgfxHeap heapWithDescriptor:descriptor
	                             device:self
	                         serializer:[self serializer]];
}

@end
