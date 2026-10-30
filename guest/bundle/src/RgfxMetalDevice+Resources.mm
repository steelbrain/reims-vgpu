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
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {
/// Whether `options` names Shared storage (the storage mode the host sees in place over
/// a region block). MTLResourceStorageModeShared is 0 at the storage-mode shift.
bool rgfxIsSharedStorage(MTLResourceOptions options)
{
	return ((options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift) ==
	       MTLStorageModeShared;
}
} // namespace

@implementation RgfxMetalDevice (Resources)

// --- buffers -----------------------------------------------------------------

// A Shared buffer backed by a region block (the host sees it in place), or nil when
// the region had no room. `initialBytes` (may be null) seeds the block.
- (nullable id<MTLBuffer>)rgfxSharedBufferOfLength:(NSUInteger)length
                                          options:(MTLResourceOptions)options
                                      initialBytes:(nullable const void *)initialBytes
{
	RgfxRegion *region = [self region];
	if (region == nullptr) {
		return nil;
	}
	RgfxRegion::Block block = region->allocate(length == 0 ? 1 : length);
	if (!block.ok) {
		return nil;
	}
	void *contents = const_cast<uint8_t *>(block.ptr);
	if (length > 0) {
		if (initialBytes != nullptr) {
			std::memcpy(contents, initialBytes, length);
		} else {
			std::memset(contents, 0, length);
		}
	}
	// The region owns the block (no free list yet, a follow-up), so no deallocator.
	return [RgfxBuffer bufferOverHostMemoryWithContents:contents
	                                      regionOffset:block.offset
	                                            length:length
	                                           options:options
	                                       deallocator:nil
	                                            device:self
	                                        serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// A Shared buffer lives in the device region so the host sees it in place; any other
	// storage mode is a device buffer (createBuffer) with its own backing.
	if (rgfxIsSharedStorage(options) && [self region] != nullptr) {
		id<MTLBuffer> shared = [self rgfxSharedBufferOfLength:length
		                                             options:options
		                                        initialBytes:nullptr];
		if (shared != nil) {
			return shared;
		}
	}
	return [RgfxBuffer bufferWithLength:length
	                           options:options
	                            device:self
	                        serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer
                             length:(NSUInteger)length
                            options:(MTLResourceOptions)options
{
	if (rgfxIsSharedStorage(options) && [self region] != nullptr) {
		id<MTLBuffer> shared = [self rgfxSharedBufferOfLength:length
		                                             options:options
		                                        initialBytes:pointer];
		if (shared != nil) {
			return shared;
		}
	}
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
	// No-copy wraps the caller's pages as the buffer's backing, which the host sees in
	// place only when those pages already lie in the device region (a buffer the caller
	// allocated from our region). Then we wrap them via createBufferOverHostMemory at
	// their offset and call the caller's deallocator when the buffer dies. Pages outside
	// the region cannot be aliased into the host's view without a copy, which would break
	// the caller's aliasing contract, so those still return nil (a documented limitation).
	RgfxRegion *region = [self region];
	if (region == nullptr || pointer == nullptr || !region->contains(pointer, length)) {
		(void)deallocator;
		return nil;
	}
	uint64_t offset = region->offsetOf(pointer);
	void (^wrapped)(void) = nil;
	if (deallocator != nil) {
		wrapped = ^{ deallocator(pointer, length); };
	}
	return [RgfxBuffer bufferOverHostMemoryWithContents:pointer
	                                      regionOffset:offset
	                                            length:length
	                                           options:options
	                                       deallocator:wrapped
	                                            device:self
	                                        serializer:[self serializer]];
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
	// A Shared heap lives in the device region so the host sees its sub-resources in
	// place; any other storage mode is a device heap (createHeap).
	RgfxRegion *region = [self region];
	if (descriptor != nil && rgfxIsSharedStorage(descriptor.resourceOptions) &&
	    region != nullptr) {
		RgfxRegion::Block block =
		    region->allocate(descriptor.size == 0 ? 1 : static_cast<uint64_t>(descriptor.size));
		if (block.ok) {
			std::memset(const_cast<uint8_t *>(block.ptr), 0, block.size);
			id<MTLHeap> shared = [RgfxHeap heapOverHostMemoryWithDescriptor:descriptor
			                                                  regionOffset:block.offset
			                                                        device:self
			                                                    serializer:[self serializer]];
			if (shared != nil) {
				return shared;
			}
		}
	}
	return [RgfxHeap heapWithDescriptor:descriptor
	                             device:self
	                         serializer:[self serializer]];
}

@end
