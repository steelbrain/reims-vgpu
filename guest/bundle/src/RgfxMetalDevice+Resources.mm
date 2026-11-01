/*
 * RgfxMetalDevice (Resources): see RgfxMetalDevice+Resources.h. The device's
 * MTLDevice resource creators, each a thin forwarder to a wrapper factory, routing
 * through `-serializer`. Our own ObjC++.
 */

#import "RgfxMetalDevice+Resources.h"

#import "RgfxBuffer.h"
#import "RgfxDepthStencilState.h"
#import "RgfxHeap.h"
#import "RgfxIOSurface.h"
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

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// A Shared buffer over a block of the device region, which the host sees in place, a Managed
	// one with its CPU copy there, a Private one the GPU's alone; none where Metal makes none, nor
	// where the region has no room (RgfxBuffer.h).
	return [RgfxBuffer bufferForDeviceWithLength:length
	                                     options:options
	                                       bytes:nullptr
	                                   maxLength:[self maxBufferLength]
	                                      region:[self region]
	                                      device:self
	                                  serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer
                             length:(NSUInteger)length
                            options:(MTLResourceOptions)options
{
	// As newBufferWithLength:options:, its block seeded with the bytes; none for a Private one,
	// whose bytes Metal's validation refuses, nor for no bytes at all, which Metal's validation
	// refuses too and its copy faults at (measured on the M3 Max).
	if (pointer == nullptr)
		return nil;
	return [RgfxBuffer bufferForDeviceWithLength:length
	                                     options:options
	                                       bytes:pointer
	                                   maxLength:[self maxBufferLength]
	                                      region:[self region]
	                                      device:self
	                                  serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer
                                   length:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                              deallocator:(void (^)(void *, NSUInteger))deallocator
{
	// Over the caller's bytes in place where they lie in a block of ours, else over a block of
	// their own synced with them (RgfxBuffer.h).
	return [RgfxBuffer bufferWithBytesNoCopy:pointer
	                                  length:length
	                                 options:options
	                             deallocator:deallocator
	                                  region:[self region]
	                                  device:self
	                              serializer:[self serializer]];
}

// --- textures ----------------------------------------------------------------

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	// A Managed texture's CPU copy is a region block; one of a format or shape no CPU copy
	// holds is made without one.
	if (descriptor != nil && descriptor.storageMode == MTLStorageModeManaged &&
	    [self region] != nullptr && [self serializer] != nullptr) {
		RgfxTexture *managed = [RgfxTexture managedTextureWithDescriptor:descriptor
		                                                          region:[self region]
		                                                          device:self
		                                                      serializer:[self serializer]];
		if (managed != nil) {
			return managed;
		}
	}
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
	// Over this process's copy of the plane in the device region, synced with the
	// surface's pages (RgfxIOSurface.h).
	return [RgfxIOSurfaceTexture textureWithDescriptor:descriptor
	                                         iosurface:iosurface
	                                             plane:plane
	                                            region:[self region]
	                                            device:self
	                                        serializer:[self serializer]];
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
		// The heap's memory is the block, on the host too, whole pages of Metal's 16 KiB.
		const uint64_t page = 0x4000;
		const uint64_t size = descriptor.size == 0 ? 1 : static_cast<uint64_t>(descriptor.size);
		RgfxRegion::Block block = region->allocate((size + page - 1) & ~(page - 1), page);
		if (block.ok) {
			uint8_t *bytes = const_cast<uint8_t *>(block.ptr);
			std::memset(bytes, 0, block.size);
			// The block goes back to the region once the host has dropped the heap.
			uint64_t offset = block.offset;
			id<MTLHeap> shared = [RgfxHeap heapOverHostMemoryWithDescriptor:descriptor
			                                                  regionOffset:offset
			                                                      contents:bytes
			                                                   deallocator:^{ region->release(offset); }
			                                                        device:self
			                                                    serializer:[self serializer]];
			if (shared != nil) {
				return shared;
			}
			region->release(offset); // no heap lies over the block
		}
	}
	return [RgfxHeap heapWithDescriptor:descriptor
	                             device:self
	                         serializer:[self serializer]];
}

@end
