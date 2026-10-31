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

/// Whether `options` names Managed storage, whose CPU copy is a region block the host copies
/// to and from the GPU's (RgfxTexture's managedTextureWithDescriptor, RgfxBuffer's
/// didModifyRange:).
bool rgfxIsManagedStorage(MTLResourceOptions options)
{
	return ((options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift) ==
	       MTLStorageModeManaged;
}
} // namespace

@implementation RgfxMetalDevice (Resources)

// --- buffers -----------------------------------------------------------------

// A Shared or Managed buffer backed by a region block (the host sees a Shared one in place,
// and a Managed one's CPU copy there), or nil when the region had no room. `initialBytes`
// (may be null) seeds the block.
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
	// The block goes back to the region once the host has dropped the buffer.
	uint64_t offset = block.offset;
	id<MTLBuffer> buffer = [RgfxBuffer bufferOverHostMemoryWithContents:contents
	                                                      regionOffset:offset
	                                                            length:length
	                                                           options:options
	                                                       deallocator:^{ region->release(offset); }
	                                                            device:self
	                                                        serializer:[self serializer]];
	if (buffer == nil)
		region->release(offset); // no buffer lies over the block
	return buffer;
}

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// A Shared buffer lives in the device region so the host sees it in place, and a Managed
	// one's CPU copy does, which the host copies to the GPU's; any other storage mode is a
	// device buffer (createBuffer) with its own backing.
	if ((rgfxIsSharedStorage(options) || rgfxIsManagedStorage(options)) &&
	    [self region] != nullptr) {
		id<MTLBuffer> shared = [self rgfxSharedBufferOfLength:length
		                                             options:options
		                                        initialBytes:nullptr];
		if (shared != nil) {
			return shared;
		}
	}
	// A Managed buffer without a block would be bytes the GPU never sees: none, as Metal
	// answers a buffer it has no memory for.
	if (rgfxIsManagedStorage(options)) {
		return nil;
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
	if ((rgfxIsSharedStorage(options) || rgfxIsManagedStorage(options)) &&
	    [self region] != nullptr) {
		id<MTLBuffer> shared = [self rgfxSharedBufferOfLength:length
		                                             options:options
		                                        initialBytes:pointer];
		if (shared != nil) {
			return shared;
		}
	}
	if (rgfxIsManagedStorage(options)) {
		return nil;
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
