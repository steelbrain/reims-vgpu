/*
 * RgfxMetalDevice (Resources): see RgfxMetalDevice+Resources.h. The device's
 * MTLDevice resource creators, each a thin forwarder to a wrapper factory, routing
 * through `-serializer`. Our own ObjC++.
 */

#import "RgfxMetalDevice+Resources.h"

#import "RgfxBlockCall.h"
#import "RgfxBuffer.h"
#import "RgfxHeap.h"
#import "RgfxIOSurface.h"
#import "RgfxIndirectCommandBuffer.h"
#import "RgfxMetalPrivate.h"
#import "RgfxStateCache.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include "RgfxObjectModel.hpp" // RGFX_OBJ_NONE
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

@implementation RgfxMetalDevice (Resources)

// --- buffers -----------------------------------------------------------------

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// A Shared buffer in a slab of the device region, which the host sees in place, a Managed
	// one with its CPU copy there, a Private one the GPU's alone; none where Metal makes none, nor
	// where the region has no room (RgfxBuffer.h).
	return [RgfxBuffer bufferForDeviceWithLength:length
	                                     options:options
	                                       bytes:nullptr
	                                   maxLength:[self maxBufferLength]
	                                       slabs:[self slabs]
	                                      device:self
	                                  serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer
                             length:(NSUInteger)length
                            options:(MTLResourceOptions)options
{
	// As newBufferWithLength:options:, its range seeded with the bytes; none for a Private one,
	// whose bytes Metal's validation refuses, nor for no bytes at all, which Metal's validation
	// refuses too and its copy faults at (measured on the M3 Max).
	if (pointer == nullptr)
		return nil;
	return [RgfxBuffer bufferForDeviceWithLength:length
	                                     options:options
	                                       bytes:pointer
	                                   maxLength:[self maxBufferLength]
	                                       slabs:[self slabs]
	                                      device:self
	                                  serializer:[self serializer]];
}

- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer
                                   length:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                              deallocator:(void (^)(void *, NSUInteger))deallocator
{
	// Over the caller's bytes in place where they lie in a block of ours or the kext wires their
	// pages, else over a range of a slab synced with them (RgfxBuffer.h).
	return [RgfxBuffer bufferWithBytesNoCopy:pointer
	                                  length:length
	                                 options:options
	                             deallocator:deallocator
	                                  region:[self region]
	                                   slabs:[self slabs]
	                                  device:self
	                              serializer:[self serializer]];
}

// --- textures ----------------------------------------------------------------

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	// Where Metal's validation ends the process, so does the bundle, sending nothing; and none of
	// options with bit 18, as the x86 Macs make none (RgfxMakesNoResourceOf).
	rgfxEndAtInvalidTextureDescriptor(descriptor, self);
	if (descriptor != nil && RgfxMakesNoResourceOf(descriptor.resourceOptions))
		return nil;
	// A Managed texture's CPU copy is a Shared buffer of the slabs where the host reads a copy
	// wherever its buffer lies (RGFX_FEATURE_MANAGED_TEXTURE_SLABS), so the copies take the
	// process's own pages and not the region every process shares; else a region block. One of
	// a format or shape no CPU copy holds is made without one.
	if (descriptor != nil && descriptor.storageMode == MTLStorageModeManaged &&
	    [self region] != nullptr && [self serializer] != nullptr) {
		const bool inSlabs = ([self caps]->features() & RGFX_FEATURE_MANAGED_TEXTURE_SLABS) != 0;
		RgfxTexture *managed = [RgfxTexture managedTextureWithDescriptor:descriptor
		                                                          region:[self region]
		                                                           slabs:inSlabs ? [self slabs]
		                                                                         : nullptr
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
	// Where Metal ends the process, so does the bundle, sending nothing; else over this
	// process's copy of the plane in the device region, synced with the surface's pages
	// (RgfxIOSurface.h).
	rgfxEndAtInvalidSurfaceTexture(descriptor, iosurface, plane, self);
	return [RgfxIOSurfaceTexture textureWithDescriptor:descriptor
	                                         iosurface:iosurface
	                                             plane:plane
	                                            region:[self region]
	                                            device:self
	                                        serializer:[self serializer]];
}

// The same with a slice (measured: _MTLDevice's on macOS 27's x86-64 slice,
// `@48@0:8@16^{__IOSurface=}24Q32Q40`). On the M3 Max, slices 0, 1 and 2 of a one-plane surface
// each answer a texture over the surface's plane that reads what the plane's texture writes, so
// every slice answers the plane's texture.
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                 iosurface:(IOSurfaceRef)iosurface
                                     plane:(NSUInteger)plane
                                     slice:(NSUInteger)slice
{
	(void)slice;
	return [self newTextureWithDescriptor:descriptor iosurface:iosurface plane:plane];
}

// A tiled texture over the caller's bytes (measured: IOGPUMetalDevice's on macOS 27's x86-64 slice,
// `@64@0:8^v16Q24@?32@40Q48Q56`, and `@56@0:8^v16Q24@32Q40Q48` without the deallocator, which
// calls the first). The base makes a buffer of its own buffer class over the bytes and asks it
// for the texture (newTiledTextureWithDescriptor:offset:bytesPerRow:). Ours is the same over the
// device's no-copy buffer: a Shared one, as a tiled texture needs (RgfxBuffer.mm), with the
// descriptor's other resource options. The deallocator goes to that buffer, which calls it with
// the bytes and their length; one whose own signature takes other arguments is refused.
- (id<MTLTexture>)newTiledTextureWithBytesNoCopy:(void *)bytes
                                          length:(NSUInteger)length
                                     deallocator:(id)deallocator
                                      descriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
{
	if (deallocator != nil &&
	    ![RgfxBlockArgumentTypes(deallocator) isEqualToArray:@[ @"^v", @"Q" ]]) {
		RGFX_UNSUPPORTED("tiled textures with a deallocator of another type");
		return nil;
	}
	if (descriptor == nil)
		return nil;
	const MTLResourceOptions options =
	    (descriptor.resourceOptions & ~MTLResourceStorageModeMask) | MTLResourceStorageModeShared;
	id<MTLBuffer> buffer = [self newBufferWithBytesNoCopy:bytes
	                                               length:length
	                                              options:options
	                                          deallocator:deallocator];
	if (![(id)buffer isKindOfClass:[RgfxBuffer class]])
		return nil;
	return [(id<RgfxMetalBufferInternals>)buffer newTiledTextureWithDescriptor:descriptor
	                                                                   offset:offset
	                                                              bytesPerRow:bytesPerRow];
}

- (id<MTLTexture>)newTiledTextureWithBytesNoCopy:(void *)bytes
                                          length:(NSUInteger)length
                                      descriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
{
	return [self newTiledTextureWithBytesNoCopy:bytes
	                                     length:length
	                                deallocator:nil
	                                 descriptor:descriptor
	                                     offset:offset
	                                bytesPerRow:bytesPerRow];
}

// --- samplers ----------------------------------------------------------------

// An equal descriptor's sampler where the device holds it, as Metal's (RgfxStateCache.h); none
// where the wire handshake failed.
- (id<MTLSamplerState>)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)descriptor
{
	return [[self stateCache] samplerWithDescriptor:descriptor];
}

// --- depth/stencil state -----------------------------------------------------

// As newSamplerStateWithDescriptor:.
- (id<MTLDepthStencilState>)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
	return [[self stateCache] depthStencilStateWithDescriptor:descriptor];
}

// --- indirect command buffers ------------------------------------------------

// Made on the host where it makes them (INDIRECT_COMMAND_BUFFERS, which the kext takes for the
// bundle); nil where it does not, as before the wire carried them.
- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:
                                    (MTLIndirectCommandBufferDescriptor *)descriptor
                                                       maxCommandCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
	if (([self caps]->features() & RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS) == 0) {
		RGFX_UNSUPPORTED("indirect command buffers (the host offers none)");
		return nil;
	}
	// Slots where the host takes them, in blocks of the process's region.
	const bool slots = ([self caps]->features() & RGFX_FEATURE_ICB_SLOTS) != 0;
	return [RgfxIndirectCommandBuffer icbWithDescriptor:descriptor
	                                    maxCommandCount:maxCount
	                                            options:options
	                                             device:self
	                                         serializer:[self serializer]
	                                        slotsRegion:slots ? [self region] : nullptr];
}

// --- heaps -------------------------------------------------------------------

- (id<MTLHeap>)newHeapWithDescriptor:(MTLHeapDescriptor *)descriptor
{
	// A Shared heap lives in the device region so the host sees its sub-resources in place, or
	// is nil; any other storage mode is a device heap (createHeap).
	return [RgfxHeap heapForDeviceWithDescriptor:descriptor
	                                   maxLength:[self maxBufferLength]
	                                      region:[self region]
	                                      device:self
	                                  serializer:[self serializer]];
}

@end
