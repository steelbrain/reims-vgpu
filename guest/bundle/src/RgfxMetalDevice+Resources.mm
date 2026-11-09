/*
 * RgfxMetalDevice (Resources): see RgfxMetalDevice+Resources.h. The device's
 * MTLDevice resource creators, each a thin forwarder to a wrapper factory, routing
 * through `-serializer`. Each nil one answers is logged (noteNil). Our own ObjC++.
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

#import <objc/message.h>
#include <os/log.h>

#include <cstdio>
#include <mutex>
#include <set>
#include <string>

namespace {

// A nil the device's creator `creator` answers for `cause`, asked for `asked`: logged once at the
// info level (`log show --info`), by creator, cause and what was asked, so a process that ends at
// a nil it was given (14.6.1's WindowServer ends at a texture's, in SkyLight's
// MetalBacking::AbortWithTextureInfo) has the device's answer named beside it. A creator asked
// the same every frame logs once; the set is bounded, as a long session asks many shapes.
void noteNil(const char *creator, const char *cause, const std::string &asked)
{
	static std::mutex lock;
	static std::set<std::string> *seen = new std::set<std::string>(); // never destroyed
	{
		std::lock_guard<std::mutex> g(lock);
		if (seen->size() >= 4096)
			seen->clear();
		if (!seen->insert(std::string(creator) + "|" + cause + "|" + asked).second)
			return;
	}
	os_log_info(OS_LOG_DEFAULT,
	            "RGFX: the device answers nil at %{public}s (%{public}s): %{public}s", creator,
	            cause, asked.c_str());
}

// What a texture's descriptor asks, its protection options among it where the descriptor has
// them (MTLTextureDescriptorInternal's protectionOptions, read from 14.6.1's Metal).
std::string textureAsked(MTLTextureDescriptor *d)
{
	if (d == nil)
		return "no descriptor";
	unsigned long long protection = 0;
	if ([d respondsToSelector:@selector(protectionOptions)])
		protection =
		    ((unsigned long long (*)(id, SEL))objc_msgSend)(d, @selector(protectionOptions));
	char line[320];
	snprintf(line, sizeof(line),
	         "a texture of format %lu, type %lu, %lux%lux%lu, array length %lu, %lu levels, "
	         "%lu samples, usage 0x%lx, storage %lu, options 0x%lx, protection 0x%llx",
	         (unsigned long)d.pixelFormat, (unsigned long)d.textureType, (unsigned long)d.width,
	         (unsigned long)d.height, (unsigned long)d.depth, (unsigned long)d.arrayLength,
	         (unsigned long)d.mipmapLevelCount, (unsigned long)d.sampleCount,
	         (unsigned long)d.usage, (unsigned long)d.storageMode,
	         (unsigned long)d.resourceOptions, protection);
	return line;
}

// What a buffer's creator asks.
std::string bufferAsked(NSUInteger length, MTLResourceOptions options)
{
	char line[96];
	snprintf(line, sizeof(line), "a buffer of %lu bytes, options 0x%lx", (unsigned long)length,
	         (unsigned long)options);
	return line;
}

// Why a device buffer's factory answered nil for `length` and `options` (RgfxBuffer's
// bufferForDeviceWithLength:..., which says no more).
const char *bufferCause(NSUInteger length, MTLResourceOptions options, NSUInteger maxLength)
{
	if (length == 0)
		return "no bytes";
	if (maxLength != 0 && length > maxLength)
		return "more bytes than maxBufferLength";
	if (RgfxMakesNoResourceOf(options))
		return "options with bit 18";
	return "no room in the slabs, or the device never got the create";
}

} // namespace

@implementation RgfxMetalDevice (Resources)

// --- buffers -----------------------------------------------------------------

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// A Shared buffer in a slab of the device region, which the host sees in place, a Managed
	// one with its CPU copy there, a Private one the GPU's alone; none where Metal makes none, nor
	// where the region has no room (RgfxBuffer.h).
	id<MTLBuffer> buffer = [RgfxBuffer bufferForDeviceWithLength:length
	                                                     options:options
	                                                       bytes:nullptr
	                                                   maxLength:[self maxBufferLength]
	                                                       slabs:[self slabs]
	                                                      device:self
	                                                  serializer:[self serializer]];
	if (buffer == nil)
		noteNil("newBufferWithLength:options:",
		        bufferCause(length, options, [self maxBufferLength]), bufferAsked(length, options));
	return buffer;
}

- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer
                             length:(NSUInteger)length
                            options:(MTLResourceOptions)options
{
	// As newBufferWithLength:options:, its range seeded with the bytes; none for a Private one,
	// whose bytes Metal's validation refuses, nor for no bytes at all, which Metal's validation
	// refuses too and its copy faults at (measured on the M3 Max).
	if (pointer == nullptr) {
		noteNil("newBufferWithBytes:length:options:", "no bytes", bufferAsked(length, options));
		return nil;
	}
	id<MTLBuffer> buffer = [RgfxBuffer bufferForDeviceWithLength:length
	                                                     options:options
	                                                       bytes:pointer
	                                                   maxLength:[self maxBufferLength]
	                                                       slabs:[self slabs]
	                                                      device:self
	                                                  serializer:[self serializer]];
	if (buffer == nil)
		noteNil("newBufferWithBytes:length:options:",
		        (options & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate
		            ? "bytes for a Private buffer"
		            : bufferCause(length, options, [self maxBufferLength]),
		        bufferAsked(length, options));
	return buffer;
}

- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer
                                   length:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                              deallocator:(void (^)(void *, NSUInteger))deallocator
{
	// Over the caller's bytes in place where they lie in a block of ours or the kext wires their
	// pages, else over a range of a slab synced with them (RgfxBuffer.h).
	id<MTLBuffer> buffer = [RgfxBuffer bufferWithBytesNoCopy:pointer
	                                                  length:length
	                                                 options:options
	                                             deallocator:deallocator
	                                                  region:[self region]
	                                                   slabs:[self slabs]
	                                                  device:self
	                                              serializer:[self serializer]];
	if (buffer == nil)
		noteNil("newBufferWithBytesNoCopy:length:options:deallocator:",
		        "the buffer's factory made none", bufferAsked(length, options));
	return buffer;
}

// --- textures ----------------------------------------------------------------

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	// Where Metal's validation ends the process, so does the bundle, sending nothing; and none of
	// options with bit 18, as the x86 Macs make none (RgfxMakesNoResourceOf).
	rgfxEndAtInvalidTextureDescriptor(descriptor, self);
	const char *creator = "newTextureWithDescriptor:";
	if (descriptor != nil && RgfxMakesNoResourceOf(descriptor.resourceOptions)) {
		noteNil(creator, "options with bit 18", textureAsked(descriptor));
		return nil;
	}
	// None of a two-plane format but the XR10 ones, or Unspecialized, on an Apple GPU
	// (rgfxDeviceMakesNoTextureOf).
	if (rgfxDeviceMakesNoTextureOf(descriptor, self)) {
		noteNil(creator, "a two-plane format or Unspecialized on an Apple GPU",
		        textureAsked(descriptor));
		return nil;
	}
	// None of more bytes than maxBufferLength, as the Intel 630 makes none
	// (rgfxTexturePastLength).
	if (descriptor != nil && rgfxTexturePastLength(descriptor, [self maxBufferLength])) {
		noteNil(creator, "more bytes than maxBufferLength", textureAsked(descriptor));
		return nil;
	}
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
	RgfxTexture *texture = [RgfxTexture textureWithDescriptor:descriptor
	                                                     heap:RGFX_OBJ_NONE
	                                               heapOffset:0
	                                                   device:self
	                                               serializer:[self serializer]];
	if (texture == nil)
		noteNil(creator, "no route, no resource id left, or the device never got the create",
		        textureAsked(descriptor));
	return texture;
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
	const char *creator = "newTiledTextureWithBytesNoCopy:length:deallocator:descriptor:offset:"
	                      "bytesPerRow:";
	if (deallocator != nil &&
	    ![RgfxBlockArgumentTypes(deallocator) isEqualToArray:@[ @"^v", @"Q" ]]) {
		RGFX_UNSUPPORTED("tiled textures with a deallocator of another type");
		noteNil(creator, "a deallocator of another type", textureAsked(descriptor));
		return nil;
	}
	if (descriptor == nil) {
		noteNil(creator, "no descriptor", textureAsked(descriptor));
		return nil;
	}
	const MTLResourceOptions options =
	    (descriptor.resourceOptions & ~MTLResourceStorageModeMask) | MTLResourceStorageModeShared;
	id<MTLBuffer> buffer = [self newBufferWithBytesNoCopy:bytes
	                                               length:length
	                                              options:options
	                                          deallocator:deallocator];
	if (![(id)buffer isKindOfClass:[RgfxBuffer class]]) {
		noteNil(creator, "no buffer over the bytes", textureAsked(descriptor));
		return nil;
	}
	id<MTLTexture> texture =
	    [(id<RgfxMetalBufferInternals>)buffer newTiledTextureWithDescriptor:descriptor
	                                                                 offset:offset
	                                                            bytesPerRow:bytesPerRow];
	if (texture == nil)
		noteNil(creator, "the buffer's tiled texture creator made none", textureAsked(descriptor));
	return texture;
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
	id<MTLSamplerState> state = [[self stateCache] samplerWithDescriptor:descriptor];
	if (state == nil)
		noteNil("newSamplerStateWithDescriptor:", "no state made", "a sampler");
	return state;
}

// --- depth/stencil state -----------------------------------------------------

// As newSamplerStateWithDescriptor:.
- (id<MTLDepthStencilState>)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
	id<MTLDepthStencilState> state = [[self stateCache] depthStencilStateWithDescriptor:descriptor];
	if (state == nil)
		noteNil("newDepthStencilStateWithDescriptor:", "no state made", "a depth-stencil state");
	return state;
}

// --- indirect command buffers ------------------------------------------------

// Made on the host where it makes them (INDIRECT_COMMAND_BUFFERS, which the kext takes for the
// bundle); nil where it does not, as before the wire carried them.
- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:
                                    (MTLIndirectCommandBufferDescriptor *)descriptor
                                                       maxCommandCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
	const char *creator = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";
	char asked[96];
	snprintf(asked, sizeof(asked), "an indirect command buffer of %lu commands, options 0x%lx",
	         (unsigned long)maxCount, (unsigned long)options);
	if (([self caps]->features() & RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS) == 0) {
		RGFX_UNSUPPORTED("indirect command buffers (the host offers none)");
		noteNil(creator, "the host offers none", asked);
		return nil;
	}
	// Slots where the host takes them, in blocks of the process's region.
	const bool slots = ([self caps]->features() & RGFX_FEATURE_ICB_SLOTS) != 0;
	id<MTLIndirectCommandBuffer> icb =
	    [RgfxIndirectCommandBuffer icbWithDescriptor:descriptor
	                                 maxCommandCount:maxCount
	                                         options:options
	                                          device:self
	                                      serializer:[self serializer]
	                                     slotsRegion:slots ? [self region] : nullptr];
	if (icb == nil)
		noteNil(creator, "the factory made none", asked);
	return icb;
}

// --- heaps -------------------------------------------------------------------

- (id<MTLHeap>)newHeapWithDescriptor:(MTLHeapDescriptor *)descriptor
{
	// A Shared heap lives in the device region so the host sees its sub-resources in place, or
	// is nil; any other storage mode is a device heap (createHeap).
	id<MTLHeap> heap = [RgfxHeap heapForDeviceWithDescriptor:descriptor
	                                               maxLength:[self maxBufferLength]
	                                                  region:[self region]
	                                                  device:self
	                                              serializer:[self serializer]];
	if (heap == nil) {
		char asked[128];
		snprintf(asked, sizeof(asked), "a heap of %lu bytes, type %ld, storage %lu",
		         (unsigned long)descriptor.size, (long)descriptor.type,
		         (unsigned long)descriptor.storageMode);
		noteNil("newHeapWithDescriptor:", "the factory made none", asked);
	}
	return heap;
}

@end
