/*
 * create_tests: what each create answers where the device never gets its message (the kext's
 * submit refuses it, as it does past a limit), part of the Metal shell's check (shell_tests.mm
 * calls rgfxCreateTests; guest/bundle/Makefile `test`). Each is nil, as Metal answers for an
 * object it cannot make, and gives back what it took: no object is left on nothing, which the
 * host would never know, and a command buffer naming one would fail there.
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <vector>

#import "RgfxBuffer.h"
#import "RgfxDepthStencilState.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxHeap.h"
#import "RgfxIOSurface.h"
#import "RgfxSampler.h"
#import "RgfxSharedEvent.h"
#import "RgfxStateCache.h"
#import "RgfxTexture.h"

#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxCreateTests();

namespace {

// A device that takes the next `allow` messages and refuses every one after, as the kext's
// submit does past a limit, though it takes every destroy while `refuseDestroys` is false: a
// serialiser over a control ring that forwards each message. Never freed: objects a test lets go
// post their destroys through it after the test.
struct Refusing {
	uint32_t allow = 0;
	uint64_t seqno = 0;
	bool refuseDestroys = false;
	std::vector<uint32_t> taken;   // the types of the messages taken but destroys, in order
	std::vector<uint64_t> destroyed; // the objects of the destroys taken, in order
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxRingWriter &control = *new RgfxRingWriter(&Refusing::forward, this, 0x1000);
	RgfxSerializer &ser = *new RgfxSerializer(control, ids);

	static bool forward(void *context, const void *message, uint32_t size, uint64_t *seqno,
	                    uint32_t *status)
	{
		(void)size;
		(void)status;
		Refusing *r = static_cast<Refusing *>(context);
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		if (h.type != RGFX_CMD_OBJECT_DESTROY) {
			if (r->allow == 0)
				return false;
			r->allow--;
			r->taken.push_back(h.type);
		} else {
			if (r->refuseDestroys)
				return false;
			RgfxObjectDestroy d = {};
			std::memcpy(&d, message, sizeof(d));
			r->destroyed.push_back(d.obj_id);
		}
		*seqno = ++r->seqno;
		return true;
	}
};

// A region whose blocks are placed one after another past the floor, with those live. Never
// freed: objects a test lets go give theirs back after it.
struct Blocks {
	std::vector<uint8_t> mem;
	RgfxRegion region;
	uint64_t next = RGFX_ALLOCATIONS_OFFSET;
	std::vector<uint64_t> live;

	explicit Blocks(uint64_t bytes)
	    : mem(RGFX_ALLOCATIONS_OFFSET + bytes, 0), region(mem.data(), mem.size())
	{
		region.setBlockSource(&Blocks::alloc, &Blocks::release, this);
	}
	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		Blocks *b = static_cast<Blocks *>(c);
		const uint64_t at = (b->next + align - 1) & ~(align - 1);
		if (at + size > b->mem.size())
			return false;
		b->next = at + size;
		b->live.push_back(at);
		*offset = at;
		return true;
	}
	static void release(void *c, uint64_t offset)
	{
		std::vector<uint64_t> &live = static_cast<Blocks *>(c)->live;
		for (auto it = live.begin(); it != live.end(); ++it) {
			if (*it == offset) {
				live.erase(it);
				return;
			}
		}
		CHECK(false); // a block released twice, or never placed
	}
};

MTLTextureDescriptor *texture2D(MTLStorageMode storage)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:4
	                                                      height:4
	                                                   mipmapped:NO];
	td.storageMode = storage;
	return td;
}

// A device buffer, a Shared one in a slab of the region and one over a surface's pages: each nil,
// what it took back where it came from, and the caller's deallocator unrun (it is the caller's to
// run then). A Shared buffer whose slab's heap the device never got gives the slab's block back;
// one whose own create it never got gives its range back to a slab the class keeps.
void test_refused_buffer_create_is_nil()
{
	Refusing &r = *new Refusing();
	Blocks &b = *new Blocks(0x100000);
	RgfxRegionSlabSource &source = *new RgfxRegionSlabSource(b.region, r.ser);
	RgfxSlabs &slabs = *new RgfxSlabs(source);
	static uint8_t pages[0x1000];
	@autoreleasepool {
		r.allow = 4; // taken: each is made
		CHECK([RgfxBuffer bufferWithLength:0x1000
		                           options:MTLResourceStorageModePrivate
		                            device:nil
		                        serializer:&r.ser] != nil);
		CHECK([RgfxBuffer bufferForDeviceWithLength:0x1000
		                                    options:MTLResourceStorageModeShared
		                                      bytes:nullptr
		                                  maxLength:0
		                                      slabs:&slabs
		                                     device:nil
		                                 serializer:&r.ser] != nil);
		CHECK([RgfxBuffer bufferOverSurfacePagesWithContents:pages
		                                         blockOffset:RGFX_ALLOCATIONS_OFFSET
		                                           blockSize:sizeof(pages)
		                                              length:sizeof(pages)
		                                             options:MTLResourceStorageModeShared
		                                         deallocator:nil
		                                              device:nil
		                                          serializer:&r.ser] != nil);
		CHECK(r.taken == std::vector<uint32_t>({RGFX_CMD_BUFFER_CREATE,
		                                        RGFX_CMD_HEAP_OVER_HOST_MEMORY,
		                                        RGFX_CMD_BUFFER_AT_OFFSET,
		                                        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES}));
	}
	// The class keeps its slab, empty, for its next buffer.
	CHECK(b.live.size() == 1 && slabs.stats().emptySlabs == 1);
	RgfxSlabs &fresh = *new RgfxSlabs(source);
	@autoreleasepool {
		CHECK([RgfxBuffer bufferWithLength:0x1000
		                           options:MTLResourceStorageModePrivate
		                            device:nil
		                        serializer:&r.ser] == nil);
		CHECK([RgfxBuffer bufferForDeviceWithLength:0x1000
		                                    options:MTLResourceStorageModeShared
		                                      bytes:nullptr
		                                  maxLength:0
		                                      slabs:&slabs
		                                     device:nil
		                                 serializer:&r.ser] == nil);
		CHECK(slabs.stats().usedBytes == 0);
		CHECK([RgfxBuffer bufferForDeviceWithLength:0x1000
		                                    options:MTLResourceStorageModeShared
		                                      bytes:nullptr
		                                  maxLength:0
		                                      slabs:&fresh
		                                     device:nil
		                                 serializer:&r.ser] == nil);
		CHECK(b.live.size() == 1 && fresh.stats().slabs == 0);
		__block int letGo = 0;
		CHECK([RgfxBuffer bufferOverSurfacePagesWithContents:pages
		                                         blockOffset:RGFX_ALLOCATIONS_OFFSET
		                                           blockSize:sizeof(pages)
		                                              length:sizeof(pages)
		                                             options:MTLResourceStorageModeShared
		                                         deallocator:^{
			                                         letGo++;
		                                         }
		                                              device:nil
		                                          serializer:&r.ser] == nil);
		CHECK(letGo == 0);
	}
	CHECK(r.taken.size() == 4);
}

// A device heap is nil; a buffer at an offset of a Placement heap, and one an Automatic heap
// places, are nil, the Automatic heap's bytes free again.
void test_refused_heap_create_is_nil()
{
	Refusing &r = *new Refusing();
	Blocks &b = *new Blocks(0x100000);
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 0x10000;
	hd.storageMode = MTLStorageModePrivate;
	hd.type = MTLHeapTypePlacement;
	MTLHeapDescriptor *automatic = [hd copy];
	automatic.type = MTLHeapTypeAutomatic;
	@autoreleasepool {
		r.allow = 1;
		CHECK([RgfxHeap heapWithDescriptor:hd device:nil serializer:&r.ser] != nil);
	}
	CHECK([RgfxHeap heapWithDescriptor:hd device:nil serializer:&r.ser] == nil);
	CHECK(b.live.empty());

	r.allow = 2;
	RgfxHeap *placement = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&r.ser];
	RgfxHeap *placing = [RgfxHeap heapWithDescriptor:automatic device:nil serializer:&r.ser];
	CHECK(placement != nil && placing != nil);
	const NSUInteger room = [placing maxAvailableSizeWithAlignment:1];
	r.allow = 2;
	@autoreleasepool {
		CHECK([placement newBufferWithLength:0x1000
		                             options:MTLResourceStorageModePrivate
		                              offset:0] != nil);
		CHECK([placing newBufferWithLength:0x1000 options:MTLResourceStorageModePrivate] != nil);
	}
	CHECK([placement newBufferWithLength:0x1000 options:MTLResourceStorageModePrivate offset:0] ==
	      nil);
	CHECK([placing newBufferWithLength:0x1000 options:MTLResourceStorageModePrivate] == nil);
	CHECK([placing maxAvailableSizeWithAlignment:1] == room);
	CHECK(r.taken == std::vector<uint32_t>({RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_CREATE,
	                                        RGFX_CMD_HEAP_CREATE,
	                                        RGFX_CMD_BUFFER_AT_OFFSET,
	                                        RGFX_CMD_BUFFER_AT_OFFSET}));
}

// A texture of the device's, one over a buffer, a linear one, a view and a Managed one are nil,
// each giving back the resource id it took, and the Managed one its copy's block.
void test_refused_texture_create_is_nil()
{
	Refusing &r = *new Refusing();
	Blocks &b = *new Blocks(0x100000);
	r.allow = 2;
	RgfxBuffer *buffer = [RgfxBuffer bufferWithLength:0x1000
	                                          options:MTLResourceStorageModePrivate
	                                           device:nil
	                                       serializer:&r.ser];
	RgfxTexture *texture = [RgfxTexture textureWithDescriptor:texture2D(MTLStorageModePrivate)
	                                                     heap:RGFX_OBJ_NONE
	                                               heapOffset:0
	                                                   device:nil
	                                               serializer:&r.ser];
	CHECK(buffer != nil && texture != nil);
	auto make = [&](int kind) -> id<MTLTexture> {
		switch (kind) {
		case 0:
			return [RgfxTexture textureWithDescriptor:texture2D(MTLStorageModePrivate)
			                                     heap:RGFX_OBJ_NONE
			                               heapOffset:0
			                                   device:nil
			                               serializer:&r.ser];
		case 1:
			return [RgfxTexture textureWithDescriptor:texture2D(MTLStorageModePrivate)
			                                   buffer:buffer
			                                   offset:0
			                              bytesPerRow:256
			                                   device:nil
			                               serializer:&r.ser];
		case 2:
			return [RgfxTexture linearTextureWithDescriptor:texture2D(MTLStorageModePrivate)
			                                         buffer:buffer
			                                         offset:0
			                                    bytesPerRow:256
			                                  bytesPerImage:1024
			                                         device:nil
			                                     serializer:&r.ser];
		case 3:
			return [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
			                                  textureType:MTLTextureType2D
			                                       levels:NSMakeRange(0, 1)
			                                       slices:NSMakeRange(0, 1)];
		default:
			return [RgfxTexture managedTextureWithDescriptor:texture2D(MTLStorageModeManaged)
			                                          region:&b.region
			                                           slabs:nullptr
			                                          device:nil
			                                      serializer:&r.ser];
		}
	};
	for (int kind = 0; kind < 5; kind++) {
		// Taken, it is made; its messages counted, the last of them (the texture's) refused.
		const size_t before = r.taken.size();
		@autoreleasepool {
			r.allow = 8;
			CHECK(make(kind) != nil);
		}
		const uint32_t messages = static_cast<uint32_t>(r.taken.size() - before);
		CHECK(messages == (kind == 4 ? 2u : 1u));
		CHECK(b.live.empty());
		const uint64_t next = r.ids.allocResourceId();
		r.ids.freeResourceId(next);
		@autoreleasepool {
			r.allow = messages - 1;
			CHECK(make(kind) == nil);
			CHECK(r.allow == 0);
		}
		CHECK(r.ids.allocResourceId() == next);
		r.ids.freeResourceId(next);
		CHECK(b.live.empty());
	}
}

// A texture over a surface's plane is nil where its create is refused, after the buffer over the
// surface's block was taken.
void test_refused_iosurface_texture_create_is_nil()
{
	Refusing &r = *new Refusing();
	Blocks &b = *new Blocks(0x100000);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	auto make = [&](IOSurfaceRef surface) {
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                         iosurface:surface
		                                             plane:0
		                                            region:&b.region
		                                            device:nil
		                                        serializer:&r.ser];
	};
	IOSurfaceRef taken = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceRef refused = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(taken != nullptr && refused != nullptr);
	uint32_t messages = 0;
	@autoreleasepool {
		r.allow = 8;
		CHECK(make(taken) != nil);
		messages = static_cast<uint32_t>(r.taken.size());
		CHECK(messages >= 2 && r.taken.back() == RGFX_CMD_TEXTURE_FROM_IOSURFACE);
	}
	r.taken.clear();
	@autoreleasepool {
		r.allow = messages - 1;
		CHECK(make(refused) == nil);
		CHECK(r.allow == 0 && r.taken.size() == messages - 1);
	}
	CFRelease(taken);
	CFRelease(refused);
}

void test_refused_event_create_is_nil()
{
	Refusing &r = *new Refusing();
	r.allow = 1;
	CHECK([RgfxEvent eventWithDevice:nil serializer:&r.ser] != nil);
	CHECK([RgfxEvent eventWithDevice:nil serializer:&r.ser] == nil);
}

void test_refused_fence_create_is_nil()
{
	Refusing &r = *new Refusing();
	r.allow = 1;
	CHECK([RgfxFence fenceWithDevice:nil serializer:&r.ser] != nil);
	CHECK([RgfxFence fenceWithDevice:nil serializer:&r.ser] == nil);
}

void test_refused_shared_event_create_is_nil()
{
	Refusing &r = *new Refusing();
	r.allow = 1;
	CHECK([RgfxSharedEvent sharedEventWithInitialValue:0 device:nil serializer:&r.ser] != nil);
	CHECK([RgfxSharedEvent sharedEventWithInitialValue:0 device:nil serializer:&r.ser] == nil);
}

void test_refused_sampler_create_is_nil()
{
	Refusing &r = *new Refusing();
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	r.allow = 1;
	CHECK([RgfxSampler samplerWithDescriptor:sd device:nil serializer:&r.ser] != nil);
	CHECK([RgfxSampler samplerWithDescriptor:sd device:nil serializer:&r.ser] == nil);
}

void test_refused_depth_stencil_create_is_nil()
{
	Refusing &r = *new Refusing();
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	r.allow = 1;
	CHECK([RgfxDepthStencilState depthStencilStateWithDescriptor:dd device:nil serializer:&r.ser] !=
	      nil);
	CHECK([RgfxDepthStencilState depthStencilStateWithDescriptor:dd device:nil serializer:&r.ser] ==
	      nil);
}

// The device's state cache keeps nothing for a create the device never got: the next equal
// create is posted, and only an object made is answered again for an equal descriptor.
void test_refused_state_is_not_cached()
{
	Refusing &r = *new Refusing();
	RgfxStateCache *cache = [[RgfxStateCache alloc] initWithDevice:nil serializer:&r.ser];
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	@autoreleasepool {
		CHECK([cache samplerWithDescriptor:sd] == nil);
		CHECK([cache depthStencilStateWithDescriptor:dd] == nil);
		CHECK(r.taken.empty());
		r.allow = 2;
		id<MTLSamplerState> sampler = [cache samplerWithDescriptor:sd];
		id<MTLDepthStencilState> state = [cache depthStencilStateWithDescriptor:dd];
		CHECK(sampler != nil && state != nil);
		CHECK(r.taken == std::vector<uint32_t>(
		                     {RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_DEPTH_STENCIL_CREATE}));
		CHECK([cache samplerWithDescriptor:sd] == sampler);
		CHECK([cache depthStencilStateWithDescriptor:dd] == state);
		CHECK(r.taken.size() == 2);
	}
}

// Each create that picks a GPU address gives it back where the device never got its create: a
// device buffer, a heap of the device's and one over a region block, a buffer over a region
// block, a texture of the device's and a Managed one (its CPU copy's buffer taken, the texture
// refused). In a context whose addresses hold one pick, a device buffer made after the refusal
// finds its address.
void test_refused_create_gives_its_gpu_address_back()
{
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 0x1000;
	hd.storageMode = MTLStorageModePrivate;
	hd.type = MTLHeapTypePlacement;
	MTLHeapDescriptor *shared = [hd copy];
	shared.storageMode = MTLStorageModeShared;
	auto make = [&](Refusing &r, Blocks &b, int kind) -> id {
		switch (kind) {
		case 0:
			return [RgfxBuffer bufferWithLength:0x1000
			                            options:MTLResourceStorageModePrivate
			                             device:nil
			                         serializer:&r.ser];
		case 1:
			return [RgfxHeap heapWithDescriptor:hd device:nil serializer:&r.ser];
		case 2:
			return [RgfxHeap heapOverHostMemoryWithDescriptor:shared
			                                     regionOffset:RGFX_ALLOCATIONS_OFFSET
			                                         contents:b.mem.data() + RGFX_ALLOCATIONS_OFFSET
			                                      deallocator:nil
			                                           device:nil
			                                       serializer:&r.ser];
		case 3:
			return [RgfxBuffer bufferOverHostMemoryWithContents:b.mem.data() + RGFX_ALLOCATIONS_OFFSET
			                                       regionOffset:RGFX_ALLOCATIONS_OFFSET
			                                             length:0x1000
			                                            options:MTLResourceStorageModeShared
			                                        deallocator:nil
			                                             device:nil
			                                         serializer:&r.ser];
		case 4:
			return [RgfxTexture textureWithDescriptor:texture2D(MTLStorageModePrivate)
			                                     heap:RGFX_OBJ_NONE
			                               heapOffset:0
			                                   device:nil
			                               serializer:&r.ser];
		default:
			return [RgfxTexture managedTextureWithDescriptor:texture2D(MTLStorageModeManaged)
			                                          region:&b.region
			                                           slabs:nullptr
			                                          device:nil
			                                      serializer:&r.ser];
		}
	};
	for (int kind = 0; kind < 6; kind++) {
		Refusing &r = *new Refusing();
		r.ids.setBases(0, RGFX_CONTEXT_VA_BASE, RgfxIdSpace::kAddressGranule);
		Blocks &b = *new Blocks(0x100000);
		const uint32_t messages = kind == 5 ? 2 : 1;
		@autoreleasepool {
			r.allow = messages - 1;
			CHECK(make(r, b, kind) == nil);
			CHECK(r.allow == 0 && r.taken.size() == messages - 1);
		}
		@autoreleasepool {
			r.allow = 1;
			CHECK([RgfxBuffer bufferWithLength:0x1000
			                           options:MTLResourceStorageModePrivate
			                            device:nil
			                        serializer:&r.ser] != nil);
			CHECK(r.taken.back() == RGFX_CMD_BUFFER_CREATE);
		}
	}
}

// A destroy the device never gets waits, and what its object lay over stays held: in a context
// whose addresses hold one pick, a device buffer let go while destroys are refused leaves no
// address to the next buffer, nor a texture let go its resource id, while they wait; a commit's
// post of the waiting destroys (postWaitingDestroys) refused again keeps them waiting; once the
// device takes destroys, the next destroy posts them first, in the order they were made, and
// both go back.
void test_refused_destroy_waits()
{
	Refusing &r = *new Refusing();
	r.ids.setBases(0, RGFX_CONTEXT_VA_BASE, RgfxIdSpace::kAddressGranule);
	auto buffer = [&] {
		return [RgfxBuffer bufferWithLength:0x1000
		                            options:MTLResourceStorageModePrivate
		                             device:nil
		                         serializer:&r.ser];
	};
	const uint64_t resourceId = r.ids.allocResourceId();
	r.ids.freeResourceId(resourceId);
	RgfxBuffer *b = nil;
	RgfxTexture *t = nil;
	r.allow = 8;
	@autoreleasepool {
		b = buffer();
		t = [RgfxTexture textureWithDescriptor:texture2D(MTLStorageModePrivate)
		                                  heap:RGFX_OBJ_NONE
		                            heapOffset:0
		                                device:nil
		                            serializer:&r.ser];
	}
	CHECK(b != nil && t != nil);
	const uint64_t bufferObj = b.rgfxObjId, textureObj = t.rgfxObjId;
	CHECK(r.ser.postWaitingDestroys()); // none waits
	r.refuseDestroys = true;
	@autoreleasepool {
		t = nil;
	}
	@autoreleasepool {
		b = nil;
	}
	CHECK(r.destroyed.empty());
	@autoreleasepool {
		CHECK(buffer() == nil); // no address: the first buffer's is held
	}
	const uint64_t next = r.ids.allocResourceId();
	CHECK(next != resourceId); // held too
	r.ids.freeResourceId(next);
	CHECK(!r.ser.postWaitingDestroys() && r.destroyed.empty());

	r.refuseDestroys = false;
	uint64_t eventObj = 0;
	@autoreleasepool {
		RgfxEvent *event = [RgfxEvent eventWithDevice:nil serializer:&r.ser];
		CHECK(event != nil);
		eventObj = event.rgfxObjId;
	}
	CHECK(r.destroyed == std::vector<uint64_t>({textureObj, bufferObj, eventObj}));
	CHECK(r.ser.postWaitingDestroys());
	CHECK(r.ids.allocResourceId() == resourceId);
	r.ids.freeResourceId(resourceId);
	@autoreleasepool {
		CHECK(buffer() != nil);
	}
}

} // namespace

void rgfxCreateTests()
{
	test_refused_buffer_create_is_nil();
	test_refused_heap_create_is_nil();
	test_refused_texture_create_is_nil();
	test_refused_iosurface_texture_create_is_nil();
	test_refused_event_create_is_nil();
	test_refused_fence_create_is_nil();
	test_refused_shared_event_create_is_nil();
	test_refused_sampler_create_is_nil();
	test_refused_depth_stencil_create_is_nil();
	test_refused_state_is_not_cached();
	test_refused_create_gives_its_gpu_address_back();
	test_refused_destroy_waits();
}
