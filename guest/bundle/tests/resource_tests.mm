/*
 * resource_tests: the shell check of what the resource wrappers answer past Metal's public
 * protocols (RgfxMetalPrivate.h), which Apple's frameworks call (built into shell_tests, which
 * calls rgfxResourceTests()). Each answer checked here is Metal's on the M3 Max (macOS 27), for
 * the kind of object the wrapper stands for.
 *
 * Our own ObjC++; built on the Mac's native arch with the NSObject-based wrappers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <IOSurface/IOSurface.h>

#import "RgfxBuffer.h"
#import "RgfxEvent.h"
#import "RgfxHeap.h"
#import "RgfxIOSurface.h"
#import "RgfxMetalPrivate.h"
#import "RgfxSampler.h"
#import "RgfxSharedEvent.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

void rgfxResourceTests();
// shell_tests.mm's control ring helpers.
bool ringEmpty(TestRing &ring);
void drainRing(TestRing &ring);

namespace {

// A ring, an id space and a serialiser, as shell_tests.mm's fixture. The transport is never freed:
// an object a test lets go may dealloc after the test and post its destroy here.
struct Fixture {
	TestRing &ring;
	RgfxRingWriter &writer;
	RgfxIdSpace &ids;
	RgfxSerializer &ser;

	Fixture()
	    : ring(*new TestRing(0x8000)),
	      writer(*new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID,
	                                 TestRing::doorbell, nullptr)),
	      ids(*new RgfxIdSpace()), ser(*new RgfxSerializer(writer, ids))
	{
		ring.setConsumerIdle(1);
	}
};

MTLHeapDescriptor *heapDescriptor(MTLStorageMode storage, MTLHeapType type)
{
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = 0x100000;
	d.storageMode = storage;
	d.type = type;
	return d;
}

// The heap's members past MTLHeap.
id<RgfxMetalHeapInternals> internals(RgfxHeap *heap)
{
	return (id<RgfxMetalHeapInternals>)heap;
}

// A new MTLBufferDescriptor of `length` bytes and `options` (Metal's own class, at run time).
id bufferDescriptor(NSUInteger length, MTLResourceOptions options)
{
	id d = [NSClassFromString(@"MTLBufferDescriptor") new];
	CHECK(d != nil);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setLength:), length);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setResourceOptions:), options);
	return d;
}

// A heap answers its base address, its options with a Default hazard tracking mode resolved to
// Untracked (unfiltered the same), and keeps the purgeable state it is told, as Metal's.
void test_heap_answers()
{
	Fixture fx;
	@autoreleasepool {
		RgfxHeap *priv = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypePlacement)
		                                       device:nil
		                                   serializer:&fx.ser];
		auto made = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		CHECK(made.gpu_address != 0 && [internals(priv) gpuAddress] == made.gpu_address);
		// The wire carries the options as asked; the heap answers them resolved.
		CHECK(made.options == MTLResourceStorageModePrivate);
		CHECK(priv.resourceOptions == 0x120 && [internals(priv) unfilteredResourceOptions] == 0x120);
		CHECK(priv.hazardTrackingMode == MTLHazardTrackingModeUntracked);
		CHECK(priv.storageMode == MTLStorageModePrivate);
		// A buffer placed at 0 lies at the heap's base.
		id<MTLBuffer> at0 = [priv newBufferWithLength:0x100
		                                      options:MTLResourceStorageModePrivate
		                                       offset:0];
		CHECK(at0.gpuAddress == [internals(priv) gpuAddress]);

		MTLHeapDescriptor *sd = heapDescriptor(MTLStorageModeShared, MTLHeapTypeAutomatic);
		RgfxHeap *shared = [RgfxHeap heapWithDescriptor:sd device:nil serializer:&fx.ser];
		CHECK(shared.resourceOptions == 0x100 && [internals(shared) unfilteredResourceOptions] == 0x100);
		sd.hazardTrackingMode = MTLHazardTrackingModeTracked;
		RgfxHeap *tracked = [RgfxHeap heapWithDescriptor:sd device:nil serializer:&fx.ser];
		CHECK(tracked.resourceOptions == 0x200 && [internals(tracked) unfilteredResourceOptions] == 0x200);
		CHECK(tracked.hazardTrackingMode == MTLHazardTrackingModeTracked);

		// The state is kept and the previous one answered.
		CHECK([priv setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateEmpty] == MTLPurgeableStateVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateEmpty);
		CHECK([priv setPurgeableState:MTLPurgeableStateKeepCurrent] ==
		      MTLPurgeableStateNonVolatile);

		// No acceleration structure: nil, logged, nothing posted.
		drainRing(fx.ring);
		CHECK([internals(priv) newAccelerationStructureWithSize:0x1000 resourceIndex:0] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxHeap class],
		                               @selector(newAccelerationStructureWithSize:resourceIndex:)));
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// An Automatic heap places a buffer of a descriptor's length as it places one of a length, at the
// descriptor's alignment where it is larger, with the heap's options whatever the descriptor asks;
// a Placement heap, a length of 0, no descriptor and a descriptor over memory of its own answer
// nil, posting nothing.
void test_heap_buffer_descriptors()
{
	Fixture fx;
	@autoreleasepool {
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypeAutomatic)
		                                       device:nil
		                                   serializer:&fx.ser];
		auto made = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		auto placed = [&](uint64_t offset, uint64_t length) {
			auto m = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
			CHECK(m.heap_obj_id == heap.rgfxObjId && m.offset == offset && m.length == length);
			// Placed with the heap's options, so the host's heap takes it.
			CHECK(m.options == MTLResourceStorageModePrivate);
		};
		id<MTLBuffer> a = [heap newBufferWithLength:100 options:MTLResourceStorageModePrivate];
		placed(0, 100);
		// Shared options on a Private heap still answer a buffer, placed as one of the heap's.
		id<MTLBuffer> b = [internals(heap) newBufferWithDescriptor:bufferDescriptor(100, 0)];
		placed(256, 100);
		CHECK(b != nil && b.heap == heap && b.length == 100 && b.heapOffset == 0);
		CHECK(b.gpuAddress == made.gpu_address + 256);
		CHECK(b.storageMode == MTLStorageModePrivate && b.contents == nullptr);
		id d = bufferDescriptor(100, MTLResourceStorageModePrivate);
		((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setAlignment:), 0x10000);
		id<MTLBuffer> c = [internals(heap) newBufferWithDescriptor:d];
		placed(0x10000, 100);
		CHECK(c.gpuAddress == made.gpu_address + 0x10000 && heap.usedSize == 300);

		CHECK([internals(heap) newBufferWithDescriptor:bufferDescriptor(0, 0)] == nil);
		CHECK([internals(heap) newBufferWithDescriptor:nil] == nil);
		CHECK(ringEmpty(fx.ring));
		id own = bufferDescriptor(100, 0);
		((void (*)(id, SEL, uint64_t))objc_msgSend)(own, @selector(setPinnedGPUAddress:), 0x1000);
		CHECK([internals(heap) newBufferWithDescriptor:own] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newBufferWithDescriptor:)));
		CHECK(ringEmpty(fx.ring));

		RgfxHeap *placement = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                                  MTLHeapTypePlacement)
		                                            device:nil
		                                        serializer:&fx.ser];
		drainRing(fx.ring);
		CHECK([internals(placement) newBufferWithDescriptor:bufferDescriptor(100, 0)] == nil);
		CHECK(ringEmpty(fx.ring));
		(void)a;
	}
	drainRing(fx.ring);
}

// A buffer's members past MTLBuffer.
id<RgfxMetalBufferInternals> internals(id<MTLBuffer> buffer)
{
	return (id<RgfxMetalBufferInternals>)buffer;
}

BOOL alias(id<MTLBuffer> a, id<MTLResource> b)
{
	return [internals(a) doesAliasResource:b];
}

// A buffer in no heap answers its options resolved and as asked, an allocation of its own of its
// own size, its contents as its CPU address, NULL for a Private one, no IOSurface, its purgeable
// state, that it is complete, and that it aliases nothing.
void test_buffer_answers()
{
	Fixture fx;
	@autoreleasepool {
		auto make = [&](MTLResourceOptions options) {
			return [RgfxBuffer bufferWithLength:0x1000 options:options device:nil serializer:&fx.ser];
		};
		RgfxBuffer *plain = make(0);
		RgfxBuffer *untracked = make(MTLResourceHazardTrackingModeUntracked);
		RgfxBuffer *priv = make(MTLResourceStorageModePrivate);
		RgfxBuffer *combined = make(MTLResourceCPUCacheModeWriteCombined);
		const struct {
			RgfxBuffer *buffer;
			MTLResourceOptions asked, answered;
		} cases[] = {{plain, 0, 0x200}, {untracked, 0x100, 0x100}, {priv, 0x20, 0x220},
		             {combined, 0x1, 0x201}};
		for (const auto &c : cases) {
			CHECK(c.buffer.resourceOptions == c.answered);
			CHECK([internals(c.buffer) unfilteredResourceOptions] == c.asked);
		}
		CHECK(plain.hazardTrackingMode == MTLHazardTrackingModeTracked);
		CHECK(untracked.hazardTrackingMode == MTLHazardTrackingModeUntracked);
		CHECK(priv.storageMode == MTLStorageModePrivate &&
		      combined.cpuCacheMode == MTLCPUCacheModeWriteCombined);

		const uint64_t id0 = [internals(plain) allocationID], id1 = [internals(priv) allocationID];
		CHECK(id0 != 0 && id1 != 0 && id0 != id1);
		CHECK([internals(plain) resourceSize] == 0x1000 && [internals(plain) iosurface] == nullptr);
		CHECK(plain.contents != nullptr && [internals(plain) virtualAddress] == plain.contents);
		CHECK([internals(priv) virtualAddress] == nullptr);

		CHECK([internals(plain) isComplete] && [internals(plain) isWriteComplete]);
		[internals(plain) waitUntilComplete];
		CHECK(![internals(plain) isPurgeable]);
		[plain setPurgeableState:MTLPurgeableStateVolatile];
		CHECK([internals(plain) isPurgeable]);
		[plain setPurgeableState:MTLPurgeableStateEmpty];
		CHECK([internals(plain) isPurgeable]);
		[plain setPurgeableState:MTLPurgeableStateNonVolatile];
		CHECK(![internals(plain) isPurgeable]);

		CHECK(!alias(plain, plain) && !alias(plain, priv) && !alias(plain, nil));
		__unsafe_unretained id<MTLResource> list[] = {plain};
		CHECK([internals(plain) doesAliasAllResources:list count:0]);
		CHECK(![internals(plain) doesAliasAnyResources:list count:0]);
		CHECK(![internals(plain) doesAliasAllResources:list count:1]);
	}
	drainRing(fx.ring);
}

// A heap's buffers answer the heap's options, allocation and size, the heap's CPU address plus
// their offset (the offset itself in a Private heap), alias where their bytes meet in one heap, and
// stay NonVolatile.
void test_heap_buffers_answer_their_heap()
{
	Fixture fx;
	std::vector<uint8_t> bytes(0x100000);
	@autoreleasepool {
		MTLHeapDescriptor *pd = heapDescriptor(MTLStorageModePrivate, MTLHeapTypePlacement);
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:pd device:nil serializer:&fx.ser];
		RgfxHeap *other = [RgfxHeap heapWithDescriptor:pd device:nil serializer:&fx.ser];
		const MTLResourceOptions o = MTLResourceStorageModePrivate;
		id<MTLBuffer> b0 = [heap newBufferWithLength:0x1000 options:o offset:0];
		id<MTLBuffer> b1 = [heap newBufferWithLength:0x1000 options:o offset:0x800];
		id<MTLBuffer> b2 = [heap newBufferWithLength:0x1000 options:o offset:0x2000];
		id<MTLBuffer> b3 = [heap newBufferWithLength:0x1000
		                                     options:o | MTLResourceHazardTrackingModeTracked
		                                      offset:0x1000];
		id<MTLBuffer> o0 = [other newBufferWithLength:0x1000 options:o offset:0];
		CHECK(b3.resourceOptions == 0x120 && [internals(b3) unfilteredResourceOptions] == 0x120);
		CHECK(b3.hazardTrackingMode == MTLHazardTrackingModeUntracked);
		CHECK([internals(b0) allocationID] == [internals(b2) allocationID]);
		CHECK([internals(b0) allocationID] != [internals(o0) allocationID]);
		CHECK([internals(b2) resourceSize] == heap.size);
		CHECK([internals(b0) virtualAddress] == nullptr);
		CHECK([internals(b1) virtualAddress] == reinterpret_cast<void *>(0x800));
		CHECK([internals(b2) virtualAddress] == reinterpret_cast<void *>(0x2000));

		// Bytes that meet alias, and bytes that only touch do not.
		CHECK(alias(b0, b1) && alias(b1, b0) && alias(b0, b0));
		CHECK(!alias(b0, b2) && !alias(b1, b2) && !alias(b0, b3) && !alias(b0, o0));
		__unsafe_unretained id<MTLResource> apart[] = {b1, b2};
		__unsafe_unretained id<MTLResource> over[] = {b1, b0};
		CHECK(![internals(b0) doesAliasAllResources:apart count:2]);
		CHECK([internals(b0) doesAliasAnyResources:apart count:2]);
		CHECK([internals(b0) doesAliasAllResources:over count:2]);

		CHECK([b0 setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK(![internals(b0) isPurgeable]);
		CHECK([b0 setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateNonVolatile);

		// An Automatic heap's buffers meet only once one is made aliasable and its bytes taken again.
		RgfxHeap *automatic = [RgfxHeap
		    heapWithDescriptor:heapDescriptor(MTLStorageModePrivate, MTLHeapTypeAutomatic)
		                device:nil
		            serializer:&fx.ser];
		id<MTLBuffer> a0 = [automatic newBufferWithLength:0x2000 options:o];
		id<MTLBuffer> a1 = [automatic newBufferWithLength:0x2000 options:o];
		CHECK(!alias(a0, a1) && [internals(a1) virtualAddress] == reinterpret_cast<void *>(0x2000));
		[a0 makeAliasable];
		id<MTLBuffer> a2 = [automatic newBufferWithLength:0x2000 options:o];
		CHECK(a2.gpuAddress == a0.gpuAddress && alias(a0, a2) && alias(a2, a0) && !alias(a1, a2));

		// A Shared heap's buffer lies at the heap's bytes plus its offset.
		RgfxHeap *shared = [RgfxHeap
		    heapOverHostMemoryWithDescriptor:heapDescriptor(MTLStorageModeShared,
		                                                    MTLHeapTypePlacement)
		                        regionOffset:0
		                            contents:bytes.data()
		                         deallocator:nil
		                              device:nil
		                          serializer:&fx.ser];
		id<MTLBuffer> s0 = [shared newBufferWithLength:0x1000 options:0 offset:0x4000];
		CHECK(s0.contents == bytes.data() + 0x4000 && [internals(s0) virtualAddress] == s0.contents);
		CHECK(s0.resourceOptions == 0x100 && [internals(s0) unfilteredResourceOptions] == 0x100);
	}
	drainRing(fx.ring);
}

// A linear texture is made by Metal's linear form on the host (LINEAR_TEXTURE_FROM_BUFFER), its
// bytes per image carried as asked, a 2D array's and a texture buffer's among them, and answers
// its buffer, offset and rows as a texture over a buffer does; a type Metal's validation ends the
// process at, or more than one level, is nil, logged, posting nothing.
void test_buffer_linear_textures()
{
	Fixture fx;
	@autoreleasepool {
		RgfxBuffer *buffer = [RgfxBuffer bufferWithLength:0x10000
		                                          options:0
		                                           device:nil
		                                       serializer:&fx.ser];
		drainRing(fx.ring);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		auto linear = [&](MTLTextureDescriptor *d, uint64_t offset, uint64_t bpr, uint64_t bpi) {
			return [internals(buffer) newLinearTextureWithDescriptor:d
			                                                  offset:offset
			                                             bytesPerRow:bpr
			                                           bytesPerImage:bpi];
		};
		auto posted = [&](uint64_t offset, uint64_t bpr, uint64_t bpi, MTLTextureType type) {
			auto m = expectMessage<RgfxLinearTextureFromBuffer>(fx.ring,
			                                                    RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER);
			CHECK(m.buffer_obj_id == buffer.rgfxObjId && m.offset == offset);
			CHECK(m.bytes_per_row == bpr && m.bytes_per_image == bpi);
			CHECK(m.desc.texture_type == static_cast<uint32_t>(type) && m.resource_id != 0);
		};
		id<MTLTexture> t = linear(td, 0x1000, 256, 99999);
		posted(0x1000, 256, 99999, MTLTextureType2D);
		CHECK(t != nil && t.buffer == buffer && t.bufferOffset == 0x1000);
		CHECK(t.bufferBytesPerRow == 256 && t.textureType == MTLTextureType2D);
		td.textureType = MTLTextureType2DArray;
		td.arrayLength = 3;
		id<MTLTexture> array = linear(td, 0x2000, 256, 0x1000);
		posted(0x2000, 256, 0x1000, MTLTextureType2DArray);
		CHECK(array != nil && array.arrayLength == 3 && array.buffer == buffer);
		MTLTextureDescriptor *tb =
		    [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                           width:64
		                                                 resourceOptions:0
		                                                           usage:MTLTextureUsageShaderRead];
		id<MTLTexture> texels = linear(tb, 0, 256, 0);
		posted(0, 256, 0, MTLTextureTypeTextureBuffer);
		CHECK(texels != nil && texels.textureType == MTLTextureTypeTextureBuffer);

		td.arrayLength = 1;
		for (MTLTextureType type : {MTLTextureType3D, MTLTextureType1D, MTLTextureType1DArray,
		                            MTLTextureTypeCube, MTLTextureType2DMultisample}) {
			td.textureType = type;
			CHECK(linear(td, 0, 256, 0x1000) == nil);
		}
		td.textureType = MTLTextureType2D;
		td.mipmapLevelCount = 2;
		CHECK(linear(td, 0, 256, 0) == nil);
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// A texture's members past MTLTexture.
id<RgfxMetalTextureInternals> internals(id<MTLTexture> texture)
{
	return (id<RgfxMetalTextureInternals>)texture;
}

MTLTextureDescriptor *texture2D(MTLPixelFormat format, NSUInteger width, NSUInteger height,
                                MTLTextureUsage usage)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                             width:width
	                                                                            height:height
	                                                                         mipmapped:NO];
	d.usage = usage;
	d.storageMode = MTLStorageModePrivate;
	return d;
}

// isCompressed follows Metal's rule, case by case as measured: the usage, the size, the format,
// the storage, GPU-optimized contents, and the memory a texture lies in; a view answers its
// texture's.
void test_texture_compression()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0);
	RgfxRegion region(mem.data(), mem.size());
	@autoreleasepool {
		auto compressed = [&](MTLTextureDescriptor *d) {
			RgfxTexture *t = [RgfxTexture textureWithDescriptor:d
			                                               heap:RGFX_OBJ_NONE
			                                         heapOffset:0
			                                             device:nil
			                                         serializer:&fx.ser];
			CHECK(t != nil);
			return [internals(t) isCompressed];
		};
		const MTLTextureUsage read = MTLTextureUsageShaderRead;
		const MTLTextureUsage target = MTLTextureUsageRenderTarget;
		const struct {
			MTLPixelFormat format;
			NSUInteger width, height;
			MTLTextureUsage usage;
			BOOL answer;
		} cases[] = {
		    {MTLPixelFormatBGRA8Unorm, 16, 16, read, YES},
		    {MTLPixelFormatBGRA8Unorm, 15, 16, read, NO},
		    {MTLPixelFormatBGRA8Unorm, 16, 15, read, NO},
		    {MTLPixelFormatBGRA8Unorm, 64, 64, target, YES},
		    {MTLPixelFormatBGRA8Unorm, 64, 64, target | read, YES},
		    {MTLPixelFormatBGRA8Unorm, 64, 64, read | MTLTextureUsageShaderWrite, NO},
		    {MTLPixelFormatBGRA8Unorm, 64, 64, read | MTLTextureUsagePixelFormatView, NO},
		    {MTLPixelFormatR32Uint, 64, 64, read | MTLTextureUsageShaderAtomic, NO},
		    {MTLPixelFormatBGRA8Unorm, 64, 64, MTLTextureUsageUnknown, NO},
		    {MTLPixelFormatBC1_RGBA, 64, 64, read, NO},
		    {MTLPixelFormatASTC_4x4_LDR, 64, 64, read, NO},
		    {MTLPixelFormatETC2_RGB8, 64, 64, read, NO},
		    {MTLPixelFormatEAC_R11Unorm, 64, 64, read, NO},
		    {MTLPixelFormatPVRTC_RGBA_4BPP, 64, 64, read, NO},
		    {MTLPixelFormatGBGR422, 64, 64, read, NO},
		    {MTLPixelFormatBGRG422, 64, 64, read, NO},
		    {MTLPixelFormatDepth32Float, 64, 64, target, YES},
		    {MTLPixelFormatStencil8, 64, 64, target, YES},
		    {MTLPixelFormatDepth32Float_Stencil8, 64, 64, target, YES},
		    {MTLPixelFormatBGR10_XR, 64, 64, read, YES},
		    {MTLPixelFormatRGB9E5Float, 64, 64, read, YES},
		    {MTLPixelFormatR8Unorm, 64, 64, read, YES},
		    {MTLPixelFormatA8Unorm, 64, 64, read, YES},
		    {MTLPixelFormatRGBA32Float, 64, 64, read, YES},
		};
		for (const auto &c : cases) {
			CHECK(compressed(texture2D(c.format, c.width, c.height, c.usage)) == c.answer);
		}

		MTLTextureDescriptor *d = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		d.compressionType = MTLTextureCompressionTypeLossy;
		CHECK(compressed(d));
		d.compressionType = MTLTextureCompressionTypeLossless;
		d.allowGPUOptimizedContents = NO;
		CHECK(!compressed(d));
		d.allowGPUOptimizedContents = YES;
		for (MTLStorageMode storage : {MTLStorageModeShared, MTLStorageModeManaged}) {
			d.storageMode = storage;
			CHECK(compressed(d));
		}
		d.storageMode = MTLStorageModeMemoryless;
		d.usage = target;
		CHECK(!compressed(d));
		d.storageMode = MTLStorageModePrivate;
		d.usage = read;
		d.textureType = MTLTextureType3D;
		d.width = d.height = 16;
		d.depth = 4;
		CHECK(compressed(d));
		d.width = d.height = 8;
		d.depth = 64;
		CHECK(!compressed(d));
		d.textureType = MTLTextureType1D;
		d.width = 64;
		d.height = d.depth = 1;
		CHECK(!compressed(d));
		d.textureType = MTLTextureType2DMultisample;
		d.width = d.height = 64;
		d.sampleCount = 4;
		d.usage = target;
		CHECK(compressed(d));
		d = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		d.mipmapLevelCount = 7;
		CHECK(compressed(d));
		d.mipmapLevelCount = 1;
		d.textureType = MTLTextureTypeCube;
		CHECK(compressed(d));

		// A heap's texture follows the rule; a view answers its texture's, a texture over a buffer
		// or an IOSurface NO.
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypePlacement)
		                                       device:nil
		                                   serializer:&fx.ser];
		id<MTLTexture> placed = [heap newTextureWithDescriptor:texture2D(MTLPixelFormatBGRA8Unorm,
		                                                                 64, 64, read)
		                                                offset:0];
		CHECK([internals(placed) isCompressed]);
		id<MTLTexture> view = [placed newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(view != nil && [internals(view) isCompressed]);
		RgfxBuffer *buffer = [RgfxBuffer bufferWithLength:0x10000
		                                          options:0
		                                           device:nil
		                                       serializer:&fx.ser];
		MTLTextureDescriptor *ld = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		ld.storageMode = MTLStorageModeShared;
		id<MTLTexture> linear = [buffer newTextureWithDescriptor:ld offset:0 bytesPerRow:256];
		CHECK(linear != nil && ![internals(linear) isCompressed]);
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @64,
			(id)kIOSurfaceHeight : @64,
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
		};
		IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		CHECK(surface != nullptr);
		RgfxIOSurfaceTexture *over = [RgfxIOSurfaceTexture textureWithDescriptor:ld
		                                                               iosurface:surface
		                                                                   plane:0
		                                                                  region:&region
		                                                                  device:nil
		                                                              serializer:&fx.ser];
		CHECK(over != nil && ![internals(over) isCompressed]);
		over = nil;
		CFRelease(surface);
	}
	drainRing(fx.ring);
}

// What a texture answers past MTLTexture, for each kind the device makes: its options resolved and
// as asked, its GPU and CPU addresses, the allocation it lies in and that allocation's size, its
// number among the process's textures, and the rest of the resource SPI. A view answers its
// texture's, a texture over a buffer the buffer's, a heap's texture the heap's, and textures over
// one IOSurface plane share an allocation and an address.
void test_texture_answers()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0);
	RgfxRegion region(mem.data(), mem.size());
	std::vector<uint8_t> heapBytes(0x100000);
	@autoreleasepool {
		const MTLTextureUsage read = MTLTextureUsageShaderRead;
		auto device = [&](MTLTextureDescriptor *d) {
			return [RgfxTexture textureWithDescriptor:d
			                                     heap:RGFX_OBJ_NONE
			                               heapOffset:0
			                                   device:nil
			                               serializer:&fx.ser];
		};
		drainRing(fx.ring);
		RgfxTexture *priv = device(texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read));
		auto made = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
		CHECK(priv != nil && made.gpu_address != 0);
		CHECK([internals(priv) gpuAddress] == made.gpu_address);
		CHECK(priv.resourceOptions == 0x220 && [internals(priv) unfilteredResourceOptions] == 0x20);
		CHECK(priv.hazardTrackingMode == MTLHazardTrackingModeTracked);
		CHECK(priv.storageMode == MTLStorageModePrivate);
		CHECK([internals(priv) resourceSize] == priv.allocatedSize && priv.allocatedSize == 0x4000);
		CHECK([internals(priv) virtualAddress] == nullptr);
		CHECK(![internals(priv) isDrawable] && [internals(priv) rotation] == 0);
		[internals(priv) didModifyData];
		CHECK([internals(priv) isComplete] && [internals(priv) isWriteComplete]);
		[internals(priv) waitUntilComplete];
		CHECK(![internals(priv) isPurgeable]);
		[priv setPurgeableState:MTLPurgeableStateVolatile];
		CHECK([internals(priv) isPurgeable]);
		[priv setPurgeableState:MTLPurgeableStateNonVolatile];
		CHECK(![internals(priv) isPurgeable]);
		CHECK(![internals(priv) doesAliasResource:priv] && ![internals(priv) doesAliasResource:nil]);
		CHECK(ringEmpty(fx.ring)); // the members post nothing

		// Options as asked, resolved; a texture of each kind takes the next number.
		MTLTextureDescriptor *sd = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		sd.storageMode = MTLStorageModeShared;
		RgfxTexture *shared = device(sd);
		CHECK(shared.resourceOptions == 0x200 && [internals(shared) unfilteredResourceOptions] == 0);
		CHECK([internals(shared) virtualAddress] == nullptr);
		sd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		RgfxTexture *untracked = device(sd);
		CHECK(untracked.resourceOptions == 0x100 &&
		      [internals(untracked) unfilteredResourceOptions] == 0x100);
		CHECK(untracked.hazardTrackingMode == MTLHazardTrackingModeUntracked);
		MTLTextureDescriptor *md = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		md.storageMode = MTLStorageModeManaged;
		RgfxTexture *managed = [RgfxTexture managedTextureWithDescriptor:md
		                                                          region:&region
		                                                          device:nil
		                                                      serializer:&fx.ser];
		CHECK(managed != nil && managed.resourceOptions == 0x210);
		CHECK([internals(managed) unfilteredResourceOptions] == 0x10);
		CHECK([internals(managed) gpuAddress] != 0 &&
		      [internals(managed) gpuAddress] != [internals(priv) gpuAddress]);
		const uint64_t first = [internals(priv) uniqueIdentifier];
		CHECK(first != 0 && [internals(shared) uniqueIdentifier] == first + 1);
		CHECK([internals(untracked) uniqueIdentifier] == first + 2);
		CHECK([internals(managed) uniqueIdentifier] == first + 3);
		const uint64_t a0 = [internals(priv) allocationID], a1 = [internals(shared) allocationID];
		CHECK(a0 != 0 && a1 != 0 && a0 != a1);

		// A view, at any level and slice, answers its texture's, under a number of its own, and
		// keeps a purgeable state of its own.
		id<MTLTexture> view = [priv newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(view != nil && [internals(view) uniqueIdentifier] == first + 4);
		CHECK([internals(view) gpuAddress] == [internals(priv) gpuAddress]);
		CHECK([internals(view) allocationID] == a0);
		CHECK([internals(view) resourceSize] == [internals(priv) resourceSize]);
		CHECK(view.resourceOptions == 0x220 && [internals(view) unfilteredResourceOptions] == 0x20);
		CHECK([internals(view) virtualAddress] == nullptr && ![internals(view) isDrawable]);
		[priv setPurgeableState:MTLPurgeableStateVolatile];
		CHECK([internals(priv) isPurgeable] && ![internals(view) isPurgeable]);
		[priv setPurgeableState:MTLPurgeableStateNonVolatile];
		MTLTextureDescriptor *ad = texture2D(MTLPixelFormatRGBA8Unorm, 64, 64, read);
		ad.textureType = MTLTextureType2DArray;
		ad.arrayLength = 4;
		ad.mipmapLevelCount = 3;
		RgfxTexture *array = device(ad);
		id<MTLTexture> deep = [array newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                               textureType:MTLTextureType2DArray
		                                                    levels:NSMakeRange(1, 2)
		                                                    slices:NSMakeRange(1, 2)];
		CHECK(deep != nil && [internals(deep) gpuAddress] == [internals(array) gpuAddress]);
		CHECK([internals(deep) allocationID] == [internals(array) allocationID]);

		// A heap's texture lies at the heap's address and CPU address plus its offset, in the
		// heap's allocation, of the heap's size and options, and aliases what meets its bytes.
		drainRing(fx.ring);
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypePlacement)
		                                       device:nil
		                                   serializer:&fx.ser];
		auto heapMade = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		id<MTLTexture> placed =
		    [heap newTextureWithDescriptor:texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read)
		                            offset:0x10000];
		id<MTLBuffer> over = [heap newBufferWithLength:0x1000
		                                       options:MTLResourceStorageModePrivate
		                                        offset:0x10000];
		id<MTLBuffer> apart = [heap newBufferWithLength:0x1000
		                                        options:MTLResourceStorageModePrivate
		                                         offset:0];
		CHECK(placed != nil && [internals(placed) gpuAddress] == heapMade.gpu_address + 0x10000);
		CHECK([internals(placed) virtualAddress] == reinterpret_cast<void *>(0x10000));
		CHECK([internals(placed) allocationID] == [internals(over) allocationID]);
		CHECK([internals(placed) resourceSize] == heap.size);
		CHECK(placed.resourceOptions == 0x120 && [internals(placed) unfilteredResourceOptions] == 0x120);
		CHECK([internals(placed) doesAliasResource:over] && [internals(over) doesAliasResource:placed]);
		CHECK([internals(placed) doesAliasResource:placed]);
		CHECK(![internals(placed) doesAliasResource:apart]);
		__unsafe_unretained id<MTLResource> both[] = {over, apart};
		CHECK([internals(placed) doesAliasAnyResources:both count:2]);
		CHECK(![internals(placed) doesAliasAllResources:both count:2]);
		id<MTLTexture> placedView = [placed newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK([internals(placedView) doesAliasResource:over] &&
		      [internals(placedView) doesAliasResource:placed]);
		CHECK([internals(placedView) gpuAddress] == [internals(placed) gpuAddress]);

		// A Shared heap's texture lies at the heap's bytes plus its offset.
		RgfxHeap *sharedHeap = [RgfxHeap
		    heapOverHostMemoryWithDescriptor:heapDescriptor(MTLStorageModeShared,
		                                                    MTLHeapTypePlacement)
		                        regionOffset:0
		                            contents:heapBytes.data()
		                         deallocator:nil
		                              device:nil
		                          serializer:&fx.ser];
		MTLTextureDescriptor *shd = texture2D(MTLPixelFormatRGBA8Unorm, 64, 64, read);
		shd.storageMode = MTLStorageModeShared;
		id<MTLTexture> inShared = [sharedHeap newTextureWithDescriptor:shd offset:0x10000];
		CHECK(inShared != nil &&
		      [internals(inShared) virtualAddress] == heapBytes.data() + 0x10000);
		CHECK(inShared.resourceOptions == 0x100 &&
		      [internals(inShared) unfilteredResourceOptions] == 0x100);

		// A texture over a buffer lies in the buffer's memory: its address plus the offset, its
		// allocation, its options and its heap, and stays NonVolatile.
		RgfxBuffer *buffer = [RgfxBuffer bufferWithLength:0x10000
		                                          options:0
		                                           device:nil
		                                       serializer:&fx.ser];
		MTLTextureDescriptor *ld = texture2D(MTLPixelFormatBGRA8Unorm, 64, 64, read);
		ld.storageMode = MTLStorageModeShared;
		id<MTLTexture> linear = [buffer newTextureWithDescriptor:ld offset:0x1000 bytesPerRow:256];
		CHECK(linear != nil && [internals(linear) gpuAddress] == buffer.gpuAddress + 0x1000);
		CHECK([internals(linear) virtualAddress] ==
		      static_cast<uint8_t *>(buffer.contents) + 0x1000);
		CHECK([internals(linear) allocationID] == [internals(buffer) allocationID]);
		CHECK([internals(linear) resourceSize] == 0x10000);
		CHECK(linear.resourceOptions == 0x200 && [internals(linear) unfilteredResourceOptions] == 0);
		CHECK(linear.heap == nil && ![internals(linear) doesAliasResource:buffer]);
		[buffer setPurgeableState:MTLPurgeableStateVolatile];
		CHECK(![internals(linear) isPurgeable]);
		CHECK([linear setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK(![internals(linear) isPurgeable]);
		RgfxBuffer *privBuffer = [RgfxBuffer bufferWithLength:0x10000
		                                              options:MTLResourceStorageModePrivate
		                                               device:nil
		                                           serializer:&fx.ser];
		ld.storageMode = MTLStorageModePrivate;
		id<MTLTexture> privLinear = [privBuffer newTextureWithDescriptor:ld
		                                                          offset:0x1000
		                                                     bytesPerRow:256];
		CHECK(privLinear != nil &&
		      [internals(privLinear) virtualAddress] == reinterpret_cast<void *>(0x1000));
		CHECK(privLinear.resourceOptions == 0x220 &&
		      [internals(privLinear) unfilteredResourceOptions] == 0x20);
		id<MTLBuffer> heapBuffer = [sharedHeap newBufferWithLength:0x1000 options:0 offset:0x4000];
		ld.storageMode = MTLStorageModeShared;
		id<MTLTexture> heapLinear = [heapBuffer newTextureWithDescriptor:ld
		                                                          offset:0
		                                                     bytesPerRow:256];
		CHECK(heapLinear != nil && heapLinear.heap == sharedHeap);
		CHECK(heapLinear.heapOffset == heapBuffer.heapOffset);
		CHECK([internals(heapLinear) allocationID] == [internals(heapBuffer) allocationID]);
		CHECK([internals(heapLinear) resourceSize] == sharedHeap.size);
		CHECK([internals(heapLinear) virtualAddress] == heapBuffer.contents);
		CHECK([internals(heapLinear) doesAliasResource:heapBuffer] &&
		      [internals(heapBuffer) doesAliasResource:heapLinear]);
		id<MTLTexture> linearView = [linear newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(linearView != nil && [internals(linearView) gpuAddress] == [internals(linear) gpuAddress]);
		CHECK([internals(linearView) virtualAddress] == [internals(linear) virtualAddress]);

		// Textures over one IOSurface plane, and a view of one, share an allocation and an address,
		// with no CPU address, and alias nothing.
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @64,
			(id)kIOSurfaceHeight : @64,
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
		};
		IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		CHECK(surface != nullptr);
		ld.storageMode = MTLStorageModeShared;
		auto overSurface = [&](MTLTextureDescriptor *d) {
			return [RgfxIOSurfaceTexture textureWithDescriptor:d
			                                         iosurface:surface
			                                             plane:0
			                                            region:&region
			                                            device:nil
			                                        serializer:&fx.ser];
		};
		RgfxIOSurfaceTexture *sa = overSurface(ld);
		ld.storageMode = MTLStorageModePrivate;
		RgfxIOSurfaceTexture *sb = overSurface(ld);
		CHECK(sa != nil && sb != nil && [internals(sa) gpuAddress] != 0);
		CHECK([internals(sa) gpuAddress] == [internals(sb) gpuAddress]);
		CHECK([internals(sa) allocationID] == [internals(sb) allocationID]);
		CHECK([internals(sa) allocationID] != a0);
		CHECK([internals(sb) uniqueIdentifier] == [internals(sa) uniqueIdentifier] + 1);
		CHECK(sa.resourceOptions == 0x200 && [internals(sa) unfilteredResourceOptions] == 0);
		CHECK(sb.resourceOptions == 0x220 && [internals(sb) unfilteredResourceOptions] == 0x20);
		CHECK([internals(sa) virtualAddress] == nullptr);
		CHECK([internals(sa) resourceSize] == sa.allocatedSize);
		CHECK(![internals(sa) doesAliasResource:sb] && ![internals(sa) doesAliasResource:sa]);
		id<MTLTexture> surfaceView = [sa newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(surfaceView != nil && [internals(surfaceView) gpuAddress] == [internals(sa) gpuAddress]);
		CHECK([internals(surfaceView) allocationID] == [internals(sa) allocationID]);
		CHECK(![internals(surfaceView) isDrawable] && [internals(surfaceView) rotation] == 0);
		sa = nil;
		sb = nil;
		surfaceView = nil;
		CFRelease(surface);
		(void)apart;
	}
	drainRing(fx.ring);
}

// A fake device that answers each sampler create SAMPLER_STATE with the id it is given, the way
// the host answers equal descriptors with one id.
struct SamplerAnswer {
	uint64_t id = 0;

	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		RgfxSamplerState answer = {};
		answer.header = {RGFX_EVT_SAMPLER_STATE, sizeof(answer), 0};
		answer.status = RGFX_STATUS_OK;
		answer.gpu_resource_id = static_cast<SamplerAnswer *>(context)->id;
		CHECK(size == sizeof(RgfxSamplerCreate) && message != nullptr);
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}
};

// A sampler answers no GPU address and no pixel format, and its gpuResourceID's value as its
// identifier, as Metal's: samplers the host answers with one id answer one identifier.
void test_sampler_answers()
{
	Fixture fx;
	SamplerAnswer answer;
	fx.ser.setAsker(&SamplerAnswer::ask, &answer);
	@autoreleasepool {
		MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
		answer.id = 2;
		RgfxSampler *a = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&fx.ser];
		RgfxSampler *b = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&fx.ser];
		d.minFilter = MTLSamplerMinMagFilterLinear;
		d.supportArgumentBuffers = YES;
		answer.id = 3;
		RgfxSampler *c = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&fx.ser];
		CHECK(a != nil && b != nil && c != nil);
		id<RgfxMetalSamplerInternals> ai = (id<RgfxMetalSamplerInternals>)a;
		id<RgfxMetalSamplerInternals> bi = (id<RgfxMetalSamplerInternals>)b;
		id<RgfxMetalSamplerInternals> ci = (id<RgfxMetalSamplerInternals>)c;
		CHECK([ai uniqueIdentifier] == 2 && [bi uniqueIdentifier] == 2 && [ci uniqueIdentifier] == 3);
		CHECK([ai uniqueIdentifier] == a.gpuResourceID._impl &&
		      [ci uniqueIdentifier] == c.gpuResourceID._impl);
		CHECK([ai gpuAddress] == 0 && [ci gpuAddress] == 0);
		CHECK([ai pixelFormat] == MTLPixelFormatInvalid && [ci pixelFormat] == MTLPixelFormatInvalid);
		CHECK(ringEmpty(fx.ring)); // asked, and the members post nothing
	}
	fx.ser.setAsker(nullptr, nullptr);
	drainRing(fx.ring);
}

// A shared event answers itself as its IOSurface shared event, as Metal's; a plain event answers
// no such member.
void test_shared_event_answers()
{
	Fixture fx;
	@autoreleasepool {
		RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:0
		                                                             device:nil
		                                                         serializer:&fx.ser];
		CHECK(ev != nil);
		drainRing(fx.ring);
		CHECK([(id<RgfxMetalSharedEventInternals>)ev IOSurfaceSharedEvent] == ev);
		CHECK(ringEmpty(fx.ring));
		RgfxEvent *plain = [RgfxEvent eventWithDevice:nil serializer:&fx.ser];
		CHECK(plain != nil && ![plain respondsToSelector:@selector(IOSurfaceSharedEvent)]);
	}
	drainRing(fx.ring);
}

} // namespace

void rgfxResourceTests()
{
	test_heap_answers();
	test_heap_buffer_descriptors();
	test_buffer_answers();
	test_heap_buffers_answer_their_heap();
	test_buffer_linear_textures();
	test_texture_compression();
	test_texture_answers();
	test_sampler_answers();
	test_shared_event_answers();
}
