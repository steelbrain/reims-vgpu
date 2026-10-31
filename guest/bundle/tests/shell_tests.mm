/*
 * shell_tests: the Metal shell's user-space check, run on the Mac's native arch
 * (guest/bundle/Makefile `test`). It exercises the NSObject-based object wrappers
 * — the part of the shell that does not need Apple's private device base — and
 * pins the ROUTING ASSERTION PATTERN the four later slices copy:
 *
 *   build a ring + serialiser (the shared tests/rgfx_test_ring.hpp harness, the
 *   same one the serialiser check uses) -> instantiate a wrapper, which routes its
 *   create through the serialiser -> read the message back off the ring with
 *   expectMessage<T> and assert the exact wire fields, plus the wrapper's getters.
 *
 * The device shell (RgfxMetalDevice) is NOT built into this binary: it subclasses
 * an Apple-private class whose symbols resolve only in the guest. Only the
 * NSObject-based wrappers are tested here.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the
 * first failure.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>

#import "RgfxBuffer.h"
#import "RgfxResource.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxSharedEvent.h"
#import "RgfxDepthStencilState.h"
#import "RgfxHeap.h"
#import "RgfxSampler.h"
#import "RgfxTexture.h"
#import "RgfxArgumentEncoder.h"
#import "RgfxComputePipelineState.h"
#import "RgfxFunction.h"
#import "RgfxLibrary.h"
#import "RgfxRenderPipelineState.h"
#include <set>
#include <vector>
#import "RgfxCommandObjects.h"
#import "RgfxIOSurface.h"

#include <IOSurface/IOSurface.h>

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "RgfxCompletionThread.hpp"
#include "RgfxManagedLayout.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRetirement.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage


@interface TestHandle : RgfxResource <MTLBuffer, MTLTexture, MTLRenderPipelineState,
                                      MTLComputePipelineState, MTLDepthStencilState,
                                      MTLSamplerState, MTLFence, MTLHeap,
                                      MTLIndirectCommandBuffer>
@end

@implementation TestHandle
@end
namespace {

// A ring + id space + serialiser over a plain-buffer region, the fixture every
// wrapper routing test stands up. Mirrors how a device category would reach its
// serialiser, minus the device (nil here, as a wrapper allows).
// The transport is never freed: an object a test lets go may dealloc after the test (its
// autorelease pool drains later, or a retirement waits) and post its destroy here.
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
		ring.setConsumerIdle(1); // host idle, so each post rings the doorbell
	}
};

// A buffer wrapper's create routes an RGFX_CMD_BUFFER_CREATE onto the ring, and
// its getters answer from the created values. This is the pattern the resource
// slice replicates for textures/samplers/heaps.
void test_buffer_routing()
{
	Fixture fx;

	const NSUInteger length = 0x2000;
	const MTLResourceOptions options =
	    MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;

	RgfxBuffer *buf = [RgfxBuffer bufferWithLength:length
	                                       options:options
	                                        device:nil
	                                    serializer:&fx.ser];
	CHECK(buf != nil);

	// The wrapper's getters answer from the created values.
	CHECK(buf.rgfxObjId == 1);         // first object on this device
	CHECK([buf length] == length);     // MTLBuffer.length
	CHECK([buf gpuAddress] != 0);      // a GPU address was picked
	CHECK(([buf gpuAddress] % RgfxSerializer::kDefaultAddressAlign) == 0);
	CHECK([buf gpuAddress] >= RgfxIdSpace::kDefaultAddressBase);
	CHECK([buf resourceOptions] == options);
	CHECK(buf.device == nil);          // no device in this test
	CHECK([buf serializer] == &fx.ser); // borrowed route is wired

	// contents is a usable, zeroed, length-sized CPU backing.
	void *contents = [buf contents];
	CHECK(contents != nullptr);
	uint8_t *bytes = static_cast<uint8_t *>(contents);
	CHECK(bytes[0] == 0 && bytes[length - 1] == 0);
	bytes[0] = 0xab;
	bytes[length - 1] = 0xcd;
	CHECK(bytes[0] == 0xab && bytes[length - 1] == 0xcd);

	// The create wrote exactly one BUFFER_CREATE, and its fields match the wrapper.
	auto msg = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == buf.rgfxObjId);
	CHECK(msg.gpu_address == [buf gpuAddress]);
	CHECK(msg.length == length);
	CHECK(msg.options == static_cast<uint32_t>(options));

	// didModifyRange: is a no-op for coherent Shared storage (slice 0) and must not
	// post anything or fault. (Apple deprecated the selector on macOS 27 — Managed
	// storage has no effect on Apple Silicon — but we still implement it for
	// MTLBuffer conformance and older guests, so the call is deprecation-silenced.)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[buf didModifyRange:NSMakeRange(0, length)];
#pragma clang diagnostic pop
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // nothing more on the ring
}

// The id space is monotonic across wrappers: a second buffer gets the next objId
// and a disjoint, higher GPU address, and its own BUFFER_CREATE.
void test_second_buffer_is_monotonic()
{
	Fixture fx;

	RgfxBuffer *a = [RgfxBuffer bufferWithLength:0x1000
	                                     options:MTLResourceStorageModeShared
	                                      device:nil
	                                  serializer:&fx.ser];
	RgfxBuffer *b = [RgfxBuffer bufferWithLength:0x1000
	                                     options:MTLResourceStorageModeShared
	                                      device:nil
	                                  serializer:&fx.ser];
	CHECK(a != nil && b != nil);
	CHECK(b.rgfxObjId == a.rgfxObjId + 1);
	CHECK([b gpuAddress] >= [a gpuAddress] + 0x1000);

	auto ma = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	auto mb = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	CHECK(ma.obj_id == a.rgfxObjId && mb.obj_id == b.rgfxObjId);
	CHECK(mb.header.seqno == ma.header.seqno + 1);
}

// A wrapper with no serialiser route cannot be created (a buffer must reach the
// host), and the base label getter/setter work.
void test_wrapper_guards_and_base()
{
	RgfxBuffer *none = [RgfxBuffer bufferWithLength:0x1000
	                                        options:MTLResourceStorageModeShared
	                                         device:nil
	                                     serializer:nullptr];
	CHECK(none == nil);

	Fixture fx;
	RgfxBuffer *buf = [RgfxBuffer bufferWithLength:0x1000
	                                       options:MTLResourceStorageModeShared
	                                        device:nil
	                                    serializer:&fx.ser];
	CHECK(buf.label == nil);
	buf.label = @"scratch";
	CHECK([buf.label isEqualToString:@"scratch"]);
	// RgfxBuffer is a RgfxResource and conforms to MTLBuffer.
	CHECK([buf isKindOfClass:[RgfxResource class]]);
	CHECK([buf conformsToProtocol:@protocol(MTLBuffer)]);
}


// --- sync slice (slice 4): event / fence / shared-event wrappers -------------

// An event wrapper's create routes one RGFX_CMD_EVENT_CREATE (an RgfxObjectCreate
// carrying its objId), and the wrapper carries that id, its device and route.
void test_event_routing()
{
	Fixture fx;

	RgfxEvent *ev = [RgfxEvent eventWithDevice:nil serializer:&fx.ser];
	CHECK(ev != nil);
	CHECK(ev.rgfxObjId == 1); // first object on this device
	CHECK(ev.device == nil);
	CHECK([ev serializer] == &fx.ser);
	CHECK([ev isKindOfClass:[RgfxResource class]]);
	CHECK([ev conformsToProtocol:@protocol(MTLEvent)]);

	auto msg = expectMessage<RgfxObjectCreate>(fx.ring, RGFX_CMD_EVENT_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == ev.rgfxObjId);

	// Nothing else reached the ring.
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes));

	// A null route declines the create (an event must reach the host).
	CHECK([RgfxEvent eventWithDevice:nil serializer:nullptr] == nil);
}

// A fence wrapper's create routes one RGFX_CMD_FENCE_CREATE, same shape as an
// event.
void test_fence_routing()
{
	Fixture fx;

	RgfxFence *fence = [RgfxFence fenceWithDevice:nil serializer:&fx.ser];
	CHECK(fence != nil);
	CHECK(fence.rgfxObjId == 1);
	CHECK([fence conformsToProtocol:@protocol(MTLFence)]);

	auto msg = expectMessage<RgfxObjectCreate>(fx.ring, RGFX_CMD_FENCE_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == fence.rgfxObjId);

	CHECK([RgfxFence fenceWithDevice:nil serializer:nullptr] == nil);
}

// A shared-event wrapper's create routes one RGFX_CMD_SHARED_EVENT_CREATE carrying
// the initial value, and the wrapper's signaledValue mirror starts there.
void test_shared_event_routing()
{
	Fixture fx;

	const uint64_t initial = 0x1234;
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:initial
	                                                           device:nil
	                                                       serializer:&fx.ser];
	CHECK(ev != nil);
	CHECK(ev.rgfxObjId == 1);
	CHECK([ev conformsToProtocol:@protocol(MTLSharedEvent)]);

	auto msg = expectMessage<RgfxSharedEventCreate>(fx.ring, RGFX_CMD_SHARED_EVENT_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == ev.rgfxObjId);
	CHECK(msg.initial_value == initial);

	// The local value mirror starts at the initial value, and a CPU-side set
	// updates the mirror (no wire op for a set in this slice).
	CHECK(ev.signaledValue == initial);
	ev.signaledValue = initial + 7;
	CHECK(ev.signaledValue == initial + 7);

	// The set posts nothing on the ring (documented follow-up, no wire op yet), and
	// the listener/handle stubs are safe no-ops.
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes));
	CHECK([ev newSharedEventHandle] == nil);
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
	[ev notifyListener:listener
	           atValue:initial + 7
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             (void)v;
	             }]; // stub: must not fault

	CHECK([RgfxSharedEvent sharedEventWithInitialValue:0 device:nil serializer:nullptr] == nil);
}


// A texture wrapper's create routes an RGFX_CMD_TEXTURE_CREATE onto the ring
// (device-placed: a GPU address and a resource id are picked, heap_obj_id 0), and
// its getters answer the mapped descriptor. The RgfxBuffer routing pattern, for
// the texture's shape.
void test_texture_routing()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {

	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:1920
	                                                      height:1080
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModePrivate;

	RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
	                                                 heap:RGFX_OBJ_NONE
	                                           heapOffset:0
	                                               device:nil
	                                           serializer:&fx.ser];
	CHECK(tex != nil);

	// Getters answer the mapped descriptor.
	CHECK(tex.rgfxObjId == 1);
	CHECK([tex textureType] == MTLTextureType2D);
	CHECK([tex pixelFormat] == MTLPixelFormatBGRA8Unorm);
	CHECK([tex width] == 1920 && [tex height] == 1080 && [tex depth] == 1);
	CHECK([tex mipmapLevelCount] == 1 && [tex arrayLength] == 1 && [tex sampleCount] == 1);
	CHECK([tex usage] == (MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead));
	CHECK([tex storageMode] == MTLStorageModePrivate);
	CHECK([tex resourceOptions] == desc.resourceOptions);
	MTLTextureSwizzleChannels sw = [tex swizzle]; // the descriptor's default swizzle
	CHECK(sw.red == MTLTextureSwizzleRed && sw.green == MTLTextureSwizzleGreen);
	CHECK(sw.blue == MTLTextureSwizzleBlue && sw.alpha == MTLTextureSwizzleAlpha);
	CHECK([tex gpuResourceID]._impl != 0); // a resource id was picked
	CHECK([tex parentTexture] == nil && [tex buffer] == nil);

	// The create wrote exactly one TEXTURE_CREATE matching the wrapper.
	auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == tex.rgfxObjId);
	CHECK(msg.resource_id == [tex gpuResourceID]._impl);
	CHECK(msg.gpu_address != 0 && msg.heap_obj_id == 0); // device-placed
	CHECK(msg.desc.pixel_format == MTLPixelFormatBGRA8Unorm);
	CHECK(msg.desc.texture_type == MTLTextureType2D);
	CHECK(msg.desc.width == 1920 && msg.desc.height == 1080);
	CHECK(msg.desc.usage == (uint32_t)(MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead));
	CHECK(msg.desc.options == (uint32_t)desc.resourceOptions);
	}
}

// One chunk of texture resource ids (ids 1 + 4 * 64 to 64 + 4 * 64), as the kext would hand
// it, and none after it.
bool oneChunk(void *context, uint64_t *first_id)
{
	int *calls = static_cast<int *>(context);
	if ((*calls)++ != 0)
		return false;
	*first_id = 1 + 4 * RGFX_RESOURCE_ID_CHUNK;
	return true;
}

// A texture's resource id comes from the process's chunks and goes back as the texture goes,
// for the next texture; with every id held and no chunk left, newTexture answers nil and posts
// nothing.
void test_texture_resource_ids()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	const uint64_t first = 1 + 4 * RGFX_RESOURCE_ID_CHUNK;
	NSMutableArray<RgfxTexture *> *held = [NSMutableArray array];
	for (uint64_t i = 0; i < RGFX_RESOURCE_ID_CHUNK; i++) {
		// Each texture's autoreleased return drains here, so `held` alone keeps it.
		@autoreleasepool {
			RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
			                                                 heap:RGFX_OBJ_NONE
			                                           heapOffset:0
			                                               device:nil
			                                           serializer:&fx.ser];
			CHECK(tex != nil && [tex gpuResourceID]._impl == first + i);
			auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
			CHECK(msg.resource_id == first + i);
			[held addObject:tex];
		}
	}
	CHECK(calls == 1);
	RgfxTexture *none = [RgfxTexture textureWithDescriptor:desc
	                                                  heap:RGFX_OBJ_NONE
	                                            heapOffset:0
	                                                device:nil
	                                            serializer:&fx.ser];
	CHECK(none == nil);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!fx.ring.next(&h, &bytes));
	// The texture holding id first + 10 goes: the host drops it (no command buffer is in
	// flight), then the next texture takes its id, after the destroy on the ring.
	uint64_t goneObjId = 0;
	@autoreleasepool {
		goneObjId = held[10].rgfxObjId;
		[held removeObjectAtIndex:10];
	}
	auto gone = expectMessage<RgfxObjectDestroy>(fx.ring, RGFX_CMD_OBJECT_DESTROY);
	CHECK(gone.obj_id == goneObjId);
	RgfxTexture *again = [RgfxTexture textureWithDescriptor:desc
	                                                   heap:RGFX_OBJ_NONE
	                                             heapOffset:0
	                                                 device:nil
	                                             serializer:&fx.ser];
	CHECK(again != nil && [again gpuResourceID]._impl == first + 10);
	auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(msg.resource_id == first + 10 && msg.obj_id == again.rgfxObjId);
	}
}

// A sampler wrapper's create routes an RGFX_CMD_SAMPLER_CREATE, mapping every
// descriptor field (the LOD clamps as f32 bits, the booleans packed into flags).
void test_sampler_routing()
{
	Fixture fx;

	MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
	desc.minFilter = MTLSamplerMinMagFilterLinear;
	desc.magFilter = MTLSamplerMinMagFilterLinear;
	desc.mipFilter = MTLSamplerMipFilterLinear;
	desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
	desc.tAddressMode = MTLSamplerAddressModeRepeat;
	desc.rAddressMode = MTLSamplerAddressModeMirrorRepeat;
	desc.maxAnisotropy = 4;
	desc.compareFunction = MTLCompareFunctionLess;
	desc.lodMinClamp = 0.0f;
	desc.lodMaxClamp = 7.5f;
	desc.normalizedCoordinates = YES;
	desc.supportArgumentBuffers = YES;

	RgfxSampler *smp = [RgfxSampler samplerWithDescriptor:desc device:nil serializer:&fx.ser];
	CHECK(smp != nil && smp.rgfxObjId == 1);
	CHECK([smp conformsToProtocol:@protocol(MTLSamplerState)]);

	auto msg = expectMessage<RgfxSamplerCreate>(fx.ring, RGFX_CMD_SAMPLER_CREATE);
	CHECK(msg.obj_id == smp.rgfxObjId);
	CHECK(msg.min_filter == (uint32_t)MTLSamplerMinMagFilterLinear);
	CHECK(msg.mag_filter == (uint32_t)MTLSamplerMinMagFilterLinear);
	CHECK(msg.mip_filter == (uint32_t)MTLSamplerMipFilterLinear);
	CHECK(msg.s_address_mode == (uint32_t)MTLSamplerAddressModeClampToEdge);
	CHECK(msg.t_address_mode == (uint32_t)MTLSamplerAddressModeRepeat);
	CHECK(msg.r_address_mode == (uint32_t)MTLSamplerAddressModeMirrorRepeat);
	CHECK(msg.max_anisotropy == 4);
	CHECK(msg.compare_function == (uint32_t)MTLCompareFunctionLess);
	float lodMax;
	std::memcpy(&lodMax, &msg.lod_max_clamp_bits, 4);
	CHECK(lodMax == 7.5f);
	CHECK(msg.flags == (0x1u | 0x4u)); // normalized coords + argument-buffer usable
}

// A depth/stencil wrapper's create routes an RGFX_CMD_DEPTH_STENCIL_CREATE,
// mapping the depth test/write and both faces' stencil state.
void test_depth_stencil_routing()
{
	Fixture fx;

	MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
	desc.depthCompareFunction = MTLCompareFunctionLessEqual;
	desc.depthWriteEnabled = YES;
	MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
	front.stencilCompareFunction = MTLCompareFunctionEqual;
	front.stencilFailureOperation = MTLStencilOperationKeep;
	front.depthFailureOperation = MTLStencilOperationIncrementClamp;
	front.depthStencilPassOperation = MTLStencilOperationReplace;
	front.readMask = 0xfe;
	front.writeMask = 0xcd;
	desc.frontFaceStencil = front;

	RgfxDepthStencilState *ds =
	    [RgfxDepthStencilState depthStencilStateWithDescriptor:desc device:nil serializer:&fx.ser];
	CHECK(ds != nil && ds.rgfxObjId == 1);
	CHECK([ds conformsToProtocol:@protocol(MTLDepthStencilState)]);

	auto msg = expectMessage<RgfxDepthStencilCreate>(fx.ring, RGFX_CMD_DEPTH_STENCIL_CREATE);
	CHECK(msg.obj_id == ds.rgfxObjId);
	CHECK(msg.depth_compare == (uint32_t)MTLCompareFunctionLessEqual);
	CHECK(msg.depth_write_enabled == 1);
	CHECK(msg.front.stencil_compare == (uint32_t)MTLCompareFunctionEqual);
	CHECK(msg.front.stencil_fail_op == (uint32_t)MTLStencilOperationKeep);
	CHECK(msg.front.depth_fail_op == (uint32_t)MTLStencilOperationIncrementClamp);
	CHECK(msg.front.pass_op == (uint32_t)MTLStencilOperationReplace);
	CHECK(msg.front.read_mask == 0xfe && msg.front.write_mask == 0xcd);
	// The untouched back face carries Metal's default MTLStencilDescriptor:
	// stencilCompareFunction Always, keep operations, all-ones masks.
	CHECK(msg.back.stencil_compare == (uint32_t)MTLCompareFunctionAlways);
	CHECK(msg.back.read_mask == 0xffffffffu && msg.back.write_mask == 0xffffffffu);
}

// A heap wrapper's create routes an RGFX_CMD_HEAP_CREATE; its getters answer the
// descriptor; and a Placement heap sub-allocates a buffer (BUFFER_AT_OFFSET) and a
// texture (TEXTURE_CREATE into the heap), both wrapped. The offset-less variants
// are documented nil stubs this slice.
void test_heap_routing_and_suballoc()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {

	MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
	hd.size = 0x100000;
	hd.storageMode = MTLStorageModePrivate;
	hd.type = MTLHeapTypePlacement;

	RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
	CHECK(heap != nil && heap.rgfxObjId == 1);
	CHECK([heap size] == 0x100000);
	CHECK([heap type] == MTLHeapTypePlacement);
	CHECK([heap storageMode] == MTLStorageModePrivate);
	CHECK([heap resourceOptions] == hd.resourceOptions);

	auto hm = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
	CHECK(hm.obj_id == heap.rgfxObjId && hm.gpu_address != 0);
	CHECK(hm.size_bytes == 0x100000 && hm.heap_type == (int32_t)MTLHeapTypePlacement);
	CHECK(hm.options == (uint32_t)hd.resourceOptions);

	// Placement buffer at an offset -> BUFFER_AT_OFFSET into this heap.
	id<MTLBuffer> hb = [heap newBufferWithLength:0x400
	                                    options:MTLResourceStorageModePrivate
	                                     offset:0x2000];
	CHECK(hb != nil);
	CHECK([(RgfxBuffer *)hb rgfxObjId] == heap.rgfxObjId + 1); // next objId
	CHECK([hb length] == 0x400);
	auto bm = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(bm.obj_id == [(RgfxBuffer *)hb rgfxObjId]);
	CHECK(bm.heap_obj_id == heap.rgfxObjId);
	CHECK(bm.offset == 0x2000 && bm.length == 0x400);

	// Placement texture at an offset -> TEXTURE_CREATE into this heap.
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:256
	                                                      height:256
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> ht = [heap newTextureWithDescriptor:td offset:0x4000];
	CHECK(ht != nil);
	CHECK([ht width] == 256 && [ht height] == 256);
	auto tm = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(tm.obj_id == [(RgfxTexture *)ht rgfxObjId]);
	CHECK(tm.heap_obj_id == heap.rgfxObjId && tm.heap_offset == 0x4000);
	CHECK(tm.gpu_address == 0); // heap-placed: no device address
	CHECK(tm.desc.pixel_format == MTLPixelFormatRGBA8Unorm);

	// The offset-less (Automatic-placement) variants are nil stubs this slice.
	CHECK([heap newBufferWithLength:0x100 options:MTLResourceStorageModePrivate] == nil);
	CHECK([heap newTextureWithDescriptor:td] == nil);
	}
}

// A wrapper with no serialiser route cannot be created (each resource must reach
// the host).
void test_resource_guards()
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	CHECK([RgfxTexture textureWithDescriptor:td
	                                    heap:RGFX_OBJ_NONE
	                              heapOffset:0
	                                  device:nil
	                              serializer:nullptr] == nil);
	CHECK([RgfxSampler samplerWithDescriptor:[[MTLSamplerDescriptor alloc] init]
	                                  device:nil
	                              serializer:nullptr] == nil);
	CHECK([RgfxDepthStencilState depthStencilStateWithDescriptor:[[MTLDepthStencilDescriptor alloc]
	                                                                 init]
	                                                     device:nil
	                                                 serializer:nullptr] == nil);
	CHECK([RgfxHeap heapWithDescriptor:[[MTLHeapDescriptor alloc] init]
	                            device:nil
	                        serializer:nullptr] == nil);
}


// Reads the next message off the ring into `bytes` and checks its type; returns the
// full message bytes (head + tail) for a variable-size wire struct the fixed-size
// expectMessage<T> cannot read. Aborts (via CHECK) if there is no message.
std::vector<uint8_t> nextMessage(TestRing &ring, uint32_t type)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == type);
	return bytes;
}

// A library create forwards its bytes unchanged (RGFX_CMD_LIBRARY_CREATE), and the
// library vends a function by name as a guest-side RgfxFunction (no wire message).
void test_library_and_function_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	CHECK(lib != nil);
	CHECK(lib.rgfxObjId == 1); // first object on this device
	CHECK([lib conformsToProtocol:@protocol(MTLLibrary)]);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	RgfxLibraryCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.header.seqno == 1);
	CHECK(head.obj_id == lib.rgfxObjId);
	CHECK(head.byte_length == sizeof(libBytes));
	CHECK(std::memcmp(bytes.data() + RGFX_LIBRARY_BYTES_OFFSET, libBytes, sizeof(libBytes)) == 0);

	// A function is a guest-side reference: it holds the library objId + name, picks
	// no objId, and posts nothing.
	id<MTLFunction> fn = [lib newFunctionWithName:@"vmain"];
	CHECK(fn != nil);
	RgfxFunction *rf = (RgfxFunction *)fn;
	CHECK(rf.rgfxObjId == 0); // RGFX_OBJ_NONE: no host handle of its own
	CHECK(rf.rgfxLibraryObjId == lib.rgfxObjId);
	CHECK(std::strcmp([rf rgfxEntryPointUTF8], "vmain") == 0);
	CHECK([rf rgfxConstantCount] == 0);
	CHECK([[fn name] isEqualToString:@"vmain"]);

	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // newFunctionWithName: posts nothing

	// A library with no route cannot be created.
	RgfxLibrary *none = [RgfxLibrary libraryWithData:libBytes
	                                         length:sizeof(libBytes)
	                                            key:nullptr
	                                         device:nil
	                                     serializer:nullptr
	                                         status:&libStatus];
	CHECK(none == nil);
}

// A render pipeline create maps the MTLRenderPipelineDescriptor into the wire: the
// functions, label, colour attachment, flags and vertex descriptor; the wrapper
// holds objId + seqno and answers reflection getters with documented placeholders.
void test_render_pipeline_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {1, 2, 3, 4};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE); // drain the library create

	id<MTLFunction> vfn = [lib newFunctionWithName:@"vmain"];
	id<MTLFunction> ffn = [lib newFunctionWithName:@"fmain"];

	MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
	desc.vertexFunction = vfn;
	desc.fragmentFunction = ffn;
	desc.label = @"pipe";
	desc.rasterSampleCount = 4;
	desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	desc.colorAttachments[0].blendingEnabled = YES;
	desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
	desc.alphaToCoverageEnabled = YES;

	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.layouts[0].stride = 32;
	vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
	vd.layouts[0].stepRate = 1;
	vd.attributes[0].format = MTLVertexFormatFloat3;
	vd.attributes[0].offset = 0;
	vd.attributes[0].bufferIndex = 0;
	desc.vertexDescriptor = vd;

	NSError *err = nil;
	RgfxRenderPipelineState *ps =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(ps != nil);
	CHECK(err == nil);
	CHECK(ps.rgfxObjId == 2); // library was objId 1, the pipeline the next
	CHECK(ps.rgfxSeqno == 2); // library post was seqno 1, the pipeline the next
	CHECK([ps conformsToProtocol:@protocol(MTLRenderPipelineState)]);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_RENDER_PIPELINE_CREATE);
	RgfxRenderPipelineCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == ps.rgfxObjId);
	CHECK(head.vertex_library_obj_id == lib.rgfxObjId);
	CHECK(head.fragment_library_obj_id == lib.rgfxObjId);
	CHECK(head.raster_sample_count == 4);
	CHECK(head.color_attachment_count == 1);
	CHECK(head.colors[0].pixel_format == static_cast<uint32_t>(MTLPixelFormatBGRA8Unorm));
	CHECK(head.colors[0].blending_enabled == 1);
	CHECK(head.colors[0].source_rgb_blend_factor == static_cast<uint32_t>(MTLBlendFactorSourceAlpha));
	CHECK(head.colors[0].destination_rgb_blend_factor ==
	      static_cast<uint32_t>(MTLBlendFactorOneMinusSourceAlpha));
	CHECK(head.depth_pixel_format == static_cast<uint32_t>(MTLPixelFormatDepth32Float));
	CHECK((head.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE) != 0);
	CHECK((head.flags & RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED) != 0); // default on

	// The functions, label and vertex descriptor are in the tail at their offsets.
	CHECK(head.vertex_name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.vertex_name_offset, "vmain", 5) == 0);
	CHECK(head.fragment_name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.fragment_name_offset, "fmain", 5) == 0);
	CHECK(head.label_len == 4);
	CHECK(std::memcmp(bytes.data() + head.label_offset, "pipe", 4) == 0);

	CHECK(head.vertex_layout_count == 1);
	RgfxVertexBufferLayout vl;
	std::memcpy(&vl, bytes.data() + head.vertex_layouts_offset, sizeof(vl));
	CHECK(vl.buffer_index == 0);
	CHECK(vl.stride == 32);
	CHECK(vl.step_function == static_cast<uint32_t>(MTLVertexStepFunctionPerVertex));

	CHECK(head.vertex_attribute_count == 1);
	RgfxVertexAttribute va;
	std::memcpy(&va, bytes.data() + head.vertex_attributes_offset, sizeof(va));
	CHECK(va.attribute_index == 0);
	CHECK(va.format == static_cast<uint32_t>(MTLVertexFormatFloat3));
	CHECK(va.buffer_index == 0);

	// Reflection getters are documented placeholders until the reply reader lands.
	CHECK([ps maxTotalThreadsPerThreadgroup] == 1024);
	CHECK(ps.gpuResourceID._impl == 0);

	// No serialiser route is a refusal with an error.
	NSError *err2 = nil;
	RgfxRenderPipelineState *noRoute =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                       device:nil
	                                                   serializer:nullptr
	                                                        error:&err2];
	CHECK(noRoute == nil);
	CHECK(err2 != nil);
}

// A compute pipeline create maps the function (and the descriptor knobs) into the
// wire; the wrapper holds objId + seqno.
void test_compute_pipeline_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {9, 9, 9};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);

	id<MTLFunction> kfn = [lib newFunctionWithName:@"kmain"];

	NSError *err = nil;
	RgfxComputePipelineState *cs =
	    [RgfxComputePipelineState computePipelineStateWithFunction:kfn
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(cs != nil);
	CHECK(err == nil);
	CHECK(cs.rgfxObjId == 2);
	CHECK(cs.rgfxSeqno == 2);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == cs.rgfxObjId);
	CHECK(head.function_library_obj_id == lib.rgfxObjId);
	CHECK(head.name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.name_offset, "kmain", 5) == 0);
	CHECK(head.flags == 0);

	CHECK([cs threadExecutionWidth] == 32);           // documented placeholder
	CHECK([cs maxTotalThreadsPerThreadgroup] == 1024); // documented placeholder

	// The descriptor variant carries the ICB/exec-width flags and the thread cap.
	MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
	desc.computeFunction = kfn;
	desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
	desc.maxTotalThreadsPerThreadgroup = 256;
	NSError *err2 = nil;
	RgfxComputePipelineState *cs2 =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:desc
	                                                         device:nil
	                                                     serializer:&fx.ser
	                                                          error:&err2];
	CHECK(cs2 != nil);
	CHECK(err2 == nil);
	auto bytes2 = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head2;
	std::memcpy(&head2, bytes2.data(), sizeof(head2));
	CHECK((head2.flags & RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH) != 0);
	CHECK(head2.max_total_threads_per_threadgroup == 256);

	// A compute create with no function is a refusal with an error. (The nil goes
	// through a variable so the nonnull parameter's literal-nil diagnostic does not
	// fire; a guest can still reach this path with a nil id at runtime.)
	NSError *err3 = nil;
	id<MTLFunction> nilFn = nil;
	RgfxComputePipelineState *noFn =
	    [RgfxComputePipelineState computePipelineStateWithFunction:nilFn
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err3];
	CHECK(noFn == nil);
	CHECK(err3 != nil);
}

// The argument encoder is a documented stub: it holds its argument count, carries no
// host handle, and answers placeholder layout queries without posting anything.
void test_argument_encoder_stub()
{
	RgfxArgumentEncoder *enc = [RgfxArgumentEncoder argumentEncoderWithArguments:@[]
	                                                                     device:nil
	                                                                 serializer:nullptr];
	CHECK(enc != nil);
	CHECK(enc.rgfxObjId == 0); // no host handle in this slice
	CHECK(enc.rgfxArgumentCount == 0);
	CHECK([enc encodedLength] == 0);
	CHECK([enc conformsToProtocol:@protocol(MTLArgumentEncoder)]);
}



static TestHandle *handle(uint64_t objId)
{
	return [[TestHandle alloc] initWithObjId:objId device:nil serializer:nullptr];
}

// The host-side op-stream walker, the same minimal consumer as the serialiser check
// (serializer_tests.cpp) and crates/.../decode/cmdbuf.rs.
struct OpWalk {
	const uint8_t *ops;
	size_t size;
	size_t pos = 0;
	OpWalk(const uint8_t *p, size_t n) : ops(p), size(n) {}
	bool done() const { return pos >= size; }
	bool next(uint16_t *tag, uint32_t *len, const uint8_t **body, size_t *body_len)
	{
		if (size - pos < sizeof(RgfxOpHeader))
			return false;
		RgfxOpHeader h;
		std::memcpy(&h, ops + pos, sizeof(h));
		if (h.len < RGFX_OP_BODY_OFFSET || (h.len % 8) != 0 || h.len > size - pos)
			return false;
		*tag = h.tag;
		*len = h.len;
		*body = ops + pos + RGFX_OP_BODY_OFFSET;
		*body_len = h.len - RGFX_OP_BODY_OFFSET;
		pos += h.len;
		return true;
	}
};

static uint16_t opGroup(uint16_t tag) { return static_cast<uint16_t>(tag >> 12); }

// The queue/command-buffer/encoder object graph routes the whole Metal call stream
// through RgfxEncoder onto the queue's submission ring. This drives a command buffer
// through each encoder type and asserts the exact op stream the host decoder expects —
// the routing-assertion pattern of this slice.
void test_command_graph()
{
	// Control ring for the serialiser (QUEUE_CREATE), and a separate submission ring
	// the queue's command buffers post on (COMMAND_BUFFER).
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);

	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, /*ring_id*/ 1, TestRing::doorbell,
	                            nullptr);

	// The queue: its create routes a QUEUE_CREATE on the control ring and picks objId 1.
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	CHECK(queue.rgfxObjId == 1);
	CHECK([queue conformsToProtocol:@protocol(MTLCommandQueue)]);
	auto qc = expectMessage<RgfxQueueCreate>(control, RGFX_CMD_QUEUE_CREATE);
	CHECK(qc.obj_id == 1);
	CHECK(qc.submit_data_size == submit.dataSize);

	// A command buffer: next objId (2), bound to the queue, not yet enqueued.
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	CHECK(cb != nil);
	CHECK([cb conformsToProtocol:@protocol(MTLCommandBuffer)]);
	CHECK(((RgfxResource *)cb).rgfxObjId == 2);
	CHECK(cb.commandQueue == queue);
	CHECK(cb.retainedReferences == YES);
	CHECK(cb.status == MTLCommandBufferStatusNotEnqueued);

	// --- render encoder ---
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
	rp.renderTargetWidth = 1920;
	rp.renderTargetHeight = 1080;
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
	CHECK(renc != nil);
	CHECK(renc.device == nil);
	[renc setRenderPipelineState:handle(7)];
	[renc setVertexBuffer:handle(10) offset:16 atIndex:0];
	[renc setViewport:(MTLViewport){ 0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0 }];
	{
		id<MTLBuffer> bufs[2] = { handle(10), handle(11) };
		NSUInteger offs[2] = { 0, 256 };
		[renc setVertexBuffers:bufs offsets:offs withRange:NSMakeRange(0, 2)];
	}
	[renc setFragmentTexture:handle(42) atIndex:3];
	[renc setScissorRect:(MTLScissorRect){ 1, 2, 3, 4 }];
	[renc setDepthStencilState:handle(8)];
	[renc setCullMode:MTLCullModeBack];
	[renc useResource:handle(9) usage:MTLResourceUsageRead];
	[renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[renc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
	                 indexCount:6
	                  indexType:MTLIndexTypeUInt16
	                indexBuffer:handle(12)
	          indexBufferOffset:0
	              instanceCount:2];
	[renc endEncoding];

	// --- compute encoder ---
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	CHECK(cenc != nil);
	[cenc setComputePipelineState:handle(20)];
	[cenc setBuffer:handle(21) offset:64 atIndex:0];
	[cenc setTexture:handle(22) atIndex:1];
	[cenc dispatchThreadgroups:MTLSizeMake(16, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
	[cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[cenc updateFence:handle(30)];
	[cenc endEncoding];

	// --- blit encoder ---
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	CHECK(benc != nil);
	[benc copyFromBuffer:handle(40)
	        sourceOffset:16
	            toBuffer:handle(41)
	   destinationOffset:32
	                size:4096];
	[benc fillBuffer:handle(40) range:NSMakeRange(0, 256) value:0xab];
	[benc generateMipmapsForTexture:handle(50)];
	[benc synchronizeTexture:handle(51) slice:0 level:1];
	[benc waitForFence:handle(60)];
	[benc endEncoding];

	// Holding a completed handler must not fault; commit posts the command buffer.
	[cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull done) {
		(void)done;
	}];
	[cb commit];
	CHECK(cb.status == MTLCommandBufferStatusCommitted);

	// Read the command buffer off the submission ring and check its header.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(h.size == bytes.size());
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	CHECK(cbh.queue_obj_id == 1);
	CHECK(cbh.cb_obj_id == 2);
	CHECK(cbh.op_bytes == bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET);

	// The exact op sequence the three encoders produced.
	const uint16_t want[] = {
		RGFX_OP_ENCODER_BEGIN_RENDER,
		RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
		RGFX_OP_RENDER_SET_VERTEX_BUFFER,
		RGFX_OP_RENDER_SET_VIEWPORT,
		RGFX_OP_RENDER_SET_VERTEX_BUFFERS,
		RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
		RGFX_OP_RENDER_SET_SCISSOR_RECT,
		RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE,
		RGFX_OP_RENDER_SET_CULL_MODE,
		RGFX_OP_RENDER_USE_RESOURCES,
		RGFX_OP_RENDER_DRAW_PRIMITIVES,
		RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES,
		RGFX_OP_ENCODER_END,
		RGFX_OP_ENCODER_BEGIN_COMPUTE,
		RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
		RGFX_OP_COMPUTE_SET_BUFFERS,
		RGFX_OP_COMPUTE_SET_TEXTURE,
		RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
		RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE,
		RGFX_OP_COMPUTE_UPDATE_FENCE,
		RGFX_OP_ENCODER_END,
		RGFX_OP_ENCODER_BEGIN_BLIT,
		RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER,
		RGFX_OP_BLIT_FILL_BUFFER,
		RGFX_OP_BLIT_GENERATE_MIPMAPS,
		RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE,
		RGFX_OP_BLIT_WAIT_FOR_FENCE,
		RGFX_OP_ENCODER_END,
	};
	const size_t want_n = sizeof(want) / sizeof(want[0]);

	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	int open = 0, begins = 0, ends = 0;
	size_t i = 0;
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		CHECK(i < want_n);
		CHECK(tag == want[i]);

		if (opGroup(tag) == RGFX_OP_GROUP_FRAMING) {
			if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
				CHECK(open == 0);
				open = 1;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_COMPUTE) {
				CHECK(open == 0);
				open = 2;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_BLIT) {
				CHECK(open == 0);
				open = 3;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_END) {
				CHECK(open != 0);
				open = 0;
				ends++;
			}
		} else {
			CHECK(static_cast<int>(opGroup(tag)) == open);
		}

		// Representative bodies across the three encoders, with the extracted handles.
		if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
			CHECK(len == 8 + sizeof(RgfxRenderPassDesc) + sizeof(RgfxRenderPassColorAttachment));
			RgfxRenderPassDesc pass;
			std::memcpy(&pass, body, sizeof(pass));
			CHECK(pass.color_count == 1);
			CHECK(pass.flags == RGFX_PASS_FLAG_RT_SIZE);
			CHECK(pass.render_target_width == 1920 && pass.render_target_height == 1080);
			RgfxRenderPassColorAttachment col;
			std::memcpy(&col, body + RGFX_RENDER_PASS_COLOR_OFFSET, sizeof(col));
			CHECK(col.texture_obj_id == 42);
			CHECK(col.load_action == MTLLoadActionClear);
			CHECK(col.store_action == MTLStoreActionStore);
			double a;
			std::memcpy(&a, &col.clear_color_bits[3], 8);
			CHECK(a == 1.0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFER) {
			RgfxOpSetVertexBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.buffer_obj_id == 10 && b.offset == 16 && b.index == 0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFERS) {
			CHECK(len == 8 + sizeof(RgfxOpSetVertexBuffers) + 2 * sizeof(RgfxBufferBinding));
			RgfxOpSetVertexBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.range_location == 0 && head.range_length == 2);
			RgfxBufferBinding e0, e1;
			std::memcpy(&e0, body + sizeof(head), sizeof(e0));
			std::memcpy(&e1, body + sizeof(head) + sizeof(e0), sizeof(e1));
			CHECK(e0.buffer_obj_id == 10 && e0.offset == 0);
			CHECK(e1.buffer_obj_id == 11 && e1.offset == 256);
		} else if (tag == RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE) {
			RgfxOpSetFragmentTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 42 && b.index == 3);
		} else if (tag == RGFX_OP_RENDER_USE_RESOURCES) {
			RgfxOpUseResources head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.count == 1 && head.usage == MTLResourceUsageRead);
			RgfxResourceBinding e;
			std::memcpy(&e, body + sizeof(head), sizeof(e));
			CHECK(e.obj_id == 9 && e.kind == RGFX_RESOURCE_KIND_BUFFER);
		} else if (tag == RGFX_OP_RENDER_DRAW_PRIMITIVES) {
			RgfxOpDrawPrimitives b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.primitive_type == MTLPrimitiveTypeTriangle);
			CHECK(b.vertex_start == 0 && b.vertex_count == 3);
		} else if (tag == RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES) {
			RgfxOpDrawIndexedPrimitivesInstances b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.index_count == 6 && b.index_buffer_obj_id == 12 && b.instance_count == 2);
		} else if (tag == RGFX_OP_COMPUTE_SET_BUFFERS) {
			RgfxComputeOpSetBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.location == 0 && head.length == 1);
			RgfxComputeBufferBinding e;
			std::memcpy(&e, body + sizeof(head), sizeof(e));
			CHECK(e.buffer_obj_id == 21 && e.offset == 64);
		} else if (tag == RGFX_OP_COMPUTE_SET_TEXTURE) {
			RgfxComputeOpSetTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 22 && b.index == 1);
		} else if (tag == RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS) {
			RgfxComputeOpDispatchThreadgroups b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.threadgroups_width == 16 && b.threadgroups_height == 8);
			CHECK(b.threads_width == 32);
		} else if (tag == RGFX_OP_COMPUTE_UPDATE_FENCE) {
			RgfxComputeOpUpdateFence b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.fence_obj_id == 30);
		} else if (tag == RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER) {
			RgfxBlitOpCopyBufferToBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.source_buffer_obj_id == 40 && b.destination_buffer_obj_id == 41);
			CHECK(b.source_offset == 16 && b.destination_offset == 32 && b.size == 4096);
		} else if (tag == RGFX_OP_BLIT_FILL_BUFFER) {
			RgfxBlitOpFillBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.buffer_obj_id == 40 && b.range_length == 256 && b.value == 0xab);
		} else if (tag == RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE) {
			RgfxBlitOpSynchronizeTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 51 && b.slice == 0 && b.level == 1);
		} else if (tag == RGFX_OP_BLIT_WAIT_FOR_FENCE) {
			RgfxBlitOpWaitForFence b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.fence_obj_id == 60);
		}
		i++;
	}
	CHECK(i == want_n);
	CHECK(open == 0);
	CHECK(begins == 3 && ends == 3);
	CHECK(walk.pos == cbh.op_bytes && cbh.op_bytes % 8 == 0);

	// commit is once: a second commit posts nothing more.
	[cb commit];
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!submit.next(&extra, &extraBytes));
}

// A parallel render encoder owns one render pass: its sub-encoders forward ops to the
// shared RgfxEncoder without their own framing, and the parallel encoder writes the
// single ENCODER_END.
void test_parallel_render_encoder()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	(void)expectMessage<RgfxQueueCreate>(control, RGFX_CMD_QUEUE_CREATE);
	id<MTLCommandBuffer> cb = [queue commandBuffer];

	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	id<MTLParallelRenderCommandEncoder> penc =
	    [cb parallelRenderCommandEncoderWithDescriptor:rp];
	CHECK(penc != nil);
	id<MTLRenderCommandEncoder> sub1 = [penc renderCommandEncoder];
	[sub1 setRenderPipelineState:handle(7)];
	[sub1 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[sub1 endEncoding]; // sub-encoder writes NO framing
	id<MTLRenderCommandEncoder> sub2 = [penc renderCommandEncoder];
	[sub2 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:3];
	[sub2 endEncoding];
	[penc endEncoding]; // the one ENCODER_END for the pass
	[cb commit];

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes));
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	const uint16_t want[] = {
		RGFX_OP_ENCODER_BEGIN_RENDER,     RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
		RGFX_OP_RENDER_DRAW_PRIMITIVES,   RGFX_OP_RENDER_DRAW_PRIMITIVES,
		RGFX_OP_ENCODER_END,
	};
	const size_t want_n = sizeof(want) / sizeof(want[0]);
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	size_t i = 0;
	int begins = 0, ends = 0;
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		CHECK(i < want_n && tag == want[i]);
		if (tag == RGFX_OP_ENCODER_BEGIN_RENDER)
			begins++;
		if (tag == RGFX_OP_ENCODER_END)
			ends++;
		i++;
	}
	CHECK(i == want_n);
	CHECK(begins == 1 && ends == 1); // one pass, one begin/end though two sub-encoders
}

// A queue must have a route, an id space and a ring.
void test_queue_guards()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:nullptr] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:nullptr
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:nullptr
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
}

// =============================================================================
// The transport layer: the region allocator and the reply reader
// =============================================================================

// The region allocator hands out aligned, non-overlapping blocks past the floor
// (RGFX_ALLOCATIONS_OFFSET), never touching the host-reserved control rings / caps
// page below it, and places ring pairs with power-of-two byte rings.
void test_region_allocator()
{
	std::vector<uint8_t> mem(0x800000, 0xff); // 8 MiB, pre-dirtied like BAR memory
	RgfxRegion region(mem.data(), mem.size());
	CHECK(region.floor() == RGFX_ALLOCATIONS_OFFSET);
	CHECK(region.size() == mem.size());

	RgfxRegion::Block a = region.allocate(0x1000);
	CHECK(a.ok);
	CHECK(a.offset >= RGFX_ALLOCATIONS_OFFSET); // never below the floor
	CHECK(a.offset % 0x1000 == 0);              // aligned (default page)
	CHECK(a.ptr == region.base() + a.offset);
	CHECK(a.size == 0x1000);

	RgfxRegion::Block b = region.allocate(0x800, 256);
	CHECK(b.ok);
	CHECK(b.offset % 256 == 0);
	CHECK(a.offset + a.size <= b.offset); // non-overlap with a

	RgfxRegion::Block c = region.allocate(0x1234, 64);
	CHECK(c.ok);
	CHECK(b.offset + b.size <= c.offset); // non-overlap with b
	CHECK(region.used() >= (c.offset - region.floor()) + c.size);

	// A non-power-of-two alignment, and a request past the region's end, both fail.
	CHECK(!region.allocate(16, 3).ok);
	CHECK(!region.allocate(mem.size(), 0x1000).ok);

	// A submission + completion ring pair: both offsets past the floor, non-overlapping,
	// each a ring header plus a power-of-two byte ring.
	RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
	CHECK(rings.ok);
	CHECK(rings.submit_offset >= RGFX_ALLOCATIONS_OFFSET);
	CHECK(rings.submit_data_size == 0x8000);
	CHECK(rings.complete_data_size == 0x8000);
	CHECK(rings.complete_offset >= rings.submit_offset + RGFX_RING_HEADER_SIZE + 0x8000);
	// A completion ring of 0 places none (the single-ring model).
	RgfxRegion::RingPair submitOnly = region.allocateRingPair(0x1000, 0);
	CHECK(submitOnly.ok);
	CHECK(submitOnly.complete_offset == 0);
	// A non-power-of-two byte ring is refused.
	CHECK(!region.allocateRingPair(0x7000, 0).ok);

	// contains / offsetOf: inside the allocable region, below the floor is out.
	CHECK(region.contains(mem.data() + RGFX_ALLOCATIONS_OFFSET, 16));
	CHECK(!region.contains(mem.data(), 16)); // below the floor
	CHECK(!region.contains(mem.data() + mem.size() - 8, 16)); // runs off the end
	CHECK(region.offsetOf(mem.data() + a.offset) == a.offset);
}

// newCommandQueue's path: a queue built over a ring pair the region allocator placed
// carries those offsets in its RGFX_CMD_QUEUE_CREATE (the device category does the same
// with its region; here the factory is driven directly with a fixture region).
void test_queue_carries_allocated_offsets()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);

	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
	CHECK(rings.ok);

	// The submission ring's writer over the region at the allocated submit offset.
	RgfxRingWriter submitWriter(region.base() + rings.submit_offset, rings.submit_data_size, 1,
	                            TestRing::doorbell, nullptr);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:rings.submit_offset
	                                                completeOffset:rings.complete_offset
	                                              completeDataSize:rings.complete_data_size
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	RgfxQueueCreate qc = expectMessage<RgfxQueueCreate>(control, RGFX_CMD_QUEUE_CREATE);
	CHECK(qc.submit_offset == rings.submit_offset);
	CHECK(qc.complete_offset == rings.complete_offset);
	CHECK(qc.submit_data_size == rings.submit_data_size);
	CHECK(qc.complete_data_size == rings.complete_data_size);
	CHECK(qc.submit_offset >= RGFX_ALLOCATIONS_OFFSET);
}

// A host command-buffer completion the reader dispatches: the CB's completed handler
// fires with the buffer, its status flips, and an error completion carries the error.
void test_reply_reader_completion()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	queue.replyReader = &reader;
	(void)expectMessage<RgfxQueueCreate>(control, RGFX_CMD_QUEUE_CREATE);

	// --- a clean completion ---
	__block int fired = 0;
	__block id<MTLCommandBuffer> handlerGot = nil;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		fired++;
		handlerGot = done;
	}];
	// No time is known before the commit, and the GPU's none before the completion.
	CHECK(cb.kernelStartTime == 0 && cb.kernelEndTime == 0);
	CHECK(cb.GPUStartTime == 0 && cb.GPUEndTime == 0);
	[cb commit];
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	CHECK(cb.kernelStartTime > 0 && cb.kernelEndTime >= cb.kernelStartTime);
	CHECK(cb.GPUStartTime == 0 && cb.GPUEndTime == 0);

	// Read the submission seqno the host's completion must carry.
	RgfxMsgHeader submitHdr = {};
	std::vector<uint8_t> submitBytes;
	CHECK(submit.next(&submitHdr, &submitBytes));
	CHECK(submitHdr.type == RGFX_CMD_COMMAND_BUFFER);
	uint64_t seqno = submitHdr.seqno;

	// Nothing has arrived: no dispatch, handler unfired.
	CHECK(reader.pump() == 0);
	CHECK(fired == 0);
	CHECK(reader.awaitingCompletion(seqno));

	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = seqno;
	done.cb_obj_id = cbObjId;
	done.status = RGFX_CB_OK;
	// A 2 ms run on the device's clock.
	done.gpu_start_ns = 5'000'000'000;
	done.gpu_end_ns = 5'002'000'000;
	CHECK(completion.postFromHost(&done, sizeof(done)));
	CHECK(reader.pump() == 1);
	CHECK(fired == 1);
	CHECK(handlerGot == cb);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(cb.error == nil);
	// The run ends at the delivery, in the guest's time, and lasts what the device measured,
	// starting no earlier than the CPU posted the buffer.
	CHECK(cb.GPUEndTime >= cb.GPUStartTime && cb.GPUStartTime >= cb.kernelEndTime);
	CHECK(cb.GPUEndTime - cb.GPUStartTime <= 0.002 + 1e-9);
	CHECK(!reader.awaitingCompletion(seqno)); // the one-shot was removed

	// --- an error completion on a second buffer, waited on ---
	id<MTLCommandBuffer> cb2 = [queue commandBuffer];
	uint64_t cb2ObjId = ((RgfxResource *)cb2).rgfxObjId;
	[cb2 commit];
	RgfxMsgHeader submitHdr2 = {};
	std::vector<uint8_t> submitBytes2;
	CHECK(submit.next(&submitHdr2, &submitBytes2));
	uint64_t seqno2 = submitHdr2.seqno;

	RgfxCbComplete failed = {};
	failed.header.type = RGFX_EVT_CB_COMPLETE;
	failed.header.size = sizeof(failed);
	failed.header.seqno = seqno2;
	failed.cb_obj_id = cb2ObjId;
	failed.status = RGFX_CB_TIMEOUT;
	failed.error_code = 2; // MTLCommandBufferErrorTimeout
	CHECK(completion.postFromHost(&failed, sizeof(failed)));
	[cb2 waitUntilCompleted]; // pumps until the completion lands
	CHECK(cb2.status == MTLCommandBufferStatusError);
	CHECK(cb2.error != nil);
	CHECK(cb2.error.code == 2);
	// No run length known: the run starts as it ends.
	CHECK(cb2.GPUEndTime > 0 && cb2.GPUStartTime == cb2.GPUEndTime);

	// A descriptor's error options are kept for the buffer to answer, and no buffer has logs.
	MTLCommandBufferDescriptor *desc = [[MTLCommandBufferDescriptor alloc] init];
	desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
	id<MTLCommandBuffer> cb3 = [queue commandBufferWithDescriptor:desc];
	CHECK(cb3.errorOptions == MTLCommandBufferErrorOptionEncoderExecutionStatus);
	CHECK(cb.errorOptions == MTLCommandBufferErrorOptionNone);
	NSUInteger logs = 0;
	for (id log in cb3.logs) {
		(void)log;
		logs++;
	}
	CHECK(logs == 0);
}

// A host pipeline reflection the reader dispatches fills the pipeline-state wrapper, so
// its reflection getters report the host's values instead of placeholders.
void test_reply_reader_reflection()
{
	Fixture fx; // control ring + serialiser

	const uint8_t libBytes[] = { 1, 2, 3, 4 };
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	id<MTLFunction> cfn = [lib newFunctionWithName:@"kmain"];

	NSError *err = nil;
	RgfxComputePipelineState *ps =
	    [RgfxComputePipelineState computePipelineStateWithFunction:cfn
	                                                        device:nil
	                                                    serializer:&fx.ser
	                                                         error:&err];
	CHECK(ps != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);

	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);
	[ps rgfxRegisterWithReplyReader:&reader];
	CHECK(reader.awaitingReflection(ps.rgfxSeqno));

	// Before the reply the getters answer placeholders.
	CHECK([ps maxTotalThreadsPerThreadgroup] == 1024);
	CHECK(ps.gpuResourceID._impl == 0);

	RgfxPipelineReflection refl = {};
	refl.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	refl.header.size = sizeof(refl);
	refl.header.seqno = ps.rgfxSeqno;
	refl.obj_id = ps.rgfxObjId;
	refl.gpu_resource_id = 0xABCDEF;
	refl.max_total_threads_per_threadgroup = 512;
	refl.static_threadgroup_memory_length = 2048;
	refl.thread_execution_width = 64;
	refl.status = RGFX_PIPELINE_OK;
	refl.kind = RGFX_PIPELINE_KIND_COMPUTE;
	CHECK(completion.postFromHost(&refl, sizeof(refl)));
	CHECK(reader.pump() == 1);

	CHECK([ps maxTotalThreadsPerThreadgroup] == 512);
	CHECK([ps threadExecutionWidth] == 64);
	CHECK([ps staticThreadgroupMemoryLength] == 2048);
	CHECK(ps.gpuResourceID._impl == 0xABCDEF);
	CHECK(!reader.awaitingReflection(ps.rgfxSeqno)); // one-shot removed
	CHECK(![ps rgfxRefused]);

	// A refused create: the state reports the refusal, which the device's creators answer as
	// nil and an NSError.
	RgfxComputePipelineState *refusedPs =
	    [RgfxComputePipelineState computePipelineStateWithFunction:cfn
	                                                        device:nil
	                                                    serializer:&fx.ser
	                                                         error:&err];
	CHECK(refusedPs != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	[refusedPs rgfxRegisterWithReplyReader:&reader];
	CHECK(![refusedPs rgfxRefused]);
	RgfxPipelineReflection no = {};
	no.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	no.header.size = sizeof(no);
	no.header.seqno = refusedPs.rgfxSeqno;
	no.obj_id = refusedPs.rgfxObjId;
	no.status = RGFX_PIPELINE_REFUSED;
	no.kind = RGFX_PIPELINE_KIND_COMPUTE;
	CHECK(completion.postFromHost(&no, sizeof(no)));
	CHECK(reader.pump() == 1);
	CHECK([refusedPs rgfxRefused]);
}

// A host event-signalled message the reader dispatches updates the shared-event value
// mirror and fires the notifyListener blocks that have been reached.
void test_reply_reader_event()
{
	// The reader is declared before the inner pool so it outlives the autoreleased
	// event, whose dealloc deregisters its listener from the reader (as the device's
	// reader outlives its events in the guest).
	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);
	@autoreleasepool {
	Fixture fx;

	const uint64_t initial = 5;
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:initial
	                                                           device:nil
	                                                       serializer:&fx.ser];
	CHECK(ev != nil);
	(void)expectMessage<RgfxSharedEventCreate>(fx.ring, RGFX_CMD_SHARED_EVENT_CREATE);

	[ev rgfxRegisterWithReplyReader:&reader];
	CHECK(ev.signaledValue == initial);

	// A listener that fires only once the value reaches 50.
	__block uint64_t notified = 0;
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
	[ev notifyListener:listener
	           atValue:50
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             notified = v;
	             }];

	// The host raises the event to 42: the mirror updates, the 50-listener holds.
	RgfxEventSignalled sig = {};
	sig.header.type = RGFX_EVT_EVENT_SIGNALLED;
	sig.header.size = sizeof(sig);
	sig.header.seqno = 0;
	sig.obj_id = ev.rgfxObjId;
	sig.value = 42;
	CHECK(completion.postFromHost(&sig, sizeof(sig)));
	CHECK(reader.pump() == 1);
	CHECK(ev.signaledValue == 42);
	CHECK(notified == 0);

	// The host raises it to 50: the mirror updates and the listener fires.
	RgfxEventSignalled sig2 = {};
	sig2.header.type = RGFX_EVT_EVENT_SIGNALLED;
	sig2.header.size = sizeof(sig2);
	sig2.obj_id = ev.rgfxObjId;
	sig2.value = 50;
	CHECK(completion.postFromHost(&sig2, sizeof(sig2)));
	CHECK(reader.pump() == 1);
	CHECK(ev.signaledValue == 50);
	CHECK(notified == 50);
	} // @autoreleasepool: ev drains here, while reader still lives
}

// --- the completion thread ---------------------------------------------------

// The kext's side of the completion thread, in user space: the wake count the thread
// sleeps on (kRgfxAccelMethodWaitCompletions), and the host's post, which raises the
// vector where the ring is armed and so wakes the thread (queueInterrupt).
struct FakeKext {
	std::mutex lock;
	std::condition_variable changed;
	uint64_t count = 0;
	bool closed = false;
	int waits = 0;

	void wake()
	{
		std::lock_guard<std::mutex> hold(lock);
		count++;
		changed.notify_all();
	}

	void close()
	{
		std::lock_guard<std::mutex> hold(lock);
		closed = true;
		changed.notify_all();
	}

	int waitCount()
	{
		std::lock_guard<std::mutex> hold(lock);
		return waits;
	}

	static RgfxCompletionThread::WaitResult wait(void *context, uint64_t last, uint64_t *now)
	{
		FakeKext *k = static_cast<FakeKext *>(context);
		std::unique_lock<std::mutex> hold(k->lock);
		k->waits++;
		k->changed.wait(hold, [&] { return k->count != last || k->closed; });
		*now = k->count;
		return k->closed ? RgfxCompletionThread::kStop : RgfxCompletionThread::kWoke;
	}
};

uint32_t ringArmed(TestRing &ring)
{
	return __atomic_load_n(reinterpret_cast<uint32_t *>(ring.mem.data() +
	                                                    __builtin_offsetof(RgfxRingHeader, armed)),
	                       __ATOMIC_SEQ_CST);
}

// The host's post (ring.rs Producer::post): the record published, a full fence, then
// `armed` read and cleared, the vector raised where it was set.
void hostPost(TestRing &ring, FakeKext &kext, const void *msg, uint32_t size)
{
	CHECK(ring.postFromHost(msg, size));
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	uint32_t *armed = reinterpret_cast<uint32_t *>(ring.mem.data() +
	                                               __builtin_offsetof(RgfxRingHeader, armed));
	if (__atomic_exchange_n(armed, 0u, __ATOMIC_SEQ_CST) == 1)
		kext.wake();
}

RgfxCbComplete cbComplete(uint64_t seqno, uint64_t cbObjId)
{
	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = seqno;
	done.cb_obj_id = cbObjId;
	done.status = RGFX_CB_OK;
	return done;
}

// Waits up to 5 s for `pred`, polling.
template <typename F> bool eventually(F pred)
{
	for (int i = 0; i < 5000; i++) {
		if (pred())
			return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return pred();
}

// A queue over its own submission and completion rings, as the device makes one, with an
// owned reader the completion thread can drive.
// The rings and the serialiser are never freed, as Fixture's: the queue lives on while a
// command buffer the test let go holds it.
struct QueueFixture {
	TestRing &control = *new TestRing(0x8000);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit;
	RgfxRingWriter &submitWriter;
	TestRing &completion = *new TestRing(0x8000);
	RgfxCommandQueue *queue = nil;

	// A queue whose submission ring has `submitData` bytes, made with `maxCount`, its doorbell
	// `doorbell` (with `doorbellContext`).
	explicit QueueFixture(uint32_t submitData = 0x8000, NSUInteger maxCount = 0,
	                      RgfxDoorbell doorbell = TestRing::doorbell, void *doorbellContext = nullptr)
	    : submit(*new TestRing(submitData)),
	      submitWriter(*new RgfxRingWriter(submit.base(), submit.dataSize, 1, doorbell,
	                                       doorbellContext))
	{
		control.setConsumerIdle(1);
		submit.setConsumerIdle(1);
		queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                 submitOffset:0
		                               completeOffset:0
		                             completeDataSize:0
		                                          ids:&ids
		                                     maxCount:maxCount
		                                       device:nil
		                                   serializer:&ser];
		[queue adoptReplyReader:new RgfxReplyReader(completion.base(), completion.dataSize)];
	}

	// The submission seqno of the next command buffer committed on the queue.
	uint64_t nextSeqno()
	{
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes));
		CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
		return h.seqno;
	}
};

// armAndCheck sets the ring's `armed` and says whether records wait; a ringless reader
// has nothing to arm.
void test_reply_reader_arm()
{
	RgfxReplyReader ringless;
	CHECK(!ringless.armAndCheck());
	TestRing ring(0x1000);
	RgfxReplyReader reader(ring.base(), ring.dataSize);
	CHECK(!reader.driven());
	CHECK(!reader.armAndCheck());
	CHECK(ringArmed(ring) == 1);
	RgfxCbComplete done = cbComplete(7, 1);
	CHECK(ring.postFromHost(&done, sizeof(done)));
	CHECK(reader.armAndCheck()); // a record waits: pump before sleeping
	reader.pump();
	CHECK(!reader.armAndCheck());
	reader.setDriven(true);
	CHECK(reader.driven());
}

// With nobody waiting, the completion thread runs each command buffer's scheduled then
// completed handlers as the host posts its completion, in the ring's order, on its own
// thread, and leaves the ring drained and armed.
void test_completion_thread_runs_handlers()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext(); // outlives the thread, which may sleep in it
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	thread.add([f.queue sharedReplyReader]);
	CHECK(![f.queue sharedReplyReader]->driven()); // not started: the waiters pump
	CHECK(thread.start());
	CHECK(thread.running());
	CHECK([f.queue sharedReplyReader]->driven());
	CHECK(ringArmed(f.completion) == 1);
	CHECK(eventually([&] { return kext->waitCount() >= 1; })); // asleep in the kext

	std::mutex lock;
	std::vector<std::string> order;
	__block std::mutex *orderLock = &lock;
	__block std::vector<std::string> *orderLog = &order;
	__block int offThread = 0;
	dispatch_semaphore_t last = dispatch_semaphore_create(0);
	NSMutableArray<id<MTLCommandBuffer>> *cbs = [NSMutableArray array];
	std::vector<uint64_t> seqnos;
	for (int i = 0; i < 3; i++) {
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		std::string n = std::to_string(i);
		[cb addScheduledHandler:^(id<MTLCommandBuffer> done) {
			(void)done;
			std::lock_guard<std::mutex> hold(*orderLock);
			orderLog->push_back("s" + n);
			if (!RgfxCompletionThread::onCurrentThread())
				offThread++;
		}];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
			std::lock_guard<std::mutex> hold(*orderLock);
			orderLog->push_back("c" + n + (done.status == MTLCommandBufferStatusCompleted ? "" : "!"));
			if (!RgfxCompletionThread::onCurrentThread())
				offThread++;
			if (n == "2")
				dispatch_semaphore_signal(last);
		}];
		[cb commit];
		[cbs addObject:cb];
		seqnos.push_back(f.nextSeqno());
	}
	// The first completion wakes the thread; the next two land while it may be busy, and
	// are taken by the same pass or the next.
	for (int i = 0; i < 3; i++) {
		RgfxCbComplete done = cbComplete(seqnos[i], ((RgfxResource *)cbs[i]).rgfxObjId);
		hostPost(f.completion, *kext, &done, sizeof(done));
	}
	CHECK(dispatch_semaphore_wait(last, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	{
		std::lock_guard<std::mutex> hold(lock);
		std::vector<std::string> want = {"s0", "c0", "s1", "c1", "s2", "c2"};
		CHECK(order == want);
		CHECK(offThread == 0);
	}
	for (id<MTLCommandBuffer> cb in cbs)
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(eventually([&] { return f.completion.consumerCount() == f.completion.produced; }));
	CHECK(eventually([&] { return ringArmed(f.completion) == 1; })); // armed for the next

	// A waiter after the fact returns at once; the handlers ran once.
	[cbs[0] waitUntilCompleted];
	{
		std::lock_guard<std::mutex> hold(lock);
		CHECK(order.size() == 6);
	}
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// waitUntilCompleted on a driven reader blocks until the thread delivers, and the
// handlers run once, before the wait returns.
void test_wait_with_the_completion_thread()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);
	CHECK([f.queue sharedReplyReader]->driven());

	std::atomic<int> fired{0};
	std::atomic<int> *firedp = &fired;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // the waiter must outwait it
		(*firedp)++;
	}];
	[cb commit];
	uint64_t seqno = f.nextSeqno();
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	TestRing *ring = &f.completion;
	std::thread host([ring, kext, seqno, cbObjId] {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		RgfxCbComplete done = cbComplete(seqno, cbObjId);
		hostPost(*ring, *kext, &done, sizeof(done));
	});
	[cb waitUntilCompleted];
	CHECK(fired.load() == 1); // the handler ran before the wait returned
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	host.join();
	[cb waitUntilCompleted];
	CHECK(fired.load() == 1); // and only once

	// A handler waiting for a later buffer, on the completion thread: it takes the ring
	// itself, so neither hangs.
	id<MTLCommandBuffer> first = [f.queue commandBuffer];
	id<MTLCommandBuffer> second = [f.queue commandBuffer];
	std::atomic<int> nested{0};
	std::atomic<int> *nestedp = &nested;
	dispatch_semaphore_t both = dispatch_semaphore_create(0);
	[first addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		[second waitUntilCompleted];
		*nestedp = second.status == MTLCommandBufferStatusCompleted ? 1 : 2;
		dispatch_semaphore_signal(both);
	}];
	[first commit];
	uint64_t s1 = f.nextSeqno();
	[second commit];
	uint64_t s2 = f.nextSeqno();
	RgfxCbComplete d1 = cbComplete(s1, ((RgfxResource *)first).rgfxObjId);
	hostPost(f.completion, *kext, &d1, sizeof(d1));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	RgfxCbComplete d2 = cbComplete(s2, ((RgfxResource *)second).rgfxObjId);
	hostPost(f.completion, *kext, &d2, sizeof(d2));
	CHECK(dispatch_semaphore_wait(both, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(nested.load() == 1);
	[second waitUntilCompleted];
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// Once the kext ends the wait (the process's context is gone) or the thread is
// destroyed, the readers go undriven, and waitUntilCompleted takes the ring itself.
void test_completion_thread_stops()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);
	CHECK([f.queue sharedReplyReader]->driven());
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	CHECK(![f.queue sharedReplyReader]->driven());

	__block int fired = 0;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		fired++;
	}];
	[cb commit];
	RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted]; // pumps: nothing else takes the ring
	CHECK(fired == 1);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);

	// A thread destroyed while it sleeps lets its readers go at once.
	QueueFixture g;
	FakeKext *kext2 = new FakeKext();
	{
		RgfxCompletionThread doomed(&FakeKext::wait, kext2);
		CHECK(doomed.start());
		doomed.add([g.queue sharedReplyReader]);
		CHECK([g.queue sharedReplyReader]->driven());
		CHECK(eventually([&] { return kext2->waitCount() >= 1; }));
	}
	CHECK(![g.queue sharedReplyReader]->driven());
	kext2->wake(); // the detached thread wakes, sees it is stopping, and ends
	}
}

// A command buffer too large for half its queue's ring puts its whole COMMAND_BUFFER message
// in a region block, which the ring names with a COMMAND_BUFFER_IN_REGION whose seqno the
// completion carries; the block goes back as the buffer completes.
struct BlockLog {
	std::vector<uint64_t> allocated;
	std::vector<uint64_t> released;
	uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x10000;
	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		BlockLog *log = static_cast<BlockLog *>(c);
		uint64_t at = (log->next + align - 1) & ~(align - 1);
		log->next = at + size;
		log->allocated.push_back(at);
		*offset = at;
		return true;
	}
	static void release(void *c, uint64_t offset)
	{
		static_cast<BlockLog *>(c)->released.push_back(offset);
	}
};

void test_command_buffer_out_of_line()
{
	@autoreleasepool {
	QueueFixture f(0x1000); // up to 2 KiB on the ring
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	f.queue.region = &region;

	// A small buffer still rides the ring.
	id<MTLCommandBuffer> small = [f.queue commandBuffer];
	[small commit];
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(f.submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(blocks.allocated.empty());
	RgfxCbComplete smallDone = cbComplete(h.seqno, ((RgfxResource *)small).rgfxObjId);
	CHECK(f.completion.postFromHost(&smallDone, sizeof(smallDone)));
	[small waitUntilCompleted];

	// 200 fills: far past half the ring.
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	for (int i = 0; i < 200; i++)
		[blit fillBuffer:handle(40) range:NSMakeRange(4 * i, 4) value:(uint8_t)i];
	[blit endEncoding];
	__block int fired = 0;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		fired++;
	}];
	[cb commit];
	CHECK(f.submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
	CHECK(h.size == sizeof(RgfxCommandBufferInRegion));
	RgfxCommandBufferInRegion ref = {};
	std::memcpy(&ref, bytes.data(), sizeof(ref));
	CHECK(blocks.allocated.size() == 1 && ref.region_offset == blocks.allocated[0]);
	CHECK(ref.byte_length > 2048 && ref.byte_length % 8 == 0);

	// The block holds the whole message: a COMMAND_BUFFER of that length, seqno 0, its ops.
	RgfxCommandBuffer body = {};
	std::memcpy(&body, mem.data() + ref.region_offset, sizeof(body));
	CHECK(body.header.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(body.header.size == ref.byte_length);
	CHECK(body.header.seqno == 0);
	CHECK(body.cb_obj_id == ((RgfxResource *)cb).rgfxObjId);
	CHECK(body.op_bytes == ref.byte_length - RGFX_COMMAND_BUFFER_OPS_OFFSET);
	RgfxOpHeader first = {};
	std::memcpy(&first, mem.data() + ref.region_offset + RGFX_COMMAND_BUFFER_OPS_OFFSET,
	            sizeof(first));
	CHECK(first.tag == RGFX_OP_ENCODER_BEGIN_BLIT);

	// Its completion carries the reference's seqno; the block goes back then.
	CHECK(blocks.released.empty());
	RgfxCbComplete done = cbComplete(h.seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(fired == 1);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(blocks.released.size() == 1 && blocks.released[0] == ref.region_offset);

	// With no region, a buffer too large for the ring fails as out of memory, off the
	// committing thread, and never reaches the ring.
	QueueFixture g(0x1000);
	id<MTLCommandBuffer> lost = [g.queue commandBuffer];
	id<MTLBlitCommandEncoder> b2 = [lost blitCommandEncoder];
	for (int i = 0; i < 200; i++)
		[b2 fillBuffer:handle(40) range:NSMakeRange(4 * i, 4) value:1];
	[b2 endEncoding];
	dispatch_semaphore_t failed = dispatch_semaphore_create(0);
	__block int failedOnMain = -1;
	[lost addCompletedHandler:^(id<MTLCommandBuffer> d) {
		failedOnMain = pthread_main_np();
		(void)d;
		dispatch_semaphore_signal(failed);
	}];
	[lost commit];
	CHECK(dispatch_semaphore_wait(failed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(lost.status == MTLCommandBufferStatusError);
	CHECK(lost.error.code == MTLCommandBufferErrorOutOfMemory);
	CHECK(failedOnMain == 0);
	CHECK(!g.submit.next(&h, &bytes));
	f.queue.region = nullptr;
	}
}

// A buffer's completion is expected before the host can see the buffer: at the doorbell that
// publishing it rings, the reader already awaits its seqno.
struct ExpectAtDoorbell {
	TestRing *submit = nullptr;
	RgfxReplyReader *reader = nullptr;
	int rung = 0;
	bool awaited = false;
	uint64_t seqno = 0;
	static void ring(void *c, uint32_t)
	{
		ExpectAtDoorbell *e = static_cast<ExpectAtDoorbell *>(c);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		if (e->submit->next(&h, &bytes)) {
			e->rung++;
			e->seqno = h.seqno;
			e->awaited = e->reader->awaitingCompletion(h.seqno);
		}
	}
};

void test_completion_expected_before_publish()
{
	@autoreleasepool {
	ExpectAtDoorbell e;
	QueueFixture f(0x8000, 0, &ExpectAtDoorbell::ring, &e);
	e.submit = &f.submit;
	e.reader = [f.queue sharedReplyReader].get();
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb commit];
	CHECK(e.rung == 1);
	CHECK(e.awaited);
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	RgfxCbComplete done = cbComplete(e.seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	}
}

// -commandBuffer blocks while maxCommandBufferCount buffers of the queue are alive and not
// completed, as Metal's does; a buffer that completes, or one released uncommitted, gives its
// slot back; and a completed handler that makes the next buffer does not wait for its own.
void test_command_buffer_back_pressure()
{
	@autoreleasepool {
	QueueFixture f(0x8000, 2);
	CHECK(f.queue.maxCommandBufferCount == 2);
	QueueFixture d;
	CHECK(d.queue.maxCommandBufferCount == 64);

	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);

	id<MTLCommandBuffer> a = [f.queue commandBuffer];
	id<MTLCommandBuffer> b = [f.queue commandBuffer];
	[a commit];
	uint64_t sa = f.nextSeqno();
	[b commit];
	uint64_t sb = f.nextSeqno();

	// A third waits for a's completion, posted 50 ms from now.
	RgfxCommandQueue *queue = f.queue;
	std::atomic<bool> got{false};
	std::atomic<bool> *gotp = &got;
	id<MTLCommandBuffer> third = nil;
	id<MTLCommandBuffer> __strong *thirdp = &third;
	dispatch_semaphore_t made = dispatch_semaphore_create(0);
	std::thread maker([queue, gotp, made, thirdp] {
		@autoreleasepool {
			*thirdp = [queue commandBuffer];
			gotp->store(true);
			dispatch_semaphore_signal(made);
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	CHECK(!got.load()); // still blocked: two in flight
	RgfxCbComplete done = cbComplete(sa, ((RgfxResource *)a).rgfxObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	CHECK(dispatch_semaphore_wait(made, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	maker.join();
	CHECK(got.load() && third != nil);

	// b and the third hold both slots. Released uncommitted, the third gives its slot back.
	third = nil;
	id<MTLCommandBuffer> fourth = nil;
	@autoreleasepool {
		fourth = [f.queue commandBuffer]; // did not block
	}
	CHECK(fourth != nil);
	RgfxCbComplete bDone = cbComplete(sb, ((RgfxResource *)b).rgfxObjId);
	hostPost(f.completion, *kext, &bDone, sizeof(bDone));
	[b waitUntilCompleted];
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));

	// With one slot, a completed handler that makes the next buffer has the slot back.
	QueueFixture g(0x8000, 1);
	id<MTLCommandBuffer> first = [g.queue commandBuffer];
	__block id<MTLCommandBuffer> next = nil;
	RgfxCommandQueue *gq = g.queue;
	[first addCompletedHandler:^(id<MTLCommandBuffer> x) {
		(void)x;
		next = [gq commandBuffer];
	}];
	[first commit];
	RgfxCbComplete d1 = cbComplete(g.nextSeqno(), ((RgfxResource *)first).rgfxObjId);
	CHECK(g.completion.postFromHost(&d1, sizeof(d1)));
	[first waitUntilCompleted]; // pumps: the handler runs here, and gets its buffer
	CHECK(next != nil);
	}
}

// A release waits for every command buffer begun before it, and runs at once with none in
// flight; one begun after it does not hold it; releases run in the order retired.
void test_retirement()
{
	RgfxRetirement r;
	std::vector<int> ran;
	r.retire([&] { ran.push_back(0); });
	CHECK(ran == std::vector<int>({0}));
	uint64_t a = r.begin();
	uint64_t b = r.begin();
	r.retire([&] { ran.push_back(1); });
	uint64_t c = r.begin(); // after the release: does not hold it
	r.retire([&] { ran.push_back(2); });
	CHECK(r.waiting() == 2);
	r.end(b);
	CHECK(ran.size() == 1);
	r.end(a);
	CHECK(ran == std::vector<int>({0, 1})); // 2 waits for c
	r.end(c);
	CHECK(ran == std::vector<int>({0, 1, 2}));
	CHECK(r.waiting() == 0);
}

// A texture let go while a command buffer is in flight stays on the host until the buffer
// completes: its OBJECT_DESTROY comes after the buffer's handlers, and its resource id goes
// to the next texture only after that.
void test_texture_destroy_waits_for_buffers()
{
	// Nothing left in flight by earlier tests.
	CHECK(RgfxRetirement::process().waiting() == 0 && RgfxRetirement::process().live() == 0);
	@autoreleasepool {
	QueueFixture f;
	(void)expectMessage<RgfxQueueCreate>(f.control, RGFX_CMD_QUEUE_CREATE);
	int calls = 0;
	f.ids.setResourceIdSource(&oneChunk, &calls);
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO];
	uint64_t texObjId = 0, texResourceId = 0;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	@autoreleasepool {
		RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
		                                                 heap:RGFX_OBJ_NONE
		                                           heapOffset:0
		                                               device:nil
		                                           serializer:&f.ser];
		CHECK(tex != nil);
		texObjId = tex.rgfxObjId;
		texResourceId = [tex gpuResourceID]._impl;
		(void)expectMessage<RgfxTextureCreate>(f.control, RGFX_CMD_TEXTURE_CREATE);
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit generateMipmapsForTexture:tex];
		[blit endEncoding];
		[cb commit];
	} // the app lets the texture go while the buffer is in flight
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!f.control.next(&h, &bytes)); // no destroy yet
	CHECK(RgfxRetirement::process().waiting() == 1);
	// A texture made meanwhile gets another id: the first's is still the host's.
	@autoreleasepool {
		RgfxTexture *other = [RgfxTexture textureWithDescriptor:desc
		                                                   heap:RGFX_OBJ_NONE
		                                             heapOffset:0
		                                                 device:nil
		                                             serializer:&f.ser];
		CHECK(other != nil && [other gpuResourceID]._impl != texResourceId);
		(void)expectMessage<RgfxTextureCreate>(f.control, RGFX_CMD_TEXTURE_CREATE);
	}
	CHECK(RgfxRetirement::process().waiting() == 2);

	__block bool destroyedBeforeHandler = true;
	TestRing *control = &f.control;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> d) {
		(void)d;
		RgfxMsgHeader hh = {};
		std::vector<uint8_t> bb;
		destroyedBeforeHandler = control->next(&hh, &bb);
	}];
	RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(!destroyedBeforeHandler);
	auto gone = expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY);
	CHECK(gone.obj_id == texObjId);
	(void)expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY); // `other`'s
	CHECK(RgfxRetirement::process().waiting() == 0);
	}
}

// A buffer whose completion never comes, released with its queue and reader, ends
// nothing (its completion group is left at dealloc).
void test_uncompleted_buffer_releases()
{
	@autoreleasepool {
		QueueFixture f;
		{
			RgfxReplyReader reader(f.completion.base(), f.completion.dataSize);
			f.queue.replyReader = &reader;
			@autoreleasepool {
				id<MTLCommandBuffer> cb = [f.queue commandBuffer];
				[cb commit];
				(void)f.nextSeqno();
				CHECK(cb.status == MTLCommandBufferStatusCommitted);
			}
			CHECK(RgfxRetirement::process().live() == 1);
			f.queue.replyReader = nullptr;
		} // the reader goes, and with it its handler, the buffer's last hold
		CHECK(RgfxRetirement::process().live() == 0); // its ticket ended at its dealloc
	}
}

} // namespace

// The layout of a Managed texture's CPU copy, for the cases crates/reims-graphics/src/managed.rs
// holds too: a 2D RGBA8 texture with three levels, a BC1 one whose sides are no multiple of its
// blocks, a cube array, and a 3D one; and the blocks of a few formats.
void test_managed_layout()
{
	auto desc = [](uint32_t type, uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
	               uint32_t array) {
		RgfxTextureDescriptor t = {};
		t.pixel_format = 70;
		t.texture_type = type;
		t.width = w;
		t.height = h;
		t.depth = d;
		t.mipmap_level_count = levels;
		t.array_length = array;
		t.sample_count = 1;
		return t;
	};
	auto level = [](const RgfxManagedLayout &l, uint32_t i, uint64_t offset, uint64_t row,
	                uint64_t image, uint64_t slice) {
		const RgfxManagedLevel &lv = l.level(i);
		return lv.offset == offset && lv.bytesPerRow == row && lv.bytesPerImage == image &&
		       lv.bytesPerSlice == slice;
	};
	RgfxFormatBlock rgba = {}, bc1 = {}, astc = {}, none = {};
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatRGBA8Unorm, &rgba) && rgba.bytes == 4 &&
	      rgba.width == 1 && rgba.linear);
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatBC1_RGBA, &bc1) && bc1.bytes == 8 &&
	      bc1.width == 4 && bc1.height == 4 && !bc1.linear);
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatASTC_10x8_LDR, &astc) && astc.bytes == 16 &&
	      astc.width == 10 && astc.height == 8);
	CHECK(!rgfxPixelFormatBlock(MTLPixelFormatDepth32Float, &none));
	CHECK(!rgfxPixelFormatBlock(191, &none));

	RgfxManagedLayout l;
	CHECK(l.init(desc(2, 100, 60, 1, 3, 1), rgba));
	CHECK(level(l, 0, 0, 512, 30720, 30720) && level(l, 1, 30720, 256, 7680, 7680) &&
	      level(l, 2, 38400, 256, 3840, 3840));
	CHECK(l.size() == 42240 && l.slices() == 1);
	CHECK(l.init(desc(2, 30, 30, 1, 2, 1), bc1));
	CHECK(level(l, 0, 0, 256, 2048, 2048) && level(l, 1, 2048, 256, 1024, 1024));
	uint64_t at = 0;
	const uint64_t bcOrigin[3] = {4, 8, 0}, bcSize[3] = {26, 22, 1}, offBlock[3] = {2, 0, 0};
	CHECK(l.region(0, 0, bcOrigin, bcSize, &at) && at == 512 + 8);
	CHECK(!l.region(0, 0, offBlock, bcSize, &at));
	CHECK(l.init(desc(6, 16, 16, 1, 2, 2), rgba));
	CHECK(l.slices() == 12 && level(l, 0, 0, 256, 4096, 4096) &&
	      level(l, 1, 49152, 256, 2048, 2048) && l.size() == 49152 + 12 * 2048);
	const uint64_t o[3] = {2, 1, 0}, sz[3] = {4, 3, 1};
	CHECK(l.region(1, 3, o, sz, &at) && at == 49152 + 3 * 2048 + 256 + 8);
	CHECK(!l.region(1, 12, o, sz, &at) && !l.region(2, 0, o, sz, &at));
	CHECK(l.init(desc(7, 8, 8, 4, 2, 1), rgba));
	CHECK(level(l, 0, 0, 256, 2048, 8192) && level(l, 1, 8192, 256, 1024, 2048));
	CHECK(l.init(desc(2, 8, 8, 4, 1, 1), rgba) && l.size() == 2048);
	CHECK(!l.init(desc(2, 8, 8, 1, 5, 1), rgba)); // more levels than 8 halves to
	CHECK(!l.init(desc(2, 0, 8, 1, 1, 1), rgba));
}

// A Managed buffer's contents are a region block: didModifyRange: tells the host the range
// (BUFFER_DID_MODIFY_RANGE) and a Shared buffer's posts nothing.
void test_managed_buffer_range()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegion::Block block = region.allocate(0x1000);
	CHECK(block.ok);
	const MTLResourceOptions managed = MTLResourceStorageModeManaged;
	RgfxBuffer *buf = [RgfxBuffer
	    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
	                        regionOffset:block.offset
	                              length:0x1000
	                             options:managed
	                         deallocator:nil
	                              device:nil
	                          serializer:&fx.ser];
	CHECK(buf != nil && [buf contents] == block.ptr);
	CHECK([buf storageMode] == MTLStorageModeManaged &&
	      [buf cpuCacheMode] == MTLCPUCacheModeDefaultCache &&
	      [buf hazardTrackingMode] == MTLHazardTrackingModeDefault);
	auto create =
	    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(create.options == static_cast<uint32_t>(managed));
	CHECK(create.region_offset == block.offset);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[buf didModifyRange:NSMakeRange(0x40, 0x20)];
	[buf didModifyRange:NSMakeRange(0x40, 0)];
#pragma clang diagnostic pop
	auto range =
	    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
	CHECK(range.obj_id == buf.rgfxObjId && range.location == 0x40 && range.length == 0x20);
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // an empty range posts nothing

	RgfxBuffer *shared = [RgfxBuffer
	    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
	                        regionOffset:block.offset
	                              length:0x1000
	                             options:MTLResourceStorageModeShared
	                         deallocator:nil
	                              device:nil
	                          serializer:&fx.ser];
	expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[shared didModifyRange:NSMakeRange(0, 0x10)];
#pragma clang diagnostic pop
	CHECK(!fx.ring.next(&extra, &extraBytes));
}

// A Managed texture's CPU copy is a block of the region, made over a Shared buffer and named by
// its MANAGED_TEXTURE_CREATE: replaceRegion: writes it at the layout's rows and posts the region
// (TEXTURE_REPLACE_REGION); getBytes: reads it back with no message; a region past the level
// writes nothing and posts nothing. A format no CPU copy holds makes no Managed texture.
void test_managed_texture_cpu_copy()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0xff); // pre-dirtied, as BAR memory
	RgfxRegion region(mem.data(), mem.size());
	@autoreleasepool {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:100
		                                                      height:60
		                                                   mipmapped:YES];
		desc.mipmapLevelCount = 3;
		desc.storageMode = MTLStorageModeManaged;
		RgfxTexture *tex = [RgfxTexture managedTextureWithDescriptor:desc
		                                                      region:&region
		                                                      device:nil
		                                                  serializer:&fx.ser];
		CHECK(tex != nil);
		CHECK([tex storageMode] == MTLStorageModeManaged);
		auto cpu =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(cpu.length == 42240 && cpu.options == MTLResourceStorageModeShared);
		auto create = expectMessage<RgfxManagedTextureCreate>(fx.ring,
		                                                      RGFX_CMD_MANAGED_TEXTURE_CREATE);
		CHECK(create.obj_id == tex.rgfxObjId && create.cpu_buffer_obj_id == cpu.obj_id);
		CHECK(create.resource_id == [tex gpuResourceID]._impl && create.gpu_address != 0);
		CHECK(create.bytes_per_block == 4 && create.block_width == 1 &&
		      create.block_height == 1 && create.flags == RGFX_MANAGED_TEXTURE_LINEAR);
		CHECK(create.desc.mipmap_level_count == 3 &&
		      create.desc.options == static_cast<uint32_t>(desc.resourceOptions));
		uint8_t *copy = mem.data() + cpu.region_offset;
		CHECK(copy[0] == 0 && copy[42239] == 0); // a new Managed texture reads zero

		// Level 1, a 4x3 region at (2, 1): its rows at the layout's 256-byte pitch.
		uint8_t pixels[3][16];
		for (int r = 0; r < 3; r++)
			for (int i = 0; i < 16; i++)
				pixels[r][i] = static_cast<uint8_t>(r * 16 + i + 1);
		[tex replaceRegion:MTLRegionMake2D(2, 1, 4, 3)
		       mipmapLevel:1
		         withBytes:pixels
		       bytesPerRow:16];
		for (int r = 0; r < 3; r++)
			CHECK(std::memcmp(copy + 30720 + (1 + r) * 256 + 8, pixels[r], 16) == 0);
		CHECK(copy[30720 + 256 + 7] == 0 && copy[30720 + 256 + 24] == 0);
		auto replaced = expectMessage<RgfxTextureReplaceRegion>(fx.ring,
		                                                        RGFX_CMD_TEXTURE_REPLACE_REGION);
		CHECK(replaced.obj_id == tex.rgfxObjId && replaced.level == 1 && replaced.slice == 0);
		CHECK(replaced.origin[0] == 2 && replaced.origin[1] == 1 && replaced.origin[2] == 0);
		CHECK(replaced.size[0] == 4 && replaced.size[1] == 3 && replaced.size[2] == 1);

		// getBytes: reads the copy, here as the GPU's synchronize would have left it.
		copy[30720 + 2 * 256 + 16] = 0xab;
		uint8_t back[2][8] = {};
		[tex getBytes:back bytesPerRow:8 fromRegion:MTLRegionMake2D(3, 1, 2, 2) mipmapLevel:1];
		CHECK(back[0][0] == pixels[0][4] && back[1][0] == pixels[1][4] && back[1][4] == 0xab);
		RgfxMsgHeader extra = {};
		std::vector<uint8_t> extraBytes;
		CHECK(!fx.ring.next(&extra, &extraBytes)); // getBytes: posts nothing

		// A region past the level writes nothing and posts nothing.
		const uint8_t before = copy[30720 + 49 * 4];
		[tex replaceRegion:MTLRegionMake2D(49, 0, 2, 1)
		       mipmapLevel:1
		         withBytes:pixels
		       bytesPerRow:16];
		CHECK(copy[30720 + 49 * 4] == before);
		CHECK(!fx.ring.next(&extra, &extraBytes));

		// A depth texture keeps no CPU copy.
		MTLTextureDescriptor *depth =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		depth.storageMode = MTLStorageModeManaged;
		CHECK([RgfxTexture managedTextureWithDescriptor:depth
		                                         region:&region
		                                         device:nil
		                                     serializer:&fx.ser] == nil);
		CHECK(!fx.ring.next(&extra, &extraBytes));
	}
}

// --- host objects go as the app lets them go ----------------------------------------------

// The obj_id of the OBJECT_DESTROY next on `ring`.
uint64_t nextDestroyed(TestRing &ring)
{
	return expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY).obj_id;
}

bool ringEmpty(TestRing &ring)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	return !ring.next(&h, &bytes);
}

// Reads every message waiting on `ring` (the creates a test does not check).
void drainRing(TestRing &ring)
{
	while (!ringEmpty(ring)) {
	}
}

bool retirementIdle()
{
	return RgfxRetirement::process().waiting() == 0 && RgfxRetirement::process().live() == 0;
}

// A Managed texture's CPU copy goes with it: the texture's destroy, then its copy's buffer's,
// then the copy's block goes back.
void test_managed_texture_copy_block_after_destroys()
{
	CHECK(retirementIdle());
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Log {
		BlockLog blocks;
		TestRing *ring;
		std::vector<uint64_t> seenAtRelease; // the objects destroyed on the ring by each release
		static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
		{
			return BlockLog::alloc(&static_cast<Log *>(c)->blocks, size, align, offset);
		}
		static void release(void *c, uint64_t offset)
		{
			Log *log = static_cast<Log *>(c);
			BlockLog::release(&log->blocks, offset);
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			while (log->ring->next(&h, &bytes)) {
				RgfxObjectDestroy d = {};
				if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() >= sizeof(d)) {
					std::memcpy(&d, bytes.data(), sizeof(d));
					log->seenAtRelease.push_back(d.obj_id);
				}
			}
		}
	} log;
	log.ring = &fx.ring;
	region.setBlockSource(&Log::alloc, &Log::release, &log);
	uint64_t texId = 0, bufId = 0;
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeManaged;
		RgfxTexture *t = [RgfxTexture managedTextureWithDescriptor:td
		                                                    region:&region
		                                                    device:nil
		                                                serializer:&fx.ser];
		CHECK(t != nil);
		texId = t.rgfxObjId;
		bufId = expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY)
		            .obj_id;
		drainRing(fx.ring);
		CHECK(log.blocks.allocated.size() == 1 && log.blocks.released.empty());
	}
	CHECK(log.blocks.released == log.blocks.allocated);
	CHECK(log.seenAtRelease == std::vector<uint64_t>({texId, bufId}));
	CHECK(ringEmpty(fx.ring));
}

// A heap's GPU address goes back once its destroy is retired: churned 1 GiB heaps, each made
// while the one before it lives, take spans that never meet a live one and are picked again
// once the picks have gone round the context's 64 GiB, so the churn never runs out. One a
// command buffer in flight holds back keeps its span till the buffer completes.
void test_addresses_reused_after_destroy()
{
	CHECK(retirementIdle());
	Fixture fx;
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = uint64_t{1} << 30;
	d.type = MTLHeapTypePlacement;
	const uint64_t span = RgfxIdSpace::addressSpan(d.size);
	auto heapAddress = [&] {
		return expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE).gpu_address;
	};
	std::set<uint64_t> seen;
	RgfxHeap *held = nil;
	uint64_t heldAt = 0;
	bool reused = false;
	for (int i = 0; i < 200; i++) {
		@autoreleasepool {
			RgfxHeap *heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
			uint64_t at = heapAddress();
			CHECK(heap != nil && at != 0 && at % RgfxIdSpace::kAddressGranule == 0);
			CHECK(at + span <= heldAt || heldAt + span <= at); // apart from the live one's span
			reused |= !seen.insert(at).second;
			held = heap; // the one before goes
			heldAt = at;
			drainRing(fx.ring);
		}
	}
	CHECK(reused && seen.size() < 200 && held != nil);
	@autoreleasepool {
		held = nil;
	}
	drainRing(fx.ring);

	// A destroy held back by a command buffer in flight keeps its span till the buffer ends:
	// with every other span taken, only its own is picked once the buffer is done.
	uint64_t ticket = 0, keptAt = 0;
	@autoreleasepool {
		RgfxHeap *kept = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
		keptAt = heapAddress();
		CHECK(kept != nil);
		ticket = RgfxRetirement::process().begin();
		kept = nil;
	}
	CHECK(ringEmpty(fx.ring)); // its destroy waits
	int taken = 0;
	for (uint64_t at; (at = fx.ids.allocAddress(d.size, 1)) != 0; taken++)
		CHECK(at != keptAt);
	CHECK(taken > 0);
	RgfxRetirement::process().end(ticket);
	CHECK(nextDestroyed(fx.ring) != 0 && ringEmpty(fx.ring));
	CHECK(fx.ids.allocAddress(d.size, 1) == keptAt);
	CHECK(retirementIdle());
}

// Each host object the bundle makes is destroyed as the app lets it go, at once with no
// command buffer in flight: samplers, depth-stencil states, events, fences, shared events,
// buffers, heaps, libraries, and render and compute pipelines.
void test_objects_destroyed_as_let_go()
{
	CHECK(retirementIdle());
	Fixture fx;
	auto made = [&](id object) {
		CHECK(object != nil);
		uint64_t objId = ((RgfxResource *)object).rgfxObjId;
		CHECK(objId != 0);
		drainRing(fx.ring);
		return objId;
	};
	uint64_t objId = 0;
	@autoreleasepool {
		objId = made([RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new]
		                                         device:nil
		                                     serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxDepthStencilState
		    depthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]
		                             device:nil
		                         serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxEvent eventWithDevice:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxFence fenceWithDevice:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxSharedEvent sharedEventWithInitialValue:3 device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxBuffer bufferWithLength:256 options:0 device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x10000;
		objId = made([RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));

	const uint8_t libBytes[] = {1, 2, 3, 4};
	uint32_t status = 0;
	uint64_t libId = 0;
	@autoreleasepool {
		libId = made([RgfxLibrary libraryWithData:libBytes
		                                   length:sizeof(libBytes)
		                                      key:nullptr
		                                   device:nil
		                               serializer:&fx.ser
		                                   status:&status]);
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring));

	RgfxLibrary *lib = nil;
	@autoreleasepool {
		lib = [RgfxLibrary libraryWithData:libBytes
		                            length:sizeof(libBytes)
		                               key:nullptr
		                            device:nil
		                        serializer:&fx.ser
		                            status:&status];
		drainRing(fx.ring);
	}
	@autoreleasepool {
		MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
		desc.vertexFunction = [lib newFunctionWithName:@"vmain"];
		desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		NSError *err = nil;
		objId = made([RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
		                                                                 device:nil
		                                                             serializer:&fx.ser
		                                                                  error:&err]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		NSError *err = nil;
		objId = made([RgfxComputePipelineState
		    computePipelineStateWithFunction:[lib newFunctionWithName:@"kmain"]
		                              device:nil
		                          serializer:&fx.ser
		                               error:&err]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	libId = lib.rgfxObjId;
	@autoreleasepool {
		lib = nil;
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring)); // with its last function
	CHECK(retirementIdle());
}

// A function keeps its library on the host, so a pipeline made from it after the app let the
// library go still names a library the host holds; the pipeline needs the library no more.
void test_function_keeps_its_library()
{
	CHECK(retirementIdle());
	Fixture fx;
	const uint8_t libBytes[] = {5, 6, 7};
	uint32_t status = 0;
	uint64_t libId = 0;
	id<MTLFunction> fn = nil;
	@autoreleasepool {
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
		                                        length:sizeof(libBytes)
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		libId = lib.rgfxObjId;
		fn = [lib newFunctionWithName:@"kmain"];
		drainRing(fx.ring);
	}
	CHECK(ringEmpty(fx.ring)); // the function keeps it
	RgfxComputePipelineState *ps = nil;
	@autoreleasepool {
		NSError *err = nil;
		ps = [RgfxComputePipelineState computePipelineStateWithFunction:fn
		                                                         device:nil
		                                                     serializer:&fx.ser
		                                                          error:&err];
		CHECK(ps != nil);
		auto bytes = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
		RgfxComputePipelineCreate create;
		std::memcpy(&create, bytes.data(), sizeof(create));
		CHECK(create.function_library_obj_id == libId);
	}
	@autoreleasepool {
		fn = nil;
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring)); // with its last function
	uint64_t pipeId = ps.rgfxObjId;
	@autoreleasepool {
		ps = nil;
	}
	CHECK(nextDestroyed(fx.ring) == pipeId && ringEmpty(fx.ring));
}

// A buffer or texture placed in a heap keeps the heap (its `heap`): the heap goes from the
// host only after the last of them.
void test_heap_outlives_its_resources()
{
	CHECK(retirementIdle());
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	RgfxHeap *heap = nil;
	id<MTLBuffer> buf = nil;
	id<MTLTexture> tex = nil;
	@autoreleasepool {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x100000;
		heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
		buf = [heap newBufferWithLength:256 options:0 offset:0];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:8
		                                                      height:8
		                                                   mipmapped:NO];
		tex = [heap newTextureWithDescriptor:td offset:0x10000];
		CHECK(buf != nil && tex != nil);
		drainRing(fx.ring);
	}
	uint64_t heapId = heap.rgfxObjId;
	uint64_t bufId = ((RgfxResource *)buf).rgfxObjId;
	uint64_t texId = ((RgfxResource *)tex).rgfxObjId;
	@autoreleasepool {
		CHECK(buf.heap == heap && tex.heap == heap);
		heap = nil;
	}
	CHECK(ringEmpty(fx.ring)); // its resources keep it
	@autoreleasepool {
		buf = nil;
	}
	CHECK(nextDestroyed(fx.ring) == bufId && ringEmpty(fx.ring));
	@autoreleasepool {
		tex = nil;
	}
	CHECK(nextDestroyed(fx.ring) == texId);
	CHECK(nextDestroyed(fx.ring) == heapId && ringEmpty(fx.ring));
}

// The region memory under a buffer or heap over host memory goes back only once the host has
// dropped the object, and only once every command buffer in flight at its release completed.
void test_region_memory_given_back_after_destroy()
{
	CHECK(retirementIdle());
	Fixture fx;
	TestRing *ring = &fx.ring;
	__block int released = 0;
	__block bool destroyedFirst = false;
	void (^deallocator)(void) = ^{
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		destroyedFirst = ring->next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY;
		released++;
	};
	static uint8_t page[0x1000];
	uint64_t ticket = RgfxRetirement::process().begin(); // a command buffer in flight
	@autoreleasepool {
		RgfxBuffer *b = [RgfxBuffer bufferOverHostMemoryWithContents:page
		                                                regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET
		                                                      length:sizeof(page)
		                                                     options:MTLResourceStorageModeShared
		                                                 deallocator:deallocator
		                                                      device:nil
		                                                  serializer:&fx.ser];
		CHECK(b != nil);
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x10000;
		d.storageMode = MTLStorageModeShared;
		RgfxHeap *h = [RgfxHeap heapOverHostMemoryWithDescriptor:d
		                                            regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x1000
		                                             deallocator:deallocator
		                                                  device:nil
		                                              serializer:&fx.ser];
		CHECK(h != nil);
		drainRing(fx.ring);
	}
	CHECK(released == 0 && ringEmpty(fx.ring)); // the buffer in flight may name them
	RgfxRetirement::process().end(ticket);
	CHECK(released == 2 && destroyedFirst);
	CHECK(ringEmpty(fx.ring));
}

// An IOSurface texture's copy goes as the texture goes: the texture, then the copy's buffer,
// from the host, and only then the copy's block back to the region.
void test_iosurface_copy_block_after_destroys()
{
	CHECK(retirementIdle());
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Log {
		BlockLog blocks;
		TestRing *ring;
		std::vector<uint64_t> seenAtRelease; // the objects destroyed on the ring by each release
		static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
		{
			return BlockLog::alloc(&static_cast<Log *>(c)->blocks, size, align, offset);
		}
		static void release(void *c, uint64_t offset)
		{
			Log *log = static_cast<Log *>(c);
			BlockLog::release(&log->blocks, offset);
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			while (log->ring->next(&h, &bytes)) {
				RgfxObjectDestroy d = {};
				if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() >= sizeof(d)) {
					std::memcpy(&d, bytes.data(), sizeof(d));
					log->seenAtRelease.push_back(d.obj_id);
				}
			}
		}
	} log;
	log.ring = &fx.ring;
	region.setBlockSource(&Log::alloc, &Log::release, &log);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	uint64_t texId = 0, bufId = 0;
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&fx.ser];
		CHECK(t != nil);
		texId = t.rgfxObjId;
		bufId = expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY)
		            .obj_id;
		drainRing(fx.ring);
		CHECK(log.blocks.allocated.size() == 1);
	}
	CHECK(log.blocks.released == log.blocks.allocated);
	CHECK(log.seenAtRelease == std::vector<uint64_t>({texId, bufId}));
	CHECK(ringEmpty(fx.ring));
	CFRelease(surface);
}

// A queue's command buffers keep it alive; once they are gone it ends on the device, its
// completion ring read no more, and only then does what it holds go back.
void test_queue_ends_after_its_buffers()
{
	CHECK(retirementIdle());
	@autoreleasepool {
		// Made in a pool of its own, so no autorelease keeps the queue past the test's let-go.
		QueueFixture *made = nullptr;
		@autoreleasepool {
			made = new QueueFixture();
		}
		QueueFixture &f = *made;
		(void)expectMessage<RgfxQueueCreate>(f.control, RGFX_CMD_QUEUE_CREATE);
		uint64_t queueId = f.queue.rgfxObjId;
		std::shared_ptr<RgfxReplyReader> reader = [f.queue sharedReplyReader];
		int released = 0;
		bool endedFirst = false;
		TestRing *control = &f.control;
		[f.queue rgfxSetRelease:[&released, &endedFirst, control] {
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			endedFirst = control->next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY;
			released++;
		}];
		id<MTLCommandBuffer> cb = nil;
		@autoreleasepool {
			cb = [f.queue commandBuffer];
			[cb commit];
			f.queue = nil;
		}
		@autoreleasepool {
			CHECK(cb.commandQueue != nil); // the buffer keeps it
		}
		CHECK(ringEmpty(f.control) && released == 0);
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(ringEmpty(f.control) && released == 0);
		@autoreleasepool {
			cb = nil;
		}
		CHECK(released == 1 && endedFirst);
		CHECK(queueId != 0);
		// Its completion ring is read no more.
		uint64_t consumer = f.completion.consumerCount();
		RgfxCbComplete late = cbComplete(99, 1);
		CHECK(f.completion.postFromHost(&late, sizeof(late)));
		CHECK(reader->pump() == 0 && !reader->armAndCheck());
		CHECK(f.completion.consumerCount() == consumer);
	}
	CHECK(retirementIdle());
}

// Where a closer is set the queue ends through it (the kext's QUEUE_DESTROY) and nothing is
// posted on the ring; without one, an OBJECT_DESTROY ends it.
void test_serializer_destroys_queues()
{
	struct Closed {
		std::vector<uint64_t> ids;
		static void close(void *c, RgfxObjId objId) { static_cast<Closed *>(c)->ids.push_back(objId); }
	} closed;
	Fixture fx;
	fx.ser.destroyQueue(7);
	CHECK(nextDestroyed(fx.ring) == 7 && ringEmpty(fx.ring));
	fx.ser.setQueueCloser(&Closed::close, &closed);
	fx.ser.destroyQueue(8);
	CHECK(closed.ids == std::vector<uint64_t>({8}) && ringEmpty(fx.ring));
}

// An object let go after a command buffer was made, before its commit, stays on the host
// until the buffer completes, as it may encode it meanwhile; one never committed holds
// nothing back once it goes.
void test_objects_outlive_an_uncommitted_buffer()
{
	CHECK(retirementIdle());
	@autoreleasepool {
		QueueFixture f;
		drainRing(f.control);
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		uint64_t samplerId = 0;
		@autoreleasepool {
			RgfxSampler *s = [RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new]
			                                             device:nil
			                                         serializer:&f.ser];
			samplerId = s.rgfxObjId;
			drainRing(f.control);
		}
		CHECK(ringEmpty(f.control));
		[cb commit];
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(nextDestroyed(f.control) == samplerId && ringEmpty(f.control));

		uint64_t other = 0;
		@autoreleasepool {
			id<MTLCommandBuffer> idle = [f.queue commandBuffer];
			@autoreleasepool {
				RgfxSampler *s = [RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new]
				                                             device:nil
				                                         serializer:&f.ser];
				other = s.rgfxObjId;
				drainRing(f.control);
			}
			CHECK(ringEmpty(f.control));
			(void)idle;
		}
		CHECK(nextDestroyed(f.control) == other && ringEmpty(f.control));
	}
	CHECK(retirementIdle());
}

// While one thread runs the releases due, a release retired meanwhile waits behind them,
// though no ticket is live, and runs on that thread after them; a release that retires
// another runs it after itself.
void test_retirement_keeps_order_while_running()
{
	RgfxRetirement r;
	std::vector<int> ran;
	std::atomic<bool> running{false}, go{false};
	uint64_t t = r.begin();
	r.retire([&] {
		running = true;
		while (!go)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		ran.push_back(1);
	});
	std::thread::id ranOn;
	std::thread a([&] { r.end(t); });
	std::thread::id aId = a.get_id();
	CHECK(eventually([&] { return running.load(); }));
	r.retire([&] {
		ran.push_back(2);
		ranOn = std::this_thread::get_id();
	});
	CHECK(r.waiting() == 1 && r.live() == 0);
	go = true;
	a.join();
	CHECK(ran == std::vector<int>({1, 2}));
	CHECK(ranOn == aId);
	r.retire([&] {
		r.retire([&] { ran.push_back(4); });
		ran.push_back(3);
	});
	CHECK(ran == std::vector<int>({1, 2, 3, 4}));
	CHECK(r.waiting() == 0);
}

// A closed reader takes nothing more off its ring: no handler runs, the consumer stays, and
// the ring is not armed.
void test_reply_reader_close()
{
	TestRing ring(0x1000);
	RgfxReplyReader reader(ring.base(), ring.dataSize);
	int fired = 0;
	reader.expectCompletion(7, [&](const RgfxCbComplete &) { fired++; });
	reader.close();
	RgfxCbComplete done = cbComplete(7, 1);
	CHECK(ring.postFromHost(&done, sizeof(done)));
	CHECK(reader.pump() == 0 && fired == 0);
	CHECK(!reader.armAndCheck());
	CHECK(ringArmed(ring) == 0);
	CHECK(ring.consumerCount() == 0);
}

int main()
{
	@autoreleasepool {
		test_buffer_routing();
		test_second_buffer_is_monotonic();
		test_wrapper_guards_and_base();
		test_event_routing();
		test_fence_routing();
		test_shared_event_routing();
		test_texture_routing();
		test_texture_resource_ids();
		test_sampler_routing();
		test_depth_stencil_routing();
		test_heap_routing_and_suballoc();
		test_resource_guards();
		test_library_and_function_routing();
		test_render_pipeline_routing();
		test_compute_pipeline_routing();
		test_argument_encoder_stub();
		test_command_graph();
		test_parallel_render_encoder();
		test_queue_guards();
		test_region_allocator();
		test_queue_carries_allocated_offsets();
		test_reply_reader_completion();
		test_reply_reader_reflection();
		test_reply_reader_event();
		test_reply_reader_arm();
		test_completion_thread_runs_handlers();
		test_wait_with_the_completion_thread();
		test_completion_thread_stops();
		test_uncompleted_buffer_releases();
		test_command_buffer_out_of_line();
		test_completion_expected_before_publish();
		test_command_buffer_back_pressure();
		test_retirement();
		test_texture_destroy_waits_for_buffers();
		test_managed_layout();
		test_managed_buffer_range();
		test_managed_texture_cpu_copy();
		test_objects_destroyed_as_let_go();
		test_function_keeps_its_library();
		test_heap_outlives_its_resources();
		test_region_memory_given_back_after_destroy();
		test_iosurface_copy_block_after_destroys();
		test_queue_ends_after_its_buffers();
		test_serializer_destroys_queues();
		test_objects_outlive_an_uncommitted_buffer();
		test_retirement_keeps_order_while_running();
		test_reply_reader_close();
		test_addresses_reused_after_destroy();
		test_managed_texture_copy_block_after_destroys();
	}
	std::printf("shell_tests: OK (%d checks)\n", g_checks);
	return 0;
}
