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
#include <vector>
#import "RgfxCommandObjects.h"

#include "RgfxObjectModel.hpp"
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
struct Fixture {
	TestRing ring;
	RgfxRingWriter writer;
	RgfxIdSpace ids;
	RgfxSerializer ser;

	Fixture()
	    : ring(0x8000), writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                           nullptr),
	      ids(), ser(writer, ids)
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
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                        device:nil
	                                    serializer:&fx.ser];
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
	                                         device:nil
	                                     serializer:nullptr];
	CHECK(none == nil);
}

// A render pipeline create maps the MTLRenderPipelineDescriptor into the wire: the
// functions, label, colour attachment, flags and vertex descriptor; the wrapper
// holds objId + seqno and answers reflection getters with documented placeholders.
void test_render_pipeline_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {1, 2, 3, 4};
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                        device:nil
	                                    serializer:&fx.ser];
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
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                        device:nil
	                                    serializer:&fx.ser];
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
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter controlWriter(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                             TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(controlWriter, ids);

	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, /*ring_id*/ 1, TestRing::doorbell,
	                            nullptr);

	// The queue: its create routes a QUEUE_CREATE on the control ring and picks objId 1.
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
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
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter controlWriter(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                             TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
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
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter controlWriter(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                             TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(controlWriter, ids);
	TestRing submit(0x8000);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:nullptr] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                        ids:nullptr
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:nullptr
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
}

} // namespace

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
	}
	std::printf("shell_tests: OK (%d checks)\n", g_checks);
	return 0;
}
