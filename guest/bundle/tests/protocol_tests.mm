/*
 * protocol_tests: the shell check of the Metal protocol members the bundle's classes answer
 * beyond their slices' own (built into shell_tests, which calls rgfxProtocolTests()). Each
 * member with behaviour is driven here: the state getters' answers, the members on the wire
 * (their ops, read back off the rings as the host decoder reads them), and the members the
 * device does not support (nil, an NSError, or nothing done, and logged once:
 * RgfxUnsupported.h).
 *
 * Our own ObjC++; built on the Mac's native arch with the NSObject-based wrappers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#import "RgfxArgumentEncoder.h"
#import "RgfxBuffer.h"
#import "RgfxCommandObjects.h"
#import "RgfxComputePipelineState.h"
#import "RgfxDepthStencilState.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxFunction.h"
#import "RgfxHeap.h"
#import "RgfxIndirectCommandBuffer.h"
#import "RgfxInterop.h"
#import "RgfxBlockCall.h"
#import "RgfxLibrary.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxResource.h"
#import "RgfxSampler.h"
#import "RgfxSharedEvent.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include "RgfxHeapSizes.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxUploads.h"
#include "rgfx_argument_reply.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

void rgfxProtocolTests();

namespace {

// One op of a command buffer's stream: its tag and body.
struct Op {
	uint16_t tag;
	std::vector<uint8_t> body;
	template <typename T> T as() const
	{
		T v;
		CHECK(body.size() >= sizeof(T));
		std::memcpy(&v, body.data(), sizeof(T));
		return v;
	}
	// The `i`th element of `T` after a `Head` (an array-tailed op's elements).
	template <typename Head, typename T> T elem(size_t i) const
	{
		T v;
		CHECK(body.size() >= sizeof(Head) + (i + 1) * sizeof(T));
		std::memcpy(&v, body.data() + sizeof(Head) + i * sizeof(T), sizeof(T));
		return v;
	}
};

// A queue as the device makes one: its create on a control ring, its command buffers on a
// submission ring, the host's completions on a completion ring, and an 8 MiB region for its
// blocks.
struct QueueFixture {
	TestRing control;
	RgfxRingWriter controlWriter;
	RgfxIdSpace ids;
	RgfxSerializer ser;
	TestRing submit;
	RgfxRingWriter submitWriter;
	TestRing completion;
	RgfxReplyReader reader;
	std::vector<uint8_t> mem;
	RgfxRegion region;
	RgfxCommandQueue *queue;

	explicit QueueFixture(id<MTLDevice> device = nil)
	    : control(0x8000),
	      controlWriter(control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                    nullptr),
	      ids(), ser(controlWriter, ids), submit(0x8000),
	      submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr),
	      completion(0x8000), reader(completion.base(), completion.dataSize), mem(0x800000, 0),
	      region(mem.data(), mem.size())
	{
		control.setConsumerIdle(1);
		submit.setConsumerIdle(1);
		// The factory's autoreleased reference drains here, so the fixture's own is the queue's
		// last one, let go before its serialiser.
		@autoreleasepool {
			queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
			                                 submitOffset:0
			                               completeOffset:0
			                             completeDataSize:0
			                                          ids:&ids
			                                     maxCount:0
			                                       device:device
			                                   serializer:&ser];
		}
		CHECK(queue != nil);
		queue.replyReader = &reader;
		queue.region = &region;
		(void)expectQueueCreate(control);
	}

	// The ops of the next command buffer on the submission ring, and its seqno.
	std::vector<Op> nextOps(uint64_t *seqno)
	{
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes));
		CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
		RgfxCommandBuffer cbh = {};
		std::memcpy(&cbh, bytes.data(), sizeof(cbh));
		*seqno = h.seqno;
		std::vector<Op> ops;
		size_t pos = RGFX_COMMAND_BUFFER_OPS_OFFSET;
		while (pos < bytes.size()) {
			RgfxOpHeader oh;
			std::memcpy(&oh, bytes.data() + pos, sizeof(oh));
			CHECK(oh.len >= RGFX_OP_BODY_OFFSET && pos + oh.len <= bytes.size());
			Op op;
			op.tag = oh.tag;
			op.body.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos + RGFX_OP_BODY_OFFSET),
			               bytes.begin() + static_cast<std::ptrdiff_t>(pos + oh.len));
			ops.push_back(op);
			pos += oh.len;
		}
		return ops;
	}

	// The host completes the command buffer committed with `seqno`, which then has `status`.
	void complete(id<MTLCommandBuffer> cb, uint64_t seqno,
	              MTLCommandBufferStatus status = MTLCommandBufferStatusCompleted)
	{
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(cb.status == status);
	}

	// The region bytes at `offset` in the chunk buffer created at `regionOffset`.
	const uint8_t *regionAt(uint64_t regionOffset, uint64_t offset)
	{
		return mem.data() + regionOffset + offset;
	}
};

const Op *findOp(const std::vector<Op> &ops, uint16_t tag, size_t nth = 0)
{
	for (const Op &op : ops) {
		if (op.tag == tag && nth-- == 0)
			return &op;
	}
	return nullptr;
}

MTLRenderPassDescriptor *onePass(id<MTLTexture> target)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = target;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	return rp;
}

// A texture wrapper over no host object, of `desc`'s shape, for the encoders to name.
RgfxTexture *shapeTexture(MTLTextureDescriptor *d, uint64_t objId)
{
	RgfxTextureDescriptor w = {};
	w.pixel_format = static_cast<uint32_t>(d.pixelFormat);
	w.texture_type = static_cast<uint32_t>(d.textureType);
	w.width = static_cast<uint32_t>(d.width);
	w.height = static_cast<uint32_t>(d.height);
	w.depth = static_cast<uint32_t>(d.depth);
	w.mipmap_level_count = static_cast<uint32_t>(d.mipmapLevelCount);
	w.array_length = static_cast<uint32_t>(d.arrayLength);
	w.sample_count = static_cast<uint32_t>(d.sampleCount);
	return [[RgfxTexture alloc] initWithObjId:objId
	                               gpuAddress:0
	                               resourceId:0
	                               descriptor:w
	                                   device:nil
	                               serializer:nullptr];
}

// --- the encoders ----------------------------------------------------------------

// Every encoder's inline bytes land in one upload chunk of the command buffer's, at 256-byte
// boundaries, bound as a buffer at their offset; the chunk is made once (one
// BUFFER_OVER_HOST_MEMORY), given back when the host completes the buffer, and the next buffer
// fills it again from its start.
void test_inline_bytes()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO],
	    0x42);

	const uint32_t a[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
	const float f[3] = {1.0f, 2.0f, 3.0f};
	const uint8_t t[8] = {9, 8, 7, 6, 5, 4, 3, 2};

	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	[renc setVertexBytes:a length:sizeof(a) atIndex:1];
	[renc setFragmentBytes:f length:sizeof(f) atIndex:2];
	[renc setVertexBytes:a length:8 attributeStride:4 atIndex:3];
	[renc setTileBytes:t length:sizeof(t) atIndex:0];
	[renc endEncoding];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	[cenc setBytes:t length:sizeof(t) atIndex:5];
	[cenc setBytes:a length:12 attributeStride:8 atIndex:6];
	[cenc endEncoding];

	// One chunk was made, over a region block.
	auto chunk = expectMessage<RgfxBufferOverHostMemory>(fx.control, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(chunk.length == RgfxUploadArena::kUploadChunkSize);
	CHECK(chunk.region_offset >= RGFX_ALLOCATIONS_OFFSET);
	CHECK(chunk.options == MTLResourceStorageModeShared);
	RgfxMsgHeader none = {};
	std::vector<uint8_t> noneBytes;
	CHECK(!fx.control.next(&none, &noneBytes));

	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);

	const Op *v = findOp(ops, RGFX_OP_RENDER_SET_VERTEX_BUFFER);
	CHECK(v != nullptr);
	auto vb = v->as<RgfxOpSetVertexBuffer>();
	CHECK(vb.buffer_obj_id == chunk.obj_id && vb.offset == 0 && vb.index == 1);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 0), a, sizeof(a)) == 0);

	const Op *fr = findOp(ops, RGFX_OP_RENDER_SET_FRAGMENT_BUFFER);
	CHECK(fr != nullptr);
	auto fb = fr->as<RgfxOpSetFragmentBuffer>();
	CHECK(fb.buffer_obj_id == chunk.obj_id && fb.offset == 256 && fb.index == 2);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 256), f, sizeof(f)) == 0);

	const Op *vs = findOp(ops, RGFX_OP_RENDER_SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES);
	CHECK(vs != nullptr);
	auto vsh = vs->as<RgfxOpSetVertexBuffersAttributeStrides>();
	CHECK(vsh.range_location == 3 && vsh.range_length == 1);
	auto vse = vs->elem<RgfxOpSetVertexBuffersAttributeStrides, RgfxBufferStrideBinding>(0);
	CHECK(vse.buffer_obj_id == chunk.obj_id && vse.offset == 512 && vse.stride == 4);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 512), a, 8) == 0);

	const Op *tl = findOp(ops, RGFX_OP_RENDER_SET_TILE_BUFFERS);
	CHECK(tl != nullptr);
	auto tle = tl->elem<RgfxOpSetTileBuffers, RgfxBufferBinding>(0);
	CHECK(tle.buffer_obj_id == chunk.obj_id && tle.offset == 768);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 768), t, sizeof(t)) == 0);

	const Op *cs = findOp(ops, RGFX_OP_COMPUTE_SET_BUFFERS);
	CHECK(cs != nullptr);
	auto csh = cs->as<RgfxComputeOpSetBuffers>();
	CHECK(csh.location == 5 && csh.length == 1);
	auto cse = cs->elem<RgfxComputeOpSetBuffers, RgfxComputeBufferBinding>(0);
	CHECK(cse.buffer_obj_id == chunk.obj_id && cse.offset == 1024);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 1024), t, sizeof(t)) == 0);

	const Op *css = findOp(ops, RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES);
	CHECK(css != nullptr);
	auto csse =
	    css->elem<RgfxComputeOpSetBuffersAttributeStrides, RgfxComputeBufferStrideBinding>(0);
	CHECK(csse.buffer_obj_id == chunk.obj_id && csse.offset == 1280 && csse.stride == 8);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 1280), a, 12) == 0);

	// While the host has not completed the buffer, the next one takes a chunk of its own.
	id<MTLCommandBuffer> cb2 = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc2 = [cb2 renderCommandEncoderWithDescriptor:onePass(target)];
	[renc2 setVertexBytes:f length:sizeof(f) atIndex:0];
	[renc2 endEncoding];
	auto chunk2 =
	    expectMessage<RgfxBufferOverHostMemory>(fx.control, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(chunk2.obj_id != chunk.obj_id && chunk2.region_offset != chunk.region_offset);
	[cb2 commit];
	uint64_t seqno2 = 0;
	(void)fx.nextOps(&seqno2);

	// Once the host completes the first, its chunk is reused from its start: no new create.
	fx.complete(cb, seqno);
	id<MTLCommandBuffer> cb3 = [fx.queue commandBuffer];
	id<MTLComputeCommandEncoder> cenc3 = [cb3 computeCommandEncoder];
	[cenc3 setBytes:a length:sizeof(a) atIndex:0];
	[cenc3 endEncoding];
	CHECK(!fx.control.next(&none, &noneBytes));
	[cb3 commit];
	uint64_t seqno3 = 0;
	std::vector<Op> ops3 = fx.nextOps(&seqno3);
	auto cse3 = findOp(ops3, RGFX_OP_COMPUTE_SET_BUFFERS)
	                ->elem<RgfxComputeOpSetBuffers, RgfxComputeBufferBinding>(0);
	CHECK(cse3.buffer_obj_id == chunk.obj_id && cse3.offset == 0);
	CHECK(std::memcmp(fx.regionAt(chunk.region_offset, 0), a, sizeof(a)) == 0);

	// Bytes larger than a chunk take a chunk of their own size, rounded up to a page.
	std::vector<uint8_t> big(RgfxUploadArena::kUploadChunkSize + 100, 0x5a);
	id<MTLCommandBuffer> cb4 = [fx.queue commandBuffer];
	id<MTLComputeCommandEncoder> cenc4 = [cb4 computeCommandEncoder];
	[cenc4 setBytes:big.data() length:big.size() atIndex:0];
	[cenc4 endEncoding];
	auto chunk4 =
	    expectMessage<RgfxBufferOverHostMemory>(fx.control, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(chunk4.length == RgfxUploadArena::kUploadChunkSize + 0x1000);
	CHECK(std::memcmp(fx.regionAt(chunk4.region_offset, 0), big.data(), big.size()) == 0);
	fx.complete(cb2, seqno2);
	fx.complete(cb3, seqno3);
	} // @autoreleasepool
}

// Without a region the bytes have nowhere to go: no binding is made (and it is logged), and the
// command buffer fails with MTLCommandBufferErrorOutOfMemory.
void test_inline_bytes_without_region()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	fx.queue.region = nullptr;
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	const uint32_t a = 7;
	[renc setVertexBytes:&a length:sizeof(a) atIndex:0];
	[renc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	CHECK(findOp(ops, RGFX_OP_RENDER_SET_VERTEX_BUFFER) == nullptr);
	// It ran without the binding, so it fails as Metal's does when memory runs out.
	fx.complete(cb, seqno, MTLCommandBufferStatusError);
	CHECK(cb.error != nil && [cb.error.domain isEqualToString:MTLCommandBufferErrorDomain] &&
	      cb.error.code == MTLCommandBufferErrorOutOfMemory);
	CHECK([cb.error.localizedDescription
	    isEqualToString:@"Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"]);
	} // @autoreleasepool
}

// The render barriers: the scope barrier as given; a resource barrier as the scope of the
// resources' kinds; the texture barrier as the render targets' fragment barrier.
void test_render_barriers()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:0x50
	                                            gpuAddress:0
	                                                length:16
	                                               options:0
	                                                device:nil
	                                            serializer:nullptr];
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	[renc memoryBarrierWithScope:MTLBarrierScopeBuffers
	                 afterStages:MTLRenderStageVertex
	                beforeStages:MTLRenderStageFragment];
	id<MTLResource> onlyBuffer[1] = {buffer};
	[renc memoryBarrierWithResources:onlyBuffer
	                           count:1
	                     afterStages:MTLRenderStageFragment
	                    beforeStages:MTLRenderStageVertex];
	id<MTLResource> both[2] = {buffer, target};
	[renc memoryBarrierWithResources:both
	                           count:2
	                     afterStages:MTLRenderStageVertex
	                    beforeStages:MTLRenderStageVertex];
	[renc memoryBarrierWithResources:both count:0 afterStages:0 beforeStages:0]; // nothing
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[renc textureBarrier];
#pragma clang diagnostic pop
	[renc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	auto b0 = findOp(ops, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, 0)->as<RgfxOpMemoryBarrierWithScope>();
	CHECK(b0.scope == MTLBarrierScopeBuffers && b0.after == MTLRenderStageVertex &&
	      b0.before == MTLRenderStageFragment);
	auto b1 = findOp(ops, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, 1)->as<RgfxOpMemoryBarrierWithScope>();
	CHECK(b1.scope == MTLBarrierScopeBuffers && b1.after == MTLRenderStageFragment &&
	      b1.before == MTLRenderStageVertex);
	auto b2 = findOp(ops, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, 2)->as<RgfxOpMemoryBarrierWithScope>();
	CHECK(b2.scope == (MTLBarrierScopeBuffers | MTLBarrierScopeTextures |
	                   MTLBarrierScopeRenderTargets));
	auto b3 = findOp(ops, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, 3)->as<RgfxOpMemoryBarrierWithScope>();
	CHECK(b3.scope == MTLBarrierScopeRenderTargets && b3.after == MTLRenderStageFragment &&
	      b3.before == MTLRenderStageFragment);
	CHECK(findOp(ops, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, 4) == nullptr);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// A compute pass opened with a dispatch type carries it, and the encoder answers it; one
// opened without answers Serial.
void test_compute_dispatch_type()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLComputeCommandEncoder> serial = [cb computeCommandEncoder];
	CHECK(serial.dispatchType == MTLDispatchTypeSerial);
	[serial endEncoding];
	id<MTLComputeCommandEncoder> conc =
	    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	CHECK(conc.dispatchType == MTLDispatchTypeConcurrent);
	[conc endEncoding];
	MTLComputePassDescriptor *cpd = [MTLComputePassDescriptor computePassDescriptor];
	cpd.dispatchType = MTLDispatchTypeConcurrent;
	id<MTLComputeCommandEncoder> viaDesc = [cb computeCommandEncoderWithDescriptor:cpd];
	CHECK(viaDesc.dispatchType == MTLDispatchTypeConcurrent);
	[viaDesc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	auto p1 = findOp(ops, RGFX_OP_ENCODER_BEGIN_COMPUTE, 1)->as<RgfxComputePassDesc>();
	CHECK(p1.dispatch_type == MTLDispatchTypeConcurrent);
	auto p2 = findOp(ops, RGFX_OP_ENCODER_BEGIN_COMPUTE, 2)->as<RgfxComputePassDesc>();
	CHECK(p2.dispatch_type == MTLDispatchTypeConcurrent);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// copyFromTexture:toTexture: copies as many whole surfaces as both have: lined up at the
// level of one that is the other's first level's size, the levels both have from there, and
// the slices both have (six faces a cube); textures no level of which lines up copy nothing.
void test_blit_copy_texture_to_texture()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	MTLTextureDescriptor *big =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:YES]; // 7 levels
	MTLTextureDescriptor *small =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:YES]; // 5 levels
	small.textureType = MTLTextureType2DArray;
	small.arrayLength = 3;
	MTLTextureDescriptor *cube = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                                   size:16
	                                                                              mipmapped:NO];
	MTLTextureDescriptor *odd =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:24
	                                                      height:16
	                                                   mipmapped:NO];
	RgfxTexture *tBig = shapeTexture(big, 0x60);
	RgfxTexture *tSmall = shapeTexture(small, 0x61);
	RgfxTexture *tCube = shapeTexture(cube, 0x62);
	RgfxTexture *tOdd = shapeTexture(odd, 0x63);

	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	[benc copyFromTexture:tBig toTexture:tSmall]; // big's level 2 is small's level 0
	[benc copyFromTexture:tSmall toTexture:tBig]; // small's level 0 is big's level 2
	[benc copyFromTexture:tCube toTexture:tCube]; // 6 faces, 1 level
	[benc copyFromTexture:tOdd toTexture:tBig];   // nothing lines up: no op
	[benc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	auto c0 = findOp(ops, RGFX_OP_BLIT_COPY_TEXTURE_SLICES, 0)->as<RgfxBlitOpCopyTextureSlices>();
	CHECK(c0.source_texture_obj_id == 0x60 && c0.destination_texture_obj_id == 0x61);
	CHECK(c0.source_level == 2 && c0.destination_level == 0);
	CHECK(c0.level_count == 5 && c0.slice_count == 1);
	auto c1 = findOp(ops, RGFX_OP_BLIT_COPY_TEXTURE_SLICES, 1)->as<RgfxBlitOpCopyTextureSlices>();
	CHECK(c1.source_level == 0 && c1.destination_level == 2);
	CHECK(c1.level_count == 5 && c1.slice_count == 1);
	auto c2 = findOp(ops, RGFX_OP_BLIT_COPY_TEXTURE_SLICES, 2)->as<RgfxBlitOpCopyTextureSlices>();
	CHECK(c2.slice_count == 6 && c2.level_count == 1);
	CHECK(findOp(ops, RGFX_OP_BLIT_COPY_TEXTURE_SLICES, 3) == nullptr);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// The tile size an encoder answers: its pass descriptor's; the device's own choice, which is
// not on the wire, answers 0 and is logged.
void test_render_tile_size()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO],
	    0x42);
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	MTLRenderPassDescriptor *rp = onePass(target);
	rp.tileWidth = 16;
	rp.tileHeight = 32;
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
	CHECK(renc.tileWidth == 16 && renc.tileHeight == 32);
	[renc endEncoding];
	id<MTLParallelRenderCommandEncoder> par = [cb parallelRenderCommandEncoderWithDescriptor:rp];
	id<MTLRenderCommandEncoder> sub = [par renderCommandEncoder];
	CHECK(sub.tileWidth == 16 && sub.tileHeight == 32);
	[par endEncoding];
	id<MTLRenderCommandEncoder> chosen = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	CHECK(chosen.tileWidth == 0);
	CHECK(RgfxUnsupportedWasLogged([RgfxRenderCommandEncoder class], @selector(tileWidth)));
	[chosen endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	(void)fx.nextOps(&seqno);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// --- trace ids, encoders' command buffers, unrecognised selectors ----------------------

// The trace ids Metal answers (RgfxInterop.h): on queues, command buffers, every encoder (a
// parallel pass's sub-encoders too), heaps and shared events; each taken as the object is made,
// so in creation order, and the same each time it is asked. Each encoder answers the command
// buffer that made it.
void test_trace_ids()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO],
	    0x42);
	auto trace = [](id object) {
		return ((uint64_t(*)(id, SEL))objc_msgSend)(object, @selector(globalTraceObjectID));
	};
	std::vector<uint64_t> made;
	made.push_back(trace(fx.queue));
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	made.push_back(trace(cb));
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	made.push_back(trace(renc));
	CHECK([(id)renc commandBuffer] == cb);
	[renc endEncoding];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	made.push_back(trace(cenc));
	CHECK([(id)cenc commandBuffer] == cb);
	[cenc endEncoding];
	id<MTLComputeCommandEncoder> conc =
	    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	made.push_back(trace(conc));
	CHECK([(id)conc commandBuffer] == cb);
	[conc endEncoding];
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	made.push_back(trace(benc));
	CHECK([(id)benc commandBuffer] == cb);
	[benc endEncoding];
	id<MTLParallelRenderCommandEncoder> par =
	    [cb parallelRenderCommandEncoderWithDescriptor:onePass(target)];
	made.push_back(trace(par));
	CHECK([(id)par commandBuffer] == cb);
	id<MTLRenderCommandEncoder> sub = [par renderCommandEncoder];
	made.push_back(trace(sub));
	CHECK([(id)sub commandBuffer] == cb);
	[sub endEncoding];
	[par endEncoding];
	MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
	hd.size = 0x10000;
	RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
	CHECK(heap != nil);
	made.push_back(trace(heap));
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:0
	                                                             device:nil
	                                                         serializer:&fx.ser];
	CHECK(ev != nil);
	made.push_back(trace(ev));
	// Unique, in creation order, and stable.
	for (size_t i = 0; i < made.size(); i++)
		CHECK(made[i] != 0 && (i == 0 || made[i] > made[i - 1]));
	CHECK(trace(fx.queue) == made[0] && trace(cb) == made[1] && trace(sub) == made[7]);
	CHECK(trace(ev) == made.back());
	// The kinds Metal gives no trace id answer none.
	for (Class kind : {[RgfxBuffer class], [RgfxTexture class], [RgfxSampler class],
	                   [RgfxFence class], [RgfxEvent class], [RgfxLibrary class],
	                   [RgfxFunction class], [RgfxRenderPipelineState class],
	                   [RgfxComputePipelineState class], [RgfxDepthStencilState class],
	                   [RgfxArgumentEncoder class]})
		CHECK(![kind instancesRespondToSelector:@selector(globalTraceObjectID)]);
	[cb commit];
	uint64_t seqno = 0;
	(void)fx.nextOps(&seqno);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// A selector a class does not answer is logged with the class, once, and then raises as
// NSObject does.
void test_unrecognized_selectors_logged()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	SEL missing = NSSelectorFromString(@"rgfxNoSuchMember");
	auto send = [missing](id object) {
		BOOL raised = NO;
		@try {
			((void (*)(id, SEL))objc_msgSend)(object, missing);
		} @catch (NSException *e) {
			raised = [e.name isEqualToString:NSInvalidArgumentException];
		}
		return raised;
	};
	CHECK(!RgfxUnrecognizedWasLogged([RgfxComputeCommandEncoder class], missing));
	CHECK(send(cenc));
	CHECK(RgfxUnrecognizedWasLogged([RgfxComputeCommandEncoder class], missing));
	CHECK(send(cenc)); // still raises, logged once
	// A subclass of the resource root logs its own class.
	CHECK(send(cb));
	CHECK(RgfxUnrecognizedWasLogged([RgfxCmdBuffer class], missing));
	CHECK(!RgfxUnrecognizedWasLogged([RgfxResource class], missing));
	[cenc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	(void)fx.nextOps(&seqno);
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// The members Apple's frameworks send queues, command buffers and encoders beyond their
// protocols (selector-sweep), each as Metal's answers it.
void test_framework_members()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO],
	    0x42);
	RgfxTexture *image = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO],
	    0x43);
	id queue = fx.queue;
	// The queue: no perf samples, no counters, statistics taken.
	__block int perfCalls = 0;
	((void (*)(id, SEL, void (^)(id)))objc_msgSend)(queue, @selector(addPerfSampleHandler:),
	                                                ^(id sample) {
		                                                (void)sample;
		                                                perfCalls++;
	                                                });
	auto counters = [queue](NSArray *list) {
		return ((int (*)(id, SEL, NSArray *, NSUInteger))objc_msgSend)(
		    queue, @selector(requestCounters:withIndex:), list, 0);
	};
	CHECK(counters(nil) == -1 && counters(@[]) == -1 && counters(@[ @0 ]) == -1);
	((void (*)(id, SEL, BOOL))objc_msgSend)(queue, @selector(setStatEnabled:), NO);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(queue, @selector(setStatLocations:), 1);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(queue, @selector(setStatOptions:), 1);
	CHECK(!RgfxUnsupportedWasLogged([RgfxCommandQueue class], @selector(setStatEnabled:)));
	((void (*)(id, SEL, BOOL))objc_msgSend)(queue, @selector(setStatEnabled:), YES);
	CHECK(RgfxUnsupportedWasLogged([RgfxCommandQueue class], @selector(setStatEnabled:)));

	// A fragment texture and its sampler set together are the two sets.
	std::vector<Op> together, apart;
	for (int k = 0; k < 2; k++) {
		id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
		id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
		if (k == 0) {
			((void (*)(id, SEL, id, NSUInteger, id, NSUInteger))objc_msgSend)(
			    renc, @selector(setFragmentTexture:atTextureIndex:samplerState:atSamplerIndex:),
			    image, 3, nil, 5);
		} else {
			[renc setFragmentTexture:image atIndex:3];
			[renc setFragmentSamplerState:nil atIndex:5];
		}
		// Every line width is posted, as its bits.
		((void (*)(id, SEL, float))objc_msgSend)(renc, @selector(setLineWidth:), 1.0f);
		((void (*)(id, SEL, float))objc_msgSend)(renc, @selector(setLineWidth:), 3.5f);
		CHECK(!RgfxUnsupportedWasLogged([RgfxRenderCommandEncoder class], @selector(setLineWidth:)));
		((void (*)(id, SEL, id))objc_msgSend)(renc, @selector(useResidencySet:), nil);
		CHECK(!RgfxUnsupportedWasLogged([RgfxRenderCommandEncoder class],
		                                @selector(useResidencySet:)));
		[renc endEncoding];
		[cb commit];
		uint64_t seqno = 0;
		(k == 0 ? together : apart) = fx.nextOps(&seqno);
		fx.complete(cb, seqno);
	}
	for (uint16_t tag : {RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE, RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES}) {
		const Op *a = findOp(together, tag);
		const Op *b = findOp(apart, tag);
		CHECK(a != nullptr && b != nullptr && a->body == b->body);
	}
	CHECK(findOp(together, RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE)->as<RgfxOpSetFragmentTexture>().index == 3);
	const float widths[] = {1.0f, 3.5f};
	for (size_t n = 0; n < 2; n++) {
		const Op *w = findOp(together, RGFX_OP_RENDER_SET_LINE_WIDTH, n);
		CHECK(w != nullptr && w->body.size() == sizeof(RgfxOpSetLineWidth));
		RgfxOpSetLineWidth b = w->as<RgfxOpSetLineWidth>();
		uint32_t bits;
		std::memcpy(&bits, &widths[n], sizeof(bits));
		CHECK(b.line_width_bits == bits && b.reserved_04 == 0);
	}
	CHECK(findOp(together, RGFX_OP_RENDER_SET_LINE_WIDTH, 2) == nullptr);

	// The command buffer: its dictionary, its agent-masked signal, statistics, validate.
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	auto userDictionary = [cb] {
		return ((NSMutableDictionary * (*)(id, SEL)) objc_msgSend)(cb, @selector(userDictionary));
	};
	NSMutableDictionary *dict = userDictionary();
	CHECK([dict isKindOfClass:[NSMutableDictionary class]] && userDictionary() == dict);
	dict[@"key"] = @1;
	CHECK([userDictionary()[@"key"] isEqual:@1]);
	((void (*)(id, SEL))objc_msgSend)(cb, @selector(validate));
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(cb, @selector(setStatLocations:), 1);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(cb, @selector(setStatOptions:), 1);
	CHECK(cb.status == MTLCommandBufferStatusNotEnqueued);
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	((void (*)(id, SEL, MTLSize, MTLRegion))objc_msgSend)(
	    renc, @selector(dispatchThreadsPerTile:inRegion:), MTLSizeMake(8, 8, 1),
	    MTLRegionMake2D(0, 0, 16, 16));
	CHECK(RgfxUnsupportedWasLogged([RgfxRenderCommandEncoder class],
	                               @selector(dispatchThreadsPerTile:inRegion:)));
	[renc endEncoding];
	// The compute encoder: no loop is ever open.
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	CHECK(((BOOL (*)(id, SEL))objc_msgSend)(cenc, @selector(encodeEndWhile)) == NO);
	((void (*)(id, SEL, id, NSUInteger, MTLCompareFunction, uint32_t))objc_msgSend)(
	    cenc, @selector(encodeStartWhile:offset:comparison:referenceValue:), nil, 0,
	    MTLCompareFunctionNever, 0);
	CHECK(RgfxUnsupportedWasLogged([RgfxComputeCommandEncoder class],
	                               @selector(encodeStartWhile:offset:comparison:referenceValue:)));
	CHECK(((BOOL (*)(id, SEL))objc_msgSend)(cenc, @selector(encodeEndWhile)) == NO);
	((void (*)(id, SEL, id))objc_msgSend)(cenc, @selector(useResidencySet:), nil);
	CHECK(!RgfxUnsupportedWasLogged([RgfxComputeCommandEncoder class], @selector(useResidencySet:)));
	[cenc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	(void)fx.nextOps(&seqno);
	fx.complete(cb, seqno);
	CHECK(perfCalls == 0);
	} // @autoreleasepool
}

// --- programmable sample positions ---------------------------------------------------

} // namespace

// A device that answers only which texture sample counts it makes: 1, 2 and 4, as the M3 Max's.
@interface RgfxSampleCountDevice : NSObject
- (BOOL)supportsTextureSampleCount:(NSUInteger)sampleCount;
@end
@implementation RgfxSampleCountDevice
- (BOOL)supportsTextureSampleCount:(NSUInteger)sampleCount
{
	return sampleCount == 1 || sampleCount == 2 || sampleCount == 4;
}
@end

namespace {

// The sample positions record after the pass's colour records in a BEGIN_RENDER body; its count
// is 0 where the pass flag is clear.
RgfxRenderPassSamplePositions passPositions(const Op &begin)
{
	RgfxRenderPassSamplePositions p = {};
	const RgfxRenderPassDesc d = begin.as<RgfxRenderPassDesc>();
	const size_t at = RGFX_RENDER_PASS_COLOR_OFFSET +
	                  static_cast<size_t>(d.color_count) * sizeof(RgfxRenderPassColorAttachment);
	if ((d.flags & RGFX_PASS_FLAG_SAMPLE_POSITIONS) == 0) {
		CHECK(begin.body.size() == at);
		return p;
	}
	CHECK(begin.body.size() == at + sizeof(p));
	if (begin.body.size() >= at + sizeof(p))
		std::memcpy(&p, begin.body.data() + at, sizeof(p));
	return p;
}

bool positionIs(const RgfxRenderPassSamplePositions &p, uint32_t i, float x, float y)
{
	float gx, gy;
	std::memcpy(&gx, &p.position_bits[i][0], sizeof(gx));
	std::memcpy(&gy, &p.position_bits[i][1], sizeof(gy));
	return gx == x && gy == y;
}

// A pass's sample positions travel after its colour records, in render and parallel passes; a
// pass without them carries none, and a count the device makes no texture of answers no encoder
// and sends nothing, as Metal's validation stops it. setDepthStoreActionOptions: is its op.
void test_sample_positions()
{
	RgfxSampleCountDevice *device = [RgfxSampleCountDevice new];
	QueueFixture fx((id<MTLDevice>)device);
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO];
	td.textureType = MTLTextureType2DMultisample;
	td.sampleCount = 4;
	RgfxTexture *target = shapeTexture(td, 0x42);
	const MTLSamplePosition four[4] = {
	    MTLSamplePositionMake(0.125f, 0.375f), MTLSamplePositionMake(0.625f, 0.125f),
	    MTLSamplePositionMake(0.375f, 0.875f), MTLSamplePositionMake(0.875f, 0.625f)};
	const MTLSamplePosition two[2] = {MTLSamplePositionMake(0.25f, 0.25f),
	                                  MTLSamplePositionMake(0.75f, 0.75f)};
	MTLSamplePosition eight[8];
	for (int i = 0; i < 8; i++)
		eight[i] = MTLSamplePositionMake((i + 0.5f) / 8.0f, (7 - i + 0.5f) / 8.0f);

	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	MTLRenderPassDescriptor *rp = onePass(target);
	[rp setSamplePositions:four count:4];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
	CHECK(renc != nil);
	[renc setDepthStoreActionOptions:MTLStoreActionOptionCustomSamplePositions];
	[renc endEncoding];
	[rp setSamplePositions:two count:2];
	id<MTLParallelRenderCommandEncoder> par = [cb parallelRenderCommandEncoderWithDescriptor:rp];
	CHECK(par != nil);
	[par setDepthStoreActionOptions:MTLStoreActionOptionNone];
	[par endEncoding];
	id<MTLRenderCommandEncoder> plain = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	[plain endEncoding];
	[rp setSamplePositions:eight count:8];
	CHECK([cb renderCommandEncoderWithDescriptor:rp] == nil);
	CHECK([cb parallelRenderCommandEncoderWithDescriptor:rp] == nil);
	[cb commit];

	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	const Op *b0 = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 0);
	const Op *b1 = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 1);
	const Op *b2 = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 2);
	CHECK(b0 != nullptr && b1 != nullptr && b2 != nullptr);
	CHECK(findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 3) == nullptr); // the 8-sample passes
	if (b0 != nullptr && b1 != nullptr && b2 != nullptr) {
		const RgfxRenderPassSamplePositions p0 = passPositions(*b0);
		CHECK(p0.count == 4);
		CHECK(positionIs(p0, 0, 0.125f, 0.375f) && positionIs(p0, 1, 0.625f, 0.125f));
		CHECK(positionIs(p0, 2, 0.375f, 0.875f) && positionIs(p0, 3, 0.875f, 0.625f));
		CHECK(p0.position_bits[4][0] == 0 && p0.position_bits[15][1] == 0);
		const RgfxRenderPassSamplePositions p1 = passPositions(*b1);
		CHECK(p1.count == 2);
		CHECK(positionIs(p1, 0, 0.25f, 0.25f) && positionIs(p1, 1, 0.75f, 0.75f));
		CHECK(passPositions(*b2).count == 0);
	}
	const Op *o0 = findOp(ops, RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS, 0);
	const Op *o1 = findOp(ops, RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS, 1);
	CHECK(o0 != nullptr && o1 != nullptr);
	if (o0 != nullptr && o1 != nullptr) {
		CHECK(o0->body.size() == sizeof(RgfxOpSetDepthStoreActionOptions));
		CHECK(o0->as<RgfxOpSetDepthStoreActionOptions>().store_action_options ==
		      MTLStoreActionOptionCustomSamplePositions);
		CHECK(o1->as<RgfxOpSetDepthStoreActionOptions>().store_action_options ==
		      MTLStoreActionOptionNone);
	}
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// A pass's visibility result buffer and its type travel after its colour records, in render
// and parallel passes, and the modes after it; a pass whose buffer is not ours, or that has none,
// carries none.
void test_visibility_result_buffer()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	RgfxBuffer *results = [[RgfxBuffer alloc] initWithObjId:0x50
	                                             gpuAddress:0
	                                                 length:16
	                                                options:0
	                                                 device:nil
	                                             serializer:nullptr];
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	MTLRenderPassDescriptor *rp = onePass(target);
	rp.visibilityResultBuffer = results;
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
	[renc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:0];
	[renc setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:8];
	[renc endEncoding];
	const bool typed =
	    [MTLRenderPassDescriptor instancesRespondToSelector:@selector(setVisibilityResultType:)];
	if (typed)
		rp.visibilityResultType = MTLVisibilityResultTypeAccumulate;
	id<MTLParallelRenderCommandEncoder> par = [cb parallelRenderCommandEncoderWithDescriptor:rp];
	[par endEncoding];
	[[cb renderCommandEncoderWithDescriptor:onePass(target)] endEncoding];
	[cb commit];

	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	const size_t at = RGFX_RENDER_PASS_COLOR_OFFSET + sizeof(RgfxRenderPassColorAttachment);
	const uint32_t types[2] = {0, typed ? 1u : 0u};
	for (uint32_t i = 0; i < 2; i++) {
		const Op *b = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, i);
		CHECK(b != nullptr);
		if (b == nullptr)
			continue;
		CHECK(b->as<RgfxRenderPassDesc>().flags == RGFX_PASS_FLAG_VISIBILITY_RESULT);
		CHECK(b->body.size() == at + sizeof(RgfxRenderPassVisibility));
		if (b->body.size() < at + sizeof(RgfxRenderPassVisibility))
			continue;
		RgfxRenderPassVisibility v = {};
		std::memcpy(&v, b->body.data() + at, sizeof(v));
		CHECK(v.buffer_obj_id == 0x50 && v.result_type == types[i] && v.reserved_0c == 0);
	}
	const Op *plain = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 2);
	CHECK(plain != nullptr && plain->body.size() == at);
	const Op *m0 = findOp(ops, RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE, 0);
	const Op *m1 = findOp(ops, RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE, 1);
	CHECK(m0 != nullptr && m1 != nullptr);
	if (m0 != nullptr && m1 != nullptr) {
		CHECK(m0->as<RgfxOpSetVisibilityResultMode>().mode == MTLVisibilityResultModeCounting);
		CHECK(m0->as<RgfxOpSetVisibilityResultMode>().offset == 0);
		CHECK(m1->as<RgfxOpSetVisibilityResultMode>().mode == MTLVisibilityResultModeBoolean);
		CHECK(m1->as<RgfxOpSetVisibilityResultMode>().offset == 8);
	}
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// A pass that supports colour attachment mapping (macOS 26) travels as the same pass without it,
// in render and parallel passes: each output lands in the attachment of its index, as on the M3
// Max (fillRenderPass), and nothing is logged.
void test_color_attachment_mapping_pass()
{
	if (![MTLRenderPassDescriptor instancesRespondToSelector:@selector(
	                                  setSupportColorAttachmentMapping:)])
		return;
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	MTLRenderPassDescriptor *mapped = onePass(target);
	mapped.supportColorAttachmentMapping = YES;
	[[cb renderCommandEncoderWithDescriptor:mapped] endEncoding];
	[[cb parallelRenderCommandEncoderWithDescriptor:mapped] endEncoding];
	[[cb renderCommandEncoderWithDescriptor:onePass(target)] endEncoding];
	[cb commit];

	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	const Op *plain = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, 2);
	CHECK(plain != nullptr);
	for (uint32_t i = 0; i < 2 && plain != nullptr; i++) {
		const Op *b = findOp(ops, RGFX_OP_ENCODER_BEGIN_RENDER, i);
		CHECK(b != nullptr && b->body == plain->body);
	}
	CHECK(!RgfxUnsupportedWasLogged([RgfxCmdBuffer class],
	                                @selector(renderCommandEncoderWithDescriptor:)));
	CHECK(!RgfxUnsupportedWasLogged([RgfxCmdBuffer class],
	                                @selector(parallelRenderCommandEncoderWithDescriptor:)));
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// The members of features the device answers NO for do nothing on the wire: a nil or default
// argument is exact and not logged; any other use is logged once.
void test_encoders_unsupported()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	Class r = [RgfxRenderCommandEncoder class];
	[renc setVertexAccelerationStructure:nil atBufferIndex:0];
	[renc setFragmentVisibleFunctionTable:nil atBufferIndex:0];
	[renc setObjectBuffer:nil offset:0 atIndex:0];
	[renc setDepthTestMinBound:0.0f maxBound:1.0f];
	[renc setColorAttachmentMap:nil];
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setVertexAccelerationStructure:atBufferIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setFragmentVisibleFunctionTable:atBufferIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setObjectBuffer:offset:atIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setDepthTestMinBound:maxBound:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setColorAttachmentMap:)));
	[renc setDepthTestMinBound:0.25f maxBound:0.5f];
	[renc drawMeshThreads:MTLSizeMake(1, 1, 1)
	    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
	      threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
	[renc setMeshBytes:&r length:sizeof(r) atIndex:0];
	CHECK(RgfxUnsupportedWasLogged(r, @selector(setDepthTestMinBound:maxBound:)));
	CHECK(RgfxUnsupportedWasLogged(
	    r, @selector(drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:)));
	CHECK(RgfxUnsupportedWasLogged(r, @selector(setMeshBytes:length:atIndex:)));
	[renc endEncoding];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	[cenc setAccelerationStructure:nil atBufferIndex:1];
	id<MTLVisibleFunctionTable> noTables[1] = {nil};
	[cenc setVisibleFunctionTables:noTables withBufferRange:NSMakeRange(0, 0)];
	CHECK(!RgfxUnsupportedWasLogged([RgfxComputeCommandEncoder class],
	                                @selector(setAccelerationStructure:atBufferIndex:)));
	[cenc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	// Only the framing and the queue barrier fences (test_queue_barriers) reached the wire.
	for (const Op &op : ops) {
		CHECK(op.tag == RGFX_OP_ENCODER_BEGIN_RENDER || op.tag == RGFX_OP_ENCODER_BEGIN_COMPUTE ||
		      op.tag == RGFX_OP_RENDER_UPDATE_FENCE || op.tag == RGFX_OP_COMPUTE_UPDATE_FENCE ||
		      op.tag == RGFX_OP_ENCODER_END);
	}
	fx.complete(cb, seqno);
	} // @autoreleasepool
}

// --- queue barriers -----------------------------------------------------------------

// The fence id of fence op `op` (every fence op's body starts with it).
uint64_t fenceOf(const Op &op)
{
	uint64_t id = 0;
	std::memcpy(&id, op.body.data(), sizeof(id));
	return id;
}

// The fence ops of `ops` between the nth (0-based) encoder begin and its end, as (tag, fence).
std::vector<std::pair<uint16_t, uint64_t>> encoderFences(const std::vector<Op> &ops, int nth)
{
	std::vector<std::pair<uint16_t, uint64_t>> out;
	int begins = -1;
	bool in = false;
	for (const Op &op : ops) {
		if (op.tag == RGFX_OP_ENCODER_BEGIN_RENDER || op.tag == RGFX_OP_ENCODER_BEGIN_COMPUTE ||
		    op.tag == RGFX_OP_ENCODER_BEGIN_BLIT) {
			in = ++begins == nth;
		} else if (op.tag == RGFX_OP_ENCODER_END) {
			in = false;
		} else if (in && (op.tag == RGFX_OP_RENDER_UPDATE_FENCE ||
		                  op.tag == RGFX_OP_RENDER_WAIT_FOR_FENCE ||
		                  op.tag == RGFX_OP_COMPUTE_UPDATE_FENCE ||
		                  op.tag == RGFX_OP_COMPUTE_WAIT_FOR_FENCE ||
		                  op.tag == RGFX_OP_BLIT_UPDATE_FENCE ||
		                  op.tag == RGFX_OP_BLIT_WAIT_FOR_FENCE)) {
			out.push_back({op.tag, fenceOf(op)});
		}
	}
	return out;
}

// barrierAfterQueueStages:beforeStages: waits for every encoder made before it on the queue:
// each encoder updates one of the queue's eight barrier fences at its end, and first waits
// for the encoder that last updated it; a barrier waits for the seven others an earlier encoder
// updated, once, in render, compute, blit and parallel passes, across command buffers.
void test_queue_barriers()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	RgfxTexture *target = shapeTexture(
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO],
	    0x42);
	// Encoders 0..2: render, compute, then a blit whose barrier waits for 0 and 1.
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:onePass(target)];
	[renc endEncoding];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	[cenc endEncoding];
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	[benc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll];
	[benc barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageBlit]; // nothing more
	[benc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	auto e0 = encoderFences(ops, 0), e1 = encoderFences(ops, 1), e2 = encoderFences(ops, 2);
	CHECK(e0.size() == 1 && e0[0].first == RGFX_OP_RENDER_UPDATE_FENCE);
	CHECK(e1.size() == 1 && e1[0].first == RGFX_OP_COMPUTE_UPDATE_FENCE);
	const uint64_t f0 = e0[0].second, f1 = e1[0].second;
	CHECK(f0 != 0 && f1 != 0 && f0 != f1);
	CHECK(e2.size() == 3);
	CHECK(e2[0].first == RGFX_OP_BLIT_WAIT_FOR_FENCE && e2[0].second == f0);
	CHECK(e2[1].first == RGFX_OP_BLIT_WAIT_FOR_FENCE && e2[1].second == f1);
	CHECK(e2[2].first == RGFX_OP_BLIT_UPDATE_FENCE && e2[2].second != f0 && e2[2].second != f1);
	const uint64_t f2 = e2[2].second;
	// A render pass's fences cover all its stages.
	RgfxOpUpdateFence ru = {};
	std::memcpy(&ru, findOp(ops, RGFX_OP_RENDER_UPDATE_FENCE)->body.data(), sizeof(ru));
	CHECK(ru.stages == (MTLRenderStageVertex | MTLRenderStageFragment));
	fx.complete(cb, seqno);

	// Encoders 3..7 in a second buffer take the other five fences; 8 takes encoder 0's again,
	// so it waits for 0 first, and its barrier waits for the other seven (1..7). A parallel
	// pass waits and updates as one encoder, and its sub-encoder's barrier waits as it would.
	id<MTLCommandBuffer> cb2 = [fx.queue commandBuffer];
	for (int i = 3; i < 8; i++) {
		[[cb2 blitCommandEncoder] endEncoding];
	}
	id<MTLParallelRenderCommandEncoder> par =
	    [cb2 parallelRenderCommandEncoderWithDescriptor:onePass(target)];
	id<MTLRenderCommandEncoder> sub = [par renderCommandEncoder];
	[sub barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageVertex];
	[sub endEncoding];
	[par endEncoding];
	[cb2 commit];
	ops = fx.nextOps(&seqno);
	std::vector<uint64_t> fences = {f0, f1, f2};
	for (int i = 0; i < 5; i++) {
		auto e = encoderFences(ops, i);
		CHECK(e.size() == 1 && e[0].first == RGFX_OP_BLIT_UPDATE_FENCE);
		fences.push_back(e[0].second);
	}
	auto e8 = encoderFences(ops, 5);
	CHECK(e8.size() == 9);
	CHECK(e8[0].first == RGFX_OP_RENDER_WAIT_FOR_FENCE && e8[0].second == f0); // its own, first
	for (int t = 1; t < 8; t++) {
		CHECK(e8[t].first == RGFX_OP_RENDER_WAIT_FOR_FENCE && e8[t].second == fences[t]);
	}
	CHECK(e8[8].first == RGFX_OP_RENDER_UPDATE_FENCE && e8[8].second == f0);
	fx.complete(cb2, seqno);
	} // @autoreleasepool
}


// --- the command buffer and the queue ---------------------------------------------

// The encoders of features the device does not support are nil, logged; residency sets
// (which the device does not make) change nothing; a capture boundary marks nothing.
void test_command_buffer_and_queue_members()
{
	QueueFixture fx;
	// Drained while fx lives: what the test lets go (a command buffer, the queue it holds)
	// posts its destroy through fx.
	@autoreleasepool {
	id<MTLCommandBuffer> cb = [fx.queue commandBuffer];
	CHECK([cb accelerationStructureCommandEncoder] == nil);
	CHECK([cb accelerationStructureCommandEncoderWithDescriptor:
	              [MTLAccelerationStructurePassDescriptor accelerationStructurePassDescriptor]] ==
	      nil);
	CHECK([cb resourceStateCommandEncoder] == nil);
	CHECK([cb resourceStateCommandEncoderWithDescriptor:
	              [MTLResourceStatePassDescriptor resourceStatePassDescriptor]] == nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxCmdBuffer class],
	                               @selector(accelerationStructureCommandEncoder)));
	CHECK(RgfxUnsupportedWasLogged([RgfxCmdBuffer class], @selector(resourceStateCommandEncoder)));
	id<MTLResidencySet> noSets[1] = {nil};
	[cb useResidencySets:noSets count:0];
	CHECK(!RgfxUnsupportedWasLogged([RgfxCmdBuffer class], @selector(useResidencySets:count:)));
	[fx.queue addResidencySets:noSets count:0];
	[fx.queue removeResidencySets:noSets count:0];
	CHECK(!RgfxUnsupportedWasLogged([RgfxCommandQueue class], @selector(addResidencySets:count:)));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[fx.queue insertDebugCaptureBoundary];
#pragma clang diagnostic pop
	// None of it reached a ring.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!fx.control.next(&h, &bytes));
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	CHECK(ops.empty());
	fx.complete(cb, seqno);
	} // @autoreleasepool
}


// --- textures ---------------------------------------------------------------------

// The descriptor's members the wire does not carry answer as made; a texture that is not a
// view, sparse, shared or remote answers as Metal's such texture does; views, remote views and
// shared handles are nil, logged.
void test_texture_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:32
	                                                      height:32
	                                                   mipmapped:NO];
	d.allowGPUOptimizedContents = NO;
	d.compressionType = MTLTextureCompressionTypeLossy;
	RgfxTexture *t = [RgfxTexture textureWithDescriptor:d
	                                               heap:RGFX_OBJ_NONE
	                                         heapOffset:0
	                                             device:nil
	                                         serializer:&ser];
	CHECK(t != nil);
	CHECK(t.allowGPUOptimizedContents == NO);
	CHECK(t.compressionType == MTLTextureCompressionTypeLossy);

	// Metal's defaults where no descriptor was taken.
	RgfxTexture *plain = shapeTexture(d, 0x70);
	CHECK(plain.allowGPUOptimizedContents == YES);
	CHECK(plain.compressionType == MTLTextureCompressionTypeLossless);

	CHECK(t.iosurfacePlane == 0);
	CHECK(t.isShareable == NO);
	CHECK(t.sparseTextureTier == MTLTextureSparseTierNone);
	CHECK(t.isSparse == NO && t.firstMipmapInTail == 0 && t.tailSizeInBytes == 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CHECK(t.rootResource == nil); // no parent, no buffer
	CHECK(t.remoteStorageTexture == nil);
	CHECK([t newRemoteTextureViewForDevice:(id<MTLDevice>)t] == nil);
#pragma clang diagnostic pop

	Class c = [RgfxTexture class];
	CHECK([t newSharedTextureHandle] == nil);
	CHECK(RgfxUnsupportedWasLogged(c, @selector(newSharedTextureHandle)));
	} // @autoreleasepool
}


// --- buffers and heaps ------------------------------------------------------------

// A buffer that is not sparse or remote answers as Metal's; debug markers change nothing on
// the wire; a remote view or a tensor over it is nil (with an NSError where the member has an
// error out), logged.
void test_buffer_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	RgfxBuffer *b = [RgfxBuffer bufferWithLength:256
	                                     options:MTLResourceStorageModeShared
	                                      device:nil
	                                  serializer:&ser];
	(void)expectMessage<RgfxBufferCreate>(control, RGFX_CMD_BUFFER_CREATE);
	CHECK(b.sparseBufferTier == MTLBufferSparseTierNone);
	[b addDebugMarker:@"marker" range:NSMakeRange(0, 16)];
	[b removeAllDebugMarkers];
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!control.next(&h, &bytes));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CHECK(b.remoteStorageBuffer == nil);
	CHECK([b newRemoteBufferViewForDevice:(id<MTLDevice>)b] == nil);
#pragma clang diagnostic pop
	NSError *error = nil;
	MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
	CHECK([b newTensorWithDescriptor:td offset:0 error:&error] == nil);
	CHECK(error != nil && [error.domain isEqualToString:RgfxMetalErrorDomain] &&
	      error.code == RgfxMetalErrorUnsupported);
	CHECK([b newTensorWithDescriptor:td offset:0 error:nullptr] == nil); // no error out: fine
	CHECK(RgfxUnsupportedWasLogged([RgfxBuffer class], @selector(newTensorWithDescriptor:offset:error:)));

	// A texture over the buffer: a TEXTURE_FROM_BUFFER naming it, with the offset and the rows;
	// the texture keeps the buffer and answers where it lies.
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:4
	                                                      height:2
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModeShared;
	id<MTLTexture> t = [b newTextureWithDescriptor:d offset:64 bytesPerRow:64];
	CHECK(t != nil);
	RgfxTextureFromBuffer m =
	    expectMessage<RgfxTextureFromBuffer>(control, RGFX_CMD_TEXTURE_FROM_BUFFER);
	CHECK(m.buffer_obj_id == b.rgfxObjId && m.offset == 64 && m.bytes_per_row == 64);
	CHECK(m.desc.width == 4 && m.desc.height == 2);
	CHECK(t.buffer == b && t.bufferOffset == 64 && t.bufferBytesPerRow == 64);
	} // @autoreleasepool
}

// A heap's allocated size is its size; acceleration structures are nil, logged; a Placement
// heap makes nothing at an offset it picks (only an Automatic heap does), and logs no buffer.
void test_heap_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
	hd.size = 0x100000;
	hd.type = MTLHeapTypePlacement;
	RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&ser];
	CHECK(heap != nil);
	CHECK(heap.allocatedSize == 0x100000);
	CHECK([heap newAccelerationStructureWithSize:4096] == nil);
	CHECK([heap newAccelerationStructureWithSize:4096 offset:0] == nil);
	MTLAccelerationStructureDescriptor *ad = [[MTLAccelerationStructureDescriptor alloc] init];
	CHECK([heap newAccelerationStructureWithDescriptor:ad] == nil);
	CHECK([heap newAccelerationStructureWithDescriptor:ad offset:0] == nil);
	CHECK([heap newBufferWithLength:64 options:0] == nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newAccelerationStructureWithSize:)));
	CHECK([heap newTextureWithDescriptor:[MTLTextureDescriptor
	                                         texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
	                                                                      width:4
	                                                                     height:4
	                                                                  mipmapped:NO]] == nil);
	CHECK(!RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newBufferWithLength:options:)));
	} // @autoreleasepool
}


// --- pipeline states ---------------------------------------------------------------

// What the create asked answers as asked; a pipeline a device makes has no reflection object
// and no validation; a vertex/fragment pipeline answers 0 for the mesh, object and tile
// stages; the host's memory and imageblock answers are not on the wire (0, logged); function
// handles, binary functions and function tables are nil (with an NSError where there is an
// error out), logged.
void test_pipeline_state_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	const uint8_t libBytes[] = {9, 9, 9};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&ser
	                                        status:&libStatus];
	id<MTLFunction> kfn = [lib newFunctionWithName:@"kmain"];
	id<MTLFunction> vfn = [lib newFunctionWithName:@"vmain"];
	id<MTLFunction> ffn = [lib newFunctionWithName:@"fmain"];

	// --- compute ---
	MTLComputePipelineDescriptor *cd = [[MTLComputePipelineDescriptor alloc] init];
	cd.computeFunction = kfn;
	cd.supportIndirectCommandBuffers = YES;
	cd.requiredThreadsPerThreadgroup = MTLSizeMake(8, 4, 1);
	NSError *err = nil;
	RgfxComputePipelineState *cs =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:cd
	                                                         device:nil
	                                                     serializer:&ser
	                                                          error:&err];
	CHECK(cs != nil);
	CHECK(cs.supportIndirectCommandBuffers == YES);
	MTLSize req = cs.requiredThreadsPerThreadgroup;
	CHECK(req.width == 8 && req.height == 4 && req.depth == 1);
	CHECK(cs.forwardProgressUsage == MTLForwardProgressUsageAutomatic);
	CHECK(cs.shaderValidation == MTLShaderValidationDisabled);
	CHECK(cs.reflection == nil);
	CHECK(cs.allocatedSize == 0);
	CHECK([cs imageblockMemoryLengthForDimensions:MTLSizeMake(8, 8, 1)] == 0);
	CHECK(RgfxUnsupportedWasLogged([RgfxComputePipelineState class], @selector(allocatedSize)));
	CHECK([cs functionHandleWithFunction:kfn] == nil);
	CHECK([cs functionHandleWithName:@"kmain"] == nil);
	CHECK([cs newVisibleFunctionTableWithDescriptor:[MTLVisibleFunctionTableDescriptor
	                                                    visibleFunctionTableDescriptor]] == nil);
	CHECK([cs newIntersectionFunctionTableWithDescriptor:
	              [MTLIntersectionFunctionTableDescriptor intersectionFunctionTableDescriptor]] ==
	      nil);
	err = nil;
	CHECK([cs newComputePipelineStateWithAdditionalBinaryFunctions:@[ kfn ] error:&err] == nil);
	CHECK(err != nil && err.code == RgfxMetalErrorUnsupported);
	// A function-only pipeline answers Metal's defaults.
	RgfxComputePipelineState *plain =
	    [RgfxComputePipelineState computePipelineStateWithFunction:kfn
	                                                       device:nil
	                                                   serializer:&ser
	                                                        error:nullptr];
	CHECK(plain.supportIndirectCommandBuffers == NO);
	MTLSize none = plain.requiredThreadsPerThreadgroup;
	CHECK(none.width == 0 && none.height == 0 && none.depth == 0);
	// Metal's private members: no GPU address, and an identifier unique to the object.
	auto u64 = [](id object, NSString *name) {
		return ((uint64_t(*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
	};
	CHECK(u64(cs, @"gpuAddress") == 0 && u64(plain, @"gpuAddress") == 0);
	CHECK(u64(cs, @"uniqueIdentifier") == reinterpret_cast<uintptr_t>((__bridge void *)cs));
	CHECK(u64(cs, @"uniqueIdentifier") == u64(cs, @"uniqueIdentifier"));
	CHECK(u64(cs, @"uniqueIdentifier") != u64(plain, @"uniqueIdentifier"));

	// --- render ---
	MTLRenderPipelineDescriptor *rd = [[MTLRenderPipelineDescriptor alloc] init];
	rd.vertexFunction = vfn;
	rd.fragmentFunction = ffn;
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	rd.supportIndirectCommandBuffers = YES;
	RgfxRenderPipelineState *rs =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:rd
	                                                       device:nil
	                                                   serializer:&ser
	                                                        error:nullptr];
	CHECK(rs != nil);
	CHECK(rs.supportIndirectCommandBuffers == YES);
	CHECK(rs.shaderValidation == MTLShaderValidationDisabled);
	CHECK(rs.reflection == nil);
	CHECK(rs.maxTotalThreadsPerObjectThreadgroup == 0 && rs.maxTotalThreadsPerMeshThreadgroup == 0);
	CHECK(rs.objectThreadExecutionWidth == 0 && rs.meshThreadExecutionWidth == 0);
	CHECK(rs.maxTotalThreadgroupsPerMeshGrid == 0);
	CHECK(rs.requiredThreadsPerTileThreadgroup.width == 0);
	CHECK(rs.requiredThreadsPerObjectThreadgroup.width == 0);
	CHECK(rs.requiredThreadsPerMeshThreadgroup.width == 0);
	CHECK(u64(rs, @"gpuAddress") == 0);
	CHECK(u64(rs, @"uniqueIdentifier") == reinterpret_cast<uintptr_t>((__bridge void *)rs));
	// The two members a compute pipeline declares, which Metal's render pipelines answer too.
	id<MTLComputePipelineState> asCompute = (id<MTLComputePipelineState>)rs;
	MTLSize rsRequired = asCompute.requiredThreadsPerThreadgroup;
	CHECK(rsRequired.width == 0 && rsRequired.height == 0 && rsRequired.depth == 0);
	CHECK(asCompute.staticThreadgroupMemoryLength == 0);
	CHECK(rs.allocatedSize == 0);
	CHECK([rs functionHandleWithFunction:vfn stage:MTLRenderStageVertex] == nil);
	CHECK([rs newRenderPipelineDescriptorForSpecialization] == nil);
	err = nil;
	CHECK([rs newRenderPipelineStateWithAdditionalBinaryFunctions:
	              [[MTLRenderPipelineFunctionsDescriptor alloc] init]
	                                                        error:&err] == nil);
	CHECK(err != nil && err.code == RgfxMetalErrorUnsupported);
	CHECK(RgfxUnsupportedWasLogged([RgfxRenderPipelineState class],
	                               @selector(newRenderPipelineDescriptorForSpecialization)));

	// --- depth/stencil ---
	RgfxDepthStencilState *ds =
	    [RgfxDepthStencilState depthStencilStateWithDescriptor:[[MTLDepthStencilDescriptor alloc] init]
	                                                    device:nil
	                                                serializer:&ser];
	// No resource id, as on both x86 Macs: it raises an unrecognized selector, unlogged.
	bool raised = false;
	@try {
		(void)ds.gpuResourceID;
	} @catch (NSException *e) {
		raised = [e.name isEqualToString:NSInvalidArgumentException];
	}
	CHECK(raised);
	CHECK(!RgfxUnsupportedWasLogged([RgfxDepthStencilState class], @selector(gpuResourceID)));
	} // @autoreleasepool
}


// --- libraries, argument encoders and shared events -----------------------------------

// An intersection function is nil with Metal's own unsupported library error, through both
// forms; a function's reflection is nil, logged.
void test_library_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	@autoreleasepool {
		const uint8_t libBytes[] = {1, 2, 3};
		uint32_t libStatus = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
		                                        length:sizeof(libBytes)
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&ser
		                                        status:&libStatus];
		MTLIntersectionFunctionDescriptor *d = [[MTLIntersectionFunctionDescriptor alloc] init];
		d.name = @"hit";
		NSError *err = nil;
		CHECK([lib newIntersectionFunctionWithDescriptor:d error:&err] == nil);
		CHECK(err != nil && [err.domain isEqualToString:MTLLibraryErrorDomain] &&
		      err.code == MTLLibraryErrorUnsupported);
		__block BOOL called = NO;
		__block NSError *handlerErr = nil;
		[lib newIntersectionFunctionWithDescriptor:d
		                         completionHandler:^(id<MTLFunction> fn, NSError *e) {
			                         called = fn == nil;
			                         handlerErr = e;
		                         }];
		CHECK(RgfxWaitForCreations(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)));
		CHECK(called && handlerErr.code == MTLLibraryErrorUnsupported);
		CHECK([lib reflectionForFunctionWithName:@"vmain"] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxLibrary class],
		                               @selector(reflectionForFunctionWithName:)));
	} // @autoreleasepool: the library drains while ser lives
}

// A fake device's sampler creates: each answered SAMPLER_STATE with `status` and the next id from
// `id` on, the creates it was asked kept.
struct SamplerAsker {
	uint64_t id = 1;
	uint32_t status = RGFX_STATUS_OK;
	bool answers = true;
	std::vector<RgfxSamplerCreate> asked;

	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		SamplerAsker *self = static_cast<SamplerAsker *>(context);
		RgfxSamplerCreate m = {};
		CHECK(size == sizeof(m));
		std::memcpy(&m, message, sizeof(m));
		self->asked.push_back(m);
		if (!self->answers)
			return false;
		RgfxSamplerState answer = {};
		answer.header = {RGFX_EVT_SAMPLER_STATE, sizeof(answer), 11};
		answer.status = self->status;
		answer.gpu_resource_id = self->status == RGFX_STATUS_OK ? self->id++ : 0;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}
};

// A sampler takes the resource id its create was answered with, and is nil where the device
// refused it or no answer came back; with no asker its create goes on the ring unanswered.
void test_sampler_resource_ids()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	SamplerAsker asker;
	asker.id = 7;
	ser.setAsker(&SamplerAsker::ask, &asker);
	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
	d.minFilter = MTLSamplerMinMagFilterLinear;
	d.supportArgumentBuffers = YES;
	d.lodMaxClamp = 4.0f;
	RgfxSampler *a = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&ser];
	RgfxSampler *b = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&ser];
	CHECK(a != nil && b != nil);
	CHECK(a.gpuResourceID._impl == 7 && b.gpuResourceID._impl == 8);
	CHECK(asker.asked.size() == 2);
	const RgfxSamplerCreate &m = asker.asked[0];
	CHECK(m.header.type == RGFX_CMD_SAMPLER_CREATE && m.header.size == sizeof(m));
	CHECK(m.obj_id == a.rgfxObjId && asker.asked[1].obj_id == b.rgfxObjId);
	CHECK(m.min_filter == MTLSamplerMinMagFilterLinear && (m.flags & 0x4) != 0);
	float lod = 0;
	std::memcpy(&lod, &m.lod_max_clamp_bits, sizeof(lod));
	CHECK(lod == 4.0f);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!control.next(&h, &bytes)); // asked, not posted

	asker.status = RGFX_STATUS_REFUSED;
	CHECK([RgfxSampler samplerWithDescriptor:d device:nil serializer:&ser] == nil);
	asker.answers = false;
	CHECK([RgfxSampler samplerWithDescriptor:d device:nil serializer:&ser] == nil);
	CHECK(asker.asked.size() == 4);

	ser.setAsker(nullptr, nullptr);
	RgfxSampler *plain = [RgfxSampler samplerWithDescriptor:d device:nil serializer:&ser];
	CHECK(plain != nil && plain.gpuResourceID._impl == 0);
	(void)expectMessage<RgfxSamplerCreate>(control, RGFX_CMD_SAMPLER_CREATE);
	} // @autoreleasepool
}

// The word at `offset` of `buffer`'s contents.
uint64_t wordAt(id<MTLBuffer> buffer, uint64_t offset)
{
	uint64_t v = 0;
	std::memcpy(&v, static_cast<uint8_t *>(buffer.contents) + offset, sizeof(v));
	return v;
}

// An argument encoder writes each argument's word where its layout puts the index, in the
// argument buffer at the offset it was set to: a buffer's GPU address plus the offset, a
// texture's or pipeline's resource id, 0 for nil, at the arguments' offset 0 for an index no
// argument has, each word of a range in turn; nothing past the buffer's end, nor with no
// argument buffer set. A depth/stencil state and a sampler with no resource id log that the
// wire carries none; what the device makes no object of takes only nil, which writes 0.
void test_argument_encoder_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	Class c = [RgfxArgumentEncoder class];
	auto probe = probeReply().bytes();
	RgfxArgumentEncoder *enc =
	    [[RgfxArgumentEncoder alloc] initWithLayout:RgfxArgLayout::parse(probe.data(), probe.size())
	                                         device:nil
	                                     serializer:&ser];
	CHECK(enc.rgfxObjId == 0 && enc.encodedLength == 64 && enc.alignment == 16);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!control.next(&h, &bytes)); // an encoder makes nothing on the device

	// No argument buffer: nothing to write into.
	[enc setBuffer:nil offset:0 atIndex:4];
	CHECK([enc constantDataAtIndex:3] == nullptr);

	RgfxBuffer *args = [[RgfxBuffer alloc] initWithObjId:0x50
	                                          gpuAddress:0x1000000
	                                              length:256
	                                             options:MTLResourceStorageModeShared
	                                              device:nil
	                                          serializer:nullptr];
	RgfxBuffer *data = [[RgfxBuffer alloc] initWithObjId:0x51
	                                          gpuAddress:0x2000000
	                                              length:64
	                                             options:MTLResourceStorageModeShared
	                                              device:nil
	                                          serializer:nullptr];
	std::memset(args.contents, 0xaa, 256);
	[enc setArgumentBuffer:args offset:64];
	[enc setBuffer:data offset:16 atIndex:4];
	CHECK(wordAt(args, 64 + 48) == 0x2000010);
	RgfxTextureDescriptor tw = {};
	tw.pixel_format = 70;
	tw.texture_type = 2;
	tw.width = tw.height = 4;
	tw.depth = tw.mipmap_level_count = tw.array_length = tw.sample_count = 1;
	RgfxTexture *named = [[RgfxTexture alloc] initWithObjId:0x61
	                                             gpuAddress:0
	                                             resourceId:0x1234
	                                             descriptor:tw
	                                                 device:nil
	                                             serializer:nullptr];
	[enc setTexture:named atIndex:0];
	CHECK(wordAt(args, 64) == 0x1234);
	[enc setTexture:nil atIndex:0];
	CHECK(wordAt(args, 64) == 0);
	CHECK(static_cast<uint8_t *>([enc constantDataAtIndex:3]) ==
	      static_cast<uint8_t *>(args.contents) + 64 + 32);
	// An index no argument has lands at the arguments' offset 0, as Metal's encoder writes it.
	[enc setBuffer:data offset:0 atIndex:9];
	CHECK(wordAt(args, 64) == 0x2000000);
	CHECK(static_cast<uint8_t *>([enc constantDataAtIndex:9]) ==
	      static_cast<uint8_t *>(args.contents) + 64);
	CHECK(wordAt(args, 56) == 0xaaaaaaaaaaaaaaaaull); // nothing before the arguments

	// The third element of an array of argument buffers; nothing past the buffer's end.
	[enc setArgumentBuffer:args startOffset:0 arrayElement:2];
	[enc setBuffer:data offset:8 atIndex:4];
	CHECK(wordAt(args, 128 + 48) == 0x2000008);
	[enc setArgumentBuffer:args offset:240];
	[enc setBuffer:data offset:0 atIndex:4];
	CHECK(wordAt(args, 240) == 0xaaaaaaaaaaaaaaaaull);

	// A nested argument buffer's encoder, from the layout; none where the index holds none.
	id<MTLArgumentEncoder> inner = [enc newArgumentEncoderForBufferAtIndex:1];
	CHECK(inner != nil && inner.encodedLength == 16 && inner.alignment == 8);
	CHECK([enc newArgumentEncoderForBufferAtIndex:4] == nil);
	CHECK([enc newArgumentEncoderForBufferAtIndex:0] == nil);
	CHECK([enc newArgumentEncoderForBufferAtIndex:7] == nil);
	[inner setArgumentBuffer:args offset:16];
	[inner setTexture:named atIndex:0];
	CHECK(wordAt(args, 16) == 0x1234);

	// A sampler's word is the resource id the host answered its create with.
	[enc setArgumentBuffer:args offset:0];
	SamplerAsker samplers;
	samplers.id = 0x2a;
	ser.setAsker(&SamplerAsker::ask, &samplers);
	RgfxSampler *smp = [RgfxSampler samplerWithDescriptor:[[MTLSamplerDescriptor alloc] init]
	                                               device:nil
	                                           serializer:&ser];
	ser.setAsker(nullptr, nullptr);
	CHECK(smp != nil && smp.gpuResourceID._impl == 0x2a);
	std::memset(args.contents, 0x5c, 24);
	[enc setSamplerState:nil atIndex:2];
	CHECK(wordAt(args, 16) == 0);
	[enc setSamplerState:smp atIndex:2];
	CHECK(wordAt(args, 16) == 0x2a);
	CHECK(!RgfxUnsupportedWasLogged(c, @selector(setSamplerState:atIndex:)));
	// A depth/stencil state has no resource id, as on both x86 Macs: its gpuResourceID raises an
	// unrecognized selector, and setting a state, nil among them, raises one of the encoder's,
	// though it responds, before anything is written.
	RgfxDepthStencilState *ds =
	    [RgfxDepthStencilState depthStencilStateWithDescriptor:[[MTLDepthStencilDescriptor alloc] init]
	                                                    device:nil
	                                                serializer:&ser];
	CHECK([ds respondsToSelector:@selector(gpuResourceID)]);
	CHECK([enc respondsToSelector:@selector(setDepthStencilState:atIndex:)]);
	auto raisesUnrecognized = [](NSString *selector, void (^call)(void)) {
		bool raised = false;
		@try {
			call();
		} @catch (NSException *e) {
			raised = [e.name isEqualToString:NSInvalidArgumentException] &&
			         [e.reason containsString:[selector stringByAppendingString:
			                                                 @"]: unrecognized selector"]];
		}
		return raised;
	};
	CHECK(raisesUnrecognized(@"gpuResourceID", ^{
		(void)ds.gpuResourceID;
	}));
	std::memset(args.contents, 0x5c, 8);
	CHECK(raisesUnrecognized(@"setDepthStencilState:atIndex:", ^{
		[enc setDepthStencilState:ds atIndex:0];
	}));
	CHECK(wordAt(args, 0) == 0x5c5c5c5c5c5c5c5cull);
	CHECK(raisesUnrecognized(@"setDepthStencilState:atIndex:", ^{
		[enc setDepthStencilState:nil atIndex:0];
	}));
	CHECK(wordAt(args, 0) == 0x5c5c5c5c5c5c5c5cull);
	CHECK(!RgfxUnsupportedWasLogged(c, @selector(setDepthStencilState:atIndex:)));
	// An indirect command buffer writes its gpuResourceID, the host's id, as a pipeline; nil 0.
	{
		RgfxIcbState state = {};
		state.header = {RGFX_EVT_ICB_STATE, sizeof(state), 1};
		state.status = RGFX_STATUS_OK;
		state.gpu_resource_id = 0x3c;
		static RgfxIcbState answer;
		answer = state;
		ser.setAsker(
		    [](void *, const void *, uint32_t, void *reply, uint32_t, uint32_t *size) {
			    std::memcpy(reply, &answer, sizeof(answer));
			    *size = sizeof(answer);
			    return true;
		    },
		    nullptr);
		MTLIndirectCommandBufferDescriptor *icbd = [MTLIndirectCommandBufferDescriptor new];
		icbd.commandTypes = MTLIndirectCommandTypeDraw;
		RgfxIndirectCommandBuffer *icb = [RgfxIndirectCommandBuffer icbWithDescriptor:icbd
		                                                              maxCommandCount:2
		                                                                      options:0
		                                                                       device:nil
		                                                                   serializer:&ser];
		ser.setAsker(nullptr, nullptr);
		CHECK(icb != nil);
		[enc setIndirectCommandBuffer:icb atIndex:0];
		CHECK(wordAt(args, 0) == 0x3c);
	}
	[enc setIndirectCommandBuffer:nil atIndex:0];
	CHECK(wordAt(args, 0) == 0);
	CHECK(!RgfxUnsupportedWasLogged(c, @selector(setIndirectCommandBuffer:atIndex:)));
	// What the device makes no object of: nil writes 0, unlogged.
	std::memset(args.contents, 0x5c, 8);
	[enc setAccelerationStructure:nil atIndex:0];
	[enc setVisibleFunctionTable:nil atIndex:0];
	CHECK(wordAt(args, 0) == 0);
	// The atBufferIndex: and withBufferRange: names write as atIndex: and withRange:.
	auto tableAt = [&](const char *name) {
		std::memset(args.contents, 0x5c, 16);
		((void (*)(id, SEL, id, NSUInteger))objc_msgSend)(enc, sel_registerName(name), nil, 1);
		return wordAt(args, 0) == 0x5c5c5c5c5c5c5c5cull && wordAt(args, 8) == 0;
	};
	CHECK(tableAt("setVisibleFunctionTable:atBufferIndex:"));
	CHECK(tableAt("setIntersectionFunctionTable:atBufferIndex:"));
	auto tablesIn = [&](const char *name) {
		std::memset(args.contents, 0x5c, 24);
		const id none[2] = {nil, nil};
		((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(enc, sel_registerName(name), none,
		                                                         NSMakeRange(1, 2));
		return wordAt(args, 0) == 0x5c5c5c5c5c5c5c5cull && wordAt(args, 8) == 0 &&
		       wordAt(args, 16) == 0;
	};
	CHECK(tablesIn("setVisibleFunctionTables:withBufferRange:"));
	CHECK(tablesIn("setIntersectionFunctionTables:withBufferRange:"));

	// A range writes each word in turn: the descriptors' layout of a texture, four pointers,
	// a float4 and a float (measured: M3 Max, macOS 27).
	ArgReply arr;
	arr.layout(80, 16)
	    .member(0, RGFX_ARGUMENT_TEXTURE, 0, 8)
	    .member(1, RGFX_ARGUMENT_BUFFER, 8, 8, 4, 8)
	    .member(5, RGFX_ARGUMENT_CONSTANT, 48, 16)
	    .member(6, RGFX_ARGUMENT_CONSTANT, 64, 4);
	auto ab = arr.bytes();
	RgfxArgumentEncoder *ranged =
	    [[RgfxArgumentEncoder alloc] initWithLayout:RgfxArgLayout::parse(ab.data(), ab.size())
	                                         device:nil
	                                     serializer:nullptr];
	CHECK(ranged.encodedLength == 80);
	[ranged setArgumentBuffer:args offset:0];
	id<MTLBuffer> four[4] = {data, nil, data, data};
	NSUInteger offsets[4] = {0, 8, 16, 24};
	[ranged setBuffers:four offsets:offsets withRange:NSMakeRange(1, 4)];
	CHECK(wordAt(args, 8) == 0x2000000 && wordAt(args, 16) == 0);
	CHECK(wordAt(args, 24) == 0x2000010 && wordAt(args, 32) == 0x2000018);
	id<MTLTexture> textures[2] = {named, nil};
	std::memset(args.contents, 0xaa, 16);
	[ranged setTextures:textures withRange:NSMakeRange(0, 2)];
	CHECK(wordAt(args, 0) == 0x1234 && wordAt(args, 8) == 0);
	} // @autoreleasepool
}

// A fake device's argument layouts, written into the region's block the query names: the
// whole layout where it fits, else the bytes it needs and TOO_SMALL; `status` where it is set.
struct RegionAsker {
	uint8_t *region = nullptr;
	std::vector<uint8_t> reply;
	std::vector<RgfxArgumentLayoutQuery> asked;
	std::vector<std::string> names;
	uint32_t status = RGFX_STATUS_OK;

	static bool ask(void *context, const void *message, uint32_t size, void *answer,
	                uint32_t capacity, uint32_t *answer_size)
	{
		RegionAsker *self = static_cast<RegionAsker *>(context);
		RgfxArgumentLayoutQuery q = {};
		CHECK(size >= sizeof(q));
		std::memcpy(&q, message, sizeof(q));
		CHECK(q.header.type == RGFX_CMD_ARGUMENT_LAYOUT);
		self->asked.push_back(q);
		self->names.emplace_back(static_cast<const char *>(message) + q.name_offset, q.name_len);
		RgfxComplete c = {};
		c.header = {RGFX_EVT_COMPLETE, sizeof(c), 7};
		c.status = self->status;
		if (c.status == RGFX_STATUS_OK && self->reply.size() > q.reply_length) {
			const uint64_t needed = self->reply.size();
			std::memcpy(self->region + q.reply_offset, &needed, sizeof(needed));
			c.status = RGFX_STATUS_TOO_SMALL;
		} else if (c.status == RGFX_STATUS_OK) {
			std::memcpy(self->region + q.reply_offset, self->reply.data(), self->reply.size());
		}
		CHECK(capacity >= sizeof(c));
		std::memcpy(answer, &c, sizeof(c));
		*answer_size = sizeof(c);
		return true;
	}
};

// The region's blocks, counted: a bump of whole pages past the reserved head.
struct CountedBlocks {
	uint64_t next = RGFX_ALLOCATIONS_OFFSET;
	int live = 0;

	static bool alloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
	{
		CountedBlocks *self = static_cast<CountedBlocks *>(context);
		self->next = (self->next + align - 1) & ~(align - 1);
		*offset = self->next;
		self->next += (size + 4095) & ~uint64_t(4095);
		self->live++;
		return true;
	}
	static void release(void *context, uint64_t)
	{
		static_cast<CountedBlocks *>(context)->live--;
	}
};

// The device's and a function's encoders take their layout from the host's answer, asked into
// a block of the region that goes back once the answer is read, and again, once, into a block
// of the size the host says it needs; nil where the host refuses, or there is no region.
void test_argument_encoder_query()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + (8u << 20), 0);
	RgfxRegion region(mem.data(), mem.size());
	CountedBlocks blocks;
	region.setBlockSource(&CountedBlocks::alloc, &CountedBlocks::release, &blocks);
	RegionAsker asker;
	asker.region = mem.data();
	asker.reply = probeReply().bytes();
	ser.setAsker(&RegionAsker::ask, &asker);

	// Drained while the transport lives: what the test lets go posts its destroy through ser.
	@autoreleasepool {
	MTLArgumentDescriptor *t = [MTLArgumentDescriptor argumentDescriptor];
	t.dataType = MTLDataTypeTexture;
	t.index = 0;
	t.textureType = MTLTextureType2D;
	t.access = MTLBindingAccessReadOnly;
	MTLArgumentDescriptor *p = [MTLArgumentDescriptor argumentDescriptor];
	p.dataType = MTLDataTypePointer;
	p.index = 1;
	p.arrayLength = 4;
	RgfxArgumentEncoder *enc = [RgfxArgumentEncoder encoderOfArguments:@[ t, p ]
	                                                            region:&region
	                                                            device:nil
	                                                        serializer:&ser];
	CHECK(enc != nil && enc.encodedLength == 64 && enc.alignment == 16);
	CHECK(asker.asked.size() == 1 && blocks.live == 0);
	const RgfxArgumentLayoutQuery &q = asker.asked[0];
	CHECK(q.source == RGFX_ARGUMENT_SOURCE_DESCRIPTORS && q.count == 2 && q.reply_length == 4096);
	CHECK(q.reply_offset >= RGFX_ALLOCATIONS_OFFSET);

	// A layout past the first block: asked again into one of the size the host needs.
	ArgReply big;
	big.layout(8 * 100, 8);
	for (uint32_t i = 0; i < 100; i++)
		big.member(i, RGFX_ARGUMENT_BUFFER, 8 * i, 8);
	asker.reply = big.bytes();
	CHECK(asker.reply.size() > 4096);
	enc = [RgfxArgumentEncoder encoderOfArguments:@[ p ] region:&region device:nil serializer:&ser];
	CHECK(enc != nil && enc.encodedLength == 800 && enc.rgfxLayout->members.size() == 100);
	CHECK(asker.asked.size() == 3 && asker.asked[2].reply_length >= asker.reply.size());
	CHECK(blocks.live == 0);

	// A function's buffer argument, under its library, name and constants.
	RgfxResource *lib = [[RgfxResource alloc] initWithObjId:0x70 device:nil serializer:&ser];
	RgfxFunctionConstant fc = {};
	fc.index = 1;
	fc.data_type = 53; // MTLDataTypeBool
	fc.value[0] = 1;
	RgfxFunction *fn = [RgfxFunction functionWithLibrary:lib
	                                                name:@"argument_probe"
	                                           constants:&fc
	                                               count:1
	                                              device:nil
	                                          serializer:&ser];
	asker.reply = probeReply().bytes();
	enc = [RgfxArgumentEncoder encoderOfFunction:fn
	                                 bufferIndex:2
	                                      region:&region
	                                      device:nil
	                                  serializer:&ser];
	CHECK(enc != nil && enc.encodedLength == 64);
	const RgfxArgumentLayoutQuery &fq = asker.asked.back();
	CHECK(fq.source == RGFX_ARGUMENT_SOURCE_FUNCTION && fq.library_obj_id == 0x70);
	CHECK(fq.buffer_index == 2 && fq.constant_count == 1);
	CHECK(asker.names.back() == "argument_probe");

	// Refused, or no region: no encoder, and no block kept.
	asker.status = RGFX_STATUS_REFUSED;
	CHECK([RgfxArgumentEncoder encoderOfArguments:@[ t ] region:&region device:nil
	                                   serializer:&ser] == nil);
	CHECK([RgfxArgumentEncoder encoderOfFunction:fn
	                                 bufferIndex:0
	                                      region:&region
	                                      device:nil
	                                  serializer:&ser] == nil);
	CHECK(blocks.live == 0);
	const size_t asked = asker.asked.size();
	CHECK([RgfxArgumentEncoder encoderOfArguments:@[ t ] region:nullptr device:nil
	                                   serializer:&ser] == nil);
	// A function with no device to hold a region makes none, nor answers a reflection.
	MTLAutoreleasedArgument reflection = (MTLArgument *)@"unset";
	CHECK([fn newArgumentEncoderWithBufferIndex:0 reflection:&reflection] == nil);
	CHECK(reflection == nil && asker.asked.size() == asked);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!control.next(&h, &bytes)); // queries go through the asker, never the ring
	} // @autoreleasepool
}

// An encoder of a buffer binding (newArgumentEncoderWithBufferBinding:) and an encoder's
// reflection, over every_access of tests/air_arguments.metal and its library's reflection, Metal's
// own over the same bytes: a binding of the library's reflection is the function's own encoder,
// asked of the host by the function, as both x86 Macs answer the same encoder for the two; a
// binding the device did not answer (Metal's own pipeline reflection here) is laid out from
// descriptors of its struct's members; a plain buffer's binding has none, as the AMD answers. The
// reflection out is the library reflection's binding of that buffer.
void test_argument_encoders_of_bindings()
{
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	TestRing control(0x40000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + (8u << 20), 0);
	RgfxRegion region(mem.data(), mem.size());
	CountedBlocks blocks;
	region.setBlockSource(&CountedBlocks::alloc, &CountedBlocks::release, &blocks);
	RegionAsker asker;
	asker.region = mem.data();
	asker.reply = probeReply().bytes();
	ser.setAsker(&RegionAsker::ask, &asker);
	@autoreleasepool {
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
	                                         length:bytes.length
	                                            key:nullptr
	                                         device:nil
	                                     serializer:&ser
	                                         status:&status];
	CHECK(lib != nil);
	[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		     (void)bytes;
	     })
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
		       return [metal newLibraryWithData:d error:nil];
	       }];
	MTLFunctionReflection *reflection = [lib reflectionForFunctionWithName:@"every_access"];
	id<MTLBufferBinding> argumentBuffer = nil, plain = nil;
	for (id<MTLBinding> b in reflection.bindings) {
		if (b.type == MTLBindingTypeBuffer && b.index == 2)
			argumentBuffer = (id<MTLBufferBinding>)b;
		if (b.type == MTLBindingTypeBuffer && b.index == 0)
			plain = (id<MTLBufferBinding>)b;
	}
	CHECK(argumentBuffer != nil && plain != nil);
	CHECK(argumentBuffer.bufferPointerType.elementIsArgumentBuffer);

	// The library's binding: the function's own encoder of buffer 2.
	RgfxArgumentEncoder *enc = [RgfxArgumentEncoder encoderOfBinding:argumentBuffer
	                                                           region:&region
	                                                           device:nil
	                                                       serializer:&ser];
	CHECK(enc != nil && enc.encodedLength == 64 && asker.asked.size() == 1);
	CHECK(asker.asked.back().source == RGFX_ARGUMENT_SOURCE_FUNCTION);
	CHECK(asker.asked.back().buffer_index == 2 && asker.asked.back().library_obj_id == lib.rgfxObjId);
	CHECK(asker.names.back() == "every_access" && blocks.live == 0);
	// A plain buffer's binding: none, and nothing asked.
	CHECK([RgfxArgumentEncoder encoderOfBinding:plain region:&region device:nil serializer:&ser] == nil);
	CHECK(asker.asked.size() == 1);
	// The function's encoders at its plain buffers, 0 and 1: of no arguments, 0 bytes long, as
	// both x86 Macs make (oracle_bundle argbuf.index.1), and nothing asked.
	RgfxFunction *everyAccess = (RgfxFunction *)[lib newFunctionWithName:@"every_access"];
	CHECK(everyAccess != nil && [everyAccess rgfxAirFunction] != nullptr);
	for (NSUInteger index : {NSUInteger(0), NSUInteger(1)}) {
		id<MTLArgumentEncoder> none = [everyAccess newArgumentEncoderWithBufferIndex:index];
		CHECK(none != nil && none.encodedLength == 0 && none.alignment == 1);
	}
	CHECK(asker.asked.size() == 1);

	// A binding the device did not answer: descriptors of its struct's members, by their indices.
	id<MTLLibrary> metals = [metal newLibraryWithData:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		                                  (void)bytes;
	                                  })
	                                            error:nil];
	MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
	bool scale = true;
	[values setConstantValue:&scale type:MTLDataTypeBool atIndex:0];
	id<MTLFunction> metalsFunction = [metals newFunctionWithName:@"every_access" constantValues:values error:nil];
	MTLComputePipelineReflection *pipelineReflection = nil;
	id<MTLComputePipelineState> pipeline =
	    [metal newComputePipelineStateWithFunction:metalsFunction
	                                       options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
	                                    reflection:&pipelineReflection
	                                         error:nil];
	CHECK(pipeline != nil);
	id<MTLBufferBinding> foreign = nil;
	for (id<MTLBinding> b in pipelineReflection.bindings)
		if (b.type == MTLBindingTypeBuffer && b.index == 2)
			foreign = (id<MTLBufferBinding>)b;
	CHECK(foreign != nil);
	NSArray<MTLArgumentDescriptor *> *descriptors = RgfxDescriptorsOfArguments(foreign.bufferStructType);
	CHECK(descriptors.count == 2);
	CHECK(descriptors[0].index == 0 && descriptors[0].dataType == MTLDataTypeTexture &&
	      descriptors[0].textureType == MTLTextureType2D &&
	      descriptors[0].access == MTLBindingAccessWriteOnly);
	CHECK(descriptors[1].index == 1 && descriptors[1].dataType == MTLDataTypePointer);
	enc = [RgfxArgumentEncoder encoderOfBinding:foreign region:&region device:nil serializer:&ser];
	CHECK(enc != nil && asker.asked.size() == 2);
	CHECK(asker.asked.back().source == RGFX_ARGUMENT_SOURCE_DESCRIPTORS && asker.asked.back().count == 2);

	// The reflection out: the library reflection's binding of the buffer.
	id<MTLFunction> fn = [lib newFunctionWithName:@"every_access"];
	CHECK([(id)fn isKindOfClass:[RgfxFunction class]]);
	enc = [RgfxArgumentEncoder encoderOfFunction:(RgfxFunction *)fn
	                                 bufferIndex:2
	                                      region:&region
	                                      device:nil
	                                  serializer:&ser];
	CHECK(enc != nil);
	MTLArgument *of2 = [(RgfxFunction *)fn rgfxArgumentOfBufferAtIndex:2];
	CHECK(of2 != nil && [of2.name isEqualToString:argumentBuffer.name] && of2.index == 2 &&
	      of2.type == MTLArgumentTypeBuffer && of2.bufferDataType == MTLDataTypeStruct);
	CHECK([(RgfxFunction *)fn rgfxArgumentOfBufferAtIndex:7] == nil);
	} // @autoreleasepool
	RgfxMsgHeader h = {};
	std::vector<uint8_t> drained;
	while (control.next(&h, &drained)) {
	}
}

// waitUntilSignaledValue:timeoutMS: answers YES at once for a value reached, NO once the
// timeout passes, and YES when the value is reached meanwhile: by the CPU's set on another
// thread, or by the host's EVENT_SIGNALLED, which the wait takes off a reader no thread drives.
void test_shared_event_wait()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);
	@autoreleasepool {
		RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:5
		                                                             device:nil
		                                                         serializer:&ser];
		CHECK([ev waitUntilSignaledValue:5 timeoutMS:0] == YES);
		CHECK([ev waitUntilSignaledValue:6 timeoutMS:20] == NO);

		std::thread setter([ev] {
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			ev.signaledValue = 9;
		});
		CHECK([ev waitUntilSignaledValue:9 timeoutMS:5000] == YES);
		setter.join();

		[ev rgfxRegisterWithReplyReader:&reader];
		RgfxEventSignalled sig = {};
		sig.header.type = RGFX_EVT_EVENT_SIGNALLED;
		sig.header.size = sizeof(sig);
		sig.obj_id = ev.rgfxObjId;
		sig.value = 12;
		CHECK(completion.postFromHost(&sig, sizeof(sig)));
		CHECK([ev waitUntilSignaledValue:12 timeoutMS:5000] == YES);
		CHECK(ev.signaledValue == 12);
		CHECK([ev waitUntilSignaledValue:13 timeoutMS:10] == NO);
	} // @autoreleasepool: ev drains while ser and reader live
}

// --- heap placement sizes ------------------------------------------------------------

// A heap buffer is its length at the host's alignment (Metal's 256 where the host gave none).
// A heap texture's bound holds Metal's answer for shapes of each kind (Metal's sizes measured on
// an M3 Max, macOS 27), at Metal's coarsest alignment; a descriptor of no texture answers 0.
void test_heap_sizes()
{
	RgfxHeapSizeAlign b = rgfxHeapBufferSizeAndAlign(100001, 256);
	CHECK(b.size == 100001 && b.align == 256);
	b = rgfxHeapBufferSizeAndAlign(7, 0);
	CHECK(b.size == 7 && b.align == kRgfxHeapBufferAlign);
	b = rgfxHeapBufferSizeAndAlign(0, 4096);
	CHECK(b.size == 0 && b.align == 4096);

	struct Shape {
		uint32_t format, type;
		uint64_t w, h, d;
		uint32_t levels;
		uint64_t array, samples, metal; // Metal's size
	};
	const Shape shapes[] = {
	    {10, 2, 1, 1, 1, 1, 1, 1, 128},                 // R8Unorm 1x1
	    {10, 2, 100, 37, 1, 7, 1, 1, 11776},            // R8Unorm, mipmapped
	    {10, 2, 1920, 1080, 1, 11, 1, 1, 3457408},      // the bound's closest (1.06x)
	    {70, 2, 1920, 1080, 1, 1, 1, 1, 8486912},       // RGBA8Unorm
	    {70, 4, 1920, 1080, 1, 1, 1, 4, 33947648},      // RGBA8Unorm, 4 samples
	    {252, 4, 1920, 1080, 1, 1, 1, 4, 33947648},     // Depth32Float, 4 samples
	    {130, 2, 1920, 1080, 1, 1, 1, 1, 1179648},      // BC1_RGBA
	    {115, 3, 333, 777, 1, 10, 3, 1, 10469376},      // RGBA16Float array
	    {70, 5, 64, 64, 1, 7, 1, 1, 198912},            // RGBA8Unorm cube
	    {10, 7, 256, 256, 256, 9, 1, 1, 25919488},      // R8Unorm 3D, mipmapped
	    {125, 1, 16384, 1, 1, 1, 9, 1, 2359296},        // RGBA32Float 1D array
	    {70, 2, 16384, 16384, 1, 15, 1, 1, 1442841088}, // RGBA8Unorm 16K, mipmapped
	};
	for (const Shape &t : shapes) {
		RgfxHeapSizeAlign sa = {};
		CHECK(rgfxHeapTextureBound(t.format, t.type, t.w, t.h, t.d, t.levels, t.array, t.samples,
		                           &sa));
		CHECK(sa.size >= t.metal && sa.align == kRgfxHeapTextureGranule);
		CHECK(sa.size % sa.align == 0);
	}
	// A large texture's bound stays near Metal's size.
	RgfxHeapSizeAlign big = {};
	CHECK(rgfxHeapTextureBound(70, 2, 1920, 1080, 1, 1, 1, 1, &big));
	CHECK(big.size < 8486912ull * 5 / 4);

	RgfxHeapSizeAlign none = {};
	CHECK(!rgfxHeapTextureBound(70, 2, 0, 16, 1, 1, 1, 1, &none));
	CHECK(none.size == 0);
	CHECK(!rgfxHeapTextureBound(70, 2, 16, 16, 1, 0, 1, 1, &none));
	CHECK(!rgfxHeapTextureBound(125, 3, 1ull << 40, 1ull << 40, 1, 1, 1ull << 20, 1, &none));
	CHECK(none.size == 0);
	// A format the CPU copies hold no block of still has one: depth, stencil, PVRTC, any other.
	CHECK(rgfxHeapFormatBlock(260).bytes == 8 && rgfxHeapFormatBlock(253).bytes == 1);
	CHECK(rgfxHeapFormatBlock(160).width == 8 && rgfxHeapFormatBlock(0xffff).bytes == 16);
}

// The IOSurfaceAllocSize the device answers for a texture (copyIOSurfaceSharedTextureProperties:):
// its linear rows of blocks at the host's alignment, through every level, slice and depth image.
// A 64 x 64 BGRA8 texture and a two-element array of them take what both x86 Macs answer (16384
// and 32768: oracle_bundle props2).
void test_surface_texture_alloc_sizes()
{
	struct Shape {
		uint32_t format, type;
		uint64_t w, h, d;
		uint32_t levels;
		uint64_t array, samples, align, size;
	};
	const Shape shapes[] = {
	    {80, 2, 64, 64, 1, 1, 1, 1, 256, 16384},   // BGRA8Unorm, as both x86 Macs
	    {80, 3, 64, 64, 1, 1, 2, 1, 256, 32768},   // an array of two, as both x86 Macs
	    {80, 2, 100, 30, 1, 1, 1, 1, 256, 15360},  // rows of 400 bytes at 512
	    {80, 2, 100, 30, 1, 1, 1, 1, 16, 12000},   // at 16, rows of 400
	    {80, 2, 1, 1, 1, 1, 1, 1, 256, 256},       // one texel, one row
	    {10, 2, 17, 9, 1, 1, 1, 1, 256, 2304},     // R8Unorm
	    {130, 2, 64, 64, 1, 1, 1, 1, 256, 4096},   // BC1: 16 rows of 16 blocks of 8 bytes
	    {80, 2, 64, 64, 1, 7, 1, 1, 256, 32512},   // mipmapped: 127 rows of 256
	    {10, 7, 16, 16, 4, 1, 1, 1, 256, 16384},   // R8Unorm 3D: four images
	    {10, 7, 16, 16, 4, 2, 1, 1, 256, 20480},   // and a second level, of two images of 8 rows
	    {70, 5, 16, 16, 1, 1, 1, 1, 256, 24576},   // RGBA8Unorm cube: six faces
	    {252, 4, 32, 32, 1, 1, 1, 4, 256, 32768},  // Depth32Float, four samples
	};
	for (const Shape &t : shapes) {
		uint64_t size = 1;
		CHECK(rgfxSurfaceTextureAllocSize(t.format, t.type, t.w, t.h, t.d, t.levels, t.array,
		                                  t.samples, t.align, &size));
		CHECK(size == t.size);
	}
	uint64_t none = 1;
	CHECK(!rgfxSurfaceTextureAllocSize(80, 2, 0, 16, 1, 1, 1, 1, 256, &none) && none == 0);
	CHECK(!rgfxSurfaceTextureAllocSize(80, 2, 16, 16, 1, 0, 1, 1, 256, &none));
	CHECK(!rgfxSurfaceTextureAllocSize(80, 2, 16, 16, 1, 1, 1, 1, 0, &none));
	CHECK(!rgfxSurfaceTextureAllocSize(125, 3, 1ull << 40, 1ull << 40, 1, 1, 1ull << 20, 1, 256,
	                                   &none));
	CHECK(none == 0);
}

} // namespace

void rgfxProtocolTests()
{
	test_inline_bytes();
	test_inline_bytes_without_region();
	test_render_barriers();
	test_compute_dispatch_type();
	test_blit_copy_texture_to_texture();
	test_render_tile_size();
	test_trace_ids();
	test_unrecognized_selectors_logged();
	test_framework_members();
	test_sample_positions();
	test_visibility_result_buffer();
	test_color_attachment_mapping_pass();
	test_encoders_unsupported();
	test_command_buffer_and_queue_members();
	test_texture_members();
	test_buffer_members();
	test_heap_members();
	test_pipeline_state_members();
	test_library_members();
	test_argument_encoder_members();
	test_argument_encoder_query();
	test_argument_encoders_of_bindings();
	test_sampler_resource_ids();
	test_shared_event_wait();
	test_heap_sizes();
	test_surface_texture_alloc_sizes();
	test_queue_barriers();
}
