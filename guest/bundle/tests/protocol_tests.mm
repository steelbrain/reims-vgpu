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

#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#import "RgfxArgumentEncoder.h"
#import "RgfxBuffer.h"
#import "RgfxCommandObjects.h"
#import "RgfxComputePipelineState.h"
#import "RgfxDepthStencilState.h"
#import "RgfxHeap.h"
#import "RgfxLibrary.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxResource.h"
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

	QueueFixture()
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
		queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                 submitOffset:0
		                               completeOffset:0
		                             completeDataSize:0
		                                          ids:&ids
		                                     maxCount:0
		                                       device:nil
		                                   serializer:&ser];
		CHECK(queue != nil);
		queue.replyReader = &reader;
		queue.region = &region;
		(void)expectMessage<RgfxQueueCreate>(control, RGFX_CMD_QUEUE_CREATE);
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
}

// Without a region the bytes have nowhere to go: no binding is made (and it is logged), and the
// command buffer fails with MTLCommandBufferErrorOutOfMemory.
void test_inline_bytes_without_region()
{
	QueueFixture fx;
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
}

// The render barriers: the scope barrier as given; a resource barrier as the scope of the
// resources' kinds; the texture barrier as the render targets' fragment barrier.
void test_render_barriers()
{
	QueueFixture fx;
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
}

// A compute pass opened with a dispatch type carries it, and the encoder answers it; one
// opened without answers Serial.
void test_compute_dispatch_type()
{
	QueueFixture fx;
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
}

// copyFromTexture:toTexture: copies as many whole surfaces as both have: lined up at the
// level of one that is the other's first level's size, the levels both have from there, and
// the slices both have (six faces a cube); textures no level of which lines up copy nothing.
void test_blit_copy_texture_to_texture()
{
	QueueFixture fx;
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
}

// The tile size an encoder answers: its pass descriptor's; the device's own choice, which is
// not on the wire, answers 0 and is logged.
void test_render_tile_size()
{
	QueueFixture fx;
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
}

// The members of features the device answers NO for do nothing on the wire: a nil or default
// argument is exact and not logged; any other use is logged once.
void test_encoders_unsupported()
{
	QueueFixture fx;
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[renc setDepthStoreActionOptions:MTLStoreActionOptionNone];
#pragma clang diagnostic pop
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setVertexAccelerationStructure:atBufferIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setFragmentVisibleFunctionTable:atBufferIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setObjectBuffer:offset:atIndex:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setDepthTestMinBound:maxBound:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setColorAttachmentMap:)));
	CHECK(!RgfxUnsupportedWasLogged(r, @selector(setDepthStoreActionOptions:)));
	[renc setDepthTestMinBound:0.25f maxBound:0.5f];
	[renc drawMeshThreads:MTLSizeMake(1, 1, 1)
	    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
	      threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
	[renc setMeshBytes:&r length:sizeof(r) atIndex:0];
	[renc barrierAfterQueueStages:MTLStageVertex beforeStages:MTLStageFragment];
	CHECK(RgfxUnsupportedWasLogged(r, @selector(setDepthTestMinBound:maxBound:)));
	CHECK(RgfxUnsupportedWasLogged(
	    r, @selector(drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:)));
	CHECK(RgfxUnsupportedWasLogged(r, @selector(setMeshBytes:length:atIndex:)));
	CHECK(RgfxUnsupportedWasLogged(r, @selector(barrierAfterQueueStages:beforeStages:)));
	[renc endEncoding];
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	[cenc setAccelerationStructure:nil atBufferIndex:1];
	id<MTLVisibleFunctionTable> noTables[1] = {nil};
	[cenc setVisibleFunctionTables:noTables withBufferRange:NSMakeRange(0, 0)];
	CHECK(!RgfxUnsupportedWasLogged([RgfxComputeCommandEncoder class],
	                                @selector(setAccelerationStructure:atBufferIndex:)));
	[cenc barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch];
	CHECK(RgfxUnsupportedWasLogged([RgfxComputeCommandEncoder class],
	                               @selector(barrierAfterQueueStages:beforeStages:)));
	[cenc endEncoding];
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	[benc barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageBlit];
	CHECK(RgfxUnsupportedWasLogged([RgfxBlitCommandEncoder class],
	                               @selector(barrierAfterQueueStages:beforeStages:)));
	[benc endEncoding];
	[cb commit];
	uint64_t seqno = 0;
	std::vector<Op> ops = fx.nextOps(&seqno);
	// Only the framing and nothing else reached the wire.
	for (const Op &op : ops) {
		CHECK(op.tag == RGFX_OP_ENCODER_BEGIN_RENDER || op.tag == RGFX_OP_ENCODER_BEGIN_COMPUTE ||
		      op.tag == RGFX_OP_ENCODER_BEGIN_BLIT || op.tag == RGFX_OP_ENCODER_END);
	}
	fx.complete(cb, seqno);
}


// --- the command buffer and the queue ---------------------------------------------

// The encoders of features the device does not support are nil, logged; residency sets
// (which the device does not make) change nothing; a capture boundary marks nothing.
void test_command_buffer_and_queue_members()
{
	QueueFixture fx;
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
}

// A heap's allocated size is its size; acceleration structures and the automatic placements
// (no host size and alignment on the wire) are nil, logged.
void test_heap_members()
{
	TestRing control(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter writer(control.base(), control.dataSize, RGFX_CONTROL_RING_ID,
	                      TestRing::doorbell, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
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
	CHECK(RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newBufferWithLength:options:)));
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
	CHECK(ds.gpuResourceID._impl == 0);
	CHECK(RgfxUnsupportedWasLogged([RgfxDepthStencilState class], @selector(gpuResourceID)));
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
		CHECK(called && handlerErr.code == MTLLibraryErrorUnsupported);
		CHECK([lib reflectionForFunctionWithName:@"vmain"] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxLibrary class],
		                               @selector(reflectionForFunctionWithName:)));
	} // @autoreleasepool: the library drains while ser lives
}

// An argument encoder lays out nothing (its encoded length is 0) and writes nothing; each
// member that is asked something logs it, a nil argument does not.
void test_argument_encoder_members()
{
	MTLArgumentDescriptor *a = [MTLArgumentDescriptor argumentDescriptor];
	a.dataType = MTLDataTypePointer;
	a.index = 0;
	RgfxArgumentEncoder *enc = [RgfxArgumentEncoder argumentEncoderWithArguments:@[ a ]
	                                                                      device:nil
	                                                                  serializer:nullptr];
	Class c = [RgfxArgumentEncoder class];
	[enc setBuffer:nil offset:0 atIndex:0];
	CHECK(!RgfxUnsupportedWasLogged(c, @selector(setBuffer:offset:atIndex:)));
	CHECK(enc.encodedLength == 0);
	CHECK(RgfxUnsupportedWasLogged(c, @selector(encodedLength)));
	id<MTLTexture> none[1] = {nil};
	[enc setTextures:none withRange:NSMakeRange(0, 1)];
	CHECK(RgfxUnsupportedWasLogged(c, @selector(setTextures:withRange:)));
	CHECK([enc newArgumentEncoderForBufferAtIndex:0] == nil);
	CHECK([enc constantDataAtIndex:0] == nullptr);
	[enc setIndirectCommandBuffer:nil atIndex:0];
	CHECK(!RgfxUnsupportedWasLogged(c, @selector(setIndirectCommandBuffer:atIndex:)));
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

} // namespace

void rgfxProtocolTests()
{
	test_inline_bytes();
	test_inline_bytes_without_region();
	test_render_barriers();
	test_compute_dispatch_type();
	test_blit_copy_texture_to_texture();
	test_render_tile_size();
	test_encoders_unsupported();
	test_command_buffer_and_queue_members();
	test_texture_members();
	test_buffer_members();
	test_heap_members();
	test_pipeline_state_members();
	test_library_members();
	test_argument_encoder_members();
	test_shared_event_wait();
	test_heap_sizes();
}
