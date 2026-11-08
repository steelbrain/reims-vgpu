// The indirect command buffer's checks (tests/shell_tests.mm's main calls rgfxIcbTests()): what
// the buffer and its commands answer, the records each encode sends and when they go. Each shape
// is one oracle_icb measured (guest/bundle/tools/oracle/oracle_icb.m) on the AMD 5300M and the
// Intel 630 (macOS 26.7.1) and the M3 Max (macOS 27.0.1); where they differ, the lane's choice,
// as RgfxIndirectCommandBuffer.h lists.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <objc/message.h>

#include <cstdint>
#include <cstring>
#include <vector>

#import "RgfxBuffer.h"
#import "RgfxCommandObjects.h"
#import "RgfxIndirectCommandBuffer.h"
#import "RgfxUnsupported.h"

#include "RgfxIcbSlotTable.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing

void rgfxIcbTests();
bool ringEmpty(TestRing &ring);
void drainRing(TestRing &ring);

namespace {

// A ring, an id space and a serialiser, never freed (an object a test lets go may post its
// destroy after the test), with a fake host that answers each ICB_CREATE with `id` and counts
// the creates, refusing them where `refuse` is set.
struct Fixture {
	TestRing &ring;
	RgfxRingWriter &writer;
	RgfxIdSpace &ids;
	RgfxSerializer &ser;
	uint64_t id = 7;
	bool refuse = false;
	RgfxIcbCreate last = {};

	explicit Fixture(uint32_t ringBytes = 0x8000, bool answers = true)
	    : ring(*new TestRing(ringBytes)),
	      writer(*new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID,
	                                 TestRing::doorbell, nullptr)),
	      ids(*new RgfxIdSpace()), ser(*new RgfxSerializer(writer, ids))
	{
		ring.setConsumerIdle(1);
		if (answers)
			ser.setAsker(&Fixture::ask, this);
	}

	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		Fixture *self = static_cast<Fixture *>(context);
		CHECK(size == sizeof(RgfxIcbCreate) && capacity >= sizeof(RgfxIcbState));
		std::memcpy(&self->last, message, sizeof(self->last));
		CHECK(self->last.header.type == RGFX_CMD_ICB_CREATE);
		RgfxIcbState answer = {};
		answer.header = {RGFX_EVT_ICB_STATE, sizeof(answer), 1};
		answer.status = self->refuse ? RGFX_STATUS_REFUSED : RGFX_STATUS_OK;
		answer.gpu_resource_id = self->refuse ? 0 : self->id;
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}

	RgfxIndirectCommandBuffer *make(MTLIndirectCommandBufferDescriptor *d, NSUInteger count,
	                                MTLResourceOptions options = 0)
	{
		return [RgfxIndirectCommandBuffer icbWithDescriptor:d
		                                    maxCommandCount:count
		                                            options:options
		                                             device:nil
		                                         serializer:&ser];
	}

	// As make, its CPU encodes as slots in blocks of `region` where its shape takes them.
	RgfxIndirectCommandBuffer *makeSlots(MTLIndirectCommandBufferDescriptor *d, NSUInteger count,
	                                     RgfxRegion &region)
	{
		return [RgfxIndirectCommandBuffer icbWithDescriptor:d
		                                    maxCommandCount:count
		                                            options:0
		                                             device:nil
		                                         serializer:&ser
		                                        slotsRegion:&region];
	}
};

// A region whose blocks a stand-in kext places from the floor, its bytes 0xee until a block's
// owner writes them, and the blocks live.
struct SlotRegion {
	std::vector<uint8_t> mem;
	RgfxRegion region;
	uint64_t next = RGFX_ALLOCATIONS_OFFSET;
	std::vector<uint64_t> live;

	explicit SlotRegion(uint64_t bytes)
	    : mem(RGFX_ALLOCATIONS_OFFSET + bytes, 0xee), region(mem.data(), mem.size())
	{
		region.setBlockSource(&SlotRegion::alloc, &SlotRegion::release, this);
	}
	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		SlotRegion *r = static_cast<SlotRegion *>(c);
		const uint64_t at = (r->next + align - 1) & ~(align - 1);
		if (at + size > r->mem.size())
			return false;
		r->next = at + size;
		r->live.push_back(at);
		*offset = at;
		return true;
	}
	static void release(void *c, uint64_t offset)
	{
		std::vector<uint64_t> &live = static_cast<SlotRegion *>(c)->live;
		for (auto it = live.begin(); it != live.end(); ++it) {
			if (*it == offset) {
				live.erase(it);
				return;
			}
		}
		CHECK(false); // a block released twice, or never placed
	}
	// The `T` at region offset `offset`.
	template <typename T> T at(uint64_t offset) const
	{
		T v;
		std::memcpy(&v, mem.data() + offset, sizeof(v));
		return v;
	}
};

MTLIndirectCommandBufferDescriptor *renderDescriptor()
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed;
	d.maxVertexBufferBindCount = 2;
	d.maxFragmentBufferBindCount = 1;
	return d;
}

MTLIndirectCommandBufferDescriptor *computeDescriptor()
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
	d.maxKernelBufferBindCount = 2;
	return d;
}

RgfxBuffer *buffer(uint64_t objId)
{
	return [[RgfxBuffer alloc] initWithObjId:objId
	                              gpuAddress:0x100000 * objId
	                                  length:256
	                                 options:MTLResourceStorageModeShared
	                                  device:nil
	                              serializer:nullptr];
}

// One record read back off an ICB_WRITE.
struct Record {
	uint16_t op;
	uint32_t index;
	std::vector<uint8_t> body;

	template <typename T> T as() const
	{
		CHECK(body.size() == sizeof(T));
		T v;
		std::memcpy(&v, body.data(), sizeof(v));
		return v;
	}
};

// The next message on `ring`, which must be an ICB_WRITE into `icb`: its records, in order.
std::vector<Record> nextWrite(TestRing &ring, RgfxObjId icb, uint32_t *size = nullptr)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_ICB_WRITE && bytes.size() >= sizeof(RgfxIcbWrite));
	if (size != nullptr)
		*size = h.size;
	RgfxIcbWrite head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == icb);
	std::vector<Record> out;
	size_t at = sizeof(RgfxIcbWrite);
	while (at < bytes.size()) {
		RgfxIcbRecord r;
		CHECK(at + sizeof(r) <= bytes.size());
		std::memcpy(&r, bytes.data() + at, sizeof(r));
		CHECK(r.size >= sizeof(r) && r.size % 8 == 0 && at + r.size <= bytes.size());
		out.push_back({r.op, r.command_index,
		               std::vector<uint8_t>(bytes.begin() + at + sizeof(r), bytes.begin() + at + r.size)});
		at += r.size;
	}
	return out;
}

// The next message on `ring`, which must be an ICB_SLOTS: its runs, in order.
std::vector<RgfxIcbSlotRun> nextSlots(TestRing &ring)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_ICB_SLOTS && bytes.size() >= sizeof(RgfxIcbSlots));
	RgfxIcbSlots head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(bytes.size() == sizeof(head) + head.run_count * sizeof(RgfxIcbSlotRun));
	std::vector<RgfxIcbSlotRun> runs(head.run_count);
	std::memcpy(runs.data(), bytes.data() + sizeof(head), runs.size() * sizeof(RgfxIcbSlotRun));
	CHECK(runs.empty() || head.obj_id == runs[0].icb_obj_id);
	return runs;
}

// --- the create ------------------------------------------------------------------------

// The create carries every field of the descriptor, and for the SPI members this Mac's fresh
// descriptor's answers, the ones measured on the AMD and the Intel too: 1 scissor rect, 1
// viewport, their states inherited, no override; and a fresh descriptor's 26 inherits (YES).
void test_create_carries_the_descriptor()
{
	MTLIndirectCommandBufferDescriptor *fresh = [MTLIndirectCommandBufferDescriptor new];
	RgfxIcbCreate m = RgfxIcbCreateOf(fresh, 4, 0);
	CHECK(m.max_command_count == 4 && m.options == 0 && m.command_types == 0);
	const uint32_t inherited =
	    RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE | RGFX_ICB_INHERIT_DEPTH_BIAS |
	    RGFX_ICB_INHERIT_DEPTH_CLIP_MODE | RGFX_ICB_INHERIT_CULL_MODE |
	    RGFX_ICB_INHERIT_FRONT_FACING_WINDING | RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE |
	    RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES | RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS |
	    RGFX_ICB_INHERIT_SCISSOR_RECTS | RGFX_ICB_INHERIT_VIEWPORTS | RGFX_ICB_INHERIT_BLEND_COLOR;
	CHECK(m.flags == inherited);
	CHECK(m.max_vertex_buffer_bind_count == 0 && m.max_fragment_buffer_bind_count == 0 &&
	      m.max_kernel_buffer_bind_count == 0 && m.max_kernel_threadgroup_memory_bind_count == 31);
	CHECK(m.max_scissor_rect_count == 1 && m.max_viewport_count == 1 &&
	      m.allow_override_render_states == 0);

	MTLIndirectCommandBufferDescriptor *d = renderDescriptor();
	d.commandTypes |= MTLIndirectCommandTypeConcurrentDispatch;
	d.inheritBuffers = YES;
	d.inheritPipelineState = YES;
	d.maxKernelBufferBindCount = 5;
	d.maxObjectBufferBindCount = 6;
	d.maxMeshBufferBindCount = 7;
	d.maxObjectThreadgroupMemoryBindCount = 8;
	d.supportDynamicAttributeStride = YES;
	d.inheritCullMode = NO;
	m = RgfxIcbCreateOf(d, 1ull << 33, MTLResourceStorageModePrivate);
	CHECK(m.max_command_count == 1ull << 33 && m.options == MTLResourceStorageModePrivate);
	CHECK(m.command_types == (MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed |
	                          MTLIndirectCommandTypeConcurrentDispatch));
	CHECK(m.flags == ((inherited & ~RGFX_ICB_INHERIT_CULL_MODE) | RGFX_ICB_INHERIT_BUFFERS |
	                  RGFX_ICB_INHERIT_PIPELINE_STATE | RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE));
	CHECK(m.max_vertex_buffer_bind_count == 2 && m.max_fragment_buffer_bind_count == 1 &&
	      m.max_kernel_buffer_bind_count == 5 && m.max_object_buffer_bind_count == 6 &&
	      m.max_mesh_buffer_bind_count == 7 && m.max_object_threadgroup_memory_bind_count == 8);

	// Through the serialiser: the create goes to the host whole, with its picked id.
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(d, 16, MTLResourceStorageModePrivate);
		CHECK(icb != nil);
		CHECK(fx.last.obj_id == icb.rgfxObjId && fx.last.max_command_count == 16 &&
		      fx.last.command_types == m.command_types && fx.last.flags == m.flags &&
		      fx.last.max_kernel_buffer_bind_count == 5);
		CHECK(ringEmpty(fx.ring)); // asked, not posted
	}
	drainRing(fx.ring);
}

// The buffer answers the host's id, its size and options, and what every Mac's answers: no heap,
// no label, not aliasable, a Default hazard mode as Tracked, the purgeable states in turn; and 64
// bytes a command in whole pages. A refused create, and one with no descriptor, are nil.
void test_buffer_answers()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(renderDescriptor(), 16);
		CHECK(icb != nil && icb.size == 16 && icb.gpuResourceID._impl == 7);
		// Metal's buffer SPI: the index, the handle and the identifier are gpuResourceID's value; no
		// GPU address the guest sees.
		for (NSString *member in @[ @"resourceIndex", @"gpuHandle", @"uniqueIdentifier" ])
			CHECK(((uint64_t (*)(id, SEL))objc_msgSend)(icb, NSSelectorFromString(member)) == 7);
		CHECK(((uint64_t (*)(id, SEL))objc_msgSend)(icb, NSSelectorFromString(@"gpuAddress")) == 0);
		// No driver header of the buffer in the guest.
		void *header = &fx;
		uint64_t headerSize = 1;
		((void (*)(id, SEL, void **, uint64_t *))objc_msgSend)(
		    icb, NSSelectorFromString(@"getHeader:headerSize:"), &header, &headerSize);
		CHECK(header == nullptr && headerSize == 0);
		((void (*)(id, SEL, void **, uint64_t *))objc_msgSend)(
		    icb, NSSelectorFromString(@"getHeader:headerSize:"), nullptr, nullptr);
		// The resource SPI: an allocation of its own, aliasing nothing; complete, written and not
		// purgeable; its options as asked.
		auto u64 = [](id object, NSString *name) {
			return ((uint64_t (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
		};
		auto flag = [](id object, NSString *name) {
			return ((BOOL (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
		};
		CHECK(flag(icb, @"isComplete") && flag(icb, @"isWriteComplete") && !flag(icb, @"isPurgeable"));
		CHECK(u64(icb, @"unfilteredResourceOptions") == 0);
		((void (*)(id, SEL))objc_msgSend)(icb, NSSelectorFromString(@"waitUntilComplete"));
		CHECK(icb.resourceOptions == 0x200 && icb.storageMode == MTLStorageModeShared &&
		      icb.cpuCacheMode == MTLCPUCacheModeDefaultCache &&
		      icb.hazardTrackingMode == MTLHazardTrackingModeTracked);
		CHECK(icb.heap == nil && icb.heapOffset == 0 && icb.label == nil && !icb.isAliasable);
		CHECK(icb.allocatedSize == 4096);
		CHECK([icb setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateNonVolatile);
		CHECK([icb setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK([icb setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateVolatile);

		fx.id = 8;
		RgfxIndirectCommandBuffer *priv = fx.make(computeDescriptor(), 1024,
		                                          MTLResourceStorageModePrivate);
		CHECK(priv.gpuResourceID._impl == 8 && priv.resourceOptions == 0x220 &&
		      priv.storageMode == MTLStorageModePrivate && priv.allocatedSize == 65536);
		CHECK(u64(icb, @"allocationID") != 0 && u64(icb, @"allocationID") != u64(priv, @"allocationID"));
		CHECK(!((BOOL (*)(id, SEL, id))objc_msgSend)(icb, NSSelectorFromString(@"doesAliasResource:"), priv));
		CHECK(u64(priv, @"unfilteredResourceOptions") == MTLResourceStorageModePrivate);
		CHECK(fx.make(renderDescriptor(), 4, MTLResourceHazardTrackingModeUntracked)
		          .resourceOptions == 0x100);
		CHECK(fx.make(renderDescriptor(), 4, MTLResourceCPUCacheModeWriteCombined)
		          .resourceOptions == 0x201);
		CHECK(fx.make(renderDescriptor(), 0).allocatedSize == 4096);

		fx.refuse = true;
		CHECK(fx.make(renderDescriptor(), 4) == nil);
		fx.refuse = false;
		CHECK([RgfxIndirectCommandBuffer icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *_Nonnull)nil
		                                   maxCommandCount:4
		                                           options:0
		                                            device:nil
		                                        serializer:&fx.ser] == nil);
		CHECK([RgfxIndirectCommandBuffer icbWithDescriptor:renderDescriptor()
		                                   maxCommandCount:4
		                                           options:0
		                                            device:nil
		                                        serializer:nullptr] == nil);
	}
	drainRing(fx.ring);
}

// --- the commands -----------------------------------------------------------------------

// A command past the size, or of a kind the buffer's types lack, is nil (the Intel's), and each
// lookup a new object (the AMD's and the M3's).
void test_command_lookups()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *render = fx.make(renderDescriptor(), 4);
		RgfxIndirectCommandBuffer *compute = fx.make(computeDescriptor(), 4);
		MTLIndirectCommandBufferDescriptor *both = renderDescriptor();
		both.commandTypes |= MTLIndirectCommandTypeConcurrentDispatchThreads;
		RgfxIndirectCommandBuffer *mixed = fx.make(both, 4);
		MTLIndirectCommandBufferDescriptor *none = [MTLIndirectCommandBufferDescriptor new];
		RgfxIndirectCommandBuffer *empty = fx.make(none, 4);

		id<MTLIndirectRenderCommand> a = [render indirectRenderCommandAtIndex:3];
		id<MTLIndirectRenderCommand> b = [render indirectRenderCommandAtIndex:3];
		CHECK(a != nil && b != nil && a != b);
		CHECK([render indirectRenderCommandAtIndex:4] == nil);
		CHECK([render indirectComputeCommandAtIndex:0] == nil);
		CHECK([compute indirectComputeCommandAtIndex:3] != nil);
		CHECK([compute indirectComputeCommandAtIndex:4] == nil);
		CHECK([compute indirectRenderCommandAtIndex:0] == nil);
		CHECK([mixed indirectRenderCommandAtIndex:0] != nil &&
		      [mixed indirectComputeCommandAtIndex:0] != nil);
		CHECK([empty indirectRenderCommandAtIndex:0] == nil &&
		      [empty indirectComputeCommandAtIndex:0] == nil);
		CHECK(ringEmpty(fx.ring)); // a lookup sends nothing

		// Metal's command SPI: what reads back what a command was given answers a fresh
		// command's, logged; what Metal's base render command raises for raises; nothing is sent.
		auto obj = [](id object, NSString *name) {
			return ((id (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
		};
		auto u64 = [](id object, NSString *name) {
			return ((uint64_t (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
		};
		auto at = [](id object, NSString *name) {
			return ((uint64_t (*)(id, SEL, NSUInteger))objc_msgSend)(object, NSSelectorFromString(name), 0);
		};
		auto flag = [](id object, NSString *name) {
			return ((BOOL (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(name));
		};
		for (NSString *member in @[ @"drawArguments", @"drawIndexedArguments", @"drawPatchesArguments",
		                            @"drawIndexedPatchesArguments", @"getTessellationFactorArguments" ])
			CHECK(obj(a, member) == nil);
		CHECK(u64(a, @"getCommandType") == 0 && u64(a, @"getOptimizedStatus") == 0 &&
		      u64(a, @"getPipelineStateUniqueIdentifier") == 0 && !flag(a, @"hasBarrier"));
		for (NSString *member in @[ @"getVertexBufferAtIndex:", @"getVertexBufferAddressAtIndex:",
		                            @"getVertexAttributeStrideAtIndex:", @"getFragmentBufferAtIndex:",
		                            @"getFragmentBufferAddressAtIndex:" ])
			CHECK(at(a, member) == 0);
		CHECK(RgfxUnsupportedWasLogged([RgfxIndirectRenderCommand class], NSSelectorFromString(@"getCommandType")));
		auto raises = [](void (^body)(void)) {
			@try {
				body();
			} @catch (NSException *e) {
				return [e.reason containsString:@"unrecognized selector"];
			}
			return false;
		};
		CHECK(raises(^{
			(void)u64(a, @"getCullMode");
		}));
		CHECK(raises(^{
			(void)obj(a, @"getViewports");
		}));
		CHECK(raises(^{
			(void)at(a, @"getMeshBufferAddressAtIndex:");
		}));
		CHECK(raises(^{
			((void (*)(id, SEL, MTLScissorRect))objc_msgSend)(a, NSSelectorFromString(@"setScissorRect:"),
			                                                  MTLScissorRect{0, 0, 1, 1});
		}));
		id<MTLIndirectComputeCommand> k = [compute indirectComputeCommandAtIndex:0];
		for (NSString *member in @[ @"dispatchThreadgroupsArguments", @"dispatchThreadsArguments",
		                            @"getImageBlockSize" ])
			CHECK(obj(k, member) == nil);
		CHECK(u64(k, @"getCommandType") == 0 && u64(k, @"getPipelineStateUniqueIdentifier") == 0 &&
		      !flag(k, @"hasBarrier"));
		for (NSString *member in @[ @"getKernelBufferAtIndex:", @"getKernelBufferAddressAtIndex:",
		                            @"getKernelAttributeStrideAtIndex:", @"getThreadgroupMemoryLengthAtIndex:" ])
			CHECK(at(k, member) == 0);
		CHECK(RgfxUnsupportedWasLogged([RgfxIndirectComputeCommand class], NSSelectorFromString(@"getCommandType")));
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// Each render setter and draw is one record on its command, in call order, with every argument;
// nothing goes until a flush, and then one message holds them.
void test_render_records()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(renderDescriptor(), 4);
		RgfxBuffer *vb = buffer(0x31), *fb = buffer(0x32), *ib = buffer(0x33), *tf = buffer(0x34);
		id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:2];
		[c setRenderPipelineState:(id<MTLRenderPipelineState> _Nonnull)nil];
		[c setVertexBuffer:vb offset:16 atIndex:1];
		[c setVertexBuffer:vb offset:32 attributeStride:12 atIndex:0];
		[c setFragmentBuffer:fb offset:8 atIndex:0];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip
		      vertexStart:1
		      vertexCount:4
		    instanceCount:2
		     baseInstance:3];
		[c drawIndexedPrimitives:MTLPrimitiveTypeTriangle
		              indexCount:6
		               indexType:MTLIndexTypeUInt32
		             indexBuffer:ib
		       indexBufferOffset:4
		           instanceCount:1
		              baseVertex:-4
		            baseInstance:5];
		[c drawPatches:4
		                                patchStart:1
		                                patchCount:2
		                          patchIndexBuffer:nil
		                    patchIndexBufferOffset:0
		                             instanceCount:1
		                              baseInstance:0
		                  tessellationFactorBuffer:tf
		            tessellationFactorBufferOffset:64
		    tessellationFactorBufferInstanceStride:12];
		[c drawIndexedPatches:3
		                                patchStart:0
		                                patchCount:1
		                          patchIndexBuffer:ib
		                    patchIndexBufferOffset:8
		                   controlPointIndexBuffer:ib
		             controlPointIndexBufferOffset:24
		                             instanceCount:2
		                              baseInstance:1
		                  tessellationFactorBuffer:tf
		            tessellationFactorBufferOffset:0
		    tessellationFactorBufferInstanceStride:0];
		[c setBarrier];
		[c clearBarrier];
		[[icb indirectRenderCommandAtIndex:0] reset];
		CHECK(ringEmpty(fx.ring));

		const RgfxIcbFlush flush = fx.ser.flushIcbWrites();
		CHECK(flush.messages == 1);
		uint32_t size = 0;
		std::vector<Record> r = nextWrite(fx.ring, icb.rgfxObjId, &size);
		CHECK(flush.bytes == size);
		CHECK(r.size() == 11);
		for (size_t i = 0; i < 10; i++)
			CHECK(r[i].index == 2);
		CHECK(r[0].op == RGFX_ICB_OP_RENDER_PIPELINE && r[0].as<RgfxIcbObject>().obj_id == 0);
		const RgfxIcbBuffer v = r[1].as<RgfxIcbBuffer>();
		CHECK(r[1].op == RGFX_ICB_OP_VERTEX_BUFFER && v.obj_id == 0x31 && v.offset == 16 &&
		      v.index == 1);
		const RgfxIcbBufferStride s = r[2].as<RgfxIcbBufferStride>();
		CHECK(r[2].op == RGFX_ICB_OP_VERTEX_BUFFER_STRIDE && s.obj_id == 0x31 && s.offset == 32 &&
		      s.stride == 12 && s.index == 0);
		const RgfxIcbBuffer f = r[3].as<RgfxIcbBuffer>();
		CHECK(r[3].op == RGFX_ICB_OP_FRAGMENT_BUFFER && f.obj_id == 0x32 && f.offset == 8);
		const RgfxIcbDraw d = r[4].as<RgfxIcbDraw>();
		CHECK(r[4].op == RGFX_ICB_OP_DRAW && d.primitive_type == MTLPrimitiveTypeTriangleStrip &&
		      d.vertex_start == 1 && d.vertex_count == 4 && d.instance_count == 2 &&
		      d.base_instance == 3);
		const RgfxIcbDrawIndexed di = r[5].as<RgfxIcbDrawIndexed>();
		CHECK(r[5].op == RGFX_ICB_OP_DRAW_INDEXED && di.index_count == 6 &&
		      di.index_type == MTLIndexTypeUInt32 && di.index_buffer_obj_id == 0x33 &&
		      di.index_buffer_offset == 4 && di.base_vertex == -4 && di.base_instance == 5);
		const RgfxIcbDrawPatches p = r[6].as<RgfxIcbDrawPatches>();
		CHECK(r[6].op == RGFX_ICB_OP_DRAW_PATCHES && p.number_of_patch_control_points == 4 &&
		      p.patch_start == 1 && p.patch_count == 2 && p.patch_index_buffer_obj_id == 0 &&
		      p.tessellation_factor_buffer_obj_id == 0x34 &&
		      p.tessellation_factor_buffer_offset == 64 && p.instance_stride == 12);
		const RgfxIcbDrawIndexedPatches ip = r[7].as<RgfxIcbDrawIndexedPatches>();
		CHECK(r[7].op == RGFX_ICB_OP_DRAW_INDEXED_PATCHES && ip.patches.patch_index_buffer_obj_id == 0x33 &&
		      ip.patches.patch_index_buffer_offset == 8 && ip.control_point_index_buffer_obj_id == 0x33 &&
		      ip.control_point_index_buffer_offset == 24 && ip.patches.instance_count == 2);
		CHECK(r[8].op == RGFX_ICB_OP_RENDER_SET_BARRIER && r[8].body.empty());
		CHECK(r[9].op == RGFX_ICB_OP_RENDER_CLEAR_BARRIER && r[9].body.empty());
		CHECK(r[10].op == RGFX_ICB_OP_RENDER_RESET && r[10].index == 0 && r[10].body.empty());

		// Nothing waits now: a flush posts nothing.
		CHECK(fx.ser.flushIcbWrites().messages == 0 && ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// Each compute setter and dispatch is one record, as the render ones.
void test_compute_records()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(computeDescriptor(), 4);
		RgfxBuffer *kb = buffer(0x41);
		id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:1];
		[c setComputePipelineState:(id<MTLComputePipelineState> _Nonnull)nil];
		[c setKernelBuffer:kb offset:64 atIndex:1];
		[c setKernelBuffer:kb offset:0 attributeStride:16 atIndex:0];
		[c setThreadgroupMemoryLength:256 atIndex:0];
		[c setStageInRegion:MTLRegionMake3D(1, 2, 3, 4, 5, 6)];
		[c setImageblockWidth:8 height:16];
		[c concurrentDispatchThreadgroups:MTLSizeMake(2, 3, 4) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[c concurrentDispatchThreads:MTLSizeMake(100, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
		[c setBarrier];
		[c clearBarrier];
		[c reset];
		fx.ser.flushIcbWrites();
		std::vector<Record> r = nextWrite(fx.ring, icb.rgfxObjId);
		CHECK(r.size() == 11);
		for (const Record &x : r)
			CHECK(x.index == 1);
		CHECK(r[0].op == RGFX_ICB_OP_COMPUTE_PIPELINE);
		const RgfxIcbBuffer k = r[1].as<RgfxIcbBuffer>();
		CHECK(r[1].op == RGFX_ICB_OP_KERNEL_BUFFER && k.obj_id == 0x41 && k.offset == 64 &&
		      k.index == 1);
		const RgfxIcbBufferStride ks = r[2].as<RgfxIcbBufferStride>();
		CHECK(r[2].op == RGFX_ICB_OP_KERNEL_BUFFER_STRIDE && ks.stride == 16 && ks.index == 0);
		const RgfxIcbThreadgroupMemory tg = r[3].as<RgfxIcbThreadgroupMemory>();
		CHECK(r[3].op == RGFX_ICB_OP_THREADGROUP_MEMORY && tg.length == 256 && tg.index == 0);
		const RgfxIcbRegion rg = r[4].as<RgfxIcbRegion>();
		CHECK(r[4].op == RGFX_ICB_OP_STAGE_IN_REGION && rg.origin[0] == 1 && rg.origin[2] == 3 &&
		      rg.size[0] == 4 && rg.size[2] == 6);
		const RgfxIcbImageblock ib = r[5].as<RgfxIcbImageblock>();
		CHECK(r[5].op == RGFX_ICB_OP_IMAGEBLOCK && ib.width == 8 && ib.height == 16);
		const RgfxIcbDispatch tgs = r[6].as<RgfxIcbDispatch>();
		CHECK(r[6].op == RGFX_ICB_OP_DISPATCH_THREADGROUPS && tgs.grid[0] == 2 && tgs.grid[1] == 3 &&
		      tgs.grid[2] == 4 && tgs.threadgroup[0] == 64);
		const RgfxIcbDispatch th = r[7].as<RgfxIcbDispatch>();
		CHECK(r[7].op == RGFX_ICB_OP_DISPATCH_THREADS && th.grid[0] == 100 && th.threadgroup[0] == 32);
		CHECK(r[8].op == RGFX_ICB_OP_COMPUTE_SET_BARRIER && r[9].op == RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER &&
		      r[10].op == RGFX_ICB_OP_COMPUTE_RESET);
	}
	drainRing(fx.ring);
}

// The macOS 26 render-state setters raise an unrecognized selector, though the command answers
// respondsToSelector: YES for them, as on the AMD and the Intel; the mesh setters do nothing and
// log it. Neither sends a record.
void test_setters_the_lane_does_not_carry()
{
	Fixture fx;
	@autoreleasepool {
		MTLIndirectCommandBufferDescriptor *d = renderDescriptor();
		d.commandTypes |= MTLIndirectCommandTypeDrawMeshThreadgroups;
		RgfxIndirectCommandBuffer *icb = fx.make(d, 2);
		id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:0];
		const SEL raise[] = {@selector(setDepthStencilState:), @selector(setDepthClipMode:),
		                     @selector(setCullMode:), @selector(setFrontFacingWinding:),
		                     @selector(setTriangleFillMode:)};
		for (SEL sel : raise) {
			CHECK([c respondsToSelector:sel]);
			bool raised = false;
			@try {
				((void (*)(id, SEL, NSUInteger))objc_msgSend)(c, sel, 0);
			} @catch (NSException *e) {
				raised = [e.name isEqualToString:NSInvalidArgumentException];
			}
			CHECK(raised);
		}
		CHECK([c respondsToSelector:@selector(setDepthBias:slopeScale:clamp:)]);
		bool raised = false;
		@try {
			[c setDepthBias:1 slopeScale:1 clamp:0];
		} @catch (NSException *e) {
			raised = [e.name isEqualToString:NSInvalidArgumentException];
		}
		CHECK(raised);

		[c setObjectBuffer:buffer(0x51) offset:0 atIndex:0];
		[c setMeshBuffer:buffer(0x52) offset:0 atIndex:0];
		[c setObjectThreadgroupMemoryLength:64 atIndex:0];
		[c drawMeshThreadgroups:MTLSizeMake(1, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
		[c drawMeshThreads:MTLSizeMake(1, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
		CHECK(RgfxUnsupportedWasLogged([RgfxIndirectRenderCommand class],
		                               @selector(drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:)));
		CHECK(fx.ser.flushIcbWrites().messages == 0 && ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// --- when the records go -------------------------------------------------------------------

// Records of two buffers go in two messages, each with its own in call order; a destroy posts
// the waiting records before it, as one may name the object it drops.
void test_records_go_before_a_destroy()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *a = fx.make(renderDescriptor(), 4);
		RgfxIndirectCommandBuffer *b = fx.make(computeDescriptor(), 4);
		RgfxBuffer *vb = buffer(0x61);
		[[a indirectRenderCommandAtIndex:0] setVertexBuffer:vb offset:0 atIndex:0];
		[[b indirectComputeCommandAtIndex:1] setKernelBuffer:vb offset:0 atIndex:0];
		[[a indirectRenderCommandAtIndex:1] drawPrimitives:MTLPrimitiveTypePoint
		                                       vertexStart:0
		                                       vertexCount:1
		                                     instanceCount:1
		                                      baseInstance:0];
		CHECK(ringEmpty(fx.ring));
		CHECK(fx.ser.destroy(0x61));
		std::vector<Record> ra = nextWrite(fx.ring, a.rgfxObjId);
		CHECK(ra.size() == 2 && ra[0].op == RGFX_ICB_OP_VERTEX_BUFFER && ra[0].index == 0 &&
		      ra[1].op == RGFX_ICB_OP_DRAW && ra[1].index == 1);
		std::vector<Record> rb = nextWrite(fx.ring, b.rgfxObjId);
		CHECK(rb.size() == 1 && rb[0].op == RGFX_ICB_OP_KERNEL_BUFFER && rb[0].index == 1);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(fx.ring.next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY);
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// A buffer's message that would pass the largest the control ring takes (half a ring's bytes)
// goes at once, and its records go on in a new one: none is lost or reordered.
void test_a_full_message_goes_at_once()
{
	Fixture fx(0x1000); // messages of at most 2 KiB
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(renderDescriptor(), 1000);
		// Each draw is a 32-byte record: 63 fit after the 24-byte message header.
		for (uint32_t i = 0; i < 100; i++) {
			[[icb indirectRenderCommandAtIndex:i] drawPrimitives:MTLPrimitiveTypePoint
			                                         vertexStart:i
			                                         vertexCount:1
			                                       instanceCount:1
			                                        baseInstance:0];
		}
		uint32_t size = 0;
		std::vector<Record> first = nextWrite(fx.ring, icb.rgfxObjId, &size);
		CHECK(first.size() == 63 && size <= 0x800);
		CHECK(ringEmpty(fx.ring));
		fx.ser.flushIcbWrites();
		std::vector<Record> rest = nextWrite(fx.ring, icb.rgfxObjId);
		CHECK(rest.size() == 37);
		for (uint32_t i = 0; i < 100; i++) {
			const Record &r = i < 63 ? first[i] : rest[i - 63];
			CHECK(r.index == i && r.as<RgfxIcbDraw>().vertex_start == i);
		}
	}
	drainRing(fx.ring);
}

// A command buffer's commit posts the waiting records on the control ring before the buffer
// goes on its queue's ring, so it sees every encode made before it.
void test_a_commit_posts_the_records_first()
{
	Fixture fx;
	TestRing &submit = *new TestRing(0x8000);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	submit.setConsumerIdle(1);
	@autoreleasepool {
		RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                                   submitOffset:0
		                                                 completeOffset:0
		                                               completeDataSize:0
		                                                            ids:&fx.ids
		                                                       maxCount:0
		                                                         device:nil
		                                                     serializer:&fx.ser];
		drainRing(fx.ring); // the queue's create
		RgfxIndirectCommandBuffer *icb = fx.make(renderDescriptor(), 4);
		[[icb indirectRenderCommandAtIndex:0] drawPrimitives:MTLPrimitiveTypePoint
		                                         vertexStart:0
		                                         vertexCount:1
		                                       instanceCount:1
		                                        baseInstance:0];
		CHECK(ringEmpty(fx.ring));
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[cb commit];
		std::vector<Record> r = nextWrite(fx.ring, icb.rgfxObjId);
		CHECK(r.size() == 1 && r[0].op == RGFX_ICB_OP_DRAW);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
	}
	drainRing(fx.ring);
}

// --- ranges ------------------------------------------------------------------------------

// A range is clamped to the buffer's commands; a reset of a range resets each kind of command
// the buffer has, and of none past its size.
void test_ranges_are_clamped()
{
	Fixture fx;
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.make(renderDescriptor(), 8);
		NSRange r = [icb rgfxClampRange:NSMakeRange(2, 4)];
		CHECK(r.location == 2 && r.length == 4);
		r = [icb rgfxClampRange:NSMakeRange(6, 100)];
		CHECK(r.location == 6 && r.length == 2);
		r = [icb rgfxClampRange:NSMakeRange(20, 4)];
		CHECK(r.location == 8 && r.length == 0);
		r = RgfxIcbClamp(nil, NSMakeRange(20, 4));
		CHECK(r.location == 20 && r.length == 4); // not ours: as given

		[icb resetWithRange:NSMakeRange(5, 10)];
		[icb resetWithRange:NSMakeRange(9, 1)]; // past the size: nothing
		MTLIndirectCommandBufferDescriptor *both = renderDescriptor();
		both.commandTypes |= MTLIndirectCommandTypeConcurrentDispatch;
		RgfxIndirectCommandBuffer *mixed = fx.make(both, 4);
		[mixed resetWithRange:NSMakeRange(0, 4)];
		fx.ser.flushIcbWrites();
		std::vector<Record> a = nextWrite(fx.ring, icb.rgfxObjId);
		CHECK(a.size() == 1 && a[0].op == RGFX_ICB_OP_RENDER_RESET_RANGE && a[0].index == 5 &&
		      a[0].as<RgfxIcbRange>().length == 3);
		std::vector<Record> m = nextWrite(fx.ring, mixed.rgfxObjId);
		CHECK(m.size() == 2 && m[0].op == RGFX_ICB_OP_RENDER_RESET_RANGE &&
		      m[1].op == RGFX_ICB_OP_COMPUTE_RESET_RANGE && m[1].as<RgfxIcbRange>().length == 4);
	}
	drainRing(fx.ring);
}


// --- slots ------------------------------------------------------------------------------

// The bytes of a render slot of renderDescriptor()'s shape (2 vertex buffers, 1 fragment
// buffer), and of a compute slot of computeDescriptor()'s (2 kernel buffers, 31 threadgroup
// memory lengths).
constexpr uint64_t kRenderSlot = sizeof(RgfxIcbRenderSlot) + 3 * sizeof(RgfxIcbSlotBuffer);
constexpr uint64_t kComputeSlot = sizeof(RgfxIcbComputeSlot) + 2 * sizeof(RgfxIcbSlotBuffer) + 128;

// Where the device took ICB_SLOTS, a buffer's slots are placed at its first command lookup,
// zeroed, and each encode is written into its command's slot as the wire lays it out (each
// field, then its bit), posting nothing; the flush names the written commands as runs of 64-
// command chunks in one ICB_SLOTS, and the next flush names only what was written since.
void test_slots_hold_the_encodes()
{
	Fixture fx;
	SlotRegion sr(1 << 20);
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.makeSlots(renderDescriptor(), 200, sr.region);
		CHECK(icb != nil && sr.live.empty()); // no block until a command is looked up
		RgfxBuffer *pso = buffer(0x70), *vb = buffer(0x71), *fb = buffer(0x72), *ib = buffer(0x73);
		id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:0];
		CHECK(sr.live.size() == 1);
		const uint64_t block = sr.live[0];
		for (uint64_t i = 0; i < 200 * kRenderSlot; i++)
			CHECK(sr.mem[block + i] == 0);
		[c setRenderPipelineState:(id<MTLRenderPipelineState>)pso];
		[c setVertexBuffer:vb offset:16 atIndex:1];
		[c setVertexBuffer:vb offset:8 attributeStride:24 atIndex:0];
		[c setFragmentBuffer:fb offset:32 atIndex:0];
		[c drawIndexedPrimitives:MTLPrimitiveTypeTriangle
		              indexCount:6
		               indexType:MTLIndexTypeUInt32
		             indexBuffer:ib
		       indexBufferOffset:64
		           instanceCount:2
		              baseVertex:-4
		            baseInstance:1];
		[c setBarrier];
		[[icb indirectRenderCommandAtIndex:130] drawPrimitives:MTLPrimitiveTypePoint
		                                           vertexStart:3
		                                           vertexCount:5
		                                         instanceCount:1
		                                          baseInstance:0];
		CHECK(ringEmpty(fx.ring)); // nothing posted while encoding

		const RgfxIcbRenderSlot s = sr.at<RgfxIcbRenderSlot>(block);
		CHECK(s.mask == (RGFX_ICB_SLOT_PIPELINE | RGFX_ICB_SLOT_ACTION | RGFX_ICB_SLOT_BARRIER |
		                 RGFX_ICB_SLOT_BUFFER_0 | RGFX_ICB_SLOT_BUFFER_0 << 1));
		CHECK(s.mask_b == 1 && s.pipeline == 0x70 && s.op == RGFX_ICB_OP_DRAW_INDEXED &&
		      s.barrier == 1 && s.strided == 1);
		const uint64_t bindings = block + sizeof(RgfxIcbRenderSlot);
		const RgfxIcbSlotBuffer v0 = sr.at<RgfxIcbSlotBuffer>(bindings);
		const RgfxIcbSlotBuffer v1 = sr.at<RgfxIcbSlotBuffer>(bindings + 24);
		const RgfxIcbSlotBuffer f0 = sr.at<RgfxIcbSlotBuffer>(bindings + 48);
		CHECK(v0.obj_id == 0x71 && v0.offset == 8 && v0.stride == 24);
		CHECK(v1.obj_id == 0x71 && v1.offset == 16 && v1.stride == 0);
		CHECK(f0.obj_id == 0x72 && f0.offset == 32);
		RgfxIcbDrawIndexed d;
		std::memcpy(&d, s.action, sizeof(d));
		CHECK(d.primitive_type == MTLPrimitiveTypeTriangle && d.index_count == 6 &&
		      d.index_type == MTLIndexTypeUInt32 && d.instance_count == 2 &&
		      d.index_buffer_obj_id == 0x73 && d.index_buffer_offset == 64 &&
		      d.base_vertex == -4 && d.base_instance == 1);
		uint8_t past[72 - sizeof(d)];
		std::memcpy(past, reinterpret_cast<const uint8_t *>(s.action) + sizeof(d), sizeof(past));
		for (uint8_t b : past)
			CHECK(b == 0);
		const RgfxIcbRenderSlot t = sr.at<RgfxIcbRenderSlot>(block + 130 * kRenderSlot);
		RgfxIcbDraw draw;
		std::memcpy(&draw, t.action, sizeof(draw));
		CHECK(t.mask == RGFX_ICB_SLOT_ACTION && t.op == RGFX_ICB_OP_DRAW &&
		      draw.vertex_start == 3 && draw.vertex_count == 5);

		const RgfxIcbFlush flush = fx.ser.flushIcbWrites();
		CHECK(flush.messages == 1);
		std::vector<RgfxIcbSlotRun> runs = nextSlots(fx.ring);
		CHECK(runs.size() == 2 && ringEmpty(fx.ring));
		for (const RgfxIcbSlotRun &r : runs) {
			CHECK(r.icb_obj_id == icb.rgfxObjId && r.slot_bytes == kRenderSlot &&
			      r.kind == RGFX_ICB_SLOTS_RENDER && r.bindings_a == 2 && r.bindings_b == 1);
			CHECK(r.region_offset == block + r.first_command * kRenderSlot);
		}
		CHECK(runs[0].first_command == 0 && runs[0].command_count == 64);
		CHECK(runs[1].first_command == 128 && runs[1].command_count == 64);
		CHECK(fx.ser.flushIcbWrites().messages == 0 && ringEmpty(fx.ring));

		// Written again: only its chunk; the last chunk ends at the size.
		[c setFragmentBuffer:vb offset:0 atIndex:0];
		[[icb indirectRenderCommandAtIndex:199] setBarrier];
		fx.ser.flushIcbWrites();
		runs = nextSlots(fx.ring);
		CHECK(runs.size() == 2 && runs[0].first_command == 0 && runs[0].command_count == 64 &&
		      runs[1].first_command == 192 && runs[1].command_count == 8);
	}
	drainRing(fx.ring);
}

// A compute slot holds each field of its command, as a render slot does.
void test_compute_slots_hold_the_encodes()
{
	Fixture fx;
	SlotRegion sr(1 << 20);
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.makeSlots(computeDescriptor(), 4, sr.region);
		RgfxBuffer *pso = buffer(0x70), *kb = buffer(0x71);
		id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:2];
		const uint64_t block = sr.live.at(0) + 2 * kComputeSlot;
		[c setComputePipelineState:(id<MTLComputePipelineState>)pso];
		[c setKernelBuffer:kb offset:8 atIndex:0];
		[c setKernelBuffer:kb offset:16 attributeStride:12 atIndex:1];
		[c setThreadgroupMemoryLength:256 atIndex:1];
		[c setThreadgroupMemoryLength:64 atIndex:30];
		[c setStageInRegion:MTLRegionMake3D(1, 2, 3, 4, 5, 6)];
		[c setImageblockWidth:16 height:8];
		[c concurrentDispatchThreads:MTLSizeMake(64, 2, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
		[c clearBarrier];
		CHECK(ringEmpty(fx.ring));
		const RgfxIcbComputeSlot s = sr.at<RgfxIcbComputeSlot>(block);
		CHECK(s.mask == (RGFX_ICB_SLOT_PIPELINE | RGFX_ICB_SLOT_ACTION | RGFX_ICB_SLOT_BARRIER |
		                 RGFX_ICB_SLOT_STAGE_IN | RGFX_ICB_SLOT_IMAGEBLOCK |
		                 RGFX_ICB_SLOT_BUFFER_0 | RGFX_ICB_SLOT_BUFFER_0 << 1));
		CHECK(s.mask_b == ((1ull << 1) | (1ull << 30)) && s.pipeline == 0x70 &&
		      s.op == RGFX_ICB_OP_DISPATCH_THREADS && s.barrier == 0 && s.strided == 2);
		CHECK(s.dispatch.grid[0] == 64 && s.dispatch.grid[1] == 2 && s.dispatch.threadgroup[0] == 32);
		CHECK(s.stage_in.origin[2] == 3 && s.stage_in.size[0] == 4 && s.imageblock.width == 16 &&
		      s.imageblock.height == 8);
		const uint64_t bindings = block + sizeof(RgfxIcbComputeSlot);
		const RgfxIcbSlotBuffer k0 = sr.at<RgfxIcbSlotBuffer>(bindings);
		const RgfxIcbSlotBuffer k1 = sr.at<RgfxIcbSlotBuffer>(bindings + 24);
		CHECK(k0.obj_id == 0x71 && k0.offset == 8 && k1.offset == 16 && k1.stride == 12);
		CHECK(sr.at<uint32_t>(bindings + 48 + 4) == 256 && sr.at<uint32_t>(bindings + 48 + 120) == 64);
		fx.ser.flushIcbWrites();
		std::vector<RgfxIcbSlotRun> runs = nextSlots(fx.ring);
		CHECK(runs.size() == 1 && runs[0].first_command == 0 && runs[0].command_count == 4 &&
		      runs[0].kind == RGFX_ICB_SLOTS_COMPUTE && runs[0].slot_bytes == kComputeSlot &&
		      runs[0].bindings_a == 2 && runs[0].bindings_b == 31);
	}
	drainRing(fx.ring);
}

// A command's reset leaves its slot the reset alone, so what was set before it goes and what is
// set after it follows it; a reset of a range leaves each slot of it the range reset alone.
void test_a_reset_in_a_slot()
{
	Fixture fx;
	SlotRegion sr(1 << 20);
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.makeSlots(renderDescriptor(), 200, sr.region);
		RgfxBuffer *pso = buffer(0x70), *vb = buffer(0x71);
		id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:3];
		const uint64_t block = sr.live.at(0);
		[c setRenderPipelineState:(id<MTLRenderPipelineState>)pso];
		[c setFragmentBuffer:vb offset:0 atIndex:0];
		[c reset];
		[c setVertexBuffer:vb offset:0 atIndex:0];
		RgfxIcbRenderSlot s = sr.at<RgfxIcbRenderSlot>(block + 3 * kRenderSlot);
		CHECK(s.mask == (RGFX_ICB_SLOT_RESET | RGFX_ICB_SLOT_BUFFER_0) && s.mask_b == 0);
		[[icb indirectRenderCommandAtIndex:61] setFragmentBuffer:vb offset:0 atIndex:0];
		[icb resetWithRange:NSMakeRange(60, 10)];
		for (uint64_t i = 60; i < 70; i++) {
			s = sr.at<RgfxIcbRenderSlot>(block + i * kRenderSlot);
			CHECK(s.mask == RGFX_ICB_SLOT_RESET_RANGE && s.mask_b == 0);
		}
		CHECK(sr.at<uint64_t>(block + 70 * kRenderSlot) == 0);
		[icb resetWithRange:NSMakeRange(190, 100)]; // clamped to the size
		CHECK(sr.at<uint64_t>(block + 199 * kRenderSlot) == RGFX_ICB_SLOT_RESET_RANGE);
		CHECK(ringEmpty(fx.ring));
		fx.ser.flushIcbWrites();
		std::vector<RgfxIcbSlotRun> runs = nextSlots(fx.ring);
		// Chunks 0 to 3 written, the last ending at the size: one run.
		CHECK(runs.size() == 1 && runs[0].first_command == 0 && runs[0].command_count == 200);
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// A call a slot cannot hold (a binding past the slot's count) turns the buffer to records for
// good: its written slots are posted at once, ahead of the record, which waits for the flush,
// and every later encode is a record.
void test_slots_turn_to_records()
{
	Fixture fx;
	SlotRegion sr(1 << 20);
	@autoreleasepool {
		RgfxIndirectCommandBuffer *icb = fx.makeSlots(renderDescriptor(), 8, sr.region);
		RgfxBuffer *pso = buffer(0x70), *vb = buffer(0x71);
		id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:0];
		const uint64_t block = sr.live.at(0);
		[c setRenderPipelineState:(id<MTLRenderPipelineState>)pso];
		CHECK(ringEmpty(fx.ring));
		[c setVertexBuffer:vb offset:4 atIndex:5];
		std::vector<RgfxIcbSlotRun> runs = nextSlots(fx.ring);
		CHECK(runs.size() == 1 && runs[0].first_command == 0 && runs[0].command_count == 8);
		CHECK(ringEmpty(fx.ring));
		[[icb indirectRenderCommandAtIndex:1] setRenderPipelineState:(id<MTLRenderPipelineState>)pso];
		[c reset];
		[icb resetWithRange:NSMakeRange(2, 2)];
		CHECK(sr.at<uint64_t>(block) == RGFX_ICB_SLOT_PIPELINE &&
		      sr.at<uint64_t>(block + kRenderSlot) == 0 &&
		      sr.at<uint64_t>(block + 2 * kRenderSlot) == 0); // the slots are left as they were
		fx.ser.flushIcbWrites();
		std::vector<Record> r = nextWrite(fx.ring, icb.rgfxObjId);
		CHECK(r.size() == 4 && r[0].op == RGFX_ICB_OP_VERTEX_BUFFER && r[0].index == 0 &&
		      r[0].as<RgfxIcbBuffer>().index == 5 && r[1].op == RGFX_ICB_OP_RENDER_PIPELINE &&
		      r[1].index == 1 && r[2].op == RGFX_ICB_OP_RENDER_RESET &&
		      r[3].op == RGFX_ICB_OP_RENDER_RESET_RANGE && r[3].index == 2);
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// Whether `icb` writes slots: a draw on command 0 posts an ICB_SLOTS at the flush, not an
// ICB_WRITE (the destroys of buffers let go before it are passed over).
bool writesSlots(Fixture &fx, RgfxIndirectCommandBuffer *icb)
{
	drainRing(fx.ring);
	[[icb indirectRenderCommandAtIndex:0] drawPrimitives:MTLPrimitiveTypePoint
	                                         vertexStart:0
	                                         vertexCount:1
	                                       instanceCount:1
	                                        baseInstance:0];
	fx.ser.flushIcbWrites();
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	int slots = 0, writes = 0;
	while (fx.ring.next(&h, &bytes)) {
		slots += h.type == RGFX_CMD_ICB_SLOTS;
		writes += h.type == RGFX_CMD_ICB_WRITE;
	}
	CHECK(slots + writes == 1);
	return slots == 1;
}

// A buffer takes slots only where the device took them, its commands are of one kind, its
// bindings fit a slot, its slots fit RgfxIcbSlotTable::kMaxBytes and the process's slots fit
// RgfxSerializer::kIcbSlotBytes; else records. A buffer's destroy posts its slots first, then
// the destroy, and its block and bytes go back.
void test_which_buffers_take_slots()
{
	Fixture fx;
	SlotRegion sr(48ull << 20);
	@autoreleasepool {
		CHECK(!writesSlots(fx, fx.make(renderDescriptor(), 8)));
		CHECK(writesSlots(fx, fx.makeSlots(renderDescriptor(), 8, sr.region)));
		MTLIndirectCommandBufferDescriptor *both = renderDescriptor();
		both.commandTypes |= MTLIndirectCommandTypeConcurrentDispatch;
		CHECK(!writesSlots(fx, fx.makeSlots(both, 8, sr.region)));
		MTLIndirectCommandBufferDescriptor *wide = renderDescriptor();
		wide.maxVertexBufferBindCount = 33;
		CHECK(!writesSlots(fx, fx.makeSlots(wide, 8, sr.region)));
		drainRing(fx.ring);
	}
	drainRing(fx.ring);
	CHECK(sr.live.empty());
	const NSUInteger most = RgfxIcbSlotTable::kMaxBytes / kRenderSlot;
	@autoreleasepool {
		CHECK(!writesSlots(fx, fx.makeSlots(renderDescriptor(), most + 1, sr.region)));
		NSMutableArray *held = [NSMutableArray array];
		for (int i = 0; i < 4; i++) {
			RgfxIndirectCommandBuffer *icb = fx.makeSlots(renderDescriptor(), most, sr.region);
			CHECK(writesSlots(fx, icb));
			[held addObject:icb];
		}
		CHECK(sr.live.size() == 4);
		// The process's slots are full: the fifth takes records.
		CHECK(!writesSlots(fx, fx.makeSlots(renderDescriptor(), most, sr.region)));
		drainRing(fx.ring);
		// One goes: its slots first, then its destroy; its block and bytes back.
		RgfxObjId gone = 0;
		@autoreleasepool {
			RgfxIndirectCommandBuffer *last = held.lastObject;
			gone = last.rgfxObjId;
			[[last indirectRenderCommandAtIndex:5] setBarrier];
			[held removeLastObject];
		}
		std::vector<RgfxIcbSlotRun> runs = nextSlots(fx.ring);
		CHECK(runs.size() == 1 && runs[0].icb_obj_id == gone);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(fx.ring.next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY);
		CHECK(sr.live.size() == 3);
		CHECK(writesSlots(fx, fx.makeSlots(renderDescriptor(), most, sr.region)));
		drainRing(fx.ring);
	}
	drainRing(fx.ring);
	CHECK(sr.live.empty());
}

} // namespace

void rgfxIcbTests()
{
	test_create_carries_the_descriptor();
	test_buffer_answers();
	test_command_lookups();
	test_render_records();
	test_compute_records();
	test_setters_the_lane_does_not_carry();
	test_records_go_before_a_destroy();
	test_a_full_message_goes_at_once();
	test_a_commit_posts_the_records_first();
	test_ranges_are_clamped();
	test_slots_hold_the_encodes();
	test_compute_slots_hold_the_encodes();
	test_a_reset_in_a_slot();
	test_slots_turn_to_records();
	test_which_buffers_take_slots();
}
