/*
 * RgfxIndirectCommandBuffer: see RgfxIndirectCommandBuffer.h. Our own ObjC++; the descriptor's
 * SPI members are read only where it answers them.
 */

#import "RgfxIndirectCommandBuffer.h"
#include "RgfxLogOnce.hpp"
#import "RgfxInterop.h"

#import "RgfxResource.h" // RgfxResolveHazardTracking
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include <objc/message.h>

#include <algorithm>
#include <atomic>
#include <mutex>

#include "RgfxIcbSlotTable.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

// The command types each kind of command draws or dispatches (MTLIndirectCommandType).
constexpr uint64_t kRenderTypes = MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed |
                                  MTLIndirectCommandTypeDrawPatches |
                                  MTLIndirectCommandTypeDrawIndexedPatches |
                                  MTLIndirectCommandTypeDrawMeshThreadgroups |
                                  MTLIndirectCommandTypeDrawMeshThreads;
constexpr uint64_t kComputeTypes =
    MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;

// The bytes a command takes (allocatedSize), and the page they are counted in.
constexpr uint64_t kCommandBytes = 64;
constexpr uint64_t kPageBytes = 4096;

// No mesh render pipeline the device makes can be used in an indirect command buffer: its
// creators end the process at one that supports them, as the x86 Macs' Metal does (measured:
// guest/bundle/tools/oracle/oracle_mesh on the AMD 5300M and the Intel 630, macOS 26.7.1).
constexpr const char *kMesh = "mesh shading in indirect command buffers (no mesh pipeline supports them)";
constexpr const char *kReadBack = "reading back what an indirect command was given";

// The SPI member `name` of `d` where it answers it, else `fallback`.
NSUInteger spi(MTLIndirectCommandBufferDescriptor *d, const char *name, NSUInteger fallback)
{
	SEL sel = sel_registerName(name);
	if (![d respondsToSelector:sel])
		return fallback;
	return ((NSUInteger(*)(id, SEL))objc_msgSend)(d, sel);
}

// The low 32 bits of a count, as the native records hold it.
inline uint32_t u32(NSUInteger v)
{
	return static_cast<uint32_t>(v);
}

} // namespace

RgfxIcbCreate RgfxIcbCreateOf(MTLIndirectCommandBufferDescriptor *d, NSUInteger maxCount,
                              MTLResourceOptions options)
{
	RgfxIcbCreate m = {};
	m.max_command_count = maxCount;
	m.options = options;
	m.command_types = static_cast<uint32_t>(d.commandTypes);
	uint32_t f = 0;
	f |= d.inheritPipelineState ? RGFX_ICB_INHERIT_PIPELINE_STATE : 0;
	f |= d.inheritBuffers ? RGFX_ICB_INHERIT_BUFFERS : 0;
	// Each property newer than the 10.14 descriptor through spi(), which answers the fallback where
	// the release lacks the selector: supportRayTracing reached it in macOS 13, the dynamic
	// attribute stride in 14, and the colour-attachment mapping and the inherit* below in 26. A
	// property the release lacks keeps its default (off here, or the base bind counts below).
	f |= spi(d, "supportRayTracing", NO) ? RGFX_ICB_SUPPORT_RAY_TRACING : 0;
	f |= spi(d, "supportDynamicAttributeStride", NO) ? RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE : 0;
	f |= spi(d, "supportColorAttachmentMapping", NO) ? RGFX_ICB_SUPPORT_COLOR_ATTACHMENT_MAPPING : 0;
	f |= spi(d, "inheritDepthStencilState", NO) ? RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE : 0;
	f |= spi(d, "inheritDepthBias", NO) ? RGFX_ICB_INHERIT_DEPTH_BIAS : 0;
	f |= spi(d, "inheritDepthClipMode", NO) ? RGFX_ICB_INHERIT_DEPTH_CLIP_MODE : 0;
	f |= spi(d, "inheritCullMode", NO) ? RGFX_ICB_INHERIT_CULL_MODE : 0;
	f |= spi(d, "inheritFrontFacingWinding", NO) ? RGFX_ICB_INHERIT_FRONT_FACING_WINDING : 0;
	f |= spi(d, "inheritTriangleFillMode", NO) ? RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE : 0;
	f |= spi(d, "inheritStencilReferenceValues", YES) ? RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES
	                                                   : 0;
	f |= spi(d, "inheritDepthTestBounds", YES) ? RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS : 0;
	f |= spi(d, "inheritScissorRects", YES) ? RGFX_ICB_INHERIT_SCISSOR_RECTS : 0;
	f |= spi(d, "inheritViewports", YES) ? RGFX_ICB_INHERIT_VIEWPORTS : 0;
	f |= spi(d, "inheritBlendColor", YES) ? RGFX_ICB_INHERIT_BLEND_COLOR : 0;
	m.flags = f;
	m.max_vertex_buffer_bind_count = u32(d.maxVertexBufferBindCount);
	m.max_fragment_buffer_bind_count = u32(d.maxFragmentBufferBindCount);
	m.max_kernel_buffer_bind_count = u32(d.maxKernelBufferBindCount);
	// The object and mesh bind counts reached the descriptor in macOS 14; spi answers 0 before it.
	m.max_kernel_threadgroup_memory_bind_count = u32(spi(d, "maxKernelThreadgroupMemoryBindCount", 0));
	m.max_object_buffer_bind_count = u32(spi(d, "maxObjectBufferBindCount", 0));
	m.max_mesh_buffer_bind_count = u32(spi(d, "maxMeshBufferBindCount", 0));
	m.max_object_threadgroup_memory_bind_count =
	    u32(spi(d, "maxObjectThreadgroupMemoryBindCount", 0));
	m.max_scissor_rect_count = u32(spi(d, "maxScissorRectCount", 1));
	m.max_viewport_count = u32(spi(d, "maxViewportCount", 1));
	m.allow_override_render_states = spi(d, "allowOverrideRenderStates", NO) ? 1 : 0;
	return m;
}

NSRange RgfxIcbClamp(id icb, NSRange range)
{
	if (![icb isKindOfClass:[RgfxIndirectCommandBuffer class]])
		return range;
	return [(RgfxIndirectCommandBuffer *)icb rgfxClampRange:range];
}

// Each command names its buffer's slots where it writes them (RgfxIcbSlotTable.hpp), which the
// buffer, held by the command, keeps.
@interface RgfxIndirectRenderCommand () {
@package
	RgfxIndirectCommandBuffer *_icb;
	RgfxIcbSlotTable *_slots;
	uint32_t _index;
}
@end

@interface RgfxIndirectComputeCommand () {
@package
	RgfxIndirectCommandBuffer *_icb;
	RgfxIcbSlotTable *_slots;
	uint32_t _index;
}
@end

@implementation RgfxIndirectCommandBuffer {
	NSUInteger _size;
	MTLResourceOptions _options;
	uint64_t _types;
	uint64_t _resourceId; // the host's answer; 0 where none came (a create on a test's ring)
	// Slots: the region they are placed in (null where the device took no ICB_SLOTS), their
	// kind and bindings, and their stride, 0 where the buffer takes records; made at the first
	// CPU encode (rgfxSlots), under _slotLock.
	RgfxRegion *_slotRegion;
	uint16_t _slotKind;
	uint32_t _slotA, _slotB;
	std::atomic<uint32_t> _slotStride;
	std::atomic<RgfxIcbSlotTable *> _slots;
	uint64_t _slotBlock;
	std::mutex _slotLock;
}

// The host drops the buffer as the app lets it go, after any command buffer that names it (each
// keeps it while it runs), and after the writes waiting for it (RgfxSerializer::destroy), its
// slots' among them; the slots' block goes back after the destroy.
- (void)dealloc
{
	RgfxIcbSlotTable *slots = _slots.load(std::memory_order_acquire);
	if (slots == nullptr) {
		[self rgfxRetireThen:nullptr];
		return;
	}
	RgfxSerializer *serializer = _serializer;
	RgfxRegion *region = _slotRegion;
	const uint64_t block = _slotBlock;
	[self rgfxRetireThen:[serializer, region, slots, block] {
		serializer->forgetIcbSlots(slots);
		delete slots;
		region->release(block);
	}];
}

+ (instancetype)icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                  maxCommandCount:(NSUInteger)maxCount
                          options:(MTLResourceOptions)options
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
{
	return [self icbWithDescriptor:descriptor
	               maxCommandCount:maxCount
	                       options:options
	                        device:device
	                    serializer:serializer
	                   slotsRegion:nullptr];
}

+ (instancetype)icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                  maxCommandCount:(NSUInteger)maxCount
                          options:(MTLResourceOptions)options
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
                      slotsRegion:(RgfxRegion *)region
{
	// Every Mac ends the process here, before anything else of the call (measured: an assertion
	// and SIGABRT on the AMD 5300M and the Intel 630, a trap on the M3 Max).
	const NSUInteger storage = (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
	if (storage == MTLStorageModeMemoryless) {
		RgfxMetalAssertionFailed(
		    "-[MTLDevice newIndirectCommandBufferWithDescriptor:maxCommandCount:options:]",
		    @"an indirect command buffer cannot be Memoryless (storageMode 3)");
	}
	if (serializer == nullptr || descriptor == nil)
		return nil;
	RgfxCreated created = serializer->createIcb(RgfxIcbCreateOf(descriptor, maxCount, options));
	// A create the device never got, one the host refused, or one not answered made no buffer:
	// nil, as Metal answers one it cannot make. One the kext timed out on is destroyed behind.
	if (serializer->refused(created) || created.status != RGFX_STATUS_OK) {
		static char site;
		if (rgfxLogFirst(&site, created.status))
			NSLog(@"RGFX: newIndirectCommandBufferWithDescriptor: no buffer of %lu commands "
			      @"(status 0x%x)",
			      (unsigned long)maxCount, created.status);
		return nil;
	}
	RgfxIndirectCommandBuffer *icb = [[RgfxIndirectCommandBuffer alloc] initWithObjId:created.obj_id
	                                                                            device:device
	                                                                        serializer:serializer];
	if (icb) {
		icb->_size = maxCount;
		icb->_options = options;
		icb->_types = descriptor.commandTypes;
		icb->_resourceId = created.resource_id;
		[icb rgfxPlanSlots:descriptor region:region];
	}
	return icb;
}

// Whether the buffer takes slots, and of what shape: where `region` is given, its commands are
// of one kind, its bindings fit a slot and its slots fit kMaxBytes. Else records.
- (void)rgfxPlanSlots:(MTLIndirectCommandBufferDescriptor *)d region:(RgfxRegion *)region
{
	_slotRegion = region;
	_slotStride.store(0, std::memory_order_relaxed);
	if (region == nullptr || _size == 0 || _size > UINT32_MAX)
		return;
	if ((_types & ~kRenderTypes) == 0) {
		_slotKind = RGFX_ICB_SLOTS_RENDER;
		_slotA = u32(std::min<NSUInteger>(d.maxVertexBufferBindCount, UINT32_MAX));
		_slotB = u32(std::min<NSUInteger>(d.maxFragmentBufferBindCount, UINT32_MAX));
	} else if ((_types & ~kComputeTypes) == 0) {
		_slotKind = RGFX_ICB_SLOTS_COMPUTE;
		_slotA = u32(std::min<NSUInteger>(d.maxKernelBufferBindCount, UINT32_MAX));
		_slotB = u32(std::min<NSUInteger>(d.maxKernelThreadgroupMemoryBindCount, UINT32_MAX));
	} else {
		return;
	}
	const uint32_t stride = RgfxIcbSlotTable::strideFor(_slotKind, _slotA, _slotB);
	if (stride != 0 && uint64_t(_size) * stride <= RgfxIcbSlotTable::kMaxBytes)
		_slotStride.store(stride, std::memory_order_relaxed);
}

// The buffer's slots, made at the first call that may write them, zeroed: null where it takes
// records (rgfxPlanSlots:, or where the process's slots or the region have no room for them).
- (RgfxIcbSlotTable *)rgfxSlots
{
	RgfxIcbSlotTable *slots = _slots.load(std::memory_order_acquire);
	if (slots != nullptr || _slotStride.load(std::memory_order_relaxed) == 0)
		return slots;
	std::lock_guard<std::mutex> lock(_slotLock);
	slots = _slots.load(std::memory_order_relaxed);
	const uint32_t stride = _slotStride.load(std::memory_order_relaxed);
	if (slots != nullptr || stride == 0)
		return slots;
	const uint64_t bytes = uint64_t(_size) * stride;
	if (!_serializer->reserveIcbSlotBytes(bytes)) {
		_slotStride.store(0, std::memory_order_relaxed);
		return nullptr;
	}
	const RgfxRegion::Block block = _slotRegion->allocate(bytes);
	if (!block.ok) {
		_serializer->releaseIcbSlotBytes(bytes);
		_slotStride.store(0, std::memory_order_relaxed);
		return nullptr;
	}
	std::memset(const_cast<uint8_t *>(block.ptr), 0, bytes);
	slots = new RgfxIcbSlotTable(_serializer, _objId, _slotKind, _slotA, _slotB, u32(_size), stride,
	                         block.ptr, block.offset);
	_slotBlock = block.offset;
	_slots.store(slots, std::memory_order_release);
	return slots;
}

- (NSRange)rgfxClampRange:(NSRange)range
{
	const NSUInteger location = std::min<NSUInteger>(range.location, _size);
	return NSMakeRange(location, std::min<NSUInteger>(range.length, _size - location));
}

// --- MTLIndirectCommandBuffer ----------------------------------------------------

- (NSUInteger)size
{
	return _size;
}

// The id the host's device numbered the buffer by.
- (MTLResourceID)gpuResourceID
{
	MTLResourceID rid;
	rid._impl = _resourceId;
	return rid;
}

// Metal's IOAccel buffer answers gpuResourceID and gpuHandle with resourceIndex (read from code,
// MTLIOAccelIndirectCommandBuffer, macOS 27.0.1), and both x86 Macs' fresh buffer answers one
// value as its handle, its resource index and its identifier (measured: macOS 26.7.1,
// oracle_spi_members answers): each is the id the host numbered the buffer by.
- (uint64_t)resourceIndex
{
	return _resourceId;
}

- (uint64_t)gpuHandle
{
	return [self resourceIndex];
}

- (uint64_t)uniqueIdentifier
{
	return [self resourceIndex];
}

// The buffer's commands live on the host, so the guest sees no GPU address of them: 0, where both
// x86 Macs answer their buffer's (measured: macOS 26.7.1, oracle_spi_members answers). A shader
// reaches a buffer of the device's by its gpuResourceID.
- (uint64_t)gpuAddress
{
	return 0;
}

// Metal's IOAccel buffer answers a pointer to its driver's header of the buffer and the header's
// size (read from code, MTLIOAccelIndirectCommandBuffer, macOS 27.0.1): the buffer's commands live
// on the host, which keeps no such header in the guest, so these answer none, of size 0.
- (void)getHeader:(void **)header headerSize:(uint64_t *)headerSize
{
	RGFX_UNSUPPORTED("an indirect command buffer's driver header");
	if (header != nullptr) {
		*header = nullptr;
	}
	if (headerSize != nullptr) {
		*headerSize = 0;
	}
}

// Every command of the range loses what it was given, on the host's buffer: in its slots, or a
// reset of each kind of command the buffer has.
- (void)resetWithRange:(NSRange)range
{
	range = [self rgfxClampRange:range];
	if (_serializer == nullptr || range.length == 0)
		return;
	RgfxIcbSlotTable *slots = [self rgfxSlots];
	if (slots != nullptr && slots->active()) {
		slots->resetRange(u32(range.location), u32(range.length));
		return;
	}
	const RgfxIcbRange body = {u32(range.length), 0};
	if (_types & kRenderTypes) {
		_serializer->icbRecord(_objId, RGFX_ICB_OP_RENDER_RESET_RANGE, u32(range.location), &body,
		                       sizeof(body));
	}
	if (_types & kComputeTypes) {
		_serializer->icbRecord(_objId, RGFX_ICB_OP_COMPUTE_RESET_RANGE, u32(range.location),
		                       &body, sizeof(body));
	}
}

// A new command each call; nil past the size or where the buffer draws nothing.
- (id<MTLIndirectRenderCommand>)indirectRenderCommandAtIndex:(NSUInteger)commandIndex
{
	if (commandIndex >= _size || (_types & kRenderTypes) == 0)
		return nil;
	RgfxIndirectRenderCommand *c = [RgfxIndirectRenderCommand new];
	c->_icb = self;
	c->_slots = [self rgfxSlots];
	c->_index = u32(commandIndex);
	return c;
}

// As indirectRenderCommandAtIndex:, for a buffer that dispatches.
- (id<MTLIndirectComputeCommand>)indirectComputeCommandAtIndex:(NSUInteger)commandIndex
{
	if (commandIndex >= _size || (_types & kComputeTypes) == 0)
		return nil;
	RgfxIndirectComputeCommand *c = [RgfxIndirectComputeCommand new];
	c->_icb = self;
	c->_slots = [self rgfxSlots];
	c->_index = u32(commandIndex);
	return c;
}

// --- MTLResource -----------------------------------------------------------------

// The options as Metal answers them: those it was made with, a Default hazard tracking mode
// resolved to Tracked (measured: 0x200 for options of 0, 0x220 for Private, 0x100 Untracked).
- (MTLResourceOptions)resourceOptions
{
	return RgfxResolveHazardTracking(_options, MTLResourceHazardTrackingModeTracked);
}

// The options as asked, as both x86 Macs' buffers answer (0 for a buffer made with none; measured:
// macOS 26.7.1, oracle_spi_members ask answers).
- (MTLResourceOptions)unfilteredResourceOptions
{
	return _options;
}

// The resource SPI a buffer and a texture answer (RgfxResource.h), which both x86 Macs' buffers
// answer too, from Metal's IOAccel resource and indirect command buffer (measured with
// respondsToSelector:, macOS 26.7.1, oracle_spi_members ask): a fresh one is complete, written and
// not purgeable (measured: its answers).
RGFX_RESOURCE_SPI

- (MTLStorageMode)storageMode
{
	return static_cast<MTLStorageMode>((self.resourceOptions & MTLResourceStorageModeMask) >>
	                                   MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
	return static_cast<MTLCPUCacheMode>((self.resourceOptions & MTLResourceCPUCacheModeMask) >>
	                                    MTLResourceCPUCacheModeShift);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
	return static_cast<MTLHazardTrackingMode>((self.resourceOptions &
	                                           MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

// 64 bytes a command in whole pages, one at least: the AMD 5300M's rate for a draw command (4096
// bytes for 16 of them; 68 bytes a command at 1024, 66 at 2048), inside what Macs answer (the M3
// 672 bytes a command, the Intel 630 3428).
- (NSUInteger)allocatedSize
{
	const uint64_t commands = std::min<uint64_t>(_size, UINT64_MAX / kCommandBytes - kPageBytes);
	const uint64_t bytes = (commands * kCommandBytes + kPageBytes - 1) / kPageBytes * kPageBytes;
	return static_cast<NSUInteger>(std::max(bytes, kPageBytes));
}

@end

// --- the commands ------------------------------------------------------------------

namespace {

// A record of `op` on `index` of `icb`, its body `body`.
template <typename T>
inline void record(RgfxIndirectCommandBuffer *icb, uint16_t op, uint32_t index, const T &body)
{
	static_assert(sizeof(T) % 8 == 0, "a record's body is whole words");
	RgfxSerializer *s = icb->_serializer;
	if (s != nullptr)
		s->icbRecord(icb->_objId, op, index, &body, sizeof(body));
}

// A record of `op` on `index` of `icb`, with no body.
inline void bare(RgfxIndirectCommandBuffer *icb, uint16_t op, uint32_t index)
{
	RgfxSerializer *s = icb->_serializer;
	if (s != nullptr)
		s->icbRecord(icb->_objId, op, index, nullptr, 0);
}


inline RgfxIcbDrawPatches patches(NSUInteger controlPoints, NSUInteger start, NSUInteger count,
                                  id<MTLBuffer> patchIndexBuffer, NSUInteger patchIndexOffset,
                                  NSUInteger instanceCount, NSUInteger baseInstance,
                                  id<MTLBuffer> factors, NSUInteger factorsOffset,
                                  NSUInteger instanceStride)
{
	RgfxIcbDrawPatches p = {};
	p.number_of_patch_control_points = u32(controlPoints);
	p.patch_start = u32(start);
	p.patch_count = u32(count);
	p.instance_count = u32(instanceCount);
	p.base_instance = u32(baseInstance);
	p.instance_stride = u32(instanceStride);
	p.patch_index_buffer_obj_id = RgfxResourceId(patchIndexBuffer);
	p.patch_index_buffer_offset = patchIndexOffset;
	p.tessellation_factor_buffer_obj_id = RgfxResourceId(factors);
	p.tessellation_factor_buffer_offset = factorsOffset;
	return p;
}

inline RgfxIcbDispatch dispatch(MTLSize grid, MTLSize threadgroup)
{
	return {{u32(grid.width), u32(grid.height), u32(grid.depth)},
	        {u32(threadgroup.width), u32(threadgroup.height), u32(threadgroup.depth)}};
}

// The command's slots where its buffer still writes them, else null: the call is a record.
inline RgfxIcbSlotTable *live(RgfxIcbSlotTable *slots)
{
	return slots != nullptr && slots->active() ? slots : nullptr;
}

// A draw or a dispatch, `op` with `body`, on `index` of `icb`: into its slot, or a record.
template <typename T>
inline void action(RgfxIndirectCommandBuffer *icb, RgfxIcbSlotTable *slots, uint16_t op,
                   uint32_t index, const T &body)
{
	if (RgfxIcbSlotTable *s = live(slots)) {
		s->setAction(index, op, &body, sizeof(body));
		return;
	}
	record(icb, op, index, body);
}

} // namespace

@implementation RgfxIndirectRenderCommand
RGFX_LOG_UNRECOGNIZED

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
	const uint64_t pipeline = RgfxResourceId(pipelineState);
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setPipeline(_index, pipeline);
		return;
	}
	record(_icb, RGFX_ICB_OP_RENDER_PIPELINE, _index, RgfxIcbObject{pipeline});
}

- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	const uint64_t b = RgfxResourceId(buffer);
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setBuffer(_index, index, b, offset, 0, false))
		return;
	record(_icb, RGFX_ICB_OP_VERTEX_BUFFER, _index, RgfxIcbBuffer{b, offset, u32(index), 0});
}

- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	const uint64_t b = RgfxResourceId(buffer);
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setFragmentBuffer(_index, index, b, offset))
		return;
	record(_icb, RGFX_ICB_OP_FRAGMENT_BUFFER, _index, RgfxIcbBuffer{b, offset, u32(index), 0});
}

- (void)setVertexBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	const uint64_t b = RgfxResourceId(buffer);
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setBuffer(_index, index, b, offset, stride, true))
		return;
	const RgfxIcbBufferStride body = {b, offset, stride, u32(index), 0};
	record(_icb, RGFX_ICB_OP_VERTEX_BUFFER_STRIDE, _index, body);
}

- (void)drawPatches:(NSUInteger)numberOfPatchControlPoints
                                patchStart:(NSUInteger)patchStart
                                patchCount:(NSUInteger)patchCount
                          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
                    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
                             instanceCount:(NSUInteger)instanceCount
                              baseInstance:(NSUInteger)baseInstance
                  tessellationFactorBuffer:(id<MTLBuffer>)buffer
            tessellationFactorBufferOffset:(NSUInteger)offset
    tessellationFactorBufferInstanceStride:(NSUInteger)instanceStride
{
	action(_icb, _slots, RGFX_ICB_OP_DRAW_PATCHES, _index,
	       patches(numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer,
	               patchIndexBufferOffset, instanceCount, baseInstance, buffer, offset,
	               instanceStride));
}

- (void)drawIndexedPatches:(NSUInteger)numberOfPatchControlPoints
                                patchStart:(NSUInteger)patchStart
                                patchCount:(NSUInteger)patchCount
                          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
                    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
                   controlPointIndexBuffer:(id<MTLBuffer>)controlPointIndexBuffer
             controlPointIndexBufferOffset:(NSUInteger)controlPointIndexBufferOffset
                             instanceCount:(NSUInteger)instanceCount
                              baseInstance:(NSUInteger)baseInstance
                  tessellationFactorBuffer:(id<MTLBuffer>)buffer
            tessellationFactorBufferOffset:(NSUInteger)offset
    tessellationFactorBufferInstanceStride:(NSUInteger)instanceStride
{
	RgfxIcbDrawIndexedPatches body = {};
	body.patches = patches(numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer,
	                       patchIndexBufferOffset, instanceCount, baseInstance, buffer, offset,
	                       instanceStride);
	body.control_point_index_buffer_obj_id = RgfxResourceId(controlPointIndexBuffer);
	body.control_point_index_buffer_offset = controlPointIndexBufferOffset;
	action(_icb, _slots, RGFX_ICB_OP_DRAW_INDEXED_PATCHES, _index, body);
}

- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
          baseInstance:(NSUInteger)baseInstance
{
	const RgfxIcbDraw body = {static_cast<uint32_t>(primitiveType), u32(vertexStart),
	                          u32(vertexCount), u32(instanceCount), u32(baseInstance), 0};
	action(_icb, _slots, RGFX_ICB_OP_DRAW, _index, body);
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
                instanceCount:(NSUInteger)instanceCount
                   baseVertex:(NSInteger)baseVertex
                 baseInstance:(NSUInteger)baseInstance
{
	RgfxIcbDrawIndexed body = {};
	body.primitive_type = static_cast<uint32_t>(primitiveType);
	body.index_count = u32(indexCount);
	body.index_type = static_cast<uint32_t>(indexType);
	body.instance_count = u32(instanceCount);
	body.index_buffer_obj_id = RgfxResourceId(indexBuffer);
	body.index_buffer_offset = indexBufferOffset;
	body.base_vertex = static_cast<int32_t>(baseVertex);
	body.base_instance = u32(baseInstance);
	action(_icb, _slots, RGFX_ICB_OP_DRAW_INDEXED, _index, body);
}

// Sent on to the host's buffer, as the M3 Max takes them (the AMD 5300M faults, the Intel 630
// raises an unrecognized selector).
- (void)setBarrier
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setBarrier(_index, true);
		return;
	}
	bare(_icb, RGFX_ICB_OP_RENDER_SET_BARRIER, _index);
}

- (void)clearBarrier
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setBarrier(_index, false);
		return;
	}
	bare(_icb, RGFX_ICB_OP_RENDER_CLEAR_BARRIER, _index);
}

- (void)reset
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->reset(_index);
		return;
	}
	bare(_icb, RGFX_ICB_OP_RENDER_RESET, _index);
}

// --- mesh commands: no mesh pipeline supports indirect command buffers, so none of these draws -

- (void)setObjectThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED(kMesh);
}

- (void)setObjectBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED(kMesh);
}

- (void)setMeshBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED(kMesh);
}

- (void)drawMeshThreadgroups:(MTLSize)threadgroupsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	RGFX_UNSUPPORTED(kMesh);
}

- (void)drawMeshThreads:(MTLSize)threadsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	RGFX_UNSUPPORTED(kMesh);
}

// --- macOS 26's render states: the AMD 5300M and the Intel 630 raise an unrecognized selector
// for each (and answer respondsToSelector: YES), and so do these ---------------------------

- (void)setDepthStencilState:(id<MTLDepthStencilState>)depthStencilState
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setDepthBias:(float)depthBias slopeScale:(float)slopeScale clamp:(float)clamp
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setDepthClipMode:(MTLDepthClipMode)depthClipMode
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setCullMode:(MTLCullMode)cullMode
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setFrontFacingWinding:(MTLWinding)frontFacingWinding
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setTriangleFillMode:(MTLTriangleFillMode)fillMode
{
	[self doesNotRecognizeSelector:_cmd];
}

// --- Metal's indirect render command SPI that reads back what was encoded: both x86 Macs'
// commands answer what each holds (measured: macOS 26.7.1, tools/oracle/oracle_icb.m spi: the
// command type, the pipeline's identifier, each buffer's GPU address and the draw's arguments,
// none of them after a reset). A command here keeps what it was given in its buffer's slots or
// on the host, not where these could read it back, so each answers what a fresh command of the
// Intel 630's answers, logged, which a fresh one of the AMD 5300M's does too but where noted.

// nil (the AMD 5300M's fresh command answers an object of zeros).
- (nullable id)drawArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)drawIndexedArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)drawPatchesArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)drawIndexedPatchesArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)getTessellationFactorArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (uint64_t)getCommandType
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (uint64_t)getOptimizedStatus
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

// 0 (the AMD 5300M's fresh command answers ~0).
- (uint64_t)getPipelineStateUniqueIdentifier
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

// NO, as the AMD 5300M's fresh command answers (the Intel 630's raises an unrecognized selector).
- (BOOL)hasBarrier
{
	RGFX_UNSUPPORTED(kReadBack);
	return NO;
}

- (void *)getVertexBufferAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return nullptr;
}

- (uint64_t)getVertexBufferAddressAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

// 0, as the AMD 5300M's fresh command answers (the Intel 630's answers ~0).
- (uint64_t)getVertexAttributeStrideAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (void *)getFragmentBufferAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return nullptr;
}

- (uint64_t)getFragmentBufferAddressAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

// --- Metal's indirect render command SPI that its base command raises an unrecognized selector
// for (it sends itself doesNotRecognizeSelector:; read from code, _MTLIndirectRenderCommand, macOS
// 27.0.1), which neither x86 Mac's command overrides, and the mesh ones, whose draws no command
// here makes: so do these -------------------------------------------------------------------

- (nullable id)getBlendColor
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getCullMode
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (nullable id)getDepthBiasInfo
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getDepthClipMode
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (nullable id)getDepthStencilState
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getDepthStencilStateUniqueIdentifier
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (nullable id)getDepthTestBounds
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getFrontFacingWinding
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (nullable id)getScissorRects
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (nullable id)getStencilReferenceValues
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getTriangleFillMode
{
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (nullable id)getViewports
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (nullable id)drawMeshThreadgroupsArguments
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (nullable id)drawMeshThreadsArguments
{
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (uint64_t)getMeshBufferAddressAtIndex:(NSUInteger)index
{
	(void)index;
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (uint64_t)getObjectBufferAddressAtIndex:(NSUInteger)index
{
	(void)index;
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (uint64_t)getObjectThreadgroupMemoryLengthAtIndex:(NSUInteger)index
{
	(void)index;
	[self doesNotRecognizeSelector:_cmd];
	return 0;
}

- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setDepthTestMinBound:(float)minBound maxBound:(float)maxBound
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setScissorRect:(MTLScissorRect)rect
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setScissorRects:(const MTLScissorRect *)scissorRects count:(NSUInteger)count
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setStencilReferenceValue:(uint32_t)referenceValue
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setStencilFrontReferenceValue:(uint32_t)frontReferenceValue
                   backReferenceValue:(uint32_t)backReferenceValue
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setToolsDispatchBufferSPI:(NSUInteger)buffer atIndex:(NSUInteger)index stages:(NSUInteger)stages
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setViewport:(MTLViewport)viewport
{
	[self doesNotRecognizeSelector:_cmd];
}

- (void)setViewports:(const MTLViewport *)viewports count:(NSUInteger)count
{
	[self doesNotRecognizeSelector:_cmd];
}

@end

@implementation RgfxIndirectComputeCommand
RGFX_LOG_UNRECOGNIZED

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipelineState
{
	const uint64_t pipeline = RgfxResourceId(pipelineState);
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setPipeline(_index, pipeline);
		return;
	}
	record(_icb, RGFX_ICB_OP_COMPUTE_PIPELINE, _index, RgfxIcbObject{pipeline});
}

- (void)setKernelBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	const uint64_t b = RgfxResourceId(buffer);
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setBuffer(_index, index, b, offset, 0, false))
		return;
	record(_icb, RGFX_ICB_OP_KERNEL_BUFFER, _index, RgfxIcbBuffer{b, offset, u32(index), 0});
}

- (void)setKernelBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	const uint64_t b = RgfxResourceId(buffer);
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setBuffer(_index, index, b, offset, stride, true))
		return;
	const RgfxIcbBufferStride body = {b, offset, stride, u32(index), 0};
	record(_icb, RGFX_ICB_OP_KERNEL_BUFFER_STRIDE, _index, body);
}

- (void)concurrentDispatchThreadgroups:(MTLSize)threadgroupsPerGrid
                 threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	action(_icb, _slots, RGFX_ICB_OP_DISPATCH_THREADGROUPS, _index,
	       dispatch(threadgroupsPerGrid, threadsPerThreadgroup));
}

- (void)concurrentDispatchThreads:(MTLSize)threadsPerGrid
            threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	action(_icb, _slots, RGFX_ICB_OP_DISPATCH_THREADS, _index,
	       dispatch(threadsPerGrid, threadsPerThreadgroup));
}

- (void)setBarrier
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setBarrier(_index, true);
		return;
	}
	bare(_icb, RGFX_ICB_OP_COMPUTE_SET_BARRIER, _index);
}

- (void)clearBarrier
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setBarrier(_index, false);
		return;
	}
	bare(_icb, RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER, _index);
}

- (void)setImageblockWidth:(NSUInteger)width height:(NSUInteger)height
{
	const RgfxIcbImageblock body = {u32(width), u32(height)};
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setImageblock(_index, body);
		return;
	}
	record(_icb, RGFX_ICB_OP_IMAGEBLOCK, _index, body);
}

- (void)reset
{
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->reset(_index);
		return;
	}
	bare(_icb, RGFX_ICB_OP_COMPUTE_RESET, _index);
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	RgfxIcbSlotTable *s = live(_slots);
	if (s != nullptr && s->setThreadgroupMemory(_index, index, u32(length)))
		return;
	record(_icb, RGFX_ICB_OP_THREADGROUP_MEMORY, _index,
	       RgfxIcbThreadgroupMemory{u32(length), u32(index)});
}

- (void)setStageInRegion:(MTLRegion)region
{
	const RgfxIcbRegion body = {
	    {u32(region.origin.x), u32(region.origin.y), u32(region.origin.z)},
	    {u32(region.size.width), u32(region.size.height), u32(region.size.depth)}};
	if (RgfxIcbSlotTable *s = live(_slots)) {
		s->setStageIn(_index, body);
		return;
	}
	record(_icb, RGFX_ICB_OP_STAGE_IN_REGION, _index, body);
}


// --- Metal's indirect compute command SPI that reads back what was encoded: both x86 Macs'
// commands answer what each holds (measured: macOS 26.7.1, tools/oracle/oracle_icb.m spi). As the
// render command's (above), a command here answers what a fresh command answers, logged: the AMD
// 5300M's, where the two differ.

// nil (the Intel 630's fresh command answers an object of zeros).
- (nullable id)dispatchThreadgroupsArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)dispatchThreadsArguments
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (nullable id)getImageBlockSize
{
	RGFX_UNSUPPORTED(kReadBack);
	return nil;
}

- (uint64_t)getCommandType
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (uint64_t)getOptimizedStatus
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

// 0, as the Intel 630's fresh command answers (the AMD 5300M's answers ~0).
- (uint64_t)getPipelineStateUniqueIdentifier
{
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (BOOL)hasBarrier
{
	RGFX_UNSUPPORTED(kReadBack);
	return NO;
}

- (MTLRegion)getStageInRegion
{
	RGFX_UNSUPPORTED(kReadBack);
	return MTLRegionMake3D(0, 0, 0, 0, 0, 0);
}

- (void *)getKernelBufferAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return nullptr;
}

- (uint64_t)getKernelBufferAddressAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (uint64_t)getKernelAttributeStrideAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}

- (uint64_t)getThreadgroupMemoryLengthAtIndex:(NSUInteger)index
{
	(void)index;
	RGFX_UNSUPPORTED(kReadBack);
	return 0;
}
@end
