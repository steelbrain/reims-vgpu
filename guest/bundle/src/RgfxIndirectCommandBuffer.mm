/*
 * RgfxIndirectCommandBuffer: see RgfxIndirectCommandBuffer.h. Our own ObjC++; the descriptor's
 * SPI members are read only where it answers them.
 */

#import "RgfxIndirectCommandBuffer.h"

#import "RgfxResource.h" // RgfxResolveHazardTracking
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include <objc/message.h>

#include <algorithm>

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

constexpr const char *kMesh = "mesh shading (no mesh pipeline create on the wire)";

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
	f |= d.supportRayTracing ? RGFX_ICB_SUPPORT_RAY_TRACING : 0;
	f |= d.supportDynamicAttributeStride ? RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE : 0;
	f |= d.supportColorAttachmentMapping ? RGFX_ICB_SUPPORT_COLOR_ATTACHMENT_MAPPING : 0;
	f |= d.inheritDepthStencilState ? RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE : 0;
	f |= d.inheritDepthBias ? RGFX_ICB_INHERIT_DEPTH_BIAS : 0;
	f |= d.inheritDepthClipMode ? RGFX_ICB_INHERIT_DEPTH_CLIP_MODE : 0;
	f |= d.inheritCullMode ? RGFX_ICB_INHERIT_CULL_MODE : 0;
	f |= d.inheritFrontFacingWinding ? RGFX_ICB_INHERIT_FRONT_FACING_WINDING : 0;
	f |= d.inheritTriangleFillMode ? RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE : 0;
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
	m.max_kernel_threadgroup_memory_bind_count = u32(d.maxKernelThreadgroupMemoryBindCount);
	m.max_object_buffer_bind_count = u32(d.maxObjectBufferBindCount);
	m.max_mesh_buffer_bind_count = u32(d.maxMeshBufferBindCount);
	m.max_object_threadgroup_memory_bind_count = u32(d.maxObjectThreadgroupMemoryBindCount);
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

@interface RgfxIndirectRenderCommand () {
@package
	RgfxIndirectCommandBuffer *_icb;
	uint32_t _index;
}
@end

@interface RgfxIndirectComputeCommand () {
@package
	RgfxIndirectCommandBuffer *_icb;
	uint32_t _index;
}
@end

@implementation RgfxIndirectCommandBuffer {
	NSUInteger _size;
	MTLResourceOptions _options;
	uint64_t _types;
	uint64_t _resourceId; // the host's answer; 0 where none came (a create on a test's ring)
}

// The host drops the buffer as the app lets it go, after any command buffer that names it (each
// keeps it while it runs), and after the records waiting for it (RgfxSerializer::destroy).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                  maxCommandCount:(NSUInteger)maxCount
                          options:(MTLResourceOptions)options
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
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
		NSLog(@"RGFX: newIndirectCommandBufferWithDescriptor: no buffer of %lu commands (status "
		      @"0x%x)",
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
	}
	return icb;
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

// Every command of the range loses what it was given, on the host's buffer: a reset of each
// kind of command the buffer has.
- (void)resetWithRange:(NSRange)range
{
	range = [self rgfxClampRange:range];
	if (_serializer == nullptr || range.length == 0)
		return;
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

inline RgfxIcbBuffer bufferBody(id<MTLBuffer> b, NSUInteger offset, NSUInteger index)
{
	return {RgfxResourceId(b), offset, u32(index), 0};
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

} // namespace

@implementation RgfxIndirectRenderCommand

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
	record(_icb, RGFX_ICB_OP_RENDER_PIPELINE, _index, RgfxIcbObject{RgfxResourceId(pipelineState)});
}

- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	record(_icb, RGFX_ICB_OP_VERTEX_BUFFER, _index, bufferBody(buffer, offset, index));
}

- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	record(_icb, RGFX_ICB_OP_FRAGMENT_BUFFER, _index, bufferBody(buffer, offset, index));
}

- (void)setVertexBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	const RgfxIcbBufferStride body = {RgfxResourceId(buffer), offset, stride, u32(index), 0};
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
	record(_icb, RGFX_ICB_OP_DRAW_PATCHES, _index,
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
	record(_icb, RGFX_ICB_OP_DRAW_INDEXED_PATCHES, _index, body);
}

- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
          baseInstance:(NSUInteger)baseInstance
{
	const RgfxIcbDraw body = {static_cast<uint32_t>(primitiveType), u32(vertexStart),
	                          u32(vertexCount), u32(instanceCount), u32(baseInstance), 0};
	record(_icb, RGFX_ICB_OP_DRAW, _index, body);
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
	record(_icb, RGFX_ICB_OP_DRAW_INDEXED, _index, body);
}

// Sent on to the host's buffer, as the M3 Max takes them (the AMD 5300M faults, the Intel 630
// raises an unrecognized selector).
- (void)setBarrier
{
	bare(_icb, RGFX_ICB_OP_RENDER_SET_BARRIER, _index);
}

- (void)clearBarrier
{
	bare(_icb, RGFX_ICB_OP_RENDER_CLEAR_BARRIER, _index);
}

- (void)reset
{
	bare(_icb, RGFX_ICB_OP_RENDER_RESET, _index);
}

// --- mesh commands: the device makes no mesh pipeline, so none of these can draw ------------

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

@end

@implementation RgfxIndirectComputeCommand

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipelineState
{
	record(_icb, RGFX_ICB_OP_COMPUTE_PIPELINE, _index, RgfxIcbObject{RgfxResourceId(pipelineState)});
}

- (void)setKernelBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	record(_icb, RGFX_ICB_OP_KERNEL_BUFFER, _index, bufferBody(buffer, offset, index));
}

- (void)setKernelBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	const RgfxIcbBufferStride body = {RgfxResourceId(buffer), offset, stride, u32(index), 0};
	record(_icb, RGFX_ICB_OP_KERNEL_BUFFER_STRIDE, _index, body);
}

- (void)concurrentDispatchThreadgroups:(MTLSize)threadgroupsPerGrid
                 threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	record(_icb, RGFX_ICB_OP_DISPATCH_THREADGROUPS, _index,
	       dispatch(threadgroupsPerGrid, threadsPerThreadgroup));
}

- (void)concurrentDispatchThreads:(MTLSize)threadsPerGrid
            threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	record(_icb, RGFX_ICB_OP_DISPATCH_THREADS, _index,
	       dispatch(threadsPerGrid, threadsPerThreadgroup));
}

- (void)setBarrier
{
	bare(_icb, RGFX_ICB_OP_COMPUTE_SET_BARRIER, _index);
}

- (void)clearBarrier
{
	bare(_icb, RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER, _index);
}

- (void)setImageblockWidth:(NSUInteger)width height:(NSUInteger)height
{
	record(_icb, RGFX_ICB_OP_IMAGEBLOCK, _index, RgfxIcbImageblock{u32(width), u32(height)});
}

- (void)reset
{
	bare(_icb, RGFX_ICB_OP_COMPUTE_RESET, _index);
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	record(_icb, RGFX_ICB_OP_THREADGROUP_MEMORY, _index,
	       RgfxIcbThreadgroupMemory{u32(length), u32(index)});
}

- (void)setStageInRegion:(MTLRegion)region
{
	const RgfxIcbRegion body = {
	    {u32(region.origin.x), u32(region.origin.y), u32(region.origin.z)},
	    {u32(region.size.width), u32(region.size.height), u32(region.size.depth)}};
	record(_icb, RGFX_ICB_OP_STAGE_IN_REGION, _index, body);
}

@end
