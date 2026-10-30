/*
 * RgfxEncoder: building the command-buffer op stream and posting it on a queue's
 * submission ring (RgfxEncoder.hpp). Each op is an RgfxOpHeader {tag, flags, len}
 * then its body; `len` is the whole op, a multiple of 8. The framing and body
 * layouts mirror the host decoder,
 * crates/reims-vgpu/src/graphics/decode/{cmdbuf,render,compute,blit}.rs.
 */

#include "RgfxEncoder.hpp"

#include <cstring>

namespace {

/// The f32 bits of `value`, as a float field travels on the wire.
uint32_t f32bits(float value)
{
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

/// The f64 bits of `value`, as an `MTLViewport`/`MTLClearColor` field travels.
uint64_t f64bits(double value)
{
	uint64_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

/// `n` rounded up to a multiple of 8, as every op's `len` is.
size_t roundUp8(size_t n)
{
	return (n + 7) & ~static_cast<size_t>(7);
}

} // namespace

RgfxEncoder::RgfxEncoder(RgfxRingWriter &ring, RgfxObjId queue_obj_id, RgfxObjId cb_obj_id)
    : m_ring(ring), m_head{}, m_buf()
{
	m_head.queue_obj_id = queue_obj_id;
	m_head.cb_obj_id = cb_obj_id;
	m_head.op_bytes = 0;
	// Reserve the fixed header; its RgfxMsgHeader is stamped by the ring writer on
	// post, and op_bytes is filled by finish(). The op stream starts right after.
	m_buf.resize(RGFX_COMMAND_BUFFER_OPS_OFFSET, 0);
	std::memcpy(m_buf.data(), &m_head, sizeof(m_head));
}

void RgfxEncoder::appendBytes(const void *src, size_t n)
{
	const uint8_t *b = static_cast<const uint8_t *>(src);
	m_buf.insert(m_buf.end(), b, b + n);
}

void RgfxEncoder::appendOp(uint16_t tag)
{
	RgfxOpHeader h = {};
	h.tag = tag;
	h.flags = 0;
	h.len = RGFX_OP_BODY_OFFSET; // header only, 8
	appendBytes(&h, sizeof(h));
}

void RgfxEncoder::appendOp(uint16_t tag, const void *body, size_t n)
{
	RgfxOpHeader h = {};
	h.tag = tag;
	h.flags = 0;
	h.len = static_cast<uint32_t>(RGFX_OP_BODY_OFFSET + n);
	appendBytes(&h, sizeof(h));
	appendBytes(body, n);
}

void RgfxEncoder::appendArrayOp(uint16_t tag, const void *head, size_t head_n, const void *elems,
                                uint32_t count, size_t stride)
{
	RgfxOpHeader h = {};
	h.tag = tag;
	h.flags = 0;
	h.len = static_cast<uint32_t>(RGFX_OP_BODY_OFFSET + head_n + static_cast<size_t>(count) * stride);
	appendBytes(&h, sizeof(h));
	appendBytes(head, head_n);
	if (count && stride)
		appendBytes(elems, static_cast<size_t>(count) * stride);
}

uint64_t RgfxEncoder::opBytes() const
{
	return m_buf.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET;
}

bool RgfxEncoder::fitsRing() const
{
	return m_buf.size() <= UINT32_MAX && m_ring.fits(static_cast<uint32_t>(m_buf.size()));
}

bool RgfxEncoder::finish(uint64_t *out_seqno, const std::function<void()> &whileWaiting,
                         const std::function<void(uint64_t)> &beforePublish)
{
	if (!fitsRing())
		return false;
	m_head.op_bytes = opBytes();
	std::memcpy(m_buf.data(), &m_head, sizeof(m_head));
	return m_ring.post(RGFX_CMD_COMMAND_BUFFER, m_buf.data(),
	                   static_cast<uint32_t>(m_buf.size()), out_seqno, whileWaiting, beforePublish);
}

bool RgfxEncoder::finishInRegion(volatile uint8_t *block, uint64_t region_offset,
                                 uint64_t *out_seqno, const std::function<void()> &whileWaiting,
                                 const std::function<void(uint64_t)> &beforePublish)
{
	if (m_buf.size() > UINT32_MAX)
		return false;
	m_head.op_bytes = opBytes();
	m_head.header.type = RGFX_CMD_COMMAND_BUFFER;
	m_head.header.size = static_cast<uint32_t>(m_buf.size());
	m_head.header.seqno = 0;
	std::memcpy(m_buf.data(), &m_head, sizeof(m_head));
	// Write-back memory the host reads once the reference's `producer` is published (a
	// release store), so the copy is ordered before it.
	std::memcpy(const_cast<uint8_t *>(block), m_buf.data(), m_buf.size());
	RgfxCommandBufferInRegion ref = {};
	ref.region_offset = region_offset;
	ref.byte_length = m_buf.size();
	return m_ring.post(RGFX_CMD_COMMAND_BUFFER_IN_REGION, &ref, sizeof(ref), out_seqno,
	                   whileWaiting, beforePublish);
}

// --- framing ----------------------------------------------------------------

void RgfxEncoder::beginRenderPass(RgfxRenderPassDesc desc,
                                  const RgfxRenderPassColorAttachment *colors, uint32_t color_count)
{
	desc.color_count = color_count;
	appendArrayOp(RGFX_OP_ENCODER_BEGIN_RENDER, &desc, sizeof(desc), colors, color_count,
	              sizeof(RgfxRenderPassColorAttachment));
}

void RgfxEncoder::beginComputePass()
{
	appendOp(RGFX_OP_ENCODER_BEGIN_COMPUTE); // empty body: Metal's default pass
}

void RgfxEncoder::beginComputePass(uint32_t dispatch_type)
{
	RgfxComputePassDesc d = {};
	d.dispatch_type = dispatch_type;
	d.flags = RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE;
	appendOp(RGFX_OP_ENCODER_BEGIN_COMPUTE, &d, sizeof(d));
}

void RgfxEncoder::beginBlitPass()
{
	appendOp(RGFX_OP_ENCODER_BEGIN_BLIT);
}

void RgfxEncoder::endEncoder()
{
	appendOp(RGFX_OP_ENCODER_END);
}

// --- render ops -------------------------------------------------------------

void RgfxEncoder::renderSetRenderPipelineState(RgfxObjId pipeline)
{
	RgfxOpSetRenderPipelineState b = {};
	b.pipeline_obj_id = pipeline;
	appendOp(RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, &b, sizeof(b));
}

void RgfxEncoder::renderSetVertexBuffer(RgfxObjId buffer, uint64_t offset, uint64_t index)
{
	RgfxOpSetVertexBuffer b = {};
	b.buffer_obj_id = buffer;
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_VERTEX_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::renderSetFragmentBuffer(RgfxObjId buffer, uint64_t offset, uint64_t index)
{
	RgfxOpSetFragmentBuffer b = {};
	b.buffer_obj_id = buffer;
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_FRAGMENT_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::renderSetFragmentTexture(RgfxObjId texture, uint64_t index)
{
	RgfxOpSetFragmentTexture b = {};
	b.texture_obj_id = texture;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::renderSetViewport(double origin_x, double origin_y, double width, double height,
                                    double znear, double zfar)
{
	RgfxOpSetViewport b = {};
	b.origin_x_bits = f64bits(origin_x);
	b.origin_y_bits = f64bits(origin_y);
	b.width_bits = f64bits(width);
	b.height_bits = f64bits(height);
	b.znear_bits = f64bits(znear);
	b.zfar_bits = f64bits(zfar);
	appendOp(RGFX_OP_RENDER_SET_VIEWPORT, &b, sizeof(b));
}

void RgfxEncoder::renderSetScissorRect(uint64_t x, uint64_t y, uint64_t width, uint64_t height)
{
	RgfxOpSetScissorRect b = {};
	b.x = x;
	b.y = y;
	b.width = width;
	b.height = height;
	appendOp(RGFX_OP_RENDER_SET_SCISSOR_RECT, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPrimitives(uint64_t primitive_type, uint64_t vertex_start,
                                       uint64_t vertex_count)
{
	RgfxOpDrawPrimitives b = {};
	b.primitive_type = primitive_type;
	b.vertex_start = vertex_start;
	b.vertex_count = vertex_count;
	appendOp(RGFX_OP_RENDER_DRAW_PRIMITIVES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPrimitives(uint64_t primitive_type, uint64_t index_count,
                                              uint64_t index_type, RgfxObjId index_buffer,
                                              uint64_t index_buffer_offset)
{
	RgfxOpDrawIndexedPrimitives b = {};
	b.primitive_type = primitive_type;
	b.index_count = index_count;
	b.index_type = index_type;
	b.index_buffer_obj_id = index_buffer;
	b.index_buffer_offset = index_buffer_offset;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPrimitivesInstances(uint64_t primitive_type, uint64_t vertex_start,
                                                uint64_t vertex_count, uint64_t instance_count)
{
	RgfxOpDrawPrimitivesInstances b = {};
	b.primitive_type = primitive_type;
	b.vertex_start = vertex_start;
	b.vertex_count = vertex_count;
	b.instance_count = instance_count;
	appendOp(RGFX_OP_RENDER_DRAW_PRIMITIVES_INSTANCES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPrimitivesBase(uint64_t primitive_type, uint64_t vertex_start,
                                           uint64_t vertex_count, uint64_t instance_count,
                                           uint64_t base_instance)
{
	RgfxOpDrawPrimitivesBase b = {};
	b.primitive_type = primitive_type;
	b.vertex_start = vertex_start;
	b.vertex_count = vertex_count;
	b.instance_count = instance_count;
	b.base_instance = base_instance;
	appendOp(RGFX_OP_RENDER_DRAW_PRIMITIVES_BASE, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPrimitivesIndirect(uint64_t primitive_type, RgfxObjId indirect_buffer,
                                               uint64_t indirect_buffer_offset)
{
	RgfxOpDrawPrimitivesIndirect b = {};
	b.primitive_type = primitive_type;
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_RENDER_DRAW_PRIMITIVES_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPrimitivesInstances(uint64_t primitive_type, uint64_t index_count,
                                                       uint64_t index_type, RgfxObjId index_buffer,
                                                       uint64_t index_buffer_offset,
                                                       uint64_t instance_count)
{
	RgfxOpDrawIndexedPrimitivesInstances b = {};
	b.primitive_type = primitive_type;
	b.index_count = index_count;
	b.index_type = index_type;
	b.index_buffer_obj_id = index_buffer;
	b.index_buffer_offset = index_buffer_offset;
	b.instance_count = instance_count;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPrimitivesBase(uint64_t primitive_type, uint64_t index_count,
                                                  uint64_t index_type, RgfxObjId index_buffer,
                                                  uint64_t index_buffer_offset,
                                                  uint64_t instance_count, int64_t base_vertex,
                                                  uint64_t base_instance)
{
	RgfxOpDrawIndexedPrimitivesBase b = {};
	b.primitive_type = primitive_type;
	b.index_count = index_count;
	b.index_type = index_type;
	b.index_buffer_obj_id = index_buffer;
	b.index_buffer_offset = index_buffer_offset;
	b.instance_count = instance_count;
	b.base_vertex = base_vertex;
	b.base_instance = base_instance;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_BASE, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPrimitivesIndirect(uint64_t primitive_type, uint64_t index_type,
                                                      RgfxObjId index_buffer,
                                                      uint64_t index_buffer_offset,
                                                      RgfxObjId indirect_buffer,
                                                      uint64_t indirect_buffer_offset)
{
	RgfxOpDrawIndexedPrimitivesIndirect b = {};
	b.primitive_type = primitive_type;
	b.index_type = index_type;
	b.index_buffer_obj_id = index_buffer;
	b.index_buffer_offset = index_buffer_offset;
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPatches(uint64_t number_of_patch_control_points, uint64_t patch_start,
                                    uint64_t patch_count, RgfxObjId patch_index_buffer,
                                    uint64_t patch_index_buffer_offset, uint64_t instance_count,
                                    uint64_t base_instance)
{
	RgfxOpDrawPatches b = {};
	b.number_of_patch_control_points = number_of_patch_control_points;
	b.patch_start = patch_start;
	b.patch_count = patch_count;
	b.patch_index_buffer_obj_id = patch_index_buffer;
	b.patch_index_buffer_offset = patch_index_buffer_offset;
	b.instance_count = instance_count;
	b.base_instance = base_instance;
	appendOp(RGFX_OP_RENDER_DRAW_PATCHES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawPatchesIndirect(uint64_t number_of_patch_control_points,
                                            RgfxObjId patch_index_buffer,
                                            uint64_t patch_index_buffer_offset,
                                            RgfxObjId indirect_buffer,
                                            uint64_t indirect_buffer_offset)
{
	RgfxOpDrawPatchesIndirect b = {};
	b.number_of_patch_control_points = number_of_patch_control_points;
	b.patch_index_buffer_obj_id = patch_index_buffer;
	b.patch_index_buffer_offset = patch_index_buffer_offset;
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_RENDER_DRAW_PATCHES_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPatches(uint64_t number_of_patch_control_points,
                                           uint64_t patch_start, uint64_t patch_count,
                                           RgfxObjId patch_index_buffer,
                                           uint64_t patch_index_buffer_offset,
                                           RgfxObjId control_point_index_buffer,
                                           uint64_t control_point_index_buffer_offset,
                                           uint64_t instance_count, uint64_t base_instance)
{
	RgfxOpDrawIndexedPatches b = {};
	b.number_of_patch_control_points = number_of_patch_control_points;
	b.patch_start = patch_start;
	b.patch_count = patch_count;
	b.patch_index_buffer_obj_id = patch_index_buffer;
	b.patch_index_buffer_offset = patch_index_buffer_offset;
	b.control_point_index_buffer_obj_id = control_point_index_buffer;
	b.control_point_index_buffer_offset = control_point_index_buffer_offset;
	b.instance_count = instance_count;
	b.base_instance = base_instance;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PATCHES, &b, sizeof(b));
}

void RgfxEncoder::renderDrawIndexedPatchesIndirect(uint64_t number_of_patch_control_points,
                                                   RgfxObjId patch_index_buffer,
                                                   uint64_t patch_index_buffer_offset,
                                                   RgfxObjId control_point_index_buffer,
                                                   uint64_t control_point_index_buffer_offset,
                                                   RgfxObjId indirect_buffer,
                                                   uint64_t indirect_buffer_offset)
{
	RgfxOpDrawIndexedPatchesIndirect b = {};
	b.number_of_patch_control_points = number_of_patch_control_points;
	b.patch_index_buffer_obj_id = patch_index_buffer;
	b.patch_index_buffer_offset = patch_index_buffer_offset;
	b.control_point_index_buffer_obj_id = control_point_index_buffer;
	b.control_point_index_buffer_offset = control_point_index_buffer_offset;
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_RENDER_DRAW_INDEXED_PATCHES_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::renderDispatchThreadsPerTile(uint64_t width, uint64_t height, uint64_t depth)
{
	RgfxOpDispatchThreadsPerTile b = {};
	b.width = width;
	b.height = height;
	b.depth = depth;
	appendOp(RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE, &b, sizeof(b));
}

void RgfxEncoder::renderDispatchThreadsPerTileInRegion(uint64_t width, uint64_t height,
                                                       uint64_t depth, uint64_t origin_x,
                                                       uint64_t origin_y, uint64_t origin_z,
                                                       uint64_t region_width, uint64_t region_height,
                                                       uint64_t region_depth)
{
	RgfxOpDispatchThreadsPerTileInRegion b = {};
	b.width = width;
	b.height = height;
	b.depth = depth;
	b.origin_x = origin_x;
	b.origin_y = origin_y;
	b.origin_z = origin_z;
	b.region_width = region_width;
	b.region_height = region_height;
	b.region_depth = region_depth;
	appendOp(RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION, &b, sizeof(b));
}

void RgfxEncoder::renderDispatchThreadsPerTileInRegionIndex(
    uint64_t width, uint64_t height, uint64_t depth, uint64_t origin_x, uint64_t origin_y,
    uint64_t origin_z, uint64_t region_width, uint64_t region_height, uint64_t region_depth,
    uint32_t render_target_array_index)
{
	RgfxOpDispatchThreadsPerTileInRegionIndex b = {};
	b.width = width;
	b.height = height;
	b.depth = depth;
	b.origin_x = origin_x;
	b.origin_y = origin_y;
	b.origin_z = origin_z;
	b.region_width = region_width;
	b.region_height = region_height;
	b.region_depth = region_depth;
	b.render_target_array_index = render_target_array_index;
	appendOp(RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION_INDEX, &b, sizeof(b));
}

void RgfxEncoder::renderExecuteCommandsInBuffer(RgfxObjId indirect_command_buffer,
                                                uint64_t range_location, uint64_t range_length)
{
	RgfxOpExecuteCommandsInBuffer b = {};
	b.indirect_command_buffer_obj_id = indirect_command_buffer;
	b.range_location = range_location;
	b.range_length = range_length;
	appendOp(RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::renderExecuteCommandsIndirect(RgfxObjId indirect_command_buffer,
                                                RgfxObjId indirect_range_buffer,
                                                uint64_t indirect_buffer_offset)
{
	RgfxOpExecuteCommandsIndirect b = {};
	b.indirect_command_buffer_obj_id = indirect_command_buffer;
	b.indirect_range_buffer_obj_id = indirect_range_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::renderMemoryBarrierWithScope(uint64_t scope, uint64_t after, uint64_t before)
{
	RgfxOpMemoryBarrierWithScope b = {};
	b.scope = scope;
	b.after = after;
	b.before = before;
	appendOp(RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE, &b, sizeof(b));
}

void RgfxEncoder::renderUpdateFence(RgfxObjId fence, uint64_t stages)
{
	RgfxOpUpdateFence b = {};
	b.fence_obj_id = fence;
	b.stages = stages;
	appendOp(RGFX_OP_RENDER_UPDATE_FENCE, &b, sizeof(b));
}

void RgfxEncoder::renderWaitForFence(RgfxObjId fence, uint64_t stages)
{
	RgfxOpWaitForFence b = {};
	b.fence_obj_id = fence;
	b.stages = stages;
	appendOp(RGFX_OP_RENDER_WAIT_FOR_FENCE, &b, sizeof(b));
}

void RgfxEncoder::renderSetCullMode(uint64_t cull_mode)
{
	RgfxOpSetCullMode b = {};
	b.cull_mode = cull_mode;
	appendOp(RGFX_OP_RENDER_SET_CULL_MODE, &b, sizeof(b));
}

void RgfxEncoder::renderSetFrontFacingWinding(uint64_t winding)
{
	RgfxOpSetFrontFacingWinding b = {};
	b.front_facing_winding = winding;
	appendOp(RGFX_OP_RENDER_SET_FRONT_FACING_WINDING, &b, sizeof(b));
}

void RgfxEncoder::renderSetTriangleFillMode(uint64_t fill_mode)
{
	RgfxOpSetTriangleFillMode b = {};
	b.fill_mode = fill_mode;
	appendOp(RGFX_OP_RENDER_SET_TRIANGLE_FILL_MODE, &b, sizeof(b));
}

void RgfxEncoder::renderSetDepthClipModeSpi(uint64_t mode)
{
	RgfxOpSetDepthClipModeSpi b = {};
	b.mode = mode;
	appendOp(RGFX_OP_RENDER_SET_DEPTH_CLIP_MODE_SPI, &b, sizeof(b));
}

void RgfxEncoder::renderSetProvokingVertexMode(uint64_t mode)
{
	RgfxOpSetProvokingVertexMode b = {};
	b.mode = mode;
	appendOp(RGFX_OP_RENDER_SET_PROVOKING_VERTEX_MODE, &b, sizeof(b));
}

void RgfxEncoder::renderSetDepthStoreAction(uint64_t store_action)
{
	RgfxOpSetDepthStoreAction b = {};
	b.store_action = store_action;
	appendOp(RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION, &b, sizeof(b));
}

void RgfxEncoder::renderSetStencilStoreAction(uint64_t store_action)
{
	RgfxOpSetStencilStoreAction b = {};
	b.store_action = store_action;
	appendOp(RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION, &b, sizeof(b));
}

void RgfxEncoder::renderSetStencilStoreActionOptions(uint64_t store_action_options)
{
	RgfxOpSetStencilStoreActionOptions b = {};
	b.store_action_options = store_action_options;
	appendOp(RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION_OPTIONS, &b, sizeof(b));
}

void RgfxEncoder::renderSetColorStoreAction(uint64_t store_action, uint64_t color_attachment_index)
{
	RgfxOpSetColorStoreAction b = {};
	b.store_action = store_action;
	b.color_attachment_index = color_attachment_index;
	appendOp(RGFX_OP_RENDER_SET_COLOR_STORE_ACTION, &b, sizeof(b));
}

void RgfxEncoder::renderSetColorStoreActionOptions(uint64_t store_action_options,
                                                   uint64_t color_attachment_index)
{
	RgfxOpSetColorStoreActionOptions b = {};
	b.store_action_options = store_action_options;
	b.color_attachment_index = color_attachment_index;
	appendOp(RGFX_OP_RENDER_SET_COLOR_STORE_ACTION_OPTIONS, &b, sizeof(b));
}

void RgfxEncoder::renderSetVisibilityResultMode(uint64_t mode, uint64_t offset)
{
	RgfxOpSetVisibilityResultMode b = {};
	b.mode = mode;
	b.offset = offset;
	appendOp(RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE, &b, sizeof(b));
}

void RgfxEncoder::renderSetVertexAmplificationMode(uint64_t mode, uint64_t value)
{
	RgfxOpSetVertexAmplificationMode b = {};
	b.mode = mode;
	b.value = value;
	appendOp(RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_MODE, &b, sizeof(b));
}

void RgfxEncoder::renderSetVertexBufferOffset(uint64_t offset, uint64_t index)
{
	RgfxOpSetVertexBufferOffset b = {};
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET, &b, sizeof(b));
}

void RgfxEncoder::renderSetFragmentBufferOffset(uint64_t offset, uint64_t index)
{
	RgfxOpSetFragmentBufferOffset b = {};
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_FRAGMENT_BUFFER_OFFSET, &b, sizeof(b));
}

void RgfxEncoder::renderSetTileBufferOffset(uint64_t offset, uint64_t index)
{
	RgfxOpSetTileBufferOffset b = {};
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_TILE_BUFFER_OFFSET, &b, sizeof(b));
}

void RgfxEncoder::renderSetVertexBufferOffsetAttributeStride(uint64_t offset, uint64_t stride,
                                                             uint64_t index)
{
	RgfxOpSetVertexBufferOffsetAttributeStride b = {};
	b.offset = offset;
	b.stride = stride;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET_ATTRIBUTE_STRIDE, &b, sizeof(b));
}

void RgfxEncoder::renderSetThreadgroupMemoryLength(uint64_t length, uint64_t offset, uint64_t index)
{
	RgfxOpSetThreadgroupMemoryLength b = {};
	b.length = length;
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_RENDER_SET_THREADGROUP_MEMORY_LENGTH, &b, sizeof(b));
}

void RgfxEncoder::renderSetAlphaTestReferenceValue(float reference_value)
{
	RgfxOpSetAlphaTestReferenceValue b = {};
	b.reference_value_bits = f32bits(reference_value);
	appendOp(RGFX_OP_RENDER_SET_ALPHA_TEST_REFERENCE_VALUE, &b, sizeof(b));
}

void RgfxEncoder::renderSetLineWidth(float line_width)
{
	RgfxOpSetLineWidth b = {};
	b.line_width_bits = f32bits(line_width);
	appendOp(RGFX_OP_RENDER_SET_LINE_WIDTH, &b, sizeof(b));
}

void RgfxEncoder::renderSetPointSize(float point_size)
{
	RgfxOpSetPointSize b = {};
	b.point_size_bits = f32bits(point_size);
	appendOp(RGFX_OP_RENDER_SET_POINT_SIZE, &b, sizeof(b));
}

void RgfxEncoder::renderSetTessellationFactorScale(float scale)
{
	RgfxOpSetTessellationFactorScale b = {};
	b.scale_bits = f32bits(scale);
	appendOp(RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_SCALE, &b, sizeof(b));
}

void RgfxEncoder::renderSetBlendColor(float red, float green, float blue, float alpha)
{
	RgfxOpSetBlendColorRed b = {};
	b.red_bits = f32bits(red);
	b.green_bits = f32bits(green);
	b.blue_bits = f32bits(blue);
	b.alpha_bits = f32bits(alpha);
	appendOp(RGFX_OP_RENDER_SET_BLEND_COLOR_RED, &b, sizeof(b));
}

void RgfxEncoder::renderSetDepthBias(float depth_bias, float slope_scale, float clamp)
{
	RgfxOpSetDepthBias b = {};
	b.depth_bias_bits = f32bits(depth_bias);
	b.slope_scale_bits = f32bits(slope_scale);
	b.clamp_bits = f32bits(clamp);
	appendOp(RGFX_OP_RENDER_SET_DEPTH_BIAS, &b, sizeof(b));
}

void RgfxEncoder::renderSetViewportTransformEnabled(bool enabled)
{
	RgfxOpSetViewportTransformEnabled b = {};
	b.enabled = enabled ? 1u : 0u;
	appendOp(RGFX_OP_RENDER_SET_VIEWPORT_TRANSFORM_ENABLED, &b, sizeof(b));
}

void RgfxEncoder::renderSetPrimitiveRestartEnabled(uint64_t index, bool enabled)
{
	RgfxOpSetPrimitiveRestartEnabled b = {};
	b.index = index;
	b.enabled = enabled ? 1u : 0u;
	appendOp(RGFX_OP_RENDER_SET_PRIMITIVE_RESTART_ENABLED, &b, sizeof(b));
}

void RgfxEncoder::renderSetStencilReferenceValues(uint32_t front_reference_value,
                                                  uint32_t back_reference_value)
{
	RgfxOpSetStencilFrontReferenceValue b = {};
	b.front_reference_value = front_reference_value;
	b.back_reference_value = back_reference_value;
	appendOp(RGFX_OP_RENDER_SET_STENCIL_FRONT_REFERENCE_VALUE, &b, sizeof(b));
}

void RgfxEncoder::renderSetDepthCleared()
{
	appendOp(RGFX_OP_RENDER_SET_DEPTH_CLEARED); // no body
}

void RgfxEncoder::renderSetStencilCleared()
{
	appendOp(RGFX_OP_RENDER_SET_STENCIL_CLEARED); // no body
}

void RgfxEncoder::renderSetDepthStencilState(RgfxObjId depth_stencil_state)
{
	RgfxOpSetDepthStencilState b = {};
	b.depth_stencil_state_obj_id = depth_stencil_state;
	appendOp(RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE, &b, sizeof(b));
}

void RgfxEncoder::renderSetColorResolveTexture(RgfxObjId texture, uint64_t slice,
                                               uint64_t depth_plane, uint64_t level, uint64_t index,
                                               uint32_t y_invert)
{
	RgfxOpSetColorResolveTexture b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.depth_plane = depth_plane;
	b.level = level;
	b.index = index;
	b.y_invert = y_invert;
	appendOp(RGFX_OP_RENDER_SET_COLOR_RESOLVE_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::renderSetDepthResolveTexture(RgfxObjId texture, uint64_t slice,
                                               uint64_t depth_plane, uint64_t level,
                                               uint32_t y_invert)
{
	RgfxOpSetDepthResolveTexture b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.depth_plane = depth_plane;
	b.level = level;
	b.y_invert = y_invert;
	appendOp(RGFX_OP_RENDER_SET_DEPTH_RESOLVE_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::renderSetStencilResolveTexture(RgfxObjId texture, uint64_t slice,
                                                 uint64_t depth_plane, uint64_t level,
                                                 uint32_t y_invert)
{
	RgfxOpSetStencilResolveTexture b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.depth_plane = depth_plane;
	b.level = level;
	b.y_invert = y_invert;
	appendOp(RGFX_OP_RENDER_SET_STENCIL_RESOLVE_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::renderSetTessellationFactorBuffer(RgfxObjId buffer, uint64_t offset,
                                                    uint64_t instance_stride)
{
	RgfxOpSetTessellationFactorBuffer b = {};
	b.buffer_obj_id = buffer;
	b.offset = offset;
	b.instance_stride = instance_stride;
	appendOp(RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::renderSetVertexBuffers(uint64_t range_location, const RgfxBufferBinding *bindings,
                                         uint32_t count)
{
	RgfxOpSetVertexBuffers h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_BUFFERS, &h, sizeof(h), bindings, count,
	              sizeof(RgfxBufferBinding));
}

void RgfxEncoder::renderSetFragmentBuffers(uint64_t range_location,
                                           const RgfxBufferBinding *bindings, uint32_t count)
{
	RgfxOpSetFragmentBuffers h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_FRAGMENT_BUFFERS, &h, sizeof(h), bindings, count,
	              sizeof(RgfxBufferBinding));
}

void RgfxEncoder::renderSetTileBuffers(uint64_t range_location, const RgfxBufferBinding *bindings,
                                       uint32_t count)
{
	RgfxOpSetTileBuffers h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_TILE_BUFFERS, &h, sizeof(h), bindings, count,
	              sizeof(RgfxBufferBinding));
}

void RgfxEncoder::renderSetVertexBuffersAttributeStrides(uint64_t range_location,
                                                         const RgfxBufferStrideBinding *bindings,
                                                         uint32_t count)
{
	RgfxOpSetVertexBuffersAttributeStrides h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES, &h, sizeof(h), bindings,
	              count, sizeof(RgfxBufferStrideBinding));
}

void RgfxEncoder::renderSetVertexTextures(uint64_t range_location, const RgfxTextureBinding *textures,
                                          uint32_t count)
{
	RgfxOpSetVertexTextures h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_TEXTURES, &h, sizeof(h), textures, count,
	              sizeof(RgfxTextureBinding));
}

void RgfxEncoder::renderSetFragmentTextures(uint64_t range_location,
                                            const RgfxTextureBinding *textures, uint32_t count)
{
	RgfxOpSetFragmentTextures h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_FRAGMENT_TEXTURES, &h, sizeof(h), textures, count,
	              sizeof(RgfxTextureBinding));
}

void RgfxEncoder::renderSetTileTextures(uint64_t range_location, const RgfxTextureBinding *textures,
                                        uint32_t count)
{
	RgfxOpSetTileTextures h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_TILE_TEXTURES, &h, sizeof(h), textures, count,
	              sizeof(RgfxTextureBinding));
}

void RgfxEncoder::renderSetVertexSamplerStates(uint64_t range_location,
                                               const RgfxSamplerBinding *samplers, uint32_t count)
{
	RgfxOpSetVertexSamplerStates h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES, &h, sizeof(h), samplers, count,
	              sizeof(RgfxSamplerBinding));
}

void RgfxEncoder::renderSetFragmentSamplerStates(uint64_t range_location,
                                                 const RgfxSamplerBinding *samplers, uint32_t count)
{
	RgfxOpSetFragmentSamplerStates h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES, &h, sizeof(h), samplers, count,
	              sizeof(RgfxSamplerBinding));
}

void RgfxEncoder::renderSetTileSamplerStates(uint64_t range_location,
                                             const RgfxSamplerBinding *samplers, uint32_t count)
{
	RgfxOpSetTileSamplerStates h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES, &h, sizeof(h), samplers, count,
	              sizeof(RgfxSamplerBinding));
}

void RgfxEncoder::renderSetVertexSamplerStatesLodClamps(uint64_t range_location,
                                                        const RgfxSamplerLodBinding *samplers,
                                                        uint32_t count)
{
	RgfxOpSetVertexSamplerStatesLodClamps h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES_LOD_CLAMPS, &h, sizeof(h), samplers,
	              count, sizeof(RgfxSamplerLodBinding));
}

void RgfxEncoder::renderSetFragmentSamplerStatesLodClamps(uint64_t range_location,
                                                          const RgfxSamplerLodBinding *samplers,
                                                          uint32_t count)
{
	RgfxOpSetFragmentSamplerStatesLodClamps h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES_LOD_CLAMPS, &h, sizeof(h), samplers,
	              count, sizeof(RgfxSamplerLodBinding));
}

void RgfxEncoder::renderSetTileSamplerStatesLodClamps(uint64_t range_location,
                                                      const RgfxSamplerLodBinding *samplers,
                                                      uint32_t count)
{
	RgfxOpSetTileSamplerStatesLodClamps h = {};
	h.range_location = range_location;
	h.range_length = count;
	appendArrayOp(RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES_LOD_CLAMPS, &h, sizeof(h), samplers, count,
	              sizeof(RgfxSamplerLodBinding));
}

void RgfxEncoder::renderSetScissorRects(const RgfxScissorRectElem *rects, uint32_t count)
{
	RgfxOpSetScissorRects h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_RENDER_SET_SCISSOR_RECTS, &h, sizeof(h), rects, count,
	              sizeof(RgfxScissorRectElem));
}

void RgfxEncoder::renderSetViewports(const RgfxViewportElem *viewports, uint32_t count)
{
	RgfxOpSetViewports h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VIEWPORTS, &h, sizeof(h), viewports, count,
	              sizeof(RgfxViewportElem));
}

void RgfxEncoder::renderSetVertexAmplificationCount(const RgfxViewMapping *mappings, uint32_t count)
{
	// The amplification count is the number of view mappings that follow.
	RgfxOpSetVertexAmplificationCount h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_COUNT, &h, sizeof(h), mappings, count,
	              sizeof(RgfxViewMapping));
}

void RgfxEncoder::renderUseHeaps(const RgfxHeapBinding *heaps, uint32_t count)
{
	RgfxOpUseHeaps h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_RENDER_USE_HEAPS, &h, sizeof(h), heaps, count, sizeof(RgfxHeapBinding));
}

void RgfxEncoder::renderUseHeapsStages(const RgfxHeapBinding *heaps, uint32_t count, uint64_t stages)
{
	RgfxOpUseHeapsStages h = {};
	h.count = count;
	h.stages = stages;
	appendArrayOp(RGFX_OP_RENDER_USE_HEAPS_STAGES, &h, sizeof(h), heaps, count,
	              sizeof(RgfxHeapBinding));
}

void RgfxEncoder::renderUseResources(const RgfxResourceBinding *resources, uint32_t count,
                                     uint64_t usage)
{
	RgfxOpUseResources h = {};
	h.count = count;
	h.usage = usage;
	appendArrayOp(RGFX_OP_RENDER_USE_RESOURCES, &h, sizeof(h), resources, count,
	              sizeof(RgfxResourceBinding));
}

void RgfxEncoder::renderUseResourcesStages(const RgfxResourceBinding *resources, uint32_t count,
                                           uint64_t usage, uint64_t stages)
{
	RgfxOpUseResourcesStages h = {};
	h.count = count;
	h.usage = usage;
	h.stages = stages;
	appendArrayOp(RGFX_OP_RENDER_USE_RESOURCES_STAGES, &h, sizeof(h), resources, count,
	              sizeof(RgfxResourceBinding));
}

// --- compute ops ------------------------------------------------------------

void RgfxEncoder::computeSetComputePipelineState(RgfxObjId pipeline)
{
	RgfxComputeOpSetComputePipelineState b = {};
	b.pipeline_obj_id = pipeline;
	appendOp(RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, &b, sizeof(b));
}

void RgfxEncoder::computeSetBufferOffset(uint64_t offset, uint64_t index)
{
	RgfxComputeOpSetBufferOffset b = {};
	b.offset = offset;
	b.index = index;
	appendOp(RGFX_OP_COMPUTE_SET_BUFFER_OFFSET, &b, sizeof(b));
}

void RgfxEncoder::computeSetBufferOffsetAttributeStride(uint64_t offset, uint64_t stride,
                                                        uint64_t index)
{
	RgfxComputeOpSetBufferOffsetAttributeStride b = {};
	b.offset = offset;
	b.stride = stride;
	b.index = index;
	appendOp(RGFX_OP_COMPUTE_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE, &b, sizeof(b));
}

void RgfxEncoder::computeSetBuffers(uint64_t location,
                                    const RgfxComputeBufferBinding *bindings,
                                    uint32_t count)
{
	RgfxComputeOpSetBuffers h = {};
	h.location = location;
	h.length = count;
	appendArrayOp(RGFX_OP_COMPUTE_SET_BUFFERS, &h, sizeof(h), bindings, count,
	              sizeof(RgfxComputeBufferBinding));
}

void RgfxEncoder::computeSetBuffersAttributeStrides(
    uint64_t location, const RgfxComputeBufferStrideBinding *bindings, uint32_t count)
{
	RgfxComputeOpSetBuffersAttributeStrides h = {};
	h.location = location;
	h.length = count;
	appendArrayOp(RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES, &h, sizeof(h), bindings, count,
	              sizeof(RgfxComputeBufferStrideBinding));
}

void RgfxEncoder::computeSetTexture(RgfxObjId texture, uint64_t index)
{
	RgfxComputeOpSetTexture b = {};
	b.texture_obj_id = texture;
	b.index = index;
	appendOp(RGFX_OP_COMPUTE_SET_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::computeSetTextures(uint64_t location,
                                     const RgfxComputeTextureBinding *textures,
                                     uint32_t count)
{
	RgfxComputeOpSetTextures h = {};
	h.location = location;
	h.length = count;
	appendArrayOp(RGFX_OP_COMPUTE_SET_TEXTURES, &h, sizeof(h), textures, count,
	              sizeof(RgfxComputeTextureBinding));
}

void RgfxEncoder::computeSetSamplerStates(uint64_t location,
                                          const RgfxComputeSamplerBinding *samplers,
                                          uint32_t count)
{
	RgfxComputeOpSetSamplerStates h = {};
	h.location = location;
	h.length = count;
	appendArrayOp(RGFX_OP_COMPUTE_SET_SAMPLER_STATES, &h, sizeof(h), samplers, count,
	              sizeof(RgfxComputeSamplerBinding));
}

void RgfxEncoder::computeSetSamplerStatesLodClamps(
    uint64_t location, const RgfxComputeSamplerLodBinding *samplers, uint32_t count)
{
	RgfxComputeOpSetSamplerStatesLodClamps h = {};
	h.location = location;
	h.length = count;
	appendArrayOp(RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS, &h, sizeof(h), samplers, count,
	              sizeof(RgfxComputeSamplerLodBinding));
}

void RgfxEncoder::computeSetThreadgroupMemoryLength(uint64_t length, uint64_t index)
{
	RgfxComputeOpSetThreadgroupMemoryLength b = {};
	b.length = length;
	b.index = index;
	appendOp(RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH, &b, sizeof(b));
}

void RgfxEncoder::computeSetStageInRegion(uint64_t origin_x, uint64_t origin_y, uint64_t origin_z,
                                          uint64_t width, uint64_t height, uint64_t depth)
{
	RgfxComputeOpSetStageInRegion b = {};
	b.origin_x = origin_x;
	b.origin_y = origin_y;
	b.origin_z = origin_z;
	b.width = width;
	b.height = height;
	b.depth = depth;
	appendOp(RGFX_OP_COMPUTE_SET_STAGE_IN_REGION, &b, sizeof(b));
}

void RgfxEncoder::computeSetStageInRegionIndirect(RgfxObjId indirect_buffer,
                                                  uint64_t indirect_buffer_offset)
{
	RgfxComputeOpSetStageInRegionIndirect b = {};
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::computeSetImageblockWidth(uint64_t width, uint64_t height)
{
	RgfxComputeOpSetImageblockWidth b = {};
	b.width = width;
	b.height = height;
	appendOp(RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH, &b, sizeof(b));
}

void RgfxEncoder::computeDispatchThreadgroups(uint64_t threadgroups_width,
                                              uint64_t threadgroups_height,
                                              uint64_t threadgroups_depth, uint64_t threads_width,
                                              uint64_t threads_height, uint64_t threads_depth)
{
	RgfxComputeOpDispatchThreadgroups b = {};
	b.threadgroups_width = threadgroups_width;
	b.threadgroups_height = threadgroups_height;
	b.threadgroups_depth = threadgroups_depth;
	b.threads_width = threads_width;
	b.threads_height = threads_height;
	b.threads_depth = threads_depth;
	appendOp(RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, &b, sizeof(b));
}

void RgfxEncoder::computeDispatchThreadgroupsIndirect(RgfxObjId indirect_buffer,
                                                      uint64_t indirect_buffer_offset,
                                                      uint64_t threads_width, uint64_t threads_height,
                                                      uint64_t threads_depth)
{
	RgfxComputeOpDispatchThreadgroupsIndirect b = {};
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	b.threads_width = threads_width;
	b.threads_height = threads_height;
	b.threads_depth = threads_depth;
	appendOp(RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::computeDispatchThreads(uint64_t grid_width, uint64_t grid_height,
                                         uint64_t grid_depth, uint64_t threads_width,
                                         uint64_t threads_height, uint64_t threads_depth)
{
	RgfxComputeOpDispatchThreads b = {};
	b.grid_width = grid_width;
	b.grid_height = grid_height;
	b.grid_depth = grid_depth;
	b.threads_width = threads_width;
	b.threads_height = threads_height;
	b.threads_depth = threads_depth;
	appendOp(RGFX_OP_COMPUTE_DISPATCH_THREADS, &b, sizeof(b));
}

void RgfxEncoder::computeDispatchThreadsIndirect(RgfxObjId indirect_buffer,
                                                 uint64_t indirect_buffer_offset)
{
	RgfxComputeOpDispatchThreadsIndirect b = {};
	b.indirect_buffer_obj_id = indirect_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::computeUseResources(const RgfxComputeResourceRef *resources,
                                      uint32_t count, uint64_t usage)
{
	RgfxComputeOpUseResources h = {};
	h.count = count;
	h.usage = usage;
	appendArrayOp(RGFX_OP_COMPUTE_USE_RESOURCES, &h, sizeof(h), resources, count,
	              sizeof(RgfxComputeResourceRef));
}

void RgfxEncoder::computeUseHeaps(const RgfxComputeHeapBinding *heaps, uint32_t count)
{
	RgfxComputeOpUseHeaps h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_COMPUTE_USE_HEAPS, &h, sizeof(h), heaps, count,
	              sizeof(RgfxComputeHeapBinding));
}

void RgfxEncoder::computeMemoryBarrierResources(
    const RgfxComputeResourceRef *resources, uint32_t count)
{
	RgfxComputeOpMemoryBarrierResources h = {};
	h.count = count;
	appendArrayOp(RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES, &h, sizeof(h), resources, count,
	              sizeof(RgfxComputeResourceRef));
}

void RgfxEncoder::computeMemoryBarrierScope(uint64_t scope)
{
	RgfxComputeOpMemoryBarrierScope b = {};
	b.scope = scope;
	appendOp(RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE, &b, sizeof(b));
}

void RgfxEncoder::computeUpdateFence(RgfxObjId fence)
{
	RgfxComputeOpUpdateFence b = {};
	b.fence_obj_id = fence;
	appendOp(RGFX_OP_COMPUTE_UPDATE_FENCE, &b, sizeof(b));
}

void RgfxEncoder::computeWaitForFence(RgfxObjId fence)
{
	RgfxComputeOpWaitForFence b = {};
	b.fence_obj_id = fence;
	appendOp(RGFX_OP_COMPUTE_WAIT_FOR_FENCE, &b, sizeof(b));
}

void RgfxEncoder::computeExecuteCommandsIndirect(RgfxObjId indirect_command_buffer,
                                                 RgfxObjId indirect_range_buffer,
                                                 uint64_t indirect_buffer_offset)
{
	RgfxComputeOpExecuteCommandsIndirect b = {};
	b.indirect_commandbuffer_obj_id = indirect_command_buffer;
	b.indirect_range_buffer_obj_id = indirect_range_buffer;
	b.indirect_buffer_offset = indirect_buffer_offset;
	appendOp(RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT, &b, sizeof(b));
}

void RgfxEncoder::computeExecuteCommandsInBuffer(RgfxObjId indirect_command_buffer,
                                                 uint64_t location, uint64_t length)
{
	RgfxComputeOpExecuteCommandsInBuffer b = {};
	b.indirect_command_buffer_obj_id = indirect_command_buffer;
	b.location = location;
	b.length = length;
	appendOp(RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::computeEncodeStartIf(RgfxObjId buffer, uint64_t offset, uint64_t comparison,
                                       uint32_t reference_value)
{
	RgfxComputeOpEncodeStartIf b = {};
	b.buffer_obj_id = buffer;
	b.offset = offset;
	b.comparison = comparison;
	b.reference_value = reference_value;
	appendOp(RGFX_OP_COMPUTE_ENCODE_START_IF, &b, sizeof(b));
}

void RgfxEncoder::computeEncodeStartWhile(RgfxObjId buffer, uint64_t offset, uint64_t comparison,
                                          uint32_t reference_value)
{
	RgfxComputeOpEncodeStartWhile b = {};
	b.buffer_obj_id = buffer;
	b.offset = offset;
	b.comparison = comparison;
	b.reference_value = reference_value;
	appendOp(RGFX_OP_COMPUTE_ENCODE_START_WHILE, &b, sizeof(b));
}

void RgfxEncoder::computeEncodeStartElse()
{
	appendOp(RGFX_OP_COMPUTE_ENCODE_START_ELSE); // no body
}

void RgfxEncoder::computeEncodeStartDoWhile()
{
	appendOp(RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE); // no body
}

void RgfxEncoder::computeCompressedTextureReinterpretationFlush()
{
	appendOp(RGFX_OP_COMPUTE_COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH); // no body
}

// --- blit ops ---------------------------------------------------------------

void RgfxEncoder::blitCopyBufferToBuffer(RgfxObjId source_buffer, RgfxObjId destination_buffer,
                                         uint64_t source_offset, uint64_t destination_offset,
                                         uint64_t size)
{
	RgfxBlitOpCopyBufferToBuffer b = {};
	b.source_buffer_obj_id = source_buffer;
	b.destination_buffer_obj_id = destination_buffer;
	b.source_offset = source_offset;
	b.destination_offset = destination_offset;
	b.size = size;
	appendOp(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitCopyBufferToTexture(
    RgfxObjId source_buffer, RgfxObjId destination_texture, uint64_t source_offset,
    uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_size_width,
    uint64_t source_size_height, uint64_t source_size_depth, uint64_t destination_slice,
    uint64_t destination_level, uint64_t destination_origin_x, uint64_t destination_origin_y,
    uint64_t destination_origin_z)
{
	RgfxBlitOpCopyBufferToTexture b = {};
	b.source_buffer_obj_id = source_buffer;
	b.destination_texture_obj_id = destination_texture;
	b.source_offset = source_offset;
	b.source_bytes_per_row = source_bytes_per_row;
	b.source_bytes_per_image = source_bytes_per_image;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_slice = destination_slice;
	b.destination_level = destination_level;
	b.destination_origin_x = destination_origin_x;
	b.destination_origin_y = destination_origin_y;
	b.destination_origin_z = destination_origin_z;
	appendOp(RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::blitCopyBufferToTextureOptions(
    RgfxObjId source_buffer, RgfxObjId destination_texture, uint64_t source_offset,
    uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_size_width,
    uint64_t source_size_height, uint64_t source_size_depth, uint64_t destination_slice,
    uint64_t destination_level, uint64_t destination_origin_x, uint64_t destination_origin_y,
    uint64_t destination_origin_z, uint64_t options)
{
	RgfxBlitOpCopyBufferToTextureOptions b = {};
	b.source_buffer_obj_id = source_buffer;
	b.destination_texture_obj_id = destination_texture;
	b.source_offset = source_offset;
	b.source_bytes_per_row = source_bytes_per_row;
	b.source_bytes_per_image = source_bytes_per_image;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_slice = destination_slice;
	b.destination_level = destination_level;
	b.destination_origin_x = destination_origin_x;
	b.destination_origin_y = destination_origin_y;
	b.destination_origin_z = destination_origin_z;
	b.options = options;
	appendOp(RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE_OPTIONS, &b, sizeof(b));
}

void RgfxEncoder::blitCopyTextureToBuffer(
    RgfxObjId source_texture, RgfxObjId destination_buffer, uint64_t source_slice,
    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
    uint64_t source_size_depth, uint64_t destination_offset, uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image)
{
	RgfxBlitOpCopyTextureToBuffer b = {};
	b.source_texture_obj_id = source_texture;
	b.destination_buffer_obj_id = destination_buffer;
	b.source_slice = source_slice;
	b.source_level = source_level;
	b.source_origin_x = source_origin_x;
	b.source_origin_y = source_origin_y;
	b.source_origin_z = source_origin_z;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_offset = destination_offset;
	b.destination_bytes_per_row = destination_bytes_per_row;
	b.destination_bytes_per_image = destination_bytes_per_image;
	appendOp(RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitCopyTextureToBufferOptions(
    RgfxObjId source_texture, RgfxObjId destination_buffer, uint64_t source_slice,
    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
    uint64_t source_size_depth, uint64_t destination_offset, uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image, uint64_t options)
{
	RgfxBlitOpCopyTextureToBufferOptions b = {};
	b.source_texture_obj_id = source_texture;
	b.destination_buffer_obj_id = destination_buffer;
	b.source_slice = source_slice;
	b.source_level = source_level;
	b.source_origin_x = source_origin_x;
	b.source_origin_y = source_origin_y;
	b.source_origin_z = source_origin_z;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_offset = destination_offset;
	b.destination_bytes_per_row = destination_bytes_per_row;
	b.destination_bytes_per_image = destination_bytes_per_image;
	b.options = options;
	appendOp(RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER_OPTIONS, &b, sizeof(b));
}

void RgfxEncoder::blitCopyTextureRegion(
    RgfxObjId source_texture, RgfxObjId destination_texture, uint64_t source_slice,
    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
    uint64_t source_size_depth, uint64_t destination_slice, uint64_t destination_level,
    uint64_t destination_origin_x, uint64_t destination_origin_y, uint64_t destination_origin_z)
{
	RgfxBlitOpCopyTextureRegion b = {};
	b.source_texture_obj_id = source_texture;
	b.destination_texture_obj_id = destination_texture;
	b.source_slice = source_slice;
	b.source_level = source_level;
	b.source_origin_x = source_origin_x;
	b.source_origin_y = source_origin_y;
	b.source_origin_z = source_origin_z;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_slice = destination_slice;
	b.destination_level = destination_level;
	b.destination_origin_x = destination_origin_x;
	b.destination_origin_y = destination_origin_y;
	b.destination_origin_z = destination_origin_z;
	appendOp(RGFX_OP_BLIT_COPY_TEXTURE_REGION, &b, sizeof(b));
}

void RgfxEncoder::blitCopyTextureRegionOptions(
    RgfxObjId source_texture, RgfxObjId destination_texture, uint64_t source_slice,
    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
    uint64_t source_size_depth, uint64_t destination_slice, uint64_t destination_level,
    uint64_t destination_origin_x, uint64_t destination_origin_y, uint64_t destination_origin_z,
    uint64_t options)
{
	RgfxBlitOpCopyTextureRegionOptions b = {};
	b.source_texture_obj_id = source_texture;
	b.destination_texture_obj_id = destination_texture;
	b.source_slice = source_slice;
	b.source_level = source_level;
	b.source_origin_x = source_origin_x;
	b.source_origin_y = source_origin_y;
	b.source_origin_z = source_origin_z;
	b.source_size_width = source_size_width;
	b.source_size_height = source_size_height;
	b.source_size_depth = source_size_depth;
	b.destination_slice = destination_slice;
	b.destination_level = destination_level;
	b.destination_origin_x = destination_origin_x;
	b.destination_origin_y = destination_origin_y;
	b.destination_origin_z = destination_origin_z;
	b.options = options;
	appendOp(RGFX_OP_BLIT_COPY_TEXTURE_REGION_OPTIONS, &b, sizeof(b));
}

void RgfxEncoder::blitCopyTextureSlices(RgfxObjId source_texture, RgfxObjId destination_texture,
                                        uint64_t source_slice, uint64_t source_level,
                                        uint64_t destination_slice, uint64_t destination_level,
                                        uint64_t slice_count, uint64_t level_count)
{
	RgfxBlitOpCopyTextureSlices b = {};
	b.source_texture_obj_id = source_texture;
	b.destination_texture_obj_id = destination_texture;
	b.source_slice = source_slice;
	b.source_level = source_level;
	b.destination_slice = destination_slice;
	b.destination_level = destination_level;
	b.slice_count = slice_count;
	b.level_count = level_count;
	appendOp(RGFX_OP_BLIT_COPY_TEXTURE_SLICES, &b, sizeof(b));
}

void RgfxEncoder::blitCopyIndirectCommandBuffer(RgfxObjId source, RgfxObjId destination,
                                                uint64_t source_range_location,
                                                uint64_t source_range_length,
                                                uint64_t destination_index)
{
	RgfxBlitOpCopyIndirectCommandBuffer b = {};
	b.source_obj_id = source;
	b.destination_obj_id = destination;
	b.source_range_location = source_range_location;
	b.source_range_length = source_range_length;
	b.destination_index = destination_index;
	appendOp(RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitFillBuffer(RgfxObjId buffer, uint64_t range_location, uint64_t range_length,
                                 uint8_t value)
{
	RgfxBlitOpFillBuffer b = {};
	b.buffer_obj_id = buffer;
	b.range_location = range_location;
	b.range_length = range_length;
	b.value = value;
	appendOp(RGFX_OP_BLIT_FILL_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitFillBufferPattern4(RgfxObjId buffer, uint64_t range_location,
                                         uint64_t range_length, uint32_t pattern)
{
	RgfxBlitOpFillBufferPattern4 b = {};
	b.buffer_obj_id = buffer;
	b.range_location = range_location;
	b.range_length = range_length;
	b.pattern = pattern;
	appendOp(RGFX_OP_BLIT_FILL_BUFFER_PATTERN4, &b, sizeof(b));
}

void RgfxEncoder::blitFillTexture(RgfxObjId texture, uint64_t level, uint64_t slice,
                                  uint64_t region_origin_x, uint64_t region_origin_y,
                                  uint64_t region_origin_z, uint64_t region_size_width,
                                  uint64_t region_size_height, uint64_t region_size_depth,
                                  const void *bytes, uint64_t length)
{
	// Head at body offset 0, the `length` fill bytes at RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET
	// (== sizeof head), then pad the op to a multiple of 8.
	RgfxBlitOpFillTexture h = {};
	h.texture_obj_id = texture;
	h.level = level;
	h.slice = slice;
	h.region_origin_x = region_origin_x;
	h.region_origin_y = region_origin_y;
	h.region_origin_z = region_origin_z;
	h.region_size_width = region_size_width;
	h.region_size_height = region_size_height;
	h.region_size_depth = region_size_depth;
	h.length = length;

	size_t body_n = roundUp8(sizeof(h) + static_cast<size_t>(length));
	RgfxOpHeader oh = {};
	oh.tag = RGFX_OP_BLIT_FILL_TEXTURE;
	oh.flags = 0;
	oh.len = static_cast<uint32_t>(RGFX_OP_BODY_OFFSET + body_n);
	appendBytes(&oh, sizeof(oh));
	appendBytes(&h, sizeof(h));
	if (length)
		appendBytes(bytes, static_cast<size_t>(length));
	// Zero-pad the fill bytes to the op's 8-aligned length.
	size_t pad = body_n - (sizeof(h) + static_cast<size_t>(length));
	for (size_t i = 0; i < pad; i++)
		m_buf.push_back(0);
}

void RgfxEncoder::blitFillTextureColor(RgfxObjId texture, uint64_t level, uint64_t slice,
                                       uint64_t region_origin_x, uint64_t region_origin_y,
                                       uint64_t region_origin_z, uint64_t region_size_width,
                                       uint64_t region_size_height, uint64_t region_size_depth,
                                       double red, double green, double blue, double alpha,
                                       uint64_t pixel_format)
{
	RgfxBlitOpFillTextureColor b = {};
	b.texture_obj_id = texture;
	b.level = level;
	b.slice = slice;
	b.region_origin_x = region_origin_x;
	b.region_origin_y = region_origin_y;
	b.region_origin_z = region_origin_z;
	b.region_size_width = region_size_width;
	b.region_size_height = region_size_height;
	b.region_size_depth = region_size_depth;
	b.color_bits[0] = f64bits(red);
	b.color_bits[1] = f64bits(green);
	b.color_bits[2] = f64bits(blue);
	b.color_bits[3] = f64bits(alpha);
	b.pixel_format = pixel_format;
	appendOp(RGFX_OP_BLIT_FILL_TEXTURE_COLOR, &b, sizeof(b));
}

void RgfxEncoder::blitGenerateMipmaps(RgfxObjId texture)
{
	RgfxBlitOpGenerateMipmaps b = {};
	b.texture_obj_id = texture;
	appendOp(RGFX_OP_BLIT_GENERATE_MIPMAPS, &b, sizeof(b));
}

void RgfxEncoder::blitInvalidateCompressedTexture(RgfxObjId texture)
{
	RgfxBlitOpInvalidateCompressedTexture b = {};
	b.texture_obj_id = texture;
	appendOp(RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::blitInvalidateCompressedTextureSlice(RgfxObjId texture, uint64_t slice,
                                                       uint64_t level)
{
	RgfxBlitOpInvalidateCompressedTextureSlice b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.level = level;
	appendOp(RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE_SLICE, &b, sizeof(b));
}

void RgfxEncoder::blitOptimizeForCpu(RgfxObjId texture)
{
	RgfxBlitOpOptimizeForCpu b = {};
	b.texture_obj_id = texture;
	appendOp(RGFX_OP_BLIT_OPTIMIZE_FOR_CPU, &b, sizeof(b));
}

void RgfxEncoder::blitOptimizeForCpuSlice(RgfxObjId texture, uint64_t slice, uint64_t level)
{
	RgfxBlitOpOptimizeForCpuSlice b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.level = level;
	appendOp(RGFX_OP_BLIT_OPTIMIZE_FOR_CPU_SLICE, &b, sizeof(b));
}

void RgfxEncoder::blitOptimizeForGpu(RgfxObjId texture)
{
	RgfxBlitOpOptimizeForGpu b = {};
	b.texture_obj_id = texture;
	appendOp(RGFX_OP_BLIT_OPTIMIZE_FOR_GPU, &b, sizeof(b));
}

void RgfxEncoder::blitOptimizeForGpuSlice(RgfxObjId texture, uint64_t slice, uint64_t level)
{
	RgfxBlitOpOptimizeForGpuSlice b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.level = level;
	appendOp(RGFX_OP_BLIT_OPTIMIZE_FOR_GPU_SLICE, &b, sizeof(b));
}

void RgfxEncoder::blitOptimizeIndirectCommandBuffer(RgfxObjId icb, uint64_t range_location,
                                                    uint64_t range_length)
{
	RgfxBlitOpOptimizeIndirectCommandBuffer b = {};
	b.icb_obj_id = icb;
	b.range_location = range_location;
	b.range_length = range_length;
	appendOp(RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitResetCommandsInBuffer(RgfxObjId icb, uint64_t range_location,
                                            uint64_t range_length)
{
	RgfxBlitOpResetCommandsInBuffer b = {};
	b.icb_obj_id = icb;
	b.range_location = range_location;
	b.range_length = range_length;
	appendOp(RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER, &b, sizeof(b));
}

void RgfxEncoder::blitSynchronizeResource(RgfxObjId resource, uint32_t resource_kind)
{
	RgfxBlitOpSynchronizeResource b = {};
	b.resource_obj_id = resource;
	b.resource_kind = resource_kind;
	appendOp(RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE, &b, sizeof(b));
}

void RgfxEncoder::blitSynchronizeTexture(RgfxObjId texture, uint64_t slice, uint64_t level)
{
	RgfxBlitOpSynchronizeTexture b = {};
	b.texture_obj_id = texture;
	b.slice = slice;
	b.level = level;
	appendOp(RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE, &b, sizeof(b));
}

void RgfxEncoder::blitUpdateFence(RgfxObjId fence)
{
	RgfxBlitOpUpdateFence b = {};
	b.fence_obj_id = fence;
	appendOp(RGFX_OP_BLIT_UPDATE_FENCE, &b, sizeof(b));
}

void RgfxEncoder::blitWaitForFence(RgfxObjId fence)
{
	RgfxBlitOpWaitForFence b = {};
	b.fence_obj_id = fence;
	appendOp(RGFX_OP_BLIT_WAIT_FOR_FENCE, &b, sizeof(b));
}
