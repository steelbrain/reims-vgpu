/*
 * RgfxEncoder: turns a guest command buffer's encoder operations into the
 * reims-graphics op-stream wire format and posts the whole command buffer on a
 * queue's submission ring (docs/graphics-kext/design.md, "Stage 3 wire protocol:
 * the Metal-call stream"). It mirrors the host command-buffer decoder it feeds,
 * crates/reims-vgpu/src/graphics/decode/{cmdbuf,render,compute,blit}.rs, byte for
 * byte.
 *
 * The builder opens one `RGFX_CMD_COMMAND_BUFFER` message for a (queue, command
 * buffer) pair, then the guest appends encoder passes and, inside each, the
 * encoder's commands, and finally posts it:
 *   - `beginRenderPass`/`beginComputePass`/`beginBlitPass` and `endEncoder`
 *     write the framing ops (`RGFX_OP_ENCODER_*`), and `signalEvent`/`waitEvent`
 *     the buffer's own event ops between encoders (`RGFX_OP_SIGNAL_EVENT`,
 *     `RGFX_OP_WAIT_EVENT`);
 *   - one method per render (`render*`), compute (`compute*`) and blit (`blit*`)
 *     op writes an `RgfxOpHeader` then that op's body, 8-byte aligned, exactly as
 *     the host reads it; array-tailed ops take a count and an element array and
 *     write head then elements;
 *   - `finish` fills `op_bytes` and posts via the ring writer.
 *
 * Every wire write goes through a struct type so the layout stays locked. The
 * render-encoder op bodies (`RgfxOp*`), the compute-encoder op bodies
 * (`RgfxComputeOp*`) and the blit-encoder op bodies (`RgfxBlitOp*`), together with
 * the pass and element records, all come from the committed header
 * (`reims_graphics_wire.h`), generated from the crates/reims-graphics/src/ops
 * modules. The
 * op structs carry a per-group prefix because several names (`UpdateFence`,
 * `WaitForFence`, `ExecuteCommandsInBuffer`, ...) recur across the render, compute
 * and blit groups with different fields, which a flat C namespace cannot hold.
 * RgfxWireLayout's static_asserts pin every size against the header.
 *
 * This is our own code: it knows nothing of Apple's Metal SPI. The Metal device
 * shell (built later) translates each encoder selector into these plain fields and
 * calls the matching method while the matching encoder is open.
 */

#ifndef RGFX_ENCODER_HPP_
#define RGFX_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "RgfxMemoryShadow.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "reims_graphics_wire.h"

/// A block of the device region a command buffer's message lies in, from its queue's pool
/// (RgfxMessageBlocks): its region offset, its CPU address and its size.
struct RgfxMessageBlock {
	uint64_t offset;
	volatile uint8_t *ptr;
	uint64_t size;
};

/// Where a command buffer's message goes once it is too large for its queue's ring: blocks of
/// the device region, which the host reads it from (COMMAND_BUFFER_IN_REGION). In the bundle,
/// the queue's RgfxBodyPool (RgfxUploads.h). Locked: any thread's encoder takes and gives back.
class RgfxMessageBlocks {
public:
	/// A block of at least `size` bytes; false where the region has no room.
	virtual bool take(uint64_t size, RgfxMessageBlock *block) = 0;
	/// Takes back a block no message posted is in, or whose command buffer is done.
	virtual void giveBack(const RgfxMessageBlock &block) = 0;

protected:
	~RgfxMessageBlocks() = default;
};

/// The kinds of render state an encoder sets that Metal gives a defined default at each
/// encoder's start (MTLRenderCommandEncoder.h), each with the op that sets it: what a parallel
/// pass sets back between its sub-encoders (RgfxEncoder::appendSegment). Bindings are not among
/// them: a function that reads an index nothing was bound at is outside Metal's contract.
enum : uint32_t {
	kRgfxRenderStateCullMode = 1u << 0,          // None
	kRgfxRenderStateFrontFacing = 1u << 1,       // Clockwise
	kRgfxRenderStateFillMode = 1u << 2,          // Fill
	kRgfxRenderStateDepthClipMode = 1u << 3,     // Clip
	kRgfxRenderStateDepthBias = 1u << 4,         // 0, 0, 0
	kRgfxRenderStateBlendColor = 1u << 5,        // 0, 0, 0, 0
	kRgfxRenderStateStencilReference = 1u << 6,  // 0, 0
	kRgfxRenderStateViewport = 1u << 7,          // the target's size, depth 0 to 1
	kRgfxRenderStateScissor = 1u << 8,           // the target's size
	kRgfxRenderStateDepthStencil = 1u << 9,      // nil: the default depth-stencil state
	kRgfxRenderStateVisibility = 1u << 10,       // Disabled, offset 0
	kRgfxRenderStateTessellationScale = 1u << 11 // 1.0
};

/// The size of a render pass's target, which Metal's default viewport and scissor rectangle
/// cover: what a parallel pass's sub-encoder starts from (RgfxEncoder::appendSegment).
struct RgfxRenderTargetSize {
	uint64_t width;
	uint64_t height;
};

/// Builds one `RGFX_CMD_COMMAND_BUFFER` message — its header, then a stream of
/// encoder ops — and posts it on a queue's submission ring. One encoder is open
/// at a time, between a `begin*` and `endEncoder`; the op methods are called while
/// the matching encoder is open, as the host's walk expects (`decode::cmdbuf`).
///
/// The message is written once, where it is posted from: a buffer of the builder's own while
/// it fits its queue's ring, else, once it outgrows the ring, a block of the queue's region
/// pool (setMessageBlocks), where it grows by doubling and from which the host reads it with no
/// copy of the guest's. Each op takes its whole length at once (a pointer bump) and is stored
/// with fixed-size copies. Room before the header takes the SURFACE_USE ops a commit puts
/// first (prependSurfaceUses), so they move nothing.
///
/// A segment (SegmentOf) is a stream of ops alone, with no header: a parallel render pass's
/// sub-encoder writes one of its own, on its own thread, and the pass joins them into its
/// command buffer's stream in the order they were made (appendSegment).
///
/// Thread safety: none; one thread writes a builder at a time, as Metal's command buffer and
/// encoders are used. A segment is its own builder.
class RgfxEncoder {
public:
	/// A builder that will post to `ring` (a queue's submission ring) a command
	/// buffer `cb_obj_id` on the queue `queue_obj_id`. `ring` outlives the builder.
	/// The message has room for `reserve` bytes before it grows: its queue's guess
	/// (RgfxCommandQueue's rgfxMessageReserve), so its ops are not copied as it grows.
	RgfxEncoder(RgfxRingWriter &ring, RgfxObjId queue_obj_id, RgfxObjId cb_obj_id,
	            size_t reserve = 0);

	/// What a segment is made of: the builder whose command buffer it is part of.
	struct SegmentOf {
		const RgfxEncoder &parent;
	};
	/// A segment of `of.parent`'s command buffer: an op stream with no header, which keeps what
	/// it names where the parent does (setKeeper, keepStamp), and is joined into the parent by
	/// appendSegment.
	explicit RgfxEncoder(SegmentOf of);

	~RgfxEncoder();
	RgfxEncoder(const RgfxEncoder &) = delete;
	RgfxEncoder &operator=(const RgfxEncoder &) = delete;

	/// Where the message moves once it no longer fits the ring: `blocks`, taking a block of at
	/// least `hint` bytes first (the queue's guess of the message's size). A message starts in
	/// a buffer of the builder's own whatever the guess, so one that fits the ring rides it; it
	/// is copied once, when it outgrows the ring, at most the ring's half.
	void setMessageBlocks(std::shared_ptr<RgfxMessageBlocks> blocks, uint64_t hint)
	{
		m_blocks = std::move(blocks);
		m_blockHint = hint;
	}

	// --- framing --------------------------------------------------------------

	/// Opens a render encoder for `desc` and its `color_count`
	/// [`RgfxRenderPassColorAttachment`] records, then the pass's sample positions where
	/// `positions` is not null, then its visibility result buffer where `visibility` is not null.
	/// `desc.color_count` is set from `color_count`, `RGFX_PASS_FLAG_SAMPLE_POSITIONS` from
	/// `positions` and `RGFX_PASS_FLAG_VISIBILITY_RESULT` from `visibility`, so each always agrees
	/// with what follows.
	void beginRenderPass(RgfxRenderPassDesc desc, const RgfxRenderPassColorAttachment *colors,
	                     uint32_t color_count,
	                     const RgfxRenderPassSamplePositions *positions = nullptr,
	                     const RgfxRenderPassVisibility *visibility = nullptr);
	/// Opens a compute encoder for Metal's default pass (empty body).
	void beginComputePass();
	/// Opens a compute encoder whose pass sets `MTLDispatchType` `dispatch_type`.
	void beginComputePass(uint32_t dispatch_type);
	/// Opens a blit encoder (no body).
	void beginBlitPass();
	/// Opens a video encoder on the video session `session` (RGFX_OP_ENCODER_BEGIN_VIDEO), the
	/// command buffer's only encoder.
	void beginVideoPass(RgfxObjId session);
	/// Ends the open encoder (`endEncoding`).
	void endEncoder();
	/// The buffer signals `event` (an event or a shared event) to `value`
	/// (`encodeSignalEvent:value:`), between encoders.
	void signalEvent(RgfxObjId event, uint64_t value);
	/// The buffer's later work waits until `event` reaches `value`
	/// (`encodeWaitForEvent:value:`), between encoders.
	void waitEvent(RgfxObjId event, uint64_t value);

	// --- video ops (RGFX_OP_VIDEO_*, group 4) ----------------------------------

	/// Decodes an H.264 access unit (RGFX_OP_VIDEO_DECODE_H264): `body` then the `size` bytes of
	/// SPS and PPS NAL units at `parameterSets` (none where `size` is 0), padded to whole words.
	void videoDecodeH264(const RgfxVideoOpDecodeH264 &body, const uint8_t *parameterSets,
	                     uint32_t size);
	/// Decodes an HEVC access unit (RGFX_OP_VIDEO_DECODE_H265), as videoDecodeH264 does: `body`
	/// then the `size` bytes of parameter sets at `parameterSets`, padded to whole words.
	void videoDecodeH265(const RgfxVideoOpDecodeH265 &body, const uint8_t *parameterSets,
	                     uint32_t size);

	// --- render ops (RGFX_OP_RENDER_*, group 1) --------------------------------

	void renderSetRenderPipelineState(RgfxObjId pipeline);
	void renderSetVertexBuffer(RgfxObjId buffer, uint64_t offset, uint64_t index);
	void renderSetFragmentBuffer(RgfxObjId buffer, uint64_t offset, uint64_t index);
	void renderSetFragmentTexture(RgfxObjId texture, uint64_t index);
	void renderSetViewport(double origin_x, double origin_y, double width, double height,
	                       double znear, double zfar);
	void renderSetScissorRect(uint64_t x, uint64_t y, uint64_t width, uint64_t height);
	void renderDrawPrimitives(uint64_t primitive_type, uint64_t vertex_start,
	                          uint64_t vertex_count);
	void renderDrawIndexedPrimitives(uint64_t primitive_type, uint64_t index_count,
	                                 uint64_t index_type, RgfxObjId index_buffer,
	                                 uint64_t index_buffer_offset);
	void renderDrawPrimitivesInstances(uint64_t primitive_type, uint64_t vertex_start,
	                                   uint64_t vertex_count, uint64_t instance_count);
	void renderDrawPrimitivesBase(uint64_t primitive_type, uint64_t vertex_start,
	                              uint64_t vertex_count, uint64_t instance_count,
	                              uint64_t base_instance);
	void renderDrawPrimitivesIndirect(uint64_t primitive_type, RgfxObjId indirect_buffer,
	                                  uint64_t indirect_buffer_offset);
	void renderDrawIndexedPrimitivesInstances(uint64_t primitive_type, uint64_t index_count,
	                                          uint64_t index_type, RgfxObjId index_buffer,
	                                          uint64_t index_buffer_offset,
	                                          uint64_t instance_count);
	void renderDrawIndexedPrimitivesBase(uint64_t primitive_type, uint64_t index_count,
	                                     uint64_t index_type, RgfxObjId index_buffer,
	                                     uint64_t index_buffer_offset, uint64_t instance_count,
	                                     int64_t base_vertex, uint64_t base_instance);
	void renderDrawIndexedPrimitivesIndirect(uint64_t primitive_type, uint64_t index_type,
	                                         RgfxObjId index_buffer, uint64_t index_buffer_offset,
	                                         RgfxObjId indirect_buffer,
	                                         uint64_t indirect_buffer_offset);
	void renderDrawPatches(uint64_t number_of_patch_control_points, uint64_t patch_start,
	                       uint64_t patch_count, RgfxObjId patch_index_buffer,
	                       uint64_t patch_index_buffer_offset, uint64_t instance_count,
	                       uint64_t base_instance);
	void renderDrawPatchesIndirect(uint64_t number_of_patch_control_points,
	                               RgfxObjId patch_index_buffer, uint64_t patch_index_buffer_offset,
	                               RgfxObjId indirect_buffer, uint64_t indirect_buffer_offset);
	void renderDrawIndexedPatches(uint64_t number_of_patch_control_points, uint64_t patch_start,
	                              uint64_t patch_count, RgfxObjId patch_index_buffer,
	                              uint64_t patch_index_buffer_offset,
	                              RgfxObjId control_point_index_buffer,
	                              uint64_t control_point_index_buffer_offset,
	                              uint64_t instance_count, uint64_t base_instance);
	void renderDrawIndexedPatchesIndirect(uint64_t number_of_patch_control_points,
	                                      RgfxObjId patch_index_buffer,
	                                      uint64_t patch_index_buffer_offset,
	                                      RgfxObjId control_point_index_buffer,
	                                      uint64_t control_point_index_buffer_offset,
	                                      RgfxObjId indirect_buffer,
	                                      uint64_t indirect_buffer_offset);
	void renderDispatchThreadsPerTile(uint64_t width, uint64_t height, uint64_t depth);
	void renderDispatchThreadsPerTileInRegion(uint64_t width, uint64_t height, uint64_t depth,
	                                          uint64_t origin_x, uint64_t origin_y,
	                                          uint64_t origin_z, uint64_t region_width,
	                                          uint64_t region_height, uint64_t region_depth);
	void renderDispatchThreadsPerTileInRegionIndex(uint64_t width, uint64_t height, uint64_t depth,
	                                               uint64_t origin_x, uint64_t origin_y,
	                                               uint64_t origin_z, uint64_t region_width,
	                                               uint64_t region_height, uint64_t region_depth,
	                                               uint32_t render_target_array_index);
	void renderExecuteCommandsInBuffer(RgfxObjId indirect_command_buffer, uint64_t range_location,
	                                   uint64_t range_length);
	void renderExecuteCommandsIndirect(RgfxObjId indirect_command_buffer,
	                                   RgfxObjId indirect_range_buffer,
	                                   uint64_t indirect_buffer_offset);
	void renderMemoryBarrierWithScope(uint64_t scope, uint64_t after, uint64_t before);
	void renderUpdateFence(RgfxObjId fence, uint64_t stages);
	void renderWaitForFence(RgfxObjId fence, uint64_t stages);
	void renderSetCullMode(uint64_t cull_mode);
	void renderSetFrontFacingWinding(uint64_t winding);
	void renderSetTriangleFillMode(uint64_t fill_mode);
	void renderSetDepthClipModeSpi(uint64_t mode);
	void renderSetProvokingVertexMode(uint64_t mode);
	void renderSetDepthStoreAction(uint64_t store_action);
	void renderSetStencilStoreAction(uint64_t store_action);
	void renderSetStencilStoreActionOptions(uint64_t store_action_options);
	void renderSetDepthStoreActionOptions(uint64_t store_action_options);
	void renderSetColorStoreAction(uint64_t store_action, uint64_t color_attachment_index);
	void renderSetColorStoreActionOptions(uint64_t store_action_options,
	                                      uint64_t color_attachment_index);
	void renderSetVisibilityResultMode(uint64_t mode, uint64_t offset);
	void renderSetVertexAmplificationMode(uint64_t mode, uint64_t value);
	void renderSetVertexBufferOffset(uint64_t offset, uint64_t index);
	void renderSetFragmentBufferOffset(uint64_t offset, uint64_t index);
	void renderSetTileBufferOffset(uint64_t offset, uint64_t index);
	void renderSetVertexBufferOffsetAttributeStride(uint64_t offset, uint64_t stride,
	                                                uint64_t index);
	void renderSetThreadgroupMemoryLength(uint64_t length, uint64_t offset, uint64_t index);
	void renderSetAlphaTestReferenceValue(float reference_value);
	void renderSetLineWidth(float line_width);
	void renderSetPointSize(float point_size);
	void renderSetTessellationFactorScale(float scale);
	void renderSetBlendColor(float red, float green, float blue, float alpha);
	void renderSetDepthBias(float depth_bias, float slope_scale, float clamp);
	void renderSetViewportTransformEnabled(bool enabled);
	void renderSetPrimitiveRestartEnabled(uint64_t index, bool enabled);
	void renderSetStencilReferenceValues(uint32_t front_reference_value,
	                                     uint32_t back_reference_value);
	void renderSetDepthCleared();   // no body
	void renderSetStencilCleared(); // no body
	void renderSetDepthStencilState(RgfxObjId depth_stencil_state);
	void renderSetColorResolveTexture(RgfxObjId texture, uint64_t slice, uint64_t depth_plane,
	                                  uint64_t level, uint64_t index, uint32_t y_invert);
	void renderSetDepthResolveTexture(RgfxObjId texture, uint64_t slice, uint64_t depth_plane,
	                                  uint64_t level, uint32_t y_invert);
	void renderSetStencilResolveTexture(RgfxObjId texture, uint64_t slice, uint64_t depth_plane,
	                                    uint64_t level, uint32_t y_invert);
	void renderSetTessellationFactorBuffer(RgfxObjId buffer, uint64_t offset,
	                                       uint64_t instance_stride);
	void renderSetVertexBuffers(uint64_t range_location, const RgfxBufferBinding *bindings,
	                            uint32_t count);
	void renderSetFragmentBuffers(uint64_t range_location, const RgfxBufferBinding *bindings,
	                              uint32_t count);
	void renderSetTileBuffers(uint64_t range_location, const RgfxBufferBinding *bindings,
	                          uint32_t count);
	void renderSetVertexBuffersAttributeStrides(uint64_t range_location,
	                                            const RgfxBufferStrideBinding *bindings,
	                                            uint32_t count);
	void renderSetVertexTextures(uint64_t range_location, const RgfxTextureBinding *textures,
	                             uint32_t count);
	void renderSetFragmentTextures(uint64_t range_location, const RgfxTextureBinding *textures,
	                               uint32_t count);
	void renderSetTileTextures(uint64_t range_location, const RgfxTextureBinding *textures,
	                           uint32_t count);
	void renderSetVertexSamplerStates(uint64_t range_location, const RgfxSamplerBinding *samplers,
	                                  uint32_t count);
	void renderSetFragmentSamplerStates(uint64_t range_location, const RgfxSamplerBinding *samplers,
	                                    uint32_t count);
	void renderSetTileSamplerStates(uint64_t range_location, const RgfxSamplerBinding *samplers,
	                                uint32_t count);
	void renderSetVertexSamplerStatesLodClamps(uint64_t range_location,
	                                           const RgfxSamplerLodBinding *samplers,
	                                           uint32_t count);
	void renderSetFragmentSamplerStatesLodClamps(uint64_t range_location,
	                                             const RgfxSamplerLodBinding *samplers,
	                                             uint32_t count);
	void renderSetTileSamplerStatesLodClamps(uint64_t range_location,
	                                         const RgfxSamplerLodBinding *samplers, uint32_t count);
	void renderSetScissorRects(const RgfxScissorRectElem *rects, uint32_t count);
	void renderSetViewports(const RgfxViewportElem *viewports, uint32_t count);
	void renderSetVertexAmplificationCount(const RgfxViewMapping *mappings, uint32_t count);
	void renderUseHeaps(const RgfxHeapBinding *heaps, uint32_t count);
	void renderUseHeapsStages(const RgfxHeapBinding *heaps, uint32_t count, uint64_t stages);
	void renderUseResources(const RgfxResourceBinding *resources, uint32_t count, uint64_t usage);
	void renderUseResourcesStages(const RgfxResourceBinding *resources, uint32_t count,
	                              uint64_t usage, uint64_t stages);

	// --- compute ops (RGFX_OP_COMPUTE_*, group 2) ------------------------------

	void computeSetComputePipelineState(RgfxObjId pipeline);
	void computeSetBufferOffset(uint64_t offset, uint64_t index);
	void computeSetBufferOffsetAttributeStride(uint64_t offset, uint64_t stride, uint64_t index);
	void computeSetBuffers(uint64_t location, const RgfxComputeBufferBinding *bindings,
	                       uint32_t count);
	void computeSetBuffersAttributeStrides(
	    uint64_t location, const RgfxComputeBufferStrideBinding *bindings,
	    uint32_t count);
	void computeSetTexture(RgfxObjId texture, uint64_t index);
	void computeSetTextures(uint64_t location,
	                        const RgfxComputeTextureBinding *textures, uint32_t count);
	void computeSetSamplerStates(uint64_t location,
	                             const RgfxComputeSamplerBinding *samplers,
	                             uint32_t count);
	void computeSetSamplerStatesLodClamps(
	    uint64_t location, const RgfxComputeSamplerLodBinding *samplers,
	    uint32_t count);
	void computeSetThreadgroupMemoryLength(uint64_t length, uint64_t index);
	void computeSetStageInRegion(uint64_t origin_x, uint64_t origin_y, uint64_t origin_z,
	                             uint64_t width, uint64_t height, uint64_t depth);
	void computeSetStageInRegionIndirect(RgfxObjId indirect_buffer,
	                                     uint64_t indirect_buffer_offset);
	void computeSetImageblockWidth(uint64_t width, uint64_t height);
	void computeDispatchThreadgroups(uint64_t threadgroups_width, uint64_t threadgroups_height,
	                                 uint64_t threadgroups_depth, uint64_t threads_width,
	                                 uint64_t threads_height, uint64_t threads_depth);
	void computeDispatchThreadgroupsIndirect(RgfxObjId indirect_buffer,
	                                         uint64_t indirect_buffer_offset, uint64_t threads_width,
	                                         uint64_t threads_height, uint64_t threads_depth);
	void computeDispatchThreads(uint64_t grid_width, uint64_t grid_height, uint64_t grid_depth,
	                            uint64_t threads_width, uint64_t threads_height,
	                            uint64_t threads_depth);
	void computeDispatchThreadsIndirect(RgfxObjId indirect_buffer, uint64_t indirect_buffer_offset);
	void computeUseResources(const RgfxComputeResourceRef *resources, uint32_t count,
	                         uint64_t usage);
	void computeUseHeaps(const RgfxComputeHeapBinding *heaps, uint32_t count);
	void computeMemoryBarrierResources(const RgfxComputeResourceRef *resources,
	                                   uint32_t count);
	void computeMemoryBarrierScope(uint64_t scope);
	void computeUpdateFence(RgfxObjId fence);
	void computeWaitForFence(RgfxObjId fence);
	void computeExecuteCommandsIndirect(RgfxObjId indirect_command_buffer,
	                                    RgfxObjId indirect_range_buffer,
	                                    uint64_t indirect_buffer_offset);
	void computeExecuteCommandsInBuffer(RgfxObjId indirect_command_buffer, uint64_t location,
	                                    uint64_t length);
	void computeEncodeStartIf(RgfxObjId buffer, uint64_t offset, uint64_t comparison,
	                          uint32_t reference_value);
	void computeEncodeStartWhile(RgfxObjId buffer, uint64_t offset, uint64_t comparison,
	                             uint32_t reference_value);
	void computeEncodeStartElse();                       // no body
	void computeEncodeStartDoWhile();                    // no body
	void computeCompressedTextureReinterpretationFlush(); // no body

	// --- blit ops (RGFX_OP_BLIT_*, group 3) ------------------------------------

	void blitCopyBufferToBuffer(RgfxObjId source_buffer, RgfxObjId destination_buffer,
	                            uint64_t source_offset, uint64_t destination_offset, uint64_t size);
	void blitCopyBufferToTexture(RgfxObjId source_buffer, RgfxObjId destination_texture,
	                             uint64_t source_offset, uint64_t source_bytes_per_row,
	                             uint64_t source_bytes_per_image, uint64_t source_size_width,
	                             uint64_t source_size_height, uint64_t source_size_depth,
	                             uint64_t destination_slice, uint64_t destination_level,
	                             uint64_t destination_origin_x, uint64_t destination_origin_y,
	                             uint64_t destination_origin_z);
	void blitCopyBufferToTextureOptions(
	    RgfxObjId source_buffer, RgfxObjId destination_texture, uint64_t source_offset,
	    uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_size_width,
	    uint64_t source_size_height, uint64_t source_size_depth, uint64_t destination_slice,
	    uint64_t destination_level, uint64_t destination_origin_x, uint64_t destination_origin_y,
	    uint64_t destination_origin_z, uint64_t options);
	void blitCopyTextureToBuffer(RgfxObjId source_texture, RgfxObjId destination_buffer,
	                             uint64_t source_slice, uint64_t source_level,
	                             uint64_t source_origin_x, uint64_t source_origin_y,
	                             uint64_t source_origin_z, uint64_t source_size_width,
	                             uint64_t source_size_height, uint64_t source_size_depth,
	                             uint64_t destination_offset, uint64_t destination_bytes_per_row,
	                             uint64_t destination_bytes_per_image);
	void blitCopyTextureToBufferOptions(
	    RgfxObjId source_texture, RgfxObjId destination_buffer, uint64_t source_slice,
	    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
	    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
	    uint64_t source_size_depth, uint64_t destination_offset, uint64_t destination_bytes_per_row,
	    uint64_t destination_bytes_per_image, uint64_t options);
	void blitCopyTextureRegion(RgfxObjId source_texture, RgfxObjId destination_texture,
	                           uint64_t source_slice, uint64_t source_level, uint64_t source_origin_x,
	                           uint64_t source_origin_y, uint64_t source_origin_z,
	                           uint64_t source_size_width, uint64_t source_size_height,
	                           uint64_t source_size_depth, uint64_t destination_slice,
	                           uint64_t destination_level, uint64_t destination_origin_x,
	                           uint64_t destination_origin_y, uint64_t destination_origin_z);
	void blitCopyTextureRegionOptions(
	    RgfxObjId source_texture, RgfxObjId destination_texture, uint64_t source_slice,
	    uint64_t source_level, uint64_t source_origin_x, uint64_t source_origin_y,
	    uint64_t source_origin_z, uint64_t source_size_width, uint64_t source_size_height,
	    uint64_t source_size_depth, uint64_t destination_slice, uint64_t destination_level,
	    uint64_t destination_origin_x, uint64_t destination_origin_y, uint64_t destination_origin_z,
	    uint64_t options);
	void blitCopyTextureSlices(RgfxObjId source_texture, RgfxObjId destination_texture,
	                           uint64_t source_slice, uint64_t source_level,
	                           uint64_t destination_slice, uint64_t destination_level,
	                           uint64_t slice_count, uint64_t level_count);
	void blitCopyIndirectCommandBuffer(RgfxObjId source, RgfxObjId destination,
	                                   uint64_t source_range_location, uint64_t source_range_length,
	                                   uint64_t destination_index);
	void blitFillBuffer(RgfxObjId buffer, uint64_t range_location, uint64_t range_length,
	                    uint8_t value);
	void blitFillBufferPattern4(RgfxObjId buffer, uint64_t range_location, uint64_t range_length,
	                            uint32_t pattern);
	/// `fillTexture:...:bytes:length:`. The `length` fill bytes trail the fixed head
	/// at [`RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET`]; the op is padded to 8 bytes.
	void blitFillTexture(RgfxObjId texture, uint64_t level, uint64_t slice, uint64_t region_origin_x,
	                     uint64_t region_origin_y, uint64_t region_origin_z,
	                     uint64_t region_size_width, uint64_t region_size_height,
	                     uint64_t region_size_depth, const void *bytes, uint64_t length);
	void blitFillTextureColor(RgfxObjId texture, uint64_t level, uint64_t slice,
	                          uint64_t region_origin_x, uint64_t region_origin_y,
	                          uint64_t region_origin_z, uint64_t region_size_width,
	                          uint64_t region_size_height, uint64_t region_size_depth,
	                          double red, double green, double blue, double alpha,
	                          uint64_t pixel_format);
	void blitGenerateMipmaps(RgfxObjId texture);
	void blitInvalidateCompressedTexture(RgfxObjId texture);
	void blitInvalidateCompressedTextureSlice(RgfxObjId texture, uint64_t slice, uint64_t level);
	void blitOptimizeForCpu(RgfxObjId texture);
	void blitOptimizeForCpuSlice(RgfxObjId texture, uint64_t slice, uint64_t level);
	void blitOptimizeForGpu(RgfxObjId texture);
	void blitOptimizeForGpuSlice(RgfxObjId texture, uint64_t slice, uint64_t level);
	void blitOptimizeIndirectCommandBuffer(RgfxObjId icb, uint64_t range_location,
	                                       uint64_t range_length);
	void blitResetCommandsInBuffer(RgfxObjId icb, uint64_t range_location, uint64_t range_length);
	void blitSynchronizeResource(RgfxObjId resource, uint32_t resource_kind);
	void blitSynchronizeTexture(RgfxObjId texture, uint64_t slice, uint64_t level);
	void blitUpdateFence(RgfxObjId fence);
	void blitWaitForFence(RgfxObjId fence);

	// --- finish ---------------------------------------------------------------

	/// Sets `op_bytes` and posts the command buffer, waiting for room
	/// (RgfxRingWriter::post, `whileWaiting` called as it waits): on the ring where the message
	/// fits it (fitsRing()), else as a COMMAND_BUFFER_IN_REGION naming its block, the one it was
	/// built in, or one taken now from setMessageBlocks's pool, into which it is copied. The
	/// block must stay the process's until the buffer completes: it goes to `*held` (when not
	/// null) before the message is posted, whether or not the post succeeds, and its holder
	/// gives it back to the pool then; else the builder holds it until it goes. Returns whether the message was posted: false for one no block could be had for,
	/// or too long for a header's `size`. On success `*out_seqno` (when not null) holds the
	/// stamped seqno, which the completion carries. `beforePublish` gets the seqno before the
	/// host can see the buffer (RgfxRingWriter::tryPost), so its completion is expected first.
	bool finish(uint64_t *out_seqno = nullptr, const std::function<void()> &whileWaiting = {},
	            const std::function<void(uint64_t)> &beforePublish = {},
	            RgfxMessageBlock *held = nullptr);

	/// Whether the finished message goes on the ring (RgfxRingWriter::fits), not in a block.
	bool fitsRing() const;

	/// The finished message's length: the header, the op stream and the surface uses before it.
	uint64_t messageSize() const { return RGFX_COMMAND_BUFFER_OPS_OFFSET + opBytes(); }

	/// The bytes the message holds before it grows.
	size_t capacity() const
	{
		return static_cast<size_t>(m_end - m_ops) + RGFX_COMMAND_BUFFER_OPS_OFFSET + m_prefix;
	}

	/// Whether the message is being built in a block of the region pool (setMessageBlocks).
	bool inBlock() const { return m_inBlock; }

	/// The op-stream bytes appended so far (the message size less the header).
	uint64_t opBytes() const { return static_cast<uint64_t>(m_cur - m_ops) + m_prefix; }

	/// The op stream built so far, opBytes() long: the surface uses put first, then the ops.
	/// For tests and diagnostics.
	const uint8_t *ops() const { return m_ops - m_prefix; }

	// --- IOSurface copies this command buffer writes -------------------------

	/// Records that the command buffer writes the IOSurface copy `shadow` (opaque
	/// here: the Metal shell's RgfxSurfaceShadow), which its completion writes back
	/// to the surface's pages. Held until taken.
	void noteSurfaceWrite(std::shared_ptr<void> shadow)
	{
		if (m_surfaceWrites.empty() || m_surfaceWrites.back() != shadow)
			m_surfaceWrites.push_back(std::move(shadow));
	}
	/// The copies recorded so far, which the caller now holds.
	std::vector<std::shared_ptr<void>> takeSurfaceWrites() { return std::move(m_surfaceWrites); }
	/// Records that the command buffer uses the IOSurface view `shadow`, whose block is cached
	/// on the host (opaque here: the Metal shell's RgfxSurfaceShadow), once for a run of the
	/// same view. Held until taken.
	void noteSurfaceUse(std::shared_ptr<void> shadow)
	{
		if (m_surfaceUses.empty() || m_surfaceUses.back() != shadow)
			m_surfaceUses.push_back(std::move(shadow));
	}
	/// The uses recorded so far, which the caller now holds.
	std::vector<std::shared_ptr<void>> takeSurfaceUses() { return std::move(m_surfaceUses); }
	/// Records that the command buffer uses the IOSurface view `shadow`, a copy of the surface's
	/// pages, which its commit syncs from them where a CPU wrote them since (the Metal shell's
	/// RgfxSurfaceShadow, opaque here; its caller notes each once a command buffer, by
	/// keepStamp). Held until taken.
	void noteSurfaceSync(std::shared_ptr<void> shadow) { m_surfaceSyncs.push_back(std::move(shadow)); }
	/// The copies recorded so far, which the caller now holds.
	std::vector<std::shared_ptr<void>> takeSurfaceSyncs() { return std::move(m_surfaceSyncs); }
	/// Asks the host for the buffer's submission (`RGFX_OP_SUBMITTED`, where the device has
	/// RGFX_FEATURE_CB_SUBMITTED): it posts RGFX_EVT_CB_SUBMITTED ahead of the buffer's other
	/// records.
	void appendSubmitted() { appendOp(RGFX_OP_SUBMITTED); }
	/// Puts the `count` `RGFX_OP_SURFACE_USE` ops of bodies `uses` at the op stream's start,
	/// before its first encoder, where the host reads them before the buffer's work.
	void prependSurfaceUses(const RgfxOpSurfaceUse *uses, size_t count);
	/// Puts one RGFX_OP_SURFACE_ORDER naming `waits` (`count` of them, none among them) before
	/// the op stream's first op, as prependSurfaceUses does.
	void prependSurfaceOrder(const RgfxSurfaceOrderWait *waits, size_t count);
	/// Where the command buffer keeps the objects its encoders name, as Metal's command buffer
	/// with retained references keeps what it encodes: `keep(context, object)` for each one
	/// (once for a run of the same object), from the encode until the buffer completes or goes.
	/// A parallel pass's segments keep there too, each on its own thread, so `keep` is locked.
	/// None set: nothing is kept.
	void setKeeper(void (*keep)(void *context, const void *object), void *context)
	{
		m_keep = keep;
		m_keepContext = context;
	}
	/// Keeps `object` where the command buffer keeps what it encodes (setKeeper).
	void keep(const void *object)
	{
		if (m_keep == nullptr || object == nullptr || object == m_lastKept)
			return;
		m_lastKept = object; // kept, so no other object takes its address meanwhile
		m_keep(m_keepContext, object);
	}
	/// The command buffer's own stamp, which no other command buffer's builder has, and which
	/// its segments share: an object that records the stamp of the last builder to keep it
	/// (the Metal shell's RgfxResource) is kept once a command buffer, at its first naming.
	uint64_t keepStamp() const { return m_keepStamp; }
	/// The id of `object` where the builder named it before and it was kept (namedNote): a
	/// command buffer keeps what it names until it completes, so no other object takes its
	/// address meanwhile. A few recent objects are held, by address.
	bool namedBefore(const void *object, uint64_t *id) const
	{
		const Named &n = m_named[namedSlot(object)];
		if (n.object != object)
			return false;
		*id = n.id;
		return true;
	}
	/// Notes that `object`, named `id`, is kept (keep), and that a naming of it needs nothing
	/// noted but its id: none of its copies a commit or a completion syncs (the Metal shell's
	/// IOSurface and app-memory copies). Nothing where nothing is kept (no keeper).
	void namedNote(const void *object, uint64_t id)
	{
		if (m_keep != nullptr)
			m_named[namedSlot(object)] = {object, id};
	}
	/// Records that the command buffer uses the copy of app memory `shadow` (a no-copy
	/// buffer's; RgfxMemoryShadow.hpp), which its commit syncs in, and, where `writes`, its
	/// completion writes back. Held until taken.
	void noteShadowUse(const std::shared_ptr<RgfxMemoryShadow> &shadow, bool writes)
	{
		if (!m_shadowUses.empty() && m_shadowUses.back().shadow == shadow) {
			m_shadowUses.back().writes = m_shadowUses.back().writes || writes;
			return;
		}
		m_shadowUses.push_back({shadow, writes});
	}
	/// The uses recorded so far, which the caller now holds.
	std::vector<RgfxShadowUse> takeShadowUses() { return std::move(m_shadowUses); }

	// --- parallel render passes ---------------------------------------------------

	/// The render state, of the kinds Metal gives a defined default (kRgfxRenderState*), that
	/// this builder's ops have set: what a parallel pass resets after the segment.
	uint32_t renderStateSet() const { return m_renderStateSet; }

	/// Joins `segment` (one of this command buffer's, SegmentOf) at the stream's end: its ops,
	/// then the surface and app-memory uses it recorded, which this builder now holds. Before
	/// its ops, the render state an earlier segment set (`reset`, kRgfxRenderState*) is set
	/// back to Metal's defaults with the ops that set it, the viewport and scissor rectangle
	/// to `target`'s whole size, as each sub-encoder of a parallel pass starts from them.
	void appendSegment(RgfxEncoder &segment, uint32_t reset, const RgfxRenderTargetSize &target);

private:
	/// Room for `bytes` more of the prefix, before the ops: the ops move up where the room the
	/// header leaves is short. Where the bytes go.
	uint8_t *prependRoom(size_t bytes);
	/// The bytes before the header a commit's surface uses take with no move: 64 of them.
	static constexpr size_t kHeadroom = 64 * (RGFX_OP_BODY_OFFSET + sizeof(RgfxOpSurfaceUse));

	// Takes `n` bytes at the stream's end (growing it where it has less room) and answers them.
	__attribute__((always_inline)) uint8_t *reserve(size_t n)
	{
		if (__builtin_expect(static_cast<size_t>(m_end - m_cur) < n, 0))
			grow(n);
		uint8_t *at = m_cur;
		m_cur += n;
		return at;
	}
	// Writes an op's header, of `len` bytes in all, at `at`.
	__attribute__((always_inline)) static void putHeader(uint8_t *at, uint16_t tag, size_t len)
	{
		RgfxOpHeader h;
		h.tag = tag;
		h.flags = 0;
		h.len = static_cast<uint32_t>(len);
		std::memcpy(at, &h, sizeof(h));
	}
	// Appends an op with no body (header only, len 8).
	__attribute__((always_inline)) void appendOp(uint16_t tag)
	{
		putHeader(reserve(RGFX_OP_BODY_OFFSET), tag, RGFX_OP_BODY_OFFSET);
	}
	// Appends an op whose body is `n` bytes at `body` (an 8-multiple); len = 8 + n.
	__attribute__((always_inline)) void appendOp(uint16_t tag, const void *body, size_t n)
	{
		uint8_t *at = reserve(RGFX_OP_BODY_OFFSET + n);
		putHeader(at, tag, RGFX_OP_BODY_OFFSET + n);
		std::memcpy(at + RGFX_OP_BODY_OFFSET, body, n);
	}
	// Appends an op whose body is `body`, a wire struct.
	template <typename T> __attribute__((always_inline)) void appendBody(uint16_t tag, const T &body)
	{
		static_assert(sizeof(T) % 8 == 0, "an op body is a multiple of 8 bytes");
		appendOp(tag, &body, sizeof(T));
	}
	// Appends an array-tailed op: `head` (`head_n` bytes) then `count` elements of
	// `stride` bytes at `elems`; len = 8 + head_n + count*stride.
	void appendArrayOp(uint16_t tag, const void *head, size_t head_n, const void *elems,
	                   uint32_t count, size_t stride);
	// Makes room for `n` more bytes: a larger buffer of the builder's own, or a block of the
	// region pool once the message outgrows the ring, its bytes so far copied there once.
	void grow(size_t n);
	// Moves the bytes so far to a block of at least `size` bytes; false, nothing moved, where
	// none was had.
	bool moveToBlock(size_t size);
	// Moves the bytes so far to a buffer of the builder's own of `size` bytes.
	void moveToHeap(size_t size);
	// Gives back the block the builder holds, if any.
	void giveBackBlock();
	// A video decode op of `tag`: its header, the `n` bytes of its body at `body`, then the
	// `size` bytes of parameter sets at `parameterSets`, padded with zeros to whole words.
	void appendVideoDecode(uint16_t tag, const void *body, size_t n, const uint8_t *parameterSets,
	                       uint32_t size);

	struct Named {
		const void *object;
		uint64_t id;
	};
	static constexpr size_t kNamedSlots = 32; // 2^(64 - 59)
	static size_t namedSlot(const void *object)
	{
		// Fibonacci hashing: objects a malloc zone lays out a power of two apart spread too.
		return static_cast<size_t>((reinterpret_cast<uint64_t>(object) * 0x9e3779b97f4a7c15ull) >> 59);
	}

	RgfxRingWriter *m_ring; // null for a segment
	RgfxCommandBuffer m_head;
	// The storage: a buffer of the builder's own (malloc) or a region block (m_inBlock), from
	// m_store, m_end the end; the op stream from m_ops (kHeadroom and the header before it,
	// none for a segment) to m_cur. The surface uses and the surface order put first lie in the
	// m_prefix bytes before m_ops (prependRoom).
	uint8_t *m_store = nullptr;
	uint8_t *m_ops = nullptr;
	uint8_t *m_cur = nullptr;
	uint8_t *m_end = nullptr;
	size_t m_prefix = 0;
	bool m_inBlock = false;
	RgfxMessageBlock m_block = {};
	std::shared_ptr<RgfxMessageBlocks> m_blocks;
	uint64_t m_blockHint = 0;
	uint32_t m_renderStateSet = 0;
	uint64_t m_keepStamp;
	std::vector<std::shared_ptr<void>> m_surfaceWrites;
	std::vector<std::shared_ptr<void>> m_surfaceUses;
	std::vector<std::shared_ptr<void>> m_surfaceSyncs;
	std::vector<RgfxShadowUse> m_shadowUses;
	void (*m_keep)(void *, const void *) = nullptr;
	void *m_keepContext = nullptr;
	const void *m_lastKept = nullptr;
	Named m_named[kNamedSlots] = {};
};

#endif // RGFX_ENCODER_HPP_
