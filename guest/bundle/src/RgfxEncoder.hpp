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
#include <functional>
#include <memory>
#include <vector>

#include "RgfxMemoryShadow.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "reims_graphics_wire.h"

/// Builds one `RGFX_CMD_COMMAND_BUFFER` message — its header, then a stream of
/// encoder ops — and posts it on a queue's submission ring. One encoder is open
/// at a time, between a `begin*` and `endEncoder`; the op methods are called while
/// the matching encoder is open, as the host's walk expects (`decode::cmdbuf`).
class RgfxEncoder {
public:
	/// A builder that will post to `ring` (a queue's submission ring) a command
	/// buffer `cb_obj_id` on the queue `queue_obj_id`. `ring` outlives the builder.
	/// The message has room for `reserve` bytes before it grows: its queue's guess
	/// (RgfxCommandQueue's rgfxMessageReserve), so its ops are not copied as it grows.
	RgfxEncoder(RgfxRingWriter &ring, RgfxObjId queue_obj_id, RgfxObjId cb_obj_id,
	            size_t reserve = 0);

	// --- framing --------------------------------------------------------------

	/// Opens a render encoder for `desc` and its `color_count`
	/// [`RgfxRenderPassColorAttachment`] records, then the pass's sample positions where
	/// `positions` is not null. `desc.color_count` is set from `color_count`, and
	/// `RGFX_PASS_FLAG_SAMPLE_POSITIONS` from `positions`, so each always agrees with what follows.
	void beginRenderPass(RgfxRenderPassDesc desc, const RgfxRenderPassColorAttachment *colors,
	                     uint32_t color_count,
	                     const RgfxRenderPassSamplePositions *positions = nullptr);
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

	/// Sets `op_bytes` and posts the command buffer on the ring, waiting for room
	/// (RgfxRingWriter::post, `whileWaiting` called as it waits). Returns whether it
	/// reached the ring: false, posting nothing, for a message the ring does not fit
	/// (fitsRing()), which goes out of line with finishInRegion. On success
	/// `*out_seqno` (when not null) holds the stamped seqno. The built message stays
	/// readable through [`message`] afterwards. `beforePublish` gets the seqno before the
	/// host can see the buffer (RgfxRingWriter::tryPost), so its completion is expected first.
	bool finish(uint64_t *out_seqno = nullptr, const std::function<void()> &whileWaiting = {},
	            const std::function<void(uint64_t)> &beforePublish = {});

	/// Whether the finished message fits the ring (RgfxRingWriter::fits).
	bool fitsRing() const;

	/// The finished message's length: messageSize() bytes go to finishInRegion's block.
	uint64_t messageSize() const { return m_buf.size(); }

	/// The bytes the message holds before it grows.
	size_t capacity() const { return m_buf.capacity(); }

	/// Out of line: writes the whole COMMAND_BUFFER message (its header's `type` and `size`
	/// stamped, `seqno` 0) to `block`, the CPU address of region offset `region_offset`, of at
	/// least messageSize() bytes, then posts a COMMAND_BUFFER_IN_REGION naming it, waiting
	/// for room as finish() does; its seqno is the one the completion carries. The block must
	/// stay the process's until the buffer completes. False for a message too long for a
	/// header's `size`.
	bool finishInRegion(volatile uint8_t *block, uint64_t region_offset, uint64_t *out_seqno,
	                    const std::function<void()> &whileWaiting = {},
	                    const std::function<void(uint64_t)> &beforePublish = {});

	/// The command-buffer bytes built so far: the `RgfxCommandBuffer` header (its
	/// `op_bytes` is set by [`finish`]) then the op stream. For tests and
	/// diagnostics.
	const std::vector<uint8_t> &message() const { return m_buf; }
	/// The op-stream bytes appended so far (the message size less the header).
	uint64_t opBytes() const;

	// --- IOSurface copies this command buffer writes -------------------------

	/// Records that the command buffer writes the IOSurface copy `shadow` (opaque
	/// here: the Metal shell's RgfxSurfaceShadow), which its completion writes back
	/// to the surface's pages. Held until taken.
	void noteSurfaceWrite(std::shared_ptr<void> shadow) { m_surfaceWrites.push_back(std::move(shadow)); }
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
	/// Asks the host for the buffer's submission (`RGFX_OP_SUBMITTED`, where the device has
	/// RGFX_FEATURE_CB_SUBMITTED): it posts RGFX_EVT_CB_SUBMITTED ahead of the buffer's other
	/// records.
	void appendSubmitted() { appendOp(RGFX_OP_SUBMITTED); }
	/// Puts the `count` `RGFX_OP_SURFACE_USE` ops of bodies `uses` at the op stream's start,
	/// before its first encoder, where the host reads them before the buffer's work.
	void prependSurfaceUses(const RgfxOpSurfaceUse *uses, size_t count);
	/// Where the command buffer keeps the objects its encoders name, as Metal's command buffer
	/// with retained references keeps what it encodes: `keep(context, object)` for each one
	/// (once for a run of the same object), from the encode until the buffer completes or goes.
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

private:
	// Appends an op with no body (header only, len 8).
	void appendOp(uint16_t tag);
	// Appends an op whose body is `n` bytes at `body` (an 8-multiple); len = 8 + n.
	void appendOp(uint16_t tag, const void *body, size_t n);
	// Appends an array-tailed op: `head` (`head_n` bytes) then `count` elements of
	// `stride` bytes at `elems`; len = 8 + head_n + count*stride.
	void appendArrayOp(uint16_t tag, const void *head, size_t head_n, const void *elems,
	                   uint32_t count, size_t stride);
	// Appends raw bytes to the message buffer.
	void appendBytes(const void *src, size_t n);

	RgfxRingWriter &m_ring;
	RgfxCommandBuffer m_head;
	std::vector<uint8_t> m_buf;
	std::vector<std::shared_ptr<void>> m_surfaceWrites;
	std::vector<std::shared_ptr<void>> m_surfaceUses;
	std::vector<RgfxShadowUse> m_shadowUses;
	void (*m_keep)(void *, const void *) = nullptr;
	void *m_keepContext = nullptr;
	const void *m_lastKept = nullptr;
};

#endif // RGFX_ENCODER_HPP_
