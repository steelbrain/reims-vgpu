//! `id<MTLRenderCommandEncoder>`: core's `RenderEncoder`, whose commands (`RenderCmd`) are one
//! exhaustive `match`, one typed call per variant, and its tile-size getters.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandEncoder as _, MTLCullMode, MTLIndexType, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderStages, MTLResourceUsage, MTLStoreAction,
    MTLStoreActionOptions, MTLTriangleFillMode, MTLVisibilityResultMode, MTLWinding,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::render::RenderCmd;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::gpu::{array, check_len, objects, resource};
use crate::msg::encode::ToMtl;
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::RenderEncoderSpi;

/// `id<MTLRenderCommandEncoder>`.
pub type MetalRenderEncoder = Obj<ProtocolObject<dyn MTLRenderCommandEncoder>>;

impl reims_vgpu_core::platform::render::RenderEncoder<MetalPlatform> for MetalRenderEncoder {
    /// One call per command: a raise inside Metal is the caller's to catch. A command only the
    /// decoder encodes, always under a handler, is `Caught`; one only the display's frame
    /// encodes, with none, is `Fatal`; one both encode is `On(on)`, as the caller says. A
    /// `…:withRange:` command whose array is shorter than the range dies before the call
    /// ([`check_len`]). Raw guest values go to Metal unvalidated, which is what objc2-metal's
    /// `unsafe` marks: Metal checks them itself, and raises or faults the GPU on what it
    /// refuses; each C array has at least as many elements as its count or range.
    #[expect(clippy::too_many_lines, reason = "one arm per command")]
    #[expect(
        deprecated,
        reason = "`useHeaps:count:` and `useResources:count:usage:` are the selectors the guest encodes"
    )]
    fn encode(&self, on: OnFault, cmd: RenderCmd<'_, MetalPlatform>) -> RaiseResult<()> {
        let e = self.get();
        // SAFETY (every `unsafe` call below): as above.
        match cmd {
            RenderCmd::DispatchThreadsPerTile { threads_per_tile } => {
                let size = threads_per_tile.mtl();
                msg::send("dispatchThreadsPerTile:", Policy::Caught, || {
                    e.dispatchThreadsPerTile(size)
                })
            }
            RenderCmd::DispatchThreadsPerTileInRegion {
                threads_per_tile,
                region,
            } => {
                let (size, region) = (threads_per_tile.mtl(), region.mtl());
                msg::send("dispatchThreadsPerTile:inRegion:", Policy::Caught, || {
                    e.dispatchThreadsPerTile_inRegion(size, region)
                })
            }
            RenderCmd::DispatchThreadsPerTileInRegionIndex {
                threads_per_tile,
                region,
                render_target_array_index,
            } => {
                let (size, region) = (threads_per_tile.mtl(), region.mtl());
                msg::send(
                    "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
                    Policy::Caught,
                    || {
                        e.dispatchThreadsPerTile_inRegion_withRenderTargetArrayIndex(
                            size,
                            region,
                            render_target_array_index,
                        )
                    },
                )
            }
            RenderCmd::DrawIndexedPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let patch_index = patch_index_buffer.map(Obj::get);
                let control_point_index = control_point_index_buffer.get();
                let indirect = indirect_buffer.get();
                msg::send(
                    "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPatches_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_indirectBuffer_indirectBufferOffset(
                        number_of_patch_control_points.to_usize(),
                        patch_index,
                        patch_index_buffer_offset.to_usize(),
                        control_point_index,
                        control_point_index_buffer_offset.to_usize(),
                        indirect,
                        indirect_buffer_offset.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let patch_index = patch_index_buffer.map(Obj::get);
                let control_point_index = control_point_index_buffer.get();
                msg::send(
                    "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance(
                        number_of_patch_control_points.to_usize(),
                        patch_start.to_usize(),
                        patch_count.to_usize(),
                        patch_index,
                        patch_index_buffer_offset.to_usize(),
                        control_point_index,
                        control_point_index_buffer_offset.to_usize(),
                        instance_count.to_usize(),
                        base_instance.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (index_type, index) = (MTLIndexType(index_type.to_usize()), index_buffer.get());
                msg::send(
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            primitive,
                            index_count.to_usize(),
                            index_type,
                            index,
                            index_buffer_offset.to_usize(),
                        )
                    },
                )
            }
            RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (index_type, index) = (MTLIndexType(index_type.to_usize()), index_buffer.get());
                msg::send(
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                        primitive,
                        index_count.to_usize(),
                        index_type,
                        index,
                        index_buffer_offset.to_usize(),
                        instance_count.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawIndexedPrimitivesBase {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (index_type, index) = (MTLIndexType(index_type.to_usize()), index_buffer.get());
                // `NSInteger` is `i64` on arm64: lossless.
                let base_vertex = base_vertex as isize;
                msg::send(
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive,
                        index_count.to_usize(),
                        index_type,
                        index,
                        index_buffer_offset.to_usize(),
                        instance_count.to_usize(),
                        base_vertex,
                        base_instance.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type,
                index_buffer,
                index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (index_type, index) = (MTLIndexType(index_type.to_usize()), index_buffer.get());
                let indirect = indirect_buffer.get();
                msg::send(
                    "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        primitive,
                        index_type,
                        index,
                        index_buffer_offset.to_usize(),
                        indirect,
                        indirect_buffer_offset.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawMeshThreadgroups {
                threadgroups_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            } => {
                let (grid, object, mesh) = (
                    threadgroups_per_grid.mtl(),
                    threads_per_object_threadgroup.mtl(),
                    threads_per_mesh_threadgroup.mtl(),
                );
                msg::send(
                    "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
                    Policy::Caught,
                    || {
                        e.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                            grid, object, mesh,
                        )
                    },
                )
            }
            RenderCmd::DrawMeshThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            } => {
                let (indirect, offset) = (indirect_buffer.get(), indirect_buffer_offset.to_usize());
                let (object, mesh) = (
                    threads_per_object_threadgroup.mtl(),
                    threads_per_mesh_threadgroup.mtl(),
                );
                msg::send(
                    "drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
                    Policy::Caught,
                    || unsafe {
                        e.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                            indirect, offset, object, mesh,
                        )
                    },
                )
            }
            RenderCmd::DrawMeshThreads {
                threads_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            } => {
                let (grid, object, mesh) = (
                    threads_per_grid.mtl(),
                    threads_per_object_threadgroup.mtl(),
                    threads_per_mesh_threadgroup.mtl(),
                );
                msg::send(
                    "drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
                    Policy::Caught,
                    || {
                        e.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                            grid, object, mesh,
                        )
                    },
                )
            }
            RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let (patch_index, indirect) =
                    (patch_index_buffer.map(Obj::get), indirect_buffer.get());
                msg::send(
                    "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.drawPatches_patchIndexBuffer_patchIndexBufferOffset_indirectBuffer_indirectBufferOffset(
                        number_of_patch_control_points.to_usize(),
                        patch_index,
                        patch_index_buffer_offset.to_usize(),
                        indirect,
                        indirect_buffer_offset.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let patch_index = patch_index_buffer.map(Obj::get);
                msg::send(
                    "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:",
                    Policy::Caught,
                    || unsafe {
                        e.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance(
                        number_of_patch_control_points.to_usize(),
                        patch_start.to_usize(),
                        patch_count.to_usize(),
                        patch_index,
                        patch_index_buffer_offset.to_usize(),
                        instance_count.to_usize(),
                        base_instance.to_usize(),
                    )
                    },
                )
            }
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (indirect, offset) = (indirect_buffer.get(), indirect_buffer_offset.to_usize());
                msg::send(
                    "drawPrimitives:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.drawPrimitives_indirectBuffer_indirectBufferOffset(
                            primitive, indirect, offset,
                        )
                    },
                )
            }
            RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (start, count) = (vertex_start.to_usize(), vertex_count.to_usize());
                let drawn = msg::send(
                    "drawPrimitives:vertexStart:vertexCount:",
                    Policy::On(on),
                    || unsafe { e.drawPrimitives_vertexStart_vertexCount(primitive, start, count) },
                );
                crate::gpu::rate_map::drew(e, primitive, count, 1, &drawn);
                drawn
            }
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (start, count) = (vertex_start.to_usize(), vertex_count.to_usize());
                let instances = instance_count.to_usize();
                let drawn = msg::send(
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:",
                    Policy::Caught,
                    || unsafe {
                        e.drawPrimitives_vertexStart_vertexCount_instanceCount(
                            primitive, start, count, instances,
                        )
                    },
                );
                crate::gpu::rate_map::drew(e, primitive, count, instances, &drawn);
                drawn
            }
            RenderCmd::DrawPrimitivesBase {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (start, count) = (vertex_start.to_usize(), vertex_count.to_usize());
                let (instances, base) = (instance_count.to_usize(), base_instance.to_usize());
                let drawn = msg::send(
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                    Policy::Caught,
                    || unsafe {
                        e.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            primitive, start, count, instances, base,
                        )
                    },
                );
                crate::gpu::rate_map::drew(e, primitive, count, instances, &drawn);
                drawn
            }
            // A pass over a map that is not immutable, in which the guest drew no triangles, draws
            // no vertices first ([`crate::gpu::rate_map`]).
            RenderCmd::EndEncoding => {
                crate::gpu::rate_map::end_pass(on, e)?;
                msg::send("endEncoding", Policy::On(on), || e.endEncoding())
            }
            RenderCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => {
                let (icb, buffer) = (indirect_commandbuffer.get(), indirect_range_buffer.get());
                let offset = indirect_buffer_offset.to_usize();
                msg::send(
                    "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(
                            icb, buffer, offset,
                        )
                    },
                )
            }
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => {
                let (icb, range) = (indirect_command_buffer.get(), execution_range.mtl());
                msg::send(
                    "executeCommandsInBuffer:withRange:",
                    Policy::Caught,
                    || unsafe { e.executeCommandsInBuffer_withRange(icb, range) },
                )
            }
            RenderCmd::MemoryBarrierWithScope {
                scope,
                after,
                before,
            } => {
                let scope = MTLBarrierScope(scope.to_usize());
                let (after, before) = (
                    MTLRenderStages(after.to_usize()),
                    MTLRenderStages(before.to_usize()),
                );
                msg::send(
                    "memoryBarrierWithScope:afterStages:beforeStages:",
                    Policy::Caught,
                    || e.memoryBarrierWithScope_afterStages_beforeStages(scope, after, before),
                )
            }
            RenderCmd::SetAlphaTestReferenceValue { reference_value } => {
                msg::send("setAlphaTestReferenceValue:", Policy::Caught, || {
                    e.setAlphaTestReferenceValue(reference_value)
                })
            }
            RenderCmd::SetBlendColorRed {
                red,
                green,
                blue,
                alpha,
            } => msg::send("setBlendColorRed:green:blue:alpha:", Policy::Caught, || {
                e.setBlendColorRed_green_blue_alpha(red, green, blue, alpha)
            }),
            RenderCmd::SetColorResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert,
                index,
            } => {
                let texture = texture.map(Obj::get);
                msg::send(
                    "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
                    Policy::Caught,
                    || {
                        e.setColorResolveTexture_slice_depthPlane_level_yInvert_atIndex(
                            texture,
                            slice,
                            depth_plane,
                            level,
                            y_invert,
                            index,
                        )
                    },
                )
            }
            RenderCmd::SetColorStoreAction {
                store_action,
                color_attachment_index,
            } => {
                let (action, index) = (
                    MTLStoreAction(store_action.to_usize()),
                    color_attachment_index.to_usize(),
                );
                msg::send("setColorStoreAction:atIndex:", Policy::Caught, || unsafe {
                    e.setColorStoreAction_atIndex(action, index)
                })
            }
            RenderCmd::SetColorStoreActionOptions {
                store_action_options,
                color_attachment_index,
            } => {
                let options = MTLStoreActionOptions(store_action_options.to_usize());
                let index = color_attachment_index.to_usize();
                msg::send(
                    "setColorStoreActionOptions:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setColorStoreActionOptions_atIndex(options, index) },
                )
            }
            RenderCmd::SetCullMode { cull_mode } => {
                let mode = MTLCullMode(cull_mode.to_usize());
                msg::send("setCullMode:", Policy::Caught, || e.setCullMode(mode))
            }
            RenderCmd::SetDepthBias {
                depth_bias,
                slope_scale,
                clamp,
            } => msg::send("setDepthBias:slopeScale:clamp:", Policy::Caught, || {
                e.setDepthBias_slopeScale_clamp(depth_bias, slope_scale, clamp)
            }),
            RenderCmd::SetDepthCleared => {
                msg::send("setDepthCleared", Policy::Caught, || e.setDepthCleared())
            }
            RenderCmd::SetDepthClipModeSpi { mode } => {
                msg::send("setDepthClipModeSPI:", Policy::Caught, || {
                    e.setDepthClipModeSPI(mode)
                })
            }
            RenderCmd::SetDepthResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert,
            } => {
                let texture = texture.map(Obj::get);
                msg::send(
                    "setDepthResolveTexture:slice:depthPlane:level:yInvert:",
                    Policy::Caught,
                    || {
                        e.setDepthResolveTexture_slice_depthPlane_level_yInvert(
                            texture,
                            slice,
                            depth_plane,
                            level,
                            y_invert,
                        )
                    },
                )
            }
            RenderCmd::SetDepthStencilState {
                depth_stencil_state,
            } => {
                let state = depth_stencil_state.map(Obj::get);
                msg::send("setDepthStencilState:", Policy::Caught, || {
                    e.setDepthStencilState(state)
                })
            }
            RenderCmd::SetDepthStoreAction { store_action } => {
                let action = MTLStoreAction(store_action.to_usize());
                msg::send("setDepthStoreAction:", Policy::Caught, || {
                    e.setDepthStoreAction(action)
                })
            }
            RenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send(
                    "setFragmentBuffer:offset:atIndex:",
                    Policy::Fatal,
                    || unsafe { e.setFragmentBuffer_offset_atIndex(buffer, offset, index) },
                )
            }
            RenderCmd::SetFragmentBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send(
                    "setFragmentBufferOffset:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setFragmentBufferOffset_atIndex(offset, index) },
                )
            }
            RenderCmd::SetFragmentBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setFragmentBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setFragmentBuffers_offsets_withRange(list, offsets, range)
                })
            }
            RenderCmd::SetFragmentSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setFragmentSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        list, min, max, range,
                    )
                })
            }
            RenderCmd::SetFragmentSamplerStates { samplers, range } => {
                let sel = "setFragmentSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setFragmentSamplerStates_withRange(list, range)
                })
            }
            RenderCmd::SetFragmentTexture { texture, index } => {
                let (texture, index) = (texture.map(Obj::get), index.to_usize());
                msg::send("setFragmentTexture:atIndex:", Policy::Fatal, || unsafe {
                    e.setFragmentTexture_atIndex(texture, index)
                })
            }
            RenderCmd::SetFragmentTextures { textures, range } => {
                let sel = "setFragmentTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setFragmentTextures_withRange(list, range)
                })
            }
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding,
            } => {
                let winding = MTLWinding(front_facing_winding.to_usize());
                msg::send("setFrontFacingWinding:", Policy::Caught, || {
                    e.setFrontFacingWinding(winding)
                })
            }
            RenderCmd::SetLineWidth { line_width } => {
                msg::send("setLineWidth:", Policy::Caught, || {
                    e.setLineWidth(line_width)
                })
            }
            RenderCmd::SetMeshBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send("setMeshBufferOffset:atIndex:", Policy::Caught, || unsafe {
                    e.setMeshBufferOffset_atIndex(offset, index)
                })
            }
            RenderCmd::SetMeshBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setMeshBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setMeshBuffers_offsets_withRange(list, offsets, range)
                })
            }
            RenderCmd::SetMeshSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setMeshSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        list, min, max, range,
                    )
                })
            }
            RenderCmd::SetMeshSamplerStates { samplers, range } => {
                let sel = "setMeshSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setMeshSamplerStates_withRange(list, range)
                })
            }
            RenderCmd::SetMeshTextures { textures, range } => {
                let sel = "setMeshTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setMeshTextures_withRange(list, range)
                })
            }
            RenderCmd::SetObjectBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send(
                    "setObjectBufferOffset:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setObjectBufferOffset_atIndex(offset, index) },
                )
            }
            RenderCmd::SetObjectBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setObjectBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setObjectBuffers_offsets_withRange(list, offsets, range)
                })
            }
            RenderCmd::SetObjectSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setObjectSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        list, min, max, range,
                    )
                })
            }
            RenderCmd::SetObjectSamplerStates { samplers, range } => {
                let sel = "setObjectSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setObjectSamplerStates_withRange(list, range)
                })
            }
            RenderCmd::SetObjectTextures { textures, range } => {
                let sel = "setObjectTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setObjectTextures_withRange(list, range)
                })
            }
            RenderCmd::SetObjectThreadgroupMemoryLength { length, index } => {
                let (length, index) = (length.to_usize(), index.to_usize());
                msg::send(
                    "setObjectThreadgroupMemoryLength:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setObjectThreadgroupMemoryLength_atIndex(length, index) },
                )
            }
            RenderCmd::SetPointSize { point_size } => {
                msg::send("setPointSize:", Policy::Caught, || {
                    e.setPointSize(point_size)
                })
            }
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index } => {
                msg::send("setPrimitiveRestartEnabled:index:", Policy::Caught, || {
                    e.setPrimitiveRestartEnabled_index(enabled, index)
                })
            }
            RenderCmd::SetProvokingVertexMode { mode } => {
                msg::send("setProvokingVertexMode:", Policy::Caught, || {
                    e.setProvokingVertexMode(mode)
                })
            }
            RenderCmd::SetRenderPipelineState { pipeline_state } => {
                let state = pipeline_state.map(Obj::get);
                msg::send("setRenderPipelineState:", Policy::On(on), || {
                    RenderEncoderSpi::setRenderPipelineState(e, state)
                })
            }
            RenderCmd::SetScissorRect { rect } => {
                let rect = rect.mtl();
                msg::send("setScissorRect:", Policy::Caught, || e.setScissorRect(rect))
            }
            RenderCmd::SetScissorRects { scissor_rects } => {
                let (list, count) = (array(scissor_rects), scissor_rects.len());
                msg::send("setScissorRects:count:", Policy::Caught, || unsafe {
                    e.setScissorRects_count(list, count)
                })
            }
            RenderCmd::SetStencilCleared => msg::send("setStencilCleared", Policy::Caught, || {
                e.setStencilCleared()
            }),
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value,
                back_reference_value,
            } => msg::send(
                "setStencilFrontReferenceValue:backReferenceValue:",
                Policy::Caught,
                || {
                    e.setStencilFrontReferenceValue_backReferenceValue(
                        front_reference_value,
                        back_reference_value,
                    )
                },
            ),
            RenderCmd::SetStencilResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert,
            } => {
                let texture = texture.map(Obj::get);
                msg::send(
                    "setStencilResolveTexture:slice:depthPlane:level:yInvert:",
                    Policy::Caught,
                    || {
                        e.setStencilResolveTexture_slice_depthPlane_level_yInvert(
                            texture,
                            slice,
                            depth_plane,
                            level,
                            y_invert,
                        )
                    },
                )
            }
            RenderCmd::SetStencilStoreAction { store_action } => {
                let action = MTLStoreAction(store_action.to_usize());
                msg::send("setStencilStoreAction:", Policy::Caught, || {
                    e.setStencilStoreAction(action)
                })
            }
            RenderCmd::SetStencilStoreActionOptions {
                store_action_options,
            } => {
                let options = MTLStoreActionOptions(store_action_options.to_usize());
                msg::send("setStencilStoreActionOptions:", Policy::Caught, || {
                    e.setStencilStoreActionOptions(options)
                })
            }
            RenderCmd::SetDepthStoreActionOptions {
                store_action_options,
            } => {
                let options = MTLStoreActionOptions(store_action_options.to_usize());
                msg::send("setDepthStoreActionOptions:", Policy::Caught, || {
                    e.setDepthStoreActionOptions(options)
                })
            }
            RenderCmd::SetTessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => {
                let buffer = buffer.map(Obj::get);
                let (offset, stride) = (offset.to_usize(), instance_stride.to_usize());
                msg::send(
                    "setTessellationFactorBuffer:offset:instanceStride:",
                    Policy::Caught,
                    || unsafe {
                        e.setTessellationFactorBuffer_offset_instanceStride(buffer, offset, stride)
                    },
                )
            }
            RenderCmd::SetTessellationFactorScale { scale } => {
                msg::send("setTessellationFactorScale:", Policy::Caught, || {
                    e.setTessellationFactorScale(scale)
                })
            }
            RenderCmd::SetThreadgroupMemoryLength {
                length,
                offset,
                index,
            } => {
                let (length, offset, index) =
                    (length.to_usize(), offset.to_usize(), index.to_usize());
                msg::send(
                    "setThreadgroupMemoryLength:offset:atIndex:",
                    Policy::Caught,
                    || unsafe {
                        e.setThreadgroupMemoryLength_offset_atIndex(length, offset, index)
                    },
                )
            }
            RenderCmd::SetTileBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send("setTileBufferOffset:atIndex:", Policy::Caught, || unsafe {
                    e.setTileBufferOffset_atIndex(offset, index)
                })
            }
            RenderCmd::SetTileBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setTileBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setTileBuffers_offsets_withRange(list, offsets, range)
                })
            }
            RenderCmd::SetTileSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setTileSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        list, min, max, range,
                    )
                })
            }
            RenderCmd::SetTileSamplerStates { samplers, range } => {
                let sel = "setTileSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setTileSamplerStates_withRange(list, range)
                })
            }
            RenderCmd::SetTileTextures { textures, range } => {
                let sel = "setTileTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setTileTextures_withRange(list, range)
                })
            }
            RenderCmd::SetTriangleFillMode { fill_mode } => {
                let mode = MTLTriangleFillMode(fill_mode.to_usize());
                msg::send("setTriangleFillMode:", Policy::Caught, || {
                    e.setTriangleFillMode(mode)
                })
            }
            RenderCmd::SetVertexAmplificationCount { view_mappings } => {
                let (count, list) = (view_mappings.len(), array(view_mappings));
                msg::send(
                    "setVertexAmplificationCount:viewMappings:",
                    Policy::Caught,
                    || unsafe { e.setVertexAmplificationCount_viewMappings(count, list.as_ptr()) },
                )
            }
            RenderCmd::SetVertexAmplificationMode { mode, value } => {
                msg::send("setVertexAmplificationMode:value:", Policy::Caught, || {
                    e.setVertexAmplificationMode_value(mode, value)
                })
            }
            RenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send(
                    "setVertexBuffer:offset:atIndex:",
                    Policy::Fatal,
                    || unsafe { e.setVertexBuffer_offset_atIndex(buffer, offset, index) },
                )
            }
            RenderCmd::SetVertexBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send(
                    "setVertexBufferOffset:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setVertexBufferOffset_atIndex(offset, index) },
                )
            }
            RenderCmd::SetVertexBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => {
                let (offset, stride, index) =
                    (offset.to_usize(), stride.to_usize(), index.to_usize());
                msg::send(
                    "setVertexBufferOffset:attributeStride:atIndex:",
                    Policy::Caught,
                    || unsafe {
                        e.setVertexBufferOffset_attributeStride_atIndex(offset, stride, index)
                    },
                )
            }
            RenderCmd::SetVertexBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => {
                let sel = "setVertexBuffers:offsets:attributeStrides:withRange:";
                check_len(sel, strides.len(), range);
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, strides) = (array(&buffers), array(offsets), array(strides));
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setVertexBuffers_offsets_attributeStrides_withRange(
                        list, offsets, strides, range,
                    )
                })
            }
            RenderCmd::SetVertexBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setVertexBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setVertexBuffers_offsets_withRange(list, offsets, range)
                })
            }
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setVertexSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        list, min, max, range,
                    )
                })
            }
            RenderCmd::SetVertexSamplerStates { samplers, range } => {
                let sel = "setVertexSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setVertexSamplerStates_withRange(list, range)
                })
            }
            RenderCmd::SetVertexTextures { textures, range } => {
                let sel = "setVertexTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setVertexTextures_withRange(list, range)
                })
            }
            RenderCmd::SetViewport { viewport } => {
                let viewport = viewport.mtl();
                msg::send("setViewport:", Policy::On(on), || e.setViewport(viewport))
            }
            RenderCmd::SetViewportTransformEnabled { enabled } => {
                msg::send("setViewportTransformEnabled:", Policy::Caught, || {
                    e.setViewportTransformEnabled(enabled)
                })
            }
            RenderCmd::SetViewports { viewports } => {
                let (list, count) = (array(viewports), viewports.len());
                msg::send("setViewports:count:", Policy::Caught, || unsafe {
                    e.setViewports_count(list, count)
                })
            }
            RenderCmd::SetVisibilityResultMode { mode, offset } => {
                let (mode, offset) = (MTLVisibilityResultMode(mode.to_usize()), offset.to_usize());
                msg::send("setVisibilityResultMode:offset:", Policy::Caught, || {
                    e.setVisibilityResultMode_offset(mode, offset)
                })
            }
            RenderCmd::UpdateFence { fence, stages } => {
                let (fence, stages) = (fence.get(), MTLRenderStages(stages.to_usize()));
                msg::send("updateFence:afterStages:", Policy::Caught, || {
                    e.updateFence_afterStages(fence, stages)
                })
            }
            RenderCmd::UseHeaps { heaps } => {
                let heaps = objects(heaps.iter().map(|o| o.map(Obj::get)));
                let (list, count) = (array(&heaps), heaps.len());
                msg::send("useHeaps:count:", Policy::Caught, || unsafe {
                    e.useHeaps_count(list, count)
                })
            }
            RenderCmd::UseHeapsStages { heaps, stages } => {
                let heaps = objects(heaps.iter().map(|o| o.map(Obj::get)));
                let (list, count) = (array(&heaps), heaps.len());
                let stages = MTLRenderStages(stages.to_usize());
                msg::send("useHeaps:count:stages:", Policy::Caught, || unsafe {
                    e.useHeaps_count_stages(list, count, stages)
                })
            }
            RenderCmd::UseResources { resources, usage } => {
                let resources = objects(resources.iter().map(|r| r.map(resource)));
                let (list, count) = (array(&resources), resources.len());
                let usage = MTLResourceUsage(usage.to_usize());
                msg::send("useResources:count:usage:", Policy::Caught, || unsafe {
                    e.useResources_count_usage(list, count, usage)
                })
            }
            RenderCmd::UseResourcesStages {
                resources,
                usage,
                stages,
            } => {
                let resources = objects(resources.iter().map(|r| r.map(resource)));
                let (list, count) = (array(&resources), resources.len());
                let (usage, stages) = (
                    MTLResourceUsage(usage.to_usize()),
                    MTLRenderStages(stages.to_usize()),
                );
                msg::send(
                    "useResources:count:usage:stages:",
                    Policy::Caught,
                    || unsafe { e.useResources_count_usage_stages(list, count, usage, stages) },
                )
            }
            RenderCmd::WaitForFence { fence, stages } => {
                let (fence, stages) = (fence.get(), MTLRenderStages(stages.to_usize()));
                msg::send("waitForFence:beforeStages:", Policy::Caught, || {
                    e.waitForFence_beforeStages(fence, stages)
                })
            }
        }
    }
    fn tile_height(&self) -> u64 {
        let e = self.get();
        msg::get("tileHeight", || e.tileHeight()).to_u64()
    }
    fn tile_width(&self) -> u64 {
        let e = self.get();
        msg::get("tileWidth", || e.tileWidth()).to_u64()
    }
}

#[cfg(test)]
mod tests {

    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLClearColor, MTLDevice as _, MTLOrigin, MTLPixelFormat, MTLRegion,
        MTLRenderPipelineDescriptor, MTLSize, MTLStorageMode, MTLTexture as _,
    };
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::command::CommandBuffer as _;
    use reims_vgpu_core::platform::render::*;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::*;

    use super::*;
    use crate::gpu::resource::MetalRenderPipeline;
    use crate::gpu::resource::testkit::*;

    /// Every RenderEncoder method, pipeline first, endEncoding last; the SPI the host lacks raises
    #[test]
    fn render_encoder_sends() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&k.render_pipeline),
            },
        );
        assert!(r.is_ok(), "set_render_pipeline_state: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 0,
                index_type: 0,
                index_buffer: &k.buffer,
                index_buffer_offset: 0,
            },
        );
        assert!(r.is_ok(), "draw_indexed_primitives: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type: 0,
                index_type: 0,
                index_buffer: &k.buffer,
                index_buffer_offset: 0,
                indirect_buffer: &k.buffer,
                indirect_buffer_offset: 0,
            },
        );
        assert!(
            r.is_ok(),
            "draw_indexed_primitives_indirect_buffer_indirect_buffer_offset: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type: 0,
                index_count: 0,
                index_type: 0,
                index_buffer: &k.buffer,
                index_buffer_offset: 0,
                instance_count: 1,
            },
        );
        assert!(r.is_ok(), "draw_indexed_primitives_instance_count: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPrimitivesBase {
                primitive_type: 0,
                index_count: 0,
                index_type: 0,
                index_buffer: &k.buffer,
                index_buffer_offset: 0,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        );
        assert!(
            r.is_ok(),
            "draw_indexed_primitives_instance_count_base_vertex_base_instance: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type: 3,
                indirect_buffer: &k.buffer,
                indirect_buffer_offset: 0,
            },
        );
        assert!(r.is_ok(), "draw_primitives: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawPrimitives {
                primitive_type: 0,
                vertex_start: 0,
                vertex_count: 1,
            },
        );
        assert!(
            r.is_ok(),
            "draw_primitives_vertex_start_vertex_count: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawPrimitivesInstances {
                primitive_type: 0,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
            },
        );
        assert!(
            r.is_ok(),
            "draw_primitives_vertex_start_vertex_count_instance_count: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::DrawPrimitivesBase {
                primitive_type: 0,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
                base_instance: 0,
            },
        );
        assert!(
            r.is_ok(),
            "draw_primitives_vertex_start_vertex_count_instance_count_base_instance: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &k.icb_render,
                execution_range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "execute_commands_in_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer: &k.icb_render,
                indirect_range_buffer: &k.buffer,
                indirect_buffer_offset: 0,
            },
        );
        assert!(
            r.is_ok(),
            "execute_commands_in_buffer_indirect_buffer_indirect_buffer_offset: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::MemoryBarrierWithScope {
                scope: 0,
                after: 0,
                before: 0,
            },
        );
        assert!(r.is_ok(), "memory_barrier_with_scope: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetAlphaTestReferenceValue {
                reference_value: 1.0,
            },
        );
        assert!(r.is_ok(), "set_alpha_test_reference_value: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetBlendColorRed {
                red: 1.0,
                green: 1.0,
                blue: 1.0,
                alpha: 1.0,
            },
        );
        assert!(r.is_ok(), "set_blend_color_red: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetColorResolveTexture {
                texture: Some(&k.texture),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_color_resolve_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetColorStoreAction {
                store_action: 0,
                color_attachment_index: 0,
            },
        );
        assert!(r.is_ok(), "set_color_store_action: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetColorStoreActionOptions {
                store_action_options: 0,
                color_attachment_index: 0,
            },
        );
        assert!(r.is_ok(), "set_color_store_action_options: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetCullMode { cull_mode: 0 });
        assert!(r.is_ok(), "set_cull_mode: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetDepthBias {
                depth_bias: 1.0,
                slope_scale: 1.0,
                clamp: 1.0,
            },
        );
        assert!(r.is_ok(), "set_depth_bias: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetDepthCleared);
        assert!(r.is_ok(), "set_depth_cleared: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetDepthClipModeSpi { mode: 0 });
        assert!(r.is_ok(), "set_depth_clip_mode_spi: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetDepthResolveTexture {
                texture: Some(&k.texture),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
        );
        assert!(r.is_ok(), "set_depth_resolve_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: Some(&k.depth_stencil),
            },
        );
        assert!(r.is_ok(), "set_depth_stencil_state: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetDepthStoreAction { store_action: 0 },
        );
        assert!(r.is_ok(), "set_depth_store_action: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&k.buffer),
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_fragment_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentBufferOffset {
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_fragment_buffer_offset: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentBuffers {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_fragment_buffers: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentSamplerStates {
                samplers: &[Some(&k.sampler)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_fragment_sampler_states: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentSamplerStatesLodClamps {
                samplers: &[Some(&k.sampler)],
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[0.0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(
            r.is_ok(),
            "set_fragment_sampler_states_lod_min_clamps_lod_max_clamps: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentTexture {
                texture: Some(&k.texture),
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_fragment_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFragmentTextures {
                textures: &[Some(&k.texture)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_fragment_textures: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding: 0,
            },
        );
        assert!(r.is_ok(), "set_front_facing_winding: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetLineWidth { line_width: 1.0 });
        assert!(r.is_ok(), "set_line_width: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetPointSize { point_size: 1.0 });
        assert!(r.is_ok(), "set_point_size: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: false,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_primitive_restart_enabled: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetProvokingVertexMode { mode: 0 },
        );
        assert!(r.is_ok(), "set_provoking_vertex_mode: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetScissorRect {
                rect: MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
            },
        );
        assert!(r.is_ok(), "set_scissor_rect: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetScissorRects {
                scissor_rects: &[MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                }],
            },
        );
        assert!(r.is_ok(), "set_scissor_rects: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::SetStencilCleared);
        assert!(r.is_ok(), "set_stencil_cleared: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value: 0,
                back_reference_value: 0,
            },
        );
        assert!(r.is_ok(), "set_stencil_front_reference_value: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetStencilResolveTexture {
                texture: Some(&k.texture),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
        );
        assert!(r.is_ok(), "set_stencil_resolve_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetStencilStoreAction { store_action: 0 },
        );
        assert!(r.is_ok(), "set_stencil_store_action: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetStencilStoreActionOptions {
                store_action_options: 0,
            },
        );
        assert!(r.is_ok(), "set_stencil_store_action_options: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetDepthStoreActionOptions {
                store_action_options: 0,
            },
        );
        assert!(r.is_ok(), "set_depth_store_action_options: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTessellationFactorBuffer {
                buffer: Some(&k.buffer),
                offset: 0,
                instance_stride: 16,
            },
        );
        assert!(r.is_ok(), "set_tessellation_factor_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTessellationFactorScale { scale: 1.0 },
        );
        assert!(r.is_ok(), "set_tessellation_factor_scale: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetThreadgroupMemoryLength {
                length: 16,
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_threadgroup_memory_length: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTileBufferOffset {
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_tile_buffer_offset: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTileBuffers {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_tile_buffers: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTileSamplerStates {
                samplers: &[Some(&k.sampler)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_tile_sampler_states: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTileSamplerStatesLodClamps {
                samplers: &[Some(&k.sampler)],
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[0.0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(
            r.is_ok(),
            "set_tile_sampler_states_lod_min_clamps_lod_max_clamps: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTileTextures {
                textures: &[Some(&k.texture)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_tile_textures: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetTriangleFillMode { fill_mode: 0 },
        );
        assert!(r.is_ok(), "set_triangle_fill_mode: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexAmplificationMode { mode: 0, value: 0 },
        );
        assert!(r.is_ok(), "set_vertex_amplification_mode: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBuffer {
                buffer: Some(&k.buffer),
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_vertex_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBufferOffset {
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_vertex_buffer_offset: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBufferOffsetAttributeStride {
                offset: 0,
                stride: 16,
                index: 0,
            },
        );
        assert!(
            r.is_ok(),
            "set_vertex_buffer_offset_attribute_stride: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBuffers {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_vertex_buffers: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBuffersAttributeStrides {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                strides: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_vertex_buffers_attribute_strides: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexSamplerStates {
                samplers: &[Some(&k.sampler)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_vertex_sampler_states: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers: &[Some(&k.sampler)],
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[0.0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(
            r.is_ok(),
            "set_vertex_sampler_states_lod_min_clamps_lod_max_clamps: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVertexTextures {
                textures: &[Some(&k.texture)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_vertex_textures: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetViewport {
                viewport: MtlViewport {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width: 16.0,
                    height: 16.0,
                    znear: 0.0,
                    zfar: 1.0,
                },
            },
        );
        assert!(r.is_ok(), "set_viewport: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetViewportTransformEnabled { enabled: false },
        );
        assert!(r.is_ok(), "set_viewport_transform_enabled: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetViewports {
                viewports: &[MtlViewport {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width: 16.0,
                    height: 16.0,
                    znear: 0.0,
                    zfar: 1.0,
                }],
            },
        );
        assert!(r.is_ok(), "set_viewports: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::SetVisibilityResultMode { mode: 0, offset: 0 },
        );
        assert!(r.is_ok(), "set_visibility_result_mode: {r:?}");
        assert_eq!(x.tile_height(), raw_u64(id(&x), "tileHeight"));
        assert_eq!(x.tile_width(), raw_u64(id(&x), "tileWidth"));
        let r = x.encode(
            OnFault::Return,
            RenderCmd::UpdateFence {
                fence: &k.fence,
                stages: 1,
            },
        );
        assert!(r.is_ok(), "update_fence: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::UseHeaps {
                heaps: &[Some(&k.heap)],
            },
        );
        assert!(r.is_ok(), "use_heaps: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::UseHeapsStages {
                heaps: &[Some(&k.heap)],
                stages: 1,
            },
        );
        assert!(r.is_ok(), "use_heaps_stages: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&k.buffer))],
                usage: 1,
            },
        );
        assert!(r.is_ok(), "use_resources: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::UseResourcesStages {
                resources: &[Some(ResourceRef::Buffer(&k.buffer))],
                usage: 1,
                stages: 1,
            },
        );
        assert!(r.is_ok(), "use_resources_stages: {r:?}");
        let r = x.encode(
            OnFault::Return,
            RenderCmd::WaitForFence {
                fence: &k.fence,
                stages: 1,
            },
        );
        assert!(r.is_ok(), "wait_for_fence: {r:?}");
        let r = x.encode(OnFault::Return, RenderCmd::EndEncoding);
        assert!(r.is_ok(), "end_encoding: {r:?}");
    }

    /// setVertexAmplificationCount:viewMappings: passes the count and core's array
    #[test]
    fn vertex_amplification_count() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        let m = VertexAmplificationViewMapping {
            viewport_array_index_offset: 0,
            render_target_array_index_offset: 0,
        };
        x.encode(
            OnFault::Return,
            RenderCmd::SetVertexAmplificationCount {
                view_mappings: &[m],
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetVertexAmplificationCount { view_mappings: &[] },
        )
        .unwrap();
        x.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
    }

    /// dispatchThreadsPerTile:, dispatchThreadsPerTile:inRegion: (SPI) and
    /// dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex: (SPI, a 32-bit index) with a
    /// tile pipeline bound (AGX faults without one)
    #[test]
    fn tile_dispatches() {
        let k = kit();
        let tile = tile_pipeline(&k.device);
        let cb = command_buffer(&k.queue);
        let x = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&tile),
            },
        )
        .unwrap();
        let t = MtlSize::new(x.tile_width(), x.tile_height(), 1);
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(16, 16, 1));
        x.encode(
            OnFault::Return,
            RenderCmd::DispatchThreadsPerTile {
                threads_per_tile: t,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DispatchThreadsPerTileInRegion {
                threads_per_tile: t,
                region,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DispatchThreadsPerTileInRegionIndex {
                threads_per_tile: t,
                region,
                render_target_array_index: 0,
            },
        )
        .unwrap();
        x.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
    }

    /// The four patch draws (direct and indirect, indexed and not) with a tessellated pipeline, its
    /// control points and factors bound (AGX faults without one)
    #[test]
    fn patch_draws() {
        let k = kit();
        let pipeline = patch_pipeline(&k.device);
        let points = buffer(&k.device, 4096);
        let factors = buffer(&k.device, 4096);
        let indirect = buffer(&k.device, 4096);
        let indices = buffer(&k.device, 4096);
        let cb = command_buffer(&k.queue);
        let x = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetVertexBuffer {
                buffer: Some(&points),
                offset: 0,
                index: 0,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetTessellationFactorBuffer {
                buffer: Some(&factors),
                offset: 0,
                instance_stride: 0,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DrawPatches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer: None,
                patch_index_buffer_offset: 0,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points: 3,
                patch_index_buffer: None,
                patch_index_buffer_offset: 0,
                indirect_buffer: &indirect,
                indirect_buffer_offset: 0,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPatches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer: None,
                patch_index_buffer_offset: 0,
                control_point_index_buffer: &indices,
                control_point_index_buffer_offset: 0,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DrawIndexedPatchesIndirect {
                number_of_patch_control_points: 3,
                patch_index_buffer: None,
                patch_index_buffer_offset: 0,
                control_point_index_buffer: &indices,
                control_point_index_buffer_offset: 0,
                indirect_buffer: &indirect,
                indirect_buffer_offset: 0,
            },
        )
        .unwrap();
        x.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
    }

    /// The widths [`line_widths_draw_as_metal`] draws at, one picture each, left to right.
    const LINE_WIDTHS: [f32; 6] = [0.0, 0.5, 1.0, 2.0, 3.5, 8.0];

    /// What Metal draws of a list of two lines, from (1, 3) to (15, 3), along a pixel edge, and
    /// from (1, 8) to (15, 15), at each of [`LINE_WIDTHS`] on a 16x16 target, `#` for a pixel drawn
    /// (measured on the M3 Max at macOS 27.0.1 by a program of its own that sends Metal's
    /// encoder `setLineWidth:`)
    const LINES_DRAWN: &str = "
................ ................ ................ ................ ................ .##############.
................ ................ ................ ................ .##############. .##############.
................ ................ .##############. .##############. .##############. .##############.
................ ................ ................ .##############. .##############. .##############.
................ ................ ................ ................ .##############. .##############.
................ ................ ................ ................ ................ .##############.
................ ................ ................ ................ ................ .##############.
................ ................ ................ .#.............. .##............. .#######........
................ .#.............. .##............. .###............ .####........... .#########......
................ ...#............ ...##........... ..####.......... .######......... .###########....
................ .....#.......... .....##......... ....####........ ..#######....... .#############..
................ .......#........ .......##....... ......####...... ....#######..... .##############.
................ .........#...... .........##..... ........####.... ......#######... ..#############.
................ ...........#.... ...........##... ..........####.. ........#######. ....###########.
................ .............#.. .............##. ............###. ..........#####. ......#########.
................ ................ ................ ..............#. ............###. ........#######.
";

    /// What Metal draws of a strip of three vertices, (3.5, 1), (3.5, 9.5) and (13.5, 14.5), the
    /// second segment through pixel centres, as [`LINES_DRAWN`]
    const LINE_STRIP_DRAWN: &str = "
................ ................ ...#............ ................ ................ ................
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #######.........
...#............ ...#............ ...#............ ..##............ ..###........... #########.......
...#............ ...#............ ...#............ ...##........... ...####......... ...########.....
.....#.......... .....#.......... ....##.......... ...####......... ...######....... ...##########...
.......#........ .......#........ ......##........ .....####....... ....#######..... ...##########...
.........#...... .........#...... ........##...... .......####..... ......#######... ...##########...
...........#.... ...........#.... ..........##.... .........####... ........#####... ...##########...
................ ................ ............#... ...........##... ..........###... .....########...
................ ................ ................ ................ ............#... .......######...
";

    /// A line is drawn as wide as `setLineWidth:` sets, pixel for pixel as Metal draws it at that
    /// width: by the diamond-exit rule at 1, and otherwise as a band that many pixels across the
    /// line's minor axis whose pixel centres are drawn (nothing at 0 or 0.5 along a pixel edge,
    /// 4 rows at 3.5), the centres on the line itself at 0; each segment of a strip for itself
    #[test]
    fn line_widths_draw_as_metal() {
        const MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; };
vertex V lv(uint i [[vertex_id]], const device float2 *q [[buffer(0)]]) {
    V o; o.p = float4(q[i].x / 8 - 1, 1 - q[i].y / 8, 0, 1); return o;
}
fragment float4 f() { return float4(1); }
";
        let k = kit();
        let d = k.device.get();
        let lib = Obj::from_retained(
            d.newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
                .unwrap(),
        );
        let rd = MTLRenderPipelineDescriptor::new();
        rd.setVertexFunction(Some(&function(&lib, "lv")));
        rd.setFragmentFunction(Some(&function(&lib, "f")));
        // SAFETY: index 0 of the colour attachments.
        let ca = unsafe { rd.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline: MetalRenderPipeline =
            Obj::from_retained(d.newRenderPipelineStateWithDescriptor_error(&rd).unwrap());
        let lines = [[1.0, 3.0], [15.0, 3.0], [1.0, 8.0], [15.0, 15.0]];
        let strip = [[3.5, 1.0], [3.5, 9.5], [13.5, 14.5]];
        for (primitive_type, points, drawn) in [
            (MTLPrimitiveType::Line, &lines[..], LINES_DRAWN),
            (MTLPrimitiveType::LineStrip, &strip[..], LINE_STRIP_DRAWN),
        ] {
            let pictures: Vec<Vec<String>> = LINE_WIDTHS
                .iter()
                .map(|&width| draw_lines(&k, &pipeline, primitive_type, points, width))
                .collect();
            let rows: Vec<String> = (0..16)
                .map(|y| {
                    let row: Vec<&str> = pictures.iter().map(|p| p[y].as_str()).collect();
                    row.join(" ")
                })
                .collect();
            assert_eq!(
                rows.join("\n"),
                drawn.trim(),
                "{primitive_type:?} at {LINE_WIDTHS:?}"
            );
        }
    }

    /// Draws `points` as `primitive_type` lines `width` wide through the platform's encoder, in
    /// `pipeline` (`lv`: pixel coordinates on a 16x16 target), onto a 16x16 BGRA8 target cleared
    /// to 0, and answers its rows, `#` for a pixel drawn in white, `.` for one left clear, and `?`
    /// for any other
    fn draw_lines(
        k: &Kit,
        pipeline: &MetalRenderPipeline,
        primitive_type: MTLPrimitiveType,
        points: &[[f32; 2]],
        width: f32,
    ) -> Vec<String> {
        let vertices = buffer(&k.device, 4096);
        let bytes = std::mem::size_of_val(points);
        // SAFETY: a 4096-byte `Shared` buffer, which holds the points, written before any
        // command buffer reads it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                points.as_ptr().cast::<u8>(),
                vertices.contents().as_ptr::<u8>(),
                bytes,
            );
        }
        let target = texture(&k.device, MTLStorageMode::Shared);
        let rp = render_pass(&target);
        // SAFETY: index 0 of the colour attachments.
        let ca = unsafe { rp.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 0.0,
        });
        let pass = Obj::from_retained(rp);
        let cb = command_buffer(&k.queue);
        let x = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        let cmds = [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(pipeline),
            },
            RenderCmd::SetLineWidth { line_width: width },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&vertices),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: primitive_type.0.to_u64(),
                vertex_start: 0,
                vertex_count: points.len().to_u64(),
            },
            RenderCmd::EndEncoding,
        ];
        for cmd in cmds {
            x.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
        let mut px = [0u32; 16 * 16];
        // SAFETY: the whole of a 16x16 target of 4 bytes a pixel, into as many bytes, after its
        // command buffer completed.
        unsafe {
            target.get().getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(px.as_mut_ptr().cast()).unwrap(),
                16 * 4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                },
                0,
            );
        }
        px.chunks(16)
            .map(|row| {
                row.iter()
                    .map(|&p| match p {
                        0 => '.',
                        u32::MAX => '#',
                        _ => '?',
                    })
                    .collect()
            })
            .collect()
    }

    /// A pass that clears its target and draws nothing leaves the clear colour (the encoder core
    /// makes and ends is a real pass)
    #[test]
    fn render_pass_runs() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&k.render_pipeline),
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::SetViewport {
                viewport: MtlViewport {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width: 16.0,
                    height: 16.0,
                    znear: 0.0,
                    zfar: 1.0,
                },
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            RenderCmd::DrawPrimitives {
                primitive_type: 0,
                vertex_start: 0,
                vertex_count: 1,
            },
        )
        .unwrap();
        x.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
    }
}
