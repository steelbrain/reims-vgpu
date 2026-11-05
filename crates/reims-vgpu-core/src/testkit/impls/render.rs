//! Fake: the `platform/render.rs` traits over the fake models (architecture 13.2).
//!
//! Every render command is ordered: each logs its selector and arguments (`fake::calllog`) and
//! consults the script (`fake::script`); see `fake::impls` for the outcome channels. The tile
//! size getters are free, and answer the fake's constant.

use super::super::calllog::Arg;
use super::super::objects::{Fk, kind};
use super::super::platform::FakePlatform;
use super::res_ids;
use crate::platform::render::*;
use crate::raise::{OnFault, RaiseResult};
use crate::testkit::script::Policy;

impl RenderEncoder<FakePlatform> for Fk<kind::RenderEncoder> {
    fn encode(&self, on: OnFault, cmd: RenderCmd<'_, FakePlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        match cmd {
            RenderCmd::DispatchThreadsPerTile { threads_per_tile } => {
        self.act(Policy::Caught, sel, vec![Arg::val(&threads_per_tile)])
    }
            RenderCmd::DispatchThreadsPerTileInRegion { threads_per_tile, region } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::val(&threads_per_tile), Arg::val(&region)],
        )
    }
            RenderCmd::DispatchThreadsPerTileInRegionIndex { threads_per_tile, region, render_target_array_index } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::val(&threads_per_tile),
                Arg::val(&region),
                Arg::U32(render_target_array_index),
            ],
        )
    }
            RenderCmd::DrawIndexedPatchesIndirect { number_of_patch_control_points, patch_index_buffer, patch_index_buffer_offset, control_point_index_buffer, control_point_index_buffer_offset, indirect_buffer, indirect_buffer_offset } => {
        self.act(Policy::Caught, "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:", vec![Arg::U64(number_of_patch_control_points), Arg::opt(patch_index_buffer), Arg::U64(patch_index_buffer_offset), Arg::obj(control_point_index_buffer), Arg::U64(control_point_index_buffer_offset), Arg::obj(indirect_buffer), Arg::U64(indirect_buffer_offset)])
    }
            RenderCmd::DrawIndexedPatches { number_of_patch_control_points, patch_start, patch_count, patch_index_buffer, patch_index_buffer_offset, control_point_index_buffer, control_point_index_buffer_offset, instance_count, base_instance } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(number_of_patch_control_points), Arg::U64(patch_start), Arg::U64(patch_count), Arg::opt(patch_index_buffer), Arg::U64(patch_index_buffer_offset), Arg::obj(control_point_index_buffer), Arg::U64(control_point_index_buffer_offset), Arg::U64(instance_count), Arg::U64(base_instance)])
    }
            RenderCmd::DrawIndexedPrimitives { primitive_type, index_count, index_type, index_buffer, index_buffer_offset } => {
        self.act(Policy::Caught,
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            vec![
                Arg::U64(primitive_type),
                Arg::U64(index_count),
                Arg::U64(index_type),
                Arg::obj(index_buffer),
                Arg::U64(index_buffer_offset),
            ],
        )
    }
            RenderCmd::DrawIndexedPrimitivesInstances { primitive_type, index_count, index_type, index_buffer, index_buffer_offset, instance_count } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(primitive_type), Arg::U64(index_count), Arg::U64(index_type), Arg::obj(index_buffer), Arg::U64(index_buffer_offset), Arg::U64(instance_count)])
    }
            RenderCmd::DrawIndexedPrimitivesBase { primitive_type, index_count, index_type, index_buffer, index_buffer_offset, instance_count, base_vertex, base_instance } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(primitive_type), Arg::U64(index_count), Arg::U64(index_type), Arg::obj(index_buffer), Arg::U64(index_buffer_offset), Arg::U64(instance_count), Arg::I64(base_vertex), Arg::U64(base_instance)])
    }
            RenderCmd::DrawIndexedPrimitivesIndirect { primitive_type, index_type, index_buffer, index_buffer_offset, indirect_buffer, indirect_buffer_offset } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(primitive_type), Arg::U64(index_type), Arg::obj(index_buffer), Arg::U64(index_buffer_offset), Arg::obj(indirect_buffer), Arg::U64(indirect_buffer_offset)])
    }
            RenderCmd::DrawPatchesIndirect { number_of_patch_control_points, patch_index_buffer, patch_index_buffer_offset, indirect_buffer, indirect_buffer_offset } => {
        self.act(Policy::Caught, "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:", vec![Arg::U64(number_of_patch_control_points), Arg::opt(patch_index_buffer), Arg::U64(patch_index_buffer_offset), Arg::obj(indirect_buffer), Arg::U64(indirect_buffer_offset)])
    }
            RenderCmd::DrawPatches { number_of_patch_control_points, patch_start, patch_count, patch_index_buffer, patch_index_buffer_offset, instance_count, base_instance } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(number_of_patch_control_points), Arg::U64(patch_start), Arg::U64(patch_count), Arg::opt(patch_index_buffer), Arg::U64(patch_index_buffer_offset), Arg::U64(instance_count), Arg::U64(base_instance)])
    }
            RenderCmd::DrawMeshThreadgroups { threadgroups_per_grid, threads_per_object_threadgroup, threads_per_mesh_threadgroup } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::val(&threadgroups_per_grid),
                Arg::val(&threads_per_object_threadgroup),
                Arg::val(&threads_per_mesh_threadgroup),
            ],
        )
    }
            RenderCmd::DrawMeshThreadgroupsIndirect { indirect_buffer, indirect_buffer_offset, threads_per_object_threadgroup, threads_per_mesh_threadgroup } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::obj(indirect_buffer),
                Arg::U64(indirect_buffer_offset),
                Arg::val(&threads_per_object_threadgroup),
                Arg::val(&threads_per_mesh_threadgroup),
            ],
        )
    }
            RenderCmd::DrawMeshThreads { threads_per_grid, threads_per_object_threadgroup, threads_per_mesh_threadgroup } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::val(&threads_per_grid),
                Arg::val(&threads_per_object_threadgroup),
                Arg::val(&threads_per_mesh_threadgroup),
            ],
        )
    }
            RenderCmd::DrawPrimitivesIndirect { primitive_type, indirect_buffer, indirect_buffer_offset } => {
        self.act(Policy::Caught,
            "drawPrimitives:indirectBuffer:indirectBufferOffset:",
            vec![
                Arg::U64(primitive_type),
                Arg::obj(indirect_buffer),
                Arg::U64(indirect_buffer_offset),
            ],
        )
    }
            RenderCmd::DrawPrimitives { primitive_type, vertex_start, vertex_count } => {
        self.act(Policy::On(on),
            sel,
            vec![
                Arg::U64(primitive_type),
                Arg::U64(vertex_start),
                Arg::U64(vertex_count),
            ],
        )
    }
            RenderCmd::DrawPrimitivesInstances { primitive_type, vertex_start, vertex_count, instance_count } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::U64(primitive_type),
                Arg::U64(vertex_start),
                Arg::U64(vertex_count),
                Arg::U64(instance_count),
            ],
        )
    }
            RenderCmd::DrawPrimitivesBase { primitive_type, vertex_start, vertex_count, instance_count, base_instance } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::U64(primitive_type),
                Arg::U64(vertex_start),
                Arg::U64(vertex_count),
                Arg::U64(instance_count),
                Arg::U64(base_instance),
            ],
        )
    }
            RenderCmd::EndEncoding => self.act(Policy::On(on), sel, vec![]),
            RenderCmd::ExecuteCommandsIndirect { indirect_commandbuffer, indirect_range_buffer, indirect_buffer_offset } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::obj(indirect_commandbuffer),
                Arg::obj(indirect_range_buffer),
                Arg::U64(indirect_buffer_offset),
            ],
        )
    }
            RenderCmd::ExecuteCommandsInBuffer { indirect_command_buffer, execution_range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::obj(indirect_command_buffer),
                Arg::val(&execution_range),
            ],
        )
    }
            RenderCmd::MemoryBarrierWithScope { scope, after, before } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(scope), Arg::U64(after), Arg::U64(before)],
        )
    }
            RenderCmd::SetAlphaTestReferenceValue { reference_value } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::F32(reference_value)],
        )
    }
            RenderCmd::SetBlendColorRed { red, green, blue, alpha } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::F32(red),
                Arg::F32(green),
                Arg::F32(blue),
                Arg::F32(alpha),
            ],
        )
    }
            RenderCmd::SetColorResolveTexture { texture, slice, depth_plane, level, y_invert, index } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::opt(texture),
                Arg::U64(slice),
                Arg::U64(depth_plane),
                Arg::U64(level),
                Arg::Bool(y_invert),
                Arg::U64(index),
            ],
        )
    }
            RenderCmd::SetColorStoreAction { store_action, color_attachment_index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(store_action), Arg::U64(color_attachment_index)],
        )
    }
            RenderCmd::SetColorStoreActionOptions { store_action_options, color_attachment_index } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::U64(store_action_options),
                Arg::U64(color_attachment_index),
            ],
        )
    }
            RenderCmd::SetCullMode { cull_mode } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(cull_mode)])
    }
            RenderCmd::SetDepthBias { depth_bias, slope_scale, clamp } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::F32(depth_bias), Arg::F32(slope_scale), Arg::F32(clamp)],
        )
    }
            RenderCmd::SetDepthCleared => {
        self.act(Policy::Caught, sel, vec![])
    }
            RenderCmd::SetDepthClipModeSpi { mode } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(mode)])
    }
            RenderCmd::SetDepthResolveTexture { texture, slice, depth_plane, level, y_invert } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::opt(texture),
                Arg::U64(slice),
                Arg::U64(depth_plane),
                Arg::U64(level),
                Arg::Bool(y_invert),
            ],
        )
    }
            RenderCmd::SetDepthStencilState { depth_stencil_state } => {
        self.act(Policy::Caught, sel, vec![Arg::opt(depth_stencil_state)])
    }
            RenderCmd::SetDepthStoreAction { store_action } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(store_action)])
    }
            RenderCmd::SetDepthStoreActionOptions { store_action_options } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(store_action_options)])
    }
            RenderCmd::SetFragmentBuffer { buffer, offset, index } => {
        self.act(Policy::Fatal,
            sel,
            vec![Arg::opt(buffer), Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetFragmentBufferOffset { offset, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetFragmentBuffers { buffers, offsets, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(buffers),
                Arg::U64s(offsets.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetFragmentSamplerStatesLodClamps { samplers, lod_min_clamps, lod_max_clamps, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(samplers),
                Arg::F32s(lod_min_clamps.to_vec()),
                Arg::F32s(lod_max_clamps.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetFragmentSamplerStates { samplers, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(samplers), Arg::val(&range)],
        )
    }
            RenderCmd::SetFragmentTexture { texture, index } => {
        self.act(Policy::Fatal,
            sel,
            vec![Arg::opt(texture), Arg::U64(index)],
        )
    }
            RenderCmd::SetFragmentTextures { textures, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(textures), Arg::val(&range)],
        )
    }
            RenderCmd::SetFrontFacingWinding { front_facing_winding } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(front_facing_winding)],
        )
    }
            RenderCmd::SetLineWidth { line_width } => {
        self.act(Policy::Caught, sel, vec![Arg::F32(line_width)])
    }
            RenderCmd::SetPointSize { point_size } => {
        self.act(Policy::Caught, sel, vec![Arg::F32(point_size)])
    }
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::Bool(enabled), Arg::U64(index)],
        )
    }
            RenderCmd::SetProvokingVertexMode { mode } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(mode)])
    }
            RenderCmd::SetRenderPipelineState { pipeline_state } => {
        self.act(Policy::On(on), sel, vec![Arg::opt(pipeline_state)])
    }
            RenderCmd::SetScissorRect { rect } => {
        self.act(Policy::Caught, sel, vec![Arg::val(&rect)])
    }
            RenderCmd::SetScissorRects { scissor_rects } => {
        self.act(Policy::Caught, sel, vec![Arg::val(scissor_rects)])
    }
            RenderCmd::SetStencilCleared => {
        self.act(Policy::Caught, sel, vec![])
    }
            RenderCmd::SetStencilFrontReferenceValue { front_reference_value, back_reference_value } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::U32(front_reference_value),
                Arg::U32(back_reference_value),
            ],
        )
    }
            RenderCmd::SetStencilResolveTexture { texture, slice, depth_plane, level, y_invert } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::opt(texture),
                Arg::U64(slice),
                Arg::U64(depth_plane),
                Arg::U64(level),
                Arg::Bool(y_invert),
            ],
        )
    }
            RenderCmd::SetStencilStoreAction { store_action } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(store_action)])
    }
            RenderCmd::SetStencilStoreActionOptions { store_action_options } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(store_action_options)],
        )
    }
            RenderCmd::SetTessellationFactorBuffer { buffer, offset, instance_stride } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::opt(buffer),
                Arg::U64(offset),
                Arg::U64(instance_stride),
            ],
        )
    }
            RenderCmd::SetTessellationFactorScale { scale } => {
        self.act(Policy::Caught, sel, vec![Arg::F32(scale)])
    }
            RenderCmd::SetThreadgroupMemoryLength { length, offset, index } => {
        self.act(Policy::Caught,
            "setThreadgroupMemoryLength:offset:atIndex:",
            vec![Arg::U64(length), Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetMeshBufferOffset { offset, index }
            | RenderCmd::SetObjectBufferOffset { offset, index }
            | RenderCmd::SetObjectThreadgroupMemoryLength { length: offset, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetMeshBuffers { buffers, offsets, range }
            | RenderCmd::SetObjectBuffers { buffers, offsets, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(buffers),
                Arg::U64s(offsets.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetMeshSamplerStatesLodClamps { samplers, lod_min_clamps, lod_max_clamps, range }
            | RenderCmd::SetObjectSamplerStatesLodClamps { samplers, lod_min_clamps, lod_max_clamps, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(samplers),
                Arg::F32s(lod_min_clamps.to_vec()),
                Arg::F32s(lod_max_clamps.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetMeshSamplerStates { samplers, range }
            | RenderCmd::SetObjectSamplerStates { samplers, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(samplers), Arg::val(&range)],
        )
    }
            RenderCmd::SetMeshTextures { textures, range }
            | RenderCmd::SetObjectTextures { textures, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(textures), Arg::val(&range)],
        )
    }
            RenderCmd::SetTileBufferOffset { offset, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetTileBuffers { buffers, offsets, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(buffers),
                Arg::U64s(offsets.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetTileSamplerStatesLodClamps { samplers, lod_min_clamps, lod_max_clamps, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(samplers),
                Arg::F32s(lod_min_clamps.to_vec()),
                Arg::F32s(lod_max_clamps.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetTileSamplerStates { samplers, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(samplers), Arg::val(&range)],
        )
    }
            RenderCmd::SetTileTextures { textures, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(textures), Arg::val(&range)],
        )
    }
            RenderCmd::SetTriangleFillMode { fill_mode } => {
        self.act(Policy::Caught, sel, vec![Arg::U64(fill_mode)])
    }
            RenderCmd::SetVertexAmplificationCount { view_mappings } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::val(view_mappings)],
        )
    }
            RenderCmd::SetVertexAmplificationMode { mode, value } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(mode), Arg::U64(value)],
        )
    }
            RenderCmd::SetVertexBuffer { buffer, offset, index } => {
        self.act(Policy::Fatal,
            sel,
            vec![Arg::opt(buffer), Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetVertexBufferOffset { offset, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(offset), Arg::U64(index)],
        )
    }
            RenderCmd::SetVertexBufferOffsetAttributeStride { offset, stride, index } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(offset), Arg::U64(stride), Arg::U64(index)],
        )
    }
            RenderCmd::SetVertexBuffersAttributeStrides { buffers, offsets, strides, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(buffers),
                Arg::U64s(offsets.to_vec()),
                Arg::U64s(strides.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetVertexBuffers { buffers, offsets, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(buffers),
                Arg::U64s(offsets.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetVertexSamplerStatesLodClamps { samplers, lod_min_clamps, lod_max_clamps, range } => {
        self.act(Policy::Caught,
            sel,
            vec![
                Arg::objs(samplers),
                Arg::F32s(lod_min_clamps.to_vec()),
                Arg::F32s(lod_max_clamps.to_vec()),
                Arg::val(&range),
            ],
        )
    }
            RenderCmd::SetVertexSamplerStates { samplers, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(samplers), Arg::val(&range)],
        )
    }
            RenderCmd::SetVertexTextures { textures, range } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(textures), Arg::val(&range)],
        )
    }
            RenderCmd::SetViewport { viewport } => {
        self.act(Policy::On(on), sel, vec![Arg::val(&viewport)])
    }
            RenderCmd::SetViewportTransformEnabled { enabled } => {
        self.act(Policy::Caught, sel, vec![Arg::Bool(enabled)])
    }
            RenderCmd::SetViewports { viewports } => {
        self.act(Policy::Caught, sel, vec![Arg::val(viewports)])
    }
            RenderCmd::SetVisibilityResultMode { mode, offset } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::U64(mode), Arg::U64(offset)],
        )
    }
            RenderCmd::UpdateFence { fence, stages } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::obj(fence), Arg::U64(stages)],
        )
    }
            RenderCmd::UseHeaps { heaps } => {
        self.act(Policy::Caught, sel, vec![Arg::objs(heaps)])
    }
            RenderCmd::UseHeapsStages { heaps, stages } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::objs(heaps), Arg::U64(stages)],
        )
    }
            RenderCmd::UseResources { resources, usage } => {
        self.act(Policy::Caught,
            sel,
            vec![res_ids(resources), Arg::U64(usage)],
        )
    }
            RenderCmd::UseResourcesStages { resources, usage, stages } => {
        self.act(Policy::Caught,
            sel,
            vec![res_ids(resources), Arg::U64(usage), Arg::U64(stages)],
        )
    }
            RenderCmd::WaitForFence { fence, stages } => {
        self.act(Policy::Caught,
            sel,
            vec![Arg::obj(fence), Arg::U64(stages)],
        )
    }
        }
    }

    fn tile_height(&self) -> u64 {
        32
    }

    fn tile_width(&self) -> u64 {
        32
    }
}
