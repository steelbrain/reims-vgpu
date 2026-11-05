//! Platform trait `RenderEncoder`: every `MTLRenderCommandEncoder` selector core sends (D4).
//!
//! `RenderCmd::selector` names each command's selector. `RenderPipeline`'s own selectors are in
//! `platform/resource.rs`.
//!
//! Conventions (architecture 4.2): a method with arguments, and every encode, returns
//! `RaiseResult<T>`; the zero-argument getters `tile_width`/`tile_height` return `u64`. An object
//! argument is `Option<&P::X>` unless every caller is shown never to pass nil, and the `nil:` line
//! of such a method says why. A C array passed with a count or a range is a slice whose length is
//! that count or `range.length`; its elements are `Option` because they are filled from lookups
//! that return nil. An SPI selector the encoder does not implement raises
//! (`NSInvalidArgumentException`), which the render decoder's `@catch` turns into an exec fault,
//! so every SPI method returns `RaiseResult`. The five records whose selectors the host's encoder
//! does not implement raise the same themselves, without the send (`deserializer/render/state.rs`).

// One method per selector and one argument per keyword (D4): the patch draws take nine.
#![allow(clippy::too_many_arguments)]

use super::blit::ResourceRef;
use super::types::{
    MtlRegion, MtlScissorRect, MtlSize, MtlViewport, NsRange, VertexAmplificationViewMapping,
};
use super::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};

/// One `MTLRenderCommandEncoder` command.
///
/// One variant per selector, whose fields are the selector's arguments in order;
/// [`RenderEncoder::encode`] sends one.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the selector's arguments
pub enum RenderCmd<'a, P: Platform> {
    /// `dispatchThreadsPerTile:`
    DispatchThreadsPerTile { threads_per_tile: MtlSize },
    /// `dispatchThreadsPerTile:inRegion:` (SPI)
    /// Undeclared in every header: types from core's send, checked against the runtime encodings
    /// (`metal/tests/spi_encodings.rs`). The region is in pixels: every tile it touches runs a
    /// full tile of threads (measured on the M3 Max, macOS 26.6).
    DispatchThreadsPerTileInRegion {
        threads_per_tile: MtlSize,
        region: MtlRegion,
    },
    /// `dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` (SPI)
    /// Undeclared in every header: types from core's send, checked against the runtime encodings
    /// (`metal/tests/spi_encodings.rs`). The index is `unsigned int` (runtime `I`), and the
    /// command record's field is a `uint32_t`, sent as it is (D41).
    DispatchThreadsPerTileInRegionIndex {
        threads_per_tile: MtlSize,
        region: MtlRegion,
        render_target_array_index: u32,
    },
    /// `drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:`
    /// nil: `control_point_index_buffer`, `indirect_buffer` — never nil: the decoder raises on a
    /// nil indirect or control-point index buffer first.
    /// nil: `patch_index_buffer` is nil when the command's reference is 0 (no lookup is made).
    DrawIndexedPatchesIndirect {
        number_of_patch_control_points: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &'a P::Buffer,
        control_point_index_buffer_offset: u64,
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:`
    /// nil: `control_point_index_buffer` — never nil: the decoder raises on a nil control-point
    /// index buffer first.
    /// nil: `patch_index_buffer` is nil when the command's reference is 0 (no lookup is made).
    DrawIndexedPatches {
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &'a P::Buffer,
        control_point_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`
    /// nil: `index_buffer` — never nil: the decoder raises first ("indexed draw without an index
    /// buffer").
    DrawIndexedPrimitives {
        primitive_type: u64,
        index_count: u64,
        index_type: u64,
        index_buffer: &'a P::Buffer,
        index_buffer_offset: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`
    /// nil: `index_buffer` — never nil: the decoder raises first ("indexed draw without an index
    /// buffer").
    DrawIndexedPrimitivesInstances {
        primitive_type: u64,
        index_count: u64,
        index_type: u64,
        index_buffer: &'a P::Buffer,
        index_buffer_offset: u64,
        instance_count: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`
    /// nil: `index_buffer` — never nil: the decoder raises first ("indexed draw without an index
    /// buffer").
    DrawIndexedPrimitivesBase {
        primitive_type: u64,
        index_count: u64,
        index_type: u64,
        index_buffer: &'a P::Buffer,
        index_buffer_offset: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    /// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`
    /// nil: `index_buffer`, `indirect_buffer` — never nil: the decoder raises on a nil indirect or
    /// index buffer first.
    DrawIndexedPrimitivesIndirect {
        primitive_type: u64,
        index_type: u64,
        index_buffer: &'a P::Buffer,
        index_buffer_offset: u64,
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
    /// The graphics lane's only.
    DrawMeshThreadgroups {
        threadgroups_per_grid: MtlSize,
        threads_per_object_threadgroup: MtlSize,
        threads_per_mesh_threadgroup: MtlSize,
    },
    /// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
    /// nil: `indirect_buffer` — never nil: the graphics lane's decoder raises on a nil indirect
    /// buffer first.
    DrawMeshThreadgroupsIndirect {
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
        threads_per_object_threadgroup: MtlSize,
        threads_per_mesh_threadgroup: MtlSize,
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
    /// The graphics lane's only.
    DrawMeshThreads {
        threads_per_grid: MtlSize,
        threads_per_object_threadgroup: MtlSize,
        threads_per_mesh_threadgroup: MtlSize,
    },
    /// `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:`
    /// nil: `indirect_buffer` — never nil: the decoder raises on a nil indirect buffer first.
    /// nil: `patch_index_buffer` is nil when the command's reference is 0 (no lookup is made).
    DrawPatchesIndirect {
        number_of_patch_control_points: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:`
    /// nil: `patch_index_buffer` is nil when the command's reference is 0 (no lookup is made).
    DrawPatches {
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`
    /// nil: `indirect_buffer` — never nil: the decoder raises first ("indirect draw without an
    /// indirect buffer").
    DrawPrimitivesIndirect {
        primitive_type: u64,
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:`
    DrawPrimitives {
        primitive_type: u64,
        vertex_start: u64,
        vertex_count: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:`
    DrawPrimitivesInstances {
        primitive_type: u64,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`
    DrawPrimitivesBase {
        primitive_type: u64,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// `endEncoding`
    /// aborts: a memoryless attachment stored with a resolve (AGX framebuffer at the end of the
    /// pass: render_pass_memoryless, render_pass_memoryless_stencil_resolve)
    EndEncoding,
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`
    /// nil: `indirect_commandbuffer`, `indirect_range_buffer` — never nil: the decoder raises on
    /// either nil first.
    ExecuteCommandsIndirect {
        indirect_commandbuffer: &'a P::Icb,
        indirect_range_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `executeCommandsInBuffer:withRange:`
    /// nil: `indirect_command_buffer` — never nil: the decoder raises on nil first ("no indirect
    /// command buffer for reference …").
    ExecuteCommandsInBuffer {
        indirect_command_buffer: &'a P::Icb,
        execution_range: NsRange,
    },
    /// `memoryBarrierWithScope:afterStages:beforeStages:`
    MemoryBarrierWithScope { scope: u64, after: u64, before: u64 },
    /// `setAlphaTestReferenceValue:` (SPI)
    SetAlphaTestReferenceValue { reference_value: f32 },
    /// `setBlendColorRed:green:blue:alpha:`
    SetBlendColorRed {
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    },
    /// `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` (SPI)
    SetColorResolveTexture {
        texture: Option<&'a P::Texture>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
        index: u64,
    },
    /// `setColorStoreAction:atIndex:`
    SetColorStoreAction {
        store_action: u64,
        color_attachment_index: u64,
    },
    /// `setColorStoreActionOptions:atIndex:`
    SetColorStoreActionOptions {
        store_action_options: u64,
        color_attachment_index: u64,
    },
    /// `setCullMode:`
    SetCullMode { cull_mode: u64 },
    /// `setDepthBias:slopeScale:clamp:`
    SetDepthBias {
        depth_bias: f32,
        slope_scale: f32,
        clamp: f32,
    },
    /// `setDepthCleared` (SPI)
    SetDepthCleared,
    /// `setDepthClipModeSPI:` (SPI)
    SetDepthClipModeSpi { mode: u64 },
    /// `setDepthResolveTexture:slice:depthPlane:level:yInvert:` (SPI)
    SetDepthResolveTexture {
        texture: Option<&'a P::Texture>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
    },
    /// `setDepthStencilState:`
    SetDepthStencilState {
        depth_stencil_state: Option<&'a P::DepthStencilState>,
    },
    /// `setDepthStoreAction:`
    SetDepthStoreAction { store_action: u64 },
    /// `setDepthStoreActionOptions:`
    SetDepthStoreActionOptions { store_action_options: u64 },
    /// `setFragmentBuffer:offset:atIndex:`
    SetFragmentBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        index: u64,
    },
    /// `setFragmentBufferOffset:atIndex:`
    SetFragmentBufferOffset { offset: u64, index: u64 },
    /// `setFragmentBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetFragmentBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetFragmentSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setFragmentSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetFragmentSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setFragmentTexture:atIndex:`
    SetFragmentTexture {
        texture: Option<&'a P::Texture>,
        index: u64,
    },
    /// `setFragmentTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetFragmentTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setFrontFacingWinding:`
    SetFrontFacingWinding { front_facing_winding: u64 },
    /// `setLineWidth:` (SPI)
    SetLineWidth { line_width: f32 },
    /// `setMeshBufferOffset:atIndex:` (the graphics lane's only, as every mesh and object
    /// stage command)
    SetMeshBufferOffset { offset: u64, index: u64 },
    /// `setMeshBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetMeshBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetMeshSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setMeshSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetMeshSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setMeshTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetMeshTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setObjectBufferOffset:atIndex:`
    SetObjectBufferOffset { offset: u64, index: u64 },
    /// `setObjectBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetObjectBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetObjectSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setObjectSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetObjectSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setObjectTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetObjectTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setObjectThreadgroupMemoryLength:atIndex:`
    SetObjectThreadgroupMemoryLength { length: u64, index: u64 },
    /// `setPointSize:` (SPI)
    SetPointSize { point_size: f32 },
    /// `setPrimitiveRestartEnabled:index:` (SPI)
    SetPrimitiveRestartEnabled { enabled: bool, index: u64 },
    /// `setProvokingVertexMode:` (SPI)
    SetProvokingVertexMode { mode: u64 },
    /// `setRenderPipelineState:`
    SetRenderPipelineState {
        pipeline_state: Option<&'a P::RenderPipeline>,
    },
    /// `setScissorRect:`
    SetScissorRect { rect: MtlScissorRect },
    /// `setScissorRects:count:`
    SetScissorRects { scissor_rects: &'a [MtlScissorRect] },
    /// `setStencilCleared` (SPI)
    SetStencilCleared,
    /// `setStencilFrontReferenceValue:backReferenceValue:`
    SetStencilFrontReferenceValue {
        front_reference_value: u32,
        back_reference_value: u32,
    },
    /// `setStencilResolveTexture:slice:depthPlane:level:yInvert:` (SPI)
    SetStencilResolveTexture {
        texture: Option<&'a P::Texture>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
    },
    /// `setStencilStoreAction:`
    SetStencilStoreAction { store_action: u64 },
    /// `setStencilStoreActionOptions:`
    SetStencilStoreActionOptions { store_action_options: u64 },
    /// `setTessellationFactorBuffer:offset:instanceStride:`
    SetTessellationFactorBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        instance_stride: u64,
    },
    /// `setTessellationFactorScale:`
    SetTessellationFactorScale { scale: f32 },
    /// `setThreadgroupMemoryLength:offset:atIndex:`
    SetThreadgroupMemoryLength {
        length: u64,
        offset: u64,
        index: u64,
    },
    /// `setTileBufferOffset:atIndex:`
    SetTileBufferOffset { offset: u64, index: u64 },
    /// `setTileBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetTileBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetTileSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setTileSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetTileSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setTileTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetTileTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setTriangleFillMode:`
    SetTriangleFillMode { fill_mode: u64 },
    /// `setVertexAmplificationCount:viewMappings:`
    SetVertexAmplificationCount {
        view_mappings: &'a [VertexAmplificationViewMapping],
    },
    /// `setVertexAmplificationMode:value:` (SPI)
    SetVertexAmplificationMode { mode: u64, value: u64 },
    /// `setVertexBuffer:offset:atIndex:`
    SetVertexBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        index: u64,
    },
    /// `setVertexBufferOffset:atIndex:`
    SetVertexBufferOffset { offset: u64, index: u64 },
    /// `setVertexBufferOffset:attributeStride:atIndex:`
    SetVertexBufferOffsetAttributeStride {
        offset: u64,
        stride: u64,
        index: u64,
    },
    /// `setVertexBuffers:offsets:attributeStrides:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`. `strides`: its length is `range.length`.
    SetVertexBuffersAttributeStrides {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        strides: &'a [u64],
        range: NsRange,
    },
    /// `setVertexBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetVertexBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetVertexSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setVertexSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetVertexSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setVertexTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetVertexTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setViewport:`
    SetViewport { viewport: MtlViewport },
    /// `setViewportTransformEnabled:` (SPI)
    SetViewportTransformEnabled { enabled: bool },
    /// `setViewports:count:`
    SetViewports { viewports: &'a [MtlViewport] },
    /// `setVisibilityResultMode:offset:`
    SetVisibilityResultMode { mode: u64, offset: u64 },
    /// `updateFence:afterStages:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    UpdateFence { fence: &'a P::Fence, stages: u64 },
    /// `useHeaps:count:`
    UseHeaps { heaps: &'a [Option<&'a P::Heap>] },
    /// `useHeaps:count:stages:`
    UseHeapsStages {
        heaps: &'a [Option<&'a P::Heap>],
        stages: u64,
    },
    /// `useResources:count:usage:`
    UseResources {
        resources: &'a [Option<ResourceRef<'a, P>>],
        usage: u64,
    },
    /// `useResources:count:usage:stages:`
    UseResourcesStages {
        resources: &'a [Option<ResourceRef<'a, P>>],
        usage: u64,
        stages: u64,
    },
    /// `waitForFence:beforeStages:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    WaitForFence { fence: &'a P::Fence, stages: u64 },
}

impl<P: Platform> RenderCmd<'_, P> {
    /// The selector this sends.
    pub fn selector(&self) -> &'static str {
        match self {
            Self::DispatchThreadsPerTile { .. } => "dispatchThreadsPerTile:",
            Self::DispatchThreadsPerTileInRegion { .. } => "dispatchThreadsPerTile:inRegion:",
            Self::DispatchThreadsPerTileInRegionIndex { .. } => {
                "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:"
            }
            Self::DrawIndexedPatchesIndirect { .. } => {
                "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:"
            }
            Self::DrawIndexedPatches { .. } => {
                "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:"
            }
            Self::DrawIndexedPrimitives { .. } => {
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
            }
            Self::DrawIndexedPrimitivesInstances { .. } => {
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
            }
            Self::DrawIndexedPrimitivesBase { .. } => {
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"
            }
            Self::DrawIndexedPrimitivesIndirect { .. } => {
                "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:"
            }
            Self::DrawMeshThreadgroups { .. } => {
                "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"
            }
            Self::DrawMeshThreadgroupsIndirect { .. } => {
                "drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"
            }
            Self::DrawMeshThreads { .. } => {
                "drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"
            }
            Self::DrawPatchesIndirect { .. } => {
                "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:"
            }
            Self::DrawPatches { .. } => {
                "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:"
            }
            Self::DrawPrimitivesIndirect { .. } => {
                "drawPrimitives:indirectBuffer:indirectBufferOffset:"
            }
            Self::DrawPrimitives { .. } => "drawPrimitives:vertexStart:vertexCount:",
            Self::DrawPrimitivesInstances { .. } => {
                "drawPrimitives:vertexStart:vertexCount:instanceCount:"
            }
            Self::DrawPrimitivesBase { .. } => {
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"
            }
            Self::EndEncoding => "endEncoding",
            Self::ExecuteCommandsIndirect { .. } => {
                "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:"
            }
            Self::ExecuteCommandsInBuffer { .. } => "executeCommandsInBuffer:withRange:",
            Self::MemoryBarrierWithScope { .. } => {
                "memoryBarrierWithScope:afterStages:beforeStages:"
            }
            Self::SetAlphaTestReferenceValue { .. } => "setAlphaTestReferenceValue:",
            Self::SetBlendColorRed { .. } => "setBlendColorRed:green:blue:alpha:",
            Self::SetColorResolveTexture { .. } => {
                "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:"
            }
            Self::SetColorStoreAction { .. } => "setColorStoreAction:atIndex:",
            Self::SetColorStoreActionOptions { .. } => "setColorStoreActionOptions:atIndex:",
            Self::SetCullMode { .. } => "setCullMode:",
            Self::SetDepthBias { .. } => "setDepthBias:slopeScale:clamp:",
            Self::SetDepthCleared => "setDepthCleared",
            Self::SetDepthClipModeSpi { .. } => "setDepthClipModeSPI:",
            Self::SetDepthResolveTexture { .. } => {
                "setDepthResolveTexture:slice:depthPlane:level:yInvert:"
            }
            Self::SetDepthStencilState { .. } => "setDepthStencilState:",
            Self::SetDepthStoreAction { .. } => "setDepthStoreAction:",
            Self::SetDepthStoreActionOptions { .. } => "setDepthStoreActionOptions:",
            Self::SetFragmentBuffer { .. } => "setFragmentBuffer:offset:atIndex:",
            Self::SetFragmentBufferOffset { .. } => "setFragmentBufferOffset:atIndex:",
            Self::SetFragmentBuffers { .. } => "setFragmentBuffers:offsets:withRange:",
            Self::SetFragmentSamplerStatesLodClamps { .. } => {
                "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetFragmentSamplerStates { .. } => "setFragmentSamplerStates:withRange:",
            Self::SetFragmentTexture { .. } => "setFragmentTexture:atIndex:",
            Self::SetFragmentTextures { .. } => "setFragmentTextures:withRange:",
            Self::SetFrontFacingWinding { .. } => "setFrontFacingWinding:",
            Self::SetLineWidth { .. } => "setLineWidth:",
            Self::SetMeshBufferOffset { .. } => "setMeshBufferOffset:atIndex:",
            Self::SetMeshBuffers { .. } => "setMeshBuffers:offsets:withRange:",
            Self::SetMeshSamplerStatesLodClamps { .. } => {
                "setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetMeshSamplerStates { .. } => "setMeshSamplerStates:withRange:",
            Self::SetMeshTextures { .. } => "setMeshTextures:withRange:",
            Self::SetObjectBufferOffset { .. } => "setObjectBufferOffset:atIndex:",
            Self::SetObjectBuffers { .. } => "setObjectBuffers:offsets:withRange:",
            Self::SetObjectSamplerStatesLodClamps { .. } => {
                "setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetObjectSamplerStates { .. } => "setObjectSamplerStates:withRange:",
            Self::SetObjectTextures { .. } => "setObjectTextures:withRange:",
            Self::SetObjectThreadgroupMemoryLength { .. } => {
                "setObjectThreadgroupMemoryLength:atIndex:"
            }
            Self::SetPointSize { .. } => "setPointSize:",
            Self::SetPrimitiveRestartEnabled { .. } => "setPrimitiveRestartEnabled:index:",
            Self::SetProvokingVertexMode { .. } => "setProvokingVertexMode:",
            Self::SetRenderPipelineState { .. } => "setRenderPipelineState:",
            Self::SetScissorRect { .. } => "setScissorRect:",
            Self::SetScissorRects { .. } => "setScissorRects:count:",
            Self::SetStencilCleared => "setStencilCleared",
            Self::SetStencilFrontReferenceValue { .. } => {
                "setStencilFrontReferenceValue:backReferenceValue:"
            }
            Self::SetStencilResolveTexture { .. } => {
                "setStencilResolveTexture:slice:depthPlane:level:yInvert:"
            }
            Self::SetStencilStoreAction { .. } => "setStencilStoreAction:",
            Self::SetStencilStoreActionOptions { .. } => "setStencilStoreActionOptions:",
            Self::SetTessellationFactorBuffer { .. } => {
                "setTessellationFactorBuffer:offset:instanceStride:"
            }
            Self::SetTessellationFactorScale { .. } => "setTessellationFactorScale:",
            Self::SetThreadgroupMemoryLength { .. } => "setThreadgroupMemoryLength:offset:atIndex:",
            Self::SetTileBufferOffset { .. } => "setTileBufferOffset:atIndex:",
            Self::SetTileBuffers { .. } => "setTileBuffers:offsets:withRange:",
            Self::SetTileSamplerStatesLodClamps { .. } => {
                "setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetTileSamplerStates { .. } => "setTileSamplerStates:withRange:",
            Self::SetTileTextures { .. } => "setTileTextures:withRange:",
            Self::SetTriangleFillMode { .. } => "setTriangleFillMode:",
            Self::SetVertexAmplificationCount { .. } => "setVertexAmplificationCount:viewMappings:",
            Self::SetVertexAmplificationMode { .. } => "setVertexAmplificationMode:value:",
            Self::SetVertexBuffer { .. } => "setVertexBuffer:offset:atIndex:",
            Self::SetVertexBufferOffset { .. } => "setVertexBufferOffset:atIndex:",
            Self::SetVertexBufferOffsetAttributeStride { .. } => {
                "setVertexBufferOffset:attributeStride:atIndex:"
            }
            Self::SetVertexBuffersAttributeStrides { .. } => {
                "setVertexBuffers:offsets:attributeStrides:withRange:"
            }
            Self::SetVertexBuffers { .. } => "setVertexBuffers:offsets:withRange:",
            Self::SetVertexSamplerStatesLodClamps { .. } => {
                "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetVertexSamplerStates { .. } => "setVertexSamplerStates:withRange:",
            Self::SetVertexTextures { .. } => "setVertexTextures:withRange:",
            Self::SetViewport { .. } => "setViewport:",
            Self::SetViewportTransformEnabled { .. } => "setViewportTransformEnabled:",
            Self::SetViewports { .. } => "setViewports:count:",
            Self::SetVisibilityResultMode { .. } => "setVisibilityResultMode:offset:",
            Self::UpdateFence { .. } => "updateFence:afterStages:",
            Self::UseHeaps { .. } => "useHeaps:count:",
            Self::UseHeapsStages { .. } => "useHeaps:count:stages:",
            Self::UseResources { .. } => "useResources:count:usage:",
            Self::UseResourcesStages { .. } => "useResources:count:usage:stages:",
            Self::WaitForFence { .. } => "waitForFence:beforeStages:",
        }
    }
}

/// One draw of [`RenderEncoder::encode_draws`]: [`RenderCmd::SetVertexBuffer`] of `buffer`,
/// `offset` and `index`, then [`RenderCmd::DrawPrimitives`] of `primitive_type`, `vertex_start`
/// and `vertex_count`.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the two commands' arguments
pub struct VertexDraw<'a, P: Platform> {
    pub buffer: Option<&'a P::Buffer>,
    pub offset: u64,
    pub index: u64,
    pub primitive_type: u64,
    pub vertex_start: u64,
    pub vertex_count: u64,
}

impl<P: Platform> Clone for VertexDraw<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Platform> Copy for VertexDraw<'_, P> {}

impl<'a, P: Platform> VertexDraw<'a, P> {
    /// Its `setVertexBuffer:offset:atIndex:`.
    pub fn vertex_buffer(&self) -> RenderCmd<'a, P> {
        RenderCmd::SetVertexBuffer {
            buffer: self.buffer,
            offset: self.offset,
            index: self.index,
        }
    }

    /// Its `drawPrimitives:vertexStart:vertexCount:`.
    pub fn primitives(&self) -> RenderCmd<'a, P> {
        RenderCmd::DrawPrimitives {
            primitive_type: self.primitive_type,
            vertex_start: self.vertex_start,
            vertex_count: self.vertex_count,
        }
    }
}

/// [`RenderEncoder::encode_draws`] as each platform's but the one that takes them its own way
/// sends them: each draw's two commands through [`RenderEncoder::encode`], in order, up to the
/// first that raises.
pub fn encode_each<P: Platform, E: RenderEncoder<P> + ?Sized>(
    enc: &E,
    on: OnFault,
    draws: &[VertexDraw<'_, P>],
) -> RaiseResult<()> {
    for draw in draws {
        enc.encode(on, draw.vertex_buffer())?;
        enc.encode(on, draw.primitives())?;
    }
    Ok(())
}

/// The render encoder `-renderCommandEncoderWithDescriptor:` answers (`CommandBuffer`).
pub trait RenderEncoder<P: Platform>: Handle {
    /// Sends one [`RenderCmd`]. `on` says what a raise in a command reached both under a
    /// handler and with none (the display's frame commands and `EndEncoding`) does; the other
    /// commands are reached one way only.
    fn encode(&self, on: OnFault, cmd: RenderCmd<'_, P>) -> RaiseResult<()>;
    /// Sends each of `draws`, one after another, as [`encode_each`] does: each command does what
    /// it does alone, in its order, and the first that raises ends them. A platform may take
    /// them its own way, where that is all it does.
    fn encode_draws(&self, on: OnFault, draws: &[VertexDraw<'_, P>]) -> RaiseResult<()> {
        encode_each(self, on, draws)
    }
    /// Runs `run`, a run of [`Self::encode`]s this thread sends the encoder one after another,
    /// with nothing else sent it between them. Each command does what it does alone, in its
    /// order; a platform may hold the encoder for the run, so that each command takes less.
    fn encoding<T>(&self, run: impl FnOnce() -> T) -> T
    where
        Self: Sized,
    {
        run()
    }
    /// sel: tileHeight
    /// Declared on MTLRenderCommandEncoder (protocol).
    /// A zero-argument getter: a raise dies at the send (4.2 item 3).
    fn tile_height(&self) -> u64;
    /// sel: tileWidth
    /// Declared on MTLRenderCommandEncoder (protocol).
    /// A zero-argument getter: a raise dies at the send (4.2 item 3).
    fn tile_width(&self) -> u64;
}
