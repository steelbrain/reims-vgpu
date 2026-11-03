//! The render-encoder op decoder: one arm per [`reims_graphics::ops::render`] op,
//! turning a wire op into a `RenderCmd` on the open `P::RenderEncoder`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The command-buffer
//! walk ([`super::cmdbuf`]) routes every render-group op
//! (`tag >> 12 == `[`RGFX_OP_GROUP_RENDER`](reims_graphics::ops::RGFX_OP_GROUP_RENDER))
//! here.
//!
//! Each arm reads its op's body with zerocopy, resolves its object references
//! through the [object-resolution API](super) (`require_*`/`get_*`), and sends the
//! matching [`RenderCmd`] with [`RenderEncoder::encode`]. A body too short for its
//! op faults the exec ([`super::op_fault`]); a bad reference faults through the
//! resolution API.
//!
//! This covers the whole `RenderCommand` set: the state setters, every draw,
//! indexed draw, patch draw and indirect draw, the per-tile dispatches, the
//! `useResource(s)`/`useHeap(s)` binds, the memory barrier, fence update and wait,
//! `executeCommands`, the colour/depth/stencil store actions and resolve textures,
//! tessellation, the viewport and scissor arrays, amplification and visibility. One
//! command — `endEncoding` — has no op here: it is the framing `ENCODER_END`, which
//! the command-buffer walk ([`super::cmdbuf`]) sends when it drops the encoder.
//!
//! The array-tailed binds (`setVertexBuffers:`, `useResources:`, `setViewports:`, …)
//! read a fixed head then the trailing element array ([`elems`]), resolve each
//! element's reference, and pass the encode a local `Vec` of the decoder's handles,
//! borrowed for the op.

// The render-op tags and body structs (`Rgfx*`): the whole render group, so a glob
// (as the Metal platform's command impl globs `platform::render`).
use reims_graphics::ops::render::*;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::types::{
    MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, MtlViewport, NsRange,
    VertexAmplificationViewMapping,
};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use zerocopy::FromBytes;

use super::{GraphicsDecoder, op_fault};

/// Reads op body `T` from `payload`, or an exec fault when it is too short.
#[inline]
fn body<T: FromBytes>(tag: u16, payload: &[u8]) -> RaiseResult<T> {
    match T::read_from_prefix(payload) {
        Ok((t, _)) => Ok(t),
        Err(_) => Err(short_body(tag)),
    }
}

/// The exec fault of render op `tag`, whose body is too short: out of line, as no well-formed
/// op meets it.
#[cold]
#[inline(never)]
fn short_body(tag: u16) -> Raise {
    op_fault(format_args!("render op {tag:#06x}: body too short"))
}

/// Reads `count` trailing elements of `T` from `payload`, the first at `head` (the
/// head struct's `size_of`). An exec fault when the body is too short for them. The
/// claimed `count` is checked against the body before any allocation.
fn elems<T: FromBytes>(tag: u16, payload: &[u8], head: usize, count: usize) -> RaiseResult<Vec<T>> {
    let stride = size_of::<T>();
    let avail = payload.len().saturating_sub(head) / stride;
    if count > avail {
        return Err(op_fault(format_args!(
            "render op {tag:#06x}: body too short for its {count}-element array"
        )));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = head + i * stride;
        // The bound above guarantees the slice is one element wide; a defensive
        // fault stands in for the unreachable short read.
        let Ok(e) = T::read_from_bytes(&payload[off..off + stride]) else {
            return Err(op_fault(format_args!(
                "render op {tag:#06x}: malformed array element"
            )));
        };
        out.push(e);
    }
    Ok(out)
}

/// Resolves one resource-array element: `None` for 0, else the object its
/// [kind](reims_graphics::ops::render::RGFX_RESOURCE_KIND_BUFFER) names (a miss or an
/// unknown kind faults).
fn resolve_resource<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    e: &RgfxResourceBinding,
) -> RaiseResult<Option<ResourceRef<'a, P>>> {
    if e.obj_id == 0 {
        return Ok(None);
    }
    Ok(Some(match e.kind {
        RGFX_RESOURCE_KIND_NONE => return Ok(None),
        RGFX_RESOURCE_KIND_BUFFER => ResourceRef::Buffer(objs.require_buffer(e.obj_id)?),
        RGFX_RESOURCE_KIND_TEXTURE => ResourceRef::Texture(objs.require_texture(e.obj_id)?),
        RGFX_RESOURCE_KIND_ICB => ResourceRef::Icb(objs.require_indirect_command_buffer(e.obj_id)?),
        other => {
            return Err(op_fault(format_args!(
                "useResources: resource {} has unknown kind {other}",
                e.obj_id
            )));
        }
    }))
}

/// Resolves a buffer-binding array into its objects (nil for 0) and their offsets, in
/// order.
#[allow(clippy::type_complexity)] // the pair mirrors the `buffers`/`offsets` arguments
fn resolve_buffers<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    es: &[RgfxBufferBinding],
) -> RaiseResult<(Vec<Option<&'a P::Buffer>>, Vec<u64>)> {
    let mut named = Vec::with_capacity(es.len());
    let mut offsets = Vec::with_capacity(es.len());
    for e in es {
        named.push(objs.get_buffer(e.buffer_obj_id)?);
        offsets.push(e.offset);
    }
    Ok((named, offsets))
}

/// Resolves a texture-binding array into its objects (nil for 0), in order.
fn resolve_textures<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    es: &[RgfxTextureBinding],
) -> RaiseResult<Vec<Option<&'a P::Texture>>> {
    es.iter()
        .map(|e| objs.get_texture(e.texture_obj_id))
        .collect()
}

/// Resolves a sampler-binding array into its objects (nil for 0), in order.
fn resolve_samplers<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    es: &[RgfxSamplerBinding],
) -> RaiseResult<Vec<Option<&'a P::SamplerState>>> {
    es.iter()
        .map(|e| objs.get_sampler(e.sampler_obj_id))
        .collect()
}

/// Resolves a sampler-with-LOD-clamps array into its objects (nil for 0) and the min
/// and max LOD clamps, each in order.
#[allow(clippy::type_complexity)] // the triple mirrors the three array arguments
fn resolve_samplers_lod<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    es: &[RgfxSamplerLodBinding],
) -> RaiseResult<(Vec<Option<&'a P::SamplerState>>, Vec<f32>, Vec<f32>)> {
    let mut named = Vec::with_capacity(es.len());
    let mut mins = Vec::with_capacity(es.len());
    let mut maxs = Vec::with_capacity(es.len());
    for e in es {
        named.push(objs.get_sampler(e.sampler_obj_id)?);
        mins.push(f32::from_bits(e.lod_min_clamp_bits));
        maxs.push(f32::from_bits(e.lod_max_clamp_bits));
    }
    Ok((named, mins, maxs))
}

/// Resolves a heap array into its objects (nil for 0), in order.
fn resolve_heaps<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    es: &[RgfxHeapBinding],
) -> RaiseResult<Vec<Option<&'a P::Heap>>> {
    es.iter().map(|e| objs.get_heap(e.heap_obj_id)).collect()
}

/// Decodes one render-encoder op: `tag` its op, `payload` its body (the bytes
/// after the [`RgfxOpHeader`](reims_graphics::ops::RgfxOpHeader)). `objs` resolves
/// object references, `enc` is the open render encoder, and `on` says how an
/// encode's raise is taken. `Ok(true)` when the op was handled, `Ok(false)` for an
/// unknown tag (the walk then faults); a platform raise or a bad reference
/// propagates as `Err`.
pub(crate) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    enc: &P::RenderEncoder,
    on: OnFault,
) -> RaiseResult<bool> {
    match tag {
        RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE => {
            let m: RgfxOpSetRenderPipelineState = body(tag, payload)?;
            let pipeline = objs.get_render_pipeline(m.pipeline_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: pipeline,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_BUFFER => {
            let m: RgfxOpSetVertexBuffer = body(tag, payload)?;
            let buffer = objs.get_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetVertexBuffer {
                    buffer,
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_BUFFER => {
            let m: RgfxOpSetFragmentBuffer = body(tag, payload)?;
            let buffer = objs.get_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetFragmentBuffer {
                    buffer,
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE => {
            let m: RgfxOpSetFragmentTexture = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetFragmentTexture {
                    texture,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VIEWPORT => {
            let m: RgfxOpSetViewport = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetViewport {
                    viewport: MtlViewport {
                        origin_x: f64::from_bits(m.origin_x_bits),
                        origin_y: f64::from_bits(m.origin_y_bits),
                        width: f64::from_bits(m.width_bits),
                        height: f64::from_bits(m.height_bits),
                        znear: f64::from_bits(m.znear_bits),
                        zfar: f64::from_bits(m.zfar_bits),
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_SCISSOR_RECT => {
            let m: RgfxOpSetScissorRect = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetScissorRect {
                    rect: MtlScissorRect {
                        x: m.x,
                        y: m.y,
                        width: m.width,
                        height: m.height,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_PRIMITIVES => {
            let m: RgfxOpDrawPrimitives = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DrawPrimitives {
                    primitive_type: m.primitive_type,
                    vertex_start: m.vertex_start,
                    vertex_count: m.vertex_count,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES => {
            let m: RgfxOpDrawIndexedPrimitives = body(tag, payload)?;
            // The index buffer must name an object (an indexed draw without one).
            let index_buffer = objs.require_buffer(m.index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPrimitives {
                    primitive_type: m.primitive_type,
                    index_count: m.index_count,
                    index_type: m.index_type,
                    index_buffer,
                    index_buffer_offset: m.index_buffer_offset,
                },
            )?;
        }

        // ---- Draws ----
        RGFX_OP_RENDER_DRAW_PRIMITIVES_INSTANCES => {
            let m: RgfxOpDrawPrimitivesInstances = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DrawPrimitivesInstances {
                    primitive_type: m.primitive_type,
                    vertex_start: m.vertex_start,
                    vertex_count: m.vertex_count,
                    instance_count: m.instance_count,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_PRIMITIVES_BASE => {
            let m: RgfxOpDrawPrimitivesBase = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DrawPrimitivesBase {
                    primitive_type: m.primitive_type,
                    vertex_start: m.vertex_start,
                    vertex_count: m.vertex_count,
                    instance_count: m.instance_count,
                    base_instance: m.base_instance,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_PRIMITIVES_INDIRECT => {
            let m: RgfxOpDrawPrimitivesIndirect = body(tag, payload)?;
            // The indirect buffer must name an object.
            let indirect_buffer = objs.require_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawPrimitivesIndirect {
                    primitive_type: m.primitive_type,
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES => {
            let m: RgfxOpDrawIndexedPrimitivesInstances = body(tag, payload)?;
            let index_buffer = objs.require_buffer(m.index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPrimitivesInstances {
                    primitive_type: m.primitive_type,
                    index_count: m.index_count,
                    index_type: m.index_type,
                    index_buffer,
                    index_buffer_offset: m.index_buffer_offset,
                    instance_count: m.instance_count,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_BASE => {
            let m: RgfxOpDrawIndexedPrimitivesBase = body(tag, payload)?;
            let index_buffer = objs.require_buffer(m.index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPrimitivesBase {
                    primitive_type: m.primitive_type,
                    index_count: m.index_count,
                    index_type: m.index_type,
                    index_buffer,
                    index_buffer_offset: m.index_buffer_offset,
                    instance_count: m.instance_count,
                    base_vertex: m.base_vertex,
                    base_instance: m.base_instance,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INDIRECT => {
            let m: RgfxOpDrawIndexedPrimitivesIndirect = body(tag, payload)?;
            let index_buffer = objs.require_buffer(m.index_buffer_obj_id)?;
            let indirect_buffer = objs.require_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPrimitivesIndirect {
                    primitive_type: m.primitive_type,
                    index_type: m.index_type,
                    index_buffer,
                    index_buffer_offset: m.index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }

        // ---- Patches ----
        RGFX_OP_RENDER_DRAW_PATCHES => {
            let m: RgfxOpDrawPatches = body(tag, payload)?;
            let patch_index_buffer = objs.get_buffer(m.patch_index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawPatches {
                    number_of_patch_control_points: m.number_of_patch_control_points,
                    patch_start: m.patch_start,
                    patch_count: m.patch_count,
                    patch_index_buffer,
                    patch_index_buffer_offset: m.patch_index_buffer_offset,
                    instance_count: m.instance_count,
                    base_instance: m.base_instance,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_PATCHES_INDIRECT => {
            let m: RgfxOpDrawPatchesIndirect = body(tag, payload)?;
            let patch_index_buffer = objs.get_buffer(m.patch_index_buffer_obj_id)?;
            let indirect_buffer = objs.require_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawPatchesIndirect {
                    number_of_patch_control_points: m.number_of_patch_control_points,
                    patch_index_buffer,
                    patch_index_buffer_offset: m.patch_index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PATCHES => {
            let m: RgfxOpDrawIndexedPatches = body(tag, payload)?;
            let patch_index_buffer = objs.get_buffer(m.patch_index_buffer_obj_id)?;
            let control_point_index_buffer =
                objs.require_buffer(m.control_point_index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPatches {
                    number_of_patch_control_points: m.number_of_patch_control_points,
                    patch_start: m.patch_start,
                    patch_count: m.patch_count,
                    patch_index_buffer,
                    patch_index_buffer_offset: m.patch_index_buffer_offset,
                    control_point_index_buffer,
                    control_point_index_buffer_offset: m.control_point_index_buffer_offset,
                    instance_count: m.instance_count,
                    base_instance: m.base_instance,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PATCHES_INDIRECT => {
            let m: RgfxOpDrawIndexedPatchesIndirect = body(tag, payload)?;
            let patch_index_buffer = objs.get_buffer(m.patch_index_buffer_obj_id)?;
            let control_point_index_buffer =
                objs.require_buffer(m.control_point_index_buffer_obj_id)?;
            let indirect_buffer = objs.require_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPatchesIndirect {
                    number_of_patch_control_points: m.number_of_patch_control_points,
                    patch_index_buffer,
                    patch_index_buffer_offset: m.patch_index_buffer_offset,
                    control_point_index_buffer,
                    control_point_index_buffer_offset: m.control_point_index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }

        // ---- Dispatch threads per tile ----
        RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE => {
            let m: RgfxOpDispatchThreadsPerTile = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DispatchThreadsPerTile {
                    threads_per_tile: MtlSize {
                        width: m.width,
                        height: m.height,
                        depth: m.depth,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION => {
            let m: RgfxOpDispatchThreadsPerTileInRegion = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DispatchThreadsPerTileInRegion {
                    threads_per_tile: MtlSize {
                        width: m.width,
                        height: m.height,
                        depth: m.depth,
                    },
                    region: MtlRegion {
                        origin: MtlOrigin {
                            x: m.origin_x,
                            y: m.origin_y,
                            z: m.origin_z,
                        },
                        size: MtlSize {
                            width: m.region_width,
                            height: m.region_height,
                            depth: m.region_depth,
                        },
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION_INDEX => {
            let m: RgfxOpDispatchThreadsPerTileInRegionIndex = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DispatchThreadsPerTileInRegionIndex {
                    threads_per_tile: MtlSize {
                        width: m.width,
                        height: m.height,
                        depth: m.depth,
                    },
                    region: MtlRegion {
                        origin: MtlOrigin {
                            x: m.origin_x,
                            y: m.origin_y,
                            z: m.origin_z,
                        },
                        size: MtlSize {
                            width: m.region_width,
                            height: m.region_height,
                            depth: m.region_depth,
                        },
                    },
                    render_target_array_index: m.render_target_array_index,
                },
            )?;
        }

        // ---- Execute indirect command buffers ----
        RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER => {
            let m: RgfxOpExecuteCommandsInBuffer = body(tag, payload)?;
            let icb = objs.require_indirect_command_buffer(m.indirect_command_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: icb,
                    execution_range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT => {
            let m: RgfxOpExecuteCommandsIndirect = body(tag, payload)?;
            let icb = objs.require_indirect_command_buffer(m.indirect_command_buffer_obj_id)?;
            let range_buffer = objs.require_buffer(m.indirect_range_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::ExecuteCommandsIndirect {
                    indirect_commandbuffer: icb,
                    indirect_range_buffer: range_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }

        // ---- Barrier and fences ----
        RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE => {
            let m: RgfxOpMemoryBarrierWithScope = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::MemoryBarrierWithScope {
                    scope: m.scope,
                    after: m.after,
                    before: m.before,
                },
            )?;
        }
        RGFX_OP_RENDER_UPDATE_FENCE => {
            let m: RgfxOpUpdateFence = body(tag, payload)?;
            // The fence must name an object (core gets it from a non-nil getter).
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(
                on,
                RenderCmd::UpdateFence {
                    fence,
                    stages: m.stages,
                },
            )?;
        }
        RGFX_OP_RENDER_WAIT_FOR_FENCE => {
            let m: RgfxOpWaitForFence = body(tag, payload)?;
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(
                on,
                RenderCmd::WaitForFence {
                    fence,
                    stages: m.stages,
                },
            )?;
        }

        // ---- Scalar state setters ----
        RGFX_OP_RENDER_SET_CULL_MODE => {
            let m: RgfxOpSetCullMode = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetCullMode {
                    cull_mode: m.cull_mode,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRONT_FACING_WINDING => {
            let m: RgfxOpSetFrontFacingWinding = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetFrontFacingWinding {
                    front_facing_winding: m.front_facing_winding,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TRIANGLE_FILL_MODE => {
            let m: RgfxOpSetTriangleFillMode = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetTriangleFillMode {
                    fill_mode: m.fill_mode,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_DEPTH_CLIP_MODE_SPI => {
            let m: RgfxOpSetDepthClipModeSpi = body(tag, payload)?;
            enc.encode(on, RenderCmd::SetDepthClipModeSpi { mode: m.mode })?;
        }
        RGFX_OP_RENDER_SET_PROVOKING_VERTEX_MODE => {
            let m: RgfxOpSetProvokingVertexMode = body(tag, payload)?;
            enc.encode(on, RenderCmd::SetProvokingVertexMode { mode: m.mode })?;
        }
        RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION => {
            let m: RgfxOpSetDepthStoreAction = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetDepthStoreAction {
                    store_action: m.store_action,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION => {
            let m: RgfxOpSetStencilStoreAction = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetStencilStoreAction {
                    store_action: m.store_action,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION_OPTIONS => {
            let m: RgfxOpSetStencilStoreActionOptions = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetStencilStoreActionOptions {
                    store_action_options: m.store_action_options,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS => {
            let m: RgfxOpSetDepthStoreActionOptions = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetDepthStoreActionOptions {
                    store_action_options: m.store_action_options,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_COLOR_STORE_ACTION => {
            let m: RgfxOpSetColorStoreAction = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetColorStoreAction {
                    store_action: m.store_action,
                    color_attachment_index: m.color_attachment_index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_COLOR_STORE_ACTION_OPTIONS => {
            let m: RgfxOpSetColorStoreActionOptions = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetColorStoreActionOptions {
                    store_action_options: m.store_action_options,
                    color_attachment_index: m.color_attachment_index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE => {
            let m: RgfxOpSetVisibilityResultMode = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetVisibilityResultMode {
                    mode: m.mode,
                    offset: m.offset,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_MODE => {
            let m: RgfxOpSetVertexAmplificationMode = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetVertexAmplificationMode {
                    mode: m.mode,
                    value: m.value,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET => {
            let m: RgfxOpSetVertexBufferOffset = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetVertexBufferOffset {
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_BUFFER_OFFSET => {
            let m: RgfxOpSetFragmentBufferOffset = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetFragmentBufferOffset {
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TILE_BUFFER_OFFSET => {
            let m: RgfxOpSetTileBufferOffset = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetTileBufferOffset {
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET_ATTRIBUTE_STRIDE => {
            let m: RgfxOpSetVertexBufferOffsetAttributeStride = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetVertexBufferOffsetAttributeStride {
                    offset: m.offset,
                    stride: m.stride,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_THREADGROUP_MEMORY_LENGTH => {
            let m: RgfxOpSetThreadgroupMemoryLength = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetThreadgroupMemoryLength {
                    length: m.length,
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }

        // ---- Float setters (arguments as `f32` bits) ----
        RGFX_OP_RENDER_SET_ALPHA_TEST_REFERENCE_VALUE => {
            let m: RgfxOpSetAlphaTestReferenceValue = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetAlphaTestReferenceValue {
                    reference_value: f32::from_bits(m.reference_value_bits),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_LINE_WIDTH => {
            let m: RgfxOpSetLineWidth = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetLineWidth {
                    line_width: f32::from_bits(m.line_width_bits),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_POINT_SIZE => {
            let m: RgfxOpSetPointSize = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetPointSize {
                    point_size: f32::from_bits(m.point_size_bits),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_SCALE => {
            let m: RgfxOpSetTessellationFactorScale = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetTessellationFactorScale {
                    scale: f32::from_bits(m.scale_bits),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_BLEND_COLOR_RED => {
            let m: RgfxOpSetBlendColorRed = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetBlendColorRed {
                    red: f32::from_bits(m.red_bits),
                    green: f32::from_bits(m.green_bits),
                    blue: f32::from_bits(m.blue_bits),
                    alpha: f32::from_bits(m.alpha_bits),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_DEPTH_BIAS => {
            let m: RgfxOpSetDepthBias = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetDepthBias {
                    depth_bias: f32::from_bits(m.depth_bias_bits),
                    slope_scale: f32::from_bits(m.slope_scale_bits),
                    clamp: f32::from_bits(m.clamp_bits),
                },
            )?;
        }

        // ---- Boolean setters ----
        RGFX_OP_RENDER_SET_VIEWPORT_TRANSFORM_ENABLED => {
            let m: RgfxOpSetViewportTransformEnabled = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetViewportTransformEnabled {
                    enabled: m.enabled != 0,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_PRIMITIVE_RESTART_ENABLED => {
            let m: RgfxOpSetPrimitiveRestartEnabled = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetPrimitiveRestartEnabled {
                    enabled: m.enabled != 0,
                    index: m.index,
                },
            )?;
        }

        // ---- Stencil reference and clear flags ----
        RGFX_OP_RENDER_SET_STENCIL_FRONT_REFERENCE_VALUE => {
            let m: RgfxOpSetStencilFrontReferenceValue = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetStencilFrontReferenceValue {
                    front_reference_value: m.front_reference_value,
                    back_reference_value: m.back_reference_value,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_DEPTH_CLEARED => {
            enc.encode(on, RenderCmd::SetDepthCleared)?;
        }
        RGFX_OP_RENDER_SET_STENCIL_CLEARED => {
            enc.encode(on, RenderCmd::SetStencilCleared)?;
        }

        // ---- Depth/stencil state and resolve textures ----
        RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE => {
            let m: RgfxOpSetDepthStencilState = body(tag, payload)?;
            let state = objs.get_depth_stencil(m.depth_stencil_state_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetDepthStencilState {
                    depth_stencil_state: state,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_COLOR_RESOLVE_TEXTURE => {
            let m: RgfxOpSetColorResolveTexture = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetColorResolveTexture {
                    texture,
                    slice: m.slice,
                    depth_plane: m.depth_plane,
                    level: m.level,
                    y_invert: m.y_invert != 0,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_DEPTH_RESOLVE_TEXTURE => {
            let m: RgfxOpSetDepthResolveTexture = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetDepthResolveTexture {
                    texture,
                    slice: m.slice,
                    depth_plane: m.depth_plane,
                    level: m.level,
                    y_invert: m.y_invert != 0,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_STENCIL_RESOLVE_TEXTURE => {
            let m: RgfxOpSetStencilResolveTexture = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetStencilResolveTexture {
                    texture,
                    slice: m.slice,
                    depth_plane: m.depth_plane,
                    level: m.level,
                    y_invert: m.y_invert != 0,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_BUFFER => {
            let m: RgfxOpSetTessellationFactorBuffer = body(tag, payload)?;
            let buffer = objs.get_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetTessellationFactorBuffer {
                    buffer,
                    offset: m.offset,
                    instance_stride: m.instance_stride,
                },
            )?;
        }

        // ---- Array-tailed: buffers ----
        RGFX_OP_RENDER_SET_VERTEX_BUFFERS => {
            let m: RgfxOpSetVertexBuffers = body(tag, payload)?;
            let es: Vec<RgfxBufferBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexBuffers>(),
                m.range_length as usize,
            )?;
            let (named, offsets) = resolve_buffers(objs, &es)?;
            let buffers: Vec<Option<&P::Buffer>> = named;
            enc.encode(
                on,
                RenderCmd::SetVertexBuffers {
                    buffers: &buffers,
                    offsets: &offsets,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_BUFFERS => {
            let m: RgfxOpSetFragmentBuffers = body(tag, payload)?;
            let es: Vec<RgfxBufferBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetFragmentBuffers>(),
                m.range_length as usize,
            )?;
            let (named, offsets) = resolve_buffers(objs, &es)?;
            let buffers: Vec<Option<&P::Buffer>> = named;
            enc.encode(
                on,
                RenderCmd::SetFragmentBuffers {
                    buffers: &buffers,
                    offsets: &offsets,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TILE_BUFFERS => {
            let m: RgfxOpSetTileBuffers = body(tag, payload)?;
            let es: Vec<RgfxBufferBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetTileBuffers>(),
                m.range_length as usize,
            )?;
            let (named, offsets) = resolve_buffers(objs, &es)?;
            let buffers: Vec<Option<&P::Buffer>> = named;
            enc.encode(
                on,
                RenderCmd::SetTileBuffers {
                    buffers: &buffers,
                    offsets: &offsets,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES => {
            let m: RgfxOpSetVertexBuffersAttributeStrides = body(tag, payload)?;
            let es: Vec<RgfxBufferStrideBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexBuffersAttributeStrides>(),
                m.range_length as usize,
            )?;
            let mut named = Vec::with_capacity(es.len());
            let mut offsets = Vec::with_capacity(es.len());
            let mut strides = Vec::with_capacity(es.len());
            for e in &es {
                named.push(objs.get_buffer(e.buffer_obj_id)?);
                offsets.push(e.offset);
                strides.push(e.stride);
            }
            let buffers: Vec<Option<&P::Buffer>> = named;
            enc.encode(
                on,
                RenderCmd::SetVertexBuffersAttributeStrides {
                    buffers: &buffers,
                    offsets: &offsets,
                    strides: &strides,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }

        // ---- Array-tailed: textures ----
        RGFX_OP_RENDER_SET_VERTEX_TEXTURES => {
            let m: RgfxOpSetVertexTextures = body(tag, payload)?;
            let es: Vec<RgfxTextureBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexTextures>(),
                m.range_length as usize,
            )?;
            let named = resolve_textures(objs, &es)?;
            let textures: Vec<Option<&P::Texture>> = named;
            enc.encode(
                on,
                RenderCmd::SetVertexTextures {
                    textures: &textures,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURES => {
            let m: RgfxOpSetFragmentTextures = body(tag, payload)?;
            let es: Vec<RgfxTextureBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetFragmentTextures>(),
                m.range_length as usize,
            )?;
            let named = resolve_textures(objs, &es)?;
            let textures: Vec<Option<&P::Texture>> = named;
            enc.encode(
                on,
                RenderCmd::SetFragmentTextures {
                    textures: &textures,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TILE_TEXTURES => {
            let m: RgfxOpSetTileTextures = body(tag, payload)?;
            let es: Vec<RgfxTextureBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetTileTextures>(),
                m.range_length as usize,
            )?;
            let named = resolve_textures(objs, &es)?;
            let textures: Vec<Option<&P::Texture>> = named;
            enc.encode(
                on,
                RenderCmd::SetTileTextures {
                    textures: &textures,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }

        // ---- Array-tailed: samplers ----
        RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES => {
            let m: RgfxOpSetVertexSamplerStates = body(tag, payload)?;
            let es: Vec<RgfxSamplerBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexSamplerStates>(),
                m.range_length as usize,
            )?;
            let named = resolve_samplers(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetVertexSamplerStates {
                    samplers: &samplers,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES => {
            let m: RgfxOpSetFragmentSamplerStates = body(tag, payload)?;
            let es: Vec<RgfxSamplerBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetFragmentSamplerStates>(),
                m.range_length as usize,
            )?;
            let named = resolve_samplers(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetFragmentSamplerStates {
                    samplers: &samplers,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES => {
            let m: RgfxOpSetTileSamplerStates = body(tag, payload)?;
            let es: Vec<RgfxSamplerBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetTileSamplerStates>(),
                m.range_length as usize,
            )?;
            let named = resolve_samplers(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetTileSamplerStates {
                    samplers: &samplers,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES_LOD_CLAMPS => {
            let m: RgfxOpSetVertexSamplerStatesLodClamps = body(tag, payload)?;
            let es: Vec<RgfxSamplerLodBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexSamplerStatesLodClamps>(),
                m.range_length as usize,
            )?;
            let (named, mins, maxs) = resolve_samplers_lod(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetVertexSamplerStatesLodClamps {
                    samplers: &samplers,
                    lod_min_clamps: &mins,
                    lod_max_clamps: &maxs,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES_LOD_CLAMPS => {
            let m: RgfxOpSetFragmentSamplerStatesLodClamps = body(tag, payload)?;
            let es: Vec<RgfxSamplerLodBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetFragmentSamplerStatesLodClamps>(),
                m.range_length as usize,
            )?;
            let (named, mins, maxs) = resolve_samplers_lod(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetFragmentSamplerStatesLodClamps {
                    samplers: &samplers,
                    lod_min_clamps: &mins,
                    lod_max_clamps: &maxs,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES_LOD_CLAMPS => {
            let m: RgfxOpSetTileSamplerStatesLodClamps = body(tag, payload)?;
            let es: Vec<RgfxSamplerLodBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetTileSamplerStatesLodClamps>(),
                m.range_length as usize,
            )?;
            let (named, mins, maxs) = resolve_samplers_lod(objs, &es)?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                RenderCmd::SetTileSamplerStatesLodClamps {
                    samplers: &samplers,
                    lod_min_clamps: &mins,
                    lod_max_clamps: &maxs,
                    range: NsRange {
                        location: m.range_location,
                        length: m.range_length,
                    },
                },
            )?;
        }

        // ---- Array-tailed: viewports, scissor rects, amplification ----
        RGFX_OP_RENDER_SET_SCISSOR_RECTS => {
            let m: RgfxOpSetScissorRects = body(tag, payload)?;
            let es: Vec<RgfxScissorRectElem> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetScissorRects>(),
                m.count as usize,
            )?;
            let rects: Vec<MtlScissorRect> = es
                .iter()
                .map(|e| MtlScissorRect {
                    x: e.x,
                    y: e.y,
                    width: e.width,
                    height: e.height,
                })
                .collect();
            enc.encode(
                on,
                RenderCmd::SetScissorRects {
                    scissor_rects: &rects,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VIEWPORTS => {
            let m: RgfxOpSetViewports = body(tag, payload)?;
            let es: Vec<RgfxViewportElem> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetViewports>(),
                m.count as usize,
            )?;
            let viewports: Vec<MtlViewport> = es
                .iter()
                .map(|e| MtlViewport {
                    origin_x: f64::from_bits(e.origin_x_bits),
                    origin_y: f64::from_bits(e.origin_y_bits),
                    width: f64::from_bits(e.width_bits),
                    height: f64::from_bits(e.height_bits),
                    znear: f64::from_bits(e.znear_bits),
                    zfar: f64::from_bits(e.zfar_bits),
                })
                .collect();
            enc.encode(
                on,
                RenderCmd::SetViewports {
                    viewports: &viewports,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_COUNT => {
            let m: RgfxOpSetVertexAmplificationCount = body(tag, payload)?;
            let es: Vec<RgfxViewMapping> = elems(
                tag,
                payload,
                size_of::<RgfxOpSetVertexAmplificationCount>(),
                m.count as usize,
            )?;
            let view_mappings: Vec<VertexAmplificationViewMapping> = es
                .iter()
                .map(|e| VertexAmplificationViewMapping {
                    viewport_array_index_offset: e.viewport_array_index_offset,
                    render_target_array_index_offset: e.render_target_array_index_offset,
                })
                .collect();
            enc.encode(
                on,
                RenderCmd::SetVertexAmplificationCount {
                    view_mappings: &view_mappings,
                },
            )?;
        }

        // ---- Array-tailed: heaps and resources ----
        RGFX_OP_RENDER_USE_HEAPS => {
            let m: RgfxOpUseHeaps = body(tag, payload)?;
            let es: Vec<RgfxHeapBinding> =
                elems(tag, payload, size_of::<RgfxOpUseHeaps>(), m.count as usize)?;
            let named = resolve_heaps(objs, &es)?;
            let heaps: Vec<Option<&P::Heap>> = named;
            enc.encode(on, RenderCmd::UseHeaps { heaps: &heaps })?;
        }
        RGFX_OP_RENDER_USE_HEAPS_STAGES => {
            let m: RgfxOpUseHeapsStages = body(tag, payload)?;
            let es: Vec<RgfxHeapBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpUseHeapsStages>(),
                m.count as usize,
            )?;
            let named = resolve_heaps(objs, &es)?;
            let heaps: Vec<Option<&P::Heap>> = named;
            enc.encode(
                on,
                RenderCmd::UseHeapsStages {
                    heaps: &heaps,
                    stages: m.stages,
                },
            )?;
        }
        RGFX_OP_RENDER_USE_RESOURCES => {
            let m: RgfxOpUseResources = body(tag, payload)?;
            let es: Vec<RgfxResourceBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpUseResources>(),
                m.count as usize,
            )?;
            let resources: Vec<Option<ResourceRef<'_, P>>> = es
                .iter()
                .map(|e| resolve_resource(objs, e))
                .collect::<RaiseResult<_>>()?;
            enc.encode(
                on,
                RenderCmd::UseResources {
                    resources: &resources,
                    usage: m.usage,
                },
            )?;
        }
        RGFX_OP_RENDER_USE_RESOURCES_STAGES => {
            let m: RgfxOpUseResourcesStages = body(tag, payload)?;
            let es: Vec<RgfxResourceBinding> = elems(
                tag,
                payload,
                size_of::<RgfxOpUseResourcesStages>(),
                m.count as usize,
            )?;
            let resources: Vec<Option<ResourceRef<'_, P>>> = es
                .iter()
                .map(|e| resolve_resource(objs, e))
                .collect::<RaiseResult<_>>()?;
            enc.encode(
                on,
                RenderCmd::UseResourcesStages {
                    resources: &resources,
                    usage: m.usage,
                    stages: m.stages,
                },
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::wire::{
        RGFX_CMD_BUFFER_CREATE, RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_FENCE_CREATE,
        RGFX_CMD_HEAP_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_TEXTURE_CREATE, RgfxBufferCreate,
        RgfxDepthStencilCreate, RgfxHeapCreate, RgfxMsgHeader, RgfxObjectCreate, RgfxSamplerCreate,
        RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::values::IcbDesc;
    use reims_vgpu_core::platform::{Handle, Platform};
    use reims_vgpu_core::testkit::{Arg, FakePlatform};
    use zerocopy::IntoBytes;

    use super::super::ObjHandle;
    use super::*;

    /// The op decode's fault policy, as the walk passes it.
    const ON: OnFault = OnFault::Return;

    type Fp = FakePlatform;

    /// Object ids the setup creates in the decoder's table.
    const BUF_VERTEX: u64 = 2;
    const BUF_INDEX: u64 = 3;
    const BUF_INDIRECT: u64 = 4;
    const BUF_AUX: u64 = 5;
    const TEX_A: u64 = 10;
    const TEX_B: u64 = 11;
    const SAMP_A: u64 = 20;
    const SAMP_B: u64 = 21;
    const DEPTH_STENCIL: u64 = 30;
    const FENCE: u64 = 40;
    const HEAP: u64 = 45;
    const ICB: u64 = 50;

    /// A decoder whose table holds the objects the ops below reference: four
    /// buffers, two textures, two samplers, a depth/stencil state, a fence, a heap
    /// and an indirect command buffer. Everything the op stream can create goes
    /// through [`GraphicsDecoder::decode`]; the indirect command buffer, which the
    /// control ring cannot yet create, is placed straight into the table.
    fn setup() -> (Arc<Fp>, GraphicsDecoder<Fp>) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<Fp>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

        for id in [BUF_VERTEX, BUF_INDEX, BUF_INDIRECT, BUF_AUX] {
            let b = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                length: 65536,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, b.as_bytes()).unwrap(), Ok(()));
        }
        // The textures' resource ids are their object ids, in chunk 0.
        dec.hold_resource_ids(0, 0);
        for id in [TEX_A, TEX_B] {
            let t = RgfxTextureCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_CREATE,
                    size: size_of::<RgfxTextureCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                desc: RgfxTextureDescriptor {
                    pixel_format: 80,
                    texture_type: 2,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 5,
                    ..RgfxTextureDescriptor::default()
                },
                resource_id: id,
                ..RgfxTextureCreate::default()
            };
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        for id in [SAMP_A, SAMP_B] {
            let s = RgfxSamplerCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_SAMPLER_CREATE,
                    size: size_of::<RgfxSamplerCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                max_anisotropy: 1,
                lod_max_clamp_bits: f32::MAX.to_bits(),
                ..RgfxSamplerCreate::default()
            };
            assert_eq!(dec.decode(&device, s.as_bytes()).unwrap(), Ok(()));
        }
        let ds = RgfxDepthStencilCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DEPTH_STENCIL_CREATE,
                size: size_of::<RgfxDepthStencilCreate>() as u32,
                seqno: 0,
            },
            obj_id: DEPTH_STENCIL,
            ..RgfxDepthStencilCreate::default()
        };
        assert_eq!(dec.decode(&device, ds.as_bytes()).unwrap(), Ok(()));
        let fence = RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FENCE_CREATE,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id: FENCE,
        };
        assert_eq!(dec.decode(&device, fence.as_bytes()).unwrap(), Ok(()));
        let heap = RgfxHeapCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_CREATE,
                size: size_of::<RgfxHeapCreate>() as u32,
                seqno: 0,
            },
            obj_id: HEAP,
            size_bytes: 1 << 20,
            heap_type: 1,
            ..RgfxHeapCreate::default()
        };
        assert_eq!(dec.decode(&device, heap.as_bytes()).unwrap(), Ok(()));

        // The control ring cannot yet create an indirect command buffer; place one
        // in the table directly so the execute and useResources ops can name it.
        let icb = device
            .new_indirect_command_buffer(&IcbDesc::default(), 4, 0)
            .unwrap()
            .unwrap();
        dec.insert(ICB, ObjHandle::Icb(icb)).unwrap().unwrap();

        (p, dec)
    }

    /// A render encoder to decode ops onto, and the command buffer that holds it
    /// open (returned so it outlives the encoder).
    fn encoder(
        p: &Arc<Fp>,
    ) -> (
        <Fp as Platform>::CommandBuffer,
        <Fp as Platform>::RenderEncoder,
    ) {
        let q = p.device().new_command_queue().unwrap();
        let cb = q.command_buffer(ON).unwrap().unwrap();
        let pass = p.new_render_pass_descriptor();
        let enc = cb.render_encoder(ON, &pass).unwrap().unwrap();
        (cb, enc)
    }

    /// Concatenates an array-tailed op's head and its element records.
    fn tailed(head: &[u8], elems: &[&[u8]]) -> Vec<u8> {
        let mut v = head.to_vec();
        for e in elems {
            v.extend_from_slice(e);
        }
        v
    }

    /// Every render op the decoder now covers drives exactly its selector, in order.
    #[test]
    fn every_render_op_drives_its_selector() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);

        let range = RgfxOpSetVertexBuffers {
            range_location: 0,
            range_length: 1,
        };
        // Each entry is (tag, body, the selector the op must send).
        let entries: Vec<(u16, Vec<u8>, &str)> = vec![
            (
                RGFX_OP_RENDER_DRAW_PRIMITIVES_INSTANCES,
                RgfxOpDrawPrimitivesInstances {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 2,
                }
                .as_bytes()
                .to_vec(),
                "drawPrimitives:vertexStart:vertexCount:instanceCount:",
            ),
            (
                RGFX_OP_RENDER_DRAW_PRIMITIVES_BASE,
                RgfxOpDrawPrimitivesBase {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 2,
                    base_instance: 1,
                }
                .as_bytes()
                .to_vec(),
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
            ),
            (
                RGFX_OP_RENDER_DRAW_PRIMITIVES_INDIRECT,
                RgfxOpDrawPrimitivesIndirect {
                    primitive_type: 3,
                    indirect_buffer_obj_id: BUF_INDIRECT,
                    indirect_buffer_offset: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawPrimitives:indirectBuffer:indirectBufferOffset:",
            ),
            (
                RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES,
                RgfxOpDrawIndexedPrimitivesInstances {
                    primitive_type: 3,
                    index_count: 6,
                    index_type: 1,
                    index_buffer_obj_id: BUF_INDEX,
                    index_buffer_offset: 0,
                    instance_count: 2,
                }
                .as_bytes()
                .to_vec(),
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:",
            ),
            (
                RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_BASE,
                RgfxOpDrawIndexedPrimitivesBase {
                    primitive_type: 3,
                    index_count: 6,
                    index_type: 1,
                    index_buffer_obj_id: BUF_INDEX,
                    index_buffer_offset: 0,
                    instance_count: 2,
                    base_vertex: -3,
                    base_instance: 1,
                }
                .as_bytes()
                .to_vec(),
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
            ),
            (
                RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INDIRECT,
                RgfxOpDrawIndexedPrimitivesIndirect {
                    primitive_type: 3,
                    index_type: 1,
                    index_buffer_obj_id: BUF_INDEX,
                    index_buffer_offset: 0,
                    indirect_buffer_obj_id: BUF_INDIRECT,
                    indirect_buffer_offset: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
            ),
            (
                RGFX_OP_RENDER_DRAW_PATCHES,
                RgfxOpDrawPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count: 1,
                    patch_index_buffer_obj_id: 0,
                    patch_index_buffer_offset: 0,
                    instance_count: 1,
                    base_instance: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:",
            ),
            (
                RGFX_OP_RENDER_DRAW_PATCHES_INDIRECT,
                RgfxOpDrawPatchesIndirect {
                    number_of_patch_control_points: 3,
                    patch_index_buffer_obj_id: BUF_AUX,
                    patch_index_buffer_offset: 0,
                    indirect_buffer_obj_id: BUF_INDIRECT,
                    indirect_buffer_offset: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
            ),
            (
                RGFX_OP_RENDER_DRAW_INDEXED_PATCHES,
                RgfxOpDrawIndexedPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count: 1,
                    patch_index_buffer_obj_id: 0,
                    patch_index_buffer_offset: 0,
                    control_point_index_buffer_obj_id: BUF_AUX,
                    control_point_index_buffer_offset: 0,
                    instance_count: 1,
                    base_instance: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:",
            ),
            (
                RGFX_OP_RENDER_DRAW_INDEXED_PATCHES_INDIRECT,
                RgfxOpDrawIndexedPatchesIndirect {
                    number_of_patch_control_points: 3,
                    patch_index_buffer_obj_id: 0,
                    patch_index_buffer_offset: 0,
                    control_point_index_buffer_obj_id: BUF_AUX,
                    control_point_index_buffer_offset: 0,
                    indirect_buffer_obj_id: BUF_INDIRECT,
                    indirect_buffer_offset: 0,
                }
                .as_bytes()
                .to_vec(),
                "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
            ),
            (
                RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE,
                RgfxOpDispatchThreadsPerTile {
                    width: 16,
                    height: 16,
                    depth: 1,
                }
                .as_bytes()
                .to_vec(),
                "dispatchThreadsPerTile:",
            ),
            (
                RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION,
                RgfxOpDispatchThreadsPerTileInRegion {
                    width: 16,
                    height: 16,
                    depth: 1,
                    origin_x: 0,
                    origin_y: 0,
                    origin_z: 0,
                    region_width: 64,
                    region_height: 64,
                    region_depth: 1,
                }
                .as_bytes()
                .to_vec(),
                "dispatchThreadsPerTile:inRegion:",
            ),
            (
                RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION_INDEX,
                RgfxOpDispatchThreadsPerTileInRegionIndex {
                    width: 16,
                    height: 16,
                    depth: 1,
                    origin_x: 0,
                    origin_y: 0,
                    origin_z: 0,
                    region_width: 64,
                    region_height: 64,
                    region_depth: 1,
                    render_target_array_index: 2,
                    reserved_4c: 0,
                }
                .as_bytes()
                .to_vec(),
                "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
            ),
            (
                RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER,
                RgfxOpExecuteCommandsInBuffer {
                    indirect_command_buffer_obj_id: ICB,
                    range_location: 0,
                    range_length: 2,
                }
                .as_bytes()
                .to_vec(),
                "executeCommandsInBuffer:withRange:",
            ),
            (
                RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT,
                RgfxOpExecuteCommandsIndirect {
                    indirect_command_buffer_obj_id: ICB,
                    indirect_range_buffer_obj_id: BUF_AUX,
                    indirect_buffer_offset: 0,
                }
                .as_bytes()
                .to_vec(),
                "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
            ),
            (
                RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE,
                RgfxOpMemoryBarrierWithScope {
                    scope: 1,
                    after: 2,
                    before: 4,
                }
                .as_bytes()
                .to_vec(),
                "memoryBarrierWithScope:afterStages:beforeStages:",
            ),
            (
                RGFX_OP_RENDER_UPDATE_FENCE,
                RgfxOpUpdateFence {
                    fence_obj_id: FENCE,
                    stages: 1,
                }
                .as_bytes()
                .to_vec(),
                "updateFence:afterStages:",
            ),
            (
                RGFX_OP_RENDER_WAIT_FOR_FENCE,
                RgfxOpWaitForFence {
                    fence_obj_id: FENCE,
                    stages: 1,
                }
                .as_bytes()
                .to_vec(),
                "waitForFence:beforeStages:",
            ),
            (
                RGFX_OP_RENDER_SET_CULL_MODE,
                RgfxOpSetCullMode { cull_mode: 2 }.as_bytes().to_vec(),
                "setCullMode:",
            ),
            (
                RGFX_OP_RENDER_SET_FRONT_FACING_WINDING,
                RgfxOpSetFrontFacingWinding {
                    front_facing_winding: 1,
                }
                .as_bytes()
                .to_vec(),
                "setFrontFacingWinding:",
            ),
            (
                RGFX_OP_RENDER_SET_TRIANGLE_FILL_MODE,
                RgfxOpSetTriangleFillMode { fill_mode: 1 }
                    .as_bytes()
                    .to_vec(),
                "setTriangleFillMode:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_CLIP_MODE_SPI,
                RgfxOpSetDepthClipModeSpi { mode: 1 }.as_bytes().to_vec(),
                "setDepthClipModeSPI:",
            ),
            (
                RGFX_OP_RENDER_SET_PROVOKING_VERTEX_MODE,
                RgfxOpSetProvokingVertexMode { mode: 1 }.as_bytes().to_vec(),
                "setProvokingVertexMode:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION,
                RgfxOpSetDepthStoreAction { store_action: 1 }
                    .as_bytes()
                    .to_vec(),
                "setDepthStoreAction:",
            ),
            (
                RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION,
                RgfxOpSetStencilStoreAction { store_action: 1 }
                    .as_bytes()
                    .to_vec(),
                "setStencilStoreAction:",
            ),
            (
                RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION_OPTIONS,
                RgfxOpSetStencilStoreActionOptions {
                    store_action_options: 1,
                }
                .as_bytes()
                .to_vec(),
                "setStencilStoreActionOptions:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS,
                RgfxOpSetDepthStoreActionOptions {
                    store_action_options: 1,
                }
                .as_bytes()
                .to_vec(),
                "setDepthStoreActionOptions:",
            ),
            (
                RGFX_OP_RENDER_SET_COLOR_STORE_ACTION,
                RgfxOpSetColorStoreAction {
                    store_action: 1,
                    color_attachment_index: 0,
                }
                .as_bytes()
                .to_vec(),
                "setColorStoreAction:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_COLOR_STORE_ACTION_OPTIONS,
                RgfxOpSetColorStoreActionOptions {
                    store_action_options: 1,
                    color_attachment_index: 0,
                }
                .as_bytes()
                .to_vec(),
                "setColorStoreActionOptions:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE,
                RgfxOpSetVisibilityResultMode { mode: 1, offset: 8 }
                    .as_bytes()
                    .to_vec(),
                "setVisibilityResultMode:offset:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_MODE,
                RgfxOpSetVertexAmplificationMode { mode: 1, value: 2 }
                    .as_bytes()
                    .to_vec(),
                "setVertexAmplificationMode:value:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET,
                RgfxOpSetVertexBufferOffset {
                    offset: 32,
                    index: 1,
                }
                .as_bytes()
                .to_vec(),
                "setVertexBufferOffset:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_FRAGMENT_BUFFER_OFFSET,
                RgfxOpSetFragmentBufferOffset {
                    offset: 32,
                    index: 1,
                }
                .as_bytes()
                .to_vec(),
                "setFragmentBufferOffset:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_TILE_BUFFER_OFFSET,
                RgfxOpSetTileBufferOffset {
                    offset: 32,
                    index: 1,
                }
                .as_bytes()
                .to_vec(),
                "setTileBufferOffset:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET_ATTRIBUTE_STRIDE,
                RgfxOpSetVertexBufferOffsetAttributeStride {
                    offset: 32,
                    stride: 16,
                    index: 1,
                }
                .as_bytes()
                .to_vec(),
                "setVertexBufferOffset:attributeStride:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_THREADGROUP_MEMORY_LENGTH,
                RgfxOpSetThreadgroupMemoryLength {
                    length: 256,
                    offset: 0,
                    index: 0,
                }
                .as_bytes()
                .to_vec(),
                "setThreadgroupMemoryLength:offset:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_ALPHA_TEST_REFERENCE_VALUE,
                RgfxOpSetAlphaTestReferenceValue {
                    reference_value_bits: 0.5f32.to_bits(),
                    reserved_04: 0,
                }
                .as_bytes()
                .to_vec(),
                "setAlphaTestReferenceValue:",
            ),
            (
                RGFX_OP_RENDER_SET_LINE_WIDTH,
                RgfxOpSetLineWidth {
                    line_width_bits: 2.0f32.to_bits(),
                    reserved_04: 0,
                }
                .as_bytes()
                .to_vec(),
                "setLineWidth:",
            ),
            (
                RGFX_OP_RENDER_SET_POINT_SIZE,
                RgfxOpSetPointSize {
                    point_size_bits: 3.0f32.to_bits(),
                    reserved_04: 0,
                }
                .as_bytes()
                .to_vec(),
                "setPointSize:",
            ),
            (
                RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_SCALE,
                RgfxOpSetTessellationFactorScale {
                    scale_bits: 1.5f32.to_bits(),
                    reserved_04: 0,
                }
                .as_bytes()
                .to_vec(),
                "setTessellationFactorScale:",
            ),
            (
                RGFX_OP_RENDER_SET_BLEND_COLOR_RED,
                RgfxOpSetBlendColorRed {
                    red_bits: 1.0f32.to_bits(),
                    green_bits: 0.0f32.to_bits(),
                    blue_bits: 0.0f32.to_bits(),
                    alpha_bits: 1.0f32.to_bits(),
                }
                .as_bytes()
                .to_vec(),
                "setBlendColorRed:green:blue:alpha:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_BIAS,
                RgfxOpSetDepthBias {
                    depth_bias_bits: 0.1f32.to_bits(),
                    slope_scale_bits: 1.0f32.to_bits(),
                    clamp_bits: 0.0f32.to_bits(),
                    reserved_0c: 0,
                }
                .as_bytes()
                .to_vec(),
                "setDepthBias:slopeScale:clamp:",
            ),
            (
                RGFX_OP_RENDER_SET_VIEWPORT_TRANSFORM_ENABLED,
                RgfxOpSetViewportTransformEnabled {
                    enabled: 1,
                    reserved_04: 0,
                }
                .as_bytes()
                .to_vec(),
                "setViewportTransformEnabled:",
            ),
            (
                RGFX_OP_RENDER_SET_PRIMITIVE_RESTART_ENABLED,
                RgfxOpSetPrimitiveRestartEnabled {
                    index: 0xffff,
                    enabled: 1,
                    reserved_0c: 0,
                }
                .as_bytes()
                .to_vec(),
                "setPrimitiveRestartEnabled:index:",
            ),
            (
                RGFX_OP_RENDER_SET_STENCIL_FRONT_REFERENCE_VALUE,
                RgfxOpSetStencilFrontReferenceValue {
                    front_reference_value: 1,
                    back_reference_value: 2,
                }
                .as_bytes()
                .to_vec(),
                "setStencilFrontReferenceValue:backReferenceValue:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_CLEARED,
                Vec::new(),
                "setDepthCleared",
            ),
            (
                RGFX_OP_RENDER_SET_STENCIL_CLEARED,
                Vec::new(),
                "setStencilCleared",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE,
                RgfxOpSetDepthStencilState {
                    depth_stencil_state_obj_id: DEPTH_STENCIL,
                }
                .as_bytes()
                .to_vec(),
                "setDepthStencilState:",
            ),
            (
                RGFX_OP_RENDER_SET_COLOR_RESOLVE_TEXTURE,
                RgfxOpSetColorResolveTexture {
                    texture_obj_id: TEX_A,
                    slice: 0,
                    depth_plane: 0,
                    level: 0,
                    index: 0,
                    y_invert: 1,
                    reserved_2c: 0,
                }
                .as_bytes()
                .to_vec(),
                "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
            ),
            (
                RGFX_OP_RENDER_SET_DEPTH_RESOLVE_TEXTURE,
                RgfxOpSetDepthResolveTexture {
                    texture_obj_id: TEX_A,
                    slice: 0,
                    depth_plane: 0,
                    level: 0,
                    y_invert: 0,
                    reserved_24: 0,
                }
                .as_bytes()
                .to_vec(),
                "setDepthResolveTexture:slice:depthPlane:level:yInvert:",
            ),
            (
                RGFX_OP_RENDER_SET_STENCIL_RESOLVE_TEXTURE,
                RgfxOpSetStencilResolveTexture {
                    texture_obj_id: TEX_B,
                    slice: 0,
                    depth_plane: 0,
                    level: 0,
                    y_invert: 0,
                    reserved_24: 0,
                }
                .as_bytes()
                .to_vec(),
                "setStencilResolveTexture:slice:depthPlane:level:yInvert:",
            ),
            (
                RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_BUFFER,
                RgfxOpSetTessellationFactorBuffer {
                    buffer_obj_id: BUF_AUX,
                    offset: 0,
                    instance_stride: 0,
                }
                .as_bytes()
                .to_vec(),
                "setTessellationFactorBuffer:offset:instanceStride:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_BUFFERS,
                tailed(
                    range.as_bytes(),
                    &[RgfxBufferBinding {
                        buffer_obj_id: BUF_VERTEX,
                        offset: 0,
                    }
                    .as_bytes()],
                ),
                "setVertexBuffers:offsets:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_FRAGMENT_BUFFERS,
                tailed(
                    range.as_bytes(),
                    &[RgfxBufferBinding {
                        buffer_obj_id: BUF_VERTEX,
                        offset: 32,
                    }
                    .as_bytes()],
                ),
                "setFragmentBuffers:offsets:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_TILE_BUFFERS,
                tailed(
                    range.as_bytes(),
                    &[RgfxBufferBinding {
                        buffer_obj_id: BUF_VERTEX,
                        offset: 0,
                    }
                    .as_bytes()],
                ),
                "setTileBuffers:offsets:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES,
                tailed(
                    range.as_bytes(),
                    &[RgfxBufferStrideBinding {
                        buffer_obj_id: BUF_VERTEX,
                        offset: 0,
                        stride: 16,
                    }
                    .as_bytes()],
                ),
                "setVertexBuffers:offsets:attributeStrides:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_TEXTURES,
                tailed(
                    range.as_bytes(),
                    &[RgfxTextureBinding {
                        texture_obj_id: TEX_A,
                    }
                    .as_bytes()],
                ),
                "setVertexTextures:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_FRAGMENT_TEXTURES,
                tailed(
                    range.as_bytes(),
                    &[RgfxTextureBinding {
                        texture_obj_id: TEX_B,
                    }
                    .as_bytes()],
                ),
                "setFragmentTextures:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_TILE_TEXTURES,
                tailed(
                    range.as_bytes(),
                    &[RgfxTextureBinding {
                        texture_obj_id: TEX_A,
                    }
                    .as_bytes()],
                ),
                "setTileTextures:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerBinding {
                        sampler_obj_id: SAMP_A,
                    }
                    .as_bytes()],
                ),
                "setVertexSamplerStates:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerBinding {
                        sampler_obj_id: SAMP_B,
                    }
                    .as_bytes()],
                ),
                "setFragmentSamplerStates:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerBinding {
                        sampler_obj_id: SAMP_A,
                    }
                    .as_bytes()],
                ),
                "setTileSamplerStates:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES_LOD_CLAMPS,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerLodBinding {
                        sampler_obj_id: SAMP_A,
                        lod_min_clamp_bits: 0.0f32.to_bits(),
                        lod_max_clamp_bits: 8.0f32.to_bits(),
                    }
                    .as_bytes()],
                ),
                "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES_LOD_CLAMPS,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerLodBinding {
                        sampler_obj_id: SAMP_B,
                        lod_min_clamp_bits: 0.0f32.to_bits(),
                        lod_max_clamp_bits: 8.0f32.to_bits(),
                    }
                    .as_bytes()],
                ),
                "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES_LOD_CLAMPS,
                tailed(
                    range.as_bytes(),
                    &[RgfxSamplerLodBinding {
                        sampler_obj_id: SAMP_A,
                        lod_min_clamp_bits: 0.0f32.to_bits(),
                        lod_max_clamp_bits: 8.0f32.to_bits(),
                    }
                    .as_bytes()],
                ),
                "setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
            ),
            (
                RGFX_OP_RENDER_SET_SCISSOR_RECTS,
                tailed(
                    RgfxOpSetScissorRects { count: 1 }.as_bytes(),
                    &[RgfxScissorRectElem {
                        x: 0,
                        y: 0,
                        width: 100,
                        height: 100,
                    }
                    .as_bytes()],
                ),
                "setScissorRects:count:",
            ),
            (
                RGFX_OP_RENDER_SET_VIEWPORTS,
                tailed(
                    RgfxOpSetViewports { count: 1 }.as_bytes(),
                    &[RgfxViewportElem {
                        origin_x_bits: 0.0f64.to_bits(),
                        origin_y_bits: 0.0f64.to_bits(),
                        width_bits: 640.0f64.to_bits(),
                        height_bits: 480.0f64.to_bits(),
                        znear_bits: 0.0f64.to_bits(),
                        zfar_bits: 1.0f64.to_bits(),
                    }
                    .as_bytes()],
                ),
                "setViewports:count:",
            ),
            (
                RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_COUNT,
                tailed(
                    RgfxOpSetVertexAmplificationCount { count: 2 }.as_bytes(),
                    &[
                        RgfxViewMapping {
                            viewport_array_index_offset: 0,
                            render_target_array_index_offset: 0,
                        }
                        .as_bytes(),
                        RgfxViewMapping {
                            viewport_array_index_offset: 1,
                            render_target_array_index_offset: 1,
                        }
                        .as_bytes(),
                    ],
                ),
                "setVertexAmplificationCount:viewMappings:",
            ),
            (
                RGFX_OP_RENDER_USE_HEAPS,
                tailed(
                    RgfxOpUseHeaps { count: 1 }.as_bytes(),
                    &[RgfxHeapBinding { heap_obj_id: HEAP }.as_bytes()],
                ),
                "useHeaps:count:",
            ),
            (
                RGFX_OP_RENDER_USE_HEAPS_STAGES,
                tailed(
                    RgfxOpUseHeapsStages {
                        count: 1,
                        stages: 1,
                    }
                    .as_bytes(),
                    &[RgfxHeapBinding { heap_obj_id: HEAP }.as_bytes()],
                ),
                "useHeaps:count:stages:",
            ),
            (
                RGFX_OP_RENDER_USE_RESOURCES,
                tailed(
                    RgfxOpUseResources { count: 3, usage: 1 }.as_bytes(),
                    &[
                        RgfxResourceBinding {
                            obj_id: BUF_VERTEX,
                            kind: RGFX_RESOURCE_KIND_BUFFER,
                            reserved_0c: 0,
                        }
                        .as_bytes(),
                        RgfxResourceBinding {
                            obj_id: TEX_A,
                            kind: RGFX_RESOURCE_KIND_TEXTURE,
                            reserved_0c: 0,
                        }
                        .as_bytes(),
                        RgfxResourceBinding {
                            obj_id: ICB,
                            kind: RGFX_RESOURCE_KIND_ICB,
                            reserved_0c: 0,
                        }
                        .as_bytes(),
                    ],
                ),
                "useResources:count:usage:",
            ),
            (
                RGFX_OP_RENDER_USE_RESOURCES_STAGES,
                tailed(
                    RgfxOpUseResourcesStages {
                        count: 1,
                        usage: 1,
                        stages: 1,
                    }
                    .as_bytes(),
                    &[RgfxResourceBinding {
                        obj_id: BUF_AUX,
                        kind: RGFX_RESOURCE_KIND_BUFFER,
                        reserved_0c: 0,
                    }
                    .as_bytes()],
                ),
                "useResources:count:usage:stages:",
            ),
        ];

        let mark = p.call_log().mark();
        let mut expected = Vec::new();
        for (tag, body, sel) in &entries {
            assert!(
                decode_op(*tag, body, &dec, &enc, ON).unwrap(),
                "op {tag:#06x} ({sel}) was not handled",
            );
            expected.push(*sel);
        }

        let got = p.call_log().sels_since(mark);
        assert_eq!(got, expected);
        // The whole remaining set was exercised.
        assert_eq!(expected.len(), 72);
    }

    /// An array-tailed op resolves each element and passes the slices through: the
    /// buffers (by identity, a nil kept), the offsets and the range.
    #[test]
    fn set_vertex_buffers_passes_its_elements() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);

        let body = tailed(
            RgfxOpSetVertexBuffers {
                range_location: 3,
                range_length: 2,
            }
            .as_bytes(),
            &[
                RgfxBufferBinding {
                    buffer_obj_id: BUF_VERTEX,
                    offset: 16,
                }
                .as_bytes(),
                RgfxBufferBinding {
                    buffer_obj_id: 0,
                    offset: 0,
                }
                .as_bytes(),
            ],
        );
        let mark = p.call_log().mark();
        assert!(decode_op(RGFX_OP_RENDER_SET_VERTEX_BUFFERS, &body, &dec, &enc, ON).unwrap());

        let vertex_id = dec.get_buffer(BUF_VERTEX).unwrap().unwrap().id();
        let calls = p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "setVertexBuffers:offsets:withRange:");
        assert_eq!(
            calls[0].args,
            vec![
                Arg::Objs(vec![Some(vertex_id), None]),
                Arg::U64s(vec![16, 0]),
                Arg::val(&NsRange {
                    location: 3,
                    length: 2,
                }),
            ],
        );
    }

    /// A `useResources:` array resolves a buffer, a texture and an indirect command
    /// buffer each by its kind, keeping their identities and the usage.
    #[test]
    fn use_resources_resolves_each_kind() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);

        let body = tailed(
            RgfxOpUseResources { count: 3, usage: 3 }.as_bytes(),
            &[
                RgfxResourceBinding {
                    obj_id: BUF_VERTEX,
                    kind: RGFX_RESOURCE_KIND_BUFFER,
                    reserved_0c: 0,
                }
                .as_bytes(),
                RgfxResourceBinding {
                    obj_id: TEX_A,
                    kind: RGFX_RESOURCE_KIND_TEXTURE,
                    reserved_0c: 0,
                }
                .as_bytes(),
                RgfxResourceBinding {
                    obj_id: ICB,
                    kind: RGFX_RESOURCE_KIND_ICB,
                    reserved_0c: 0,
                }
                .as_bytes(),
            ],
        );
        let mark = p.call_log().mark();
        assert!(decode_op(RGFX_OP_RENDER_USE_RESOURCES, &body, &dec, &enc, ON).unwrap());

        let buf_id = dec.get_buffer(BUF_VERTEX).unwrap().unwrap().id();
        let tex_id = dec.get_texture(TEX_A).unwrap().unwrap().id();
        let icb_id = dec.require_indirect_command_buffer(ICB).unwrap().id();
        let calls = p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "useResources:count:usage:");
        assert_eq!(
            calls[0].args,
            vec![
                Arg::Objs(vec![Some(buf_id), Some(tex_id), Some(icb_id)]),
                Arg::U64(3),
            ],
        );
    }

    /// A body too short for its op faults the exec, and no command is sent.
    #[test]
    fn a_short_body_faults() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);
        let mark = p.call_log().mark();
        // `DrawPrimitivesInstances` needs 32 bytes; give it 16.
        let short = vec![0u8; 16];
        assert!(
            decode_op(
                RGFX_OP_RENDER_DRAW_PRIMITIVES_INSTANCES,
                &short,
                &dec,
                &enc,
                ON
            )
            .is_err()
        );
        // An array op whose head claims more elements than the body carries.
        let claims_two = tailed(
            RgfxOpSetVertexBuffers {
                range_location: 0,
                range_length: 2,
            }
            .as_bytes(),
            &[RgfxBufferBinding {
                buffer_obj_id: BUF_VERTEX,
                offset: 0,
            }
            .as_bytes()],
        );
        assert!(
            decode_op(
                RGFX_OP_RENDER_SET_VERTEX_BUFFERS,
                &claims_two,
                &dec,
                &enc,
                ON
            )
            .is_err()
        );
        assert!(p.call_log().since(mark).is_empty());
    }

    /// A reference the host does not hold faults the exec: a required buffer, a
    /// required fence, and a resource array element.
    #[test]
    fn a_bad_reference_raises() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);

        let indirect = RgfxOpDrawPrimitivesIndirect {
            primitive_type: 3,
            indirect_buffer_obj_id: 999,
            indirect_buffer_offset: 0,
        };
        assert!(
            decode_op(
                RGFX_OP_RENDER_DRAW_PRIMITIVES_INDIRECT,
                indirect.as_bytes(),
                &dec,
                &enc,
                ON,
            )
            .is_err()
        );

        let fence = RgfxOpUpdateFence {
            fence_obj_id: 999,
            stages: 1,
        };
        assert!(
            decode_op(
                RGFX_OP_RENDER_UPDATE_FENCE,
                fence.as_bytes(),
                &dec,
                &enc,
                ON
            )
            .is_err()
        );

        // A resource array whose element names no object of its kind.
        let bad_res = tailed(
            RgfxOpUseResources { count: 1, usage: 1 }.as_bytes(),
            &[RgfxResourceBinding {
                obj_id: 999,
                kind: RGFX_RESOURCE_KIND_BUFFER,
                reserved_0c: 0,
            }
            .as_bytes()],
        );
        assert!(decode_op(RGFX_OP_RENDER_USE_RESOURCES, &bad_res, &dec, &enc, ON).is_err());

        // An unknown resource kind also faults.
        let bad_kind = tailed(
            RgfxOpUseResources { count: 1, usage: 1 }.as_bytes(),
            &[RgfxResourceBinding {
                obj_id: BUF_VERTEX,
                kind: 7,
                reserved_0c: 0,
            }
            .as_bytes()],
        );
        assert!(decode_op(RGFX_OP_RENDER_USE_RESOURCES, &bad_kind, &dec, &enc, ON).is_err());
    }

    /// A tag the render decoder does not know is unhandled (the walk then faults).
    #[test]
    fn an_unknown_tag_is_not_handled() {
        let (p, dec) = setup();
        let (_cb, enc) = encoder(&p);
        assert!(!decode_op(0x1fff, &[], &dec, &enc, ON).unwrap());
    }
}
