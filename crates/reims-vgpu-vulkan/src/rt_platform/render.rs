//! The rt platform's render encoders ([`RtRenderEncoder`]): rt's, one dynamic rendering of the
//! pass its descriptor keeps (`super::descriptor`), begun as the encoder is made, into which each
//! draw records at once.
//!
//! - **Each command is rt's call.** A list (`set*Buffers:offsets:withRange:`,
//!   `set*Textures:withRange:`, `set*SamplerStates:withRange:` and its form with LOD clamps) is one
//!   call per entry, in its order; the shorter draw forms are the longest one with one instance and
//!   bases of 0.
//! - **A pipeline state the encoder sets is kept by its command buffer** until the command buffer
//!   completes ([`Kept`]), as the compute encoder's are (`super::compute`).
//! - **Attribute strides** (`setVertexBufferOffset:attributeStride:atIndex:`,
//!   `setVertexBuffers:offsets:attributeStrides:withRange:`) bind as the calls without them: only a
//!   layout of `MTLBufferLayoutStrideDynamic` reads a stride, and rt refuses such a state as it is
//!   made.
//! - **Nothing to record**: `useResources:` and `useHeaps:` in each form (rt's pass orders every
//!   earlier command before it and itself before every later one, and every resource is
//!   resident), but that a tiled texture `useResources:` names is a use of it (`super::tiled`), and
//!   a texture it names at one of the platform's own indices a read through it (`Kept::reads`); the
//!   store action options (rt renders one sample, and the one option is custom
//!   sample positions); the object, mesh and tile functions' buffers, textures, samplers and
//!   threadgroup memory, and the tile threadgroup memory length (rt makes no state with such
//!   functions); and, each at Metal's default, a resolve texture of nil, a visibility result mode
//!   of `Disabled`, a line width of 1, primitive restart enabled, the viewport transform enabled,
//!   one vertex amplification view and a tessellation factor scale of 1.
//! - **Unbuilt** ([`Unbuilt::RenderCommand`]), each encoding nothing, counted: indirect draws,
//!   indexed patch draws, mesh draws, tile dispatches, a resolve texture, a visibility result
//!   mode but `Disabled`, and the private settings of other than Metal's default rt has none of
//!   (the alpha test's reference, a line width, a point size, the provoking vertex, primitive
//!   restart disabled, the viewport transform disabled, vertex amplification, and an attachment
//!   marked cleared). `tileWidth` and `tileHeight` answer 0, counted under
//!   [`Unbuilt::RenderPass`]: rt's pass has no tiles. A tessellation factor scale other than 1
//!   is counted under [`Unbuilt::TessellationFactorScale`]: rt has no call of it, and the
//!   encoder's later patch draws read their factors unscaled.
//! - **Metal's refusals**, each encoding nothing, counted (`rt-refused-metal`): a draw with no
//!   pipeline state set, or with nil set since; a value no Metal enumeration has (a primitive,
//!   index or fill type, a cull mode, a winding, a depth clip mode, a store action); an index, a
//!   count or a base past rt's 32 bits; a list of another length than its range; and a command
//!   after `endEncoding`.
//!
//! A command rt refuses encodes nothing, counted under the refusal's kind; in a list, the entries
//! before the refused one are bound.

use metal2vulkan::metal::{
    CullMode, DepthClipMode, IndexType, PrimitiveType, TriangleFillMode, Winding,
};
use metal2vulkan_runtime as rt;

use crate::core::platform::blit::ResourceRef;
use crate::core::platform::render::{RenderCmd, RenderEncoder};
use crate::core::platform::types::{MtlScissorRect, MtlViewport};
use crate::core::platform::{Handle, ObjId};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;

use super::blit::{Stop, report};
use super::command::{Kept, OpenMark};
use super::compute::{slots, small};
use super::copied::Used;
use super::diag::{self, Bound, Target, Use};
use super::render_pipeline::RtRenderPipeline;
use super::texture::RtTexture;
use super::unbuilt::{Unbuilt, refuse};
use super::{RtBuffer, RtObj, RtPlatform};

/// A count, a base or an index past rt's 32 bits.
const WIDE: Stop = Stop::Metal("a count, a base or an index past 32 bits");

/// `MTLVisibilityResultModeDisabled`.
const VISIBILITY_DISABLED: u64 = 0;

/// `id<MTLRenderCommandEncoder>` on metal2vulkan-runtime.
pub struct RtRenderEncoder {
    /// rt's encoder and what the encoder set.
    open: Mutex<Open>,
    /// Its command buffer's mark of it open, until it ends.
    mark: OpenMark,
    /// What its command buffer keeps until it completes.
    kept: Kept,
    /// Its pass's targets, for the boot diagnostics (`diag::attachment_read`,
    /// `diag::fetched_attachment`).
    targets: Vec<Target>,
}

/// An encoder's rt encoder, until `endEncoding` ends it, and its pipeline state.
struct Open {
    /// rt's encoder.
    rt: Option<rt::RenderCommandEncoder>,
    /// The pipeline state set last, which the command buffer keeps: none before one is set, or
    /// once nil is.
    pipeline: Option<ObjId>,
    /// What the boot diagnostics read of a draw rt refuses (`diag::attachment_read`).
    seen: Seen,
}

/// What an encoder set that the boot diagnostics read of a refused draw: the textures bound, the
/// pipeline state, and whether the depth stencil state writes depth.
#[derive(Default)]
struct Seen {
    bound: Bound,
    state: Option<RtObj<RtRenderPipeline>>,
    depth_write: bool,
}

impl RtRenderEncoder {
    /// The encoder over rt's `encoder`, open as `mark` marks it, its pipeline states kept in
    /// `kept`, of a pass of `targets`.
    pub fn new(
        encoder: rt::RenderCommandEncoder,
        mark: OpenMark,
        kept: Kept,
        targets: Vec<Target>,
    ) -> RtRenderEncoder {
        RtRenderEncoder {
            open: Mutex::new(Open {
                rt: Some(encoder),
                pipeline: None,
                seen: Seen::default(),
            }),
            mark,
            kept,
            targets,
        }
    }

    /// Notes `texture` bound to slot `slot` of stage `stage` (0 vertex, 1 fragment) for the boot
    /// diagnostics, and as a tiled texture's use (`super::tiled`).
    fn bound(&self, seen: &mut Seen, stage: usize, slot: u32, texture: Option<&RtObj<RtTexture>>) {
        seen.bound.set(stage, slot, texture);
        if let Some(t) = texture {
            self.kept.used(t, Use::Sampled);
        }
    }

    /// rt refused a draw as an attachment read through a texture: notes it for the boot
    /// diagnostics (`diag::attachment_read`).
    fn attachment_read(&self, seen: &Seen) {
        let Some(state) = &seen.state else {
            return;
        };
        let vertex = state.rt().texture_accesses(rt::Stage::Vertex);
        let fragment = state.rt().texture_accesses(rt::Stage::Fragment);
        let pipeline = state.id().0;
        let accesses = [vertex.as_slice(), fragment.as_slice()];
        diag::attachment_read(
            pipeline,
            accesses,
            &seen.bound,
            &self.targets,
            seen.depth_write,
        );
    }

    /// rt refused a draw as reading a colour attachment through its texture: notes it for the boot
    /// diagnostics (`diag::fetched_attachment`).
    fn fetched_attachment(&self, seen: &Seen) {
        if let Some(state) = &seen.state {
            diag::fetched_attachment(state.id().0, &self.targets);
        }
    }

    /// Encodes `cmd` but `endEncoding`, which the caller ends.
    fn command(&self, open: &mut Open, cmd: RenderCmd<'_, RtPlatform>) -> Result<(), Stop> {
        let Open {
            rt: encoder,
            pipeline,
            seen,
        } = open;
        let Some(e) = encoder.as_mut() else {
            return Err(Stop::Metal("a command after endEncoding"));
        };
        match cmd {
            RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
            } => {
                set(*pipeline)?;
                let (start, count) = (small(vertex_start)?, small(vertex_count)?);
                e.draw_primitives(primitive(primitive_type)?, start, count, 1, 0)?;
            }
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
            } => {
                set(*pipeline)?;
                let (start, count) = (small(vertex_start)?, small(vertex_count)?);
                let instances = small(instance_count)?;
                e.draw_primitives(primitive(primitive_type)?, start, count, instances, 0)?;
            }
            RenderCmd::DrawPrimitivesBase {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                set(*pipeline)?;
                let (start, count) = (small(vertex_start)?, small(vertex_count)?);
                let (instances, base) = (small(instance_count)?, small(base_instance)?);
                e.draw_primitives(primitive(primitive_type)?, start, count, instances, base)?;
            }
            RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
            } => {
                set(*pipeline)?;
                let indices = indices(index_buffer, index_buffer_offset, index_type)?;
                let (kind, count) = (primitive(primitive_type)?, small(index_count)?);
                e.draw_indexed_primitives(kind, count, indices, 1, 0, 0)?;
            }
            RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
            } => {
                set(*pipeline)?;
                let indices = indices(index_buffer, index_buffer_offset, index_type)?;
                let (kind, count) = (primitive(primitive_type)?, small(index_count)?);
                let instances = small(instance_count)?;
                e.draw_indexed_primitives(kind, count, indices, instances, 0, 0)?;
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
                set(*pipeline)?;
                let indices = indices(index_buffer, index_buffer_offset, index_type)?;
                let (kind, count) = (primitive(primitive_type)?, small(index_count)?);
                let instances = small(instance_count)?;
                let bases = (
                    i32::try_from(base_vertex).map_err(|_| WIDE)?,
                    small(base_instance)?,
                );
                e.draw_indexed_primitives(kind, count, indices, instances, bases.0, bases.1)?;
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
                set(*pipeline)?;
                let points = small(number_of_patch_control_points)?;
                let (start, count) = (small(patch_start)?, small(patch_count)?);
                let patch_indices = patch_index_buffer.map(|b| (b.rt(), patch_index_buffer_offset));
                let (instances, base) = (small(instance_count)?, small(base_instance)?);
                e.draw_patches(points, start, count, patch_indices, instances, base)?;
            }
            RenderCmd::DrawPrimitivesIndirect { .. }
            | RenderCmd::DrawIndexedPrimitivesIndirect { .. }
            | RenderCmd::DrawPatchesIndirect { .. }
            | RenderCmd::DrawIndexedPatchesIndirect { .. }
            | RenderCmd::DrawIndexedPatches { .. }
            | RenderCmd::DrawMeshThreadgroups { .. }
            | RenderCmd::DrawMeshThreadgroupsIndirect { .. }
            | RenderCmd::DrawMeshThreads { .. }
            | RenderCmd::DispatchThreadsPerTile { .. }
            | RenderCmd::DispatchThreadsPerTileInRegion { .. }
            | RenderCmd::DispatchThreadsPerTileInRegionIndex { .. }
            | RenderCmd::SetAlphaTestReferenceValue { .. }
            | RenderCmd::SetPointSize { .. }
            | RenderCmd::SetProvokingVertexMode { .. }
            | RenderCmd::SetVertexAmplificationMode { .. }
            | RenderCmd::SetDepthCleared
            | RenderCmd::SetStencilCleared => return Err(Stop::Unbuilt(Unbuilt::RenderCommand)),
            RenderCmd::SetColorResolveTexture { texture, .. }
            | RenderCmd::SetDepthResolveTexture { texture, .. }
            | RenderCmd::SetStencilResolveTexture { texture, .. } => {
                unbuilt_unless(texture.is_none())?;
            }
            RenderCmd::SetVisibilityResultMode { mode, .. } => {
                unbuilt_unless(mode == VISIBILITY_DISABLED)?;
            }
            RenderCmd::SetLineWidth { line_width } => unbuilt_unless(line_width == 1.0)?,
            RenderCmd::SetTessellationFactorScale { scale } if scale != 1.0 => {
                return Err(Stop::Unbuilt(Unbuilt::TessellationFactorScale));
            }
            RenderCmd::SetTessellationFactorScale { .. } => {}
            RenderCmd::SetPrimitiveRestartEnabled { enabled, .. } => unbuilt_unless(enabled)?,
            RenderCmd::SetViewportTransformEnabled { enabled } => unbuilt_unless(enabled)?,
            RenderCmd::SetVertexAmplificationCount { view_mappings } => {
                unbuilt_unless(view_mappings.len() <= 1)?;
            }
            RenderCmd::UseResources { resources, .. }
            | RenderCmd::UseResourcesStages { resources, .. } => {
                for resource in resources.iter().flatten() {
                    if let ResourceRef::Texture(texture) = resource {
                        self.kept.takes().used(texture);
                        self.kept.reads().note(texture.index_use());
                    }
                }
            }
            RenderCmd::EndEncoding
            | RenderCmd::UseHeaps { .. }
            | RenderCmd::UseHeapsStages { .. }
            | RenderCmd::SetColorStoreActionOptions { .. }
            | RenderCmd::SetDepthStoreActionOptions { .. }
            | RenderCmd::SetStencilStoreActionOptions { .. }
            | RenderCmd::SetThreadgroupMemoryLength { .. }
            | RenderCmd::SetObjectThreadgroupMemoryLength { .. }
            | RenderCmd::SetMeshBufferOffset { .. }
            | RenderCmd::SetMeshBuffers { .. }
            | RenderCmd::SetMeshSamplerStatesLodClamps { .. }
            | RenderCmd::SetMeshSamplerStates { .. }
            | RenderCmd::SetMeshTextures { .. }
            | RenderCmd::SetObjectBufferOffset { .. }
            | RenderCmd::SetObjectBuffers { .. }
            | RenderCmd::SetObjectSamplerStatesLodClamps { .. }
            | RenderCmd::SetObjectSamplerStates { .. }
            | RenderCmd::SetObjectTextures { .. }
            | RenderCmd::SetTileBufferOffset { .. }
            | RenderCmd::SetTileBuffers { .. }
            | RenderCmd::SetTileSamplerStatesLodClamps { .. }
            | RenderCmd::SetTileSamplerStates { .. }
            | RenderCmd::SetTileTextures { .. } => {}
            RenderCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                ..
            } => indirect_commandbuffer.unreachable(),
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                ..
            } => indirect_command_buffer.unreachable(),
            RenderCmd::MemoryBarrierWithScope { .. } => e.memory_barrier()?,
            RenderCmd::SetBlendColorRed {
                red,
                green,
                blue,
                alpha,
            } => e.set_blend_color(red, green, blue, alpha),
            RenderCmd::SetColorStoreAction {
                store_action,
                color_attachment_index,
            } => {
                let index = usize::try_from(color_attachment_index).map_err(|_| WIDE)?;
                e.set_color_store_action(store(store_action)?, index)?;
            }
            RenderCmd::SetDepthStoreAction { store_action } => {
                e.set_depth_store_action(store(store_action)?)?;
            }
            RenderCmd::SetStencilStoreAction { store_action } => {
                e.set_stencil_store_action(store(store_action)?)?;
            }
            RenderCmd::SetCullMode { cull_mode } => {
                e.set_cull_mode(named::<CullMode>(cull_mode, "a cull mode")?);
            }
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding,
            } => {
                let winding = named::<Winding>(front_facing_winding, "a winding")?;
                e.set_front_facing_winding(winding);
            }
            RenderCmd::SetTriangleFillMode { fill_mode } => {
                e.set_triangle_fill_mode(named::<TriangleFillMode>(fill_mode, "a fill mode")?)?;
            }
            RenderCmd::SetDepthClipModeSpi { mode } => {
                e.set_depth_clip_mode(named::<DepthClipMode>(mode, "a depth clip mode")?)?;
            }
            RenderCmd::SetDepthBias {
                depth_bias,
                slope_scale,
                clamp,
            } => e.set_depth_bias(depth_bias, slope_scale, clamp)?,
            RenderCmd::SetDepthStencilState {
                depth_stencil_state,
            } => {
                let state = depth_stencil_state.map(|s| &**s);
                e.set_depth_stencil_state(state)?;
                seen.depth_write = state.is_some_and(|s| s.descriptor().depth_write_enabled);
            }
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value,
                back_reference_value,
            } => e.set_stencil_front_back_reference_value(
                front_reference_value,
                back_reference_value,
            ),
            RenderCmd::SetRenderPipelineState { pipeline_state } => {
                let Some(state) = pipeline_state else {
                    *pipeline = None;
                    seen.state = None;
                    return Ok(());
                };
                e.set_render_pipeline_state(state.rt())?;
                let id = state.id();
                if *pipeline != Some(id) {
                    self.kept.keep(id, Box::new(state.clone_ref()));
                    *pipeline = Some(id);
                    seen.state = Some(state.clone_ref());
                }
            }
            RenderCmd::SetScissorRect { rect } => e.set_scissor_rect(scissor(rect)),
            RenderCmd::SetScissorRects { scissor_rects } => {
                let rects: Vec<rt::ScissorRect> =
                    (scissor_rects.iter().copied()).map(scissor).collect();
                e.set_scissor_rects(&rects)?;
            }
            RenderCmd::SetViewport { viewport: v } => e.set_viewport(viewport(v))?,
            RenderCmd::SetViewports { viewports } => {
                let all: Vec<rt::Viewport> = viewports.iter().copied().map(viewport).collect();
                e.set_viewports(&all)?;
            }
            RenderCmd::SetTessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => {
                let buffer = buffer.map(|b| b.rt());
                e.set_tessellation_factor_buffer(buffer, offset, instance_stride)?;
            }
            RenderCmd::UpdateFence { fence, .. } => e.update_fence(fence)?,
            RenderCmd::WaitForFence { fence, .. } => e.wait_for_fence(fence)?,
            RenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => e.set_vertex_buffer(buffer.map(|b| b.rt()), offset, small(index)?)?,
            RenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => e.set_fragment_buffer(buffer.map(|b| b.rt()), offset, small(index)?)?,
            RenderCmd::SetVertexBufferOffset { offset, index }
            | RenderCmd::SetVertexBufferOffsetAttributeStride { offset, index, .. } => {
                e.set_vertex_buffer_offset(offset, small(index)?)?;
            }
            RenderCmd::SetFragmentBufferOffset { offset, index } => {
                e.set_fragment_buffer_offset(offset, small(index)?)?;
            }
            RenderCmd::SetVertexBuffers {
                buffers,
                offsets,
                range,
            } => {
                let at = slots(range, &[buffers.len(), offsets.len()])?;
                for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(at) {
                    e.set_vertex_buffer(buffer.map(|b| b.rt()), offset, index)?;
                }
            }
            RenderCmd::SetVertexBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => {
                let at = slots(range, &[buffers.len(), offsets.len(), strides.len()])?;
                for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(at) {
                    e.set_vertex_buffer(buffer.map(|b| b.rt()), offset, index)?;
                }
            }
            RenderCmd::SetFragmentBuffers {
                buffers,
                offsets,
                range,
            } => {
                let at = slots(range, &[buffers.len(), offsets.len()])?;
                for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(at) {
                    e.set_fragment_buffer(buffer.map(|b| b.rt()), offset, index)?;
                }
            }
            RenderCmd::SetFragmentTexture { texture, index } => {
                let index = small(index)?;
                e.set_fragment_texture(texture.map(|t| t.rt()), index)?;
                self.bound(seen, 1, index, texture);
            }
            RenderCmd::SetVertexTextures { textures, range } => {
                let at = slots(range, &[textures.len()])?;
                for (texture, index) in textures.iter().zip(at) {
                    e.set_vertex_texture(texture.map(|t| t.rt()), index)?;
                    self.bound(seen, 0, index, *texture);
                }
            }
            RenderCmd::SetFragmentTextures { textures, range } => {
                let at = slots(range, &[textures.len()])?;
                for (texture, index) in textures.iter().zip(at) {
                    e.set_fragment_texture(texture.map(|t| t.rt()), index)?;
                    self.bound(seen, 1, index, *texture);
                }
            }
            RenderCmd::SetVertexSamplerStates { samplers, range } => {
                let at = slots(range, &[samplers.len()])?;
                for (sampler, index) in samplers.iter().zip(at) {
                    e.set_vertex_sampler_state(sampler.map(|s| s.rt()), index)?;
                    self.kept.reads().note(sampler.and_then(|s| s.index_use()));
                }
            }
            RenderCmd::SetFragmentSamplerStates { samplers, range } => {
                let at = slots(range, &[samplers.len()])?;
                for (sampler, index) in samplers.iter().zip(at) {
                    e.set_fragment_sampler_state(sampler.map(|s| s.rt()), index)?;
                    self.kept.reads().note(sampler.and_then(|s| s.index_use()));
                }
            }
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let lengths = [samplers.len(), lod_min_clamps.len(), lod_max_clamps.len()];
                let at = slots(range, &lengths)?;
                let clamps = lod_min_clamps.iter().zip(lod_max_clamps);
                for ((sampler, (&min, &max)), index) in samplers.iter().zip(clamps).zip(at) {
                    self.kept.reads().note(sampler.and_then(|s| s.index_use()));
                    let sampler = sampler.map(|s| s.rt());
                    e.set_vertex_sampler_state_with_lod_clamps(sampler, min, max, index)?;
                }
            }
            RenderCmd::SetFragmentSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let lengths = [samplers.len(), lod_min_clamps.len(), lod_max_clamps.len()];
                let at = slots(range, &lengths)?;
                let clamps = lod_min_clamps.iter().zip(lod_max_clamps);
                for ((sampler, (&min, &max)), index) in samplers.iter().zip(clamps).zip(at) {
                    self.kept.reads().note(sampler.and_then(|s| s.index_use()));
                    let sampler = sampler.map(|s| s.rt());
                    e.set_fragment_sampler_state_with_lod_clamps(sampler, min, max, index)?;
                }
            }
        }
        Ok(())
    }
}

/// `Ok` where a pipeline state is set ([`Open::pipeline`]); a draw without one is Metal's refusal.
fn set(pipeline: Option<ObjId>) -> Result<(), Stop> {
    match pipeline {
        Some(_) => Ok(()),
        None => Err(Stop::Metal("a draw with no pipeline state set")),
    }
}

/// `Ok` where a setting rt has none of is at Metal's default (`at_default`); unbuilt otherwise.
fn unbuilt_unless(at_default: bool) -> Result<(), Stop> {
    match at_default {
        true => Ok(()),
        false => Err(Stop::Unbuilt(Unbuilt::RenderCommand)),
    }
}

/// The Metal enumeration `T`'s value of `raw`; one it has none of, which `what` names, is Metal's
/// refusal.
fn named<T: TryFrom<u32>>(raw: u64, what: &'static str) -> Result<T, Stop> {
    let value = u32::try_from(raw)
        .ok()
        .and_then(|raw| T::try_from(raw).ok());
    value.ok_or(Stop::Metal(what))
}

/// The primitive type of the draw `cmd`; `None` for a command other than a draw of one.
fn drawn_primitive(cmd: &RenderCmd<'_, RtPlatform>) -> Option<u64> {
    match *cmd {
        RenderCmd::DrawPrimitives { primitive_type, .. }
        | RenderCmd::DrawPrimitivesInstances { primitive_type, .. }
        | RenderCmd::DrawPrimitivesBase { primitive_type, .. }
        | RenderCmd::DrawPrimitivesIndirect { primitive_type, .. }
        | RenderCmd::DrawIndexedPrimitives { primitive_type, .. }
        | RenderCmd::DrawIndexedPrimitivesInstances { primitive_type, .. }
        | RenderCmd::DrawIndexedPrimitivesBase { primitive_type, .. }
        | RenderCmd::DrawIndexedPrimitivesIndirect { primitive_type, .. } => Some(primitive_type),
        _ => None,
    }
}

/// What a draw asked of rt, for the process's first line of each kind of rt's refusals of it
/// (`super::unbuilt::refusal_detail`): its primitive type (Metal's number; none for a command
/// other than a draw of one) and the vertex function of the pipeline state set (none before
/// one is set). The function's name is the guest's.
struct DrawDetail<'a> {
    /// The draw's primitive type.
    primitive: Option<u64>,
    /// The name of the vertex function of the pipeline state set.
    vertex: Option<&'a str>,
}

impl std::fmt::Display for DrawDetail<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.primitive {
            Some(primitive) => write!(f, "a draw of primitive type {primitive}")?,
            None => f.write_str("a command other than a draw of a primitive type")?,
        }
        match self.vertex {
            Some(vertex) => write!(f, ", the pipeline state's vertex function {vertex}"),
            None => f.write_str(", no pipeline state set"),
        }
    }
}

/// `MTLPrimitiveType` of `raw`.
fn primitive(raw: u64) -> Result<PrimitiveType, Stop> {
    named(raw, "a primitive type Metal has none of")
}

/// rt's store action of `MTLStoreAction` `raw`.
fn store(raw: u64) -> Result<rt::StoreAction, Stop> {
    match raw {
        0 => Ok(rt::StoreAction::DontCare),
        1 => Ok(rt::StoreAction::Store),
        2 => Ok(rt::StoreAction::MultisampleResolve),
        3 => Ok(rt::StoreAction::StoreAndMultisampleResolve),
        4 => Ok(rt::StoreAction::Unknown),
        5 => Ok(rt::StoreAction::CustomSampleDepthStore),
        _ => Err(Stop::Metal("a store action Metal has none of")),
    }
}

/// The index buffer of an indexed draw: `buffer` at `offset`, of `MTLIndexType` `index_type`.
fn indices(
    buffer: &RtObj<RtBuffer>,
    offset: u64,
    index_type: u64,
) -> Result<rt::IndexBuffer<'_>, Stop> {
    Ok(rt::IndexBuffer {
        buffer: buffer.rt(),
        offset,
        index_type: named::<IndexType>(index_type, "an index type Metal has none of")?,
    })
}

/// rt's viewport of Metal's.
fn viewport(v: MtlViewport) -> rt::Viewport {
    rt::Viewport {
        origin_x: v.origin_x,
        origin_y: v.origin_y,
        width: v.width,
        height: v.height,
        znear: v.znear,
        zfar: v.zfar,
    }
}

/// rt's scissor rectangle of Metal's.
fn scissor(r: MtlScissorRect) -> rt::ScissorRect {
    rt::ScissorRect {
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
    }
}

/// Notes the copies `cmd` uses in `used` ([`Used`]): those the buffers it sets or draws from and
/// the heaps and resources it declares lie over, each in a use that can write it but an index,
/// indirect or tessellation factor buffer's and a resource's declared for reading alone. Its
/// textures are noted as they bind ([`Kept::used`]), its pass's attachments at its making
/// ([`super::PassFields::note_copies`]).
fn note_copies(used: &Used, cmd: &RenderCmd<'_, RtPlatform>) {
    if !super::copied::any() {
        return;
    }
    match cmd {
        RenderCmd::DrawIndexedPatchesIndirect {
            patch_index_buffer,
            control_point_index_buffer,
            indirect_buffer,
            ..
        } => {
            patch_index_buffer.iter().for_each(|b| used.buffer_read(b));
            used.buffer_read(control_point_index_buffer);
            used.buffer_read(indirect_buffer);
        }
        RenderCmd::DrawIndexedPatches {
            patch_index_buffer,
            control_point_index_buffer,
            ..
        } => {
            patch_index_buffer.iter().for_each(|b| used.buffer_read(b));
            used.buffer_read(control_point_index_buffer);
        }
        RenderCmd::DrawIndexedPrimitivesIndirect {
            index_buffer,
            indirect_buffer,
            ..
        } => {
            used.buffer_read(index_buffer);
            used.buffer_read(indirect_buffer);
        }
        RenderCmd::DrawPatchesIndirect {
            patch_index_buffer,
            indirect_buffer,
            ..
        } => {
            patch_index_buffer.iter().for_each(|b| used.buffer_read(b));
            used.buffer_read(indirect_buffer);
        }
        RenderCmd::DrawPatches {
            patch_index_buffer: buffer,
            ..
        }
        | RenderCmd::SetTessellationFactorBuffer { buffer, .. } => {
            buffer.iter().for_each(|b| used.buffer_read(b));
        }
        RenderCmd::SetFragmentBuffer { buffer, .. } | RenderCmd::SetVertexBuffer { buffer, .. } => {
            buffer.iter().for_each(|b| used.buffer(b));
        }
        RenderCmd::DrawIndexedPrimitives {
            index_buffer: buffer,
            ..
        }
        | RenderCmd::DrawIndexedPrimitivesInstances {
            index_buffer: buffer,
            ..
        }
        | RenderCmd::DrawIndexedPrimitivesBase {
            index_buffer: buffer,
            ..
        }
        | RenderCmd::DrawPrimitivesIndirect {
            indirect_buffer: buffer,
            ..
        }
        | RenderCmd::ExecuteCommandsIndirect {
            indirect_range_buffer: buffer,
            ..
        } => used.buffer_read(buffer),
        RenderCmd::SetFragmentBuffers { buffers, .. }
        | RenderCmd::SetVertexBuffersAttributeStrides { buffers, .. }
        | RenderCmd::SetVertexBuffers { buffers, .. } => {
            buffers.iter().flatten().for_each(|b| used.buffer(b));
        }
        RenderCmd::UseHeaps { heaps } | RenderCmd::UseHeapsStages { heaps, .. } => {
            heaps.iter().flatten().for_each(|h| used.heap(h));
        }
        RenderCmd::UseResources { resources, usage }
        | RenderCmd::UseResourcesStages {
            resources, usage, ..
        } => {
            resources
                .iter()
                .flatten()
                .for_each(|r| used.resource(r, *usage));
        }
        _ => {}
    }
}

impl RenderEncoder<RtPlatform> for RtObj<RtRenderEncoder> {
    /// rt's command (the module's docs); `endEncoding` ends rt's encoder.
    fn encode(&self, on: OnFault, cmd: RenderCmd<'_, RtPlatform>) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(@cmd self, cmd);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            let sel = cmd.selector();
            let ending = matches!(cmd, RenderCmd::EndEncoding);
            note_copies(self.kept.copies(), &cmd);
            let primitive = drawn_primitive(&cmd);
            let mut open = self.open.lock();
            let stopped = self.command(&mut open, cmd).err();
            if let Some(Stop::Rt(error @ rt::Error::Unsupported(rt::Gap::Draw { kind }))) = &stopped
            {
                let vertex = open.seen.state.as_ref().map(|state| state.vertex_name());
                let what = DrawDetail { primitive, vertex };
                super::unbuilt::refusal_detail(sel, error, Some(&what));
                match kind {
                    rt::DrawKind::AttachmentRead => self.attachment_read(&open.seen),
                    rt::DrawKind::FetchedAttachment => self.fetched_attachment(&open.seen),
                    _ => {}
                }
            }
            if ending && let Some(e) = open.rt.take() {
                e.end_encoding();
                self.mark.end();
            }
            drop(open);
            report(sel, stopped);
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// 0, counted ([`Unbuilt::RenderPass`]): rt's pass has no tiles.
    fn tile_height(&self) -> u64 {
        refuse(Unbuilt::RenderPass, "tileHeight");
        0
    }

    /// 0, counted ([`Unbuilt::RenderPass`]): rt's pass has no tiles.
    fn tile_width(&self) -> u64 {
        refuse(Unbuilt::RenderPass, "tileWidth");
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::num::ToU64 as _;
    use crate::core::platform::Platform as _;
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::descriptor::RenderPassDescriptor as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, Library as _};
    use crate::core::platform::types::{ClearColor, CommandBufferStatus, NsRange};
    use crate::core::platform::values::{
        AttributeDesc, BufferLayoutDesc, ColorAttachmentDesc, ColorAttachmentSet,
        FullRenderPipelineDesc, PassSet, TextureDesc, VertexDesc,
    };
    use crate::rt_platform::unbuilt::RtRefusal;
    use crate::rt_platform::{PassFields, RtCommandBuffer, RtTexture};

    /// A vertex function fed through a vertex descriptor: attribute 0 a `float2` position, 1 a
    /// normalized `uchar4` colour.
    const VDESC: &[u8] = include_bytes!("../../../../tests/shaders/render_vdesc.metallib");

    /// A fragment function that returns the interpolated colour.
    const COLOR: &[u8] = include_bytes!("../../../../tests/shaders/render_color.metallib");

    /// `MTLPrimitiveTypeTriangle`.
    const TRIANGLE: u64 = 3;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A refused draw's detail names its primitive type and the vertex function of the pipeline
    /// state set, or that the command draws no primitive type, or that no state is set.
    #[test]
    fn a_refused_draw_s_detail_names_its_primitive_and_vertex_function() {
        let draw = drawn_primitive(&RenderCmd::<RtPlatform>::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 3,
        });
        assert_eq!(draw, Some(0));
        let what = DrawDetail {
            primitive: draw,
            vertex: Some("vertex_points"),
        };
        assert_eq!(
            what.to_string(),
            "a draw of primitive type 0, the pipeline state's vertex function vertex_points"
        );
        let what = DrawDetail {
            primitive: drawn_primitive(&RenderCmd::<RtPlatform>::EndEncoding),
            vertex: None,
        };
        assert_eq!(
            what.to_string(),
            "a command other than a draw of a primitive type, no pipeline state set"
        );
    }

    /// A pass of `texture` as colour attachment 0, of `load_action` (cleared to opaque red) and a
    /// `Store` store.
    fn pass(texture: &RtObj<RtTexture>, load_action: u64) -> RtObj<PassFields> {
        let pass = RtObj::new(PassFields::default());
        let set = ColorAttachmentSet {
            texture: Some(Some(texture)),
            load_action: Some(load_action),
            store_action: Some(1),
            clear_color: Some(ClearColor {
                red: 1.0,
                alpha: 1.0,
                ..ClearColor::default()
            }),
            ..ColorAttachmentSet::default()
        };
        let set = pass.set_color_attachment(OnFault::Return, 0, &set);
        assert_eq!(set, Ok(()));
        pass
    }

    /// Encodes `cmds` into one render encoder of `pass`, in a command buffer `made`, ends it,
    /// commits the command buffer and waits for it: its status then.
    fn run(
        made: &RtObj<RtCommandBuffer>,
        pass: &RtObj<PassFields>,
        cmds: Vec<RenderCmd<'_, RtPlatform>>,
    ) -> CommandBufferStatus {
        let on = OnFault::Return;
        let encoder = made.render_encoder(on, pass).expect("never a raise");
        let encoder = encoder.expect("a render encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, RenderCmd::EndEncoding), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        made.status()
    }

    /// Render encoders clear, load and store their pass's attachment, and draw through a
    /// pipeline state and a vertex buffer bound to them; what rt has no form of is counted as
    /// its piece, what Metal's validation stops as Metal's refusal, and the command buffer still
    /// completes.
    #[test]
    fn render_encoders_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let done = CommandBufferStatus::COMPLETED;
        let metal = RtRefusal::Refused.code();
        let function = |bytes: &[u8], name: &str| {
            let library = device.new_library(bytes).expect("never a raise");
            let library = library.expect("a metallib").expect("a library");
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the table")
        };
        let (vdesc, color) = (
            function(VDESC, "render_vdesc"),
            function(COLOR, "render_color"),
        );
        let mut desc = FullRenderPipelineDesc::new(&vdesc, Some(&color));
        desc.vertex_descriptor = Some(VertexDesc {
            layouts: vec![BufferLayoutDesc {
                buffer_index: 0,
                stride: 12,
                step_function: 1,
                step_rate: 1,
            }],
            attributes: vec![
                AttributeDesc {
                    attribute_index: 0,
                    format: 29,
                    offset: 0,
                    buffer_index: 0,
                },
                AttributeDesc {
                    attribute_index: 1,
                    format: 9,
                    offset: 8,
                    buffer_index: 0,
                },
            ],
        });
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 70,
                ..ColorAttachmentDesc::default()
            },
        )];
        let state = device.new_render_pipeline_state_full(&desc);
        let state = state.expect("a render state").expect("a render state");

        // One green triangle over the whole target: (-1, -1), (3, -1), (-1, 3).
        let mut vertices = Vec::new();
        for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)] {
            vertices.extend_from_slice(&x.to_le_bytes());
            vertices.extend_from_slice(&y.to_le_bytes());
            vertices.extend_from_slice(&[0, 255, 0, 255]);
        }
        let buffer = device.new_buffer_of_length(on, vertices.len().to_u64(), 0);
        let buffer = buffer.expect("never a raise").expect("a buffer");
        // SAFETY: the Shared buffer's contents are mapped, `vertices` long.
        unsafe {
            let at = buffer.contents().as_ptr::<u8>();
            at.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        }
        let target = device.new_texture(
            on,
            &platform.texture_descriptor(&TextureDesc {
                width: 4,
                height: 4,
                usage: 5,
                ..TextureDesc::default()
            }),
        );
        let target = target.expect("never a raise").expect("a render target");
        let texels = || {
            let mut texels = [0u8; 64];
            target.read_pixels(&mut texels, 16);
            texels
        };
        let every = |texel: [u8; 4]| -> [u8; 64] { std::array::from_fn(|i| texel[i % 4]) };
        let (red, green) = ([255, 0, 0, 255], [0, 255, 0, 255]);

        // A cleared pass with no draw stores its clear colour.
        assert_eq!(run(&made(), &pass(&target, 2), Vec::new()), done);
        assert_eq!(texels(), every(red));

        // A draw over the loaded target.
        let draw = vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&buffer),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
        ];
        assert_eq!(run(&made(), &pass(&target, 1), draw), done);
        assert_eq!(texels(), every(green));

        // A loaded pass with no draw keeps what was stored.
        assert_eq!(run(&made(), &pass(&target, 1), Vec::new()), done);
        assert_eq!(texels(), every(green));

        // A pass with a visibility result buffer, its set counted nowhere, draws over its cleared
        // target with no mode enabled, and the buffer's bytes stay as they were.
        let results = device.new_buffer_of_length(on, 16, 0);
        let results = results.expect("never a raise").expect("a buffer");
        // SAFETY: the Shared buffer's contents are mapped, 16 bytes long.
        unsafe { results.contents().as_ptr::<u8>().write_bytes(0xa5, 16) };
        let passes = counted(Unbuilt::RenderPass.code());
        let kept = pass(&target, 2);
        kept.set_pass(&PassSet {
            visibility_result_buffer: Some(Some(&results)),
            ..PassSet::default()
        });
        assert_eq!(counted(Unbuilt::RenderPass.code()), passes);
        let d = kept.rt_pass().expect("rt's descriptor");
        assert!(d.visibility_result_buffer.is_some());
        let draw = vec![
            RenderCmd::SetVisibilityResultMode { mode: 0, offset: 8 },
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&buffer),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
        ];
        assert_eq!(run(&made(), &kept, draw), done);
        assert_eq!(texels(), every(green));
        // SAFETY: as above; the command buffer has completed.
        let bytes = unsafe { std::slice::from_raw_parts(results.contents().as_ptr::<u8>(), 16) };
        assert_eq!(bytes, [0xa5; 16]);

        // A pass Metal refuses, and one rt refuses: nil, counted.
        let before = counted(metal);
        assert!(matches!(
            made().render_encoder(on, &pass(&target, 7)),
            Ok(None)
        ));
        assert_eq!(counted(metal), before + 1);
        let gaps = counted(RtRefusal::Gap.code());
        let empty = platform.new_render_pass_descriptor();
        assert!(matches!(made().render_encoder(on, &empty), Ok(None)));
        assert_eq!(counted(RtRefusal::Gap.code()), gaps + 1);

        // What rt has no form of, what Metal's validation stops, and what records nothing; the
        // draws stopped draw nothing over the cleared target.
        let (unbuilt, pass_unbuilt) = (Unbuilt::RenderCommand.code(), Unbuilt::RenderPass.code());
        let (before, unbuilt_before, pass_before) =
            (counted(metal), counted(unbuilt), counted(pass_unbuilt));
        let scale = Unbuilt::TessellationFactorScale.code();
        let scale_before = counted(scale);
        let commands = made();
        let encoder = commands.render_encoder(on, &pass(&target, 2));
        let encoder = encoder.expect("never a raise").expect("a render encoder");
        assert_eq!(encoder.tile_width(), 0);
        assert_eq!(encoder.tile_height(), 0);
        let cmds = [
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&buffer),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 9,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::DrawPrimitivesInstances {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1 << 32,
            },
            RenderCmd::SetCullMode { cull_mode: 7 },
            RenderCmd::SetVertexBuffers {
                buffers: &[Some(&buffer)],
                offsets: &[],
                range: NsRange::new(0, 1),
            },
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type: TRIANGLE,
                indirect_buffer: &buffer,
                indirect_buffer_offset: 0,
            },
            RenderCmd::SetVisibilityResultMode { mode: 1, offset: 0 },
            RenderCmd::SetLineWidth { line_width: 2.0 },
            RenderCmd::SetAlphaTestReferenceValue {
                reference_value: 0.5,
            },
            RenderCmd::SetVisibilityResultMode { mode: 0, offset: 0 },
            RenderCmd::SetLineWidth { line_width: 1.0 },
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: true,
                index: 0,
            },
            RenderCmd::UseResources {
                resources: &[None],
                usage: 1,
            },
            RenderCmd::SetTessellationFactorScale { scale: 2.0 },
            RenderCmd::SetTessellationFactorScale { scale: 1.0 },
            RenderCmd::SetRenderPipelineState {
                pipeline_state: None,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
            RenderCmd::SetBlendColorRed {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            },
        ];
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(counted(metal), before + 7);
        assert_eq!(counted(unbuilt), unbuilt_before + 4);
        assert_eq!(counted(pass_unbuilt), pass_before + 2);
        assert_eq!(counted(scale), scale_before + 1);
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(commands.status(), done);
        assert_eq!(texels(), every(red));
    }

    /// A pass whose colour attachment, a 4-sample texture, resolves into its resolve texture
    /// leaves the cleared colour there, as Metal's resolve does; one that resolves with no
    /// resolve texture is Metal's refusal, nil and counted.
    #[test]
    fn a_pass_resolves_into_its_resolve_texture_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let texture = |desc: TextureDesc| {
            let made = device.new_texture(on, &platform.texture_descriptor(&desc));
            made.expect("never a raise").expect("a texture")
        };
        // `MTLTextureType2DMultisample` of 4 samples, render target usage and Private storage.
        let samples = texture(TextureDesc {
            texture_type: 4,
            sample_count: 4,
            width: 4,
            height: 4,
            usage: 4,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let resolved = texture(TextureDesc {
            width: 4,
            height: 4,
            usage: 5,
            ..TextureDesc::default()
        });
        // A cleared pass of `samples` with store action `store` (opaque red), resolving into
        // `into`.
        let pass = |store: u64, into: Option<&RtObj<RtTexture>>| {
            let pass = RtObj::new(PassFields::default());
            let set = ColorAttachmentSet {
                texture: Some(Some(&samples)),
                resolve_texture: Some(into),
                load_action: Some(2),
                store_action: Some(store),
                clear_color: Some(ClearColor {
                    red: 1.0,
                    alpha: 1.0,
                    ..ClearColor::default()
                }),
                ..ColorAttachmentSet::default()
            };
            assert_eq!(pass.set_color_attachment(on, 0, &set), Ok(()));
            pass
        };
        let resolve = 2;
        let done = CommandBufferStatus::COMPLETED;
        assert_eq!(
            run(&made(), &pass(resolve, Some(&resolved)), Vec::new()),
            done
        );
        let mut texels = [0u8; 64];
        resolved.read_pixels(&mut texels, 16);
        let red: [u8; 64] = std::array::from_fn(|i| [255, 0, 0, 255][i % 4]);
        assert_eq!(texels, red);

        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        let refused = made().render_encoder(on, &pass(resolve, None));
        assert!(matches!(refused, Ok(None)));
        assert_eq!(counted(metal), before + 1);
    }
}
