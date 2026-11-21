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
//! - **Indirect draws** of primitives, indexed primitives and patches are rt's: the GPU reads the
//!   draw's arguments (`MTLDrawPrimitivesIndirectArguments`,
//!   `MTLDrawIndexedPrimitivesIndirectArguments`, `MTLDrawPatchIndirectArguments`) from the
//!   buffer at the offset as the draw runs, so arguments of no vertices, indices or patches draw
//!   nothing, and rt's command buffer keeps the buffer until it retires. An offset rt refuses (one
//!   not a multiple of 4, or without the arguments' 16 or 20 bytes after it in the buffer) is
//!   rt's refusal of Metal's (`Refusal::IndirectOffset`), counted (`rt-refused-metal`), and a patch
//!   index buffer rt's gap, as the direct patch draw's is.
//! - **Attribute strides** (`setVertexBufferOffset:attributeStride:atIndex:`,
//!   `setVertexBuffers:offsets:attributeStrides:withRange:`) bind as the calls without them: only a
//!   layout of `MTLBufferLayoutStrideDynamic` reads a stride, and rt refuses such a state as it is
//!   made.
//! - **Executions of an indirect command buffer** (`executeCommandsInBuffer:withRange:`, and
//!   `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` of a range the GPU reads from
//!   the buffer as the execution runs) are rt's (`super::icb`): each command draws with the
//!   encoder's pipeline state and render states and the buffers the command's encoding bound, and
//!   the command buffer keeps the buffer's records, and a range buffer, until it retires. Without
//!   a pipeline state set it is Metal's refusal, as a draw is; rt refuses a range past the
//!   buffer's commands and a range buffer's offset that is not a multiple of 4 or without 8
//!   bytes after it (`Refusal::CommandRange`), counted (`rt-refused-metal`), and a state made
//!   without `supportIndirectCommandBuffers` or one whose commands carry more buffer slots than
//!   rt's push range holds or the device's tokens take (`Gap::IndirectCommandBuffers`, the
//!   module `super::icb`'s "The commands' slots"), counted (`rt-refused-gap`), each drawing
//!   nothing. The first line of each kind of rt's refusals of an execution names the buffer's
//!   bind counts and the device's tokens (`super::icb::ExecutionDetail`).
//! - **`useResources:`** in each form is rt's `use_resource` of each resource it names but nil, as
//!   the compute encoder's is (`super::compute`): the command buffer keeps the resource until it
//!   retires, and rt records nothing. A tiled texture it names is a use of it (`super::tiled`), and
//!   a texture at one of the platform's own indices a read through it (`Kept::reads`). After a
//!   usage that writes a texture made for shader writes with no storage element, each draw of the
//!   encoder whose vertex or fragment function may reach a texture beyond its slots (an argument
//!   buffer's `access::write` or `access::read_write` texture, a resource ID) is rt's refusal
//!   (`Gap::StorageElement`), encoding nothing, counted (`rt-refused-gap`), as the compute
//!   encoder's dispatch is: Metal writes such a texture there, and Vulkan would drop the writes.
//! - **Nothing to record**: `useHeaps:` in each form (rt's pass orders every earlier command before
//!   it and itself before every later one, and every resource is resident); the store action
//!   options (rt renders one sample, and the one option is custom sample positions); the object,
//!   mesh and tile functions' buffers, textures, samplers and threadgroup memory, and the tile
//!   threadgroup memory length (rt makes no state with such functions); and, each at Metal's
//!   default, a resolve texture of nil, a line width of 1, primitive restart enabled, the viewport
//!   transform enabled, one vertex amplification view and a tessellation factor scale of 1.
//! - **The visibility result mode** is rt's: `Disabled`, at any offset, records nothing, and
//!   `Boolean` and `Counting` are rt's gap (`Gap::Draw` of `VisibilityResults`), encoding nothing,
//!   counted (`rt-refused-gap`), so the encoder's later draws write no result.
//! - **The alpha test's reference** (`setAlphaTestReferenceValue:`) is rt's, which a state of
//!   rt's private alpha test reads at its draws. No state of the platform's has that test
//!   (`super::render_pipeline`'s private state has none), so no draw reads the reference.
//! - **Unbuilt** ([`Unbuilt::RenderCommand`]), each encoding nothing, counted: indexed patch
//!   draws, direct and indirect (rt has none, and refuses every state of a control point index
//!   type), mesh draws, tile dispatches, a resolve texture, and the private settings of other than
//!   Metal's default rt has none of (a point size, the provoking vertex, primitive restart
//!   disabled, the viewport transform disabled, vertex amplification, and an attachment marked
//!   cleared). So is a line width other than 1, which is set nowhere, so the encoder's later lines
//!   draw 1 pixel wide: rt keeps such a width but refuses each draw of lines under it
//!   (`Gap::Draw` of `LineWidth`), which would draw them not at all. `tileWidth` and `tileHeight`
//!   answer 0, counted under
//!   [`Unbuilt::RenderPass`]: rt's pass has no tiles. A tessellation factor scale other than 1
//!   is counted under [`Unbuilt::TessellationFactorScale`]: rt has no call of it, and the
//!   encoder's later patch draws read their factors unscaled.
//! - **Metal's refusals**, each encoding nothing, counted (`rt-refused-metal`): a draw with no
//!   pipeline state set, or with nil set since; a value no Metal enumeration has (a primitive,
//!   index or fill type, a cull mode, a winding, a depth clip mode, a store action, a visibility
//!   result mode, a resource usage); an index, a count or a base past rt's 32 bits, and a range
//!   of commands past them; a list of another length than its range; and a command after
//!   `endEncoding`.
//!
//! A command rt refuses encodes nothing, counted under the refusal's kind; in a list, the entries
//! before the refused one are bound.

use metal2vulkan::metal::{
    CullMode, DepthClipMode, IndexType, PrimitiveType, TriangleFillMode, Winding,
};
use metal2vulkan_runtime as rt;

use crate::core::platform::render::{RenderCmd, RenderEncoder};
use crate::core::platform::types::{MtlScissorRect, MtlViewport};
use crate::core::platform::{Handle, ObjId};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;

use super::blit::{Stop, report};
use super::command::{Kept, OpenMark};
use super::compute::{declared, resource_usage, slots, small};
use super::copied::Used;
use super::diag::{self, Bound, Target, Use};
use super::fault::{EncoderKind, EncoderNames, Naming};
use super::render_pipeline::{Raster, RtRenderPipeline};
use super::texture::RtTexture;
use super::unbuilt::{Unbuilt, refuse};
use super::{RtBuffer, RtObj, RtPlatform};

/// A count, a base or an index past rt's 32 bits.
const WIDE: Stop = Stop::Metal("a count, a base or an index past 32 bits");

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
    /// What its pass and commands name, for a lost device's lines (`super::fault`'s docs, "The
    /// command buffers in flight"), which join its command buffer's as it ends or goes.
    naming: Naming,
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
        let names = EncoderNames {
            attachments: targets.iter().map(|t| (t.point, t.id)).collect(),
            ..EncoderNames::new(EncoderKind::Render)
        };
        RtRenderEncoder {
            open: Mutex::new(Open {
                rt: Some(encoder),
                pipeline: None,
                seen: Seen::default(),
                naming: Naming::new(kept.encoded(), names),
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
            ..
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
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                set(*pipeline)?;
                let kind = primitive(primitive_type)?;
                e.draw_primitives_indirect(kind, indirect_buffer.rt(), indirect_buffer_offset)?;
            }
            RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type,
                index_buffer,
                index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                set(*pipeline)?;
                let indices = indices(index_buffer, index_buffer_offset, index_type)?;
                let kind = primitive(primitive_type)?;
                let (arguments, offset) = (indirect_buffer.rt(), indirect_buffer_offset);
                e.draw_indexed_primitives_indirect(kind, indices, arguments, offset)?;
            }
            RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                set(*pipeline)?;
                let points = small(number_of_patch_control_points)?;
                let patch_indices = patch_index_buffer.map(|b| (b.rt(), patch_index_buffer_offset));
                let (arguments, offset) = (indirect_buffer.rt(), indirect_buffer_offset);
                e.draw_patches_indirect(points, patch_indices, arguments, offset)?;
            }
            RenderCmd::DrawIndexedPatchesIndirect { .. }
            | RenderCmd::DrawIndexedPatches { .. }
            | RenderCmd::DrawMeshThreadgroups { .. }
            | RenderCmd::DrawMeshThreadgroupsIndirect { .. }
            | RenderCmd::DrawMeshThreads { .. }
            | RenderCmd::DispatchThreadsPerTile { .. }
            | RenderCmd::DispatchThreadsPerTileInRegion { .. }
            | RenderCmd::DispatchThreadsPerTileInRegionIndex { .. }
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
            RenderCmd::SetVisibilityResultMode { mode, offset } => {
                e.set_visibility_result_mode(visibility_mode(mode)?, offset)?;
            }
            RenderCmd::SetAlphaTestReferenceValue { reference_value } => {
                e.set_alpha_test_reference_value(reference_value);
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
            RenderCmd::UseResources { resources, usage }
            | RenderCmd::UseResourcesStages {
                resources, usage, ..
            } => {
                let usage = resource_usage(usage)?;
                for resource in resources.iter().flatten() {
                    e.use_resource(declared(&self.kept, resource), usage)?;
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
                indirect_range_buffer,
                indirect_buffer_offset,
            } => {
                set(*pipeline)?;
                let (commands, ranges) = (indirect_commandbuffer.rt(), indirect_range_buffer.rt());
                e.execute_commands_in_buffer_with_range(commands, ranges, indirect_buffer_offset)?;
            }
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => {
                set(*pipeline)?;
                let range = super::icb::commands(execution_range)?;
                e.execute_commands_in_buffer(indirect_command_buffer.rt(), range)?;
            }
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
            } => e.set_depth_bias(depth_bias, slope_scale, clamp),
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

/// What an execution `cmd` asks of rt ([`super::icb::ExecutionDetail`]), for the process's first
/// line of each kind of rt's refusals of one; none for a command other than an execution.
fn execution(cmd: &RenderCmd<'_, RtPlatform>) -> Option<super::icb::ExecutionDetail> {
    match cmd {
        RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: icb,
            ..
        }
        | RenderCmd::ExecuteCommandsIndirect {
            indirect_commandbuffer: icb,
            ..
        } => Some(*icb.detail()),
        _ => None,
    }
}

/// What a draw asked of rt, for the process's first line of each kind of rt's refusals of it
/// (`super::unbuilt::refusal_detail`), as structure alone: its primitive type (Metal's number;
/// none for a command other than a draw of one) and the rasterization fields of the pipeline
/// state set (none before one is set). Whether the state's vertex function writes
/// `[[point_size]]` is rt's to know (its translator's), which no call of rt's answers here, so the
/// line says it is not known.
struct DrawDetail {
    /// The draw's primitive type.
    primitive: Option<u64>,
    /// The rasterization fields of the pipeline state set.
    raster: Option<Raster>,
}

impl std::fmt::Display for DrawDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.primitive {
            Some(primitive) => write!(f, "a draw of primitive type {primitive}")?,
            None => f.write_str("a command other than a draw of a primitive type")?,
        }
        match self.raster {
            Some(raster) => write!(
                f,
                ", the pipeline state's {raster}, its vertex function's point size write not known \
                 here"
            ),
            None => f.write_str(", no pipeline state set"),
        }
    }
}

/// `MTLPrimitiveType` of `raw`.
fn primitive(raw: u64) -> Result<PrimitiveType, Stop> {
    named(raw, "a primitive type Metal has none of")
}

/// rt's visibility result mode of `MTLVisibilityResultMode` `raw`.
fn visibility_mode(raw: u64) -> Result<rt::VisibilityResultMode, Stop> {
    match raw {
        0 => Ok(rt::VisibilityResultMode::Disabled),
        1 => Ok(rt::VisibilityResultMode::Boolean),
        2 => Ok(rt::VisibilityResultMode::Counting),
        _ => Err(Stop::Metal("a visibility result mode Metal has none of")),
    }
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

/// Notes in `names` what `cmd` names (`super::fault`'s docs, "The command buffers in flight"): the
/// buffers it binds to a vertex or fragment slot or as the tessellation factor buffer, and the
/// heaps and resources it declares, by their guest ids (rt keeps no mesh, object or tile buffers).
fn note_names(names: &mut EncoderNames, cmd: &RenderCmd<'_, RtPlatform>) {
    match cmd {
        RenderCmd::SetFragmentBuffer { buffer, .. }
        | RenderCmd::SetVertexBuffer { buffer, .. }
        | RenderCmd::SetTessellationFactorBuffer { buffer, .. } => {
            if let Some(buffer) = buffer {
                names.bound_buffers.add(buffer.named().id());
            }
        }
        RenderCmd::SetFragmentBuffers { buffers, .. }
        | RenderCmd::SetVertexBuffersAttributeStrides { buffers, .. }
        | RenderCmd::SetVertexBuffers { buffers, .. } => {
            for buffer in buffers.iter().flatten() {
                names.bound_buffers.add(buffer.named().id());
            }
        }
        RenderCmd::UseHeaps { heaps } | RenderCmd::UseHeapsStages { heaps, .. } => {
            for heap in heaps.iter().flatten() {
                names.heaps.add(heap.named().id());
            }
        }
        RenderCmd::UseResources { resources, .. }
        | RenderCmd::UseResourcesStages { resources, .. } => {
            super::compute::note_declared(names, resources);
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
            let execution = execution(&cmd);
            let mut open = self.open.lock();
            if let Some(names) = open.naming.names() {
                note_names(names, &cmd);
            }
            let stopped = self.command(&mut open, cmd).err();
            if let Some(Stop::Rt(error @ rt::Error::Unsupported(rt::Gap::Draw { kind }))) = &stopped
            {
                let raster = open.seen.state.as_ref().map(|state| state.raster());
                let what = DrawDetail { primitive, raster };
                super::unbuilt::refusal_detail(sel, error, Some(&what));
                match kind {
                    rt::DrawKind::AttachmentRead => self.attachment_read(&open.seen),
                    rt::DrawKind::FetchedAttachment => self.fetched_attachment(&open.seen),
                    _ => {}
                }
            }
            let icb_gap = |error: &rt::Error| {
                matches!(
                    error,
                    rt::Error::Unsupported(rt::Gap::IndirectCommandBuffers { .. })
                )
            };
            if let (Some(Stop::Rt(error)), Some(what)) = (&stopped, &execution)
                && icb_gap(error)
            {
                super::unbuilt::refusal_detail(sel, error, Some(what));
            }
            if ending && let Some(e) = open.rt.take() {
                e.end_encoding();
                self.mark.end();
                open.naming.end();
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
    use crate::core::platform::blit::ResourceRef;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::descriptor::RenderPassDescriptor as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::icb::IndirectCommandBuffer as _;
    use crate::core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use crate::core::platform::types::{ClearColor, CommandBufferStatus, MtlSize, NsRange};
    use crate::core::platform::values::IcbDesc;
    use crate::core::platform::values::{
        AttributeDesc, BufferLayoutDesc, ColorAttachmentDesc, ColorAttachmentSet,
        FullRenderPipelineDesc, PassSet, TextureDesc, VertexDesc,
    };
    use crate::rt_platform::unbuilt::RtRefusal;
    use crate::rt_platform::{
        PassFields, RtCommandBuffer, RtComputePipeline, RtDevice, RtFunction, RtIcb, RtTexture,
    };

    /// A vertex function fed through a vertex descriptor: attribute 0 a `float2` position, 1 a
    /// normalized `uchar4` colour.
    const VDESC: &[u8] = include_bytes!("../../../../tests/shaders/render_vdesc.metallib");

    /// A fragment function that returns the interpolated colour.
    const COLOR: &[u8] = include_bytes!("../../../../tests/shaders/render_color.metallib");

    /// A post-tessellation vertex function of triangle patches, `render_tess`, that reads each
    /// patch's three `float2` control points from buffer 0 at `3 * patch_id` and colours it
    /// opaque blue.
    const TESS: &[u8] = include_bytes!("../../../../tests/shaders/render_tess.metallib");

    /// A fragment function that reaches a texture beyond its slots: it writes `(1, 0, 1, 1)` to
    /// texel `(0, 0)` of the texture whose resource ID the argument buffer at fragment buffer 0
    /// holds, and returns opaque green.
    const ARGWRITE: &[u8] = include_bytes!("../../../../tests/shaders/render_argwrite.metallib");

    /// A kernel that writes each texel of texture 0 its grid reaches (`tex_write`).
    const TEXWRITE: &[u8] = include_bytes!("../../../../tests/shaders/compute_texwrite.metallib");

    /// Kernels that encode indexed triangles into the indirect command buffer whose
    /// `gpuResourceID` buffer 0 holds, command `tid` drawing the three 16-bit indices of buffer 3
    /// from `3 * tid` with the `float4` positions of buffer 1 at vertex buffer 0 and its own
    /// `float4` colour of buffer 2 at fragment buffer 0 (`icb_encode`), and at vertex buffers 0 to
    /// 28 (`icb_encode_wide`); vertex functions that place each vertex at its position read from
    /// vertex buffer 0 (`icb_vertex`) or as the sum of buffers 0 to 27 (`icb_vertex_28`) or 0 to
    /// 28 (`icb_vertex_29`); and a fragment function that returns the colour at fragment buffer 0
    /// (`icb_fragment`).
    const ICB: &[u8] = include_bytes!("../../../../tests/shaders/icb_indexed.metallib");

    /// `MTLPrimitiveTypeTriangle`.
    const TRIANGLE: u64 = 3;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A refused draw's detail names its primitive type and the rasterization fields of the
    /// pipeline state set, and no name, or that the command draws no primitive type, or that no
    /// state is set.
    #[test]
    fn a_refused_draw_s_detail_names_its_primitive_and_rasterization_fields() {
        let draw = drawn_primitive(&RenderCmd::<RtPlatform>::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 3,
        });
        assert_eq!(draw, Some(0));
        let what = DrawDetail {
            primitive: draw,
            raster: Some(Raster {
                enabled: true,
                topology: metal2vulkan::metal::PrimitiveTopologyClass::Unspecified,
                samples: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                open_gl_mode: true,
            }),
        };
        assert_eq!(
            what.to_string(),
            "a draw of primitive type 0, the pipeline state's rasterization enabled true, input \
             primitive topology Unspecified, raster sample count 1, alpha to coverage false, alpha \
             to one false, OpenGL mode true, its vertex function's point size write not known here"
        );
        let what = DrawDetail {
            primitive: drawn_primitive(&RenderCmd::<RtPlatform>::EndEncoding),
            raster: None,
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

    /// The function `name` of the metallib `bytes`, made on `device`.
    fn function(device: &RtObj<RtDevice>, bytes: &[u8], name: &str) -> RtObj<RtFunction> {
        let library = device.new_library(bytes).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let made = library.new_function(OnFault::Return, name);
        made.expect("never a raise")
            .expect("a function of the table")
    }

    /// The state of `render_vdesc` and `fragment` on `device`, into one `RGBA8Unorm` colour
    /// attachment: its vertex descriptor reads a `float2` position at byte 0 and a normalized
    /// `uchar4` colour at byte 8 of each 12 bytes of buffer 0.
    fn vdesc_state(
        device: &RtObj<RtDevice>,
        fragment: &RtObj<RtFunction>,
    ) -> RtObj<RtRenderPipeline> {
        let vdesc = function(device, VDESC, "render_vdesc");
        let mut desc = FullRenderPipelineDesc::new(&vdesc, Some(fragment));
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
        state.expect("a render state").expect("a render state")
    }

    /// One opaque green triangle over the whole of a target, (-1, -1), (3, -1), (-1, 3), in a
    /// Shared buffer of `device` that [`vdesc_state`]'s vertex descriptor reads.
    fn triangle(device: &RtObj<RtDevice>) -> RtObj<RtBuffer> {
        shared(device, &vertices(&[[0, 255, 0, 255]]))
    }

    /// For each of `colours`, a triangle over the whole of a target, (-1, -1), (3, -1), (-1, 3),
    /// in that colour, as [`vdesc_state`]'s vertex descriptor reads them: three vertices each.
    fn vertices(colours: &[[u8; 4]]) -> Vec<u8> {
        let mut vertices = Vec::new();
        for colour in colours {
            for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)] {
                vertices.extend_from_slice(&x.to_le_bytes());
                vertices.extend_from_slice(&y.to_le_bytes());
                vertices.extend_from_slice(colour);
            }
        }
        vertices
    }

    /// A Shared buffer of `device` that holds `bytes`.
    fn shared(device: &RtObj<RtDevice>, bytes: &[u8]) -> RtObj<RtBuffer> {
        let buffer = device.new_buffer_of_length(OnFault::Return, bytes.len().to_u64(), 0);
        let buffer = buffer.expect("never a raise").expect("a buffer");
        // SAFETY: the Shared buffer's contents are mapped, `bytes` long.
        unsafe {
            let at = buffer.contents().as_ptr::<u8>();
            at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
        buffer
    }

    /// `words`' little-endian bytes.
    fn words(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
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
        let state = vdesc_state(&device, &function(&device, COLOR, "render_color"));
        let buffer = triangle(&device);
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

        // Passes Metal refuses: an unknown load action, and a pass of no attachment at its
        // default size and sample count of 0. Nil, counted.
        let before = counted(metal);
        assert!(matches!(
            made().render_encoder(on, &pass(&target, 7)),
            Ok(None)
        ));
        let empty = platform.new_render_pass_descriptor();
        assert!(matches!(made().render_encoder(on, &empty), Ok(None)));
        assert_eq!(counted(metal), before + 2);

        // A pass of no attachment with a size and a sample count runs, and leaves the target as
        // it was.
        empty.set_pass(&PassSet {
            render_target_width: Some(4),
            render_target_height: Some(4),
            default_raster_sample_count: Some(1),
            ..PassSet::default()
        });
        assert_eq!(run(&made(), &empty, Vec::new()), done);
        assert_eq!(counted(metal), before + 2);
        assert_eq!(texels(), every(green));

        // What rt has no form of, what Metal's validation stops, and what records nothing; the
        // draws stopped draw nothing over the cleared target.
        let (unbuilt, pass_unbuilt) = (Unbuilt::RenderCommand.code(), Unbuilt::RenderPass.code());
        let (before, unbuilt_before, pass_before) =
            (counted(metal), counted(unbuilt), counted(pass_unbuilt));
        let scale = Unbuilt::TessellationFactorScale.code();
        let scale_before = counted(scale);
        let gap = RtRefusal::Gap.code();
        let gap_before = counted(gap);
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
            RenderCmd::UseResourcesStages {
                resources: &[Some(ResourceRef::Buffer(&buffer))],
                usage: 8,
                stages: 2,
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
        assert_eq!(counted(metal), before + 8);
        assert_eq!(counted(unbuilt), unbuilt_before + 1);
        assert_eq!(counted(gap), gap_before + 1);
        assert_eq!(counted(pass_unbuilt), pass_before + 2);
        assert_eq!(counted(scale), scale_before + 1);
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(commands.status(), done);
        assert_eq!(texels(), every(red));
    }

    /// The commands that set `state` and `vertices` at vertex buffer 0, then `draw`.
    fn bound<'a>(
        state: &'a RtObj<RtRenderPipeline>,
        vertices: &'a RtObj<RtBuffer>,
        draw: RenderCmd<'a, RtPlatform>,
    ) -> Vec<RenderCmd<'a, RtPlatform>> {
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(vertices),
                offset: 0,
                index: 0,
            },
            draw,
        ]
    }

    /// Indirect draws of primitives, indexed primitives and patches each draw what the arguments
    /// written into their buffer say, read at the draw's offset as the draw runs: a vertex start,
    /// an index start and a base vertex, a patch start. Arguments of no vertices or patches draw
    /// nothing. An offset off a multiple of 4, or without the arguments' bytes after it, is
    /// Metal's refusal, and a patch index buffer rt's gap, each counted and drawing nothing.
    /// None is counted unbuilt.
    #[test]
    fn indirect_draws_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let done = CommandBufferStatus::COMPLETED;
        let (metal, gap) = (RtRefusal::Refused.code(), RtRefusal::Gap.code());
        let unbuilt = Unbuilt::RenderCommand.code();
        let before = (counted(metal), counted(gap), counted(unbuilt));
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
        // Each draw's texels, in a pass of its own that clears the target to opaque red.
        let drawn = |cmds| {
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            let mut texels = [0u8; 64];
            target.read_pixels(&mut texels, 16);
            texels
        };
        let every = |texel: [u8; 4]| -> [u8; 64] { std::array::from_fn(|i| texel[i % 4]) };
        let (red, green, blue) = ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]);

        // Vertices 0 to 2 a green triangle over the target, 3 to 5 a blue one; indices 5, 5, 5
        // (one vertex thrice, which draws no pixel), then 0, 1, 2.
        let color = function(&device, COLOR, "render_color");
        let state = &vdesc_state(&device, &color);
        let triangles = &shared(&device, &vertices(&[green, blue]));
        let index_bytes: Vec<u8> = [5u16, 5, 5, 0, 1, 2]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let indices = &shared(&device, &index_bytes);
        // A word no draw reads; at byte 4 `MTLDrawPrimitivesIndirectArguments` of 3 vertices from
        // vertex 3 (vertex count, instance count, vertex start, base instance); at byte 20 the
        // same of no vertices; at byte 36 `MTLDrawIndexedPrimitivesIndirectArguments` of 3
        // indices from index 3 with a base vertex of 3 (index count, instance count, index start,
        // base vertex, base instance). 56 bytes.
        let arguments = &shared(
            &device,
            &words(&[0xeeee_eeee, 3, 1, 3, 0, 0, 1, 0, 0, 3, 1, 3, 3, 0]),
        );
        let draw = move |offset: u64| {
            let draw = RenderCmd::DrawPrimitivesIndirect {
                primitive_type: TRIANGLE,
                indirect_buffer: arguments,
                indirect_buffer_offset: offset,
            };
            bound(state, triangles, draw)
        };
        assert_eq!(drawn(draw(4)), every(blue));
        assert_eq!(drawn(draw(20)), every(red));
        let indexed = RenderCmd::DrawIndexedPrimitivesIndirect {
            primitive_type: TRIANGLE,
            index_type: 0,
            index_buffer: indices,
            index_buffer_offset: 0,
            indirect_buffer: arguments,
            indirect_buffer_offset: 36,
        };
        assert_eq!(drawn(bound(state, triangles, indexed)), every(blue));
        assert_eq!(counted(metal), before.0);

        // An offset 2 bytes past a multiple of 4, and one 12 bytes before the buffer's end.
        assert_eq!(drawn(draw(2)), every(red));
        assert_eq!(drawn(draw(44)), every(red));
        assert_eq!(counted(metal), before.0 + 2);

        // Patch 0's control points all (0, 0), which draws no pixel; patch 1's a triangle over
        // the target. Factors of 1 (`half`, edges and inside) for every patch, Metal's default
        // step function reading the first patch's for each.
        let tess = function(&device, TESS, "render_tess");
        let mut desc = FullRenderPipelineDesc::new(&tess, Some(&color));
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 70,
                ..ColorAttachmentDesc::default()
            },
        )];
        let patches = device.new_render_pipeline_state_full(&desc);
        let patches = &patches.expect("a render state").expect("a render state");
        let mut point_bytes = vec![0u8; 24];
        for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)] {
            point_bytes.extend_from_slice(&x.to_le_bytes());
            point_bytes.extend_from_slice(&y.to_le_bytes());
        }
        let points = &shared(&device, &point_bytes);
        let factors = &shared(&device, &words(&[0x3c00_3c00, 0x3c00_3c00]));
        // A word no draw reads; at byte 4 `MTLDrawPatchIndirectArguments` of 1 patch from patch 1
        // (patch count, instance count, patch start, base instance); at byte 20 of no patches.
        let patch_arguments = &shared(&device, &words(&[0xeeee_eeee, 1, 1, 1, 0, 0, 1, 1, 0]));
        let patch_draw = move |offset: u64, indexed: bool| {
            let draw = RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points: 3,
                patch_index_buffer: indexed.then_some(indices),
                patch_index_buffer_offset: 0,
                indirect_buffer: patch_arguments,
                indirect_buffer_offset: offset,
            };
            let mut cmds = bound(patches, points, draw);
            cmds.insert(
                2,
                RenderCmd::SetTessellationFactorBuffer {
                    buffer: Some(factors),
                    offset: 0,
                    instance_stride: 0,
                },
            );
            cmds
        };
        assert_eq!(drawn(patch_draw(4, false)), every(blue));
        assert_eq!(drawn(patch_draw(20, false)), every(red));
        assert_eq!(drawn(patch_draw(4, true)), every(red));
        assert_eq!(counted(gap), before.1 + 1);
        assert_eq!(counted(metal), before.0 + 2);
        assert_eq!(counted(unbuilt), before.2);
    }

    /// A pass of a render target array length of 2 from slice 0 of a 2D array texture of 2
    /// slices clears and stores both slices, and a draw whose vertex function writes no
    /// `[[render_target_array_index]]` renders into slice 0 alone. The length is counted nowhere.
    /// A length of 2 from the texture's last slice, which leaves the pass no second slice, is
    /// Metal's refusal: nil, counted.
    #[test]
    fn a_layered_pass_renders_each_slice_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let done = CommandBufferStatus::COMPLETED;
        let (metal, unbuilt) = (RtRefusal::Refused.code(), Unbuilt::RenderPass.code());
        let before = (counted(metal), counted(unbuilt));
        // `MTLTextureType2DArray` of 2 slices, `RGBA8Unorm`, shader read and render target usage.
        let target = device.new_texture(
            on,
            &platform.texture_descriptor(&TextureDesc {
                texture_type: 3,
                array_length: 2,
                width: 4,
                height: 4,
                usage: 5,
                ..TextureDesc::default()
            }),
        );
        let target = target.expect("never a raise").expect("a render target");
        let texels = |slice: u32| {
            let mut texels = [0u8; 64];
            let region = rt::Region {
                origin: [0; 3],
                size: [4, 4, 1],
            };
            let read = target.rt().get_bytes(&mut texels, 16, 64, region, 0, slice);
            read.expect("the slice's texels");
            texels
        };
        let every = |texel: [u8; 4]| -> [u8; 64] { std::array::from_fn(|i| texel[i % 4]) };
        let (red, green, blue) = ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]);
        // A pass of `slice` of the target as colour attachment 0, cleared to `clear` and stored,
        // of render target array length `layers`.
        let layered = |slice: u64, clear: ClearColor, layers: u64| {
            let pass = RtObj::new(PassFields::default());
            let set = ColorAttachmentSet {
                texture: Some(Some(&target)),
                slice: Some(slice),
                load_action: Some(2),
                store_action: Some(1),
                clear_color: Some(clear),
                ..ColorAttachmentSet::default()
            };
            assert_eq!(pass.set_color_attachment(on, 0, &set), Ok(()));
            pass.set_pass(&PassSet {
                render_target_array_length: Some(layers),
                ..PassSet::default()
            });
            pass
        };
        let opaque = |red, green, blue| ClearColor {
            red,
            green,
            blue,
            alpha: 1.0,
        };

        // Slice 1 alone, cleared blue.
        assert_eq!(
            run(&made(), &layered(1, opaque(0.0, 0.0, 1.0), 0), Vec::new()),
            done
        );
        assert_eq!(texels(1), every(blue));

        // Both slices from slice 0, cleared red, and a green triangle into slice 0.
        let state = vdesc_state(&device, &function(&device, COLOR, "render_color"));
        let buffer = triangle(&device);
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
        assert_eq!(
            run(&made(), &layered(0, opaque(1.0, 0.0, 0.0), 2), draw),
            done
        );
        assert_eq!(texels(0), every(green));
        assert_eq!(texels(1), every(red));
        assert_eq!(counted(unbuilt), before.1);

        // Two slices from slice 1, the last.
        assert_eq!(counted(metal), before.0);
        let refused = made().render_encoder(on, &layered(1, opaque(1.0, 0.0, 0.0), 2));
        assert!(matches!(refused, Ok(None)));
        assert_eq!(counted(metal), before.0 + 1);
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

    /// The draw commands of `state` over the triangle at vertex buffer 0, with `argument` at
    /// fragment buffer 0.
    fn drawn<'a>(
        state: &'a RtObj<RtRenderPipeline>,
        vertices: &'a RtObj<RtBuffer>,
        argument: &'a RtObj<RtBuffer>,
    ) -> [RenderCmd<'a, RtPlatform>; 4] {
        [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(vertices),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(argument),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
        ]
    }

    /// A render encoder's `useResources:` reaches rt: once it declares a write of a texture with
    /// no storage element, a draw whose fragment function writes a texture beyond its slots
    /// (through an argument buffer) is rt's refusal, which draws nothing and is counted under
    /// `rt-refused-gap`, as the compute encoder's dispatch of a kernel that writes the same
    /// texture at a slot is, and exactly where that dispatch is. The same draw before the
    /// declaration, after a declaration for reading alone, and after a declared write of a
    /// texture with a storage element draws, and so does a draw of a function that reaches no
    /// texture beyond its slots after the declared write.
    #[test]
    fn a_draw_after_a_declared_write_of_a_texture_with_no_storage_element_is_refused_on_vulkan() {
        // `MTLResourceUsageRead`, `MTLResourceUsageWrite` and `MTLRenderStageFragment`.
        const READ: u64 = 1;
        const WRITE: u64 = 2;
        const FRAGMENT: u64 = 2;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let done = CommandBufferStatus::COMPLETED;
        let gap = RtRefusal::Gap.code();
        let texture = |desc: TextureDesc| {
            let made = device.new_texture(on, &platform.texture_descriptor(&desc));
            made.expect("never a raise").expect("a texture")
        };
        let writing = vdesc_state(&device, &function(&device, ARGWRITE, "render_argwrite"));
        let plain = vdesc_state(&device, &function(&device, COLOR, "render_color"));
        let tex_write = {
            let kernel = function(&device, TEXWRITE, "tex_write");
            let made = device.new_compute_pipeline_state_for_function(Some(&kernel));
            made.expect("a compute state").expect("a compute state")
        };
        let vertices = triangle(&device);
        let target = texture(TextureDesc {
            width: 4,
            height: 4,
            usage: 5,
            ..TextureDesc::default()
        });
        let drawn_into = || {
            let mut texels = [0u8; 64];
            target.read_pixels(&mut texels, 16);
            texels
        };
        let every = |texel: [u8; 4]| -> [u8; 64] { std::array::from_fn(|i| texel[i % 4]) };
        let (red, green) = (every([255, 0, 0, 255]), every([0, 255, 0, 255]));
        // A new 1 x 1 Shared `RGBA8Unorm` texture of shader-read and shader-write usage, which
        // reads zero, and a Shared argument buffer of its resource ID.
        let fresh = || {
            let written = texture(TextureDesc {
                usage: 3,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            let argument = device.new_buffer_of_length(on, 8, 0);
            let argument = argument.expect("never a raise").expect("a buffer");
            let id = written.gpu_resource_id().to_le_bytes();
            // SAFETY: the Shared buffer's contents are mapped, 8 bytes long.
            unsafe {
                let at = argument.contents().as_ptr::<u8>();
                at.copy_from_nonoverlapping(id.as_ptr(), id.len());
            }
            (written, argument)
        };
        let texel = |written: &RtObj<RtTexture>| {
            let mut texel = [0u8; 4];
            written.read_pixels(&mut texel, 4);
            texel
        };
        let magenta = [255, 0, 255, 255];
        // Whether the compute encoder's dispatch of `tex_write` with `t` at texture 0 is rt's
        // gap: a texture with no storage element.
        let stores_none = |t: &RtObj<RtTexture>| {
            let before = counted(gap);
            let commands = made();
            let encoder = commands.compute_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a compute encoder");
            let cmds = [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&tex_write),
                },
                ComputeCmd::SetTexture {
                    texture: Some(t),
                    index: 0,
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(4, 1, 1),
                    threads_per_threadgroup: MtlSize::new(4, 1, 1),
                },
                ComputeCmd::EndEncoding,
            ];
            for cmd in cmds {
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(commands.commit(on), Ok(()));
            assert_eq!(commands.wait_until_completed(on), Ok(()));
            assert_eq!(commands.status(), done);
            match counted(gap) - before {
                0 => false,
                1 => true,
                n => panic!("{n} gaps counted for one dispatch"),
            }
        };

        // `RGBA8Unorm`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and `A8Unorm`, 4 x 1, Private, of
        // shader-read and shader-write usage.
        let mut refused = Vec::new();
        for format in [70, 71, 81, 1] {
            let candidate = texture(TextureDesc {
                pixel_format: format,
                width: 4,
                usage: 3,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            let unstored = stores_none(&candidate);
            let declared = [Some(ResourceRef::Texture(&candidate))];

            // A declared write, then the draw: rt's gap where the texture has no storage
            // element, drawing nothing and writing nothing; drawn elsewhere.
            let (written, argument) = fresh();
            let declared_written = [Some(ResourceRef::Texture(&written))];
            let before = counted(gap);
            let mut cmds = vec![
                RenderCmd::UseResources {
                    resources: &declared_written,
                    usage: WRITE,
                },
                RenderCmd::UseResourcesStages {
                    resources: &declared,
                    usage: WRITE,
                    stages: FRAGMENT,
                },
            ];
            cmds.extend(drawn(&writing, &vertices, &argument));
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            assert_eq!(counted(gap), before + u64::from(unstored), "{format}");
            let (drawn_texels, written_texel) = match unstored {
                true => (red, [0; 4]),
                false => (green, magenta),
            };
            assert_eq!(drawn_into(), drawn_texels, "{format}");
            assert_eq!(texel(&written), written_texel, "{format}");

            // Declared for reading alone, beside a buffer: the draw draws.
            let (written, argument) = fresh();
            let declared_written = [Some(ResourceRef::Texture(&written))];
            let read = [
                Some(ResourceRef::Buffer(&argument)),
                Some(ResourceRef::Texture(&candidate)),
            ];
            let before = counted(gap);
            let mut cmds = vec![
                RenderCmd::UseResources {
                    resources: &declared_written,
                    usage: WRITE,
                },
                RenderCmd::UseResources {
                    resources: &read,
                    usage: READ,
                },
            ];
            cmds.extend(drawn(&writing, &vertices, &argument));
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            assert_eq!(counted(gap), before, "{format}");
            assert_eq!(drawn_into(), green, "{format}");
            assert_eq!(texel(&written), magenta, "{format}");

            // The draw before a declared write draws; the same draw after it is rt's gap where
            // the texture has no storage element.
            let (written, argument) = fresh();
            let declared_written = [Some(ResourceRef::Texture(&written))];
            let before = counted(gap);
            let mut cmds = vec![RenderCmd::UseResources {
                resources: &declared_written,
                usage: WRITE,
            }];
            cmds.extend(drawn(&writing, &vertices, &argument));
            cmds.push(RenderCmd::UseResources {
                resources: &declared,
                usage: WRITE,
            });
            cmds.push(RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            });
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            assert_eq!(counted(gap), before + u64::from(unstored), "{format}");
            assert_eq!(drawn_into(), green, "{format}");
            assert_eq!(texel(&written), magenta, "{format}");

            // After a declared write, a draw of a function that reaches no texture beyond its
            // slots draws.
            let before = counted(gap);
            let mut cmds = vec![RenderCmd::UseResources {
                resources: &declared,
                usage: WRITE,
            }];
            cmds.extend(drawn(&plain, &vertices, &argument));
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            assert_eq!(counted(gap), before, "{format}");
            assert_eq!(drawn_into(), green, "{format}");
            if unstored {
                refused.push(format);
            }
        }
        eprintln!("the pixel formats a declared write refused draws after: {refused:?}");
    }

    /// `MTLIndirectCommandTypeDrawIndexed`.
    const DRAW_INDEXED: u64 = 2;

    /// A descriptor of indexed draws that inherit the pipeline state and bind `vertex` vertex
    /// buffers and one fragment buffer each.
    fn icb_desc(vertex: u64) -> IcbDesc {
        IcbDesc {
            command_types: DRAW_INDEXED,
            inherit_pipeline_state: true,
            max_vertex_buffer_bind_count: vertex,
            max_fragment_buffer_bind_count: 1,
            ..IcbDesc::default()
        }
    }

    /// `device`'s `maxIndirectCommandsTokenCount`, rt's floor's reading of it; `None` on a device
    /// without `VK_EXT_device_generated_commands`.
    fn tokens(device: &RtObj<RtDevice>) -> Option<u32> {
        device.facts().indirect_tokens
    }

    /// The state of `vertex` and `icb_fragment` on `device` into one `RGBA8Unorm` colour
    /// attachment, made for indirect command buffers where `flagged`.
    fn icb_state(device: &RtObj<RtDevice>, vertex: &str, flagged: bool) -> RtObj<RtRenderPipeline> {
        let vertex = function(device, ICB, vertex);
        let fragment = function(device, ICB, "icb_fragment");
        let mut desc = FullRenderPipelineDesc::new(&vertex, Some(&fragment));
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 70,
                ..ColorAttachmentDesc::default()
            },
        )];
        desc.support_indirect_command_buffers = flagged;
        let state = device.new_render_pipeline_state_full(&desc);
        state.expect("a render state").expect("a render state")
    }

    /// The buffers a kernel of [`ICB`] encodes commands of: for each of `colours`, a triangle over
    /// the whole of a target, (-1, -1), (3, -1), (-1, 3), as three `float4` positions, its three
    /// 16-bit indices and its colour as a `float4`.
    struct IcbScene {
        positions: RtObj<RtBuffer>,
        colours: RtObj<RtBuffer>,
        indices: RtObj<RtBuffer>,
    }

    impl IcbScene {
        fn new(device: &RtObj<RtDevice>, colours: &[[f32; 4]]) -> IcbScene {
            let floats = |values: &[f32]| -> Vec<u8> {
                values.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            let corner = [
                [-1.0f32, -1.0, 0.0, 1.0],
                [3.0, -1.0, 0.0, 1.0],
                [-1.0, 3.0, 0.0, 1.0],
            ];
            let positions: Vec<f32> = (colours.iter())
                .flat_map(|_| corner.iter().flatten().copied())
                .collect();
            let indices: Vec<u8> = (0..3 * colours.len())
                .flat_map(|i| u16::try_from(i).expect("a 16-bit index").to_le_bytes())
                .collect();
            let colours: Vec<f32> = colours.iter().flatten().copied().collect();
            IcbScene {
                positions: shared(device, &floats(&positions)),
                colours: shared(device, &floats(&colours)),
                indices: shared(device, &indices),
            }
        }

        /// `count` threads of `kernel` encode commands 0 to `count` of `icb`, declared written,
        /// in a command buffer `made` that completes.
        fn encode(
            &self,
            (device, made): (&RtObj<RtDevice>, &RtObj<RtCommandBuffer>),
            kernel: &RtObj<RtComputePipeline>,
            icb: &RtObj<RtIcb>,
            count: u64,
        ) {
            let on = OnFault::Return;
            let arguments = shared(device, &icb.gpu_resource_id().to_le_bytes());
            let encoder = made.compute_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a compute encoder");
            let buffers = [
                Some(&arguments),
                Some(&self.positions),
                Some(&self.colours),
                Some(&self.indices),
            ];
            let declared = [Some(ResourceRef::Icb(icb))];
            let cmds = [
                ComputeCmd::SetComputePipelineState {
                    state: Some(kernel),
                },
                ComputeCmd::SetBuffers {
                    buffers: &buffers,
                    offsets: &[0; 4],
                    range: NsRange::new(0, 4),
                },
                ComputeCmd::UseResources {
                    resources: &declared,
                    usage: 2,
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(count, 1, 1),
                    threads_per_threadgroup: MtlSize::new(count, 1, 1),
                },
                ComputeCmd::EndEncoding,
            ];
            for cmd in cmds {
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(made.commit(on), Ok(()));
            assert_eq!(made.wait_until_completed(on), Ok(()));
            assert_eq!(made.status(), CommandBufferStatus::COMPLETED);
        }
    }

    /// The commands that set `state` and execute commands `location` to `location + length` of
    /// `icb`.
    fn executed<'a>(
        state: &'a RtObj<RtRenderPipeline>,
        icb: &'a RtObj<RtIcb>,
        (location, length): (u64, u64),
    ) -> Vec<RenderCmd<'a, RtPlatform>> {
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: NsRange::new(location, length),
            },
        ]
    }

    /// Indirect command buffers are rt's (`super::super::icb`): a buffer of indexed draws that
    /// inherit the pipeline state is made with Metal's answers, a kernel encodes its commands
    /// through its `gpuResourceID`, and an execution of a range the CPU gives or a buffer holds
    /// draws those commands with the encoder's pipeline state, the last drawn on top; an empty
    /// range draws nothing. A blit's reset of a command leaves it drawing nothing until a kernel
    /// encodes it again, and an optimize and a synchronize record nothing. A range past the
    /// commands, a range buffer's offset off a multiple of 4 or without 8 bytes after it, and an
    /// execution with no pipeline state set or in a compute encoder are Metal's refusals; a state
    /// made without `supportIndirectCommandBuffers` and a buffer of other commands, of commands
    /// that set their own pipeline state, of 32 vertex buffers or of no commands are rt's gaps;
    /// the CPU's encodes, reset and copies and a buffer of a resource index are unbuilt; each is
    /// counted and draws or makes nothing. On a device without
    /// `VK_EXT_device_generated_commands` the make answers nil, rt's gap, which the test prints
    /// and stops at.
    #[test]
    fn indirect_command_buffers_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let done = CommandBufferStatus::COMPLETED;
        let (metal, gap) = (RtRefusal::Refused.code(), RtRefusal::Gap.code());
        let unbuilt = Unbuilt::Icb.code();
        let before = (counted(metal), counted(gap), counted(unbuilt));
        let make = |desc: &IcbDesc, count: u64| {
            let made = device.new_indirect_command_buffer(desc, count, 0);
            made.expect("never a raise")
        };
        let Some(icb) = make(&icb_desc(1), 2) else {
            assert_eq!(
                tokens(&device),
                None,
                "a device of the extension made no buffer"
            );
            assert_eq!(counted(gap), before.1 + 1);
            println!("no VK_EXT_device_generated_commands: the make answers nil, rt's gap");
            return;
        };
        assert_eq!(icb.size(), 2);
        assert_ne!(icb.gpu_resource_id(), 0);
        assert_eq!(icb.resource_index(), 0);
        let other = make(&icb_desc(1), 1).expect("a second buffer");
        assert!(other.unique_identifier() > icb.unique_identifier());

        // Draws of no index buffer, commands that set their own pipeline state, 32 vertex
        // buffers and no commands: rt's gaps. A resource index: unbuilt.
        let drawing = IcbDesc {
            command_types: 1,
            ..icb_desc(1)
        };
        let own_states = IcbDesc {
            inherit_pipeline_state: false,
            ..icb_desc(1)
        };
        let indexed = IcbDesc {
            resource_index: Some(0),
            ..icb_desc(1)
        };
        for (desc, count) in [(&drawing, 2), (&own_states, 2), (&icb_desc(32), 2)] {
            assert!(make(desc, count).is_none());
        }
        assert!(make(&icb_desc(1), 0).is_none());
        assert!(make(&indexed, 2).is_none());
        assert_eq!(counted(gap), before.1 + 4);
        assert_eq!(counted(unbuilt), before.2 + 1);

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
        // The texels of a pass of its own that clears the target to opaque red, then `cmds`.
        let drawn = |cmds| {
            assert_eq!(run(&made(), &pass(&target, 2), cmds), done);
            let mut texels = [0u8; 64];
            target.read_pixels(&mut texels, 16);
            texels
        };
        let every = |texel: [u8; 4]| -> [u8; 64] { std::array::from_fn(|i| texel[i % 4]) };
        let (red, green, blue) = ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]);

        // Command 0 a green triangle over the target, command 1 a blue one.
        let scene = IcbScene::new(&device, &[[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
        let kernel = function(&device, ICB, "icb_encode");
        let kernel = device.new_compute_pipeline_state_for_function(Some(&kernel));
        let kernel = kernel.expect("a compute state").expect("a compute state");
        scene.encode((&device, &made()), &kernel, &icb, 2);
        let state = &icb_state(&device, "icb_vertex", true);
        assert_eq!(drawn(executed(state, &icb, (0, 2))), every(blue));
        assert_eq!(drawn(executed(state, &icb, (0, 1))), every(green));
        assert_eq!(drawn(executed(state, &icb, (1, 1))), every(blue));
        assert_eq!(drawn(executed(state, &icb, (0, 0))), every(red));
        // A word no execution reads, then the ranges (0, 1) at byte 4 and (1, 1) at byte 12.
        let ranges = &shared(&device, &words(&[0xeeee_eeee, 0, 1, 1, 1]));
        let ranged = |offset: u64| {
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                RenderCmd::ExecuteCommandsIndirect {
                    indirect_commandbuffer: &icb,
                    indirect_range_buffer: ranges,
                    indirect_buffer_offset: offset,
                },
            ]
        };
        assert_eq!(drawn(ranged(4)), every(green));
        assert_eq!(drawn(ranged(12)), every(blue));
        assert_eq!(counted(metal), before.0);
        assert_eq!(counted(gap), before.1 + 4);

        // Metal's refusals: a range past the commands and one past 32 bits, a range buffer's
        // offset off a multiple of 4 and one 4 bytes before its end, no pipeline state set.
        assert_eq!(drawn(executed(state, &icb, (1, 2))), every(red));
        assert_eq!(drawn(executed(state, &icb, (u64::MAX, 2))), every(red));
        assert_eq!(drawn(ranged(2)), every(red));
        assert_eq!(drawn(ranged(16)), every(red));
        let unset = executed(state, &icb, (0, 2)).split_off(1);
        assert_eq!(drawn(unset), every(red));
        assert_eq!(counted(metal), before.0 + 5);
        // rt's gap: a state made without `supportIndirectCommandBuffers`.
        let unflagged = &icb_state(&device, "icb_vertex", false);
        assert_eq!(drawn(executed(unflagged, &icb, (0, 2))), every(red));
        assert_eq!(counted(gap), before.1 + 5);

        // A blit's reset of command 1, an optimize and a synchronize: command 0 draws alone.
        let blit = |cmds: Vec<BlitCmd<'_, RtPlatform>>| {
            let commands = made();
            let encoder = commands.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            for cmd in cmds {
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(commands.commit(on), Ok(()));
            assert_eq!(commands.wait_until_completed(on), Ok(()));
            assert_eq!(commands.status(), done);
        };
        blit(vec![
            BlitCmd::ResetCommandsInBuffer {
                buffer: &icb,
                range: NsRange::new(1, 1),
            },
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: &icb,
                range: NsRange::new(0, 2),
            },
            BlitCmd::SynchronizeResource {
                resource: Some(ResourceRef::Icb(&icb)),
            },
        ]);
        assert_eq!(drawn(executed(state, &icb, (0, 2))), every(green));
        assert_eq!(drawn(executed(state, &icb, (1, 1))), every(red));
        // A reset past the commands: Metal's refusal. A copy between buffers: unbuilt.
        blit(vec![
            BlitCmd::ResetCommandsInBuffer {
                buffer: &icb,
                range: NsRange::new(1, 2),
            },
            BlitCmd::CopyIndirectCommandBuffer {
                source: &icb,
                source_range: NsRange::new(0, 1),
                destination: &icb,
                destination_index: 1,
            },
        ]);
        assert_eq!(counted(metal), before.0 + 6);
        assert_eq!(counted(unbuilt), before.2 + 2);
        // The kernel encodes command 1 again.
        scene.encode((&device, &made()), &kernel, &icb, 2);
        assert_eq!(drawn(executed(state, &icb, (0, 2))), every(blue));

        // The CPU's encodes and reset: unbuilt.
        assert!(matches!(icb.render_command(0), Ok(None)));
        assert!(matches!(icb.compute_command(0), Ok(None)));
        assert_eq!(icb.reset_with_range(0, 2), Ok(()));
        assert_eq!(counted(unbuilt), before.2 + 5);
        assert_eq!(drawn(executed(state, &icb, (0, 2))), every(blue));

        // An execution in a compute encoder: Metal's refusal.
        let commands = made();
        let encoder = commands.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        let execution = ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(0, 2),
        };
        assert_eq!(encoder.encode(on, execution), Ok(()));
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(commands.status(), done);
        assert_eq!(counted(metal), before.0 + 7);
        assert_eq!(counted(gap), before.1 + 5);
    }

    /// The most vertex buffer slots of a command that rt's push range holds: a 25th vertex slot
    /// falls past the state's 256 push bytes into its parameter block. The bound is on the vertex
    /// slots, not on a total: measured on an RTX 3090 and a Radeon RX 7900 XTX with a fragment
    /// function reading one slot more, and on the 3090 a state reading 25 vertex slots below the
    /// buffer's bind count is refused with a fragment bind count of 0 as of 1.
    const PUSHED_VERTEX_SLOTS: u32 = 24;

    /// An execution carries each buffer slot its state's functions read below the buffer's bind
    /// counts as a slot word of rt's push range, and takes a token of the device's for each and
    /// three more: one of a state of more slots than the push range holds is rt's gap
    /// (`IndirectCommandBuffers` of `DrawInput`), and one of more than the device's
    /// `maxIndirectCommandsTokenCount` leaves is too (of `Tokens`); each draws nothing and is
    /// counted. Commands of 24 to 30 slots (`icb_vertex_23` to `icb_vertex_29`, and the fragment
    /// function's one), each executed on its own: those whose vertex function reads at most
    /// [`PUSHED_VERTEX_SLOTS`] draw where the device's tokens take them.
    #[test]
    fn an_execution_of_more_slots_than_rt_carries_draws_nothing_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let Some(tokens) = tokens(&device) else {
            println!("no VK_EXT_device_generated_commands: no execution to measure");
            return;
        };
        let gap = RtRefusal::Gap.code();
        let made_icb = device.new_indirect_command_buffer(&icb_desc(29), 1, 0);
        let icb = made_icb
            .expect("never a raise")
            .expect("a buffer of 29 vertex buffers");
        let scene = IcbScene::new(&device, &[[0.0, 1.0, 0.0, 1.0]]);
        let kernel = function(&device, ICB, "icb_encode_wide");
        let kernel = device.new_compute_pipeline_state_for_function(Some(&kernel));
        let kernel = kernel.expect("a compute state").expect("a compute state");
        scene.encode((&device, &made()), &kernel, &icb, 1);
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
        let kinds = ["DrawInput", "Tokens"].map(|kind| format!("IndirectCommandBuffers-{kind}"));
        let named = || (kinds.each_ref()).map(|name| crate::rt_platform::diag::gaps_named(name));
        let (mut seen, mut wanted) = (Vec::new(), Vec::new());
        for vertex in 23..=29u32 {
            let slots = vertex + 1;
            let state = icb_state(&device, &format!("icb_vertex_{vertex}"), true);
            let before = (counted(gap), named());
            let cmds = executed(&state, &icb, (0, 1));
            assert_eq!(
                run(&made(), &pass(&target, 2), cmds),
                CommandBufferStatus::COMPLETED
            );
            let mut texels = [0u8; 64];
            target.read_pixels(&mut texels, 16);
            let after = named();
            let drawn = texels.chunks(4).all(|texel| texel == [0, 255, 0, 255]);
            let red = texels.chunks(4).all(|texel| texel == [255, 0, 0, 255]);
            let kinds = [after[0] - before.1[0], after[1] - before.1[1]];
            seen.push((slots, drawn, red, counted(gap) - before.0, kinds));
            let pushed = vertex <= PUSHED_VERTEX_SLOTS;
            let taken = slots + 3 <= tokens;
            let kinds = [u64::from(!pushed), u64::from(pushed && !taken)];
            let fits = pushed && taken;
            wanted.push((slots, fits, !fits, u64::from(!fits), kinds));
        }
        // Each: slots, drawn, nothing drawn, gaps, and gaps of DrawInput and of Tokens.
        println!("{tokens} tokens: {seen:?}");
        assert_eq!(seen, wanted, "{tokens} tokens");
    }
}
