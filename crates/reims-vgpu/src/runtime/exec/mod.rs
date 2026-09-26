//! CmdExecIndirect2: load streams, multi-attachment clears, Metal draw attempt.
//!
//! Clear-only passes write guest mapping pages (archive render_clear).
//! Draws try Metal encode when pipeline MTLBs resolve; otherwise color targets
//! are still marked dirty for DisplaySwap.

// This rail's preflight, named rather than re-exported flat — reached only
// through `Backend`, so this module still names no rail.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

// The backend the process executes on, reached only through the trait: this
// module names no rail.
use crate::backend::Backend as _;
use crate::model::DeviceState;
use crate::protocol::draw::DrawArgs;
use crate::protocol::endian::{ld32, ld64};
use crate::protocol::fifo::{decode_exec_resource_table, ExecResourceDesc};
use crate::protocol::fifo::{
    CHILD_EXEC_INDIRECT_CMDBUF_COUNT, CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN,
    CHILD_EXEC_INDIRECT_CMDBUF_GVA, CHILD_EXEC_INDIRECT_CMDBUF_LENGTH,
    CHILD_EXEC_INDIRECT_HEADER_LEN, CHILD_EXEC_INDIRECT_RESOURCE_COUNT,
    CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN, CHILD_EXEC_INDIRECT_TASK_ID,
};
use crate::protocol::pixel_format::{self, ClearImageEncoding};
use crate::runtime::blit_exec::{self, BlitStatus};
use crate::runtime::compute_exec::{self, ComputeStatus};
use crate::runtime::decode::blit_spi;
use crate::runtime::decode::compute_spi::{self, Kind as ComputeKind};
use crate::runtime::decode::render_spi::{self, Kind as SpiKind};
use crate::runtime::draw::{
    self, BindTable, BufferBind, EncodeStatus, IndexedDrawInfo, SamplerBind, TextureBind,
    MAX_BUFFER_BIND_SLOTS, MAX_SAMPLER_BIND_SLOTS, MAX_TEXTURE_BIND_SLOTS,
};
use crate::runtime::fence_exec;
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapping_write;
use crate::runtime::mipmap::{self, MipmapStatus};
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};
use crate::runtime::render_pass::{
    self, attachment_subresource_is_bindable, color_attachment_subresource_is_bindable,
    ColorAttachment, DepthAttachment, LevelSupport, ScissorRect, StencilAttachment,
};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};
use reims_vgpu_core::operation::{OperationClass, OperationHome};
use reims_vgpu_protocol::closure::Rail;
use reims_vgpu_protocol::decode::blit::BlitRecord;
use reims_vgpu_protocol::decode::sync::SyncRecord;
use reims_vgpu_protocol::pass_action::{
    store_action_publishes_single_sample, MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_LOAD,
    MTL_STORE_ACTION_MULTISAMPLE_RESOLVE, MTL_STORE_ACTION_STORE,
    MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
};
use reims_vgpu_protocol::render::{RenderKind as ProtoRenderKind, ShaderStage};
use reims_vgpu_protocol::resource_state::ContentDirective;
use reims_vgpu_protocol::segment::{SegmentBody, SegmentKind, SegmentLifetime, SegmentStream};
use reims_vgpu_wire::ops::blit as wire_blit;
use reims_vgpu_wire::ops::render as wire_render;
use reims_vgpu_wire::ops::render_pass as wire_pass;
use reims_vgpu_wire::ops::tile as wire_tile;
use std::sync::Arc;

/// Pending render-pass ICB execute (range form or indirect range buffer).
#[derive(Clone, Debug, Default)]
struct RenderIcbExecute {
    icb_ref: u32,
    is_range: bool,
    range_location: u64,
    range_length: u64,
    args_buffer_ref: u32,
    args_buffer_offset: u64,
}

/// One draw recorded with the bind state at that point (archive DrawRec / multi-draw job).
///
/// Archive `apple_pv_gpu_render_worker_run` executes **every** draw in order,
/// seeding draw N from draw N-1's writeback. Product previously kept only
/// `last_draw`, which dropped the logo when the pill was the final draw in the
/// same stream (journal: logo RG8 168×206 + pill → one mapper-ref-texture FB).
#[derive(Clone, Debug, Default)]
struct PendingDraw {
    pipeline_ref: u32,
    draw: DrawArgs,
    indexed: Option<IndexedDrawInfo>,
    vertex_buffers: BindTable<BufferBind>,
    fragment_buffers: BindTable<BufferBind>,
    vertex_textures: BindTable<TextureBind>,
    fragment_textures: BindTable<TextureBind>,
    vertex_samplers: BindTable<SamplerBind>,
    fragment_samplers: BindTable<SamplerBind>,
    /// Every viewport this draw was recorded with, in the guest's order. Empty
    /// means the stream bound none and the backend's full-target default
    /// stands — what `None` used to mean, at a capacity of one.
    viewports: Vec<[f64; 6]>,
    /// Every scissor rect this draw was recorded with. See [`Self::viewports`].
    scissors: Vec<ScissorRect>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    /// `setTriangleFillMode:` — `MTLTriangleFillMode`, `None` where the stream
    /// bound none and the Metal default (fill) applies.
    fill_mode: Option<u32>,
    /// `setDepthClipMode:` — `MTLDepthClipMode`, `None` for the Metal default
    /// (clip).
    depth_clip_mode: Option<u32>,
    /// `setLineWidth:` — the width the stream last set, `None` where it set
    /// none and the Metal default (1.0) applies.
    line_width: Option<f32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
    /// The occlusion query armed when this draw was recorded, snapshotted from
    /// [`StreamAccum::visibility`]. `None` is the Metal default,
    /// `MTLVisibilityResultModeDisabled`.
    visibility: Option<draw::VisibilityArming>,
}

#[derive(Clone, Debug, Default)]
struct StreamAccum {
    pipeline_ref: u32,
    /// Every colour attachment whose `load_action` is `Clear`, in stream order.
    ///
    /// Membership is the **load** action alone, because this is the pass's
    /// CLEAR seed and `MTLLoadActionClear` means the attachment starts at the
    /// record's clear value whatever becomes of it afterwards. Use
    /// [`StreamAccum::clears_reaching_guest_pages`] — not this — wherever the
    /// clear colour would be written into the guest's own pages.
    clears: Vec<ColorAttachment>,
    /// Color targets as (pass slot index, attachment). Slot maps to Metal color(i).
    color_slots: Vec<(u32, ColorAttachment)>,
    color_targets: Vec<u32>,
    /// All draws in stream order (archive multi-draw job).
    draws: Vec<PendingDraw>,
    saw_draw: bool,
    /// Every render ICB execute (`0x14`/`0x15`) in this stream, in stream
    /// order.
    ///
    /// A list rather than a latch because `executeCommandsInBuffer:` is work,
    /// not state: a second record does not replace the first, it asks for a
    /// second execution. See the loop that drains this in [`finish_stream`] for
    /// what a capacity of one used to cost.
    execute_icb: Vec<RenderIcbExecute>,
    vertex_buffers: BindTable<BufferBind>,
    fragment_buffers: BindTable<BufferBind>,
    vertex_textures: BindTable<TextureBind>,
    fragment_textures: BindTable<TextureBind>,
    vertex_samplers: BindTable<SamplerBind>,
    fragment_samplers: BindTable<SamplerBind>,
    /// Every viewport the stream bound, in the guest's order. Empty means the
    /// stream bound none and the backend's full-target default stands.
    viewports: Vec<[f64; 6]>,
    /// Every scissor rect the stream bound, in the guest's order.
    scissors: Vec<ScissorRect>,
    indexed: Option<IndexedDrawInfo>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    /// `setTriangleFillMode:` — `MTLTriangleFillMode`, `None` where the stream
    /// bound none and the Metal default (fill) applies.
    fill_mode: Option<u32>,
    /// `setDepthClipMode:` — `MTLDepthClipMode`, `None` for the Metal default
    /// (clip).
    depth_clip_mode: Option<u32>,
    /// `setLineWidth:` — the width the stream last set, `None` where it set
    /// none and the Metal default (1.0) applies.
    line_width: Option<f32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
    /// Serializer ref of the pass's `visibilityResultBuffer`, `0` for a pass
    /// that named none.
    ///
    /// A *pass* property, set once by the `RenderPass` arm, where
    /// [`Self::visibility`] beside it is encoder state each `0x84` replaces.
    /// Both are needed to write anything: the mode says what to count and this
    /// says where the guest will read it.
    visibility_buffer_ref: u32,
    /// The occlusion query currently armed, replaced by each
    /// `setVisibilityResultMode:offset:`.
    ///
    /// Encoder state, so one slot is the contract rather than a bound: a second
    /// record genuinely replaces the first. What *accumulates* across draws is
    /// the count in the guest's buffer, not the arming.
    visibility: Option<draw::VisibilityArming>,
    /// Draw records this stream decoded but did not keep because no pipeline was
    /// latched. See [`StreamDrawDrop`]; reported once per stream by
    /// [`note_stream_draw_drops`].
    dropped_no_pipeline: u32,
    /// Draw records this stream decoded but did not keep because they asked for
    /// zero vertices.
    ///
    /// Split from [`Self::dropped_no_pipeline`] because the two are opposite
    /// findings that were folded into one number for as long as the emitter has
    /// existed: a zero count is a **legal empty draw** and nothing is lost, while
    /// an unlatched pipeline is a **draw the guest asked for and this device
    /// dropped**. [`StreamDrawDrop::Unbound`]'s own doc said to read the rate to
    /// tell them apart, which cannot work when both increment the same field —
    /// a workload emitting thousands of legal empty draws reads identically to
    /// one losing thousands of real ones.
    dropped_zero_count: u32,
    /// Something the guest asked this stream for that its state cannot carry.
    ///
    /// Every arm that sets this used to note its loss and carry on, and all of
    /// them cost the same thing: the pass ran, the guest was told nothing, and
    /// the pixels are not the ones it asked for. See [`StreamRefusal`] for what
    /// each arm loses and why none of them can be told apart downstream.
    ///
    /// Recording it lets [`StreamAccum::bind_snapshot`] refuse. That is the
    /// funnel both consumers of the stream's state pass through — a decoded draw
    /// and an end-of-stream ICB execute — which is why the refusal lives there
    /// and not in either backend's encoder.
    ///
    /// **Sticky, and it cannot go stale.** There is no retirement path and none
    /// is needed: this field describes the accumulator beside it, a
    /// `StreamAccum` is built fresh per stream and dropped at [`finish_stream`],
    /// so the refusal and the state it describes have exactly the same life. The
    /// compute rail's equivalent needs a `clear_refusal_at` because a
    /// `ComputeAccum` outlives many dispatches; a render pass's state does not
    /// outlive the pass.
    unrepresentable: Option<StreamRefusal>,
}

impl StreamAccum {
    /// The subset of [`Self::clears`] whose colour the guest may read back, so
    /// writing it into the guest's pages is publishing the pass's result rather
    /// than inventing one.
    ///
    /// `MTLStoreActionDontCare` says the pass's result for that attachment is
    /// dropped. Landing the clear colour in guest memory anyway would be this
    /// device deciding what the guest sees where the guest said it does not
    /// care — a content invention, and the exact thing the seed list must not
    /// be used for.
    ///
    /// One method rather than the predicate written at each `apply_clear` loop,
    /// because there are two of them — the clear-only stream and the draw-failure
    /// fallback — and they have to agree about what "the guest can read this"
    /// means.
    fn clears_reaching_guest_pages(&self) -> impl Iterator<Item = &ColorAttachment> {
        self.clears
            .iter()
            .filter(|att| store_action_publishes_single_sample(att.store_action))
    }

    /// The stream's bind state as a `PendingDraw`, or what makes it
    /// unrepresentable.
    ///
    /// Two things need it and must not disagree: a decoded draw, which fills
    /// in `pipeline_ref` and `draw` on top, and an ICB execute, which inherits
    /// the state as it stands at end of stream and supplies neither. Both must
    /// also refuse on the same terms, which is why the check is here rather than
    /// at either of them: a snapshot of state that is missing something the
    /// guest asked for is not this stream's state, and a draw encoded from it
    /// computes the wrong pixels with nothing to say so.
    ///
    /// Draws recorded *before* the refusal are untouched. They snapshotted state
    /// that was still complete, so they are the guest's own work and they stand;
    /// only the ones that would read the gap are refused.
    fn bind_snapshot(&self) -> Result<PendingDraw, StreamRefusal> {
        if let Some(refused) = self.unrepresentable {
            return Err(refused);
        }
        Ok(PendingDraw {
            indexed: self.indexed.clone(),
            vertex_buffers: self.vertex_buffers.clone(),
            fragment_buffers: self.fragment_buffers.clone(),
            vertex_textures: self.vertex_textures.clone(),
            fragment_textures: self.fragment_textures.clone(),
            vertex_samplers: self.vertex_samplers.clone(),
            fragment_samplers: self.fragment_samplers.clone(),
            viewports: self.viewports.clone(),
            scissors: self.scissors.clone(),
            blend_color: self.blend_color,
            cull_mode: self.cull_mode,
            front_facing: self.front_facing,
            fill_mode: self.fill_mode,
            depth_clip_mode: self.depth_clip_mode,
            line_width: self.line_width,
            depth_bias: self.depth_bias,
            depth_stencil_ref: self.depth_stencil_ref,
            stencil_ref: self.stencil_ref,
            depth_attach: self.depth_attach,
            stencil_attach: self.stencil_attach,
            visibility: self.visibility,
            ..Default::default()
        })
    }
}

/// Why a decoded `RenderKind::Draw` record never became a `PendingDraw`.
///
/// A serialized Metal render stream is one render pass, and every draw in it
/// contributes to one attachment set. Dropping any of them leaves the pixels
/// that draw would have written as whatever the earlier records put there —
/// which, for a compositor doing per-element damage draws, is a rectangle of
/// the target holding the wrong picture and holding it until the next full
/// redraw.
///
/// There used to be a second arm here: a `MAX_DRAWS_PER_STREAM = 64` ceiling
/// that truncated the list inside a bare `if` with no `else`. It is gone. The
/// number was this crate's, not the protocol's — its comment named an archive
/// environment variable rather than a wire field — and a live boot found streams
/// pressing right against it (8013 streams at 33–63 draws, two truncated, one
/// losing four draws). What bounds the list now is the stream itself: a draw
/// record has a minimum encoded length, so the record count cannot exceed the
/// stream bytes this crate already holds in memory, and [`BindTable`] keeps the
/// per-record cost at one pointer per stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamDrawDrop {
    /// The record arrived with no pipeline bound: a `SetPipeline` this decoder
    /// failed to latch, and therefore a **lost draw**.
    ///
    /// This used to also carry the zero-primitive-count case, which is a legal
    /// empty draw that loses nothing, and told the reader to separate the two by
    /// the rate. That could not work — both incremented one field — so the two
    /// now count apart at the check and only this one is a loss.
    Unbound { dropped: u32 },
    /// A depth or stencil attachment this device cannot honour as decoded.
    ///
    /// The pass still runs, and it runs *without* the attachment — so depth or
    /// stencil testing silently disappears for every draw in it, which shows up
    /// as wrong occlusion rather than as a missing frame. Both conditions are
    /// real Metal that this device does not implement: a non-zero `level` binds
    /// a mip of the depth texture, and a non-zero `resolve_texture_ref` is a
    /// multisample depth resolve. Naming them is what separates "the guest
    /// never asked" from "the guest asked and we dropped it".
    DepthStencilUnsupported {
        aspect: &'static str,
        level: u32,
        slice: u32,
        depth_plane: u32,
        resolve_texture_ref: u32,
    },
    /// A colour attachment naming a subresource this device renders past.
    ///
    /// The same shape as [`Self::DepthStencilUnsupported`] and it was invisible
    /// for longer, because the fields did not exist: `level`, `slice` and
    /// `depth_plane` are three sixteen-bit fields of the pass record and this
    /// device read only the first of them, thirty-two bits wide, so a slice
    /// arrived folded into the level and a depth plane was never decoded at all.
    ///
    /// The pass still runs, into **level 0, slice 0, plane 0** of the named
    /// texture. So a guest rendering a cube face, a texture-array layer or a mip
    /// gets its work — into the wrong subresource, overwriting face 0 every
    /// time. That is wrong pixels rather than missing ones, which is why it is
    /// fail-visible: nothing downstream can tell it happened.
    ///
    /// `resolve_texture_ref` is the fourth shape and it was the last to be
    /// tested. [`Self::DepthStencilUnsupported`] carried it from the start; this
    /// arm did not, so a multisample colour pass — attachment texture
    /// multisampled, `storeAction = MultisampleResolve`, `resolveTexture` naming
    /// where the single-sampled result goes — was admitted, rendered at one
    /// sample into the attachment, and its resolve target left holding whatever
    /// it held before. The guest reads the resolve target.
    ColorSubresourceUnsupported {
        slot: u32,
        level: u32,
        slice: u32,
        depth_plane: u32,
        resolve_texture_ref: u32,
    },
    /// A pass declaring more render-target array layers than this device draws.
    ///
    /// Layered rendering: one draw is broadcast to the layers its vertex stage
    /// selects with `[[render_target_array_index]]`, and this device binds the
    /// attachment whole and draws into layer 0. So it is
    /// [`Self::ColorSubresourceUnsupported`] again with the coordinate chosen
    /// per draw instead of per pass — geometry meant for layer 3 lands on top
    /// of layer 0's content, and layers 1..n keep whatever they held through a
    /// `Clear` the guest asked to apply to all of them.
    ///
    /// It counted rather than refused for as long as the two arms beside it did,
    /// on an argument they no longer make: rendering it anyway is wrong content
    /// written over right content in a layer the pass did not name, and nothing
    /// downstream can tell, because a pass that touched only layer 0 is exactly
    /// what a guest that asked for one layer also produces.
    PassArrayLengthUnsupported { length: u64 },
    /// A pass declaring a default raster sample count this device cannot
    /// rasterize at.
    ///
    /// `defaultRasterSampleCount` is how many fragments the rasterizer produces
    /// per pixel for a pass whose coverage does not come from an attachment. No
    /// render rail here rasterizes above one sample, so a pass asking for four
    /// gets one — and the difference is not a quality setting: coverage decides
    /// which fragments run, so a shader that blends by coverage, an occlusion
    /// query that counts samples, and any edge the guest expected to be
    /// resolved all come back with a different answer than the one it asked
    /// for.
    ///
    /// Refused rather than counted for the reason
    /// [`Self::PassArrayLengthUnsupported`] gives: a pass rendered at one sample
    /// is exactly what a guest asking for one sample also produces, so nothing
    /// downstream can tell the substitution happened.
    ///
    /// The device advertises `DEVICE_INFO_KEY_MAX_SAMPLE_COUNT` above 1, so a
    /// guest is entitled to ask. This is the refusal that says what that
    /// advertisement costs when it does.
    PassRasterSampleCountUnsupported { count: u64 },
    /// An attachment's store-action *options* asking for something beyond the
    /// store action itself.
    ///
    /// `MTLStoreActionOptions` declares exactly one flag,
    /// `CustomSamplePositions`, and it asks that a multisample resolve use the
    /// pass's programmable sample positions. This device sets none — the pass
    /// record that carries them is dropped — so the resolve it would produce
    /// is not the one the option names, and a value outside the declared set
    /// is not an `MTLStoreActionOptions` at all.
    ///
    /// `MTLStoreActionOptionNone` is the API default and asks for nothing, so
    /// it is honoured rather than refused, on exactly the reading
    /// [`Self::PassRasterSampleCountUnsupported`] takes about a count of 1.
    /// Everything else is refused rather than counted, because a resolve done
    /// with the default sample positions is byte-for-byte what a guest that
    /// asked for the default also gets — nothing downstream can tell the
    /// substitution happened.
    StoreActionOptionsUnsupported {
        /// Which attachment the record names: `color`, `depth` or `stencil`.
        aspect: &'static str,
        /// The colour attachment index. Zero for the depth and stencil forms,
        /// which name their attachment by being themselves and carry no index.
        slot: u32,
        options: u64,
    },
}

impl crate::observe::Decline for StreamDrawDrop {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unbound { .. } => "stream_draw_dropped_unbound",
            Self::DepthStencilUnsupported { .. } => "stream_depth_stencil_unsupported",
            Self::ColorSubresourceUnsupported { .. } => "stream_color_subresource_unsupported",
            Self::PassArrayLengthUnsupported { .. } => "stream_pass_array_length_unsupported",
            Self::PassRasterSampleCountUnsupported { .. } => {
                "stream_pass_raster_sample_count_unsupported"
            }
            Self::StoreActionOptionsUnsupported { .. } => "stream_store_action_options_unsupported",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unbound { dropped } => vec![("dropped", dropped.to_string())],
            Self::DepthStencilUnsupported {
                aspect,
                level,
                slice,
                depth_plane,
                resolve_texture_ref,
            } => vec![
                ("aspect", (*aspect).to_string()),
                ("level", level.to_string()),
                ("slice", slice.to_string()),
                ("plane", depth_plane.to_string()),
                ("resolve", format!("{resolve_texture_ref:#x}")),
            ],
            Self::ColorSubresourceUnsupported {
                slot,
                level,
                slice,
                depth_plane,
                resolve_texture_ref,
            } => vec![
                ("slot", slot.to_string()),
                ("level", level.to_string()),
                ("slice", slice.to_string()),
                ("plane", depth_plane.to_string()),
                ("resolve", format!("{resolve_texture_ref:#x}")),
            ],
            Self::PassArrayLengthUnsupported { length } => {
                vec![("length", length.to_string())]
            }
            Self::PassRasterSampleCountUnsupported { count } => {
                vec![("count", count.to_string())]
            }
            Self::StoreActionOptionsUnsupported {
                aspect,
                slot,
                options,
            } => vec![
                ("aspect", (*aspect).to_string()),
                ("slot", slot.to_string()),
                ("options", format!("{options:#x}")),
            ],
        }
    }
}

impl StreamDrawDrop {
    /// The `fail_once` latch for this drop.
    ///
    /// Keyed on the fields that decide the arm and not on the task or the
    /// texture, because the question every one of these answers is which
    /// *shape* a guest asks for, not how many objects it asks for it on. A
    /// per-task latch would emit on every pass in every stream of a guest that
    /// uses mip-1 depth throughout.
    ///
    /// One definition for three emitters. The two pass arms had a copy each at
    /// their own emitter and [`note_draw_refused`] would have been the third —
    /// which is exactly where a latch quietly stops matching its sibling and
    /// one of them starts emitting per pass.
    ///
    /// [`Self::Unbound`] carries the stream's own count and is reported once per
    /// stream rather than latched, so its latch is the count: a stream that
    /// dropped a different number of draws is a different reading.
    pub(super) fn latch(self) -> u64 {
        match self {
            Self::Unbound { dropped } => u64::from(dropped),
            Self::DepthStencilUnsupported {
                aspect,
                level,
                slice,
                depth_plane,
                resolve_texture_ref,
            } => {
                u64::from(level) << 32
                    | u64::from(slice) << 16
                    | u64::from(depth_plane) << 8
                    | u64::from(resolve_texture_ref != 0) << 1
                    | u64::from(aspect == "stencil")
            }
            Self::ColorSubresourceUnsupported {
                slot,
                level,
                slice,
                depth_plane,
                resolve_texture_ref,
            } => {
                // The resolve ref contributes whether it is set, not which
                // texture it names, on the same reading its sibling above takes:
                // what this latch separates is which *shape* of attachment a
                // guest asks for, and one bit is the whole answer for a field
                // with no coordinate in it. Bit 63, above the slot, so it cannot
                // collide with a coordinate.
                u64::from(resolve_texture_ref != 0) << 63
                    | u64::from(slot) << 48
                    | u64::from(level) << 32
                    | u64::from(slice) << 16
                    | u64::from(depth_plane)
            }
            // The layer count itself: a guest asking for 6 layers and one asking
            // for 2 are different readings, and how many a pass declares is the
            // whole of what this arm has to say.
            Self::PassArrayLengthUnsupported { length } => length,
            // The requested count, on the same reading as the layer count
            // above: a guest asking for 2 samples and one asking for 8 are
            // different readings, and the count is the whole of what this arm
            // reports.
            Self::PassRasterSampleCountUnsupported { count } => count,
            // The options value and which attachment asked for it. A guest
            // asking for custom sample positions on colour 0 and one asking on
            // the depth attachment are different readings, and the two forms
            // do not even share a record length.
            Self::StoreActionOptionsUnsupported {
                aspect,
                slot,
                options,
            } => {
                options << 8
                    | u64::from(slot) << 2
                    | match aspect {
                        "depth" => 1,
                        "stencil" => 2,
                        _ => 0,
                    }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecResult {
    pub task_id: u32,
    pub streams_loaded: u32,
    /// Immutable shader translation is still running off the FIFO scheduler.
    /// The caller must keep this packet at the channel head and retry it.
    pub deferred: bool,
    pub texture_refs: Vec<u32>,
    pub mapper_ref_texture_mappings: Vec<u32>,
    pub saw_draw: bool,
    pub clears_applied: u32,
    pub metal_draws_ok: u32,
    pub metal_draws_fail: u32,
    /// Render-pass attachment sets resolved from guest objects. One Metal
    /// render stream has one fixed attachment set regardless of draw count.
    pub render_attachment_resolves: u32,
    /// Guest-visible color attachment Stores issued at render-pass completion.
    /// Multi-draw records stay resident; one pass must not full-frame import
    /// the same attachment after every draw.
    pub render_guest_stores: u32,
    /// Explicit nil entries in render bind ranges. These must remove prior
    /// slot state rather than silently retaining a stale resource.
    pub buffer_unbinds: u32,
    pub texture_unbinds: u32,
    pub sampler_unbinds: u32,
    /// Control-flow SPI encode failures (`0xdc`–`0xe2`).
    pub compute_control_fail: u32,
    /// ICB materialize+execute failures (`0xe4`/`0xe5`).
    pub compute_icb_fail: u32,
    /// Render ICB execute ok / fail (`0x14`/`0x15`).
    pub render_icb_ok: u32,
    pub render_icb_fail: u32,
    /// Wall-clock for the whole synchronous packet body. A packet holding the
    /// device lock past `SYNC_EXEC_STALL_US` starves the guest's read-to-clear
    /// completion registers; the drain reports that as a typed TRANSPORT line.
    pub total_us: u64,
}

/// How many command buffers one exec packet carried, in buckets.
///
/// **Asked because the semantic model's walk takes one buffer and this wire
/// carries a table of them.** `reims_vgpu_core::walk::exec` consumes an
/// `ExecBuilder` and finishes it, so a caller with two command buffers has one
/// transaction and no way to walk both into it — and whether that matters is
/// exactly this distribution. A stream of single-buffer submissions makes the
/// signature adequate; a tail above one makes it short of the wire, and the
/// tail is what a mean would hide.
///
/// It is not a hypothetical tail. The ceiling this loop used to carry truncated
/// at sixteen, and the submission that exposed it declared **seventeen** — see
/// `every_declared_command_buffer_is_visited_not_just_the_first_sixteen`. What
/// is missing is a current number, on the guest and workload the replacement
/// architecture is being cut over against.
///
/// Counted after the load, so the bucket is buffers this device could read
/// rather than buffers the header declared. The two differ exactly when a
/// descriptor was skipped, and `exec_cmdbuf` says which.
///
/// # What a driven boot answered
///
/// x86 Vulkan, macos-15, three rounds of five applications:
/// **`exec_cmdbufs_1=20 006` and every other bucket silent**, against
/// `packet_class_exec=20 006`. Every exec packet this guest sends carries
/// exactly one command buffer.
///
/// So `reims_vgpu_core::walk::exec`'s one-buffer signature is adequate for the
/// guest the cutover is being measured against, and the bridge that eventually
/// builds a `Payload::Exec` has one stream to hand it. That is a reading and not
/// a contract: the seventeen-buffer submission this loop's removed ceiling
/// truncated is in this repository's history, so the multi-buffer path is real
/// and `walk::command_buffer` is what it walks through. A bucket above `_1`
/// appearing is the day a caller must use it.
fn note_command_stream_count(loaded: usize) {
    // The denominator for `exec_stream_framed`: how many streams this device
    // was handed, against how many times it framed one. The bands below say
    // how a submission is shaped; this says how much there was to walk, and a
    // ratio above one is a re-scan.
    crate::runtime::drain::note_store_route_n("exec_streams_loaded", loaded as u64);
    crate::runtime::drain::note_store_route(match loaded {
        0 => "exec_cmdbufs_0",
        1 => "exec_cmdbufs_1",
        2..=4 => "exec_cmdbufs_2_4",
        5..=16 => "exec_cmdbufs_5_16",
        17..=64 => "exec_cmdbufs_17_64",
        _ => "exec_cmdbufs_over_64",
    });
}

/// Read every command buffer an `EXEC_INDIRECT2` payload's descriptor table
/// names out of the task's address space.
///
/// **The exec class's third input, as a value.** The payload carries a header, a
/// resource table and `cmdbuf_count` descriptors of `{gva, length}`; the streams
/// themselves live in the guest's own address space, so producing them is a page
/// table walk per buffer and not a slice of the packet. That is the input
/// [`crate::runtime::ingress::ExecStreams::streams`] takes, and the reason the
/// exec class cannot cross that bridge as a function of the drained packet the
/// way the other four do — it is a function of the drained packet *and these
/// bytes*. Named here so that whoever hands the bridge the streams calls one
/// function rather than restating this loop, and so the loop has a test of its
/// own.
///
/// `cbufs_off` and `cmdbuf_count` are the caller's because the caller has
/// already proved the table fits — `need = cbufs_off + count * DESC_LEN` against
/// `payload.len()`. Re-deriving them here would be a second bound that could
/// disagree with the one the caller returned on.
///
/// A buffer that cannot be read is **skipped and reported**, never fatal: a
/// zero-length descriptor, a length the host process cannot address, and a GVA
/// that does not walk are three different guest-visible losses and each says
/// which. The result is therefore shorter than the table whenever one happens,
/// and the caller's `streams_loaded` against `cmdbuf_count` is where that shows.
fn load_command_streams<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    payload: &[u8],
    cbufs_off: u64,
    cmdbuf_count: u32,
) -> Vec<Vec<u8>> {
    // Every command buffer the header declares, because the caller's `need`
    // already bounded how many there can be: the guest cannot claim a table
    // longer than the descriptors it actually supplied, so `cmdbuf_count` is
    // capped by `payload.len() / CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN` and
    // `with_capacity` cannot be talked into an allocation the payload does not
    // back.
    //
    // A fixed ceiling used to sit here and truncate with `.min()`, above the
    // check that already bounded the same number. Nothing derived it — a
    // submission of 17 lost its last command buffer entirely, before the loop,
    // with no fail line, which is a whole packet of guest draws vanishing into a
    // silently shorter table.
    let n_cb = cmdbuf_count as usize;
    let page_shift = state.page_shift;
    let mut streams = Vec::with_capacity(n_cb);
    for i in 0..n_cb {
        // The caller's `need` already pinned the whole table: i < n_cb <=
        // cmdbuf_count, so off + DESC_LEN = cbufs_off + (i + 1) * DESC_LEN <=
        // need <= payload.len(). The bounds check that stood here could not
        // fire, and its `break` would have dropped every remaining command
        // buffer with no line if it ever had.
        let off = (cbufs_off + i as u64 * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as u64) as usize;
        let gva = ld64(&payload[off + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..]);
        let length = ld64(&payload[off + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..]);
        if length == 0 {
            crate::observe::fail(format!(
                "exec_cmdbuf skip task={task_id} i={i} gva={gva:#x} len=0"
            ));
            continue;
        }
        // Guest length is authoritative — no product MiB budget. Fail only if
        // the host process cannot address the allocation.
        let Some(stream_len) = crate::runtime::draw::host_alloc_len(length) else {
            crate::observe::fail(format!(
                "exec_cmdbuf skip task={task_id} i={i} gva={gva:#x} len={length} (host_len)"
            ));
            continue;
        };
        let mut stream = vec![0u8; stream_len];
        // Product x86 uses page_shift=12; the unshifted helper defaults to arm14
        // and silently fails every stream load on Ventura/Tahoe x86.
        if gva_mem::read_task_gva_by_id(host, &state.tasks, task_id, gva, &mut stream, page_shift)
            .is_err()
        {
            crate::observe::fail(format!(
                "exec_cmdbuf gva_fail task={task_id} i={i} gva={gva:#x} len={length} shift={page_shift}"
            ));
            continue;
        }
        streams.push(stream);
    }
    streams
}

/// One `CmdExecIndirect2` packet, read.
///
/// **The seam between deciding and doing.** Everything above it is a function
/// of the packet's bytes and guest memory: which task, what the resource table
/// declares, and the command buffers themselves. Everything below it is host
/// work. The replacement architecture puts an admission decision between the
/// two — a packet may be judged and then *parked*, and executed later — so the
/// two halves cannot be one straight line through a single function, and the
/// buffers have to be a value that outlives the reading.
///
/// That is also why there is no separate double-load to remove: the streams are
/// loaded once, here, and whoever executes them is handed the same `Vec`.
#[derive(Debug)]
pub struct ExecSubmission {
    task_id: u32,
    /// Per resource this submission touches, who owns the authoritative bytes
    /// afterwards. Applied by [`consume_resource_table`] *before* any of the
    /// submission's work runs.
    resource_descs: Vec<ExecResourceDesc>,
    /// The command buffers the header declares, in the order it declares them.
    streams: Vec<Vec<u8>>,
}

impl ExecSubmission {
    /// The task whose object list the refs inside these streams index.
    ///
    /// From the exec header and not from the FIFO — see
    /// `crate::runtime::ingress::ExecStreams::task`.
    #[must_use]
    pub const fn task_id(&self) -> u32 {
        self.task_id
    }

    /// The command buffers, in the order the header declares them.
    #[must_use]
    pub fn streams(&self) -> &[Vec<u8>] {
        &self.streams
    }

    /// A submission stated outright rather than read out of a packet.
    ///
    /// For the suites that are about what a *read* submission does downstream
    /// — parking, admission, execution — where reaching them through
    /// [`read_submission`] would mean building a guest page table and an exec
    /// header to state two command buffers. The reading has its own tests;
    /// this is for everything the reading feeds.
    #[cfg(test)]
    pub(crate) fn stated(task_id: u32, streams: Vec<Vec<u8>>) -> Self {
        Self {
            task_id,
            resource_descs: Vec::new(),
            streams,
        }
    }

    /// Host bytes this submission is holding, for a caller accounting for what
    /// a parked position retains.
    ///
    /// The command buffers, which are the allocation: the resource
    /// descriptors are bounded by the table the header declares and are one
    /// small `Vec` beside megabytes of stream.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.streams.iter().map(Vec::len).sum()
    }
}

/// Read one `CmdExecIndirect2` packet into the submission it describes.
///
/// `None` for every reason the packet does not describe one, each of which is
/// already on the always-on channel where it is refused. `measured_ns`
/// accumulates this call's own spans so [`note_exec_header`] can derive the
/// leftover; see that function for why the header phase is a subtraction.
fn read_submission<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    payload: &[u8],
    out: &mut ExecResult,
    measured_ns: &mut u64,
) -> Option<ExecSubmission> {
    if payload.len() < CHILD_EXEC_INDIRECT_HEADER_LEN as usize {
        return None;
    }
    let raw_task = ld32(&payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..]);
    // The resolver guarantees a live slot or nothing, so there is no second
    // liveness check here. The refusal is always-on: an exec packet the crate
    // drops is a whole command stream of guest work lost, and it used to leave
    // no line at all.
    let Some(task_id) = resolve_task_word(&state.tasks, TaskWordSite::ExecIndirect2, raw_task)
    else {
        out.task_id = raw_task;
        crate::observe::fail(format!(
            "exec_indirect2 no_such_task task={raw_task} tasks={} plen={}",
            state.tasks.live_count(),
            payload.len()
        ));
        return None;
    };
    out.task_id = task_id;

    let resource_count = ld32(&payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..]);
    let cmdbuf_count = ld32(&payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..]);
    let resources_len = resource_count as u64 * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as u64;
    let cbufs_off = CHILD_EXEC_INDIRECT_HEADER_LEN as u64 + resources_len;
    let need = cbufs_off + cmdbuf_count as u64 * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as u64;
    if need > payload.len() as u64 {
        crate::observe::fail(format!(
            "exec_indirect2 short_payload task={task_id} res={resource_count} cbufs={cmdbuf_count} need={need} plen={}",
            payload.len()
        ));
        return None;
    }
    if cmdbuf_count == 0 {
        crate::observe::fail(format!(
            "exec_indirect2 zero_cbufs task={task_id} res={resource_count} plen={}",
            payload.len()
        ));
        return None;
    }

    // The guest declares, per resource this submission touches, who owns the
    // authoritative bytes afterwards. `need` above already proved the table fits,
    // so a refusal here means the header and the decoder disagree about the
    // layout — which is a fail line, never a silent empty table.
    let resource_descs = decode_exec_resource_table(payload).unwrap_or_else(|refusal| {
        crate::observe::fail(format!(
            "exec_res_table {} task={task_id} res={resource_count} plen={}",
            refusal.slug(),
            payload.len()
        ));
        Vec::new()
    });

    // This call's measured spans, summed, so `Header` can be the leftover. The
    // census's own totals cover the whole window and cannot answer for one call.
    let load_started = std::time::Instant::now();
    let streams = load_command_streams(state, host, task_id, payload, cbufs_off, cmdbuf_count);
    out.streams_loaded += u32::try_from(streams.len()).unwrap_or(u32::MAX);
    note_command_stream_count(streams.len());
    let load_ns = load_started.elapsed().as_nanos() as u64;
    *measured_ns += load_ns;
    crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Load, load_ns);

    Some(ExecSubmission {
        task_id,
        resource_descs,
        streams,
    })
}

/// Start every cold translation this submission needs, and say whether any is
/// still running.
///
/// **The middle of read / plan / execute**, and the reason the three are three.
/// Cold AIR translation is immutable CPU work over bytes the submission already
/// holds: it needs no protocol ownership, it mutates no guest state, and it can
/// therefore run at a moment of the caller's choosing. Execution cannot — it
/// consumes the resource table and encodes draws.
///
/// `true` means a referenced render stage is still translating. The caller must
/// then run *nothing* of this submission: the packet stays unconsumed so the
/// guest replays it, and replay must not duplicate clears, fences, dispatches
/// or guest writeback.
///
/// Every stream is scanned, not just up to the first pending one, so the
/// translations proceed in parallel and the submission is retried once rather
/// than once per stream. That is the rail's own contract — see
/// `crate::backend::Backend::preflight_translations` — and it is the property
/// that makes calling this early worth anything.
/// `resolved` is the walk's own answer about this packet, and its render
/// pipeline leases are the same names admission readies on this function's
/// verdict. Passing it rather than letting the rail rescan the bytes is what
/// makes "nothing is pending" a statement about exactly the leases it will be
/// used for — see [`vulkan::preflight_render_translations`].
pub fn preflight_submission<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    submission: &ExecSubmission,
    resolved: &reims_vgpu_core::exec::ExecWork,
    measured_ns: &mut u64,
) -> Vec<u32> {
    let preflight_started = std::time::Instant::now();
    // The render half of the packet's leases, and the guest's own ref, which is
    // the slot half of the name: the generation is the model's business and no
    // MTLB is keyed by it. Compute leases are held back because the compute
    // pre-scan takes its own inputs — a compute ref handed to the render arm
    // would fail to load a render MTLB pair and be counted as an unloadable
    // one, which is a real reading this would fill with a benign population.
    let refs: Vec<u32> = resolved
        .render_pipeline_leases()
        .iter()
        .map(|lease| lease.slot.0)
        .collect();
    // The compute half, from the same walk. A kernel is keyed by its
    // threadgroup size as well as its ref, which is why this is a pair list
    // and the render half is not; both are the model's answer rather than a
    // rescan of the packet's bytes.
    let dispatches: Vec<(u32, [u32; 3])> = resolved
        .compute_dispatch_translations()
        .into_iter()
        .filter_map(|(pipeline, size)| {
            let extent = [
                u32::try_from(size.width).ok()?,
                u32::try_from(size.height).ok()?,
                u32::try_from(size.depth).ok()?,
            ];
            // A grid with a zero edge dispatches nothing, and the translator
            // has no local size to key on. The model states the extent the
            // guest stated; deciding it is not a translation input is this
            // rail's business.
            (extent.iter().all(|d| *d != 0)).then_some((pipeline.slot.0, extent))
        })
        .collect();
    let pending = crate::backend::selected().preflight_translations(
        state,
        host,
        submission.task_id,
        &refs,
        &dispatches,
    );
    // Timed unconditionally: the phase is the drain's own accounting of where
    // an exec call's time went, and a rail that preflights nothing has to show
    // as the zero it costs rather than as an absent column the leftover
    // `Header` silently absorbs.
    let preflight_ns = preflight_started.elapsed().as_nanos() as u64;
    *measured_ns += preflight_ns;
    crate::runtime::drain::note_exec_phase(
        crate::runtime::drain::ExecPhase::Preflight,
        preflight_ns,
    );
    pending
}

/// Run one read and planned submission's host work.
///
/// Everything here mutates something the guest can see, which is what makes it
/// the third step rather than part of the second: the resource table is
/// consumed before the first record, and the records encode.
fn execute_submission<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    submission: &ExecSubmission,
    resolved: Option<&reims_vgpu_core::exec::ExecWork>,
    out: &mut ExecResult,
    measured_ns: &mut u64,
) {
    let ExecSubmission {
        task_id,
        resource_descs,
        streams,
    } = submission;
    let task_id = *task_id;

    // Before any of this submission's work runs. Each record states what was
    // true of its resource *before* the submission, so a pending window holding
    // pixels the guest has since overwritten has to go now — landing it later
    // would replace the guest's own bytes with a frame the guest has declared
    // stale.
    let guest_reads = consume_resource_table(state, task_id, resource_descs);

    // One cursor for the whole packet, not one per buffer: a packet's
    // command-buffer table is one submission and the model resolved all of it
    // into one flat record order, so a cursor that restarted per buffer would
    // compare the second buffer's records against the first buffer's answers.
    let mut resolved = resolved.map(ResolvedCursor::new);

    for stream in streams {
        let mut acc = StreamAccum::default();
        let walk_started = std::time::Instant::now();
        walk_stream(
            state,
            host,
            task_id,
            stream,
            out,
            &mut acc,
            resolved.as_mut(),
        );
        let walk_ns = walk_started.elapsed().as_nanos() as u64;
        *measured_ns += walk_ns;
        crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Walk, walk_ns);
        let finish_started = std::time::Instant::now();
        finish_stream(state, host, task_id, out, &acc);
        let finish_ns = finish_started.elapsed().as_nanos() as u64;
        *measured_ns += finish_ns;
        crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Finish, finish_ns);
    }
    // After the streams, because the frame the guest is about to read is the
    // one this submission just rendered: the table is consumed before any
    // record runs, when that frame's debt does not exist yet.
    for object_id in guest_reads {
        crate::runtime::resource_validity::deliver_for_guest_read(state, host, task_id, object_id);
    }
}

/// Run one submission whose plan step has already answered, from the inputs it
/// was admitted with.
///
/// **The only door into execution.** The door for a caller that planned
/// earlier — which is every admitted packet, because whether its translations
/// are done is what decided it could run at all. Planning again here would be
/// asking a question already answered and paying for it once per pipeline on
/// the hottest path this device has.
///
/// The caller owes the answer: a submission run without its plan step having
/// said `false` may encode against a shader that is still being translated.
///
/// # The tiling closes over two clocks, and it has to
///
/// [`ExecPhase::Header`] is a leftover — a span's worth of time minus the
/// phases that measured themselves — and it used to be derived from one clock
/// because reading and running were one call. They are not: a parked packet is
/// read when it arrives and run when the model releases it, and a single clock
/// spanning both would charge `Header` with the whole of the wait.
///
/// So each half closes its own leftover. The four measured phases are still
/// counted exactly once and `Header` is still everything else, which is the
/// property that made the tiling worth having; what changed is that it is now
/// the sum of two leftovers rather than one. `total_us` is this half's, which
/// is the half `sync_exec_stalled` is about.
pub fn execute_planned<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    inputs: RetainedInputs<'_>,
    mut out: ExecResult,
) -> ExecResult {
    let started = std::time::Instant::now();
    let mut measured_ns = 0u64;
    // Read, plan, execute. The plan step is separate because it is the only one
    // of the three that a caller may run at a moment of its own choosing: it is
    // pure CPU work over bytes the submission already holds, where the read
    // walks the guest's page tables and the execution consumes the resource
    // table.
    execute_submission(
        state,
        host,
        inputs.submission,
        Some(inputs.resolved),
        &mut out,
        &mut measured_ns,
    );
    note_exec_header(started, measured_ns);
    if !out.deferred {
        out.total_us = elapsed_us(started);
    }
    out
}

/// Read one `CmdExecIndirect2` packet's submission and nothing more.
///
/// The half a drain performs at *arrival*: the streams live in the task's
/// address space and the guest may reuse that memory once the ring head has
/// advanced past the packet, so whoever runs the submission later has to have
/// been handed these bytes rather than reading them again.
///
/// It closes its own `ExecPhase::Header` leftover, because the running half
/// happens at a different moment and one clock across both would charge the
/// leftover with the wait — see [`execute_planned`].
#[must_use]
pub fn read_exec_submission<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    payload: &[u8],
) -> (Option<ExecSubmission>, ExecResult) {
    let started = std::time::Instant::now();
    let mut out = ExecResult::default();
    let mut measured_ns = 0u64;
    let submission = read_submission(state, host, payload, &mut out, &mut measured_ns);
    note_exec_header(started, measured_ns);
    (submission, out)
}

/// Close the [`ExecPhase`] tiling of one exec half at one of its return
/// points.
///
/// [`ExecPhase::Header`] is the **leftover**, not a span: it is the function's
/// own elapsed time minus the four that measured themselves, so the five sum to
/// the opcode's `op0x37_us` whatever path the call took. Deriving it rather than
/// wrapping the header parse is what makes the tiling closed — a cost in a
/// corner nobody thought to list still lands here instead of vanishing, which is
/// the property that made the child-FIFO tiling answer on one boot.
///
/// `measured_ns` is **this call's** four spans summed, not the census's running
/// totals: the census accumulates across every packet in the window, so
/// subtracting it from one call's clock would be subtracting the whole second.
/// The subtraction is saturating anyway, because an underflow would print as a
/// colossal `header_us` rather than as the zero it means.
fn note_exec_header(exec_started: std::time::Instant, measured_ns: u64) {
    let total = exec_started.elapsed().as_nanos() as u64;
    crate::runtime::drain::note_exec_phase(
        crate::runtime::drain::ExecPhase::Header,
        total.saturating_sub(measured_ns),
    );
}

/// Apply every record of one submission's resource table.
///
/// The table is the guest's own statement about who owns each resource's
/// authoritative bytes, and `clear_host_valid` is its consume-once notification
/// that it CPU-wrote one — delivered here and nowhere else.
///
/// # What "did not apply" means, in two kinds
///
/// The table's ids are the **task's object-ref space**, not the mapping space.
/// Measured over one boot's 6 823 records: 72 % are live object refs, 20 % are
/// mappings, 19 % resolve nowhere, and `texture_to_mapping` answered for exactly
/// none. So most records name resources that have no surface state to apply a
/// validity quad to — buffers, heaps, pipelines — and that is the protocol
/// working, not a loss.
///
/// The two are therefore counted apart. `validity_no_surface` is the expected
/// majority; `validity_unknown_object` is a record naming an id no registry has
/// heard of. Merging them would bury the second under the first at roughly four
/// to one.
///
/// **That four-to-one is the x86 pathway's, and arm64 is not close to it.** Two
/// driven arm64/Vulkan boots read `no_surface`/`unknown` of 1342/926 and
/// 713/536 — about **1.4 to 1** on both, with the two workloads deliberately
/// different. So the majority is much thinner here, and a reader who takes the
/// ratio above as the protocol's shape will either think this pathway is broken
/// or fail to notice that it is not the same. Neither reading is available from
/// one host: the numbers are the same counters measuring the same thing, and
/// what differs is how much of a submission's residency list has already been
/// named by an executed command when the table arrives. What would still be the
/// finding on *either* pathway is the one named above — this count staying high
/// for ids that later do execute — and nothing measures that yet on either.
///
/// `validity_unknown_object` is **not** by itself a defect either, and a reader
/// scoring it needs to know why: `DeviceState::objects` is populated lazily, by
/// `objects::resolve_mapper_ref_texture` and `resolve_backing_ex` at the moment a
/// decoded command names a ref. A resource the guest has created in its own
/// object list but has not yet named in an executed stream is absent from the
/// set by construction. The table names the submission's whole residency list,
/// which is a superset of what its command buffers reference. What *would* be
/// the finding is this count staying high for ids that later do execute.
///
/// # What `set_host_valid` means, and how that is known
///
/// It licenses exactly the resources the submission stores into. That was an
/// inference from IOAccel resource-list usage until a census correlated the two
/// sides over one driven boot: of 19 135 stores, **zero** landed on a resource
/// the table had not licensed, and the records that both license a resource and
/// name a mapping this device holds (1 382 vs 1 380 licensed-and-stored) are the
/// render targets. `clear_host_valid` is the other direction and arrives 15 423
/// times in the same boot — one per guest CPU write, never resent.
///
/// The census that measured it is gone; a correlation with no counter-examples
/// over 19 135 trials is a finding, not a thing to keep re-deriving per frame.
/// Returns the objects whose records asked for the guest's copy
/// ([`crate::runtime::resource_validity::ValidityOutcome::guest_read_requested`]),
/// for the caller to deliver once the submission has run.
fn consume_resource_table(
    state: &mut DeviceState,
    task_id: u32,
    descs: &[ExecResourceDesc],
) -> Vec<u32> {
    use crate::runtime::resource_validity::{apply, ValiditySite};
    let mut guest_reads = Vec::new();
    let mut no_surface = 0u32;
    let mut unknown = 0u32;
    for d in descs {
        if d.tail_nonzero_bytes() > 0 {
            crate::observe::Emit::decline("exec_res_table", &ResourceTableDecline::TailPopulated)
                .field("task", task_id)
                .field("object", d.object_id)
                .field("tail_nz", d.tail_nonzero_bytes())
                .fail_once(0);
        }
        let outcome = apply(state, task_id, d.object_id, d.ops, ValiditySite::ExecTable);
        if outcome.guest_read_requested {
            guest_reads.push(d.object_id);
        }
        if !outcome.missed {
            continue;
        }
        if state.objects.contains(&(task_id, d.object_id)) {
            no_surface = no_surface.saturating_add(1);
        } else {
            unknown = unknown.saturating_add(1);
        }
    }
    // Rate-summarised on the per-second store-route window: this is the hottest
    // opcode in the device and a per-record line would bury the fail view.
    crate::runtime::drain::note_store_route_n("validity_no_surface", no_surface as u64);
    crate::runtime::drain::note_store_route_n("validity_unknown_object", unknown as u64);
    guest_reads
}

/// The one part of an `EXEC_INDIRECT2` resource-table record this device cannot
/// act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResourceTableDecline {
    /// A record set one of the trailing 16 bytes, whose meaning is unrecovered.
    ///
    /// Zero across 84 868 records on the Ventura 13.7.8 x86 build, so ignoring
    /// them costs nothing *there*. A build that starts using them is a statement
    /// this device is discarding, which is why it raises a line rather than
    /// passing unread — once per boot, because the field is a property of the
    /// guest build and not of the record that happened to carry it first.
    TailPopulated,
}

impl crate::observe::Decline for ResourceTableDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::TailPopulated => "exec_res_tail_populated",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

fn elapsed_us(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// The two inputs an admitted packet was retained with.
///
/// One type rather than two arguments, because they are one fact: the command
/// buffers are what the packet was admitted *against* and the resolved records
/// are what it was admitted *as*, both taken at arrival before the guest could
/// rewrite either. A caller holding one without the other would be running
/// bytes its records never named.
#[derive(Clone, Copy)]
pub struct RetainedInputs<'a> {
    pub submission: &'a ExecSubmission,
    pub resolved: &'a reims_vgpu_core::exec::ExecWork,
}

/// Where an executing walk stands in the records the model already resolved
/// this packet into.
///
/// **The walks are the same walk.** Both this device's executor and
/// `reims_vgpu_core::walk` frame a command buffer with
/// `reims_vgpu_protocol::segment::SegmentStream` and its records with
/// `reims_vgpu_wire::op::OpStream`, step a protection envelope over without
/// opening an encoder for it, and visit every framed record of every buffer in
/// table order. So the two agree about *where* a record is, and
/// [`reims_vgpu_core::stream::StreamPosition`] is that agreement written down:
/// a segment ordinal that counts encoder segments across the whole packet, and
/// a record ordinal that restarts at each segment boundary.
///
/// **Keyed by position, not by arrival count.** A flat "nth record here is the
/// nth record there" cursor holds today — measured at 275 782 records with no
/// disagreement on a driven macos-15 boot, and 182 820 on macos-26 — but it
/// holds only while the two walks admit exactly the same records. A transaction
/// that one day omits a record the executor still reaches would put every later
/// record off by one, and each would be read as the wrong operation rather than
/// as a gap. A position lookup reads that as one absent twin and nothing else,
/// which is the difference between a gap and a corruption.
///
/// The class the ledger gives the opcode is compared against the class the
/// model resolved at every step. A disagreement is counted by name rather than
/// acted on: until that count is a measured zero, nothing here decides
/// anything.
struct ResolvedCursor<'a> {
    /// Ascending by position, which is the order [`ExecWork::records`] yields
    /// and therefore needs no sort.
    records: Vec<&'a reims_vgpu_core::exec::StreamRecord>,
    at: reims_vgpu_core::stream::StreamPosition,
}

impl<'a> ResolvedCursor<'a> {
    fn new(work: &'a reims_vgpu_core::exec::ExecWork) -> Self {
        Self {
            records: work.records().collect(),
            at: reims_vgpu_core::stream::StreamPosition {
                segment: 0,
                record: 0,
            },
        }
    }

    /// A segment's records are over.
    ///
    /// The segment ordinal advances and the record ordinal restarts, which is
    /// `reims_vgpu_core::stream`'s own rule: an encoder spanning three segments
    /// still gives its records three distinct segment indices, because where a
    /// record was written is not the same question as which encoder ran it. A
    /// protection envelope opens no encoder and so is not a segment here — the
    /// walk steps over it without reaching this.
    fn end_segment(&mut self) {
        self.at.segment += 1;
        self.at.record = 0;
    }

    /// The resolved operation standing where this walk is, if the two agree
    /// about what it is.
    ///
    /// Advances whatever the answer, because the walks advance together.
    fn step(
        &mut self,
        kind: SegmentKind,
        opcode: u32,
    ) -> Option<&'a reims_vgpu_core::exec::ResolvedOperation> {
        let at = self.at;
        self.at.record += 1;
        let Ok(index) = self.records.binary_search_by_key(&at, |record| record.at) else {
            crate::runtime::drain::note_store_route("resolved_walk_no_record");
            return None;
        };
        let record = self.records[index];
        let Some(expected) = stream_class(kind, opcode) else {
            // The ledger has not settled this opcode as a stream record, so
            // this walk has no class to compare against and says so rather than
            // reporting the model's answer as wrong.
            crate::runtime::drain::note_store_route("resolved_walk_unjudged");
            return None;
        };
        if record.op.class() != expected {
            crate::runtime::drain::note_store_route("resolved_walk_class_differs");
            return None;
        }
        crate::runtime::drain::note_store_route("resolved_walk_aligned");
        Some(&record.op)
    }
}

/// The stream class an opcode belongs to on the rail its segment names.
///
/// The ledger's own answer, through the same two steps
/// `reims_vgpu_core::resolve::operation` takes, so "which class is this record"
/// has one owner and this is a reader of it rather than a second table.
fn stream_class(kind: SegmentKind, opcode: u32) -> Option<OperationClass> {
    let row = reims_vgpu_protocol::closure::find(kind.rail(), opcode)?;
    match reims_vgpu_core::operation::classify(row) {
        Some(reims_vgpu_core::operation::OperationHome::Stream(class)) => Some(class),
        Some(reims_vgpu_core::operation::OperationHome::ObjectLifecycle) | None => None,
    }
}

/// Walk every record in one segment, handing each handler its opcode and its
/// command bytes.
///
/// Lifting this out of `walk_stream`'s five near-identical arms gives the framing
/// decoder exactly one emission site. Each arm previously swallowed its refusals
/// twice over: `if let Ok(r) = decode_first_record(..)` dropped a malformed first
/// record with no line at all, and `Err(_) => break` made a truncated or
/// self-inconsistent segment indistinguishable from `Done` — so every remaining
/// record in that segment went unexecuted and unreported.
///
/// Slicing here rather than in each handler is what makes the record's extent a
/// framing property instead of five re-derivations of it. `decode_next_record`
/// already refuses `record_len > command_end - cursor` and `validate_segment`
/// refuses `command_end > bytes.len()`, so `bytes_offset + length` is inside
/// `stream` by construction — the five copies of that same bounds check each
/// had a silent `return` behind a branch none of them could take.
///
/// # The record count is the denominator for `exec_phase walk_us`
///
/// `walk_us` is the largest single span this device reports — 31.6 s of a 45.6 s
/// driven macos-13 Maps window, against 6.4 s of actual drawing — and until this
/// counter existed there was no way to tell which of two very different readings
/// it was. 857 us per stream is either tens of microseconds spent on each of a
/// few dozen records, which points at one expensive handler, or a fraction of a
/// microsecond spent on each of tens of thousands, which points at the guest
/// simply sending that many. Counted per segment family, because a blit record
/// and a render record cost nothing like the same and the mix is what says which
/// of the two the wall clock belongs to.
fn walk_segment_records(
    kind: SegmentKind,
    commands_offset: u32,
    commands: &[u8],
    mut handle: impl FnMut(u32, &[u8]),
) {
    let mut records = 0u64;
    let (route, route_us) = match kind {
        SegmentKind::Render => ("walk_records_render", "walk_render_us"),
        SegmentKind::Blit => ("walk_records_blit", "walk_blit_us"),
        SegmentKind::Compute => ("walk_records_compute", "walk_compute_us"),
        SegmentKind::Event | SegmentKind::Info => ("walk_records_other", "walk_other_us"),
    };
    // One clock pair per *segment*, not per record. A stream carries at most a
    // handful of segments and tens of thousands of records, so this splits
    // `exec_phase walk_us` by family for a cost that does not show up, where
    // per-record timing would cost more than the handlers it measured.
    let started = std::time::Instant::now();
    let mut ops = reims_vgpu_wire::op::OpStream::new(commands);
    let mut refusal = None;
    for next in ops.by_ref() {
        match next {
            Ok(op) => {
                records += 1;
                let start = op.offset;
                handle(op.opcode(), &commands[start..start + op.length() as usize]);
            }
            // The iterator stops after yielding a refusal, so this is the last
            // item and the loop ends with it.
            Err(e) => refusal = Some(RecordFraming::from(e)),
        }
    }
    crate::runtime::drain::note_store_route_n(route, records);
    crate::runtime::drain::note_store_route_us(route_us, started.elapsed().as_micros() as u64);
    if let Some(status) = refusal {
        if let Some(e) = crate::observe::Emit::refusal("stream_record_fail", &status) {
            // Latch per segment family: a guest re-submitting a malformed
            // stream sends it on every frame and the second line carries
            // nothing the first did not. Keying on the family still tells
            // a broken blit segment from a broken render one, which
            // keying on the reason alone would hide.
            e.field("seg", kind.name())
                .field("seg_cmd_off", commands_offset)
                .field("seg_cmd_len", commands.len())
                .field("cursor", ops.consumed())
                .fail_once(u64::from(kind.wire_type()));
        }
    }
}

/// Why a segment's records stopped framing.
///
/// **The name is this device's; the fact is `reims_vgpu_wire`'s.** That crate
/// has no dependencies on purpose, so it cannot carry a `Decline`, and the
/// device is the layer that has to say which check stopped a walk. Everything
/// numeric a reader needs is on the emitted line, so this is a name and not a
/// second reading of the bytes.
///
/// The two length refusals the device's own framer had — "below its own header"
/// and "past the segment's end" — are one variant here, because
/// [`reims_vgpu_wire::op::op`] makes one check of both. Which of the two it was
/// is still legible: `length` under eight is the first and `length` over
/// `remaining` is the second, and both are on the line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordFraming {
    ShortHeader {
        need: usize,
        have: usize,
    },
    BadLength {
        opcode: u32,
        length: u32,
        remaining: usize,
    },
    /// Neither is reachable from a record walk — `op` views at offset zero of a
    /// slice and counts nothing — and both are named rather than folded into a
    /// catch-all, so a walk that started producing one would say so instead of
    /// borrowing another check's name.
    OutOfRange {
        offset: usize,
        len: usize,
    },
    CountOverflow {
        count: usize,
        elem: usize,
    },
}

impl From<reims_vgpu_wire::WireError> for RecordFraming {
    fn from(e: reims_vgpu_wire::WireError) -> Self {
        use reims_vgpu_wire::WireError as W;
        match e {
            W::Short { need, have } => Self::ShortHeader { need, have },
            W::BadLength {
                opcode,
                length,
                remaining,
            } => Self::BadLength {
                opcode,
                length,
                remaining,
            },
            W::OutOfRange { offset, len } => Self::OutOfRange { offset, len },
            W::CountOverflow { count, elem } => Self::CountOverflow { count, elem },
        }
    }
}

impl crate::observe::Refusal for RecordFraming {
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ShortHeader { .. } => "stream_rec_short_header",
            Self::BadLength { .. } => "stream_rec_bad_length",
            Self::OutOfRange { .. } => "stream_rec_view_out_of_range",
            Self::CountOverflow { .. } => "stream_rec_count_overflow",
        })
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::ShortHeader { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::BadLength {
                opcode,
                length,
                remaining,
            } => vec![
                ("opcode", format!("{opcode:#x}")),
                ("length", length.to_string()),
                ("remaining", remaining.to_string()),
            ],
            Self::OutOfRange { offset, len } => {
                vec![("offset", offset.to_string()), ("len", len.to_string())]
            }
            Self::CountOverflow { count, elem } => {
                vec![("count", count.to_string()), ("elem", elem.to_string())]
            }
        }
    }
}

/// The per-record work an exec stream's segments cause, separated from whoever
/// walks the stream.
///
/// `walk_stream` used to hold both halves: the framing loop, and inside each of
/// its five arms the handler call plus — for compute — the segment-spanning
/// encoder that one record opens and the segment's end commits. They are two
/// different things. `reims_vgpu_core::walk` frames and records the same
/// segments from the same `reims_vgpu_protocol::segment::SegmentStream`, and a
/// device driven from that walk needs this half without this loop.
///
/// Splitting them here is what makes the walker substitutable rather than
/// duplicated. `walk.rs`'s own module documentation names the class of defect
/// that matters: two walks over one stream are two chances to disagree about
/// where a record was, and the dispatch is the part that would have had to be
/// written twice for them to disagree at all.
///
/// The sink owns exactly the state that spans records — the open compute
/// segment — and nothing that spans streams.
struct StreamSink<'a, M: HostMemory + HostOps> {
    state: &'a mut DeviceState,
    host: &'a mut M,
    task_id: u32,
    out: &'a mut ExecResult,
    acc: &'a mut StreamAccum,
    /// `Some` between the [`Self::begin_segment`] and [`Self::end_segment`] of a
    /// compute segment, and `None` everywhere else. A compute record arriving
    /// while this is `None` is a walker that skipped the opening, which is a
    /// loss and is named rather than absorbed.
    compute: Option<crate::runtime::compute_session::ComputeSegment>,
}

impl<'a, M: HostMemory + HostOps> StreamSink<'a, M> {
    /// A segment of `kind` is about to deliver its records.
    fn begin_segment(&mut self, kind: SegmentKind) {
        if self.compute.is_some() {
            // The previous segment's encoder was never committed. Committing it
            // here is the only reading that does not drop the work the guest
            // already recorded into it, and the line says the walker paired its
            // calls wrongly rather than leaving that silent.
            crate::observe::fail(format!(
                "exec_segment_unended task={} opening={} (a segment opened while a compute \
                 segment was still open, so its encoder had not been committed)",
                self.task_id,
                kind.name()
            ));
            self.end_segment();
        }
        if matches!(kind, SegmentKind::Compute) {
            self.compute = Some(crate::runtime::compute_session::ComputeSegment::default());
        }
    }

    /// One record of the open segment.
    ///
    /// `kind` is the segment's, which is the only defensible source for the
    /// rail a record is read on — the same rule `reims_vgpu_core::walk` states
    /// for `resolve::operation`, and the reason it is a parameter here rather
    /// than something this method could derive from the opcode.
    fn record(&mut self, kind: SegmentKind, opcode: u32, cmd: &[u8]) {
        match kind {
            SegmentKind::Render => handle_render_record(
                self.state,
                self.host,
                self.task_id,
                opcode,
                cmd,
                self.out,
                self.acc,
            ),
            SegmentKind::Blit => {
                handle_blit_record(self.state, self.host, self.task_id, opcode, cmd)
            }
            SegmentKind::Compute => {
                let Some(compute) = self.compute.as_mut() else {
                    crate::observe::fail(format!(
                        "exec_compute_record_unopened task={} op={opcode:#x} len={} (a compute \
                         record arrived with no compute segment open, so its dispatch is lost)",
                        self.task_id,
                        cmd.len()
                    ));
                    return;
                };
                handle_compute_record(
                    self.state,
                    self.host,
                    self.task_id,
                    opcode,
                    cmd,
                    self.out,
                    compute,
                );
            }
            SegmentKind::Event => handle_event_record(self.state, self.task_id, cmd),
            SegmentKind::Info => {
                handle_info_record(self.state, self.host, self.task_id, opcode, cmd);
            }
        }
    }

    /// The open segment has delivered its last record.
    ///
    /// Only compute has anything to do here, and it is a commit rather than a
    /// teardown: the session's whole multi-record encoder is the work, so a
    /// failure at this point loses all of it and is the one thing this method
    /// reports.
    fn end_segment(&mut self) {
        let Some(mut segment) = self.compute.take() else {
            return;
        };
        let Some(status) = crate::runtime::compute_session::finish_session(
            &mut segment.session,
            self.state,
            self.host,
            self.task_id,
        ) else {
            return;
        };
        if !matches!(status, ComputeStatus::Ok) {
            self.out.compute_control_fail += 1;
            // Segment-end commit: the whole multi-record session's work is
            // gone, and this counter was its only trace.
            if let Some(e) = crate::observe::Emit::refusal("compute_session_finish", &status) {
                e.field("task", self.task_id)
                    .fail_once(u64::from(self.task_id));
            }
        }
    }
}

fn walk_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    stream: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
    mut resolved: Option<&mut ResolvedCursor<'_>>,
) {
    // **The one place this crate frames an exec stream, counted.** Seam 6 asks
    // for zero re-scans of already resolved EXEC bytes, and the shape the
    // violation took here was two rail pre-scans walking the same stream
    // through the same framer before execution walked it a third time. So the
    // measurable form of that zero is "one framing per stream per submission",
    // and this is the number it is read off — on a boot, against
    // `exec_streams_loaded`; off the VM, by
    // fn `a_submission_frames_each_stream_once_and_preflight_frames_none`.
    crate::runtime::drain::note_store_route("exec_stream_framed");
    // The outermost frame in the crate, and it is
    // `reims_vgpu_protocol::segment`'s. This device had its own segment framer
    // whose per-segment index was found by re-walking the stream from zero — a
    // quadratic scan over a stream a driven boot puts ninety-five thousand
    // segments in. `SegmentStream` counts the index it already knows.
    let mut segments = match SegmentStream::new(stream) {
        Ok(segments) => segments,
        Err(refusal) => {
            // A stream that will not frame executes *nothing*, and that used to
            // be indistinguishable from an idle guest: no records, no work, no
            // line.
            crate::observe::Emit::decline("stream_frame_fail", &refusal)
                .field("task", task_id)
                .field("bytes", stream.len())
                .fail_once(u64::from(task_id));
            return;
        }
    };
    let mut sink = StreamSink {
        state,
        host,
        task_id,
        out,
        acc,
        compute: None,
    };
    for framed in segments.by_ref() {
        let framed = match framed {
            Ok(framed) => framed,
            // **The iterator stops here, and every later segment goes
            // unexecuted.** That is `reims_vgpu_protocol::segment`'s reading and
            // it is stricter than the framer it replaces, which reported an
            // unknown segment type and then walked on to the next one. Its
            // reason is on `FramingRefusal::UnknownType`: a family whose record
            // framing is unknown can be skipped only on its declared length,
            // and skipping on that hands the following segment an encoder state
            // derived from bytes nothing here understands. The reference host
            // rejects a non-continuation type it has no decoder for rather than
            // stepping over it.
            Err(refusal) => {
                crate::observe::Emit::decline("stream_frame_fail", &refusal)
                    .field("task", task_id)
                    .field("bytes", stream.len())
                    .field("segments_before", segments.segments())
                    .fail_once(u64::from(task_id));
                return;
            }
        };
        let (kind, commands) = match framed.body {
            SegmentBody::Encoder { kind, commands } => (kind, commands),
            // `-beginSegment:protectionOptions:` emits a segment-level envelope
            // before the real segment, and skipping it is contract-correct — so
            // it is control flow and stays silent, where a line would land in
            // the sink on every healthy frame that carries one. This device
            // implements no protection domain, so nothing acts on the value;
            // that it is *read* rather than stepped over is what lets that
            // sentence be a choice rather than a limitation.
            SegmentBody::ProtectionEnvelope { .. } => continue,
        };
        // The encoder-lifetime census, counted once per segment. The two
        // pre-scans on the Vulkan rail used to walk the same stream through the
        // same framer, so every segment of every stream was counted three times
        // on that rail and the reading was a rail-dependent multiple of the
        // truth.
        crate::runtime::drain::note_store_route(segment_chain_route(framed.lifetime));
        sink.begin_segment(kind);
        walk_segment_records(kind, framed.commands_offset, commands, |op, cmd| {
            // Stepped for every record this walk reaches, whether or not the
            // arm below has anything to do with the answer: the two walks stay
            // in correspondence by advancing together, and a step taken only
            // for the classes that consume it would be a cursor that means
            // something different in every segment.
            if let Some(cursor) = resolved.as_deref_mut() {
                let _ = cursor.step(kind, op);
            }
            sink.record(kind, op, cmd);
        });
        sink.end_segment();
        if let Some(cursor) = resolved.as_deref_mut() {
            cursor.end_segment();
        }
    }
}

/// Every census route [`segment_chain_route`] can answer, in the order
/// `(continues_previous, continues_into_next)` counts up.
///
/// Exported so a reading is over a named set rather than over whichever names a
/// grep of the log happened to find, and so the four cannot be spelled twice.
pub const SEGMENT_CHAIN_ROUTES: [&str; 4] = [
    "seg_chain_none",
    "seg_chain_next",
    "seg_chain_prev",
    "seg_chain_both",
];

/// Which of [`SEGMENT_CHAIN_ROUTES`] a segment's encoder lifetime selects.
///
/// Takes the [`SegmentLifetime`] rather than two bytes side by side. The edge is
/// recorded from both ends of it — the serializer writes the `beginSegment:`
/// `BOOL` into the header it opens and reaches back to mark the preceding one —
/// so a caller handed the two halves separately could carry one and drop the
/// other. The `!= 0` reading of each byte is the protocol crate's and is not
/// repeated here.
#[must_use]
pub fn segment_chain_route(lifetime: SegmentLifetime) -> &'static str {
    let index =
        usize::from(lifetime.continues_previous) << 1 | usize::from(lifetime.continues_into_next);
    SEGMENT_CHAIN_ROUTES[index]
}

fn handle_info_record<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
) {
    use crate::runtime::icb::{
        apply_icb_host_resource_info, decode_icb_host_resource_info, INFO_OP_ICB_HOST_RESOURCE,
    };
    let bytes = cmd_bytes;
    if opcode == INFO_OP_ICB_HOST_RESOURCE {
        // `icb_backing_fail` was a counter with no reason beside it: an ICB
        // whose command memory never bound looked identical whether the payload
        // was malformed, the buffer was short, or the pathway has no ICB
        // execution at all. Latched per ICB ref — the guest re-sends `0x1d1`
        // for the same ICB, so an unlatched line would be one per frame.
        //
        // `apply_icb_host_resource_info` now always refuses: `0x1d1` is a query
        // whose answer this device does not compute. The reply pair is logged
        // because it is where the answer *would* go, not because anything reads
        // it — the previous reading bound it as the ICB's command memory.
        match decode_icb_host_resource_info(bytes) {
            Ok(info) => match apply_icb_host_resource_info(state, host, task_id, &info) {
                Ok(_) => {}
                Err(e) => {
                    crate::observe::Emit::decline("icb_backing", &e)
                        .field("task", task_id)
                        .field("icb", info.icb_ref)
                        .field("reply_buf", info.reply_buffer_ref)
                        .field("reply_off", info.reply_offset)
                        .fail_once(info.icb_ref as u64);
                }
            },
            Err(e) => {
                crate::observe::Emit::decline("icb_backing", &e)
                    .field("task", task_id)
                    .field("len", bytes.len())
                    .fail_once(bytes.len() as u64);
            }
        }
    }
    // Every info record is a question, `0x1d1` included: the arm above logs
    // where the answer would go and does not compute one, so the reply buffer
    // is left holding whatever it held. Reporting them all through one site
    // keeps that true — an arm that starts answering `0x1d1` has to return
    // before it reaches here, which is a change this line makes visible rather
    // than one it hides.
    note_info_record_unanswered(task_id, opcode, bytes.len());
}

/// One event-segment record, lifted by the protocol crate and executed here.
///
/// **The lift is `reims_vgpu_protocol::decode::sync`'s now.** This device
/// carried its own event decoder, which framed the same three opcodes and
/// carried its own copy of the blit encoder's fence numbers in order to refuse
/// them. That is the wire's question with a contract answer, and the protocol
/// crate is where it is answered for every rail at once — so the boundary
/// probes, the cross-encoder refusals and the three-opcode window stop being
/// this crate's to keep true.
///
/// **`waitForEvent:value:timeoutMS:` is refused at the lift and not after it.**
/// The row is settled: this device runs no clock against the guest's, so
/// executing the bounded wait as the unbounded one it resembles turns a guest's
/// timeout into a hang. `event_kind` gives it no kind and the lift refuses it
/// `RefusedByContract`. The old path decoded it, planned it, and refused it at
/// the end — one settled row refused in two places, and the one further from
/// the ledger was the one a reader found first.
fn handle_event_record(state: &mut DeviceState, task_id: u32, cmd_bytes: &[u8]) {
    let refuse = |decline: &dyn crate::observe::Decline| {
        // A malformed or refused event record drops a guest signal or wait
        // outright, so the loss is named rather than counted. The record's
        // own refusal says which check refused it, on which rail, at which
        // opcode.
        crate::observe::Emit::decline("event_record", decline)
            .field("task", task_id)
            .field("len", cmd_bytes.len())
            .fail();
    };
    let op = match reims_vgpu_protocol::decode::op(cmd_bytes, 0) {
        Ok(op) => op,
        Err(_) => {
            // The header itself did not frame. `walk_segment_records` has
            // already framed this record once to hand it over, so this is
            // unreachable rather than a guest case — and reported, because a
            // frame that stopped agreeing with itself between two reads is not
            // something to pass over.
            crate::observe::fail(format!(
                "event_record_unframed task={task_id} len={} (the segment walk framed this record                  and the op header did not)",
                cmd_bytes.len()
            ));
            return;
        }
    };
    let record = match reims_vgpu_protocol::decode::sync::decode(
        reims_vgpu_protocol::closure::Rail::Event,
        &op,
    ) {
        Ok(reims_vgpu_protocol::decode::sync::SyncRecord::Event(record)) => record,
        Ok(other) => {
            // The event rail lifts one record class. A fence or a barrier here
            // would mean `decode::sync` and `event_kind` disagree about which
            // rail owns which opcode.
            crate::observe::fail(format!(
                "event_record_not_an_event task={task_id} op={:#x} kind={} (the event rail lifted                  a record that is not an event)",
                op.opcode(),
                match other {
                    reims_vgpu_protocol::decode::sync::SyncRecord::Fence(_) => "fence",
                    reims_vgpu_protocol::decode::sync::SyncRecord::Barrier(_) => "barrier",
                    reims_vgpu_protocol::decode::sync::SyncRecord::Event(_) => "event",
                }
            ));
            return;
        }
        Err(refusal) => {
            refuse(&refusal);
            return;
        }
    };
    // Refusals are emitted by `execute_event` itself, against the ref that
    // failed; there is nothing left for this caller to report.
    fence_exec::execute_event(state, task_id, &record);
}

fn handle_compute_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
    out: &mut ExecResult,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) {
    // **Which class of record this is, is the closure ledger's answer.** The
    // compute *rail* carries six: the encoder's own records, the fence pair,
    // the barriers, the content-representation flush, the sequencing SPI
    // (control flow and indirect-command execution), and the residency
    // declarations it inherits from the shared encoder base class. This device
    // answered all six with one `Kind`, decoded in the same pass that read the
    // fields — so a record's class and its layout were one verdict.
    //
    // Three of the six are answered here, and they are the three that own
    // nothing: the barriers and the flush are proven no-ops, and the residency
    // pair is declined. None of them touches `seg.acc`, `seg.session` or
    // `seg.block`, which is what makes them separable from the rest of the rail
    // rather than merely convenient to move first. The encoder's records, the
    // fence pair and the sequencing SPI all still go through this device's own
    // decoder below: the first because its executor takes the flat
    // `compute::Command`, and the other two because the ledger has not settled
    // them and the protocol crate will not lift a row it has not settled.
    match reims_vgpu_protocol::closure::find(Rail::Compute, opcode)
        .and_then(reims_vgpu_core::operation::classify)
    {
        // `memoryBarrierWithResources:` and `memoryBarrierWithScope:`.
        Some(OperationHome::Stream(OperationClass::Barrier)) => {
            return note_compute_barrier(task_id, opcode, cmd_bytes);
        }
        // `insertCompressedTextureReinterpretationFlush`, the compute rail's
        // one content-representation directive.
        Some(OperationHome::Stream(OperationClass::ResourceState)) => {
            return note_compute_content_directive(task_id, opcode, cmd_bytes);
        }
        // `None` is a row the ledger has not settled, and the compute rail has
        // thirteen. Two of them are the unqualified `useHeaps:count:` and
        // `useResources:count:usage:` inherited from the serializer's encoder
        // base class, and they are declined rather than executed — so their
        // layout can come from `decode::residency::lift`, which answers what
        // the guest wrote without claiming the row is settled. The other eleven
        // feed live executors and stay below.
        None if reims_vgpu_protocol::decode::residency::is_residency(Rail::Compute, opcode) => {
            return note_compute_residency(task_id, opcode, cmd_bytes);
        }
        _ => {}
    }
    // The seventeen records the ledger has settled. `classify` put them here
    // and `protocol::decode::compute` owns their layout, so the class and the
    // fields are two answers from two owners instead of one `Kind` assigned
    // while the fields were being read.
    if matches!(
        reims_vgpu_protocol::closure::find(Rail::Compute, opcode)
            .and_then(reims_vgpu_core::operation::classify),
        Some(OperationHome::Stream(OperationClass::Compute))
    ) {
        let Some(framed) = frame_compute_record(task_id, opcode, cmd_bytes) else {
            return;
        };
        match reims_vgpu_protocol::decode::compute::decode(&framed) {
            Ok(record) => {
                let pipeline_ref = seg.acc.pipeline_ref;
                match compute_exec::apply_record(state, host, task_id, &record, seg) {
                    // `None` is an accumulator-only record, not a loss: the
                    // record was applied, `apply_record` simply had no
                    // execution status to report for it.
                    None | Some(ComputeStatus::Ok) => {}
                    Some(st) => note_compute_refusal(st, task_id, pipeline_ref, &record.kind()),
                }
            }
            Err(refusal) => {
                // The pass's dispatch type is the one enumerated field on this
                // encoder, and a word outside `MTLDispatchType` is refused at
                // the lift rather than folded onto its nearest neighbour. The
                // census slug the device used to raise when it substituted
                // `Serial` is kept, because what that counter was for — the
                // evidence that decides whether an unrecognised ordinal is a
                // guest asking for something new or this device reading the
                // wrong offset — is still exactly what a firing would mean.
                if matches!(
                    refusal,
                    reims_vgpu_protocol::decode::DecodeRefusal::UndefinedOrdinal {
                        field: "dispatch_type",
                        ..
                    }
                ) {
                    crate::runtime::drain::note_store_route("compute_dispatch_type_unknown");
                }
                note_compute_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
            }
        }
        return;
    }
    // What is left is the eleven rows the ledger has **not** settled — the
    // fence pair, the seven control-flow records and the two indirect-command
    // executions — and this device decodes them itself. They are not declines
    // like the blit rail's unsettled four: each one drives real work, on
    // evidence this project gathered outside the ledger, and holding them back
    // is what keeps a settled record and an unsettled one from sharing a
    // decoder.
    let cmd = match compute_spi::decode(cmd_bytes) {
        Ok(c) => c,
        // Same silent drop as the render path above.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("compute_decode", &status) {
                // Latched per (reason, opcode): the guest re-encodes the same
                // stream every frame, so an unclassified opcode would arrive
                // once per draw. Magnitude is the encoder's fail counter's job.
                e.field("opcode", format!("{:#x}", opcode))
                    .field("len", cmd_bytes.len())
                    .fail_once(opcode as u64);
            }
            return;
        }
    };
    match cmd.kind {
        ComputeKind::UpdateFence | ComputeKind::WaitFence => {
            let action = if cmd.kind == ComputeKind::UpdateFence {
                FenceAction::Update
            } else {
                FenceAction::Wait
            };
            fence_exec::execute_fence(
                state,
                task_id,
                FenceDomain::ComputeFence,
                cmd.fence_ref,
                action,
            );
        }
        ComputeKind::ControlStartDoWhile
        | ComputeKind::ControlEndDoWhile
        | ComputeKind::ControlStartWhile
        | ComputeKind::ControlEndWhile
        | ComputeKind::ControlStartIf
        | ComputeKind::ControlStartElse
        | ComputeKind::ControlEndIf => {
            // Denominator in front of the call, for the same reason as
            // `icb_exec_seen`: `compute_control_fail` only ever reaches the
            // always-on sink on a packet that already failed, so a control
            // record that works is unobservable and the rail reads as dead
            // whether it is dead or perfect.
            crate::runtime::drain::note_store_route("compute_ctrl_seen");
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_sequencing_record(state, host, task_id, &cmd, seg) {
                ComputeStatus::Ok => {}
                st => {
                    out.compute_control_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, &cmd.kind);
                }
            }
        }
        ComputeKind::ExecuteCommandsInBuffer | ComputeKind::ExecuteCommandsInBufferIndirect => {
            crate::runtime::drain::note_store_route("compute_icb_seen");
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_sequencing_record(state, host, task_id, &cmd, seg) {
                ComputeStatus::Ok => {}
                st => {
                    out.compute_icb_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, &cmd.kind);
                }
            }
        }
        // A settled row reaching this decoder is the routing above disagreeing
        // with the ledger, not a guest case.
        other => crate::observe::fail(format!(
            "compute_record_misrouted task={task_id} opcode={opcode:#x} kind={other:?} (the \
             ledger settled this row and it reached the unsettled decoder)"
        )),
    }
}

/// Frame one compute record for a protocol decoder, or say the frame disagreed
/// with itself.
///
/// `walk_segment_records` framed this record to hand it over, so a header that
/// does not frame here is two reads of one header disagreeing rather than a
/// guest case — which is why it is a `fail` and not a decline.
fn frame_compute_record<'a>(
    task_id: u32,
    opcode: u32,
    cmd_bytes: &'a [u8],
) -> Option<reims_vgpu_wire::op::Op<'a>> {
    match reims_vgpu_protocol::decode::op(cmd_bytes, 0) {
        Ok(framed) => Some(framed),
        Err(_) => {
            crate::observe::fail(format!(
                "compute_record_unframed task={task_id} opcode={opcode:#x} len={} (the segment \
                 walk framed this record and the op header did not)",
                cmd_bytes.len()
            ));
            None
        }
    }
}

/// Count and report one compute record a protocol decoder refused.
///
/// Latched, because the guest re-encodes the same stream every frame and an
/// unliftable record would otherwise be one line per segment. The token is the
/// opcode, widened by the offending value when the refusal names one: two
/// different undeclared dispatch types are two different questions about the
/// contract, and collapsing them onto the opcode would answer only the first.
///
/// The refusal renders its own rail and opcode, so neither is repeated here.
fn note_compute_record_refused(
    task_id: u32,
    opcode: u32,
    len: usize,
    refusal: &reims_vgpu_protocol::decode::DecodeRefusal,
) {
    use reims_vgpu_protocol::decode::DecodeRefusal;
    let token = match refusal {
        DecodeRefusal::UndefinedOrdinal { value, .. } => {
            u64::from(opcode) | (u64::from(*value) << 32)
        }
        _ => u64::from(opcode),
    };
    crate::observe::Emit::decline("compute_record", refusal)
        .field("task", task_id)
        .field("len", len)
        .fail_once(token);
}

/// Price one compute-encoder barrier, which this device answers by doing
/// nothing.
///
/// The no-op is sound at pass granularity and stronger than that on the Vulkan
/// rail, where `backend::vulkan::engine::exec_compute::execute_compute_inner`
/// begins, ends and submits one command buffer per dispatch — so consecutive
/// dispatches are separated by a queue submission rather than by a barrier
/// inside one. Under `-setSupportsComputePassDescriptorDispatchType:` Apple's
/// serializer emits a scope barrier after **every** dispatch and every ICB
/// execution of a serial pass, so this counter reading high is that capability
/// being on rather than a defect.
///
/// It is still decoded rather than skipped on the opcode: the barrier's shape
/// is what would have to be read the day the no-op stops being sound, and a
/// record whose body does not frame is a different event from one that does.
fn note_compute_barrier(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    let Some(framed) = frame_compute_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    match reims_vgpu_protocol::decode::sync::decode(Rail::Compute, &framed) {
        // One slug for both shapes, as before the ledger answered the class:
        // the claim being priced is "this device ordered nothing here", and it
        // is the same claim for a resource list and for a scope word.
        Ok(SyncRecord::Barrier(_)) => {
            crate::runtime::drain::note_store_route("compute_noop_barrier");
        }
        // `barrier_kind` answers for exactly the two barrier opcodes on this
        // rail, so the class and the lift agreeing is the protocol crate's
        // invariant. Unreachable, and named rather than ignored.
        Ok(_) => crate::observe::fail(format!(
            "compute_barrier_not_a_barrier task={task_id} opcode={opcode:#x} (the ledger calls \
             this a barrier and the lift produced another ordering record)"
        )),
        Err(refusal) => note_compute_record_refused(task_id, opcode, cmd_bytes.len(), &refusal),
    }
}

/// Price the compute rail's one content-representation directive.
///
/// `insertCompressedTextureReinterpretationFlush` names no resource and takes
/// no argument. It is a proven no-op here because this device never
/// materialises Apple's lossless-compression metadata, so there is no second
/// representation for the flush to make visible.
fn note_compute_content_directive(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    let Some(framed) = frame_compute_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    match reims_vgpu_protocol::decode::resource_state::decode(Rail::Compute, &framed) {
        Ok(record) => crate::runtime::drain::note_store_route(match record.directive {
            ContentDirective::FlushCompressedReinterpretation => {
                "compute_noop_flush_compressed_reinterpretation"
            }
            // The four directives that name a resource; they are the blit
            // rail's and do not arrive here. Healthy zeroes, kept apart from
            // the flush so a reading cannot mistake one for the other.
            ContentDirective::Synchronize => "compute_noop_synchronize",
            ContentDirective::OptimizeForCpu => "compute_noop_optimize_for_cpu",
            ContentDirective::OptimizeForGpu => "compute_noop_optimize_for_gpu",
            ContentDirective::InvalidateCompressed => "compute_noop_invalidate_compressed",
        }),
        Err(refusal) => note_compute_record_refused(task_id, opcode, cmd_bytes.len(), &refusal),
    }
}

/// A compute residency declaration whose usage the no-op argument does not
/// cover.
///
/// The reasoning and the vocabulary are the render rail's — see
/// [`report::ResidencyWriteDeclared`], which carries it — with one difference
/// this rail owns. The compute encoder inherits only the **unqualified** residency selectors,
/// so no record here carries a stage argument and there is no stage half to
/// report; the usage half is the whole declaration, and it is the half that
/// decides whether answering by doing nothing is sound.
struct ComputeResidencyDeclared {
    opcode: u32,
    count: usize,
    usage: reims_vgpu_protocol::residency::ResourceUsage,
}

impl crate::observe::Decline for ComputeResidencyDeclared {
    fn slug(&self) -> &'static str {
        use reims_vgpu_protocol::residency::UsageClass;
        match self.usage.classify() {
            UsageClass::Undeclared => "compute_residency_usage_undeclared",
            _ => "compute_residency_write_dropped",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("op", format!("{:#x}", self.opcode)),
            ("count", self.count.to_string()),
            ("usage", format!("{:#x}", self.usage.0)),
            (
                "undeclared_usage",
                format!("{:#x}", self.usage.undeclared_bits()),
            ),
        ]
    }
}

/// Price one compute residency declaration by what it declared.
///
/// This device resolves every binding per dispatch, so there is nothing for a
/// residency hint to keep resident and the answer is to do nothing. That
/// argument is load-bearing, and the census is what prices it: a declaration
/// that only reads is free to drop, while a dispatch writing through a path
/// this rail did not bind loses content the guest expects to read back, which
/// is not what a *hint* costs.
///
/// **The heap form carries no usage and the resource form always does**, and
/// that is now the record's own shape rather than a convention: the heap
/// selector has no usage argument at all, so there is no class to report and
/// nothing to weigh. It used to arrive as a zero in a `resource_usage` field
/// the decoder filled in for it, which made "declared nothing" and "declared
/// no usage" the same word.
fn note_compute_residency(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    use reims_vgpu_protocol::decode::residency::ResidencySubject;
    use reims_vgpu_protocol::residency::UsageClass;

    let Some(framed) = frame_compute_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    // `lift` rather than `decode`: the row is unresolved, so `decode` refuses
    // it on principle and is right to. What this arm wants is the layout
    // question answered on its own — it is a census, not a claim that the
    // contract is settled.
    let record = match reims_vgpu_protocol::decode::residency::lift(Rail::Compute, &framed) {
        Ok(record) => record,
        Err(refusal) => {
            note_compute_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
            return;
        }
    };
    let usage = match (record.subject, record.usage) {
        (ResidencySubject::Heaps, _) => {
            crate::runtime::drain::note_store_route("compute_residency_heap");
            return;
        }
        (ResidencySubject::Resources, Some(usage)) => usage,
        // Unreachable on this rail: both of its residency selectors are
        // unqualified, and the resource form of that pair carries a usage word.
        // Named rather than folded onto an empty usage, which would report a
        // declaration the guest did not make.
        (ResidencySubject::Resources, None) => {
            crate::observe::fail(format!(
                "compute_residency_without_usage task={task_id} opcode={opcode:#x} (a resource \
                 residency declaration on this rail carries a usage word and this one did not)"
            ));
            return;
        }
    };
    let class = usage.classify();
    crate::runtime::drain::note_store_route(match class {
        UsageClass::Empty => "compute_residency_empty",
        UsageClass::ReadOnly => "compute_residency_read",
        UsageClass::Writes => "compute_residency_write",
        UsageClass::Undeclared => "compute_residency_undeclared",
    });
    if matches!(class, UsageClass::Empty | UsageClass::ReadOnly) {
        return;
    }
    // Latched on the declaration: the same kernel asks for the same thing every
    // frame, and a second shape is the event.
    let decline = ComputeResidencyDeclared {
        opcode,
        count: record.refs.len(),
        usage,
    };
    crate::observe::Emit::decline("compute_residency", &decline).fail_once(u64::from(usage.0));
}

/// An indirect-command-buffer record this rail decoded and did not apply.
///
/// Two slugs rather than one, because the two losses are not the same loss: a
/// dropped `resetCommandsInBuffer:` leaves commands live that the guest retired,
/// and a dropped `copyIndirectCommandBuffer:` leaves the destination holding
/// whatever it held before. One slug for both is exactly the collapse
/// [`crate::observe::Decline`]'s own doc refuses — you watch it fire and still
/// cannot tell which buffer is wrong.
struct IcbRecordDropped(u32);

impl crate::observe::Decline for IcbRecordDropped {
    fn slug(&self) -> &'static str {
        match self.0 {
            wire_blit::OPCODE_RESET_ICB => "blit_icb_reset_dropped",
            wire_blit::OPCODE_COPY_ICB => "blit_icb_copy_dropped",
            // `0x138` cannot arrive: the optimize hint is answered by the no-op
            // arm before this one. So this names a record that reached an ICB
            // kind without being one of the three, which would be a decoder bug
            // rather than a dropped command. A healthy zero.
            _ => "blit_icb_unclassified",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("opcode", format!("{:#x}", self.0))]
    }
}

/// A texture fill this rail decoded and did not apply.
///
/// Two slugs, on the same reasoning as [`IcbRecordDropped`]: the colour form
/// and the staged-bytes form are lost the same way but cost different things to
/// implement. The colour form needs a clear-colour-to-pixel-format converter
/// this device does not have; the bytes form needs the staging buffer read and
/// the pattern tiled across the region, and nothing converted. A single count
/// could not tell which of those a driven boot is asking for.
struct TextureFillDropped(blit_spi::FillSource);

impl crate::observe::Decline for TextureFillDropped {
    fn slug(&self) -> &'static str {
        match self.0 {
            blit_spi::FillSource::Color => "blit_fill_texture_color_dropped",
            blit_spi::FillSource::Bytes => "blit_fill_texture_bytes_dropped",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("source", format!("{:?}", self.0))]
    }
}

fn handle_blit_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
) {
    // `walk_blit_us` charges this rail 33.3 s of a 45 s driven Maps window and
    // every clock inside `execute_blit` accounts for 0.14 s of it. The gap has
    // to be in this function, and only two things here are outside that call:
    // the decode above, and the `Fence` arm, which reaches
    // `execute_blit_fence` directly rather than through `execute_blit`. A
    // blocking fence wait costs exactly what is missing and does no work while
    // it costs it, which is why no copy clock can see it.
    //
    // Timed at the closure `walk_segment_records` calls, so decode is inside the
    // span and no arm can leave without being charged.
    let record_started = std::time::Instant::now();
    // **Which class of record this is, is the closure ledger's answer.** The
    // blit *rail* carries five: transfers, fences, indirect-command mutations,
    // content-representation directives, and the mipmap generation that sits
    // with the transfers. This device answered it with its own `Kind`, decoded
    // in the same pass that read the fields — so a record's class and its
    // layout were one verdict and a misclassification read fields at the wrong
    // offsets. `reims_vgpu_core::operation::classify` answers the class from
    // the ledger row, and each class is then lifted by the protocol decoder
    // that owns its layout.
    //
    // Three of the five are lifted that way here. The transfers still go
    // through this device's own decoder: their executor takes the flat
    // `blit::Command` and moving it to `BlitRecord` is its own step.
    let framed = match reims_vgpu_protocol::decode::op(cmd_bytes, 0) {
        Ok(framed) => framed,
        Err(_) => {
            // `walk_segment_records` framed this record to hand it over, so a
            // header that does not frame here is a frame disagreeing with
            // itself between two reads rather than a guest case.
            crate::observe::fail(format!(
                "blit_record_unframed task={task_id} opcode={opcode:#x} len={} (the segment walk \
                 framed this record and the op header did not)",
                cmd_bytes.len()
            ));
            return;
        }
    };
    let class = reims_vgpu_protocol::closure::find(Rail::Blit, opcode)
        .and_then(reims_vgpu_core::operation::classify);
    let record_class = match class {
        Some(OperationHome::Stream(class)) => Some(class),
        // The blit rail has no object-lifecycle records — that home is the root
        // rail's whole side — and `None` is a row the ledger has not settled.
        // Both fall to the unclassified arm below rather than being read as
        // some class they are not.
        Some(OperationHome::ObjectLifecycle) | None => None,
    };
    match record_class {
        // The content-representation directives: optimize for CPU or GPU,
        // synchronize, invalidate the compression metadata. Nine opcodes across
        // the rails and three payload shapes between them, and
        // `reims_vgpu_protocol::resource_state::content_request` is the one
        // place that maps an opcode to a directive and a target.
        //
        // Every one is a no-op on this device and each for a stated reason, so
        // what this arm owes is a census that keeps them apart. It used to be
        // keyed on this device's `Kind`, which folded the four directives into
        // two variants — `Resource` and `Image` — plus a third arm for the
        // compressed invalidate; the directive's own name is the key now, so a
        // reading can say which directive a workload issues rather than which
        // wire shape it used.
        Some(OperationClass::ResourceState) => {
            match reims_vgpu_protocol::decode::resource_state::decode(Rail::Blit, &framed) {
                Ok(record) => {
                    // `optimize*`/`synchronize*` are protocol no-ops on the
                    // unified-memory path: this device writes the guest's pages
                    // directly, so there is no second representation to move
                    // content between. `invalidateCompressedTexture:` is the
                    // same statement about Apple's lossless-compression
                    // metadata, which this device never materialises.
                    crate::runtime::drain::note_store_route(match record.directive {
                        ContentDirective::Synchronize => "blit_noop_synchronize",
                        ContentDirective::OptimizeForCpu => "blit_noop_optimize_for_cpu",
                        ContentDirective::OptimizeForGpu => "blit_noop_optimize_for_gpu",
                        ContentDirective::InvalidateCompressed => "blit_noop_invalidate_compressed",
                        // The compute encoder's flush; it names no resource and
                        // does not arrive on this rail. A healthy zero.
                        ContentDirective::FlushCompressedReinterpretation => {
                            "blit_noop_flush_compressed_reinterpretation"
                        }
                    });
                }
                Err(refusal) => {
                    note_blit_record_refused(task_id, opcode, cmd_bytes.len(), &refusal)
                }
            }
        }
        Some(OperationClass::Fence) => {
            match reims_vgpu_protocol::decode::sync::decode(Rail::Blit, &framed) {
                Ok(SyncRecord::Fence(record)) => {
                    // Log from the *blit* status, before the remap. The remap
                    // folds two meanings into `FenceStatus::Missing` — an absent
                    // object and a zero fence ref — and only the blit rail's own
                    // reason can tell them apart.
                    let blit_st = blit_exec::execute_blit_fence(state, task_id, &record);
                    if let Some(e) = crate::observe::Emit::refusal("blit_fence_fail", &blit_st) {
                        e.field("opcode", format!("{opcode:#x}")).fail();
                    }
                }
                // `fence_kind` answers for exactly the two fence opcodes on this
                // rail, so the class and the lift agreeing is the protocol
                // crate's invariant. Unreachable, and named rather than ignored.
                Ok(_) => crate::observe::fail(format!(
                    "blit_fence_not_a_fence task={task_id} opcode={opcode:#x} (the ledger calls \
                     this a fence and the lift produced another ordering record)"
                )),
                Err(refusal) => {
                    note_blit_record_refused(task_id, opcode, cmd_bytes.len(), &refusal)
                }
            }
        }
        // The three indirect-command-buffer records. All three used to be
        // refused before decode under one shared reason, which said three
        // different things with one word — and only two of them are losses.
        //
        // `optimizeIndirectCommandBuffer:` is Metal's hint that a range will be
        // reused, so skipping it is semantically correct and costs speed alone;
        // it is counted so the census still shows the traffic. The other two
        // change what a later `executeCommandsInBuffer:` will run: a reset the
        // device drops leaves commands live that the guest retired, and a copy
        // it drops leaves the destination holding whatever it held before. Both
        // are stale commands executing, which is worse than a dropped one, so
        // they stay fail-visible as well as counted.
        //
        // Counted rather than executed on purpose. `runtime::icb` materializes
        // host ICBs on the Metal arm only, and it reads 0.00% on a driven x86
        // boot — so the count is what says whether an executor is worth building,
        // and for which of the two.
        Some(OperationClass::IndirectCommand) => {
            match reims_vgpu_protocol::decode::icb::decode(Rail::Blit, &framed) {
                Ok(record) => note_blit_icb_dropped(task_id, opcode, &record),
                Err(refusal) => {
                    note_blit_record_refused(task_id, opcode, cmd_bytes.len(), &refusal)
                }
            }
        }
        // The transfers, and — deliberately in the same arm — the rows the
        // ledger has **not settled**.
        //
        // `classify` answers `None` for an unresolved row, because a model that
        // promises ordering and completion for everything in its vocabulary may
        // not admit an operation it cannot describe. This device is not that
        // model: it decodes those records and *declines* them, and the decline's
        // count is the evidence that will settle the row. So an unsettled row
        // goes to this rail's own decoder, which is the only thing that can
        // name it — routing it to a generic line instead would delete the
        // instrument that says how much of it a workload issues.
        //
        // The two are one arm because they share one decoder. Which of them a
        // record is, is still asked: the decline arms inside are exactly the
        // unresolved rows.
        // The transfers. Nine records lifted by the protocol decoder that owns
        // their layouts, and executed from the enum — the record's class and
        // its field offsets are no longer one verdict.
        Some(OperationClass::Blit) => handle_blit_transfer_record(
            state,
            host,
            task_id,
            opcode,
            &framed,
            cmd_bytes.len(),
            record_started,
        ),
        // The rows the ledger has **not settled**.
        //
        // `classify` answers `None` for an unresolved row, because a model that
        // promises ordering and completion for everything in its vocabulary may
        // not admit an operation it cannot describe. This device is not that
        // model: it decodes those records and *declines* them, and the decline's
        // count is the evidence that will settle the row. So an unsettled row
        // goes to this rail's own decoder, which is the only thing that can
        // name it — routing it to a generic line instead would delete the
        // instrument that says how much of it a workload issues.
        None => handle_blit_unsettled_record(task_id, opcode, cmd_bytes),
        // A class the blit rail does not carry at all. `classify` maps this
        // rail's settled opcodes onto four classes and this is none of them, so
        // reaching here means `classify` and this dispatch disagree about which
        // records the rail carries.
        Some(other) => {
            crate::observe::fail(format!(
                "blit_record_wrong_class task={task_id} opcode={opcode:#x} len={} class={} (the \
                 ledger gives this record a class the blit rail does not carry, so no decoder \
                 here owns its layout)",
                cmd_bytes.len(),
                other.name()
            ));
        }
    }
    if !matches!(record_class, Some(OperationClass::Blit)) {
        // The transfer arm charges its own clock inside
        // `handle_blit_transfer_record`, which is the only arm whose bucket
        // depends on which record it lifted.
        let route = match record_class {
            Some(OperationClass::Fence) => "blitrec_fence_us",
            Some(OperationClass::ResourceState) => "blitrec_noop_us",
            _ => "blitrec_other_us",
        };
        crate::runtime::drain::note_store_route_us(
            route,
            record_started.elapsed().as_micros() as u64,
        );
        crate::runtime::drain::note_store_route(match record_class {
            Some(OperationClass::Fence) => "blitrec_fence_n",
            Some(OperationClass::ResourceState) => "blitrec_noop_n",
            _ => "blitrec_other_n",
        });
    }
}

/// Report a blit-rail record whose bytes are not the record its class names.
///
/// One site, because every class above loses the same thing when its lift
/// refuses — the guest's command, with no line unless one is written here — and
/// the refusal itself already says which check refused, on which rail, at which
/// opcode.
fn note_blit_record_refused(
    task_id: u32,
    opcode: u32,
    len: usize,
    refusal: &reims_vgpu_protocol::decode::DecodeRefusal,
) {
    crate::observe::Emit::decline("blit_record", refusal)
        .field("task", task_id)
        .field("opcode", format!("{opcode:#x}"))
        .field("len", len)
        .fail();
}

/// Count and report one indirect-command-buffer record this rail did not apply.
fn note_blit_icb_dropped(
    task_id: u32,
    opcode: u32,
    record: &reims_vgpu_protocol::decode::icb::IcbRecord,
) {
    use crate::observe::Decline as _;
    use reims_vgpu_protocol::decode::icb::IcbRecord;
    // The hint is semantically free to skip; the two mutations are not.
    if matches!(record, IcbRecord::Optimize { .. }) {
        crate::runtime::drain::note_store_route("blit_noop_icb_optimize");
        return;
    }
    let decline = IcbRecordDropped(opcode);
    crate::runtime::drain::note_store_route(decline.slug());
    crate::observe::Emit::decline("blit_icb", &decline)
        .field("task", task_id)
        .field("record", format!("{record:?}"))
        .fail_once(u64::from(opcode));
}

/// The transfers: nine records that move bytes, lifted by the layer that owns
/// their layouts.
///
/// The nine field orders do not rhyme — a buffer copy puts both refs first, a
/// texture-to-buffer copy narrows its options word to sixteen bits where its
/// sibling uses thirty-two, and the region copy's `options:` form is a
/// different opcode at a different length. None of that is derivable from the
/// selector, so all of it belongs to `reims_vgpu_protocol::decode::blit`, whose
/// fields come from the pinned wire views. What is left here is the routing:
/// which executor a lifted record reaches, and what is said when one refuses.
///
/// `generateMipmapsForTexture:` is a transfer by opcode and a filter chain by
/// execution, so it is answered here and not by `execute_blit` — the record
/// names one texture and `runtime::mipmap` owns the chain.
fn handle_blit_transfer_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    opcode: u32,
    framed: &reims_vgpu_wire::op::Op<'_>,
    cmd_len: usize,
    record_started: std::time::Instant,
) {
    let record = match reims_vgpu_protocol::decode::blit::decode(framed) {
        Ok(record) => record,
        Err(refusal) => {
            note_blit_record_refused(task_id, opcode, cmd_len, &refusal);
            return;
        }
    };
    if let BlitRecord::GenerateMipmaps(m) = record {
        match mipmap::generate_mipmaps_linear(state, host, task_id, m.texture_ref) {
            MipmapStatus::Ok => {}
            st => {
                // Was `st={st:?}` with no `reason=` at all, so none of the
                // eight outcomes was greppable and the Debug spelling was
                // the only handle on which check refused.
                if let Some(e) = crate::observe::Emit::refusal("blit_generate_mipmaps", &st) {
                    e.field("resource", m.texture_ref).fail();
                }
            }
        }
    } else {
        match blit_exec::execute_blit(state, host, task_id, &record) {
            BlitStatus::Ok | BlitStatus::ZeroExtent => {}
            st => {
                // Icon/upload path often uses blit copies; fail-visible for RE.
                // The reason names the specific failing site inside blit_exec
                // that produced the coarse `st` — 177 checks collapse into six
                // classes, so the class alone says almost nothing. It rides in
                // the value: `BlitStatus::Failed` has nowhere to put a refusal
                // that did not name its check, so there is no site here that
                // can render a bare `reason=`.
                //
                // The endpoints and extents are the record's own Debug rather
                // than a hand-picked list of fields: the fields a record has
                // are the fields it has, and a list written here could name one
                // the record does not carry.
                let (src_ref, dst_ref) = record.refs();
                let object_type = |r: Option<u32>| {
                    r.and_then(|r| objects::lookup_list_entry(state, host, task_id, r))
                        .map_or(0, |e| e.object_type)
                };
                let src_ty = object_type(src_ref);
                let dst_ty = object_type(dst_ref);
                if let Some(e) = crate::observe::Emit::refusal("blit_fail", &st) {
                    e.field("st", format!("{st:?}"))
                        .field("kind", format!("{:?}", record.kind()))
                        .field("opcode", format!("{opcode:#x}"))
                        .field("src_ty", src_ty)
                        .field("dst_ty", dst_ty)
                        .field("record", format!("{record:?}"))
                        .fail();
                }
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        transfer_route(record.kind(), "us"),
        record_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route(transfer_route(record.kind(), "n"));
}

/// The census bucket a lifted transfer is charged to.
///
/// One function for the microsecond clock and the count, because the two used
/// to be written as two matches over the same tag and a record could be counted
/// in one bucket and timed in another.
fn transfer_route(kind: reims_vgpu_protocol::blit::BlitKind, suffix: &str) -> &'static str {
    use reims_vgpu_protocol::blit::BlitKind;
    match (kind, suffix) {
        (BlitKind::FillBuffer | BlitKind::FillBufferPattern4, "us") => "blitrec_fill_us",
        (BlitKind::FillBuffer | BlitKind::FillBufferPattern4, _) => "blitrec_fill_n",
        (BlitKind::GenerateMipmaps, "us") => "blitrec_noop_us",
        (BlitKind::GenerateMipmaps, _) => "blitrec_noop_n",
        (_, "us") => "blitrec_copy_us",
        (_, _) => "blitrec_copy_n",
    }
}

/// The rows the closure ledger has **not settled**, which this device decodes
/// for itself and declines.
///
/// This is not a wiring step waiting to happen. An unresolved row has no
/// established contract, so no layer above the wire may give it a shape — and
/// this device's decline of it, counted per record, is the measurement that
/// settles it. The indirect-command-buffer reset and copy and the two texture
/// fills are here for that reason and not because nothing has got round to
/// them.
fn handle_blit_unsettled_record(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    let cmd = match blit_spi::decode(cmd_bytes) {
        Ok(c) => c,
        // Was `Err(_) => return`: a decoded blit record dropped with no line at
        // all, which on a live boot is indistinguishable from a segment that
        // carried no blit work. The status names which of the checks refused.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("blit_decode", &status) {
                e.field("opcode", format!("{opcode:#x}"))
                    .field("len", cmd_bytes.len())
                    .fail();
            }
            return;
        }
    };
    use crate::observe::Decline as _;
    match cmd {
        // The indirect-command-buffer records the ledger has not settled: the
        // reset and the copy. Both change what a later `executeCommandsInBuffer:`
        // will run — a reset the device drops leaves commands live that the
        // guest retired, and a copy it drops leaves the destination holding
        // whatever it held before. Both are stale commands executing, which is
        // worse than a dropped one, so they are fail-visible as well as
        // counted.
        //
        // `optimizeIndirectCommandBuffer:` is not here: its row *is* settled —
        // a proven no-op, because the hint costs speed alone — so the caller
        // lifts it through `reims_vgpu_protocol::decode::icb` and counts it
        // there.
        //
        // Counted rather than executed on purpose. `runtime::icb` materializes
        // host ICBs on the Metal arm only, and it reads 0.00% on a driven x86
        // boot — so the count is what says whether an executor is worth building,
        // and for which of the two.
        blit_spi::UnsettledRecord::IcbMutation {
            range_location,
            range_length,
            ..
        } => {
            let decline = IcbRecordDropped(opcode);
            crate::runtime::drain::note_store_route(decline.slug());
            crate::observe::Emit::decline("blit_icb", &decline)
                .field("task", task_id)
                .field("range_loc", range_location)
                .field("range_len", range_length)
                .fail_once(u64::from(opcode));
        }
        // `fillTexture:…:color:` and `fillTexture:…:bytes:length:`. These are
        // writes the guest expects to land, so a dropped one leaves the region
        // holding what it held before and the guest reads back content it
        // believes it just wrote. Counted and fail-visible, with the extent
        // named, because the extent is what decides whether an executor is
        // worth building.
        //
        // Not executed here on purpose. A texture fill needs the destination
        // resolved through the backing/5/11 rails, the region walked per row,
        // and — for the colour form — the clear colour converted into the
        // texture's pixel format, which is a converter this device does not
        // have. The count is what says whether to build one, and for which of
        // the two sources.
        blit_spi::UnsettledRecord::TextureFill {
            source,
            texture,
            level,
            slice,
            size,
            ..
        } => {
            let decline = TextureFillDropped(source);
            crate::runtime::drain::note_store_route(decline.slug());
            crate::observe::Emit::decline("blit_fill_texture", &decline)
                .field("task", task_id)
                .field("texture", texture)
                .field("level", level)
                .field("slice", slice)
                .field(
                    "extent",
                    format!("{}x{}x{}", size.width, size.height, size.depth),
                )
                .fail_once(u64::from(opcode));
        }
    }
}

/// Frame one render record for a protocol decoder, or say the frame disagreed
/// with itself.
fn frame_render_record<'a>(
    task_id: u32,
    opcode: u32,
    cmd_bytes: &'a [u8],
) -> Option<reims_vgpu_wire::op::Op<'a>> {
    match reims_vgpu_protocol::decode::op(cmd_bytes, 0) {
        Ok(framed) => Some(framed),
        Err(_) => {
            crate::observe::fail(format!(
                "render_record_unframed task={task_id} opcode={opcode:#x} len={} (the segment \
                 walk framed this record and the op header did not)",
                cmd_bytes.len()
            ));
            None
        }
    }
}

/// Count and report one render record a protocol decoder refused.
///
/// Latched on the opcode: the guest re-encodes the same stream every frame, and
/// this rail is the hottest in the crate.
fn note_render_record_refused(
    task_id: u32,
    opcode: u32,
    len: usize,
    refusal: &reims_vgpu_protocol::decode::DecodeRefusal,
) {
    crate::observe::Emit::decline("render_record", refusal)
        .field("task", task_id)
        .field("len", len)
        .fail_once(u64::from(opcode));
}

/// Order one render-encoder fence.
///
/// The encoder numbers its own fences and the three rails' pairs are nowhere
/// near each other, so the rail is what says which pair an opcode belongs to.
/// This arm used to read the direction from the opcode a second time and carry
/// an `_ =>` for the case where a record classed as a fence was neither an
/// update nor a wait — a state `reims_vgpu_protocol::sync::fence_kind` makes
/// unrepresentable, because the same function decides both that this *is* a
/// fence and which side of one it is.
fn handle_render_fence(state: &mut DeviceState, task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    match reims_vgpu_protocol::decode::sync::decode(Rail::Render, &framed) {
        Ok(SyncRecord::Fence(record)) => {
            // `updateFence:afterStages:` is what a guest uses to order work
            // inside one render encoder against a later one, so a dropped one
            // is lost encoder synchronisation on every pass that asked for it.
            let action = match record.kind {
                reims_vgpu_protocol::sync::FenceKind::Update => FenceAction::Update,
                reims_vgpu_protocol::sync::FenceKind::Wait => FenceAction::Wait,
            };
            fence_exec::execute_fence(
                state,
                task_id,
                FenceDomain::RenderFence,
                record.fence_ref,
                action,
            );
        }
        // `fence_kind` answers for exactly the two fence opcodes on this rail,
        // so the class and the lift agreeing is the protocol crate's invariant.
        // Unreachable, and named rather than ignored.
        Ok(_) => crate::observe::fail(format!(
            "render_fence_not_a_fence task={task_id} opcode={opcode:#x} (the ledger calls this a \
             fence and the lift produced another ordering record)"
        )),
        Err(refusal) => note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal),
    }
}

/// Price one render-encoder barrier, which this device answers by doing
/// nothing.
///
/// One slug for all three shapes, as before the ledger answered the class: the
/// claim being priced is "this device ordered nothing here", and it is the same
/// claim for a resource list, a scope word and `textureBarrier`.
fn note_render_barrier(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    match reims_vgpu_protocol::decode::sync::decode(Rail::Render, &framed) {
        Ok(SyncRecord::Barrier(_)) => {
            crate::runtime::drain::note_store_route("render_noop_barrier");
        }
        Ok(_) => crate::observe::fail(format!(
            "render_barrier_not_a_barrier task={task_id} opcode={opcode:#x} (the ledger calls \
             this a barrier and the lift produced another ordering record)"
        )),
        Err(refusal) => note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal),
    }
}

/// Price one render residency declaration by what it declared.
///
/// **A heap declaration carries no usage and a stage-less form carries no
/// stages**, and both are now the record's own shape rather than zeros the
/// decoder filled in. The flat command had a `residency_usage` and a
/// `residency_stages` on every record, so "the guest declared nothing" and "the
/// selector has no such argument" were the same word on the counters whose
/// whole job is telling them apart — and this rail has all four forms, the
/// qualified pair that carries stages and the unqualified pair inherited from
/// the encoder base class that does not.
fn note_render_residency(task_id: u32, opcode: u32, cmd_bytes: &[u8]) {
    use reims_vgpu_protocol::decode::residency::ResidencySubject;

    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    // `lift` rather than `decode`: every residency row is unresolved, so
    // `decode` refuses them on principle and is right to. This arm wants the
    // layout question answered on its own — it is a census, not a claim that
    // the contract is settled.
    match reims_vgpu_protocol::decode::residency::lift(Rail::Render, &framed) {
        Ok(record) => note_residency_declaration(
            task_id,
            matches!(record.subject, ResidencySubject::Heaps),
            opcode,
            record.refs.len(),
            record.usage,
            record.stages,
        ),
        Err(refusal) => note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal),
    }
}

/// Whether this record's whole effect is one field of [`StreamAccum`] that no
/// other record class writes.
///
/// **This is the group's disjointness claim, and it is a claim about
/// `StreamAccum` rather than about the wire.** The render encoder's records all
/// land in one accumulator, so "touches no stream state" — the test W8 and W10
/// could apply to their classes — cannot separate anything here. What separates
/// these is that each is the *sole writer* of the field it sets: nothing else
/// on the rail assigns `pipeline_ref`, `cull_mode`, `viewports`, `visibility`
/// or the rest, so the record's decoder can change without any other record
/// class being able to disagree with it about a value. `bind_snapshot` reads
/// them all and reads the same value whichever decoder produced it.
///
/// The store actions fail that test and are not here: `SetStoreAction` mutates
/// `color_slots`, `depth_attach` and `stencil_attach`, which the pass
/// descriptor writes, so those three move as one group with the descriptor.
/// The binds write the six bind tables and `unrepresentable`, and the draws
/// read the whole accumulator and push onto `draws`; both are later groups.
///
/// Exhaustive rather than `_ => false`, so a kind added to the contract has to
/// be classified here instead of silently joining the legacy path.
const fn is_render_stream_state(kind: ProtoRenderKind) -> bool {
    use ProtoRenderKind as K;
    match kind {
        K::SetRenderPipelineState
        | K::SetDepthStencilState
        | K::SetStencilReference
        | K::SetBlendColor
        | K::SetCullMode
        | K::SetFrontFacingWinding
        | K::SetDepthClipMode
        | K::SetTriangleFillMode
        | K::SetDepthBias
        | K::SetLineWidth
        | K::SetViewport
        | K::SetViewports
        | K::SetScissorRect
        | K::SetScissorRects
        | K::SetVisibilityResultMode => true,

        K::Draw
        | K::DrawWide
        | K::DrawInstanced
        | K::DrawInstancedWide
        | K::DrawInstancedBase
        | K::DrawInstancedBaseWide
        | K::DrawIndexed
        | K::DrawIndexedWide
        | K::DrawIndexedInstanced
        | K::DrawIndexedInstancedWide
        | K::DrawIndexedInstancedBase
        | K::DrawIndexedInstancedBaseWide
        | K::DrawIndirect
        | K::DrawIndexedIndirect
        | K::WriteDescriptor
        | K::SetColorStoreAction
        | K::SetDepthStoreAction
        | K::SetStencilStoreAction
        | K::SetVertexBuffers
        | K::SetVertexBuffersWithStride
        | K::SetVertexBufferOffset
        | K::SetVertexBufferOffsetStride
        | K::SetVertexSamplers
        | K::SetVertexSamplersWithLod
        | K::SetVertexTextures
        | K::SetFragmentBuffers
        | K::SetFragmentBufferOffset
        | K::SetFragmentSamplers
        | K::SetFragmentSamplersWithLod
        | K::SetFragmentTextures => false,
    }
}

/// Apply one render-encoder state record, lifted by the protocol crate.
///
/// Every arm here is an assignment to one accumulator field, which is what
/// [`is_render_stream_state`] selected on. The three ordinals this device does
/// not parse — cull mode, winding, depth-clip and fill mode — stay raw for the
/// reason they always have: only the running rail knows whether it can spell
/// the answer, so only the rail can refuse one by name.
fn handle_render_stream_state(task_id: u32, opcode: u32, cmd_bytes: &[u8], acc: &mut StreamAccum) {
    use reims_vgpu_protocol::decode::render::RenderRecord;

    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    let record = match reims_vgpu_protocol::decode::render::decode(&framed) {
        Ok(record) => record,
        Err(refusal) => {
            return note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
        }
    };
    // The ordinal a raw state record carries, narrowed the way this device has
    // always narrowed them: `u32::MAX` for a word that does not fit, so a wide
    // value reaches the backend as an out-of-contract number that says its own
    // name rather than as its own low half — which for a multiple of 2^32 would
    // be the *default*, the one answer that renders with nothing in the log.
    let narrow = |mode: u64| u32::try_from(mode).unwrap_or(u32::MAX);

    match record {
        RenderRecord::SetPipeline(r) => {
            // Applied whatever the ref, ref 0 included: `acc.pipeline_ref == 0`
            // is a state the draw arm knows and declines under
            // `dropped_no_pipeline`, where dropping the record would leave the
            // *previous* pipeline latched and encode the next draw against it.
            if r.pipeline_ref == 0 && crate::observe::first_sight("render_set_pipeline_zero", 0) {
                crate::observe::fail(
                    "stream_set_pipeline reason=render_set_pipeline_zero_ref \
                     (a render pipeline was set to ref 0; the pass is now unbound \
                     and its draws decline as dropped_no_pipeline)",
                );
            }
            acc.pipeline_ref = r.pipeline_ref;
        }
        RenderRecord::SetDepthStencilState(r) => acc.depth_stencil_ref = r.state_ref,
        RenderRecord::SetStencilReference(r) => acc.stencil_ref = Some((r.front, r.back)),
        RenderRecord::SetBlendColor(r) => {
            acc.blend_color = Some([
                f32::from_bits(r.red_bits),
                f32::from_bits(r.green_bits),
                f32::from_bits(r.blue_bits),
                f32::from_bits(r.alpha_bits),
            ]);
        }
        RenderRecord::SetCullMode(r) => acc.cull_mode = Some(narrow(r.mode)),
        RenderRecord::SetFrontFacingWinding(r) => acc.front_facing = Some(narrow(r.winding)),
        RenderRecord::SetDepthClipMode(r) => acc.depth_clip_mode = Some(narrow(r.mode)),
        RenderRecord::SetTriangleFillMode(r) => acc.fill_mode = Some(narrow(r.mode)),
        RenderRecord::SetDepthBias(r) => {
            acc.depth_bias = Some([
                f32::from_bits(r.bias_bits),
                f32::from_bits(r.slope_scale_bits),
                f32::from_bits(r.clamp_bits),
            ]);
        }
        // `setLineWidth:` alone. `setTessellationFactorScale:` shares this wire
        // form and *not* this record — the ledger has not settled it, so it has
        // no `RenderKind` and never reaches here. It stays on this device's own
        // decoder with its own census, which is exactly the split the two
        // selectors deserve: one is state a rail may be able to carry, the other
        // has no carrier on either.
        RenderRecord::SetLineWidth(r) => acc.line_width = Some(f32::from_bits(r.width_bits)),
        RenderRecord::SetViewports(ports) => {
            // A plural record of count zero. The singular form is this slice at
            // length one, so an empty one can only be the plural, and it is the
            // one shape where "replace the state" and "the guest bound none"
            // are the same assignment for opposite reasons. The previous state
            // stands and the record is named, which is the reading the legacy
            // decoder's `ErrBadLength` had without saying which record it was.
            if ports.is_empty() {
                return note_empty_viewport_or_scissor(task_id, "viewport", opcode);
            }
            acc.viewports.clear();
            acc.viewports
                .extend(ports.iter().map(render_pass::viewport_from_wire));
        }
        RenderRecord::SetScissorRects(rects) => {
            if rects.is_empty() {
                return note_empty_viewport_or_scissor(task_id, "scissor", opcode);
            }
            // All-or-nothing on an empty rect: `setScissorRects:count:` replaces
            // the state atomically and slot order is what a shader's
            // `[[viewport_array_index]]` selects, so an array cannot be adopted
            // with the empty slots left out, and adopting them as written would
            // make exactly those slots clip however the backend reads a zero
            // rect.
            let lifted: Vec<ScissorRect> =
                rects.iter().map(render_pass::scissor_from_wire).collect();
            match lifted.iter().find(|r| r.is_empty()) {
                Some(empty) => note_empty_scissor(task_id, *empty),
                None => acc.scissors = lifted,
            }
        }
        RenderRecord::SetVisibilityResultMode(r) => {
            // `MTLVisibilityResultModeDisabled` is 0, and it is the guest
            // disarming the query rather than an unknown value: subsequent draws
            // simply carry none.
            acc.visibility = (r.mode != 0).then(|| draw::VisibilityArming {
                mode: narrow(r.mode),
                offset: r.offset,
            });
        }
        // `is_render_stream_state` selected these fifteen kinds and
        // `decode::render` maps each of them to the arm above. Arriving here is
        // the two disagreeing rather than a guest case, so it is named instead
        // of being answered a second time.
        other => crate::observe::fail(format!(
            "render_state_record_not_state task={task_id} opcode={opcode:#x} \
             (the rail routed this row to the stream-state arm and the lift \
             produced {other:?})"
        )),
    }
}

/// Whether this record binds into one of the six argument tables, or moves an
/// offset inside one.
///
/// The second group of the render encoder's cutover. Its fields are the six
/// bind tables and `unrepresentable` — and `unrepresentable` is shared with the
/// pass descriptor, which is why the claim here is not the sole-writer one
/// [`is_render_stream_state`] makes. It is the weaker one that still forbids a
/// disagreement: `unrepresentable` is a **first-wins latch** whose insert order
/// is the stream's own record order, which no change of decoder can alter, so
/// two writers cannot produce two answers the way two writers of a value could.
///
/// Exhaustive over `RenderKind` for the reason [`is_render_stream_state`] is.
const fn is_render_bind(kind: ProtoRenderKind) -> bool {
    use ProtoRenderKind as K;
    match kind {
        K::SetVertexBuffers
        | K::SetVertexBuffersWithStride
        | K::SetVertexBufferOffset
        | K::SetVertexBufferOffsetStride
        | K::SetVertexTextures
        | K::SetVertexSamplers
        | K::SetVertexSamplersWithLod
        | K::SetFragmentBuffers
        | K::SetFragmentBufferOffset
        | K::SetFragmentTextures
        | K::SetFragmentSamplers
        | K::SetFragmentSamplersWithLod => true,

        K::Draw
        | K::DrawWide
        | K::DrawInstanced
        | K::DrawInstancedWide
        | K::DrawInstancedBase
        | K::DrawInstancedBaseWide
        | K::DrawIndexed
        | K::DrawIndexedWide
        | K::DrawIndexedInstanced
        | K::DrawIndexedInstancedWide
        | K::DrawIndexedInstancedBase
        | K::DrawIndexedInstancedBaseWide
        | K::DrawIndirect
        | K::DrawIndexedIndirect
        | K::WriteDescriptor
        | K::SetColorStoreAction
        | K::SetDepthStoreAction
        | K::SetStencilStoreAction
        | K::SetRenderPipelineState
        | K::SetDepthStencilState
        | K::SetStencilReference
        | K::SetBlendColor
        | K::SetCullMode
        | K::SetFrontFacingWinding
        | K::SetDepthClipMode
        | K::SetTriangleFillMode
        | K::SetDepthBias
        | K::SetLineWidth
        | K::SetViewport
        | K::SetViewports
        | K::SetScissorRect
        | K::SetScissorRects
        | K::SetVisibilityResultMode => false,
    }
}

/// Apply one render-encoder bind or offset record, lifted by the protocol
/// crate.
///
/// **The stage is no longer a field this device fills in.** It comes from
/// `RenderKind::bind_stage`, which is total over the bind rows, so a bind whose
/// stage is neither vertex nor fragment cannot be constructed — the state the
/// flat command reached as `Stage::Unknown`, and which `apply_binds` answered
/// by counting a clear against a table it could not name.
///
/// **A `None` from `make` is the wire's zero ref and nothing else.** Each of
/// the five entry shapes says which field carries that ref, so a record with no
/// stride field cannot produce a stride of zero and one with no LOD clamp
/// cannot produce a clamp of zero — the two states the flat command's
/// `has_attribute_stride`/`has_lod_clamp` flags stood beside.
fn handle_render_binds<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    use reims_vgpu_protocol::decode::render::RenderRecord;

    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    let record = match reims_vgpu_protocol::decode::render::decode(&framed) {
        Ok(record) => record,
        Err(refusal) => {
            return note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
        }
    };
    match record {
        RenderRecord::BindBuffers(r) => {
            let cleared = apply_binds(
                r.entries,
                r.first,
                BindTarget {
                    stage: r.stage,
                    class: BindClass::Buffer,
                },
                BindTables {
                    vertex: &mut acc.vertex_buffers,
                    fragment: &mut acc.fragment_buffers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, e| {
                    let buffer_ref = e.buffer_ref.get();
                    (buffer_ref != 0).then_some(BufferBind {
                        index,
                        buffer_ref,
                        resource: objects::resolve_resource(state, host, task_id, buffer_ref).ok(),
                        offset: e.offset.get(),
                        // The record with no stride field cannot state one. The
                        // slot keeps whatever an earlier strided bind left.
                        attribute_stride: None,
                    })
                },
            );
            out.buffer_unbinds = out.buffer_unbinds.saturating_add(cleared);
        }
        RenderRecord::BindBuffersWithStride(r) => {
            let cleared = apply_binds(
                r.entries,
                r.first,
                BindTarget {
                    // No fragment form exists: the API has no fragment attribute
                    // stride, which is why the payload names no stage.
                    stage: ShaderStage::Vertex,
                    class: BindClass::Buffer,
                },
                BindTables {
                    vertex: &mut acc.vertex_buffers,
                    fragment: &mut acc.fragment_buffers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, e| {
                    let buffer_ref = e.buffer_ref.get();
                    (buffer_ref != 0).then_some(BufferBind {
                        index,
                        buffer_ref,
                        resource: objects::resolve_resource(state, host, task_id, buffer_ref).ok(),
                        offset: e.offset.get(),
                        attribute_stride: Some(e.attribute_stride.get()),
                    })
                },
            );
            out.buffer_unbinds = out.buffer_unbinds.saturating_add(cleared);
        }
        RenderRecord::BindTextures(r) => {
            let cleared = apply_binds(
                r.entries,
                r.first,
                BindTarget {
                    stage: r.stage,
                    class: BindClass::Texture,
                },
                BindTables {
                    vertex: &mut acc.vertex_textures,
                    fragment: &mut acc.fragment_textures,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, e| {
                    let texture_ref = e.object_ref.get();
                    if texture_ref == 0 {
                        return None;
                    }
                    if !out.texture_refs.contains(&texture_ref) {
                        out.texture_refs.push(texture_ref);
                    }
                    if let Some(m) =
                        objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref)
                    {
                        if !out.mapper_ref_texture_mappings.contains(&m) {
                            out.mapper_ref_texture_mappings.push(m);
                        }
                    } else if objects::resolve_backing(state, host, texture_ref) {
                        // x86 backing: object ref is surface_id / mapping_id.
                        if !out.mapper_ref_texture_mappings.contains(&texture_ref) {
                            out.mapper_ref_texture_mappings.push(texture_ref);
                        }
                    }
                    Some(TextureBind {
                        index,
                        texture_ref,
                        resource: objects::resolve_resource(state, host, task_id, texture_ref).ok(),
                    })
                },
            );
            out.texture_unbinds = out.texture_unbinds.saturating_add(cleared);
        }
        RenderRecord::BindSamplers(r) => {
            let cleared = apply_binds(
                r.entries,
                r.first,
                BindTarget {
                    stage: r.stage,
                    class: BindClass::Sampler,
                },
                BindTables {
                    vertex: &mut acc.vertex_samplers,
                    fragment: &mut acc.fragment_samplers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, e| {
                    let sampler_ref = e.object_ref.get();
                    (sampler_ref != 0).then_some(SamplerBind {
                        index,
                        sampler_ref,
                        // The record carries no clamp pair, so the slot gets
                        // none. This used to be a zero beside a `has_lod_clamp`
                        // flag, which made "unclamped" and "clamped to zero"
                        // the same slot.
                        lod_clamp: None,
                    })
                },
            );
            out.sampler_unbinds = out.sampler_unbinds.saturating_add(cleared);
        }
        RenderRecord::BindSamplersWithLod(r) => {
            let cleared = apply_binds(
                r.entries,
                r.first,
                BindTarget {
                    stage: r.stage,
                    class: BindClass::Sampler,
                },
                BindTables {
                    vertex: &mut acc.vertex_samplers,
                    fragment: &mut acc.fragment_samplers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, e| {
                    let sampler_ref = e.sampler_ref.get();
                    // The clamp pair travels with the ref in one entry, so a
                    // slot can no longer be paired with another slot's clamp —
                    // the flat command carried two lists and zipped them.
                    (sampler_ref != 0).then_some(SamplerBind {
                        index,
                        sampler_ref,
                        lod_clamp: Some((
                            e.lod_min_clamp.get().to_bits(),
                            e.lod_max_clamp.get().to_bits(),
                        )),
                    })
                },
            );
            out.sampler_unbinds = out.sampler_unbinds.saturating_add(cleared);
        }
        RenderRecord::RebindBufferOffset(r) => {
            if r.index >= BindClass::Buffer.table() {
                // The slot is outside the table, so the bind that would have
                // occupied it was already dropped by `apply_binds` and counted
                // under `render_buffer_bind_slot_past_table`. This is the
                // *second* record the guest spends on that slot, counted
                // separately because these are different records.
                crate::runtime::drain::note_store_route("render_buffer_offset_slot_past_table");
                let over = BufferOffsetSlotPastTable {
                    stage: r.stage,
                    index: r.index,
                };
                crate::observe::Emit::decline("render_buffer_offset", &over)
                    .fail_once((u64::from(r.stage as u32) << 32) | u64::from(r.index));
                acc.unrepresentable
                    .get_or_insert(StreamRefusal::BufferOffset(over));
                return;
            }
            let list = match r.stage {
                ShaderStage::Vertex => Arc::make_mut(&mut acc.vertex_buffers),
                ShaderStage::Fragment => Arc::make_mut(&mut acc.fragment_buffers),
            };
            match list.iter_mut().find(|b| b.index == r.index) {
                Some(b) => {
                    b.offset = r.offset;
                    // Only when this record carried one. The plain and strided
                    // rebinds are different opcodes, and the plain one must not
                    // clear a stride an earlier bind established — which is what
                    // `stride: Option<u64>` says where a flag beside a zero did
                    // not.
                    if let Some(stride) = r.stride {
                        b.attribute_stride = Some(stride);
                    }
                }
                // A healthy zero, and a sharp one. Metal requires a buffer
                // already bound at the index before
                // `setVertexBufferOffset:atIndex:`, and a render encoder's bind
                // state does not outlive the encoder, so the guest and this
                // table should agree on which slots are live. A firing means
                // they do not, and the offset lands on nothing.
                None => {
                    crate::runtime::drain::note_store_route("render_buffer_offset_slot_unbound")
                }
            }
        }
        // `is_render_bind` selected these twelve kinds and `decode::render`
        // maps each of them to an arm above. Arriving here is the two
        // disagreeing rather than a guest case.
        other => crate::observe::fail(format!(
            "render_bind_record_not_a_bind task={task_id} opcode={opcode:#x} \
             (the rail routed this row to the bind arm and the lift produced {other:?})"
        )),
    }
}

/// Whether this record states the pass's attachments, or overrides one
/// attachment's store action.
///
/// The third group. Its fields are `clears`, `color_slots`, `color_targets`,
/// `depth_attach`, `stencil_attach` and `visibility_buffer_ref` — and the three
/// store-action rows are *in* the group rather than beside it precisely because
/// they mutate `color_slots`, `depth_attach` and `stencil_attach`, which the
/// descriptor writes. Splitting them would put two decoders on one attachment's
/// `store_action`, which is the disagreement the plan forbids.
///
/// `unrepresentable` is shared with the bind group and is a first-wins latch, as
/// [`is_render_bind`] records.
///
/// Exhaustive over `RenderKind` for the reason [`is_render_stream_state`] is.
const fn is_render_pass_state(kind: ProtoRenderKind) -> bool {
    use ProtoRenderKind as K;
    match kind {
        K::WriteDescriptor
        | K::SetColorStoreAction
        | K::SetDepthStoreAction
        | K::SetStencilStoreAction => true,

        K::Draw
        | K::DrawWide
        | K::DrawInstanced
        | K::DrawInstancedWide
        | K::DrawInstancedBase
        | K::DrawInstancedBaseWide
        | K::DrawIndexed
        | K::DrawIndexedWide
        | K::DrawIndexedInstanced
        | K::DrawIndexedInstancedWide
        | K::DrawIndexedInstancedBase
        | K::DrawIndexedInstancedBaseWide
        | K::DrawIndirect
        | K::DrawIndexedIndirect
        | K::SetVertexBuffers
        | K::SetVertexBuffersWithStride
        | K::SetVertexBufferOffset
        | K::SetVertexBufferOffsetStride
        | K::SetVertexSamplers
        | K::SetVertexSamplersWithLod
        | K::SetVertexTextures
        | K::SetFragmentBuffers
        | K::SetFragmentBufferOffset
        | K::SetFragmentSamplers
        | K::SetFragmentSamplersWithLod
        | K::SetFragmentTextures
        | K::SetRenderPipelineState
        | K::SetDepthStencilState
        | K::SetStencilReference
        | K::SetBlendColor
        | K::SetCullMode
        | K::SetFrontFacingWinding
        | K::SetDepthClipMode
        | K::SetTriangleFillMode
        | K::SetDepthBias
        | K::SetLineWidth
        | K::SetViewport
        | K::SetViewports
        | K::SetScissorRect
        | K::SetScissorRects
        | K::SetVisibilityResultMode => false,
    }
}

/// Apply the pass descriptor or one store-action override, lifted by the
/// protocol crate.
///
/// **The descriptor is one wire view now, not three offset computations.**
/// `decode_{depth,stencil,color}_attachment` each re-derived the section's base
/// from `size_of` sums this module also wrote down; the record's own
/// `RenderPassBody` carries `depth`, `stencil` and a `[ColorAttachmentBody; 8]`,
/// so the eight slots are an iteration and the offsets are not restated.
///
/// **Carries one behaviour change.** A pass record shorter than
/// `RenderPassBody` is refused rather than read at the offsets that happen to
/// fit. The legacy decoder accepted anything from `PASS_MIN_PAYLOAD` — depth,
/// stencil and one colour slot — and answered the other seven slots as
/// unattached, which is indistinguishable from a guest that attached one. Apple's
/// serializer writes all 592 bytes; the short form was a tolerance for this
/// repo's own fixtures, and a truncated descriptor read as a shorter one is a
/// pass rendered with attachments the guest did not retire. `render_record`
/// names it under the row's own opcode.
fn handle_render_pass_state<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    use reims_vgpu_protocol::decode::render::RenderRecord;
    use reims_vgpu_protocol::render::StoreActionTarget;

    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    let record = match reims_vgpu_protocol::decode::render::decode(&framed) {
        Ok(record) => record,
        Err(refusal) => {
            return note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
        }
    };
    match record {
        RenderRecord::WriteDescriptor(r) => {
            let descriptor = r.descriptor;
            let target_extent = (
                descriptor.render_target_width.get(),
                descriptor.render_target_height.get(),
            );
            // The pass's own tail, decoded and not applied. Four counters
            // rather than one, because they name four different losses and one
            // of them is not a loss at all when it is zero.
            //
            // `render_target_width`/`height` are the guest's explicit extent
            // and this device renders at the attachment's instead, which is a
            // silent over-render whenever the two differ. `array_length` is
            // layered rendering. The visibility buffer is the other half of
            // `setVisibilityResultMode:offset:` — that record already counts
            // its own drop, and this counts the buffer it would have written
            // to, so the two should track and a divergence means one of the
            // arms is wrong. All four report only a non-default value: a pass
            // that asks for the API default is asking for what already happens.
            //
            // The extent one is **not** a healthy zero and the others are. On a
            // driven arm64/Vulkan boot it reads 1 575 over 127 one-second
            // windows while the visibility buffer, the array length and the
            // colour subresource all read 0 — so the macOS window server states
            // an explicit pass extent on essentially every pass, and this
            // device renders at the attachment's instead. Whether that is a
            // loss is settled by `note_pass_extent_coverage`'s bands and not by
            // this count: the two agree, so this is the denominator of a
            // measurement rather than an alarm.
            // Kept, not counted: this is where the pass says which guest buffer
            // its occlusion counts land in, and `finish_stream` writes them
            // there. `0` is a pass that named none, which leaves the arming
            // below with nowhere to write.
            acc.visibility_buffer_ref = descriptor.visibility_result_buffer_ref.get();
            // Refused rather than drawn into layer 0, the decision the colour
            // subresource arm below already made for the same shape of loss:
            // the layer a draw selects is a coordinate the pass did not name,
            // so rendering anyway lands geometry meant for one layer on top of
            // another's correct content.
            if descriptor.render_target_array_length.get() > 1 {
                let drop = note_pass_array_length_unsupported(
                    task_id,
                    descriptor.render_target_array_length.get(),
                );
                acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
            }
            if descriptor.render_target_width.get() != 0
                || descriptor.render_target_height.get() != 0
            {
                note_pass_target_extent();
            }
            {
                // A depth or stencil attachment this device cannot bind used to
                // be left out and the pass run without it, which turns depth
                // testing off for every draw in it: the near geometry stops
                // occluding the far, and the colour target — which was correct
                // before the pass — is overwritten with a picture assembled in
                // the wrong order. That is not a degraded frame, it is wrong
                // content written over right content, and nothing downstream can
                // tell because a pass with no depth attachment is exactly what a
                // guest that wanted none also produces.
                let depth = render_pass::depth_from_wire(&descriptor.depth);
                if depth.texture_ref != 0 {
                    if attachment_subresource_is_bindable(depth.into(), LevelSupport::LevelZeroOnly)
                    {
                        acc.depth_attach = Some(depth);
                    } else {
                        let drop = note_depth_stencil_unsupported(task_id, "depth", &depth.into());
                        acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
                    }
                }
                let stencil = render_pass::stencil_from_wire(&descriptor.stencil);
                if stencil.texture_ref != 0 {
                    if attachment_subresource_is_bindable(
                        stencil.into(),
                        LevelSupport::LevelZeroOnly,
                    ) {
                        acc.stencil_attach = Some(stencil);
                    } else {
                        let drop =
                            note_depth_stencil_unsupported(task_id, "stencil", &stencil.into());
                        acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
                    }
                }
                for (i, slot_body) in descriptor.color.iter().enumerate() {
                    let att = render_pass::color_from_wire(slot_body);
                    if att.texture_ref == 0 {
                        continue;
                    }
                    let slot = i as u32;
                    // A slice or depth plane is rendered past rather than into,
                    // and the pass is refused
                    // for it. This used to be reported and then rendered anyway,
                    // on the argument that dropping the pass "would trade wrong
                    // pixels for none, which is worse". That argument does not
                    // survive asking *whose* pixels: the pass does not land in
                    // the guest's slice 3 and come out wrong, it lands in
                    // **slice 0 of the same texture**, overwriting the image the
                    // guest is sampling there. A cube face becomes face 0 every
                    // time. That is wrong content written over right content,
                    // which is worse than none — and unlike none it also
                    // corrupts a resource the guest did not name in this pass.
                    //
                    // A **mip level** is the one coordinate that is not in that
                    // class, which is why this arm passes `AnyLevel`: the linear
                    // rung of `render_target` resolves the named level's own
                    // plane out of the guest allocation, so the pass renders
                    // into it rather than over level 0. macOS 26's compositor
                    // renders a blur pyramid level by level and every one of
                    // those passes was being dropped here.
                    //
                    // A resolve destination is not a source coordinate. It stays
                    // on the attachment so the backend can perform the
                    // end-of-pass resolve or refuse that exact operation.
                    if !color_attachment_subresource_is_bindable(att.into()) {
                        let drop = note_color_subresource_unsupported(task_id, slot, &att);
                        acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
                    }
                    if !acc
                        .color_slots
                        .iter()
                        .any(|(s, a)| *s == slot || a.texture_ref == att.texture_ref)
                    {
                        acc.color_slots.push((slot, att));
                    } else if let Some(entry) = acc.color_slots.iter_mut().find(|(s, _)| *s == slot)
                    {
                        entry.1 = att;
                    }
                    let published_ref = if att.resolve_texture_ref != 0 {
                        att.resolve_texture_ref
                    } else {
                        att.texture_ref
                    };
                    if !acc.color_targets.contains(&published_ref) {
                        acc.color_targets.push(published_ref);
                    }
                    if !out.texture_refs.contains(&att.texture_ref) {
                        out.texture_refs.push(att.texture_ref);
                    }
                    if att.resolve_texture_ref != 0
                        && !out.texture_refs.contains(&att.resolve_texture_ref)
                    {
                        out.texture_refs.push(att.resolve_texture_ref);
                    }
                    if let Some(m) =
                        objects::resolve_mapper_ref_texture(state, host, task_id, published_ref)
                    {
                        note_pass_extent_for_slot(state, task_id, slot, m, target_extent);
                        if !out.mapper_ref_texture_mappings.contains(&m) {
                            out.mapper_ref_texture_mappings.push(m);
                        }
                    } else if objects::resolve_backing(state, host, published_ref) {
                        // A backing attachment is its own mapping id — the arm
                        // below pushes `att.texture_ref` where the mapper-ref-texture arm
                        // pushes the id it resolved to.
                        note_pass_extent_for_slot(
                            state,
                            task_id,
                            slot,
                            published_ref,
                            target_extent,
                        );
                        if !out.mapper_ref_texture_mappings.contains(&published_ref) {
                            out.mapper_ref_texture_mappings.push(published_ref);
                        }
                    }
                    // The load action decides this, and only the load action.
                    //
                    // A `Clear` + non-`Store` attachment used to be dropped from
                    // this list entirely, which conflated the two jobs the list
                    // does: it is the pass's CLEAR **seed** for the draws, and
                    // it is the set whose colour may be **published** to guest
                    // pages. `MTLStoreAction` governs only the second. Dropping
                    // it from both meant a drawn pass began on the attachment's
                    // stale contents — wrong for anything that blends, depth-
                    // tests, or draws less than the full extent — and the store
                    // action never licensed that.
                    //
                    // macOS 26 asks for the pair 23 times in a 25 s drag and
                    // macOS 14 twice, against zero on 11/12/13; the branch was
                    // written as a healthy-zero alarm and those are firings.
                    // `clears_reaching_guest_pages` is where the store action is
                    // honoured instead.
                    if att.load_action == MTL_LOAD_ACTION_CLEAR {
                        acc.clears.push(att);
                    }
                }
            }
            // The `color0` block that stood here is gone with the second reading it
            // needed. It re-pushed slot 0 onto `clears` from a field the decoder had
            // lifted separately, guarded by "it is not already there" — and the loop
            // above pushes exactly that attachment under exactly that condition, so the
            // guard could never pass. It was reachable only when the loop and the field
            // disagreed about slot 0, which is a state one wire view cannot be in.
        }
        RenderRecord::SetStoreAction(r) => {
            // The store action is a `u16` in every attachment struct, so a mode
            // that does not fit is not narrowed into a different action; it is
            // left alone and named.
            let Ok(action) = u16::try_from(r.action) else {
                crate::runtime::drain::note_store_route("render_store_action_out_of_range");
                crate::observe::fail(format!(
                    "render_store_action fail reason=render_store_action_out_of_range \
                     op={opcode:#x} mode={} target={:?}",
                    r.action, r.target
                ));
                return;
            };
            match r.target {
                // By pass slot, which is what the record's index names and what
                // `color_slots` is keyed by — not by position, since a pass
                // declaring slots 0 and 3 has two entries.
                StoreActionTarget::Color(index) => {
                    match acc.color_slots.iter_mut().find(|(slot, _)| *slot == index) {
                        Some((_, att)) => att.store_action = action,
                        // A slot the pass never declared. The override has
                        // nothing to override and inventing an attachment for it
                        // would give the draw a target the guest did not ask
                        // for, so it is named instead.
                        None => {
                            crate::runtime::drain::note_store_route(
                                "render_store_action_slot_undeclared",
                            );
                            crate::observe::fail(format!(
                                "render_store_action fail \
                                 reason=render_store_action_slot_undeclared \
                                 index={index} declared={}",
                                acc.color_slots.len()
                            ));
                        }
                    }
                }
                // Neither of these carries an index: there is one depth and one
                // stencil attachment, so the record names only the action. The
                // fourth case the opcode match used to carry — a store action
                // for none of the three — is gone with `StoreActionTarget`,
                // which is total over the three rows.
                StoreActionTarget::Depth => match acc.depth_attach.as_mut() {
                    Some(d) => d.store_action = action,
                    None => note_store_action_no_attachment("depth", action),
                },
                StoreActionTarget::Stencil => match acc.stencil_attach.as_mut() {
                    Some(s) => s.store_action = action,
                    None => note_store_action_no_attachment("stencil", action),
                },
            }
        }
        // `is_render_pass_state` selected these four kinds and `decode::render`
        // maps each of them to an arm above.
        other => crate::observe::fail(format!(
            "render_pass_record_not_pass_state task={task_id} opcode={opcode:#x} \
             (the rail routed this row to the pass arm and the lift produced {other:?})"
        )),
    }
}

/// Whether this record is a draw.
///
/// The fourth and last of the render encoder's groups, and the one that decides
/// pixels. Its fields are `draws`, `saw_draw`, `indexed`, `dropped_no_pipeline`
/// and `dropped_zero_count`; it *reads* everything the other three groups
/// wrote, through `bind_snapshot`, which is why it moves last — a reader can
/// only be trusted once every writer it reads is settled.
///
/// Exhaustive over `RenderKind` for the reason [`is_render_stream_state`] is.
const fn is_render_draw(kind: ProtoRenderKind) -> bool {
    kind.draw_shape().is_some()
}

/// The count a draw record carries, at the width `DrawArgs` holds.
///
/// The wide encodings exist because the guest had a value above 16 bits, not
/// above 32: a vertex or index count of four billion is not a draw any GPU
/// completes. Truncating one would draw the wrong geometry in silence, so it is
/// refused by name instead.
fn draw_count(task_id: u32, opcode: u32, what: &str, value: u64) -> Option<u32> {
    match u32::try_from(value) {
        Ok(v) => Some(v),
        Err(_) => {
            crate::runtime::drain::note_store_route("render_draw_count_out_of_range");
            if crate::observe::first_sight("render_draw_count_out_of_range", u64::from(opcode)) {
                crate::observe::fail(format!(
                    "render_draw fail reason=render_draw_count_out_of_range \
                     task={task_id} op={opcode:#x} field={what} value={value}"
                ));
            }
            None
        }
    }
}

/// How many instances a draw record asked for.
///
/// **`None` and `Some(0)` are different answers and this is the single site that
/// says so.** `None` is the plain selector, which carries no `instanceCount:`
/// argument at all — Metal's own default is one instance, so one is what the
/// guest asked for. `Some(0)` is a guest that wrote the argument and wrote zero,
/// which draws nothing; that is the draw it asked for, and it is passed through.
///
/// The `.max(1)` this replaces did not distinguish them. It sat in the decoder,
/// applied to both, and three arms of this device disagreed with it and with
/// each other: `backend::metal::render` refuses a zero by name — a refusal the
/// clamp made unreachable — and `runtime::icb` decodes the same argument out of
/// an ICB slot and hands it straight to Metal with no clamp at all, so the
/// device already shipped a zero instance count on the path that did not come
/// through here.
///
/// **The clamp read zero.** Driven x86/Vulkan boots, Ventura desktop, Safari
/// window drag: `draw_instance_count_zero` never fired in the thousands. So this
/// changes nothing on a measured workload and stops the three arms disagreeing
/// on an unmeasured one. The census survives, at the lift rather than in a
/// decoder.
fn draw_instances(instances: reims_vgpu_protocol::decode::render::Instancing) -> Option<u32> {
    let Some(count) = instances.count else {
        return Some(1);
    };
    if count == 0 {
        crate::runtime::drain::note_store_route("draw_instance_count_zero");
    }
    u32::try_from(count).ok()
}

/// Record one draw, lifted by the protocol crate.
fn handle_render_draw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    kind: ProtoRenderKind,
    opcode: u32,
    cmd_bytes: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    use reims_vgpu_protocol::decode::render::{DrawRecord, RenderRecord};

    let Some(framed) = frame_render_record(task_id, opcode, cmd_bytes) else {
        return;
    };
    let record = match reims_vgpu_protocol::decode::render::decode(&framed) {
        Ok(RenderRecord::Draw(draw)) => draw,
        Ok(other) => {
            return crate::observe::fail(format!(
                "render_draw_record_not_a_draw task={task_id} opcode={opcode:#x} \
                 (the rail routed this row to the draw arm and the lift produced {other:?})"
            ));
        }
        Err(refusal) => {
            return note_render_record_refused(task_id, opcode, cmd_bytes.len(), &refusal);
        }
    };

    // The wide-encoding census. `RenderKind` is what carries it — the record
    // widens both encodings' counts to one width on purpose, so the encoding is
    // a question only the kind can answer, and the kind is the same answer that
    // said this row is a draw.
    if kind.is_wide_encoding() {
        if let DrawRecord::Indexed(d) = &record {
            crate::observe::line(format!(
                "render_wide_indexed task={task_id} target_refs={:?} pipeline={} prim={} \
                 index_type={} index_ref={} count={} offset={:#x}",
                acc.color_targets,
                acc.pipeline_ref,
                d.primitive,
                d.index.index_type.ordinal(),
                d.index.buffer_ref,
                d.index_count,
                d.index.offset
            ));
        }
    }

    match record {
        DrawRecord::PrimitivesIndirect(_) | DrawRecord::IndexedIndirect(_) => {
            return execute_indirect_draw(state, host, task_id, opcode, &record, acc);
        }
        _ => {}
    }

    acc.saw_draw = true;
    out.saw_draw = true;

    let (primitive, count, first_vertex, instances) = match &record {
        DrawRecord::Primitives(d) => {
            // Not `None`-by-omission: a non-indexed draw arriving after an
            // indexed one in the same stream must not inherit its index buffer.
            acc.indexed = None;
            (d.primitive, d.vertex_count, d.vertex_start, d.instances)
        }
        DrawRecord::Indexed(d) => {
            if d.index.buffer_ref == 0 {
                // An indexed record naming no index buffer.
                // `drawIndexedPrimitives:` takes its index buffer as an argument
                // and there is no bound-index-buffer state for a zero ref to
                // mean, so the record is malformed.
                //
                // Still named rather than declined, and the reason has changed:
                // it used to be that `index_buffer_ref` was read at an offset
                // this device computed per draw form, so a wrong offset would
                // read 0 and declining would turn a decode fault into a blank
                // frame. The offsets are the wire crate's now, fixture-derived
                // and shared by both encodings, so that risk is gone — but the
                // counter has never fired, and a decline is a frame this device
                // stops drawing. The reading is what would argue for it.
                // A zero count is not this case: an indexed draw of no indices
                // names no buffer because it reads none.
                if d.index_count != 0 {
                    note_indexed_draw_without_buffer(
                        task_id,
                        opcode,
                        u32::try_from(d.index_count).unwrap_or(u32::MAX),
                    );
                }
                acc.indexed = None;
            } else {
                let Some(index_count) = draw_count(task_id, opcode, "index_count", d.index_count)
                else {
                    return;
                };
                acc.indexed = Some(IndexedDrawInfo {
                    index_type: d.index.index_type.ordinal(),
                    index_count,
                    index_buffer_ref: d.index.buffer_ref,
                    index_buffer_offset: d.index.offset,
                    base_vertex: d.base_vertex,
                });
            }
            (d.primitive, d.index_count, 0, d.instances)
        }
        // Taken above.
        DrawRecord::PrimitivesIndirect(_) | DrawRecord::IndexedIndirect(_) => return,
    };

    let Some(count) = draw_count(task_id, opcode, "count", count) else {
        return;
    };
    let Some(first_vertex) = draw_count(task_id, opcode, "vertex_start", first_vertex) else {
        return;
    };
    let Some(instance_count) = draw_instances(instances) else {
        return;
    };
    let Some(base_instance) = draw_count(task_id, opcode, "base_instance", instances.base) else {
        return;
    };

    if acc.pipeline_ref == 0 {
        acc.dropped_no_pipeline = acc.dropped_no_pipeline.saturating_add(1);
    } else if count == 0 {
        acc.dropped_zero_count = acc.dropped_zero_count.saturating_add(1);
    } else {
        match acc.bind_snapshot() {
            Ok(snapshot) => acc.draws.push(PendingDraw {
                pipeline_ref: acc.pipeline_ref,
                draw: DrawArgs {
                    vertex_count: count,
                    instance_count,
                    primitive_type: primitive,
                    first_vertex,
                    base_instance,
                },
                ..snapshot
            }),
            Err(over) => note_draw_refused(over, acc.pipeline_ref, "draw"),
        }
    }
}

fn handle_render_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    opcode: u32,
    cmd_bytes: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    // **Which class of record this is, is the closure ledger's answer.** The
    // render rail carries the widest set on this device: the encoder's own
    // records, the fence pair, the barriers, the indirect-command executions,
    // the residency declarations inherited from the encoder base class, and
    // twenty-six rows the ledger has not settled. This device answered all of
    // them with one `Kind`, decoded in the same pass that read the fields, so a
    // record's class and its layout were one verdict.
    //
    // Three are answered here, and they are the three that own nothing on this
    // rail. The fences reach `fence_exec`, which keeps its own generations; the
    // barriers and the residency declarations are counted. **None of them
    // touches `StreamAccum`**, which is what makes them separable from the rest
    // of the rail rather than merely convenient to move first — the
    // indirect-command executions look equally self-contained and are not,
    // because they push onto `acc.execute_icb`.
    match reims_vgpu_protocol::closure::find(Rail::Render, opcode)
        .and_then(reims_vgpu_core::operation::classify)
    {
        Some(OperationHome::Stream(OperationClass::Fence)) => {
            return handle_render_fence(state, task_id, opcode, cmd_bytes);
        }
        Some(OperationHome::Stream(OperationClass::Barrier)) => {
            return note_render_barrier(task_id, opcode, cmd_bytes);
        }
        // `None` is a row the ledger has not settled. The four residency
        // declarations are among them and are *declined* rather than executed,
        // so their layout can come from `decode::residency::lift` — which
        // answers what the guest wrote without claiming the row is settled.
        // The other twenty-two unsettled rows drive work or are counted from
        // fields this device reads itself, and stay below.
        None if reims_vgpu_protocol::decode::residency::is_residency(Rail::Render, opcode) => {
            return note_render_residency(task_id, opcode, cmd_bytes);
        }
        _ => {}
    }
    // **The encoder's own state records.** Fifteen rows whose whole effect is
    // one field of `StreamAccum` that no other record class writes — see
    // [`is_render_stream_state`], which is where that claim lives and what
    // makes this group separable from the binds, the draws and the pass
    // descriptor still below.
    //
    // Routed on the kind rather than on `classify`, because every encoder
    // record on this rail is one `OperationClass::Render` and the class cannot
    // tell fifteen of them from the other thirty. `RenderKind::of_opcode`
    // answering at all is the ledger's settlement: the rows it does not name
    // are the ones the contract has not closed, and they keep this device's own
    // decoder below.
    if let Some(kind) = ProtoRenderKind::of_opcode(opcode) {
        if is_render_stream_state(kind) {
            return handle_render_stream_state(task_id, opcode, cmd_bytes, acc);
        }
        // **The encoder's argument tables.** Twelve rows writing the six bind
        // tables and the `unrepresentable` latch — see [`is_render_bind`] for
        // why a latch two classes insert into is still a group boundary the
        // rest of the rail cannot disagree across.
        if is_render_bind(kind) {
            return handle_render_binds(state, host, task_id, opcode, cmd_bytes, out, acc);
        }
        // **The pass descriptor and its three store-action overrides.** One
        // group because the overrides mutate attachments the descriptor writes
        // — see [`is_render_pass_state`].
        if is_render_pass_state(kind) {
            return handle_render_pass_state(state, host, task_id, opcode, cmd_bytes, out, acc);
        }
        // **The draws.** The group that reads what the other three wrote, which
        // is why it moves last — see [`is_render_draw`].
        if is_render_draw(kind) {
            return handle_render_draw(state, host, task_id, kind, opcode, cmd_bytes, out, acc);
        }
    }
    let cmd = match render_spi::decode(cmd_bytes) {
        Ok(c) => c,
        // `ErrUnknownOpcode` keeps the deduped fail-visible line *and the wire
        // capture* the legacy decoder's `OtherAccepted` catch-all gave it: an
        // opcode no render row names is the one case where the bytes are the
        // whole diagnostic, and a typed refusal alone would say a record
        // arrived and nothing about its layout.
        Err(render_spi::DecodeStatus::ErrUnknownOpcode) => {
            note_unimplemented_render_opcode(opcode, cmd_bytes, task_id, acc);
            return;
        }
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("render_decode", &status) {
                // Latched per (reason, opcode): the guest re-encodes the same
                // stream every frame, and this is the hottest rail in the crate.
                e.field("opcode", format!("{:#x}", opcode))
                    .field("len", cmd_bytes.len())
                    .fail_once(opcode as u64);
            }
            return;
        }
    };
    match cmd.kind {
        SpiKind::ExecuteCommands => {
            if cmd.indirect_command_buffer_ref == 0 {
                note_unnamed_icb_execute(task_id, &cmd);
                return;
            }
            acc.execute_icb.push(RenderIcbExecute {
                icb_ref: cmd.indirect_command_buffer_ref,
                is_range: cmd.icb_is_range,
                range_location: cmd.icb_range_location,
                range_length: cmd.icb_range_length,
                args_buffer_ref: cmd.icb_args_buffer_ref,
                args_buffer_offset: cmd.icb_args_buffer_offset,
            });
        }
        // Two kinds the product answers by doing nothing, counted separately.
        // They used to fall into the catch-all below, which made them
        // indistinguishable from a record that was handled — and unlike a
        // `SetBuffer` these carry ordering and lifetime the guest expects us to
        // honour, so silence was the wrong answer even though doing nothing is
        // the right one.
        //
        // The arguments are the compute rail's, which reached the same two
        // conclusions first. Residency: `useResource:`/`useHeap:` are hints for
        // a driver that pages resources, and this product resolves every
        // binding per draw, so there is nothing for them to keep resident —
        // for the half of the family that declares a *read*. The write half is
        // not covered by that argument and is now told apart from it; see
        // `report::note_residency_declaration`. Barriers: the render rail
        // submits and waits at pass granularity, so a barrier inside the pass
        // is implied by the boundary.
        //
        // These counters exist to price those arguments rather than to doubt
        // them. A large residency count is the cost of resolving per draw; a
        // large barrier count is what the pass-granularity submit is buying.
        // The render states this rail decodes and does not apply. Each reports
        // only when the guest asked for something *other* than the API default,
        // because asking for the default is asking for what we already do — so
        // these are healthy zeros, and a non-zero reading is the measured
        // argument for implementing that state.
        //
        // That distinction is the point of decoding them at all. They all used
        // to reach `OtherAccepted`, and `0x7c` alone fires thousands of times
        // per app render, so the one line it produced said a record had arrived
        // and nothing about whether any of them mattered.
        //
        // `SetRasterState` was two of them and is no longer here: the counters
        // it raised are what argued for plumbing it, both halves now reach a
        // backend, and the row itself has since moved to
        // `handle_render_stream_state`.
        SpiKind::SetTessellationFactorScale => {
            // `setLineWidth:` shared this wire form and does not share its
            // settlement: the line width has a `RenderKind` and reaches
            // `handle_render_stream_state`, and this does not and reaches here.
            // The two selectors are two modules now, which is the whole reason
            // the opcode no longer has to be read a second time to tell them
            // apart.
            //
            // The tessellation factor scale has no carrier on either rail. Its
            // loss is counted only where the value is not the 1.0 default —
            // compared exactly rather than with a tolerance, because the guest
            // wrote a literal and the question is whether it wrote *the*
            // literal.
            if cmd.float_value != 1.0 {
                crate::runtime::drain::note_store_route("render_tessellation_scale_dropped");
            }
        }
        SpiKind::SetVertexAmplification => {
            // Amplification makes one vertex invocation produce several views,
            // so a dropped record renders one view where the guest asked for
            // many. Both forms have an API default that means "no
            // amplification" — a count of 1, and mode 0 — and asking for the
            // default is asking for what this rail already does, so only the
            // rest is a loss.
            let asked_for_more = match cmd.opcode {
                wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => cmd.count > 1,
                _ => cmd.mode != 0 || cmd.amplification_value != 0,
            };
            if asked_for_more {
                crate::runtime::drain::note_store_route("render_vertex_amplification_dropped");
            }
            // The count is not the whole record. Its view mappings offset the
            // viewport and render-target *array indices* the views rasterise
            // into, so a count of one whose mapping is not the identity is a
            // draw aimed at a different array slice — a loss that the count
            // alone reads as the API default and says nothing about.
            //
            // Its own route rather than the one above, because the two are
            // different losses: that one renders one view where several were
            // asked for, and this one renders the right number of views into
            // the wrong slice. A reading that merged them could not say which
            // had happened.
            if cmd.amplification_offsets_views {
                crate::runtime::drain::note_store_route(
                    "render_vertex_amplification_view_mapping_dropped",
                );
            }
        }
        SpiKind::TileBind => {
            // A bind against the tile argument tables. There is no default a
            // bind could be sitting at, so this counts unconditionally: it is
            // an upper bound on tile resources the guest attached and this rail
            // did not, the same footing as `render_store_action_override_dropped`.
            //
            // Counted rather than applied, and the reason is the same one the
            // decoder gives for not reusing `Kind::SetBuffer`: this device has
            // no tile argument table to bind into. Routing these into the
            // vertex or fragment table would not be a partial implementation,
            // it would be a wrong one.
            //
            // Split by which table, because they are not interchangeable when
            // an implementation is costed — a tile buffer bind is imageblock
            // storage, a tile texture bind is a sampled attachment.
            crate::runtime::drain::note_store_route(match cmd.opcode {
                wire_tile::OPCODE_SET_TILE_BUFFER | wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET => {
                    "render_tile_buffer_bind_dropped"
                }
                wire_tile::OPCODE_SET_TILE_TEXTURE => "render_tile_texture_bind_dropped",
                // Imageblock memory, not an argument-table slot: this one is
                // the tile shader's scratch storage, so it is priced on its own
                // rather than with the buffer binds it sits next to.
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY => {
                    "render_tile_threadgroup_memory_dropped"
                }
                _ => "render_tile_sampler_bind_dropped",
            });
        }
        SpiKind::TileDispatch => {
            // A tile shader the guest asked to run. Like an indirect draw and
            // unlike the unapplied states, this is work rather than state, so
            // it keeps the deduped fail-visible line as well as the count.
            //
            // The one healthy zero here is a genuinely empty grid: Metal
            // dispatches nothing when any dimension of `threadsPerTile` is 0,
            // so dropping such a record loses nothing and counting it would
            // inflate the loss estimate this counter exists to be.
            if cmd.tile_threads.iter().all(|&n| n != 0) {
                crate::runtime::drain::note_store_route("render_tile_dispatch_dropped");
                note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
            }
        }
        SpiKind::SetStoreActionOptions => {
            // The options sibling of the store action beside it, which *is*
            // applied. `MTLStoreActionOptions` declares exactly one flag,
            // `CustomSamplePositions`, asking that a multisample resolve use
            // the pass's programmable sample positions — which this device does
            // not set (`render_pass_sample_positions_dropped`).
            //
            // `MTLStoreActionOptionNone` is honoured. This doc used to argue
            // there was "no default to compare against", because a guest that
            // writes 0 is still overriding whatever the pass descriptor said —
            // and that is true and does not matter: overriding to the value
            // that asks for nothing is asking for nothing, and this device's
            // resolve already uses the default sample positions the option
            // would have to change. It is the same reading the pass's default
            // raster sample count takes about a count of 1, and it is why that
            // arm counts nothing at its API default either.
            //
            // Everything else is refused rather than counted, on the reading
            // `StreamDrawDrop::StoreActionOptionsUnsupported` records: a
            // resolve produced with the default sample positions is
            // byte-for-byte what a guest asking for the default also gets, so
            // nothing downstream can tell the substitution happened. Undeclared
            // bits reach the same refusal — masking them away would read the
            // capture's own `0x1111` as a request for custom sample positions
            // the guest never made.
            let honoured = reims_vgpu_protocol::pass_action::StoreActionOptions::parse(cmd.mode)
                .is_some_and(
                    reims_vgpu_protocol::pass_action::StoreActionOptions::asks_for_nothing,
                );
            if !honoured {
                let (aspect, slot) = match cmd.opcode {
                    wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS => ("depth", 0),
                    wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => ("stencil", 0),
                    _ => ("color", cmd.first),
                };
                let drop = note_store_action_options_unsupported(task_id, aspect, slot, cmd.mode);
                acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
            }
        }
        SpiKind::DrawPatches => {
            // A tessellated draw. Geometry the guest asked for and did not get,
            // so it counts unconditionally and keeps the deduped fail-visible
            // line, on the same footing as the indirect draws — there is no
            // default a draw could be sitting at.
            //
            // Split by form because they are not equally far from being
            // executable: the two direct forms carry their patch counts on the
            // wire, while the indirect pair reads them from a buffer the GPU
            // may not have written yet.
            crate::runtime::drain::note_store_route(match cmd.opcode {
                wire_render::OPCODE_DRAW_PATCHES_INDIRECT
                | wire_render::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    "render_draw_patches_indirect_dropped"
                }
                _ => "render_draw_patches_dropped",
            });
            note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
        }
        SpiKind::SetTessellationFactorBuffer => {
            // The state half of a tessellated draw. Unapplied like the draws
            // themselves, so this should track `render_draw_patches_dropped`;
            // the two being far apart would mean one of the two arms is wrong
            // rather than that the guest is doing something unusual.
            crate::runtime::drain::note_store_route("render_tessellation_factor_buffer_dropped");
        }
        SpiKind::RenderPassProperty => {
            // One of the six records `writeDescriptor` emits beside the pass
            // descriptor. Every one is behind a serializer capability that
            // defaults off, so these are healthy zeros: a non-zero reading is
            // the first evidence this project would have that a guest
            // negotiates one of the sixteen flags, which nothing in this device
            // currently observes.
            //
            // Counted per opcode rather than under one name, because the six
            // are not equally costly to drop. The rate map and the sample
            // positions change *where fragments land*; the raster sample count
            // changes how many there are; the three tile ones are tile-shader
            // pass geometry this device has no executor for at all.
            //
            // The sample count is the one of the six that is refused rather
            // than counted, and it takes its own arm below for that reason. The
            // other five still count: three are tile-shader pass geometry with
            // no executor to refuse *for*, and the rate map and the sample
            // positions move fragments within a pixel rather than changing
            // which pixels a draw covers — a loss that has never been read
            // against a boot, so refusing on it would trade a measured
            // degradation for an unmeasured refusal.
            if cmd.opcode == wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT {
                // `MTLRenderPassDescriptor.defaultRasterSampleCount` defaults to
                // 1, which is what this device already does, so only a request
                // above it is a loss. A zero is not a Metal sample count at all;
                // it reaches the refusal rather than the silent arm, because a
                // record this device cannot honour is not made honourable by
                // naming an impossible value.
                if cmd.mode != 1 {
                    let drop = note_pass_raster_sample_count_unsupported(task_id, cmd.mode);
                    acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
                }
            } else {
                crate::runtime::drain::note_store_route(match cmd.opcode {
                    wire_pass::OPCODE_RASTERIZATION_RATE_MAP => "render_pass_rate_map_dropped",
                    wire_pass::OPCODE_SAMPLE_POSITIONS => "render_pass_sample_positions_dropped",
                    wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH => "render_pass_imageblock_dropped",
                    wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
                        "render_pass_threadgroup_memory_dropped"
                    }
                    _ => "render_pass_tile_size_dropped",
                });
                // Only the five that are still dropped. The sample count has an
                // executor arm now — it is honoured at 1 and refused above it —
                // so reporting it as `accepted_without_executor` beside its own
                // typed decline would name the same record twice and disagree
                // with itself about whether anything read it.
                note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
            }
        }
        SpiKind::TileDimensionsQuery => {
            // Not a dropped command — a *wrong answer*. The guest handed over a
            // buffer for this device to write the tile width and height into
            // and will read it back regardless of whether anything was written,
            // so ignoring the record leaves the guest treating whatever its ring
            // last held as a tile geometry. There is no default and no healthy
            // zero, which is why this one is fail-visible on its own line
            // naming where the answer was expected rather than through the
            // deduped opcode path.
            crate::runtime::drain::note_store_route("render_tile_dimensions_unanswered");
            crate::observe::fail(format!(
                "render_tile_dimensions reason=render_tile_dimensions_unanswered \
                 task={task_id} buffer={} offset={:#x}",
                cmd.buffer_ref, cmd.buffer_offset
            ));
        }
        // `decode` never produces it — it is the `Default` a caller builds a
        // command from — so this arm is the type's shape rather than a record.
        SpiKind::Unknown => {}
    }
}

use crate::runtime::draw::BindTableClass as BindClass;

/// The census vocabulary for a bind slot this device's argument table could not
/// hold.
///
/// The type and its bound live in [`crate::runtime::draw`], beside the three
/// constants; this `impl` is this module's own addition — the census
/// vocabulary for a slot the table could not hold. [`apply_binds`] gates each
/// class on [`BindClass::table`], its own bound. It used to gate all three on
/// one constant, which was Metal's *buffer* table applied to buffers, textures
/// and samplers alike — defensible as a *bound*, since it was the smallest of
/// the three, but the wrong number for two classes by construction and never
/// defensible as a *counter*.
///
/// Apple's serializer truncates a plural bind at the stage's argument table, and
/// [`reims_vgpu_wire::ops::bind_limit`] measured those three tables at 128
/// textures, 31 buffers and 16 samplers. All three of this device's bounds now
/// sit at or above Apple's, pinned by the `const` assertions below, so a slot
/// dropped here cannot come from a conforming Apple stream in any class — but
/// what a reading would *mean* still differs by class:
///
/// * **Texture** — the bound is Apple's whole 128-entry table. It was 32, the
///   width of a descriptor binding band, and slots 32..127 were guest work with
///   nowhere to go until `spirv_bind::widen_sampled_bands` closed the gap.
/// * **Buffer** — 31 is exactly the serializer's own buffer bound, with no
///   margin at all, so a non-zero reading is either a guest writing its own
///   stream or a decode that mis-sized the table.
/// * **Sampler** — same, one step further: Apple truncates at 16, half the
///   bound, so this can only fire on a stream Apple's serializer did not write.
///
/// One slug for all three said "31 slots were lost" and could not say which
/// table to widen, which is the whole reason the counter exists. Splitting it is
/// the same lesson `BlitEncoderSPI` taught one layer up — a family is not
/// uniform in what its loss means.
impl BindClass {
    /// The census name for slots this class lost to [`BindClass::table`].
    ///
    /// Also the `reason=` slug of [`BindSlotPastTable`], deliberately: the two
    /// name one event, and a reader who greps the fail log for a slug should
    /// find the same string beside a running total in the census. What they
    /// count differs — one line per distinct `(stage, slot)` this boot against a
    /// cumulative per-window slot count — which is exactly why both exist.
    fn past_table_route(self) -> &'static str {
        match self {
            BindClass::Buffer => "render_buffer_bind_slot_past_table",
            BindClass::Texture => "render_texture_bind_slot_past_table",
            BindClass::Sampler => "render_sampler_bind_slot_past_table",
        }
    }

    /// The size of Apple's own argument table for this class, as measured in
    /// [`reims_vgpu_wire::ops::bind_limit`].
    ///
    /// On the line because it is what makes a reading actionable without going
    /// back to the source: `table=31 apple_table=128` is guest work Apple's
    /// serializer is entitled to emit and this device cannot hold, while
    /// `table=31 apple_table=16` cannot come from an Apple guest at all and
    /// points at a decode that mis-sized the record.
    fn apple_table(self) -> u32 {
        use reims_vgpu_wire::ops::bind_limit;
        match self {
            BindClass::Buffer => bind_limit::BUFFER,
            BindClass::Texture => bind_limit::TEXTURE,
            BindClass::Sampler => bind_limit::SAMPLER,
        }
    }

    /// The census name for the band a record's requested reach falls in.
    ///
    /// `reach` is `first + count`, the exclusive end of the slot run the guest
    /// asked for. The drop counters above say only that traffic crossed the
    /// bound, and a zero from them is not interpretable on its own: every
    /// record reaching slot 30 and every record reaching slot 4 both read zero,
    /// and only one of those says the bound has headroom. That is the same
    /// shape as `pass_scissor_union` and `pass_extent_full` — a census of what
    /// the guest asked for, kept beside the counter for what it lost.
    ///
    /// The bands are Apple's own three argument tables rather than round
    /// numbers, so each one means something:
    ///
    /// * `le16` — inside all three of Apple's tables, so inside any bound this
    ///   device could plausibly adopt.
    /// * `le_table` — above Apple's 16-entry sampler table, inside its buffer
    ///   table and inside this class's own bound. This is headroom being spent.
    /// * `over_table` — past this device's bound. Fires on exactly the records
    ///   the sibling `*_bind_slot_past_table` counts slots for, so the two
    ///   reconcile: records here, slots there.
    ///
    /// # What a driven boot reads, and why it settles the widening question
    ///
    /// arm64 / MoltenVK-Vulkan, `vm/boot-arm64.sh --device reims-vgpu-mmio
    /// --testing`, driven with `window-drag-probe` repositioning a Safari
    /// window; 325 census windows, peak 1 205 draws in a window, 325 523 bind
    /// records:
    ///
    /// | class | `le16` | `le_table` | `over_table` |
    /// |---|---|---|---|
    /// | buffer | 188 072 | 5 104 | 0 |
    /// | texture | 84 692 | 0 | 0 |
    /// | sampler | 47 655 | 0 | 0 |
    ///
    /// **No class has a gap left to widen.** All three of this device's tables
    /// now meet or exceed Apple's own — texture 128 against 128, buffer 31
    /// against 31, sampler 32 against 16 — so a record reaching past one is a
    /// record Apple's serializer cannot emit, and `over_table` is a healthy
    /// zero rather than headroom being measured. The texture band that closed
    /// the last of it lives in [`crate::runtime::spirv_bind`] as `[32,160)`,
    /// held there by a `const` assertion that
    /// [`crate::runtime::draw::MAX_TEXTURE_BIND_SLOTS`] reads its value from,
    /// so the two cannot part without failing the build.
    ///
    /// Not one texture bind in the table above reaches even slot 17, which is
    /// why this cost nothing to confirm — but the reading that matters is the
    /// pin, not the counter: a zero here would look identical if the band were
    /// still 32 wide.
    ///
    /// **The table actually running near its ceiling is the buffer one**, which
    /// no reading of the loss counters could have said: 2.6 % of buffer binds
    /// reach into 17..31, and 31 is *exactly* Apple's own buffer bound, so this
    /// device fits it with no loss and no margin at all. If a later serializer
    /// raises that table, this rail starts dropping on the first record rather
    /// than degrading — which is what makes the build gate beside
    /// [`BindClass`] load-bearing rather than tidy.
    ///
    /// The standing caveat applies: one workload, one pathway. A guest binding
    /// many textures at once — a deferred renderer, an atlas-heavy engine — is
    /// exactly where `render_bind_reach_texture_le_table` would move first, and
    /// it is the band to watch rather than the drop counter.
    fn reach_route(self, reach: u32) -> &'static str {
        use reims_vgpu_wire::ops::bind_limit;
        match (self, reach) {
            (BindClass::Buffer, r) if r <= bind_limit::SAMPLER => "render_bind_reach_buffer_le16",
            (BindClass::Buffer, r) if r <= MAX_BUFFER_BIND_SLOTS => {
                "render_bind_reach_buffer_le_table"
            }
            (BindClass::Buffer, _) => "render_bind_reach_buffer_over_table",
            (BindClass::Texture, r) if r <= bind_limit::SAMPLER => "render_bind_reach_texture_le16",
            (BindClass::Texture, r) if r <= MAX_TEXTURE_BIND_SLOTS => {
                "render_bind_reach_texture_le_table"
            }
            (BindClass::Texture, _) => "render_bind_reach_texture_over_table",
            (BindClass::Sampler, r) if r <= bind_limit::SAMPLER => "render_bind_reach_sampler_le16",
            (BindClass::Sampler, r) if r <= MAX_SAMPLER_BIND_SLOTS => {
                "render_bind_reach_sampler_le_table"
            }
            (BindClass::Sampler, _) => "render_bind_reach_sampler_over_table",
        }
    }
}

/// A render bind record whose slot run reached past [`BindClass::table`], so the
/// walk stopped and the rest of the record was dropped.
///
/// # Why this is on the fail channel and not only in the census
///
/// The sibling counter [`BindClass::past_table_route`] has always been here, and
/// a census counter is not the always-on failure path: it lands in a one-second
/// `OFF` line among a hundred other routes, and a route reading zero is simply
/// absent from it. So the first time a guest lost a texture bind, nothing in
/// `/tmp/reims-vgpu-fail.log` would have said so — the reader had to already
/// suspect it and diff two census lines to find out.
///
/// The compute rail reached the opposite conclusion about the identical loss:
/// `compute_exec`'s `ComputeBindOverflow` puts a slot past
/// `MAX_COMPUTE_*_SLOTS` on the fail channel, deduped per `(table, index)`,
/// with the comment "wrong compute output with no other symptom, previously
/// silent". Two arms, one rule about one wire form, and the arm that a boot
/// actually walks was the quiet one. This closes that.
///
/// Latched per `(stage, first refused slot)` rather than per record: a guest
/// that binds a 40-slot texture range does it every frame, and the second line
/// carries nothing the first did not. Magnitude is what the counter is for.
///
/// **A reading here is the argument for widening the table**, for the class
/// named by the slug — see [`BindClass::reach_route`] for what a driven boot
/// measured and why one workload's zero is not a reason to leave it unwatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BindSlotPastTable {
    class: BindClass,
    stage: ShaderStage,
    /// The first slot the walk refused — the guest's own index, not the
    /// position within the record, so it can be read against the table size.
    index: u32,
    /// Entries dropped with it, this record. The record is walked in slot
    /// order, so everything from `index` on is lost together.
    slots: u32,
}

impl crate::observe::Decline for BindSlotPastTable {
    fn slug(&self) -> &'static str {
        self.class.past_table_route()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("stage", self.stage.name().to_string()),
            ("index", self.index.to_string()),
            ("slots", self.slots.to_string()),
            ("table", self.class.table().to_string()),
            ("apple_table", self.class.apple_table().to_string()),
        ]
    }
}

/// Report a draw refused because the stream's state is missing something the
/// guest asked for.
///
/// The same decline the decode already emitted, re-emitted under a different
/// tag: the first line says what was lost, this one says what the loss then
/// cost, and the two share a slug on purpose so one grep finds both halves of
/// one event.
///
/// Latched per refusal, not per draw. A stream refuses once and then refuses
/// every draw after it, and the second line carries nothing the first did not;
/// `render_draw_refused_unrepresentable` is the magnitude.
///
/// `site` separates the two consumers of the stream's state, because what the
/// guest loses differs: a decoded draw loses one draw, and an ICB execute loses
/// whatever the command buffer held.
fn note_draw_refused(refusal: StreamRefusal, pipeline_ref: u32, site: &'static str) {
    crate::runtime::drain::note_store_route("render_draw_refused_unrepresentable");
    let emit = match refusal {
        StreamRefusal::Bind(over) => crate::observe::Emit::decline("render_draw", &over),
        StreamRefusal::Pass(drop) => crate::observe::Emit::decline("render_draw", &drop),
        StreamRefusal::BufferOffset(over) => crate::observe::Emit::decline("render_draw", &over),
    };
    emit.field("site", site)
        .field("pipeline_ref", pipeline_ref)
        .fail_once(refusal.latch());
}

impl StreamRefusal {
    /// The `fail_once` latch for this refusal.
    ///
    /// Distinct per *condition* rather than per stream, so a guest that binds
    /// past the table on every frame gets one line and a guest that then also
    /// names a mip gets a second. The two arms cannot collide: the pass arm sets
    /// the top bit and the offset arm the one below it, neither of which the
    /// bind arm's `(stage, index)` pair can reach.
    fn latch(self) -> u64 {
        match self {
            Self::Bind(over) => (u64::from(over.stage as u32) << 32) | u64::from(over.index),
            Self::Pass(drop) => 1 << 63 | drop.latch(),
            Self::BufferOffset(over) => {
                1 << 62 | (u64::from(over.stage as u32) << 32) | u64::from(over.index)
            }
        }
    }
}

// The three relations that make each `*_bind_slot_past_table` slug readable in a
// driven boot's census, pinned at build time because both sides can move
// independently: a new macOS serializer can change Apple's argument tables, and
// widening a host table moves that class's constant. Either would silently
// re-point what the census means, so this is a build gate rather than a test —
// the same reason `reims_vgpu_wire::Wire::ASSERT_ALIGN_1` is one.
//
// Textures: the bound IS Apple's table now, so no texture bind an Apple guest
// can emit is refused. This used to be `>`, and the gap it recorded — slots
// 32..127, dropped because the descriptor binding band was 32 wide — is what
// `spirv_bind::widen_sampled_bands` closed. A `<` here would mean this device
// accepts a slot it cannot name; a `>` would mean the gap is back.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::TEXTURE == MAX_TEXTURE_BIND_SLOTS);
// Buffers: two independent derivations of one table size — Apple's serializer
// truncates there and Metal's `REIMS_VGPU_METAL_MAX_BUFFERS` stops there.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::BUFFER == MAX_BUFFER_BIND_SLOTS);
// Samplers: Apple truncates well below the bound, so this slug cannot fire on a
// stream Apple's serializer wrote. A reading is a guest writing its own stream,
// or a decode that mis-sized the table.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::SAMPLER < MAX_SAMPLER_BIND_SLOTS);
// The two band bounds are the *encoding's*, so they must stay equal to the
// distance between the bands they name. A texture index at
// `MAX_TEXTURE_BIND_SLOTS` would carry sampler 0's descriptor binding, and a
// sampler index at `MAX_SAMPLER_BIND_SLOTS` would carry the first ColorInput's;
// either collision is silent, because a flat binding number cannot say which
// class wrote it.
const _: () = assert!(
    crate::runtime::spirv_bind::TEXTURE_BINDING_BASE + MAX_TEXTURE_BIND_SLOTS
        == crate::runtime::spirv_bind::SAMPLER_BINDING_BASE
);
const _: () = assert!(
    crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + MAX_SAMPLER_BIND_SLOTS
        == crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE
);

/// Which bind table a record names: the stage picks vertex or fragment, the
/// class picks buffer, texture or sampler.
///
/// The two travel together because [`apply_binds`] needs both to say where a
/// slot went and, when the slot is past the bound, which of the three tables
/// lost it.
#[derive(Clone, Copy, Debug)]
struct BindTarget {
    stage: ShaderStage,
    class: BindClass,
}

/// Why one stream's state cannot be encoded as the guest described it.
///
/// The three arms are decoded at three different points and none of them can be
/// noticed downstream, which is what they have in common and why they share one
/// field. A shader that does not sample the missing texture, a pass that draws
/// into the base level of the texture it was given, a pass with no depth
/// attachment — each is byte-for-byte indistinguishable from the state the guest
/// asked for, right up until the pixels are wrong.
///
/// Each arm used to note its loss and let the pass run. What that bought, in
/// every case, was **wrong content written over content that was right**: the
/// subresource arm overwrites base level 0 of a texture whose mip the guest
/// named, and the depth arm draws with occlusion turned off into a colour target
/// that was correct before. Refusing leaves the guest's own bytes where they
/// are, which is the answer a GPU gives and the answer that can be seen in a
/// log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamRefusal {
    /// A bind slot past its class's argument table.
    ///
    /// [`apply_binds`] stops a record's walk there — forced, there is no slot to
    /// put it in — and the six tables then carry state the guest did not ask
    /// for.
    ///
    /// [`crate::runtime::draw::first_bind_past_table`] cannot catch this. It
    /// reads the six tables of a *built request*, and this bind is precisely the
    /// one that never entered them, which is why that check calls itself a
    /// backstop and why the refusal has to be recorded here instead.
    Bind(BindSlotPastTable),
    /// A pass attachment this device would have bound past: a colour
    /// subresource it renders into the base of, or a depth/stencil form it
    /// leaves out of the pass entirely.
    ///
    /// Carried as the [`StreamDrawDrop`] arm that decoded it, so the refusal
    /// line names the same fields the pass census already reports.
    Pass(StreamDrawDrop),
    /// A `SetBufferOffset` naming a slot past the buffer table.
    ///
    /// Its own variant rather than folded into [`Self::Bind`] because they are
    /// different records with different counters, and sharing one would put two
    /// checks behind one `reason=` slug and one `fail_once` latch.
    BufferOffset(BufferOffsetSlotPastTable),
}

/// A `SetBufferOffset` record naming a slot the buffer table does not have.
///
/// The offset update has nowhere to land, and this used to be a census counter
/// and nothing else — which is the same gap [`BindSlotPastTable`]'s own doc
/// argues about for the bind: a route reading zero is simply absent from a
/// one-second `OFF` line among a hundred others, so the first time a guest lost
/// one, `/tmp/reims-vgpu-fail.log` said nothing.
///
/// The counter stays and says how much; this says which slot, once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferOffsetSlotPastTable {
    stage: ShaderStage,
    /// The slot the record named. `cmd.first` is the whole of it — this wire
    /// form updates one slot, so there is no run to report a length for.
    index: u32,
}

impl crate::observe::Decline for BufferOffsetSlotPastTable {
    fn slug(&self) -> &'static str {
        "render_buffer_offset_slot_past_table"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("stage", self.stage.name().to_string()),
            ("index", self.index.to_string()),
            ("table", BindClass::Buffer.table().to_string()),
            ("apple_table", BindClass::Buffer.apple_table().to_string()),
        ]
    }
}

/// The [`StreamAccum`] state one bind record writes: the two stage tables a
/// slot may land in, and the place a slot that lands in neither is recorded.
///
/// The three travel as one because they are written together and no caller has
/// a reason to pass two of them. They are also three disjoint fields of one
/// accumulator, which is what lets a caller hand out all three at once.
struct BindTables<'a, B> {
    vertex: &'a mut BindTable<B>,
    fragment: &'a mut BindTable<B>,
    /// Where [`apply_binds`] leaves a slot past [`BindClass::table`]. See
    /// [`StreamAccum::unrepresentable`] for why it is recorded rather than only
    /// counted.
    refused: &'a mut Option<StreamRefusal>,
}

/// Apply one `Set{Buffer,Texture,Sampler}` record to a stage's bind table.
///
/// All three carry the same wire form: `count` consecutive slots starting at
/// `first`, where a zero object ref clears the slot it names and any other ref
/// replaces whatever occupied it. Slots at or past the class's own
/// [`BindClass::table`] are outside the encoder's table and end the walk. Only the vertex and fragment
/// stages have tables here; a record for any other stage still counts its
/// clears, because a slot the guest cleared is cleared whether or not we model
/// the table it lived in.
///
/// `make` builds the bind for a live slot and returns `None` for the zero ref,
/// which keeps the ref field's name — and any side registration, such as the
/// texture arm's mapper-ref-texture mapping list — with the caller. The clear count comes
/// back as a return value rather than through an `&mut` counter so `make` can
/// hold the rest of `ExecResult`.
fn apply_binds<T, B: Clone>(
    entries: &[T],
    first: u32,
    target: BindTarget,
    tables: BindTables<'_, B>,
    slot: impl Fn(&B) -> u32,
    mut make: impl FnMut(u32, &T) -> Option<B>,
) -> u32 {
    let BindTarget { stage, class } = target;
    let BindTables {
        vertex,
        fragment,
        refused,
    } = tables;
    // Once per record, before the walk, so it reports what the guest asked for
    // rather than what survived the bound. An empty entry list is not a request
    // and `first` alone is not a reach.
    if let Some(last) = entries.len().checked_sub(1) {
        let reach = first.saturating_add(last as u32).saturating_add(1);
        crate::runtime::drain::note_store_route(class.reach_route(reach));
    }
    let mut cleared = 0u32;
    for (i, entry) in entries.iter().enumerate() {
        let index = first.saturating_add(i as u32);
        if index >= class.table() {
            // The walk stops here, and it used to stop in silence — a `break`
            // that dropped every remaining slot with nothing to say so.
            //
            // The bound is `class.table()`, one constant per class. It
            // used to be a single 31 — Metal's *buffer* index cap — applied to
            // all three tables, where Apple's texture limit is 128 and its
            // sampler limit 16, so it was the wrong number for two of the three
            // by construction. What still refuses a texture is the descriptor
            // binding band's width, and `setVertexTextures:withRange:` over a
            // range of 40 is a record Apple's serializer can produce.
            //
            // **This has not been observed to fire.** Driven x86/PCI boot,
            // window-drag probe against Safari, `reach_route` census over 18 044
            // bind records:
            //
            //     texture  le16=5519  le_table=0  over_table=0
            //     buffer   le16=9275  le_table=0  over_table=0
            //     sampler  le16=3250  le_table=0  over_table=0
            //
            // and all three `*_bind_slot_past_table` counters absent. Every
            // record this guest issued ended at slot 16 or below — not merely
            // inside the bound, but inside the *smallest* of Apple's three
            // tables, with 15 slots of headroom nothing touched. Read the reach
            // bands and not just the drop counters: a zero drop count alone
            // cannot tell a record stopping at slot 4 from one stopping at 30,
            // which is why the bands are here.
            //
            // So "the serializer can emit a range of 40" is a statement about
            // Apple's encoder, not a reading of this workload, and it is not on
            // its own an argument for widening. One workload on one pathway
            // proves one workload on one pathway; a heavier guest may differ.
            //
            // Raising the cap means widening the backends' tables, which is a
            // change with its own measurement; naming the loss is not. A
            // non-zero reading from the counter below — or from `le_table`,
            // which fires one band earlier and is the leading indicator — is the
            // argument for doing the widening, for the table [`BindClass`]
            // names, which is why there are three slugs rather than one.
            //
            // The counter alone was still not the always-on failure path, which
            // is what `AGENTS.md` asks a dropped guest record for, and which the
            // compute rail already gives the same loss. Both, now: the line says
            // *which* bind was lost the first time it happens, the counter says
            // how much. See [`BindSlotPastTable`].
            let slots = (entries.len() - i) as u32;
            crate::runtime::drain::note_store_route_n(class.past_table_route(), u64::from(slots));
            let over = BindSlotPastTable {
                class,
                stage,
                index,
                slots,
            };
            crate::observe::Emit::decline("render_bind_overflow", &over)
                .fail_once((u64::from(stage as u32) << 32) | u64::from(index));
            // The walk cannot refuse anything — a bind record has no draw to
            // refuse — so it records, and [`StreamAccum::bind_snapshot`] refuses
            // every draw that would have read the gap. The first one is kept
            // rather than the last: it is the refusal the earliest later draw
            // would read, and the rest are the same record.
            refused.get_or_insert(StreamRefusal::Bind(over));
            break;
        }
        let bind = make(index, entry);
        // Two arms and no third: `ShaderStage` is the render opcode set's whole
        // vocabulary, so "the record named no stage" — which used to reach here
        // as `Stage::Unknown` and count the clear against a table it could not
        // name — is not a state a lifted bind can be in.
        let list = match stage {
            ShaderStage::Vertex => Arc::make_mut(vertex),
            ShaderStage::Fragment => Arc::make_mut(fragment),
        };
        let Some(bind) = bind else {
            list.retain(|b| slot(b) != index);
            cleared = cleared.saturating_add(1);
            continue;
        };
        match list.iter_mut().find(|b| slot(b) == index) {
            Some(occupant) => *occupant = bind,
            None => list.push(bind),
        }
    }
    cleared
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamDrawDelta {
    ok: u32,
    fail: u32,
}

fn stream_draw_delta(out: &ExecResult, at_entry: (u32, u32)) -> StreamDrawDelta {
    StreamDrawDelta {
        ok: out.metal_draws_ok.saturating_sub(at_entry.0),
        fail: out.metal_draws_fail.saturating_sub(at_entry.1),
    }
}

fn finish_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    out: &mut ExecResult,
    acc: &StreamAccum,
) {
    let draws_at_entry = (out.metal_draws_ok, out.metal_draws_fail);
    let clears_at_entry = out.clears_applied;
    // Opens in `Prelude` and is charged to whichever part is open until it
    // drops, so the six tile this function rather than sampling it. See
    // [`finish_phase`] for what the split is for.
    let mut fin = finish_phase::FinishTimer::open();
    note_stream_draw_drops(task_id, acc);
    // Archive ApplePVGPUDrawJob: clear/load seed is private initial_rgba for the
    // async job; guest pages are written once at completion. Apply clear-to-guest
    // only for clear-only streams (no draws). When draws run, CLEAR is the Metal
    // pass seed inside encode (mrt_draw_request solid seed) — not a pre-draw
    // guest store that would expose intermediate pixels to DisplaySwap.
    let will_draw = acc.saw_draw && !acc.color_slots.is_empty() && !acc.draws.is_empty();
    if !will_draw {
        for att in acc.clears_reaching_guest_pages() {
            if apply_clear(state, host, task_id, att) {
                out.clears_applied += 1;
            }
        }
    }

    // Render ICB execute (`0x14`/`0x15`) — open pass over color slots and run ICB.
    // Counted in FRONT of every gate below, because the only always-on report
    // of this rail (`exec_summary`'s `icb_ok`/`icb_fail`) is emitted solely for
    // packets that already failed, so an ICB that succeeds is invisible there
    // and the whole rail reads as "never runs". This says how often the decoded
    // stream asks for one at all, which is the denominator `runtime/icb`
    // (2818 product lines) has never had.
    //
    // # Measured absent on three driven x86 / Vulkan boots
    //
    // The third is the one that carries the weight, because the first two were
    // compositing-only and could not tell "the guest does not use ICB" from
    // "this workload never reaches Metal":
    //
    //   1. Wikipedia + apple.com + System Settings, three title-bar drags.
    //   2. apple.com, four page-downs.
    //   3. Chess (SceneKit 3D) + Maps + the WebGL aquarium rendering live —
    //      **66 512 draws** and 74.9 ms of `compute_us` across the boot.
    //
    // `icb_exec_seen`, `compute_icb_seen` and `compute_ctrl_seen` are all absent
    // from every one. Across the whole accumulated fail log the subsystem has
    // never emitted a line of its own either (every "icb" string in it is a
    // field *name* on an `exec_summary` line).
    //
    // **This is still not a licence to delete `runtime/icb`, and the precedent
    // that settles it is `ffe31d4`**: `mrt_draw_multi` also measured zero, and
    // that session kept MRT *rendering* because it is decoded contract, cutting
    // only the speculative sampling side-map built around it.
    // `ExecuteCommandsInBuffer` is likewise a real Metal opcode in the decoded
    // stream — a guest that issues one against a decoder we deleted loses work
    // silently, which is the one outcome the ground rules forbid outright. What
    // the reading does license is scrutiny of any layer built *around* the
    // decode on speculation rather than on decoded fields.
    //
    // arm64 is unmeasured; these are x86 / Vulkan readings only.
    //
    // # A stream may ask for several, and every one of them runs
    //
    // `executeCommandsInBuffer:` is not a state a later record replaces — it is
    // work, and Metal's ordinary ICB shape is one buffer per object batch, so
    // several in one encoder is the expected case rather than the odd one.
    // This used to be an `Option` assigned with `=`, which made the stream's
    // capacity for them **one**: a second record overwrote the first and the
    // first's commands never ran, with no counter and no line. That is a bound
    // with no constant to name it, which is why none of the five bound scans
    // could see it.
    //
    // The list is bounded by the stream the way [`StreamDrawDrop`] describes
    // for `draws`: a record has a minimum encoded length, so the count cannot
    // exceed the stream bytes already in memory.
    //
    // Records 2+ open their pass with `MTL_LOAD_ACTION_LOAD` and no clears.
    // Each execute writes back before the next builds its request, so the LOAD
    // seed is the previous execute's output — the clear belongs to the pass,
    // which began at the first one, and re-running it would wipe what the ICB
    // before it drew.
    for (icb_index, exec) in acc.execute_icb.iter().enumerate() {
        crate::runtime::drain::note_store_route("icb_exec_seen");
        if !acc.color_slots.is_empty() {
            // `mrt_draw_request` gates on a non-zero pipeline ref, and an
            // ICB-only execute has none in the stream — its PSO lives inside
            // the filled slots — so 1 stands in and only the colour list is
            // taken. That case also takes the default single-triangle geometry
            // rather than the stream's last draw, because the ICB carries its
            // own. Otherwise the last pass's geometry describes the pass this
            // ICB runs inside.
            let (pipeline, args) = if acc.pipeline_ref != 0 {
                let args = acc.draws.last().map_or(
                    DrawArgs {
                        vertex_count: 1,
                        instance_count: 1,
                        primitive_type: 3,
                        first_vertex: 0,
                        base_instance: 0,
                    },
                    |pd| pd.draw,
                );
                (acc.pipeline_ref, args)
            } else {
                (
                    1,
                    DrawArgs {
                        vertex_count: 1,
                        instance_count: 1,
                        primitive_type: 3,
                        first_vertex: 0,
                        base_instance: 0,
                    },
                )
            };
            // The first execute opens the pass, so it takes the stream's load
            // actions and its clears. Every later one composites onto what the
            // pass already holds.
            let loading_slots;
            let (slots, clears): (&[(u32, ColorAttachment)], &[ColorAttachment]) = if icb_index == 0
            {
                (&acc.color_slots, &acc.clears)
            } else {
                loading_slots = color_slots_loading(&acc.color_slots);
                (&loading_slots, &[])
            };
            let req = draw::mrt_draw_request(state, host, task_id, pipeline, slots, clears, args);
            // ICB execute inherits stream bind state at end of stream, and both
            // branches below inherit the same six tables — the last draw's
            // snapshot is those tables as they stood when it was recorded, and
            // nothing between then and here can have refilled a slot the walk
            // refused. So a bind the tables could not hold is asked about once,
            // ahead of both, rather than only on the second branch.
            let inherited = acc.bind_snapshot();
            if let Err(over) = inherited {
                out.render_icb_fail += 1;
                note_draw_refused(over, pipeline, "icb_execute");
            } else if let (Some(mut req), Ok(snapshot)) = (req, inherited) {
                if let Some(pd) = acc.draws.last() {
                    fill_draw_binds_from_pending(&mut req, pd);
                } else {
                    fill_draw_binds_from_pending(&mut req, &snapshot);
                }
                let (loc, len) = if exec.is_range {
                    (exec.range_location, exec.range_length)
                } else {
                    // Indirect: stage 8-byte range from guest buffer.
                    match read_icb_exec_range(
                        state,
                        host,
                        task_id,
                        exec.args_buffer_ref,
                        exec.args_buffer_offset,
                    ) {
                        Some(v) => v,
                        None => {
                            // Sibling ICB arms all log; this one only bumped the
                            // counter (ICB audit) — name the reason.
                            crate::observe::fail(format!(
                                "render_icb fail reason=exec_range_read args_ref={} args_off={}",
                                exec.args_buffer_ref, exec.args_buffer_offset
                            ));
                            out.render_icb_fail += 1;
                            dirty_color_targets(state, host, task_id, &acc.color_targets);
                            // `continue`, not `return`. One execute whose range
                            // could not be read is one execute lost; it says
                            // nothing about the next one's args buffer, and it
                            // used to abandon the whole packet — including the
                            // stream's own draws below — because there could
                            // only ever be one of these.
                            continue;
                        }
                    }
                };
                match crate::backend::selected().encode_icb_execute_and_writeback(
                    state,
                    host,
                    &req,
                    exec.icb_ref,
                    loc,
                    len,
                ) {
                    EncodeStatus::Ok => {
                        crate::runtime::drain::note_store_route("icb_exec_ok");
                        out.render_icb_ok += 1;
                    }
                    st => {
                        out.render_icb_fail += 1;
                        // Was `st={st:?}` — the variant, Debug-rendered, with no
                        // `reason=` at all, so ten distinct checks in
                        // `encode_icb_execute_and_writeback` (plus every ICB
                        // refusal forwarded into it) shared four names and none
                        // of them was greppable. Latched per ICB: the guest
                        // re-executes the same one every frame.
                        if let Some(e) = crate::observe::Emit::refusal("render_icb", &st) {
                            e.field("icb_ref", exec.icb_ref)
                                .field("loc", loc)
                                .field("len", len)
                                .field("colors", acc.color_slots.len())
                                .fail_once(exec.icb_ref as u64);
                        }
                        dirty_color_targets(state, host, task_id, &acc.color_targets);
                    }
                }
            } else {
                out.render_icb_fail += 1;
                crate::observe::fail(format!(
                    "render_icb fail reason=mrt_request icb_ref={} colors={}",
                    exec.icb_ref,
                    acc.color_slots.len()
                ));
            }
        } else {
            out.render_icb_fail += 1;
            crate::observe::fail("render_icb fail reason=no_color_slots");
        }
        // ICB execute is the primary work; still allow a co-recorded draw below if present.
    }

    if acc.saw_draw && !acc.color_slots.is_empty() && !acc.draws.is_empty() {
        // Archive multi-draw (apple-pv-gpu-exec DrawJob): every honorable draw of
        // one exec packet targets one surface in decode order; the worker threads
        // each record's RGBA output as the next record's initial content; guest
        // writeback + completion stamp happen once for the final image.
        //
        // Chain in-process color0 RGBA8 between encodes (no float16 guest round-
        // trip between draws). Only the last successful encode stores to guest.
        let draw_list: Vec<&PendingDraw> = acc
            .draws
            .iter()
            .filter(|pd| pd.pipeline_ref != 0 && pd.draw.vertex_count > 0)
            .collect();
        let mut chain_rgba: Option<Vec<u8>> = None;
        // Occlusion counts, keyed by the guest byte offset each lands at.
        //
        // Summed rather than replaced because one Metal counter can span
        // several draws and every backend here runs one query per draw: Metal
        // accumulates into the buffer word itself, so the equivalent is the sum
        // of what each draw passed. Several offsets in one pass are legal and
        // independent, which is why this is a map and not a scalar.
        let mut visibility_counts: std::collections::BTreeMap<u64, u64> =
            std::collections::BTreeMap::new();
        // Resident render-pass chain: intermediate records keep their content
        // on the engine target (no CPU chain buffer); records 2+ LoadFromTarget.
        let mut resident_chain = false;
        let mut saw_nometal = false;
        let first_draw = draw_list.first().copied();
        let mut first_req = first_draw.and_then(|pd| {
            out.render_attachment_resolves = out.render_attachment_resolves.saturating_add(1);
            draw::mrt_draw_request(
                state,
                host,
                task_id,
                pd.pipeline_ref,
                &acc.color_slots,
                &acc.clears,
                pd.draw,
            )
        });
        // A serialized Metal render stream is one render pass: its attachment
        // descriptors are fixed while pipeline, binds, and draw arguments may
        // change per record. Keep a seedless template so records 2+ do not
        // re-walk the same guest object list/page tables (or clone a full-frame
        // GVA LOAD seed). The resident target itself preserves record order.
        let attachment_template = first_req.as_ref().map(render_pass_attachment_template);
        if first_draw.is_some() && first_req.is_none() {
            let refs: Vec<u32> = acc.color_slots.iter().map(|(_, a)| a.texture_ref).collect();
            crate::observe::fail(format!(
                "metal_draw mrt_request fail task={task_id} pipe={} slots={refs:?} di=0/{}",
                first_draw.map(|pd| pd.pipeline_ref).unwrap_or(0),
                draw_list.len()
            ));
            out.metal_draws_fail = out.metal_draws_fail.saturating_add(1);
            dirty_color_targets(state, host, task_id, &acc.color_targets);
        }
        for (di, pd) in draw_list.iter().enumerate() {
            fin.enter(crate::runtime::drain::FinishPhase::Retarget);
            let mut req = if di == 0 {
                let Some(req) = first_req.take() else {
                    break;
                };
                req
            } else {
                let Some(template) = attachment_template.as_ref() else {
                    break;
                };
                retarget_render_pass_draw(template, pd)
            };
            {
                fin.enter(crate::runtime::drain::FinishPhase::Binds);
                fill_draw_binds_from_pending(&mut req, pd);
                (req.continues_render_pass, req.render_pass_continues) =
                    render_pass_chain_position(di, draw_list.len());
                // A resident mapper-ref-texture target carries attachment contents between
                // records without a CPU chain buffer. Like a native Metal render
                // pass, only the final record performs the guest-visible Store;
                // importing a full frame after every draw held DeviceInner for
                // seconds and starved the guest completion/status registers.
                let unified = req
                    .colors
                    .first()
                    .map(|c| c.mapping_id != 0)
                    .unwrap_or(false);
                // Records 2+ of a chain composite over the prior record: force
                // loadAction=Load on every color. Leaving the pass action alone
                // on a mapper-ref-texture target let a CLEAR re-run before each record,
                // wiping the full composite drawn by record 1 (live poison=1:
                // mid peak 10.9M native → 2.5M after later records).
                if di > 0 {
                    for c in &mut req.colors {
                        c.load_action = MTL_LOAD_ACTION_LOAD;
                    }
                    // Chain from the engine resident when available; otherwise
                    // seed from the prior encode output (archive "thread each
                    // record's output as next initial content"). MoltenVK's
                    // portability path returns CPU pixels for mapper-ref-texture mappings,
                    // so `unified` does not imply that a resident exists.
                    // Moved, not cloned (multi-MiB).
                    match multi_draw_chain_source(resident_chain, chain_rgba.is_some()) {
                        MultiDrawChainSource::Resident => {
                            req.chain_from_resident = true;
                        }
                        MultiDrawChainSource::Cpu => {
                            if let Some(c0) = req.colors.first_mut() {
                                c0.target_seed =
                                    chain_rgba.take().map(crate::runtime::draw::LoadSeed::rgba8);
                            }
                        }
                        MultiDrawChainSource::Missing => {
                            crate::observe::fail(format!(
                                "multi_draw_chain_break reason=prior_output_missing \
                                 task={task_id} pipe={} di={di}/{} unified={}",
                                pd.pipeline_ref,
                                draw_list.len(),
                                unified as u8
                            ));
                        }
                    }
                }
                let (do_writeback, force_full_store) = multi_draw_store_plan(draw_list.len(), di);
                if do_writeback {
                    out.render_guest_stores = out.render_guest_stores.saturating_add(1);
                }
                let draw_started = std::time::Instant::now();
                fin.enter(crate::runtime::drain::FinishPhase::Encode);
                let encode = crate::backend::selected().encode_draw_chain(
                    state,
                    host,
                    &mut req,
                    do_writeback,
                    force_full_store,
                );
                fin.enter(crate::runtime::drain::FinishPhase::Result);
                // Read before the status is matched: a draw whose Store failed
                // still ran its query, and the count is the guest's answer
                // either way.
                match (req.visibility, req.visibility_samples) {
                    (Some(arming), Some(samples)) => {
                        let slot = visibility_counts.entry(arming.offset).or_default();
                        *slot = slot.saturating_add(samples);
                    }
                    // Armed and unanswered: the draw that ran did not record the
                    // query, so the guest will read its own stale word and cull
                    // on it. Both backends record one now, so what is left here
                    // is the refusal cases — a Vulkan host without
                    // `occlusionQueryPrecise` asked for a counting query, a mode
                    // ordinal neither table converts, an encode that failed
                    // before the pass ran — and any draw form whose encoder does
                    // not carry the arming at all. Detected here rather than in
                    // each backend because the question is the same on all three
                    // pathways: was the query the guest armed actually run.
                    (Some(arming), None) => {
                        crate::runtime::drain::note_store_route("visibility_query_unanswered");
                        if crate::observe::first_sight(
                            "visibility_query_unanswered",
                            u64::from(arming.mode),
                        ) {
                            crate::observe::fail(format!(
                                "visibility_query_unanswered \
                                 reason=visibility_query_unanswered task={task_id} \
                                 pipe={} mode={} off={:#x} (the guest armed an \
                                 occlusion query and this backend ran none; it will \
                                 read whatever its buffer already held)",
                                pd.pipeline_ref, arming.mode, arming.offset
                            ));
                        }
                    }
                    (None, _) => {}
                }
                crate::runtime::drain::note_drain_phase(
                    crate::runtime::drain::DrainPhase::Draw,
                    draw_started,
                );
                match encode {
                    (EncodeStatus::Ok, Some(rgba)) => {
                        out.metal_draws_ok += 1;
                        if !resident_chain {
                            chain_rgba = Some(rgba);
                        }
                    }
                    (EncodeStatus::Ok, None) if req.chain_resident_established => {
                        // Resident render-pass chain intermediate: content stays
                        // on the engine target; the next record loads it there.
                        out.metal_draws_ok += 1;
                        resident_chain = true;
                    }
                    (EncodeStatus::Ok, None) => {
                        // Intermediate must return color0 for chaining; treat as
                        // break so we do not composite later draws on a missing seed.
                        out.metal_draws_ok += 1;
                        if !do_writeback && !unified {
                            // Every draw after this one is dropped, so say so.
                            // The two sibling break arms below report through
                            // `note_draw_encode_fail`; this one encoded `Ok` and
                            // so has no `EncodeStatus` to carry a reason, which
                            // is exactly how it stayed silent while losing the
                            // rest of the packet.
                            crate::observe::Emit::decline(
                                "draw_chain_abandon",
                                &ChainAbandonDecline {
                                    index: di,
                                    total: draw_list.len(),
                                    pipeline_ref: pd.pipeline_ref,
                                },
                            )
                            .field("task", task_id)
                            .fail_once(pd.pipeline_ref as u64);
                            // Land any earlier chain image before abandoning —
                            // same as the hard-fail path below. Dropping the
                            // chain left dual-mid pages black while gen advanced.
                            land_chain_before_abandon(
                                state,
                                host,
                                task_id,
                                acc,
                                &req,
                                &mut chain_rgba,
                                ChainEnd {
                                    cause: draw::ChainAbandonCause::NoColor0,
                                    resident: resident_chain,
                                },
                            );
                            break;
                        }
                    }
                    (st @ EncodeStatus::NoMetal(_), _) => {
                        saw_nometal = true;
                        out.metal_draws_fail += 1;
                        note_draw_encode_fail(task_id, pd.pipeline_ref, st, di, draw_list.len());
                        land_chain_before_abandon(
                            state,
                            host,
                            task_id,
                            acc,
                            &req,
                            &mut chain_rgba,
                            ChainEnd {
                                cause: draw::ChainAbandonCause::NoMetal,
                                resident: resident_chain,
                            },
                        );
                        break;
                    }
                    // `Ok` and the distinct clear-fallback `NoMetal` recovery
                    // are exhausted above. Every remaining status is a typed
                    // terminal refusal, including the Metal-only carrier when
                    // that feature exists.
                    (st, _) => {
                        out.metal_draws_fail += 1;
                        note_draw_encode_fail(task_id, pd.pipeline_ref, st, di, draw_list.len());
                        // If earlier GVA draws produced a chain image, land it
                        // before abandoning the packet. Unified targets already
                        // landed each record in guest memory — never write the
                        // (zero) chain buffer over them.
                        land_chain_before_abandon(
                            state,
                            host,
                            task_id,
                            acc,
                            &req,
                            &mut chain_rgba,
                            ChainEnd {
                                cause: draw::ChainAbandonCause::TerminalRefusal,
                                resident: resident_chain,
                            },
                        );
                        break;
                    }
                }
            }
        }
        fin.enter(crate::runtime::drain::FinishPhase::Tail);
        write_visibility_results(state, host, task_id, acc, &visibility_counts);
        // Encode never landed Stores (NoMetal stubs, missing MTLB/pipeline, or
        // mrt resolve fail). Honor CLEAR load+store into guest/host pages so
        // dual-buffer display mids at least hold the pass clear color (archive
        // CLEAR seed — not a content heuristic). Applies for any draw-fail
        // class, not only NoMetal: mrt_request fail used to skip this and left
        // mid pages empty → nz_swing thrash on x86 Linux product.
        let stream_draws = stream_draw_delta(out, draws_at_entry);
        if stream_draws.ok == 0 && !acc.clears.is_empty() {
            for att in acc.clears_reaching_guest_pages() {
                if apply_clear(state, host, task_id, att) {
                    out.clears_applied = out.clears_applied.saturating_add(1);
                }
            }
            let stream_clears = out.clears_applied.saturating_sub(clears_at_entry);
            if stream_clears > 0 || saw_nometal || stream_draws.fail > 0 {
                crate::observe::fail(format!(
                    "draw_fail_clear_fallback task={task_id} clears={} draws_fail={} nometal={}",
                    stream_clears, stream_draws.fail, saw_nometal as u8
                ));
            }
        }
    }
}

/// Land this stream's occlusion counts in the guest's `visibilityResultBuffer`.
///
/// The guest reads this buffer with its own CPU and culls on what it finds, so
/// a count this device does not write is not a picture that comes out wrong —
/// it is the guest acting on whatever it last initialised. That is why every
/// refusal below is fail-visible: dropping the write silently is the one
/// outcome the ground rules forbid.
///
/// Each result is a little-endian `u64` at `base + offset`, the width
/// `MTLVisibilityResultMode` documents for both of its modes.
///
/// The span is resolved once, here, rather than per draw.
/// `objects::resolve_buffer_span` is the same resolver an indirect-draw buffer
/// and a vertex bind go through, so a guest naming a non-buffer or an unbacked
/// object refuses by that rail's own name instead of a literal invented here.
fn write_visibility_results<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &StreamAccum,
    counts: &std::collections::BTreeMap<u64, u64>,
) {
    if counts.is_empty() {
        return;
    }
    // A pass that armed a query and named no buffer has nowhere to put the
    // answer. The two halves are decoded from separate records, so this device
    // can see a pairing neither record states on its own.
    if acc.visibility_buffer_ref == 0 {
        crate::runtime::drain::note_store_route("visibility_result_no_buffer");
        crate::observe::fail(format!(
            "visibility_result_unwritable reason=visibility_result_no_buffer \
             task={task_id} results={} (a draw armed an occlusion query and the \
             pass named no visibilityResultBuffer; the counts are lost)",
            counts.len()
        ));
        return;
    }
    let (base, size) = match crate::runtime::objects::resolve_buffer_span(
        state,
        host,
        task_id,
        acc.visibility_buffer_ref,
    ) {
        Ok(v) => v,
        Err(refusal) => {
            // Mapped into this rail's own vocabulary rather than reported as
            // one slug, for the reason `resolve_buffer_span` gives: a ref
            // naming nothing, a ref holding some other object, a descriptor
            // that would not decode and one naming no allocation are four
            // different findings, and collapsing them names the last.
            let reason = match refusal {
                crate::runtime::objects::BufferSpanRefusal::Rung(rung) => {
                    crate::observe::ladder_slugs!("visibility_buf")(rung)
                }
                crate::runtime::objects::BufferSpanRefusal::Decode => {
                    crate::observe::ladder_slug!("visibility_buf", desc_decode)
                }
                crate::runtime::objects::BufferSpanRefusal::NoBacking => {
                    "visibility_buf_no_backing"
                }
            };
            crate::runtime::drain::note_store_route(reason);
            crate::observe::fail(format!(
                "visibility_result_unwritable reason={reason} task={task_id} buf={} \
                 results={} (the pass named a visibilityResultBuffer this device \
                 cannot resolve; the counts are lost)",
                acc.visibility_buffer_ref,
                counts.len()
            ));
            return;
        }
    };
    for (&offset, &samples) in counts {
        // Bound each word against the buffer the guest actually allocated. The
        // offset is decoded guest data and the two halves arrive in separate
        // records, so nothing before this point has compared them.
        let Some(end) = offset.checked_add(8) else {
            continue;
        };
        if end > size {
            crate::runtime::drain::note_store_route("visibility_result_offset_past_buffer");
            crate::observe::fail(format!(
                "visibility_result_unwritable reason=visibility_result_offset_past_buffer \
                 task={task_id} buf={} off={offset:#x} size={size} (count {samples} lost)",
                acc.visibility_buffer_ref
            ));
            continue;
        }
        if let Err(e) = crate::runtime::gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            base.saturating_add(offset),
            &samples.to_le_bytes(),
            None,
        ) {
            crate::runtime::drain::note_store_route("visibility_result_write_failed");
            crate::observe::fail(format!(
                "visibility_result_unwritable reason=visibility_result_write_failed \
                 task={task_id} buf={} off={offset:#x} err={e:?}",
                acc.visibility_buffer_ref
            ));
        }
    }
}

/// Why a draw list stopped early while every draw in it had encoded `Ok`.
///
/// This is the one abandon path that no counter can see. `metal_draws_fail`
/// stays 0, so `packet_failed` is false and even the packet-level
/// `exec_indirect2` line is suppressed; the draws after this point are dropped
/// with the packet still reported as successful.
#[derive(Debug)]
struct ChainAbandonDecline {
    /// Index of the record that returned no chain image, and the list length.
    /// A break at 0 of 8 loses a whole composite; a break at 7 of 8 loses one
    /// draw, and the two are not the same defect.
    index: usize,
    total: usize,
    pipeline_ref: u32,
}

impl crate::observe::Decline for ChainAbandonDecline {
    fn slug(&self) -> &'static str {
        "draw_chain_abandoned_without_color0"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("di", format!("{}/{}", self.index, self.total)),
            ("lost", (self.total - self.index - 1).to_string()),
            ("pipe", self.pipeline_ref.to_string()),
        ]
    }
}

/// Seedless fixed-attachment template for records after the first draw in one
/// serialized Metal render pass. Construct fields explicitly so a multi-MiB
/// CPU LOAD seed is not cloned merely to reuse attachment identity/geometry.
/// Position one draw in the decoded Metal render encoder that owns it.
/// A one-draw encoder has neither edge; longer encoders expose exactly one
/// start, one end, and a continuation on both sides of every middle draw.
fn render_pass_chain_position(index: usize, len: usize) -> (bool, bool) {
    debug_assert!(index < len);
    (index > 0, index + 1 < len)
}

fn render_pass_attachment_template(first: &draw::DrawEncodeRequest) -> draw::DrawEncodeRequest {
    let colors = first
        .colors
        .iter()
        .map(|c| draw::ColorRtRequest {
            slot: c.slot,
            texture_ref: c.texture_ref,
            mapping_id: c.mapping_id,
            target_gva: c.target_gva,
            row_stride: c.row_stride,
            width: c.width,
            height: c.height,
            format: c.format,
            sample_count: c.sample_count,
            load_action: MTL_LOAD_ACTION_LOAD,
            store_action: c.store_action,
            clear_color: c.clear_color,
            target_seed: None,
            multisample_source_ref: c.multisample_source_ref,
        })
        .collect();
    draw::DrawEncodeRequest {
        task_id: first.task_id,
        colors,
        ..Default::default()
    }
}

fn retarget_render_pass_draw(
    template: &draw::DrawEncodeRequest,
    draw: &PendingDraw,
) -> draw::DrawEncodeRequest {
    let mut req = template.clone();
    req.pipeline_ref = draw.pipeline_ref;
    req.vertex_count = draw.draw.vertex_count;
    req.instance_count = draw.draw.instance_count;
    req.primitive_type = draw.draw.primitive_type;
    req.first_vertex = draw.draw.first_vertex;
    req.base_instance = draw.draw.base_instance;
    req
}

/// Record a draw whose counts live in a guest buffer rather than in the record.
///
/// `drawPrimitives:indirectBuffer:indirectBufferOffset:` and its indexed
/// sibling. Both used to raise a counter and reach
/// `note_unimplemented_render_opcode`, so the geometry the guest asked for was
/// never drawn — the arm's own comment said it could not be, "because the
/// vertex and instance counts are in the indirect buffer … and this rail
/// replays counts it has read".
///
/// It can be, and the reason is the argument the comment did not follow
/// through: **this rail needs the count on the CPU whatever it does.** The
/// vertex buffers are staged by extent, and the extent is a function of the
/// vertex count, so even a real `vkCmdDrawIndirect` would have had to read the
/// block to know how many bytes to stage. Once it is read, the draw is an
/// ordinary one and takes every rail an ordinary one takes.
///
/// What that costs, stated rather than assumed: the counts are a **snapshot**
/// taken when this record is decoded. A guest that writes them from a compute
/// kernel in the same submission is relying on this device having executed and
/// written back that dispatch first, which it does — compute segments complete
/// before the render stream that follows them — but it is an ordering property
/// of the device rather than of the Metal API, and a design that stopped
/// completing compute before render would break this silently.
/// Record one indirect draw, whose counts live in a guest buffer.
///
/// **Which form this is comes from the record's own variant**, not from the
/// opcode read a second time: an `IndexedIndirect` carries an `IndexRef` and a
/// `PrimitivesIndirect` does not, so "the indexed form with no index buffer"
/// and "the unindexed form" stopped being one shape with a flag beside it.
fn execute_indirect_draw<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    opcode: u32,
    record: &reims_vgpu_protocol::decode::render::DrawRecord,
    acc: &mut StreamAccum,
) {
    use crate::protocol::draw::indirect;
    use reims_vgpu_protocol::decode::render::DrawRecord;

    let (primitive, arguments, index) = match record {
        DrawRecord::PrimitivesIndirect(d) => (d.primitive, d.arguments, None),
        DrawRecord::IndexedIndirect(d) => (d.primitive, d.arguments, Some(d.index)),
        // Taken by the caller's direct arms.
        _ => return,
    };
    let block_len = if index.is_some() {
        indirect::INDEXED_LEN
    } else {
        indirect::UNINDEXED_LEN
    };
    let block = match crate::runtime::compute_exec::read_buffer_window(
        state,
        host,
        task_id,
        arguments.buffer_ref,
        arguments.offset,
        block_len,
    ) {
        Ok(block) => block,
        Err(status) => {
            // The buffer is the whole draw here — there is no fallback count in
            // the record to fall back to — so a read that fails is a refused
            // draw, and `read_buffer_window`'s status already names which rung
            // of the resolve refused. Latched per buffer ref because a guest
            // re-issues the same indirect draw every frame.
            note_indirect_draw_refused(task_id, opcode, arguments, status);
            return;
        }
    };

    let (args, index_start, base_vertex) = match index {
        Some(_) => match indirect::indexed(&block, primitive) {
            Some(v) => (v.0, v.1, v.2),
            None => return,
        },
        None => match indirect::unindexed(&block, primitive) {
            Some(args) => (args, 0, 0),
            None => return,
        },
    };

    acc.saw_draw = true;
    if let Some(index) = index {
        // `indexStart` counts indices, not bytes. The loader is given a byte
        // offset, so it is scaled here by the width the record's own index type
        // declares — `IndexType::bytes`, which is the same table
        // `translate::raster::index_type` reads, rather than a `1 => 4, _ => 2`
        // restatement whose `_` arm answered for an ordinal the record can no
        // longer carry.
        acc.indexed = Some(IndexedDrawInfo {
            index_type: index.index_type.ordinal(),
            index_count: args.vertex_count,
            index_buffer_ref: index.buffer_ref,
            index_buffer_offset: index
                .offset
                .saturating_add(u64::from(index_start).saturating_mul(index.index_type.bytes())),
            base_vertex: i64::from(base_vertex),
        });
    } else {
        // Not `None`-by-omission: an unindexed indirect draw arriving after an
        // indexed one in the same stream must not inherit its index buffer,
        // which is the same rule the direct draw arm applies in its `else`.
        acc.indexed = None;
    }

    // A zero count here is the guest's own, read from its own buffer, and it is
    // a legal empty draw rather than a record this device failed to decode — so
    // it takes the zero-count counter the way a zero-count direct draw does, and
    // the unlatched-pipeline reading beside it stays a loss on both arms.
    if acc.pipeline_ref == 0 {
        acc.dropped_no_pipeline = acc.dropped_no_pipeline.saturating_add(1);
        return;
    }
    if args.vertex_count == 0 {
        acc.dropped_zero_count = acc.dropped_zero_count.saturating_add(1);
        return;
    }
    match acc.bind_snapshot() {
        Ok(snapshot) => acc.draws.push(PendingDraw {
            pipeline_ref: acc.pipeline_ref,
            draw: args,
            ..snapshot
        }),
        Err(over) => note_draw_refused(over, acc.pipeline_ref, "draw_indirect"),
    }
}

fn read_icb_exec_range<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
) -> Option<(u64, u64)> {
    use crate::runtime::compute_exec::read_buffer_window;
    // `read_buffer_window` returns exactly the requested 8 bytes or an error,
    // so both reads are in range; the `try_into().ok()?` pair that used to
    // wrap them could only ever be `Ok`.
    let raw = read_buffer_window(state, host, task_id, buffer_ref, offset, 8).ok()?;
    Some((u64::from(ld32(&raw)), u64::from(ld32(&raw[4..]))))
}

/// Guest store plan for multi-draw record `di` of `draw_count` (0-based).
///
/// Archive DrawJob: one writeback of the final image. Multi-draw builds that
/// image in host memory; the last record must full-frame store even if its
/// scissor is partial (else wallpaper chained earlier never reaches guest).
pub(crate) fn multi_draw_store_plan(draw_count: usize, di: usize) -> (bool, bool) {
    if draw_count == 0 {
        return (false, false);
    }
    let last_i = draw_count - 1;
    let do_writeback = di == last_i;
    let force_full_store = do_writeback && draw_count > 1;
    (do_writeback, force_full_store)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MultiDrawChainSource {
    Resident,
    Cpu,
    Missing,
}

/// The same colour slots, opening with `LOAD` instead of whatever the stream
/// asked for.
///
/// One render pass has one load action per attachment, taken when the pass
/// begins. This device opens a fresh host pass per ICB execute, so the second
/// and later ones have to be told that their pass is a continuation — the
/// alternative is a `CLEAR` re-running mid-pass and wiping what the execute
/// before it drew, which is the same failure the multi-draw chain describes at
/// `di > 0`.
///
/// The clear colour is carried through untouched. It is not read on the `LOAD`
/// path, and blanking it here would put an invented value in the record that a
/// later reader of the request would have no way to distinguish from a decoded
/// one.
fn color_slots_loading(slots: &[(u32, ColorAttachment)]) -> Vec<(u32, ColorAttachment)> {
    slots
        .iter()
        .map(|&(slot, att)| {
            (
                slot,
                ColorAttachment {
                    load_action: MTL_LOAD_ACTION_LOAD,
                    ..att
                },
            )
        })
        .collect()
}

fn multi_draw_chain_source(resident_chain: bool, cpu_chain_ready: bool) -> MultiDrawChainSource {
    if resident_chain {
        MultiDrawChainSource::Resident
    } else if cpu_chain_ready {
        MultiDrawChainSource::Cpu
    } else {
        MultiDrawChainSource::Missing
    }
}

fn fill_draw_binds_from_pending(req: &mut draw::DrawEncodeRequest, pd: &PendingDraw) {
    req.vertex_buffers.clone_from(&pd.vertex_buffers);
    req.fragment_buffers.clone_from(&pd.fragment_buffers);
    req.vertex_textures.clone_from(&pd.vertex_textures);
    req.fragment_textures.clone_from(&pd.fragment_textures);
    req.vertex_samplers.clone_from(&pd.vertex_samplers);
    req.fragment_samplers.clone_from(&pd.fragment_samplers);
    req.viewports.clone_from(&pd.viewports);
    req.scissors.clone_from(&pd.scissors);
    req.indexed = pd.indexed.clone();
    req.blend_color = pd.blend_color;
    req.cull_mode = pd.cull_mode;
    req.front_facing = pd.front_facing;
    req.fill_mode = pd.fill_mode;
    req.depth_clip_mode = pd.depth_clip_mode;
    req.line_width = pd.line_width;
    req.depth_bias = pd.depth_bias;
    req.depth_stencil_ref = pd.depth_stencil_ref;
    req.stencil_ref = pd.stencil_ref;
    req.depth_attach = pd.depth_attach;
    req.stencil_attach = pd.stencil_attach;
    req.visibility = pd.visibility;
    // Cleared with the arming it belongs to. `req` is reused across the draws
    // of a chain, so a stale count from draw N-1 would otherwise be read as
    // draw N's — and an occlusion count that is silently the previous draw's is
    // the exact shape of wrong this rail exists to avoid.
    req.visibility_samples = None;
}

fn dirty_color_targets<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    refs: &[u32],
) {
    for &tex_ref in refs {
        if let Some(mid) = objects::resolve_mapper_ref_texture(state, host, task_id, tex_ref) {
            // The guest pages are the only copy of a mapper-ref-texture surface, so there
            // is no mirror to drop — only bump gen for scanout skips.
            let _ = state.mark_mapping_written(mid);
        } else if objects::resolve_backing(state, host, tex_ref) {
            let _ = state.mark_mapping_written(tex_ref);
        }
    }
}

/// How a packet's chain ended: which break stopped it, and whether the last
/// record left its pixels on the engine-resident target rather than in guest
/// memory. Both are answers to "what state was the chain in when it broke", and
/// the recovery rail needs each for a different reason — `resident` decides
/// whether a readback is owed at all, `cause` is what the refusal reports.
#[derive(Clone, Copy)]
struct ChainEnd {
    cause: draw::ChainAbandonCause,
    resident: bool,
}

/// Land the chain image this packet has produced before abandoning it.
///
/// Three records break a multi-draw chain: a typed terminal refusal, the
/// `NoMetal` carrier, and an intermediate that returned no colour0. All three
/// leave earlier GVA draws' pixels only on the engine target, and dropping
/// them left dual-mid pages black while the content generation advanced — so
/// the resident is read back and written out first, and the colour targets are
/// marked written either way.
///
/// Unified targets already landed each record in guest memory and must never
/// take the (zero) chain buffer over them; the one caller where that is
/// possible gates on it.
fn land_chain_before_abandon<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &StreamAccum,
    req: &draw::DrawEncodeRequest,
    chain_rgba: &mut Option<Vec<u8>>,
    end: ChainEnd,
) {
    // The one caller that has no identity to be handed: the chain broke, so no
    // span carries the key its last good record registered. Which is why the
    // rail derives the target itself here — see
    // `Backend::read_abandoned_chain_rgba`, and
    // `draw::M2vDrawSpan::ResidentSurfaceStore` for what sharing that
    // derivation with the callers that *do* hold a key once cost.
    if end.resident && chain_rgba.is_none() {
        *chain_rgba = crate::backend::selected().read_abandoned_chain_rgba(state, req);
    }
    if let Some(rgba) = chain_rgba.take() {
        let _ =
            draw::writeback_chain_rgba(state, host, task_id, &acc.color_slots, &rgba, end.cause);
    }
    dirty_color_targets(state, host, task_id, &acc.color_targets);
}

/// Where a clear-only pass publishes its single-sample result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClearPublish {
    /// Publish into the attachment's own texture, exactly as declared.
    Direct,
    /// Publish into the resolve texture instead of the multisample one.
    Resolved(u32),
    /// Preserve the multisample attachment and publish its resolved value.
    /// These are two distinct destinations and neither substitutes for the
    /// other.
    StoredAndResolved { source: u32, resolve: u32 },
    /// This store action publishes no single-sample result, or there is no
    /// attachment texture at all. Not a loss: the guest asked for nothing.
    NotPublished,
    /// A resolve-carrying store action naming no resolve texture. The guest
    /// asked for a resolve and gave nowhere to put it.
    ResolveTargetMissing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StoreAndResolveClearDecline {
    source: u32,
    resolve: u32,
}

impl crate::observe::Decline for StoreAndResolveClearDecline {
    fn slug(&self) -> &'static str {
        "clear_store_and_multisample_resolve_unsupported"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("source", self.source.to_string()),
            ("resolve", self.resolve.to_string()),
        ]
    }
}

/// Which texture a clear-only pass's colour attachment publishes into.
///
/// `MTLLoadActionClear` with no draws leaves every sample holding `clearColor`,
/// so a multisample resolve publishes that colour into `resolveTexture`.
/// `MTLStoreActionStoreAndMultisampleResolve` additionally preserves the source
/// attachment; it is therefore a distinct two-destination result.
fn clear_publish_target(att: &ColorAttachment) -> ClearPublish {
    if att.texture_ref == 0 || !store_action_publishes_single_sample(att.store_action) {
        return ClearPublish::NotPublished;
    }
    if matches!(
        att.store_action,
        MTL_STORE_ACTION_MULTISAMPLE_RESOLVE | MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
    ) {
        if att.resolve_texture_ref == 0 {
            return ClearPublish::ResolveTargetMissing;
        }
        if att.store_action == MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE {
            return ClearPublish::StoredAndResolved {
                source: att.texture_ref,
                resolve: att.resolve_texture_ref,
            };
        }
        return ClearPublish::Resolved(att.resolve_texture_ref);
    }
    ClearPublish::Direct
}

fn apply_clear<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    att: &ColorAttachment,
) -> bool {
    let target = match clear_publish_target(att) {
        // Declared single-sample: published exactly as the guest stated it,
        // level and all.
        ClearPublish::Direct => *att,
        // A resolve: the clear lands in the resolve texture as an ordinary
        // single-sample store. Level zero because a resolve target has one.
        ClearPublish::Resolved(texture_ref) => ColorAttachment {
            texture_ref,
            resolve_texture_ref: 0,
            level: 0,
            store_action: MTL_STORE_ACTION_STORE,
            ..*att
        },
        ClearPublish::StoredAndResolved { source, resolve } => {
            // This helper can publish one single-sample texture. Writing only
            // `resolve` would silently discard the independently retained
            // multisample `source`, while treating the source as a linear image
            // would write only one sample. Refuse the unsupported pair as one
            // contract operation.
            crate::observe::Emit::decline(
                "render_clear",
                &StoreAndResolveClearDecline { source, resolve },
            )
            .fail();
            return false;
        }
        ClearPublish::NotPublished => return false,
        ClearPublish::ResolveTargetMissing => {
            crate::observe::fail(format!(
                "render_clear reason=clear_multisample_resolve_target_missing source={} \
                 store={}",
                att.texture_ref, att.store_action
            ));
            return false;
        }
    };
    // Prefer full draw-path resolve (mapper-ref-texture or normal-texture GVA wallpaper targets).
    let Some(req) =
        // A clear-only pass: no pipeline and no geometry, so every draw
        // argument including the base instance is zero by construction.
        draw::color_target_request(state, host, task_id, target, 0, 0, 1, 0, 0, 0)
    else {
        // A clear whose color target cannot resolve (mapping unresolved, geometry
        // missing) is dropped here with no other trace — the "background didn't
        // clear cleanly" class. Make it visible, deduped per target.
        note_clear_dropped(
            "target_unresolved",
            att.texture_ref,
            "color_target_request=none",
        );
        return false;
    };
    let c0 = req.colors.first().unwrap_or_else(|| unreachable!());
    // A multisample attachment has no single-sample linear publication, and
    // this is the first point at which that is knowable: `clear_publish_target`
    // above decides from the store action alone, and the sample count arrives
    // with the resolved target.
    //
    // The rule is the one the `StoredAndResolved` arm already states —
    // "treating the source as a linear image would write only one sample" — and
    // it applies just as much to a plain `MTLStoreActionStore` on a texture
    // whose descriptor declares four samples. That arm could not reach this
    // case because the store action does not name it.
    //
    // The guest sizes and strides these allocations for their samples. On rail
    // macos-15 the 300x300 four-sample tiles carry `bpr = 4800`, exactly four
    // times a 300-wide BGRA8 tight row, against `bpr = 1216` on the
    // single-sample surfaces beside them. So writing a single-sample image here
    // is not a partial answer: it fills 1200 bytes of every 4800-byte row with
    // a solid colour and leaves the rest, in a sample layout this device has
    // never established. Refusing leaves the guest the bytes it already had,
    // which is what every other refusal on this rail promises.
    //
    // This narrows the clear rail; it does not close it. A resolve destination
    // is single-sample by construction and still publishes, through
    // `ClearPublish::Resolved` above.
    if c0.sample_count > 1 {
        note_clear_dropped(
            "clear_multisample_source_not_linear",
            att.texture_ref,
            &format!(
                "samples={} {}x{} bpr={} gva={:#x} mid={} store={} (the guest \
                 strided this span for its samples; a one-sample image is the \
                 wrong content for it, not a partial one)",
                c0.sample_count,
                c0.width,
                c0.height,
                c0.row_stride,
                c0.target_gva,
                c0.mapping_id,
                att.store_action
            ),
        );
        return false;
    }
    // Format and clear representation are one contract decision. Continuous
    // colour keeps the semantic RGBA8 carrier the existing converters consume;
    // integer targets carry their own texels, where `1` remains the integer 1.
    let Some(clear) =
        pixel_format::solid_clear_image(c0.format, c0.width, c0.height, &att.clear_color)
    else {
        note_clear_dropped(
            "target_clear_image_unrepresentable",
            att.texture_ref,
            "the admitted target has no CPU clear representation",
        );
        return false;
    };
    if c0.target_gva != 0 {
        let frame = match clear.encoding() {
            ClearImageEncoding::Rgba8 => draw::FrameRows::Rgba8(clear.pixels()),
            ClearImageEncoding::Native => draw::FrameRows::Native(clear.pixels()),
        };
        let ok = draw::write_gva_frame_within(
            state,
            host,
            task_id,
            c0.target_gva,
            c0.width,
            c0.height,
            c0.row_stride,
            c0.format,
            frame,
            None,
        )
        .is_ok();
        if ok {
            crate::runtime::surface_cache::forget_gva_copies(
                state,
                task_id,
                c0.target_gva,
                att.texture_ref,
            );
        }
        return ok;
    }
    if c0.mapping_id == 0 {
        return false;
    }
    let ok = match clear.encoding() {
        ClearImageEncoding::Rgba8 => mapping_write::write_rgba8_image_changed(
            state,
            host,
            c0.mapping_id,
            clear.pixels(),
            None,
            c0.width,
            c0.height,
            mapping_write::FramePublication::HostCache,
        ),
        ClearImageEncoding::Native => mapping_write::write_native_image(
            state,
            host,
            c0.mapping_id,
            clear.pixels(),
            clear.row_bytes(),
            c0.width,
            c0.height,
            c0.format,
        ),
    };
    if ok {
        state.note_surface_clear(c0.mapping_id);
    }
    ok
}

pub(crate) mod finish_phase;

mod report;
use report::{
    note_clear_dropped, note_color_subresource_unsupported, note_compute_refusal,
    note_depth_stencil_unsupported, note_draw_encode_fail, note_empty_scissor,
    note_empty_viewport_or_scissor, note_indexed_draw_without_buffer, note_indirect_draw_refused,
    note_info_record_unanswered, note_pass_array_length_unsupported, note_pass_extent_for_slot,
    note_pass_raster_sample_count_unsupported, note_pass_target_extent, note_residency_declaration,
    note_store_action_no_attachment, note_store_action_options_unsupported, note_stream_draw_drops,
    note_unimplemented_render_opcode, note_unnamed_icb_execute,
};
// The unimplemented-opcode latch is test-only on both sides, so its import has
// to carry the same gate the items do.
#[cfg(test)]
use report::{
    note_pass_extent_coverage, pass_extent_band, reset_unimplemented_opcode_dedup_for_test,
    PASS_EXTENT_SLUGS, UNIMPL_TEST_LOCK,
};

#[cfg(test)]
mod tests;
