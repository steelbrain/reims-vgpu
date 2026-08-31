//! CmdExecIndirect2: load streams, multi-attachment clears, Metal draw attempt.
//!
//! Clear-only passes write guest mapping pages (archive render_clear).
//! Draws try Metal encode when pipeline MTLBs resolve; otherwise color targets
//! are still marked dirty for DisplaySwap.

use crate::contract::draw::DrawArgs;
use crate::contract::endian::{ld32, ld64};
use crate::contract::pass_action::{
    store_action_publishes_single_sample, MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_LOAD,
    MTL_STORE_ACTION_MULTISAMPLE_RESOLVE, MTL_STORE_ACTION_STORE,
    MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
};
use crate::contract::pixel_format::{self, ClearImageEncoding};
use crate::model::DeviceState;
use crate::runtime::blit_exec::{self, BlitStatus};
use crate::runtime::compute_exec::{self, ComputeStatus};
use crate::runtime::decode::blit::{self, Kind as BlitKind};
use crate::runtime::decode::compute::{self, Kind as ComputeKind};
use crate::runtime::decode::event as event_decode;
use crate::runtime::decode::fifo::{decode_exec_resource_table, ExecResourceDesc};
use crate::runtime::decode::fifo::{
    CHILD_EXEC_INDIRECT_CMDBUF_COUNT, CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN,
    CHILD_EXEC_INDIRECT_CMDBUF_GVA, CHILD_EXEC_INDIRECT_CMDBUF_LENGTH,
    CHILD_EXEC_INDIRECT_HEADER_LEN, CHILD_EXEC_INDIRECT_RESOURCE_COUNT,
    CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN, CHILD_EXEC_INDIRECT_TASK_ID,
};
use crate::runtime::decode::render::{
    self, attachment_subresource_is_bindable, color_attachment_subresource_is_bindable,
    decode_color_attachment, decode_depth_attachment, decode_stencil_attachment, ColorAttachment,
    DepthAttachment, Kind as RenderKind, LevelSupport, ScissorRect, Stage, StencilAttachment,
    PASS_MAX_COLOR_ATTACHMENTS,
};
use crate::runtime::decode::stream::{
    self, decode_first_record, decode_next_record, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE,
    SEGMENT_TYPE_EVENT, SEGMENT_TYPE_INFO, SEGMENT_TYPE_RENDER,
};
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
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};
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
/// same stream (journal: logo RG8 168×206 + pill → one type-11 FB).
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
    pub type11_mappings: Vec<u32>,
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

pub fn process_exec_indirect2<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    payload: &[u8],
) -> ExecResult {
    let exec_started = std::time::Instant::now();
    let mut out = ExecResult::default();
    if payload.len() < CHILD_EXEC_INDIRECT_HEADER_LEN as usize {
        return out;
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
        return out;
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
        return out;
    }
    if cmdbuf_count == 0 {
        crate::observe::fail(format!(
            "exec_indirect2 zero_cbufs task={task_id} res={resource_count} plen={}",
            payload.len()
        ));
        return out;
    }

    // The guest declares, per resource this submission touches, who owns the
    // authoritative bytes afterwards. `need` above already proved the table fits,
    // so a refusal here means the header and the decoder disagree about the
    // layout — which is a fail line, never a silent empty table.
    let resource_descs = decode_exec_resource_table(payload).unwrap_or_else(|| {
        crate::observe::fail(format!(
            "exec_res_table decode_fail task={task_id} res={resource_count} plen={}",
            payload.len()
        ));
        Vec::new()
    });

    // Every command buffer the header declares, because `need` above already
    // bounded how many there can be: the guest cannot claim a table longer than
    // the descriptors it actually supplied, so `cmdbuf_count` is capped by
    // `payload.len() / CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN` and `with_capacity`
    // below cannot be talked into an allocation the payload does not back.
    //
    // A fixed ceiling used to sit here and truncate with `.min()`, above the
    // check that already bounded the same number. Nothing derived it — a
    // submission of 17 lost its last command buffer entirely, before the loop,
    // with no fail line, which is a whole packet of guest draws vanishing into a
    // silently shorter table.
    let n_cb = cmdbuf_count as usize;
    let page_shift = state.page_shift;
    let mut streams = Vec::with_capacity(n_cb);
    // This call's measured spans, summed, so `Header` can be the leftover. The
    // census's own totals cover the whole window and cannot answer for one call.
    let mut measured_ns = 0u64;
    let load_started = std::time::Instant::now();
    for i in 0..n_cb {
        // `need` already pinned the whole table: i < n_cb <= cmdbuf_count, so
        // off + DESC_LEN = cbufs_off + (i + 1) * DESC_LEN <= need <=
        // payload.len(). The bounds check that stood here could not fire, and
        // its `break` would have dropped every remaining command buffer with no
        // line if it ever had.
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
        out.streams_loaded += 1;
        streams.push(stream);
    }
    let load_ns = load_started.elapsed().as_nanos() as u64;
    measured_ns += load_ns;
    crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Load, load_ns);

    // Plan before execute: cold AIR translation is immutable CPU work and can
    // run without protocol ownership. Keep the packet unconsumed until every
    // referenced render stage is ready, so replay cannot duplicate clears,
    // fences, compute dispatches, or guest writeback.
    #[cfg(feature = "backend-vulkan")]
    let translation_pending = {
        let preflight_started = std::time::Instant::now();
        let pending = streams.iter().fold(false, |pending, stream| {
            let render_pending = preflight_render_translations(state, host, task_id, stream);
            let compute_pending = preflight_compute_translations(state, host, task_id, stream);
            render_pending || compute_pending || pending
        });
        let preflight_ns = preflight_started.elapsed().as_nanos() as u64;
        measured_ns += preflight_ns;
        crate::runtime::drain::note_exec_phase(
            crate::runtime::drain::ExecPhase::Preflight,
            preflight_ns,
        );
        pending
    };
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    let translation_pending = false;
    if translation_pending {
        out.deferred = true;
        note_exec_header(exec_started, measured_ns);
        return out;
    }

    // Before any of this submission's work runs. Each record states what was
    // true of its resource *before* the submission, so a pending window holding
    // pixels the guest has since overwritten has to go now — landing it later
    // would replace the guest's own bytes with a frame the guest has declared
    // stale.
    consume_resource_table(state, task_id, &resource_descs);

    for stream in streams {
        let mut acc = StreamAccum::default();
        let walk_started = std::time::Instant::now();
        walk_stream(state, host, task_id, &stream, &mut out, &mut acc);
        let walk_ns = walk_started.elapsed().as_nanos() as u64;
        measured_ns += walk_ns;
        crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Walk, walk_ns);
        let finish_started = std::time::Instant::now();
        finish_stream(state, host, task_id, &mut out, &acc);
        let finish_ns = finish_started.elapsed().as_nanos() as u64;
        measured_ns += finish_ns;
        crate::runtime::drain::note_exec_phase(crate::runtime::drain::ExecPhase::Finish, finish_ns);
    }
    note_exec_header(exec_started, measured_ns);
    out.total_us = elapsed_us(exec_started);
    out
}

/// Close the [`ExecPhase`] tiling of `process_exec_indirect2` at one of its
/// return points.
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
/// `objects::resolve_type11_ref` and `resolve_type4_surface_ex` at the moment a
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
fn consume_resource_table(state: &mut DeviceState, task_id: u32, descs: &[ExecResourceDesc]) {
    use crate::runtime::resource_validity::{apply, ValiditySite};
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

#[cfg(feature = "backend-vulkan")]
fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    let refs_started = std::time::Instant::now();
    let pipelines = render_pipeline_refs(stream);
    note_preflight_part(
        PreflightPart::Refs,
        refs_started.elapsed().as_nanos() as u64,
    );
    let mut pending = false;
    for pipeline_ref in pipelines {
        note_preflight_pipe();
        // The draw path's own memo already knows whether these two shaders are
        // translated, and answers for ~0.6 us against the 4.3 us of guest
        // resolves below. `translations_ready` states why that is not a weaker
        // answer — chiefly that the translate cache never evicts, so a shader
        // this memo saw translated is still translated.
        if crate::runtime::pipeline_resolve::translations_ready(state, host, task_id, pipeline_ref)
        {
            continue;
        }
        let air_started = std::time::Instant::now();
        // The MTLB containers, not owned copies of the AIR inside them: the two
        // `ensure_cached_async` calls below borrow, digest and drop, so copying
        // first would allocate twice per pipeline ref for bytes nothing keeps.
        let pair = draw::load_render_mtlb_pair(state, host, task_id, pipeline_ref);
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Ok((v_mtlb, f_mtlb)) = pair else {
            // Normal execution emits the precise pipeline/MTLB failure. A
            // missing plan input is deterministic, not asynchronous work.
            continue;
        };
        // A container whose AIR will not extract is the same "deterministic
        // missing plan input" as one that would not load: normal execution
        // reports it precisely, and there is no asynchronous work to await.
        let (Ok(v_air), Ok(f_air)) = (
            crate::runtime::mtlb::extract_air(&v_mtlb),
            crate::runtime::mtlb::extract_air(&f_mtlb),
        ) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        if !crate::runtime::m2v_cache::ensure_cached_async(
            v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) {
            pending = true;
        }
        if !crate::runtime::m2v_cache::ensure_cached_async(
            f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ) {
            pending = true;
        }
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
    }
    pending
}

#[cfg(feature = "backend-vulkan")]
fn render_pipeline_refs(stream: &[u8]) -> Vec<u32> {
    // Deliberately silent on a framing refusal: this is a speculative pre-scan of
    // the very stream `walk_stream` is about to frame and report on. Logging here
    // would double every `stream_frame_fail` line for no added information.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut pipelines = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_RENDER {
            continue;
        }
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = render::decode(bytes) {
                    if cmd.kind == RenderKind::SetPipeline
                        && cmd.pipeline_ref != 0
                        && !pipelines.contains(&cmd.pipeline_ref)
                    {
                        pipelines.push(cmd.pipeline_ref);
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }

    pipelines
}

#[cfg(feature = "backend-vulkan")]
fn preflight_compute_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    let refs_started = std::time::Instant::now();
    let inputs = compute_translation_inputs(stream);
    note_preflight_part(
        PreflightPart::Refs,
        refs_started.elapsed().as_nanos() as u64,
    );
    let mut pending = false;
    for (pipeline_ref, local_size) in inputs {
        note_preflight_pipe();
        let air_started = std::time::Instant::now();
        let loaded = compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
            .and_then(|pipeline| {
                crate::runtime::mtlb::load_mtlb(
                    state,
                    host,
                    task_id,
                    pipeline.kernel_func_ref,
                    crate::runtime::mtlb::AirLoadRail::Compute,
                )
            });
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Some(mtlb) = loaded else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        let cached =
            crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref);
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
        if !cached {
            pending = true;
        }
    }
    pending
}

/// Structurally collect compute pipeline + LocalSize pairs in command order.
/// Threads-indirect carries LocalSize in guest argument memory rather than the
/// stream record, so it deliberately remains on the synchronous fallback.
#[cfg(feature = "backend-vulkan")]
fn compute_translation_inputs(stream: &[u8]) -> Vec<(u32, [u32; 3])> {
    // Silent for the same reason as `render_pipeline_refs`: a pre-scan whose
    // framing refusal `walk_stream` will report once, with the task attached.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut inputs = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_COMPUTE {
            continue;
        }
        let mut pipeline_ref = 0u32;
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = compute::decode(bytes) {
                    match cmd.kind {
                        ComputeKind::Pipeline => pipeline_ref = cmd.pipeline_ref,
                        ComputeKind::DispatchThreadgroups
                        | ComputeKind::DispatchThreadgroupsIndirect
                        | ComputeKind::DispatchThreads => {
                            let dims = cmd.threads_per_threadgroup;
                            let local_size = [
                                u32::try_from(dims.x).ok(),
                                u32::try_from(dims.y).ok(),
                                u32::try_from(dims.z).ok(),
                            ];
                            if pipeline_ref != 0 {
                                if let [Some(x), Some(y), Some(z)] = local_size {
                                    let item = (pipeline_ref, [x, y, z]);
                                    if x != 0 && y != 0 && z != 0 && !inputs.contains(&item) {
                                        inputs.push(item);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }
    inputs
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
fn walk_segment_records(stream: &[u8], seg: &stream::Segment, mut handle: impl FnMut(u32, &[u8])) {
    let mut cursor = 0usize;
    let mut records = 0u64;
    let mut next = decode_first_record(stream, seg, &mut cursor);
    let (route, route_us) = match seg.type_ {
        SEGMENT_TYPE_RENDER => ("walk_records_render", "walk_render_us"),
        SEGMENT_TYPE_BLIT => ("walk_records_blit", "walk_blit_us"),
        SEGMENT_TYPE_COMPUTE => ("walk_records_compute", "walk_compute_us"),
        _ => ("walk_records_other", "walk_other_us"),
    };
    // One clock pair per *segment*, not per record. A stream carries at most a
    // handful of segments and tens of thousands of records, so this splits
    // `exec_phase walk_us` by family for a cost that does not show up, where
    // per-record timing would cost more than the handlers it measured.
    let started = std::time::Instant::now();
    loop {
        match next {
            Ok(rec) => {
                records += 1;
                let start = rec.bytes_offset as usize;
                handle(rec.opcode, &stream[start..start + rec.length as usize]);
                next = decode_next_record(stream, seg, &mut cursor);
            }
            // `Done` is end-of-segment and yields `None` here, so the normal exit
            // path stays silent; anything else names the check that refused.
            Err(status) => {
                crate::runtime::drain::note_store_route_n(route, records);
                crate::runtime::drain::note_store_route_us(
                    route_us,
                    started.elapsed().as_micros() as u64,
                );
                if let Some(e) = crate::observe::Emit::refusal("stream_record_fail", &status) {
                    // Latch per segment family: a guest re-submitting a malformed
                    // stream sends it on every frame and the second line carries
                    // nothing the first did not. Keying on the family still tells
                    // a broken blit segment from a broken render one, which
                    // keying on the reason alone would hide.
                    e.field("seg", stream::segment_type_name(u32::from(seg.type_)))
                        .field("seg_off", seg.offset)
                        .field("seg_len", seg.length)
                        .field("cursor", cursor)
                        .fail_once(u64::from(seg.type_));
                }
                return;
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
) {
    let segs = match stream::iter_segments(stream) {
        Ok(s) => s,
        Err(status) => {
            // The outermost frame in the crate. A stream that will not frame
            // executes *nothing* — and until now that was indistinguishable from
            // an idle guest: no records, no work, no line.
            if let Some(e) = crate::observe::Emit::refusal("stream_frame_fail", &status) {
                e.field("task", task_id)
                    .field("bytes", stream.len())
                    .fail_once(u64::from(task_id));
            }
            return;
        }
    };
    for seg in segs {
        if let Some(e) =
            crate::observe::Emit::refusal("stream_segment", &stream::segment_disposition(seg.type_))
        {
            e.field("seg_type", seg.type_)
                .field("seg_off", seg.offset)
                .field("seg_len", seg.length)
                .fail_once(u64::from(seg.type_));
            continue;
        }
        match seg.type_ {
            SEGMENT_TYPE_RENDER => {
                walk_segment_records(stream, &seg, |op, cmd| {
                    handle_render_record(state, host, task_id, op, cmd, out, acc)
                });
            }
            SEGMENT_TYPE_BLIT => {
                walk_segment_records(stream, &seg, |op, cmd| {
                    handle_blit_record(state, host, task_id, op, cmd)
                });
            }
            SEGMENT_TYPE_COMPUTE => {
                let mut compute = crate::runtime::compute_session::ComputeSegment::default();
                walk_segment_records(stream, &seg, |op, cmd| {
                    handle_compute_record(state, host, task_id, op, cmd, out, &mut compute)
                });
                if let Some(st) = crate::runtime::compute_session::finish_session(
                    &mut compute.session,
                    state,
                    host,
                    task_id,
                ) {
                    if !matches!(st, ComputeStatus::Ok) {
                        out.compute_control_fail += 1;
                        // Segment-end commit: the whole multi-record session's
                        // work is gone, and this counter was its only trace.
                        if let Some(e) =
                            crate::observe::Emit::refusal("compute_session_finish", &st)
                        {
                            e.field("task", task_id).fail_once(u64::from(task_id));
                        }
                    }
                }
            }
            SEGMENT_TYPE_EVENT => {
                walk_segment_records(stream, &seg, |_op, cmd| {
                    handle_event_record(state, task_id, cmd)
                });
            }
            SEGMENT_TYPE_INFO => {
                walk_segment_records(stream, &seg, |op, cmd| {
                    handle_info_record(state, host, task_id, op, cmd)
                });
            }
            // Unreachable: `segment_disposition` already answered `Walk` for
            // exactly the five families above, and `continue`d on the rest.
            _ => {}
        }
    }
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
        // was malformed, the type-1 buffer was short, or the pathway has no ICB
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
}

fn handle_event_record(state: &mut DeviceState, task_id: u32, cmd_bytes: &[u8]) {
    let cmd = match event_decode::decode(cmd_bytes) {
        Ok(c) => c,
        Err(status) => {
            // A malformed event record drops a guest signal or wait outright.
            // The `Err(_)` here used to feed a counter nothing read, so the loss
            // left no line at all; the decoder's own typed refusal names which
            // of its five checks rejected the bytes.
            if let Some(e) = crate::observe::Emit::refusal("event_decode", &status) {
                e.field("task", task_id)
                    .field("len", cmd_bytes.len())
                    .fail();
            }
            return;
        }
    };
    // Refusals are emitted by `execute_event` itself, against the ref that
    // failed; there is nothing left for this caller to report.
    fence_exec::execute_event(state, task_id, &cmd);
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
    let cmd = match compute::decode(cmd_bytes) {
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
        ComputeKind::BufferBind | ComputeKind::BufferBindAttributeStride => {
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
        }
        ComputeKind::TextureBind => {
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
        }
        ComputeKind::SamplerBind | ComputeKind::SamplerLod => {
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
        }
        ComputeKind::Pipeline
        | ComputeKind::BufferOffset
        | ComputeKind::BufferOffsetAttributeStride
        | ComputeKind::DispatchType
        | ComputeKind::StageInRegion
        | ComputeKind::StageInRegionIndirect
        | ComputeKind::ThreadgroupMemory
        | ComputeKind::ImageblockDimensions
        | ComputeKind::BarrierResources
        | ComputeKind::BarrierScope
        | ComputeKind::UseHeaps
        | ComputeKind::UseResources
        | ComputeKind::CompressedTextureFlush => {
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
        }
        ComputeKind::DispatchThreadgroups
        | ComputeKind::DispatchThreads
        | ComputeKind::DispatchThreadgroupsIndirect
        | ComputeKind::DispatchThreadsIndirect => {
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                // `None` is an accumulator-only record kind, not a loss: the
                // record was applied, `apply_record` simply had no execution
                // status to report for it.
                None | Some(ComputeStatus::Ok) => {}
                Some(st) => note_compute_refusal(st, task_id, pipeline_ref, cmd.kind),
            }
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
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                None | Some(ComputeStatus::Ok) => {}
                Some(st) => {
                    out.compute_control_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, cmd.kind);
                }
            }
        }
        ComputeKind::ExecuteCommandsInBuffer | ComputeKind::ExecuteCommandsInBufferIndirect => {
            crate::runtime::drain::note_store_route("compute_icb_seen");
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                None | Some(ComputeStatus::Ok) => {}
                Some(st) => {
                    out.compute_icb_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, cmd.kind);
                }
            }
        }
        _ => {}
    }
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
struct TextureFillDropped(blit::FillSource);

impl crate::observe::Decline for TextureFillDropped {
    fn slug(&self) -> &'static str {
        match self.0 {
            blit::FillSource::Color => "blit_fill_texture_color_dropped",
            blit::FillSource::Bytes => "blit_fill_texture_bytes_dropped",
            // Unreachable while both decode arms set the source: `FillSource`
            // defaults to `None` and only a `Kind::FillTexture` gets here. A
            // firing means a third fill form reached this kind without naming
            // where its value comes from. A healthy zero.
            blit::FillSource::None => "blit_fill_texture_source_unset",
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
    let cmd = match blit::decode(cmd_bytes) {
        Ok(c) => c,
        // Was `Err(_) => return`: a decoded blit record dropped with no line at
        // all, which on a live boot is indistinguishable from a segment that
        // carried no blit work. The status names which of the four checks
        // refused.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("blit_decode", &status) {
                e.field("opcode", format!("{:#x}", opcode))
                    .field("len", cmd_bytes.len())
                    .fail();
            }
            return;
        }
    };
    match cmd.kind {
        BlitKind::Resource if cmd.opcode == wire_blit::OPCODE_GENERATE_MIPMAPS => {
            match mipmap::generate_mipmaps_linear(state, host, task_id, cmd.resource) {
                MipmapStatus::Ok => {}
                st => {
                    // Was `st={st:?}` with no `reason=` at all, so none of the
                    // eight outcomes was greppable and the Debug spelling was
                    // the only handle on which check refused.
                    if let Some(e) = crate::observe::Emit::refusal("blit_generate_mipmaps", &st) {
                        e.field("resource", cmd.resource).fail();
                    }
                }
            }
        }
        // optimize*/synchronize* are protocol no-ops on the unified-memory path.
        BlitKind::Resource | BlitKind::Image => {}
        // The three indirect-command-buffer records. All three used to be
        // refused before decode under one shared reason, which said three
        // different things with one word — and only two of them are losses.
        //
        // `optimizeIndirectCommandBuffer:` is Metal's hint that a range will be
        // reused, so skipping it is semantically correct and costs speed alone;
        // it joins the no-ops above and is counted so the census still shows the
        // traffic. The other two change what a later `executeCommandsInBuffer:`
        // will run: a reset the device drops leaves commands live that the guest
        // retired, and a copy it drops leaves the destination holding whatever it
        // held before. Both are stale commands executing, which is worse than a
        // dropped one, so they stay fail-visible as well as counted.
        //
        // Counted rather than executed on purpose. `runtime::icb` materializes
        // host ICBs on the Metal arm only, and it reads 0.00% on a driven x86
        // boot — so the count is what says whether an executor is worth building,
        // and for which of the two.
        BlitKind::IcbRange if cmd.opcode == wire_blit::OPCODE_OPTIMIZE_ICB => {
            crate::runtime::drain::note_store_route("blit_noop_icb_optimize");
        }
        BlitKind::IcbRange | BlitKind::IcbCopy => {
            use crate::observe::Decline as _;
            let decline = IcbRecordDropped(cmd.opcode);
            crate::runtime::drain::note_store_route(decline.slug());
            crate::observe::Emit::decline("blit_icb", &decline)
                .field("task", task_id)
                .field("range_loc", cmd.range_location)
                .field("range_len", cmd.range_length)
                .fail_once(cmd.opcode as u64);
        }
        BlitKind::Fence => {
            // Log from the *blit* status, before the remap. The remap folds two
            // meanings into `FenceStatus::Missing` — an absent object and a zero
            // fence ref — and only the blit rail's own reason can tell them
            // apart; `Refusal for BlitStatus` reproduces this site's previous
            // log condition exactly.
            let blit_st = blit_exec::execute_blit_fence(state, task_id, &cmd);
            if let Some(e) = crate::observe::Emit::refusal("blit_fence_fail", &blit_st) {
                e.field("opcode", format!("{:#x}", cmd.opcode)).fail();
            }
        }
        // `invalidateCompressedTexture:` and its `slice:level:` form. Apple's
        // lossless-compression metadata is a property of a *host* texture's
        // backing, and this device writes the guest's pages directly — there is
        // no compressed representation here to mark stale, so skipping it is
        // semantically correct and it joins the `optimize*`/`synchronize*`
        // no-ops. It is counted rather than folded into them because the two
        // records share the `Ref` and `RefSliceLevel` wire shapes with those
        // selectors: without its own route a compressed-texture invalidate
        // would be indistinguishable from a synchronize that genuinely needed
        // nothing done, and "this workload issues none" would be unprovable.
        BlitKind::InvalidateCompressedTexture => {
            crate::runtime::drain::note_store_route("blit_noop_invalidate_compressed");
        }
        // `fillTexture:…:color:` and `fillTexture:…:bytes:length:`. Unlike the
        // invalidate above these are writes the guest expects to land, so a
        // dropped one leaves the region holding what it held before and the
        // guest reads back content it believes it just wrote. Counted and
        // fail-visible, with the extent named, because the extent is what
        // decides whether an executor is worth building.
        //
        // Not executed here on purpose. A texture fill needs the destination
        // resolved through the type-4/5/11 rails, the region walked per row,
        // and — for the colour form — the clear colour converted into the
        // texture's pixel format, which is a converter this device does not
        // have. The count is what says whether to build one, and for which of
        // the two sources.
        BlitKind::FillTexture => {
            use crate::observe::Decline as _;
            let decline = TextureFillDropped(cmd.fill_source);
            crate::runtime::drain::note_store_route(decline.slug());
            crate::observe::Emit::decline("blit_fill_texture", &decline)
                .field("task", task_id)
                .field("texture", cmd.texture)
                .field("level", cmd.level)
                .field("slice", cmd.slice)
                .field(
                    "extent",
                    format!(
                        "{}x{}x{}",
                        cmd.fill_size.width, cmd.fill_size.height, cmd.fill_size.depth
                    ),
                )
                .fail_once(cmd.opcode as u64);
        }
        BlitKind::FillBuffer | BlitKind::FillBufferPattern4 | BlitKind::Copy => {
            match blit_exec::execute_blit(state, host, task_id, &cmd) {
                BlitStatus::Ok | BlitStatus::ZeroExtent => {}
                st => {
                    // Icon/upload path often uses blit copies; fail-visible for RE.
                    // The reason names the specific failing site inside blit_exec
                    // that produced the coarse `st` — 177 checks collapse into
                    // eight statuses, so the status alone says almost nothing.
                    // `Refusal` supplies it, and an uninstrumented site now reads
                    // `blit_unattributed` rather than rendering a bare `reason=`.
                    let src_ty = objects::lookup_list_entry(state, host, task_id, cmd.source)
                        .map(|e| e.object_type)
                        .unwrap_or(0);
                    let dst_ty = objects::lookup_list_entry(state, host, task_id, cmd.destination)
                        .map(|e| e.object_type)
                        .unwrap_or(0);
                    if let Some(e) = crate::observe::Emit::refusal("blit_fail", &st) {
                        e.field("st", format!("{st:?}"))
                            .field("kind", format!("{:?}", cmd.kind))
                            .field("opcode", format!("{:#x}", cmd.opcode))
                            .field("src", cmd.source)
                            .field("src_ty", src_ty)
                            .field("dst", cmd.destination)
                            .field("dst_ty", dst_ty)
                            .field("off", cmd.source_offset)
                            .field(
                                "lvl",
                                format!("{}/{}", cmd.destination_level, cmd.source_level),
                            )
                            .field(
                                "size",
                                format!(
                                    "{}x{}x{}",
                                    cmd.source_size.width,
                                    cmd.source_size.height,
                                    cmd.source_size.depth
                                ),
                            )
                            .fail();
                    }
                }
            }
        }
        BlitKind::Unknown => {
            crate::observe::fail(format!(
                "blit unknown opcode={:#x} len={}",
                cmd.opcode,
                cmd_bytes.len()
            ));
        }
    }
    crate::runtime::drain::note_store_route_us(
        match cmd.kind {
            BlitKind::Fence => "blitrec_fence_us",
            BlitKind::Copy => "blitrec_copy_us",
            BlitKind::FillBuffer | BlitKind::FillBufferPattern4 => "blitrec_fill_us",
            BlitKind::Resource | BlitKind::Image => "blitrec_noop_us",
            _ => "blitrec_other_us",
        },
        record_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route(match cmd.kind {
        BlitKind::Fence => "blitrec_fence_n",
        BlitKind::Copy => "blitrec_copy_n",
        BlitKind::FillBuffer | BlitKind::FillBufferPattern4 => "blitrec_fill_n",
        BlitKind::Resource | BlitKind::Image => "blitrec_noop_n",
        _ => "blitrec_other_n",
    });
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
    let cmd = match render::decode(cmd_bytes) {
        Ok(c) => c,
        // Was `Err(_) => return`: a malformed render command dropped with no
        // line, on the hottest path in the crate. Indistinguishable from a
        // segment that simply carried no render work.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("render_decode", &status) {
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
        RenderKind::SetPipeline => {
            // Apply what the record decoded, ref 0 included. This used to be
            // guarded `if cmd.pipeline_ref != 0`, and the match's last arm is a
            // bare `_ => {}`, so a zero ref left the *previous* pipeline latched
            // and the next draw encoded against it — a wrong frame with nothing
            // on any channel. Dropping the record is not the neutral choice it
            // looks like: `acc.pipeline_ref == 0` is already a state the draw arm
            // knows, where it declines as `dropped_no_pipeline` and says so. Letting
            // the zero through routes this into that named decline instead of
            // into a stale bind.
            //
            // A healthy zero: `setRenderPipelineState:` takes a non-null
            // pipeline, so Apple's serializer has no reason to emit this record
            // with ref 0. That is what makes applying the decoded value safe — on
            // a stream that never sends it, the two behaviors are identical — and
            // it is why a firing is worth a line rather than a silent drop.
            //
            // Measured zero on a driven x86/PCI boot (Ventura guest, 25 s Safari
            // window drag, ~500 host-window draws), which is the reading that
            // makes this arm's removal of the old `if cmd.pipeline_ref != 0`
            // guard inert on that workload rather than merely argued. One boot on
            // one pathway: it does not prove the arm never fires, it says the
            // desktop compositor does not take it.
            if cmd.pipeline_ref == 0 && crate::observe::first_sight("render_set_pipeline_zero", 0) {
                crate::observe::fail(
                    "stream_set_pipeline reason=render_set_pipeline_zero_ref \
                     (a render pipeline was set to ref 0; the pass is now unbound \
                     and its draws decline as dropped_no_pipeline)",
                );
            }
            acc.pipeline_ref = cmd.pipeline_ref;
        }
        RenderKind::SetBuffer => {
            // Slots first..first+n from the archive layout's entry array.
            // `render::decode` refuses `count == 0` with `ErrBadLength`, and
            // sets `cmd.buffer_ref` from `buffer_binds.first()`, so a decoded
            // SetBuffer always carries at least one entry and there is no
            // single-entry wire form to fall back to.
            let cleared = apply_binds(
                &cmd.buffer_binds,
                cmd.first,
                BindTarget {
                    stage: cmd.stage,
                    class: BindClass::Buffer,
                },
                BindTables {
                    vertex: &mut acc.vertex_buffers,
                    fragment: &mut acc.fragment_buffers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, b| {
                    (b.buffer_ref != 0).then_some(BufferBind {
                        index,
                        buffer_ref: b.buffer_ref,
                        resource: objects::resolve_resource(state, host, task_id, b.buffer_ref)
                            .ok(),
                        offset: b.offset,
                        attribute_stride: b.attribute_stride,
                    })
                },
            );
            out.buffer_unbinds = out.buffer_unbinds.saturating_add(cleared);
        }
        RenderKind::SetBufferOffset => {
            // Archive apply_buffer_offset: update offset on an already-bound slot.
            if cmd.first >= BindClass::Buffer.table() {
                // The slot is outside the table, so the bind that would have
                // occupied it was already dropped by `apply_binds` and counted
                // under `render_buffer_bind_slot_past_table`. This is the
                // *second* record the guest spends on that slot. Counted
                // separately rather than folded in, because these are different
                // records.
                //
                // In a conforming stream the bind came first and already refused
                // the draws — Metal requires a buffer bound at the index before
                // `setVertexBufferOffset:atIndex:`. This does not rely on that:
                // an offset record naming a slot this device has no table entry
                // for is on its own a record it cannot carry, and a stream where
                // the bind did *not* come first is exactly the one where relying
                // on it would be wrong.
                crate::runtime::drain::note_store_route("render_buffer_offset_slot_past_table");
                let over = BufferOffsetSlotPastTable {
                    stage: cmd.stage,
                    index: cmd.first,
                };
                crate::observe::Emit::decline("render_buffer_offset", &over)
                    .fail_once((u64::from(cmd.stage as u32) << 32) | u64::from(cmd.first));
                acc.unrepresentable
                    .get_or_insert(StreamRefusal::BufferOffset(over));
                return;
            }
            let list = match cmd.stage {
                Stage::Vertex => Arc::make_mut(&mut acc.vertex_buffers),
                Stage::Fragment => Arc::make_mut(&mut acc.fragment_buffers),
                // An offset update names a slot in a table; with no stage
                // there is no table, and inventing one would move somebody
                // else's binding.
                Stage::Unknown => return,
            };
            match list.iter_mut().find(|b| b.index == cmd.first) {
                Some(b) => {
                    b.offset = cmd.buffer_offset;
                    // Only when this record carried one.
                    // `setVertexBufferOffset:atIndex:` and its strided sibling
                    // are different opcodes, and the plain one must not clear a
                    // stride an earlier bind established.
                    if let Some(stride) = cmd.attribute_stride {
                        b.attribute_stride = Some(stride);
                    }
                }
                // A healthy zero, and a sharp one. Metal requires a buffer
                // already bound at the index before
                // `setVertexBufferOffset:atIndex:`, and a render encoder's bind
                // state does not outlive the encoder, so the guest and this
                // table should agree on which slots are live. A firing means
                // they do not — a bind this device dropped, refused or never
                // decoded — and the offset lands on nothing.
                None => {
                    crate::runtime::drain::note_store_route("render_buffer_offset_slot_unbound")
                }
            }
        }
        RenderKind::SetTexture => {
            // As for SetBuffer: `ref_binds` is never empty on a decoded record,
            // and the clone the removed fallback needed went with it.
            let cleared = apply_binds(
                &cmd.ref_binds,
                cmd.first,
                BindTarget {
                    stage: cmd.stage,
                    class: BindClass::Texture,
                },
                BindTables {
                    vertex: &mut acc.vertex_textures,
                    fragment: &mut acc.fragment_textures,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, texture_ref| {
                    if texture_ref == 0 {
                        return None;
                    }
                    if !out.texture_refs.contains(&texture_ref) {
                        out.texture_refs.push(texture_ref);
                    }
                    if let Some(m) = objects::resolve_type11_ref(state, host, task_id, texture_ref)
                    {
                        if !out.type11_mappings.contains(&m) {
                            out.type11_mappings.push(m);
                        }
                    } else if objects::resolve_type4_surface(state, host, texture_ref) {
                        // x86 type-4: object ref is surface_id / mapping_id.
                        if !out.type11_mappings.contains(&texture_ref) {
                            out.type11_mappings.push(texture_ref);
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
        RenderKind::SetSampler => {
            // The two LOD forms carry a clamp pair per entry; the plain forms
            // carry none, and `sampler_lod_binds` is empty for them. Zipped
            // rather than indexed into inside the builder, so a record whose
            // two lists ever disagreed in length binds the slots it has refs
            // for and clamps only those it has clamps for, instead of
            // panicking or pairing a slot with another slot's clamp.
            let entries: Vec<(u32, Option<(u32, u32)>)> = cmd
                .ref_binds
                .iter()
                .enumerate()
                .map(|(i, &r)| (r, cmd.sampler_lod_binds.get(i).copied()))
                .collect();
            let cleared = apply_binds(
                &entries,
                cmd.first,
                BindTarget {
                    stage: cmd.stage,
                    class: BindClass::Sampler,
                },
                BindTables {
                    vertex: &mut acc.vertex_samplers,
                    fragment: &mut acc.fragment_samplers,
                    refused: &mut acc.unrepresentable,
                },
                |b| b.index,
                |index, (sampler_ref, lod_clamp)| {
                    (sampler_ref != 0).then_some(SamplerBind {
                        index,
                        sampler_ref,
                        lod_clamp,
                    })
                },
            );
            out.sampler_unbinds = out.sampler_unbinds.saturating_add(cleared);
        }
        RenderKind::SetViewport => {
            // The whole array, in the guest's order. `setViewports:count:`
            // replaces the viewport state rather than adding to it, so this
            // assigns rather than extends — a record of two after a record of
            // five leaves two, which is what Metal does.
            acc.viewports.clone_from(&cmd.viewports);
        }
        RenderKind::SetScissor => {
            // All-or-nothing on an empty rect, which is the singular arm's rule
            // read at array width. `setScissorRects:count:` replaces the state
            // atomically and slot order is meaningful — it is what a shader's
            // `[[viewport_array_index]]` selects — so an array cannot be adopted
            // with the empty slots left out, and adopting them as written would
            // make exactly those slots clip however the backend reads a zero
            // rect. Neither is expressible here, so the record is refused whole
            // and the previous state stands, as one empty rect always has.
            if let Some(empty) = cmd.scissors.iter().find(|r| r.is_empty()) {
                note_empty_scissor(task_id, *empty);
            } else {
                acc.scissors.clone_from(&cmd.scissors);
            }
        }
        // No `if cmd.has_blend_color` on these five. Each of the five kinds has
        // exactly one producer in `decode::render`, which sets the kind and the
        // flag in the same block, so the flag was true whenever the arm was
        // reached and the guard could not fail. It was not free: the match's last
        // arm is a bare `_ => {}`, so the shape said a guest could set a cull mode
        // this device then discarded, when no such loss was possible. A record
        // too short to hold
        // the field never gets here at all — the wire view refuses it and
        // `decode` returns `ErrShort` before a kind is assigned.
        RenderKind::SetBlendColor => {
            acc.blend_color = Some(cmd.blend_color);
        }
        RenderKind::SetCullMode => {
            acc.cull_mode = Some(cmd.cull_mode);
        }
        RenderKind::SetFrontFacing => {
            acc.front_facing = Some(cmd.front_facing);
        }
        RenderKind::SetDepthBias => {
            acc.depth_bias = Some(cmd.depth_bias);
        }
        RenderKind::SetDepthStencil => {
            acc.depth_stencil_ref = cmd.depth_stencil_ref;
        }
        RenderKind::SetStencilReference => {
            acc.stencil_ref = Some((cmd.stencil_ref_front, cmd.stencil_ref_back));
        }
        RenderKind::RenderPass => {
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
            acc.visibility_buffer_ref = cmd.pass_visibility_result_buffer_ref;
            // Refused rather than drawn into layer 0, the decision the colour
            // subresource arm below already made for the same shape of loss:
            // the layer a draw selects is a coordinate the pass did not name,
            // so rendering anyway lands geometry meant for one layer on top of
            // another's correct content.
            if cmd.pass_render_target_array_length > 1 {
                let drop = note_pass_array_length_unsupported(
                    task_id,
                    cmd.pass_render_target_array_length,
                );
                acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
            }
            if cmd.pass_render_target_width != 0 || cmd.pass_render_target_height != 0 {
                note_pass_target_extent();
            }
            // Full multi-attachment: re-decode all color slots from payload.
            if cmd_bytes.len() >= 8 {
                let payload = &cmd_bytes[8..];
                // A depth or stencil attachment this device cannot bind used to
                // be left out and the pass run without it, which turns depth
                // testing off for every draw in it: the near geometry stops
                // occluding the far, and the colour target — which was correct
                // before the pass — is overwritten with a picture assembled in
                // the wrong order. That is not a degraded frame, it is wrong
                // content written over right content, and nothing downstream can
                // tell because a pass with no depth attachment is exactly what a
                // guest that wanted none also produces.
                let depth = decode_depth_attachment(payload);
                if depth.texture_ref != 0 {
                    if attachment_subresource_is_bindable(depth.into(), LevelSupport::LevelZeroOnly)
                    {
                        acc.depth_attach = Some(depth);
                    } else {
                        let drop = note_depth_stencil_unsupported(task_id, "depth", &depth.into());
                        acc.unrepresentable.get_or_insert(StreamRefusal::Pass(drop));
                    }
                }
                let stencil = decode_stencil_attachment(payload);
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
                for i in 0..PASS_MAX_COLOR_ATTACHMENTS {
                    let att = decode_color_attachment(payload, i);
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
                        objects::resolve_type11_ref(state, host, task_id, published_ref)
                    {
                        note_pass_extent_for_slot(state, task_id, slot, m, &cmd);
                        if !out.type11_mappings.contains(&m) {
                            out.type11_mappings.push(m);
                        }
                    } else if objects::resolve_type4_surface(state, host, published_ref) {
                        // A type-4 attachment is its own mapping id — the arm
                        // below pushes `att.texture_ref` where the type-11 arm
                        // pushes the id it resolved to.
                        note_pass_extent_for_slot(state, task_id, slot, published_ref, &cmd);
                        if !out.type11_mappings.contains(&published_ref) {
                            out.type11_mappings.push(published_ref);
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
            // Also keep color0 from command for convenience.
            if cmd.color0.texture_ref != 0
                && cmd.color0.load_action == MTL_LOAD_ACTION_CLEAR
                && store_action_publishes_single_sample(cmd.color0.store_action)
                && !acc
                    .clears
                    .iter()
                    .any(|a| a.texture_ref == cmd.color0.texture_ref)
            {
                acc.clears.push(cmd.color0);
            }
        }
        RenderKind::Draw => {
            if cmd.opcode == wire_render::OPCODE_DRAW_INDEXED_WIDE {
                crate::observe::line(format!(
                    "render_wide_indexed task={task_id} target_refs={:?} pipeline={} prim={} index_type={} index_ref={} count={} offset={:#x}",
                    acc.color_targets,
                    acc.pipeline_ref,
                    cmd.primitive_type,
                    cmd.index_type,
                    cmd.index_buffer_ref,
                    cmd.index_count,
                    cmd.index_buffer_offset
                ));
            }
            acc.saw_draw = true;
            out.saw_draw = true;
            let count = if cmd.index_count != 0 {
                cmd.index_count
            } else {
                cmd.vertex_count
            };
            if cmd.index_count != 0 && cmd.index_buffer_ref != 0 {
                acc.indexed = Some(IndexedDrawInfo {
                    index_type: cmd.index_type,
                    index_count: cmd.index_count,
                    index_buffer_ref: cmd.index_buffer_ref,
                    index_buffer_offset: cmd.index_buffer_offset,
                    base_vertex: cmd.base_vertex,
                });
            } else {
                // An indexed opcode whose record named no index buffer falls
                // through to a *non-indexed* draw of `index_count` vertices,
                // because `count` above took `index_count` and `indexed` is
                // None. That is not a form Metal has:
                // `drawIndexedPrimitives` takes its index buffer as an
                // argument and there is no bound-index-buffer state for a zero
                // ref to mean, so the record is malformed and this is the
                // device inventing a different draw call from it.
                //
                // Named rather than declined, deliberately. Declining is the
                // contract-faithful answer, but `index_buffer_ref` is read at a
                // payload offset that differs per draw form, so if any of those
                // offsets is wrong the ref reads 0 and declining would turn a
                // decode fault into a blank frame. This counter says first
                // whether the cell is reached at all.
                if cmd.index_count != 0 && is_indexed_draw_opcode(opcode) {
                    note_indexed_draw_without_buffer(task_id, opcode, cmd.index_count);
                }
                acc.indexed = None;
            }
            // Snapshot bind state for this draw (archive multi-draw job).
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
                            instance_count: cmd.instance_count,
                            primitive_type: cmd.primitive_type,
                            first_vertex: cmd.vertex_start,
                            base_instance: cmd.base_instance,
                        },
                        ..snapshot
                    }),
                    Err(over) => note_draw_refused(over, acc.pipeline_ref, "draw"),
                }
            }
        }
        RenderKind::ExecuteCommands => {
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
        RenderKind::Fence => {
            // The render encoder's own fence opcodes, not the blit encoder's.
            // Each encoder numbers its selectors in its own space and the two
            // fence pairs are nowhere near each other, so matching a render
            // opcode against `wire_blit`'s constants never succeeded and sent
            // every render fence to the arm below. `updateFence:afterStages:`
            // is what a guest uses to order work inside one render encoder
            // against a later one, so what it dropped was encoder
            // synchronisation on every pass that asked for it.
            let action = match cmd.opcode {
                wire_render::OPCODE_UPDATE_FENCE => FenceAction::Update,
                wire_render::OPCODE_WAIT_FOR_FENCE => FenceAction::Wait,
                opcode => {
                    // A render fence record whose opcode is neither update nor
                    // wait drops the guest's encoder synchronisation. The
                    // counter that stood here had no reader, so this was silent.
                    crate::observe::fail(format!(
                        "render_fence_opcode reason=render_fence_opcode_unknown \
                         task={task_id} opcode={opcode:#x} fence={}",
                        cmd.fence_ref
                    ));
                    return;
                }
            };
            fence_exec::execute_fence(
                state,
                task_id,
                FenceDomain::RenderFence,
                cmd.fence_ref,
                action,
            );
        }
        RenderKind::OtherAccepted => {
            // An undecoded render opcode: the decoder accepts it (catch-all)
            // but no executor exists, so the guest command is effectively
            // dropped. That MUST stay fail-visible — but a per-draw op such as
            // 0x7c fires thousands of times per app render, so emitting per
            // record floods /tmp/reims-vgpu-fail.log (measured ~2620 lines from six
            // app launches). Dedup to ONE line per distinct opcode (the set is
            // tiny and boot-stable) and capture the raw wire on first sighting
            // so the layout can be decoded offline. Unknown wire stays unknown;
            // we never invent semantics for it.
            note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
        }
        // Two kinds the product answers by doing nothing, counted separately.
        // They used to fall into the catch-all below, which made them
        // indistinguishable from a record that was handled — and unlike a
        // `SetBuffer` these carry ordering and lifetime the guest expects us to
        // honour, so silence was the wrong answer even though doing nothing is
        // the right one.
        //
        // The arguments are the compute rail's, which reached the same two
        // conclusions first (`compute_noop_residency_hint`,
        // `compute_noop_barrier`). Residency: `useResource:`/`useHeap:` are
        // hints for a driver that pages resources, and this product resolves
        // every binding per draw, so there is nothing for them to keep
        // resident. Barriers: the render rail submits and waits at pass
        // granularity, so a barrier inside the pass is implied by the boundary.
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
        // it raised are what argued for plumbing it, and both halves now reach
        // a backend.
        RenderKind::SetRasterState => {
            // Two selectors share the one-`NSUInteger` record; the opcode says
            // which. Both are latched whatever the value, including the Metal
            // default — a stream that sets Lines and then sets Fill again is
            // asking for Fill, and dropping the second record would leave the
            // rest of the pass wireframed.
            //
            // The ordinal is carried raw and translated per backend, the way
            // `cull_mode` and `front_facing` beside it are: only the backend
            // knows whether the host can spell the answer, so only the backend
            // can refuse by name.
            let slot = match cmd.opcode {
                wire_render::OPCODE_SET_TRIANGLE_FILL_MODE => &mut acc.fill_mode,
                _ => &mut acc.depth_clip_mode,
            };
            // The record's field is 64-bit and the ordinals are small, but a
            // guest writes what it likes. `u32::MAX` is not a value of either
            // Metal enum, so a wide word reaches the backend as an
            // out-of-contract value that says its own name, rather than as its
            // own low half — which for a multiple of 2^32 would be the
            // *default*, the one answer that renders with nothing in the log.
            *slot = Some(u32::try_from(cmd.mode).unwrap_or(u32::MAX));
        }
        RenderKind::SetFloatState => {
            // Both default to 1.0. Compared exactly rather than with a
            // tolerance: the guest wrote a literal and the question is whether
            // it wrote *the* literal, not whether it is close to it.
            if cmd.float_value != 1.0 {
                crate::runtime::drain::note_store_route(match cmd.opcode {
                    wire_render::OPCODE_SET_LINE_WIDTH => "render_line_width_dropped",
                    _ => "render_tessellation_scale_dropped",
                });
            }
        }
        RenderKind::SetStoreAction => {
            // `setColorStoreAction:atIndex:` and its depth and stencil siblings
            // replace what the render-pass descriptor declared for one
            // attachment. All three of those declared actions are honoured —
            // colour in `encode_draw_chain`'s writeback loop, depth and stencil
            // through `draw::depth_stencil` — so dropping the override was a
            // real loss in both directions, and the expensive one is a pass
            // declared `DontCare` and overridden to `Store`: content the guest
            // asked to keep and never got back.
            //
            // The store action is a `u16` in every attachment struct, so a mode
            // that does not fit is not narrowed into a different action; it is
            // left alone and named, the same reading `SetIntState` takes of its
            // own low half.
            let Ok(action) = u16::try_from(cmd.mode) else {
                crate::runtime::drain::note_store_route("render_store_action_out_of_range");
                crate::observe::fail(format!(
                    "render_store_action fail reason=render_store_action_out_of_range \
                     op={:#x} mode={} index={}",
                    cmd.opcode, cmd.mode, cmd.first
                ));
                return;
            };
            match cmd.opcode {
                wire_render::OPCODE_SET_COLOR_STORE_ACTION => {
                    // By pass slot, which is what the record's index names and
                    // what `color_slots` is keyed by — not by position, since a
                    // pass declaring slots 0 and 3 has two entries.
                    match acc
                        .color_slots
                        .iter_mut()
                        .find(|(slot, _)| *slot == cmd.first)
                    {
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
                                 index={} declared={}",
                                cmd.first,
                                acc.color_slots.len()
                            ));
                        }
                    }
                }
                // Neither of these carries an index: there is one depth and one
                // stencil attachment, so the record names only the action.
                wire_render::OPCODE_SET_DEPTH_STORE_ACTION => match acc.depth_attach.as_mut() {
                    Some(d) => d.store_action = action,
                    None => note_store_action_no_attachment("depth", action),
                },
                wire_render::OPCODE_SET_STENCIL_STORE_ACTION => match acc.stencil_attach.as_mut() {
                    Some(s) => s.store_action = action,
                    None => note_store_action_no_attachment("stencil", action),
                },
                // Not a catch-all standing in for the stencil arm: the decoder
                // maps exactly three opcodes to this kind, so a fourth reaching
                // here means the decoder grew an arm this one did not, and the
                // guest's store action lands on nothing.
                op => {
                    crate::runtime::drain::note_store_route("render_store_action_opcode_unknown");
                    crate::observe::fail(format!(
                        "render_store_action fail reason=render_store_action_opcode_unknown \
                         op={op:#x} action={action}"
                    ));
                }
            }
        }
        RenderKind::SetVertexAmplification => {
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
        }
        RenderKind::SetVisibilityResultMode => {
            // `MTLVisibilityResultModeDisabled` is 0, and it is the guest
            // disarming the query rather than an unknown value: subsequent draws
            // simply carry none. The record's field is 64-bit and the ordinals
            // are small, but a guest writes what it likes — a wide word reaches
            // the backend as an out-of-contract value that says its own name,
            // the same treatment `fill_mode` gives its ordinal, rather than as
            // its own low half.
            acc.visibility = (cmd.mode != 0).then(|| draw::VisibilityArming {
                mode: u32::try_from(cmd.mode).unwrap_or(u32::MAX),
                offset: cmd.visibility_result_offset,
            });
        }
        RenderKind::DrawIndirect => {
            execute_indirect_draw(state, host, task_id, &cmd, acc);
        }
        RenderKind::UseResource | RenderKind::UseHeap => {
            crate::runtime::drain::note_store_route("render_noop_residency_hint");
        }
        RenderKind::Barrier => {
            crate::runtime::drain::note_store_route("render_noop_barrier");
        }
        // The tile-shader family. Nine opcodes that used to reach
        // `OtherAccepted` together, split here into the three different things
        // they actually are — because "a tile record arrived" is not a
        // measurement anyone can act on.
        RenderKind::TileBind => {
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
        RenderKind::TileDispatch => {
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
        RenderKind::SetStoreActionOptions => {
            // The options sibling of the store action beside it, which *is*
            // applied now — this is the half that is not. `MTLStoreActionOptions`
            // carries `CustomSamplePositions`, asking that a multisample resolve
            // use the pass's programmable sample positions, and this device
            // neither sets those (`render_pass_sample_positions_dropped`) nor
            // renders at more than one sample per pixel, where the option means
            // nothing. Applying it here would be recording a number no resolve
            // reads.
            //
            // No default to compare against — `MTLStoreActionOptionNone` is 0,
            // but a guest that writes 0 is still overriding whatever the pass
            // descriptor said.
            crate::runtime::drain::note_store_route("render_store_action_options_dropped");
        }
        RenderKind::DrawPatches => {
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
        RenderKind::SetTessellationFactorBuffer => {
            // The state half of a tessellated draw. Unapplied like the draws
            // themselves, so this should track `render_draw_patches_dropped`;
            // the two being far apart would mean one of the two arms is wrong
            // rather than that the guest is doing something unusual.
            crate::runtime::drain::note_store_route("render_tessellation_factor_buffer_dropped");
        }
        RenderKind::RenderPassProperty => {
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
        RenderKind::TileDimensionsQuery => {
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
        _ => {}
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
    stage: Stage,
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
            (
                "stage",
                match self.stage {
                    Stage::Vertex => "vertex",
                    Stage::Fragment => "fragment",
                    Stage::Unknown => "unknown",
                }
                .to_string(),
            ),
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
    stage: Stage,
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
    stage: Stage,
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
            (
                "stage",
                match self.stage {
                    Stage::Vertex => "vertex",
                    Stage::Fragment => "fragment",
                    Stage::Unknown => "unknown",
                }
                .to_string(),
            ),
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
/// texture arm's type-11 mapping list — with the caller. The clear count comes
/// back as a return value rather than through an `&mut` counter so `make` can
/// hold the rest of `ExecResult`.
fn apply_binds<T: Copy, B: Clone>(
    entries: &[T],
    first: u32,
    target: BindTarget,
    tables: BindTables<'_, B>,
    slot: impl Fn(&B) -> u32,
    mut make: impl FnMut(u32, T) -> Option<B>,
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
    for (i, entry) in entries.iter().copied().enumerate() {
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
        let list = match stage {
            Stage::Vertex => Arc::make_mut(vertex),
            Stage::Fragment => Arc::make_mut(fragment),
            // No table to bind into, but a slot the guest cleared is still
            // cleared: the count is what the record said, not what we modelled.
            Stage::Unknown => {
                cleared = cleared.saturating_add(bind.is_none() as u32);
                continue;
            }
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
                match draw::encode_icb_execute_and_writeback(
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
                // A resident type-11 target carries attachment contents between
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
                // on a type-11 target let a CLEAR re-run before each record,
                // wiping the full composite drawn by record 1 (live poison=1:
                // mid peak 10.9M native → 2.5M after later records).
                if di > 0 {
                    for c in &mut req.colors {
                        c.load_action = MTL_LOAD_ACTION_LOAD;
                    }
                    // Chain from the engine resident when available; otherwise
                    // seed from the prior encode output (archive "thread each
                    // record's output as next initial content"). MoltenVK's
                    // portability path returns CPU pixels for type-11 mappings,
                    // so `unified` does not imply that a resident exists.
                    // Moved, not cloned (multi-MiB).
                    match multi_draw_chain_source(resident_chain, chain_rgba.is_some()) {
                        MultiDrawChainSource::Resident => {
                            req.chain_from_resident = true;
                        }
                        MultiDrawChainSource::Cpu => {
                            if let Some(c0) = req.colors.first_mut() {
                                c0.target_seed_rgba = chain_rgba.take();
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
                let encode =
                    draw::encode_draw_chain(state, host, &mut req, do_writeback, force_full_store);
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
            target_seed_rgba: None,
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
fn execute_indirect_draw<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    cmd: &render::Command,
    acc: &mut StreamAccum,
) {
    use crate::contract::draw::indirect;

    let indexed_form = cmd.opcode == wire_render::OPCODE_DRAW_INDEXED_INDIRECT;
    let block_len = if indexed_form {
        indirect::INDEXED_LEN
    } else {
        indirect::UNINDEXED_LEN
    };
    let block = match crate::runtime::compute_exec::read_buffer_window(
        state,
        host,
        task_id,
        cmd.indirect_buffer_ref,
        cmd.indirect_buffer_offset,
        block_len,
    ) {
        Ok(block) => block,
        Err(status) => {
            // The buffer is the whole draw here — there is no fallback count in
            // the record to fall back to — so a read that fails is a refused
            // draw, and `read_buffer_window`'s status already names which rung
            // of the resolve refused. Latched per buffer ref because a guest
            // re-issues the same indirect draw every frame.
            note_indirect_draw_refused(task_id, cmd, status);
            return;
        }
    };

    let (args, index_start, base_vertex) = if indexed_form {
        match indirect::indexed(&block, cmd.primitive_type) {
            Some(v) => (v.0, v.1, v.2),
            None => return,
        }
    } else {
        match indirect::unindexed(&block, cmd.primitive_type) {
            Some(args) => (args, 0, 0),
            None => return,
        }
    };

    acc.saw_draw = true;
    if indexed_form {
        // `indexStart` counts indices, not bytes. The loader is given a byte
        // offset, so it is scaled here by the width the record's own
        // `index_type` declares — the same two widths `translate::raster::
        // index_type` accepts, and an unknown one is left to the loader's
        // typed refusal rather than being guessed at as 2.
        let stride = match cmd.index_type {
            1 => 4u64, // MTLIndexTypeUInt32
            _ => 2,    // MTLIndexTypeUInt16, and Metal's default
        };
        acc.indexed = Some(IndexedDrawInfo {
            index_type: cmd.index_type,
            index_count: args.vertex_count,
            index_buffer_ref: cmd.index_buffer_ref,
            index_buffer_offset: cmd
                .index_buffer_offset
                .saturating_add(u64::from(index_start).saturating_mul(stride)),
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
        if let Some(mid) = objects::resolve_type11_ref(state, host, task_id, tex_ref) {
            // The guest pages are the only copy of a type-11 surface, so there
            // is no mirror to drop — only bump gen for scanout skips.
            let _ = state.mark_mapping_written(mid);
        } else if objects::resolve_type4_surface(state, host, tex_ref) {
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
    // The one caller that has no identity to be handed. The chain broke, so no
    // span carries the key its last good record registered, and the abandoning
    // read has to name the resident from the state it can still see. That is a
    // second derivation and it is spelled out here rather than hidden inside
    // `read_resident_chain`, because every *other* caller has the draw's own
    // key and a shared re-derivation would silently give them this one's answer
    // — see `draw::M2vDrawSpan::ResidentSurfaceStore` for what that cost.
    #[cfg(feature = "backend-vulkan")]
    if end.resident && chain_rgba.is_none() {
        if let Some(identity) = draw::render_chain_identity(state, req) {
            *chain_rgba = draw::read_resident_chain(req, &identity);
        }
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = (req, end.resident);
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
    // Prefer full draw-path resolve (type-11 or type-2/3 GVA wallpaper targets).
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
    is_indexed_draw_opcode, note_clear_dropped, note_color_subresource_unsupported,
    note_compute_refusal, note_depth_stencil_unsupported, note_draw_encode_fail,
    note_empty_scissor, note_indexed_draw_without_buffer, note_indirect_draw_refused,
    note_pass_array_length_unsupported, note_pass_extent_for_slot,
    note_pass_raster_sample_count_unsupported, note_pass_target_extent,
    note_store_action_no_attachment, note_stream_draw_drops, note_unimplemented_render_opcode,
    note_unnamed_icb_execute,
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
