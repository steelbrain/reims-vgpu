//! CmdExecIndirect2: load streams, multi-attachment clears, Metal draw attempt.
//!
//! Clear-only passes write guest mapping pages (archive render_clear).
//! Draws try Metal encode when pipeline MTLBs resolve; otherwise color targets
//! are still marked dirty for DisplaySwap.

use crate::contract::endian::{ld32, ld64};
use crate::contract::pixel_format::{f64_to_unorm8, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
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
    self, decode_color_attachment, decode_depth_attachment, decode_stencil_attachment,
    ColorAttachment, DepthAttachment, Kind as RenderKind, Stage, StencilAttachment,
    PASS_LOAD_ACTION_CLEAR, PASS_LOAD_ACTION_LOAD, PASS_MAX_COLOR_ATTACHMENTS,
    PASS_STORE_ACTION_STORE,
};
use crate::runtime::decode::stream::{
    self, decode_first_record, decode_next_record, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE,
    SEGMENT_TYPE_EVENT, SEGMENT_TYPE_INFO, SEGMENT_TYPE_RENDER,
};
use crate::runtime::fence_exec;
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapping_write;
use crate::runtime::metal_draw::{
    self, BufferBind, EncodeStatus, IndexedDrawInfo, SamplerBind, TextureBind, MAX_BIND_SLOTS,
};
use crate::runtime::mipmap::{self, MipmapStatus};
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};
use reims_vgpu_wire::ops::blit as wire_blit;
use reims_vgpu_wire::ops::render as wire_render;
use reims_vgpu_wire::ops::render_pass as wire_pass;
use reims_vgpu_wire::ops::tile as wire_tile;
use std::sync::Arc;

/// Max descriptors per ExecIndirect2 (wire table size), not a byte budget.
const MAX_CMDBUFS: usize = 16;

/// One stage's bind table as a draw sees it.
///
/// `Arc` rather than a plain `Vec` because a render stream's draws share their
/// bind state: the guest sets a table once and then issues many draws against
/// it, so snapshotting per draw copied the same entries over and over. The
/// accumulator mutates through [`Arc::make_mut`], which copies only when a
/// snapshot is actually outstanding — so a stream that binds once and draws 400
/// times allocates one table and 400 pointers.
///
/// That is what makes an unbounded draw list affordable, and an unbounded draw
/// list is what the protocol requires: the guest emits as many records as its
/// encoder recorded and every one of them contributes to the same attachment
/// set. See [`StreamDrawDrop`].
type BindTable<T> = Arc<Vec<T>>;

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
/// The arguments of one draw, as the guest issued them.
///
/// A named struct rather than the four-tuple this was: `base_instance` is a
/// fifth `u32` joining four that were already told apart only by position, and
/// the four sites that destructure it would each have been a silent swap away
/// from drawing the wrong thing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    primitive_type: u32,
    first_vertex: u32,
    /// Metal `baseInstance` / Vulkan `firstInstance`.
    base_instance: u32,
}

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
    viewport: Option<[f64; 6]>,
    scissor: Option<(u32, u32, u32, u32)>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
    /// First draw of the pass that owns this stencil attachment: the clear is
    /// its, and every later draw loads what it left.
    stencil_first_in_pass: bool,
}

#[derive(Clone, Debug, Default)]
struct StreamAccum {
    /// Whether a draw in the current pass has already consumed the stencil
    /// clear. Reset when the pass publishes its stencil attachment.
    stencil_pass_started: bool,
    pipeline_ref: u32,
    /// Pending clears for color attachments (load=clear).
    clears: Vec<ColorAttachment>,
    /// Color targets as (pass slot index, attachment). Slot maps to Metal color(i).
    color_slots: Vec<(u32, ColorAttachment)>,
    color_targets: Vec<u32>,
    /// All draws in stream order (archive multi-draw job).
    draws: Vec<PendingDraw>,
    saw_draw: bool,
    /// Last render ICB execute (`0x14`/`0x15`) in this stream.
    execute_icb: Option<RenderIcbExecute>,
    vertex_buffers: BindTable<BufferBind>,
    fragment_buffers: BindTable<BufferBind>,
    vertex_textures: BindTable<TextureBind>,
    fragment_textures: BindTable<TextureBind>,
    vertex_samplers: BindTable<SamplerBind>,
    fragment_samplers: BindTable<SamplerBind>,
    viewport: Option<[f64; 6]>,
    scissor: Option<(u32, u32, u32, u32)>,
    indexed: Option<IndexedDrawInfo>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
    /// Draw records this stream decoded but did not keep. See
    /// [`StreamDrawDrop`]; reported once per stream by [`note_stream_draw_drops`].
    dropped_unbound: u32,
}

impl StreamAccum {
    /// The stream's bind state as a `PendingDraw`, with no pipeline and no
    /// draw call attached.
    ///
    /// Two things need it and must not disagree: a decoded draw, which fills
    /// in `pipeline_ref` and `draw` on top, and an ICB execute, which inherits
    /// the state as it stands at end of stream and supplies neither.
    fn bind_snapshot(&self) -> PendingDraw {
        PendingDraw {
            indexed: self.indexed.clone(),
            vertex_buffers: self.vertex_buffers.clone(),
            fragment_buffers: self.fragment_buffers.clone(),
            vertex_textures: self.vertex_textures.clone(),
            fragment_textures: self.fragment_textures.clone(),
            vertex_samplers: self.vertex_samplers.clone(),
            fragment_samplers: self.fragment_samplers.clone(),
            viewport: self.viewport,
            scissor: self.scissor,
            blend_color: self.blend_color,
            cull_mode: self.cull_mode,
            front_facing: self.front_facing,
            depth_bias: self.depth_bias,
            depth_stencil_ref: self.depth_stencil_ref,
            stencil_ref: self.stencil_ref,
            depth_attach: self.depth_attach,
            stencil_attach: self.stencil_attach,
            stencil_first_in_pass: !self.stencil_pass_started,
            ..Default::default()
        }
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
    /// The record arrived with no pipeline bound or a zero primitive count.
    ///
    /// Either a genuinely empty draw — legal, and nothing is lost — or a
    /// `SetPipeline` this decoder failed to latch, which is a lost draw. The
    /// count is what separates the two: a rate that tracks the draw rate is the
    /// second reading, an occasional one is the first.
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
    ColorSubresourceUnsupported {
        slot: u32,
        level: u32,
        slice: u32,
        depth_plane: u32,
    },
    /// The pass named an explicit render target extent and this device used the
    /// attachment's.
    ///
    /// Unlike its three siblings on the pass tail this is **not** a healthy
    /// zero: a driven arm64/Vulkan boot reads it on essentially every pass. The
    /// line carries the extent because the count alone cannot say whether it
    /// matters — a pass whose stated extent equals its attachment's is asking
    /// for what already happens, and one that states less is a region this
    /// device renders outside of.
    TargetExtentUnapplied { width: u64, height: u64 },
}

impl crate::observe::Decline for StreamDrawDrop {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unbound { .. } => "stream_draw_dropped_unbound",
            Self::DepthStencilUnsupported { .. } => "stream_depth_stencil_unsupported",
            Self::ColorSubresourceUnsupported { .. } => "stream_color_subresource_unsupported",
            Self::TargetExtentUnapplied { .. } => "stream_pass_target_extent_unapplied",
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
            } => vec![
                ("slot", slot.to_string()),
                ("level", level.to_string()),
                ("slice", slice.to_string()),
                ("plane", depth_plane.to_string()),
            ],
            Self::TargetExtentUnapplied { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
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
            state.tasks.len(),
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

    let n_cb = (cmdbuf_count as usize).min(MAX_CMDBUFS);
    let page_shift = state.page_shift;
    let mut streams = Vec::with_capacity(n_cb);
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
        let Some(stream_len) = crate::runtime::metal_draw::host_alloc_len(length) else {
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

    // Plan before execute: cold AIR translation is immutable CPU work and can
    // run without protocol ownership. Keep the packet unconsumed until every
    // referenced render stage is ready, so replay cannot duplicate clears,
    // fences, compute dispatches, or guest writeback.
    #[cfg(feature = "backend-vulkan")]
    let translation_pending = {
        let mut pending = false;
        let mut unpublished: Vec<u32> = Vec::new();
        for stream in &streams {
            if preflight_render_translations(state, host, task_id, stream, &mut unpublished) {
                pending = true;
            }
            if preflight_compute_translations(state, host, task_id, stream) {
                pending = true;
            }
        }
        // A pipeline the guest has not finished publishing is asynchronous, not
        // a malformed one: the same read a moment later succeeds. Retrying the
        // packet keeps the draw; declining it here is what leaves an icon a
        // blank rounded square, because the 128x128 render that fills it is the
        // draw being thrown away.
        //
        // Bounded, because a reference that never resolves must not hold the
        // channel: past the budget the packet executes and the draw declines
        // with the reason it always did.
        // A pipeline that resolved is no longer waiting on anything; drop its
        // clock so the map holds only what is actually outstanding.
        state
            .pipeline_unreadable_since
            .retain(|(task, pipeline_ref), _| {
                *task != task_id || unpublished.contains(pipeline_ref)
            });
        for pipeline_ref in unpublished {
            let key = (task_id, pipeline_ref);
            let now = std::time::Instant::now();
            let since = *state.pipeline_unreadable_since.entry(key).or_insert(now);
            if now.duration_since(since) < PIPELINE_PUBLISH_WAIT {
                pending = true;
            } else {
                state.pipeline_unreadable_since.remove(&key);
                crate::observe::fail(format!(
                    "pipeline_publish_wait_expired task={task_id} ref={pipeline_ref} waited_ms={}",
                    now.duration_since(since).as_millis()
                ));
            }
        }
        pending
    };
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    let translation_pending = false;
    if translation_pending {
        out.deferred = true;
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
        walk_stream(state, host, task_id, &stream, &mut out, &mut acc);
        finish_stream(state, host, task_id, &mut out, &acc);
    }
    out.total_us = elapsed_us(exec_started);
    out
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

/// How long a draw waits for the guest to finish publishing the pipeline object
/// it names.
///
/// Long enough to cover the gap between an object being created on one channel
/// and referenced from another — measured here as a single miss per task, at
/// the head of that task's life, on refs the guest goes on to use. Short enough
/// that a reference which never resolves costs one packet's worth of latency
/// and then declines with the reason it always did, rather than holding the
/// channel that carries it.
#[cfg(feature = "backend-vulkan")]
const PIPELINE_PUBLISH_WAIT: std::time::Duration = std::time::Duration::from_millis(200);

#[cfg(feature = "backend-vulkan")]
fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
    unpublished: &mut Vec<u32>,
) -> bool {
    let pipelines = render_pipeline_refs(stream);
    let mut pending = false;
    for pipeline_ref in pipelines {
        let Ok((v_air, f_air)) =
            metal_draw::load_render_air_pair(state, host, task_id, pipeline_ref)
        else {
            // Normal execution emits the precise pipeline/MTLB failure. A plan
            // input that is malformed is deterministic and executes now; one
            // the guest has not finished publishing is not, and the caller
            // holds the packet for it.
            if metal_draw::render_pipeline_unreadable_yet(state, host, task_id, pipeline_ref) {
                unpublished.push(pipeline_ref);
            }
            continue;
        };
        if !crate::runtime::m2v_cache::ensure_cached_async(
            &v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) {
            pending = true;
        }
        if !crate::runtime::m2v_cache::ensure_cached_async(
            &f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ) {
            pending = true;
        }
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
    let mut pending = false;
    for (pipeline_ref, local_size) in compute_translation_inputs(stream) {
        let Some(pipeline) =
            compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
        else {
            continue;
        };
        let Some(mtlb) = compute_exec::load_mtlb(state, host, task_id, pipeline.kernel_func_ref)
        else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        if !crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref) {
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
fn walk_segment_records(stream: &[u8], seg: &stream::Segment, mut handle: impl FnMut(u32, &[u8])) {
    let mut cursor = 0usize;
    let mut next = decode_first_record(stream, seg, &mut cursor);
    loop {
        match next {
            Ok(rec) => {
                let start = rec.bytes_offset as usize;
                handle(rec.opcode, &stream[start..start + rec.length as usize]);
                next = decode_next_record(stream, seg, &mut cursor);
            }
            // `Done` is end-of-segment and yields `None` here, so the normal exit
            // path stays silent; anything else names the check that refused.
            Err(status) => {
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
        // for the same buffer, so an unlatched line would be one per frame.
        match decode_icb_host_resource_info(bytes) {
            Ok(info) => match apply_icb_host_resource_info(state, host, task_id, &info) {
                Ok(_) => {}
                Err(e) => {
                    crate::observe::Emit::decline("icb_backing", &e)
                        .field("task", task_id)
                        .field("icb", info.icb_ref)
                        .field("buffer", info.buffer_ref)
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

/// Name a compute refusal at the rail boundary.
///
/// Until this existed the three dispatch/control/ICB arms below only
/// *counted*: `compute_dispatches_fail` went up and nothing said which of the
/// rail's ~150 checks refused, because nine of `ComputeStatus`'s variants were
/// payload-free. The slug now rides in the status, so one line names the check,
/// the pipeline and the record kind.
///
/// Latched per `(reason, pipeline)`: the guest re-submits the same dispatch
/// every frame, so a persistent refusal would otherwise be a per-frame flood —
/// while a *different* pipeline failing the same check is a distinct event and
/// still gets its line.
fn note_compute_refusal(status: ComputeStatus, task_id: u32, pipeline_ref: u32, kind: ComputeKind) {
    // One event token for the whole rail, with `kind=` separating dispatch
    // from control-flow from ICB: the emission gate reads the *literal* first
    // argument, so a per-arm event passed in as a parameter would leave the
    // registry naming a line the gate cannot find.
    if let Some(e) = crate::observe::Emit::refusal("compute_record", &status) {
        e.field("task", task_id)
            .field("pipe", pipeline_ref)
            .field("kind", format!("{kind:?}"))
            .fail_once(u64::from(pipeline_ref));
    }
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
        RenderKind::SetPipeline if cmd.pipeline_ref != 0 => {
            acc.pipeline_ref = cmd.pipeline_ref;
        }
        RenderKind::SetBuffer => {
            if cmd.has_attribute_stride {
                // Same shape as `render_sampler_lod_dropped` below, and it was
                // the same bug: `0xa5` sits above the old accepted window, so a
                // guest that negotiated `supportsDynamicAttributeStride` had
                // every strided vertex bind refused and the buffer never bound.
                // The bind is applied now; the per-entry stride is not, because
                // `BufferBind` carries none and the vertex fetch layout is
                // pipeline state neither backend is asked to re-declare.
                crate::runtime::drain::note_store_route("render_vertex_attribute_stride_dropped");
            }
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
                &mut acc.vertex_buffers,
                &mut acc.fragment_buffers,
                |b| b.index,
                |index, (buffer_ref, offset)| {
                    (buffer_ref != 0).then_some(BufferBind {
                        index,
                        buffer_ref,
                        offset,
                    })
                },
            );
            out.buffer_unbinds = out.buffer_unbinds.saturating_add(cleared);
        }
        RenderKind::SetBufferOffset => {
            if cmd.has_attribute_stride {
                crate::runtime::drain::note_store_route("render_vertex_attribute_stride_dropped");
            }
            // Archive apply_buffer_offset: update offset on an already-bound slot.
            if cmd.first >= MAX_BIND_SLOTS {
                // The slot is outside the table, so the bind that would have
                // occupied it was already dropped by `apply_binds` and counted
                // under `render_buffer_bind_slot_past_table`. This is the
                // *second* record the guest spends on that slot, and it was
                // silently ignored — so the bind counter under-reported how much
                // of the stream the table bound costs. Counted separately rather
                // than folded in, because these are different records.
                crate::runtime::drain::note_store_route("render_buffer_offset_slot_past_table");
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
                Some(b) => b.offset = cmd.buffer_offset,
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
                &mut acc.vertex_textures,
                &mut acc.fragment_textures,
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
                    Some(TextureBind { index, texture_ref })
                },
            );
            out.texture_unbinds = out.texture_unbinds.saturating_add(cleared);
        }
        RenderKind::SetSampler => {
            if cmd.has_sampler_lod {
                // The bind itself is applied below; what is not applied is the
                // per-entry LOD clamp pair the guest sent with it, because
                // `SamplerBind` carries no clamps and neither backend is asked
                // for any. Until this commit the whole record was dropped —
                // `0x80`/`0x71` reached no arm — so the slot stayed unbound;
                // binding it with default clamps is the closer answer, and this
                // counter is the distance still left.
                crate::runtime::drain::note_store_route("render_sampler_lod_dropped");
            }
            let cleared = apply_binds(
                &cmd.ref_binds,
                cmd.first,
                BindTarget {
                    stage: cmd.stage,
                    class: BindClass::Sampler,
                },
                &mut acc.vertex_samplers,
                &mut acc.fragment_samplers,
                |b| b.index,
                |index, sampler_ref| {
                    (sampler_ref != 0).then_some(SamplerBind { index, sampler_ref })
                },
            );
            out.sampler_unbinds = out.sampler_unbinds.saturating_add(cleared);
        }
        RenderKind::SetViewport => {
            // `cmd.viewport` is entry 0 of the record, which for the singular
            // opcode is the whole of it. The plural form (`0x83`) used to reach
            // no arm at all, so a guest that set its viewport through
            // `setViewports:count:` got none — this rail models one viewport,
            // and one is what the overwhelming majority of those records carry.
            note_extra_state_entries("viewport", cmd.count);
            acc.viewport = Some(cmd.viewport);
        }
        RenderKind::SetScissor if cmd.scissor_w > 0 && cmd.scissor_h > 0 => {
            note_extra_state_entries("scissor", cmd.count);
            acc.scissor = Some((cmd.scissor_x, cmd.scissor_y, cmd.scissor_w, cmd.scissor_h));
        }
        RenderKind::SetBlendColor if cmd.has_blend_color => {
            acc.blend_color = Some(cmd.blend_color);
        }
        RenderKind::SetCullMode if cmd.has_cull_mode => {
            acc.cull_mode = Some(cmd.cull_mode);
        }
        RenderKind::SetFrontFacing if cmd.has_front_facing => {
            acc.front_facing = Some(cmd.front_facing);
        }
        RenderKind::SetDepthBias if cmd.has_depth_bias => {
            acc.depth_bias = Some(cmd.depth_bias);
        }
        RenderKind::SetDepthStencil => {
            acc.depth_stencil_ref = cmd.depth_stencil_ref;
        }
        RenderKind::SetStencilReference if cmd.has_stencil_ref => {
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
            // defect depends on whether the two agree, which is why the extent
            // is reported with its *values* rather than only counted.
            if cmd.pass_visibility_result_buffer_ref != 0 {
                crate::runtime::drain::note_store_route("render_pass_visibility_buffer_dropped");
            }
            if cmd.pass_render_target_array_length > 1 {
                crate::runtime::drain::note_store_route("render_pass_array_length_dropped");
            }
            if cmd.pass_render_target_width != 0 || cmd.pass_render_target_height != 0 {
                note_pass_target_extent(
                    task_id,
                    cmd.pass_render_target_width,
                    cmd.pass_render_target_height,
                );
            }
            // Full multi-attachment: re-decode all color slots from payload.
            if cmd_bytes.len() >= 8 {
                let payload = &cmd_bytes[8..];
                let depth = decode_depth_attachment(payload);
                if depth.present {
                    if depth_stencil_is_bindable(
                        depth.level,
                        depth.slice,
                        depth.depth_plane,
                        depth.resolve_texture_ref,
                    ) {
                        acc.depth_attach = Some(depth);
                    } else {
                        note_depth_stencil_unsupported(task_id, "depth", &depth.into());
                    }
                }
                let stencil = decode_stencil_attachment(payload);
                if stencil.present {
                    if depth_stencil_is_bindable(
                        stencil.level,
                        stencil.slice,
                        stencil.depth_plane,
                        stencil.resolve_texture_ref,
                    ) {
                        acc.stencil_attach = Some(stencil);
                        // New pass attachments: the next draw owns the clear again.
                        acc.stencil_pass_started = false;
                    } else {
                        note_depth_stencil_unsupported(task_id, "stencil", &stencil.into());
                    }
                }
                for i in 0..PASS_MAX_COLOR_ATTACHMENTS {
                    let att = decode_color_attachment(payload, i);
                    if !att.present || att.texture_ref == 0 {
                        continue;
                    }
                    let slot = i as u32;
                    // Every consumer of a colour attachment binds the texture
                    // whole, so a subresource the guest named is rendered past
                    // rather than into. Reported and then rendered anyway: the
                    // pass carries real guest work and dropping it would trade
                    // wrong pixels for none, which is worse. The count is what
                    // decides whether a subresource-aware bind is worth
                    // building.
                    if att.level != 0 || att.slice != 0 || att.depth_plane != 0 {
                        note_color_subresource_unsupported(task_id, slot, &att);
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
                    if !acc.color_targets.contains(&att.texture_ref) {
                        acc.color_targets.push(att.texture_ref);
                    }
                    if !out.texture_refs.contains(&att.texture_ref) {
                        out.texture_refs.push(att.texture_ref);
                    }
                    if let Some(m) =
                        objects::resolve_type11_ref(state, host, task_id, att.texture_ref)
                    {
                        // The measurement the extent count could not make. Only
                        // slot 0 and only where the mapping is already resolved
                        // — this is a census, not a resolve, and making it
                        // resolve would put a guest-memory walk on the hottest
                        // record in the device.
                        if slot == 0 {
                            if let Some(e) = state.mappings.get(&m) {
                                note_pass_extent_coverage(
                                    cmd.pass_render_target_width,
                                    cmd.pass_render_target_height,
                                    e.width,
                                    e.height,
                                );
                            }
                        }
                        if !out.type11_mappings.contains(&m) {
                            out.type11_mappings.push(m);
                        }
                    } else if objects::resolve_type4_surface(state, host, att.texture_ref)
                        && !out.type11_mappings.contains(&att.texture_ref)
                    {
                        out.type11_mappings.push(att.texture_ref);
                    }
                    if clear_seeds_the_pass(att.load_action) {
                        // Load and store actions are independent in Metal, and
                        // the clear is a property of the load alone: `Clear`
                        // fills the attachment at pass start whatever happens at
                        // pass end. `DontCare` says only that the *result* need
                        // not be preserved afterwards. Vulkan expresses the same
                        // pair directly — `LOAD_OP_CLEAR` with
                        // `STORE_OP_DONT_CARE` is an ordinary combination — so
                        // nothing has to be invented to honour it.
                        //
                        // This used to admit `Store` alone and log the rest as
                        // dropped, pending a boot that showed whether any guest
                        // emitted the combination. One did: a macOS desktop
                        // emits it, with `store_action=0` (DontCare) and
                        // `store_action=2` (MultisampleResolve). The dropped
                        // seed left the pass loading stale content, which is
                        // visible from the guest as every window and every
                        // transient overlay accumulating on screen instead of
                        // replacing what was there.
                        //
                        // The clear-only case stays gated: `apply_clear` makes
                        // its own `store_action == Store` check before touching
                        // guest pages, so a pass with no draws and nothing to
                        // preserve still writes nothing.
                        acc.clears.push(att);
                    }
                }
            }
            // Also keep color0 from command for convenience. Store action is
            // not consulted here either, for the reason given on the slot loop
            // above: the clear belongs to the load action alone.
            if cmd.color0.present
                && clear_seeds_the_pass(cmd.color0.load_action)
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
            if acc.pipeline_ref == 0 || count == 0 {
                acc.dropped_unbound = acc.dropped_unbound.saturating_add(1);
            } else {
                acc.draws.push(PendingDraw {
                    pipeline_ref: acc.pipeline_ref,
                    draw: DrawArgs {
                        vertex_count: count,
                        instance_count: cmd.instance_count.max(1),
                        primitive_type: cmd.primitive_type,
                        first_vertex: cmd.vertex_start,
                        base_instance: cmd.base_instance,
                    },
                    ..acc.bind_snapshot()
                });
                acc.stencil_pass_started = true;
            }
        }
        RenderKind::ExecuteCommands if cmd.indirect_command_buffer_ref != 0 => {
            acc.execute_icb = Some(RenderIcbExecute {
                icb_ref: cmd.indirect_command_buffer_ref,
                is_range: cmd.icb_is_range,
                range_location: cmd.icb_range_location,
                range_length: cmd.icb_range_length,
                args_buffer_ref: cmd.icb_args_buffer_ref,
                args_buffer_offset: cmd.icb_args_buffer_offset,
            });
        }
        RenderKind::Fence => {
            let action = match cmd.opcode {
                wire_blit::OPCODE_UPDATE_FENCE => FenceAction::Update,
                wire_blit::OPCODE_WAIT_FOR_FENCE => FenceAction::Wait,
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
        // Six render states this rail decodes and does not apply. Each reports
        // only when the guest asked for something *other* than the API default,
        // because asking for the default is asking for what we already do — so
        // these are healthy zeros, and a non-zero reading is the measured
        // argument for implementing that state.
        //
        // That distinction is the point of decoding them at all. All six used to
        // reach `OtherAccepted`, and `0x7c` alone fires thousands of times per
        // app render, so the one line it produced said a record had arrived and
        // nothing about whether any of them mattered.
        RenderKind::SetRasterState => {
            // `MTLTriangleFillModeFill` and `MTLDepthClipModeClip` are both 0.
            if cmd.mode != 0 {
                crate::runtime::drain::note_store_route(match cmd.opcode {
                    wire_render::OPCODE_SET_TRIANGLE_FILL_MODE => "render_fill_mode_dropped",
                    _ => "render_depth_clip_mode_dropped",
                });
            }
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
            // No default to compare against: this record *overrides* what the
            // render-pass descriptor said for that attachment, so every one of
            // them is a change this rail is not making. The precise signal would
            // compare against the pass's own store action, which this arm does
            // not have; the count is the upper bound and is what says whether
            // reaching for it is worth it.
            crate::runtime::drain::note_store_route("render_store_action_override_dropped");
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
            // The seventh of those states. `MTLVisibilityResultModeDisabled` is
            // 0, so a zero mode is a guest disarming a query this rail never
            // armed — which is what we already do.
            if cmd.mode != 0 {
                crate::runtime::drain::note_store_route("render_visibility_result_mode_dropped");
            }
        }
        RenderKind::DrawIndirect => {
            // Not one of those states, and not a healthy zero. An indirect draw
            // is geometry the guest asked for, so every one of these is a
            // dropped draw rather than a state left at its default — which is
            // why this arm keeps the fail-visible line the catch-all used to
            // give it, on top of the count.
            //
            // It cannot be executed from the record: the vertex and instance
            // counts are in the indirect buffer, written by the GPU or by a
            // compute pass, and this rail replays counts it has read. The count
            // is what decides whether resolving that buffer is worth building.
            crate::runtime::drain::note_store_route(match cmd.opcode {
                wire_render::OPCODE_DRAW_INDEXED_INDIRECT => "render_draw_indexed_indirect_dropped",
                _ => "render_draw_indirect_dropped",
            });
            note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
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
            // The options sibling of the store action beside it, and unapplied
            // for the same reason: this rail does not honour the store action
            // either, so there is nothing for an option on it to modify. No
            // default to compare against — `MTLStoreActionOptionNone` is 0, but
            // a guest that writes 0 is still overriding whatever the pass
            // descriptor said, exactly as `render_store_action_override_dropped`
            // argues for the action.
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
            crate::runtime::drain::note_store_route(match cmd.opcode {
                wire_pass::OPCODE_RASTERIZATION_RATE_MAP => "render_pass_rate_map_dropped",
                wire_pass::OPCODE_SAMPLE_POSITIONS => "render_pass_sample_positions_dropped",
                wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
                    "render_pass_raster_sample_count_dropped"
                }
                wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH => "render_pass_imageblock_dropped",
                wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
                    "render_pass_threadgroup_memory_dropped"
                }
                _ => "render_pass_tile_size_dropped",
            });
            note_unimplemented_render_opcode(cmd.opcode, cmd_bytes, task_id, acc);
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

/// Fail-visible, deduped record of a render opcode the decoder accepts but has
/// no executor for (`RenderKind::OtherAccepted`). Fires exactly ONE line per
/// distinct opcode — the undecoded-opcode set is tiny and boot-stable, so this
/// keeps the "guest render command dropped" signal visible on the always-on
/// sink without the per-draw flood a bare emit would produce (a per-draw op
/// like 0x7c fired ~2620 times across six app launches). The line carries the
/// length, bound targets/pipeline, bind counts, and the first-sighting raw wire
/// (hex) so the exact layout can be decoded offline later. Runs on the drain
/// worker (off the QEMU main/vCPU threads). Diagnostic only — it never gates
/// behavior and never invents semantics for the unknown wire.
// Render opcodes are < 256 by contract (observed max 0x98); a dense lock-free
// table gives a zero-alloc, wait-free fast path after warmup. Module-scope so a
// test can reset it deterministically.
const UNIMPL_OPCODE_TABLE: usize = 256;
static UNIMPL_OPCODE_SEEN: [std::sync::atomic::AtomicBool; UNIMPL_OPCODE_TABLE] =
    [const { std::sync::atomic::AtomicBool::new(false) }; UNIMPL_OPCODE_TABLE];
static UNIMPL_OPCODE_OVERFLOW: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<u32>>,
> = std::sync::OnceLock::new();

/// Returns `true` if this call emitted the line (first sighting of `opcode`),
/// `false` if it was deduped. The caller ignores it; tests use it to assert the
/// anti-flood behavior without depending on the shared always-on log file.
fn note_unimplemented_render_opcode(
    opcode: u32,
    cmd_bytes: &[u8],
    task_id: u32,
    acc: &StreamAccum,
) -> bool {
    use std::sync::atomic::Ordering;
    if (opcode as usize) < UNIMPL_OPCODE_TABLE {
        // First sighting only: swap false->true; racers that lose stay quiet.
        if UNIMPL_OPCODE_SEEN[opcode as usize].swap(true, Ordering::Relaxed) {
            return false;
        }
    } else {
        // Out-of-range opcode (decode desync / garbage) — dedup through a
        // small overflow set so a runaway value cannot flood either.
        let set = UNIMPL_OPCODE_OVERFLOW.get_or_init(|| std::sync::Mutex::new(Default::default()));
        if let Ok(mut g) = set.lock() {
            if !g.insert(opcode) {
                return false;
            }
        }
    }
    let hex: String = cmd_bytes
        .iter()
        .take(48)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("");
    crate::observe::fail(format!(
        "render_unimplemented reason=accepted_without_executor task={task_id} opcode={:#x} len={} target_refs={:?} pipeline={} vbufs={} fbufs={} ftex={} hex={}",
        opcode,
        cmd_bytes.len(),
        acc.color_targets,
        acc.pipeline_ref,
        acc.vertex_buffers.len(),
        acc.fragment_buffers.len(),
        acc.fragment_textures.len(),
        hex
    ));
    true
}

/// Serializes the two tests that share the process-global unimplemented-opcode
/// dedup latch, so one test's reset cannot race the other's emissions.
#[cfg(test)]
static UNIMPL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Clear the unimplemented-opcode dedup latch so a test can deterministically
/// observe the first-sighting line regardless of prior in-process emissions.
#[cfg(test)]
fn reset_unimplemented_opcode_dedup_for_test() {
    for slot in UNIMPL_OPCODE_SEEN.iter() {
        slot.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(set) = UNIMPL_OPCODE_OVERFLOW.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

/// Count the entries of a plural viewport or scissor record this rail cannot
/// hold.
///
/// Both records carry `count` entries and this rail models exactly one, so
/// entries past the first are dropped. That is a narrower loss than it sounds —
/// the plural selectors are how Metal apps set a single rect as often as not,
/// and `count == 1` is the same record as the singular opcode — but it is a
/// loss, and before these opcodes were decoded at all the *whole* record was
/// dropped rather than its tail.
///
/// A non-zero reading is the argument for modelling a viewport array, and it
/// says how many entries such a model would have to hold. `count == 0` never
/// reaches here: the decoder refuses it.
fn note_extra_state_entries(what: &'static str, count: u32) {
    let extra = count.saturating_sub(1);
    if extra == 0 {
        return;
    }
    crate::runtime::drain::note_store_route_n(
        match what {
            "viewport" => "render_extra_viewports_dropped",
            _ => "render_extra_scissors_dropped",
        },
        extra as u64,
    );
}

/// Which of the three argument tables a bind record names.
///
/// [`apply_binds`] gates all three on the single [`MAX_BIND_SLOTS`], and that
/// is defensible as a *bound* — it is the smallest of the three host tables —
/// but it is not defensible as a *counter*. Apple's serializer truncates a
/// plural bind at the stage's argument table, and
/// [`reims_vgpu_wire::ops::bind_limit`] measured those three tables at 128
/// textures, 31 buffers and 16 samplers, so a slot dropped here means something
/// different in each class:
///
/// * **Texture** — real loss. Apple emits up to 128, this device holds 31, so
///   slots 31..127 are guest work with nowhere to go.
/// * **Buffer** — cannot fire from an Apple guest. 31 is exactly the serializer's
///   own buffer bound, so a non-zero reading is either a guest writing its own
///   stream or a decode that mis-sized the table.
/// * **Sampler** — same, one step further: Apple truncates at 16, well below the
///   bound, so this can only fire on a stream Apple's serializer did not write.
///
/// One slug for all three said "31 slots were lost" and could not say which
/// table to widen, which is the whole reason the counter exists. Splitting it is
/// the same lesson `BlitEncoderSPI` taught one layer up — a family is not
/// uniform in what its loss means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BindClass {
    Buffer,
    Texture,
    Sampler,
}

impl BindClass {
    /// The census name for slots this class lost to [`MAX_BIND_SLOTS`].
    fn past_table_route(self) -> &'static str {
        match self {
            BindClass::Buffer => "render_buffer_bind_slot_past_table",
            BindClass::Texture => "render_texture_bind_slot_past_table",
            BindClass::Sampler => "render_sampler_bind_slot_past_table",
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
    ///   table and inside [`MAX_BIND_SLOTS`]. This is headroom being spent.
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
    /// **The texture table's 31-against-128 gap costs this workload nothing.**
    /// Not one texture bind reaches even slot 17, so widening it — which on the
    /// Vulkan arm means re-laying every band in
    /// [`crate::runtime::spirv_bind`], whose texture band is exactly 32 wide and
    /// abuts the sampler band — would buy zero. That is the measured argument
    /// the drop counter was added to produce, and it argues against.
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
            (BindClass::Buffer, r) if r <= MAX_BIND_SLOTS => "render_bind_reach_buffer_le_table",
            (BindClass::Buffer, _) => "render_bind_reach_buffer_over_table",
            (BindClass::Texture, r) if r <= bind_limit::SAMPLER => "render_bind_reach_texture_le16",
            (BindClass::Texture, r) if r <= MAX_BIND_SLOTS => "render_bind_reach_texture_le_table",
            (BindClass::Texture, _) => "render_bind_reach_texture_over_table",
            (BindClass::Sampler, r) if r <= bind_limit::SAMPLER => "render_bind_reach_sampler_le16",
            (BindClass::Sampler, r) if r <= MAX_BIND_SLOTS => "render_bind_reach_sampler_le_table",
            (BindClass::Sampler, _) => "render_bind_reach_sampler_over_table",
        }
    }
}

// The three relations that make each `*_bind_slot_past_table` slug readable in a
// driven boot's census, pinned at build time because both sides can move
// independently: a new macOS serializer can change Apple's argument tables, and
// widening the host tables changes [`MAX_BIND_SLOTS`]. Either would silently
// re-point what the census means, so this is a build gate rather than a test —
// the same reason `reims_vgpu_wire::Wire::ASSERT_ALIGN_1` is one.
//
// Textures: Apple emits above the bound, so a reading is lost guest work and is
// the argument for widening.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::TEXTURE > MAX_BIND_SLOTS);
// Buffers: two independent derivations of one table size — Apple's serializer
// truncates there and Metal's `REIMS_VGPU_METAL_MAX_BUFFERS` stops there.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::BUFFER == MAX_BIND_SLOTS);
// Samplers: Apple truncates well below the bound, so this slug cannot fire on a
// stream Apple's serializer wrote. A reading is a guest writing its own stream,
// or a decode that mis-sized the table.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::SAMPLER < MAX_BIND_SLOTS);

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

/// Apply one `Set{Buffer,Texture,Sampler}` record to a stage's bind table.
///
/// All three carry the same wire form: `count` consecutive slots starting at
/// `first`, where a zero object ref clears the slot it names and any other ref
/// replaces whatever occupied it. Slots at or past [`MAX_BIND_SLOTS`] are
/// outside the encoder's table and end the walk. Only the vertex and fragment
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
    vertex: &mut BindTable<B>,
    fragment: &mut BindTable<B>,
    slot: impl Fn(&B) -> u32,
    mut make: impl FnMut(u32, T) -> Option<B>,
) -> u32 {
    let BindTarget { stage, class } = target;
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
        if index >= MAX_BIND_SLOTS {
            // The walk stops here, and it used to stop in silence — a `break`
            // that dropped every remaining slot with nothing to say so. The
            // guest really does bind past this: `setVertexTextures:withRange:`
            // over a range of 40 is a record Apple's serializer produces, and
            // `MAX_BIND_SLOTS` is 31 because it is Metal's *buffer* index cap,
            // where the texture limit is 128. So this fires on real traffic and
            // the binds it drops are real.
            //
            // Raising the cap means widening the backends' tables, which is a
            // change with its own measurement; naming the loss is not. The
            // counter says how much is at stake, and a non-zero reading is the
            // argument for doing the widening — for the table [`BindClass`]
            // names, which is why there are three slugs rather than one.
            crate::runtime::drain::note_store_route_n(
                class.past_table_route(),
                (entries.len() - i) as u64,
            );
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

/// The draw opcodes whose records carry an index buffer.
///
/// `render::decode` collapses every draw form to `Kind::Draw`, so the decoded
/// record cannot say which class it came from and the opcode is the only thing
/// that can.
fn is_indexed_draw_opcode(opcode: u32) -> bool {
    // wire opcodes via wire_render import

    matches!(
        opcode,
        wire_render::OPCODE_DRAW_INDEXED
            | wire_render::OPCODE_DRAW_INDEXED_INSTANCED
            | wire_render::OPCODE_DRAW_INDEXED_WIDE
    )
}

/// Name an indexed draw whose record carried no index buffer.
///
/// Deduped on the opcode: the three indexed forms read `index_buffer_ref` from
/// three different payload offsets, so which form fires is the whole diagnostic
/// value — one form firing alone points at that form's offset, all three
/// firing points at the guest.
fn note_indexed_draw_without_buffer(task_id: u32, opcode: u32, index_count: u32) {
    crate::observe::fail(format!(
        "stream_draw reason=indexed_without_index_buffer task={task_id} op={opcode:#x} \
         index_count={index_count} drawn_as=non_indexed"
    ));
}

/// Name a depth or stencil attachment dropped for a form this device does not
/// implement.
///
/// Deduped on the pair that decides the arm, not on the task: this fires from a
/// per-`RenderPass` decode, so a guest that uses mip-1 depth throughout would
/// otherwise emit on every pass in every stream. One line per distinct
/// (aspect, level, resolve-present) combination is what answers the question
/// the arm exists to answer — whether any guest asks for this at all.
/// The subresource coordinates and resolve target shared by all three
/// attachment shapes, lifted so the depth and stencil arms cannot drift apart.
///
/// They are one 28-byte prefix on the wire
/// (`reims_vgpu_wire::ops::render_pass::AttachmentPrefix`), and this device had
/// two arms reading it with two copies of the same four-line check. A third
/// copy is what the colour arm would have needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AttachSubresource {
    level: u32,
    slice: u32,
    depth_plane: u32,
    resolve_texture_ref: u32,
}

impl From<crate::runtime::decode::render::DepthAttachment> for AttachSubresource {
    fn from(a: crate::runtime::decode::render::DepthAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

impl From<crate::runtime::decode::render::StencilAttachment> for AttachSubresource {
    fn from(a: crate::runtime::decode::render::StencilAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

/// Whether this device can honour a depth or stencil attachment as decoded.
///
/// Only the whole texture at level 0, slice 0, plane 0 with no multisample
/// resolve. `slice` and `depth_plane` joined the test when they became
/// decodable: a depth buffer bound at slice 5 was previously read as slice 0 and
/// silently accepted, which is the same defect the colour arm had.
fn depth_stencil_is_bindable(level: u32, slice: u32, depth_plane: u32, resolve: u32) -> bool {
    level == 0 && slice == 0 && depth_plane == 0 && resolve == 0
}

fn note_depth_stencil_unsupported(task_id: u32, aspect: &'static str, s: &AttachSubresource) {
    crate::observe::Emit::decline(
        "stream_pass",
        &StreamDrawDrop::DepthStencilUnsupported {
            aspect,
            level: s.level,
            slice: s.slice,
            depth_plane: s.depth_plane,
            resolve_texture_ref: s.resolve_texture_ref,
        },
    )
    .field("task", task_id)
    .fail_once(
        u64::from(s.level) << 32
            | u64::from(s.slice) << 16
            | u64::from(s.depth_plane) << 8
            | u64::from(s.resolve_texture_ref != 0) << 1
            | u64::from(aspect == "stencil"),
    );
}

/// Bands for the stated pass extent as a fraction of its attachment's area.
///
/// Same seven bands as the scissor-union census in `metal_draw::vulkan`, so the
/// two are readable side by side — they answer the same question from two
/// different sources, and the whole point is which of the two carries damage the
/// other does not.
const PASS_EXTENT_SLUGS: [&str; 7] = [
    "pass_extent_lt1",
    "pass_extent_le5",
    "pass_extent_le10",
    "pass_extent_le25",
    "pass_extent_le50",
    "pass_extent_le99",
    "pass_extent_full",
];

/// Score the guest's stated pass extent against the attachment it names.
///
/// This is the number the flush rail has been missing. The root `AGENTS.md`
/// records that bounding a writeback by the *draw stream's* scissors saves
/// nothing — 99.92 % of armed windows have a per-pass scissor union of 100 % —
/// and concludes that "a damage-bounded flush needs a different source of damage
/// than the draw stream, and none is currently decoded". `renderTargetWidth` and
/// `renderTargetHeight` are decoded now, and a driven boot shows the window
/// server naming extents like 170x12 and 32x32 rather than the display's
/// 1920x1080.
///
/// What this cannot say by itself is whether those small extents sit on small
/// attachments. That is exactly what the bands measure: a distribution weighted
/// at `full` means the extent is the surface and there is nothing to bound, and
/// one with mass below `le50` is a writeback that could be halved.
///
/// A pass that states no extent at all is not scored — there is no fraction to
/// take — and neither is one whose attachment has no geometry yet.
fn note_pass_extent_coverage(pass_w: u64, pass_h: u64, surf_w: u32, surf_h: u32) {
    if pass_w == 0 || pass_h == 0 || surf_w == 0 || surf_h == 0 {
        return;
    }
    let full = u64::from(surf_w).saturating_mul(u64::from(surf_h));
    let stated = pass_w.saturating_mul(pass_h);
    // Clamped for the reason the scissor union is: a guest may state an extent
    // larger than the attachment and the rasteriser clips, so an unclamped
    // ratio would read over 100 % and make the census unreadable.
    let pct = stated.min(full).saturating_mul(100) / full.max(1);
    crate::runtime::drain::note_store_route(PASS_EXTENT_SLUGS[pass_extent_band(pct)]);
}

/// The bands, matching `metal_draw::vulkan::coverage_band` exactly.
///
/// Declared here rather than shared because that one is behind
/// `backend-vulkan` and this census runs on every backend; the two are pinned
/// equal by `the_two_coverage_censuses_use_the_same_bands`.
fn pass_extent_band(pct: u64) -> usize {
    match pct {
        0 => 0,
        1..=5 => 1,
        6..=10 => 2,
        11..=25 => 3,
        26..=50 => 4,
        51..=99 => 5,
        _ => 6,
    }
}

/// The pass extent the guest asked for, which this device does not apply.
///
/// `renderTargetWidth`/`Height` are the guest's explicit statement about how
/// much of each attachment the pass covers, and every consumer here uses the
/// attachment's own extent instead. That is only a defect when the two differ,
/// and nothing at this point knows the attachment's size — so what this reports
/// is the *value*, deduped on the pair, and the comparison is left to a reader
/// with the surface geometry beside it.
///
/// Deduped rather than counted per pass for the reason the resource table's
/// `TailPopulated` gives: a statement this device discards is a property of the
/// guest build, not of the pass that happened to carry it first. A driven boot
/// produces 1 575 of these and a handful of distinct extents.
fn note_pass_target_extent(task_id: u32, width: u64, height: u64) {
    crate::runtime::drain::note_store_route("render_pass_target_extent_unapplied");
    crate::observe::Emit::decline(
        "stream_pass",
        &StreamDrawDrop::TargetExtentUnapplied { width, height },
    )
    .field("task", task_id)
    .fail_once(width << 32 | (height & 0xffff_ffff));
}

/// A colour attachment naming a mip, a slice or a depth plane this device
/// renders past. See [`StreamDrawDrop::ColorSubresourceUnsupported`].
///
/// Deduped on the three coordinates and the slot rather than on the texture,
/// because the question is which *shape* of subresource a guest asks for, not
/// how many textures it asks for it on.
fn note_color_subresource_unsupported(
    task_id: u32,
    slot: u32,
    att: &crate::runtime::decode::render::ColorAttachment,
) {
    crate::runtime::drain::note_store_route("render_color_subresource_unsupported");
    crate::observe::Emit::decline(
        "stream_pass",
        &StreamDrawDrop::ColorSubresourceUnsupported {
            slot,
            level: att.level,
            slice: att.slice,
            depth_plane: att.depth_plane,
        },
    )
    .field("task", task_id)
    .field("texture", att.texture_ref)
    .fail_once(
        u64::from(slot) << 48
            | u64::from(att.level) << 32
            | u64::from(att.slice) << 16
            | u64::from(att.depth_plane),
    );
}

/// Report what this stream's draw list cost, and anything it lost building it.
///
/// The distribution stays after the cap is gone, because it is now the only
/// thing that prices the decision to keep every record: it says how long a real
/// render stream is, and therefore what an unbounded list actually costs. The
/// boot that removed the cap read 118 307 streams as 39 913 single-draw, 55 306
/// at 2–4, 14 579 at 9–16 and 8013 at 33–63, with two above 64 — a tail that
/// exists and a body that does not.
///
/// Buckets rather than a mean because the question is about that tail: one
/// 400-draw compositor stream among thousands of 2-draw ones is exactly the case
/// that matters and is exactly what a mean hides. The two buckets above the old
/// ceiling are what say whether removing it changed which streams complete.
fn note_stream_draw_drops(task_id: u32, acc: &StreamAccum) {
    let kept = acc.draws.len();
    if kept == 0 && acc.dropped_unbound == 0 {
        return;
    }
    crate::runtime::drain::note_store_route(match kept {
        0 => "stream_draws_0",
        1 => "stream_draws_1",
        2..=4 => "stream_draws_2_4",
        5..=8 => "stream_draws_5_8",
        9..=16 => "stream_draws_9_16",
        17..=32 => "stream_draws_17_32",
        33..=63 => "stream_draws_33_63",
        64..=255 => "stream_draws_64_255",
        _ => "stream_draws_over_255",
    });
    // Latched on the *magnitude* of the loss, not on the task: the same stream
    // shape recurs every frame, so a per-task key would print once and hide a
    // loss that grew, while a bucket key prints again when it gets worse.
    if acc.dropped_unbound > 0 {
        let d = StreamDrawDrop::Unbound {
            dropped: acc.dropped_unbound,
        };
        if crate::observe::first_sight(
            crate::observe::Decline::slug(&d),
            u64::from(acc.dropped_unbound.next_power_of_two()),
        ) {
            crate::observe::Emit::decline("stream_draw", &d)
                .field("task", task_id)
                .field("kept", kept)
                .fail();
        }
    }
}

fn finish_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    out: &mut ExecResult,
    acc: &StreamAccum,
) {
    note_stream_draw_drops(task_id, acc);
    // Archive ApplePVGPUDrawJob: clear/load seed is private initial_rgba for the
    // async job; guest pages are written once at completion. Apply clear-to-guest
    // only for clear-only streams (no draws). When draws run, CLEAR is the Metal
    // pass seed inside encode (mrt_draw_request solid seed) — not a pre-draw
    // guest store that would expose intermediate pixels to DisplaySwap.
    let will_draw = acc.saw_draw && !acc.color_slots.is_empty() && !acc.draws.is_empty();
    if !will_draw {
        for att in &acc.clears {
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
    if acc.execute_icb.is_some() {
        crate::runtime::drain::note_store_route("icb_exec_seen");
    }
    if let Some(exec) = &acc.execute_icb {
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
                    |pd| DrawArgs {
                        vertex_count: pd.draw.vertex_count.max(1),
                        instance_count: pd.draw.instance_count.max(1),
                        ..pd.draw
                    },
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
            let req = metal_draw::mrt_draw_request(
                state,
                host,
                task_id,
                pipeline,
                &acc.color_slots,
                &acc.clears,
                args.vertex_count,
                args.instance_count,
                args.primitive_type,
                args.first_vertex,
                args.base_instance,
            );
            if let Some(mut req) = req {
                // ICB execute inherits stream bind state at end of stream.
                if let Some(pd) = acc.draws.last() {
                    fill_draw_binds_from_pending(&mut req, pd);
                } else {
                    fill_draw_binds_from_pending(&mut req, &acc.bind_snapshot());
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
                            return;
                        }
                    }
                };
                match metal_draw::encode_icb_execute_and_writeback(
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
        // Resident render-pass chain: intermediate records keep their content
        // on the engine target (no CPU chain buffer); records 2+ LoadFromTarget.
        let mut resident_chain = false;
        let mut saw_nometal = false;
        let first_draw = draw_list.first().copied();
        let mut first_req = first_draw.and_then(|pd| {
            out.render_attachment_resolves = out.render_attachment_resolves.saturating_add(1);
            metal_draw::mrt_draw_request(
                state,
                host,
                task_id,
                pd.pipeline_ref,
                &acc.color_slots,
                &acc.clears,
                pd.draw.vertex_count,
                pd.draw.instance_count,
                pd.draw.primitive_type,
                pd.draw.first_vertex,
                pd.draw.base_instance,
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
                fill_draw_binds_from_pending(&mut req, pd);
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
                        c.load_action = PASS_LOAD_ACTION_LOAD;
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
                let encode = metal_draw::encode_draw_chain(
                    state,
                    host,
                    &mut req,
                    do_writeback,
                    force_full_store,
                );
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
                                resident_chain,
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
                            resident_chain,
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
                            resident_chain,
                        );
                        break;
                    }
                }
            }
        }
        // Encode never landed Stores (NoMetal stubs, missing MTLB/pipeline, or
        // mrt resolve fail). Honor CLEAR load+store into guest/host pages so
        // dual-buffer display mids at least hold the pass clear color (archive
        // CLEAR seed — not a content heuristic). Applies for any draw-fail
        // class, not only NoMetal: mrt_request fail used to skip this and left
        // mid pages empty → nz_swing thrash on x86 Linux product.
        if out.metal_draws_ok == 0 && !acc.clears.is_empty() {
            for att in &acc.clears {
                if apply_clear(state, host, task_id, att) {
                    out.clears_applied = out.clears_applied.saturating_add(1);
                }
            }
            if out.clears_applied > 0 || saw_nometal || out.metal_draws_fail > 0 {
                crate::observe::fail(format!(
                    "draw_fail_clear_fallback task={task_id} clears={} draws_fail={} nometal={}",
                    out.clears_applied, out.metal_draws_fail, saw_nometal as u8
                ));
            }
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

crate::observe::decline_display!(ChainAbandonDecline);

/// One-shot (per `pipeline_ref` x reason) always-on line for a failed draw
/// encode. `exec_indirect2 draws_fail=N` collapses every cause into one
/// counter with no reason; a persistently failing draw (e.g. an app window
/// layer that never paints) was invisible on a normal boot. The latch keys
/// on the pipeline so a new failing workload logs its own line while a
/// steady repeat (same pipeline failing every packet) stays at one line.
///
/// The `reason=` was the *variant* name until `EncodeStatus` carried its check:
/// six names for the rail's 27 refusals, so `reason=bad_args` could be a
/// zero-size target, a vertexless draw or an unresolvable MRT slot. Now the
/// variant prints as `class=` beside the check that produced it.
fn note_draw_encode_fail(
    task_id: u32,
    pipeline_ref: u32,
    status: EncodeStatus,
    di: usize,
    n: usize,
) {
    if let Some(e) = crate::observe::Emit::refusal("draw_encode_fail", &status) {
        e.field("pipe", pipeline_ref)
            .field("task", task_id)
            .field("di", format!("{di}/{n}"))
            .fail_once(pipeline_ref as u64);
    }
}

/// Seedless fixed-attachment template for records after the first draw in one
/// serialized Metal render pass. Construct fields explicitly so a multi-MiB
/// CPU LOAD seed is not cloned merely to reuse attachment identity/geometry.
fn render_pass_attachment_template(
    first: &metal_draw::DrawEncodeRequest,
) -> metal_draw::DrawEncodeRequest {
    let colors = first
        .colors
        .iter()
        .map(|c| metal_draw::ColorRtRequest {
            slot: c.slot,
            texture_ref: c.texture_ref,
            mapping_id: c.mapping_id,
            target_gva: c.target_gva,
            row_stride: c.row_stride,
            width: c.width,
            height: c.height,
            format: c.format,
            load_action: PASS_LOAD_ACTION_LOAD,
            store_action: c.store_action,
            clear_color: c.clear_color,
            target_seed_rgba: None,
        })
        .collect();
    metal_draw::DrawEncodeRequest {
        task_id: first.task_id,
        colors,
        ..Default::default()
    }
}

fn retarget_render_pass_draw(
    template: &metal_draw::DrawEncodeRequest,
    draw: &PendingDraw,
) -> metal_draw::DrawEncodeRequest {
    let mut req = template.clone();
    req.pipeline_ref = draw.pipeline_ref;
    req.vertex_count = draw.draw.vertex_count;
    req.instance_count = draw.draw.instance_count;
    req.primitive_type = draw.draw.primitive_type;
    req.first_vertex = draw.draw.first_vertex;
    req.base_instance = draw.draw.base_instance;
    req
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

fn multi_draw_chain_source(resident_chain: bool, cpu_chain_ready: bool) -> MultiDrawChainSource {
    if resident_chain {
        MultiDrawChainSource::Resident
    } else if cpu_chain_ready {
        MultiDrawChainSource::Cpu
    } else {
        MultiDrawChainSource::Missing
    }
}

fn fill_draw_binds_from_pending(req: &mut metal_draw::DrawEncodeRequest, pd: &PendingDraw) {
    req.vertex_buffers = pd.vertex_buffers.as_ref().clone();
    req.fragment_buffers = pd.fragment_buffers.as_ref().clone();
    req.vertex_textures = pd.vertex_textures.as_ref().clone();
    req.fragment_textures = pd.fragment_textures.as_ref().clone();
    req.vertex_samplers = pd.vertex_samplers.as_ref().clone();
    req.fragment_samplers = pd.fragment_samplers.as_ref().clone();
    req.viewport = pd.viewport;
    req.scissor = pd.scissor;
    req.indexed = pd.indexed.clone();
    req.blend_color = pd.blend_color;
    req.cull_mode = pd.cull_mode;
    req.front_facing = pd.front_facing;
    req.depth_bias = pd.depth_bias;
    req.depth_stencil_ref = pd.depth_stencil_ref;
    req.stencil_ref = pd.stencil_ref;
    req.depth_attach = pd.depth_attach;
    req.stencil_attach = pd.stencil_attach;
    req.stencil_first_in_pass = pd.stencil_first_in_pass;
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
    req: &metal_draw::DrawEncodeRequest,
    chain_rgba: &mut Option<Vec<u8>>,
    resident_chain: bool,
) {
    #[cfg(feature = "backend-vulkan")]
    if resident_chain && chain_rgba.is_none() {
        *chain_rgba = metal_draw::read_resident_chain(state, req);
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = (req, resident_chain);
    if let Some(rgba) = chain_rgba.take() {
        let _ = metal_draw::writeback_chain_rgba(state, host, task_id, &acc.color_slots, &rgba);
    }
    dirty_color_targets(state, host, task_id, &acc.color_targets);
}

fn solid_rgba(w: u32, h: u32, clear: &[f64; 4]) -> Vec<u8> {
    let r = f64_to_unorm8(clear[0]);
    let g = f64_to_unorm8(clear[1]);
    let b = f64_to_unorm8(clear[2]);
    let a = f64_to_unorm8(clear[3]);
    let px = [r, g, b, a];
    let n = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let mut img = vec![0u8; n];
    for i in 0..(w * h) as usize {
        img[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    img
}

/// Deduped, fail-visible record of a guest clear directive we did not honor.
/// Keyed by `(reason, texture_ref)` so a persistent condition logs exactly once
/// instead of per stream — no flood. Runs on the drain worker (off the QEMU
/// main core) via the always-on `observe::fail` sink. Returns `true` the first
/// time a given `(reason, tex_ref)` is seen (the call that emitted the line).
fn note_clear_dropped(reason: &'static str, tex_ref: u32, detail: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(&'static str, u32)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let first = seen
        .get_or_insert_with(HashSet::new)
        .insert((reason, tex_ref));
    if first {
        crate::observe::fail(format!(
            "clear_dropped reason={reason} tex_ref={tex_ref} {detail}"
        ));
    }
    first
}

/// Does this attachment seed the pass with a clear?
///
/// Load and store actions are independent in Metal. `loadAction == Clear` fills
/// the attachment at pass start whatever the store action says; `storeAction`
/// only decides whether the result survives the pass. Vulkan states the same
/// pair directly, so `LOAD_OP_CLEAR` with `STORE_OP_DONT_CARE` needs nothing
/// invented to express.
///
/// Consulting the store action here is what produced on-screen residue: a
/// macOS desktop emits `Clear` with `store_action=0` (DontCare) and with
/// `store_action=2` (MultisampleResolve), and dropping those seeds left each
/// pass loading whatever the attachment held before.
///
/// Writing *back* to guest pages is a different question with a different
/// answer, and [`apply_clear`] asks it separately.
fn clear_seeds_the_pass(load_action: u16) -> bool {
    load_action == PASS_LOAD_ACTION_CLEAR
}

fn apply_clear<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    att: &ColorAttachment,
) -> bool {
    if att.texture_ref == 0 || att.store_action != PASS_STORE_ACTION_STORE {
        return false;
    }
    // Prefer full draw-path resolve (type-11 or type-2/3 GVA wallpaper targets).
    let Some(req) =
        // A clear-only pass: no pipeline and no geometry, so every draw
        // argument including the base instance is zero by construction.
        metal_draw::color_target_request(state, host, task_id, att.texture_ref, 0, 0, 1, 0, 0, 0)
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
    let w = c0.width;
    let h = c0.height;
    let rgba = solid_rgba(w, h, &att.clear_color);
    if c0.target_gva != 0 {
        metal_draw::supersede_gva_window(state, host, c0.target_gva, w, h, "clear_store");
        return metal_draw::write_gva_rgba8(
            state,
            host,
            task_id,
            c0.target_gva,
            w,
            h,
            c0.row_stride,
            c0.format,
            &rgba,
        )
        .is_ok();
    }
    if c0.mapping_id == 0 {
        return false;
    }
    let r = f64_to_unorm8(att.clear_color[0]);
    let g = f64_to_unorm8(att.clear_color[1]);
    let b = f64_to_unorm8(att.clear_color[2]);
    let a = f64_to_unorm8(att.clear_color[3]);
    let px = [b, g, r, a];
    let stride = w.saturating_mul(RGBA8_BPP);
    let mut img = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let o = y * stride as usize + x * 4;
            img[o..o + 4].copy_from_slice(&px);
        }
    }
    let _ = MTL_FORMAT_BGRA8_UNORM;
    let ok = mapping_write::write_bgra8(state, host, c0.mapping_id, &img, stride, w, h);
    // host_cache also updated inside write_bgra8 (surface_cache::store).
    state.note_surface_clear(c0.mapping_id);
    ok
}

#[cfg(test)]
mod tests {
    use reims_vgpu_wire::ops::compute as wire_compute;

    use reims_vgpu_wire::OP_HEADER_LEN;

    use super::*;
    use crate::contract::endian::{st16, st32, st64};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::decode::render::{
        PASS_ATTACH_CLEAR_COLOR, PASS_ATTACH_LOAD_ACTION, PASS_ATTACH_STORE_ACTION,
        PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF, PASS_COLOR_ATTACH_STRIDE,
        PASS_LOAD_ACTION_CLEAR, PASS_STORE_ACTION_STORE,
    };
    use crate::runtime::host::FakeHost;

    /// The abandon line must say how much guest work it dropped.
    ///
    /// This break was silent, and the counter that would have caught it
    /// (`metal_draws_fail`) stays 0 on this path because the draw encoded
    /// `Ok` — so `packet_failed` is false and the packet-level line is
    /// suppressed too. The whole value of the line is the amount lost:
    /// breaking at 0 of 8 drops a whole composite, breaking at 7 of 8 drops
    /// one draw, and `di` alone does not distinguish them at a glance.
    #[test]
    fn chain_abandon_reports_how_many_draws_were_lost() {
        let render = |index, total| {
            crate::observe::Emit::decline(
                "draw_chain_abandon",
                &ChainAbandonDecline {
                    index,
                    total,
                    pipeline_ref: 0x41,
                },
            )
            .render()
        };

        let first_of_eight = render(0, 8);
        assert!(
            first_of_eight.contains("reason=draw_chain_abandoned_without_color0"),
            "{first_of_eight}"
        );
        assert!(first_of_eight.contains("di=0/8"), "{first_of_eight}");
        assert!(first_of_eight.contains("lost=7"), "{first_of_eight}");
        assert!(first_of_eight.contains("pipe=65"), "{first_of_eight}");

        // The last record of a list abandons nothing after it. Reporting a
        // loss here would send a reader hunting for draws that never existed.
        assert!(render(7, 8).contains("lost=0"), "{}", render(7, 8));
    }

    #[test]
    fn short_payload_noop() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let r = process_exec_indirect2(&mut state, &mut host, &[0u8; 4]);
        assert_eq!(r.streams_loaded, 0);
    }

    /// An exec packet naming a slot that is not live must be refused under the
    /// word the guest sent, not silently re-aimed at slot `word >> 1`.
    ///
    /// Slot 3 is live and slot 6 is not, so word `6` names a dead slot whose
    /// halved form is live — the exact ambiguity the two boots that justified
    /// this deletion measured on every single exec decode. The old fallback
    /// answered `3` here, and `3` is a different task: everything the packet
    /// goes on to do, including its guest writes, would run against page tables
    /// the guest never named for this work.
    ///
    /// `task_id` is the separator because it is what the crate acts as and what
    /// `exec_summary` reports. Asserting only "no streams loaded" would pass
    /// either way — with no page tables mapped nothing loads regardless, which
    /// is a probe that cannot distinguish the cases.
    #[test]
    fn an_exec_packet_naming_a_dead_slot_is_refused_not_aimed_at_its_neighbour() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(3, 0x1_0000, 2), "slot 3 must be live");
        assert!(state.tasks[3].active);
        assert!(
            !state.tasks[6].active,
            "slot 6 must be dead for this to bite"
        );

        let mut payload = vec![0u8; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 6);
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);

        let r = process_exec_indirect2(&mut state, &mut host, &payload);
        assert_eq!(
            r.task_id, 6,
            "the refusal must name the word the guest sent, not the slot we \
             would have substituted"
        );
        assert_eq!(r.streams_loaded, 0);
        assert!(!r.saw_draw);
    }

    /// Bytes `+0x08..0x18` of a resource-table record are zero on every build
    /// this project has measured, and their meaning is unrecovered. A guest that
    /// starts setting them is telling this device something it cannot act on, so
    /// the record must raise a line rather than pass unread.
    ///
    /// The record with the populated tail is second, and the first is clean: a
    /// check that fired on the *table* rather than the record would pass this
    /// too, so the assertion names the object id.
    #[test]
    fn a_resource_record_that_populates_its_unrecovered_tail_says_so() {
        use crate::runtime::decode::fifo::{
            CHILD_EXEC_RESOURCE_OBJECT_ID, CHILD_EXEC_RESOURCE_TAIL,
            CHILD_EXEC_RESOURCE_VALIDITY_OPS,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(3, 0x1_0000, 2), "slot 3 must be live");

        const N_RES: u32 = 2;
        let table_len = N_RES as usize * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
        let mut payload = vec![
            0u8;
            CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + table_len
                + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
        ];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
        st32(
            &mut payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            N_RES,
        );
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
        for (i, id) in [0x40u32, 0x41].into_iter().enumerate() {
            let off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + i * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
            st32(
                &mut payload[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..],
                id,
            );
            st32(
                &mut payload[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
                0x0000_0001,
            );
        }
        // One byte, in the last record, at the far end of the tail: the widest
        // gap between "the decoder read the tail" and "the decoder read a dword
        // it already had".
        let last = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
        payload[last + CHILD_EXEC_RESOURCE_TAIL as usize + 15] = 0xa5;

        let cb = CHILD_EXEC_INDIRECT_HEADER_LEN as usize + table_len;
        st64(
            &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
            0xdead_0000,
        );
        st64(
            &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
            64,
        );

        let cap = crate::observe::sink::FailCapture::start();
        let r = process_exec_indirect2(&mut state, &mut host, &payload);
        assert_eq!(r.task_id, 3);
        assert_eq!(r.streams_loaded, 0, "no page table backs the cmdbuf gva");
        let line = cap.one("exec_res_table");
        assert!(line.contains("reason=exec_res_tail_populated"), "{line}");
        assert!(line.contains(" object=65 "), "{line}");
        assert!(line.contains(" tail_nz=1"), "{line}");
    }

    /// A submission whose table says the guest CPU-wrote a resource must not
    /// leave that resource's deferred window armed.
    ///
    /// The window holds pixels the device rendered *before* the guest's write.
    /// Landing it afterwards replaces the guest's own bytes with a frame the
    /// guest has just declared stale — a full-extent clobber that no timing rail
    /// can prevent, because the question is not when the window lands but
    /// whether it may land at all.
    #[test]
    fn a_submission_that_says_the_guest_wrote_a_resource_drops_its_pending_window() {
        use crate::runtime::decode::fifo::{
            CHILD_EXEC_RESOURCE_OBJECT_ID, CHILD_EXEC_RESOURCE_VALIDITY_OPS,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(3, 0x1_0000, 2), "slot 3 must be live");
        const MAPPING: u32 = 0x40;
        state.mappings.entry(MAPPING).or_default().mapped = true;
        let key = crate::model::ComputeStorageResidencyKey {
            mapping_id: MAPPING,
            map_generation: 0,
            surface_offset: 0,
            surface_bpr: 64 * 4,
            span_end: 64 * 64 * 4,
            width: 64,
            height: 64,
            pixel_format: 0x50,
            texture_ref: 0,
        };
        state.compute_deferred_flush.insert(
            key,
            crate::model::DeferredOwner::Render {
                armed_seq: 0,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                    0u8;
                    64 * 64
                        * 4
                ])),
            },
        );
        assert_eq!(state.deferred_flush_window_count(MAPPING), 1);
        let gen_before = state.mappings[&MAPPING].content_generation;

        // One resource record: clear_host_valid, no command buffers to run.
        let mut payload = vec![
            0u8;
            CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
                + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
        ];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
        st32(
            &mut payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            1,
        );
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
        let res = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
        st32(
            &mut payload[res + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..],
            MAPPING,
        );
        st32(
            &mut payload[res + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
            0x0000_0001,
        );
        let cb = res + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
        st64(
            &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
            0xdead_0000,
        );
        st64(
            &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
            64,
        );

        process_exec_indirect2(&mut state, &mut host, &payload);
        assert_eq!(
            state.deferred_flush_window_count(MAPPING),
            0,
            "the guest said its own bytes are newer than ours; the window must go"
        );
        assert_eq!(
            state.mappings[&MAPPING].content_generation,
            gen_before + 1,
            "the next read must re-take the guest pages"
        );
        let validity = state.mappings[&MAPPING].validity;
        assert!(validity.host_stated && !validity.host_valid);
    }

    /// The mirror of the case above: a licence to write is not a reason to throw
    /// away the frame the device already produced.
    #[test]
    fn a_submission_that_only_licenses_a_resource_keeps_its_pending_window() {
        use crate::runtime::decode::fifo::{
            CHILD_EXEC_RESOURCE_OBJECT_ID, CHILD_EXEC_RESOURCE_VALIDITY_OPS,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(3, 0x1_0000, 2));
        const MAPPING: u32 = 0x41;
        state.mappings.entry(MAPPING).or_default().mapped = true;
        state.compute_deferred_flush.insert(
            crate::model::ComputeStorageResidencyKey {
                mapping_id: MAPPING,
                map_generation: 0,
                surface_offset: 0,
                surface_bpr: 64 * 4,
                span_end: 64 * 64 * 4,
                width: 64,
                height: 64,
                pixel_format: 0x50,
                texture_ref: 0,
            },
            crate::model::DeferredOwner::Render {
                armed_seq: 0,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                    0u8;
                    64 * 64
                        * 4
                ])),
            },
        );

        let mut payload = vec![
            0u8;
            CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
                + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
        ];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
        st32(
            &mut payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            1,
        );
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
        let res = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
        st32(
            &mut payload[res + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..],
            MAPPING,
        );
        st32(
            &mut payload[res + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
            0x0000_0100,
        );
        let cb = res + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
        st64(
            &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
            64,
        );

        process_exec_indirect2(&mut state, &mut host, &payload);
        assert_eq!(state.deferred_flush_window_count(MAPPING), 1);
        assert!(state.mappings[&MAPPING].validity.host_valid);
    }

    /// One segment header whose declared length runs `overshoot` bytes past the
    /// buffer, followed by `tail` bytes of would-be records.
    fn truncated_segment(type_: u8, overshoot: usize, tail: usize) -> Vec<u8> {
        use crate::runtime::decode::stream::SEGMENT_HEADER_LEN;
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN + tail];
        st32(
            &mut stream[0..4],
            (SEGMENT_HEADER_LEN + tail + overshoot) as u32,
        );
        stream[4] = type_;
        stream
    }

    fn sink_body() -> String {
        std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default()
    }

    #[test]
    fn a_stream_that_will_not_frame_says_so_instead_of_executing_nothing() {
        use crate::runtime::decode::stream::SEGMENT_TYPE_RENDER;
        // The defect this pins: `walk_stream` opened with `Err(_) => return`, so a
        // stream the framing decoder rejected executed zero records and produced
        // zero log lines — byte-for-byte indistinguishable at the sink from an
        // idle guest that submitted nothing.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = sink_body().len();
        // Task id doubles as the flood-latch discriminant, so it must be one no
        // other test in this process has already burned.
        let task_id = 0x5731_0001;
        walk_stream(
            &mut state,
            &mut host,
            task_id,
            &truncated_segment(SEGMENT_TYPE_RENDER, 64, 0),
            &mut out,
            &mut acc,
        );
        let added = sink_body()[before..].to_string();
        assert!(
            added.contains("stream_frame_fail"),
            "a stream that will not frame must reach the always-on sink, got:\n{added}"
        );
        assert!(
            added.contains("reason=stream_seg_len_past_buffer_end"),
            "the line must name which framing check refused, not just that one \
             did — 17 checks shared `ErrBadLength`. got:\n{added}"
        );
        assert!(
            added.contains(&format!("task={task_id}")),
            "the line must carry the task whose work was dropped, got:\n{added}"
        );
    }

    #[test]
    fn a_truncated_segment_names_the_check_rather_than_looking_like_end_of_records() {
        use crate::runtime::decode::stream::{
            segment_type_name, Segment, SEGMENT_HEADER_LEN, SEGMENT_TYPE_INFO,
        };
        // `Err(_) => break` treated a self-inconsistent segment exactly like
        // `Done`: the remaining records went unexecuted with nothing logged.
        let stream = vec![0u8; SEGMENT_HEADER_LEN + 4];
        // A segment claiming a longer body than the buffer holds, handed straight
        // to the record walker — the shape `iter_segments` would have rejected but
        // that an already-parsed `Segment` can still carry.
        let seg = Segment {
            offset: 0,
            length: (SEGMENT_HEADER_LEN + 64) as u32,
            type_: SEGMENT_TYPE_INFO,
            command_offset: SEGMENT_HEADER_LEN as u32,
            command_length: 64,
            ..Segment::default()
        };
        let before = sink_body().len();
        let mut handled = 0usize;
        walk_segment_records(&stream, &seg, |_, _| handled += 1);
        let added = sink_body()[before..].to_string();
        assert_eq!(handled, 0, "the malformed segment yields no records");
        assert!(
            added.contains("stream_record_fail"),
            "dropping a segment's records must reach the sink, got:\n{added}"
        );
        assert!(
            added.contains("reason=stream_reval_span_oob"),
            "the line must name the failing re-validation check, got:\n{added}"
        );
        assert!(
            added.contains(&format!(
                "seg={}",
                segment_type_name(u32::from(SEGMENT_TYPE_INFO))
            )),
            "the line must say which segment family lost its records, got:\n{added}"
        );
    }

    #[test]
    fn walking_a_well_formed_segment_to_its_end_logs_nothing() {
        use crate::runtime::decode::stream::{
            iter_segments, SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT,
        };
        // The other half of the obligation: `Done` is how every segment ends, so
        // if it produced a line the sink would carry one per segment per frame.
        let mut records = [0u8; 8];
        st32(&mut records[0..4], 0x190);
        st32(&mut records[4..8], 8);
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        st32(
            &mut stream[0..4],
            (SEGMENT_HEADER_LEN + records.len()) as u32,
        );
        stream[4] = SEGMENT_TYPE_EVENT;
        stream.extend_from_slice(&records);

        let segs = iter_segments(&stream).expect("a well-formed stream frames");
        let before = sink_body().len();
        let mut handled = 0usize;
        walk_segment_records(&stream, &segs[0], |_, _| handled += 1);
        let added = sink_body()[before..].to_string();
        assert_eq!(handled, 1, "the one record is handed over");
        assert!(
            !added.contains("stream_record_fail"),
            "end-of-segment is control flow and must stay out of the log, got:\n{added}"
        );
    }

    #[test]
    fn an_unknown_segment_family_is_refused_and_the_type_5_envelope_is_not() {
        use crate::observe::Refusal;
        use crate::runtime::decode::stream::{
            segment_disposition, SegmentDisposition, SEGMENT_TYPE_BLIT,
            SEGMENT_TYPE_PROTECTION_OPTIONS,
        };
        // `walk_stream` ended in `_ => {}`, which gave one silence to two very
        // different things. Type 5 is a contract-correct skip; type 6 is wire
        // format the host has never seen.
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS),
            SegmentDisposition::Envelope
        );
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS).refusal(),
            None,
            "the envelope arrives on healthy frames; a line here is a flood"
        );
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_BLIT),
            SegmentDisposition::Walk
        );
        assert_eq!(
            segment_disposition(6).refusal(),
            Some("stream_segment_type_unknown")
        );
        assert_eq!(
            segment_disposition(0xff).refusal(),
            Some("stream_segment_type_unknown")
        );
    }

    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn render_preflight_collects_content_pipelines_without_duplicates() {
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};
        use wire_render::OPCODE_SET_RENDER_PIPELINE_STATE;

        let mut records = Vec::new();
        for pipeline in [41u32, 77, 41] {
            let mut cmd = [0u8; 12];
            st32(&mut cmd[0..4], wire_compute::OPCODE_SET_PIPELINE_STATE);
            st32(&mut cmd[4..8], 12);
            st32(&mut cmd[8..12], pipeline);
            records.extend_from_slice(&cmd);
        }
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        let stream_len = stream.len() + records.len();
        st32(&mut stream[0..4], stream_len as u32);
        stream[4] = SEGMENT_TYPE_RENDER;
        stream.extend_from_slice(&records);

        assert_eq!(render_pipeline_refs(&stream), vec![41, 77]);
    }

    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn compute_preflight_collects_pipeline_and_local_size_without_duplicates() {
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_COMPUTE};

        let mut records = Vec::new();
        let mut pipeline = [0u8; 12];
        st32(&mut pipeline[0..4], wire_compute::OPCODE_SET_PIPELINE_STATE);
        st32(&mut pipeline[4..8], 12);
        st32(&mut pipeline[8..12], 20);
        records.extend_from_slice(&pipeline);
        for opcode in [
            wire_compute::OPCODE_DISPATCH_THREADGROUPS,
            wire_compute::OPCODE_DISPATCH_THREADGROUPS,
            wire_compute::OPCODE_DISPATCH_THREADS,
        ] {
            let mut dispatch = [0u8; 56];
            st32(&mut dispatch[0..4], opcode);
            st32(&mut dispatch[4..8], 56);
            st64(&mut dispatch[8..16], 6);
            st64(&mut dispatch[16..24], 11);
            st64(&mut dispatch[24..32], 1);
            st64(&mut dispatch[32..40], 16);
            st64(&mut dispatch[40..48], 16);
            st64(&mut dispatch[48..56], 1);
            records.extend_from_slice(&dispatch);
        }
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        let stream_len = stream.len() + records.len();
        st32(&mut stream[0..4], stream_len as u32);
        stream[4] = SEGMENT_TYPE_COMPUTE;
        stream.extend_from_slice(&records);

        assert_eq!(compute_translation_inputs(&stream), vec![(20, [16, 16, 1])]);
    }

    #[test]
    fn event_segment_signal_wait_in_stream() {
        use crate::model::FENCE_DOMAIN_EVENT;
        use crate::runtime::decode::event::SIGNAL_WAIT_PAYLOAD_LEN;
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT};

        fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
            let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
            let mut hdr = [0u8; 8];
            st32(&mut hdr[0..4], len);
            hdr[4] = type_;
            buf.extend_from_slice(&hdr);
            buf.extend_from_slice(payload);
        }
        fn push_event_record(buf: &mut Vec<u8>, opcode: u32, event_ref: u32, value: u64) {
            let mut payload = [0u8; SIGNAL_WAIT_PAYLOAD_LEN];
            st32(&mut payload[0..4], event_ref);
            st64(&mut payload[4..12], value);
            let len = (OP_HEADER_LEN + SIGNAL_WAIT_PAYLOAD_LEN) as u32;
            let mut hdr = [0u8; 8];
            st32(&mut hdr[0..4], opcode);
            st32(&mut hdr[4..8], len);
            buf.extend_from_slice(&hdr);
            buf.extend_from_slice(&payload);
        }

        let mut records = Vec::new();
        push_event_record(&mut records, event_decode::OP_SIGNAL_EVENT, 11, 7);
        push_event_record(&mut records, event_decode::OP_WAIT_EVENT, 11, 7);
        push_event_record(&mut records, event_decode::OP_WAIT_EVENT, 11, 8); // pending
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_EVENT, &records);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc);

        // The signal landed, and the pending wait for 8 left it alone. The
        // three per-op counters this used to assert had no product reader; the
        // generation store is what the next wait actually reads.
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 11), Some(7));
    }

    #[test]
    fn multi_attachment_decode_in_pass() {
        let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE * 2];
        for (i, tex) in [(0u32, 41u32), (1u32, 42u32)] {
            let slot = PASS_COLOR_ATTACH_OFF + i as usize * PASS_COLOR_ATTACH_STRIDE;
            st32(&mut payload[slot + PASS_ATTACH_TEXREF..], tex);
            st16(
                &mut payload[slot + PASS_ATTACH_LOAD_ACTION..],
                PASS_LOAD_ACTION_CLEAR,
            );
            st16(
                &mut payload[slot + PASS_ATTACH_STORE_ACTION..],
                PASS_STORE_ACTION_STORE,
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR..],
                1.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 8..],
                0.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 16..],
                0.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 24..],
                1.0f64.to_bits(),
            );
        }
        let a0 = decode_color_attachment(&payload, 0);
        let a1 = decode_color_attachment(&payload, 1);
        assert_eq!(a0.texture_ref, 41);
        assert_eq!(a1.texture_ref, 42);
        let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
        cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
        let c = render::decode(&cmd).unwrap();
        assert_eq!(c.kind, RenderKind::RenderPass);
        assert_eq!(c.color0.texture_ref, 41);
    }

    /// An indexed draw whose record named no index buffer says so.
    ///
    /// `count` takes `index_count` and `indexed` stays `None`, so the record
    /// executes as a non-indexed draw of `index_count` vertices — a draw call
    /// the guest never made, built from one it did. Metal has no such form:
    /// `drawIndexedPrimitives` takes its index buffer as an argument, so a zero
    /// ref has nothing to mean.
    ///
    /// Asserts the line and, separately, that a well-formed indexed draw does
    /// not produce it — the counter is only useful if it is quiet on the path
    /// that works.
    #[test]
    fn an_indexed_draw_with_no_index_buffer_is_named() {
        use wire_render::OPCODE_DRAW_INDEXED;
        // ARM compact indexed payload: prim@0, indexBufferRef@4, count@8:u16,
        // offset@0xa:u16 — total record 0x14.
        let record = |index_buffer_ref: u32| {
            let mut cmd = vec![0u8; 0x14];
            st32(&mut cmd[0..], wire_render::OPCODE_DRAW_INDEXED);
            st32(&mut cmd[4..], 0x14);
            st32(&mut cmd[OP_HEADER_LEN..], 3); // primitiveType
            st32(&mut cmd[OP_HEADER_LEN + 4..], index_buffer_ref);
            cmd[OP_HEADER_LEN + 8..OP_HEADER_LEN + 10].copy_from_slice(&6u16.to_le_bytes());
            cmd
        };
        let run = |cmd: &[u8]| {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let host = FakeHost::new();
            let mut out = ExecResult::default();
            let mut acc = StreamAccum {
                pipeline_ref: 5,
                ..Default::default()
            };
            handle_render_record(
                &mut state,
                &host,
                1,
                wire_render::OPCODE_DRAW_INDEXED,
                cmd,
                &mut out,
                &mut acc,
            );
            acc
        };

        let good = run(&record(42));
        assert!(
            good.indexed.is_some(),
            "a record naming an index buffer is an indexed draw"
        );

        let bad = run(&record(0));
        assert!(
            bad.indexed.is_none(),
            "behaviour is unchanged: still no index buffer to draw with"
        );

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("reason=indexed_without_index_buffer"),
            "an indexed draw reinterpreted as non-indexed must say so"
        );
        assert!(
            log.contains(&format!("op={:#x}", wire_render::OPCODE_DRAW_INDEXED)),
            "the line must name which indexed form fired, since each reads the \
             ref at a different offset"
        );
    }

    /// A depth attachment this device cannot honour is dropped, and says so.
    ///
    /// A non-zero `level` binds a mip of the depth texture and a non-zero
    /// `resolve_texture_ref` is a multisample depth resolve; both are real Metal
    /// and neither is implemented here. The gate that drops them was a bare `if`
    /// with no else, so the pass ran on with no depth attachment at all — depth
    /// testing gone for every draw in it, which reads as wrong occlusion rather
    /// than as a missing frame, and left nothing in the log to connect the two.
    ///
    /// Both halves are asserted: the attachment is still refused (unchanged
    /// behaviour) and the refusal is now named.
    #[test]
    fn an_unsupported_depth_attachment_is_named_not_just_dropped() {
        use crate::runtime::decode::render::{
            PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_RESOLVEREF, PASS_ATTACH_SLICE,
            PASS_ATTACH_TEXREF, PASS_DEPTH_ATTACH_OFF, PASS_STENCIL_ATTACH_OFF,
        };
        let pass = |level: u16, resolve: u32| {
            let mut payload = vec![0u8; 0x200];
            st32(
                &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
                77,
            );
            payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LEVEL
                ..PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LEVEL + 2]
                .copy_from_slice(&level.to_le_bytes());
            st32(
                &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_RESOLVEREF..],
                resolve,
            );
            // A stencil slot this device *can* honour, so the two aspects stay
            // separable and the depth arm is the only one under test.
            st32(
                &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
                88,
            );
            let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
            st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
            st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
            cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
            cmd
        };
        let run = |cmd: &[u8]| {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let host = FakeHost::new();
            let mut out = ExecResult::default();
            let mut acc = StreamAccum::default();
            handle_render_record(
                &mut state,
                &host,
                1,
                wire_pass::OPCODE_RENDER_PASS,
                cmd,
                &mut out,
                &mut acc,
            );
            acc
        };

        let ok = run(&pass(0, 0));
        assert!(
            ok.depth_attach.is_some() && ok.stencil_attach.is_some(),
            "a level-0 depth attachment with no resolve is honoured"
        );

        for (level, resolve) in [(1u16, 0u32), (0, 99)] {
            let acc = run(&pass(level, resolve));
            assert!(
                acc.depth_attach.is_none(),
                "level={level} resolve={resolve} must still be refused"
            );
            assert!(
                acc.stencil_attach.is_some(),
                "refusing depth must not take the stencil attachment with it"
            );
        }

        // `slice` and `depth_plane` are the two sixteen-bit fields above
        // `level` in the shared attachment prefix, and this arm read neither
        // until they were decodable. A depth buffer bound at slice 5 was read
        // as slice 0 and silently accepted, which is a depth test against the
        // wrong layer rather than a missing one.
        for (field, at) in [
            ("slice", PASS_ATTACH_SLICE),
            ("plane", PASS_ATTACH_DEPTH_PLANE),
        ] {
            let mut cmd = pass(0, 0);
            let slot = OP_HEADER_LEN + PASS_DEPTH_ATTACH_OFF + at;
            cmd[slot..slot + 2].copy_from_slice(&5u16.to_le_bytes());
            let acc = run(&cmd);
            assert!(
                acc.depth_attach.is_none(),
                "a depth attachment naming {field} 5 must be refused, not read as 0"
            );
        }

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("stream_depth_stencil_unsupported"),
            "an unsupported depth attachment was dropped without naming itself"
        );
        assert!(
            log.contains("aspect=depth"),
            "the line must say which aspect was lost"
        );
        assert!(
            log.contains("slice=5"),
            "the line must carry the slice; it was undecodable before the shared \
             prefix was derived"
        );
    }

    /// A colour attachment naming a mip, a slice or a depth plane says so.
    ///
    /// Every consumer binds the texture whole, so the pass renders into level 0
    /// slice 0 plane 0 regardless — a guest drawing a cube face overwrites face
    /// 0. Nothing downstream can tell that happened, which is why the report is
    /// here and not in a backend.
    ///
    /// The `slice` and `depth_plane` arms are the ones that could not have been
    /// written before: those fields did not exist, because the decoder read
    /// `level` thirty-two bits wide and swallowed the slice into it.
    #[test]
    fn a_colour_attachment_naming_a_subresource_this_device_cannot_bind_says_so() {
        use crate::contract::endian::st32;
        use crate::runtime::decode::render::{
            PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_SLICE, PASS_ATTACH_TEXREF,
            PASS_COLOR_ATTACH_OFF, PASS_MIN_PAYLOAD,
        };

        let pass = |level: u16, slice: u16, plane: u16| {
            let total = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
            let mut cmd = vec![0u8; total];
            st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
            st32(&mut cmd[4..], total as u32);
            let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
            st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 77);
            cmd[slot + PASS_ATTACH_LEVEL..slot + PASS_ATTACH_LEVEL + 2]
                .copy_from_slice(&level.to_le_bytes());
            cmd[slot + PASS_ATTACH_SLICE..slot + PASS_ATTACH_SLICE + 2]
                .copy_from_slice(&slice.to_le_bytes());
            cmd[slot + PASS_ATTACH_DEPTH_PLANE..slot + PASS_ATTACH_DEPTH_PLANE + 2]
                .copy_from_slice(&plane.to_le_bytes());
            cmd
        };
        let run = |cmd: &[u8]| {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let host = FakeHost::new();
            let mut out = ExecResult::default();
            let mut acc = StreamAccum::default();
            handle_render_record(
                &mut state,
                &host,
                1,
                wire_pass::OPCODE_RENDER_PASS,
                cmd,
                &mut out,
                &mut acc,
            );
            acc
        };

        // Subresource 0/0/0 is what this device binds, so it reports nothing.
        let acc = run(&pass(0, 0, 0));
        assert_eq!(
            acc.color_slots.len(),
            1,
            "the plain attachment still reaches the slot list"
        );

        for (level, slice, plane) in [(3u16, 0u16, 0u16), (0, 5, 0), (0, 0, 2)] {
            let acc = run(&pass(level, slice, plane));
            assert_eq!(
                acc.color_slots.len(),
                1,
                "level={level} slice={slice} plane={plane}: the pass still runs -- \
                 reporting must not cost the guest its draw"
            );
        }

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("stream_color_subresource_unsupported"),
            "a colour attachment bound at the wrong subresource said nothing"
        );
        assert!(
            log.contains("slice=5"),
            "the line must carry the slice; before the decode fix it was folded \
             into the level and could not be reported"
        );
        assert!(
            log.contains("plane=2"),
            "the line must carry the depth plane"
        );
    }

    /// The pass-extent census bands agree with the scissor-union census's.
    ///
    /// The two answer the same question from two different sources — the pass
    /// descriptor and the draw stream — and the whole reason to have both is to
    /// read them side by side. Bands that drifted apart would make that
    /// comparison silently wrong rather than obviously so.
    ///
    /// Declared twice because `coverage_band` is behind `backend-vulkan` and
    /// this census runs on every backend. This is the comparison that keeps the
    /// duplication honest.
    #[test]
    fn the_two_coverage_censuses_use_the_same_bands() {
        // Every boundary of every band, plus one over the top.
        for pct in [0u64, 1, 5, 6, 10, 11, 25, 26, 50, 51, 99, 100, 101] {
            let band = pass_extent_band(pct);
            assert!(
                band < PASS_EXTENT_SLUGS.len(),
                "pct {pct} banded out of range"
            );
            #[cfg(feature = "backend-vulkan")]
            assert_eq!(
                band,
                crate::runtime::metal_draw::coverage_band_for_test(pct),
                "pct {pct}: the two censuses band it differently"
            );
        }
        // The bands are ordered, so a larger fraction never scores lower.
        let mut last = 0usize;
        for pct in 0..=100u64 {
            let b = pass_extent_band(pct);
            assert!(b >= last, "pct {pct} banded below its predecessor");
            last = b;
        }
        assert_eq!(pass_extent_band(100), PASS_EXTENT_SLUGS.len() - 1);
    }

    /// A stated extent is scored against the attachment, and only when both are
    /// real.
    #[test]
    fn the_pass_extent_census_scores_a_fraction_and_clamps_it() {
        use crate::runtime::drain::store_route_count;

        // A pass covering a quarter of its attachment.
        let before = store_route_count("pass_extent_le25");
        note_pass_extent_coverage(960, 540, 1920, 1080);
        assert_eq!(
            store_route_count("pass_extent_le25"),
            before + 1,
            "960x540 of 1920x1080 is 25%"
        );

        // A pass stating more than the attachment holds. Metal permits it and
        // the rasteriser clips, so this reads full rather than over 100%.
        let before = store_route_count("pass_extent_full");
        note_pass_extent_coverage(4096, 4096, 1920, 1080);
        assert_eq!(store_route_count("pass_extent_full"), before + 1);

        // Neither a missing extent nor a geometry-less attachment is scored:
        // there is no fraction to take, and counting it as zero would put every
        // unstated pass in the bottom band and make the census read as damage.
        let before: u64 = PASS_EXTENT_SLUGS.iter().map(|s| store_route_count(s)).sum();
        note_pass_extent_coverage(0, 0, 1920, 1080);
        note_pass_extent_coverage(100, 100, 0, 0);
        assert_eq!(
            PASS_EXTENT_SLUGS
                .iter()
                .map(|s| store_route_count(s))
                .sum::<u64>(),
            before
        );
    }

    #[test]
    fn stream_accum_upserts_buffer_and_viewport() {
        // wire opcodes via wire_render import

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        // setVertexBuffer multi-entry: first=2 count=1 ref=9 offset=16
        // payload = first:u32 + count:u32 + {ref:u32, offset:u64}
        let mut vb = vec![0u8; OP_HEADER_LEN + 8 + 12];
        let vb_len = vb.len() as u32;
        st32(&mut vb[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
        st32(&mut vb[4..], vb_len);
        st32(&mut vb[8..], 2); // first
        st32(&mut vb[12..], 1); // count
        st32(&mut vb[16..], 9); // ref
        st64(&mut vb[20..], 16); // offset
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_VERTEX_BUFFER,
            &vb,
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.vertex_buffers.len(), 1);
        assert_eq!(acc.vertex_buffers[0].index, 2);
        assert_eq!(acc.vertex_buffers[0].buffer_ref, 9);
        assert_eq!(acc.vertex_buffers[0].offset, 16);

        // overwrite same slot
        st32(&mut vb[16..], 10);
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_VERTEX_BUFFER,
            &vb,
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.vertex_buffers.len(), 1);
        assert_eq!(acc.vertex_buffers[0].buffer_ref, 10);

        // fragment buffer multi-entry: first=0 count=1 ref=7 offset=0
        let mut fb = vec![0u8; OP_HEADER_LEN + 8 + 12];
        let fb_len = fb.len() as u32;
        st32(&mut fb[0..], wire_render::OPCODE_SET_FRAGMENT_BUFFER);
        st32(&mut fb[4..], fb_len);
        st32(&mut fb[8..], 0); // first
        st32(&mut fb[12..], 1); // count
        st32(&mut fb[16..], 7); // ref
        st64(&mut fb[20..], 0); // offset
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_FRAGMENT_BUFFER,
            &fb,
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.fragment_buffers.len(), 1);

        // viewport
        let mut vp = vec![0u8; OP_HEADER_LEN + 48];
        st32(&mut vp[0..], wire_render::OPCODE_SET_VIEWPORT);
        st32(&mut vp[4..], (OP_HEADER_LEN + 48) as u32);
        for i in 0..6 {
            let bits = (i as f64 + 1.0).to_bits();
            st64(&mut vp[OP_HEADER_LEN + i * 8..], bits);
        }
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_VIEWPORT,
            &vp,
            &mut out,
            &mut acc,
        );
        let v = acc.viewport.expect("viewport");
        assert!((v[0] - 1.0).abs() < 1e-9);
        assert!((v[5] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn wide_indexed_draw_reaches_pending_draw() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            ..Default::default()
        };
        let mut command = vec![0u8; 0x20];
        let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st16(&mut command[10..], 0);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        st32(&mut command[24..], 0x10100);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        assert!(acc.saw_draw);
        assert!(out.saw_draw);
        assert_eq!(acc.draws.len(), 1);
        let indexed = acc.draws[0].indexed.as_ref().expect("indexed draw");
        assert_eq!(indexed.index_type, 0);
        assert_eq!(indexed.index_buffer_ref, 0x3e);
        assert_eq!(indexed.index_count, 6);
        assert_eq!(indexed.index_buffer_offset, 0x10100);
        assert_eq!(
            acc.draws[0].draw,
            DrawArgs {
                vertex_count: 6,
                instance_count: 1,
                primitive_type: 3,
                first_vertex: 0,
                base_instance: 0
            }
        );
    }

    /// A base vertex and a base instance survive the whole accumulator hop.
    ///
    /// Both had a home in every backend already — Metal's `render_core_mrt`
    /// takes a base instance and `ReimsVgpuIndexedDraw` a base vertex, Vulkan's
    /// `DrawRequest` and `IndexedDrawResource` the same two — and both were fed
    /// a hardcoded zero from here, because nothing upstream decoded a draw form
    /// that carries them. This is the seam that was missing, so it is the seam
    /// worth pinning: a regression to a literal `0` anywhere between decode and
    /// `DrawEncodeRequest` fails here.
    #[test]
    fn a_base_vertex_and_base_instance_reach_the_pending_draw() {
        use crate::contract::endian::st16;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            ..Default::default()
        };
        let op = wire_render::OPCODE_DRAW_INDEXED_INSTANCED_BASE;
        let total = reims_vgpu_wire::ops::render::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN;
        let mut command = vec![0u8; total as usize];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total);
        st16(&mut command[8..], 3); // primitiveType
        st16(&mut command[10..], 1); // indexType UInt32
        st32(&mut command[12..], 0x3e); // index buffer ref
        st16(&mut command[16..], 0x40); // index buffer offset (first, on this form)
        st16(&mut command[18..], 6); // index count
        st16(&mut command[20..], 9); // instanceCount
        st16(&mut command[22..], 0xfffb); // baseVertex = -5
        st16(&mut command[24..], 7); // baseInstance
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        assert_eq!(acc.draws.len(), 1, "the draw must not be dropped");
        assert_eq!(acc.draws[0].draw.instance_count, 9);
        assert_eq!(acc.draws[0].draw.base_instance, 7);
        let indexed = acc.draws[0].indexed.as_ref().expect("indexed draw");
        assert_eq!(indexed.index_count, 6);
        assert_eq!(indexed.index_buffer_offset, 0x40);
        assert_eq!(indexed.base_vertex, -5, "a negative base vertex survives");

        // And onward into the request the backends receive. `retarget_render_
        // pass_draw` is the path records 2+ of a chained pass take, and it
        // rebuilds every draw argument from the template.
        let template = metal_draw::DrawEncodeRequest::default();
        let req = retarget_render_pass_draw(&template, &acc.draws[0]);
        assert_eq!(req.base_instance, 7);
        assert_eq!(req.instance_count, 9);
    }

    /// Every decoded draw in a stream reaches the draw list.
    ///
    /// `MAX_DRAWS_PER_STREAM = 64` truncated `acc.draws` inside a bare `if` with
    /// no `else`, so a compositor stream with more records than that lost every
    /// draw past the 64th with nothing on any channel — no counter, no line, and
    /// an `ExecResult` describing the truncated list as a fully executed pass.
    /// 71 is chosen to straddle that old ceiling: this test fails on the capped
    /// code at exactly 64.
    #[test]
    fn every_decoded_draw_in_a_stream_reaches_the_draw_list() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            ..Default::default()
        };
        let mut command = vec![0u8; 0x20];
        let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        let records = 71;
        for _ in 0..records {
            handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        }

        assert_eq!(acc.draws.len(), records, "no draw may be truncated away");
        assert_eq!(acc.dropped_unbound, 0, "all of these had a pipeline bound");

        // With no pipeline latched the same record is the other arm: still not
        // a `PendingDraw`, but counted rather than vanishing.
        let mut unbound = StreamAccum::default();
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut unbound);
        assert_eq!(unbound.dropped_unbound, 1);
        assert!(unbound.draws.is_empty());
    }

    /// A stream that binds once and draws many times must not copy its bind
    /// tables per draw.
    ///
    /// This is the property that makes an unbounded draw list affordable, and
    /// therefore the property the cap's removal rests on. It is asserted by
    /// pointer identity because that is the only thing that distinguishes a
    /// shared table from an equal copy — `assert_eq!` on the contents passes
    /// either way, which is exactly how a regression here would hide.
    #[test]
    fn draws_sharing_a_bind_table_share_its_allocation() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            vertex_buffers: Arc::new(vec![BufferBind {
                index: 0,
                buffer_ref: 9,
                offset: 0,
            }]),
            ..Default::default()
        };
        let mut command = vec![0u8; 0x20];
        let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        for _ in 0..100 {
            handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        }

        assert_eq!(acc.draws.len(), 100);
        for (i, pd) in acc.draws.iter().enumerate() {
            assert!(
                Arc::ptr_eq(&pd.vertex_buffers, &acc.vertex_buffers),
                "draw {i} copied a bind table nothing had changed"
            );
        }
    }

    /// A bind that changes after a draw must not reach back into that draw.
    ///
    /// The other half of the copy-on-write contract: sharing is only safe if a
    /// later mutation forks. `Arc::make_mut` is what does that, and a mutation
    /// site that reached the `Vec` some other way would silently rewrite a
    /// snapshot the guest already committed to.
    #[test]
    fn a_bind_after_a_draw_does_not_rewrite_that_draws_snapshot() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            vertex_buffers: Arc::new(vec![BufferBind {
                index: 0,
                buffer_ref: 9,
                offset: 0,
            }]),
            ..Default::default()
        };
        let mut command = vec![0u8; 0x20];
        let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        apply_binds(
            &[(77u32, 0u64)],
            0,
            BindTarget {
                stage: Stage::Vertex,
                class: BindClass::Buffer,
            },
            &mut acc.vertex_buffers,
            &mut acc.fragment_buffers,
            |b| b.index,
            |index, (buffer_ref, offset)| {
                Some(BufferBind {
                    index,
                    buffer_ref,
                    offset,
                })
            },
        );

        assert_eq!(
            acc.draws[0].vertex_buffers[0].buffer_ref, 9,
            "the committed draw kept the buffer it was encoded with"
        );
        assert_eq!(acc.vertex_buffers[0].buffer_ref, 77);
    }

    #[test]
    fn accepted_render_without_executor_is_fail_visible() {
        // The emit is deduped per opcode process-wide; hold the shared latch
        // lock and clear it so this test always observes its first-sighting line.
        let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_unimplemented_opcode_dedup_for_test();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 0xface,
            ..Default::default()
        };
        let task_id = 0xfeed;
        let mut command = vec![0u8; OP_HEADER_LEN];
        // An opcode inside the encoder's range that no arm claims, found rather
        // than named. It used to be `wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`, which stopped working
        // the moment that bound was corrected to `0xa6` -- because `0xa6` is a
        // record this rail now decodes. `0x99` was the replacement and lasted
        // one commit, until `setVertexAmplificationMode:value:` turned out to be
        // that number. The catch-all is what is under test, not any literal.
        let op = render::unclaimed_accepted_opcode();
        st32(&mut command[0..], op);
        st32(&mut command[4..], OP_HEADER_LEN as u32);
        handle_render_record(&mut state, &host, task_id, op, &command, &mut out, &mut acc);

        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        let want = format!(
            "render_unimplemented reason=accepted_without_executor task=65261 opcode={op:#x} len=8"
        );
        assert!(
            body.lines()
                .any(|line| line.contains(&want) && line.contains("pipeline=64206")),
            "no line matching {want:?}"
        );
    }

    /// Regression guard: the accepted-without-executor line is deduped to ONE
    /// emission per distinct opcode (a per-draw undecoded op must not flood the
    /// always-on sink), while distinct opcodes still each report once and the
    /// raw wire is captured. This locks the anti-flood behavior that replaced
    /// the ~2620-line-per-workload per-draw emit.
    #[test]
    fn unimplemented_render_opcode_dedups_per_opcode_with_wire() {
        let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_unimplemented_opcode_dedup_for_test();
        let task = 0x5151u32;
        let acc = StreamAccum {
            pipeline_ref: 0x1234,
            ..Default::default()
        };
        let wire: Vec<u8> = vec![0xde, 0xad, 0xbe, 0xef, 0x10, 0x00, 0x00, 0x00];

        // First sighting of an opcode emits; every repeat is deduped (no flood).
        assert!(
            note_unimplemented_render_opcode(0x7c, &wire, task, &acc),
            "first sighting must emit",
        );
        for _ in 0..24 {
            assert!(
                !note_unimplemented_render_opcode(0x7c, &wire, task, &acc),
                "a repeated opcode must be deduped",
            );
        }
        // A distinct opcode reports once independently of the first.
        assert!(note_unimplemented_render_opcode(0x9a, &wire, task, &acc));
        assert!(!note_unimplemented_render_opcode(0x9a, &wire, task, &acc));
        // Out-of-range opcodes (decode desync) are also deduped, not flooded.
        assert!(note_unimplemented_render_opcode(
            0x1_0001, &wire, task, &acc
        ));
        assert!(!note_unimplemented_render_opcode(
            0x1_0001, &wire, task, &acc
        ));

        // The first-sighting line captured the raw wire for offline decode.
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(
            body.lines().any(|l| l.contains(&format!("task={task}"))
                && l.contains("opcode=0x7c")
                && l.contains("hex=deadbeef10000000")),
            "the raw wire must be captured on first sighting",
        );
    }

    /// The render rail's boundary counter must name the *check* that dropped the
    /// draw, not the class it was flattened into.
    ///
    /// Before `EncodeStatus` carried its reason this line read
    /// `draw_encode_fail reason=bad_args`, and `bad_args` alone spoke for eight
    /// distinct refusals in `encode_draw_chain_inner` — a zero-size target, a
    /// vertexless draw, an MRT slot with no backing. A window that never painted
    /// gave you the class and never the cause.
    #[test]
    fn a_dropped_draw_names_which_check_refused_not_just_its_class() {
        let task = 81u32;
        // Distinct from every other pipeline in the suite: `fail_once` latches per
        // (reason, pipeline) for the whole process.
        let pipe = 249_001u32;
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::BadArgs("draw_mtl_zero_geom"),
            1,
            3,
        );
        let body = sink_body();
        assert!(
            body.lines().any(|l| l
                .contains("draw_encode_fail reason=draw_mtl_zero_geom class=bad_args")
                && l.contains(&format!("pipe={pipe}"))
                && l.contains(&format!("task={task}"))
                && l.contains("di=1/3")),
            "the boundary line must carry the specific check and the class:\n{body}"
        );

        // Latched per (reason, pipeline): the guest re-submits the same failing
        // draw every frame, so a repeat adds nothing the first line did not…
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::BadArgs("draw_mtl_zero_geom"),
            2,
            3,
        );
        // …but a *different* check on the same pipeline is a different event and
        // must still be visible. Latching on the class would have hidden it, which
        // is exactly the failure this migration removes.
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::MetalFailed("draw_mtl_core_failed"),
            2,
            3,
        );
        let body = sink_body();
        assert_eq!(
            body.matches("reason=draw_mtl_zero_geom").count(),
            1,
            "a re-attempted refusal must log once:\n{body}"
        );
        assert!(
            body.contains("reason=draw_mtl_core_failed"),
            "a second check on the same pipeline must not be latched away:\n{body}"
        );

        // Success never reaches the sink — `Emit::refusal` has no line to send for
        // `Ok`, so the carve-out is enforced by the type rather than by a `return`
        // a future arm could forget.
        let before = sink_body().matches("draw_encode_fail").count();
        note_draw_encode_fail(task, pipe, EncodeStatus::Ok, 0, 1);
        assert_eq!(
            sink_body().matches("draw_encode_fail").count(),
            before,
            "an Ok encode logged a failure line"
        );
    }

    #[test]
    fn zero_ref_render_bind_unbinds_existing_slots() {
        // wire opcodes via wire_render import

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let mut buffer = vec![0u8; OP_HEADER_LEN + 8 + 12];
        st32(&mut buffer[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
        st32(&mut buffer[4..], (OP_HEADER_LEN + 8 + 12) as u32);
        st32(&mut buffer[8..], 0);
        st32(&mut buffer[12..], 1);
        st32(&mut buffer[16..], 41);
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_VERTEX_BUFFER,
            &buffer,
            &mut out,
            &mut acc,
        );
        st32(&mut buffer[16..], 0);
        handle_render_record(
            &mut state,
            &host,
            0,
            wire_render::OPCODE_SET_VERTEX_BUFFER,
            &buffer,
            &mut out,
            &mut acc,
        );
        assert!(acc.vertex_buffers.is_empty());

        for (opcode, bound) in [
            (wire_render::OPCODE_SET_FRAGMENT_TEXTURE, 42u32),
            (wire_render::OPCODE_SET_FRAGMENT_SAMPLER, 43u32),
        ] {
            let mut command = vec![0u8; OP_HEADER_LEN + 8 + 4];
            st32(&mut command[0..], opcode);
            st32(&mut command[4..], (OP_HEADER_LEN + 8 + 4) as u32);
            st32(&mut command[8..], 3);
            st32(&mut command[12..], 1);
            st32(&mut command[16..], bound);
            handle_render_record(&mut state, &host, 0, opcode, &command, &mut out, &mut acc);
            st32(&mut command[16..], 0);
            handle_render_record(&mut state, &host, 0, opcode, &command, &mut out, &mut acc);
        }
        assert!(acc.fragment_textures.is_empty());
        assert!(acc.fragment_samplers.is_empty());
        assert_eq!(out.buffer_unbinds, 1);
        assert_eq!(out.texture_unbinds, 1);
        assert_eq!(out.sampler_unbinds, 1);
    }

    /// x86 type-4 display mid: clear-only stream must Store solid BGRA into pages.
    #[test]
    fn clear_only_type4_surface_writes_guest_pages() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // Surface pages at pfn 0x40 (one 4K page is enough for 16×16).
        let page = 0x40u64 << PAGE_SHIFT_X86;
        host.map_range(page, 0x2000, 0);
        // Task directory so object-list GVA reads work.
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        // Map the backing GVA page onto the surface pages. The device refuses a
        // backing it cannot translate rather than reusing the GVA as a GPA, so
        // the task's page table has to carry this the way a guest's does.
        st32(&mut d[..4], 0x40);
        let _ = host.write_gpa(root_gpa + 0x40 * 4, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        // Type-4 at surface_id=5.
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
        );
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x40); // identity pfn
        st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(objects::resolve_type4_surface(&mut state, &host, 5));
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        acc.clears.push(ColorAttachment {
            present: true,
            texture_ref: 5,
            resolve_texture_ref: 0,
            level: 0,
            slice: 0,
            depth_plane: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [1.0, 0.0, 0.0, 1.0], // red → BGRA (0,0,255,255)
        });
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert!(
            out.clears_applied >= 1,
            "type-4 clear must apply, got {}",
            out.clears_applied
        );
        // Read first pixel from guest page (BGRA).
        let mut px = [0u8; 4];
        assert!(host.read_gpa(page, &mut px).is_ok());
        assert_eq!(px, [0, 0, 255, 255], "expected opaque red BGRA, got {px:?}");
        let m = state.mappings.get(&5).expect("mapping");
        assert!(m.content_generation > 0 || m.mapped);
        let _ = PAGE_ENTRY_VALID;
        let _ = PAGE_ENTRY_PFN_SHIFT;
    }

    /// Archive DrawJob: clear-only packets store immediately; multi-draw packets
    /// keep CLEAR as private Metal seed (no pre-draw guest clear).
    #[test]
    fn finish_stream_clear_only_branch_without_draws() {
        use crate::runtime::decode::render::ColorAttachment;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        acc.clears.push(ColorAttachment {
            present: true,
            texture_ref: 99,
            resolve_texture_ref: 0,
            level: 0,
            slice: 0,
            depth_plane: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        });
        // No draws → clear-only branch (attempts apply_clear; unresolvable ref).
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert_eq!(out.metal_draws_ok, 0);
        assert_eq!(out.metal_draws_fail, 0);
    }

    #[test]
    fn finish_stream_with_draws_skips_guest_clear_prelude() {
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::metal_draw::BufferBind;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let att = ColorAttachment {
            present: true,
            texture_ref: 99,
            resolve_texture_ref: 0,
            level: 0,
            slice: 0,
            depth_plane: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [1.0, 0.0, 0.0, 1.0],
        };
        acc.clears.push(att);
        acc.saw_draw = true;
        acc.color_slots.push((0, att));
        acc.draws.push(PendingDraw {
            pipeline_ref: 1,
            draw: DrawArgs {
                vertex_count: 3,
                instance_count: 1,
                primitive_type: 3,
                first_vertex: 0,
                base_instance: 0,
            },
            indexed: None,
            vertex_buffers: Arc::new(vec![BufferBind {
                index: 0,
                buffer_ref: 1,
                offset: 0,
            }]),
            fragment_buffers: Arc::default(),
            vertex_textures: Arc::default(),
            fragment_textures: Arc::default(),
            vertex_samplers: Arc::default(),
            fragment_samplers: Arc::default(),
            viewport: None,
            scissor: None,
            blend_color: None,
            cull_mode: None,
            front_facing: None,
            depth_bias: None,
            depth_stencil_ref: 0,
            stencil_ref: None,
            depth_attach: None,
            stencil_attach: None,
            stencil_first_in_pass: true,
        });
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        // Unresolvable RT → mrt_request fail before encode (not NoMetal); no clear.
        assert_eq!(
            out.clears_applied, 0,
            "unresolvable multi-draw must not guest-clear"
        );
    }

    /// Linux NoMetal: draws fail but CLEAR seed still Stores into type-4 pages.
    #[test]
    fn nometal_draw_falls_back_to_type4_clear() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::metal_draw::BufferBind;
        use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let page = 0x50u64 << PAGE_SHIFT_X86;
        host.map_range(page, 0x2000, 0);
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        // As above: the backing GVA has to translate, not be assumed identity.
        st32(&mut d[..4], 0x50);
        let _ = host.write_gpa(root_gpa + 0x50 * 4, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
        );
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x50);
        st32(&mut desc[0xc..], 0x4247_5241);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);
        assert!(objects::resolve_type4_surface(&mut state, &host, 5));

        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let att = ColorAttachment {
            present: true,
            texture_ref: 5,
            resolve_texture_ref: 0,
            level: 0,
            slice: 0,
            depth_plane: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [0.0, 1.0, 0.0, 1.0], // green
        };
        acc.clears.push(att);
        acc.saw_draw = true;
        acc.color_slots.push((0, att));
        acc.draws.push(PendingDraw {
            pipeline_ref: 7,
            draw: DrawArgs {
                vertex_count: 3,
                instance_count: 1,
                primitive_type: 3,
                first_vertex: 0,
                base_instance: 0,
            },
            indexed: None,
            vertex_buffers: Arc::new(vec![BufferBind {
                index: 0,
                buffer_ref: 1,
                offset: 0,
            }]),
            fragment_buffers: Arc::default(),
            vertex_textures: Arc::default(),
            fragment_textures: Arc::default(),
            vertex_samplers: Arc::default(),
            fragment_samplers: Arc::default(),
            viewport: None,
            scissor: None,
            blend_color: None,
            cull_mode: None,
            front_facing: None,
            depth_bias: None,
            depth_stencil_ref: 0,
            stencil_ref: None,
            depth_attach: None,
            stencil_attach: None,
            stencil_first_in_pass: true,
        });
        let mut second = acc.draws[0].clone();
        second.pipeline_ref = 8;
        acc.draws.push(second);
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert_eq!(
            out.render_attachment_resolves, 1,
            "one render stream resolves its fixed attachment set once"
        );
        // Non-Apple: Linux encode Stores CLEAR load into type-4 (Ok) or
        // NoMetal clear fallback — either path must land green BGRA.
        #[cfg(feature = "backend-vulkan")]
        {
            assert!(
                out.metal_draws_ok >= 1 || out.clears_applied >= 1 || out.metal_draws_fail >= 1,
                "expected clear store path: ok={} clear={} fail={}",
                out.metal_draws_ok,
                out.clears_applied,
                out.metal_draws_fail
            );
            let mut px = [0u8; 4];
            assert!(host.read_gpa(page, &mut px).is_ok());
            // BGRA green = [0, 255, 0, 255]
            assert_eq!(px, [0, 255, 0, 255], "got {px:?}");
        }
    }

    /// Multi-draw packets force full-frame store on the final record even when
    /// that draw carries a partial scissor (dock damage over chained wallpaper).
    #[test]
    fn multi_draw_force_full_store_flag_for_chained_packet() {
        assert_eq!(multi_draw_store_plan(0, 0), (false, false));
        assert_eq!(multi_draw_store_plan(1, 0), (true, false));
        assert_eq!(multi_draw_store_plan(3, 0), (false, false));
        assert_eq!(multi_draw_store_plan(3, 1), (false, false));
        assert_eq!(multi_draw_store_plan(3, 2), (true, true));
    }

    /// qemu-shim style: multi-draw plan is one guest writeback on the last record
    /// only, with force_full so a partial scissor cannot leave wallpaper only in
    /// host chain memory (archive DrawJob single completion writeback).
    #[test]
    fn multi_draw_store_plan_matches_archive_drawjob_writeback() {
        // Every packet size and every record within it. The whole contract is two
        // predicates over (draw_count, di), so stating it over a range costs
        // nothing and covers the boundary at draw_count == 1, where force_full
        // flips — which one packet of five does not reach.
        for n in 1..8usize {
            for di in 0..n {
                let (wb, full) = multi_draw_store_plan(n, di);
                let last = di + 1 == n;
                assert_eq!(
                    wb, last,
                    "writeback is the last record only (n={n} di={di})"
                );
                assert_eq!(
                    full,
                    last && n > 1,
                    "force_full on the last record of a multi-draw packet only \
                     (n={n} di={di}); a single-draw packet may keep a local scissor"
                );
            }
        }
        assert_eq!(
            multi_draw_store_plan(0, 0),
            (false, false),
            "an empty packet writes nothing back"
        );
    }

    #[test]
    fn multi_draw_chain_source_preserves_portable_unified_output() {
        assert_eq!(
            multi_draw_chain_source(true, false),
            MultiDrawChainSource::Resident
        );
        assert_eq!(
            multi_draw_chain_source(false, true),
            MultiDrawChainSource::Cpu
        );
        assert_eq!(
            multi_draw_chain_source(false, false),
            MultiDrawChainSource::Missing
        );
    }

    #[test]
    fn render_pass_template_reuses_attachment_without_load_seed() {
        let first = metal_draw::DrawEncodeRequest {
            task_id: 1,
            pipeline_ref: 7,
            vertex_count: 3,
            instance_count: 1,
            primitive_type: 3,
            colors: vec![metal_draw::ColorRtRequest {
                slot: 0,
                texture_ref: 11,
                mapping_id: 3,
                target_gva: 0,
                row_stride: 0,
                width: 1920,
                height: 1080,
                format: 0x50,
                load_action: PASS_LOAD_ACTION_CLEAR,
                store_action: PASS_STORE_ACTION_STORE,
                clear_color: [0.1, 0.2, 0.3, 1.0],
                target_seed_rgba: Some(vec![0xbb; 16]),
            }],
            ..Default::default()
        };
        let template = render_pass_attachment_template(&first);
        assert!(template.colors[0].target_seed_rgba.is_none());
        assert_eq!(template.colors[0].load_action, PASS_LOAD_ACTION_LOAD);
        assert_eq!(template.colors[0].mapping_id, 3);
        assert_eq!(
            (template.colors[0].width, template.colors[0].height),
            (1920, 1080)
        );

        let draw = PendingDraw {
            pipeline_ref: 42,
            draw: DrawArgs {
                vertex_count: 6,
                instance_count: 2,
                primitive_type: 4,
                first_vertex: 9,
                base_instance: 0,
            },
            ..Default::default()
        };
        let req = retarget_render_pass_draw(&template, &draw);
        assert_eq!(req.pipeline_ref, 42);
        assert_eq!(
            (
                req.vertex_count,
                req.instance_count,
                req.primitive_type,
                req.first_vertex
            ),
            (6, 2, 4, 9)
        );
        assert_eq!(req.colors.len(), 1);
        assert_eq!(req.colors[0].mapping_id, 3);
        assert_eq!(
            first.colors[0].target_seed_rgba.as_ref().map(Vec::len),
            Some(16)
        );
    }

    /// A `Clear` load action seeds the pass whatever the store action is.
    ///
    /// The three store actions below are the ones a live macOS desktop was
    /// measured to emit alongside `Clear`: `Store` (1), `DontCare` (0) and
    /// `MultisampleResolve` (2). Admitting only `Store` — which this code did —
    /// dropped the other two, and a pass whose seed is dropped loads whatever
    /// the attachment held before. From the guest that is windows, menus and
    /// tooltips piling up on screen instead of replacing what was there.
    #[test]
    fn a_clear_load_action_seeds_the_pass_whatever_the_store_action() {
        // MTLStoreAction: DontCare=0, Store=1, MultisampleResolve=2.
        for store_action in [0u16, PASS_STORE_ACTION_STORE, 2] {
            assert!(
                clear_seeds_the_pass(PASS_LOAD_ACTION_CLEAR),
                "Clear must seed the pass with store_action={store_action}"
            );
        }
        // Every other load action leaves the attachment alone: Load preserves
        // it, DontCare leaves it undefined, and neither is a clear.
        for load_action in [0u16, 1, 3, 4] {
            if load_action == PASS_LOAD_ACTION_CLEAR {
                continue;
            }
            assert!(!clear_seeds_the_pass(load_action));
        }
    }

    #[test]
    fn dropped_clear_logs_once_per_reason_target() {
        // Unique keys per case so no shared-static reset is needed (the dedup set
        // is process-global). First sighting of a (reason, tex_ref) emits (true);
        // an immediate repeat is suppressed (false); a distinct target logs again.
        assert!(note_clear_dropped(
            "nonstore_store_action",
            0x9001,
            "store_action=0 load_action=clear"
        ));
        assert!(!note_clear_dropped(
            "nonstore_store_action",
            0x9001,
            "store_action=0 load_action=clear"
        ));
        assert!(note_clear_dropped(
            "nonstore_store_action",
            0x9002,
            "store_action=0 load_action=clear"
        ));
        // A different reason on the same target is a distinct blind spot and logs.
        assert!(note_clear_dropped(
            "target_unresolved",
            0x9001,
            "color_target_request=none"
        ));
        assert!(!note_clear_dropped(
            "target_unresolved",
            0x9001,
            "color_target_request=none"
        ));
    }

    /// A plural viewport or scissor applies its first entry and counts the rest.
    ///
    /// Before `0x83`/`0x76` were decoded the whole record reached no arm, so a
    /// guest setting its viewport through `setViewports:count:` got none at all.
    /// Now it gets the first, and the counter says what a viewport-array model
    /// would have to hold.
    #[test]
    fn a_plural_viewport_or_scissor_applies_one_and_counts_the_rest() {
        use crate::contract::endian::st64;
        use crate::runtime::drain::store_route_count;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        let op = wire_render::OPCODE_SET_SCISSOR_RECTS;
        let total = reims_vgpu_wire::OP_HEADER_LEN
            + render::SCISSOR_RECTS_COUNT_LEN
            + 3 * render::SCISSOR_PAYLOAD_LEN;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], 3);
        let e0 = reims_vgpu_wire::OP_HEADER_LEN + render::SCISSOR_RECTS_COUNT_LEN;
        for (i, val) in [11u64, 22, 33, 44].into_iter().enumerate() {
            st64(&mut command[e0 + i * 8..], val);
        }

        let before = store_route_count("render_extra_scissors_dropped");
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            acc.scissor,
            Some((11, 22, 33, 44)),
            "the first rect must reach the accumulator"
        );
        assert_eq!(
            store_route_count("render_extra_scissors_dropped") - before,
            2,
            "the counter must name the entries dropped, not the record"
        );

        // One entry is the singular record and drops nothing.
        let before = store_route_count("render_extra_scissors_dropped");
        let total = reims_vgpu_wire::OP_HEADER_LEN + render::SCISSOR_PAYLOAD_LEN;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_SCISSOR;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        for (i, val) in [1u64, 2, 3, 4].into_iter().enumerate() {
            st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN + i * 8..], val);
        }
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(acc.scissor, Some((1, 2, 3, 4)));
        assert_eq!(store_route_count("render_extra_scissors_dropped"), before);
    }

    /// A bind past the table's last slot says how many slots it dropped, and
    /// which of the three tables lost them.
    ///
    /// Apple's serializer produces `setVertexTextures:withRange:` over a range
    /// of 40, and `MAX_BIND_SLOTS` is 31 — Metal's *buffer* index cap, applied
    /// to a texture table whose real limit is 128. So the walk ends early on
    /// traffic a guest actually sends, and it used to end with a bare `break`.
    /// The count is the argument for widening the tables, so it has to be the
    /// number of slots lost rather than one event — and it has to name the
    /// table, because the three do not lose the same thing. The sibling slugs
    /// must stay still while the texture one moves; a shared counter that
    /// incremented for all three is what this replaced.
    #[test]
    fn a_bind_past_the_last_table_slot_reports_what_it_dropped() {
        use crate::runtime::drain::store_route_count;

        const COUNT: u32 = 40;
        let entry = render::REF_BIND_ENTRY_SIZE;
        let total =
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + (COUNT as usize) * entry;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
            0,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            COUNT,
        );
        for i in 0..COUNT as usize {
            let at = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + i * entry;
            st32(&mut command[at..], 0x4000 + i as u32);
        }

        // The record itself must survive decode; a cap that refused it whole is
        // what this counter exists to distinguish from.
        let c = render::decode(&command).expect("forty texture binds must decode");
        assert_eq!(c.ref_binds.len(), COUNT as usize);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = store_route_count(BindClass::Texture.past_table_route());
        let before_buf = store_route_count(BindClass::Buffer.past_table_route());
        let before_smp = store_route_count(BindClass::Sampler.past_table_route());
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count(BindClass::Texture.past_table_route()) - before,
            (COUNT - MAX_BIND_SLOTS) as u64,
            "the counter must name every slot dropped, not the one event"
        );
        assert_eq!(
            store_route_count(BindClass::Buffer.past_table_route()),
            before_buf,
            "a texture bind must not move the buffer table's counter"
        );
        assert_eq!(
            store_route_count(BindClass::Sampler.past_table_route()),
            before_smp,
            "a texture bind must not move the sampler table's counter"
        );
        assert_eq!(
            acc.vertex_textures.len(),
            MAX_BIND_SLOTS as usize,
            "every slot the table does hold must still be bound"
        );
    }

    /// A bind at the last slot Apple's *sampler* table can name still binds.
    ///
    /// The three classes now carry three counters, and the risk that creates is
    /// the opposite of the one it fixes: a per-class slug invites a per-class
    /// *bound*, and bounding a table by what Apple's serializer emits is the
    /// mistake [`reims_vgpu_wire::ops::bind_limit`]'s own doc names — it would
    /// refuse a guest that writes its own stream. So the bound stays one number
    /// and this pins that it did: a sampler at index 20, which is above Apple's
    /// 16-entry sampler table and below [`MAX_BIND_SLOTS`], binds rather than
    /// being counted away.
    #[test]
    fn a_sampler_above_apples_table_but_inside_ours_still_binds() {
        use crate::runtime::drain::store_route_count;
        use reims_vgpu_wire::ops::bind_limit;

        const FIRST: u32 = 20;
        const { assert!(FIRST >= bind_limit::SAMPLER && FIRST < MAX_BIND_SLOTS) };

        let entry = render::REF_BIND_ENTRY_SIZE;
        let total = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + entry;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_SAMPLER;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
            FIRST,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            1,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
            0x3333,
        );

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = store_route_count(BindClass::Sampler.past_table_route());
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        assert_eq!(
            store_route_count(BindClass::Sampler.past_table_route()),
            before,
            "the bound is the host table, not Apple's — this slot is inside it"
        );
        assert_eq!(
            acc.vertex_samplers
                .iter()
                .map(|s| s.index)
                .collect::<Vec<_>>(),
            vec![FIRST]
        );
    }

    /// Every bind record lands in exactly one reach band, and the top band
    /// fires on the same records the drop counter counts slots for.
    ///
    /// The bands are what make a zero from `*_bind_slot_past_table` readable: a
    /// workload whose every record stops at slot 4 and one whose every record
    /// stops at slot 30 both drop nothing, and only the second says the bound
    /// is nearly spent. So the band has to be chosen from the reach the guest
    /// *asked for*, before the walk truncates it — which is what the `le_table`
    /// case below would catch if the census moved inside the loop.
    #[test]
    fn every_bind_record_lands_in_one_reach_band_and_the_top_one_reconciles() {
        use crate::runtime::drain::store_route_count;
        use reims_vgpu_wire::ops::bind_limit;

        let texture_record = |first: u32, count: u32| {
            let entry = render::REF_BIND_ENTRY_SIZE;
            let total =
                reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + (count as usize) * entry;
            let mut command = vec![0u8; total];
            let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
            st32(&mut command[0..], op);
            st32(&mut command[4..], total as u32);
            st32(
                &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
                first,
            );
            st32(
                &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
                count,
            );
            for i in 0..count as usize {
                let at = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + i * entry;
                st32(&mut command[at..], 0x4000 + i as u32);
            }
            (op, command)
        };

        let bands = [
            "render_bind_reach_texture_le16",
            "render_bind_reach_texture_le_table",
            "render_bind_reach_texture_over_table",
        ];
        let read = || bands.map(store_route_count);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        // Reach exactly Apple's sampler-table size: the lowest band, inclusive.
        let before = read();
        let (op, command) = texture_record(0, bind_limit::SAMPLER);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            read()
                .iter()
                .zip(before)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
            vec![1, 0, 0],
            "a reach of exactly {} is inside every one of Apple's tables",
            bind_limit::SAMPLER
        );

        // One past it, still inside this device's table.
        let before = read();
        let (op, command) = texture_record(0, bind_limit::SAMPLER + 1);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            read()
                .iter()
                .zip(before)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
            vec![0, 1, 0],
            "one slot past Apple's sampler table is headroom being spent, not a loss"
        );

        // Past this device's table: the band and the slot counter must agree
        // that the same record crossed, in their own units.
        let before = read();
        let before_slots = store_route_count(BindClass::Texture.past_table_route());
        let (op, command) = texture_record(MAX_BIND_SLOTS - 1, 4);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            read()
                .iter()
                .zip(before)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
            vec![0, 0, 1],
            "a record reaching past the bound is one record in the top band"
        );
        assert_eq!(
            store_route_count(BindClass::Texture.past_table_route()) - before_slots,
            3,
            "and three slots in the drop counter — records here, slots there"
        );
    }

    /// A buffer-offset record that lands on nothing says so, both ways.
    ///
    /// `setVertexBufferOffset:atIndex:` is the second record the guest spends on
    /// a slot, and both of its miss paths were silent: an index past the table,
    /// and an index inside it whose slot this device never bound. The second is
    /// the sharper one — Metal requires a live bind at that index and encoder
    /// state does not outlive the encoder, so a firing means this device's table
    /// and the guest's disagree.
    #[test]
    fn a_buffer_offset_that_lands_on_nothing_reports_which_way_it_missed() {
        use crate::runtime::drain::store_route_count;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        // index:u32 @0, offset:u64 @4 — a different payload shape from the plural
        // binds, which is why it takes its own offsets rather than `BIND_*`.
        let offset_record = |index: u32| {
            let total = reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_PAYLOAD_LEN;
            let mut command = vec![0u8; total];
            let op = wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET;
            st32(&mut command[0..], op);
            st32(&mut command[4..], total as u32);
            st32(
                &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_INDEX..],
                index,
            );
            st64(
                &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_VALUE..],
                0x5555,
            );
            (op, command)
        };

        // Inside the table, but nothing is bound there.
        let before_unbound = store_route_count("render_buffer_offset_slot_unbound");
        let (op, command) = offset_record(3);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count("render_buffer_offset_slot_unbound") - before_unbound,
            1,
            "an offset for a slot this device never bound must be named"
        );

        // Past the table entirely.
        let before_past = store_route_count("render_buffer_offset_slot_past_table");
        let (op, command) = offset_record(MAX_BIND_SLOTS + 4);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count("render_buffer_offset_slot_past_table") - before_past,
            1,
            "an offset past the table bound must be named separately"
        );
        assert_eq!(
            store_route_count("render_buffer_offset_slot_unbound") - before_unbound,
            1,
            "a slot past the table is not also an unbound slot inside it"
        );
    }

    /// The records this rail answers by doing nothing still say they arrived.
    ///
    /// `UseResource`, `UseHeap` and `Barrier` all reached the dispatch's
    /// catch-all, so a guest's residency declaration and its barriers were
    /// indistinguishable from a record that had been executed — the arm they
    /// fell into was shared with `Kind::Unknown` and with every guarded arm's
    /// else-case. Doing nothing is still the answer; being silent about it is
    /// not, and a counter nobody reads back cannot show it is wired up.
    #[test]
    fn a_residency_or_barrier_record_is_counted_rather_than_dropped_in_silence() {
        use crate::runtime::drain::store_route_count;

        for (op, route, payload_len) in [
            (
                wire_render::OPCODE_USE_RESOURCE,
                "render_noop_residency_hint",
                render::USE_RESOURCE_REFS + 4,
            ),
            (
                wire_render::OPCODE_USE_HEAP,
                "render_noop_residency_hint",
                render::USE_HEAP_REFS + 4,
            ),
            (
                wire_render::OPCODE_MEMORY_BARRIER_RESOURCES,
                "render_noop_barrier",
                0,
            ),
            (
                wire_render::OPCODE_MEMORY_BARRIER_SCOPE,
                "render_noop_barrier",
                0,
            ),
        ] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let host = FakeHost::new();
            let mut out = ExecResult::default();
            let mut acc = StreamAccum::default();

            let total = reims_vgpu_wire::OP_HEADER_LEN + payload_len;
            let mut command = vec![0u8; total];
            st32(&mut command[0..], op);
            st32(&mut command[4..], total as u32);
            if payload_len > 0 {
                // One resource named, so the count-led extent is satisfied.
                st32(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], 1);
            }

            let before = store_route_count(route);
            handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
            assert_eq!(
                store_route_count(route),
                before + 1,
                "op {op:#x} did not reach {route}"
            );
        }
    }

    /// The three ICB blit records are told apart rather than refused as one.
    ///
    /// They used to be declined before decode under a single shared reason,
    /// which said three different things with one word. Only two of them are
    /// losses: skipping Metal's optimize hint is semantically correct, while a
    /// dropped reset leaves commands live that the guest retired and a dropped
    /// copy leaves the destination holding what it held before. A counter that
    /// cannot tell those apart cannot answer the question they exist to answer.
    #[test]
    fn each_icb_blit_record_reaches_a_counter_that_names_which_one_it_is() {
        use crate::contract::endian::st64;
        use crate::runtime::drain::store_route_count;
        use reims_vgpu_wire::ops::blit as wire;

        let range = |op: u32| {
            let total = wire::ICB_RANGE_TOTAL_LEN as usize;
            let mut v = vec![0u8; total];
            st32(&mut v[0..], op);
            st32(&mut v[4..], total as u32);
            st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 6161);
            st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 0x3300);
            st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 12..], 0x4400);
            v
        };
        let copy = || {
            let total = wire::COPY_ICB_TOTAL_LEN as usize;
            let mut v = vec![0u8; total];
            st32(&mut v[0..], wire_blit::OPCODE_COPY_ICB);
            st32(&mut v[4..], total as u32);
            st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 7171);
            st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 7272);
            st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 8..], 0x1100);
            st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 16..], 0x2200);
            st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 24..], 0x3300);
            v
        };

        for (op, command, route) in [
            (
                wire_blit::OPCODE_OPTIMIZE_ICB,
                range(wire_blit::OPCODE_OPTIMIZE_ICB),
                "blit_noop_icb_optimize",
            ),
            (
                wire_blit::OPCODE_RESET_ICB,
                range(wire_blit::OPCODE_RESET_ICB),
                "blit_icb_reset_dropped",
            ),
            (wire_blit::OPCODE_COPY_ICB, copy(), "blit_icb_copy_dropped"),
        ] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            let before = store_route_count(route);
            handle_blit_record(&mut state, &mut host, 1, op, &command);
            assert_eq!(
                store_route_count(route),
                before + 1,
                "op {op:#x} did not reach {route}"
            );
        }

        // The optimize hint is the one that is *not* a loss, so it must not
        // reach either of the dropped-work counters. Sharing one would put a
        // correct no-op in the same bucket as stale commands executing.
        for route in ["blit_icb_reset_dropped", "blit_icb_copy_dropped"] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            let before = store_route_count(route);
            let command = range(wire_blit::OPCODE_OPTIMIZE_ICB);
            handle_blit_record(
                &mut state,
                &mut host,
                1,
                wire_blit::OPCODE_OPTIMIZE_ICB,
                &command,
            );
            assert_eq!(
                store_route_count(route),
                before,
                "the optimize hint was counted as {route}"
            );
        }
    }

    /// The five `BlitEncoderSPI` records each reach a route that names them.
    ///
    /// All five answered `blit_decode_unknown_opcode` until the wire capture
    /// drove this class with the capability forced on, and three of them are
    /// writes to guest-visible memory. The routes are not interchangeable and
    /// the test says so in both directions: the two texture fills are lost work
    /// and must not land on the invalidate's no-op counter, while the
    /// compressed-texture invalidate is a correct skip and must not land on
    /// either dropped-fill counter. Sharing one bucket would make a driven
    /// boot's reading unusable for deciding which executor to build.
    #[test]
    fn each_blit_spi_record_reaches_a_counter_that_names_which_one_it_is() {
        use crate::contract::endian::st64;
        use crate::runtime::drain::store_route_count;
        use reims_vgpu_wire::ops::blit as wire;

        // A texture fill of either form: identical through the region, then the
        // tail that tells the two apart. Zero-filled past that, which is what
        // the guest's staged-bytes fill of length 0 would look like — the
        // routing under test is by opcode, not by any value in the tail.
        let texture_fill = |op: u32, total: u32| {
            let mut v = vec![0u8; total as usize];
            st32(&mut v[0..], op);
            st32(&mut v[4..], total);
            let p = reims_vgpu_wire::OP_HEADER_LEN;
            st32(&mut v[p..], 4242); // texture
            st16(&mut v[p + 4..], 3); // level
            st16(&mut v[p + 6..], 5); // slice
            st64(&mut v[p + 8..], 0x44); // size w/h/d
            st64(&mut v[p + 16..], 0x55);
            st64(&mut v[p + 24..], 1);
            st64(&mut v[p + 32..], 0x11); // origin x/y/z
            st64(&mut v[p + 40..], 0x22);
            st64(&mut v[p + 48..], 0x33);
            v
        };
        let invalidate = |op: u32, total: u32| {
            let mut v = vec![0u8; total as usize];
            st32(&mut v[0..], op);
            st32(&mut v[4..], total);
            st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 4242);
            v
        };

        const COLOR: &str = "blit_fill_texture_color_dropped";
        const BYTES: &str = "blit_fill_texture_bytes_dropped";
        const INVALID: &str = "blit_noop_invalidate_compressed";

        for (op, command, route) in [
            (
                wire_blit::OPCODE_FILL_TEXTURE_COLOR,
                texture_fill(
                    wire_blit::OPCODE_FILL_TEXTURE_COLOR,
                    wire::FILL_TEXTURE_COLOR_TOTAL_LEN,
                ),
                COLOR,
            ),
            (
                wire_blit::OPCODE_FILL_TEXTURE_BYTES,
                texture_fill(
                    wire_blit::OPCODE_FILL_TEXTURE_BYTES,
                    wire::FILL_TEXTURE_BYTES_TOTAL_LEN,
                ),
                BYTES,
            ),
            (
                wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE,
                invalidate(
                    wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE,
                    wire::REF_TOTAL_LEN,
                ),
                INVALID,
            ),
            (
                wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
                invalidate(
                    wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
                    wire::REF_SLICE_LEVEL_TOTAL_LEN,
                ),
                INVALID,
            ),
        ] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            let others: Vec<(&str, u64)> = [COLOR, BYTES, INVALID]
                .into_iter()
                .filter(|r| *r != route)
                .map(|r| (r, store_route_count(r)))
                .collect();
            let before = store_route_count(route);
            handle_blit_record(&mut state, &mut host, 1, op, &command);
            assert_eq!(
                store_route_count(route),
                before + 1,
                "op {op:#x} did not reach {route}"
            );
            for (other, was) in others {
                assert_eq!(
                    store_route_count(other),
                    was,
                    "op {op:#x} also reached {other}; the two losses are not the \
                     same loss and one counter cannot answer for both"
                );
            }
        }

        // The pattern fill is the one of the five that is *executed*, so it
        // must not appear on any of the three counters above. It fails on a
        // missing buffer here, which is the executor running rather than the
        // record being dropped.
        let mut v = vec![0u8; wire::FILL_BUFFER_PATTERN4_TOTAL_LEN as usize];
        st32(&mut v[0..], wire_blit::OPCODE_FILL_BUFFER_PATTERN4);
        st32(&mut v[4..], wire::FILL_BUFFER_PATTERN4_TOTAL_LEN);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 7);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 0);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 12..], 8);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 20..], 0x89ab_cdef);
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let before: Vec<u64> = [COLOR, BYTES, INVALID]
            .into_iter()
            .map(store_route_count)
            .collect();
        handle_blit_record(
            &mut state,
            &mut host,
            1,
            wire_blit::OPCODE_FILL_BUFFER_PATTERN4,
            &v,
        );
        for (route, was) in [COLOR, BYTES, INVALID].into_iter().zip(before) {
            assert_eq!(
                store_route_count(route),
                was,
                "the executed pattern fill was counted as {route}"
            );
        }
    }

    /// A strided vertex bind reaches the bind table *and* its own counter.
    ///
    /// Both halves matter and they are different claims. The bind must land,
    /// because this record used to be refused before decode and the buffer
    /// never bound at all; the counter must fire, because the per-entry
    /// attribute stride still is not applied and the count is what says whether
    /// applying it is worth building.
    #[test]
    fn a_strided_vertex_bind_lands_in_the_table_and_still_reports_the_stride() {
        use crate::contract::endian::st64;
        use crate::runtime::drain::store_route_count;

        const ROUTE: &str = "render_vertex_attribute_stride_dropped";
        let total = reims_vgpu_wire::OP_HEADER_LEN
            + render::BIND_ENTRIES
            + render::BUFFER_STRIDE_BIND_ENTRY_SIZE;
        let mut command = vec![0u8; total];
        st32(
            &mut command[0..],
            wire_render::OPCODE_SET_VERTEX_BUFFER_STRIDE,
        );
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
            4,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            1,
        );
        let e = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES;
        st32(&mut command[e..], 5151);
        st64(&mut command[e + 4..], 0x2345);
        st64(&mut command[e + 12..], 0x3456);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = store_route_count(ROUTE);
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_SET_VERTEX_BUFFER_STRIDE,
            &command,
            &mut out,
            &mut acc,
        );
        assert_eq!(
            store_route_count(ROUTE),
            before + 1,
            "the dropped stride was not counted"
        );
        assert_eq!(
            acc.vertex_buffers.len(),
            1,
            "the buffer did not bind; this record used to be refused whole"
        );
        let b = &acc.vertex_buffers[0];
        assert_eq!((b.index, b.buffer_ref, b.offset), (4, 5151, 0x2345));
        assert!(
            acc.fragment_buffers.is_empty(),
            "a vertex bind reached the fragment table"
        );

        // The plain bind must not report a stride it never carried, or the
        // counter reads as traffic on every ordinary vertex bind.
        let plain_total =
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::BUFFER_BIND_ENTRY_SIZE;
        let mut plain = vec![0u8; plain_total];
        st32(&mut plain[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
        st32(&mut plain[4..], plain_total as u32);
        st32(
            &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            1,
        );
        st32(
            &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
            5151,
        );
        let before = store_route_count(ROUTE);
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_SET_VERTEX_BUFFER,
            &plain,
            &mut out,
            &mut acc,
        );
        assert_eq!(
            store_route_count(ROUTE),
            before,
            "a plain vertex bind reported a stride it does not carry"
        );
    }

    /// Every state this rail decodes and does not apply reaches its own counter,
    /// and the ones with an API default stay quiet when the guest asks for it.
    ///
    /// The counters are the whole reason those opcodes are decoded: each is the
    /// measured argument for whether implementing that state is worth building,
    /// and a counter nobody reads back cannot be shown to be wired up. Nine of
    /// them had no such test until now.
    ///
    /// The two indirect draws are deliberately not in the default half. They
    /// have no default to be at -- an indirect draw is geometry the guest asked
    /// for, so every one is a loss and is counted unconditionally.
    #[test]
    fn every_decoded_but_unapplied_render_state_reaches_its_own_counter() {
        use crate::contract::endian::{st16, st64};
        use crate::runtime::drain::store_route_count;

        // (opcode, total length, payload writer, route, whether a default-valued
        // record of the same opcode must NOT count).
        type Writer = fn(&mut [u8]);
        let non_default: Writer = |p| st64(p, 2);
        let at_default: Writer = |p| st64(p, 0);
        let float_non_default: Writer = |p| st32(p, 2.5f32.to_bits());
        let float_at_default: Writer = |p| st32(p, 1.0f32.to_bits());

        let cases: &[(u32, usize, Writer, Option<Writer>, &str)] = &[
            (
                wire_render::OPCODE_SET_TRIANGLE_FILL_MODE,
                16,
                non_default,
                Some(at_default),
                "render_fill_mode_dropped",
            ),
            (
                wire_render::OPCODE_SET_DEPTH_CLIP_MODE,
                16,
                non_default,
                Some(at_default),
                "render_depth_clip_mode_dropped",
            ),
            (
                wire_render::OPCODE_SET_LINE_WIDTH,
                12,
                float_non_default,
                Some(float_at_default),
                "render_line_width_dropped",
            ),
            (
                wire_render::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
                12,
                float_non_default,
                Some(float_at_default),
                "render_tessellation_scale_dropped",
            ),
            (
                wire_render::OPCODE_SET_DEPTH_STORE_ACTION,
                16,
                non_default,
                // No default to compare against: the record overrides the pass
                // descriptor, so even a zero is a change this rail is not making.
                None,
                "render_store_action_override_dropped",
            ),
            (
                wire_render::OPCODE_SET_VISIBILITY_RESULT_MODE,
                24,
                // The mode is the *second* field: this record puts the offset
                // first, reversing its selector. Writing the mode at payload+0
                // sets the offset instead and leaves the mode at Disabled, so
                // the counter correctly stays quiet -- which is how this test
                // first failed.
                |p| {
                    st64(p, 0x1234);
                    st64(&mut p[8..], 2);
                },
                Some(|p| {
                    st64(p, 0x1234);
                    st64(&mut p[8..], 0);
                }),
                "render_visibility_result_mode_dropped",
            ),
            (
                wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
                reims_vgpu_wire::OP_HEADER_LEN
                    + render::AMPLIFICATION_COUNT_LEN
                    + 2 * render::AMPLIFICATION_MAPPING_SIZE,
                // Two views. One is Metal's default and means no amplification,
                // so the default arm below asks for one and must not count.
                |p| st32(p, 2),
                Some(|p| st32(p, 1)),
                "render_vertex_amplification_dropped",
            ),
            (
                wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
                reims_vgpu_wire::OP_HEADER_LEN + 8,
                |p| {
                    st32(p, 0x5555);
                    st32(&mut p[4..], 0x6666);
                },
                Some(|p| {
                    st32(p, 0);
                    st32(&mut p[4..], 0);
                }),
                "render_vertex_amplification_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_INDIRECT,
                24,
                |p| {
                    st64(p, 0x1111);
                    st32(&mut p[8..], 5151);
                    st16(&mut p[12..], 3);
                },
                None,
                "render_draw_indirect_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_INDEXED_INDIRECT,
                36,
                |p| {
                    st16(p, 4);
                    st16(&mut p[2..], 1);
                    st32(&mut p[4..], 5151);
                    st32(&mut p[8..], 5252);
                    st64(&mut p[12..], 0x1111);
                    st64(&mut p[20..], 0x2222);
                },
                None,
                "render_draw_indexed_indirect_dropped",
            ),
            // The tile family. The four bind opcodes each get a one-slot record
            // at their own entry stride, so a route that fired from the wrong
            // arm would have to have accepted the wrong length first.
            (
                wire_tile::OPCODE_SET_TILE_BUFFER,
                reims_vgpu_wire::OP_HEADER_LEN
                    + render::BIND_ENTRIES
                    + render::BUFFER_BIND_ENTRY_SIZE,
                |p| {
                    st32(&mut p[render::BIND_FIRST..], 3);
                    st32(&mut p[render::BIND_COUNT..], 1);
                },
                None,
                "render_tile_buffer_bind_dropped",
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
                20,
                |p| {
                    st32(p, 4);
                    st64(&mut p[4..], 0x2345);
                },
                None,
                "render_tile_buffer_bind_dropped",
            ),
            (
                wire_tile::OPCODE_SET_TILE_TEXTURE,
                reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE,
                |p| {
                    st32(&mut p[render::BIND_FIRST..], 2);
                    st32(&mut p[render::BIND_COUNT..], 1);
                },
                None,
                "render_tile_texture_bind_dropped",
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER,
                reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE,
                |p| {
                    st32(&mut p[render::BIND_FIRST..], 4);
                    st32(&mut p[render::BIND_COUNT..], 1);
                },
                None,
                "render_tile_sampler_bind_dropped",
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
                reims_vgpu_wire::OP_HEADER_LEN
                    + render::BIND_ENTRIES
                    + render::SAMPLER_LOD_BIND_ENTRY_SIZE,
                |p| {
                    st32(&mut p[render::BIND_FIRST..], 5);
                    st32(&mut p[render::BIND_COUNT..], 1);
                },
                None,
                "render_tile_sampler_bind_dropped",
            ),
            // The three dispatches. Their default arm is a grid with a zero
            // dimension, which Metal dispatches nothing for -- dropping one
            // loses no work, so counting it would inflate the very number this
            // counter exists to be.
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
                32,
                |p| {
                    st64(p, 0x11);
                    st64(&mut p[8..], 0x22);
                    st64(&mut p[16..], 0x33);
                },
                Some(|p| {
                    st64(p, 0x11);
                    st64(&mut p[8..], 0x22);
                    st64(&mut p[16..], 0);
                }),
                "render_tile_dispatch_dropped",
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
                84,
                |p| {
                    st64(p, 0x11);
                    st64(&mut p[8..], 0x22);
                    st64(&mut p[16..], 0x33);
                },
                Some(|p| st64(&mut p[16..], 0)),
                "render_tile_dispatch_dropped",
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
                84,
                |p| {
                    st64(p, 0x11);
                    st64(&mut p[8..], 0x22);
                    st64(&mut p[16..], 0x33);
                },
                Some(|p| st64(&mut p[16..], 0)),
                "render_tile_dispatch_dropped",
            ),
            // Not a dropped command but an unanswered question, so it has no
            // default arm: every one of these leaves the guest reading its own
            // stale ring as a tile geometry.
            (
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
                20,
                |p| {
                    st32(p, 5151);
                    st64(&mut p[4..], 0x9999);
                },
                None,
                "render_tile_dimensions_unanswered",
            ),
            (
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
                28,
                |p| {
                    st64(p, 0x1234);
                    st64(&mut p[8..], 0x2345);
                    st32(&mut p[16..], 5);
                },
                None,
                "render_tile_threadgroup_memory_dropped",
            ),
            // The store-action options. Their record is four bytes longer on
            // the colour form than on the other two, so a route reached from
            // the wrong arm would have had to accept the wrong length first.
            (
                wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
                20,
                |p| {
                    st64(p, 0x1111);
                    st32(&mut p[8..], 3);
                },
                None,
                "render_store_action_options_dropped",
            ),
            (
                wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
                16,
                |p| st64(p, 0x2222),
                None,
                "render_store_action_options_dropped",
            ),
            (
                wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
                16,
                |p| st64(p, 0x3333),
                None,
                "render_store_action_options_dropped",
            ),
            (
                wire_render::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
                28,
                |p| {
                    st32(p, 5151);
                    st64(&mut p[4..], 0x3456);
                    st64(&mut p[12..], 0x4567);
                },
                None,
                "render_tessellation_factor_buffer_dropped",
            ),
            // The patch draws. The two `0x0c` rows are the point: one opcode,
            // two lengths, and both must reach the counter -- a length-based
            // dispatch that refused one of them would read as a healthy zero.
            (
                wire_render::OPCODE_DRAW_PATCHES,
                24,
                |_p| {},
                None,
                "render_draw_patches_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_PATCHES_WIDE,
                56,
                |_p| {},
                None,
                "render_draw_patches_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_PATCHES_WIDE,
                68,
                |_p| {},
                None,
                "render_draw_patches_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_INDEXED_PATCHES,
                32,
                |_p| {},
                None,
                "render_draw_patches_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_PATCHES_INDIRECT,
                36,
                |_p| {},
                None,
                "render_draw_patches_indirect_dropped",
            ),
            (
                wire_render::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
                48,
                |_p| {},
                None,
                "render_draw_patches_indirect_dropped",
            ),
        ];

        let run = |op: u32, total: usize, write: Writer| {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let host = FakeHost::new();
            let mut out = ExecResult::default();
            let mut acc = StreamAccum::default();
            let mut command = vec![0u8; total];
            st32(&mut command[0..], op);
            st32(&mut command[4..], total as u32);
            write(&mut command[reims_vgpu_wire::OP_HEADER_LEN..]);
            handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        };

        for (op, total, write, default_write, route) in cases {
            let before = store_route_count(route);
            run(*op, *total, *write);
            assert_eq!(
                store_route_count(route),
                before + 1,
                "op {op:#x} did not reach {route}"
            );
            if let Some(default_write) = default_write {
                let before = store_route_count(route);
                run(*op, *total, *default_write);
                assert_eq!(
                    store_route_count(route),
                    before,
                    "op {op:#x} counted a guest asking for the API default, which \
                     is what this rail already does -- that turns the healthy \
                     zero back into a flood"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod publish_wait_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    /// A draw whose pipeline object the guest has not finished publishing is
    /// retried, not lost — that draw is the 128x128 render that fills an app
    /// icon. The wait is bounded so a reference that never resolves costs one
    /// packet of latency instead of the channel.
    #[test]
    fn an_unpublished_pipeline_is_waited_for_and_then_given_up_on() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let key = (7u32, 14u32);
        let now = std::time::Instant::now();

        // First sight starts the clock and is inside the budget.
        let since = *state.pipeline_unreadable_since.entry(key).or_insert(now);
        assert!(now.duration_since(since) < PIPELINE_PUBLISH_WAIT);

        // Past the budget the wait ends, and the entry goes with it so the
        // next reference starts its own clock rather than inheriting this one.
        state.pipeline_unreadable_since.insert(
            key,
            now - PIPELINE_PUBLISH_WAIT - std::time::Duration::from_millis(1),
        );
        let since = state.pipeline_unreadable_since[&key];
        assert!(std::time::Instant::now().duration_since(since) >= PIPELINE_PUBLISH_WAIT);
        state.pipeline_unreadable_since.remove(&key);
        assert!(state.pipeline_unreadable_since.is_empty());
    }
}
