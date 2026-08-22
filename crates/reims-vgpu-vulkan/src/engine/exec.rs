//! Record / submit (bounded fence) / readback for one draw.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::caches::{
    canonicalize_layout_bindings, AttrKey, BindingSig, ColorLoadKey, LayoutKey, ObjectCaches,
    PassKey, PipelineKey, SecondaryAttachKey, SessionCacheIndexes, MAX_SECONDARY_ATTACH,
};
use super::context::ContextOwner;
use super::counters::{CreateSite, EngineCounters};
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::draw_execution::DrawExecutionDecline;
use super::draw_validation::DrawValidationDecline;
use super::pools::{
    BatchFit, BatchTarget, BufferSlot, CbBind, ResourcePools, SampledKey, SampledSlot, TargetKey,
};
use super::stage_phase;
use super::types::{
    BufferContent, ColorWriteMask, DrawError, DrawOutput, DrawRequest, ResidentReclaim,
    SampledImageResource, SampledSource, ScissorResource, SeedOrder, TargetIdentity,
    VertexStepFunction, ViewportResource, VisibilityResultMode,
};
use super::vk_call::{VkCall, VkOp};

fn effective_line_raster_state(
    topology: reims_vgpu_core::PrimitiveTopology,
    fill_mode: reims_vgpu_core::FillMode,
    requested: reims_vgpu_core::LineWidth,
) -> (u32, bool) {
    let rasterizes_lines = matches!(
        topology,
        reims_vgpu_core::PrimitiveTopology::Line | reims_vgpu_core::PrimitiveTopology::LineStrip
    ) || fill_mode == reims_vgpu_core::FillMode::Lines;
    if !rasterizes_lines {
        return (reims_vgpu_core::LineWidth::ONE.bits(), false);
    }
    let width = requested.value();
    if width.is_nan() || width < 1.0 {
        (reims_vgpu_core::LineWidth::ONE.bits(), true)
    } else {
        (requested.bits(), false)
    }
}

fn populate_dynamic_viewport_scissors(
    req: &DrawRequest,
    viewports: &mut Vec<vk::Viewport>,
    scissors: &mut Vec<vk::Rect2D>,
) {
    let (raster_width, raster_height) = req.raster_extent();
    let default_viewport = ViewportResource {
        x: 0.0,
        y: 0.0,
        width: raster_width as f32,
        height: raster_height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let default_scissor = ScissorResource {
        x: 0,
        y: 0,
        width: raster_width,
        height: raster_height,
    };
    let slots = crate::engine::viewport_slot_count(req);
    viewports.extend((0..slots).map(|index| {
        let viewport = req
            .viewports
            .get(index)
            .copied()
            .unwrap_or(default_viewport);
        vk::Viewport {
            x: viewport.x,
            y: viewport.y + viewport.height,
            width: viewport.width,
            height: -viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        }
    }));
    scissors.extend((0..slots).map(|index| {
        let scissor = req.scissors.get(index).copied().unwrap_or(default_scissor);
        let x = scissor.x.min(raster_width);
        let y = scissor.y.min(raster_height);
        vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D {
                width: scissor.width.min(raster_width - x),
                height: scissor.height.min(raster_height - y),
            },
        }
    }));
}

fn effective_depth_bias(
    requested: Option<[f32; 3]>,
    depth_bias_clamp: bool,
) -> Result<Option<[f32; 3]>, super::reason::DrawReason> {
    let Some(values) = requested else {
        return Ok(None);
    };
    for (component, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(super::reason::DrawReason::DepthBiasNonFinite {
                component: component as u8,
                value_bits: value.to_bits(),
            });
        }
    }
    if values == [0.0; 3] {
        return Ok(None);
    }
    if values[2] != 0.0 && !depth_bias_clamp {
        return Err(super::reason::DrawReason::DepthBiasClampUnsupported {
            clamp_bits: values[2].to_bits(),
        });
    }
    Ok(Some(values))
}

fn draw_uses_blend_constants(req: &DrawRequest) -> bool {
    req.blend
        .iter()
        .chain(
            req.secondary_targets
                .iter()
                .filter_map(|target| target.blend.as_ref()),
        )
        .any(|blend| {
            [
                blend.src_color,
                blend.dst_color,
                blend.src_alpha,
                blend.dst_alpha,
            ]
            .into_iter()
            .any(reims_vgpu_core::BlendFactor::uses_blend_constant)
        })
}

/// A buffer a draw binds, and where in it the bytes start.
///
/// Two origins, and the offset is what distinguishes them. A pooled staging slot
/// always starts at zero — the pool carved it for this content alone. A guest
/// span starts wherever it sits inside its RAMBlock's import, which spans the
/// whole block and starts at the device's import granularity, while the guest's
/// allocator is aligned to neither.
///
/// Deliberately not a [`BufferSlot`]. A slot is a *pool* object: `acquire_staging`
/// enters it in `staging_live` and the ring recycles it at retire. An imported
/// RAMBlock belongs to [`super::host_ram::HostRamImports`] and lives as long as
/// the device does, so a type that could be mistaken for a slot would eventually
/// be handed to the staging free list — where it would be reissued as scratch
/// over the guest's live pages.
#[derive(Clone, Copy)]
pub(super) struct BoundBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) offset: vk::DeviceSize,
    /// This buffer directly aliases guest memory rather than owned staging.
    pub(super) guest_import: bool,
}

impl From<BufferSlot> for BoundBuffer {
    fn from(slot: BufferSlot) -> Self {
        Self {
            buffer: slot.buffer,
            offset: 0,
            guest_import: false,
        }
    }
}

/// One draw-time buffer window the GPU assembles out of the guest's own pages
/// before the render pass, because the window is not one contiguous stretch of
/// guest physical memory and a bind must name one range.
///
/// Recorded rather than executed at plan time: the copies belong in the draw's
/// own command buffer, ahead of the pass that reads them, so they cost one
/// submission with the draw instead of a submit and a fence of their own.
pub(super) struct PendingGuestGather {
    /// Device-local destination, which is what the draw actually binds.
    pub(super) dst: vk::Buffer,
    /// One entry per import the window's stretches resolved against, each with
    /// its copy regions.
    ///
    /// Grouped because two stretches need not share an import — a window
    /// straddling two RAMBlocks resolves against two `VkBuffer`s — and one
    /// `vkCmdCopyBuffer` names exactly one source. Ordinary machines have one
    /// RAMBlock and this is a single-entry `Vec`, but the grouping is what makes
    /// the two-block case land the whole window instead of the part that
    /// happened to be first.
    pub(super) sources: Vec<(vk::Buffer, Vec<vk::BufferCopy>)>,
}

/// Which consumer(s) require one physical guest-buffer gather.
///
/// A content allocation may appear as several vertex attributes and as a
/// storage descriptor in the same draw, while [`ResourcePools`] gathers it
/// once. Keeping this classification on the physical operation makes the three
/// byte counters partition `buffer_guest_gather_bytes`; counting logical binds
/// would double-charge the shared case and could choose the wrong zero-copy
/// rail.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BufferGatherRole {
    vertex: bool,
    storage: bool,
    index_alignment: Option<u64>,
}

impl BufferGatherRole {
    const VERTEX: Self = Self {
        vertex: true,
        storage: false,
        index_alignment: None,
    };
    pub(super) const STORAGE: Self = Self {
        vertex: false,
        storage: true,
        index_alignment: None,
    };
    pub(super) fn is_shared(self) -> bool {
        self.vertex as u8 + self.storage as u8 + self.index_alignment.is_some() as u8 > 1
    }

    pub(super) fn includes_index(self) -> bool {
        self.index_alignment.is_some()
    }

    pub(super) fn includes_storage(self) -> bool {
        self.storage
    }

    pub(super) fn is_storage_only(self) -> bool {
        self.storage && !self.vertex && self.index_alignment.is_none()
    }
}

/// How many pixels the attachment a pass instance is about to begin over covers,
/// as a band.
///
/// # Why a pass begin is banded at all
///
/// A render pass boundary is the largest single cost in this device on the
/// x86/Vulkan iGPU pathway, and it is measured causally rather than inferred:
/// `REIMS_VGPU_PASS_CHURN=on`, which adds one end/begin pair per merged loading
/// draw and changes nothing else, moved GPU per draw from **9.25 to 67.64 µs**
/// and drain CPU from 8.41 to 26.69 on interleaved driven macos-13 Maps boots
/// (/tmp/wb-out70, 71). At 169 345 pass begins a boot that is roughly two thirds
/// of the device's whole GPU second.
///
/// Two mechanisms would produce that number and they call for opposite work:
///
/// * a **full-surface operation** on the attachment — a `loadOp = CLEAR` the
///   driver cannot fast-clear, or a compression resolve. 1920x1080x4 is 8.3 MB,
///   which at this host's bandwidth is ~95 µs, and that is close enough to the
///   measurement to be worth refuting rather than assuming. If this is it, the
///   fix is to stop asking for the operation and every pass gets cheap.
/// * a **GPU pipeline drain and cache flush**, which is what a Vulkan render
///   pass boundary is on this driver whatever it is drawing into. If this is it,
///   no pass can be made cheap and the only lever is opening fewer of them.
///
/// The bands separate the two, because only the first scales with the
/// attachment. Regress a census window's `gpu_span busy_us` on the band counts:
/// coefficients that climb with the band are a surface operation, coefficients
/// that are flat across the bands are a drain. `passbegin_load` against
/// `passbegin_clear` beside them says whether the load action is the term,
/// which is the specific form of the first mechanism most likely to be true.
///
/// Boundaries are powers of four from 64 Ki pixels, so each band is four times
/// the traffic of the one below it and a linear cost is unmistakable. A 1080p
/// attachment is 2.07 M pixels and lands in the top band.
fn pass_begin_area_band(width: u32, height: u32) -> &'static str {
    match u64::from(width).saturating_mul(u64::from(height)) {
        0..=65_535 => "passbegin_px_lt64k",
        65_536..=262_143 => "passbegin_px_lt256k",
        262_144..=1_048_575 => "passbegin_px_lt1m",
        _ => "passbegin_px_ge1m",
    }
}

/// One draw's buffer binds partitioned by the consumer(s) each physical content
/// allocation serves.
///
/// A flat table and not a map. The population is a handful — a driven Maps boot
/// reads 2.9 vertex attributes and about six buffer binds a draw — and the
/// question asked of it is three bits wide, so an ordered map cost a node
/// allocation per *distinct* allocation on every draw plus a pointer-chasing
/// probe per bind, to answer something a linear scan over one contiguous
/// allocation answers in a few compares. `sg_roles_us` was 0.15 µs of a 9.3 µs
/// chain before this and the probes were charged to `sg_vertex`/`sg_index`/
/// `sg_storage` beside it.
///
/// Built from key derivations that take no reference to what they name — see
/// [`CbBind::key_of`] — because the `DrawRequest` this borrows outlives the
/// table, so the allocations cannot go anywhere while it is being read.
struct BufferGatherRoles {
    entries: Vec<((usize, u64, u64), BufferGatherRole)>,
}

impl BufferGatherRoles {
    fn of(req: &DrawRequest) -> Self {
        let mut entries: Vec<((usize, u64, u64), BufferGatherRole)> = Vec::with_capacity(
            req.vertex_attributes.len()
                + req.storage_buffers.len()
                + req.indexed.is_some() as usize,
        );
        // `entry`-shaped, so a content allocation named twice in one draw stays
        // one physical operation carrying both roles.
        let mut merge = |key, seed: BufferGatherRole, add: fn(&mut BufferGatherRole)| match entries
            .iter_mut()
            .find(|(held, _)| held == &key)
        {
            Some((_, role)) => add(role),
            None => entries.push((key, seed)),
        };
        for content in req.vertex_attributes.iter().map(|r| &r.content) {
            merge(CbBind::key_of(content), BufferGatherRole::VERTEX, |role| {
                role.vertex = true
            });
        }
        for content in req.storage_buffers.iter().map(|r| &r.content) {
            merge(CbBind::key_of(content), BufferGatherRole::STORAGE, |role| {
                role.storage = true
            });
        }
        if let Some(indexed) = &req.indexed {
            let alignment = indexed.index_type.byte_size() as u64;
            let key = CbBind::key_of(&indexed.content);
            let seed = BufferGatherRole {
                vertex: false,
                storage: false,
                index_alignment: Some(alignment),
            };
            match entries.iter_mut().find(|(held, _)| held == &key) {
                Some((_, role)) => role.index_alignment = Some(alignment),
                None => entries.push((key, seed)),
            }
        }
        Self { entries }
    }

    /// The role of the bind this key names. Every bind was classified by
    /// [`Self::of`] from the same `DrawRequest`, so an absent key is a caller
    /// asking about a bind that is not in this draw.
    fn role(&self, key: (usize, u64, u64)) -> Option<BufferGatherRole> {
        self.entries
            .iter()
            .find(|(held, _)| *held == key)
            .map(|(_, role)| *role)
    }

    /// How many *physical* operations the draw's binds partition into, which is
    /// the property the role split exists to preserve. Nothing on the draw path
    /// asks; the partition test does.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Offset alignment imposed by every consumer of one physical buffer bind.
fn buffer_bind_offset_alignment(role: BufferGatherRole, storage_align: u64) -> u64 {
    let storage = if role.storage { storage_align } else { 1 };
    storage.max(role.index_alignment.unwrap_or(1))
}

impl PendingGuestGather {
    /// Copy regions across every source, which is what the census counts.
    fn regions(&self) -> u64 {
        self.sources.iter().map(|(_, r)| r.len() as u64).sum()
    }
}

/// One gather turned into a compute dispatch: the descriptor set naming its
/// three buffers, and how many runs the kernel has to walk.
struct GatherDispatch {
    set: vk::DescriptorSet,
    run_count: u32,
}

/// Whether the compute gather is on. **Default on since 2026-08-11** — it takes
/// 21 % off this device's GPU work per draw for byte-identical guest output, and
/// the CPU cost that kept it off for two sessions is now paid out of drain-worker
/// headroom that did not exist when it was measured.
///
/// `off` restores the ~13 `VkBufferCopy` regions per gathered window. See
/// [`reims_vgpu_config::COMPUTE_GATHER`] for both sets of boots and for why the earlier
/// rejection was right at the time.
/// Whether the layout-churn probe is on. **Default off**, and never anything but
/// a probe: it adds two image barriers per draw and removes nothing.
///
/// See its one call site for what it prices and why the answer is not otherwise
/// obtainable. It is a switch and not a `#[cfg]` because the question it answers
/// is about the host, so it has to be askable on a host somebody else has.
fn layout_churn_probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            reims_vgpu_config::read(reims_vgpu_config::LAYOUT_CHURN).0,
            reims_vgpu_config::Switch::On
        )
    })
}

/// Whether the pass-churn probe is on. **Default off**, and never anything but a
/// probe: it adds one empty render pass instance per loading draw and removes
/// nothing.
///
/// See its one call site for what it prices, and [`reims_vgpu_config::PASS_CHURN`] for
/// why the question is not otherwise answerable without building the merge it is
/// pricing.
fn pass_churn_probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            reims_vgpu_config::read(reims_vgpu_config::PASS_CHURN).0,
            reims_vgpu_config::Switch::On
        )
    })
}

fn compute_gather_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            reims_vgpu_config::read(reims_vgpu_config::COMPUTE_GATHER).0,
            reims_vgpu_config::Switch::Off
        )
    })
}

/// One thing a draw records that a render pass instance cannot contain.
///
/// The variants are the recording sites in [`execute_draw_inner`], one each, and
/// they exist as a type rather than as the route string each used to carry so
/// that the two ladders below cannot come apart: a new obstacle fails to compile
/// until both spellings exist, where a second `get_or_insert` of a hand-written
/// name would simply be missing from one of them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PassObstacle {
    /// A target sampled by its own draw, captured before the attachment changes.
    Snapshot,
    /// The seed copy that gives a `LOAD` pass its prior content.
    Seed,
    /// A full-image attachment load materialized before a smaller MRT pass.
    AttachmentLoad,
    /// The colour target transitioned back into attachment use.
    TargetLayout,
    /// A `CLEAR` pass's colour writes waiting for whoever last read the target.
    ClearWait,
    /// A sampled resident transitioned to shader-read.
    ResidentLayout,
    /// A CPU-origin or guest-origin upload into a sampled image.
    SampledUpload,
    /// Writes through any guest-memory alias made visible to imported reads,
    /// including the first layout transition of a newly imported image.
    GuestMemoryVisibility,
    /// The compute dispatches (or transfer copies) that assemble scattered guest
    /// buffer windows.
    Gather,
    /// An MRT secondary attachment transitioned back into attachment use.
    MrtLayout,
    /// `vkCmdResetQueryPool` for an occlusion query.
    QueryReset,
}

impl PassObstacle {
    /// Whether this obstacle exists **only because the previous draw's pass was
    /// closed**.
    ///
    /// [`super::caches::ObjectCaches::get_or_create_pass`] ends every pass with
    /// `final_layout = TRANSFER_SRC_OPTIMAL`, so the next draw into that target
    /// has to barrier its colour attachments back to
    /// `COLOR_ATTACHMENT_OPTIMAL`. A pass that was never ended never moved them,
    /// so those two transitions have nothing to undo and are not recorded at
    /// all. Every other variant is work the draw owes whatever the pass did — a
    /// sampled resident still has to become shader-readable, a gather still has
    /// to run — so holding the pass open does not remove it, it only means the
    /// pass has to be closed to record it.
    fn is_pass_end_artifact(self) -> bool {
        matches!(self, Self::TargetLayout | Self::MrtLayout)
    }

    /// The `passmerge_*` route for the ladder that charges the nearest obstacle
    /// as this device records today.
    fn route(self) -> &'static str {
        match self {
            Self::Snapshot => "passmerge_outside_snapshot",
            Self::Seed => "passmerge_outside_seed",
            Self::AttachmentLoad => "passmerge_outside_attachment_load",
            Self::TargetLayout => "passmerge_outside_target_layout",
            Self::ClearWait => "passmerge_outside_clear_wait",
            Self::ResidentLayout => "passmerge_outside_resident_layout",
            Self::SampledUpload => "passmerge_outside_sampled_upload",
            Self::GuestMemoryVisibility => "passmerge_outside_guest_memory_visibility",
            Self::Gather => "passmerge_outside_gather",
            Self::MrtLayout => "passmerge_outside_mrt_layout",
            Self::QueryReset => "passmerge_outside_query_reset",
        }
    }

    /// The `passheld_*` route for the ladder that charges the nearest obstacle a
    /// held-open pass would still meet.
    fn held_route(self) -> &'static str {
        match self {
            Self::Snapshot => "passheld_outside_snapshot",
            Self::Seed => "passheld_outside_seed",
            Self::AttachmentLoad => "passheld_outside_attachment_load",
            Self::TargetLayout | Self::MrtLayout => {
                unreachable!("an attachment-layout obstacle is not recorded on the held ladder")
            }
            Self::ClearWait => "passheld_outside_clear_wait",
            Self::ResidentLayout => "passheld_outside_resident_layout",
            Self::SampledUpload => "passheld_outside_sampled_upload",
            Self::GuestMemoryVisibility => "passheld_outside_guest_memory_visibility",
            Self::Gather => "passheld_outside_gather",
            Self::QueryReset => "passheld_outside_query_reset",
        }
    }
}

/// The nearest obstacle a draw records, read twice: once as this device records
/// today, and once as it would record if a pass were never closed between two
/// draws of one command buffer.
///
/// Both are needed because the first answered its own question and hid the next
/// one. `passmerge_outside_target_layout` took 82.4 % of draws, which says the
/// attachment transition is the nearest obstacle and says **nothing** about what
/// stands behind it — and that is the number that decides whether holding the
/// pass open is worth building, because the transition is the one obstacle that
/// holding the pass open removes by construction.
///
/// Observation only. Nothing here changes what is recorded.
#[derive(Default)]
struct PassObstacles {
    first: Option<PassObstacle>,
    first_held: Option<PassObstacle>,
}

/// Whether the draw belongs inside the render pass the command buffer is
/// currently recording. Both facts are required: matching Vulkan objects do
/// a pass that an intervening outside-pass command already closed.
fn continues_open_render_pass(continues_encoder: bool, open_pass_matches: bool) -> bool {
    continues_encoder && open_pass_matches
}

impl PassObstacles {
    /// Charge one obstacle, at the site that records it and after whatever
    /// `continue` decides the site is a no-op for this draw. The first wins on
    /// each ladder, because the question is what ended the pass that was
    /// standing.
    fn note(&mut self, obstacle: PassObstacle) {
        self.first.get_or_insert(obstacle);
        if !obstacle.is_pass_end_artifact() {
            self.first_held.get_or_insert(obstacle);
        }
    }

    /// Close a pass inherited from the preceding draw before recording work
    /// Vulkan forbids inside it, then charge that work to the observation
    /// ladder. The close is deliberately coupled to the recording site so a
    /// newly-added barrier/copy/dispatch cannot accidentally execute inside a
    /// pass merely because a separate preflight forgot to predict it.
    unsafe fn before_record(
        &mut self,
        obstacle: PassObstacle,
        pools: &mut ResourcePools,
        device: &ash::Device,
        cb: vk::CommandBuffer,
    ) {
        unsafe { pools.close_open_pass(device, cb) };
        self.note(obstacle);
    }
}

/// Turn every pending gather into dispatches, or `None` to leave the whole
/// batch on the transfer regions.
///
/// # Why the whole batch and not a gather at a time
///
/// The two forms need different barriers — one is a `TRANSFER_WRITE` and the
/// other a `SHADER_WRITE` — and the loop that follows this emits exactly one
/// barrier for all of them. Mixing forms inside a batch would mean either two
/// barriers or one over-wide source scope, and this rail is not where an
/// all-or-nothing costs anything: a decline is a property of the *host* (no
/// pipeline, no import) or of the arithmetic (an unaligned run), and both are
/// stable across a command buffer rather than varying gather to gather.
///
/// This is the gather direction of the same repair `plan_guest_scatter_dispatches`
/// made to the writeback, where replacing ~200 transfer regions with one
/// dispatch was measured at +48 % frames on a driven macos-13 boot. Here it
/// replaces the 427 000 regions a second the buffer gather issues — the largest
/// remaining GPU cost on that rail — with one dispatch per gathered window.
///
/// # Safety
///
/// `gathers` must name buffers live for the whole submission being recorded.
unsafe fn plan_buffer_gather_dispatches(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    gathers: &[PendingGuestGather],
) -> Result<Option<Vec<GatherDispatch>>, DrawError> {
    use super::guest_scatter::{build_gather_run_tables, ScatterRun};
    let Some(pipeline) = (unsafe { pools.scatter_pipeline(ctx) }) else {
        return Ok(None);
    };
    // Planned for every gather before anything is allocated, so a refusal in the
    // last one does not leave the first one's staging slot and descriptor set on
    // the pools for a dispatch that will not be recorded — the same ordering
    // `plan_guest_scatter_dispatches` states.
    let mut planned: Vec<(vk::Buffer, vk::Buffer, u64, super::guest_scatter::RunTable)> =
        Vec::new();
    for gather in gathers {
        // The window's own byte count, from the regions themselves rather than
        // from the slot: `acquire_guest_gather` rounds up to a power-of-two
        // bucket, and the tighter bound is the one that catches a run reaching
        // past what this window actually covers.
        let dst_have: u64 = gather
            .sources
            .iter()
            .flat_map(|(_, copies)| copies.iter())
            .map(|c| c.dst_offset.saturating_add(c.size))
            .max()
            .unwrap_or(0);
        for (source, copies) in &gather.sources {
            let _p = super::gather_phase::Span::open(super::gather_phase::Part::Plan);
            let runs: Vec<ScatterRun> = copies
                .iter()
                .map(|c| ScatterRun {
                    // The guest import is the source here, so the copy's
                    // `src_offset` is the guest side and its `dst_offset` the
                    // device-local slot — the exact opposite of the writeback,
                    // and the whole of what `build_gather_run_tables` exchanges.
                    src: c.src_offset,
                    dst: c.dst_offset,
                    len: c.size,
                })
                .collect();
            match build_gather_run_tables(
                &runs,
                ctx.storage_buffer_offset_align,
                ctx.max_storage_buffer_range,
                dst_have,
            ) {
                Ok(built) => planned.extend(
                    built
                        .into_iter()
                        .map(|t| (*source, gather.dst, dst_have, t)),
                ),
                Err(decline) => {
                    counters
                        .buffer_gather_declined
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    reims_vgpu_observe::Emit::decline("buffer_gather_plan", &decline).fail_once(0);
                    return Ok(None);
                }
            }
        }
    }
    // One staging slot for every table this command buffer's gathers need, not
    // one apiece. A gathered window's table is ~200 bytes and this rail issues
    // ~40 000 of them a second against ~2 200 command buffers, so the acquire
    // and the write were being paid eighteen times over for bytes that fit in a
    // single slot.
    let (runs_slot, places) = {
        let _s = super::gather_phase::Span::open(super::gather_phase::Part::Stage);
        let words: Vec<&[u32]> = planned.iter().map(|(_, _, _, t)| &t.words[..]).collect();
        unsafe { super::stage_run_tables(ctx, pools, counters, &words) }?
    };
    let mut out = Vec::with_capacity(planned.len());
    for ((source, dst, dst_have, table), place) in planned.iter().zip(&places) {
        let _d = super::gather_phase::Span::open(super::gather_phase::Part::Dset);
        let set =
            unsafe { pools.alloc_scatter_descriptor_set(&ctx.device, pipeline.dsl, counters) }?;
        unsafe {
            super::guest_scatter::ScatterPipeline::write_set(
                &ctx.device,
                set,
                (*source, table.bind_offset, table.bind_range),
                (*dst, 0, *dst_have),
                (runs_slot.buffer, place.bind_offset, place.bind_range),
            );
        }
        out.push(GatherDispatch {
            set,
            run_count: table.run_count,
        });
    }
    Ok(Some(out))
}

/// Stage one draw-time buffer content into something the draw can bind,
/// deduplicating within the draw: several binds sharing one content (an `Arc`'d
/// byte allocation, or the same guest span) resolve to ONE buffer and at most
/// one copy.
///
/// A `GuestRuns` span has three dispositions on a host that can import guest
/// RAM, in decreasing order of what they cost:
///
/// 1. **Bound in place.** The window is one GPA-contiguous stretch sitting at an
///    offset the device will bind at, so the draw reads the guest's bytes where
///    the guest wrote them. Nothing is copied in either direction.
/// 2. **Gathered by the GPU.** The window is several stretches, so one
///    `vkCmdCopyBuffer` per stretch assembles it into device-local memory ahead
///    of the render pass. The bus is crossed once, by the engine that was going
///    to read those bytes anyway.
/// 3. **Gathered by the CPU.** Everything else, and the only arm on a host
///    without `VK_EXT_external_memory_host`. A `memcpy` per stretch into mapped
///    staging.
///
/// Arm 2 exists because arm 1 turned out to be unreachable on a real workload:
/// the guest backs a surface in 16 KiB physically-contiguous granules, so a
/// driven boot put 98.5 % of these windows at 9-32 stretches and **none at all**
/// at one. See `reims-vgpu::runtime::draw`'s guest-page window planner for the
/// measurement and what it cost — 3.6 GB/s of CPU `memcpy`, two thirds of every
/// draw's staging phase.
///
/// Arms 1 and 2 both read guest RAM when the command buffer *executes*, which is
/// after this device would otherwise have told the guest the packet finished.
/// [`super::quiesce_guest_reads`], called from `write_stamp`, is what makes that
/// ordering a rule rather than a short window; `snapshot_volatile` records that
/// the runtime asked for a stable snapshot, which only arm 3 still gives it.
#[allow(
    clippy::too_many_arguments,
    reason = "buffer staging carries the Vulkan context, pools, binding, and lifetime sets"
)]
/// The `range` a storage-buffer descriptor gets: the bind's own length, not the
/// rest of whatever buffer it landed in.
///
/// This used to be `vk::WHOLE_SIZE` on both the draw and the compute path, and
/// that made the `robustBufferAccess` argument for the extent-narrowing rail
/// vacuous. A staged bind's `VkBuffer` is created at a **power-of-two bucket**
/// at least as large as the bytes written, and `write_staging` deliberately does
/// not zero the tail. Robust access clamps against the *descriptor range*, so
/// with `WHOLE_SIZE` the bytes between the bind's length and the end of the
/// bucket are in bounds of the binding and return the previous tenant's data —
/// another guest draw's constants. Commit 0005766b's body claimed robust access
/// made an over-read "visibly wrong rather than unsound"; it did not, and stale
/// bytes from an unrelated bind are strictly harder to see than zeroes.
///
/// With an exact range, a read past the bind's own declared object is clamped by
/// the driver and reads zero, which is defined and diagnosable.
///
/// Zero keeps `WHOLE_SIZE`, because a zero `range` is not a legal descriptor. A
/// zero-length storage bind is degenerate either way; this is not the place to
/// refuse it.
///
/// One consequence worth stating: for an SSBO whose last member is a runtime
/// array, `OpArrayLength` now reports the length derived from this range rather
/// than from the bucket. That is the true size of what was staged, so it is the
/// answer the shader should have been getting.
pub(crate) fn descriptor_range(len: u64) -> u64 {
    if len == 0 {
        vk::WHOLE_SIZE
    } else {
        len
    }
}

/// The first binding either of a draw's retained shader variants statically
/// uses that the descriptor set layout this draw would build does not describe,
/// and which module named it.
///
/// The draw-path twin of `exec_compute::used_binding_absent_from_layout`, and it
/// exists for the reason that one does: a used binding the layout omits is not
/// undefined behaviour this device can absorb, it is `SIGFPE` inside
/// `vkCreateGraphicsPipelines` on Mesa's Intel driver, which kills the QEMU
/// process with no status to inspect and no guest packet left to fail. Refusing
/// one draw is the only outcome that keeps the VM alive and says why.
///
/// Both modules, checked in fragment-first order because that is the stage the
/// measured population fires on. The layout is shared — this device builds one
/// set for the pipeline with `VERTEX | FRAGMENT` stage flags — so a binding is
/// absent for both stages or for neither, and only the attribution differs.
///
/// The retained sets were derived from the executable SPIR-V variants, not from
/// reflection. They include only unambiguous, statically used
/// `UniformConstant` roots, so a storage buffer is never refused on a guess
/// about a root the walk cannot resolve. That is the same narrowing the compute
/// twin documents, and it keeps this a backstop rather than a second opinion
/// about every draw.
fn used_binding_absent_from_layout(
    vert_used: &[u32],
    frag_used: &[u32],
    layout: &[BindingSig],
) -> Option<(u32, bool)> {
    let absent = |used: &[u32]| -> Option<u32> {
        used.iter()
            .copied()
            .find(|binding| !layout.iter().any(|b| b.binding == *binding))
    };
    absent(frag_used)
        .map(|binding| (binding, true))
        .or_else(|| absent(vert_used).map(|binding| (binding, false)))
}

#[derive(Clone, Copy)]
struct StageBufferUse {
    usage: vk::BufferUsageFlags,
    snapshot_volatile: bool,
    gather_role: BufferGatherRole,
}

unsafe fn stage_buffer_content(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    content: &BufferContent,
    use_: StageBufferUse,
    gathers: &mut Vec<PendingGuestGather>,
) -> Result<BoundBuffer, DrawError> {
    let StageBufferUse {
        usage,
        snapshot_volatile,
        gather_role,
    } = use_;
    // Probe on the identity alone. A Metal argument table persists across the
    // draws of one encoder, so the guest re-presents the same buffers on every
    // draw and this probe almost always hits — paying `CbBind::of`'s `Arc` clone
    // before knowing that would be two atomics per bind per draw, on the path
    // that does no work.
    let key = super::pools::CbBind::key_of(content);
    if let Some(bound) = pools.cb_bound_buffer(key) {
        counters.note_buffer_bind_reused(gather_role);
        return Ok(bound);
    }
    // A miss, so this bind is about to be recorded. The identity and a reference
    // to what it names are taken together here — see [`super::pools::CbBind`].
    // The map cannot be told about a bind without being handed this, which is
    // what keeps the key's address from being recycled under a live entry.
    let bind = super::pools::CbBind::of(content);
    debug_assert_eq!(bind.key(), key, "the probe and the record name one bind");
    // Set by the one arm that returns a slot it has not filled. Read after the
    // `match` so the flag and the `note_cb_bound_buffer` below it cannot be
    // separated by a future edit that adds an arm.
    let mut gather_owed = false;
    let bound = match content {
        BufferContent::Bytes(b) => {
            let slot = {
                let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
                pools.acquire_staging(ctx, b.len() as u64, usage, counters)?
            };
            let _s = stage_phase::Span::moving(stage_phase::Part::Bytes, b.len() as u64);
            pools.write_staging(ctx, &slot, b)?;
            drop(_s);
            BoundBuffer::from(slot)
        }
        BufferContent::GuestRuns(src) => {
            let transfer = src.transfer_plan();
            // A retained packed allocation is already the persistent buffer
            // object the guest bound. When the device accepts that host
            // allocation and every consumer accepts its offset, bind it as-is.
            // Exact-window/scattered sources that cannot form one Vulkan buffer
            // continue through the ordered gather below.
            if let Some(bound) =
                unsafe { import_guest_buffer_window(ctx, pools, &transfer, gather_role) }
            {
                pools.note_guest_read_recorded();
                counters.note_buffer_guest_import(src.total_len, gather_role);
                bound
            } else if let Some((bound, pending)) =
                unsafe { gather_guest_buffer_window(ctx, pools, counters, src, &transfer, usage)? }
            {
                // The copies read guest RAM when the CB executes, exactly as a
                // direct bind does, so this owes the same quiesce.
                pools.note_guest_read_recorded();
                counters.note_buffer_guest_gather(src.total_len, pending.regions(), gather_role);
                // Counted here and not at the top of this arm, because only a
                // window that actually gathers is a window a content cache would
                // have to hold. `key.0` is the `runs` allocation's address,
                // already computed above as this draw's dedup key — the same
                // identity one scope wider. See
                // [`super::pools::buffer_gather_working_set`].
                super::pools::buffer_gather_working_set::note_gathered(key.0, key.2);
                gathers.push(pending);
                // The slot this returns is recycled and still holds the previous
                // tenant's bytes; what fills it is `pending`, and `pending` is
                // recorded into the command buffer hundreds of lines below, past
                // every recoverable refusal the sampled rungs raise. Until then
                // the memo entry about to be published is not answerable, and
                // this is what says so.
                gather_owed = true;
                bound
            } else {
                // No import on this host, or an offset it will not bind at. The
                // CPU gathers the runs into the mapped staging span, with no
                // intermediate `cpu_bytes()` heap Vec (this is the
                // deferred-submit hot path, ~4.8 binds/draw under compositing).
                let slot = {
                    let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
                    pools.acquire_staging(ctx, src.total_len, usage, counters)?
                };
                let _s = stage_phase::Span::moving(stage_phase::Part::Runs, src.total_len);
                pools.write_staging_from_runs(
                    ctx,
                    &slot,
                    &src.runs,
                    src.source_offset,
                    src.total_len,
                )?;
                drop(_s);
                if snapshot_volatile {
                    counters
                        .buffer_snapshot_binds
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                BoundBuffer::from(slot)
            }
        }
    };
    pools.note_cb_bound_buffer(bind, bound);
    if gather_owed {
        pools.note_cb_bind_owes_gather(key);
    }
    Ok(bound)
}

/// Bind a retained guest buffer allocation directly when it is one Vulkan
/// buffer and its decoded offset satisfies every consumer's alignment.
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
unsafe fn import_guest_buffer_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    transfer: &reims_vgpu_memory::GuestReadTransferPlan<'_>,
    role: BufferGatherRole,
) -> Option<BoundBuffer> {
    if !ctx.caps.host_pointer.is_available() {
        return None;
    }
    let stretch = transfer.direct()?;
    let bound = match unsafe { pools.bind_guest_ram(ctx, stretch.guest) } {
        Ok(bound) => bound,
        Err(inner) => {
            reims_vgpu_observe::Emit::decline("vk_buffer_import", &inner).fail_once(0);
            return None;
        }
    };
    let offset = bound.offset + bound.head + stretch.skip;
    let align = buffer_bind_offset_alignment(role, ctx.storage_buffer_offset_align);
    if !offset.is_multiple_of(align) {
        reims_vgpu_observe::Emit::decline(
            "vk_buffer_import",
            &BufferImportDecline::BindOffsetAlignment { offset, align },
        )
        .fail_once(offset % align);
        return None;
    }
    Some(BoundBuffer {
        buffer: bound.buffer,
        offset,
        guest_import: true,
    })
}

/// Bind a compute storage buffer directly to its retained guest allocation.
///
/// This is not a draw-buffer route: writable compute storage needs the guest
/// allocation itself so the dispatch can publish completion without a CPU
/// readback. Draw buffers instead go through [`gather_guest_buffer_window`].
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
pub(super) unsafe fn import_guest_compute_buffer_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    src: &super::types::GuestRunSource,
) -> Option<BoundBuffer> {
    if !ctx.caps.host_pointer.is_available() {
        return None;
    }
    let transfer = src.transfer_plan();
    let stretch = transfer.direct()?;
    let bound = match unsafe { pools.bind_guest_ram(ctx, stretch.guest) } {
        Ok(bound) => bound,
        Err(inner) => {
            reims_vgpu_observe::Emit::decline("vk_compute_buffer_import", &inner).fail_once(0);
            return None;
        }
    };
    let offset = bound.offset + bound.head + stretch.skip;
    let align = ctx.storage_buffer_offset_align;
    if !offset.is_multiple_of(align) {
        reims_vgpu_observe::Emit::decline(
            "vk_compute_buffer_import",
            &BufferImportDecline::BindOffsetAlignment { offset, align },
        )
        .fail_once(offset % align);
        return None;
    }
    Some(BoundBuffer {
        buffer: bound.buffer,
        offset,
        guest_import: true,
    })
}

/// Assemble a scattered guest buffer window into device-local memory with one
/// GPU copy per stretch, or say why the CPU still has to gather it.
///
/// Returns the buffer the draw binds together with the copies its command buffer
/// owes. `Ok(None)` is a routing answer and never a lost draw — the caller's CPU
/// gather reads the same bytes through the same runs — so every check here is a
/// counted decline rather than an error.
///
/// # Why the destination is device-local and not a staging slot
///
/// A staging slot is host-visible, which on a discrete host means system memory.
/// Gathering into one would have the GPU read guest RAM across the bus, write
/// the result back across it, and then read it a third time when the draw runs —
/// three crossings against the CPU gather's one, so it would be *slower* than
/// the path it replaces. Device-local makes it one crossing and leaves the
/// draw's own reads in VRAM, which is where the win is.
///
/// # Caching this copy across submissions is measured shut
///
/// `stage_buffer_content` serves a repeat inside one command buffer from
/// `cb_bound_buffers`, and `seal_entry` drops that map on every submission —
/// 83 293 seals a boot, 290 747 entries discarded. Keeping the device-local
/// buffer instead, keyed on the same `(runs allocation, total_len)`, looks like
/// the largest saving available on a device whose ceiling is the ~7.9 GB/s it
/// puts across PCIe, of which this rail is 2.74 GB/s.
///
/// It is not, and the two halves of the measurement are why. A census of key
/// recurrence over a driven Safari-drag boot put **99.2 % of gathers and 99.3 %
/// of gathered bytes** on a span this device had already copied, so the cache
/// would be *asked* almost every time. A second census then folded the runs'
/// bytes on a sampled subset of those repeats and compared each against the
/// previous gather of the same span:
///
/// ```text
/// gather_bytes_changed   5 504 samples   1 252 958 KB
/// gather_bytes_same        695 samples     142 947 KB
/// ```
///
/// **Only ~10 % of repeats are the same bytes.** The guest rewrites these spans
/// between gathers — they are overwhelmingly per-draw constant buffers, which is
/// what `try_buffer_zero_copy_resolved`'s width census independently found them
/// to be. A perfect cache with exact invalidation would therefore serve about a
/// tenth of 2.74 GB/s, some 274 MB/s of a 7.9 GB/s budget, in exchange for
/// taking on invalidation this rail does not currently need at all: today the
/// GPU reads guest RAM when the command buffer executes, so a guest CPU write
/// between two draws is picked up with nothing having to notice it. A missed
/// invalidation would be a wrong frame rather than a slow one.
///
/// The asymmetry is the finding. The recurrence number is enormous, the content
/// number is small, and only the second one sizes the cache — a repeat count on
/// its own would have justified building it.
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
unsafe fn gather_guest_buffer_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    src: &super::types::GuestRunSource,
    transfer: &reims_vgpu_memory::GuestReadTransferPlan<'_>,
    usage: vk::BufferUsageFlags,
) -> Result<Option<(BoundBuffer, PendingGuestGather)>, DrawError> {
    if !ctx.caps.host_pointer.is_available() {
        return Ok(None);
    }
    let Some(stretches) = transfer.stretches() else {
        return Ok(None);
    };
    // From here on this window costs something, so from here on it is charged.
    // Opening the span above would count every *attempt* — and on a host that
    // cannot import, every attempt returns at the first line, so `gather_n`
    // would report the CPU rail's whole traffic as gathers and `gather_b` would
    // claim bytes the GPU never moved. A driven `REIMS_VGPU_GUEST_IMPORT=off`
    // boot read `gather_n=288196` beside `buffer_guest_gathers=0` before this
    // moved.
    let _span = stage_phase::Span::moving(stage_phase::Part::Gather, src.total_len);
    // Plan before acquiring, so a window that turns out not to be gatherable
    // does not take a destination slot out of the pool to abandon it.
    let mut sources: Vec<(vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
    let mut covered = 0u64;
    for stretch in stretches {
        let bound = match unsafe { pools.bind_guest_ram(ctx, stretch.guest) } {
            Ok(bound) => bound,
            Err(inner) => {
                reims_vgpu_observe::Emit::decline("vk_buffer_gather", &inner).fail_once(0);
                return Ok(None);
            }
        };
        let copy = gather_region(&bound, &stretch);
        covered = covered.saturating_add(copy.size);
        super::group_by_buffer(&mut sources, bound.buffer, copy);
    }
    // The runs tile the window exactly, so this holds by construction — but a
    // short gather would hand the draw a buffer whose tail is whatever the
    // previous user of the slot left there, which is wrong pixels rather than
    // slow ones. Checked here because this is the last place that can see it.
    if covered != src.total_len {
        reims_vgpu_observe::Emit::decline(
            "vk_buffer_gather",
            &BufferImportDecline::GatherShort {
                covered,
                want: src.total_len,
            },
        )
        .fail_once(src.total_len);
        return Ok(None);
    }
    let slot = pools.acquire_guest_gather(ctx, src.total_len, usage, counters)?;
    Ok(Some((
        BoundBuffer::from(slot),
        PendingGuestGather {
            dst: slot.buffer,
            sources,
        },
    )))
}

/// The copy one stretch of a window contributes.
///
/// Every term is re-based and none is the number nearest to hand, which is what
/// makes this worth its own function:
///
/// * The **source** starts at `offset + head`, not `offset`. A bound range is
///   rounded out to the import's granularity, and `head` is what that rounding
///   added in front of the byte the guest actually named. Reading from `offset`
///   would start the stretch up to a granule early — the whole window shifted,
///   which is a wrong draw and not a failed one.
/// * `skip` then walks from that byte to the window's own first byte inside this
///   stretch, because a stretch is positioned against the whole allocation and
///   the window is a sub-range of it.
/// * The **size** is the clipped length, never `bound_len()`, which is the
///   granularity rounding at the other end: copying it would read guest bytes
///   past the window and write them past the destination's own end.
/// * The **destination** is the stretch's offset *within the window*, which is
///   the one thing a consumer may not compute for itself.
///
/// [`super::types::WindowStretch`] is the only producer of the last three, so
/// there is no second spelling of this arithmetic to drift from it.
fn gather_region(
    bound: &super::host_ram::BoundGuestRam,
    stretch: &super::types::WindowStretch<'_>,
) -> vk::BufferCopy {
    vk::BufferCopy::default()
        .src_offset(bound.offset + bound.head + stretch.skip)
        .dst_offset(stretch.window_offset)
        .size(stretch.len)
}

/// A check that sent a guest buffer span back to the CPU gather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferImportDecline {
    /// The compute storage span does not start at the offset alignment queried
    /// from this device.
    BindOffsetAlignment { offset: u64, align: u64 },
    /// The window's stretches did not add up to the window. A healthy zero:
    /// `references_for_runs` tiles exactly, so a firing means the runs and the
    /// length reached here from different windows.
    GatherShort { covered: u64, want: u64 },
}

impl reims_vgpu_observe::Decline for BufferImportDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::BindOffsetAlignment { .. } => "buffer_import_bind_offset_alignment",
            Self::GatherShort { .. } => "buffer_gather_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::BindOffsetAlignment { offset, align } => {
                vec![("offset", offset.to_string()), ("align", align.to_string())]
            }
            Self::GatherShort { covered, want } => {
                vec![("covered", covered.to_string()), ("want", want.to_string())]
            }
        }
    }
}

reims_vgpu_observe::decline::decline_display!(BufferImportDecline);

/// Where the buffer half of a guest-sourced sampled upload came from.
///
/// The two arms are the same `vkCmdCopyBufferToImage` over a different buffer,
/// and the whole difference is whether the CPU moved the texels first.
pub(super) enum GuestTexels {
    /// The buffer **is** the guest's pages, reached through the host-pointer
    /// import over their RAMBlock. Nothing copied them into it: the GPU reads
    /// the bytes where the guest wrote them, and `offset` is where the first
    /// texel sits inside that import.
    ///
    /// The read happens when the command buffer executes rather than when it is
    /// recorded, which is later than the guest's fence would otherwise allow —
    /// see [`crate::engine::quiesce_guest_reads`], which is
    /// what makes that legal.
    Imported { buffer: vk::Buffer, offset: u64 },
    /// The GPU assembled the window out of the import: one `vkCmdCopyBuffer` per
    /// guest stretch into a device-local slot, recorded ahead of the pass, and
    /// the buffer→image copy then names that slot.
    ///
    /// This is also the required snapshot when the destination image aliases
    /// the guest pages: the first copy completes before the image is
    /// transitioned and written, so the second copy never reads and writes
    /// overlapping memory.
    Gathered(BufferSlot),
    /// The CPU packed the texels into a pooled staging span, because this host
    /// could not reach the pages (no `VK_EXT_external_memory_host`, the rail
    /// switched off, a driver that declined the pointer, or no retained page
    /// references) or a direct copy could not name their offset. Always
    /// available, which is why the import may decline.
    Scratch(BufferSlot),
}

impl GuestTexels {
    pub(super) fn buffer(&self) -> vk::Buffer {
        match self {
            Self::Imported { buffer, .. } => *buffer,
            Self::Gathered(slot) | Self::Scratch(slot) => slot.buffer,
        }
    }

    pub(super) fn offset(&self) -> u64 {
        match self {
            Self::Imported { offset, .. } => *offset,
            // Both start at the beginning of a pooled slot the window was
            // assembled into, so the first texel is byte zero.
            Self::Gathered(_) | Self::Scratch(_) => 0,
        }
    }

    pub(super) fn is_imported(&self) -> bool {
        matches!(self, Self::Imported { .. })
    }
}

/// Every Vulkan write scope that can modify imported guest memory.
///
/// A packed guest allocation and its RAMBlock parent are distinct Vulkan
/// objects over the same physical bytes. Resource barriers on either object do
/// not publish writes made through the other, so every later imported-memory
/// consumer takes one global memory dependency over all producer forms.
pub(super) fn imported_guest_write_stage() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::HOST
        | vk::PipelineStageFlags::TRANSFER
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::ALL_GRAPHICS
}

pub(super) fn imported_guest_write_access() -> vk::AccessFlags {
    vk::AccessFlags::HOST_WRITE
        | vk::AccessFlags::TRANSFER_WRITE
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImportedGuestVisibility {
    /// Only guest CPU writes can precede these imported reads.
    HostOnly,
    /// An outstanding GPU write names at least one page this operation reads.
    GpuOverlap,
    /// An imported read or outstanding write lacks a comparable page identity.
    GpuUnknown,
}

impl ImportedGuestVisibility {
    pub(super) fn includes_gpu_writes(self) -> bool {
        !matches!(self, Self::HostOnly)
    }
}

pub(super) fn imported_guest_visibility(
    pages: &[Option<reims_vgpu_memory::GuestPageSet>],
) -> ImportedGuestVisibility {
    if !super::guest_writes_outstanding() {
        return ImportedGuestVisibility::HostOnly;
    }
    if pages.is_empty() || pages.iter().any(|pages| pages.is_none()) {
        return ImportedGuestVisibility::GpuUnknown;
    }
    let reach = super::guest_writes_reaching_sets(pages.iter().filter_map(Option::as_ref));
    match reach {
        reims_vgpu_core::GuestWriteReach::Overlap => ImportedGuestVisibility::GpuOverlap,
        reims_vgpu_core::GuestWriteReach::Disjoint => ImportedGuestVisibility::HostOnly,
        reims_vgpu_core::GuestWriteReach::Unnamed => ImportedGuestVisibility::GpuUnknown,
    }
}

pub(super) fn note_imported_guest_visibility(
    counters: &EngineCounters,
    visibility: ImportedGuestVisibility,
) {
    let counter = match visibility {
        ImportedGuestVisibility::HostOnly => &counters.guest_visibility_host_only,
        ImportedGuestVisibility::GpuOverlap => &counters.guest_visibility_gpu_overlap,
        ImportedGuestVisibility::GpuUnknown => &counters.guest_visibility_gpu_unknown,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn imported_guest_read_stage(
    visibility: ImportedGuestVisibility,
) -> vk::PipelineStageFlags {
    if visibility.includes_gpu_writes() {
        imported_guest_write_stage()
    } else {
        vk::PipelineStageFlags::HOST
    }
}

fn guest_buffer_physical_pages(
    content: &BufferContent,
) -> Option<Option<&reims_vgpu_memory::GuestPageSet>> {
    match content {
        BufferContent::Bytes(_) => None,
        BufferContent::GuestRuns(source) => Some(source.physical_pages.as_ref()),
    }
}

fn imported_target_needs_visibility(has_imported_target: bool, continues_open_pass: bool) -> bool {
    has_imported_target && !continues_open_pass
}

/// Make writes through any alias of imported guest memory visible to a GPU
/// consumer.
///
/// Guest CPU writes are host writes from Vulkan's point of view even though
/// they were not issued through this crate. The imported memory type is
/// host-coherent, so no mapped-range flush is owed; the execution and memory
/// dependency is still required before a transfer or shader reads the bytes.
pub(super) fn imported_guest_read_barrier(
    dst_access: vk::AccessFlags,
    visibility: ImportedGuestVisibility,
) -> vk::MemoryBarrier<'static> {
    let src_access = if visibility.includes_gpu_writes() {
        imported_guest_write_access()
    } else {
        vk::AccessFlags::HOST_WRITE
    };
    vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

/// Release a color-attachment Store made through an imported guest image to
/// the host readers of those same pages.
fn imported_guest_attachment_release_barrier() -> vk::MemoryBarrier<'static> {
    vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
}

/// Attachment content sources that belong to this serialized encoder segment.
///
/// Metal's load action and its source execute once, when the encoder begins.
/// Continuation requests repeat the encoder's declaration so the backend can
/// reconstruct state, but they do not repeat its begin operation. Keeping that
/// distinction in one view prevents the pass key, batching decision, and copy
/// recorder from independently deciding whether a carried seed is live.
#[derive(Clone, Copy)]
struct SegmentLoadSources<'a> {
    cpu: Option<&'a [u8]>,
    guest: Option<&'a super::types::GuestTargetSeed>,
    resident: Option<&'a super::types::TargetIdentity>,
}

impl<'a> SegmentLoadSources<'a> {
    fn for_request(req: &'a DrawRequest) -> Self {
        if req.continues_render_pass {
            return Self {
                cpu: None,
                guest: None,
                resident: None,
            };
        }
        Self {
            cpu: req.target_rgba8.as_deref().map(Vec::as_slice),
            guest: req.target_guest.as_ref().and_then(|target| target.seed()),
            resident: req.seed_from_target.as_ref(),
        }
    }

    fn has_seed(self) -> bool {
        self.cpu.is_some() || self.guest.is_some() || self.resident.is_some()
    }
}

/// Vulkan load operation for one invocation of a serialized render encoder.
///
/// The guest load action applies when the encoder begins. If this backend has
/// to begin another Vulkan pass while executing a continuation segment, that
/// split is an implementation detail: the earlier segment's attachment writes
/// are now the load source and must be preserved. Reapplying `Clear` or
/// `DontCare` at that boundary would execute the guest's begin operation twice.
fn color_load_for_segment(
    action: super::types::ColorLoadAction,
    continues_render_pass: bool,
    has_load_source: bool,
) -> ColorLoadKey {
    if continues_render_pass {
        return ColorLoadKey::Load;
    }
    match action {
        super::types::ColorLoadAction::DontCare => ColorLoadKey::DontCare,
        _ if has_load_source => ColorLoadKey::Load,
        _ => ColorLoadKey::Clear,
    }
}

enum PreparedSampled {
    Null {
        binding: u32,
        array_element: u32,
    },
    Upload {
        binding: u32,
        array_element: u32,
        image: SampledSlot,
        staging: BufferSlot,
        volume: bool,
        layers: u32,
    },
    /// Guest source: one buffer→image copy fills the sampled image out of the
    /// guest's own texel bytes. [`GuestTexels`] says how far those bytes had to
    /// travel to become a buffer the copy can name.
    ///
    /// No owned CPU byte buffer exists either way, so nothing can fingerprint
    /// this content across execution units. The copied image remains transient;
    /// only another draw in this command buffer may reuse it, and the memo is
    /// cleared when the buffer seals or records a guest-page write.
    GuestGather {
        binding: u32,
        array_element: u32,
        image: SampledSlot,
        source: GuestTexels,
        /// `bufferRowLength` for the buffer→image copy (0 = tight rows).
        row_length_texels: u32,
        /// Typed inter-subresource stride when the guest declaration carries
        /// one. `None` is the legacy consecutive-plane transfer form.
        layout: Option<reims_vgpu_memory::GuestImageLayout>,
        /// Complete allocation/view coordinates for a typed guest image. The
        /// legacy run source above has no allocation declaration and leaves
        /// this absent.
        allocation_copy: Option<GuestAllocationCopy>,
        volume: bool,
        layers: u32,
        /// Exact guest window and sampled-view identity. The pool publishes it
        /// only after the corresponding copy has been recorded.
        reuse: Box<super::pools::CbSampledGuest>,
    },
    Cached {
        binding: u32,
        array_element: u32,
        image: SampledSlot,
    },
    Resident {
        binding: u32,
        array_element: u32,
        identity: super::types::TargetIdentity,
        image: vk::Image,
        view: vk::ImageView,
        access: super::pools::ResidentAccess,
        next_access: super::pools::ResidentAccess,
        /// Levels `image` carries, and therefore the range this draw's own
        /// transition has to name. One on every resident but a guest-alias
        /// image over a mipmapped guest allocation, where a level-zero-only
        /// barrier would leave the tail levels in `UNDEFINED`.
        levels: u32,
        /// The birth copy an image aliasing guest pages owes before anything
        /// reads it, present only on the bind that finds one owed. `access`
        /// then describes what recording that copy leaves behind rather than
        /// where the image is now — see
        /// [`super::pools::AliasMaterialization`].
        materialize: Option<super::pools::AliasMaterialization>,
    },
    /// Stable pre-draw contents for a sampled attachment. Binding the live
    /// attachment for both sampling and rendering only makes the image-layout
    /// combination legal; it does not establish the fragment ordering needed
    /// when the sampled and written regions overlap.
    Snapshot {
        binding: u32,
        array_element: u32,
        identity: super::types::TargetIdentity,
        source_image: vk::Image,
        source_access: super::pools::ResidentAccess,
        next_access: super::pools::ResidentAccess,
        image: SampledSlot,
        timing: SnapshotTiming,
    },
}

#[derive(Clone, Debug)]
struct GuestAllocationCopy {
    allocation: reims_vgpu_memory::GuestImageAllocationLayout,
    view: reims_vgpu_memory::GuestImageViewRange,
    transfer_source_offset: u64,
    bytes_per_texel: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SnapshotTiming {
    Prior,
    Clear([f32; 4]),
    AfterPrimarySeed,
    Undefined,
}

impl SnapshotTiming {
    fn route(self) -> &'static str {
        match self {
            Self::Prior => "sampled_self_timing_prior",
            Self::Clear(_) => "sampled_self_timing_clear",
            Self::AfterPrimarySeed => "sampled_self_timing_seed",
            Self::Undefined => "sampled_self_timing_dontcare",
        }
    }
}

impl PreparedSampled {
    fn binding(&self) -> u32 {
        match self {
            Self::Null { binding, .. }
            | Self::Upload { binding, .. }
            | Self::Cached { binding, .. }
            | Self::Resident { binding, .. }
            | Self::Snapshot { binding, .. }
            | Self::GuestGather { binding, .. } => *binding,
        }
    }

    fn array_element(&self) -> u32 {
        match self {
            Self::Null { array_element, .. }
            | Self::Upload { array_element, .. }
            | Self::Cached { array_element, .. }
            | Self::Resident { array_element, .. }
            | Self::Snapshot { array_element, .. }
            | Self::GuestGather { array_element, .. } => *array_element,
        }
    }

    fn view(&self) -> vk::ImageView {
        match self {
            Self::Null { .. } => vk::ImageView::null(),
            Self::Upload { image, .. } => image.view,
            Self::Cached { image, .. } => image.view,
            Self::Resident { view, .. } => *view,
            Self::Snapshot { image, .. } => image.view,
            Self::GuestGather { image, .. } => image.view,
        }
    }

    fn descriptor_layout(&self) -> vk::ImageLayout {
        match self {
            Self::Resident { next_access, .. } => next_access.layout(),
            _ => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}

/// Which colour attachment this sampled resource aliases.
///
/// Aliasing always selects a pre-draw snapshot. An attachment-feedback layout
/// makes a live image legal to bind for both uses, but it does not order reads
/// against overlapping fragment writes. The decoded contract carries no
/// disjoint-region proof, so host capability cannot turn the live target into
/// an equivalent source.
fn sampled_attachment_slot(
    req: &DrawRequest,
    resource: &super::types::SampledImageResource,
) -> Option<super::types::AttachmentSlot> {
    let identity = match &resource.source {
        SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => identity,
        SampledSource::Null
        | SampledSource::Bytes(_)
        | SampledSource::GuestImage(..)
        | SampledSource::GuestRuns(..) => return None,
    };
    req.attachment_slot(identity)
}

/// `vkCmdCopyBufferToImage` requires `bufferOffset` to be a multiple of 4 and of
/// the format's texel block size. 16 is the largest uncompressed block in core
/// Vulkan and the larger of the two BC block sizes, so one check covers every
/// format the sampled pool can produce without the arm having to know which one
/// it is holding. A guest window whose first texel sits at any other offset
/// takes the CPU gather, which has no such rule.
const GUEST_IMPORT_COPY_OFFSET_ALIGN: u64 = 16;

/// Prepare a guest texel window as a buffer-to-image copy source.
///
/// A single-stretch window binds in place. A scattered window is gathered by
/// the GPU into a device-local slot. `None` means the host cannot import the guest
/// pages or the window cannot be represented, leaving the exact CPU fallback
/// to the caller.
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
pub(super) unsafe fn prepare_guest_texel_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    src: &super::types::GuestRunSource,
    gathers: &mut Vec<PendingGuestGather>,
) -> Result<Option<GuestTexels>, DrawError> {
    if !ctx.caps.host_pointer.is_available() {
        return Ok(None);
    }
    let transfer = src.transfer_plan();
    // One stretch binds in place; anything longer is assembled by the GPU, the
    // same two arms the buffer rail has.
    //
    // This rail used to stop at the first of them, on the grounds that its whole
    // traffic for a boot (211 gathers, 254 MB) was two orders of magnitude below
    // what the buffer rail moved in a second. Both halves of that were wrong.
    // The guest backs a surface in 16 KiB physically-contiguous granules, so a
    // sampled window is essentially never one stretch and the arm was
    // unreachable — a driven Safari drag read 4 imports against 4514 CPU
    // gathers moving 10.8 GB, while the buffer rail on the same boot imported
    // 322 303 windows and gathered none on the CPU. It is not a small rail; it
    // was a rail whose only zero-copy arm could not be taken.
    if let Some(stretch) = transfer.direct() {
        let bound = match unsafe { pools.bind_guest_ram(ctx, stretch.guest) } {
            Ok(bound) => bound,
            Err(inner) => {
                reims_vgpu_observe::Emit::decline("vk_sampled_import", &inner).fail_once(0);
                return Ok(None);
            }
        };
        // As on the buffer rail: the buffer spans the RAMBlock, so the first
        // texel sits at the bound range's start, plus the granularity widening,
        // plus the plane's own offset inside the allocation. A mapped sampled
        // plane names the whole allocation as its one stretch and carries the
        // plane offset in `source_offset`, so dropping that term reads every
        // such texture from the start of the allocation.
        let offset = bound.offset + bound.head + stretch.skip;
        if !offset.is_multiple_of(GUEST_IMPORT_COPY_OFFSET_ALIGN) {
            reims_vgpu_observe::Emit::decline(
                "vk_sampled_import",
                &SampledImportDecline::CopyOffsetAlignment { offset },
            )
            .fail_once(offset % GUEST_IMPORT_COPY_OFFSET_ALIGN);
            return Ok(None);
        }
        return Ok(Some(GuestTexels::Imported {
            buffer: bound.buffer,
            offset,
        }));
    }
    // Plan before acquiring, so a window that turns out not to be gatherable
    // does not take a destination slot out of the pool to abandon it.
    let Some(stretches) = transfer.stretches() else {
        return Ok(None);
    };
    let mut sources: Vec<(vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
    let mut covered = 0u64;
    for stretch in stretches {
        let bound = match unsafe { pools.bind_guest_ram(ctx, stretch.guest) } {
            Ok(bound) => bound,
            Err(inner) => {
                reims_vgpu_observe::Emit::decline("vk_sampled_import", &inner).fail_once(0);
                return Ok(None);
            }
        };
        let copy = gather_region(&bound, &stretch);
        covered = covered.saturating_add(copy.size);
        super::group_by_buffer(&mut sources, bound.buffer, copy);
    }
    // The runs tile the window exactly, so this holds by construction — but a
    // short gather would hand the draw an image whose tail is whatever the
    // previous user of the slot left there, which is wrong pixels rather than
    // slow ones. Checked here because this is the last place that can see it.
    if covered != src.total_len {
        reims_vgpu_observe::Emit::decline(
            "vk_sampled_import",
            &SampledImportDecline::GatherShort {
                covered,
                want: src.total_len,
            },
        )
        .fail_once(src.total_len);
        return Ok(None);
    }
    let slot = unsafe {
        pools.acquire_guest_gather(
            ctx,
            src.total_len,
            vk::BufferUsageFlags::TRANSFER_SRC,
            counters,
        )?
    };
    gathers.push(PendingGuestGather {
        dst: slot.buffer,
        sources,
    });
    Ok(Some(GuestTexels::Gathered(slot)))
}

/// A check that sent a guest-sourced sampled bind back to the CPU gather before
/// the import itself was ever asked.
///
/// Separate from [`super::host_ram::HostRamDecline`] because these are
/// properties of the *copy* this arm wants to record, not of the import or the
/// device: the same span would bind fine for a caller that named it differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampledImportDecline {
    /// The first texel does not sit at an offset `vkCmdCopyBufferToImage` can
    /// name. See [`GUEST_IMPORT_COPY_OFFSET_ALIGN`].
    CopyOffsetAlignment { offset: u64 },
    /// The window's stretches did not tile it, so a GPU gather would have left
    /// the tail of the destination holding the previous user's bytes. Fails
    /// visible because the CPU gather that follows produces a correct image and
    /// so nothing else would say the run list was malformed.
    GatherShort { covered: u64, want: u64 },
}

impl reims_vgpu_observe::Decline for SampledImportDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CopyOffsetAlignment { .. } => "sampled_import_copy_offset_alignment",
            Self::GatherShort { .. } => "sampled_import_gather_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CopyOffsetAlignment { offset } => vec![("offset", offset.to_string())],
            Self::GatherShort { covered, want } => {
                vec![("covered", covered.to_string()), ("want", want.to_string())]
            }
        }
    }
}

reims_vgpu_observe::decline::decline_display!(SampledImportDecline);

/// Shared validation for a draw-time buffer's content source. A `GuestRuns`
/// span must be internally consistent: the requested subrange fits in `runs`,
/// the span is non-empty, and `row_length_texels` is 0 (row strides are a
/// texture concept — buffers gather a flat byte span).
#[derive(Clone, Copy)]
enum BufferValidationRole {
    Vertex,
    Storage,
    Index,
}

fn validate_buffer_content(
    content: &BufferContent,
    role: BufferValidationRole,
    resource_index: u32,
) -> Result<(), DrawError> {
    let BufferContent::GuestRuns(src) = content else {
        return Ok(());
    };
    if src.row_length_texels != 0 {
        let decline = match role {
            BufferValidationRole::Vertex => DrawValidationDecline::VertexGuestRunsRowStride {
                location: resource_index,
                row_length_texels: src.row_length_texels,
            },
            BufferValidationRole::Storage => DrawValidationDecline::StorageGuestRunsRowStride {
                binding: resource_index,
                row_length_texels: src.row_length_texels,
            },
            BufferValidationRole::Index => DrawValidationDecline::IndexGuestRunsRowStride {
                row_length_texels: src.row_length_texels,
            },
        };
        return Err(DrawError::DrawValidation(decline));
    }
    let sum: u64 = src.runs.iter().map(|r| r.len).sum();
    let covered = src.source_offset.checked_add(src.total_len);
    if src.total_len == 0 || covered.is_none_or(|end| end > sum) {
        let decline = match role {
            BufferValidationRole::Vertex => DrawValidationDecline::VertexGuestRunsCoverage {
                location: resource_index,
                covered: sum,
                declared: src.total_len,
            },
            BufferValidationRole::Storage => DrawValidationDecline::StorageGuestRunsCoverage {
                binding: resource_index,
                covered: sum,
                declared: src.total_len,
            },
            BufferValidationRole::Index => DrawValidationDecline::IndexGuestRunsCoverage {
                covered: sum,
                declared: src.total_len,
            },
        };
        return Err(DrawError::DrawValidation(decline));
    }
    Ok(())
}

pub(crate) struct NativeRenderProgram {
    pub(crate) vertex: Arc<crate::m2v_cache::ShaderVariant>,
    pub(crate) fragment: Arc<crate::m2v_cache::ShaderVariant>,
}

fn validate_guest_sampled_source(
    image: &super::types::SampledImageResource,
    source: &reims_vgpu_memory::GuestImageSource,
    texel: usize,
) -> Result<(), DrawError> {
    let src = &source.transfer;
    if !source.allocation.is_vulkan_mip_chain(texel as u64) {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::GuestSampleAllocationInvalid {
                binding: image.binding,
                mip_levels: source.allocation.mips.len(),
                bytes_per_texel: texel as u64,
            },
        ));
    }
    if !source.view.fits(&source.allocation) {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::GuestSampleViewRangeInvalid {
                binding: image.binding,
                view: source.view,
                mip_levels: source.allocation.mips.len(),
            },
        ));
    }
    let layout = source.allocation.mips[source.view.base_mip_level as usize].layout;
    let invalid_view = || {
        DrawError::DrawValidation(DrawValidationDecline::GuestSampleViewRangeInvalid {
            binding: image.binding,
            view: source.view,
            mip_levels: source.allocation.mips.len(),
        })
    };
    let allocation_is_array = layout.is_arrayed();
    // A cube's six faces are array slices in the same order on both sides, so
    // it matches an arrayed allocation exactly as a 2-D array does. See
    // [`SampledImageResource::planes_are_array_slices`].
    let array_shape_matches = if image.planes_are_array_slices() {
        allocation_is_array && image.layers == source.view.array_layer_count
    } else {
        source.view.array_layer_count == 1 && image.layers == 1
    };
    let layout_matches = layout.width() == image.width
        && layout.height() == image.height
        && match layout {
            reims_vgpu_memory::GuestImageLayout::D1 { .. }
            | reims_vgpu_memory::GuestImageLayout::D1Array { .. } => {
                image.one_dim && !image.volume && array_shape_matches
            }
            reims_vgpu_memory::GuestImageLayout::D2 { .. }
            | reims_vgpu_memory::GuestImageLayout::D2Array { .. } => {
                !image.one_dim && !image.volume && array_shape_matches
            }
            reims_vgpu_memory::GuestImageLayout::D3 { depth, .. } => {
                !image.one_dim
                    && !image.arrayed
                    && image.volume
                    && image.layers == depth
                    && source.view.base_array_layer == 0
                    && source.view.array_layer_count == 1
            }
        };
    if !layout_matches || image.multisampled {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::GuestSampleLayoutMismatch {
                binding: image.binding,
                layout,
                width: image.width,
                height: image.height,
                layers: image.layers,
                arrayed: image.arrayed,
                volume: image.volume,
                one_dim: image.one_dim,
                multisampled: image.multisampled,
            },
        ));
    }
    let texel = texel as u64;
    let transfer_end = src
        .source_offset
        .checked_add(src.total_len)
        .ok_or_else(invalid_view)?;
    let mut required_end = src.source_offset;
    let view_end = source
        .view
        .base_mip_level
        .checked_add(source.view.mip_level_count)
        .ok_or_else(invalid_view)?;
    for mip_level in source.view.base_mip_level..view_end {
        let mip = source.allocation.mips[mip_level as usize];
        if let Some(memory) = source.direct.as_ref() {
            let Some(mip_backing) = mip.plane_in(memory.backing) else {
                return Err(invalid_view());
            };
            if mip_backing
                .visible_image_window(mip.layout, texel)
                .is_none()
            {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::GuestSampleLayoutInvalid {
                        binding: image.binding,
                        layout: mip.layout,
                        row_pitch: mip.row_pitch,
                        bytes_per_texel: texel,
                    },
                ));
            }
        }
        let (layer_displacement, viewed_layout) = match mip.layout {
            reims_vgpu_memory::GuestImageLayout::D1Array {
                width, array_pitch, ..
            } => (
                u64::from(source.view.base_array_layer)
                    .checked_mul(array_pitch)
                    .ok_or_else(invalid_view)?,
                reims_vgpu_memory::GuestImageLayout::D1Array {
                    width,
                    layers: source.view.array_layer_count,
                    array_pitch,
                },
            ),
            reims_vgpu_memory::GuestImageLayout::D2Array {
                width,
                height,
                array_pitch,
                ..
            } => (
                u64::from(source.view.base_array_layer)
                    .checked_mul(array_pitch)
                    .ok_or_else(invalid_view)?,
                reims_vgpu_memory::GuestImageLayout::D2Array {
                    width,
                    height,
                    layers: source.view.array_layer_count,
                    array_pitch,
                },
            ),
            other => (0, other),
        };
        // `src.source_offset` below counts from the same guest resource the mip
        // chain does, so this comparison stays in resource coordinates and never
        // reaches the allocation.
        let relative = mip
            .resource_relative_offset
            .checked_add(layer_displacement)
            .ok_or_else(invalid_view)?;
        let end = relative
            .checked_add(
                viewed_layout
                    .visible_span(mip.row_pitch, texel)
                    .ok_or_else(invalid_view)?,
            )
            .ok_or_else(invalid_view)?;
        if relative < src.source_offset {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::GuestSampleLength {
                    binding: image.binding,
                    actual: src.total_len,
                    expected: end.saturating_sub(src.source_offset),
                },
            ));
        }
        required_end = required_end.max(end);
    }
    if required_end > transfer_end {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::GuestSampleLength {
                binding: image.binding,
                actual: src.total_len,
                expected: required_end.saturating_sub(src.source_offset),
            },
        ));
    }
    let sum = src
        .runs
        .iter()
        .try_fold(0_u64, |sum, run| sum.checked_add(run.len))
        .ok_or_else(|| {
            DrawError::DrawValidation(DrawValidationDecline::GuestSampleCoverageOverflow {
                binding: image.binding,
                runs: src.runs.len(),
            })
        })?;
    let covered = src.source_offset.checked_add(src.total_len);
    if src.total_len == 0 || src.runs.is_empty() || covered.is_none_or(|end| end > sum) {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::GuestSampleCoverage {
                binding: image.binding,
                covered: sum,
                declared: src.total_len,
                runs: src.runs.len(),
            },
        ));
    }
    Ok(())
}

pub(crate) fn validate_v1(req: &DrawRequest) -> Result<(), DrawError> {
    if req.width == 0 || req.height == 0 {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::ZeroTargetGeometry {
                width: req.width,
                height: req.height,
            },
        ));
    }
    let (minimum_width, minimum_height) = req.minimum_attachment_extent();
    if minimum_width == 0 || minimum_height == 0 {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::ZeroTargetGeometry {
                width: minimum_width,
                height: minimum_height,
            },
        ));
    }
    if let Some(width) = req.render_target_extent.width {
        if width.get() > minimum_width {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::RenderTargetExtentExceedsAttachment {
                    axis: "width",
                    requested: width.get(),
                    limit: minimum_width,
                },
            ));
        }
    }
    if let Some(height) = req.render_target_extent.height {
        if height.get() > minimum_height {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::RenderTargetExtentExceedsAttachment {
                    axis: "height",
                    requested: height.get(),
                    limit: minimum_height,
                },
            ));
        }
    }
    if req.program.vertex.id.get() == 0 {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::MissingVertexProgram,
        ));
    }
    if req.program.fragment.id.get() == 0 {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::MissingFragmentProgram,
        ));
    }
    // Every viewport, not just the first: a NaN in slot 3 reaches
    // `vkCmdSetViewport` exactly as one in slot 0 does, and the driver's
    // behaviour on it is undefined either way.
    for vp in &req.viewports {
        if !vp.x.is_finite()
            || !vp.y.is_finite()
            || !vp.width.is_finite()
            || !vp.height.is_finite()
            || !vp.min_depth.is_finite()
            || !vp.max_depth.is_finite()
        {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonFiniteViewport,
            ));
        }
        if vp.width <= 0.0 || vp.height <= 0.0 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonPositiveViewport {
                    width_bits: vp.width.to_bits(),
                    height_bits: vp.height.to_bits(),
                },
            ));
        }
    }
    if draw_uses_blend_constants(req) && req.blend_constants.iter().any(|c| !c.is_finite()) {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::NonFiniteBlendConstants,
        ));
    }
    if let Some(target) = &req.target_rgba8 {
        // The seed is one tightly-packed RGBA8 slice of the target, and the
        // length is checked rather than taken. `w as usize * h as usize` widens
        // its operands, which reads as safe and is — but only just: two u32
        // maxima multiply to a hair under u64::MAX, so the bytes-per-texel is a
        // third factor that overflows. A refusal rather than a clamp, because
        // this length is what the next line compares the buffer against and a
        // wrapped one would let a short buffer match.
        let Some(expected) = reims_vgpu_protocol::tight_image_bytes(
            req.width,
            req.height,
            reims_vgpu_protocol::TexelLayout::Rgba8.bytes_per_texel() as usize,
        ) else {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::UnrepresentableImageBytes {
                    width: req.width,
                    height: req.height,
                    layers: 1,
                    bytes_per_texel: reims_vgpu_protocol::TexelLayout::Rgba8.bytes_per_texel(),
                },
            ));
        };
        if target.len() != expected {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetSeedLength {
                    actual: target.len(),
                    expected,
                },
            ));
        }
    }
    if let Some(seed) = req.target_guest.as_ref().and_then(|target| target.seed()) {
        let target_layout = req
            .target_identity
            .as_ref()
            .map(TargetIdentity::resident_layout)
            .unwrap_or(reims_vgpu_protocol::TexelLayout::Rgba8);
        if seed.format != target_layout {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetGuestSeedFormat {
                    source: super::super::translate::pixel::vk_texel_layout(seed.format),
                    target: super::super::translate::pixel::vk_texel_layout(target_layout),
                },
            ));
        }
        let texel = seed.format.bytes_per_texel();
        let tight_row =
            (req.width as usize)
                .checked_mul(texel as usize)
                .ok_or(DrawError::DrawValidation(
                    DrawValidationDecline::UnrepresentableImageBytes {
                        width: req.width,
                        height: req.height,
                        layers: 1,
                        bytes_per_texel: texel,
                    },
                ))?;
        let stride = if seed.source.row_length_texels == 0 {
            tight_row
        } else {
            (seed.source.row_length_texels as usize)
                .checked_mul(texel as usize)
                .ok_or(DrawError::DrawValidation(
                    DrawValidationDecline::UnrepresentableImageBytes {
                        width: req.width,
                        height: req.height,
                        layers: 1,
                        bytes_per_texel: texel,
                    },
                ))?
        };
        if stride < tight_row {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetGuestSeedRowStride { stride, tight_row },
            ));
        }
        let expected = (req.height.saturating_sub(1) as usize)
            .checked_mul(stride)
            .and_then(|prefix| prefix.checked_add(tight_row))
            .ok_or(DrawError::DrawValidation(
                DrawValidationDecline::UnrepresentableImageBytes {
                    width: req.width,
                    height: req.height,
                    layers: 1,
                    bytes_per_texel: texel,
                },
            ))?;
        if seed.source.total_len != expected as u64 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetGuestSeedLength {
                    actual: seed.source.total_len,
                    expected,
                },
            ));
        }
        let covered = seed.source.runs.iter().map(|run| run.len).sum();
        if covered != seed.source.total_len {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetGuestSeedCoverage {
                    covered,
                    declared: seed.source.total_len,
                },
            ));
        }
        if req.target_rgba8.is_some() || req.load_from_target || req.seed_from_target.is_some() {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedConflictsGuestSeed,
            ));
        }
    }
    if let Some(seed_identity) = &req.seed_from_target {
        if req.target_identity.is_none() {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedMissingTargetIdentity,
            ));
        }
        if req.target_rgba8.is_some()
            || req
                .target_guest
                .as_ref()
                .is_some_and(|target| target.seed().is_some())
        {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedConflictsCpuSeed,
            ));
        }
        if req.load_from_target {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedConflictsLoadFromTarget,
            ));
        }
        if req.target_identity.as_ref() == Some(seed_identity) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedEqualsTarget,
            ));
        }
        if req.sampled_images.iter().any(|img| match &img.source {
            SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => {
                identity == seed_identity
            }
            SampledSource::Null
            | SampledSource::Bytes(_)
            | SampledSource::GuestImage(..)
            | SampledSource::GuestRuns(..) => false,
        }) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedAlsoSampled,
            ));
        }
    }
    let no_vertex_fetch = draw_has_no_invocations(req);
    let last_record: u32 = match &req.indexed {
        Some(indexed) => {
            let need = indexed.index_count as usize * indexed.index_type.byte_size();
            if indexed.content.len() < need {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::IndexBytesShort {
                        actual: indexed.content.len(),
                        expected: need,
                    },
                ));
            }
            validate_buffer_content(&indexed.content, BufferValidationRole::Index, 0)?;
            // Indexed vertex addresses are data-dependent and the API does not
            // require a CPU scan before submission. The enabled Vulkan robust
            // buffer contract bounds any vertex fetch outside the retained
            // buffer; zero is enough here to validate the first element's
            // structural offset and stride without reading the index resource.
            0
        }
        None => req.vertex_count.saturating_sub(1),
    };
    let mut bindings = BTreeSet::new();
    let mut vertex_locations = BTreeSet::new();
    let mut vertex_bindings = BTreeSet::new();
    for attribute in &req.vertex_attributes {
        if !vertex_locations.insert(attribute.location) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateVertexLocation {
                    location: attribute.location,
                },
            ));
        }
        if !vertex_bindings.insert(attribute.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateVertexBinding {
                    binding: attribute.binding,
                },
            ));
        }
        // The rate is only a rule under the step functions that consume it.
        // `Constant` is fetched once for the whole draw and pairs with a rate of
        // zero — that is the spelling Metal requires, the decoder deliberately
        // preserves it, and this binding's divisor is 0 whatever the rate says.
        // Asking `rate == 0` alone declined that guest's draw outright, for a
        // field nothing downstream reads. `reims_vgpu_protocol` owns the pair.
        if !reims_vgpu_protocol::step_rate_in_contract(
            attribute.step_function.mtl_ordinal(),
            attribute.step_rate,
        ) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::ZeroVertexStepRate {
                    location: attribute.location,
                },
            ));
        }
        let format_size = attribute.format.byte_size();
        if attribute.stride < format_size {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexStrideTooSmall {
                    location: attribute.location,
                    stride: attribute.stride,
                    format_size,
                },
            ));
        }
        let element_end = attribute.offset.checked_add(format_size).ok_or({
            DrawError::DrawValidation(DrawValidationDecline::VertexOffsetOverflow {
                location: attribute.location,
            })
        })?;
        if element_end > attribute.stride {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexElementExceedsStride {
                    location: attribute.location,
                },
            ));
        }
        let last_element = if no_vertex_fetch {
            0
        } else {
            match attribute.step_function {
                VertexStepFunction::Constant => 0,
                VertexStepFunction::PerVertex => {
                    let first_record = if req.indexed.is_some() {
                        0
                    } else {
                        req.first_vertex as usize
                    };
                    first_record.checked_add(last_record as usize).ok_or({
                        DrawError::DrawValidation(DrawValidationDecline::VertexRangeOverflow {
                            location: attribute.location,
                        })
                    })?
                }
                VertexStepFunction::PerInstance => {
                    let instance_count = req.instance_count.unwrap_or(1);
                    let relative_element = if instance_count == 0 {
                        0
                    } else {
                        (instance_count - 1) / attribute.step_rate
                    };
                    req.base_instance.checked_add(relative_element).ok_or({
                        DrawError::DrawValidation(DrawValidationDecline::InstanceRangeOverflow {
                            location: attribute.location,
                        })
                    })? as usize
                }
            }
        };
        let required = (attribute.stride as usize)
            .checked_mul(last_element)
            .and_then(|span| (attribute.offset as usize).checked_add(span))
            .and_then(|end| end.checked_add(format_size as usize))
            .ok_or({
                DrawError::DrawValidation(DrawValidationDecline::VertexByteRangeOverflow {
                    location: attribute.location,
                })
            })?;
        if !no_vertex_fetch && attribute.content.len() < required {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexDataShort {
                    location: attribute.location,
                    actual: attribute.content.len(),
                    expected: required,
                },
            ));
        }
        validate_buffer_content(
            &attribute.content,
            BufferValidationRole::Vertex,
            attribute.location,
        )?;
        // The Constant-step base-instance shift prepends a CPU prefix to the
        // bytes at prepare time; a gathered guest span has no CPU bytes. The
        // runtime keeps Constant-step streams on the CPU path — reaching here
        // with a gather is a gate bug, rejected before any GPU work.
        if !no_vertex_fetch
            && attribute.step_function == VertexStepFunction::Constant
            && req.base_instance != 0
            && matches!(attribute.content, BufferContent::GuestRuns(_))
        {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::ConstantStepGuestRuns {
                    location: attribute.location,
                },
            ));
        }
    }
    for buffer in &req.storage_buffers {
        if !bindings.insert((buffer.binding, 0)) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateStorageDescriptorBinding {
                    binding: buffer.binding,
                },
            ));
        }
        validate_buffer_content(
            &buffer.content,
            BufferValidationRole::Storage,
            buffer.binding,
        )?;
    }
    for image in &req.sampled_images {
        if matches!(image.source, SampledSource::Null) {
            if image.descriptor_count == 0 || image.array_element >= image.descriptor_count {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::SampledArrayElementOutOfRange {
                        binding: image.binding,
                        element: image.array_element,
                        count: image.descriptor_count,
                    },
                ));
            }
            if !bindings.insert((image.binding, image.array_element)) {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::DuplicateSampledDescriptorBinding {
                        binding: image.binding,
                    },
                ));
            }
            continue;
        }
        if image.width == 0 || image.height == 0 || image.layers == 0 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledZeroGeometry {
                    binding: image.binding,
                    width: image.width,
                    height: image.height,
                    layers: image.layers,
                },
            ));
        }
        if (image.arrayed as u8 + image.volume as u8 + image.cube as u8) > 1 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledShapeConflict {
                    binding: image.binding,
                    arrayed: image.arrayed,
                    volume: image.volume,
                    cube: image.cube,
                },
            ));
        }
        // A 1D image (`texture1d` / `texture1d_array`) is a single row: it may
        // combine only with `arrayed` (the 1D-array case) and always has
        // height 1. `volume`/`cube` are 2D/3D shapes and cannot co-occur.
        if image.one_dim && (image.volume || image.cube || image.height != 1) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledShapeConflict {
                    binding: image.binding,
                    arrayed: image.arrayed,
                    volume: image.volume,
                    cube: image.cube,
                },
            ));
        }
        if image.cube && (image.layers != 6 || image.width != image.height) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledCubeGeometry {
                    binding: image.binding,
                    width: image.width,
                    height: image.height,
                    layers: image.layers,
                },
            ));
        }
        if !image.arrayed && !image.volume && !image.cube && image.layers != 1 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledNonArrayLayers {
                    binding: image.binding,
                    layers: image.layers,
                },
            ));
        }
        // The semantic request type only admits formats with a defined texel
        // layout, so a linear footprint is guaranteed before Vulkan sees it.
        let texel = image.format.layout().bytes_per_texel() as usize;
        // Four factors, so the widening the operands already carry is not
        // enough — see the target-seed check above for why two of them exhaust
        // a u64 on their own. `reims_vgpu_protocol` owns the checked form.
        let Some(expected) = reims_vgpu_protocol::tight_layered_image_bytes(
            image.width,
            image.height,
            image.layers,
            texel,
        ) else {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::UnrepresentableImageBytes {
                    width: image.width,
                    height: image.height,
                    layers: image.layers,
                    bytes_per_texel: texel as u32,
                },
            ));
        };
        match &image.source {
            SampledSource::Bytes(bytes) if bytes.len() != expected => {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::SampledBytesLength {
                        binding: image.binding,
                        actual: bytes.len(),
                        expected,
                    },
                ));
            }
            SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => {
                if image.arrayed || image.volume || image.cube || image.layers != 1 {
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::ResidentSampledNot2d {
                            binding: image.binding,
                        },
                    ));
                }
                if identity.width() != image.width || identity.height() != image.height {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::ResidentSampleGeometry {
                            binding: image.binding,
                            resident_width: identity.width(),
                            resident_height: identity.height(),
                            resource_width: image.width,
                            resource_height: image.height,
                        },
                    ));
                }
            }
            SampledSource::Null | SampledSource::Bytes(_) => {}
            SampledSource::GuestImage(source, _) => {
                validate_guest_sampled_source(image, source, texel)?;
            }
            SampledSource::GuestRuns(src, _) => {
                // Cubes, arrays and volumes are all ordinary consecutive image
                // planes here and the copy below consumes all of them. A cube
                // carries no ordering this source would have to describe: its
                // six faces are array slices in the same order on both sides,
                // so `layers` counts them and nothing is permuted. See
                // [`SampledImageResource::planes_are_array_slices`].
                let planes = image.layers as usize;
                let run_expected = if src.row_length_texels == 0 {
                    expected
                } else {
                    let stride = src.row_length_texels as usize * texel;
                    let tight_row = image.width as usize * texel;
                    if stride < tight_row {
                        return Err(DrawError::DrawValidation(
                            DrawValidationDecline::GuestSampleRowStride {
                                binding: image.binding,
                                stride,
                                tight_row,
                            },
                        ));
                    }
                    (planes - 1) * image.height as usize * stride
                        + (image.height as usize - 1) * stride
                        + tight_row
                };
                if src.total_len as usize != run_expected {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::GuestSampleLength {
                            binding: image.binding,
                            actual: src.total_len,
                            expected: run_expected as u64,
                        },
                    ));
                }
                let sum: u64 = src.runs.iter().map(|r| r.len).sum();
                let covered = src.source_offset.checked_add(src.total_len);
                if src.total_len == 0 || src.runs.is_empty() || covered.is_none_or(|end| end > sum)
                {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::GuestSampleCoverage {
                            binding: image.binding,
                            covered: sum,
                            declared: src.total_len,
                            runs: src.runs.len(),
                        },
                    ));
                }
            }
        }
        if image.descriptor_count == 0 || image.array_element >= image.descriptor_count {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledArrayElementOutOfRange {
                    binding: image.binding,
                    element: image.array_element,
                    count: image.descriptor_count,
                },
            ));
        }
        if !bindings.insert((image.binding, image.array_element)) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateSampledDescriptorBinding {
                    binding: image.binding,
                },
            ));
        }
    }
    for sampler in &req.samplers {
        if sampler.source == reims_vgpu_core::SamplerSource::Null {
            if !bindings.insert((sampler.binding, 0)) {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::DuplicateSamplerDescriptorBinding {
                        binding: sampler.binding,
                    },
                ));
            }
            continue;
        }
        let lod_min = sampler.lod_min_f32();
        let lod_max = sampler.lod_max_f32();
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::InvalidSamplerLod {
                    binding: sampler.binding,
                    lod_min_bits: sampler.lod_min,
                    lod_max_bits: sampler.lod_max,
                },
            ));
        }
        if !bindings.insert((sampler.binding, 0)) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateSamplerDescriptorBinding {
                    binding: sampler.binding,
                },
            ));
        }
    }
    Ok(())
}

/// Stage a host-written buffer into a freshly created sampled image and leave it
/// shader-readable.
///
/// Both sampled upload rails do exactly this: transition `UNDEFINED` →
/// `TRANSFER_DST_OPTIMAL`, one `vkCmdCopyBufferToImage`, then
/// `TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL` against both shader
/// stages. Keeping one copy means the barrier masks cannot drift apart between
/// them, which is the failure this shape invites: a missing `SHADER_READ` on one
/// rail is invisible on a driver that happens not to need it.
///
/// No HOST→TRANSFER barrier on either rail — writes the host made before
/// `vkQueueSubmit` are automatically visible to the device, and every staging
/// slot here is written before the submit. The guest-gather rail once opened with
/// two barriers ordering a *device-side* gather against this copy; there is no
/// device-side write to order any more.
///
/// `row_length_texels` is `VkBufferImageCopy::bufferRowLength`, where 0 means
/// "rows are tightly packed" — the CPU-origin rail always packs tightly, the
/// guest-gather rail may stride over guest row padding.
#[derive(Clone, Copy)]
struct SampledCopyGeometry {
    binding: u32,
    source_offset: u64,
    width: u32,
    height: u32,
    array_layers: u32,
    extent_depth: u32,
    row_length_texels: u32,
    guest_layout: Option<reims_vgpu_memory::GuestImageLayout>,
}

fn sampled_copy_regions(
    geometry: SampledCopyGeometry,
) -> Result<Vec<vk::BufferImageCopy>, DrawExecutionDecline> {
    let region = |buffer_offset, base_array_layer, z, region_depth| {
        vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(geometry.row_length_texels)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z })
            .image_extent(vk::Extent3D {
                width: geometry.width,
                height: geometry.height,
                depth: region_depth,
            })
    };
    let subresource_offset = |pitch: u64, subresource: u32| {
        u64::from(subresource)
            .checked_mul(pitch)
            .and_then(|relative| geometry.source_offset.checked_add(relative))
            .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
                binding: geometry.binding,
                source_offset: geometry.source_offset,
                pitch,
                subresource,
            })
    };
    match geometry.guest_layout {
        Some(reims_vgpu_memory::GuestImageLayout::D1Array {
            layers,
            array_pitch,
            ..
        })
        | Some(reims_vgpu_memory::GuestImageLayout::D2Array {
            layers,
            array_pitch,
            ..
        }) => (0..layers)
            .map(|layer| Ok(region(subresource_offset(array_pitch, layer)?, layer, 0, 1)))
            .collect(),
        Some(reims_vgpu_memory::GuestImageLayout::D3 {
            depth, depth_pitch, ..
        }) => (0..depth)
            .map(|z| Ok(region(subresource_offset(depth_pitch, z)?, 0, z as i32, 1)))
            .collect(),
        _ => Ok(vec![vk::BufferImageCopy::default()
            .buffer_offset(geometry.source_offset)
            .buffer_row_length(geometry.row_length_texels)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: geometry.array_layers,
            })
            .image_extent(vk::Extent3D {
                width: geometry.width,
                height: geometry.height,
                depth: geometry.extent_depth,
            })]),
    }
}

fn sampled_allocation_copy_regions(
    binding: u32,
    source_buffer_offset: u64,
    bytes_per_texel: u64,
    image_arrayed: bool,
    copy: &GuestAllocationCopy,
) -> Result<Vec<vk::BufferImageCopy>, DrawExecutionDecline> {
    let mut regions = Vec::new();
    let mip_end = copy
        .view
        .base_mip_level
        .checked_add(copy.view.mip_level_count)
        .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
            binding,
            source_offset: source_buffer_offset,
            pitch: 0,
            subresource: copy.view.base_mip_level,
        })?;
    for guest_mip_level in copy.view.base_mip_level..mip_end {
        let local_mip = guest_mip_level - copy.view.base_mip_level;
        let mip = copy.allocation.mips[guest_mip_level as usize];
        let relative = mip
            .resource_relative_offset
            .checked_sub(copy.transfer_source_offset)
            .and_then(|offset| source_buffer_offset.checked_add(offset))
            .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
                binding,
                source_offset: source_buffer_offset,
                pitch: mip.row_pitch,
                subresource: guest_mip_level,
            })?;
        let row_length = mip
            .row_pitch
            .checked_div(bytes_per_texel)
            .and_then(|texels| u32::try_from(texels).ok())
            .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
                binding,
                source_offset: source_buffer_offset,
                pitch: mip.row_pitch,
                subresource: guest_mip_level,
            })?;
        let row_length = if row_length == mip.layout.width() {
            0
        } else {
            row_length
        };
        let region = |buffer_offset, base_array_layer, z| {
            vk::BufferImageCopy::default()
                .buffer_offset(buffer_offset)
                .buffer_row_length(row_length)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: local_mip,
                    base_array_layer,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z })
                .image_extent(vk::Extent3D {
                    width: mip.layout.width(),
                    height: mip.layout.height(),
                    depth: 1,
                })
        };
        match mip.layout {
            reims_vgpu_memory::GuestImageLayout::D1Array { array_pitch, .. }
            | reims_vgpu_memory::GuestImageLayout::D2Array { array_pitch, .. } => {
                for local_layer in 0..copy.view.array_layer_count {
                    let guest_layer = copy.view.base_array_layer.checked_add(local_layer).ok_or(
                        DrawExecutionDecline::SampledCopyOffsetOverflow {
                            binding,
                            source_offset: relative,
                            pitch: array_pitch,
                            subresource: local_layer,
                        },
                    )?;
                    let offset = u64::from(guest_layer)
                        .checked_mul(array_pitch)
                        .and_then(|offset| relative.checked_add(offset))
                        .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
                            binding,
                            source_offset: relative,
                            pitch: array_pitch,
                            subresource: guest_layer,
                        })?;
                    regions.push(region(
                        offset,
                        if image_arrayed { local_layer } else { 0 },
                        0,
                    ));
                }
            }
            reims_vgpu_memory::GuestImageLayout::D3 {
                depth, depth_pitch, ..
            } => {
                for z in 0..depth {
                    let offset = u64::from(z)
                        .checked_mul(depth_pitch)
                        .and_then(|offset| relative.checked_add(offset))
                        .ok_or(DrawExecutionDecline::SampledCopyOffsetOverflow {
                            binding,
                            source_offset: relative,
                            pitch: depth_pitch,
                            subresource: z,
                        })?;
                    regions.push(region(offset, 0, z as i32));
                }
            }
            _ => regions.push(region(relative, 0, 0)),
        }
    }
    Ok(regions)
}

#[allow(clippy::too_many_arguments)]
unsafe fn upload_buffer_to_sampled_image(
    ctx: &super::context::DeviceContext,
    cb: vk::CommandBuffer,
    src: vk::Buffer,
    src_offset: u64,
    image: vk::Image,
    width: u32,
    height: u32,
    array_layers: u32,
    extent_depth: u32,
    row_length_texels: u32,
    guest_layout: Option<reims_vgpu_memory::GuestImageLayout>,
    binding: u32,
    // Where the copy leaves the image. Every caller's next reader is a shader,
    // but a resident registered in the registry rests in the one colour layout
    // its access tracker names rather than the dedicated read-only one, so the
    // layout is the caller's to state.
    final_layout: vk::ImageLayout,
) -> Result<(), DrawError> {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: array_layers,
    };
    let to_transfer = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(range)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_transfer,
    );
    let copy = sampled_copy_regions(SampledCopyGeometry {
        binding,
        source_offset: src_offset,
        width,
        height,
        array_layers,
        extent_depth,
        row_length_texels,
        guest_layout,
    })
    .map_err(DrawError::DrawExecution)?;
    ctx.device.cmd_copy_buffer_to_image(
        cb,
        src,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &copy,
    );
    let to_shader = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(final_layout)
        .image(image)
        .subresource_range(range)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_shader,
    );
    Ok(())
}

/// Land an aliasing image's birth copy, every level of it, from the staging
/// buffer the launder has already filled with the guest's own bytes.
///
/// Separate from the two upload helpers around it because nothing here is a
/// choice: the geometry comes from [`super::pools::AliasMaterialization`], which
/// was built from the guest's declared chain and has already been proved to
/// agree with the Vulkan image's own per-level placement. There is no format
/// reinterpretation and no layout equation to solve — only a barrier out of
/// `UNDEFINED` covering every level, one region per level, and a barrier into
/// the layout the caller says the image rests in.
///
/// The two barriers must name the whole chain and not level zero. An image born
/// `UNDEFINED` has every level in that layout, and a level left there is one the
/// guest samples as undefined texels — which no counter in this tree reports,
/// because from the device's side nothing failed.
unsafe fn materialize_alias_levels(
    ctx: &super::context::DeviceContext,
    cb: vk::CommandBuffer,
    staging: vk::Buffer,
    image: vk::Image,
    seed: &super::pools::AliasMaterialization,
    final_layout: vk::ImageLayout,
) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: seed.levels.len() as u32,
        base_array_layer: 0,
        layer_count: 1,
    };
    let to_transfer = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(range)];
    let regions: Vec<_> = seed
        .levels
        .iter()
        .enumerate()
        .map(|(level, copy)| {
            vk::BufferImageCopy::default()
                .buffer_offset(copy.relative_offset)
                // Zero means "tightly packed", which is what a row pitch equal
                // to the level's own width already says. Spelling the texel
                // count instead is equally correct; passing zero keeps the
                // padded and unpadded cases spelled the same way as the other
                // copy builders in this module.
                .buffer_row_length(if copy.row_length_texels == copy.width {
                    0
                } else {
                    copy.row_length_texels
                })
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level as u32,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: copy.width,
                    height: copy.height,
                    depth: 1,
                })
        })
        .collect();
    let to_shader = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(final_layout)
        .image(image)
        .subresource_range(range)];
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_transfer,
        );
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_shader,
        );
    }
}

struct SampledAllocationUpload<'a> {
    src: vk::Buffer,
    src_offset: u64,
    image: vk::Image,
    array_layers: u32,
    image_arrayed: bool,
    binding: u32,
    copy: &'a GuestAllocationCopy,
}

unsafe fn upload_buffer_to_sampled_allocation(
    ctx: &super::context::DeviceContext,
    cb: vk::CommandBuffer,
    upload: SampledAllocationUpload<'_>,
) -> Result<(), DrawError> {
    let SampledAllocationUpload {
        src,
        src_offset,
        image,
        array_layers,
        image_arrayed,
        binding,
        copy,
    } = upload;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: copy.view.mip_level_count,
        base_array_layer: 0,
        layer_count: array_layers,
    };
    let to_transfer = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(range)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_transfer,
    );
    let regions = sampled_allocation_copy_regions(
        binding,
        src_offset,
        copy.bytes_per_texel,
        image_arrayed,
        copy,
    )
    .map_err(DrawError::DrawExecution)?;
    ctx.device.cmd_copy_buffer_to_image(
        cb,
        src,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &regions,
    );
    let to_shader = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(image)
        .subresource_range(range)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_shader,
    );
    Ok(())
}

/// The eleven conditions that decide how a draw reaches the submission ring.
///
/// Two questions, one set of fields. `batch_eligible` asks whether this draw
/// may leave its command buffer in recording state for a successor to append
/// to; [`Self::refusal`] asks whether *this* draw may be that successor. The
/// first seven terms answer both, and while they were two hand-written lists a
/// `debug_assert` was the only thing keeping the shared prefix in step —
/// which caught a divergence in debug builds and shipped it in release.
/// Deriving `batch_eligible` from the fields the ladder reads removes the
/// second spelling instead of checking it.
///
/// # Not a term: sampling the target it renders into
///
/// It was one, and it refused 29.7 % of all draws — the single largest reason
/// a draw forced its own submission. The engine handles that case by copying
/// the resident into a pooled image and binding the copy, and that copy is
/// recorded into this draw's own command buffer between the previous draw's
/// `cmd_end_render_pass` and this one's `cmd_begin_render_pass`. Inside an open
/// batch it therefore captures the target as of this draw's position in the
/// stream, which is exactly what the draw asked for. What the refusal was
/// standing in for was the missing dependency in front of that copy; see
/// [`barrier_resident_for_transfer_read`].
///
/// # Not a term: not loading from the target
///
/// The same shape, found the same way, and it refused a further 15 % of draws
/// (16 576 on the boot that retired it). A draw that CLEARs rather than LOADs
/// discards the attachment through `initialLayout = UNDEFINED`, so joining an
/// open batch means recording a CLEAR pass after a pass that wrote the same
/// image — legal, and something the batch already did in the other order.
///
/// What made it unsafe was that the clear path's own barrier derived its source
/// scope from the target's tracked layout, and a resident a render pass just
/// filled sits in `TRANSFER_SRC_OPTIMAL`. So the clear named a transfer read as
/// what it was waiting for, while the thing it actually had to wait for was the
/// previous draw's colour writes — nothing ordered them, and inside one command
/// buffer with the producing draw a few commands back, that is the short-fuse
/// version of the same undefined behaviour. `super::pools::ResidentAccess` now
/// carries what last touched a resident separately from where it sits, so the
/// clear waits on `COLOR_ATTACHMENT_OUTPUT`/`COLOR_ATTACHMENT_WRITE` and the
/// term has nothing left to stand in for.
///
/// Nothing else consulted it: `BatchTarget` keys on identity and geometry and
/// not on the load action, an open batch accumulates only per-draw descriptor
/// sets and sampled admissions, and no completion stamp, epoch publish, pin or
/// writeback branches on it.
struct JoinTerms {
    force_loss: bool,
    quirk: bool,
    is_mrt: bool,
    /// A depth draw whose submit may not be deferred.
    ///
    /// Depth itself is not the reason and never was. A depth pass builds a
    /// per-draw framebuffer, and a deferred draw returns before the disposal
    /// block, so batching one used to leak that framebuffer once per draw —
    /// which is why this rung shared its condition with `is_mrt` and
    /// `color_input`, the other two terms of `ordinary_ad_hoc_framebuffer`.
    /// Both paths now dispose through [`dispose_ad_hoc_attachments`], so the
    /// term is only what [`reims_vgpu_config::BATCH_DEPTH`] restores for an A/B.
    depth_barred: bool,
    reads_back: bool,
    has_query: bool,
    no_identity: bool,
    no_open_batch: bool,
    batch_full: bool,
    target_switch: bool,
}

/// Whether [`reims_vgpu_config::BATCH_MIXED_TARGETS`] is switched off, read once per
/// process.
///
/// Latched for the same reason this crate's `spirv_bind` extent switch
/// is: this sits on the per-draw path and `std::env::var_os` is a lock and an
/// allocation, and the variable cannot change under a running device. The
/// refusal is named once, on the off channel, so a boot whose submission count
/// is being compared says in its own log which arm it ran.
fn batch_mixed_targets_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let (state, value) = reims_vgpu_config::read(reims_vgpu_config::BATCH_MIXED_TARGETS);
        match state {
            reims_vgpu_config::Switch::Off => {
                reims_vgpu_observe::off("batch_mixed reason=batch_mixed_targets_disabled_by_env");
                true
            }
            // An unrecognized spelling is named rather than silently read as the
            // default. It still takes the default arm: this switch may only turn
            // a rail off, and a value nobody can parse is not that.
            reims_vgpu_config::Switch::Unrecognized => {
                reims_vgpu_observe::fail(format!(
                    "batch_mixed reason=batch_mixed_targets_env_unrecognized value={}",
                    value.unwrap_or_default()
                ));
                false
            }
            reims_vgpu_config::Switch::On | reims_vgpu_config::Switch::Unset => false,
        }
    })
}

/// Whether [`reims_vgpu_config::BATCH_DEPTH`] is switched off, read once per process.
///
/// Latched for the same reason [`batch_mixed_targets_disabled`] is: this sits on
/// the per-draw path and `std::env::var_os` is a lock and an allocation.
fn batch_depth_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let (state, value) = reims_vgpu_config::read(reims_vgpu_config::BATCH_DEPTH);
        match state {
            reims_vgpu_config::Switch::Off => {
                reims_vgpu_observe::off("batch_depth reason=batch_depth_disabled_by_env");
                true
            }
            // Named rather than silently read as the default. It still takes the
            // default arm: this switch may only turn a rail off, and a value
            // nobody can parse is not that.
            reims_vgpu_config::Switch::Unrecognized => {
                reims_vgpu_observe::fail(format!(
                    "batch_depth reason=batch_depth_env_unrecognized value={}",
                    value.unwrap_or_default()
                ));
                false
            }
            reims_vgpu_config::Switch::On | reims_vgpu_config::Switch::Unset => false,
        }
    })
}

/// The `depth_barred` term, as the draw path computes it.
///
/// A function rather than an expression at the one call site so a test can reach
/// the *decision* and not just the field: a test that builds `JoinTerms` by hand
/// asserts the ladder, which would stay green if this rung went back to barring
/// every depth draw.
fn depth_bars_batching(has_depth: bool) -> bool {
    has_depth && batch_depth_disabled()
}

/// Hand this draw's ad-hoc attachment handles to the graveyard.
///
/// **Both submit paths call this and neither may inline it.** They differ only in
/// *when* their slot enters [`super::pools::ResourcePools::open_slot_mask`] —
/// `finish_entry_async` marks it pending on the submitting path, `batch_append`
/// installs the open batch on the deferred one — and the rule that this call must
/// follow that moment is the entire safety argument. Called before it, the mask
/// is empty and `dispose` destroys immediately, under a command buffer that still
/// names these handles. A second copy of the sequence is where one of the two
/// would drift off that rule.
///
/// `transient_depth` carries the draw's framebuffer whenever it has depth, so the
/// two arms are exclusive by that test rather than by re-deriving which features
/// are in play — a depth MRT draw has one framebuffer and must dispose it once.
unsafe fn dispose_ad_hoc_attachments(
    ctx: &super::context::DeviceContext,
    pools: &mut super::pools::ResourcePools,
    ordinary_ad_hoc_framebuffer: bool,
    target_fb: vk::Framebuffer,
    transient_depth: Option<(Option<OwnedDepthImage>, vk::Framebuffer)>,
) {
    unsafe {
        // The framebuffers are no longer this draw's to destroy. Both arms used
        // to hand one to the graveyard, because both were built per draw; they
        // now come from `ensure_ad_hoc_framebuffer` and are owned by that cache,
        // which destroys an entry when the first view it names is destroyed.
        // Disposing one here would leave later draws holding a freed handle —
        // the same defect the resident-depth arm below already records for its
        // image.
        let _ = (ordinary_ad_hoc_framebuffer, target_fb);
        // Only the unidentified depth case owns its image. A resident one belongs
        // to the registry and to the guest texture it is keyed on; disposing it
        // here would put the rail straight back to one allocation per draw, with
        // the added defect that the next draw would find a destroyed handle.
        if let Some((Some((dimg, dmem, dview)), _dfb)) = transient_depth {
            pools.dispose(
                &ctx.device,
                super::pools::DeferredHandle::Image {
                    image: dimg,
                    view: dview,
                    memory: dmem,
                },
            );
        }
    }
}

/// What a [`JoinTerms`] rung is a statement about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JoinScope {
    /// A property of this draw alone: it must not defer its submit at all, so
    /// it may neither open a batch nor join one.
    Draw,
    /// A property of how this draw sits against an *already open* batch. It
    /// says nothing about whether this draw may leave its own command buffer
    /// recording afterwards, so it bars joining and not opening.
    Fit,
}

/// One rung of [`JoinTerms::LADDER`]: the term, what it is a statement about,
/// and the census name it carries when it is the first to refuse.
type JoinRefusal = (fn(&JoinTerms) -> bool, JoinScope, &'static str);

impl JoinTerms {
    /// The refusals in ladder order.
    ///
    /// [`Self::batch_eligible`] is this same list filtered to [`JoinScope::Draw`]
    /// rather than a second copy of its prefix. A rung therefore cannot reach
    /// one question and miss the other, and cannot be mis-scoped by landing at
    /// the wrong index — its scope is written beside it, not inferred from
    /// where it sits.
    const LADDER: [JoinRefusal; 10] = [
        (|t| t.force_loss, JoinScope::Draw, "nojoin_force_loss"),
        (|t| t.quirk, JoinScope::Draw, "nojoin_quirk"),
        (|t| t.is_mrt, JoinScope::Draw, "nojoin_mrt"),
        (|t| t.depth_barred, JoinScope::Draw, "nojoin_depth"),
        (|t| t.reads_back, JoinScope::Draw, "nojoin_reads_back"),
        (|t| t.has_query, JoinScope::Draw, "nojoin_query"),
        (|t| t.no_identity, JoinScope::Draw, "nojoin_no_identity"),
        (|t| t.no_open_batch, JoinScope::Fit, "nojoin_no_open_batch"),
        (|t| t.batch_full, JoinScope::Fit, "nojoin_batch_full"),
        (|t| t.target_switch, JoinScope::Fit, "nojoin_target_switch"),
    ];

    /// Whether this draw may defer its submit and leave its command buffer
    /// recording for a successor.
    fn batch_eligible(&self) -> bool {
        !Self::LADDER
            .iter()
            .any(|(refuses, scope, _)| *scope == JoinScope::Draw && refuses(self))
    }

    /// Why this draw does not append to the open batch, named by its first
    /// refusing term, or `None` when it does.
    ///
    /// A joiner records into a command buffer an earlier draw left open, so
    /// everything that bars opening must bar joining. Nothing asserts that:
    /// this walks every rung and [`Self::batch_eligible`] walks a subset of the
    /// same rungs, so a `None` here cannot coexist with a refusing
    /// [`JoinScope::Draw`] rung. The `debug_assert` that used to police two
    /// hand-written lists is gone with the second list.
    fn refusal(&self) -> Option<&'static str> {
        Self::LADDER
            .iter()
            .find(|(refuses, _, _)| refuses(self))
            .map(|(_, _, name)| *name)
    }
}

fn draw_has_no_invocations(req: &DrawRequest) -> bool {
    let element_count = req
        .indexed
        .as_ref()
        .map(|i| i.index_count)
        .unwrap_or(req.vertex_count);
    element_count == 0 || req.instance_count == Some(0)
}

/// A depth image a single draw allocated for itself and must destroy after
/// submit, because the pass named no guest texture to hold it under.
///
/// The registry-resident rail has no equivalent and wants none: its image is
/// owned by the guest's texture, and a handle to dispose is exactly what must
/// not exist for it.
type OwnedDepthImage = (vk::Image, vk::DeviceMemory, vk::ImageView);

/// This draw's depth attachment view, and the image behind it *only if this
/// draw owns it*.
///
/// Two rails, and which one runs is decided by the guest rather than by this
/// device:
///
/// * The pass descriptor bound a depth texture, so the buffer has a guest
///   identity and a guest lifetime. It resolves to one registry resident per
///   guest texture, created on first use and reclaimed by age like every other
///   resident. Nothing is returned to dispose — the resident outlives the draw,
///   which is the whole point.
/// * The pass bound none, and there is no key to hold a resident under. A
///   private buffer is allocated for this draw and handed back for disposal.
///   `vk_alloc_sites transient_depth` is that rail's count and it is expected to
///   be near zero; `depth_resident` is the other's.
///
/// The framebuffer is this draw's either way and is not this function's to make:
/// it binds one specific `render_pass` alongside the colour view, and only the
/// caller knows which.
struct AcquiredDepth {
    image: vk::Image,
    access: super::pools::ResidentAccess,
    view: vk::ImageView,
    /// Set only on the transient rail — the resident rail's image belongs to the
    /// registry and must not be disposed with the draw.
    owned: Option<OwnedDepthImage>,
    /// The identity to mark once the pass has stored into it, so the *next*
    /// pass's LOAD has something to load. `None` on the transient rail, whose
    /// buffer is gone by then.
    identity: Option<super::types::TargetIdentity>,
    /// Whether the resident already held rendered contents when this draw
    /// resolved it. Always false on the transient rail: a buffer created for
    /// this draw has nothing in it, which is why that rail could never honour a
    /// LOAD and why the decline it used to raise was unconditional.
    content_ready: bool,
}

impl AcquiredDepth {
    /// Whether the render pass may declare `VK_ATTACHMENT_LOAD_OP_LOAD` for this
    /// draw's depth attachment.
    ///
    /// **The only thing `DepthAttachKey::load` may be built from.** The guest
    /// asking is one of two terms and it is the term that does not decide: a
    /// LOAD pass also declares `initial_layout` DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    /// and naming a layout an image is not in is undefined behaviour rather than
    /// a stale read — so an image nothing has rendered into cannot be loaded from
    /// whatever the guest asked for. The two terms live behind one call because
    /// they were briefly two expressions at one site, which is the shape where a
    /// later edit reaches for the guest's flag alone and gets a validation error
    /// on hosts that check and undefined contents on hosts that do not.
    fn honours_load(&self, guest_wants_load: bool) -> bool {
        guest_wants_load && self.content_ready
    }
}

unsafe fn acquire_depth_view(
    ctx: &super::context::DeviceContext,
    pools: &mut super::pools::ResourcePools,
    req: &DrawRequest,
    counters: &EngineCounters,
) -> Result<AcquiredDepth, DrawError> {
    let with_stencil = req.depth.as_ref().and_then(|d| d.stencil).is_some();
    let sample_count = req.raster_sample_count.max(1);
    let (depth_width, depth_height) = req
        .depth_attachment_extent()
        .expect("depth acquisition requires depth state");
    if let Some(identity) = req.depth.as_ref().and_then(|d| d.identity.clone()) {
        // Asked before `registry_ensure_depth`, because that call creates the
        // slot when it is absent and a fresh slot is `content_ready == false`.
        // Asking after would answer about the image this draw just made rather
        // than about the one the guest expects to load.
        let content_ready = pools.registry_content_ready(&identity);
        let (image, view) = pools.registry_ensure_depth(
            ctx,
            identity.clone(),
            depth_width,
            depth_height,
            sample_count,
            with_stencil,
            counters,
        )?;
        // A geometry or aspect change recreates the image inside that call, and
        // the recreated one holds nothing. Re-asking is what keeps this honest.
        let content_ready = content_ready && pools.registry_content_ready(&identity);
        let access = pools
            .registry_get(&identity)
            .expect("the depth resident was just ensured")
            .access;
        return Ok(AcquiredDepth {
            image,
            access,
            view,
            owned: None,
            identity: Some(identity),
            content_ready,
        });
    }
    let (dimg, dmem, dview) = pools.create_transient_depth(
        ctx,
        depth_width,
        depth_height,
        sample_count,
        with_stencil,
        counters,
    )?;
    Ok(AcquiredDepth {
        image: dimg,
        access: super::pools::ResidentAccess::Untouched,
        view: dview,
        owned: Some((dimg, dmem, dview)),
        identity: None,
        content_ready: false,
    })
}

/// A pass asked to load depth this device has nothing to load.
///
/// Two ways to reach it, and they are different findings. The pass is the
/// **first** into a depth texture, so `MTLLoadActionLoad` on undefined contents
/// is the guest's own undefined behaviour and a CLEAR is a conformant answer.
/// Or the depth resident was **reclaimed** between two passes that meant to
/// chain — real lost depth, bounded by `IDLE_MAINTENANCE_START_MS`, and the reading
/// that would justify giving depth residents an age of their own.
///
/// Latched per geometry-and-aspect rather than per pipeline: what a reader needs
/// is whether this happens at all and to what shape of attachment, and a
/// per-pipeline latch on a workload with hundreds of pipelines answers a
/// different question at a hundred times the volume.
fn note_depth_load_without_content(width: u32, height: u32, stencil: bool) {
    let key = (u64::from(width) << 32) ^ (u64::from(height) << 1) ^ u64::from(stencil);
    if reims_vgpu_observe::first_sight("depth_load_without_content", key) {
        reims_vgpu_observe::fail(format!(
            "depth_load reason=depth_load_without_content {width}x{height} \
             stencil={} (pass asked LOAD, resident holds nothing; cleared)",
            u8::from(stencil)
        ));
    }
}

/// Validate the resident state a sampled target needs at the authoritative
/// point: inside the draw's engine transaction.
///
/// Runtime preparation carries the serialized resource identity. It must not
/// query this mutable registry first: such a result can change before the draw
/// acquires the engine, while this decision cannot race a reclaim or target
/// replacement.
struct SampledResidentExpectation<'a> {
    binding: u32,
    identity: &'a TargetIdentity,
    resource_width: u32,
    resource_height: u32,
    shader_multisampled: bool,
    initialized_by_this_pass: bool,
}

fn validate_sampled_resident(
    expected: SampledResidentExpectation<'_>,
    held: Option<(bool, u32, u32, u32)>,
    prior: Option<ResidentReclaim>,
) -> Result<(), DrawExecutionDecline> {
    let SampledResidentExpectation {
        binding,
        identity,
        resource_width,
        resource_height,
        shader_multisampled,
        initialized_by_this_pass,
    } = expected;
    let Some((content_ready, resident_width, resident_height, resident_samples)) = held else {
        return Err(DrawExecutionDecline::SampledResidentMissing {
            binding,
            identity: identity.clone(),
            prior,
        });
    };
    if !content_ready && !initialized_by_this_pass {
        return Err(DrawExecutionDecline::SampledResidentNotReady {
            binding,
            identity: identity.clone(),
        });
    }
    if resident_width != resource_width || resident_height != resource_height {
        return Err(DrawExecutionDecline::SampledResidentGeometryMismatch {
            binding,
            identity: identity.clone(),
            resident_width,
            resident_height,
            resource_width,
            resource_height,
        });
    }
    if (resident_samples > 1) != shader_multisampled {
        return Err(DrawExecutionDecline::SampledResidentSampleCountMismatch {
            binding,
            identity: identity.clone(),
            resident_samples,
            shader_multisampled,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn prepare_guest_sampled_transfer(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    resource: &super::types::SampledImageResource,
    image_source: &reims_vgpu_memory::GuestImageSource,
    vouch: reims_vgpu_core::GatherVouch,
    guest_gathers: &mut Vec<PendingGuestGather>,
    sampled: &mut Vec<PreparedSampled>,
    phase: &mut super::draw_phase::DrawTimer,
) -> Result<(), DrawError> {
    let src = &image_source.transfer;
    counters.note_sampled_gather_witness(vouch);
    let sampled_key = SampledKey {
        mip_levels: image_source.view.mip_level_count,
        ..SampledKey::of(resource)
    };
    let reuse = super::pools::CbSampledGuest::image(sampled_key, image_source);
    if let Some(image) = pools.cb_sampled_guest(&reuse) {
        sampled.push(PreparedSampled::Cached {
            binding: resource.binding,
            array_element: resource.array_element,
            image,
        });
        return Ok(());
    }
    let img = pools.acquire_sampled(ctx, sampled_key, counters)?;
    phase.enter(super::draw_phase::Phase::SampledUpload);
    let source =
        match unsafe { prepare_guest_texel_window(ctx, pools, counters, src, guest_gathers)? } {
            Some(imported) => {
                pools.note_guest_read_recorded();
                counters.note_sampled_guest_import(src.total_len);
                imported
            }
            None => {
                let scratch = pools.acquire_staging(
                    ctx,
                    src.total_len,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    counters,
                )?;
                pools.write_staging_from_runs(
                    ctx,
                    &scratch,
                    &src.runs,
                    src.source_offset,
                    src.total_len,
                )?;
                counters.note_sampled_gather(src.total_len);
                GuestTexels::Scratch(scratch)
            }
        };
    sampled.push(PreparedSampled::GuestGather {
        binding: resource.binding,
        array_element: resource.array_element,
        image: img,
        source,
        row_length_texels: src.row_length_texels,
        layout: None,
        allocation_copy: Some(GuestAllocationCopy {
            allocation: image_source.allocation.clone(),
            view: image_source.view,
            transfer_source_offset: src.source_offset,
            bytes_per_texel: u64::from(resource.format.layout().bytes_per_texel()),
        }),
        volume: resource.volume,
        layers: resource.layers,
        reuse: Box::new(reuse),
    });
    phase.enter(super::draw_phase::Phase::AcquireSampled);
    Ok(())
}

pub(crate) unsafe fn execute_draw_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    indexes: &mut SessionCacheIndexes,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
    program: &NativeRenderProgram,
) -> Result<DrawOutput, DrawError> {
    // Charges this draw's wall clock to one phase at a time; commits from
    // `Drop`, so the `?` returns below keep their time.
    let mut phase = super::draw_phase::DrawTimer::start();
    validate_v1(req)?;
    let force_loss = owner.force_device_lost;
    if force_loss {
        owner.force_device_lost = false;
    }
    let ctx = owner.ensure(counters)?;
    pools.ensure_init(ctx, counters)?;

    // Draw batching (deferred submit): a draw that hands the CPU nothing
    // (skip_readback + resident target, no MRT) leaves its CB in recording
    // state for successors. A successor whose work folds into the open CB
    // appends to it, skipping slot claim and submit entirely. Commands that
    // cannot execute inside a render pass close that pass at their recording
    // site; they do not require a queue submission boundary.
    let is_mrt = !req.secondary_targets.is_empty();
    // The resolved attachment decides its own channel order, and the identity is
    // the only thing that decides it: see [`TargetIdentity::is_bgra`]. A pooled
    // draw with no identity has no destination to match and stays RGBA.
    //
    // Derived here rather than at each runtime call site so that all the draws
    // sharing one identity in a frame agree by construction. `registry_ensure`
    // destroys and recreates the image on an order mismatch, so a per-path
    // predicate that one path spells differently is a full reallocation per
    // composite, not a wrong colour.
    //
    // A `DrawRequest::output_bgra` used to sit beside this as an explicit
    // opt-in, OR-ed in here. It is gone rather than unused: once every
    // namespace with a byte-for-byte destination answers from its own key, the
    // only thing an opt-in could express is an order that *disagrees* with the
    // key — which is the per-frame reallocation the paragraph above describes,
    // spelled as a feature. No runtime caller ever set it, and the six parity
    // tests that did were all already rendering into a `Surface` identity.
    let output_bgra = req.target_identity.as_ref().is_some_and(|id| id.is_bgra());
    // Slot 0's view supplies the attachment format. The identity names the
    // allocation behind it; treating those as the same question loses Metal's
    // compatible-format texture views (most visibly UNORM versus sRGB).
    let color0_format = req.color_attachment_format.map_or_else(
        || {
            req.target_identity
                .as_ref()
                .map(|id| super::super::translate::pixel::vk_texel_layout(id.resident_layout()))
                .unwrap_or(crate::translate::pixel::RESIDENT_RGBA_FORMAT)
        },
        crate::format::vk_image_format,
    );
    // A guest-sourced sampled bind used to force the immediate-submit path.
    // Its read of guest RAM happens when the CB *executes*, and this device
    // acked the packet as soon as it was consumed, so deferred submit stretched
    // record→execute from ~0 to a whole batch and the GPU sampled half-repainted
    // a/b window buffers (large black bands under window drags, 2026-07-19 live
    // A/B). Bounding the exposure by keeping it short was never a rule, only a
    // small enough window; `write_stamp` now quiesces every recorded guest read
    // before the guest is told anything finished, which makes it a rule and
    // costs the exclusion nothing to drop.
    // A queried draw is never batched. Deferring the submit defers the fence,
    // and the query result is not readable until the command buffer has
    // completed — so a batched queried draw would return `None` for a count the
    // guest is about to read. The cost lands only on passes that arm a query.
    // `DrawRequest::writes_attachment` and not a second spelling of it. This was
    // written out here as `t == req.target_identity`, which is colour slot 0 —
    // the same rule the snapshot decision carried, one of them narrower than the
    // other, and a draw sampling its own MRT secondary was labelled a plain join
    // here while being snapshotted there. It is census-only now (the join rule
    // dropped this term), so the cost of the divergence was a wrong label rather
    // than a wrong batch, which is exactly the kind that survives.
    let samples_own_target = req.sampled_images.iter().any(|s| match &s.source {
        SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => {
            req.writes_attachment(identity)
        }
        SampledSource::Null
        | SampledSource::Bytes(_)
        | SampledSource::GuestImage(..)
        | SampledSource::GuestRuns(..) => false,
    });
    // Built once and asked twice: the join test below and the append at the end
    // of this function are the same four words, and a `BatchTarget` is how they
    // stay the same four words.
    let batch_target = req.target_identity.as_ref().map(|id| BatchTarget {
        identity: id.clone(),
        width: req.width,
        height: req.height,
        bgra: output_bgra,
    });
    let segment_load = SegmentLoadSources::for_request(req);
    // Why this draw does not append to the open batch, named by its first
    // refusing term.
    //
    // A draw that does not join is a `vkQueueSubmit` and a fence, and that is
    // what `Phase::Slot` measures the worker blocking on. A bare join rate
    // cannot say which of these terms to attack, and the rule has enough terms
    // that guessing picks the wrong one — `batch_eligible` alone folds seven
    // conditions before any of the rest are reached.
    //
    // # What it read
    //
    // Driven Safari drag, 97 986 draws, and only four of the twelve refusals
    // ever fired:
    //
    //   join_appended                42 652   43.5 %
    //   nojoin_samples_own_target    29 113   29.7 %
    //   nojoin_not_load_from_target  14 683   15.0 %
    //   nojoin_no_open_batch         11 538   11.8 %
    //
    // **All seven `batch_eligible` terms read zero.** Nothing here is refused
    // for a device-lost force, a driver quirk, MRT, a depth attachment, a
    // readback, an occlusion query or a missing identity — they are real
    // conditions this workload does not present, and a firing one is news.
    //
    // # What it reads now
    //
    // Same probe after the self-alias term was dropped, 91 495 draws:
    //
    //   join_appended                38 272   41.8 %
    //   nojoin_no_open_batch         19 836   21.7 %
    //   join_appended_self_alias     19 685   21.5 %
    //   nojoin_not_load_from_target  13 702   15.0 %
    //
    // Joins 43.5 % -> 63.3 % and `batch_flushes` 55 334 -> 33 538, a 39 %
    // cut in submissions. `nojoin_no_open_batch` nearly doubled, which is the
    // expected shape and not a regression: a self-alias draw that used to
    // stop at its own term now walks to the end of the ladder and is counted
    // there when the open batch is on another target.
    //
    // What that left was `nojoin_no_open_batch` as the largest refusal, and it
    // was a statement about `BatchTarget` rather than about the draw — the batch
    // was keyed by target identity and geometry, so a run alternating between
    // two surfaces could not batch at all even though each draw opens and ends
    // its own render pass inside the command buffer.
    //
    // # And that key was not load-bearing either
    //
    // Driven macos-13 hammer boot, 22 200 draws through the ladder, before:
    //
    //   join_appended_self_alias      9 364   42.2 %
    //   join_appended                 6 786   30.6 %
    //   nojoin_no_open_batch          5 787   26.1 %
    //   nojoin_cpu_seed                 263    1.2 %
    //
    // So on this rail one refusal was 96 % of all of them, and it named a batch
    // that was recording and had room. The target now decides nothing: a draw
    // appends to whatever batch is open, and `BatchTarget` is compared only on
    // the arm `REIMS_VGPU_BATCH_MIXED_TARGETS=off` selects. What makes that
    // sound is that no consumer of an open batch reads which image its passes
    // wrote — `batch_flush` takes the CB, the fence and the accumulated
    // descriptor sets — and the readback rail has always appended a copy of one
    // target's image to whatever batch happened to be recording (see
    // `read_target_leased`'s `batch_readback_joins`).
    //
    // The refusal that stood for two conditions is split with it:
    // `nojoin_no_open_batch` now means only that nothing is recording, and
    // `nojoin_batch_full` means `BATCH_MAX_DRAWS` is the binding constraint —
    // which is the next lever if it becomes the largest reading, and was
    // unreadable while one name covered both.
    //
    // So the batching ceiling was one term: a draw sampling the target it
    // renders into, which the GVA resident sampled rung made common. That term
    // is gone, and the reason it was ever there does not survive reading the
    // snapshot path. The snapshot is a `cmd_copy_image` recorded into *this*
    // command buffer, after the previous draw's `cmd_end_render_pass` and
    // before this draw's `cmd_begin_render_pass`, from the registry-resident
    // image every batched draw with that identity renders into — so inside an
    // open batch it already captures the target as of this draw's position in
    // the stream, which is the property a self-alias draw needs.
    //
    // What did have to change is the barrier in front of that copy. It was
    // skipped whenever the resident already sat in `TRANSFER_SRC_OPTIMAL`,
    // which is exactly the layout a render pass leaves its primary attachment
    // in — so a snapshot following a draw into the same image took no
    // dependency on it. Between two submissions that was merely undefined;
    // inside one command buffer, with the producing draw a few commands back,
    // it is the same undefined behaviour with a much shorter fuse.
    // `barrier_resident_for_transfer_read` now answers it unconditionally.
    //
    // The terms are gathered into [`JoinTerms`] rather than spelled out here,
    // because they answer two questions — may this draw *open* a batch, and may
    // it *join* the open one — and those were two hand-written lists whose
    // shared prefix a `debug_assert` had to police. `batch_eligible` is now
    // derived from the same fields the ladder reads, so an added term cannot
    // reach one and miss the other.
    // Last because it is the only thing here that looks anything up. Evaluated
    // eagerly, which costs one enum match on a draw an earlier term already
    // refused. A draw with no identity has no `BatchTarget` to ask about and is
    // refused by `no_identity` above this in the ladder, so its fit is `None`
    // and never reaches a name of its own.
    let fit = batch_target
        .as_ref()
        .map(|t| pools.batch_fit(t, batch_mixed_targets_disabled()))
        .unwrap_or(BatchFit::None);
    let terms = JoinTerms {
        force_loss,
        quirk: ctx.caps.quirks.no_deferred_draw_batching,
        is_mrt,
        depth_barred: depth_bars_batching(req.depth.is_some()),
        reads_back: !req.skip_readback,
        has_query: req.occlusion_query.is_some(),
        no_identity: req.target_identity.is_none(),
        no_open_batch: matches!(fit, BatchFit::None),
        batch_full: matches!(fit, BatchFit::Full),
        target_switch: matches!(fit, BatchFit::OtherTarget),
    };
    let batch_eligible = terms.batch_eligible();
    let no_join = terms.refusal();
    // The join arm splits by self-alias, and the two must sum to the joins.
    // `nojoin_samples_own_target` was 29.7 % of all draws and is the population
    // this ladder stopped refusing; without a name of its own on the way *in*,
    // the only visible effect would be a term that stopped firing, which reads
    // identically to a workload that stopped presenting one.
    crate::telemetry::note_route(no_join.unwrap_or(if samples_own_target {
        "join_appended_self_alias"
    } else {
        "join_appended"
    }));
    // `ColorInput` is the decoded framebuffer-fetch contract, distinct from a
    // sampled-image attachment alias. Keep its live population visible at the
    // request boundary so a missing synchronization rule cannot hide behind
    // the much larger sampled-self census.
    if req.color_input {
        crate::telemetry::note_route("fragment_color_input");
    }
    let joins = no_join.is_none();
    // The ceiling on render-pass continuation. A command buffer may carry
    // inside one decoded encoder can retain the pass instance. This split says
    // how much of batching even belongs to that population; the obstacle ladder
    // at the draw records why individual candidates still have to close.
    if joins {
        crate::telemetry::note_route(
            match batch_target.as_ref().and_then(|t| pools.batch_target_is(t)) {
                Some(true) => "join_same_target",
                Some(false) => "join_other_target",
                // A joiner with no `BatchTarget` of its own: refused by
                // `no_identity` above unless something upstream changes, so this
                // is a healthy zero rather than a third population.
                None => "join_target_absent",
            },
        );
    }
    // Claim the next ring slot — BEFORE any pool acquire, so a recycled slot
    // can never alias a still-in-flight CB. Blocks (retire) only when every
    // slot is still in flight; the wait lands in retire_wait_us. A batch
    // joiner reuses the open batch's slot instead (its CB is still recording).
    // Everything above this point is bookkeeping over the request; the claim
    // below is the only part of `Prep` that can block on the GPU. Charged apart
    // so a boot can tell "the CPU is ahead of the ring" from "preparing a draw
    // got slower", which `prep_us` alone cannot.
    phase.enter(super::draw_phase::Phase::Slot);
    // `joins` is `refusal().is_none()`, and each of the other three `BatchFit`
    // arms has its own rung, so a join *is* a `BatchFit::Open` and the pair
    // cannot be any other combination.
    let (cb, fence) = match fit {
        BatchFit::Open(cb, fence) if joins => (cb, fence),
        // A fresh command buffer, so the guest-window imports the previous one
        // pinned against eviction may be displaced again. A *joiner*
        // deliberately does not bump: it records into the open batch's CB, which
        // still names every import the draws before it were handed and has not
        // been submitted, so nothing may free one out from under it.
        _ => pools.begin_entry(ctx, counters)?,
    };
    pools.enter_render_encoder_record(cb, req.continues_render_pass);
    phase.enter(super::draw_phase::Phase::Pipeline);

    // Build layout key from storage / sampled / sampler bindings. Sized up
    // front: this grew from empty on every draw, so a draw with the eight or so
    // descriptors a Maps chain binds paid three reallocations to reach a width
    // its inputs already state.
    let mut layout_bindings = Vec::with_capacity(
        req.storage_buffers.len()
            + req.sampled_images.len()
            + req.samplers.len()
            + req.color_input as usize,
    );
    for b in &req.storage_buffers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
            count: 1,
        });
    }
    for b in &req.sampled_images {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
            count: b.descriptor_count,
        });
    }
    for b in &req.samplers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
            count: 1,
        });
    }
    if req.color_input {
        layout_bindings.push(BindingSig {
            binding: super::types::COLOR_INPUT_BINDING,
            ty: vk::DescriptorType::INPUT_ATTACHMENT.as_raw() as u32,
            stages: vk::ShaderStageFlags::FRAGMENT.as_raw(),
            count: 1,
        });
    }
    let layout_bindings = canonicalize_layout_bindings(layout_bindings)?;
    if let Some(binding) = layout_bindings.iter().find(|binding| binding.count > 1) {
        let populated = req
            .sampled_images
            .iter()
            .filter(|image| image.binding == binding.binding)
            .count() as u32;
        let unpopulated = binding.count.saturating_sub(populated);
        let dynamic_indexing = ctx.features.sampled_image_array_dynamic_indexing;
        let required_descriptors = layout_bindings
            .iter()
            .filter(|candidate| {
                vk::DescriptorType::from_raw(candidate.ty as i32)
                    == vk::DescriptorType::SAMPLED_IMAGE
            })
            .fold(0u32, |total, candidate| {
                total.saturating_add(candidate.count)
            });
        let descriptor_limit = ctx.features.sampled_image_descriptor_limit;
        if !ctx
            .features
            .sampled_descriptor_arrays(required_descriptors, unpopulated)
        {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::DescriptorArrayUnsupported {
                    binding: binding.binding,
                    count: binding.count,
                    unpopulated,
                    required_descriptors,
                    descriptor_limit,
                    partially_bound: ctx.features.descriptor_binding_partially_bound,
                    null_descriptor: ctx.features.null_descriptor,
                    dynamic_indexing,
                },
            ));
        }
    }
    // The backstop the compute path has carried since `25051457` and this one
    // did not. Both of a draw's modules are checked, because either can name a
    // binding the layout above omits and the divide-by-zero is in the driver's
    // shared layout scoring rather than in anything stage-specific.
    if let Some((binding, fragment)) = used_binding_absent_from_layout(
        &req.program.vertex.used_descriptor_bindings,
        &req.program.fragment.used_descriptor_bindings,
        &layout_bindings,
    ) {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::UsedBindingAbsentFromLayout { binding, fragment },
        ));
    }
    let layout_key = LayoutKey {
        // A render stage never culls a dispatch grid, so it exposes no
        // push-constant range and cannot share a kernel's layout.
        kernel_grid: None,
        bindings: layout_bindings,
    };
    // Resolve the serialized load source. A newly imported attachment does not
    // gain initialized Vulkan image contents merely because its memory aliases
    // guest RAM; its first LOAD is materialized below through the buffer rail.
    let load_uses_gpu_content = req.load_from_target;
    // output_bgra (computed with the batch decision above): BGRA output only
    // on the resident path (pooled targets stay RGBA); the whole
    // pass/pipeline/image chain then agrees on B8G8R8A8 so a raw image→buffer
    // copy lands guest scanout order with no CPU swizzle.
    // The seed is borrowed from `req`, never copied to the heap. It is a whole
    // frame, and `engine_delta` measures ~430 MB/s of seed uploads under a
    // browser workload — so a `Vec` here is ~430 MB/s of memcpy plus ~240
    // multi-MiB allocations a second on the drain worker that `drain_duty`
    // shows pinned at duty 0.9+. The only thing that copy bought was a buffer
    // the `output_bgra` arm could swizzle in place; that swizzle now happens
    // during the single copy into the mapped staging span, so the pixels are
    // touched once either way.
    let seed_bytes: Option<&[u8]> = if load_uses_gpu_content {
        None
    } else {
        segment_load.cpu
    };
    let has_load_source = load_uses_gpu_content
        || seed_bytes.is_some()
        || segment_load.guest.is_some()
        || segment_load.resident.is_some();
    let (framebuffer_width, framebuffer_height) = req.minimum_attachment_extent();
    let mut color0_load = color_load_for_segment(
        req.color_load_action,
        req.continues_render_pass,
        has_load_source,
    );
    let preclear_primary = color0_load == ColorLoadKey::Clear
        && (req.width > framebuffer_width || req.height > framebuffer_height);
    if preclear_primary {
        color0_load = ColorLoadKey::Load;
    }
    let mut pass_key = PassKey::single(color0_load, color0_format);
    let mut preclear_secondaries = [false; MAX_SECONDARY_ATTACH];
    for (i, sec) in req.secondary_targets.iter().enumerate() {
        if i >= MAX_SECONDARY_ATTACH {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SecondaryAttachmentCap {
                    requested: req.secondary_targets.len(),
                    cap: MAX_SECONDARY_ATTACH,
                },
            ));
        }
        let mut load = color_load_for_segment(
            sec.load_action,
            req.continues_render_pass,
            matches!(sec.load_action, super::types::ColorLoadAction::Load),
        );
        preclear_secondaries[i] = load == ColorLoadKey::Clear
            && (sec.width > framebuffer_width || sec.height > framebuffer_height);
        if preclear_secondaries[i] {
            load = ColorLoadKey::Load;
        }
        pass_key.secondary[i] = SecondaryAttachKey {
            format: crate::format::vk_image_format(sec.format),
            load,
        };
    }
    pass_key.secondary_count = req.secondary_targets.len() as u8;
    pass_key.color_input = req.color_input;
    // Depth is opt-in per draw (only a non-trivial MTLDepthStencilState reaches
    // here) and composes with MRT: the pass appends its attachment after the
    // secondaries, `clear_values` appends its clear after theirs, and the ad-hoc
    // framebuffer below is built from the same order. All three orderings are
    // written once each and agree by construction; nothing branches on whether
    // the other feature is present.
    //
    // The depth attachment is resolved here rather than beside the framebuffer
    // below, because the pass key needs an answer this device can only get from
    // the resident: whether a `MTLLoadActionLoad` can be honoured at all.
    //
    // A LOAD pass declares `initial_layout` DEPTH_STENCIL_ATTACHMENT_OPTIMAL. An
    // image that nothing has rendered into is in UNDEFINED, and naming a layout
    // an image is not in is undefined behaviour rather than a stale read — so
    // "the guest asked to load" and "there is something to load" are two
    // questions and only the second one is about this device.
    phase.enter(super::draw_phase::Phase::PipelineDepth);
    let depth_attachment = req
        .depth
        .as_ref()
        .map(|_| acquire_depth_view(ctx, pools, req, counters))
        .transpose()?;
    phase.enter(super::draw_phase::Phase::Pipeline);
    let mut preclear_depth = false;
    if let Some(d) = &req.depth {
        let mut load = depth_attachment
            .as_ref()
            .is_some_and(|a: &AcquiredDepth| a.honours_load(d.load));
        if d.load && !load {
            note_depth_load_without_content(req.width, req.height, d.stencil.is_some());
        }
        let (depth_width, depth_height) = req
            .depth_attachment_extent()
            .expect("depth state has an attachment extent");
        preclear_depth =
            !load && (depth_width > framebuffer_width || depth_height > framebuffer_height);
        if preclear_depth {
            let format = if d.stencil.is_some() {
                ctx.depth_stencil_format
            } else {
                crate::translate::pixel::TRANSIENT_DEPTH_FORMAT
            };
            if !ctx.depth_format_transfer_dst(format) {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::AttachmentWideDepthClearUnsupported {
                        format: format.as_raw(),
                    },
                ));
            }
            load = true;
        }
        pass_key.depth = Some(super::caches::DepthAttachKey {
            load,
            stencil: d.stencil.is_some(),
        });
    }
    let raster_sample_count = req.raster_sample_count.max(1);
    let color_sample_count = req.color_sample_count.max(1);
    if color_sample_count != raster_sample_count {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::MultisampleAttachmentSampleCountMismatch {
                attachment: color_sample_count,
                raster: raster_sample_count,
            },
        ));
    }
    if req.multisample_resolve && raster_sample_count == 1 {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::MultisampleSampleCountUnsupported {
                requested: raster_sample_count,
                limit: ctx.features.max_sample_count,
            },
        ));
    }
    if raster_sample_count > 1 {
        if !raster_sample_count.is_power_of_two()
            || raster_sample_count > ctx.features.max_sample_count
        {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::MultisampleSampleCountUnsupported {
                    requested: raster_sample_count,
                    limit: ctx.features.max_sample_count,
                },
            ));
        }
        if !req.secondary_targets.is_empty() || req.color_input {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::MultisampleResolveShapeUnsupported {
                    color_targets: 1u32.saturating_add(req.secondary_targets.len() as u32),
                    depth: req.depth.is_some(),
                    color_input: req.color_input,
                },
            ));
        }
        if !req.multisample_resolve {
            if req.target_identity.is_none() {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::MultisampleResidentTargetMissing {
                        sample_count: raster_sample_count,
                    },
                ));
            }
            if !req.skip_readback || segment_load.has_seed() {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::MultisampleLinearTransferUnsupported {
                        sample_count: raster_sample_count,
                    },
                ));
            }
        }
    }
    pass_key.sample_count = raster_sample_count;
    pass_key.multisample_resolve = req.multisample_resolve;
    let attr_keys: Vec<AttrKey> = req
        .vertex_attributes
        .iter()
        .map(|a| AttrKey {
            location: a.location,
            binding: a.binding,
            format: a.format,
            offset: a.offset,
            stride: a.stride,
            step_function: a.step_function,
            step_rate: a.step_rate,
        })
        .collect();

    phase.enter(super::draw_phase::Phase::PipelineShader);
    let (vert_digest, vert_module) = caches.get_or_create_shader_memoized(
        indexes,
        ctx,
        &program.vertex.words,
        counters,
        pools,
    )?;
    let (frag_digest, frag_module) = caches.get_or_create_shader_memoized(
        indexes,
        ctx,
        &program.fragment.words,
        counters,
        pools,
    )?;
    phase.enter(super::draw_phase::Phase::PipelineLayoutPass);
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let render_pass = caches.get_or_create_pass(ctx, pass_key, counters, pools)?;
    phase.enter(super::draw_phase::Phase::Pipeline);
    // How many viewport slots this draw rasterizes into, checked against the
    // host before it is baked into a pipeline. Refused rather than clamped:
    // clamping would silently drop the viewports past the host's limit, which
    // is the loss this list was widened to stop, and a `viewportCount` above
    // `maxViewports` — or above 1 without `multiViewport` — makes the pipeline
    // invalid rather than merely unsupported.
    let slot_count = super::viewport_slot_count(req);
    let slot_count_u32 = u32::try_from(slot_count).unwrap_or(u32::MAX);
    let max_slots = if ctx.features.multi_viewport {
        ctx.features.max_viewports
    } else {
        1
    };
    if slot_count_u32 > max_slots {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::ViewportSlotsUnsupported {
                requested: slot_count_u32,
                limit: max_slots,
                multi_viewport: ctx.features.multi_viewport,
            },
        ));
    }
    // Resolve the occlusion query before anything is recorded, so a host that
    // cannot count refuses the draw rather than recording one it must throw
    // away. `Boolean` needs nothing: an imprecise Vulkan occlusion query is
    // that mode exactly.
    let occlusion_flags = match req.occlusion_query {
        None => None,
        Some(VisibilityResultMode::Counting) if !ctx.features.occlusion_query_precise => {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::VisibilityCountingUnsupported {
                    occlusion_query_precise: ctx.features.occlusion_query_precise,
                },
            ));
        }
        Some(mode) => Some(crate::translate::raster::vk_query_control_flags(mode)),
    };
    let (line_width_bits, rasterizer_discard) =
        effective_line_raster_state(req.primitive_topology, req.fill_mode, req.line_width);
    let depth_bias = effective_depth_bias(req.depth_bias, ctx.features.depth_bias_clamp)
        .map_err(DrawError::Unsupported)?;
    let pipeline_key =
        PipelineKey {
            vert: vert_digest,
            frag: frag_digest,
            attrs: attr_keys,
            topology: req.primitive_topology,
            blend: req.blend.as_ref().map(super::types::blend_key),
            secondary_blend: {
                let mut per_slot = [None; MAX_SECONDARY_ATTACH];
                for (slot, target) in req
                    .secondary_targets
                    .iter()
                    .take(MAX_SECONDARY_ATTACH)
                    .enumerate()
                {
                    per_slot[slot] = target.blend.as_ref().map(super::types::blend_key);
                }
                per_slot
            },
            color_write_mask: {
                let mut per_slot = [ColorWriteMask::default(); 1 + MAX_SECONDARY_ATTACH];
                per_slot[0] = req.color_write_mask;
                for (slot, target) in req
                    .secondary_targets
                    .iter()
                    .take(MAX_SECONDARY_ATTACH)
                    .enumerate()
                {
                    per_slot[slot + 1] = target.color_write_mask;
                }
                per_slot
            },
            pass: pass_key.compatibility(),
            // Taken from the pass key this draw built, not from `pass`, which
            // erases it once feedback stops changing the render pass.
            feedback_colors: pass_key.feedback_colors,
            cull_mode: req.cull_mode,
            front_face_ccw: req.front_face_ccw,
            fill_mode: req.fill_mode,
            line_width_bits,
            rasterizer_discard,
            depth_bias_enable: depth_bias.is_some(),
            depth_clip: req.depth_clip,
            depth_test: req.depth.as_ref().map(|d| d.test_enable).unwrap_or(false),
            depth_write: req.depth.as_ref().map(|d| d.write_enable).unwrap_or(false),
            depth_compare: req
                .depth
                .as_ref()
                .map(|d| d.compare)
                .unwrap_or(super::types::SamplerCompareFunction::Always),
            stencil: req.depth.as_ref().and_then(|d| d.stencil).map(|s| {
                super::caches::StencilKey {
                    front: s.front,
                    back: s.back,
                }
            }),
            viewport_slots: slot_count_u32,
            layout: layout_key.clone(),
        };
    // One cache, consulted once. `get_or_create_pipeline` already counts the hit
    // and already checks the negative entry for a key that failed to compile.
    phase.enter(super::draw_phase::Phase::PipelineCompile);
    let pipeline = caches.get_or_create_pipeline(
        indexes,
        ctx,
        &pipeline_key,
        req.pipeline_lifetime.as_ref(),
        vert_module,
        &program.vertex.vertex_inputs,
        &program.vertex.words,
        frag_module,
        &program.fragment.words,
        pipeline_layout,
        render_pass,
        counters,
        pools,
    )?;

    // Samplers
    phase.enter(super::draw_phase::Phase::PipelineSampler);
    let mut sampler_handles = Vec::new();
    for s in &req.samplers {
        let h = if s.source == reims_vgpu_core::SamplerSource::Null {
            if !ctx.features.null_descriptor {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::NullSamplerUnsupported { binding: s.binding },
                ));
            }
            vk::Sampler::null()
        } else {
            caches.get_or_create_sampler(
                ctx,
                &super::types::sampler_state_key(s),
                counters,
                pools,
            )?
        };
        sampler_handles.push((s.binding, h));
    }

    phase.enter(super::draw_phase::Phase::Stage);
    // Vertex buffers (with Constant step shift), deduplicated by content:
    // several attributes on one interleaved stream share one staging slot.
    let no_vertex_fetch = draw_has_no_invocations(req);
    // Filled by whichever binds below are scattered guest windows, and drained
    // in the record phase ahead of the render pass. Deduplicated for free: the
    // pool's `cb_bound_buffers` returns an already-planned window's buffer
    // without reaching the gather again, so a window bound twice anywhere in
    // this command buffer is copied once.
    let mut guest_gathers: Vec<PendingGuestGather> = Vec::new();
    phase.enter(super::draw_phase::Phase::StageRoles);
    let gather_roles = BufferGatherRoles::of(req);
    phase.enter(super::draw_phase::Phase::StageVertex);
    let mut vertex_bufs = Vec::new();
    for resource in &req.vertex_attributes {
        let needs_shift = !no_vertex_fetch
            && resource.step_function == VertexStepFunction::Constant
            && req.base_instance != 0;
        let slot = if needs_shift {
            // The shifted prefix makes the content unique to this bind; the
            // runtime keeps Constant-step binds on the CPU path.
            let BufferContent::Bytes(bytes) = &resource.content else {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::ConstantVertexRequiresCpuBytes {
                        location: resource.location,
                    },
                ));
            };
            let prefix = (req.base_instance as usize)
                .checked_mul(resource.stride as usize)
                .ok_or({
                    DrawError::DrawExecution(
                        DrawExecutionDecline::ConstantVertexBaseInstanceOverflow {
                            base_instance: req.base_instance,
                            stride: resource.stride,
                        },
                    )
                })?;
            let len = prefix.checked_add(bytes.len()).ok_or_else(|| {
                DrawError::DrawExecution(DrawExecutionDecline::ConstantVertexAllocationOverflow {
                    prefix,
                    bytes_len: bytes.len(),
                })
            })?;
            let shifted = {
                let _s = stage_phase::Span::moving(stage_phase::Part::Shift, len as u64);
                let mut shifted = vec![0u8; len];
                shifted[prefix..].copy_from_slice(bytes);
                shifted
            };
            let slot = {
                let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
                pools.acquire_staging(
                    ctx,
                    shifted.len() as u64,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    counters,
                )?
            };
            let _s = stage_phase::Span::moving(stage_phase::Part::Bytes, shifted.len() as u64);
            pools.write_staging(ctx, &slot, &shifted)?;
            drop(_s);
            BoundBuffer::from(slot)
        } else {
            stage_buffer_content(
                ctx,
                pools,
                counters,
                &resource.content,
                StageBufferUse {
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                    snapshot_volatile: batch_eligible,
                    gather_role: gather_roles
                        .role(CbBind::key_of(&resource.content))
                        .expect("every vertex buffer was classified"),
                },
                &mut guest_gathers,
            )?
        };
        vertex_bufs.push((resource.binding, slot));
    }

    // Index data follows the same retained resource path as vertex/storage
    // data. A direct import binds the guest's pages; a scattered window is
    // gathered once by this command buffer; only incapable hosts CPU-stage it.
    phase.enter(super::draw_phase::Phase::StageIndex);
    let index_slot = match &req.indexed {
        Some(indexed) => Some(stage_buffer_content(
            ctx,
            pools,
            counters,
            &indexed.content,
            StageBufferUse {
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                snapshot_volatile: batch_eligible,
                gather_role: gather_roles
                    .role(CbBind::key_of(&indexed.content))
                    .expect("the index buffer was classified"),
            },
            &mut guest_gathers,
        )?),
        None => None,
    };

    // Storage buffers (deduplicated by content with the vertex streams: a
    // stage-in buffer doubling as a storage bind reuses the same slot —
    // staging slots always carry the full usage superset).
    phase.enter(super::draw_phase::Phase::StageStorage);
    counters.storage_buffer_bind_slots.fetch_add(
        req.storage_buffers.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let slot = stage_buffer_content(
            ctx,
            pools,
            counters,
            &resource.content,
            StageBufferUse {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                snapshot_volatile: batch_eligible,
                gather_role: gather_roles
                    .role(CbBind::key_of(&resource.content))
                    .expect("every storage buffer was classified"),
            },
            &mut guest_gathers,
        )?;
        storage_slots.push((resource.binding, slot, resource.content.len() as u64));
    }

    // Target seed staging (CPU import only — not LoadFromTarget).
    //
    // A seed is always eight bits per channel; the attachment need not be. A
    // buffer→image copy converts nothing and reads the *image's* texel width
    // per pixel, so staging an RGBA8 seed under a wider attachment would read
    // past the slot and seed the frame with whatever followed it. The wide arm
    // below restates the seed as the attachment's texels first; the four-byte
    // arm is unchanged, which is every attachment this device had until render
    // targets began following the guest's declared format.
    phase.enter(super::draw_phase::Phase::StageSeed);
    let seed_wide = seed_bytes.and_then(|rgba8| {
        let layout = crate::translate::pixel::texel_layout_of(color0_format)?;
        if layout.bytes_per_texel() == reims_vgpu_protocol::TexelLayout::Rgba8.bytes_per_texel() {
            return None;
        }
        Some((rgba8, layout))
    });
    let seed_slot = if let Some((rgba8, layout)) = seed_wide {
        // The seed's own order first, because `expand_rgba8_to_texel` reads
        // semantic RGBA8 — the same normalization the four-byte arm folds into
        // its copy, done here as a step because a widening pass cannot also
        // exchange in place.
        let mut semantic;
        let src = if matches!(req.target_seed_order, SeedOrder::Bgra8) {
            semantic = rgba8.to_vec();
            for px in semantic.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            &semantic[..]
        } else {
            rgba8
        };
        let pixels = req.width.saturating_mul(req.height);
        let mut wide = vec![0u8; (pixels as usize) * (layout.bytes_per_texel() as usize)];
        if !reims_vgpu_core::expand_rgba8_to_texel(layout, src, pixels, &mut wide) {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::SeedFormatUnwritable {
                    format: color0_format,
                },
            ));
        }
        let slot = {
            let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
            pools.acquire_staging(
                ctx,
                wide.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?
        };
        {
            let _s = stage_phase::Span::moving(stage_phase::Part::Bytes, wide.len() as u64);
            pools.write_staging(ctx, &slot, &wide)?;
        }
        counters.note_seed_upload(wide.len() as u64);
        Some(slot)
    } else if let Some(rgba8) = seed_bytes {
        let slot = {
            let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
            pools.acquire_staging(
                ctx,
                rgba8.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?
        };
        // Vulkan buffer→image copies do not perform format conversion, so the
        // staged bytes must already be in the attachment's physical order —
        // otherwise partial draws preserve an exact R/B-exchanged seed outside
        // their damaged geometry. The attachment is BGRA when `output_bgra`; the
        // seed states its own order. Exchange exactly when they disagree, inside
        // the copy that has to happen anyway.
        if matches!(req.target_seed_order, SeedOrder::Bgra8) != output_bgra {
            let _s = stage_phase::Span::moving(stage_phase::Part::Swap, rgba8.len() as u64);
            pools.write_staging_swap_rb(ctx, &slot, rgba8)?;
        } else {
            let _s = stage_phase::Span::moving(stage_phase::Part::Bytes, rgba8.len() as u64);
            pools.write_staging(ctx, &slot, rgba8)?;
        }
        counters.note_seed_upload(rgba8.len() as u64);
        Some(slot)
    } else {
        None
    };
    // A secondary MRT attachment is bound + rendered as attachment N of an
    // ad-hoc framebuffer built here. The primary slot 0 keeps its own single-RT
    // framebuffer (consistent with single-RT draws to the same target), so the
    // primary is ensured under a single-attachment pass even in an MRT draw;
    // the MRT render pass is used only for the ad-hoc framebuffer + pipeline.
    // The resident/pooled slot keeps a color-only framebuffer + pass; the
    // depth-carrying `render_pass` is used only for the ad-hoc framebuffer +
    // pipeline (same split MRT already uses for its secondary framebuffer).
    // A framebuffer-fetch draw also splits: the slot is ensured under the
    // color-only pass (its cached framebuffer stays input-ref-free — passes
    // with and without an input reference are NOT framebuffer-compatible),
    // and the fetch-carrying `render_pass` is used only for the ad-hoc
    // framebuffer + pipeline, exactly like MRT/depth.
    phase.enter(super::draw_phase::Phase::StagePass);
    // Whether this draw's pass shape differs from the colour-only one the target
    // slot's cached framebuffer was built against. One predicate, because the
    // two answers it feeds have to agree: which pass the slot is ensured under,
    // and whether the draw builds (and later disposes) a framebuffer of its own.
    let ordinary_ad_hoc_framebuffer = is_mrt || req.depth.is_some() || req.color_input;
    let ad_hoc_framebuffer = ordinary_ad_hoc_framebuffer || req.multisample_resolve;
    let (primary_pass, primary_pass_compatibility) = if ad_hoc_framebuffer {
        let color_only = pass_key.primary_attachment_only();
        (
            caches.get_or_create_pass(ctx, color_only, counters, pools)?,
            color_only.framebuffer_compatibility(),
        )
    } else {
        (render_pass, pass_key.framebuffer_compatibility())
    };
    phase.enter(super::draw_phase::Phase::Acquire);
    // (identity, image, tracked-layout-before-this-draw) per secondary — used
    // to barrier prior sampled reads and to mark ready afterward.
    let mut mrt_secondaries: Vec<(
        super::types::TargetIdentity,
        vk::Image,
        super::pools::ResidentAccess,
        Option<reims_vgpu_memory::GuestWritePages>,
    )> = Vec::new();
    // This draw's depth attachment, when it has one. The framebuffer is always
    // this draw's own and is always disposed after submit; the image behind it
    // is only owned here when the pass named no guest depth texture to key a
    // resident on — see `acquire_depth_view`. `None` on the 2D path so nothing
    // changes there.
    let mut transient_depth: Option<(Option<OwnedDepthImage>, vk::Framebuffer)> = None;
    // Mark everything this draw is about to read before resolving its own
    // target. The whole operation runs under the engine transaction, so an
    // idle reclaim cannot interleave with validation or binding; the early
    // mark preserves the source's recency if resolving the destination needs
    // to choose a capacity victim.
    for s in &req.sampled_images {
        match &s.source {
            SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => {
                pools.registry_note_sampled_use(identity)
            }
            SampledSource::Null
            | SampledSource::Bytes(_)
            | SampledSource::GuestImage(..)
            | SampledSource::GuestRuns(..) => {}
        }
    }
    let (
        target_image,
        mut target_fb,
        target_access,
        target_view,
        target_guest_memory,
        target_guest_write_pages,
        target_content_ready,
    ) = if let Some(identity) = &req.target_identity {
        let gen = identity.generation();
        let target_sample_count = if req.multisample_resolve {
            1
        } else {
            color_sample_count
        };
        // The slot and the view the render pass attaches. They are two
        // answers because they are two questions: the slot is the
        // allocation, and the view is the interpretation `color0_format`
        // declared over it. See `translate::pixel::ResidentFormat`.
        let (t, attachment_view) = pools.registry_ensure(
            ctx,
            identity.clone(),
            req.width,
            req.height,
            target_sample_count,
            primary_pass,
            primary_pass_compatibility,
            gen,
            color0_format,
            req.target_guest.as_ref().and_then(|target| target.memory()),
            req.load_from_target,
            counters,
        )?;
        if req.load_from_target && !t.content_ready {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::LoadTargetContentNotReady {
                    identity: identity.clone(),
                },
            ));
        }
        let primary_image = t.image;
        let primary_view = attachment_view;
        let primary_access = t.access;
        let primary_slot_fb = t.framebuffer;
        let primary_guest_memory = t.memory.guest_memory().cloned();
        let primary_guest_write_pages = t.memory.guest_write_pages().cloned();
        let primary_content_ready = t.content_ready;
        if ordinary_ad_hoc_framebuffer {
            let views = ad_hoc_attachment_views(
                ctx,
                pools,
                counters,
                req,
                primary_view,
                depth_attachment.as_ref().map(|d| d.view),
                &mut mrt_secondaries,
            )?;
            let fb = pools.ensure_ad_hoc_framebuffer(
                ctx,
                render_pass,
                &views,
                framebuffer_width,
                framebuffer_height,
                counters,
            )?;
            if let Some(d) = depth_attachment.as_ref() {
                transient_depth = Some((d.owned, fb));
            }
            (
                primary_image,
                fb,
                primary_access,
                primary_view,
                primary_guest_memory,
                primary_guest_write_pages,
                primary_content_ready,
            )
        } else {
            (
                primary_image,
                primary_slot_fb,
                primary_access,
                primary_view,
                primary_guest_memory,
                primary_guest_write_pages,
                primary_content_ready,
            )
        }
    } else {
        let target_key = TargetKey {
            width: req.width,
            height: req.height,
            with_transfer_dst: seed_bytes.is_some(),
        };
        // Acquire the pooled slot under the color-only `primary_pass` (same as
        // its cached framebuffer), and build the draw's own framebuffer under
        // `render_pass` whenever the two pass shapes differ.
        let t = pools.acquire_target(ctx, target_key, primary_pass, counters)?;
        let (pool_image, pool_view, pool_fb) = (t.image, t.view, t.framebuffer);
        if ordinary_ad_hoc_framebuffer {
            let views = ad_hoc_attachment_views(
                ctx,
                pools,
                counters,
                req,
                pool_view,
                depth_attachment.as_ref().map(|d| d.view),
                &mut mrt_secondaries,
            )?;
            let fb = pools.ensure_ad_hoc_framebuffer(
                ctx,
                render_pass,
                &views,
                framebuffer_width,
                framebuffer_height,
                counters,
            )?;
            if let Some(d) = depth_attachment.as_ref() {
                transient_depth = Some((d.owned, fb));
            }
            (
                pool_image,
                fb,
                super::pools::ResidentAccess::Untouched,
                pool_view,
                None,
                None,
                false,
            )
        } else {
            (
                pool_image,
                pool_fb,
                super::pools::ResidentAccess::Untouched,
                pool_view,
                None,
                None,
                false,
            )
        }
    };
    let _multisample_source_image = if req.multisample_resolve {
        let (image, _view, framebuffer) = pools.acquire_multisample_target(
            ctx,
            super::pools::MultisampleTargetKey {
                width: req.width,
                height: req.height,
                format: color0_format,
                samples: raster_sample_count,
                compatibility: pass_key.framebuffer_compatibility(),
                resolve_view: target_view,
                depth_view: depth_attachment.as_ref().map(|depth| depth.view),
                transient_depth: depth_attachment
                    .as_ref()
                    .is_some_and(|depth| depth.owned.is_some()),
            },
            render_pass,
            counters,
        )?;
        target_fb = framebuffer;
        Some(image)
    } else {
        None
    };
    // A fresh guest-backed resident consumes its first guest-page LOAD through
    // a stable buffer snapshot. Aliasing memory does not initialize a Vulkan
    // image; after this copy, attachment writes make the imported image itself
    // authoritative and later LOADs stay zero-copy.
    let target_guest_seed = segment_load.guest;
    let target_guest_texels = match target_guest_seed.filter(|_| !target_content_ready) {
        Some(seed) if target_guest_memory.is_some() => {
            let slot = {
                let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
                pools.acquire_staging(
                    ctx,
                    seed.source.total_len,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    counters,
                )?
            };
            {
                let _s = stage_phase::Span::moving(stage_phase::Part::Runs, seed.source.total_len);
                pools.write_staging_from_runs(
                    ctx,
                    &slot,
                    &seed.source.runs,
                    seed.source.source_offset,
                    seed.source.total_len,
                )?;
            }
            counters.note_seed_upload(seed.source.total_len);
            crate::telemetry::note_route("target_seed_guest_import_snapshot");
            Some(GuestTexels::Scratch(slot))
        }
        Some(seed) => match unsafe {
            prepare_guest_texel_window(ctx, pools, counters, &seed.source, &mut guest_gathers)?
        } {
            Some(source) => {
                pools.note_guest_read_recorded();
                crate::telemetry::note_route("target_seed_guest_import");
                crate::telemetry::note_route_n(
                    "target_seed_guest_import_bytes",
                    seed.source.total_len,
                );
                Some(source)
            }
            None => {
                let slot = {
                    let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
                    pools.acquire_staging(
                        ctx,
                        seed.source.total_len,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        counters,
                    )?
                };
                {
                    let _s =
                        stage_phase::Span::moving(stage_phase::Part::Runs, seed.source.total_len);
                    pools.write_staging_from_runs(
                        ctx,
                        &slot,
                        &seed.source.runs,
                        seed.source.source_offset,
                        seed.source.total_len,
                    )?;
                }
                counters.note_seed_upload(seed.source.total_len);
                crate::telemetry::note_route("target_seed_guest_cpu_fallback");
                Some(GuestTexels::Scratch(slot))
            }
        },
        None => None,
    };
    let loads_direct_guest =
        target_guest_memory.is_some() && target_content_ready && target_guest_seed.is_some();
    let load_uses_gpu_content = req.load_from_target || loads_direct_guest;
    // GPU seed source: resolved after registry_ensure (which protects it from
    // the capacity sweep) so the handle cannot be destroyed under this draw.
    // Every rejection is a distinct named error — the runtime pre-checks
    // readiness, so these only fire on a runtime/protocol bug.
    let seed_from_resolved: Option<(
        vk::Image,
        super::pools::ResidentAccess,
        super::pools::ResidentAccess,
    )> = if let Some(seed_identity) = segment_load.resident {
        let slot = pools.registry_get(seed_identity).ok_or_else(|| {
            DrawError::DrawExecution(DrawExecutionDecline::SeedResidentMissing {
                identity: seed_identity.clone(),
            })
        })?;
        if !slot.content_ready {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::SeedResidentNotReady {
                    identity: seed_identity.clone(),
                },
            ));
        }
        if slot.width != req.width || slot.height != req.height {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::SeedGeometryMismatch {
                    identity: seed_identity.clone(),
                    resident_width: slot.width,
                    resident_height: slot.height,
                    draw_width: req.width,
                    draw_height: req.height,
                },
            ));
        }
        if slot.scanout_order() != output_bgra {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::SeedFormatMismatch {
                    identity: seed_identity.clone(),
                    resident_bgra: slot.scanout_order(),
                    draw_bgra: output_bgra,
                },
            ));
        }
        Some((
            slot.image,
            slot.access,
            super::pools::ResidentAccess::transfer_read(),
        ))
    } else {
        None
    };

    // Resolve sampled images only after ensuring the render target so registry
    // capacity eviction cannot destroy an image already selected for this draw.
    phase.enter(super::draw_phase::Phase::AcquireSampled);
    let mut sampled = Vec::new();
    let mut attachment_snapshots: std::collections::HashMap<
        (super::types::TargetIdentity, SampledKey),
        SampledSlot,
    > = std::collections::HashMap::new();
    for resource in &req.sampled_images {
        match &resource.source {
            SampledSource::Null => {
                if !ctx.features.null_descriptor {
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::NullSampledImageUnsupported {
                            binding: resource.binding,
                        },
                    ));
                }
                sampled.push(PreparedSampled::Null {
                    binding: resource.binding,
                    array_element: resource.array_element,
                });
            }
            SampledSource::Bytes(bytes) => {
                if let Some(image) = pools.find_cached_sampled(
                    SampledKey::of(resource),
                    bytes,
                    resource.identity,
                    resource.resource_lifetime.as_ref(),
                    counters,
                ) {
                    sampled.push(PreparedSampled::Cached {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        image,
                    });
                    continue;
                }
                let img = pools.acquire_sampled(ctx, SampledKey::of(resource), counters)?;
                let st = pools.acquire_staging(
                    ctx,
                    bytes.len() as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    counters,
                )?;
                pools.write_staging(ctx, &st, bytes)?;
                counters.note_sampled_reupload(bytes.len() as u64, resource.byte_origin);
                sampled.push(PreparedSampled::Upload {
                    binding: resource.binding,
                    array_element: resource.array_element,
                    image: img,
                    staging: st,
                    volume: resource.volume,
                    layers: resource.layers,
                });
            }
            SampledSource::Target(identity) | SampledSource::Attachment { identity, .. } => {
                let attachment_initial = match &resource.source {
                    SampledSource::Attachment { initial, .. } => Some(*initial),
                    _ => None,
                };
                let primary_seed = req.color_attachment_index(identity) == Some(0)
                    && (seed_slot.is_some()
                        || target_guest_texels.is_some()
                        || seed_from_resolved.is_some());
                let initialized_by_this_pass = matches!(
                    attachment_initial,
                    Some(
                        super::types::AttachmentInitial::Clear(_)
                            | super::types::AttachmentInitial::DontCare
                    )
                ) || (attachment_initial
                    == Some(super::types::AttachmentInitial::Seed)
                    && primary_seed);
                // Reading a resident is using it. Marked before the lookup so
                // the refusal paths below cannot skip it: a resident whose
                // content is not ready yet, or whose geometry disagrees with
                // this bind, is still one the guest is actively sampling, and
                // aging it out between two attempts is how a recoverable
                // not-ready became a permanent missing.
                pools.registry_note_sampled_use(identity);
                let held = pools.registry_get(identity).map(|slot| {
                    (
                        slot.image,
                        slot.view,
                        slot.access,
                        slot.scanout_order(),
                        slot.content_ready,
                        slot.width,
                        slot.height,
                        slot.sample_count,
                    )
                });
                let prior = held
                    .is_none()
                    .then(|| pools.prior_reclaim(identity))
                    .flatten();
                if let Some((_, _, _, _, ready, width, height, samples)) = held.as_ref() {
                    if *samples > 1 {
                        crate::telemetry::note_route("sampled_resident_multisample");
                        let key = (u64::from(resource.binding) << 32) | u64::from(*samples);
                        if reims_vgpu_observe::first_sight("sampled_resident_multisample", key) {
                            reims_vgpu_observe::off(format!(
                                "sampled_resident_multisample binding={} shader_ms={} \
                                 resident_samples={} resident={}x{} ready={} identity={identity:?}",
                                resource.binding,
                                resource.multisampled,
                                samples,
                                width,
                                height,
                                ready,
                            ));
                        }
                    }
                }
                validate_sampled_resident(
                    SampledResidentExpectation {
                        binding: resource.binding,
                        identity,
                        resource_width: resource.width,
                        resource_height: resource.height,
                        shader_multisampled: resource.multisampled,
                        initialized_by_this_pass,
                    },
                    held.as_ref().map(|held| (held.4, held.5, held.6, held.7)),
                    prior,
                )
                .map_err(DrawError::DrawExecution)?;
                let (
                    source_image,
                    _source_attachment_view,
                    source_layout,
                    _source_bgra,
                    _source_ready,
                    _resident_width,
                    _resident_height,
                    _resident_samples,
                ) = held.expect("validated resident is held");
                let source_view = pools
                    .registry_sample_view(
                        ctx,
                        identity,
                        crate::format::vk_image_format(resource.format),
                        resource.swizzle,
                        counters,
                    )?
                    .expect("validated resident is held");
                // One reason a resident cannot be bound through its own view:
                // the draw is sampling an attachment it also renders into, which
                // needs a copy taken before the pass writes it.
                //
                // A **view swizzle** used to be a second reason, because the
                // registry held one image view per target and had nowhere to put
                // non-identity channels. It now holds a view per
                // format-and-mapping pair, so the mapping is expressible on the
                // direct arm and the hardware performs it at sample time. Note
                // what the copy was buying when it was needed: not correctness of
                // the channels alone but the binding's *presence*, because the
                // producer before it returned without pushing any resource, and
                // a binding absent from a layout its own module statically uses
                // is refused downstream by `used_binding_absent_from_layout` —
                // losing the whole draw.
                if let Some(self_slot) = sampled_attachment_slot(req, resource) {
                    let snapshot_timing = match attachment_initial {
                        Some(super::types::AttachmentInitial::Clear(clear)) => {
                            SnapshotTiming::Clear(clear)
                        }
                        Some(super::types::AttachmentInitial::Seed) if primary_seed => {
                            SnapshotTiming::AfterPrimarySeed
                        }
                        Some(super::types::AttachmentInitial::DontCare) => {
                            SnapshotTiming::Undefined
                        }
                        _ => SnapshotTiming::Prior,
                    };
                    // Which attachment this draw is sampling of its own. The
                    // primary is the case this arm has always taken; the other
                    // two are the ones a primary-only test let past it, so they
                    // are alarms and a zero on them is the healthy reading.
                    crate::telemetry::note_route(self_slot.sampled_self_route());
                    crate::telemetry::note_route(snapshot_timing.route());
                    let snapshot_key = SampledKey::of(resource);
                    let image = if snapshot_key.is_plain_2d_identity_view() {
                        let name = (identity.clone(), snapshot_key);
                        if let Some(existing) = attachment_snapshots.get(&name) {
                            existing.handles()
                        } else {
                            let acquired =
                                pools.acquire_attachment_snapshot(ctx, snapshot_key, counters)?;
                            attachment_snapshots.insert(name, acquired.handles());
                            acquired
                        }
                    } else {
                        // Swizzled/arrayed/volume attachment views can give one
                        // source several incompatible keys, so an attachment-
                        // count bound does not apply to them: they keep the
                        // general pool.
                        pools.acquire_sampled(ctx, snapshot_key, counters)?
                    };
                    sampled.push(PreparedSampled::Snapshot {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        identity: identity.clone(),
                        source_image,
                        source_access: source_layout,
                        next_access: super::pools::ResidentAccess::transfer_read(),
                        image,
                        timing: snapshot_timing,
                    });
                } else {
                    sampled.push(PreparedSampled::Resident {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        identity: identity.clone(),
                        image: source_image,
                        view: source_view,
                        access: source_layout,
                        next_access: super::pools::ResidentAccess::shader_read(),
                        levels: pools
                            .resident_mip_levels(identity)
                            .expect("validated resident is held"),
                        materialize: None,
                    });
                }
                counters
                    .sampled_gpu_binds
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            SampledSource::GuestImage(source, vouch) => {
                let direct = if source.direct.is_some() {
                    Some(unsafe {
                        pools.acquire_guest_sampled(ctx, source, resource, counters, |identity| {
                            req.writes_attachment(identity)
                        })
                    })
                } else {
                    None
                };
                match direct {
                    None => {
                        prepare_guest_sampled_transfer(
                            ctx,
                            pools,
                            counters,
                            resource,
                            source,
                            *vouch,
                            &mut guest_gathers,
                            &mut sampled,
                            &mut phase,
                        )?;
                    }
                    Some(Ok(super::pools::GuestSampledUse::Resident {
                        identity,
                        image,
                        view,
                        access,
                        levels,
                        materialize,
                    })) => {
                        counters.note_sampled_gather_witness(*vouch);
                        pools.note_guest_read_recorded();
                        counters.note_sampled_guest_direct(source.transfer.total_len);
                        sampled.push(PreparedSampled::Resident {
                            binding: resource.binding,
                            array_element: resource.array_element,
                            identity,
                            image,
                            view,
                            access,
                            next_access: super::pools::ResidentAccess::shader_read(),
                            levels,
                            materialize,
                        });
                    }
                    Some(Err(decline)) => {
                        reims_vgpu_observe::Emit::decline("sampled_guest_image_declined", &decline)
                            .off();
                        prepare_guest_sampled_transfer(
                            ctx,
                            pools,
                            counters,
                            resource,
                            source,
                            *vouch,
                            &mut guest_gathers,
                            &mut sampled,
                            &mut phase,
                        )?;
                    }
                }
            }
            SampledSource::GuestRuns(src, vouch) => {
                // Guest pages are live storage, not immutable content. The
                // validity transition is not a version for every unified-memory
                // CPU write, so even a telemetry `Vouched` result cannot select
                // an older copied image across command-buffer boundaries.
                counters.note_sampled_gather_witness(*vouch);
                let sampled_key = SampledKey::of(resource);
                let reuse = super::pools::CbSampledGuest::runs(sampled_key, src);
                if let Some(image) = pools.cb_sampled_guest(&reuse) {
                    sampled.push(PreparedSampled::Cached {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        image,
                    });
                    continue;
                }
                let img = pools.acquire_sampled(ctx, sampled_key, counters)?;
                // Everything from here to the end of this arm moves bytes;
                // everything above it in `AcquireSampled` decides which image
                // to move them into. The split is what separates "the driver
                // made 21 objects" from "the CPU copied 8.9 MB", and those are
                // the two candidates for a cold sampled bind. The phase is
                // re-entered per texture, which is correct — `enter`
                // accumulates, so a draw binding several gathers charges each
                // half of each bind to its own bar.
                phase.enter(super::draw_phase::Phase::SampledUpload);
                // First ask whether the bytes have to move at all. The RAMBlock
                // import is a buffer the copy can name, so where this device
                // can make one this arm moves nothing on the CPU and the
                // `memcpy` below never runs.
                //
                // The import is over guest RAM, which the GPU can write as well
                // as read. What keeps the copy inside the guest's own bytes is
                // the `GuestRef` in `src.pages`: it is range-checked against its
                // RAMBlock at construction and there is no other way to name a
                // byte in one.
                let source = match unsafe {
                    prepare_guest_texel_window(ctx, pools, counters, src, &mut guest_gathers)?
                } {
                    Some(imported) => {
                        // The read is now the command buffer's, at execute time.
                        // `write_stamp` quiesces before telling the guest these
                        // pages are free, which is what keeps that legal.
                        pools.note_guest_read_recorded();
                        counters.note_sampled_guest_import(src.total_len);
                        imported
                    }
                    None => {
                        let scratch = pools.acquire_staging(
                            ctx,
                            src.total_len,
                            vk::BufferUsageFlags::TRANSFER_SRC,
                            counters,
                        )?;
                        pools.write_staging_from_runs(
                            ctx,
                            &scratch,
                            &src.runs,
                            src.source_offset,
                            src.total_len,
                        )?;
                        // The only arm of this loop that moves bytes, and until
                        // now the only one that reported nothing — which is what
                        // let the whole of `acquire_sampled` sit unattributed.
                        counters.note_sampled_gather(src.total_len);
                        GuestTexels::Scratch(scratch)
                    }
                };
                sampled.push(PreparedSampled::GuestGather {
                    binding: resource.binding,
                    array_element: resource.array_element,
                    image: img,
                    source,
                    row_length_texels: src.row_length_texels,
                    layout: None,
                    allocation_copy: None,
                    volume: resource.volume,
                    layers: resource.layers,
                    reuse: Box::new(reuse),
                });
                // Back to the deciding half for the next texture in the loop.
                phase.enter(super::draw_phase::Phase::AcquireSampled);
            }
        }
    }

    phase.enter(super::draw_phase::Phase::AcquireReadback);
    // Sized by the attachment's own texel, not by a constant four. The copy at
    // the end of this command buffer names an image extent and no buffer row
    // length, so the GPU writes `width * height *
    // bytes_per_texel(color0_format)` bytes into this slot — a wide attachment
    // over a four-byte slot is a device-side write past the slot, not a short
    // read. The seed path above answers the same question on the way in, and
    // states the same reason.
    let rb_texel = u64::from(super::readback_bytes_per_texel(color0_format));
    let rb_size = (req.width as u64) * (req.height as u64) * rb_texel;
    let do_readback = !req.skip_readback;
    phase.note_target(req.width, req.height, if do_readback { rb_size } else { 0 });
    let readback = if do_readback {
        Some(pools.acquire_readback(ctx, rb_size, counters)?)
    } else {
        None
    };

    phase.enter(super::draw_phase::Phase::Descriptors);
    // binding state: the writes become commands in this command buffer, with no
    // separately allocated object. The layout cache made the same decision.
    let push_descriptors = layout_key.uses_push_descriptors(ctx.caps.push_descriptor);
    // Owning pool block travels alongside an allocated set so the flush-time
    // free routes back to the block it came from. A push layout owns neither.
    let mut dset_pool: Option<vk::DescriptorPool> = None;
    let dset = if dsl != vk::DescriptorSetLayout::null() && !push_descriptors {
        let (dset, pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;
        dset_pool = Some(pool);
        Some(dset)
    } else {
        None
    };
    let buffer_infos: Vec<_> = storage_slots
        .iter()
        .map(|(_, bound, len)| {
            vk::DescriptorBufferInfo::default()
                .buffer(bound.buffer)
                .offset(bound.offset)
                .range(descriptor_range(*len))
        })
        .collect();
    let sampled_infos: Vec<_> = sampled
        .iter()
        .map(|image| {
            vk::DescriptorImageInfo::default()
                .image_view(image.view())
                .image_layout(image.descriptor_layout())
        })
        .collect();
    let sampler_infos: Vec<_> = sampler_handles
        .iter()
        .map(|(_, s)| vk::DescriptorImageInfo::default().sampler(*s))
        .collect();
    // Framebuffer fetch: the input attachment IS the color target's view;
    // derive the same layout as the subpass reference. A draw that also
    // samples the target upgrades both from GENERAL to the feedback layout.
    let color_input_info = vk::DescriptorImageInfo::default()
        .image_view(target_view)
        .image_layout(pass_key.color_layout(0));
    let dst_set = dset.unwrap_or_default();
    let mut descriptor_writes = Vec::new();
    for (i, (binding, _, _)) in storage_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i])),
        );
    }
    for (i, image) in sampled.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(image.binding())
                .dst_array_element(image.array_element())
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&sampled_infos[i])),
        );
    }
    for (i, (binding, _)) in sampler_handles.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&sampler_infos[i])),
        );
    }
    if req.color_input {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(super::types::COLOR_INPUT_BINDING)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .image_info(std::slice::from_ref(&color_input_info)),
        );
    }
    if dset.is_some() {
        ctx.device.update_descriptor_sets(&descriptor_writes, &[]);
        counters
            .descriptor_set_updates
            .fetch_add(1, Ordering::Relaxed);
    }

    phase.enter(super::draw_phase::Phase::RecordBegin);
    // The ring slot's CB retired at begin_entry and its fence is unsignaled —
    // no pre-record wait remains (pre_record_wait_us stays 0 on this path).
    // A batch joiner's CB is already recording (opened by the batch opener);
    // its commands append after the previous draw's end_render_pass.
    if !joins {
        unsafe {
            pools.begin_slot_recording(
                ctx,
                cb,
                super::gpu_span::Kind::Draw,
                VkOp::ExecResetCb,
                VkOp::ExecBeginCb,
            )?
        };
    }

    phase.enter(super::draw_phase::Phase::RecordBarrier);
    // What this draw records that a render pass instance cannot contain, on the
    // two ladders [`PassObstacles`] keeps.
    //
    // Every site below that emits a barrier, a copy or a dispatch names itself
    // here, *after* whatever `continue` decides the site is a no-op for this
    // draw — a loop that skips every element records nothing and must not claim
    // it did. The first obstacle wins on each ladder, because the question is
    // whether the pass standing from the previous draw survived to here, and the
    // first thing recorded is what ended it.
    //
    // This decides nothing. It is read once, at the `vkCmdBeginRenderPass`
    // below, to charge this draw to one bucket of `passmerge_*` and one of
    // `passheld_*`.
    let mut outside_pass = PassObstacles::default();
    let echo = super::pools::PassEcho {
        cb,
        compatibility: pass_key.compatibility(),
        fb: target_fb,
        // Decides nothing — `fb` is a function of the views and already covers
        // it. It is here so the census can tell a target switch apart from one
        // target described two ways; see `ResourcePools::pass_echo_delta`.
        target_image,
        area: (framebuffer_width, framebuffer_height),
    };
    let target_feedback = pass_key.color_feedback(0);
    let target_pass_layout = pass_key.color_layout(0);
    let target_dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | if target_feedback {
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
        } else {
            vk::PipelineStageFlags::empty()
        };
    let target_dst_access = vk::AccessFlags::COLOR_ATTACHMENT_READ
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | if target_feedback {
            vk::AccessFlags::SHADER_READ
        } else {
            vk::AccessFlags::empty()
        };
    let target_dependency = if target_feedback {
        vk::DependencyFlags::FEEDBACK_LOOP_EXT
    } else {
        vk::DependencyFlags::empty()
    };

    phase.enter(super::draw_phase::Phase::RecBarrierImportedTest);
    // Publish writes made through any other Vulkan alias of guest memory before
    // this draw consumes an imported buffer or image. This is deliberately one
    // memory dependency for the draw: the physical payload may be shared by a
    // RAMBlock buffer, a packed mapping buffer, and child images, so no one
    // resource barrier can name the complete producer set.
    let reads_imported_guest = vertex_bufs.iter().any(|(_, b)| b.guest_import)
        || index_slot.is_some_and(|b| b.guest_import)
        || storage_slots.iter().any(|(_, b, _)| b.guest_import)
        || !guest_gathers.is_empty()
        || target_guest_texels
            .as_ref()
            .is_some_and(GuestTexels::is_imported)
        || target_guest_memory.is_some()
        || mrt_secondaries
            .iter()
            .any(|(_, _, _, guest)| guest.is_some())
        || sampled.iter().any(|prepared| match prepared {
            PreparedSampled::GuestGather {
                source: GuestTexels::Imported { .. },
                ..
            } => true,
            // A resident whose image *is* the guest allocation reads bytes the
            // guest CPU and every other alias of those pages may have written.
            // The registry owns whether that is so, which is why this asks it
            // rather than carrying a second flag out of the sampled loop.
            PreparedSampled::Resident { identity, .. } => {
                pools.resident_reads_imported_guest(identity)
            }
            _ => false,
        });
    if reads_imported_guest {
        phase.enter(super::draw_phase::Phase::RecBarrierReadSet);
        let mut read_pages: Vec<Option<reims_vgpu_memory::GuestPageSet>> = Vec::new();
        for attr in &req.vertex_attributes {
            read_pages
                .extend(guest_buffer_physical_pages(&attr.content).map(|pages| pages.cloned()));
        }
        if let Some(indexed) = &req.indexed {
            read_pages
                .extend(guest_buffer_physical_pages(&indexed.content).map(|pages| pages.cloned()));
        }
        for storage in &req.storage_buffers {
            read_pages
                .extend(guest_buffer_physical_pages(&storage.content).map(|pages| pages.cloned()));
        }
        for sampled in &req.sampled_images {
            let source = match &sampled.source {
                SampledSource::GuestRuns(source, _) => Some(source),
                SampledSource::GuestImage(image, _) => Some(&image.transfer),
                SampledSource::Null
                | SampledSource::Bytes(_)
                | SampledSource::Target(_)
                | SampledSource::Attachment { .. } => None,
            };
            if let Some(source) = source {
                read_pages.push(source.physical_pages.clone());
            }
        }
        if let Some(seed) = req.target_guest.as_ref().and_then(|target| target.seed()) {
            read_pages.push(seed.source.physical_pages.clone());
        }
        // A continued pass keeps using the same attachment object. Its prior
        // draw's attachment write is ordered inside that pass, not a read
        // through a second imported-memory alias. Any descriptor or other
        // attachment that aliases these pages remains in `read_pages` and
        // therefore still forces the global dependency.
        let continues_imported_target = req.continues_render_pass && pools.open_pass_echoes(&echo);
        if imported_target_needs_visibility(
            target_guest_memory.is_some(),
            continues_imported_target,
        ) {
            read_pages.push(target_guest_write_pages.clone());
        }
        read_pages.extend(
            mrt_secondaries
                .iter()
                .filter_map(|(_, _, _, pages)| pages.as_ref())
                .map(|pages| Some(pages.clone())),
        );
        phase.enter(super::draw_phase::Phase::RecBarrierVisibility);
        if read_pages.is_empty() {
            // The only imported object was the attachment of the render pass
            // that is still open. No cross-object read remains to publish.
            // Keeping this arm here also prevents an empty set from becoming
            // `GpuUnknown`, whose conservative barrier would close the pass.
            crate::telemetry::note_route("guest_visibility_same_attachment");
        } else {
            if let Some(visibility) = pools.imported_guest_barrier(cb, || {
                counters.guest_visibility_read_sets.fetch_add(
                    read_pages.len() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                counters.guest_visibility_read_pages.fetch_add(
                    read_pages
                        .iter()
                        .filter_map(Option::as_ref)
                        .map(|pages| pages.pages().len() as u64)
                        .sum(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                let visibility = imported_guest_visibility(&read_pages);
                note_imported_guest_visibility(counters, visibility);
                visibility
            }) {
                phase.enter(super::draw_phase::Phase::RecBarrierPassBreak);
                unsafe {
                    outside_pass.before_record(
                        PassObstacle::GuestMemoryVisibility,
                        pools,
                        &ctx.device,
                        cb,
                    )
                };
                let barrier = [imported_guest_read_barrier(
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::UNIFORM_READ
                        | vk::AccessFlags::INDEX_READ
                        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    visibility,
                )];
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    imported_guest_read_stage(visibility),
                    vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::ALL_GRAPHICS,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
        }
    }

    phase.enter(super::draw_phase::Phase::RecBarrierSnapshot);
    // Fallback for attachment feedback loops the optional native contract
    // cannot represent (unsupported host, depth, or a non-identity view):
    // capture the prior resident content into a same-format GPU image before
    // changing the attachment. This preserves Metal's semantics without a
    // readback or host upload.
    let mut snapshotted_targets = std::collections::HashSet::new();
    let mut snapshotted_images = std::collections::HashSet::new();
    let mut target_snapshotted = false;
    for sampled_image in &sampled {
        let PreparedSampled::Snapshot {
            identity,
            source_image,
            source_access,
            next_access,
            image,
            timing,
            ..
        } = sampled_image
        else {
            continue;
        };
        if *timing != SnapshotTiming::Prior {
            continue;
        }
        target_snapshotted = true;
        unsafe { outside_pass.before_record(PassObstacle::Snapshot, pools, &ctx.device, cb) };
        // Duplicate descriptor bindings of one attachment/key share one image.
        // Its copy and two layout transitions are commands on the image, not on
        // the binding, so recording them twice would transition
        // SHADER_READ_ONLY as though it were UNDEFINED and copy the same pixels
        // twice for no guest-visible difference.
        if !snapshotted_images.insert(image.image) {
            continue;
        }
        // Once per distinct source: duplicate bindings of one target share the
        // image, so a second barrier for it would order nothing new. The
        // *first* is unconditional — the source is a registry resident this
        // draw's own predecessor may have written, and the layout it sits in
        // says nothing about that.
        if snapshotted_targets.insert(identity.clone()) {
            barrier_resident_for_transfer_read(
                &ctx.device,
                cb,
                *source_image,
                *source_access,
                *next_access,
            );
        }
        if *timing == SnapshotTiming::Undefined {
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.image)
                .subresource_range(super::color_subresource_range())];
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            continue;
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::PREINITIALIZED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::ImageCopy::default()
            .src_subresource(super::color_subresource_layers())
            .dst_subresource(super::color_subresource_layers())
            .extent(vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image(
            cb,
            *source_image,
            next_access.layout(),
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    phase.enter(super::draw_phase::Phase::RecBarrierSeed);
    // Seed upload (CPU import).
    if let Some(seed) = &seed_slot {
        unsafe { outside_pass.before_record(PassObstacle::Seed, pools, &ctx.device, cb) };
        let (src_stage, src_access) =
            target_prior_access(target_snapshotted, target_access).source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::BufferImageCopy::default()
            .image_subresource(super::color_subresource_layers())
            .image_extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            seed.buffer,
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(target_dst_access)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &barrier,
        );
    } else if target_guest_texels.is_some() {
        // Recorded after the gather shared with sampled/buffer sources below.
    } else if let Some((seed_image, seed_access, seed_next_access)) = seed_from_resolved {
        unsafe { outside_pass.before_record(PassObstacle::Seed, pools, &ctx.device, cb) };
        // GPU present-boundary seed: resident front frame → draw target copy,
        // then the pass runs with LOAD.
        //
        // The source is a resident that a draw just produced, so it is normally
        // already in TRANSFER_SRC_OPTIMAL and gating on a transition being
        // needed skipped the dependency on exactly the frames worth copying.
        barrier_resident_for_transfer_read(
            &ctx.device,
            cb,
            seed_image,
            seed_access,
            seed_next_access,
        );
        let (dst_stage, dst_access) =
            target_prior_access(target_snapshotted, target_access).source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(dst_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            dst_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let region = [vk::ImageCopy::default()
            .src_subresource(super::color_subresource_layers())
            .dst_subresource(super::color_subresource_layers())
            .extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image(
            cb,
            seed_image,
            seed_next_access.layout(),
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(target_dst_access)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &barrier,
        );
        counters
            .seed_gpu_copies
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        counters.seed_gpu_copy_bytes.fetch_add(
            (req.width as u64) * (req.height as u64) * 4,
            std::sync::atomic::Ordering::Relaxed,
        );
    } else if load_uses_gpu_content
        && !(target_feedback && req.continues_render_pass && pools.open_pass_echoes(&echo))
        && (loads_direct_guest
            || !pass_exit_needs_no_barrier(target_prior_access(target_snapshotted, target_access)))
    {
        unsafe { outside_pass.before_record(PassObstacle::TargetLayout, pools, &ctx.device, cb) };
        // A prior direct sample may have left this target shader-readable, or a
        // readback may have left it a transfer source; transition from the
        // registry's tracked layout back to attachment use.
        let prior = target_prior_access(target_snapshotted, target_access);
        let (mut src_stage, mut src_access) = prior.source_scope();
        if loads_direct_guest {
            src_stage |= imported_guest_write_stage();
            src_access |= imported_guest_write_access();
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(target_dst_access)
            .old_layout(prior.layout())
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &barrier,
        );
    } else if !load_uses_gpu_content
        && (target_snapshotted || target_access != super::pools::ResidentAccess::Untouched)
    {
        unsafe { outside_pass.before_record(PassObstacle::ClearWait, pools, &ctx.device, cb) };
        // The Clear render pass discards prior content via initialLayout
        // UNDEFINED, so nothing here preserves pixels — but its colour writes
        // still have to wait for whoever last read them, and on this path the
        // render pass supplies no such wait. The colour-only pass declares no
        // external subpass dependency, and Vulkan's implicit one carries
        // `srcStageMask = TOP_OF_PIPE` with `srcAccessMask = 0`, which orders
        // against nothing at all.
        //
        // `target_snapshotted` is this draw's own snapshot read and names the
        // newer access. Otherwise the registry's tracked layout names the
        // previous draw's: `SHADER_READ_ONLY_OPTIMAL` when it sampled this
        // resident, `TRANSFER_SRC_OPTIMAL` when it read it back or presented
        // it. Both are reads that a clear would otherwise be free to overtake.
        //
        // A pooled or freshly created target tracks `UNDEFINED` — nothing has
        // touched it, so it is excluded rather than barriered, which keeps this
        // off the pooled path entirely.
        let (src_stage, src_access) =
            target_prior_access(target_snapshotted, target_access).source_scope();
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    phase.enter(super::draw_phase::Phase::RecBarrierMaterialize);
    // The birth copy an image aliasing guest pages owes. `VK_EXT_external_
    // memory_host` forces such an image to be born `UNDEFINED`, and the first
    // transition out of that layout is free to discard the memory it aliases —
    // so the guest's own texels, which were in those pages before the image
    // existed, have to be written back through an operation Vulkan counts as a
    // write to the image.
    //
    // The bytes are laundered through staging rather than copied straight from
    // the import buffer, because that buffer and this image are two aliases of
    // one allocation and a transfer whose regions overlap is undefined. The
    // launder reads the same bytes it writes, so the copy is byte-for-byte an
    // identity: what it buys is a defined layout over contents Vulkan agrees
    // are the image's.
    //
    // Recorded before the transition loop below, and reported to the registry
    // as leaving the image in the layout a sampled read wants, so that loop
    // finds nothing to place. The imported-memory dependency above already
    // made the guest's host writes available to this TRANSFER read.
    let mut materialized_alias = std::collections::HashSet::new();
    for image in &sampled {
        let PreparedSampled::Resident {
            identity,
            image,
            next_access,
            materialize: Some(seed),
            ..
        } = image
        else {
            continue;
        };
        // Two bindings of one texture are two descriptors over one image, and
        // the copy is owed by the image.
        if !materialized_alias.insert(identity.clone()) {
            continue;
        }
        unsafe { outside_pass.before_record(PassObstacle::SampledUpload, pools, &ctx.device, cb) };
        let staging = unsafe {
            pools.acquire_staging(
                ctx,
                seed.bytes,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                counters,
            )
        }?;
        unsafe {
            ctx.device.cmd_copy_buffer(
                cb,
                seed.source,
                staging.buffer,
                &[vk::BufferCopy {
                    src_offset: seed.source_offset,
                    dst_offset: 0,
                    size: seed.bytes,
                }],
            );
            let landed = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(staging.buffer)
                .offset(0)
                .size(seed.bytes)];
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &landed,
                &[],
            );
            materialize_alias_levels(ctx, cb, staging.buffer, *image, seed, next_access.layout());
        }
        counters.note_sampled_guest_materialized(seed.bytes);
        // Only now: until the copy is in the command buffer the resident keeps
        // owing it, so a draw abandoned above leaves the next bind to record it
        // again rather than sampling an image nothing ever wrote.
        pools.registry_note_materialized(identity, *next_access);
    }

    phase.enter(super::draw_phase::Phase::RecBarrierResident);
    // Resident samples: transition the persistent target in place. Duplicate
    // bindings of one target share the same image and therefore one barrier.
    let mut transitioned_resident = std::collections::HashSet::new();
    for image in &sampled {
        let PreparedSampled::Resident {
            identity,
            image,
            access,
            next_access,
            levels,
            ..
        } = image
        else {
            continue;
        };
        // A resident whose last touch was already a shader read needs no
        // barrier: read-after-read is not a hazard, and the layout it wants is
        // the one it is in. This is the one skip in the family that is sound,
        // and it is sound because the *access* says so — the identically-shaped
        // skip keyed on the layout alone is what `ResidentAccess` exists to
        // stop, since a layout can be reached by a write that shares its name.
        //
        // The second skip is the layout one, and it is sound only because both
        // halves are asked. `layout() == layout()` says there is nothing to
        // *place*, which is true for every resident once a colour target rests
        // in one layout; `covered_by_pass_entry` says the pass's own incoming
        // external dependency already makes the prior access *visible* to this
        // draw's sampled read, which is a separate question and the one that
        // carries the hazard. Asking only the first is exactly the mistake
        // `ResidentAccess` exists to stop — a layout can be reached by a write
        // that shares its name.
        //
        // This is what retires `passmerge_outside_resident_layout`: the barrier
        // it charged is not moved earlier or made cheaper, it stops being owed.
        if !transitioned_resident.insert(identity.clone())
            || access == next_access
            || (access.layout() == next_access.layout() && access.covered_by_pass_entry())
        {
            continue;
        }
        unsafe { outside_pass.before_record(PassObstacle::ResidentLayout, pools, &ctx.device, cb) };
        let (src_stage, src_access) = access.source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(access.layout())
            .new_layout(next_access.layout())
            .image(*image)
            // Every level, because a mipmapped guest alias is one image whose
            // tail levels are read by the same sample as level zero.
            .subresource_range(vk::ImageSubresourceRange {
                level_count: *levels,
                ..super::color_subresource_range()
            })];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    phase.enter(super::draw_phase::Phase::RecBarrierUpload);
    // CPU-origin sampled uploads.
    for image in &sampled {
        let PreparedSampled::Upload {
            binding,
            image: img,
            staging: st,
            volume,
            layers,
            ..
        } = image
        else {
            continue;
        };
        unsafe { outside_pass.before_record(PassObstacle::SampledUpload, pools, &ctx.device, cb) };
        upload_buffer_to_sampled_image(
            ctx,
            cb,
            st.buffer,
            0,
            img.image,
            img.width,
            img.height,
            if *volume { 1 } else { *layers },
            if *volume { *layers } else { 1 },
            0,
            None,
            *binding,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;
    }

    // Scattered guest buffer windows, assembled into their device-local
    // destinations before anything reads them.
    //
    // The sources are guest RAM imported from QEMU's host mapping. Guest CPU
    // stores are HOST writes in Vulkan's memory model even though this crate did
    // not issue them. Publish those writes before either the transfer copies or
    // the compute gather reads the sources.
    //
    // Either form lands byte-identical bytes; which one ran decides only this
    // barrier's source scope, and `buffer_gather_dispatches` says which it was.
    if !guest_gathers.is_empty() {
        unsafe { outside_pass.before_record(PassObstacle::Gather, pools, &ctx.device, cb) };
    }
    let gather_dispatched = if guest_gathers.is_empty() || !compute_gather_enabled() {
        false
    } else {
        match unsafe { plan_buffer_gather_dispatches(ctx, pools, counters, &guest_gathers) }? {
            Some(groups) => {
                let pipeline = unsafe { pools.scatter_pipeline(ctx) }
                    .expect("planned only after the pipeline was created");
                // Bound once for the run: `record` is 39 % of a dispatch's
                // cost and this was one of its four driver calls, repeated for
                // a handle that never changes.
                unsafe { pipeline.bind(&ctx.device, cb) };
                for g in &groups {
                    let _r = super::gather_phase::Span::open(super::gather_phase::Part::Record);
                    unsafe { pipeline.dispatch(&ctx.device, cb, g.set, g.run_count) };
                }
                counters
                    .buffer_gather_dispatches
                    .fetch_add(groups.len() as u64, std::sync::atomic::Ordering::Relaxed);
                true
            }
            None => false,
        }
    };
    if !gather_dispatched {
        for gather in &guest_gathers {
            for (source, copies) in &gather.sources {
                ctx.device.cmd_copy_buffer(cb, *source, gather.dst, copies);
            }
        }
    }
    if !guest_gathers.is_empty() {
        // `ALL_GRAPHICS` rather than the exact stages: a gathered window is
        // bound as a vertex stream or as a storage buffer, and a storage bind is
        // readable from every shader stage of the pass. Naming them individually
        // would be a list that has to be revisited whenever a new bind kind
        // reaches this rail, for a dependency the driver resolves against the
        // pass it actually recorded.
        //
        // `TRANSFER`/`TRANSFER_READ` is in the destination scope because a
        // sampled window gathers into one of these slots and is then read by the
        // buffer→image copy below, which is a transfer and not a graphics stage.
        // Without it that copy races the gather that fills it.
        //
        // The source scope follows whichever form ran, and only that form: a
        // barrier naming both would be correct but would say the copies might
        // have been a compute write on a boot where they were not, which is a
        // dependency the driver then has to honour for nothing.
        let (src_stage, src_access) = if gather_dispatched {
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            )
        };
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                    | vk::AccessFlags::INDEX_READ
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ,
            )];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }
    // Every gather this draw owed is now in the command buffer, ordered ahead of
    // the draw by the barrier above, so the bind memo entries that were waiting
    // on one are answerable. Unconditional and outside the `is_empty` guard:
    // a draw with no gathers of its own must still not carry a previous draw's
    // owed list, and the list is empty in that case anyway.
    pools.note_cb_gathers_recorded();

    // Guest-page attachment seed. Its source may be a direct RAMBlock import,
    // the device-local result of the gather just recorded, or the exact CPU
    // fallback. All three expose one buffer range and therefore share the same
    // target transition and copy.
    if let (Some(source), Some(seed)) = (&target_guest_texels, target_guest_seed) {
        unsafe { outside_pass.before_record(PassObstacle::Seed, pools, &ctx.device, cb) };
        let (src_stage, src_access) =
            target_prior_access(target_snapshotted, target_access).source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::BufferImageCopy::default()
            .buffer_offset(source.offset())
            .buffer_row_length(seed.source.row_length_texels)
            .image_subresource(super::color_subresource_layers())
            .image_extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            source.buffer(),
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(target_dst_access)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &barrier,
        );
    }

    phase.enter(super::draw_phase::Phase::RecBarrierSnapshot);
    // An attachment sampled during the draw names the attachment itself, not a
    // second CPU image. Native feedback binds it directly. The capability
    // fallback below prepares the descriptor snapshot after the attachment's
    // declared initial contents have been established: Clear fills the
    // snapshot with the same clear value, a seeded primary is copied from the
    // target after its GPU seed operation above, and DontCare transitions an
    // unwritten image so its sampled value remains explicitly undefined.
    let after_seed_snapshot = sampled.iter().any(|sampled| {
        matches!(
            sampled,
            PreparedSampled::Snapshot {
                timing: SnapshotTiming::AfterPrimarySeed,
                ..
            }
        )
    });
    if after_seed_snapshot {
        unsafe { outside_pass.before_record(PassObstacle::Snapshot, pools, &ctx.device, cb) };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(target_dst_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(target_pass_layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            target_dst_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
    for sampled_image in &sampled {
        let PreparedSampled::Snapshot { image, timing, .. } = sampled_image else {
            continue;
        };
        if *timing == SnapshotTiming::Prior || !snapshotted_images.insert(image.image) {
            continue;
        }
        unsafe { outside_pass.before_record(PassObstacle::Snapshot, pools, &ctx.device, cb) };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        match timing {
            SnapshotTiming::Clear(clear) => ctx.device.cmd_clear_color_image(
                cb,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: *clear },
                &[super::color_subresource_range()],
            ),
            SnapshotTiming::AfterPrimarySeed => {
                let copy = [vk::ImageCopy::default()
                    .src_subresource(super::color_subresource_layers())
                    .dst_subresource(super::color_subresource_layers())
                    .extent(vk::Extent3D {
                        width: image.width,
                        height: image.height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_image(
                    cb,
                    target_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy,
                );
            }
            // DontCare makes the attachment contents undefined; the image
            // still needs the layout transitions surrounding this match, but
            // there is deliberately no initialization command to record.
            SnapshotTiming::Undefined => {}
            SnapshotTiming::Prior => unreachable!("filtered above"),
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
    if after_seed_snapshot {
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(target_dst_access)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &barrier,
        );
    }

    phase.enter(super::draw_phase::Phase::RecBarrierUpload);
    // Guest-sourced sampled uploads: one buffer→image copy over either the
    // guest's imported pages or the scratch the CPU packed them into, differing
    // from the CPU-origin loop above only in `row_length_texels` striding over
    // guest row padding (0 = tight rows) and in the copy's `bufferOffset`.
    //
    // Scratch writes made before `vkQueueSubmit` are automatically visible to
    // the device. Direct imports consume the guest-memory dependency recorded
    // once above, which covers both CPU writes and GPU writes through aliases.
    let mut recorded_sampled_guest = Vec::new();
    for image in &sampled {
        let PreparedSampled::GuestGather {
            binding,
            image: img,
            source,
            row_length_texels,
            layout,
            allocation_copy,
            volume,
            layers,
            reuse,
            ..
        } = image
        else {
            continue;
        };
        unsafe { outside_pass.before_record(PassObstacle::SampledUpload, pools, &ctx.device, cb) };
        if let Some(copy) = allocation_copy {
            upload_buffer_to_sampled_allocation(
                ctx,
                cb,
                SampledAllocationUpload {
                    src: source.buffer(),
                    src_offset: source.offset(),
                    image: img.image,
                    array_layers: if *volume { 1 } else { *layers },
                    image_arrayed: img.arrayed,
                    binding: *binding,
                    copy,
                },
            )?;
        } else {
            upload_buffer_to_sampled_image(
                ctx,
                cb,
                source.buffer(),
                source.offset(),
                img.image,
                img.width,
                img.height,
                if *volume { 1 } else { *layers },
                if *volume { *layers } else { 1 },
                *row_length_texels,
                *layout,
                *binding,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?;
        }
        recorded_sampled_guest.push(((**reuse).clone(), img.handles()));
    }
    // A recoverable refusal above abandons this draw's command recording. Do
    // not publish any of its images until every owed copy is in the command
    // buffer, or a later draw could bind a recycled image's previous tenant.
    for (source, image) in recorded_sampled_guest {
        pools.note_cb_sampled_guest(source, &image);
    }

    phase.enter(super::draw_phase::Phase::RecBarrierAttachment);
    // A Vulkan render-pass clear covers only its render area. Metal instead
    // applies each attachment's load action to that attachment's full image,
    // then rasterizes against the minimum attachment extent. When one MRT
    // image is larger than that common framebuffer, materialize its clear over
    // the whole image and make the render pass LOAD the result.
    if preclear_primary {
        unsafe {
            outside_pass.before_record(PassObstacle::AttachmentLoad, pools, &ctx.device, cb);
            record_attachment_wide_clear(
                &ctx.device,
                cb,
                target_image,
                AttachmentWideColorClear {
                    prior: target_prior_access(target_snapshotted, target_access),
                    pass_layout: target_pass_layout,
                    pass_stage: target_dst_stage,
                    pass_access: target_dst_access,
                    dependency: target_dependency,
                    clear: req.target_clear,
                },
            );
        }
    }
    for (secondary_index, (_id, image, access, _guest)) in mrt_secondaries.iter().enumerate() {
        if !preclear_secondaries[secondary_index] {
            continue;
        }
        let attachment_index = secondary_index + 1;
        let feedback = pass_key.color_feedback(attachment_index);
        let pass_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | if feedback {
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
            } else {
                vk::PipelineStageFlags::empty()
            };
        let pass_access = vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | if feedback {
                vk::AccessFlags::SHADER_READ
            } else {
                vk::AccessFlags::empty()
            };
        unsafe {
            outside_pass.before_record(PassObstacle::AttachmentLoad, pools, &ctx.device, cb);
            record_attachment_wide_clear(
                &ctx.device,
                cb,
                *image,
                AttachmentWideColorClear {
                    prior: *access,
                    pass_layout: pass_key.color_layout(attachment_index),
                    pass_stage,
                    pass_access,
                    dependency: if feedback {
                        vk::DependencyFlags::FEEDBACK_LOOP_EXT
                    } else {
                        vk::DependencyFlags::empty()
                    },
                    clear: req.secondary_targets[secondary_index].clear,
                },
            );
        }
    }
    if preclear_depth {
        let depth = req.depth.as_ref().expect("preclear implies depth state");
        let attachment = depth_attachment
            .as_ref()
            .expect("preclear implies an acquired depth attachment");
        unsafe {
            outside_pass.before_record(PassObstacle::AttachmentLoad, pools, &ctx.device, cb);
            record_attachment_wide_depth_clear(
                &ctx.device,
                cb,
                attachment,
                depth.clear_value,
                depth.stencil.map(|stencil| stencil.clear_value),
            );
        }
    }

    // MRT secondary attachments that were left shader-readable (sampled by a
    // prior draw) must transition back to color-attachment use, and the write
    // must wait for that prior read (WAR). A freshly-created secondary tracks
    // UNDEFINED and needs no barrier — the render pass discards on CLEAR.
    for (secondary_index, (_id, image, access, _guest)) in mrt_secondaries.iter().enumerate() {
        if preclear_secondaries[secondary_index] {
            continue;
        }
        if *access == super::pools::ResidentAccess::Untouched {
            continue;
        }
        let attachment_index = secondary_index + 1;
        let feedback = pass_key.color_feedback(attachment_index);
        if feedback && req.continues_render_pass && pools.open_pass_echoes(&echo) {
            continue;
        }
        let dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | if feedback {
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
            } else {
                vk::PipelineStageFlags::empty()
            };
        let dst_access = vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | if feedback {
                vk::AccessFlags::SHADER_READ
            } else {
                vk::AccessFlags::empty()
            };
        unsafe { outside_pass.before_record(PassObstacle::MrtLayout, pools, &ctx.device, cb) };
        let (src_stage, src_access) = access.source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(access.layout())
            .new_layout(pass_key.color_layout(attachment_index))
            .image(*image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            if feedback {
                vk::DependencyFlags::FEEDBACK_LOOP_EXT
            } else {
                vk::DependencyFlags::empty()
            },
            &[],
            &[],
            &barrier,
        );
    }

    let clear = clear_values(req);
    phase.enter(super::draw_phase::Phase::RecordPass);
    let rp_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(target_fb)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: framebuffer_width,
                height: framebuffer_height,
            },
        })
        .clear_values(&clear);
    // The pool is created per queried draw and destroyed once this draw's fence
    // has been waited on, a few lines below — at which point the command buffer
    // that names it has completed, which is the whole of Vulkan's valid-usage
    // requirement for destroying it. That is deliberately simpler than the
    // `TimestampProbe` shape (one pool for the device's life): this pool cannot
    // be shared with a concurrent submission because there is no concurrent
    // submission to share it with, a queried draw having just been excluded
    // from batching. If `note_create` ever shows these in volume, pooling them
    // per ring slot is the next step — it is not one worth taking before a
    // guest has armed a single query.
    //
    // `vkCmdResetQueryPool` must be recorded outside a render pass instance,
    // which is why it sits here rather than beside the `vkCmdBeginQuery`.
    let occlusion = match occlusion_flags {
        None => None,
        Some(flags) => {
            let ci = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::OCCLUSION)
                .query_count(1);
            let pool = ctx
                .device
                .create_query_pool(&ci, None)
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecCreateQueryPool, e)))?;
            counters.note_create(CreateSite::QueryPool);
            unsafe { outside_pass.before_record(PassObstacle::QueryReset, pools, &ctx.device, cb) };
            ctx.device.cmd_reset_query_pool(cb, pool, 0, 1);
            Some((pool, flags))
        }
    };
    // What this draw would have needed to continue the pass its predecessor
    // left standing, charged to exactly one bucket. `join_same_target` already
    // says how many draws render into the batch's own target; this says how many
    // of those could have shared the pass, which is the number the merge is
    // worth. The ladder is ordered so each rung names the *nearest* obstacle: a
    // draw that never joined has no pass to continue whatever else is true of
    // it, and a draw whose pass instance differs would need a new one even if it
    // recorded nothing.
    //
    // Each family's buckets are exhaustive over draws that reach here, which is
    // the cheapest way to catch a mis-placed `note`: summed over a census window
    // each family equals `chain_phase chains`, and `passmerge_no_join` alone
    // equals that count less `engine_delta batch_joins`. A total that falls
    // short means a draw took a path that skips this line; a `no_join` that
    // disagrees means `joins` and the batch counters have come apart.
    //
    // `passmerge_*` describes the old end/begin boundary, including layout work
    // caused by ending the pass. `passheld_*` removes those pass-end artifacts
    // and describes the continuation implemented below. An inherited pass is
    // closed at the exact recording site of every copy, barrier, dispatch, or
    // query reset Vulkan forbids inside it; this ladder therefore remains an
    // instrument for the continuation opportunities those commands consume.
    let continues = joins && pools.pass_echoes(&echo);
    let continues_open =
        continues_open_render_pass(req.continues_render_pass, pools.open_pass_echoes(&echo));
    // `passmerge_pass_differs` is one bucket over four independent causes with
    // four different repairs, and on a driven Maps leg it is the bucket that
    // holds 82 % of the draws. Split it where it is charged, so the two stay a
    // partition of each other rather than two counts of loosely the same thing.
    if joins && !continues {
        if let Some(field) = pools.pass_echo_delta(&echo) {
            crate::telemetry::note_route(field.route());
            // `passdiff_compat` is itself one bucket over nine attachment-shape
            // fields, and it became the dominant one when the framebuffer
            // identity blocker was fixed. The finer route rides along with the
            // coarse one so the two cannot be charged apart.
            if let Some(detail) = field.detail_route() {
                crate::telemetry::note_route(detail);
            }
        }
    }
    crate::telemetry::note_route(if !joins {
        "passmerge_no_join"
    } else if !continues {
        "passmerge_pass_differs"
    } else {
        outside_pass
            .first
            .map_or("passmerge_reachable", PassObstacle::route)
    });
    crate::telemetry::note_route(if !joins {
        "passheld_no_join"
    } else if !continues {
        "passheld_pass_differs"
    } else {
        outside_pass
            .first_held
            .map_or("passheld_reachable", PassObstacle::held_route)
    });
    if continues_open {
        crate::telemetry::note_route("pass_continued");
        counters
            .render_pass_continuations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        // An encoder boundary, a different pass instance, or an intervening
        // outside-pass command closes whatever the predecessor left open. A
        // continuation reopened here must LOAD regardless of the encoder's
        // original begin action; keep that contract repair measurable.
        unsafe { pools.close_open_pass(&ctx.device, cb) };
        if req.continues_render_pass {
            crate::telemetry::note_route(match req.color_load_action {
                super::types::ColorLoadAction::Load => "passreopen_from_load",
                super::types::ColorLoadAction::Clear => "passreopen_from_clear",
                super::types::ColorLoadAction::DontCare => "passreopen_from_dontcare",
            });
        }
        crate::telemetry::note_route(pass_begin_area_band(framebuffer_width, framebuffer_height));
        crate::telemetry::note_route(match pass_key.color0_load {
            ColorLoadKey::Load => "passbegin_load",
            ColorLoadKey::Clear => "passbegin_clear",
            ColorLoadKey::DontCare => "passbegin_dontcare",
        });
        crate::telemetry::note_route("passbegin_color0_resident");
        ctx.device
            .cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
        pools.note_pass_opened(echo);
    }
    if pass_key.feedback_colors != 0 {
        // Order each feedback draw's reads after the preceding colour writes.
        // This is deliberately a memory barrier inside the render pass: an
        // image transition here would be invalid, while closing the pass would
        // throw away the continuation this extension makes possible.
        //
        // A barrier inside a render pass may only name what one of that pass's
        // own self-dependencies names, so every term here is taken from the
        // constants the self-dependency is built from rather than spelled again.
        // The two disagreeing is not a slow path, it is invalid usage.
        let (src_stage, src_access) = super::caches::COLOR_FEEDBACK_SRC;
        let (dst_stage, dst_access) = super::caches::COLOR_FEEDBACK_DST;
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            vk::DependencyFlags::BY_REGION
                | super::feedback_transition_dependency(pass_key.color_layout(0)),
            &barrier,
            &[],
            &[],
        );
    }
    phase.enter(super::draw_phase::Phase::RecordState);
    // Only if this command buffer is not already carrying it — the three
    // `dynstate_*` skips below hang off this one call, because a pipeline change
    // is what invalidates them. See `super::pools::CbGraphicsState`.
    unsafe { pools.bind_graphics_pipeline(&ctx.device, cb, counters, pipeline, pipeline_layout) };
    if let Some((pool, flags)) = occlusion {
        ctx.device.cmd_begin_query(cb, pool, 0, flags);
    }

    // Dynamic viewport/scissor. Metal NDC is Y-up and Vulkan's is Y-down, so
    // every viewport is emitted flipped: origin at the bottom edge, negative
    // height. This is a property of the two APIs, not of any guest state.
    // One count for both, because a Vulkan pipeline declares one and the
    // dynamic arrays must match it. The pipeline was built from
    // `viewport_slot_count`, so this must be the same function of the same
    // request or `vkCmdSetViewport` binds a different count than the pipeline
    // declared.
    // Built into the pools' scratch rather than into two fresh `Vec`s: these are
    // rebuilt every draw, and the comparison that decides whether the driver
    // already has them needs a buffer it can swap rather than copy.
    let (vp_scratch, sc_scratch) = pools.dynamic_scratch();
    populate_dynamic_viewport_scissors(req, vp_scratch, sc_scratch);
    unsafe { pools.set_dynamic_viewport_scissor(&ctx.device, cb, counters) };
    if let Some([constant_factor, slope_factor, clamp]) = depth_bias {
        unsafe {
            ctx.device
                .cmd_set_depth_bias(cb, constant_factor, clamp, slope_factor)
        };
    }
    if draw_uses_blend_constants(req) {
        unsafe {
            ctx.device.cmd_set_blend_constants(cb, &req.blend_constants);
        }
    }
    // Dynamic stencil reference (Metal `setStencilFrontReferenceValue:back…`)
    // — only bound for stencil pipelines, which list STENCIL_REFERENCE as a
    // dynamic state; front/back set together because Metal's split refs are one
    // guest state and a cache that held half of it would be two.
    if let Some(s) = req.depth.as_ref().and_then(|d| d.stencil) {
        unsafe {
            pools.set_dynamic_stencil_reference(
                &ctx.device,
                cb,
                counters,
                s.reference_front,
                s.reference_back,
            )
        };
    }

    if push_descriptors {
        let push_state = pools.push_descriptor_scratch();
        push_state.extend(storage_slots.iter().zip(&buffer_infos).map(
            |((binding, _, _), info)| super::pools::PushDescriptorBinding::Buffer {
                binding: *binding,
                array_element: 0,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                buffer: info.buffer,
                offset: info.offset,
                range: info.range,
            },
        ));
        push_state.extend(sampled.iter().zip(&sampled_infos).map(|(image, info)| {
            super::pools::PushDescriptorBinding::Image {
                binding: image.binding(),
                array_element: image.array_element(),
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                sampler: info.sampler,
                view: info.image_view,
                layout: info.image_layout,
            }
        }));
        push_state.extend(sampler_handles.iter().zip(&sampler_infos).map(
            |((binding, _), info)| super::pools::PushDescriptorBinding::Image {
                binding: *binding,
                array_element: 0,
                ty: vk::DescriptorType::SAMPLER,
                sampler: info.sampler,
                view: info.image_view,
                layout: info.image_layout,
            },
        ));
        if req.color_input {
            push_state.push(super::pools::PushDescriptorBinding::Image {
                binding: super::types::COLOR_INPUT_BINDING,
                array_element: 0,
                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                sampler: color_input_info.sampler,
                view: color_input_info.image_view,
                layout: color_input_info.image_layout,
            });
        }
        if pools.push_descriptors_changed(pipeline_layout, counters) {
            ctx.push_descriptor
                .as_ref()
                .expect("push layout requires enabled entry points")
                .cmd_push_descriptor_set(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_writes,
                );
            counters.descriptor_pushes.fetch_add(1, Ordering::Relaxed);
        }
    } else if let Some(dset) = dset {
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[dset],
            &[],
        );
        counters
            .descriptor_set_binds
            .fetch_add(1, Ordering::Relaxed);
    }
    phase.enter(super::draw_phase::Phase::RecordDraw);
    unsafe { pools.bind_vertex_buffers(&ctx.device, cb, counters, &vertex_bufs) };
    match (&req.indexed, &index_slot) {
        (Some(indexed), Some(ibuf)) => {
            ctx.device.cmd_bind_index_buffer(
                cb,
                ibuf.buffer,
                ibuf.offset,
                crate::translate::raster::vk_index_type(indexed.index_type),
            );
            ctx.device.cmd_draw_indexed(
                cb,
                indexed.index_count,
                req.instance_count.unwrap_or(1),
                0,
                indexed.vertex_offset,
                req.base_instance,
            );
        }
        _ => {
            ctx.device.cmd_draw(
                cb,
                req.vertex_count,
                req.instance_count.unwrap_or(1),
                req.first_vertex,
                req.base_instance,
            );
        }
    }
    // Back to the remainder: the query end, the pass-close decision and
    // everything after it are the part of recording no sub-phase names.
    phase.enter(super::draw_phase::Phase::Record);
    if let Some((pool, _)) = occlusion {
        ctx.device.cmd_end_query(cb, pool, 0);
    }
    // Direct attachment writes use the same deferred submission rail as every
    // other guest-memory write. `record_guest_write_debt` below arms their exact
    // pages before this result is returned; the batch fence retires that debt,
    // and completion stamps are queued behind the batch. Waiting here would
    // turn every Metal encoder Store into a command-buffer completion point.
    let defer_submit = batch_eligible;
    let keep_pass_open = req.render_pass_continues
        && defer_submit
        && !pass_churn_probe_enabled()
        && !layout_churn_probe_enabled();
    if keep_pass_open {
        crate::telemetry::note_route("pass_left_open");
    } else {
        unsafe { pools.close_open_pass(&ctx.device, cb) };
    }
    if pass_churn_probe_enabled() && load_uses_gpu_content && !target_feedback {
        // PROBE — `REIMS_VGPU_PASS_CHURN=on`. One extra render pass instance on
        // the target this draw just finished with, loading and storing it and
        // drawing nothing into it.
        //
        // It prices the pair this device pays on every batched draw and would
        // stop paying if the pass were held open across a batch. `passheld_*`
        // says 82 % of draws could share one instance once the guest gathers
        // recorded between them are hoisted; hoisting needs a second command
        // buffer per batch, and this says what that work would be worth before
        // any of it is built. See [`reims_vgpu_config::PASS_CHURN`].
        //
        // Pixel-neutral, which is what makes it a control rather than a change:
        // `LOAD`/`STORE` preserves the attachment and no draw is recorded inside
        // the instance. `load_uses_gpu_content` gates it because a `CLEAR` pass
        // replayed here would clear away the draw's own output.
        //
        // No layout transition rides along any more, which narrows what this
        // probe measures rather than weakening it: a pass now exits at
        // [`super::caches::color0_pass_exit_layout`], which is what a `LOAD`
        // pass names as its `initial_layout`, so this barrier carries the
        // write-after-write dependency and nothing else. What separates the arms
        // is the pass instance alone. Use `REIMS_VGPU_LAYOUT_CHURN` to price a
        // transition; that is now the only arm that has one.
        let back = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(target_pass_layout)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &back,
        );
        ctx.device
            .cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
        ctx.device.cmd_end_render_pass(cb);
    }
    if layout_churn_probe_enabled() && !target_feedback {
        // PROBE — `REIMS_VGPU_LAYOUT_CHURN=on`. One round trip of the colour
        // attachment's layout, out of [`super::caches::color0_pass_exit_layout`]
        // into `TRANSFER_SRC_OPTIMAL` and straight back, recorded where the pass
        // has just left it.
        //
        // **This is now a re-enactment of what this device used to do on every
        // draw, and it is the arm that prices what removing it bought.** A pass
        // used to exit at `TRANSFER_SRC_OPTIMAL` so a present blit or readback
        // could read it untransitioned, and every draw that loaded its target
        // barriered it back — a full-attachment transition twice per draw, for a
        // reader that ran on about 5 % of them. On hardware that keeps colour
        // compression metadata (Intel CCS, AMD DCC, every tiler) each of those is
        // a decompress or recompress of the whole attachment; on hardware that
        // does not, it is a barrier and little else. Turning this switch on
        // restores the pair without restoring anything else.
        //
        // A positive control rather than a change: the pixels are identical
        // because both layouts preserve contents and nothing is recorded between
        // the two barriers, so `us/draw` moving is the cost of two transitions
        // and nothing else. It ends where it started, so nothing downstream
        // needs to know this ran.
        let out = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(super::caches::color0_pass_exit_layout())
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &out,
        );
        let back = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(super::caches::color0_pass_exit_layout())
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &back,
        );
    }

    if let Some(ref rb) = readback {
        let target_read_layout = super::pools::ResidentAccess::transfer_read().layout();
        // This copy needs a layout transition **and** a dependency, and the
        // render pass gives it neither. Vulkan's implicit final subpass
        // dependency carries `dstStageMask = BOTTOM_OF_PIPE` and
        // `dstAccessMask = 0`: it makes the colour writes available and visible
        // to nothing. Recording the copy into the same command buffer is not a
        // dependency either — commands in one buffer are free to overlap.
        //
        // Without the dependency half the readback can sample the attachment
        // before the draw it was recorded after has finished writing it, and the
        // bytes handed back are the ones from before the draw.
        //
        // The transition half used to be free, because the pass exited at
        // `TRANSFER_SRC_OPTIMAL` and this site could take it as read — a
        // `vkCmdCopyImageToBuffer` naming a `srcImageLayout` the image is not
        // actually in is undefined behaviour, not an error. It exits at
        // [`super::caches::color0_pass_exit_layout`] now, so the transition is
        // explicit, and it is put back afterwards so that
        // `registry_mark_ready_at`'s claim about this resident stays true — that
        // call runs below and records the pass's exit layout unconditionally.
        let to_transfer = [vk::ImageMemoryBarrier::default()
            .src_access_mask(target_dst_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(target_pass_layout)
            .new_layout(target_read_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            target_dst_stage,
            vk::PipelineStageFlags::TRANSFER,
            target_dependency,
            &[],
            &[],
            &to_transfer,
        );
        let region = [vk::BufferImageCopy::default()
            .image_subresource(super::color_subresource_layers())
            .image_extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            target_image,
            target_read_layout,
            rb.buffer,
            &region,
        );
        let back_to_exit = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(target_dst_access)
            .old_layout(target_read_layout)
            .new_layout(target_pass_layout)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            target_dst_stage,
            target_dependency,
            &[],
            &[],
            &back_to_exit,
        );
    }
    if target_guest_write_pages.is_some() && !req.render_pass_continues {
        // The attachment image aliases guest RAM, so its Store is a write to
        // memory the guest vCPU and host presentation path read directly. A
        // fence orders execution but does not make the color write visible to
        // HOST by itself. Release the pass's write before ending the command
        // buffer, matching the direct compute and image-copy rails.
        let host_visible = [imported_guest_attachment_release_barrier()];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &host_visible,
            &[],
            &[],
        );
    }
    // A batch-eligible draw defers end_command_buffer + submit: its CB stays
    // in recording state for same-target successors and is submitted by
    // pools.batch_flush (next begin_entry / retire / explicit flush).
    phase.enter(super::draw_phase::Phase::Submit);
    if !defer_submit {
        // Last command before the CB ends, so the stamp bounds every draw and
        // copy this submission recorded. A deferred draw is sealed by
        // `batch_flush` instead, on the same slot.
        unsafe { pools.gpu_span_seal_current(ctx, cb) };
        if let Err(e) = ctx.device.end_command_buffer(cb) {
            return Err(DrawError::VkCall(VkCall::new(VkOp::ExecEndCb, e)));
        }
    }

    if force_loss {
        // Recycle transient resources before reporting loss.
        if let (Some(ds), Some(pool)) = (dset, dset_pool) {
            pools.free_descriptor_sets(&ctx.device, &[(ds, pool)]);
        }
        pools.recycle_staging();
        pools.recycle_readback();
        pools.recycle_sampled();
        return Err(DrawError::DeviceLost(DeviceLostDecline::ForcedDraw));
    }

    let submitted_timeline = if !defer_submit {
        let cbs = [cb];
        match ctx.submit_guest_work(&cbs, fence) {
            Ok(timeline) => timeline,
            Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
                return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                    op: DeviceLostOp::DrawSubmit,
                    result: e,
                }));
            }
            Err(e) => {
                return Err(DrawError::VkCall(VkCall::new(VkOp::ExecSubmit, e)));
            }
        }
    } else {
        None
    };
    // Submission ends here. Everything below is CPU-side publication and
    // retention work, and needs its own bar: charging it to `submit_us` makes
    // a slow registry or Store-footprint update look like driver queue cost.
    phase.enter(super::draw_phase::Phase::PostTarget);
    // CPU-side bookkeeping: the retained target's content is queue-ordered
    // (mark ready), resident sampled layouts advance to the recorded
    // post-draw layout, and the sampled images this CB fills are named for the
    // cache admission that `finish_entry_async` makes below.
    if let Some(identity) = &req.target_identity {
        // How much of the attachment this draw could have written. Nothing in
        // this device acts on it — it is the standing instrument for whether
        // bounding a writeback to a damage rect could ever pay. See
        // `EngineCounters::note_draw_coverage` for the arithmetic and for the
        // reading that retired the rail built over it.
        //
        // The pass runs with LOAD only when the attachment already held the
        // content this draw builds on, and then the scissor bounds every
        // fragment that can write. Every other load action rewrites the whole
        // attachment first: a CLEAR clears the render area, which is the full
        // target, and both seed forms fill it.
        let rewrites_whole_attachment = !load_uses_gpu_content || segment_load.has_seed();
        // `any`, not the union: one scissor reaching the whole attachment is
        // enough for this draw to have written anywhere in it. A set of rects
        // that only covers the target *together* reads as partial here, which
        // over-states how much a damage-bounded flush could save rather than
        // under-stating it — the safe direction for an instrument nothing acts
        // on, and cheaper than computing a union of arbitrary rects.
        counters.note_draw_coverage(if rewrites_whole_attachment {
            super::counters::DrawCoverage::Full
        } else if pools.bound_scissors().iter().any(|s| {
            s.offset.x <= 0
                && s.offset.y <= 0
                && s.extent.width >= req.width
                && s.extent.height >= req.height
        }) {
            super::counters::DrawCoverage::LoadedFullScissor
        } else {
            super::counters::DrawCoverage::LoadedPartialScissor
        });
        if target_feedback {
            pools.registry_mark_ready_with_access(
                identity,
                super::pools::ResidentAccess::ColorFeedback(pass_key.color_final_layout(0)),
            );
        } else {
            pools.registry_mark_ready_at(identity, pass_key.color_final_layout(0));
        }
    }
    let guest_store_window = target_guest_memory.as_ref().and_then(|memory| {
        memory.backing.visible_window(
            req.width,
            req.height,
            u64::from(crate::translate::pixel::texel_layout_of(color0_format)?.bytes_per_texel()),
        )
    });
    phase.enter(super::draw_phase::Phase::PostStore);
    if let Some(write_pages) = &target_guest_write_pages {
        super::record_guest_write_debt(pools, super::GuestWriteSource::ImportedBuffer, write_pages);
    }
    phase.enter(super::draw_phase::Phase::PostTarget);
    // MRT secondary attachments settle at COLOR_ATTACHMENT_OPTIMAL (the pass
    // final layout) and become sampleable residents; the consumer's
    // resident-sample barrier then transitions COLOR_ATTACHMENT→SHADER_READ,
    // carrying the color-write→shader-read dependency. (The ad-hoc MRT
    // framebuffer is disposed below, after `finish_entry_async` — see there.)
    if is_mrt {
        for (secondary_index, (identity, _image, _old, guest)) in mrt_secondaries.iter().enumerate()
        {
            let attachment_index = secondary_index + 1;
            if pass_key.color_feedback(attachment_index) {
                pools.registry_mark_ready_with_access(
                    identity,
                    super::pools::ResidentAccess::ColorFeedback(
                        pass_key.color_final_layout(attachment_index),
                    ),
                );
            } else {
                // The pass's own `finalLayout` for this slot, asked of the key
                // rather than respelled. The feedback arm beside it already
                // derives; this one carried a hand-written
                // `COLOR_ATTACHMENT_OPTIMAL`, which is the second spelling
                // `color0_pass_exit_layout` exists to remove — a registry record
                // naming a layout the pass did not leave the image in is a later
                // barrier's wrong `oldLayout`.
                pools.registry_mark_ready_at(
                    identity,
                    pass_key.color_final_layout(attachment_index),
                );
            }
            if let Some(write_pages) = guest {
                super::record_guest_write_debt(
                    pools,
                    super::GuestWriteSource::ImportedBuffer,
                    write_pages,
                );
            }
        }
    }
    // The depth pass stores unconditionally and settles at
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL, so after this draw the resident holds
    // depth a later pass can load and sits in the layout that pass will name.
    // Marked here rather than beside the colour target above because the two
    // are different residents with different reclaim rules — see
    // `registry_mark_depth_ready` for the sole-copy line that is deliberately
    // absent from it.
    if let Some(identity) = depth_attachment.as_ref().and_then(|d| d.identity.as_ref()) {
        pools.registry_mark_depth_ready(identity);
    }
    phase.enter(super::draw_phase::Phase::PostSampled);
    let mut sampled_retains: Vec<super::pools::SampledRetain> = Vec::new();
    for prepared in &sampled {
        // Only an `Upload` can be retained by content: it is the one arm whose
        // bytes this device holds and can compare. Every other arm either reads
        // guest pages live or names a resident the registry already owns.
        if let PreparedSampled::Upload {
            binding,
            array_element,
            image,
            ..
        } = prepared
        {
            if let Some((SampledSource::Bytes(bytes), resource_lifetime)) =
                sampled_resource_at(req, *binding, *array_element)
                    .map(|resource| (&resource.source, resource.resource_lifetime.clone()))
            {
                sampled_retains.push(super::pools::SampledRetain {
                    image: image.image,
                    content: super::pools::SampledRetainContent::Bytes(bytes.clone()),
                    resource_lifetime,
                });
            }
        }
    }
    for image in &sampled {
        match image {
            PreparedSampled::Resident {
                identity,
                next_access,
                ..
            } => pools.registry_note_access(identity, *next_access),
            PreparedSampled::Snapshot {
                identity,
                next_access,
                ..
            } if req.attachment_slot(identity).is_none() => {
                pools.registry_note_access(identity, *next_access)
            }
            _ => {}
        }
    }
    if let Some((_, _, next_access)) = seed_from_resolved {
        if let Some(seed_identity) = segment_load.resident {
            pools.registry_note_access(seed_identity, next_access);
        }
    }
    // Deferred-submit draw: park the per-draw descriptor set on the open batch
    // (opening it if this is the first), hand the batch this draw's sampled
    // images for the content cache, and return. The CPU-side bookkeeping above
    // already ran — content_ready and tracked layouts describe what the recorded
    // CB produces, and every consumer path flushes the batch before touching the
    // GPU. The cache admission happens inside `batch_append` rather than at the
    // flush precisely so the *next* draw of this batch can find these windows;
    // see its doc.
    phase.enter(super::draw_phase::Phase::PostPark);
    if defer_submit {
        let target = batch_target.expect("batch_eligible requires target identity");
        pools.batch_append(
            (cb, fence),
            target,
            dset.zip(dset_pool),
            sampled_retains,
            counters,
        );
        // After the append, never before: installing the open batch is what puts
        // this slot into `open_slot_mask`, so these handles wait for the batch's
        // own work instead of being freed under a command buffer still recording
        // them. This is the same ordering rule the submitting path meets through
        // `finish_entry_async`, which is why both go through one function.
        dispose_ad_hoc_attachments(
            ctx,
            pools,
            ordinary_ad_hoc_framebuffer,
            target_fb,
            transient_depth,
        );
        counters
            .render_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(DrawOutput {
            pixels: Vec::new(),
            pixels_bgra: output_bgra,
            // Unreachable with a query armed: `batch_eligible` excludes one, so
            // `defer_submit` is false for every queried draw. Stated as `None`
            // rather than asserted because `None` is also the honest answer if
            // that exclusion is ever relaxed — a deferred draw genuinely has no
            // count yet.
            occlusion_samples: None,
            guest_store_pages: target_guest_write_pages.clone(),
            guest_store_window: guest_store_window.clone(),
        });
    }

    // Park the owed cleanup (descriptor set, transient pool slots, cache
    // admissions) on this ring slot in every mode; whichever entry retires
    // the slot drains it. A failed wait below leaves the slot pending, so no
    // path ever reuses an unretired fence.
    let sealed = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), sampled_retains);
    pools.finish_entry_async(sealed, submitted_timeline, None);

    // Dispose the ad-hoc per-draw framebuffers (MRT and/or depth) now that
    // `finish_entry_async` has marked this slot pending: the handles park in
    // the graveyard against the slots open right now — this draw's included —
    // and are freed once those retire. Disposing BEFORE this point would
    // immediate-free them (this slot is not yet pending, so it is not in the
    // open mask) while the just-submitted CB still references them → GPU fault.
    //
    // `transient_depth` carries the same handle whenever the draw has depth, so
    // the two arms below are exclusive by that test rather than by re-deriving
    // which features are in play — a depth MRT draw has one framebuffer and must
    // dispose it once.
    dispose_ad_hoc_attachments(
        ctx,
        pools,
        ordinary_ad_hoc_framebuffer,
        target_fb,
        transient_depth,
    );

    // A draw with no pixel readback (resident target, skip_readback) hands
    // the CPU nothing — skip the post-submit fence wait and return while the
    // GPU still runs on this ring slot.
    //
    // This is the whole population on a driven x86/Vulkan session. Summed over
    // one — Safari's WebGL aquarium, Wikipedia and apple.com with page scrolls
    // and title-bar drags — `render_post_wait_skips` and `draw_phase`'s own
    // `draws` are the same number, 49 592, so every draw took this return and
    // the wait/readback tail below ran zero times. Two counters incremented at
    // two unrelated sites agreeing exactly is what makes that a proof;
    // `draw_phase wait_us=0 readback_us=0` on its own cannot tell "never
    // entered" from "entered and immeasurably fast".
    //
    // That is a reading about the workload and **not** a licence to delete the
    // tail. `skip_readback` has to be decided before submit, and a Store that
    // neither defer rail can take still has to land its pixels: an IOSurface texture Store
    // always defers (`draw::vulkan` records why), but a GVA Store whose
    // `row_stride` is short of the format's tight row bytes fails
    // `gva_store_defer_eligible` and keeps its readback. Delete this and that
    // Store loses its frame silently, which is the one outcome the ground rules
    // forbid outright. What the equality licenses is not re-measuring it.
    let Some(ref rb) = readback else {
        // A queried draw has no pixels to read back and still cannot take this
        // return: the sample count *is* its result, and it is not readable
        // until the command buffer completes. So the wait the comment above
        // says this path exists to skip is exactly what a query reinstates —
        // for queried draws only, which on every workload measured so far is
        // none of them.
        if occlusion.is_some() {
            phase.enter(super::draw_phase::Phase::Wait);
            pools.wait_entry_fence(ctx, counters, fence)?;
            return Ok(DrawOutput {
                pixels: Vec::new(),
                pixels_bgra: output_bgra,
                occlusion_samples: read_occlusion_samples(ctx, occlusion)?,
                guest_store_pages: target_guest_write_pages.clone(),
                guest_store_window: guest_store_window.clone(),
            });
        }
        counters
            .render_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(DrawOutput {
            pixels: Vec::new(),
            pixels_bgra: output_bgra,
            occlusion_samples: None,
            guest_store_pages: target_guest_write_pages.clone(),
            guest_store_window: guest_store_window.clone(),
        });
    };

    // Wait ONLY this draw's fence, not the whole ring. The readback copy is the
    // tail of this CB, and single-queue submission order already guarantees it
    // observes every prior-submitted draw's writes (the same argument
    // `read_target_inner` relies on) — so `retire_all` here would just serialize
    // the guest-blocking readback behind an unrelated in-flight heavy draw (the
    // `finish_us` tail). The cleanup is already parked with `finish_entry_async`
    // above, so the slot stays pending and the ring retires it later with no
    // extra wait (its fence is already signaled).
    phase.enter(super::draw_phase::Phase::Wait);
    pools.wait_entry_fence(ctx, counters, fence)?;

    phase.enter(super::draw_phase::Phase::Readback);
    let out = super::pools::read_back_slot(
        ctx,
        rb,
        rb_size,
        VkOp::ExecMapReadback,
        VkOp::ExecInvalidateReadback,
    )?;
    counters.note_readback(rb_size, super::counters::ReadbackSource::DrawTail);

    // Read at the attachment's width above, narrowed here to the RGBA8 a
    // `DrawOutput` consumer speaks. Shared with `read_target`'s rail so the two
    // cannot answer a wide attachment differently — which they did, one
    // quantizing and one overrunning its slot.
    let layout = crate::translate::pixel::texel_layout_of(color0_format).ok_or(
        DrawError::TargetRead(super::reason::TargetReadDecline::TexelNotFourBytes {
            format: color0_format,
        }),
    )?;
    let (pixels, pixels_bgra) = super::narrow_readback_to_rgba8(
        out,
        layout,
        color0_format,
        (req.width as u64) * (req.height as u64),
        output_bgra,
    )?;

    Ok(DrawOutput {
        pixels,
        pixels_bgra,
        occlusion_samples: read_occlusion_samples(ctx, occlusion)?,
        guest_store_pages: target_guest_write_pages,
        guest_store_window,
    })
}

fn sampled_resource_at(
    req: &DrawRequest,
    binding: u32,
    array_element: u32,
) -> Option<&SampledImageResource> {
    req.sampled_images
        .iter()
        .find(|resource| resource.binding == binding && resource.array_element == array_element)
}

/// Read the sample count a queried draw produced, and destroy its pool.
///
/// **Only call this after the draw's fence has been waited on.** Both halves
/// depend on it: `WAIT` would otherwise be the only thing keeping the read
/// honest, and `vkDestroyQueryPool` requires every submitted command naming the
/// pool to have completed. Waiting is the caller's job because the two callers
/// reach it differently — one waits because it has pixels to read back, the
/// other waits *because* of the query.
///
/// `WAIT` is passed anyway rather than relied on: the fence says the command
/// buffer finished, which is the same guarantee, and asking for both costs
/// nothing while removing the question of which one is load-bearing. Without
/// either, `vkGetQueryPoolResults` may return `VK_NOT_READY` and leave the
/// destination untouched — a zero that reads exactly like a fully occluded
/// draw.
fn read_occlusion_samples(
    ctx: &super::context::DeviceContext,
    occlusion: Option<(vk::QueryPool, vk::QueryControlFlags)>,
) -> Result<Option<u64>, DrawError> {
    let Some((pool, _)) = occlusion else {
        return Ok(None);
    };
    let mut samples = [0u64; 1];
    let read = unsafe {
        ctx.device.get_query_pool_results(
            pool,
            0,
            &mut samples,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    };
    unsafe { ctx.device.destroy_query_pool(pool, None) };
    read.map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecGetQueryPoolResults, e)))?;
    Ok(Some(samples[0]))
}

/// The attachment views of a draw's own framebuffer, in the order the render
/// pass declares them: the primary colour slot, then the secondaries, then
/// depth.
///
/// Three places build a list in this order — the pass's attachment descriptions
/// in [`ObjectCaches::get_or_create_pass`], the clear vector in [`clear_values`],
/// and this. Vulkan indexes all three positionally against each other, so a
/// fourth spelling is a mismatch nothing refuses; there is one call site per
/// target arm and neither builds its own.
///
/// The secondaries are ensured here, after the caller has ensured the primary,
/// because that order is what protects this draw's own attachments: a resident
/// just ensured sits at the back of the LRU, so a later secondary's capacity
/// sweep (which evicts front-first) cannot take the primary or an earlier
/// secondary out from under the framebuffer being built.
unsafe fn ad_hoc_attachment_views(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
    primary_view: vk::ImageView,
    depth_view: Option<vk::ImageView>,
    mrt_secondaries: &mut Vec<(
        super::types::TargetIdentity,
        vk::Image,
        super::pools::ResidentAccess,
        Option<reims_vgpu_memory::GuestWritePages>,
    )>,
) -> Result<Vec<vk::ImageView>, DrawError> {
    let mut views = vec![primary_view];
    for sec in &req.secondary_targets {
        // The primary's guard, on the slot that never had one. A secondary whose
        // `load` is set becomes an `AttachmentLoadOp::LOAD` with
        // `initialLayout = COLOR_ATTACHMENT_OPTIMAL`, which preserves whatever
        // the image already holds — and a resident is born `content_ready =
        // false` over an image recycled from the target pool, whose texels are
        // some previous identity's. The pass would hand the draw those, and
        // `registry_mark_ready_at` would then publish the result as ready to
        // sample.
        //
        // Read before `registry_ensure_attachment` rather than after, because
        // ensuring is what creates the resident: asking afterwards cannot tell a
        // slot born in this call from one the guest has been rendering into.
        let wants_load = sec.load_action == super::types::ColorLoadAction::Load;
        let (img, view) = pools.registry_ensure_attachment(
            ctx,
            sec.identity.clone(),
            sec.width,
            sec.height,
            1,
            sec.identity.generation(),
            crate::format::vk_image_format(sec.format),
            sec.target_guest.as_ref(),
            wants_load,
            counters,
        )?;
        let slot = pools
            .registry_get(&sec.identity)
            .expect("the secondary resident was ensured on the line above");
        if wants_load && !slot.content_ready {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::LoadSecondaryContentNotReady {
                    identity: sec.identity.clone(),
                },
            ));
        }
        let old_access = slot.access;
        views.push(view);
        let guest = slot.memory.guest_write_pages().cloned();
        mrt_secondaries.push((sec.identity.clone(), img, old_access, guest));
    }
    views.extend(depth_view);
    Ok(views)
}

/// One `VkClearValue` per attachment, in framebuffer order: the primary colour
/// slot, then the secondaries, then depth.
///
/// Only attachments whose `loadOp` is `CLEAR` consult their entry, but the
/// vector must cover every attachment because Vulkan indexes it positionally —
/// a short vector silently gives a later attachment an earlier one's colour.
///
/// The primary's entry used to be a hard-coded transparent black, and that was
/// the whole reason `MTLLoadActionClear` could not be honoured directly: the
/// runtime met it instead by allocating a whole-attachment RGBA8 bitmap of the
/// requested colour and handing it over as a CPU seed, which also resolved the
/// pass to LOAD. The clear now travels as
/// [`super::types::DrawRequest::target_clear`], the same shape the secondaries
/// have always used.
fn clear_values(req: &DrawRequest) -> Vec<vk::ClearValue> {
    let mut clear = vec![vk::ClearValue {
        color: vk::ClearColorValue {
            float32: req.target_clear,
        },
    }];
    for sec in &req.secondary_targets {
        clear.push(vk::ClearValue {
            color: vk::ClearColorValue { float32: sec.clear },
        });
    }
    if let Some(d) = &req.depth {
        clear.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: d.clear_value,
                stencil: d.stencil.map(|s| s.clear_value).unwrap_or(0),
            },
        });
    }
    clear
}

/// Materialize one Metal attachment-wide `LoadActionClear` before a Vulkan
/// render pass whose framebuffer is smaller than this image.
///
/// Vulkan scopes a render-pass clear to `renderArea`; Metal scopes the load to
/// the whole attachment and only constrains subsequent rasterization. The two
/// transitions make the full-image transfer clear the content consumed by a
/// `LOAD` attachment in the smaller pass.
struct AttachmentWideColorClear {
    prior: super::pools::ResidentAccess,
    pass_layout: vk::ImageLayout,
    pass_stage: vk::PipelineStageFlags,
    pass_access: vk::AccessFlags,
    dependency: vk::DependencyFlags,
    clear: [f32; 4],
}

unsafe fn record_attachment_wide_clear(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    operation: AttachmentWideColorClear,
) {
    let AttachmentWideColorClear {
        prior,
        pass_layout,
        pass_stage,
        pass_access,
        dependency,
        clear,
    } = operation;
    let (src_stage, src_access) = prior.source_scope();
    let to_clear = [vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(prior.layout())
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(super::color_subresource_range())];
    device.cmd_pipeline_barrier(
        cb,
        src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_clear,
    );
    device.cmd_clear_color_image(
        cb,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearColorValue { float32: clear },
        &[super::color_subresource_range()],
    );
    let to_pass = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(pass_access)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(pass_layout)
        .image(image)
        .subresource_range(super::color_subresource_range())];
    device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        pass_stage,
        dependency,
        &[],
        &[],
        &to_pass,
    );
}

unsafe fn record_attachment_wide_depth_clear(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    attachment: &AcquiredDepth,
    clear_depth: f32,
    clear_stencil: Option<u32>,
) {
    let aspect = vk::ImageAspectFlags::DEPTH
        | if clear_stencil.is_some() {
            vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::empty()
        };
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let (src_stage, src_access) = attachment.access.source_scope();
    let to_clear = [vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(attachment.access.layout())
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(attachment.image)
        .subresource_range(range)];
    device.cmd_pipeline_barrier(
        cb,
        src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_clear,
    );
    device.cmd_clear_depth_stencil_image(
        cb,
        attachment.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearDepthStencilValue {
            depth: clear_depth,
            stencil: clear_stencil.unwrap_or(0),
        },
        &[range],
    );
    let to_pass = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .image(attachment.image)
        .subresource_range(range)];
    device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &to_pass,
    );
}

/// The access a barrier over *this draw's own colour target* must name as its
/// source, given what the registry last recorded and what this draw has already
/// done to it.
///
/// `snapshotted` is this draw's own copy-on-sample read of the target, recorded
/// into this same command buffer *after* the registry's access was read, so it
/// names the newer touch and wins. Everything else the target could be carrying
/// is already in [`super::pools::ResidentAccess`].
///
/// # Why the write sites need a source scope at all
///
/// None of the writes this feeds preserves content: a seed covers every texel,
/// and a CLEAR pass discards through `initialLayout = UNDEFINED`. Both keep the
/// discard, because paying a driver decompress to preserve pixels the very next
/// command overwrites buys nothing. What is not discardable is the *ordering* —
/// the write must not overtake whatever last touched this image, and nothing
/// else supplies that. The colour-only render pass declares no external subpass
/// dependency, and Vulkan's implicit one carries `srcStageMask = TOP_OF_PIPE`
/// with `srcAccessMask = 0`, which orders against nothing.
///
/// Batching does not close it either. A seeded draw never joins an open batch
/// (`joins` requires `LoadFromTarget` and no seed), so it opens its own command
/// buffer, and the flush of the previous batch only *submits* it. Queue
/// submission order starts command buffers in order; it does not finish them in
/// order. So frame N+1's write could land in an icon that frame N's window pass
/// was still sampling — one composite reading a half-replaced texture, a defect
/// no population counter can see, and one that grows more likely exactly as
/// queue occupancy rises under load.
///
/// # Why the same barrier shape is right elsewhere and wrong here
///
/// `upload_buffer_to_sampled_image`, the snapshot copy, and the compute storage
/// upload all open with `UNDEFINED`/`TOP_OF_PIPE` and no source scope, and all
/// three are correct. They write **pool-owned transient** images from
/// `acquire_sampled` / `acquire_storage_image`, and a slot only re-enters those
/// free lists through `drain_cleanup`, which `retire_slot` reaches only after
/// `wait_for_fences` on the submission that last used it. A pooled image
/// therefore cannot be handed out while any GPU work still reads it, so there is
/// nothing for a source scope to name.
///
/// The registry-resident target is the exception, and by design — see
/// [`super::pools::ResidentAccess`], which is where that argument now lives in a form the
/// compiler carries.
fn target_prior_access(
    snapshotted: bool,
    tracked: super::pools::ResidentAccess,
) -> super::pools::ResidentAccess {
    if snapshotted {
        super::pools::ResidentAccess::transfer_read()
    } else {
        tracked
    }
}

/// Whether a `LOAD` pass's own external subpass dependency already covers this
/// prior access, so the draw needs no explicit barrier into attachment use.
///
/// # The one case, and why it is the common one
///
/// `ColorWrite(COLOR_ATTACHMENT_OPTIMAL)` means the last thing to touch this
/// image was a render pass leaving it at
/// [`super::caches::color0_pass_exit_layout`], which is exactly the
/// `initialLayout` a `LOAD` pass names. So there is **no layout transition to
/// perform**, and the only remaining job a barrier could do is order the
/// previous pass's colour store against this pass's load — which
/// `super::caches::external_dependencies` already does, unconditionally, on
/// every pass this device builds: its incoming dependency runs
/// `VK_SUBPASS_EXTERNAL → 0` with `COLOR_ATTACHMENT_OUTPUT` /
/// `COLOR_ATTACHMENT_WRITE` in the source scope and the attachment stages and
/// accesses in the destination. `VK_SUBPASS_EXTERNAL` as `srcSubpass` scopes
/// every command submitted before the render pass instance in submission order,
/// so the previous draw's store is inside it.
///
/// Every other access keeps its barrier and must:
///
/// - `ShaderRead` needs the transition *and* a scope the pass dependency does
///   not name — `FRAGMENT_SHADER` is not in its source stages;
/// - `TransferRead` and a snapshot need the transition;
/// - `Untouched` does not reach here, because a `LOAD` pass with nothing to load
///   is not a `LOAD` pass;
/// - `ColorWrite` at any *other* layout is depth, which is not this attachment.
///
/// # What this is worth
///
/// It is the whole of `passmerge_outside_target_layout`, which was 82 % of draws
/// on macos-13 and 29-37 % on macos-11 and macos-12. Moving the pass exit to
/// `COLOR_ATTACHMENT_OPTIMAL` removed the *transition* those draws paid; this
/// removes the `vkCmdPipelineBarrier` that was left recording nothing. The
/// counter follows honestly — a draw that records no barrier is not charged an
/// obstacle, so `passmerge_reachable` can be non-zero for the first time.
fn pass_exit_needs_no_barrier(prior: super::pools::ResidentAccess) -> bool {
    prior == super::pools::ResidentAccess::ColorWrite(super::caches::color0_pass_exit_layout())
}

/// Record the barrier that makes a registry-resident image readable as a
/// transfer source, waiting for whatever last touched it.
///
/// The two questions a call site could get wrong are both answered here rather
/// than asked: it takes no condition, so the barrier cannot be skipped because
/// the image already sits in `TRANSFER_SRC_OPTIMAL`, and it takes an
/// [`super::pools::ResidentAccess`] rather than a layout, so the scope cannot be derived from
/// where the image sits. Both mistakes were live at the copy-on-sample site, and
/// both are invisible — a resident has no fence between consecutive users, so a
/// missing dependency only shows up as a copy that raced the draw producing the
/// pixels it copied.
///
/// A barrier whose old and new layouts match performs no transition and still
/// carries the dependency, which is the case this exists for.
///
/// # Nothing else orders these reads
///
/// The present blit is the clearest case and the reason this is shared with
/// `window_present` rather than written twice. The present records into its own
/// command buffer and submits it separately; queue submission order starts
/// command buffers in order but does not finish them in order, and is not a
/// memory dependency. A render pass's implicit final subpass dependency ends at
/// `dstStageMask = BOTTOM_OF_PIPE` with `dstAccessMask = 0`, so the colour
/// writes it produced are available and visible to nothing. The failure that
/// leaves is not wrong pixels but a stale frame: the blit copies the resident as
/// it stood before the draw, and the screen shows a composite missing what was
/// just rendered into it until some later redraw publishes it.
///
/// # What the repairs in this family did and did not fix
///
/// They were found while looking for the Finder icon defect, and they are not
/// it. Three 14-round `icon-composite.sh` boots, x86 / Vulkan: **3/14 corrupt
/// rounds before any of them, 4/14 after the first, 2/14 after all five.** No
/// effect at this n, and none claimed.
///
/// They stand on their own ground instead. Every one closed a read or write of a
/// registry-resident image that took no dependency on the work that last touched
/// it, which is undefined behaviour whatever it does to an icon, and the shared
/// shape of the mistake is worth remembering because it looked correct five
/// times in a row: **a barrier was skipped, or narrowed, whenever the image was
/// already in the layout the operation wanted.** A barrier is a layout
/// transition *and* a dependency, and for a resident — which by design outlives
/// the draw, with no fence between consecutive users — the layout is the half
/// that is usually already right and the dependency is the half that is always
/// needed.
///
/// The pooled census that scored those boots, and why no counter in it can
/// resolve this class, is recorded on
/// [`crate::telemetry::note_route`].
pub(super) unsafe fn barrier_resident_for_transfer_read(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    access: super::pools::ResidentAccess,
    next_access: super::pools::ResidentAccess,
) {
    let (src_stage, src_access) = access.source_scope();
    let barrier = [vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(access.layout())
        .new_layout(next_access.layout())
        .image(image)
        .subresource_range(super::color_subresource_range())];
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_width_only_changes_line_rasterized_geometry() {
        let zero = reims_vgpu_core::LineWidth::from_f32(0.0);
        let four = reims_vgpu_core::LineWidth::from_f32(4.0);
        assert_eq!(
            effective_line_raster_state(
                reims_vgpu_core::PrimitiveTopology::Triangle,
                reims_vgpu_core::FillMode::Fill,
                zero,
            ),
            (reims_vgpu_core::LineWidth::ONE.bits(), false)
        );
        assert_eq!(
            effective_line_raster_state(
                reims_vgpu_core::PrimitiveTopology::Line,
                reims_vgpu_core::FillMode::Fill,
                zero,
            ),
            (reims_vgpu_core::LineWidth::ONE.bits(), true)
        );
        assert_eq!(
            effective_line_raster_state(
                reims_vgpu_core::PrimitiveTopology::Triangle,
                reims_vgpu_core::FillMode::Lines,
                four,
            ),
            (four.bits(), false)
        );
    }

    #[test]
    fn nonpositive_and_nan_line_widths_discard_line_fragments() {
        for value in [0.99, 0.0, -1.0, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(
                effective_line_raster_state(
                    reims_vgpu_core::PrimitiveTopology::LineStrip,
                    reims_vgpu_core::FillMode::Fill,
                    reims_vgpu_core::LineWidth::from_f32(value),
                ),
                (reims_vgpu_core::LineWidth::ONE.bits(), true),
                "value={value}"
            );
        }
        let positive_infinity = reims_vgpu_core::LineWidth::from_f32(f32::INFINITY);
        assert_eq!(
            effective_line_raster_state(
                reims_vgpu_core::PrimitiveTopology::Line,
                reims_vgpu_core::FillMode::Fill,
                positive_infinity,
            ),
            (positive_infinity.bits(), false),
            "positive infinity must reach the typed host-range refusal"
        );
    }

    #[test]
    fn depth_bias_preserves_source_order_and_refuses_only_unrepresentable_state() {
        assert_eq!(effective_depth_bias(None, false), Ok(None));
        assert_eq!(effective_depth_bias(Some([0.0; 3]), false), Ok(None));
        assert_eq!(
            effective_depth_bias(Some([1.25, 2.5, 0.0]), false),
            Ok(Some([1.25, 2.5, 0.0]))
        );
        assert!(matches!(
            effective_depth_bias(Some([1.0, 2.0, 3.0]), false),
            Err(crate::engine::reason::DrawReason::DepthBiasClampUnsupported { .. })
        ));
        assert!(matches!(
            effective_depth_bias(Some([f32::NAN, 0.0, 0.0]), true),
            Err(crate::engine::reason::DrawReason::DepthBiasNonFinite { component: 0, .. })
        ));
    }

    #[test]
    fn a_secondary_attachment_alone_can_require_encoder_blend_constants() {
        let constant_blend = reims_vgpu_core::BlendStateResource {
            src_color: reims_vgpu_core::BlendFactor::ConstantColor,
            dst_color: reims_vgpu_core::BlendFactor::Zero,
            color_op: reims_vgpu_core::BlendOp::Add,
            src_alpha: reims_vgpu_core::BlendFactor::ConstantAlpha,
            dst_alpha: reims_vgpu_core::BlendFactor::Zero,
            alpha_op: reims_vgpu_core::BlendOp::Add,
        };
        let req = DrawRequest {
            blend: None,
            secondary_targets: vec![reims_vgpu_core::SecondaryColorTarget {
                identity: reims_vgpu_core::TargetIdentity::Gva {
                    gva: 1,
                    width: 1,
                    height: 1,
                    generation: 0,
                    format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
                },
                target_guest: None,
                width: 1,
                height: 1,
                format: reims_vgpu_protocol::ImageFormat::linear(
                    reims_vgpu_protocol::TexelLayout::Rgba8,
                ),
                clear: [0.0; 4],
                load_action: reims_vgpu_core::ColorLoadAction::Clear,
                blend: Some(constant_blend),
                color_write_mask: Default::default(),
            }],
            ..Default::default()
        };
        assert!(draw_uses_blend_constants(&req));
    }

    #[test]
    fn a_non_indexed_zero_vertex_count_has_no_invocations() {
        let req = DrawRequest {
            vertex_count: 0,
            instance_count: Some(1),
            ..DrawRequest::default()
        };
        assert!(draw_has_no_invocations(&req));
    }

    #[test]
    fn an_indexed_draw_is_governed_by_its_index_count() {
        let req = DrawRequest {
            vertex_count: 0,
            instance_count: Some(1),
            indexed: Some(super::super::types::IndexedDrawResource {
                index_type: super::super::types::IndexType::U16,
                index_count: 3,
                vertex_offset: 0,
                content: BufferContent::Bytes(std::sync::Arc::new(Vec::new())),
            }),
            ..DrawRequest::default()
        };
        assert!(!draw_has_no_invocations(&req));
    }

    fn sampled_identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width: 64,
            height: 32,
            generation: 3,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }
    }

    /// A serialized target reference is admitted without a preparation-side
    /// registry query; the engine transaction remains responsible for every
    /// mutable resident condition.
    #[test]
    fn sampled_resident_state_is_validated_at_execution() {
        let identity = sampled_identity();
        let expected = |initialized_by_this_pass| SampledResidentExpectation {
            binding: 34,
            identity: &identity,
            resource_width: 64,
            resource_height: 32,
            shader_multisampled: false,
            initialized_by_this_pass,
        };
        assert!(matches!(
            validate_sampled_resident(expected(false), None, None),
            Err(DrawExecutionDecline::SampledResidentMissing {
                binding: 34,
                prior: None,
                ..
            })
        ));
        assert!(matches!(
            validate_sampled_resident(expected(false), Some((false, 64, 32, 1)), None,),
            Err(DrawExecutionDecline::SampledResidentNotReady { binding: 34, .. })
        ));
        assert!(matches!(
            validate_sampled_resident(expected(false), Some((true, 63, 32, 1)), None,),
            Err(DrawExecutionDecline::SampledResidentGeometryMismatch {
                binding: 34,
                resident_width: 63,
                resource_width: 64,
                ..
            })
        ));
        assert_eq!(
            validate_sampled_resident(expected(false), Some((true, 64, 32, 1)), None,),
            Ok(())
        );
        assert_eq!(
            validate_sampled_resident(expected(true), Some((false, 64, 32, 1)), None,),
            Ok(()),
            "the attachment load establishes content before the fragment reads it"
        );
    }

    #[test]
    fn an_open_render_pass_continues_only_within_the_same_encoder() {
        assert!(continues_open_render_pass(true, true));
        assert!(!continues_open_render_pass(false, true));
        assert!(!continues_open_render_pass(true, false));
        assert!(!continues_open_render_pass(false, false));
    }
    use crate::engine::pools::ResidentAccess;
    use crate::engine::types::{GuestRun, GuestRunSource, SampledImageResource, SampledSource};
    use reims_vgpu_observe::Decline;

    fn sig(binding: u32) -> BindingSig {
        BindingSig {
            binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
            count: 1,
        }
    }

    /// A binding a module statically uses and the layout omits is caught before
    /// the pipeline is created, and is attributed to the stage that named it.
    ///
    /// This is a host-kill guard, not a correctness nicety: on Mesa's Intel
    /// driver the omission is `(use_count << 7) / 0` inside
    /// `vkCreateGraphicsPipelines`, which takes the QEMU process down with
    /// `SIGFPE` — no Vulkan error, no guest packet to fail, no log line. The
    /// compute path has refused this since `25051457`; this arm did not, and the
    /// two arms consume the same wire form.
    #[test]
    fn a_used_binding_the_layout_omits_is_refused_before_the_pipeline_is_built() {
        let empty: [u32; 0] = [];

        assert_eq!(
            used_binding_absent_from_layout(&empty, &[33], &[sig(32)]),
            Some((33, true)),
            "binding 33 is used by the fragment module and the layout names only 32, \
             which is exactly the hole Mesa divides by"
        );

        assert_eq!(
            used_binding_absent_from_layout(&[33], &empty, &[sig(32)]),
            Some((33, false)),
            "the vertex module can name the hole just as well, and the layout is shared"
        );
    }

    /// A binding the layout provides is not a hole however the module uses it.
    /// Declared-but-unused variables are excluded while constructing the
    /// retained set and are covered at that ownership boundary.
    #[test]
    fn a_provided_binding_and_empty_used_sets_are_both_left_alone() {
        let empty: [u32; 0] = [];
        assert_eq!(
            used_binding_absent_from_layout(&empty, &[33], &[sig(32), sig(33)]),
            None,
            "the layout names 33, so there is no hole"
        );
        assert_eq!(
            used_binding_absent_from_layout(&empty, &empty, &[]),
            None,
            "a draw with no modules and no layout has nothing to refuse"
        );
    }

    /// Build a `JoinTerms` from a bitmask, one bit per ladder rung in order.
    fn join_terms(bits: u32) -> JoinTerms {
        let b = |i: usize| bits & (1 << i) != 0;
        JoinTerms {
            force_loss: b(0),
            quirk: b(1),
            is_mrt: b(2),
            depth_barred: b(3),
            reads_back: b(4),
            has_query: b(5),
            no_identity: b(6),
            no_open_batch: b(7),
            batch_full: b(8),
            target_switch: b(9),
        }
    }

    /// Every rung must be reachable as a verdict, or the census name below it
    /// is unattributable and the one above it is over-counted.
    ///
    /// The ladder returns its *first* refusing term, so a rung is reachable
    /// only from the input that refuses at it and nowhere earlier — which is
    /// exactly the single-bit mask for that rung.
    #[test]
    fn every_join_refusal_is_reachable_and_named_once() {
        let mut seen = std::collections::HashSet::new();
        for (rung, (_, _, name)) in JoinTerms::LADDER.iter().enumerate() {
            assert_eq!(
                join_terms(1 << rung).refusal(),
                Some(*name),
                "rung {rung} does not answer with its own name when it is the only term set"
            );
            assert!(seen.insert(*name), "two rungs share the census name {name}");
        }
        assert_eq!(join_terms(0).refusal(), None, "no term set is a join");
    }

    /// A depth attachment on its own must not stop a draw deferring its submit.
    ///
    /// This pins the decision and not the disposal, which is the honest scope: no
    /// test here can reach `dispose_ad_hoc_attachments`, so what it guards is the
    /// rung, and the safety argument for the rung being relaxed lives in that
    /// function's doc and in `open_slot_mask`'s.
    ///
    /// Worth a test rather than a reading of the ladder because the rung was
    /// unconditional for as long as no measured workload presented depth. A
    /// driven macos-13 Maps boot puts `nojoin_depth` at 49 014 in 45 s against 0
    /// for the sustained-animation probe, and `passheld_no_join` at 50 368 —
    /// i.e. depth was very nearly the *only* reason that workload's passes did
    /// not merge. Re-adding `req.depth.is_some()` to a `JoinScope::Draw` rung
    /// unconditionally puts that back and fails here.
    #[test]
    fn a_depth_attachment_alone_does_not_bar_batching() {
        // Through the production predicate, not the field, or this stays green
        // if the rung goes back to barring every depth draw. `BATCH_DEPTH` is
        // unset in the test process, which is the shipping arm.
        let depth_only = JoinTerms {
            depth_barred: depth_bars_batching(true),
            ..join_terms(0)
        };
        assert!(
            depth_only.batch_eligible(),
            "a draw carrying depth must be able to open or join a batch"
        );
        assert_eq!(depth_only.refusal(), None);

        // And the env switch still restores the old bar, under its own name, so
        // the two arms of an A/B remain distinguishable in the census.
        let barred = JoinTerms {
            depth_barred: true,
            ..join_terms(0)
        };
        assert!(!barred.batch_eligible());
        assert_eq!(barred.refusal(), Some("nojoin_depth"));
    }

    // There is deliberately no test that `batch_eligible` and `refusal` agree
    // about which rungs gate opening a batch. Both read `JoinScope` off the same
    // ladder entry, so any such test asserts the scope field against itself:
    // mis-scoping `nojoin_query` to `Fit` — which would let a queried draw defer
    // its submit and hand the guest a count its command buffer has not produced
    // — leaves one green. That test was written, run against exactly that
    // mutation, and deleted. The `debug_assert` it replaced had the same blind
    // spot and a release-build hole besides. What makes the two agree is that
    // there is one list.

    #[test]
    fn an_imported_guest_read_publishes_every_guest_memory_writer() {
        for access in [vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ] {
            let barrier = imported_guest_read_barrier(access, ImportedGuestVisibility::GpuOverlap);
            assert!(barrier
                .src_access_mask
                .contains(vk::AccessFlags::HOST_WRITE));
            assert!(barrier
                .src_access_mask
                .contains(vk::AccessFlags::TRANSFER_WRITE));
            assert!(barrier
                .src_access_mask
                .contains(vk::AccessFlags::SHADER_WRITE));
            assert!(barrier
                .src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE));
            assert!(barrier
                .src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE));
            assert_eq!(barrier.dst_access_mask, access);
        }
        let stages = imported_guest_write_stage();
        assert!(stages.contains(vk::PipelineStageFlags::HOST));
        assert!(stages.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(stages.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(stages.contains(vk::PipelineStageFlags::ALL_GRAPHICS));
    }

    #[test]
    fn an_imported_load_orders_both_halves_of_attachment_access() {
        let access =
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        let barrier = imported_guest_read_barrier(access, ImportedGuestVisibility::GpuOverlap);
        assert!(
            barrier
                .src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            "an earlier Vulkan alias may have produced the imported bytes"
        );
        assert_eq!(barrier.dst_access_mask, access);
    }

    #[test]
    fn an_imported_attachment_store_is_released_to_host_readers() {
        let barrier = imported_guest_attachment_release_barrier();
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        );
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags::HOST_READ);
    }

    #[test]
    fn a_disjoint_imported_read_waits_only_for_guest_cpu_writes() {
        let access = vk::AccessFlags::SHADER_READ;
        let barrier = imported_guest_read_barrier(access, ImportedGuestVisibility::HostOnly);
        assert_eq!(barrier.src_access_mask, vk::AccessFlags::HOST_WRITE);
        assert_eq!(barrier.dst_access_mask, access);
        assert_eq!(
            imported_guest_read_stage(ImportedGuestVisibility::HostOnly),
            vk::PipelineStageFlags::HOST
        );
    }

    #[test]
    fn an_open_pass_does_not_republish_its_own_attachment_as_an_alias_read() {
        assert!(imported_target_needs_visibility(true, false));
        assert!(
            !imported_target_needs_visibility(true, true),
            "continuation uses the same attachment object and remains ordered in the pass"
        );
        assert!(!imported_target_needs_visibility(false, false));
    }

    /// `bufferOffset` has a hard rule in the Vulkan spec, and this rail is the
    /// only thing that ever gives it a nonzero value: every other sampled upload
    /// starts at the head of a staging span this device allocated. A window
    /// whose first texel sits at an offset the copy cannot name has to reach the
    /// CPU gather, because the alternative is `VUID-vkCmdCopyBufferToImage`
    /// undefined behaviour on a value the guest chose.
    #[test]
    fn the_copy_offset_bound_covers_every_texel_block_the_sampled_pool_can_hold() {
        // A multiple of 16 is a multiple of 4, which is the other half of the
        // rule for every non-depth/stencil format, and 16 is the largest block
        // the pool can produce: it covers R32G32B32A32 as well as
        // BC2/BC3/BC5/BC7.
        assert_eq!(GUEST_IMPORT_COPY_OFFSET_ALIGN % 4, 0);
        assert_eq!(
            GUEST_IMPORT_COPY_OFFSET_ALIGN.max(16),
            GUEST_IMPORT_COPY_OFFSET_ALIGN
        );

        // The check the constant exists for: an offset it accepts satisfies
        // both halves of the rule, and one it rejects would not have.
        for offset in [0u64, 16, 32, 4096] {
            assert!(offset.is_multiple_of(GUEST_IMPORT_COPY_OFFSET_ALIGN));
            assert!(offset.is_multiple_of(4));
        }
        for offset in [4u64, 8, 12, 100] {
            assert!(!offset.is_multiple_of(GUEST_IMPORT_COPY_OFFSET_ALIGN));
        }
    }

    /// One import over a plausible RAMBlock, for building references the
    /// planners can be asked about. The host address is compared and never
    /// dereferenced, which is what lets a unit test hold one.
    fn window_runs(stretches: &[(u64, u64, u64)]) -> Vec<reims_vgpu_memory::GuestWindowRun> {
        use reims_vgpu_memory::{GuestRamImport, GuestRamRegion, GuestRef};
        let import = std::sync::Arc::new(
            GuestRamImport::new(
                GuestRamRegion {
                    gpa_base: 0x1_0000_0000,
                    host_va: 0x7f00_0000_0000,
                    len: 0x10_0000,
                },
                1,
            )
            .expect("region is aligned and non-empty"),
        );
        stretches
            .iter()
            .map(
                |&(window_offset, offset, len)| reims_vgpu_memory::GuestWindowRun {
                    window_offset,
                    guest: GuestRef::new(
                        std::sync::Arc::clone(&import),
                        import.slice(offset, len).expect("inside the import"),
                    )
                    .expect("the slice came from this import"),
                },
            )
            .collect()
    }

    /// A source whose window is `source_offset..source_offset + total_len` inside
    /// `pages`. `runs` is the CPU gather's view of the same bytes and is not
    /// consulted by anything under test here.
    fn source_over(
        pages: Vec<reims_vgpu_memory::GuestWindowRun>,
        source_offset: u64,
        total_len: u64,
    ) -> super::super::types::GuestRunSource {
        super::super::types::GuestRunSource {
            runs: std::sync::Arc::new(Vec::new()),
            source_offset,
            total_len,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            physical_pages: None,
        }
    }

    /// A one-stretch window starting at window byte zero is the whole window, so
    /// it binds in place. Anything else has to be assembled.
    ///
    /// The offset half is the one that can be silently wrong: a lone run that
    /// does *not* start at zero names a suffix of the window, and binding it
    /// would hand the draw the guest's bytes shifted forward — a wrong draw
    /// rather than a failed one. `references_for_runs` cannot produce that
    /// shape, and this is what keeps the reliance on that written down.
    #[test]
    fn only_a_lone_stretch_covering_byte_zero_binds_in_place() {
        assert!(source_over(window_runs(&[(0, 0, 64)]), 0, 64)
            .single_stretch()
            .is_some());
        assert!(
            source_over(window_runs(&[(16, 16, 48)]), 0, 48)
                .single_stretch()
                .is_none(),
            "a lone run starting past window byte zero is a suffix, not a window"
        );
        assert!(
            source_over(window_runs(&[(0, 0, 32), (32, 4096, 32)]), 0, 64)
                .single_stretch()
                .is_none(),
            "two stretches are two ranges and a bind names one"
        );
        assert!(source_over(window_runs(&[]), 0, 0)
            .single_stretch()
            .is_none());
    }

    /// A window inside a lone stretch binds at the window's first byte, not at
    /// the stretch's.
    ///
    /// This is the shape every mapped sampled plane has:
    /// `runtime::draw::vulkan::mapped_sampled_source` names the whole allocation
    /// as one stretch and carries the plane's own offset in `source_offset`. The
    /// sampled rail used to bind the stretch and drop that offset, so on any host
    /// that imports guest RAM but cannot alias it as a linear image — every
    /// discrete GPU — each texture was read `source_offset` bytes early and came
    /// out shifted along its rows. The buffer rail always applied the term, which
    /// is why one wire form had two answers.
    #[test]
    fn a_window_inside_a_lone_stretch_skips_to_its_own_first_byte() {
        let src = source_over(window_runs(&[(0, 0, 0x8000)]), 0x100, 0x4000);
        let stretch = src.single_stretch().expect("one stretch holds the window");
        assert_eq!(
            stretch.skip, 0x100,
            "the plane offset inside the allocation"
        );
        assert_eq!(stretch.window_offset, 0);
        assert_eq!(stretch.len, 0x4000, "the window, not the whole stretch");
    }

    /// A window that does not fit the stretch it names is malformed, and the
    /// bind is refused rather than reading past the allocation.
    #[test]
    fn a_window_past_the_end_of_its_lone_stretch_does_not_bind() {
        assert!(source_over(window_runs(&[(0, 0, 0x1000)]), 0xf00, 0x200)
            .single_stretch()
            .is_none());
    }

    /// Two offsets into one retained resource are distinct command-buffer
    /// binds, while still sharing the allocation identity that keeps the
    /// source alive. This is the engine half of buffer-plus-offset semantics.
    #[test]
    fn one_run_source_at_two_offsets_has_two_bind_keys() {
        let runs = std::sync::Arc::new(vec![GuestRun {
            host_ptr: 0x1000,
            len: 0x4000,
        }]);
        let content = |source_offset| {
            BufferContent::GuestRuns(GuestRunSource {
                runs: std::sync::Arc::clone(&runs),
                source_offset,
                total_len: 0x1000,
                row_length_texels: 0,
                pages: None,
                physical_pages: None,
            })
        };
        let a = CbBind::of(&content(0));
        let b = CbBind::of(&content(0x1000));
        assert_ne!(a.key(), b.key(), "the bind offset is part of the identity");
        assert_eq!(a.key().0, b.key().0, "both keys retain one run allocation");
    }

    /// A fallback copy from a whole-resource import must crop and rebase the
    /// requested subrange rather than copying from byte zero.
    #[test]
    fn gather_region_crops_a_retained_resource_to_the_bind_offset() {
        let src = source_over(window_runs(&[(0, 0x2000, 0x4000)]), 0x1000, 0x800);
        let stretch = src
            .window_stretches()
            .expect("the source names its pages")
            .next()
            .expect("the window intersects the one stretch");
        let bound = crate::engine::host_ram::BoundGuestRam {
            buffer: ash::vk::Buffer::null(),
            offset: 0x8000,
            len: 0x4000,
            head: 0x20,
        };
        let copy = gather_region(&bound, &stretch);
        assert_eq!(copy.src_offset, 0x9020);
        assert_eq!(copy.dst_offset, 0);
        assert_eq!(copy.size, 0x800);
    }

    /// The stretches of a window tile it exactly and nothing else: the lengths
    /// sum to `total_len`, the first lands at window byte zero, and stretches the
    /// window does not reach contribute nothing.
    ///
    /// The gather rails check `covered == total_len` before they take a
    /// destination slot, so a clipping mistake here is a decline rather than a
    /// wrong image — but only because the two agree about what the window is.
    #[test]
    fn window_stretches_tile_the_window_and_skip_what_it_does_not_reach() {
        let src = source_over(
            window_runs(&[
                (0, 0, 0x1000),
                (0x1000, 0x2000, 0x1000),
                (0x2000, 0x4000, 0x1000),
            ]),
            0x800,
            0x1000,
        );
        let got: Vec<_> = src
            .window_stretches()
            .expect("the source names its pages")
            .map(|s| (s.skip, s.window_offset, s.len))
            .collect();
        assert_eq!(
            got,
            vec![(0x800, 0, 0x800), (0, 0x800, 0x800)],
            "the window starts mid-stretch, spans into the next, and never \
             reaches the third"
        );
        assert_eq!(
            got.iter().map(|s| s.2).sum::<u64>(),
            src.total_len,
            "the stretches tile the window exactly"
        );
    }

    /// The role split is over physical gathers, not logical bindings. One
    /// interleaved allocation used by fixed-function fetch and by a shader is
    /// therefore one `Shared` window rather than one vertex plus one storage
    /// window, and the byte columns remain a partition of the actual traffic.
    #[test]
    fn buffer_gather_roles_partition_physical_content_allocations() {
        use super::super::types::{
            GuestRun, GuestRunSource, IndexType, IndexedDrawResource, StorageBufferResource,
            VertexAttributeFormat, VertexAttributeResource,
        };

        let content = |host_ptr| {
            BufferContent::GuestRuns(GuestRunSource {
                runs: std::sync::Arc::new(vec![GuestRun { host_ptr, len: 16 }]),
                source_offset: 0,
                total_len: 16,
                row_length_texels: 0,
                pages: None,
                physical_pages: None,
            })
        };
        let vertex_only = content(0x1000);
        let storage_only = content(0x2000);
        let shared = content(0x3000);
        let index_only = content(0x4000);
        let keys = [
            CbBind::of(&vertex_only).key(),
            CbBind::of(&storage_only).key(),
            CbBind::of(&shared).key(),
            CbBind::of(&index_only).key(),
        ];
        let req = DrawRequest {
            vertex_attributes: vec![
                VertexAttributeResource {
                    location: 0,
                    binding: 0,
                    format: VertexAttributeFormat::Float,
                    offset: 0,
                    stride: 4,
                    step_function: VertexStepFunction::PerVertex,
                    step_rate: 1,
                    content: vertex_only,
                },
                VertexAttributeResource {
                    location: 1,
                    binding: 1,
                    format: VertexAttributeFormat::Float,
                    offset: 0,
                    stride: 4,
                    step_function: VertexStepFunction::PerVertex,
                    step_rate: 1,
                    content: shared.clone(),
                },
            ],
            storage_buffers: vec![
                StorageBufferResource {
                    binding: 2,
                    content: storage_only,
                },
                StorageBufferResource {
                    binding: 3,
                    content: shared,
                },
            ],
            indexed: Some(IndexedDrawResource {
                index_type: IndexType::U16,
                index_count: 8,
                vertex_offset: 0,
                content: index_only,
            }),
            ..DrawRequest::default()
        };

        let roles = BufferGatherRoles::of(&req);
        assert_eq!(roles.len(), 4, "shared content must stay one operation");
        assert_eq!(roles.role(keys[0]), Some(BufferGatherRole::VERTEX));
        assert_eq!(roles.role(keys[1]), Some(BufferGatherRole::STORAGE));
        assert!(roles
            .role(keys[2])
            .expect("shared was classified")
            .is_shared());
        let index = roles
            .role(keys[3])
            .expect("the index buffer was classified");
        assert!(index.includes_index());
        assert_eq!(index.index_alignment, Some(2));
        // The table answers only about this draw's binds. An unclassified key
        // must be `None` and not the role of whichever entry a scan happened to
        // stop on — the three call sites `expect()` on exactly this.
        assert_eq!(roles.role((0xdead_beef, 0, 16)), None);
    }

    /// The bands tile every attachment size with no gap and no overlap, and a
    /// 1080p target lands in the top one.
    ///
    /// A gap here would be silent: a pass begin charged to no band simply does
    /// not appear, and the regression the bands exist for would then attribute
    /// its cost to whichever band happened to move with it. The boundary values
    /// are therefore walked rather than sampled in the middle of each band.
    #[test]
    fn the_pass_area_bands_tile_every_attachment_size() {
        // One pixel either side of each boundary, plus the degenerate and the
        // real target size.
        let cases: [(u32, u32, &str); 9] = [
            (0, 0, "passbegin_px_lt64k"),
            (256, 255, "passbegin_px_lt64k"),
            (256, 256, "passbegin_px_lt256k"),
            (512, 511, "passbegin_px_lt256k"),
            (512, 512, "passbegin_px_lt1m"),
            (1024, 1023, "passbegin_px_lt1m"),
            (1024, 1024, "passbegin_px_ge1m"),
            (1920, 1080, "passbegin_px_ge1m"),
            (u32::MAX, u32::MAX, "passbegin_px_ge1m"),
        ];
        for (w, h, want) in cases {
            assert_eq!(
                pass_begin_area_band(w, h),
                want,
                "{w}x{h} landed in the wrong band"
            );
        }
    }

    #[test]
    fn buffer_roles_preserve_every_consumers_alignment() {
        assert!(!BufferGatherRole::VERTEX.includes_index());
        assert!(BufferGatherRole::STORAGE.is_storage_only());
        let index = BufferGatherRole {
            vertex: false,
            storage: false,
            index_alignment: Some(2),
        };
        assert!(index.includes_index());
        let shared = BufferGatherRole {
            vertex: true,
            storage: true,
            index_alignment: Some(4),
        };
        assert!(shared.is_shared());
        assert!(!shared.is_storage_only());
    }

    /// A direct bind applies only the alignment required by its actual Vulkan
    /// consumers. Vertex buffers have no extra offset alignment; storage and
    /// index consumers keep their independently queried/decoded requirements.
    #[test]
    fn direct_import_alignment_follows_the_actual_consumer() {
        assert_eq!(buffer_bind_offset_alignment(BufferGatherRole::VERTEX, 4), 1);
        assert_eq!(
            buffer_bind_offset_alignment(BufferGatherRole::STORAGE, 4),
            4
        );
        let index = BufferGatherRole {
            vertex: false,
            storage: false,
            index_alignment: Some(2),
        };
        assert_eq!(buffer_bind_offset_alignment(index, 4), 2);
        let shared = BufferGatherRole {
            vertex: true,
            storage: true,
            index_alignment: Some(2),
        };
        assert_eq!(buffer_bind_offset_alignment(shared, 4), 4);
    }

    /// The two re-basings a gather region does, at the values that make them
    /// visible.
    ///
    /// Both failure modes here are silent: reading from `offset` instead of
    /// `offset + head` shifts the whole window forward by up to a granule, and
    /// copying `bound_len` instead of `requested` reads guest bytes the window
    /// never named and writes them past the destination's end. Neither produces
    /// an error — they produce wrong vertices. So the stretch is deliberately
    /// misaligned on both sides: a 24-byte request inside a 4096-granular
    /// import, which rounds out to a `head` of 24 and a bound length of 4096.
    #[test]
    fn a_gather_region_reads_the_requested_bytes_and_not_the_rounding() {
        use reims_vgpu_memory::{GuestRamImport, GuestRamRegion, GuestRef};
        let import = std::sync::Arc::new(
            GuestRamImport::new(
                GuestRamRegion {
                    gpa_base: 0x1_0000_0000,
                    host_va: 0x7f00_0000_0000,
                    len: 0x10_0000,
                },
                4096,
            )
            .expect("region is aligned and non-empty"),
        );
        let slice = import.slice(4096 + 24, 100).expect("inside the import");
        assert_eq!(slice.head(), 24, "the granularity rounding went in front");
        assert_eq!(slice.requested(), 100);
        assert_eq!(slice.bound_len(), 4096, "and out to the granule at the end");

        let src = source_over(
            vec![reims_vgpu_memory::GuestWindowRun {
                window_offset: 512,
                guest: GuestRef::new(std::sync::Arc::clone(&import), slice)
                    .expect("the slice came from this import"),
            }],
            0,
            512 + 100,
        );
        let stretch = src
            .window_stretches()
            .expect("the source names its pages")
            .next()
            .expect("the window reaches this stretch");
        let bound = super::super::host_ram::BoundGuestRam {
            buffer: {
                use ash::vk::Handle;
                vk::Buffer::from_raw(0x99)
            },
            offset: 4096,
            len: 4096,
            head: 24,
        };
        let copy = gather_region(&bound, &stretch);
        assert_eq!(
            copy.src_offset, 4120,
            "the copy must start at the byte the guest named, not at the granule \
             the bound range was rounded back to"
        );
        assert_eq!(
            copy.size, 100,
            "the copy must move the bytes the window asked for, not the granule \
             the bound range was rounded out to"
        );
        assert_eq!(
            copy.dst_offset, 512,
            "the stretch lands where it sits in the window"
        );
    }

    /// A pooled staging slot binds at zero and a guest window import does not,
    /// and every bind site has to take the offset from the same place.
    ///
    /// The conversion exists so that `From<BufferSlot>` is the *only* way a slot
    /// becomes bindable — a site that reached past it and used `slot.buffer`
    /// with a literal `0` would be right for staging and silently wrong for an
    /// import, binding the head of the guest's page instead of the span the
    /// guest named. That is a wrong draw, not a failed one.
    #[test]
    fn a_pooled_slot_binds_at_zero_and_carries_its_own_buffer() {
        let slot = BufferSlot {
            buffer: {
                use ash::vk::Handle;
                vk::Buffer::from_raw(0x1234)
            },
            memory: vk::DeviceMemory::null(),
            size: 4096,
            mapped: 0,
            coherent: true,
            cached: false,
            backing: super::super::pools::BufferBacking::Dedicated,
        };
        let bound = BoundBuffer::from(slot);
        assert_eq!(bound.buffer, slot.buffer);
        assert_eq!(bound.offset, 0);
    }

    /// Each reason a guest span fails to become a directly bound buffer is its
    /// own line in the log.
    ///
    /// The alignment refusal in particular is the one that decides whether this
    /// rail runs at all on a given host: it is a device limit against a guest
    /// allocator's choices, and neither side is under this device's control. A
    /// boot where the rail moves nothing has to be able to say whether that is
    /// because the host cannot import, or because every span this guest hands
    /// over lands mid-alignment.
    ///
    /// The short-window refusals that used to sit beside this one are gone,
    /// unrepresentable rather than untested: the range comes from a
    /// `GuestRef`, whose bound is checked where it is built and cannot be
    /// skipped, so there is no longer a way for a bind to name bytes past the
    /// end of what was resolved.
    #[test]
    fn every_buffer_import_decline_names_its_own_check() {
        let declines = [BufferImportDecline::BindOffsetAlignment {
            offset: 40,
            align: 64,
        }];
        let slugs: Vec<_> = declines.iter().map(|d| d.slug()).collect();
        let unique: std::collections::HashSet<_> = slugs.iter().collect();
        assert_eq!(slugs.len(), unique.len(), "slugs collide: {slugs:?}");
        for decline in &declines {
            assert!(!decline.fields().is_empty(), "{decline} carries no values");
            assert!(decline.slug().starts_with("buffer_import_"));
        }
        // The two rails refuse for different reasons and must not share a slug
        // with each other either — a sampled bind and a buffer bind can fail the
        // same way and still need different fixes.
        assert_ne!(
            BufferImportDecline::BindOffsetAlignment {
                offset: 1,
                align: 2
            }
            .slug(),
            SampledImportDecline::CopyOffsetAlignment { offset: 1 }.slug()
        );
    }

    /// Each reason a guest window fails to become a copy source is its own line
    /// in the log.
    ///
    /// They route to the same place — the CPU gather — so nothing downstream
    /// tells them apart, and a shared slug would leave "this host never imports
    /// a sampled window" with no way to say whether the device cannot import at
    /// all or every window this guest hands over lands on an odd offset. Those
    /// are different fixes.
    #[test]
    fn every_sampled_import_decline_names_its_own_check() {
        let declines = [SampledImportDecline::CopyOffsetAlignment { offset: 12 }];
        let slugs: Vec<_> = declines.iter().map(|d| d.slug()).collect();
        let unique: std::collections::HashSet<_> = slugs.iter().collect();
        assert_eq!(slugs.len(), unique.len(), "slugs collide: {slugs:?}");
        for decline in &declines {
            assert!(!decline.fields().is_empty(), "{decline} carries no values");
            assert!(decline.slug().starts_with("sampled_import_"));
        }
    }

    fn validation_slug(req: &DrawRequest) -> &'static str {
        match validate_v1(req) {
            Err(DrawError::DrawValidation(decline)) => decline.slug(),
            Err(other) => panic!("expected typed draw validation, got {other}"),
            Ok(()) => panic!("expected draw validation failure"),
        }
    }

    fn test_program() -> reims_vgpu_core::PreparedRenderProgram {
        reims_vgpu_core::PreparedRenderProgram {
            vertex: reims_vgpu_core::PreparedShaderStage {
                id: reims_vgpu_protocol::PreparedShaderId::new(1),
                ..Default::default()
            },
            fragment: reims_vgpu_core::PreparedShaderStage {
                id: reims_vgpu_protocol::PreparedShaderId::new(2),
                ..Default::default()
            },
        }
    }

    #[test]
    fn pass_raster_extent_must_fit_the_attachment() {
        let req = DrawRequest {
            program: test_program(),
            width: 8,
            height: 8,
            render_target_extent: reims_vgpu_core::RenderTargetExtent {
                width: std::num::NonZeroU32::new(9),
                height: None,
            },
            ..Default::default()
        };
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_render_target_extent_exceeds_attachment"
        );
    }

    #[test]
    fn zero_geometry_on_any_attachment_is_rejected_before_vulkan() {
        let mut req = DrawRequest {
            program: test_program(),
            width: 8,
            height: 8,
            ..Default::default()
        };
        req.secondary_targets
            .push(reims_vgpu_core::SecondaryColorTarget {
                identity: reims_vgpu_core::TargetIdentity::Texture {
                    ref_: 1,
                    width: 0,
                    height: 8,
                    generation: 1,
                    stencil: false,
                },
                target_guest: None,
                width: 0,
                height: 8,
                format: reims_vgpu_protocol::ImageFormat::linear(
                    reims_vgpu_protocol::TexelLayout::Rgba8,
                ),
                clear: [0.0; 4],
                load_action: reims_vgpu_core::ColorLoadAction::Clear,
                blend: None,
                color_write_mask: Default::default(),
            });
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_zero_target_geometry"
        );
    }

    #[test]
    fn pass_extent_defines_the_implicit_viewport_and_scissor() {
        let req = DrawRequest {
            width: 8,
            height: 8,
            render_target_extent: reims_vgpu_core::RenderTargetExtent {
                width: std::num::NonZeroU32::new(4),
                height: std::num::NonZeroU32::new(3),
            },
            ..Default::default()
        };
        let mut viewports = Vec::new();
        let mut scissors = Vec::new();
        populate_dynamic_viewport_scissors(&req, &mut viewports, &mut scissors);
        assert_eq!(viewports.len(), 1);
        assert_eq!(viewports[0].x, 0.0);
        assert_eq!(viewports[0].y, 3.0);
        assert_eq!(viewports[0].width, 4.0);
        assert_eq!(viewports[0].height, -3.0);
        assert_eq!(scissors[0].offset, vk::Offset2D { x: 0, y: 0 });
        assert_eq!(
            scissors[0].extent,
            vk::Extent2D {
                width: 4,
                height: 3
            }
        );
    }

    fn guest_run_req(w: u32, h: u32, total_len: u64, row_length_texels: u32) -> DrawRequest {
        DrawRequest {
            width: w,
            height: h,
            program: test_program(),
            sampled_images: vec![SampledImageResource {
                binding: 32,
                array_element: 0,
                descriptor_count: 1,
                width: w,
                height: h,
                layers: 1,
                arrayed: false,
                volume: false,
                cube: false,
                one_dim: false,
                multisampled: false,
                source: SampledSource::GuestRuns(
                    GuestRunSource {
                        runs: std::sync::Arc::new(vec![GuestRun {
                            host_ptr: 0x1000,
                            len: total_len,
                        }]),
                        source_offset: 0,
                        total_len,
                        row_length_texels,
                        // A fixture over a dummy host address names no guest
                        // RAM, so there is no reference an import could bind.
                        pages: None,
                        physical_pages: None,
                    },
                    // No witness ran for a synthetic source, so nothing vouches:
                    // the gather is the only disposition this fixture can take.
                    reims_vgpu_core::GatherVouch::Fresh,
                ),
                content: None,
                byte_origin: Default::default(),
                format: reims_vgpu_protocol::ImageFormat::linear(
                    reims_vgpu_protocol::TexelLayout::Bgra8,
                ),
                identity: None,
                resource_lifetime: None,
                swizzle: Default::default(),
            }],
            ..DrawRequest::default()
        }
    }

    fn typed_guest_image_req(
        layout: reims_vgpu_memory::GuestImageLayout,
        total_len: u64,
        row_length_texels: u32,
    ) -> DrawRequest {
        let import = std::sync::Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(
                0x7f00_0000_0000,
                0x20_000,
                0x1000,
            )
            .expect("aligned fixture allocation"),
        );
        let transfer = GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: import.host_base(),
                len: total_len,
            }]),
            source_offset: 0,
            total_len,
            row_length_texels,
            pages: None,
            physical_pages: None,
        };
        let memory = reims_vgpu_memory::GuestTargetMemory {
            backing: reims_vgpu_memory::GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0,
                resource_len: import.len(),
                plane_offset: 0,
                row_pitch: if row_length_texels == 0 {
                    u64::from(layout.width()) * 4
                } else {
                    u64::from(row_length_texels) * 4
                },
            },
            import,
            footprint: reims_vgpu_memory::GuestPageFootprint::new(
                std::sync::Arc::from([0x1000]),
                0x1000,
            )
            .expect("one-page fixture footprint"),
        };
        let source = reims_vgpu_memory::GuestImageSource::single_mip(memory, layout, transfer)
            .expect("fixture plane lies inside its resource");
        DrawRequest {
            width: layout.width(),
            height: layout.height(),
            program: test_program(),
            sampled_images: vec![SampledImageResource {
                binding: 32,
                array_element: 0,
                descriptor_count: 1,
                width: layout.width(),
                height: layout.height(),
                layers: if layout.is_volume() {
                    layout.depth()
                } else {
                    layout.array_layers()
                },
                arrayed: layout.is_arrayed(),
                volume: layout.is_volume(),
                cube: false,
                one_dim: layout.is_one_dimensional(),
                multisampled: false,
                source: SampledSource::GuestImage(source, reims_vgpu_core::GatherVouch::Fresh),
                content: None,
                byte_origin: Default::default(),
                format: reims_vgpu_protocol::ImageFormat::linear(
                    reims_vgpu_protocol::TexelLayout::Bgra8,
                ),
                identity: None,
                resource_lifetime: None,
                swizzle: Default::default(),
            }],
            ..DrawRequest::default()
        }
    }

    #[test]
    fn typed_array_source_validates_the_declared_array_pitch() {
        let layout = reims_vgpu_memory::GuestImageLayout::D1Array {
            width: 4,
            layers: 3,
            array_pitch: 32,
        };
        assert!(validate_v1(&typed_guest_image_req(layout, 80, 0)).is_ok());
        assert_eq!(
            validation_slug(&typed_guest_image_req(layout, 48, 0)),
            "vk_draw_validate_guest_sample_length"
        );
    }

    #[test]
    fn typed_volume_source_validates_the_declared_depth_pitch() {
        let layout = reims_vgpu_memory::GuestImageLayout::D3 {
            width: 2,
            height: 2,
            depth: 3,
            depth_pitch: 32,
        };
        assert!(validate_v1(&typed_guest_image_req(layout, 88, 4)).is_ok());
        assert_eq!(
            validation_slug(&typed_guest_image_req(layout, 72, 4)),
            "vk_draw_validate_guest_sample_length"
        );
    }

    /// A cube declares six faces and no array, and the six faces are the six
    /// array slices of an arrayed guest allocation in the same order on both
    /// sides. So the shape check must accept it exactly as it accepts a 2-D
    /// array; refusing it fails the whole draw, which the copying arm — where
    /// the source is `Bytes` and no such check exists — happily encodes.
    #[test]
    fn a_cube_guest_image_matches_an_arrayed_allocation() {
        let layout = reims_vgpu_memory::GuestImageLayout::D2Array {
            width: 4,
            height: 4,
            layers: 6,
            array_pitch: 64,
        };
        let mut req = typed_guest_image_req(layout, 384, 0);
        req.sampled_images[0].arrayed = false;
        req.sampled_images[0].cube = true;
        assert!(validate_v1(&req).is_ok());

        // The relaxation is the cube's alone: a cube that does not cover its
        // allocation's slices is still refused on length like any array.
        let mut short = typed_guest_image_req(layout, 320, 0);
        short.sampled_images[0].arrayed = false;
        short.sampled_images[0].cube = true;
        assert_eq!(
            validation_slug(&short),
            "vk_draw_validate_guest_sample_length"
        );
    }

    /// The untyped run source counts planes and nothing else, so a cube is six
    /// of them. This is the arm a Maps overlay batch actually took, and the
    /// refusal it used to meet was fatal to the whole draw.
    #[test]
    fn a_cube_run_source_is_six_consecutive_planes() {
        let mut req = guest_run_req(4, 4, 384, 0);
        req.sampled_images[0].layers = 6;
        req.sampled_images[0].cube = true;
        assert!(validate_v1(&req).is_ok());

        let mut short = guest_run_req(4, 4, 320, 0);
        short.sampled_images[0].layers = 6;
        short.sampled_images[0].cube = true;
        assert_eq!(
            validation_slug(&short),
            "vk_draw_validate_guest_sample_length"
        );
    }

    #[test]
    fn a_complete_mip_chain_does_not_require_a_direct_import() {
        let mut req =
            typed_guest_image_req(reims_vgpu_memory::GuestImageLayout::D1 { width: 16 }, 96, 0);
        let SampledSource::GuestImage(source, _) = &mut req.sampled_images[0].source else {
            panic!("typed fixture must carry a guest image")
        };
        source.direct = None;
        source.allocation = reims_vgpu_memory::GuestImageAllocationLayout {
            mips: std::sync::Arc::from([
                reims_vgpu_memory::GuestImageMipLayout {
                    resource_relative_offset: 0,
                    row_pitch: 64,
                    layout: reims_vgpu_memory::GuestImageLayout::D1 { width: 16 },
                },
                reims_vgpu_memory::GuestImageMipLayout {
                    resource_relative_offset: 64,
                    row_pitch: 32,
                    layout: reims_vgpu_memory::GuestImageLayout::D1 { width: 8 },
                },
            ]),
        };
        source.view.mip_level_count = 2;
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn typed_array_and_volume_copies_preserve_inter_subresource_pitch() {
        let array = sampled_copy_regions(SampledCopyGeometry {
            binding: 32,
            source_offset: 64,
            width: 4,
            height: 1,
            array_layers: 3,
            extent_depth: 1,
            row_length_texels: 0,
            guest_layout: Some(reims_vgpu_memory::GuestImageLayout::D1Array {
                width: 4,
                layers: 3,
                array_pitch: 32,
            }),
        })
        .unwrap();
        assert_eq!(
            array
                .iter()
                .map(|copy| copy.buffer_offset)
                .collect::<Vec<_>>(),
            [64, 96, 128]
        );
        assert_eq!(
            array
                .iter()
                .map(|copy| copy.image_subresource.base_array_layer)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );

        let volume = sampled_copy_regions(SampledCopyGeometry {
            binding: 32,
            source_offset: 128,
            width: 2,
            height: 2,
            array_layers: 1,
            extent_depth: 3,
            row_length_texels: 4,
            guest_layout: Some(reims_vgpu_memory::GuestImageLayout::D3 {
                width: 2,
                height: 2,
                depth: 3,
                depth_pitch: 32,
            }),
        })
        .unwrap();
        assert_eq!(
            volume
                .iter()
                .map(|copy| (copy.buffer_offset, copy.image_offset.z))
                .collect::<Vec<_>>(),
            [(128, 0), (160, 1), (192, 2)]
        );
    }

    #[test]
    fn allocation_copy_preserves_mips_array_layers_and_volume_depth() {
        let array = GuestAllocationCopy {
            allocation: reims_vgpu_memory::GuestImageAllocationLayout {
                mips: std::sync::Arc::from([
                    reims_vgpu_memory::GuestImageMipLayout {
                        // Resource-relative, as the chain's own type says. The
                        // resource itself sits wherever the allocation puts it;
                        // nothing in this copy path may see that.
                        resource_relative_offset: 0x100,
                        row_pitch: 64,
                        layout: reims_vgpu_memory::GuestImageLayout::D1Array {
                            width: 16,
                            layers: 3,
                            array_pitch: 0x200,
                        },
                    },
                    reims_vgpu_memory::GuestImageMipLayout {
                        resource_relative_offset: 0x140,
                        row_pitch: 32,
                        layout: reims_vgpu_memory::GuestImageLayout::D1Array {
                            width: 8,
                            layers: 3,
                            array_pitch: 0x200,
                        },
                    },
                ]),
            },
            view: reims_vgpu_memory::GuestImageViewRange {
                base_mip_level: 0,
                mip_level_count: 2,
                base_array_layer: 1,
                array_layer_count: 2,
            },
            transfer_source_offset: 0,
            bytes_per_texel: 4,
        };
        let regions = sampled_allocation_copy_regions(32, 0, 4, true, &array).unwrap();
        assert_eq!(
            regions
                .iter()
                .map(|region| (
                    region.buffer_offset,
                    region.image_subresource.mip_level,
                    region.image_subresource.base_array_layer,
                ))
                .collect::<Vec<_>>(),
            [(0x300, 0, 0), (0x500, 0, 1), (0x340, 1, 0), (0x540, 1, 1)]
        );

        let volume = GuestAllocationCopy {
            allocation: reims_vgpu_memory::GuestImageAllocationLayout {
                mips: std::sync::Arc::from([
                    reims_vgpu_memory::GuestImageMipLayout {
                        resource_relative_offset: 0x80,
                        row_pitch: 32,
                        layout: reims_vgpu_memory::GuestImageLayout::D3 {
                            width: 8,
                            height: 4,
                            depth: 2,
                            depth_pitch: 128,
                        },
                    },
                    reims_vgpu_memory::GuestImageMipLayout {
                        resource_relative_offset: 0x180,
                        row_pitch: 16,
                        layout: reims_vgpu_memory::GuestImageLayout::D3 {
                            width: 4,
                            height: 2,
                            depth: 1,
                            depth_pitch: 32,
                        },
                    },
                ]),
            },
            view: reims_vgpu_memory::GuestImageViewRange {
                base_mip_level: 0,
                mip_level_count: 2,
                base_array_layer: 0,
                array_layer_count: 1,
            },
            transfer_source_offset: 0,
            bytes_per_texel: 4,
        };
        let regions = sampled_allocation_copy_regions(32, 0x20, 4, false, &volume).unwrap();
        assert_eq!(
            regions
                .iter()
                .map(|region| (
                    region.buffer_offset,
                    region.image_subresource.mip_level,
                    region.image_offset.z,
                ))
                .collect::<Vec<_>>(),
            [(0xa0, 0, 0), (0x120, 0, 1), (0x1a0, 1, 0)]
        );
    }

    #[test]
    fn typed_copy_offset_overflow_is_a_named_refusal() {
        let decline = sampled_copy_regions(SampledCopyGeometry {
            binding: 32,
            source_offset: u64::MAX,
            width: 1,
            height: 1,
            array_layers: 2,
            extent_depth: 1,
            row_length_texels: 0,
            guest_layout: Some(reims_vgpu_memory::GuestImageLayout::D1Array {
                width: 1,
                layers: 2,
                array_pitch: 4,
            }),
        })
        .unwrap_err();
        assert_eq!(decline.slug(), "vk_draw_exec_sampled_copy_offset_overflow");
    }

    fn guest_target_seed_req(
        w: u32,
        h: u32,
        declared: u64,
        covered: u64,
        row_length_texels: u32,
    ) -> DrawRequest {
        DrawRequest {
            width: w,
            height: h,
            program: test_program(),
            target_guest: Some(super::super::types::GuestTargetPlan::Seed(
                super::super::types::GuestTargetSeed {
                    source: GuestRunSource {
                        runs: std::sync::Arc::new(vec![GuestRun {
                            host_ptr: 0x1000,
                            len: covered,
                        }]),
                        source_offset: 0,
                        total_len: declared,
                        row_length_texels,
                        pages: None,
                        physical_pages: None,
                    },
                    format: reims_vgpu_protocol::TexelLayout::Rgba8,
                },
            )),
            ..DrawRequest::default()
        }
    }

    #[test]
    fn attachment_seed_sources_exist_only_at_the_encoder_boundary() {
        let mut req = guest_target_seed_req(16, 16, 1024, 1024, 16);
        assert!(SegmentLoadSources::for_request(&req).guest.is_some());

        req.continues_render_pass = true;
        let continuation = SegmentLoadSources::for_request(&req);
        assert!(!continuation.has_seed());
        assert!(continuation.cpu.is_none());
        assert!(continuation.guest.is_none());
        assert!(continuation.resident.is_none());
    }

    #[test]
    fn guest_load_action_runs_only_at_the_encoder_boundary() {
        use super::super::types::ColorLoadAction;

        assert_eq!(
            color_load_for_segment(ColorLoadAction::DontCare, false, false),
            ColorLoadKey::DontCare
        );
        assert_eq!(
            color_load_for_segment(ColorLoadAction::Clear, false, false),
            ColorLoadKey::Clear
        );
        assert_eq!(
            color_load_for_segment(ColorLoadAction::Load, false, true),
            ColorLoadKey::Load
        );

        for action in [
            ColorLoadAction::Load,
            ColorLoadAction::Clear,
            ColorLoadAction::DontCare,
        ] {
            assert_eq!(
                color_load_for_segment(action, true, false),
                ColorLoadKey::Load,
                "a backend pass split must preserve the preceding guest segment for {action:?}"
            );
        }
    }

    /// Every variant a resident can be in, so the tests below enumerate rather
    /// than sample. A new variant that nothing here mentions fails to compile,
    /// which is the point: each one is a rail that can leave a resident in a
    /// state some barrier has to name.
    fn every_access() -> [ResidentAccess; 7] {
        [
            ResidentAccess::Untouched,
            ResidentAccess::HostWrite(vk::ImageLayout::PREINITIALIZED),
            ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            ResidentAccess::ColorWrite(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            ResidentAccess::ColorFeedback(vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT),
            ResidentAccess::shader_read(),
            ResidentAccess::transfer_read(),
        ]
    }

    fn target_sample(identity: super::super::types::TargetIdentity) -> SampledImageResource {
        SampledImageResource {
            binding: 32,
            array_element: 0,
            descriptor_count: 1,
            width: 16,
            height: 16,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            multisampled: false,
            source: SampledSource::Target(identity),
            content: None,
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
            identity: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        }
    }

    #[test]
    fn sampled_retain_uses_the_exact_array_element() {
        let identity = super::super::types::TargetIdentity::Surface {
            id: 7,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        let first = target_sample(identity.clone());
        let mut second = target_sample(identity);
        second.array_element = 1;
        let req = DrawRequest {
            sampled_images: vec![first, second],
            ..DrawRequest::default()
        };

        let selected = sampled_resource_at(&req, 32, 1).expect("array member one");
        assert!(std::ptr::eq(selected, &req.sampled_images[1]));
        assert!(sampled_resource_at(&req, 32, 2).is_none());
    }

    /// Every resident attachment alias selects a stable snapshot. Host support
    /// for the feedback-loop layout cannot change this decision because that
    /// feature does not supply overlapping fragment ordering.
    #[test]
    fn sampled_attachment_aliases_are_selected_for_snapshot() {
        let primary = super::super::types::TargetIdentity::Surface {
            id: 7,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        let mut req = DrawRequest {
            target_identity: Some(primary.clone()),
            ..DrawRequest::default()
        };
        let plain = target_sample(primary.clone());
        assert_eq!(
            sampled_attachment_slot(&req, &plain),
            Some(super::super::types::AttachmentSlot::Primary)
        );

        let mut non_plain = target_sample(primary);
        non_plain.arrayed = true;
        assert_eq!(
            sampled_attachment_slot(&req, &non_plain),
            Some(super::super::types::AttachmentSlot::Primary),
            "view shape changes how the snapshot is stored, not whether the live target aliases"
        );

        let secondary = secondary_with_clear([0.0; 4]);
        let secondary_identity = secondary.identity.clone();
        req.secondary_targets.push(secondary);
        assert_eq!(
            sampled_attachment_slot(&req, &target_sample(secondary_identity)),
            Some(super::super::types::AttachmentSlot::Secondary)
        );

        let unrelated = super::super::types::TargetIdentity::Surface {
            id: 99,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        assert_eq!(
            sampled_attachment_slot(&req, &target_sample(unrelated)),
            None
        );
    }

    /// The invariant the whole type exists for: **where a resident sits does not
    /// tell you what a barrier over it must wait for.**
    ///
    /// A resident sitting in `TRANSFER_SRC_OPTIMAL` may have been put there by a
    /// render pass's `final_layout`, with no transfer having run — in which case
    /// the thing to wait for is a colour attachment write — or by the present
    /// blit or a readback, in which case it is a transfer read. Two different
    /// dependencies, one layout.
    ///
    /// Anyone who re-derives a source scope from `layout()` — which is the bug
    /// this replaced, five times over — makes these two agree and fails here.
    ///
    /// **The colliding pair is not currently reachable, and the test still
    /// earns its place.** Since passes exit at
    /// [`super::caches::color0_pass_exit_layout`] no two *reachable* variants
    /// share a layout, so a `layout()`-derived scope would happen to be right
    /// today. That is a property of one constant, not of the design: the
    /// constructor still admits `ColorWrite` at any layout, and the last time
    /// that constant moved it moved *to* a transfer layout. This test is what
    /// makes the next move safe rather than the sixth instance of the bug.
    #[test]
    fn one_resident_layout_carries_two_different_dependencies() {
        let drawn = ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let read_back = ResidentAccess::transfer_read();
        assert_eq!(
            drawn.layout(),
            read_back.layout(),
            "the premise: a draw and a readback leave a resident in the same layout"
        );
        assert_ne!(
            drawn.source_scope(),
            read_back.source_scope(),
            "so a barrier's source scope cannot be a function of the layout"
        );
        let (stage, access) = drawn.source_scope();
        assert!(
            stage.contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                && access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            "a resident a draw produced must be waited for as a colour write, got {stage:?}/{access:?}"
        );
    }

    /// A resident target that a previous draw *sampled* is left in
    /// `SHADER_READ_ONLY_OPTIMAL`. Writing it is a write over pixels a reader
    /// may still be consuming, so the barrier's first scope has to name that
    /// reader. `TOP_OF_PIPE`/no-access orders nothing at all.
    #[test]
    fn writing_a_sampled_target_waits_for_the_sampler() {
        let (stage, access) = ResidentAccess::shader_read().source_scope();
        assert!(
            stage.contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "the write must be ordered after the sampling fragment shader, got {stage:?}"
        );
        assert!(!stage.contains(vk::PipelineStageFlags::TOP_OF_PIPE));
        assert_eq!(access, vk::AccessFlags::SHADER_READ);
    }

    #[test]
    fn a_transfer_read_uses_the_dedicated_layout() {
        // A transfer read is a genuine second layout: nothing renders in
        // TRANSFER_SRC_OPTIMAL, so the transition is owed and the pass has to be
        // closed for it. That is the ~1 200-a-second population the exit layout
        // was moved *towards*, and it stays.
        assert_eq!(
            ResidentAccess::transfer_read().layout(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
    }

    /// The whole of the resident-sample repair, as one relation: a colour target
    /// this device rendered into is already in the layout a sampled read wants,
    /// and the pass's own entry already makes the write visible — so the draw
    /// owes nothing.
    ///
    /// Asserted against [`super::super::caches::color0_pass_exit_layout`] rather
    /// than against `GENERAL`, so it states the relation and holds under
    /// `REIMS_VGPU_COLOR_GENERAL=off` too, where both sides move back together.
    /// Written this way because the failure it guards is the two sides moving
    /// *apart*: a sampled read that keeps its own layout brings back 25 344 pass
    /// breaks a boot, and nothing but this would say so.
    #[test]
    fn a_rendered_resident_is_already_where_a_sampled_read_wants_it() {
        let resting = super::super::caches::color0_pass_exit_layout();
        let after_a_pass = ResidentAccess::ColorWrite(resting);
        let for_a_sample = ResidentAccess::shader_read();

        if super::super::caches::single_color_layout() {
            assert_eq!(for_a_sample.layout(), resting);
            assert!(after_a_pass.covered_by_pass_entry());
        } else {
            assert_ne!(for_a_sample.layout(), resting);
        }
    }

    /// Only an access a pass's incoming external dependency genuinely names may
    /// let a sampled read skip its barrier.
    ///
    /// Written over `every_access` rather than as two assertions so a new
    /// [`ResidentAccess`] variant has to be classified here rather than silently
    /// joining the skipping side. A wrong `true` is a sampled read racing the
    /// write that produced its pixels — a stale frame, reported nowhere.
    #[test]
    fn nothing_untouched_or_host_written_is_covered_by_the_pass_entry() {
        for tracked in every_access() {
            let covered = !matches!(
                tracked,
                ResidentAccess::Untouched | ResidentAccess::HostWrite(_)
            );
            assert_eq!(tracked.covered_by_pass_entry(), covered, "{tracked:?}");
            // A covered access must still name a real prior scope, or "covered"
            // would be covering nothing.
            if covered {
                assert_ne!(
                    tracked.source_scope().0,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    "{tracked:?}"
                );
            }
        }
    }

    /// A fresh registry slot and every pooled target are untouched. Nothing has
    /// happened to the image, so there is genuinely nothing to wait for — and it
    /// is the *only* state of which that is true, which is what the clear path's
    /// skip rests on. Enumerating the rest is the check; restating the call
    /// site's condition would pass whichever way the skip went.
    #[test]
    fn untouched_is_the_only_state_with_no_prior_access() {
        for access in every_access() {
            let (stage, flags) = access.source_scope();
            if access == ResidentAccess::Untouched {
                assert_eq!(stage, vk::PipelineStageFlags::TOP_OF_PIPE);
                assert_eq!(flags, vk::AccessFlags::empty());
            } else {
                assert_ne!(
                    stage,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    "{access:?} names a prior access, so skipping its barrier drops a dependency"
                );
                assert!(
                    !flags.is_empty(),
                    "{access:?} has an access to make available"
                );
            }
        }
    }

    /// A secondary colour attachment that clears to `clear`; every other field
    /// is irrelevant to the clear-value vector and takes a neutral value.
    fn secondary_with_clear(clear: [f32; 4]) -> super::super::types::SecondaryColorTarget {
        super::super::types::SecondaryColorTarget {
            target_guest: None,
            identity: super::super::types::TargetIdentity::Surface {
                id: 1,
                width: 16,
                height: 16,
                generation: 1,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            },
            width: 16,
            height: 16,
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
            clear,
            load_action: super::super::types::ColorLoadAction::Clear,
            blend: None,
            color_write_mask: Default::default(),
        }
    }

    /// A `MTLLoadActionClear` reaches the attachment as the render pass's own
    /// clear value, not as a bitmap of that colour.
    ///
    /// The primary's entry was a hard-coded transparent black, so the only way
    /// to honour a colour was to allocate a whole-attachment RGBA8 image of it,
    /// exchange its channels and stage it to the GPU — per draw, for a constant.
    /// That also set `target_rgba8`, which is what `load_seed` means, so the
    /// pass resolved to LOAD and a draw asking to discard its attachment loaded
    /// it instead.
    ///
    /// Asserting the primary's slot follows the request is the half that fails
    /// against the old hard-coded value. Asserting the positions is the other
    /// half: Vulkan indexes this vector positionally, so a missing or reordered
    /// entry hands one attachment another's colour with nothing to refuse it.
    #[test]
    fn every_attachment_takes_its_own_clear_and_the_primary_takes_the_guests() {
        let req = DrawRequest {
            target_clear: [0.25, 0.5, 0.75, 1.0],
            secondary_targets: vec![
                secondary_with_clear([1.0, 0.0, 0.0, 1.0]),
                secondary_with_clear([0.0, 1.0, 0.0, 0.5]),
            ],
            depth: Some(super::super::types::DepthState {
                // No guest depth texture: this synthetic request exercises the
                // transient rail, which is the one that still owns its image.
                identity: None,
                test_enable: true,
                write_enable: true,
                compare: super::super::types::SamplerCompareFunction::Less,
                clear_value: 0.5,
                load: false,
                stencil: None,
            }),
            ..DrawRequest::default()
        };

        let clear = clear_values(&req);
        assert_eq!(
            clear.len(),
            4,
            "one entry per attachment: primary, two secondaries, depth"
        );
        unsafe {
            assert_eq!(
                clear[0].color.float32,
                [0.25, 0.5, 0.75, 1.0],
                "the primary takes the guest's clear colour, not a fixed black"
            );
            assert_eq!(clear[1].color.float32, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(clear[2].color.float32, [0.0, 1.0, 0.0, 0.5]);
            assert_eq!(clear[3].depth_stencil.depth, 0.5);
        }
    }

    /// A default request still clears to transparent black, which is what every
    /// draw that names no colour has always got.
    #[test]
    fn an_unstated_clear_is_transparent_black() {
        let clear = clear_values(&DrawRequest::default());
        assert_eq!(clear.len(), 1, "no secondaries and no depth is one entry");
        unsafe { assert_eq!(clear[0].color.float32, [0.0; 4]) };
    }

    /// This draw's own snapshot copy is recorded after the registry's access is
    /// read, so it names the newer touch and outranks whatever was there.
    #[test]
    fn a_snapshotted_target_waits_for_its_own_snapshot() {
        for tracked in every_access() {
            assert_eq!(
                target_prior_access(true, tracked),
                ResidentAccess::transfer_read(),
                "a snapshot of a {tracked:?} target is still the newest touch"
            );
            assert_eq!(target_prior_access(false, tracked), tracked);
        }
    }

    /// Exactly one prior access lets a `LOAD` pass skip its own barrier, and it
    /// is the one the pass's `VK_SUBPASS_EXTERNAL` dependency already covers.
    ///
    /// Written over `every_access` rather than as two assertions so a new
    /// `ResidentAccess` variant has to be classified here rather than silently
    /// joining the skipping side. A wrong `true` costs a missing layout
    /// transition, which is undefined behaviour and not an error.
    #[test]
    fn only_a_target_left_where_the_next_pass_wants_it_may_skip_its_barrier() {
        for tracked in every_access() {
            let skippable = tracked
                == ResidentAccess::ColorWrite(super::super::caches::color0_pass_exit_layout());
            assert_eq!(
                pass_exit_needs_no_barrier(tracked),
                skippable,
                "{tracked:?}"
            );
            // A snapshot is this draw's own transfer read and always needs the
            // transition back, whatever the registry was tracking.
            assert!(
                !pass_exit_needs_no_barrier(target_prior_access(true, tracked)),
                "a snapshotted {tracked:?} target still has to come back from TRANSFER_SRC"
            );
        }
    }

    #[test]
    fn guest_runs_tight_total_validates() {
        let req = guest_run_req(1240, 622, 1240 * 622 * 4, 0);
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn guest_runs_volume_validates_every_depth_plane() {
        let mut req = guest_run_req(16, 8, 16 * 8 * 4 * 3, 0);
        req.sampled_images[0].volume = true;
        req.sampled_images[0].layers = 3;
        assert!(validate_v1(&req).is_ok());

        let SampledSource::GuestRuns(source, _) = &mut req.sampled_images[0].source else {
            panic!("the fixture is guest-backed")
        };
        source.total_len -= 1;
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_length"
        );
    }

    #[test]
    fn padded_guest_run_volume_counts_interplane_row_stride() {
        // Three 4-texel-by-2-row planes at a six-texel row pitch: five full
        // padded rows followed by the final tight row.
        let mut req = guest_run_req(4, 2, 5 * 24 + 16, 6);
        req.sampled_images[0].volume = true;
        req.sampled_images[0].layers = 3;
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn sampled_subrange_validates_inside_its_resource_owned_runs() {
        let mut req = guest_run_req(4, 4, 4 * 4 * 4, 0);
        {
            let SampledSource::GuestRuns(source, _) = &mut req.sampled_images[0].source else {
                panic!("the fixture is guest-backed")
            };
            source.source_offset = 32;
            source.runs = std::sync::Arc::new(vec![GuestRun {
                host_ptr: 0x1000,
                len: source.source_offset + source.total_len,
            }]);
        }
        assert!(validate_v1(&req).is_ok());

        {
            let SampledSource::GuestRuns(source, _) = &mut req.sampled_images[0].source else {
                panic!("the fixture is guest-backed")
            };
            source.runs = std::sync::Arc::new(vec![GuestRun {
                host_ptr: 0x1000,
                len: source.source_offset + source.total_len - 1,
            }]);
        }
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_coverage"
        );
    }

    /// A native attachment seed may carry guest row padding, but its declared
    /// span and host-run coverage must describe exactly the bytes the Vulkan
    /// copy reads. This is the shape produced by IOSurface rows whose BPR is
    /// wider than their visible texels.
    #[test]
    fn a_guest_target_seed_validates_its_native_format_stride_and_coverage() {
        // Two four-texel rows at six texels per row: one 24-byte stride plus
        // the final row's 16 visible bytes. Trailing padding is not part of the
        // window.
        let req = guest_target_seed_req(4, 2, 40, 40, 6);
        assert!(validate_v1(&req).is_ok());

        let mut wrong_format = guest_target_seed_req(4, 2, 40, 40, 6);
        match wrong_format.target_guest.as_mut().unwrap() {
            super::super::types::GuestTargetPlan::Seed(seed) => {
                seed.format = reims_vgpu_protocol::TexelLayout::Bgra8;
            }
            super::super::types::GuestTargetPlan::Backing { .. } => unreachable!(),
        }
        assert_eq!(
            validation_slug(&wrong_format),
            "vk_draw_validate_target_guest_seed_format"
        );

        let short_span = guest_target_seed_req(4, 2, 39, 39, 6);
        assert_eq!(
            validation_slug(&short_span),
            "vk_draw_validate_target_guest_seed_length"
        );

        let short_coverage = guest_target_seed_req(4, 2, 40, 39, 6);
        assert_eq!(
            validation_slug(&short_coverage),
            "vk_draw_validate_target_guest_seed_coverage"
        );

        let mut conflicting = guest_target_seed_req(4, 2, 40, 40, 6);
        conflicting.target_rgba8 = Some(std::sync::Arc::new(vec![0; 4 * 2 * 4]));
        assert_eq!(
            validation_slug(&conflicting),
            "vk_draw_validate_seed_conflicts_guest_seed"
        );
    }

    /// A geometry whose byte length has no `usize` is refused, not multiplied.
    ///
    /// Both sites took the product straight, and both had enough factors to
    /// overflow: `w as usize * h as usize` widens its operands and still leaves
    /// room for nothing else, because two `u32` maxima come within nine billion
    /// of exhausting a `u64` by themselves. In a debug build that is a panic
    /// raised **inside the validator**, from the request it was handed to
    /// survive; in a release build it wraps, and the wrapped length is what the
    /// very next line compares a buffer against, so a short buffer matches.
    ///
    /// Both arms are asserted here because they are different expressions in
    /// different functions and only one of them has a layer count.
    #[test]
    fn a_geometry_with_no_representable_length_is_refused_by_name() {
        let mut req = guest_run_req(1240, 622, 1240 * 622 * 4, 0);
        // The sampled arm: four factors, and `layers` is the one that tips it.
        // `arrayed`, because a non-array image with layers != 1 is refused
        // earlier and would never reach the multiplication.
        req.sampled_images[0].arrayed = true;
        req.sampled_images[0].width = u32::MAX;
        req.sampled_images[0].height = u32::MAX;
        req.sampled_images[0].layers = u32::MAX;
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_unrepresentable_image_bytes"
        );

        // The target-seed arm, reached before the sampled one, so the geometry
        // above is left at a size that validates.
        let mut req = guest_run_req(1240, 622, 1240 * 622 * 4, 0);
        req.width = u32::MAX;
        req.height = u32::MAX;
        req.target_rgba8 = Some(std::sync::Arc::new(vec![0u8; 4]));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_unrepresentable_image_bytes"
        );
    }

    /// The refusal must not have swallowed the length check it replaced.
    ///
    /// A `let Some(_) = … else { decline }` in front of a comparison is the
    /// shape where the comparison quietly stops running, and the seed-length
    /// check is the one thing standing between a short buffer and a read past
    /// its end.
    #[test]
    fn a_representable_geometry_still_has_its_seed_length_checked() {
        let mut req = guest_run_req(4, 4, 4 * 4 * 4, 0);
        req.target_rgba8 = Some(std::sync::Arc::new(vec![0u8; 4 * 4 * 4]));
        assert!(validate_v1(&req).is_ok(), "an exactly-sized seed is fine");

        req.target_rgba8 = Some(std::sync::Arc::new(vec![0u8; 4 * 4 * 4 - 1]));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_target_seed_length",
            "a seed one byte short must still be caught"
        );
    }

    /// The Safari content-layer case: width 1240, guest stride 1280 texels.
    /// The window spans (h-1)*stride + tight last row — NOT w*h*bpp; the
    /// tight comparison rejected every padded-stride zero-copy bind and the
    /// dropped draw left the app window content permanently blank.
    #[test]
    fn guest_runs_padded_stride_validates() {
        let padded = 621 * 1280 * 4 + 1240 * 4; // 3_184_480
        let req = guest_run_req(1240, 622, padded as u64, 1280);
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn guest_runs_padded_stride_rejects_tight_total() {
        let req = guest_run_req(1240, 622, 1240 * 622 * 4, 1280);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_length"
        );
    }

    #[test]
    fn guest_runs_rejects_stride_under_width() {
        let total = 621 * 1024 * 4 + 1240 * 4;
        let req = guest_run_req(1240, 622, total as u64, 1024);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_row_stride"
        );
    }

    fn buffer_guest_runs(
        run_lens: &[u64],
        total_len: u64,
        row_length_texels: u32,
    ) -> BufferContent {
        BufferContent::GuestRuns(super::super::types::GuestRunSource {
            runs: std::sync::Arc::new(
                run_lens
                    .iter()
                    .map(|&len| GuestRun {
                        host_ptr: 0x1000,
                        len,
                    })
                    .collect(),
            ),
            source_offset: 0,
            total_len,
            row_length_texels,
            // A fixture over a dummy host address has no guest pages.
            pages: None,
            physical_pages: None,
        })
    }

    fn storage_buffer_req(content: BufferContent) -> DrawRequest {
        DrawRequest {
            width: 8,
            height: 8,
            program: test_program(),
            storage_buffers: vec![super::super::types::StorageBufferResource {
                binding: 0,
                content,
            }],
            ..DrawRequest::default()
        }
    }

    fn index_buffer_req(content: BufferContent) -> DrawRequest {
        DrawRequest {
            width: 8,
            height: 8,
            program: test_program(),
            indexed: Some(super::super::types::IndexedDrawResource {
                index_type: super::super::types::IndexType::U16,
                index_count: 3,
                vertex_offset: 0,
                content,
            }),
            ..DrawRequest::default()
        }
    }

    #[test]
    fn buffer_guest_runs_consistent_span_validates() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x3000, 0x1000], 0x4000, 0));
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn buffer_guest_runs_rejects_span_mismatch() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x3000], 0x4000, 0));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_coverage"
        );
    }

    #[test]
    fn buffer_guest_runs_rejects_row_stride() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x4000], 0x4000, 64));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_row_stride"
        );
    }

    #[test]
    fn buffer_guest_runs_rejects_empty_span() {
        let req = storage_buffer_req(buffer_guest_runs(&[], 0, 0));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_coverage"
        );
    }

    #[test]
    fn index_guest_runs_validate_the_resource_window_without_reading_it() {
        let req = index_buffer_req(buffer_guest_runs(&[6], 6, 0));
        assert!(validate_v1(&req).is_ok());

        let short = index_buffer_req(buffer_guest_runs(&[5], 5, 0));
        assert_eq!(
            validation_slug(&short),
            "vk_draw_validate_index_bytes_short"
        );

        let uncovered = index_buffer_req(buffer_guest_runs(&[5], 6, 0));
        assert_eq!(
            validation_slug(&uncovered),
            "vk_draw_validate_index_guest_runs_coverage"
        );

        let strided = index_buffer_req(buffer_guest_runs(&[6], 6, 1));
        assert_eq!(
            validation_slug(&strided),
            "vk_draw_validate_index_guest_runs_row_stride"
        );
    }

    /// A Constant-step attribute with a nonzero base instance needs the CPU
    /// prefix shift; a gathered guest span must be rejected at validate time
    /// (the runtime gate keeps those streams on the CPU path).
    #[test]
    fn buffer_guest_runs_rejects_constant_step_shift() {
        let content = buffer_guest_runs(&[48 * 4], 48 * 4, 0);
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            program: test_program(),
            vertex_count: 3,
            base_instance: 2,
            ..DrawRequest::default()
        };
        req.vertex_attributes
            .push(super::super::types::VertexAttributeResource {
                location: 0,
                binding: 0,
                format: super::super::types::VertexAttributeFormat::Float4,
                offset: 0,
                stride: 48,
                step_function: VertexStepFunction::Constant,
                step_rate: 1,
                content: content.clone(),
            });
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_constant_step_guest_runs"
        );
        // Same request with CPU bytes passes.
        req.vertex_attributes[0].content = vec![0u8; 48 * 4].into();
        assert!(validate_v1(&req).is_ok());
    }

    /// One attribute, one step function, one rate — three requests that differ
    /// only in the pair.
    fn step_pair_req(step_function: VertexStepFunction, step_rate: u32) -> DrawRequest {
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            program: test_program(),
            vertex_count: 3,
            ..DrawRequest::default()
        };
        req.vertex_attributes
            .push(super::super::types::VertexAttributeResource {
                location: 0,
                binding: 0,
                format: super::super::types::VertexAttributeFormat::Float4,
                offset: 0,
                stride: 48,
                step_function,
                step_rate,
                content: vec![0u8; 48 * 4].into(),
            });
        req
    }

    /// A constant-rate attribute is spelled with a zero rate, and that is the
    /// only step function it is spelled with.
    ///
    /// `MTLVertexBufferLayoutDescriptor.stepRate` must be 0 under
    /// `MTLVertexStepFunctionConstant`, the decoder preserves a declared zero
    /// for exactly that reason, and this binding's divisor is 0 whatever the
    /// rate says — so declining the pair lost the whole draw over a field
    /// arm asked `rate == 0` alone. The sibling half of the assertion is what
    /// keeps the repair from becoming "stop checking the rate".
    #[test]
    fn a_zero_step_rate_declines_under_every_step_function_but_constant() {
        assert!(validate_v1(&step_pair_req(VertexStepFunction::Constant, 0)).is_ok());
        for step in [
            VertexStepFunction::PerVertex,
            VertexStepFunction::PerInstance,
        ] {
            assert_eq!(
                validation_slug(&step_pair_req(step, 0)),
                "vk_draw_validate_zero_vertex_step_rate",
                "{step:?} consumes the rate, so zero is still out of contract"
            );
        }
    }

    #[test]
    fn missing_vertex_and_fragment_programs_have_distinct_reasons() {
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            program: test_program(),
            ..DrawRequest::default()
        };
        req.program.vertex.id = reims_vgpu_protocol::PreparedShaderId::new(0);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_missing_vertex_program"
        );

        req.program.vertex.id = reims_vgpu_protocol::PreparedShaderId::new(1);
        req.program.fragment.id = reims_vgpu_protocol::PreparedShaderId::new(0);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_missing_fragment_program"
        );
    }

    /// `cpu_bytes` materializes a fragmented gather exactly (diagnostic /
    /// coverage-proof view of a zero-copy bind).
    #[test]
    fn buffer_content_cpu_bytes_materializes_runs() {
        let backing: Vec<u8> = (0u8..=255).collect();
        let runs = vec![
            GuestRun {
                host_ptr: backing.as_ptr() as usize,
                len: 100,
            },
            GuestRun {
                host_ptr: backing.as_ptr() as usize + 200,
                len: 56,
            },
        ];
        let content = BufferContent::GuestRuns(super::super::types::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 90,
            total_len: 20,
            row_length_texels: 0,
            // A fixture over a host `Vec` has no guest pages.
            pages: None,
            physical_pages: None,
        });
        assert_eq!(content.len(), 20);
        let bytes = content.cpu_bytes();
        assert_eq!(&bytes[..10], &backing[90..100]);
        assert_eq!(&bytes[10..20], &backing[200..210]);
    }
}

#[cfg(test)]
mod depth_load_tests {
    use super::*;

    fn acquired(content_ready: bool) -> AcquiredDepth {
        AcquiredDepth {
            view: vk::ImageView::null(),
            image: vk::Image::null(),
            owned: None,
            identity: None,
            access: crate::engine::pools::ResidentAccess::Untouched,
            content_ready,
        }
    }

    /// A depth LOAD needs both the guest asking and a resident with something in
    /// it, and the second term is the one this device owns.
    ///
    /// The asymmetry is the point. Declaring LOAD to Vulkan also declares
    /// `initial_layout` DEPTH_STENCIL_ATTACHMENT_OPTIMAL, and an image nothing
    /// has rendered into is in UNDEFINED — so honouring the guest's flag on an
    /// empty resident is not a stale read but undefined behaviour, caught by a
    /// validation layer where one runs and silent where none does. A pass that
    /// does not ask must never be given LOAD either: that would turn a CLEAR the
    /// guest asked for into a load of last frame's depth.
    #[test]
    fn a_depth_load_needs_both_the_guest_asking_and_a_resident_holding_something() {
        assert!(
            acquired(true).honours_load(true),
            "guest asked and the resident holds depth: the one case that loads"
        );
        assert!(
            !acquired(false).honours_load(true),
            "an empty resident cannot be loaded from whatever the guest asked"
        );
        assert!(
            !acquired(true).honours_load(false),
            "a guest that asked to clear must clear, resident contents or not"
        );
        assert!(!acquired(false).honours_load(false));
    }

    /// The transient rail can never honour a LOAD, whatever the guest asked.
    ///
    /// Its buffer is created for this draw and destroyed after it, so there has
    /// never been anything in it to load. That is the same conclusion the old
    /// `depth_load_unsupported_transient` decline reached unconditionally, and it
    /// still holds for the rail that decline was named for — what changed is that
    /// the *identified* case is no longer routed through it.
    #[test]
    fn the_transient_depth_rail_never_honours_a_load() {
        let transient = AcquiredDepth {
            view: vk::ImageView::null(),
            image: vk::Image::null(),
            owned: Some((
                vk::Image::null(),
                vk::DeviceMemory::null(),
                vk::ImageView::null(),
            )),
            identity: None,
            access: crate::engine::pools::ResidentAccess::Untouched,
            content_ready: false,
        };
        assert!(!transient.honours_load(true));
    }
}
