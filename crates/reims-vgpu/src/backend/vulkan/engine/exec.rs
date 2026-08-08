//! Record / submit (bounded fence) / readback for one draw.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;

use super::caches::{
    AttrKey, BindingSig, LayoutKey, ObjectCaches, PassKey, PipelineKey, SecondaryAttachKey,
    MAX_SECONDARY_ATTACH,
};
use super::context::ContextOwner;
use super::counters::EngineCounters;
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::draw_execution::DrawExecutionDecline;
use super::draw_validation::DrawValidationDecline;
use super::pools::{BatchTarget, BufferSlot, ResourcePools, SampledKey, SampledSlot, TargetKey};
use super::stage_phase;
use super::types::{
    BufferContent, ColorWriteMask, DrawError, DrawOutput, DrawRequest, SampledSource,
    ScissorResource, SeedOrder, VertexStepFunction, ViewportResource, VisibilityResultMode,
};
use super::vk_call::{VkCall, VkOp};

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
}

impl From<BufferSlot> for BoundBuffer {
    fn from(slot: BufferSlot) -> Self {
        Self {
            buffer: slot.buffer,
            offset: 0,
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
struct PendingGuestGather {
    /// Device-local destination, which is what the draw actually binds.
    dst: vk::Buffer,
    /// One entry per import the window's stretches resolved against, each with
    /// its copy regions.
    ///
    /// Grouped because two stretches need not share an import — a window
    /// straddling two RAMBlocks resolves against two `VkBuffer`s — and one
    /// `vkCmdCopyBuffer` names exactly one source. Ordinary machines have one
    /// RAMBlock and this is a single-entry `Vec`, but the grouping is what makes
    /// the two-block case land the whole window instead of the part that
    /// happened to be first.
    sources: Vec<(vk::Buffer, Vec<vk::BufferCopy>)>,
}

impl PendingGuestGather {
    /// Copy regions across every source, which is what the census counts.
    fn regions(&self) -> u64 {
        self.sources.iter().map(|(_, r)| r.len() as u64).sum()
    }
}

/// The one bind range a window's stretches amount to, when they amount to one.
///
/// A single run starting at window byte zero *is* the whole window:
/// [`crate::runtime::guest_ram_map::references_for_runs`] guarantees the runs
/// ascend and tile the window exactly, so one of them covering byte zero leaves
/// nothing else to name. Anything longer has to be gathered, because a vertex,
/// index or storage bind names one contiguous range.
fn single_run(
    runs: &[crate::runtime::guest_ram_map::GuestWindowRun],
) -> Option<&crate::runtime::guest_ram::GuestRef> {
    match runs {
        [only] if only.window_offset == 0 => Some(&only.guest),
        _ => None,
    }
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
/// at one. See [`crate::runtime::draw::vulkan`]'s `guest_page_window` for the
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
unsafe fn stage_buffer_content(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    content: &BufferContent,
    usage: vk::BufferUsageFlags,
    snapshot_volatile: bool,
    gathers: &mut Vec<PendingGuestGather>,
) -> Result<BoundBuffer, DrawError> {
    let key = match content {
        BufferContent::Bytes(b) => (std::sync::Arc::as_ptr(b) as usize, b.len() as u64),
        BufferContent::GuestRuns(src) => (
            std::sync::Arc::as_ptr(&src.runs) as *const () as usize,
            src.total_len,
        ),
    };
    if let Some(bound) = pools.cb_bound_buffer(key) {
        counters.note_buffer_bind_reused();
        return Ok(bound);
    }
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
            // The bytes are already in memory the device can address. Bind them.
            //
            // What keeps this safe is not the mechanism — a host pointer is one
            // the GPU can write and can stray within — but the bound the
            // reference carries. `src.pages` is a
            // [`crate::runtime::guest_ram::GuestRef`], which no call site can
            // construct without the range check against the RAMBlock it names,
            // and freeing the import is what ends the access.
            if let Some(bound) = unsafe { import_guest_buffer_window(ctx, pools, src) } {
                pools.note_guest_read_recorded();
                counters.note_buffer_guest_import(src.total_len);
                bound
            } else if let Some((bound, pending)) =
                unsafe { gather_guest_buffer_window(ctx, pools, counters, src, usage)? }
            {
                // The copies read guest RAM when the CB executes, exactly as a
                // direct bind does, so this owes the same quiesce.
                pools.note_guest_read_recorded();
                counters.note_buffer_guest_gather(src.total_len, pending.regions());
                gathers.push(pending);
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
                pools.write_staging_from_runs(ctx, &slot, &src.runs, src.total_len)?;
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
    pools.note_cb_bound_buffer(key, bound);
    Ok(bound)
}

/// Bind a buffer source's guest pages directly, or say why they have to be
/// gathered.
///
/// Every `None` is a routing answer and never a lost draw: the caller's CPU
/// gather reads the same bytes through the same runs.
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
unsafe fn import_guest_buffer_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    src: &super::types::GuestRunSource,
) -> Option<BoundBuffer> {
    if !ctx.caps.host_pointer.is_available() {
        return None;
    }
    let guest_ref = single_run(src.pages.as_ref()?)?;
    let bound = match unsafe { pools.bind_guest_ram(ctx, guest_ref) } {
        Ok(bound) => bound,
        Err(inner) => {
            crate::observe::Emit::decline("vk_buffer_import", &inner).fail_once(0);
            return None;
        }
    };
    // The imported buffer spans the whole RAMBlock, so the span's first byte is
    // the bound range's start plus whatever widening it to the device's import
    // granularity added at the front.
    let offset = bound.offset + bound.head;
    // Unlike the sampled rail's copy offset, this one is a *bind*: the device
    // publishes the alignment it will accept and there is no arm that can
    // renegotiate it. A guest span that lands elsewhere is gathered.
    if !offset.is_multiple_of(ctx.guest_bind_offset_align) {
        crate::observe::Emit::decline(
            "vk_buffer_import",
            &BufferImportDecline::BindOffsetAlignment {
                offset,
                align: ctx.guest_bind_offset_align,
            },
        )
        .fail_once(offset % ctx.guest_bind_offset_align);
        return None;
    }
    Some(BoundBuffer {
        buffer: bound.buffer,
        offset,
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
    usage: vk::BufferUsageFlags,
) -> Result<Option<(BoundBuffer, PendingGuestGather)>, DrawError> {
    if !ctx.caps.host_pointer.is_available() {
        return Ok(None);
    }
    let Some(runs) = src.pages.as_ref() else {
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
    for run in runs.iter() {
        let bound = match unsafe { pools.bind_guest_ram(ctx, &run.guest) } {
            Ok(bound) => bound,
            Err(inner) => {
                crate::observe::Emit::decline("vk_buffer_gather", &inner).fail_once(0);
                return Ok(None);
            }
        };
        let copy = gather_region(&bound, run);
        covered = covered.saturating_add(copy.size);
        super::group_by_buffer(&mut sources, bound.buffer, copy);
    }
    // The runs tile the window exactly, so this holds by construction — but a
    // short gather would hand the draw a buffer whose tail is whatever the
    // previous user of the slot left there, which is wrong pixels rather than
    // slow ones. Checked here because this is the last place that can see it.
    if covered != src.total_len {
        crate::observe::Emit::decline(
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

/// The copy one stretch of a scattered window contributes.
///
/// Both offsets are re-based and neither is the number nearest to hand, which is
/// what makes this worth its own function:
///
/// * The **source** is `offset + head`, not `offset`. A bound range is rounded
///   out to the import's granularity, and `head` is what that rounding added in
///   front of the byte the guest actually named. Reading from `offset` would
///   start the stretch up to a granule early — the whole window shifted, which
///   is a wrong draw and not a failed one.
/// * The **size** is `requested()`, not `bound_len()`. The bound length is the
///   same rounding at the other end, so copying it would read guest bytes past
///   the window and write them past the destination's own end.
/// * The **destination** is the stretch's offset *within the window*, which is
///   the one thing a consumer may not compute for itself and the reason
///   [`crate::runtime::guest_ram_map::GuestWindowRun`] carries it.
fn gather_region(
    bound: &super::host_ram::BoundGuestRam,
    run: &crate::runtime::guest_ram_map::GuestWindowRun,
) -> vk::BufferCopy {
    vk::BufferCopy::default()
        .src_offset(bound.offset + bound.head)
        .dst_offset(run.window_offset)
        .size(run.guest.requested())
}

/// A check that sent a guest buffer span back to the CPU gather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferImportDecline {
    /// The span does not start at an offset this device will bind a vertex or
    /// storage buffer at. See [`super::context::DeviceContext::guest_bind_offset_align`].
    BindOffsetAlignment { offset: u64, align: u64 },
    /// The window's stretches did not add up to the window. A healthy zero:
    /// `references_for_runs` tiles exactly, so a firing means the runs and the
    /// length reached here from different windows.
    GatherShort { covered: u64, want: u64 },
}

impl crate::observe::Decline for BufferImportDecline {
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

crate::observe::decline::decline_display!(BufferImportDecline);

/// Where the buffer half of a guest-sourced sampled upload came from.
///
/// The two arms are the same `vkCmdCopyBufferToImage` over a different buffer,
/// and the whole difference is whether the CPU moved the texels first.
enum GuestTexels {
    /// The buffer **is** the guest's pages, reached through the host-pointer
    /// import over their RAMBlock. Nothing copied them into it: the GPU reads
    /// the bytes where the guest wrote them, and `offset` is where the first
    /// texel sits inside that import.
    ///
    /// The read happens when the command buffer executes rather than when it is
    /// recorded, which is later than the guest's fence would otherwise allow —
    /// see [`crate::backend::vulkan::engine::quiesce_guest_reads`], which is
    /// what makes that legal.
    Imported { buffer: vk::Buffer, offset: u64 },
    /// The GPU assembled the window out of the import: one `vkCmdCopyBuffer` per
    /// guest stretch into a device-local slot, recorded ahead of the pass, and
    /// the buffer→image copy then names that slot.
    ///
    /// The arm a real workload takes. The guest backs a surface in 16 KiB
    /// physically-contiguous granules, so a sampled window is a handful of
    /// stretches and essentially never one — which made [`Self::Imported`]
    /// unreachable and sent every bind to the CPU.
    Gathered(BufferSlot),
    /// The CPU packed the texels into a pooled staging span, because this host
    /// could not reach the pages (no `VK_EXT_external_memory_host`, the rail
    /// switched off, a driver that declined the pointer, or a window too
    /// scattered to be worth the copy regions) or the copy could not name them
    /// at the offset they sit at. Always available, which is why the import is
    /// allowed to decline for any reason at all.
    Scratch(BufferSlot),
}

impl GuestTexels {
    fn buffer(&self) -> vk::Buffer {
        match self {
            Self::Imported { buffer, .. } => *buffer,
            Self::Gathered(slot) | Self::Scratch(slot) => slot.buffer,
        }
    }

    fn offset(&self) -> u64 {
        match self {
            Self::Imported { offset, .. } => *offset,
            // Both start at the beginning of a pooled slot the window was
            // assembled into, so the first texel is byte zero.
            Self::Gathered(_) | Self::Scratch(_) => 0,
        }
    }
}

enum PreparedSampled {
    Upload {
        binding: u32,
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
    /// this content. The slot is still retained when the producer vouched for an
    /// identity — see [`super::types::SampledSource::GuestRuns`] — and the next
    /// bind of the same window under the same generation binds it back through
    /// `find_sampled_by_identity` without touching the bytes at all.
    GuestGather {
        binding: u32,
        image: SampledSlot,
        source: GuestTexels,
        /// `bufferRowLength` for the buffer→image copy (0 = tight rows).
        row_length_texels: u32,
        /// Bytes the copy names, for the cache's byte-cap accounting.
        gathered_len: usize,
    },
    Cached {
        binding: u32,
        image: SampledSlot,
    },
    Resident {
        binding: u32,
        identity: super::types::TargetIdentity,
        image: vk::Image,
        view: vk::ImageView,
        access: super::pools::ResidentAccess,
    },
    Snapshot {
        binding: u32,
        identity: super::types::TargetIdentity,
        source_image: vk::Image,
        source_access: super::pools::ResidentAccess,
        image: SampledSlot,
    },
}

impl PreparedSampled {
    fn binding(&self) -> u32 {
        match self {
            Self::Upload { binding, .. }
            | Self::Cached { binding, .. }
            | Self::Resident { binding, .. }
            | Self::Snapshot { binding, .. }
            | Self::GuestGather { binding, .. } => *binding,
        }
    }

    fn view(&self) -> vk::ImageView {
        match self {
            Self::Upload { image, .. } => image.view,
            Self::Cached { image, .. } => image.view,
            Self::Resident { view, .. } => *view,
            Self::Snapshot { image, .. } => image.view,
            Self::GuestGather { image, .. } => image.view,
        }
    }
}

/// `vkCmdCopyBufferToImage` requires `bufferOffset` to be a multiple of 4 and of
/// the format's texel block size. 16 is the largest uncompressed block in core
/// Vulkan and the larger of the two BC block sizes, so one check covers every
/// format the sampled pool can produce without the arm having to know which one
/// it is holding. A guest window whose first texel sits at any other offset
/// takes the CPU gather, which has no such rule.
const GUEST_IMPORT_COPY_OFFSET_ALIGN: u64 = 16;

/// Bind a sampled source's guest pages as a buffer the copy can read directly,
/// or say why it must be gathered on the CPU instead.
///
/// Every `None` is a routing answer and never a lost frame: the caller's CPU
/// gather reads the same bytes through the same runs. So the checks here are
/// free to be conservative — a window this refuses costs one `memcpy` that the
/// device was making unconditionally until now.
///
/// # Safety
///
/// `ctx` must own the device `pools` holds every live import against.
unsafe fn import_sampled_guest_window(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    src: &super::types::GuestRunSource,
    gathers: &mut Vec<PendingGuestGather>,
) -> Result<Option<GuestTexels>, DrawError> {
    if !ctx.caps.host_pointer.is_available() {
        return Ok(None);
    }
    let Some(runs) = src.pages.as_ref() else {
        return Ok(None);
    };
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
    if let Some(guest_ref) = single_run(runs) {
        let bound = match unsafe { pools.bind_guest_ram(ctx, guest_ref) } {
            Ok(bound) => bound,
            Err(inner) => {
                crate::observe::Emit::decline("vk_sampled_import", &inner).fail_once(0);
                return Ok(None);
            }
        };
        // As on the buffer rail: the buffer spans the RAMBlock, so the first
        // texel sits at the bound range's start plus the granularity widening.
        let offset = bound.offset + bound.head;
        if !offset.is_multiple_of(GUEST_IMPORT_COPY_OFFSET_ALIGN) {
            crate::observe::Emit::decline(
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
    let mut sources: Vec<(vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
    let mut covered = 0u64;
    for run in runs.iter() {
        let bound = match unsafe { pools.bind_guest_ram(ctx, &run.guest) } {
            Ok(bound) => bound,
            Err(inner) => {
                crate::observe::Emit::decline("vk_sampled_import", &inner).fail_once(0);
                return Ok(None);
            }
        };
        let copy = gather_region(&bound, run);
        covered = covered.saturating_add(copy.size);
        super::group_by_buffer(&mut sources, bound.buffer, copy);
    }
    // The runs tile the window exactly, so this holds by construction — but a
    // short gather would hand the draw an image whose tail is whatever the
    // previous user of the slot left there, which is wrong pixels rather than
    // slow ones. Checked here because this is the last place that can see it.
    if covered != src.total_len {
        crate::observe::Emit::decline(
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

impl crate::observe::Decline for SampledImportDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CopyOffsetAlignment { .. } => "sampled_import_copy_offset_alignment",
            Self::GatherShort { .. } => "sampled_import_gather_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CopyOffsetAlignment { offset } => vec![("offset", offset.to_string())],
            Self::GatherShort { covered, want } => vec![
                ("covered", covered.to_string()),
                ("want", want.to_string()),
            ],
        }
    }
}

crate::observe::decline::decline_display!(SampledImportDecline);

/// Shared validation for a draw-time buffer's content source. A `GuestRuns`
/// span must be internally consistent: the run lengths sum to `total_len`,
/// the span is non-empty, and `row_length_texels` is 0 (row strides are a
/// texture concept — buffers gather a flat byte span).
#[derive(Clone, Copy)]
enum BufferValidationRole {
    Vertex,
    Storage,
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
        };
        return Err(DrawError::DrawValidation(decline));
    }
    let sum: u64 = src.runs.iter().map(|r| r.len).sum();
    if sum != src.total_len || src.total_len == 0 {
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
        };
        return Err(DrawError::DrawValidation(decline));
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
    if req.vert_spirv.is_empty() {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::EmptyVertexSpirv,
        ));
    }
    if req.frag_spirv.is_empty() {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::EmptyFragmentSpirv,
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
    if let Some(blend) = req.blend {
        if blend.constants.iter().any(|c| !c.is_finite()) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonFiniteBlendConstants,
            ));
        }
    }
    if let Some(target) = &req.target_rgba8 {
        // The seed is one tightly-packed RGBA8 slice of the target, and the
        // length is checked rather than taken. `w as usize * h as usize` widens
        // its operands, which reads as safe and is — but only just: two u32
        // maxima multiply to a hair under u64::MAX, so the bytes-per-texel is a
        // third factor that overflows. A refusal rather than a clamp, because
        // this length is what the next line compares the buffer against and a
        // wrapped one would let a short buffer match.
        let Some(expected) = crate::contract::extent::tight_image_bytes(
            req.width,
            req.height,
            crate::contract::pixel_format::RGBA8_BPP as usize,
        ) else {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::UnrepresentableImageBytes {
                    width: req.width,
                    height: req.height,
                    layers: 1,
                    bytes_per_texel: crate::contract::pixel_format::RGBA8_BPP,
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
    if let Some(seed_identity) = &req.seed_from_target {
        if req.target_identity.is_none() {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedMissingTargetIdentity,
            ));
        }
        if req.target_rgba8.is_some() {
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
            SampledSource::Target(identity) => identity == seed_identity,
            SampledSource::Bytes(_) | SampledSource::GuestRuns(..) => false,
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
            if indexed.indices.len() < need {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::IndexBytesShort {
                        actual: indexed.indices.len(),
                        expected: need,
                    },
                ));
            }
            if no_vertex_fetch {
                0
            } else {
                let (min_index, max_index) = indexed.index_range();
                let first = i64::from(min_index) + i64::from(indexed.vertex_offset);
                let last = i64::from(max_index) + i64::from(indexed.vertex_offset);
                if first < 0 || last < 0 || last > u32::MAX as i64 {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::IndexedVertexRange {
                            min_index,
                            max_index,
                            vertex_offset: indexed.vertex_offset,
                        },
                    ));
                }
                last as u32
            }
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
        // field nothing downstream reads. `contract::vertex_step` owns the pair.
        if !crate::contract::vertex_step::step_rate_in_contract(
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
        if !bindings.insert(buffer.binding) {
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
        // Footprint of one texel of the image's own format. `None` means a
        // format whose bytes are not one number per texel (block-compressed,
        // multi-planar) reached a rail that sizes a linear buffer — decline by
        // name rather than compute a wrong length.
        let Some(texel) = super::super::translate::pixel::bytes_per_texel(image.format) else {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledNoLinearTexelFootprint {
                    binding: image.binding,
                    format: image.format,
                },
            ));
        };
        let texel = texel as usize;
        // Four factors, so the widening the operands already carry is not
        // enough — see the target-seed check above for why two of them exhaust
        // a u64 on their own. `contract::extent` owns the checked form.
        let Some(expected) = crate::contract::extent::tight_layered_image_bytes(
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
            SampledSource::Target(identity) => {
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
            SampledSource::Bytes(_) => {}
            SampledSource::GuestRuns(src, _) => {
                // The zero-copy gather uploads a single array layer into a
                // single-depth image (`layer_count: 1`, `depth: 1` below), so
                // it serves any shape that is one layer deep: plain 2D, a
                // single-layer 2D array, and the 1D / single-layer 1D-array
                // color-transfer LUTs. Volume and multi-layer shapes still
                // decline by name — the gather would upload only their first
                // slice.
                if image.volume || image.cube || image.layers != 1 {
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::GuestRunSampledNot2d {
                            binding: image.binding,
                        },
                    ));
                }
                // Padded layouts (`row_length_texels != 0`) span
                // `(height-1) * stride + tight_row` — the final row carries
                // only its texels (see `GuestRunSource`); tight layouts match
                // the full `width * height` window.
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
                    (image.height as usize - 1) * stride + tight_row
                };
                if src.total_len as usize != run_expected {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::GuestSampleLength {
                            binding: image.binding,
                            actual: src.total_len,
                            expected: run_expected,
                        },
                    ));
                }
                let sum: u64 = src.runs.iter().map(|r| r.len).sum();
                if sum != src.total_len || src.runs.is_empty() {
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
        if !bindings.insert(image.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateSampledDescriptorBinding {
                    binding: image.binding,
                },
            ));
        }
    }
    for sampler in &req.samplers {
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
        if !bindings.insert(sampler.binding) {
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
) {
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
    let copy = [vk::BufferImageCopy::default()
        .buffer_offset(src_offset)
        .buffer_row_length(row_length_texels)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: array_layers,
        })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: extent_depth,
        })];
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
}

/// The ten conditions that decide how a draw reaches the submission ring.
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
    has_depth: bool,
    reads_back: bool,
    has_query: bool,
    no_identity: bool,
    cpu_seed: bool,
    gpu_seed: bool,
    no_open_batch: bool,
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
        (|t| t.has_depth, JoinScope::Draw, "nojoin_depth"),
        (|t| t.reads_back, JoinScope::Draw, "nojoin_reads_back"),
        (|t| t.has_query, JoinScope::Draw, "nojoin_query"),
        (|t| t.no_identity, JoinScope::Draw, "nojoin_no_identity"),
        (|t| t.cpu_seed, JoinScope::Fit, "nojoin_cpu_seed"),
        (|t| t.gpu_seed, JoinScope::Fit, "nojoin_gpu_seed"),
        (|t| t.no_open_batch, JoinScope::Fit, "nojoin_no_open_batch"),
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
unsafe fn acquire_depth_view(
    ctx: &super::context::DeviceContext,
    pools: &mut super::pools::ResourcePools,
    req: &DrawRequest,
    counters: &EngineCounters,
) -> Result<(vk::ImageView, Option<OwnedDepthImage>), DrawError> {
    let with_stencil = req.depth.as_ref().and_then(|d| d.stencil).is_some();
    if let Some(identity) = req.depth.as_ref().and_then(|d| d.identity.clone()) {
        let (_image, view) = pools.registry_ensure_depth(
            ctx,
            identity,
            req.width,
            req.height,
            with_stencil,
            counters,
        )?;
        return Ok((view, None));
    }
    let (dimg, dmem, dview) =
        pools.create_transient_depth(ctx, req.width, req.height, with_stencil, counters)?;
    Ok((dview, Some((dimg, dmem, dview))))
}

pub(crate) unsafe fn execute_draw_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
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
    // state for same-target successors; a successor whose work folds into the
    // open CB (LoadFromTarget — no CPU/GPU seed, not sampling its own target,
    // same identity/geometry/format) appends to it, skipping slot claim and
    // submit entirely. Every other draw claims a slot via begin_entry, which
    // flushes any open batch first (queue order = record order).
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
    let samples_own_target = req.sampled_images.iter().any(|s| {
        matches!(
            (&s.source, req.target_identity.as_ref()),
            (SampledSource::Target(t), Some(own)) if t == own
        )
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
    // What that leaves is `nojoin_no_open_batch` as the largest refusal, and
    // it is a statement about `BatchTarget` rather than about the draw — the
    // batch is keyed by target identity and geometry, so a run alternating
    // between two surfaces cannot batch at all even though each draw opens
    // and ends its own render pass inside the command buffer. Nothing read so
    // far says that key is load-bearing.
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
    let terms = JoinTerms {
        force_loss,
        quirk: ctx.caps.quirks.no_deferred_draw_batching,
        is_mrt,
        has_depth: req.depth.is_some(),
        reads_back: !req.skip_readback,
        has_query: req.occlusion_query.is_some(),
        no_identity: req.target_identity.is_none(),
        cpu_seed: req.target_rgba8.is_some(),
        gpu_seed: req.seed_from_target.is_some(),
        // Last because it is the only term that looks anything up. Evaluated
        // eagerly, which costs one `Option` compare on a draw an earlier term
        // already refused.
        no_open_batch: batch_target
            .as_ref()
            .and_then(|t| pools.batch_slot(t))
            .is_none(),
    };
    let batch_eligible = terms.batch_eligible();
    let no_join = terms.refusal();
    // The join arm splits by self-alias, and the two must sum to the joins.
    // `nojoin_samples_own_target` was 29.7 % of all draws and is the population
    // this ladder stopped refusing; without a name of its own on the way *in*,
    // the only visible effect would be a term that stopped firing, which reads
    // identically to a workload that stopped presenting one.
    crate::runtime::drain::note_store_route(no_join.unwrap_or(if samples_own_target {
        "join_appended_self_alias"
    } else {
        "join_appended"
    }));
    let joins = no_join.is_none();
    // Claim the next ring slot — BEFORE any pool acquire, so a recycled slot
    // can never alias a still-in-flight CB. Blocks (retire) only when every
    // slot is still in flight; the wait lands in retire_wait_us. A batch
    // joiner reuses the open batch's slot instead (its CB is still recording).
    // Everything above this point is bookkeeping over the request; the claim
    // below is the only part of `Prep` that can block on the GPU. Charged apart
    // so a boot can tell "the CPU is ahead of the ring" from "preparing a draw
    // got slower", which `prep_us` alone cannot.
    phase.enter(super::draw_phase::Phase::Slot);
    let (cb, fence) = if joins {
        let target = batch_target.as_ref().expect("joins requires identity");
        pools.batch_slot(target).expect("joins checked batch_slot")
    } else {
        // A fresh command buffer, so the guest-window imports the previous one
        // pinned against eviction may be displaced again. A *joiner*
        // deliberately does not bump: it records into the open batch's CB, which
        // still names every import the draws before it were handed and has not
        // been submitted, so nothing may free one out from under it.
        pools.begin_entry(ctx, counters)?
    };
    phase.enter(super::draw_phase::Phase::Pipeline);

    // Build layout key from storage / sampled / sampler bindings.
    let mut layout_bindings = Vec::new();
    for b in &req.storage_buffers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    for b in &req.sampled_images {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    for b in &req.samplers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    if req.color_input {
        layout_bindings.push(BindingSig {
            binding: super::types::COLOR_INPUT_BINDING,
            ty: vk::DescriptorType::INPUT_ATTACHMENT.as_raw() as u32,
            stages: vk::ShaderStageFlags::FRAGMENT.as_raw(),
        });
    }
    layout_bindings.sort_by_key(|b| b.binding);
    let layout_key = LayoutKey {
        bindings: layout_bindings,
    };
    // Resolve load action: load_from_target > target_rgba8 > Clear black.
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
        req.target_rgba8.as_ref().map(|v| v.as_slice())
    };
    let mut pass_key = PassKey::single(
        load_uses_gpu_content || seed_bytes.is_some() || req.seed_from_target.is_some(),
        output_bgra,
    );
    for (i, sec) in req.secondary_targets.iter().enumerate() {
        if i >= MAX_SECONDARY_ATTACH {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SecondaryAttachmentCap {
                    requested: req.secondary_targets.len(),
                    cap: MAX_SECONDARY_ATTACH,
                },
            ));
        }
        pass_key.secondary[i] = SecondaryAttachKey {
            format: sec.format,
            load: sec.load,
        };
    }
    pass_key.secondary_count = req.secondary_targets.len() as u8;
    pass_key.color_input = req.color_input;
    // Depth is opt-in per draw (only a non-trivial MTLDepthStencilState reaches
    // here). Combining it with MRT is not yet supported — the ad-hoc MRT
    // framebuffer would need the depth view appended and the pass rebuilt; no
    // known workload does both, so reject rather than silently drop depth.
    if req.depth.is_some() && is_mrt {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::DepthWithSecondaryAttachments,
        ));
    }
    if let Some(d) = &req.depth {
        pass_key.depth = Some(super::caches::DepthAttachKey {
            load: d.load,
            stencil: d.stencil.is_some(),
        });
    }
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

    let (vert_digest, vert_module) =
        caches.get_or_create_shader(ctx, &req.vert_spirv, counters, pools)?;
    let (frag_digest, frag_module) =
        caches.get_or_create_shader(ctx, &req.frag_spirv, counters, pools)?;
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let render_pass = caches.get_or_create_pass(ctx, pass_key, counters, pools)?;
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
        Some(mode) => Some(crate::backend::vulkan::translate::raster::vk_query_control_flags(mode)),
    };
    let pipeline_key = PipelineKey {
        vert: vert_digest,
        frag: frag_digest,
        attrs: attr_keys,
        topology: req.primitive_topology,
        blend: req.blend.map(|b| b.key()),
        secondary_blend: {
            let mut per_slot = [None; MAX_SECONDARY_ATTACH];
            for (slot, target) in req
                .secondary_targets
                .iter()
                .take(MAX_SECONDARY_ATTACH)
                .enumerate()
            {
                per_slot[slot] = target.blend.map(|b| b.key());
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
        pass: pass_key,
        cull_mode: req.cull_mode,
        front_face_ccw: req.front_face_ccw,
        fill_mode: req.fill_mode,
        depth_clip: req.depth_clip,
        depth_test: req.depth.as_ref().map(|d| d.test_enable).unwrap_or(false),
        depth_write: req.depth.as_ref().map(|d| d.write_enable).unwrap_or(false),
        depth_compare: req
            .depth
            .as_ref()
            .map(|d| d.compare)
            .unwrap_or(super::types::SamplerCompareFunction::Always),
        stencil: req
            .depth
            .as_ref()
            .and_then(|d| d.stencil)
            .map(|s| super::caches::StencilKey {
                front: s.front,
                back: s.back,
            }),
        viewport_slots: slot_count_u32,
        layout: layout_key.clone(),
    };
    // One cache, consulted once. `get_or_create_pipeline` already counts the hit
    // and already checks the negative entry for a key that failed to compile.
    let pipeline = caches.get_or_create_pipeline(
        ctx,
        &pipeline_key,
        vert_module,
        &req.vert_spirv,
        frag_module,
        pipeline_layout,
        render_pass,
        counters,
        pools,
    )?;

    // Samplers
    let mut sampler_handles = Vec::new();
    for s in &req.samplers {
        let h = caches.get_or_create_sampler(ctx, &s.state_key(), counters, pools)?;
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
                vk::BufferUsageFlags::VERTEX_BUFFER,
                batch_eligible,
                &mut guest_gathers,
            )?
        };
        vertex_bufs.push((resource.binding, slot));
    }

    // Index buffer
    let mut index_slot = None;
    if let Some(indexed) = &req.indexed {
        let slot = {
            let _s = stage_phase::Span::open(stage_phase::Part::Acquire);
            pools.acquire_staging(
                ctx,
                indexed.indices.len() as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                counters,
            )?
        };
        let _s = stage_phase::Span::moving(stage_phase::Part::Bytes, indexed.indices.len() as u64);
        pools.write_staging(ctx, &slot, &indexed.indices)?;
        drop(_s);
        index_slot = Some(slot);
    }

    // Storage buffers (deduplicated by content with the vertex streams: a
    // stage-in buffer doubling as a storage bind reuses the same slot —
    // staging slots always carry the full usage superset).
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let slot = stage_buffer_content(
            ctx,
            pools,
            counters,
            &resource.content,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            batch_eligible,
            &mut guest_gathers,
        )?;
        storage_slots.push((resource.binding, slot));
    }

    // Target seed staging (CPU import only — not LoadFromTarget).
    let seed_slot = if let Some(rgba8) = seed_bytes {
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
    let primary_pass = if is_mrt || req.depth.is_some() || req.color_input {
        caches.get_or_create_pass(
            ctx,
            PassKey::single(pass_key.load_seed, pass_key.bgra),
            counters,
            pools,
        )?
    } else {
        render_pass
    };
    phase.enter(super::draw_phase::Phase::Acquire);
    // (identity, image, tracked-layout-before-this-draw) per secondary — used
    // to barrier prior sampled reads and to mark ready afterward.
    let mut mrt_secondaries: Vec<(
        super::types::TargetIdentity,
        vk::Image,
        super::pools::ResidentAccess,
    )> =
        Vec::new();
    // This draw's depth attachment, when it has one. The framebuffer is always
    // this draw's own and is always disposed after submit; the image behind it
    // is only owned here when the pass named no guest depth texture to key a
    // resident on — see `acquire_depth_view`. `None` on the 2D path so nothing
    // changes there.
    let mut transient_depth: Option<(Option<OwnedDepthImage>, vk::Framebuffer)> = None;
    // Mark everything this draw is about to read *before* resolving its own
    // target, so a reclaim between here and `prepare_sampled` cannot take one
    // of this draw's own sampled sources.
    //
    // The idle drain is what could: it destroys any resident untouched for
    // `IDLE_TARGET_AGE_MS`, and it runs off the poll heartbeat on another
    // thread, so a source last read a while ago is reachable right up to the
    // lookup a few hundred lines below — the gap between the
    // `resident_content_ready` guard the resolver already performs and
    // `prepare_sampled`. Marking them used closes it, because both reclaim
    // paths treat a marked resident as in use, and a source this draw is about
    // to read is by construction the most recently used thing in the registry.
    // It reuses the recency the sampled resolve already records rather than
    // threading a protected set through `registry_ensure`.
    for s in &req.sampled_images {
        if let SampledSource::Target(identity) = &s.source {
            pools.registry_note_sampled_use(identity);
        }
    }
    let (target_image, target_fb, target_access, target_view) =
        if let Some(identity) = &req.target_identity {
            let gen = identity.generation();
            let t = pools.registry_ensure(
                ctx,
                identity.clone(),
                req.width,
                req.height,
                primary_pass,
                gen,
                output_bgra,
                counters,
            )?;
            if load_uses_gpu_content && !t.content_ready {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::LoadTargetContentNotReady {
                        identity: identity.clone(),
                    },
                ));
            }
            let primary_image = t.image;
            let primary_view = t.view;
            let primary_access = t.access;
            let primary_slot_fb = t.framebuffer;
            if is_mrt {
                // Ensure each secondary resident and collect its view for the MRT
                // framebuffer. Recently-ensured residents sit at the back of the
                // LRU order, so a later secondary's capacity sweep (front-first)
                // cannot evict the primary or an earlier secondary in this draw.
                let mut views = vec![primary_view];
                for sec in &req.secondary_targets {
                    let old_access = pools
                        .registry_get(&sec.identity)
                        .map(|s| s.access)
                        .unwrap_or(super::pools::ResidentAccess::Untouched);
                    let (img, view) = pools.registry_ensure_attachment(
                        ctx,
                        sec.identity.clone(),
                        sec.width,
                        sec.height,
                        sec.identity.generation(),
                        sec.format,
                        counters,
                    )?;
                    views.push(view);
                    mrt_secondaries.push((sec.identity.clone(), img, old_access));
                }
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &views,
                    req.width,
                    req.height,
                    counters,
                )?;
                (primary_image, fb, primary_access, primary_view)
            } else if req.depth.is_some() {
                let (dview, owned) = acquire_depth_view(ctx, pools, req, counters)?;
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[primary_view, dview],
                    req.width,
                    req.height,
                    counters,
                )?;
                transient_depth = Some((owned, fb));
                (primary_image, fb, primary_access, primary_view)
            } else if req.color_input {
                // Fetch pass carries an input reference → the slot's cached
                // color-only framebuffer is incompatible; build an ad-hoc one
                // against `render_pass` (disposed deferred after submit).
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[primary_view],
                    req.width,
                    req.height,
                    counters,
                )?;
                (primary_image, fb, primary_access, primary_view)
            } else {
                (primary_image, primary_slot_fb, primary_access, primary_view)
            }
        } else {
            let target_key = TargetKey {
                width: req.width,
                height: req.height,
                with_transfer_dst: seed_bytes.is_some(),
            };
            // Acquire the pooled slot under the color-only `primary_pass` (same as
            // its cached framebuffer). For a depth draw, build a fresh ad-hoc
            // framebuffer [color, depth] under the depth `render_pass`.
            let t = pools.acquire_target(ctx, target_key, primary_pass, counters)?;
            let (pool_image, pool_view, pool_fb) = (t.image, t.view, t.framebuffer);
            if req.depth.is_some() {
                let (dview, owned) = acquire_depth_view(ctx, pools, req, counters)?;
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[pool_view, dview],
                    req.width,
                    req.height,
                    counters,
                )?;
                transient_depth = Some((owned, fb));
                (pool_image, fb, super::pools::ResidentAccess::Untouched, pool_view)
            } else if req.color_input {
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[pool_view],
                    req.width,
                    req.height,
                    counters,
                )?;
                (pool_image, fb, super::pools::ResidentAccess::Untouched, pool_view)
            } else {
                (pool_image, pool_fb, super::pools::ResidentAccess::Untouched, pool_view)
            }
        };
    // GPU seed source: resolved after registry_ensure (which protects it from
    // the capacity sweep) so the handle cannot be destroyed under this draw.
    // Every rejection is a distinct named error — the runtime pre-checks
    // readiness, so these only fire on a runtime/protocol bug.
    let seed_from_resolved: Option<(vk::Image, super::pools::ResidentAccess)> =
        if let Some(seed_identity) = &req.seed_from_target {
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
            Some((slot.image, slot.access))
        } else {
            None
        };

    // Resolve sampled images only after ensuring the render target so registry
    // capacity eviction cannot destroy an image already selected for this draw.
    phase.enter(super::draw_phase::Phase::AcquireSampled);
    let mut sampled = Vec::new();
    for resource in &req.sampled_images {
        match &resource.source {
            SampledSource::Bytes(bytes) => {
                if let Some(image) = pools.find_cached_sampled(
                    SampledKey::of(resource),
                    bytes,
                    resource.identity,
                    counters,
                ) {
                    sampled.push(PreparedSampled::Cached {
                        binding: resource.binding,
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
                counters.note_sampled_reupload(bytes.len() as u64);
                sampled.push(PreparedSampled::Upload {
                    binding: resource.binding,
                    image: img,
                    staging: st,
                    volume: resource.volume,
                    layers: resource.layers,
                });
            }
            SampledSource::Target(identity) => {
                // Reading a resident is using it. Marked before the lookup so
                // the refusal paths below cannot skip it: a resident whose
                // content is not ready yet, or whose geometry disagrees with
                // this bind, is still one the guest is actively sampling, and
                // aging it out between two attempts is how a recoverable
                // not-ready became a permanent missing.
                pools.registry_note_sampled_use(identity);
                let (source_image, source_view, source_layout, source_bgra, source_ready, sw, sh) =
                    pools
                        .registry_get(identity)
                        .map(|slot| {
                            (
                                slot.image,
                                slot.view,
                                slot.access,
                                slot.scanout_order(),
                                slot.content_ready,
                                slot.width,
                                slot.height,
                            )
                        })
                        .ok_or_else(|| {
                            DrawError::DrawExecution(DrawExecutionDecline::SampledResidentMissing {
                                binding: resource.binding,
                                identity: identity.clone(),
                                prior: pools.prior_reclaim(identity),
                            })
                        })?;
                if !source_ready {
                    return Err(DrawError::DrawExecution(
                        DrawExecutionDecline::SampledResidentNotReady {
                            binding: resource.binding,
                            identity: identity.clone(),
                        },
                    ));
                }
                if sw != resource.width || sh != resource.height {
                    return Err(DrawError::DrawExecution(
                        DrawExecutionDecline::SampledResidentGeometryMismatch {
                            binding: resource.binding,
                            identity: identity.clone(),
                            resident_width: sw,
                            resident_height: sh,
                            resource_width: resource.width,
                            resource_height: resource.height,
                        },
                    ));
                }
                if req.target_identity.as_ref() == Some(identity) {
                    let image = pools.acquire_sampled(
                        ctx,
                        SampledKey {
                            // The snapshot binds the *resident's* format, not
                            // the one the binding declared.
                            format: super::super::translate::pixel::resident_color(source_bgra),
                            ..SampledKey::of(resource)
                        },
                        counters,
                    )?;
                    sampled.push(PreparedSampled::Snapshot {
                        binding: resource.binding,
                        identity: identity.clone(),
                        source_image,
                        source_access: source_layout,
                        image,
                    });
                } else {
                    sampled.push(PreparedSampled::Resident {
                        binding: resource.binding,
                        identity: identity.clone(),
                        image: source_image,
                        view: source_view,
                        access: source_layout,
                    });
                }
                counters
                    .sampled_gpu_binds
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            SampledSource::GuestRuns(src, vouch) => {
                // The producer vouches for this identity only when both halves
                // of the guest-write witness say the window's bytes cannot have
                // moved since the gather that filled the retained image: no
                // guest store into the pages, and no write by this device
                // either. So the retained image is bound with nothing read and
                // nothing compared — which is the whole point, since reading
                // the bytes to compare them is the cost being removed.
                if let Some(image) = pools.find_gathered_sampled(
                    SampledKey::of(resource),
                    resource.identity,
                    counters,
                ) {
                    counters.note_sampled_gather_skipped(src.total_len);
                    sampled.push(PreparedSampled::Cached {
                        binding: resource.binding,
                        image,
                    });
                    continue;
                }
                // The elision did not fire, and the two reasons it can fail want
                // opposite fixes: no vouch to spend, or a vouch with nothing
                // left to spend it on. Taken here because this is the only point
                // holding both the witness's answer and the cache's.
                //
                // The witness's answer is `vouch` and never `resource.identity`.
                // Asking the identity was the same question the *producer* had
                // already answered structurally — it names every window it is
                // asked about — so the "no vouch" half read zero on every bind
                // of every boot.
                counters.note_sampled_gather_unskipped(*vouch);
                let img = pools.acquire_sampled(ctx, SampledKey::of(resource), counters)?;
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
                    import_sampled_guest_window(ctx, pools, counters, src, &mut guest_gathers)?
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
                        pools.write_staging_from_runs(ctx, &scratch, &src.runs, src.total_len)?;
                        // The only arm of this loop that moves bytes, and until
                        // now the only one that reported nothing — which is what
                        // let the whole of `acquire_sampled` sit unattributed.
                        counters.note_sampled_gather(src.total_len);
                        GuestTexels::Scratch(scratch)
                    }
                };
                sampled.push(PreparedSampled::GuestGather {
                    binding: resource.binding,
                    image: img,
                    source,
                    row_length_texels: src.row_length_texels,
                    gathered_len: src.total_len as usize,
                });
                // Back to the deciding half for the next texture in the loop.
                phase.enter(super::draw_phase::Phase::AcquireSampled);
            }
        }
    }

    phase.enter(super::draw_phase::Phase::AcquireReadback);
    let rb_size = (req.width as u64) * (req.height as u64) * 4;
    let do_readback = !req.skip_readback;
    phase.note_target(req.width, req.height, if do_readback { rb_size } else { 0 });
    let readback = if do_readback {
        Some(pools.acquire_readback(ctx, rb_size, counters)?)
    } else {
        None
    };

    phase.enter(super::draw_phase::Phase::Descriptors);
    // Descriptor set
    // Owning pool block travels alongside the set so the flush-time free routes
    // back to the block it was allocated from (arena may grow past block 0).
    let mut dset_pool: Option<vk::DescriptorPool> = None;
    let dset = if dsl != vk::DescriptorSetLayout::null() {
        let (dset, pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;
        dset_pool = Some(pool);
        let buffer_infos: Vec<_> = storage_slots
            .iter()
            .map(|(_, bound)| {
                // `WHOLE_SIZE` is the rest of the buffer from `offset`. For a
                // pooled slot that is the slot; for a guest window import it is
                // the remainder of the guest's own pages, which the shader is
                // already entitled to and which its own bounds keep it inside.
                vk::DescriptorBufferInfo::default()
                    .buffer(bound.buffer)
                    .offset(bound.offset)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let sampled_infos: Vec<_> = sampled
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let sampler_infos: Vec<_> = sampler_handles
            .iter()
            .map(|(_, s)| vk::DescriptorImageInfo::default().sampler(*s))
            .collect();
        // Framebuffer fetch: the input attachment IS the color target's view;
        // GENERAL matches the subpass references (see `get_or_create_pass`).
        let color_input_info = vk::DescriptorImageInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut writes = Vec::new();
        for (i, (binding, _)) in storage_slots.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i])),
            );
        }
        for (i, image) in sampled.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(image.binding())
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&sampled_infos[i])),
            );
        }
        for (i, (binding, _)) in sampler_handles.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_infos[i])),
            );
        }
        if req.color_input {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(super::types::COLOR_INPUT_BINDING)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(std::slice::from_ref(&color_input_info)),
            );
        }
        ctx.device.update_descriptor_sets(&writes, &[]);
        Some(dset)
    } else {
        None
    };

    phase.enter(super::draw_phase::Phase::Record);
    // The ring slot's CB retired at begin_entry and its fence is unsignaled —
    // no pre-record wait remains (pre_record_wait_us stays 0 on this path).
    // A batch joiner's CB is already recording (opened by the batch opener);
    // its commands append after the previous draw's end_render_pass.
    if !joins {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecBeginCb, e)))?;
    }

    // Metal permits a pass to sample the same texture it renders into. Vulkan
    // does not permit that attachment feedback loop on this path, so capture
    // the prior resident content into a same-format GPU image before changing
    // the attachment. This preserves the old CPU snapshot semantics without a
    // readback or host upload.
    let mut snapshotted_targets = std::collections::HashSet::new();
    let mut target_snapshotted = false;
    for sampled_image in &sampled {
        let PreparedSampled::Snapshot {
            identity,
            source_image,
            source_access,
            image,
            ..
        } = sampled_image
        else {
            continue;
        };
        target_snapshotted = true;
        // Once per distinct source: duplicate bindings of one target share the
        // image, so a second barrier for it would order nothing new. The
        // *first* is unconditional — the source is a registry resident this
        // draw's own predecessor may have written, and the layout it sits in
        // says nothing about that.
        if snapshotted_targets.insert(identity.clone()) {
            barrier_resident_for_transfer_read(&ctx.device, cb, *source_image, *source_access);
        }
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
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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

    // Seed upload (CPU import).
    if let Some(seed) = &seed_slot {
        let (src_stage, src_access) = target_prior_access(target_snapshotted, target_access).source_scope();
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
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    } else if let Some((seed_image, seed_access)) = seed_from_resolved {
        // GPU present-boundary seed: resident front frame → draw target copy,
        // then the pass runs with LOAD.
        //
        // The source is a resident that a draw just produced, so it is normally
        // already in TRANSFER_SRC_OPTIMAL and gating on a transition being
        // needed skipped the dependency on exactly the frames worth copying.
        barrier_resident_for_transfer_read(&ctx.device, cb, seed_image, seed_access);
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
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
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
    } else if load_uses_gpu_content {
        // A prior direct sample may have left this target shader-readable;
        // transition from the registry's tracked layout back to attachment use.
        let prior = target_prior_access(target_snapshotted, target_access);
        let (src_stage, src_access) = prior.source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(prior.layout())
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    } else if target_snapshotted || target_access != super::pools::ResidentAccess::Untouched {
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
        let (src_stage, src_access) = target_prior_access(target_snapshotted, target_access).source_scope();
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

    // Resident samples: transition the persistent target in place. Duplicate
    // bindings of one target share the same image and therefore one barrier.
    let mut transitioned_resident = std::collections::HashSet::new();
    for image in &sampled {
        let PreparedSampled::Resident {
            identity,
            image,
            access,
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
        if !transitioned_resident.insert(identity.clone())
            || *access == super::pools::ResidentAccess::ShaderRead
        {
            continue;
        }
        let (src_stage, src_access) = access.source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(access.layout())
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(*image)
            .subresource_range(super::color_subresource_range())];
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

    // CPU-origin sampled uploads.
    for image in &sampled {
        let PreparedSampled::Upload {
            image: img,
            staging: st,
            volume,
            layers,
            ..
        } = image
        else {
            continue;
        };
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
        );
    }

    // Scattered guest buffer windows, assembled into their device-local
    // destinations before anything reads them.
    //
    // No HOST→TRANSFER barrier: the source is the guest's own RAM through the
    // RAMBlock import and nothing in this process wrote it, and the destination
    // is device-local memory only the GPU touches. What the copies *do* owe is
    // a barrier before the draw reads them, which follows the loop — one for
    // all of them, because a per-buffer barrier would submit N of them to
    // express the same dependency.
    for gather in &guest_gathers {
        for (source, copies) in &gather.sources {
            ctx.device.cmd_copy_buffer(cb, *source, gather.dst, copies);
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
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                    | vk::AccessFlags::INDEX_READ
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ,
            )];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    // Guest-sourced sampled uploads: one buffer→image copy over either the
    // guest's imported pages or the scratch the CPU packed them into, differing
    // from the CPU-origin loop above only in `row_length_texels` striding over
    // guest row padding (0 = tight rows) and in the copy's `bufferOffset`.
    //
    // No HOST→TRANSFER barrier on either arm. For a scratch, the reason is the
    // one the loop above relies on: host writes made before `vkQueueSubmit` are
    // automatically visible to the device. For an import there is no host write
    // at all — the bytes are the guest's, already in memory the device reads
    // through the fd, and nothing in this process touched them.
    for image in &sampled {
        let PreparedSampled::GuestGather {
            image: img,
            source,
            row_length_texels,
            ..
        } = image
        else {
            continue;
        };
        upload_buffer_to_sampled_image(
            ctx,
            cb,
            source.buffer(),
            source.offset(),
            img.image,
            img.width,
            img.height,
            1,
            1,
            *row_length_texels,
        );
    }

    // MRT secondary attachments that were left shader-readable (sampled by a
    // prior draw) must transition back to color-attachment use, and the write
    // must wait for that prior read (WAR). A freshly-created secondary tracks
    // UNDEFINED and needs no barrier — the render pass discards on CLEAR.
    for (_id, image, access) in &mrt_secondaries {
        if *access == super::pools::ResidentAccess::Untouched {
            continue;
        }
        let (src_stage, src_access) = access.source_scope();
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(access.layout())
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(*image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    let clear = clear_values(req);
    let rp_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(target_fb)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: req.width,
                height: req.height,
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
            ctx.device.cmd_reset_query_pool(cb, pool, 0, 1);
            Some((pool, flags))
        }
    };
    ctx.device
        .cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
    ctx.device
        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
    if let Some((pool, flags)) = occlusion {
        ctx.device.cmd_begin_query(cb, pool, 0, flags);
    }

    // Dynamic viewport/scissor. Metal NDC is Y-up and Vulkan's is Y-down, so
    // every viewport is emitted flipped: origin at the bottom edge, negative
    // height. This is a property of the two APIs, not of any guest state.
    let default_vp = ViewportResource {
        x: 0.0,
        y: 0.0,
        width: req.width as f32,
        height: req.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let default_sc = ScissorResource {
        x: 0,
        y: 0,
        width: req.width,
        height: req.height,
    };
    // One count for both, because a Vulkan pipeline declares one and the
    // dynamic arrays must match it. The pipeline was built from
    // `viewport_slot_count`, so this must be the same function of the same
    // request or `vkCmdSetViewport` binds a different count than the pipeline
    // declared.
    let slots = crate::backend::vulkan::engine::viewport_slot_count(req);
    let viewports: Vec<vk::Viewport> = (0..slots)
        .map(|i| {
            let vp = req.viewports.get(i).copied().unwrap_or(default_vp);
            vk::Viewport {
                x: vp.x,
                y: vp.y + vp.height,
                width: vp.width,
                height: -vp.height,
                min_depth: vp.min_depth,
                max_depth: vp.max_depth,
            }
        })
        .collect();
    ctx.device.cmd_set_viewport(cb, 0, &viewports);
    let scissors: Vec<vk::Rect2D> = (0..slots)
        .map(|i| {
            let sc = req.scissors.get(i).copied().unwrap_or(default_sc);
            let x = sc.x.min(req.width);
            let y = sc.y.min(req.height);
            vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: sc.width.min(req.width - x),
                    height: sc.height.min(req.height - y),
                },
            }
        })
        .collect();
    ctx.device.cmd_set_scissor(cb, 0, &scissors);
    // Dynamic stencil reference (Metal `setStencilFrontReferenceValue:back…`)
    // — only bound for stencil pipelines, which list STENCIL_REFERENCE as a
    // dynamic state; front/back set separately to honor Metal's split refs.
    if let Some(s) = req.depth.as_ref().and_then(|d| d.stencil) {
        ctx.device
            .cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT, s.reference_front);
        ctx.device
            .cmd_set_stencil_reference(cb, vk::StencilFaceFlags::BACK, s.reference_back);
    }

    if let Some(dset) = dset {
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[dset],
            &[],
        );
    }
    for (binding, bound) in &vertex_bufs {
        ctx.device
            .cmd_bind_vertex_buffers(cb, *binding, &[bound.buffer], &[bound.offset]);
    }
    match (&req.indexed, &index_slot) {
        (Some(indexed), Some(ibuf)) => {
            ctx.device
                .cmd_bind_index_buffer(cb, ibuf.buffer, 0, indexed.index_type.vk());
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
    if let Some((pool, _)) = occlusion {
        ctx.device.cmd_end_query(cb, pool, 0);
    }
    ctx.device.cmd_end_render_pass(cb);

    if let Some(ref rb) = readback {
        // The pass resolved the colour attachment to TRANSFER_SRC_OPTIMAL, so
        // this copy needs no transition — but it does need a dependency, and
        // the render pass does not give it one. Vulkan's implicit final subpass
        // dependency carries `dstStageMask = BOTTOM_OF_PIPE` and
        // `dstAccessMask = 0`: it makes the colour writes available and visible
        // to nothing. Recording the copy into the same command buffer is not a
        // dependency either — commands in one buffer are free to overlap.
        //
        // Without this the readback can sample the attachment before the draw
        // it was recorded after has finished writing it, and the bytes handed
        // back are the ones from before the draw.
        let flush_writes = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &flush_writes,
            &[],
            &[],
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
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            rb.buffer,
            &region,
        );
    }
    // A batch-eligible draw defers end_command_buffer + submit: its CB stays
    // in recording state for same-target successors and is submitted by
    // pools.batch_flush (next begin_entry / retire / explicit flush).
    let defer_submit = batch_eligible;
    phase.enter(super::draw_phase::Phase::Submit);
    if !defer_submit {
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecEndCb, e)))?;
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

    if !defer_submit {
        let queue = ctx.queue();
        let cbs = [cb];
        let si = vk::SubmitInfo::default().command_buffers(&cbs);
        match ctx.device.queue_submit(queue, &[si], fence) {
            Ok(()) => {}
            Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
                return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                    op: DeviceLostOp::DrawSubmit,
                    result: e,
                }));
            }
            Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::ExecSubmit, e))),
        }
    }
    // CPU-side bookkeeping: the retained target's content is queue-ordered
    // (mark ready), resident sampled layouts advance to the recorded
    // post-draw layout, and upload-path sampled bytes queue for cache
    // admission at retire time.
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
        let rewrites_whole_attachment =
            !load_uses_gpu_content || seed_bytes.is_some() || req.seed_from_target.is_some();
        // `any`, not the union: one scissor reaching the whole attachment is
        // enough for this draw to have written anywhere in it. A set of rects
        // that only covers the target *together* reads as partial here, which
        // over-states how much a damage-bounded flush could save rather than
        // under-stating it — the safe direction for an instrument nothing acts
        // on, and cheaper than computing a union of arbitrary rects.
        counters.note_draw_coverage(if rewrites_whole_attachment {
            super::counters::DrawCoverage::Full
        } else if scissors.iter().any(|s| {
            s.offset.x <= 0
                && s.offset.y <= 0
                && s.extent.width >= req.width
                && s.extent.height >= req.height
        }) {
            super::counters::DrawCoverage::LoadedFullScissor
        } else {
            super::counters::DrawCoverage::LoadedPartialScissor
        });
        pools.registry_mark_ready(identity);
    }
    // MRT secondary attachments settle at COLOR_ATTACHMENT_OPTIMAL (the pass
    // final layout) and become sampleable residents; the consumer's
    // resident-sample barrier then transitions COLOR_ATTACHMENT→SHADER_READ,
    // carrying the color-write→shader-read dependency. (The ad-hoc MRT
    // framebuffer is disposed below, after `finish_entry_async` — see there.)
    if is_mrt {
        for (identity, _image, _old) in &mrt_secondaries {
            pools.registry_mark_ready_at(identity, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
    }
    let mut sampled_retains: Vec<super::pools::SampledRetain> = Vec::new();
    for prepared in &sampled {
        match prepared {
            PreparedSampled::Upload { binding, image, .. } => {
                if let Some((SampledSource::Bytes(bytes), identity)) = req
                    .sampled_images
                    .iter()
                    .find(|resource| resource.binding == *binding)
                    .map(|resource| (&resource.source, resource.identity))
                {
                    sampled_retains.push(super::pools::SampledRetain {
                        image: image.image,
                        content: super::pools::SampledRetainContent::Bytes(bytes.clone()),
                        identity,
                    });
                }
            }
            // A gather with no vouched identity is dropped by the admit, which
            // is where that decision belongs: an entry nothing can name is
            // unreachable weight in a capped cache.
            PreparedSampled::GuestGather {
                binding,
                image,
                gathered_len,
                ..
            } => {
                let identity = req
                    .sampled_images
                    .iter()
                    .find(|resource| resource.binding == *binding)
                    .and_then(|resource| resource.identity);
                sampled_retains.push(super::pools::SampledRetain {
                    image: image.image,
                    content: super::pools::SampledRetainContent::Gathered { len: *gathered_len },
                    identity,
                });
            }
            _ => {}
        }
    }
    for image in &sampled {
        if let PreparedSampled::Resident { identity, .. } = image {
            pools.registry_note_access(identity, super::pools::ResidentAccess::ShaderRead);
        }
    }
    if seed_from_resolved.is_some() {
        if let Some(seed_identity) = &req.seed_from_target {
            pools.registry_note_access(seed_identity, super::pools::ResidentAccess::TransferRead);
        }
    }
    // Deferred-submit draw: park the per-draw descriptor set and sampled
    // admissions on the open batch (opening it if this is the first) and
    // return. The CPU-side bookkeeping above already ran — content_ready and
    // tracked layouts describe what the recorded CB produces, and every
    // consumer path flushes the batch before touching the GPU.
    if defer_submit {
        let target = batch_target.expect("batch_eligible requires target identity");
        pools.batch_append(
            cb,
            fence,
            target,
            dset.zip(dset_pool),
            sampled_retains,
            counters,
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
        });
    }

    // Park the owed cleanup (descriptor set, transient pool slots, cache
    // admissions) on this ring slot in every mode; whichever entry retires
    // the slot drains it. A failed wait below leaves the slot pending, so no
    // path ever reuses an unretired fence.
    let cleanup = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), sampled_retains);
    pools.finish_entry_async(cleanup);

    // Dispose the ad-hoc per-draw framebuffers (MRT and/or depth) now that
    // `finish_entry_async` has marked this slot pending: the handles park in
    // the graveyard against the slots open right now — this draw's included —
    // and are freed once those retire. Disposing BEFORE this point would
    // immediate-free them (this slot is not yet pending, so it is not in the
    // open mask) while the just-submitted CB still references them → GPU fault.
    if is_mrt || (req.color_input && transient_depth.is_none()) {
        pools.dispose(
            &ctx.device,
            super::pools::DeferredHandle::Framebuffer(target_fb),
        );
    }
    if let Some((owned, dfb)) = transient_depth {
        pools.dispose(&ctx.device, super::pools::DeferredHandle::Framebuffer(dfb));
        // Only the unidentified case owns its image. A resident one belongs to
        // the registry and to the guest texture it is keyed on; disposing it
        // here would put the rail straight back to one allocation per draw, with
        // the added defect that the next draw would find a destroyed handle.
        if let Some((dimg, dmem, dview)) = owned {
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
    // neither defer rail can take still has to land its pixels: a type-11 Store
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
            });
        }
        counters
            .render_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(DrawOutput {
            pixels: Vec::new(),
            pixels_bgra: output_bgra,
            occlusion_samples: None,
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
    counters.note_readback(rb_size);

    Ok(DrawOutput {
        pixels: out,
        pixels_bgra: output_bgra,
        occlusion_samples: read_occlusion_samples(ctx, occlusion)?,
    })
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
        super::pools::ResidentAccess::TransferRead
    } else {
        tracked
    }
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
/// [`crate::runtime::drain::note_store_route`].
pub(super) unsafe fn barrier_resident_for_transfer_read(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    access: super::pools::ResidentAccess,
) {
    let (src_stage, src_access) = access.source_scope();
    let barrier = [vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(access.layout())
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
    use crate::backend::vulkan::engine::pools::ResidentAccess;
    use crate::backend::vulkan::engine::types::{
        GuestRun, GuestRunSource, SampledImageResource, SampledSource,
    };
    use crate::observe::Decline;

    /// Build a `JoinTerms` from a bitmask, one bit per ladder rung in order.
    fn join_terms(bits: u32) -> JoinTerms {
        let b = |i: usize| bits & (1 << i) != 0;
        JoinTerms {
            force_loss: b(0),
            quirk: b(1),
            is_mrt: b(2),
            has_depth: b(3),
            reads_back: b(4),
            has_query: b(5),
            no_identity: b(6),
            cpu_seed: b(7),
            gpu_seed: b(8),
            no_open_batch: b(9),
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

    // There is deliberately no test that `batch_eligible` and `refusal` agree
    // about which rungs gate opening a batch. Both read `JoinScope` off the same
    // ladder entry, so any such test asserts the scope field against itself:
    // mis-scoping `nojoin_query` to `Fit` — which would let a queried draw defer
    // its submit and hand the guest a count its command buffer has not produced
    // — leaves one green. That test was written, run against exactly that
    // mutation, and deleted. The `debug_assert` it replaced had the same blind
    // spot and a release-build hole besides. What makes the two agree is that
    // there is one list.

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
    fn window_runs(
        stretches: &[(u64, u64, u64)],
    ) -> Vec<crate::runtime::guest_ram_map::GuestWindowRun> {
        use crate::runtime::guest_ram::{GuestRamImport, GuestRamRegion, GuestRef};
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
                |&(window_offset, offset, len)| crate::runtime::guest_ram_map::GuestWindowRun {
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
        assert!(single_run(&window_runs(&[(0, 0, 64)])).is_some());
        assert!(
            single_run(&window_runs(&[(16, 16, 48)])).is_none(),
            "a lone run starting past window byte zero is a suffix, not a window"
        );
        assert!(
            single_run(&window_runs(&[(0, 0, 32), (32, 4096, 32)])).is_none(),
            "two stretches are two ranges and a bind names one"
        );
        assert!(single_run(&window_runs(&[])).is_none());
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
        use crate::runtime::guest_ram::{GuestRamImport, GuestRamRegion, GuestRef};
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

        let run = crate::runtime::guest_ram_map::GuestWindowRun {
            window_offset: 512,
            guest: GuestRef::new(std::sync::Arc::clone(&import), slice)
                .expect("the slice came from this import"),
        };
        let bound = super::super::host_ram::BoundGuestRam {
            buffer: {
                use ash::vk::Handle;
                vk::Buffer::from_raw(0x99)
            },
            offset: 4096,
            len: 4096,
            head: 24,
        };
        let copy = gather_region(&bound, &run);
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

    fn guest_run_req(w: u32, h: u32, total_len: u64, row_length_texels: u32) -> DrawRequest {
        DrawRequest {
            width: w,
            height: h,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
            sampled_images: vec![SampledImageResource {
                binding: 32,
                width: w,
                height: h,
                layers: 1,
                arrayed: false,
                volume: false,
                cube: false,
                one_dim: false,
                source: SampledSource::GuestRuns(GuestRunSource {
                    runs: std::sync::Arc::new(vec![GuestRun {
                        host_ptr: 0x1000,
                        len: total_len,
                    }]),
                    total_len,
                    row_length_texels,
                    // A fixture over a dummy host address names no guest
                    // RAM, so there is no reference an import could bind.
                    pages: None,
                },
                // No witness ran for a synthetic source, so nothing vouches:
                // the gather is the only disposition this fixture can take.
                crate::runtime::gather_witness::GatherVouch::Fresh,
                ),
                format: crate::backend::vulkan::translate::pixel::vk_texel_layout(
                    crate::contract::pixel_format::TexelLayout::Bgra8,
                ),
                identity: None,
                swizzle: Default::default(),
            }],
            ..DrawRequest::default()
        }
    }

    /// Every variant a resident can be in, so the tests below enumerate rather
    /// than sample. A new variant that nothing here mentions fails to compile,
    /// which is the point: each one is a rail that can leave a resident in a
    /// state some barrier has to name.
    const EVERY_ACCESS: [ResidentAccess; 5] = [
        ResidentAccess::Untouched,
        ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        ResidentAccess::ColorWrite(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ResidentAccess::ShaderRead,
        ResidentAccess::TransferRead,
    ];

    /// The invariant the whole type exists for: **where a resident sits does not
    /// tell you what a barrier over it must wait for.**
    ///
    /// A render pass resolves its primary attachment to `TRANSFER_SRC_OPTIMAL`
    /// through `final_layout` with no transfer having run, so a resident in that
    /// layout was last written by a colour attachment write — while a resident a
    /// present blit or readback just read sits in the *same* layout after a
    /// transfer read. Two different dependencies, one layout.
    ///
    /// Anyone who re-derives a source scope from `layout()` — which is the bug
    /// this replaced, five times over — makes these two agree and fails here.
    #[test]
    fn one_resident_layout_carries_two_different_dependencies() {
        let drawn = ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let read_back = ResidentAccess::TransferRead;
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
        let (stage, access) = ResidentAccess::ShaderRead.source_scope();
        assert!(
            stage.contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "the write must be ordered after the sampling fragment shader, got {stage:?}"
        );
        assert!(!stage.contains(vk::PipelineStageFlags::TOP_OF_PIPE));
        assert_eq!(access, vk::AccessFlags::SHADER_READ);
    }

    /// A fresh registry slot and every pooled target are untouched. Nothing has
    /// happened to the image, so there is genuinely nothing to wait for — and it
    /// is the *only* state of which that is true, which is what the clear path's
    /// skip rests on. Enumerating the rest is the check; restating the call
    /// site's condition would pass whichever way the skip went.
    #[test]
    fn untouched_is_the_only_state_with_no_prior_access() {
        for access in EVERY_ACCESS {
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
                assert!(!flags.is_empty(), "{access:?} has an access to make available");
            }
        }
    }

    /// A secondary colour attachment that clears to `clear`; every other field
    /// is irrelevant to the clear-value vector and takes a neutral value.
    fn secondary_with_clear(clear: [f32; 4]) -> super::super::types::SecondaryColorTarget {
        super::super::types::SecondaryColorTarget {
            identity: super::super::types::TargetIdentity::Surface {
                id: 1,
                width: 16,
                height: 16,
                generation: 1,
            },
            width: 16,
            height: 16,
            format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
            clear,
            load: false,
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
        for tracked in EVERY_ACCESS {
            assert_eq!(
                target_prior_access(true, tracked),
                ResidentAccess::TransferRead,
                "a snapshot of a {tracked:?} target is still the newest touch"
            );
            assert_eq!(target_prior_access(false, tracked), tracked);
        }
    }

    #[test]
    fn guest_runs_tight_total_validates() {
        let req = guest_run_req(1240, 622, 1240 * 622 * 4, 0);
        assert!(validate_v1(&req).is_ok());
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
            total_len,
            row_length_texels,
            // A fixture over a dummy host address has no guest pages.
            pages: None,
        })
    }

    fn storage_buffer_req(content: BufferContent) -> DrawRequest {
        DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
            storage_buffers: vec![super::super::types::StorageBufferResource {
                binding: 0,
                content,
            }],
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

    /// A Constant-step attribute with a nonzero base instance needs the CPU
    /// prefix shift; a gathered guest span must be rejected at validate time
    /// (the runtime gate keeps those streams on the CPU path).
    #[test]
    fn buffer_guest_runs_rejects_constant_step_shift() {
        let content = buffer_guest_runs(&[48 * 4], 48 * 4, 0);
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
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
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
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
    /// nothing downstream reads. The Metal arm always asked for the pair; this
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
    fn empty_vertex_and_fragment_spirv_have_distinct_reasons() {
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(Vec::new()),
            frag_spirv: std::sync::Arc::new(vec![0]),
            ..DrawRequest::default()
        };
        assert_eq!(validation_slug(&req), "vk_draw_validate_empty_vertex_spirv");

        req.vert_spirv = std::sync::Arc::new(vec![0]);
        req.frag_spirv = std::sync::Arc::new(Vec::new());
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_empty_fragment_spirv"
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
            total_len: 156,
            row_length_texels: 0,
            // A fixture over a host `Vec` has no guest pages.
            pages: None,
        });
        assert_eq!(content.len(), 156);
        let bytes = content.cpu_bytes();
        assert_eq!(&bytes[..100], &backing[..100]);
        assert_eq!(&bytes[100..156], &backing[200..256]);
    }
}
