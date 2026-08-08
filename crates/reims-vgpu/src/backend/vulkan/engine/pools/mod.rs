//! Staging / target / readback / command / descriptor pools for warm-path reuse.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use ash::vk::Handle;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::buffer_slab::{BufferSlabToken, BUFFER_SLAB_IDLE_KEEP_EMPTY};
use super::compute_execution::ComputeExecutionDecline;
use super::context::{DeviceContext, FENCE_TIMEOUT_NS};
use super::counters::EngineCounters;
use super::desc_arena::{DescriptorArena, DESC_BLOCK_MAX_SETS};
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::types::{DrawError, ResidentReclaim, StorageImageFormat, TargetIdentity};
use super::vk_call::{VkCall, VkOp};
use super::{buffer_slab, color_subresource_range, host_ram, reason, slab, types};
use crate::backend::vulkan::caps::{MappedMemoryKind, MemoryClass};
use crate::backend::vulkan::translate;
use crate::model::ComputeStorageResidencyKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct TargetKey {
    pub width: u32,
    pub height: u32,
    pub with_transfer_dst: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct BufferSlot {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    /// Host address of `memory`, mapped for the slot's whole lifetime, or 0 when
    /// the slot is not persistently mapped (the readback pools).
    ///
    /// A `usize` rather than a pointer so [`crate::backend::vulkan::engine`]'s
    /// state stays `Send` — the same reason `GuestRun::host_ptr` is one.
    ///
    /// Staging memory is HOST_VISIBLE|HOST_COHERENT by construction
    /// (`MemoryClass::Upload` requires both), so a persistent map needs no
    /// flush — exactly as the map/write/unmap form it replaces needed none.
    /// Vulkan permits a memory object to stay mapped for its lifetime, and
    /// `vkFreeMemory` unmaps implicitly, so no teardown changes.
    pub mapped: usize,
    /// Whether `memory`'s type carries `HOST_COHERENT`.
    ///
    /// `MemoryClass::Readback` prefers `HOST_CACHED` over coherence, because a
    /// cached read is an order of magnitude faster and several drivers (Intel
    /// ANV among them) expose no type carrying both. A reader of a slot whose
    /// memory is not coherent owes `vkInvalidateMappedMemoryRanges` before it
    /// touches the mapping, so the slot states the property rather than leaving
    /// each reader to re-derive it from the type index.
    pub coherent: bool,
    /// Whether `memory`'s type carries `HOST_CACHED`.
    ///
    /// The other half of what `MemoryClass::Readback` asks for, and recorded for
    /// the same reason as `coherent`: it decides whether a reader may consume
    /// the mapping in place or must stream it out first. An uncached mapping
    /// reads at roughly a tenth of memcpy speed (`ReadbackMemoryDegrade` prices
    /// it at 460 MB/s on a live iGPU), and a scattered row-by-row consumer of
    /// one would pay that rate on every row rather than once on a linear pass.
    /// So the choice is a capability question, and the slot answers it.
    pub cached: bool,
    /// Where `memory` came from, and therefore how the slot is given back.
    pub backing: BufferBacking,
}

/// Where a [`BufferSlot`]'s device memory came from.
///
/// The distinction exists because the two are freed differently and getting it
/// wrong is not a leak but a double free: a slab-backed slot's `memory` is
/// shared with every other carve in its block.
#[derive(Clone, Copy)]
pub(crate) enum BufferBacking {
    /// The slot owns `memory` outright; releasing it is `vkFreeMemory`.
    ///
    /// The readback pools are all of this kind, and one more thing rests on it
    /// than the free path: [`invalidate_slot_for_read`] names the slot's whole
    /// memory object from offset 0, which is the slot's own bytes only while
    /// this holds. Sub-allocating a readback slot would have to narrow that
    /// range to the carve first.
    Dedicated,
    /// The slot holds a sub-range of a shared HOST_VISIBLE block; releasing it
    /// returns the range to [`buffer_slab::BufferSlabPool`] and must never
    /// call `vkFreeMemory`.
    Slab(BufferSlabToken),
}

/// Give a buffer slot back: destroy its handle, then release its memory by
/// whichever route its backing says.
///
/// A free function rather than a `ResourcePools` method so a caller iterating
/// one pool field can still hand the slab the `&mut` it needs.
///
/// # Safety
/// No in-flight command buffer may still reference `slot.buffer`.
pub(crate) unsafe fn release_buffer_slot(
    device: &ash::Device,
    slabs: &mut buffer_slab::BufferSlabs,
    slot: BufferSlot,
) {
    device.destroy_buffer(slot.buffer, None);
    match slot.backing {
        BufferBacking::Dedicated => device.free_memory(slot.memory, None),
        // Routed by the token's own kind and never by the call site: a
        // `BufferSlot` does not say which pool carved it, and the two pools'
        // block indices mean different things.
        BufferBacking::Slab(token) => slabs.release(device, token),
    }
}

pub(crate) struct TargetSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
}

/// A readback slot checked out of the pool, and the token its holder returns.
pub(crate) struct LeasedReadback {
    pub token: u64,
    pub slot: BufferSlot,
}

/// Tokens of leases whose holder has finished with the mapping.
///
/// Its own lock rather than a field of [`ResourcePools`], and that is the whole
/// point of the channel. Ending a lease must never need the engine lock: the
/// thread that ends one may be racing a teardown that already holds that lock
/// and is waiting for this very lease to come back, and a return path that
/// asked for the lock would close the cycle. So a holder drops a token here and
/// walks away; the next engine-locked pool operation collects it
/// ([`ResourcePools::reclaim_returned_readback_leases`]).
///
/// Nothing under this lock ever takes another, which is what keeps it a leaf.
static RETURNED_READBACK_LEASES: parking_lot::Mutex<Vec<u64>> = parking_lot::Mutex::new(Vec::new());

/// Leases handed out and not yet returned, readable without any lock.
///
/// A teardown that is about to `vkFreeMemory` a leased slot has to know whether
/// a borrow of its mapping is still live, and it cannot ask `readback_leased`
/// for that — a token sitting in [`RETURNED_READBACK_LEASES`] is a lease whose
/// holder is gone but whose slot has not been collected yet. This counter moves
/// with the *holder*, so zero means no pointer is outstanding.
static READBACK_LEASES_OUT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Distinct token per lease, so a return can name the slot it is giving back
/// without carrying a Vulkan handle across the lock boundary.
static NEXT_READBACK_LEASE_TOKEN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// End the lease `token` names: the holder is done reading its mapping.
///
/// Device-free by construction — a readback slot owns its mapping for life, so
/// there is no `vkUnmapMemory` owed and nothing here needs the engine.
pub(crate) fn return_readback_lease(token: u64) {
    RETURNED_READBACK_LEASES.lock().push(token);
    // After the token is queued, never before: a teardown that observes zero
    // must be able to conclude the slot is collectable, and a decrement ahead
    // of the push would let it observe zero with the token still in flight.
    READBACK_LEASES_OUT.fetch_sub(1, Ordering::AcqRel);
}

/// Whether any lease holder is still reading a readback mapping.
pub(crate) fn readback_leases_outstanding() -> usize {
    READBACK_LEASES_OUT.load(Ordering::Acquire)
}

pub(crate) struct ResourcePools {
    /// Size-bucketed free host-visible buffers (TRANSFER_SRC | VERTEX | INDEX | STORAGE).
    staging_free: HashMap<u64, Vec<BufferSlot>>,
    /// In-use staging slots returned after submit/wait.
    staging_live: Vec<BufferSlot>,
    /// Size-bucketed free DEVICE_LOCAL buffers the draw-time guest gather
    /// assembles scattered windows into, and the in-use ones the ring returns
    /// after submit/wait.
    ///
    /// A second pair rather than a flag on the staging pool: these are a
    /// different memory class, and a slot handed to the wrong one would either
    /// put a CPU write into memory the host cannot address or put a gather
    /// destination in system RAM, which is the arrangement
    /// `gather_guest_buffer_window` exists to avoid.
    gather_free: HashMap<u64, Vec<BufferSlot>>,
    gather_live: Vec<BufferSlot>,
    /// Buffer binds the command buffer now recording has already staged or
    /// gathered, keyed by the content that produced them.
    ///
    /// The key is `(Arc` address`, length)` of the bind's content — the runtime
    /// holds one `Arc` per resolved `(task, reference, offset)`, so two binds of
    /// the same guest window are the same pointer and two different windows
    /// cannot collide however their bytes compare.
    ///
    /// # Why the command buffer and not the draw
    ///
    /// This was a local in `execute_draw_inner`, so it deduplicated the ~5 binds
    /// of one draw and nothing across the batch that draw joins. A window bound
    /// by four consecutive draws was copied into device-local memory four times,
    /// and a driven Safari drag moved **4.46 GB per second** that way.
    ///
    /// One copy per command buffer is what the guest's own model gives it. The
    /// bytes are read when the command buffer executes, so a guest that rewrote
    /// the window between two draws of one command buffer would already be
    /// racing itself: nothing tells it when either draw runs. Copying twice does
    /// not make that guest correct, it only makes this device slower.
    ///
    /// Same probe after the move, busiest census second:
    ///
    /// ```text
    /// buffer_guest_gathers       19 372 ->  11 894
    /// buffer_guest_gather_bytes    4.46 GB ->   2.71 GB
    /// buffer_bind_reuses                -    23 947
    /// batch_flush_draws           3 913 ->   4 635
    /// ```
    ///
    /// Two binds in three are served from a copy this command buffer already
    /// holds. Bus traffic per draw — this, plus the surface writeback going the
    /// other way — went 2.25 MB to 1.69 MB while the drag ran 18 % more draws a
    /// second, and its peak present rate went 68 Hz to 76 Hz.
    ///
    /// # What ends an entry's life
    ///
    /// The slots named here live in `staging_live` / `gather_live`, so the map
    /// is cleared wherever those are handed away or recycled —
    /// [`ResourcePools::seal_entry`] and [`ResourcePools::recycle_staging`] —
    /// and additionally whenever this device records a write **into guest
    /// pages** ([`ResourcePools::note_guest_write_recorded`]). That last one is
    /// the correctness edge: a bind after a Store into the same pages must see
    /// what the Store wrote, and reusing a copy taken before it would not.
    cb_bound_buffers: HashMap<(usize, u64), super::exec::BoundBuffer>,
    /// Staging free-list hits / misses and the miss bucket histogram; see
    /// `note_staging_miss`. Measure-only.
    staging_hits: u64,
    staging_misses: u64,
    staging_miss_bins: [usize; STAGING_BUCKET_BINS],
    staging_miss_us_bins: [u64; STAGING_BUCKET_BINS],
    /// `staging_hits + staging_misses` at the previous fired idle pass; see
    /// `note_drain_settled`.
    settled_staging_mark: u64,
    /// Target images + framebuffers keyed by geometry + render_pass identity.
    targets: HashMap<(TargetKey, u64), TargetSlot>, // u64 = render_pass as u64
    target_order: Vec<(TargetKey, u64)>,
    /// Readback buffers by size.
    readback_free: HashMap<u64, Vec<BufferSlot>>,
    readback_live: Option<BufferSlot>,
    /// Extra live readbacks (compute multi-image / multi-buffer).
    readback_multi_live: Vec<BufferSlot>,
    /// Device-local buffer the guest-page writeback detiles a frame into before
    /// scattering it into the guest's stretches.
    ///
    /// One slot, not a pool keyed by size: this rail writes one frame at a time
    /// inside a single command buffer, and a boot's frames are all the same
    /// geometry until the display mode changes. It is grown and never shrunk,
    /// so a mode change up costs one reallocation and a mode change down costs
    /// nothing. `None` until the first frame that takes the linear path.
    guest_scratch: Option<BufferSlot>,
    /// Readback slots handed to a reader that is consuming their mapping with
    /// the engine unlocked; see [`ResourcePools::lease_readback`].
    ///
    /// Deliberately in none of the three lists above. A leased slot must not
    /// reach a ring entry's `PendingGpuCleanup` (which would return it to
    /// `readback_free` when that entry retires) and must not be handed to a
    /// second acquire, because either one lets a GPU copy overwrite bytes a
    /// live borrow is still reading.
    readback_leased: Vec<LeasedReadback>,
    /// Transient sampled-image pool, keyed by exact image and view geometry.
    sampled_free: FreePool<SampledKey, SampledSlot>,
    sampled_live: Vec<SampledSlot>,
    /// Exact-content sampled images retained across draw calls. Hash narrows
    /// candidates only; a hit always requires full byte equality.
    sampled_cache: Vec<ResidentSampledSlot>,
    sampled_cache_bytes: usize,
    /// Storage-image pool for compute.
    storage_image_free: FreePool<StorageImageKey, StorageImageSlot>,
    storage_image_live: Vec<StorageImageSlot>,
    /// Protocol-identity keyed compute storage images retained across calls.
    compute_storage_registry: HashMap<ComputeStorageResidencyKey, ResidentStorageImageSlot>,
    /// Insertion order for [`Self::compute_storage_registry`], oldest *created*
    /// at the front. A `VecDeque` for the same reason as [`Self::registry_order`].
    ///
    /// **Not use order**, and it was documented as LRU while it was not. Nothing
    /// promotes an entry when a dispatch reuses it, so selecting the front
    /// evicted the oldest-*created* resident however hard the current chain was
    /// reading it. That sweep is gone — the allocation bounds this population
    /// now, see `ResourcePools::recoverable_compute_storage_residents` — and
    /// what remains reads this order only to be deterministic, oldest-created
    /// first. Recency still lives on the slot
    /// ([`ResidentStorageImageSlot::last_touch_ms`]) for the age drain, which is
    /// the one rule that still consults a stamp.
    compute_storage_order: VecDeque<ComputeStorageResidencyKey>,
    /// Identity-keyed resident target registry (workstream D).
    registry: HashMap<TargetIdentity, ResidentTargetSlot>,
    /// Insertion order for [`Self::registry`], oldest *created* at the front. A
    /// `VecDeque` because the retired cap-eviction sweep popped and rotated at
    /// the front; what reads it now — `recoverable_residents` and the idle
    /// drain — only walks it, so the container is no longer load-bearing and the
    /// order is.
    ///
    /// **Not use order.** Nothing promotes an entry when a draw reuses it, so
    /// this alone would make a session-long resident the permanent front and the
    /// first candidate of every burst. Recency lives on the slot
    /// ([`ResidentTargetSlot::last_touch_ms`]), which the idle drain reads
    /// directly rather than reordering this list — so a promotion stays off the
    /// per-bind path while this order still makes ties deterministic.
    registry_order: VecDeque<TargetIdentity>,
    /// Recently reclaimed identities and which path took each, so a draw that
    /// samples a missing resident can say whether this device ever held one.
    ///
    /// Bounded and FIFO: it answers "what happened to this just now", and an
    /// identity that has fallen out of the window is reported as no record
    /// rather than guessed at. Diagnostic only — nothing reads it to decide
    /// anything.
    reclaimed_recent: VecDeque<(TargetIdentity, ResidentReclaim, u64)>,
    /// Highest non-pinned resident population this device has held, in slots.
    /// Reported as `registry_pressure`'s `peak`, beside the bytes the same
    /// sample occupied — the pair is what says a slot count was never a proxy
    /// for VRAM. See [`ResourcePools::recoverable_residents`].
    ///
    /// Kept here rather than on `EngineCounters` for the reason the recycle
    /// stats are: this is a property of the registry, which lives on this
    /// struct under the engine lock, so there is nothing to synchronise and no
    /// atomic to pay for. `engine::counter_snapshot` merges it in.
    registry_non_pinned_peak: u64,
    /// Longest a resident had gone untouched before something read it, in
    /// milliseconds of the idle clock, for the life of the pools.
    ///
    /// The margin against `IDLE_TARGET_AGE_MS`, which is the age at which the
    /// drain destroys a resident terminally. `resident_resample_band`'s bands
    /// give the distribution and this gives the worst case; the bands could not
    /// distinguish their one over-half-cutoff sample sitting at 1.0 s from
    /// sitting at 1.9 s, and those are opposite answers to whether this cutoff
    /// has room.
    ///
    /// A high-water rather than a windowed value for the same reason
    /// `registry_non_pinned_peak` is: the question is "how close did this boot
    /// ever come", and a gap that peaks between two census samples is exactly
    /// what an instantaneous reading misses.
    resident_resample_peak_ms: u64,
    /// The live non-pinned population and what it occupies, maintained rather
    /// than walked. Both readings come off this, and
    /// `ResourcePools::registry_non_pinned_adjust` is the only writer — see
    /// `ResourcePools::non_pinned_registry_len` for why it stopped being a walk.
    registry_non_pinned: NonPinnedTotals,
    /// The same high-water in attachment bytes rather than slots, sampled from
    /// the same population at the same instant. See
    /// [`ResourcePools::non_pinned_registry_bytes`] for why a slot count cannot
    /// stand in for it and why this reads as a lower bound.
    registry_non_pinned_peak_bytes: u64,
    /// The live population both reclaim paths refuse to take because this image
    /// is the only place its pixels exist, and what it occupies. Maintained, not
    /// walked, for the same reason [`NonPinnedTotals`] is.
    ///
    /// This is the instrument that says whether protecting that class is
    /// affordable. It is the one number that can turn "never lose a frame" into
    /// "hold every frame forever": a workload whose residents are all
    /// [`ResidentTargetSlot::gpu_only_content`] gives the reclaim paths nothing
    /// to take, and the allocation-failure retry then has nothing to give back.
    /// Read the peak against `registry_non_pinned_peak` — a ratio near 1 means
    /// the reclaim paths have stopped working and the copy-out sites are what
    /// needs the attention.
    registry_sole_copy: NonPinnedTotals,
    /// High-water of [`Self::registry_sole_copy`], in slots and in attachment
    /// bytes, sampled where every admission passes.
    registry_sole_copy_peak: NonPinnedTotals,
    /// The compute-storage counterparts of [`Self::registry_sole_copy`] and
    /// [`Self::registry_sole_copy_peak`]. Kept separate rather than summed with
    /// them: the two registries hold different resources — slab suballocations
    /// against standalone `VkDeviceMemory` — and a boot needs to know which one
    /// an allocation failure would have found something in.
    compute_storage_sole_copy: NonPinnedTotals,
    compute_storage_sole_copy_peak: NonPinnedTotals,
    /// Monotonic wall-clock milliseconds for the resident-target idle drain, fed
    /// from the poll heartbeat and each publish ([`Self::advance_registry_touch_and_drain`]).
    /// Each admit/hit/present stamps its slot's `last_touch_ms` with this value;
    /// the drain reclaims non-pinned residents whose stamp is `IDLE_TARGET_AGE_MS`
    /// behind. Wall-clock (not a publish counter) so it keeps advancing when the
    /// guest stops publishing, returning idle VRAM to baseline on a static page.
    idle_clock_ms: u64,
    /// Wall-clock ms of the last reclaim pass — enforces `IDLE_DRAIN_INTERVAL_MS`
    /// spacing so the ~244 Hz poll cadence cannot empty the registry at once.
    last_drain_ms: u64,
    /// Consecutive fired idle-drain passes that reclaimed **zero** registry
    /// residents. A pass that drains ≥1 victim means the working set is still
    /// churning (active video keeps aging out old frame RTs), so we reset to 0.
    /// The HOST_VISIBLE buffer pool trim (a full `vkAllocateMemory` re-alloc on
    /// the upload hot path when it refills) only fires once this crosses
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM`, so a single quiet pass mid-video cannot
    /// steal a 64 MiB staging buffer and spike the next upload's latency. The
    /// image/slab trims stay ungated — they refill via cheap slab suballocation.
    settled_drain_passes: u32,
    /// Persistent command pool; each ring slot owns one primary CB.
    cmd_pool: vk::CommandPool,
    /// Growable descriptor-pool arena (FREE_DESCRIPTOR_SET blocks). Grows a new
    /// block on exhaustion instead of hard-failing the draw; sets are freed
    /// per entry, paired with their owning block. See [`DescriptorArena`].
    desc_arena: DescriptorArena,
    /// N-deep in-flight ring: each slot is one CB + fence + the cleanup it
    /// owes. Entries rotate through slots; a slot is reused only after its
    /// fence retires (begin_entry blocks on the oldest when the ring is full).
    slots: Vec<CmdSlot>,
    /// Slot the current (or most recently begun) entry records into.
    cur: usize,
    /// Submitted-but-unretired slot count. While nonzero, destroying any GPU
    /// object a prior CB may reference is unsafe — dispose() defers those
    /// handles to `graveyard` until the slots that could reference them retire.
    in_flight: usize,
    /// Handles displaced (cache eviction, registry replace) while GPU work was
    /// open, each paired with the [`SlotMask`] of the ring slots that were open
    /// at dispose time. A dispose site has already made the handle unreachable
    /// (it was taken out of the cache/registry that named it), so no *later*
    /// entry can bind it; only the entries recording or in flight at that
    /// instant can still reference it. Clearing a slot's bit as it retires
    /// therefore frees the handle on the last fence that could be reading it,
    /// not on the whole ring going idle.
    graveyard: Vec<(SlotMask, DeferredHandle)>,
    /// Resident-target recycle pool: images displaced from the identity registry
    /// (generation bump / geometry change / LRU), held by (geometry, format) for
    /// reuse instead of destroyed. Kills the per-frame `vkCreateImage`+
    /// `vkAllocateMemory` storm a per-frame-generation target (video) would
    /// otherwise pay (see [`TargetRecycleKey`]). Bounded per key.
    target_free: FreePool<TargetRecycleKey, FreeTargetImage>,
    /// Open draw batch: a ring slot whose CB is still recording deferred
    /// same-target draws (submit pending). While `Some`, that CB references
    /// live GPU objects exactly like an in-flight CB, so dispose/graveyard
    /// treat it as in flight; every path that claims a slot or quiesces the
    /// ring flushes it first ([`Self::batch_flush`]).
    open_batch: Option<OpenBatch>,
    /// Offset suballocator for DEVICE_LOCAL optimal images (targets, sampled,
    /// storage, resident registry). Sub-allocates many image binds from a few
    /// large `VkDeviceMemory` blocks to collapse the per-image
    /// `vkAllocateMemory` storm of a layer-tree reflow burst
    /// ([[present-thrash-proxies]] bug #2). Live sub-allocations are keyed by
    /// `vk::Image` handle, so the free path routes through it with just the
    /// image.
    slab: slab::SlabPool,
    /// Offset suballocator for the HOST_VISIBLE upload blocks every staging
    /// buffer is carved from. Same reason as `slab` one field up, against a
    /// different measurement: a staging miss cost ~0.4-0.6 ms of
    /// `vkAllocateMemory` whatever its size, and the pool takes ~1 500 of them
    /// a boot, clustered on the first composite after idle.
    slabs: buffer_slab::BufferSlabs,
    /// Every RAMBlock this device has imported as a host pointer. Not a cache:
    /// one entry per span the shim reported, held for the device's life, with
    /// no eviction — see `host_ram` for why adding one would be a mistake.
    /// Lives here rather than beside its one consumer so it is destroyed by the
    /// same teardown that destroys every other device object.
    host_ram_imports: host_ram::HostRamImports,
    /// Whether any command buffer recorded or submitted since the last quiesce
    /// **reads** guest RAM when it executes.
    ///
    /// The whole reason [`ResourcePools::quiesce_guest_reads`] exists: this
    /// device tells the guest a packet is finished before the GPU has run it, so
    /// a read that happens at execute time can land after the guest has already
    /// repainted or freed the pages. One flag rather than a per-slot mask
    /// because the answer the stamp needs is "is there any", and a quiesce
    /// retires the whole ring regardless.
    guest_reads_in_flight: bool,
    /// Whether any command buffer submitted since the last quiesce **writes**
    /// guest RAM when it executes.
    ///
    /// The mirror of `guest_reads_in_flight`, and it exists for the same
    /// sentence read the other way round. `copy_target_to_guest_pages` used to
    /// wait its own fence before returning, which made the writeback rail one
    /// blocking GPU round trip per landed window: 369 fences a second at
    /// 1 360 us each, of which the device's own timestamps priced only 636 us as
    /// the copy executing. The remaining 724 us was submit-to-start plus
    /// signal-to-wake, paid once per window, and it was 267 ms of every second.
    ///
    /// Deferring the wait does not weaken the ordering the stamp needs, because
    /// that ordering was never "each copy has landed by the time its own
    /// function returns" — it is "every copy has landed before the guest is told
    /// anything". Recording the debt here and settling it at the same choke
    /// points [`ResourcePools::quiesce_guest_reads`] is settled at collapses N
    /// waits into one and lets the copies pipeline against each other on the
    /// GPU instead of stopping the queue between them.
    guest_writes_in_flight: bool,
    /// Residents held pinned because a submitted-but-unsettled writeback copy
    /// reads them.
    ///
    /// A window's flush used to unpin its resident as soon as the copy returned,
    /// which was safe only because the copy had already executed by then. With
    /// the wait deferred, unpinning at that point would let the
    /// allocation-failure reclaim or the idle drain take an image the GPU has
    /// not read yet. The pin is transferred here instead and released by
    /// [`ResourcePools::quiesce_guest_writes`], which waits the whole ring — so
    /// the interval it covers is exactly the interval the copy can still be
    /// running in.
    ///
    /// Cannot strand a pin: nothing is pushed here without also setting
    /// `guest_writes_in_flight`, and every setting of that flag is answered by a
    /// quiesce at the next completion stamp.
    unpin_on_settle: Vec<TargetIdentity>,
    initialized: bool,
}

/// State of the deferred-submit draw batch (draw-batching increment 1): the
/// opener's ring slot CB stays in recording state across joinable same-target
/// draws; per-draw descriptor sets and sampled-cache admissions accumulate
/// here and seal as ONE entry at flush.
/// What a deferred-submit batch is a batch *of*.
///
/// One value rather than four parameters, because these four decide two
/// different things in two places — whether a draw may join the open batch
/// (`batch_slot`) and what the batch records when one opens (`batch_append`) —
/// and they were spelled out at both. Two of them are adjacent `u32`s, so a
/// `width`/`height` transposition between the question and the answer compiles
/// and produces a batch that admits draws of the wrong shape.
///
/// Derived `PartialEq` is the join test, so the fields it turns on cannot drift
/// from the fields the batch carries: adding one here makes it decide joins
/// without a second edit.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct BatchTarget {
    pub identity: TargetIdentity,
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
}

pub(crate) struct OpenBatch {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    target: BatchTarget,
    draws: u64,
    /// Per-draw descriptor sets paired with the arena block they were allocated
    /// from, so the flush-time free routes each set to its owning pool.
    dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
    sampled_retains: Vec<SampledRetain>,
}

/// One in-flight ring slot: a primary CB, its fence (created unsignaled;
/// reset immediately after every successful wait), and — while the CB is in
/// flight — the cleanup its entry owes.
struct CmdSlot {
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    pending: Option<PendingGpuCleanup>,
}

/// In-flight ring depth: the next draw/dispatch records + submits while
/// previous no-readback CBs still run, removing the retire-before-acquire
/// stall of the single-slot engine. Cross-CB ordering needs no barriers
/// beyond the recorded ones — one queue, tracked layouts — so depth only
/// trades burst headroom against cleanup latency.
///
/// Depth 8 (2026-07-19): once the bufprep staging fix cut per-draw CPU prepare,
/// the draw path blocked in `begin_entry` ~61 µs/draw on slot N+1's fence — the
/// CPU outran the 3-deep GPU pipeline under Safari fast-scroll. Deepening the ring
/// lets the CPU stay ahead so the GPU stays fed: verified `retire_wait` 61 → 17
/// µs/draw, `present_hz` ~40 → ~50, correctness clean (residue byte-flat, and
/// every staleness counter of the day reading zero — those three counters have
/// since been removed with the caches they measured, so do not go looking for
/// them). It was submit/fence-bubble-bound,
/// not GPU-compute-bound. Cost is 8 command buffers + fences + up to 8 slots'
/// pooled staging live at once — bounded, pooled. `retire_wait` still ~17 µs, so
/// a deeper ring or render-pass batching (only ~37 % of draws join a shared pass)
/// may reclaim more.
pub(crate) const RING_DEPTH: usize = 8;

/// One bit per ring slot: the set of slots a deferred handle is still waiting
/// on. Sized so the whole ring fits, which is what bounds the graveyard — a
/// handle's mask can only name slots that existed when it was disposed, and
/// every one of those retires within a ring wrap.
pub(crate) type SlotMask = u32;
const _: () = assert!(
    RING_DEPTH <= SlotMask::BITS as usize,
    "SlotMask must have one bit per ring slot"
);

/// A GPU object displaced while a CB may still reference it. Destroyed only
/// once every in-flight fence has retired.
pub(crate) enum DeferredHandle {
    Image {
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
    },
    /// A sampled-cache slot evicted by the LRU/byte cap. Instead of destroying
    /// it, the drain returns it to `sampled_free` for reuse (bounded per key) so
    /// a content-changing sampled input (live tile / video frame) re-uploads
    /// into a recycled image instead of paying a fresh `vkAllocateMemory` every
    /// frame. Routed through the same in-flight-safe deferral as destroys: an
    /// in-flight CB may still sample the evicted image, so it only rejoins the
    /// free list once `in_flight == 0`.
    RecycleSampled(SampledSlot),
    /// A resident render-target image displaced from the registry (generation
    /// bump / geometry change / LRU). Instead of destroying it, the drain
    /// returns it to `target_free` for reuse (bounded per key) so a per-frame
    /// content-changing target (video output) re-renders into a recycled image
    /// instead of paying a fresh `vkCreateImage`+`vkAllocateMemory` every
    /// frame. Same in-flight-safe deferral as destroys: an in-flight CB may
    /// still reference the displaced image, so it only rejoins the free list
    /// once `in_flight == 0`.
    RecycleTarget(FreeTargetImage),
    Framebuffer(vk::Framebuffer),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    RenderPass(vk::RenderPass),
    ShaderModule(vk::ShaderModule),
    Sampler(vk::Sampler),
}

impl ResourcePools {
    /// Terminal destroy of a deferred handle. Image variants free their backing
    /// memory through the slab suballocator (`free_image` releases the
    /// sub-range; a non-slab image falls back to a raw `vkFreeMemory` so mixed
    /// slab/1:1 images both free correctly). Non-memory objects are destroyed
    /// directly.
    unsafe fn destroy_deferred_handle(&mut self, device: &ash::Device, handle: DeferredHandle) {
        match handle {
            DeferredHandle::Image {
                image,
                view,
                memory,
            } => {
                // Destroy the image before releasing its memory: `free_image`
                // may `vkFreeMemory` the whole block if this was its last live
                // sub-allocation, and freeing memory under a live image is UB.
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                if !self.slab.free_image(device, image) {
                    device.free_memory(memory, None);
                }
            }
            DeferredHandle::RecycleSampled(slot) => {
                device.destroy_image_view(slot.view, None);
                device.destroy_image(slot.image, None);
                if !self.slab.free_image(device, slot.image) {
                    device.free_memory(slot.memory, None);
                }
            }
            DeferredHandle::RecycleTarget(img) => {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
                if !self.slab.free_image(device, img.image) {
                    device.free_memory(img.memory, None);
                }
            }
            DeferredHandle::Framebuffer(fb) => device.destroy_framebuffer(fb, None),
            DeferredHandle::Pipeline(p) => device.destroy_pipeline(p, None),
            DeferredHandle::PipelineLayout(pl) => device.destroy_pipeline_layout(pl, None),
            DeferredHandle::DescriptorSetLayout(dsl) => {
                device.destroy_descriptor_set_layout(dsl, None)
            }
            DeferredHandle::RenderPass(rp) => device.destroy_render_pass(rp, None),
            DeferredHandle::ShaderModule(s) => device.destroy_shader_module(s, None),
            DeferredHandle::Sampler(s) => device.destroy_sampler(s, None),
        }
    }
}

/// Cleanup owed by an entry that skipped its post-submit fence wait: the
/// descriptor set, every transient pool slot the CB references (moved out of
/// the live lists at seal time so a concurrent entry cannot recycle them),
/// and the render path's sampled-content cache admissions — deferred because
/// admission can EVICT (destroy) cache images the in-flight CB may sample.
pub(crate) struct PendingGpuCleanup {
    dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
    staging: Vec<BufferSlot>,
    gather: Vec<BufferSlot>,
    readback: Vec<BufferSlot>,
    sampled: Vec<SampledSlot>,
    storage_images: Vec<StorageImageSlot>,
    sampled_retains: Vec<SampledRetain>,
}

pub(crate) struct SampledSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub volume: bool,
    pub cube: bool,
    pub arrayed: bool,
    /// The image was created as a Vulkan 1D (`TYPE_1D` / `TYPE_1D_ARRAY`) image
    /// because the shader's sampled binding reflects a Metal `texture1d` /
    /// `texture1d_array` (color-transfer LUTs). Part of the pool key: a 1D view
    /// and a `height==1` 2D view are byte-identical images but incompatible
    /// descriptor types, so a recycled slot must never cross that boundary.
    pub one_dim: bool,
    pub format: ash::vk::Format,
    /// The view's component mapping, from the decoded type-8 swizzle. Part of
    /// the pool key because it is baked into the `VkImageView`: a recycled slot
    /// whose view swizzles differently would silently remap a later bind's
    /// channels. Identity is the overwhelmingly common case and keeps its own
    /// free list, so a rare swizzled bind cannot fragment the hot one.
    pub swizzle: crate::contract::pixel_format::SwizzlePlan,
}

/// Everything that has to match for one sampled image to stand in for another:
/// the extent, the Vulkan image and view types the four shape flags select, the
/// format, and the view's component mapping.
///
/// Travels as one value all the way from the call site. The four flags are the
/// reason — they are adjacent, all `bool`, and `SampledImageResource` happens to
/// declare them in a different order than this does, so a positional call had
/// to reorder them by hand and nothing would have caught it getting that wrong.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SampledKey {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) volume: bool,
    pub(crate) cube: bool,
    pub(crate) arrayed: bool,
    pub(crate) one_dim: bool,
    pub(crate) format: ash::vk::Format,
    pub(crate) swizzle: crate::contract::pixel_format::SwizzlePlan,
}

impl SampledKey {
    /// The key a decoded sampled binding asks for.
    ///
    /// A caller that needs one field to differ says so with struct-update
    /// syntax (`SampledKey { format: .., ..SampledKey::of(r) }`), which names
    /// the field it is overriding — the snapshot arm binds a resident under
    /// `resident_color` rather than the resource's own format, and that used to
    /// be visible only by counting to the eighth argument.
    pub(crate) fn of(r: &types::SampledImageResource) -> Self {
        Self {
            width: r.width,
            height: r.height,
            layers: r.layers,
            volume: r.volume,
            cube: r.cube,
            arrayed: r.arrayed,
            one_dim: r.one_dim,
            format: r.format,
            swizzle: r.swizzle,
        }
    }
}

impl SampledSlot {
    fn key(&self) -> SampledKey {
        SampledKey {
            width: self.width,
            height: self.height,
            layers: self.layers,
            volume: self.volume,
            cube: self.cube,
            arrayed: self.arrayed,
            one_dim: self.one_dim,
            format: self.format,
            swizzle: self.swizzle,
        }
    }

    fn handles(&self) -> Self {
        Self {
            image: self.image,
            memory: self.memory,
            view: self.view,
            width: self.width,
            height: self.height,
            layers: self.layers,
            volume: self.volume,
            cube: self.cube,
            arrayed: self.arrayed,
            one_dim: self.one_dim,
            format: self.format,
            swizzle: self.swizzle,
        }
    }
}

/// How a retained sampled image can be recognised again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SampledFingerprint {
    /// 128-bit digest of the retained content (see [`sampled_content_hash`]).
    ///
    /// It picks the candidate and decides nothing on its own. The bytes it was
    /// taken over are retained beside it in `ResidentSampledSlot::content`, and
    /// that compare is what answers — see that field for why the digest is not
    /// allowed to be the identity.
    Content(u128),
    /// No digest exists, because the content was gathered straight from guest
    /// RAM into a staging buffer and never materialised as CPU bytes. Such an
    /// entry is reachable only through its producer identity, which is the whole
    /// point: hashing it would mean reading the bytes this entry exists to avoid
    /// reading. It must never match a content search, or a CPU-sourced bind
    /// could pick up an image whose bytes nobody has compared.
    Gathered,
}

/// One sampled image a retired submission owes the content cache.
pub(crate) struct SampledRetain {
    pub(crate) image: vk::Image,
    /// The bytes to fingerprint, where the source had any. A guest gather has
    /// none, and carries only how many it moved so the cache's byte cap still
    /// accounts for it.
    pub(crate) content: SampledRetainContent,
    pub(crate) identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
}

/// What a retained slot's content was, for the two kinds of source.
pub(crate) enum SampledRetainContent {
    Bytes(std::sync::Arc<Vec<u8>>),
    Gathered { len: usize },
}

struct ResidentSampledSlot {
    slot: SampledSlot,
    fingerprint: SampledFingerprint,
    /// The bytes [`SampledFingerprint::Content`]'s digest was taken over, so a
    /// content match is decided by comparing them rather than by the digest
    /// alone. `None` exactly when the fingerprint is
    /// [`SampledFingerprint::Gathered`], which has no CPU bytes to retain and is
    /// reachable only by identity.
    ///
    /// # Why the digest is not the identity
    ///
    /// This used to hold nothing and `find_cached_sampled` bound the retained
    /// image on a 128-bit fingerprint match alone. Two distinct textures of the
    /// same geometry and format whose digests collided were one entry: a draw
    /// sampled pixels the guest never uploaded, with nothing to refuse and
    /// nothing to log. The standing argument was the birthday bound — about
    /// `2^-116` across a 64-entry cache — and that arithmetic is right and is
    /// the wrong shape, for the reason [`crate::backend::blob`] gives for the
    /// Metal caches it removed the same class from: it prices a failure this
    /// device cannot observe if it ever happens, and a wider digest only moves
    /// the exponent.
    ///
    /// The cost the copy was dropped for was a "cold full-frame `memcmp` on
    /// every hit", which assumed entries are frames. A driven x86/Vulkan boot
    /// under a 30 s Safari drag says otherwise: 26 697 content-path hits moving
    /// 277 MB, which is **10 KB per hit**, beside a guest-gather rail the same
    /// boot runs at 842 MB/s. The compare is not measurable against it.
    ///
    /// Retaining is a refcount bump and not a copy — the `Arc` already exists on
    /// the retire path — and it costs no new budget: `sampled_cache_bytes` has
    /// always summed `content_len` against `SAMPLED_CACHE_BYTE_CAP`, so the
    /// cache was already charging itself for bytes it did not hold. Holding
    /// them makes that accounting honest.
    content: Option<std::sync::Arc<Vec<u8>>>,
    /// Byte length of the content this slot was admitted with, for the LRU
    /// byte-cap accounting. Still carried separately because a `Gathered` entry
    /// has a length but no bytes.
    content_len: usize,
    /// Producer identity of the retained content; lets a same-identity,
    /// same-generation rebind skip the content hash + compare entirely.
    identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    /// Value of [`ResourcePools::idle_clock_ms`] at this entry's last use (admit
    /// or `find_cached_sampled` hit). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims an entry
    /// once its touch falls `IDLE_TARGET_AGE_MS` behind the clock — so a settled
    /// video session's frame textures (the ≤128 MiB sampled cache) are returned
    /// to the driver at idle instead of pinned for the guest lifetime, while an
    /// actively-sampled entry (hit every frame) never ages out. Mirrors the
    /// resident-target registry drain; the sampled cache is the analogous
    /// upload-side pool the buffer/target idle trims already cover.
    last_touch_ms: u64,
}

/// Geometry+format key for storage-image pool free lists. Compute images are
/// single-layer 2D by contract (see [`crate::backend::vulkan::engine::ComputeStorageImageResource`]),
/// so geometry is exactly width × height.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct StorageImageKey {
    pub width: u32,
    pub height: u32,
    pub format: StorageImageFormat,
    /// Read-only sampled descriptor instead of writable storage descriptor.
    pub sampled_only: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct StorageImageSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub key: StorageImageKey,
}

pub(crate) struct ResidentStorageImageUse {
    pub slot: StorageImageSlot,
    pub access: ResidentAccess,
    pub generation_match: bool,
}

struct ResidentStorageImageSlot {
    slot: StorageImageSlot,
    generation: u32,
    /// What last touched this image, and where that left it — see
    /// [`ResidentAccess`], which the target registry shares.
    access: ResidentAccess,
    /// Deferred-writeback pin: the resident is the only copy of this content
    /// (guest pages are stale) — LRU eviction must skip it until the caller
    /// flushes and unpins.
    pinned: bool,
    /// This image holds dispatch output that exists nowhere else. Both reclaim
    /// paths skip it, at any age and any population.
    ///
    /// [`ResidentTargetSlot::gpu_only_content`] on the sibling registry, for the
    /// same reason and against the same defect: `pinned == false` covers both
    /// "the writeback landed, the guest's pages hold this" and "no writeback was
    /// ever armed, so nothing outside this image ever held it". A dispatch that
    /// produces a storage image and is only ever *read* from — never re-armed —
    /// sits in the second state for its whole life.
    ///
    /// The loss here is louder than on the sibling and no less real: a later
    /// dispatch reading a destroyed identity refuses with `ResidentSampleAbsent`
    /// or `ResidentSeedGenerationLost`, so the guest's compute work is dropped
    /// rather than silently mis-served.
    ///
    /// Set by `mark_resident_storage_image` — the call every executed dispatch
    /// makes for the image it wrote — and cleared where the content has
    /// demonstrably left the image: a landed readback, or the guest deleting the
    /// object it belonged to. Not cleared by an unpin: `flush_storage_one`'s
    /// abort path and `lifecycle`'s window-cleared path both unpin without
    /// having written anything.
    ///
    /// # What it costs is not yet known on a workload that would show it
    ///
    /// Driven x86/PCI boot, `web-content-probe -n 10 --churn 1`, quiesced host:
    /// `cs_sole_copy=1/0mib`, `cs_cap_no_victim=0`, `compute_storage_evicts=0`.
    /// One protected resident under a megabyte, and the cap never wanted it.
    ///
    /// **Read that as "not exercised", not as "free".** A browser page is not a
    /// compute workload, and one resident against a 64 cap says the registry was
    /// nearly empty for the whole run. The reading that would mean something
    /// comes from a guest doing sustained compute — a video decode or filter
    /// chain — and `cs_sole_copy` against the live population is what to look at
    /// when one is available. A ratio near 1 means an allocation failure would
    /// find nothing here to give back.
    gpu_only_content: bool,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this resident's
    /// last use (admit or `acquire_resident_storage_image` hit). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the clock — so a
    /// compute-heavy burst's stale residents (a settled page's blur/decode storage
    /// images) are returned to the driver instead of pinning standalone
    /// VkDeviceMemory allocations for the
    /// guest lifetime, while an actively-dispatched resident (touched every pass)
    /// never ages out. Mirrors [`ResidentTargetSlot::last_touch_ms`].
    last_touch_ms: u64,
}

/// Persistent GPU render-target slot (identity-keyed registry, workstream D).
/// Whether a resident can be blitted to the host window exactly as it stands.
///
/// The window presenter does no format conversion and no scaling of the source
/// rect, so all four conditions are hard: content landed, guest scanout byte
/// order, and the exact geometry the present names.
///
/// It is a free function because two callers must ask the *same* question at
/// two different moments. The presenter asks it to pick a source; the device's
/// publish path asks it a frame earlier to decide whether to read the frame back
/// into host memory at all. If publish's predicate were the looser one, it would
/// elide the readback for a frame the presenter then refuses — a blank window
/// with no CPU pixels behind it, and no single site able to see the
/// disagreement.
/// Why a resident cannot carry this present, or `None` when it can.
///
/// Four independent conditions used to collapse into one `bool`, and the caller
/// fell through to the CPU-BGRA present path without saying which had failed —
/// so a boot reading `direct_frac=0.00` in every census window, against a
/// documented expectation of `1.00`, named no cause at all. Each present that
/// takes the fallback copies the whole framebuffer through host memory, so this
/// is a throughput cliff and not a cosmetic one.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentPresentDecline {
    /// The image exists but nothing has vouched for its pixels yet.
    ContentNotReady,
    /// The image's texels are not in the byte order the scanout blit reads.
    ScanoutOrder,
    /// The resident's geometry is not the geometry being presented.
    Geometry,
}

pub(crate) fn slot_present_decline(
    slot: &ResidentTargetSlot,
    width: u32,
    height: u32,
) -> Option<ResidentPresentDecline> {
    if !slot.content_ready {
        return Some(ResidentPresentDecline::ContentNotReady);
    }
    if !slot.scanout_order() {
        return Some(ResidentPresentDecline::ScanoutOrder);
    }
    if slot.width != width || slot.height != height {
        return Some(ResidentPresentDecline::Geometry);
    }
    None
}

pub(crate) fn slot_presentable(slot: &ResidentTargetSlot, width: u32, height: u32) -> bool {
    slot_present_decline(slot, width, height).is_none()
}

/// The non-pinned resident population, and the attachment bytes it holds.
///
/// One struct rather than two fields because the pair is only ever read
/// together — `registry_pressure` reports both, and a count without its bytes is
/// the reading a slot count was defended by for as long as nobody could say what
/// a slot costs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NonPinnedTotals {
    pub count: usize,
    pub bytes: u64,
}

/// Where a registry-resident image sits, and what put it there, as one value.
///
/// The two halves are inseparable because reading either one alone is the bug
/// this type exists to make unrepresentable. A tracked **layout** names where
/// the image is; a barrier's source scope has to name what last **touched** it,
/// and on this registry the two disagree by construction. A render pass moves
/// its primary attachment to `TRANSFER_SRC_OPTIMAL` through `final_layout`
/// without any transfer having run, so a resident sitting in that layout was in
/// fact last written by a colour attachment write. Deriving a barrier's source
/// scope from the layout reads that write as a transfer read and leaves the
/// colour writes free to race whatever comes next — the same stale-frame
/// failure as omitting the barrier outright, only harder to see.
///
/// A resident is the only image class where this matters. Pool-owned transients
/// re-enter their free lists only through `drain_cleanup`, which `retire_slot`
/// reaches only after `wait_for_fences` on the submission that last used them,
/// so a pooled image cannot be handed out while GPU work still reads it and
/// there is nothing for a source scope to name. A resident is keyed by
/// [`TargetIdentity`] and deliberately outlives the draw so its pixels survive
/// to the next one, which is exactly what makes it useful — and exactly why it
/// alone has to state what it is waiting for.
///
/// The enum is closed because the rails that touch a resident are: a draw's
/// render pass, an MRT secondary's render pass, a resident sample, and the
/// three transfer reads (present blit, guest-page readback, GPU seed source).
/// Every one of them ends in one of these variants, which is what lets
/// [`Self::source_scope`] be exact rather than a blunt `ALL_COMMANDS` union
/// over every write a resident could conceivably carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentAccess {
    /// Created or recycled; nothing has touched the image.
    Untouched,
    /// A render pass wrote it as a colour attachment and left it in that pass's
    /// `final_layout` — `TRANSFER_SRC_OPTIMAL` for a primary target,
    /// `COLOR_ATTACHMENT_OPTIMAL` for an MRT secondary.
    ColorWrite(vk::ImageLayout),
    /// A draw sampled it.
    ShaderRead,
    /// A transfer read it: a present blit, a guest-page readback, a GPU seed
    /// copy, or this draw's own copy-on-sample snapshot.
    TransferRead,
}

impl ResidentAccess {
    /// Where the image is — the `old_layout` a barrier over it must name.
    pub(crate) fn layout(self) -> vk::ImageLayout {
        match self {
            Self::Untouched => vk::ImageLayout::UNDEFINED,
            Self::ColorWrite(layout) => layout,
            Self::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Self::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    /// What last touched it — the `srcStageMask`/`srcAccessMask` a barrier over
    /// it must name, whichever direction that barrier goes.
    ///
    /// One answer serves reads and writes both. A read-after-write needs the
    /// write made available, which the write's own access flag supplies; a
    /// write-after-read needs an execution dependency on the read, which the
    /// read's own stage supplies. Naming the single access that actually
    /// happened covers each hazard in its own direction.
    ///
    /// Naming only the *last* access is sufficient because consecutive barriers
    /// over one resident form a dependency chain: the barrier that let access
    /// N+1 happen already named access N as its source, so a write two accesses
    /// back stays ordered through the barrier that stood between them. That
    /// holds only while every touch updates the registry, which is why this
    /// enum and the field carrying it are the same value — a rail that touches
    /// a resident without recording it is the one way to break the chain, and
    /// it cannot compile without choosing a variant.
    pub(crate) fn source_scope(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
        match self {
            Self::Untouched => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
            Self::ColorWrite(_) => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            Self::ShaderRead => (
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
            ),
            Self::TransferRead => (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
        }
    }
}

pub(crate) struct ResidentTargetSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub width: u32,
    pub height: u32,
    pub generation: u64,
    pub content_ready: bool,
    /// The mapping-level `surface_content_epoch` this image's pixels were last
    /// stamped with, or `None` when nothing has vouched for them.
    ///
    /// `None` is the fail-closed default and it is what every reset restores:
    /// slot creation, image recycle, and both `registry_mark_ready*` arms — so
    /// a draw that stores into this identity without going on to publish the
    /// mapping's content leaves the slot unvouched, and the type-11 LOAD gate
    /// falls back to its CPU seed. An `Option` rather than a sentinel because
    /// epoch 0 ("nothing published since attach") is a legal *mapping* value
    /// and a bare `0 == 0` would match an image that was never stamped at all.
    pub content_epoch: Option<u32>,
    /// What last touched this image, and where that left it. See
    /// [`ResidentAccess`] for why these are one field and not two.
    pub access: ResidentAccess,
    /// Concrete Vulkan attachment format the image was created with. Every
    /// question about this resident's channel order is asked of this field,
    /// through [`ResidentTargetSlot::scanout_order`] — a format change forces an
    /// image recreate, not just a framebuffer rebuild.
    pub color_format: vk::Format,
    /// Deferred render-Store pin count: this target's content exists only on
    /// the GPU (guest pages stale). The registry LRU sweep skips slots with a
    /// nonzero count. A count (not a bool) because a surface with several
    /// identity is pinned independently by each member's deferred window —
    /// the first member's flush must not expose the image to eviction while
    /// a peer's window is still armed.
    pub pin_count: u32,
    /// This image holds pixels that exist **nowhere else** — not in the guest's
    /// pages, not in any host-side copy. Destroying it destroys guest work.
    ///
    /// Both reclaim paths skip a slot carrying this, at any population and at
    /// any age. That is not an optimisation; it is the difference between a
    /// reclaim that costs redundant work and one that loses a frame.
    ///
    /// # Why `pin_count == 0` was not already this test
    ///
    /// It reads as though it were — the field above says "content exists only on
    /// the GPU" — but a pin is *armed* by the deferred-Store rail and released
    /// when that rail is done, so `pin_count == 0` conflates two opposite states:
    ///
    /// - was pinned, the flush landed, the guest's pages now hold these pixels.
    ///   Reclaiming costs a re-upload and nothing else.
    /// - was **never** pinned, because no writeback rail was ever armed for it.
    ///   The guest's pages hold whatever they held before, which is a different
    ///   frame or nothing at all.
    ///
    /// An MRT secondary attachment is the standing case of the second: it is
    /// registered like any colour target, rendered into, marked ready at
    /// `COLOR_ATTACHMENT_OPTIMAL`, sampled by the consumer pass — and never
    /// pinned, never stamped, never written back. Reclaimed, it does not refuse:
    /// `resolve_sampled_source` finds no resident and falls through to the
    /// mapping's guest pages, which substitutes an unrelated earlier frame with
    /// no failure anywhere. That is the class this field closes.
    ///
    /// # Polarity
    ///
    /// Set by `registry_mark_ready*` — the two calls every path leaving new
    /// pixels in a resident goes through — and cleared only by
    /// [`ResourcePools::registry_note_content_copied_out`], at a site that has
    /// just copied those pixels somewhere that outlives the image. So the
    /// default is protection, and a copy-out site nobody has taught about this
    /// costs retained VRAM rather than a lost frame. Add sites as they are
    /// proven, never to make a number look better.
    ///
    /// # What it costs, measured as a controlled A/B
    ///
    /// The standing worry about protecting a class from reclaim is that it
    /// becomes a population nothing can trim. Both arms are the same build with
    /// only the two reclaim predicates changed (the flag and its totals are
    /// maintained in both, so the control reports the population it *would*
    /// have protected). Driven x86/PCI boot each, `web-content-probe -n 10
    /// --churn 1`, run on to a settled desktop, quiesced host:
    ///
    /// ```text
    ///                                    gates off   gates on
    ///   non-pinned peak (cap 320)              191        275
    ///   sole_copy peak                     68/27mib  194/37mib
    ///   cap_no_victim                            0          0
    ///   evicts                                   0          0
    ///   slab_mib peak held                     464        464
    ///   slab_mib settled carved/held         45/72     52/208
    ///   t11sample_reclaimed_from_pages (sum)    33         39
    /// ```
    ///
    /// - **`cap_no_victim=0` on both.** Not once did the then-live capacity walk
    ///   want a victim and find every candidate protected, so the ceiling this
    ///   could have introduced was not approached, and `evicts` stayed 0 either
    ///   way. Both counters are retired with the walk; what survives is the
    ///   protected population itself, which the allocation-failure reclaim
    ///   selects against with the same predicate.
    /// - **The protected set is small surfaces.** 194 slots is 71 % of the peak
    ///   population but 37 MiB is 19 % of its bytes. A slot count alone would
    ///   overstate this nearly fourfold, which is why the totals carry bytes.
    /// - **Peak VRAM is identical (464 MiB held) and settled carved moves
    ///   45 → 52 MiB.** Settled *held* is 72 → 208, so the slab retains more
    ///   empty blocks; the bytes actually in use grow by about the 10 MiB the
    ///   protected population accounts for.
    /// - **Headroom is what this really spends.** The non-pinned peak goes
    ///   191 → 275. That was headroom against a 320-slot count at the time; the
    ///   count is gone, so what it now spends is what an allocation-failure
    ///   reclaim would have to work with, and `registry_sole_copy_peak` against
    ///   `registry_non_pinned_peak` is the ratio that says how much.
    ///
    /// ## This does **not** reduce `t11sample_reclaimed_from_pages`
    ///
    /// 33 against 39, i.e. no reduction and if anything slightly up inside the
    /// run-to-run spread. An earlier revision of this doc claimed a fall from
    /// "36-44" to 12; that was wrong. `t11sample_reclaimed_from_pages` is a
    /// `store_routes` counter and therefore **per-window**, and the 12 was a
    /// `sort -n | tail -1` over the samples — the busiest window read as a boot
    /// total. Summed, the two arms are the same.
    ///
    /// That is consistent with what the field does rather than a
    /// disappointment: the events it leaves are residents that *were* copied
    /// out, and those are exactly the ones the gate still lets the drain take.
    /// This field is a correctness change, and the table above is the argument
    /// that it is affordable — not an argument that it is also an optimisation.
    ///
    /// # Every writer does pass through the two setters
    ///
    /// The protection is only as complete as that claim, so it was audited
    /// rather than assumed. Every recorded GPU write whose destination is a
    /// registry resident's image:
    ///
    /// - the draw pass itself (`exec::execute_draw_inner`'s
    ///   `cmd_begin_render_pass`), and the two seed paths that precede it
    ///   (`cmd_copy_buffer_to_image` from a host seed, `cmd_copy_image` from
    ///   another resident) — all three covered by the one `registry_mark_ready`
    ///   after the pass;
    /// - the MRT secondary attachments, covered by the `registry_mark_ready_at`
    ///   loop beside it.
    ///
    /// The host-window present path is **not** a writer and was the specific
    /// worry: `window_present` reads residents as blit and clear *sources* and
    /// writes only the swapchain image, which is never a registry resident. The
    /// one thing it mutates on a resident is its tracked layout.
    ///
    /// Sampled and staging images are written on other paths, but they are not
    /// registry residents and hold no identity a later draw could resolve.
    pub gpu_only_content: bool,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this target's
    /// last use (admit, `registry_ensure` hit, or present touch). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the current
    /// clock — so a burst's stale targets (a settled YouTube page's thumbnail RTs)
    /// are reclaimed instead of pinning VRAM for the guest lifetime, while an
    /// actively-drawn target (touched every frame) never ages out.
    pub last_touch_ms: u64,
}

impl ResidentTargetSlot {
    /// Whether this image's texels are already in the byte order the guest
    /// scanout and the host window blit both read.
    ///
    /// Derived rather than stored. The bool used to sit beside `color_format`,
    /// written by the primary arm as the `bgra` it was asked for and by the MRT
    /// secondary arm as `format == SCANOUT_FORMAT`; the same identity can be
    /// created by either arm, so two spellings of one fact could disagree about
    /// one slot. `resident_color` maps the two `bgra` values onto two distinct
    /// formats, so this test answers both arms identically.
    pub(crate) fn scanout_order(&self) -> bool {
        self.color_format == translate::pixel::SCANOUT_FORMAT
    }

    /// The framebuffer this slot owes the deferred-destroy path, or `None` when
    /// it has none to give.
    ///
    /// **Only a resident's framebuffer is optional.** The MRT-secondary arm
    /// builds no per-slot framebuffer — a secondary attachment is only ever
    /// bound as attachment N of an ad-hoc MRT framebuffer or sampled through
    /// its view — so every slot it creates stores `VK_NULL_HANDLE` here, while
    /// the target pool creates a framebuffer beside every slot it hands out and
    /// can never store a null one.
    ///
    /// A question rather than a field for the same reason as
    /// [`Self::scanout_order`] one method up: the same identity can be created
    /// by either arm and then destroyed by any of three paths, and each of the
    /// three answered this separately. `vkDestroyFramebuffer` accepts the null
    /// handle and does nothing with it, so the two that answered it wrong were
    /// neither a crash nor a log line — they spent a graveyard entry, which is
    /// a bounded ring shared with every destroy that is real.
    ///
    /// # A driven boot does not reach the `None` arm
    ///
    /// The desktop workload creates no framebuffer-less resident at all: on a
    /// driven x86/Vulkan boot (Safari window drag, 2 892 posted events, ~37.5 Hz
    /// median present) the `vk_alloc_sites` census reads `mrt_secondary=0:0:0`,
    /// and only [`ResourcePools::registry_ensure_attachment`] allocates under that
    /// site. So every slot the drain and the recreate arms retired that boot had
    /// a real framebuffer, and all three disposal sites would have behaved
    /// identically with the check and without it.
    ///
    /// That is why the guard is held by a device-free test rather than by a
    /// boot, and why a green boot is not evidence about this arm. Whatever
    /// workload does drive Apple's multiple-render-target path is what would
    /// exercise it.
    pub(crate) fn owed_framebuffer(&self) -> Option<vk::Framebuffer> {
        (self.framebuffer != vk::Framebuffer::null()).then_some(self.framebuffer)
    }

    /// Whether this slot's image may be re-used for a request of this geometry,
    /// generation and attachment format.
    ///
    /// One predicate for both `registry_ensure*` arms, which share `registry`
    /// and so can hand each other a slot. They did not share this question: the
    /// primary arm compared `bgra`, a one-bit summary of the format, while the
    /// secondary arm compared `color_format`, which is what the slot's image was
    /// actually created with and what its own doc says reuse is keyed on. An
    /// identity crossing from the secondary path to the primary one is not
    /// hypothetical — `retire_resident`'s doc records having had to handle
    /// exactly that — and a secondary slot's `bgra` is stored as
    /// `format == SCANOUT_FORMAT`, so an `RG16Float` mask slot reads `false` and
    /// matched a primary request for RGBA8. The primary arm would then build a
    /// framebuffer with an RGBA8 render pass over an `RG16Float` view, which
    /// Vulkan does not allow the attachment formats to disagree on.
    ///
    /// `color_format` decides it for both, and it subsumes the `bgra` test:
    /// `translate::pixel::resident_color` maps the two `bgra` values onto two
    /// distinct formats, so equal formats implies equal `bgra` for anything the
    /// primary arm created.
    ///
    /// A narrowing can only cost hits, so the reading that matters is that it
    /// costs none here: on a driven x86/Vulkan boot (25 s Safari window drag,
    /// 2 680 posted events) the fail channel carries no new event, `gen_mismatch`
    /// stays 0, and the desktop composites. This workload never made the
    /// crossing this refuses, which is why it had never surfaced as a
    /// validation error.
    pub(crate) fn reusable_for(
        &self,
        width: u32,
        height: u32,
        generation: u64,
        format: vk::Format,
    ) -> bool {
        self.width == width
            && self.height == height
            && self.generation == generation
            && self.color_format == format
    }
}

/// Geometry+format key for the resident-target recycle pool (`target_free`).
/// The registry keys targets by [`TargetIdentity`], which folds `generation`
/// into Hash/Eq — a per-frame content-changing target (video output, a live
/// compositor RT) bumps its generation every frame, so every frame is a *new*
/// registry key, a `registry_ensure` miss, and a full `vkCreateImage` +
/// `vkAllocateMemory`. Recycling by (geometry, format) — which is stable across
/// those generation bumps — lets the freed image+memory+view be reused instead
/// of reallocated, so the per-frame alloc storm collapses to alloc-once.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct TargetRecycleKey {
    width: u32,
    height: u32,
    format: vk::Format,
}

/// A resident-target image+memory+view displaced from the registry (generation
/// bump / geometry change / LRU eviction) and held for reuse instead of
/// destroyed. The framebuffer is NOT retained — it binds one specific
/// `render_pass`, is disposed separately, and a reused image builds a fresh
/// one. Carries its own geometry so [`ResourcePools::try_recycle_target`] can
/// bucket it without a separate key argument (mirrors [`SampledSlot`]).
pub(crate) struct FreeTargetImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    format: vk::Format,
}

impl FreeTargetImage {
    fn key(&self) -> TargetRecycleKey {
        TargetRecycleKey {
            width: self.width,
            height: self.height,
            format: self.format,
        }
    }
}

/// Wall-clock milliseconds a non-pinned resident may go untouched before the
/// idle drain reclaims it. An actively-drawn target is touched every frame (and
/// the presented target is touched every poll) so it never ages out, while a
/// burst's stale targets (a settled page's thumbnail RTs) are reclaimed ~2 s
/// after last use — so a burst is absorbed whole and its VRAM still comes back,
/// which is what lets the population have no count bounding it at all.
///
/// **Wall-clock, not publish-count:** the drain clock is fed from the poll
/// heartbeat (`device_poll`, ~244 Hz), which ticks even when the guest stops
/// compositing and issuing present publishes. A publish-count clock froze on a
/// static page — measured at zero publishes per second — so a burst's ~260 stale
/// residents (~516 MiB) never aged out and VRAM never returned to the ~1005 MiB
/// idle baseline. Real time keeps advancing regardless of guest activity.
///
/// # This value has 20 % margin on a routine workload, and that is measured
///
/// Reclaim here is terminal — `retire_resident` writes nothing back and nothing
/// recreates a resident's content — so a resident aged out and then sampled
/// refuses permanently. What keeps that from happening is that reading a
/// resident touches it, which only works while the gap *between* reads stays
/// under this value. Nothing reported that gap until
/// `resident_resample_peak_ms` did.
///
/// Driven x86/PCI, `web-content-probe --churn 1`:
///
/// ```text
///   registry_pressure ... resample_peak_ms=1609/2000
/// ```
///
/// **1609 ms of a 2000 ms budget.** Not a contrived workload — a web page
/// loading and churning. The distribution is not the reassuring part either:
/// ~25 000 resamples land under 250 ms and then a handful jump straight to the
/// top band, so the tail is not a gentle slope this value sits far above. It is
/// a rare long gap that came within 391 ms of destroying a resident something
/// was about to read.
///
/// Two things follow, and neither is "make the number bigger":
///
/// - A zero `sampled_resident_missing` on this workload is a near miss, not
///   headroom. The margin is ~20 %, and a slower host, a heavier page or a
///   contended machine eats that — every `us=` number this device reports is
///   wall clock on a shared machine, and this cutoff is measured in the same
///   wall clock.
/// - The reclaim runs on a timer with no reference to memory pressure at all.
///   On an idle desktop (~56 residents) it destroys terminally while freeing
///   VRAM nobody wants back, which is pure downside; the burst it was written
///   for is the only time the risk buys anything.
///
/// # A memory-pressure gate on the drain was tried, measured, and reverted
///
/// The obvious answer to that last bullet — skip the drain while the population
/// is below some fraction of the slot count that then bounded it — was
/// implemented and measured on a driven x86/PCI boot with
/// `web-content-probe --churn 1`, floor at half that count:
///
/// ```text
///                                   gate off   gate on
///   slab_mib peak held (MiB)             456       584
///   slab_mib SETTLED (MiB)                64       464
///   t11sample_reclaimed_from_pages        36        26
///   distinct mappings affected             5         3
/// ```
///
/// The settled row is the verdict: **at rest the gate held 464 MiB where the
/// ungated drain holds 64 — 7.25x** — because the floor is where the drain stops,
/// so in steady state it returned no VRAM at all, which was its whole purpose.
/// What it bought was a 28 % cut in destroy-then-sample events, none of which was
/// shown to lose guest work. Do not re-add it in slot or in byte form; a byte
/// floor stops the drain at the same steady state for the same reason.
///
/// The class it was reaching for is real but narrower than a population gate can
/// address: a resident whose content exists **only** in VRAM must never be aged
/// out at all, at any population, while one the guest can re-serve from its own
/// pages costs only redundant work when it is. That distinction is a property of
/// the resident, not of how many of them there are.
///
/// Reopen signal: `past_cutoff` non-zero in the `resident_resample_*` bands, or
/// `resample_peak_ms` reaching this value, both of which mean a resident
/// survived only because the drain is throttled and had not reached it yet.
///
/// **Both have since fired and neither reopened this.** Two driven x86/PCI
/// boots, window-drag probe, quiesced: `past_cutoff` 1 of 48 738 resamples and
/// 4 of 392 534, `resample_peak_ms` 2007 and 2005. `resident_resample_band`'s
/// own doc carries why that is a cost rather than a risk — every resident the
/// drain may take has been copied out to the guest's pages, so the reading to
/// act on is `t11sample_reclaimed_from_pages` (2085 and 1937 on those boots) and
/// not these four. Do not read a `past_cutoff` line as an imminent loss.
///
/// # Uncensored, this value is 3-5x too short — and the peak above could not
/// have said so
///
/// `resident_resample_peak_ms` measures the interval between two reads of a
/// resident that survived both, so it structurally cannot report a gap longer
/// than this constant: past it the resident is gone, and the read that would
/// have closed the interval falls through to the guest's pages instead. Every
/// reading taken from that peak is therefore truncated by the very policy it is
/// being used to judge, and will always make this value look *just barely*
/// adequate.
///
/// `draw::vulkan::reclaimed_resample_band` closes the interval from the other
/// side, using the time the reclaim itself is now stamped with: a resident read
/// `since` ms after being destroyed had gone at least
/// `IDLE_TARGET_AGE_MS + since` between uses. Driven x86/PCI,
/// `web-content-probe --churn 1`, 26 fall-throughs:
///
/// ```text
///   t11sample_reclaimed_within_1x_cutoff    0     (<2 s after the destroy)
///   t11sample_reclaimed_within_2x_cutoff    2     (2-4 s after)
///   t11sample_reclaimed_within_4x_cutoff   24     (4-8 s after)
///   t11sample_reclaimed_past_4x_cutoff      0
/// ```
///
/// Not one came back within a further cutoff's worth of time, and 24 of 26 came
/// back **4-8 s after being destroyed** — so their true interval between uses is
/// `2 s + 4-8 s` = **6-10 seconds**. Individual lines put it exactly:
/// `since_reclaim_ms=4117` and `4521` on menu-bar strips, ~6.1-6.5 s. On the
/// same boot a *surviving* resident recorded `resample_peak_ms=6445`, which
/// confirms the interval directly on a resident the policy did not truncate.
///
/// ## Two later boots put more of the mass past the last band, not less
///
/// The reading above has a floor built into it — the top band was empty, so it
/// could only say "at least 4-8 s". Two driven x86/PCI boots taken after the
/// resident-registry slot counts were retired, same probe at `-n 10 --churn 1`,
/// summing the per-window route counters rather than quoting one window:
///
/// ```text
///                                   26-event boot   boot A   boot B
///   t11sample_reclaimed_from_pages              26       54       66
///     ...within_1x_cutoff                        0        1        0
///     ...within_2x_cutoff                        2        0        0
///     ...within_4x_cutoff                       24        9        5
///     ...past_4x_cutoff                          0       44       61
/// ```
///
/// **The mass moved to the top band**, which is open-ended: 44 of 54 and 61 of
/// 66 came back more than 8 s after being destroyed, so their true interval
/// between uses is over 10 s and the bands no longer bound it from above. The
/// conclusion below is unchanged in direction and stronger in degree.
///
/// **Not attributable to retiring the slot counts.** Those walks reported
/// `evicts=0` on every boot ever measured, so they were removing nothing and the
/// idle drain — the only thing that reclaims here — is untouched. The non-pinned
/// peak moved 194 -> 223 across the same change with `evicts=0` on both sides,
/// which is workload variance rather than a count that had been holding the
/// population down. What is *not* established is how much of 26 -> 54/66 is
/// variance either; three boots is not a distribution, and the earlier one was
/// taken in a different session.
///
/// None of it is lost work: `sampled_resident_missing` is 0 on both boots and
/// the fall-through re-serves from the guest's own pages. It is redundant
/// upload, and the count of it roughly doubled.
///
/// So this is not a value with a thin margin. It is between three and five times
/// shorter than the re-use interval of the surfaces it destroys, and a strip
/// redrawn every several seconds is not an exotic guest behaviour.
///
/// # The VRAM before/after, and why the value stays at 2000 anyway
///
/// The paragraph above used to end "raising it needs its own before/after on
/// VRAM, which `peak_mib` — the registry's attachment bytes, not the device's
/// footprint — cannot supply". That measurement has now been taken, against the
/// host driver's own accounting (`nvidia-smi --query-gpu=memory.used`, sampled
/// every 2 s, minus a flat 1025 MiB idle-desktop baseline). Three driven
/// x86/PCI boots, `web-content-probe -n 10 --churn 1`, QEMU relinked between
/// arms, nothing else running:
///
/// ```text
///   age_ms   peak Δ   mean Δ   at-rest Δ   reclaims   registry peak_mib
///     2000   546 MiB  249 MiB    279 MiB          6              190
///     7000   834 MiB  413 MiB    778 MiB          5              396
///    12000   886 MiB  426 MiB    883 MiB          0              441
/// ```
///
/// **Raising it buys almost nothing until it buys everything, and the price is
/// paid at rest.** 2000 → 7000 more than doubles the at-rest footprint (+499
/// MiB) and removes one reclaim out of six. Only at 12000 does the class close,
/// for +604 MiB held for the life of the guest. The reuse-interval distribution
/// has a long tail — at 7000 the survivors came back with `since_reclaim_ms` of
/// 86 and 1010, i.e. true intervals of ~7.1 and ~8.0 s — so each increment
/// chases the tail rather than clearing it.
///
/// That trade is the one `af70d69f` already rejected once, for the same reason
/// and at a similar magnitude: a gate holding 400 MiB at rest was reverted. Six
/// avoided re-uploads across a ~90 s driven run does not pay for half a
/// gigabyte of resident VRAM, and the fall-through is not a loss — the guest's
/// own pages still hold the pixels, which `ResidentTargetSlot::gpu_only_content`
/// is what guarantees. So this stays at 2000, now as a measured decision rather
/// than an unexamined one.
///
/// One reading worth keeping separately: the registry's `peak_mib` moved
/// 190 → 441 (+251 MiB) across the same span the device footprint moved
/// 279 → 883 (+604 MiB). It understates the real cost by ~2.4x, which is the
/// concrete form of the warning that it is attachment bytes and not the
/// device's footprint. Do not size this constant from it.
///
/// What would change the answer is a workload with a working set large enough
/// that the drain is reclaiming under real pressure rather than on a timer; the
/// counts here (6, 5, 0) are small enough that only the 12000 arm's zero is
/// clearly outside run-to-run noise.
///
/// # This is the crate's only eviction bound that is an age
///
/// Every other bound governing a removal here is a capacity — a count of
/// entries or bytes. This one is a deadline, and the difference matters when
/// reasoning about what a reading means: a capacity bound firing says the
/// working set outgrew the table, while this one firing says only that time
/// passed. A second age-based eviction is the thing to watch for; give it a
/// name carrying `LIMIT`, because the bound *is* a limit on idle age.
pub(crate) const IDLE_TARGET_AGE_MS: u64 = 2000;
/// Minimum wall-clock spacing between reclaim passes. The poll path calls the
/// drain ~244×/s; without this it would empty the whole registry in well under a
/// second. At `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass this bounds reclaim to
/// ~40 residents/s — a ~260-target burst drains to baseline over ~6.5 s, gently
/// (no dispose storm that would itself be a P3 hitch).
const IDLE_DRAIN_INTERVAL_MS: u64 = 100;
/// Max non-pinned residents the idle drain reclaims per pass — bounds each drain
/// pass so a large stale set (a ~600-target burst) drains gradually instead of
/// stalling one call with hundreds of image destroys.
const IDLE_TARGET_DRAIN_MAX_PER_CALL: usize = 4;
/// Reclaimed identities remembered for [`ResourcePools::reclaimed_recent`].
///
/// Sized to comfortably span one burst's reclamations so the answer is still
/// there when the next draw samples one of them, without becoming a second
/// registry. Diagnostic memory only: a `TargetIdentity` and a discriminant.
const RECLAIM_HISTORY: usize = 256;
/// Entries in the exact-content sampled cache.
///
/// **This is the cap that evicts, and its sibling never does.** Measured on one
/// driven x86/PCI boot over a full `visual-gate` run:
/// `sampled_evict_count_cap = 2436`, `sampled_evict_byte_cap` **absent** — the
/// byte cap below is three orders of magnitude from being reached and has never
/// fired. Over the same boot: 7264 identity hits, 976 content hits, 1657 misses.
///
/// **Raising it buys nothing, and that is measured rather than argued.** The
/// same boot re-run at 256 — a 4x rise, same guest image, same `visual-gate`
/// workload:
///
/// ```text
///              misses  identity_hits  content_hits  evict_count  evict_byte
///   cap  64      1657           7264           976         2436           0
///   cap 256      1701           7302           963           67         658
/// ```
///
/// Count-cap evictions fell 97 % and **the miss count did not move** — 1657
/// against 1701, which is inside the run-to-run spread the hit columns show
/// (7264/7302, 976/963, so the two workloads were comparable). The misses are
/// therefore *content* the cache has never held, not capacity it was forced to
/// drop, and no size of this constant reaches them.
///
/// Two things follow. The frame-rate plan's D1 lever — worth an estimated
/// 0.42 ms/frame on the premise that the count cap was evicting live entries —
/// is **dead**; its premise was half right and the half that mattered is false.
/// And at 256 the byte cap starts binding instead (658 evictions against 0),
/// so the headroom the plan read as unused is real only while the entry count
/// is small.
///
/// # The same answer for the gather rail, which the reading above could not see
///
/// `sampled_cache_misses` has exactly one writer, in `find_cached_sampled`.
/// `find_gathered_sampled` misses through a `?` and bumps nothing, so the "miss
/// count did not move" column above is blind to the guest-run gather rail — a
/// different population, and the one that moves 424 MB/s. Its own miss is
/// `sampled_gather_unretained` (see
/// [`EngineCounters::sampled_gather_unvouched`]). Three driven x86/PCI Safari
/// drags, quiesced, one `vk_caps` each, ratios rather than totals because the
/// runs did unequal amounts of work:
///
/// ```text
///                    unretained  skips  miss%   evict_count  evict_byte
///   64 / 128 MB            6296   4164   60.2          5949           0
///  256 / 128 MB            5591   3133   64.1          3736        4511
///  256 / 768 MB            6212   4229   59.5          4791           0
/// ```
///
/// **Not capacity.** Four times the entries and six times the bytes leave the
/// miss rate where it started, 60.2 % against 59.5 %. The middle row is why both
/// caps had to move together and is the trap in the four-line reading above:
/// raising the count cap alone just hands the evictions to the byte cap, which
/// at the ~2.29 MB a window this workload gathers binds at ~56 entries, so the
/// effective capacity of rows one and two is the same number and the comparison
/// measures nothing.
///
/// The bottom row is the one that settles it, and it also says where to look
/// next: with no byte pressure at all, a 256-entry cache is *still* evicting
/// 4791 times on the count cap. Entries are being written, never hit and
/// dropped. That is the signature of a key that does not repeat rather than of a
/// cache that is too small.
///
/// `runtime::gather_witness` was read to confirm or kill that, and it did
/// neither — it found the reason the question was open. A `(key, generation)`
/// *is* held across binds for as long as both witness halves say the bytes have
/// not moved, so an unchanged window's key does repeat by construction. But
/// every other verdict spends a fresh generation, and a bind holding one is
/// **compelled** to miss: the identity names bytes no image was ever built from.
/// So a miss is only a cache failure when the witness vouched, and the counter
/// that was supposed to say which — `sampled_gather_unvouched` — was reading a
/// structural zero. The rows above cannot distinguish a cache dropping live
/// entries from a guest rewriting the window every frame, and neither can any
/// reading taken before that counter was fixed.
///
/// Do not raise either constant on the strength of the eviction count. Both
/// causes evict identically: a compulsory miss admits a new entry under its
/// fresh identity just as a lost one does, so 4791 evictions is what a
/// perfectly-behaved cache also looks like when the content really is changing.
///
/// # With the counter fixed, the answer is 68 % compulsory
///
/// A driven boot on the repaired instrument put the gather rail's misses at
/// `unvouched` 5389 against `unretained` 2524. Two thirds of them are binds the
/// witness had already spent the generation for, so this constant could not
/// have reached them at any value — which is why the table above reads flat and
/// why the flatness was never evidence about the cache. The remaining third is
/// genuinely this cache's, and it is a third of a rail rather than a rail.
///
/// The cause is upstream of both: `gw_refused_host_write` 5156 against
/// `gw_refused_guest_store` 14 on that boot. See
/// [`EngineCounters::sampled_gather_unvouched`] for the reading and its
/// qualifiers, and prefer removing the write to enlarging the cache.
const SAMPLED_CACHE_CAP: usize = 64;
const SAMPLED_CACHE_BYTE_CAP: usize = 128 * 1024 * 1024;
/// Max recycled sampled slots retained per geometry key in `sampled_free`. A
/// content-changing input only needs a few live at once (the CB ring is 3-deep
/// plus the one being acquired); beyond that a recycled slot is destroyed so a
/// one-off geometry cannot pin memory for the whole guest lifetime. Bounds total
/// retained memory to ~(distinct geometries × cap × slot size).
const SAMPLED_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `sampled_free` across all keys. The per-key cap alone does
/// not bound a *diverse* burst: a YouTube page-load evicts hundreds of distinct
/// sampled geometries (thumbnails), each ≤ the per-key cap, so the pool grew to
/// ~593 images, each pinning a slab sub-allocation so no block could ever empty
/// (`block_frees=0`) — the VRAM-return stall. The 593 came off a `vram sfree=`
/// census line that no longer exists in the tree, so it is history rather than a
/// number a boot can be asked for; `block_frees` is still live. This
/// global cap keeps the recycle pool from exceeding the working set; evictions
/// past it are destroyed (freeing their slab range) instead of cached.
const SAMPLED_FREE_CAP_TOTAL: usize = 64;
/// Max recycled resident-target images retained per (geometry, format) key in
/// `target_free`. A per-frame content-changing target only needs a few live at
/// once (the CB ring is 3-deep plus the frame being acquired); beyond that a
/// recycled image is destroyed so a one-off geometry cannot pin VRAM for the
/// guest lifetime. Bounds retained memory to ~(distinct geometries × cap ×
/// image size).
const TARGET_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `target_free` across all keys — same diverse-burst reasoning
/// as `SAMPLED_FREE_CAP_TOTAL`.
const TARGET_FREE_CAP_TOTAL: usize = 32;
/// Live entries in the scratch target pool `targets`, past which
/// `acquire_target` evicts the oldest and destroys its image, view and
/// framebuffer.
///
/// **What the eviction costs is settled; what the number should be is not.**
/// The eviction is recomputable rather than lossy, and the key is why:
/// `(TargetKey, render_pass)` is geometry plus pass
/// identity, carrying no guest resource id, so a slot here is scratch for one
/// draw rather than any resource's content. Evicting one costs an image
/// creation, never a pixel.
///
/// The number was a bare `32` written inline with the comment `// Cap target
/// pool`, and no basis for it survives anywhere. It is left at 32 deliberately:
/// moving a bound whose reach has never been measured trades one arbitrary
/// number for another, and this one is at least the number every boot so far
/// ran on.
///
/// **The reach is now measurable and was not before.** The pool's occupancy had
/// exactly one sampling point, `vulkan_guest_reset`'s `pooled_targets=`, which
/// fires at guest reset when the pool is empty by construction — so its zero was
/// an artifact of where it was sampled, and reading it as "the cap never binds"
/// is the trap `AGENTS.md` names. `acquire_target` now bands the occupancy on
/// entry — `target_pool_depth_q1..q4`, quarters of this constant, so a series
/// taken before a change to it stays comparable in the terms that matter — and
/// counts each eviction as `target_pool_evict`. Band the requested reach from a
/// driven boot before moving this.
///
/// The band is on entry rather than beside the cap on purpose: taken after the
/// hit early-return it would count only misses, and its zero would mean either
/// "never called" or "always hit", which a reader cannot separate. Taken on
/// entry, **all four bands reading zero means the function did not run**.
///
/// **First reading, driven x86/PCI boot, web-content probe: all four bands
/// absent.** Not zero — absent, the route never appearing in any of the 71
/// census windows, while `mrt_draw_single` summed 21642 over 58 of them. So the
/// engine drew hard and `acquire_target` was never called once: on this workload
/// every draw takes the resident-target path and this pool is never populated at
/// all. The cap does not merely fail to bind; there is nothing for it to bind.
///
/// That is a narrower claim than it looks. It is one workload on one pathway,
/// and it says nothing about the arm/Metal pathway or about a guest that drives
/// the `else` branch in `exec.rs` — a draw with no resident target. Anyone
/// moving this number needs a workload that reaches here first, because on the
/// evidence available today the number is unexercised rather than adequate.
///
/// One property to weigh when it is moved: the eviction is **FIFO, not LRU** —
/// `target_order.first()` is the oldest *created*, not the least recently used.
/// If the cap ever binds, a hot geometry that was created early is evicted
/// while still in use every frame, and then re-created and pushed to the back.
/// That is a thrash rather than a cache, and it is a policy question the bands
/// above are what decide is worth answering.
const TARGET_POOL_MAX_ENTRIES: usize = 32;
/// Max recycled transient compute-storage images retained per geometry key in
/// `storage_image_free`. Same reuse logic as the render recycle pools: a
/// same-geometry compute dispatch reuses a pooled image instead of a fresh
/// `vkAllocateMemory`, but a diverse workload cannot hoard more than this per
/// geometry.
const STORAGE_IMAGE_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `storage_image_free` across all keys. Lower than the render
/// pools (`SAMPLED_FREE_CAP_TOTAL=64` / `TARGET_FREE_CAP_TOTAL=32`) because
/// compute-storage residency churns far less than per-frame render targets — and,
/// crucially, unlike the slab-backed render pools each storage slot is a
/// standalone `vkAllocateMemory` (not a slab sub-range), so an *uncapped* pool
/// leaks whole device allocations, not just slab fragmentation. Before this cap
/// the per-dispatch retire path (`drain_cleanup`) pushed unconditionally, so an
/// all-new-geometry compute workload (a diff-heavy / CoreImage / blur burst) grew
/// the pool without bound. Past the cap the displaced slot is destroyed, freeing
/// its VkDeviceMemory.
const STORAGE_IMAGE_FREE_CAP_TOTAL: usize = 16;
/// A bounded per-key free list of reusable GPU objects, with the census that
/// says whether its own caps are what is limiting reuse.
///
/// # One discipline, three pools
///
/// Resident targets, sampled images and transient compute storage all want the
/// same thing: hand a retired object back keyed by the geometry that makes it
/// reusable, take one back on the next create of that geometry, and bound the
/// hoard two ways. That policy used to be written out three times, once per
/// pool, each with its own four counters — a shape whose own doc comments said
/// "mirrors `try_recycle_sampled`". It is written once here, and the pools
/// differ only in their key type and their two caps.
///
/// It is also where a fourth pool joins. `create_transient_depth` is the one
/// render target in this engine that never recycles, and it allocates the most
/// by two orders of magnitude; its doc asks for exactly this — one discipline
/// the depth path also uses, not a fourth pool beside the others.
///
/// # The two caps
///
/// `total` is tested first and it is the one that matters. The per-key cap
/// alone does not bound a *diverse* burst: hundreds of distinct keys, each on
/// its own well under `per_key`, filled `sampled_free` to ~593 images, every one
/// pinning a slab sub-allocation so no block could ever empty. The 593 is
/// history — see `SAMPLED_FREE_CAP_TOTAL` for which half of that reading a boot
/// can still produce. `per_key` is the second bound, for one geometry churning.
///
/// A high `cap_drops` beside a high `allocs` is the reading that says a cap,
/// rather than the workload, is what stopped the reuse.
struct FreePool<K, V> {
    free: HashMap<K, Vec<V>>,
    per_key: usize,
    total: usize,
    /// A take that found an entry to reuse.
    hits: u64,
    /// A take that found none, so the caller had to create.
    allocs: u64,
    /// Entries admitted for reuse.
    admits: u64,
    /// Entries handed back for destruction because a cap was full.
    cap_drops: u64,
}

impl<K: std::hash::Hash + Eq, V> FreePool<K, V> {
    fn new(per_key: usize, total: usize) -> Self {
        Self {
            free: HashMap::new(),
            per_key,
            total,
            hits: 0,
            allocs: 0,
            admits: 0,
            cap_drops: 0,
        }
    }

    /// Retained entries across every key.
    fn len(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }

    /// Offer a retired entry for reuse. `None` means it was admitted; `Some(v)`
    /// hands it back because a cap was full and the caller must destroy it.
    /// Device-free, so the routing is unit-testable without a GPU.
    fn admit(&mut self, key: K, entry: V) -> Option<V> {
        if self.len() >= self.total {
            self.cap_drops += 1;
            return Some(entry);
        }
        let list = self.free.entry(key).or_default();
        if list.len() < self.per_key {
            list.push(entry);
            self.admits += 1;
            None
        } else {
            self.cap_drops += 1;
            Some(entry)
        }
    }

    /// Return an entry with **no cap check**.
    ///
    /// For the two end-of-submit drains — `recycle_sampled` and
    /// `recycle_storage_images` — which return every live slot at once and whose
    /// signatures carry no `ash::Device`, so they cannot destroy an entry a cap
    /// would reject. The caps therefore bound only the deferred
    /// `DeferredHandle::Recycle*` route; what bounds this one is
    /// `trim_recycle_pools` on the idle drain. That asymmetry is real and was
    /// measured: one 4x4K video session held `sfree=203` against a `total` of 64.
    fn push_uncapped(&mut self, key: K, entry: V) {
        self.free.entry(key).or_default().push(entry);
    }

    /// Take a reusable entry for `key`, splitting the reuse/fresh-create census
    /// so a boot can prove a per-frame realloc storm collapsed (`allocs` ≈ 0).
    fn take(&mut self, key: &K) -> Option<V> {
        let got = self.free.get_mut(key).and_then(Vec::pop);
        if got.is_some() {
            self.hits += 1;
        } else {
            self.allocs += 1;
        }
        got
    }

    /// Take any retained entry, for the idle trim that drains the pool toward
    /// empty. Not a reuse, so it does not move the hit/alloc split.
    fn pop_any(&mut self) -> Option<V>
    where
        K: Clone,
    {
        pop_any_pool_entry(&mut self.free)
    }

    /// Empty the pool, yielding every retained entry for the caller to destroy.
    /// The census is cumulative and survives, so a teardown does not erase what
    /// the boot measured.
    fn drain(&mut self) -> impl Iterator<Item = V> + '_ {
        self.free.drain().flat_map(|(_, list)| list)
    }

    /// Retained entries under one key — what the per-key cap bounds. Only the
    /// cap tests ask this; production code never needs a single key's depth.
    #[cfg(test)]
    fn count_for(&self, key: &K) -> usize {
        self.free.get(key).map_or(0, Vec::len)
    }

    /// `(hits, allocs, admits, cap_drops)` for the counter snapshot.
    fn stats(&self) -> (u64, u64, u64, u64) {
        (self.hits, self.allocs, self.admits, self.cap_drops)
    }
}

/// Images destroyed from the recycle pools per idle-drain pass. The recycle pools
/// exist for *active* per-frame reuse; at idle (the drain only fires after
/// `IDLE_TARGET_AGE_MS` of no touch) they are pure retained VRAM, so each pass
/// also trims them toward empty. Bounded like the registry drain so a large pool
/// drains gradually (no dispose storm) and refills a few-per-frame when activity
/// resumes (no re-alloc hitch).
const IDLE_RECYCLE_TRIM_PER_PASS: usize = 8;

/// Consecutive zero-victim idle-drain passes required before the HOST_VISIBLE
/// buffer pools (`staging_free`/`readback_free`) are trimmed. Unlike the image
/// pools (cheap slab suballocation refill), a trimmed staging buffer costs a
/// full `vkAllocateMemory` when the next upload refills it — on the upload hot
/// path that spikes inter-VBL latency. Gating on N consecutive settled passes
/// (drain interval `IDLE_DRAIN_INTERVAL_MS`) ensures a single quiet pass during
/// active video — where old frame RTs mostly but not always age out each pass —
/// cannot trigger a mid-playback buffer re-alloc. At true idle the counter
/// climbs and the buffers drain to zero within a few hundred ms of settling.
const SETTLED_PASSES_FOR_BUFFER_TRIM: u32 = 3;

/// Empty slab blocks retained at idle. `slab::SLAB_KEEP_EMPTY` (2) is the churn
/// buffer the hot release path keeps mid-burst; at *settled* idle the drain
/// trims all the way to zero so no empty `SLAB_SIZE` block sits resident for a
/// long idle desktop. The hot-path buffer still absorbs active churn (blocks
/// full of live content are never empty, so this never frees a working block);
/// only a block that has genuinely gone empty and stayed empty across the drain
/// interval is returned. Re-allocating on the next burst is measured hitch-free
/// (block allocation during a quad-4K load never moved the per-frame hitch
/// proxy), and at true idle no burst reuses a spare — so a retained spare is
/// pure waste. Minimising idle VRAM is the explicit goal.
const IDLE_SLAB_KEEP_EMPTY: usize = 0;

/// Pop one entry from the LARGEST non-empty bucket of a size-keyed recycle pool.
///
/// The buffer pools are keyed by power-of-two byte size, and the idle trim that
/// drains them exists to return host memory. Taking an arbitrary bucket returns
/// an arbitrary number of bytes per destroy, and `HashMap` order is effectively
/// random, so a pass budgeted at N destroys can spend all of them on 64-byte
/// slots and return nothing. That is not hypothetical: the staging census put
/// **11 792 of 26 624** misses in the 64-byte bucket and 1 462 more at 128 bytes
/// — over half the pool's re-allocations, each costing a ~1.4 ms
/// `vkAllocateMemory` on the upload hot path, to reclaim 64 bytes.
///
/// Largest-first makes each destroy return the most it can, so the trim reaches
/// its memory target in the fewest destroys and leaves the small, cheap-to-hold,
/// constantly-reused slots alone.
fn pop_largest_pool_entry<V>(pool: &mut HashMap<u64, Vec<V>>) -> Option<V> {
    let key = *pool
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .max_by_key(|(k, _)| **k)
        .map(|(k, _)| k)?;
    let bucket = pool.get_mut(&key)?;
    let item = bucket.pop();
    if bucket.is_empty() {
        pool.remove(&key);
    }
    item
}

/// Pop one entry from any non-empty bucket of a keyed recycle pool, removing the
/// bucket when it empties so the pool does not accumulate empty `Vec`s. `None`
/// when the whole pool is empty.
fn pop_any_pool_entry<K, V>(pool: &mut HashMap<K, Vec<V>>) -> Option<V>
where
    K: Clone + std::hash::Hash + Eq,
{
    let key = pool
        .iter()
        .find(|(_, v)| !v.is_empty())
        .map(|(k, _)| k.clone())?;
    let bucket = pool.get_mut(&key)?;
    let item = bucket.pop();
    if bucket.is_empty() {
        pool.remove(&key);
    }
    item
}
/// Bucket bins for the staging census: one per power of two up to 2^31.
pub(crate) const STAGING_BUCKET_BINS: usize = 32;
/// One `staging_pool` line per this many misses.
const STAGING_MISS_EMIT_EVERY: u64 = 512;

/// Max draws per deferred-submit batch before `batch_slot` refuses joiners
/// and the run flushes + reopens. Bounds GPU-idle latency and staging-slot
/// hoarding (see `batch_slot`) while amortizing per-draw submit overhead.
///
/// # This ceiling is not what binds, and the census says what does
///
/// Driven x86/Vulkan boot, 60 s Safari drag probe, 87.6 s of census windows:
///
/// ```text
/// batch_flushes           180775
/// batch_readback_joins 106290   58.8 % of them
/// batch_flush_draws       319685    1.77 draws per batch
/// ```
///
/// **1.77 against a ceiling of 8**, so raising this constant would change
/// nothing — a batch almost never reaches it. What ends a batch is
/// `ResourcePools::begin_entry`, which calls `batch_flush` unconditionally before
/// claiming a ring slot, and every readback claims one. `batch_readback_joins`
/// is that share measured rather than inferred: **58.8 % of batch flushes are a
/// readback cutting a run of draws short**, not a batch filling up.
///
/// That looked like two costs: an extra `vkQueueSubmit` per readback, and runs
/// of draws split across several submissions. Only the first was real.
///
/// The readback paths now record the copy into the **open batch's** command
/// buffer and submit it with them, instead of flushing the batch and submitting
/// a second buffer behind it. Across a driven boot that took submissions from
/// 287 425 to 159 901 — 44 % — with no rendering change.
///
/// **Batch length did not move: 1.77 before, 1.78 after.** The prediction that
/// it would rise toward 4.3 was wrong, and why is worth keeping, because the
/// counters do not show it. A readback still *ends* the batch — it has to,
/// because it needs a fence to wait on and `batch_flush` is what submits one.
/// Appending changes how many submissions that ending costs, not whether it
/// happens.
///
/// So batch length is set by **how often a readback occurs**, not by how a
/// readback submits. 1.78 draws per batch is the guest issuing a Store about
/// every other draw. The only thing that would lengthen these runs is deferring
/// the readback past more draws, and the completion-stamp contract forbids that:
/// a Store's pixels must be in guest RAM before the stamp that claims them.
///
/// # The readback share is now nearly all of it
///
/// Same probe after the self-alias join term was dropped from
/// `engine::exec`'s `JoinTerms` ladder:
///
/// ```text
/// batch_flushes            33538   (was 55334)
/// batch_readback_joins     30471   90.9 % of them (was 58.8 %)
/// batch_flush_draws        91495   2.73 draws per batch (was 1.77)
/// ```
///
/// So the paragraph above has become the whole answer rather than most of it:
/// **91 % of batches now end at a readback**, and the remaining 9 % is every
/// other cause put together. Batch length is set by how often the guest issues
/// a Store, the completion-stamp contract forbids deferring one, and 30 471
/// readbacks is one per command stream. That is the floor this rail can reach
/// without changing what a stamp promises — not a number to tune this constant
/// against.
///
/// Before changing this constant, read `batch_flush_draws / batch_flushes`
/// against it — while the ratio sits far below, the ceiling is not the bound.
const BATCH_MAX_DRAWS: u64 = 8;

/// 128-bit content fingerprint for the sampled cache.
///
/// The sampled cache matches an incoming blob to a retained VkImage by this
/// fingerprint alone — it no longer keeps a byte copy to `memcmp` against, so
/// the width must make an accidental collision (different content, identical
/// digest, identical geometry/format key) astronomically unlikely: at 128 bits
/// the birthday bound across the 64-entry cache is ~2^-116, far below the host
/// GPU's own soft-error rate. Two independently salted `DefaultHasher`
/// (SipHash-1-3) passes over the *warm* source bytes are still strictly cheaper
/// than the old one-hash-plus-cold-full-frame-`memcmp` (which pulled the
/// retained 8 MiB copy back through DRAM on every hit).
fn sampled_content_hash(bytes: &[u8]) -> u128 {
    let mut lo = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut lo);
    let mut hi = std::collections::hash_map::DefaultHasher::new();
    // Distinct salt so the two digests are independent (else both hashers see
    // the same input and finish() to correlated values, collapsing to 64 bits).
    hi.write_u64(0x9e37_79b9_7f4a_7c15);
    bytes.hash(&mut hi);
    ((hi.finish() as u128) << 64) | lo.finish() as u128
}

/// Which pool asked for a `vkAllocateMemory`.
///
/// A `vkAllocateMemory` per draw is the render-target-recreate bug class (Safari
/// video playback crawls when every frame reallocates its target). The count is
/// the proxy for it, and it needs the site: a staging bucket that misses its
/// free list, a per-frame sampled image, and a transient depth attachment are
/// three different defects that a single fused count cannot separate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocSite {
    StorageImage,
    /// An MRT secondary color attachment — *only* those, and the sole
    /// constructor is the MRT secondary target path.
    ///
    /// It was called `ResidentColor`, which said something else entirely.
    /// Resident color targets are allocated by `registry_ensure`, which binds
    /// them through `bind_image_slab` and so counts under `SlabBlock`: one
    /// driven boot read `slab_block=41:2568` (count:MiB) against
    /// `resident_color=0:0`. A reader asking what resident color targets cost
    /// was answered zero, four columns away from the real 2.5 GiB.
    ///
    /// Zero here means no draw ever carried a second color attachment. Same
    /// boot: `mrt_draw_single=24579` with no `secondary_mrt_drop` at all — the
    /// guest never asked for MRT, rather than asking and being degraded.
    MrtSecondary,
    /// A depth buffer allocated for one draw and destroyed after it, because
    /// the pass named no guest depth texture to key a resident on. See
    /// [`AllocSite::DepthResident`] for the rail that owns the identified case;
    /// this one should read near zero, and a rising count means guests are
    /// binding depth state without a depth attachment.
    TransientDepth,
    /// A depth buffer held in the registry under the guest texture the pass
    /// bound, for as long as the guest keeps that texture.
    ///
    /// Split from [`AllocSite::TransientDepth`] rather than replacing it because
    /// the two answer different questions and a boot series spanning the change
    /// has to stay readable: this counts allocations that amortise over a
    /// texture's life, that one counts allocations that do not amortise at all.
    /// Summing them would hide exactly the ratio the split exists to show.
    DepthResident,
    /// A HOST_VISIBLE upload block, not one staging buffer.
    ///
    /// Every staging bind is a sub-allocation out of one of these
    /// ([`buffer_slab`]), so this counts blocks: a boot that once read
    /// `staging=242:134:273` (count:MiB:ms) should read a single-digit count
    /// here. The name changed with the meaning deliberately — a reader
    /// comparing a new `staging_block` figure against an old `staging` one
    /// would be comparing block allocations against buffer allocations.
    StagingBlock,
    Readback,
    ReadbackMulti,
    SlabBlock,
    /// The guest-page writeback's device-local detiling scratch. One
    /// allocation per geometry for the life of the device, so a count above a
    /// handful means the frame size is changing every flush and the grow rule
    /// is thrashing rather than settling.
    GuestScratch,
    /// A DEVICE_LOCAL block the draw-time guest gather carves its destinations
    /// from, not one destination. Same relationship to `guest_gather` binds that
    /// [`Self::StagingBlock`] has to staging ones, and read the same way: a
    /// single-digit count for a whole boot is the allocator working.
    GuestGatherBlock,
}

const ALLOC_SITE_N: usize = 10;

impl AllocSite {
    const fn idx(self) -> usize {
        match self {
            AllocSite::StorageImage => 0,
            AllocSite::MrtSecondary => 1,
            AllocSite::TransientDepth => 2,
            AllocSite::DepthResident => 3,
            AllocSite::StagingBlock => 4,
            AllocSite::Readback => 5,
            AllocSite::ReadbackMulti => 6,
            AllocSite::SlabBlock => 7,
            AllocSite::GuestScratch => 8,
            AllocSite::GuestGatherBlock => 9,
        }
    }
}

const ALLOC_SITE_NAMES: [&str; ALLOC_SITE_N] = [
    "storage_image",
    "mrt_secondary",
    "transient_depth",
    "depth_resident",
    "staging_block",
    "readback",
    "readback_multi",
    "slab_block",
    "guest_scratch",
    "guest_gather_block",
];

static ALLOC_SITE_COUNT: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] =
    [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
static ALLOC_SITE_BYTES: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] =
    [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
/// Wall clock inside `vkAllocateMemory` itself, per site.
///
/// The function that fills these has been called `allocate_memory_timed` since
/// it was written and timed nothing: it counted allocations and summed their
/// bytes. That mattered once the count became a suspect. On a near-idle x86/PCI
/// window `draw_phase` reads `stage_us=14114` over four draws with
/// `seed_upload_bytes=0.01 MB` and `allocs=14` — 14 ms of staging with no bytes
/// to stage — which points at the allocation and cannot price it. This is the
/// price.
static ALLOC_SITE_US: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] =
    [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
/// Allocations since the last emit; one line per this many, so the rate is
/// self-clocked and an idle boot stays silent.
static ALLOC_WINDOW_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const ALLOC_WINDOW_EMIT_COUNT: u64 = 64;

pub(crate) unsafe fn allocate_memory_timed(
    ctx: &DeviceContext,
    info: &vk::MemoryAllocateInfo<'_>,
    site: AllocSite,
) -> Result<vk::DeviceMemory, vk::Result> {
    let started = std::time::Instant::now();
    let result = ctx.device.allocate_memory(info, None);
    let us = started.elapsed().as_micros() as u64;
    let i = site.idx();
    ALLOC_SITE_COUNT[i].fetch_add(1, Ordering::Relaxed);
    ALLOC_SITE_BYTES[i].fetch_add(info.allocation_size, Ordering::Relaxed);
    ALLOC_SITE_US[i].fetch_add(us, Ordering::Relaxed);
    if ALLOC_WINDOW_COUNT.fetch_add(1, Ordering::Relaxed) + 1 >= ALLOC_WINDOW_EMIT_COUNT {
        ALLOC_WINDOW_COUNT.store(0, Ordering::Relaxed);
        emit_alloc_site_census();
    }
    result
}

/// Cumulative per-site allocation census: `count:mebibytes:milliseconds`.
///
/// All three are cumulative for the boot, so a reader takes deltas between two
/// lines. The third is what says whether an allocation is a bookkeeping call or
/// a stall: divided by the count it is the per-allocation cost, and divided by
/// the mebibytes it says whether that cost is the kernel providing pages.
fn emit_alloc_site_census() {
    use std::fmt::Write as _;
    let mut line = String::from("vk_alloc_sites");
    for (i, name) in ALLOC_SITE_NAMES.iter().enumerate() {
        let _ = write!(
            line,
            " {name}={}:{}:{}",
            ALLOC_SITE_COUNT[i].load(Ordering::Relaxed),
            ALLOC_SITE_BYTES[i].load(Ordering::Relaxed) >> 20,
            ALLOC_SITE_US[i].load(Ordering::Relaxed) / 1000,
        );
    }
    crate::observe::off(line);
}

/// A single host→staging step that blocked long enough to cost frames.
///
/// `draw_phase` put ~950 ms idle stalls inside the staging span of a draw and
/// could not say which of its four calls owned them — including for a 31x24
/// cursor draw, where no plausible amount of copying explains a second. These
/// are `memcpy`s into a mapped span and an allocate/map, so at this scale the
/// cost is the *mapping*, not the bytes: a `HOST_VISIBLE|DEVICE_LOCAL` span
/// whose first touch after an idle stretch wakes the device, or guest pages the
/// host has to fault back in. Naming the call and its size separates those.
///
/// Watches from `Drop` so a `?` inside the watched call still reports. Bounded
/// per boot rather than latched per key, for the same reason `draw_stall` is:
/// the distribution is the signal, and a healthy boot produces none of these.
pub(crate) struct SlowStagingWrite {
    kind: &'static str,
    bytes: u64,
    runs: usize,
    started: std::time::Instant,
}

/// Below this a staging step is ordinary work. A frame at 60 Hz is 16.7 ms; the
/// staging-miss census already reports means in the 0.3-5 ms band, so this is
/// well clear of the healthy population and still an order of magnitude under
/// the stall being chased.
const SLOW_STAGING_WRITE_US: u64 = 20_000;
const SLOW_STAGING_LINE_CAP: u64 = 256;
static SLOW_STAGING_LINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl SlowStagingWrite {
    pub(crate) fn watch(kind: &'static str, bytes: u64, runs: usize) -> Self {
        Self {
            kind,
            bytes,
            runs,
            started: std::time::Instant::now(),
        }
    }
}

impl Drop for SlowStagingWrite {
    fn drop(&mut self) {
        let us = self.started.elapsed().as_micros() as u64;
        if us < SLOW_STAGING_WRITE_US {
            return;
        }
        let n = SLOW_STAGING_LINES.fetch_add(1, Ordering::Relaxed);
        if n >= SLOW_STAGING_LINE_CAP {
            return;
        }
        crate::observe::off(format!(
            "staging_write_slow kind={} us={us} bytes={} runs={}{}",
            self.kind,
            self.bytes,
            self.runs,
            if n + 1 == SLOW_STAGING_LINE_CAP {
                " (last: report cap reached)"
            } else {
                ""
            }
        ));
    }
}

mod images_and_registry;
mod submission_and_buffers;
/// The lease's own extent travels with its pointer; see [`ReadbackLease`].
pub(crate) use submission_and_buffers::ReadbackLease;
mod teardown;

#[cfg(test)]
mod sampled_key_tests {
    use super::SampledKey;
    use crate::backend::vulkan::engine::types::{SampledImageResource, SampledSource};
    use crate::contract::pixel_format::SwizzlePlan;

    fn resource(arrayed: bool, volume: bool, cube: bool, one_dim: bool) -> SampledImageResource {
        SampledImageResource {
            binding: 0,
            width: 7,
            height: 5,
            layers: 3,
            arrayed,
            volume,
            cube,
            one_dim,
            source: SampledSource::Bytes(std::sync::Arc::new(Vec::new())),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: SwizzlePlan::default(),
        }
    }

    /// Each shape flag reaches the key field of its own name.
    ///
    /// The four are adjacent, all `bool`, and declared in a different order on
    /// the resource (`arrayed, volume, cube, one_dim`) than on the key
    /// (`volume, cube, arrayed, one_dim`), so any permutation of them compiles
    /// and none of the callers could have noticed. One flag set at a time is
    /// the only pattern that pins all four: with two set, a swap between them
    /// is invisible.
    #[test]
    fn every_shape_flag_reaches_the_key_field_of_its_own_name() {
        // Set on the resource as (arrayed, volume, cube, one_dim); read back
        // off the key as (volume, cube, arrayed, one_dim).
        let one_hot = [
            (
                "arrayed",
                (true, false, false, false),
                (false, false, true, false),
            ),
            (
                "volume",
                (false, true, false, false),
                (true, false, false, false),
            ),
            (
                "cube",
                (false, false, true, false),
                (false, true, false, false),
            ),
            (
                "one_dim",
                (false, false, false, true),
                (false, false, false, true),
            ),
        ];
        for (name, (arrayed, volume, cube, one_dim), want) in one_hot {
            let k = SampledKey::of(&resource(arrayed, volume, cube, one_dim));
            assert_eq!(
                (k.volume, k.cube, k.arrayed, k.one_dim),
                want,
                "{name} did not reach the key field of its own name"
            );
        }
    }

    /// The extent and view fields carry across too, so a flag test passing
    /// cannot hide a crossed `width`/`height` beside it.
    #[test]
    fn the_extent_and_view_fields_carry_across() {
        let k = SampledKey::of(&resource(false, false, false, false));
        assert_eq!((k.width, k.height, k.layers), (7, 5, 3));
        assert_eq!(k.format, ash::vk::Format::R8G8B8A8_UNORM);
        assert_eq!(k.swizzle, SwizzlePlan::default());
    }
}

#[cfg(test)]
mod content_hash_tests {
    use super::sampled_content_hash;

    /// Identical bytes must fingerprint identically — this is what lets a repeat
    /// bind hit the retained image without the (now removed) full-frame memcmp.
    #[test]
    fn identical_content_hashes_equal() {
        let a = vec![0x11u8; 4096];
        let b = vec![0x11u8; 4096];
        assert_eq!(sampled_content_hash(&a), sampled_content_hash(&b));
    }

    /// A single differing byte must change the digest — a stale bind is the
    /// regression this guards (dropping the memcmp made the digest the sole
    /// arbiter of "same content").
    #[test]
    fn single_byte_change_flips_digest() {
        let mut a = vec![0x11u8; 4096];
        let base = sampled_content_hash(&a);
        a[2048] = 0x12;
        assert_ne!(base, sampled_content_hash(&a));
    }

    /// The two 64-bit halves must be independent: if the high half were just a
    /// copy of the low half the fingerprint would collapse to 64 bits and the
    /// birthday bound the memcmp removal relies on would not hold. Distinct
    /// content that happened to collide on 64 bits must still differ on 128.
    #[test]
    fn halves_are_independent() {
        // Different lengths and contents: high and low halves must not mirror.
        for bytes in [
            vec![0u8; 1],
            vec![0xffu8; 64],
            (0..=255u8).collect::<Vec<_>>(),
        ] {
            let h = sampled_content_hash(&bytes);
            let lo = h as u64;
            let hi = (h >> 64) as u64;
            assert_ne!(lo, hi, "halves collapsed for len={}", bytes.len());
        }
    }

    /// Length alone must not decide the digest (content matters within a fixed
    /// geometry/format key, where all blobs share one length).
    #[test]
    fn same_length_different_content_differs() {
        let a = vec![0xa0u8; 1024];
        let mut b = vec![0xa0u8; 1024];
        b[0] = 0xa1;
        assert_ne!(sampled_content_hash(&a), sampled_content_hash(&b));
    }
}

#[cfg(test)]
mod slot_span_extent_tests {
    use super::slot_span_fits;

    /// A slot span may reach the end of its slot and not one byte past it.
    ///
    /// The boundary is the whole test. `size == slot_size` is the common case —
    /// `acquire_staging` rounds the request up to a power-of-two bucket and
    /// records the *bucket* as the slot's size, so an exact-bucket span is every
    /// one that lands on a bucket — and a check spelled `<` rather than `<=`
    /// would refuse those while still passing any test that only tried a
    /// clearly-too-large size. Both directions ask this one function, so a `<`
    /// here would have broken every full-bucket staging write and every
    /// full-bucket readback at once.
    #[test]
    fn a_slot_span_may_fill_its_slot_and_not_pass_it() {
        assert!(slot_span_fits(0, 64));
        assert!(slot_span_fits(63, 64));
        assert!(slot_span_fits(64, 64), "an exact-bucket write must fit");
        assert!(!slot_span_fits(65, 64));
        assert!(!slot_span_fits(u64::MAX, 64));
        // A slot of zero bytes takes no write. `acquire_staging` raises every
        // request to at least four, so this is unreachable through the pool —
        // it is here because the rule must not depend on that.
        assert!(!slot_span_fits(1, 0));
        assert!(slot_span_fits(0, 0));
    }
}

#[cfg(test)]
mod pool_trim_order_tests {
    use super::{pop_any_pool_entry, pop_largest_pool_entry};
    use std::collections::HashMap;

    /// The buffer trim must return the most bytes it can per destroy.
    ///
    /// The pools are keyed by power-of-two byte size and the trim's budget is a
    /// COUNT of destroys, so which bucket it takes from decides how much memory a
    /// pass reclaims. `HashMap` iteration order is effectively random, so the
    /// arbitrary-bucket form could spend a whole pass on 64-byte slots — measured
    /// as 11 792 of 26 624 staging misses in that one bucket, each costing a
    /// ~1.4 ms `vkAllocateMemory` to recreate, to reclaim 64 bytes.
    ///
    /// Asserting the descending ORDER rather than one pop is the point: a single
    /// pop passes by luck with a random-order pool, which is exactly how the bug
    /// stayed invisible.
    #[test]
    fn buffer_trim_drains_the_largest_buckets_first() {
        let mut pool: HashMap<u64, Vec<u32>> = HashMap::new();
        for (bucket, n) in [(64u64, 40), (4096, 3), (1 << 22, 2), (256, 7)] {
            pool.insert(bucket, vec![bucket as u32; n]);
        }
        let mut order = Vec::new();
        while let Some(v) = pop_largest_pool_entry(&mut pool) {
            order.push(v as u64);
        }
        assert_eq!(order.len(), 52);
        assert!(
            order.windows(2).all(|w| w[0] >= w[1]),
            "trim order is not descending by bucket: {order:?}"
        );
        assert_eq!(order[0], 1 << 22);
        assert_eq!(order[order.len() - 1], 64);
        assert!(pool.is_empty(), "emptied buckets must be removed");

        // The arbitrary-bucket helper is still used by the image pools, whose
        // budget is not about bytes; it must keep draining everything.
        let mut pool: HashMap<u64, Vec<u32>> = HashMap::new();
        pool.insert(64, vec![1, 2]);
        pool.insert(4096, vec![3]);
        let mut n = 0;
        while pop_any_pool_entry(&mut pool).is_some() {
            n += 1;
        }
        assert_eq!(n, 3);
        assert!(pool.is_empty());
    }
}

/// Whether a span of `size` bytes lies inside a slot of `slot_size` bytes.
///
/// One rule for both directions. [`staging_write_ptr`] and [`read_back_slot`]
/// have the same shape and had the same hole — a `vkMapMemory` arm the driver
/// bounds and a persistent-mapping arm that inherits nothing — so they ask this
/// one question rather than each spelling it, and a third rail added later
/// cannot spell it differently. The readback *lease* asks it a third time, at
/// its delivery site rather than here, because the pointer it lends is a `usize`
/// the borrower reads after the engine lock is dropped.
///
/// # Which boot exercises this
///
/// All three rails are on the copying path, which a host that can import guest
/// RAM as a host pointer never takes — so the boot that exercises them is the
/// one with `REIMS_VGPU_GUEST_IMPORT=off`, per `AGENTS.md`, driven with the
/// window-drag probe rather than left idle. It took when `vk_caps` reports
/// `host_pointer_import=disabled_by_env` and one `OFF guest_ram_map
/// reason=guest_ram_map_no_backend_import` appears; nothing may then report a
/// bound import, which is what says nothing bound past the closed gate. Such a
/// boot puts the copying path's whole traffic through this one comparison —
/// `zc_buffer_gathered` CPU gathers through [`staging_write_ptr`],
/// `render_flush_leased` leases, `swap_rb_kb` through the swizzling writer, and
/// one `render_flush_gpu_declined` per lease, every flush taking the copying
/// route.
///
/// That boot has now been run — `fdd7b96f`, x86 PCI attach, Safari window-drag
/// probe, 25 s, quiesced machine, one `vk_caps`. It took: `disabled_by_env` on
/// the capability line and one off-channel `no_backend_import`. Traffic, summed
/// across the 44 per-window `store_routes` samples:
///
/// | route | count |
/// |---|---:|
/// | `zc_buffer_gathered` | 273 850 |
/// | `zc_buffer_imported` | **0** |
/// | `render_flush_leased` | 7 404 |
/// | `render_flush_gpu_declined` | 7 404 |
/// | `render_flush_gpu_direct` | **0** |
/// | `swap_rb_kb` | 273 112 |
///
/// The two zeros are the gate holding: nothing bound an import past it. The two
/// 7 404s are the identity this doc predicted — one decline per lease — and
/// they are the cheapest check that the reading is of the boot it claims,
/// because no other pairing in the census produces it.
///
/// Re-run after the device-local detiling path landed, to check that path had
/// not quietly moved traffic off these rails. It had not, and the boot is worth
/// keeping for what else it pins:
///
/// | route | count | says |
/// |---|---:|---|
/// | `zc_buffer_gathered` | 278 269 | the CPU gather carries the rail |
/// | `zc_buffer_imported` | 0 | nothing bound past the gate |
/// | `render_flush_gpu_direct` | 0 | nor did the writeback |
/// | `guest_write_linear` / `_rects` | 0 / 0 | the detiling path cannot run behind a closed gate |
/// | `render_flush_leased` | 7 524 | one per decline, again |
/// | `zc_buf_no_import` | 282 235 | and `zc_buf_scattered_*` **zero** |
///
/// That last row is the one to keep. Those same windows are scattered — a boot
/// with the import on classifies 98.5 % of them at 9–32 stretches — and on a
/// host that cannot import they are now counted as what actually refused them
/// rather than as their own shape. It is the counter-level statement of the
/// ordering rule in `guest_ram_map::standing_refusal`, and it is a stronger one
/// than the log's, because the log deduplicates and this does not.
///
/// A timing would not have survived being quoted across two builds. Counts do,
/// which is why only counts are here.
///
/// The three refusals not firing is the expected reading rather than a null one:
/// `acquire_staging` and `acquire_readback` round a request up to a power-of-two
/// bucket and record the *bucket* as the slot's size, so every live caller's
/// span is inside it. One firing means a caller reached a span the bucketing
/// does not cover, which is the thing worth knowing.
///
/// Split out as a plain function so the rule is reachable without a Vulkan
/// device, which is the same reason `ContextOwner::note_init_failure` is its
/// own: it is the one part of either call that is a decision rather than a
/// driver call, and the test below is the only thing that runs it on a host
/// with no GPU.
pub(crate) fn slot_span_fits(size: u64, slot_size: u64) -> bool {
    size <= slot_size
}

/// Host write pointer for a staging slot's first `size` bytes.
///
/// Staging slots are mapped for their lifetime at allocation, so this is a field
/// read. The fallback map exists for a slot that predates the persistent
/// mapping or was built by a path that does not map — it is the same
/// map-per-write the pools used to do everywhere, and it leaks nothing because
/// `vkFreeMemory` unmaps implicitly.
///
/// # Why the length is checked here and not left to the caller
///
/// The two arms did not answer the same question. `vkMapMemory` cannot map past
/// its memory object, so while that was the only arm every over-long write was
/// refused by the driver and no code here had to say so. The persistent mapping
/// is a field read and inherits nothing: it hands back a pointer good for
/// `slot.size` bytes to a caller that asked for `size`, and the three callers
/// then `copy_nonoverlapping` or `write_bytes` for the full `size` — past the
/// slot, into whatever the shared HOST_VISIBLE block put next to it. Their own
/// comments state the span "is exactly `bytes.len()`" and "is at least
/// `rgba.len()`", which is a claim about their callers rather than a check.
///
/// So the bound the driver used to enforce is enforced here, on both arms, and
/// a violation is a named refusal instead of a host-memory overrun. It costs one
/// comparison per staging write.
unsafe fn staging_write_ptr(
    ctx: &DeviceContext,
    slot: &BufferSlot,
    size: u64,
) -> Result<*mut u8, DrawError> {
    if !slot_span_fits(size, slot.size) {
        return Err(DrawError::DrawExecution(
            super::draw_execution::DrawExecutionDecline::StagingWriteBeyondSlot {
                size,
                slot_size: slot.size,
            },
        ));
    }
    if slot.mapped != 0 {
        return Ok(slot.mapped as *mut u8);
    }
    Ok(ctx
        .device
        .map_memory(slot.memory, 0, size, vk::MemoryMapFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsMapStaging, e)))? as *mut u8)
}

/// Copy the first `len` bytes out of a readback slot, invalidating first when the
/// slot's memory is not coherent.
///
/// **Every reader of a `MemoryClass::Readback` slot goes through here.** That
/// class ranks `HOST_CACHED` above `HOST_COHERENT` — a cached read is an order of
/// magnitude faster and several drivers expose no type carrying both — so the
/// mapping can hold cache lines predating the GPU's writes, and a reader that
/// skips `vkInvalidateMappedMemoryRanges` does not fail: it silently returns
/// whatever it last read through that pooled buffer.
///
/// Which is to say the failure mode is *the previous frame*, and it is not
/// theoretical. Adding the invalidate at only the draw rail left three others —
/// `read_target_inner`, the compute storage-buffer readback and the compute
/// storage-image readback — and two GPU parity cases went red with the second of
/// two renders reporting the first one's pixels (`a_view_swizzle_…` read the
/// identity result out of the swizzled draw; `depth_test_honored_on_resident_…`
/// scored `Less` and `Greater` as equal). One reader means the next rail added
/// cannot reintroduce that.
///
/// `WHOLE_SIZE` needs no `nonCoherentAtomSize` rounding, and the copy allocates
/// uninitialized: every one of `len` bytes is written before the length becomes
/// visible, so pre-zeroing a full frame is pure waste on a guest-blocking path.
///
/// A slot that is already mapped is read through that mapping. `vkMapMemory` on
/// an already-mapped memory object fails `VK_ERROR_MEMORY_MAP_FAILED`, and the
/// compute rail reads *back* out of slots it acquired from the staging pool
/// (`acquire_staging`, which maps for the slot's lifetime) — so the persistent
/// staging mapping had made 8 of `vk_engine_compute`'s 14 cases fail with
/// `reason=vk_compute_exec_map_storage_readback
/// vk_result=Mapping_of_a_memory_object_has_failed`, i.e. every SSBO dispatch on
/// this host. This mirrors `staging_write_ptr`'s rule for the write direction.
///
/// `map_op` and `invalidate_op` stay per-rail so an exhaustion or a driver
/// refusal still names which readback failed.
///
/// # The length is checked for the reason the write direction's is
///
/// The mirror holds all the way down, including the hole. `vkMapMemory` cannot
/// map past its memory object, so the mapping arm is bounded by the driver; the
/// persistent arm is a field read that inherits nothing, and
/// `copy_mapped_output` then reads the full `len` from it. Reading past the slot
/// is worse than writing past it in one respect — the bytes beyond belong to
/// whatever the host slab carved next in the shared block, and they end up in a
/// `Vec` this device hands back — so this refuses before the pointer is formed,
/// on both arms, through [`slot_span_fits`].
pub(super) unsafe fn read_back_slot(
    ctx: &DeviceContext,
    slot: &BufferSlot,
    len: u64,
    map_op: VkOp,
    invalidate_op: VkOp,
) -> Result<Vec<u8>, DrawError> {
    if !slot_span_fits(len, slot.size) {
        return Err(DrawError::DrawExecution(
            super::draw_execution::DrawExecutionDecline::ReadBackBeyondSlot {
                len,
                slot_size: slot.size,
            },
        ));
    }
    let persistent = slot.mapped != 0;
    let ptr = if persistent {
        slot.mapped as *const u8
    } else {
        ctx.device
            .map_memory(slot.memory, 0, len, vk::MemoryMapFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(map_op, e)))? as *const u8
    };
    // Unmapping is only ours to do when the mapping is ours; a persistent one
    // belongs to the slot and outlives this read.
    let release = |ctx: &DeviceContext| {
        if !persistent {
            ctx.device.unmap_memory(slot.memory);
        }
    };
    if let Err(e) = invalidate_slot_for_read(ctx, slot, invalidate_op) {
        release(ctx);
        return Err(e);
    }
    let out = super::exec_compute::copy_mapped_output(ptr, len as usize);
    release(ctx);
    Ok(out)
}

/// Make the GPU's writes to `slot` visible to a host read of its mapping.
///
/// A no-op on coherent memory and a `vkInvalidateMappedMemoryRanges` otherwise;
/// [`BufferSlot::coherent`] is what decides, because `MemoryClass::Readback`
/// ranks `HOST_CACHED` above coherence and several drivers expose no type
/// carrying both.
///
/// Shared by the two ways a readback is consumed — copied out through
/// [`read_back_slot`], or read in place through a lease — so that the rule for
/// when the invalidate is owed lives in one place. A leased read owes it just
/// as much as a copied one does; the only thing the lease changes is what
/// happens *after* the bytes become visible.
pub(super) unsafe fn invalidate_slot_for_read(
    ctx: &DeviceContext,
    slot: &BufferSlot,
    invalidate_op: VkOp,
) -> Result<(), DrawError> {
    if slot.coherent {
        return Ok(());
    }
    let range = vk::MappedMemoryRange::default()
        .memory(slot.memory)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    ctx.device
        .invalidate_mapped_memory_ranges(&[range])
        .map_err(|e| DrawError::VkCall(VkCall::new(invalidate_op, e)))
}

#[cfg(test)]
mod staging_mapping_tests {
    use super::{readback_leases_outstanding, return_readback_lease, DeviceContext, ResourcePools};
    use crate::backend::vulkan::engine::counters::EngineCounters;
    use ash::vk;

    /// A staging slot carries its own host mapping, and keeps it across recycle.
    ///
    /// Every staging write used to bracket itself in
    /// `vkMapMemory`/`vkUnmapMemory` — two driver round trips per buffer bind,
    /// and the outlier tranches carry ~92 000 binds. The pool's premise is that
    /// the allocation outlives the bind; so does its mapping.
    ///
    /// The recycle half is the part worth pinning: a slot that came back from the
    /// free list with `mapped == 0` would silently fall back to map-per-write and
    /// nothing else would notice.
    #[test]
    fn a_staging_slot_keeps_one_mapping_across_recycle() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP staging mapping: no device ({e})");
                return;
            }
        };
        let counters = EngineCounters::default();
        let mut pools = ResourcePools::new();
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER;

        let first = unsafe { pools.acquire_staging(&ctx, 4096, usage, &counters) }
            .expect("a 4 KiB staging slot must be available");
        assert_ne!(first.mapped, 0, "a fresh staging slot must be mapped");

        // Written through the persistent pointer, readable through it.
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        unsafe { pools.write_staging(&ctx, &first, &payload) }.expect("write must land");
        let seen = unsafe { std::slice::from_raw_parts(first.mapped as *const u8, payload.len()) };
        assert_eq!(seen, &payload[..], "the mapping does not observe the write");

        pools.recycle_staging();
        let again = unsafe { pools.acquire_staging(&ctx, 4096, usage, &counters) }
            .expect("the recycled slot must come back");
        assert_eq!(again.buffer, first.buffer, "expected the recycled slot");
        assert_eq!(
            again.mapped, first.mapped,
            "a recycled slot lost its mapping and would map per write again"
        );

        pools.recycle_staging();
        unsafe { pools.destroy_all(&ctx.device) };
        unsafe { ctx.destroy() };
    }

    /// A cold burst of staging acquires costs a handful of `vkAllocateMemory`,
    /// not one per acquire.
    ///
    /// This is the whole claim of [`buffer_slab`]. The staging pool
    /// hits ~99.6 % of the time, so what it costs is decided by its misses, and
    /// a miss used to be a full allocation — measured at a ~0.4-0.6 ms floor
    /// whatever the size (a 64-byte miss read 421 us). Every acquire below is a
    /// cold miss: distinct buckets, nothing recycled between them. Before
    /// suballocation each one allocated, so this delta was exactly `acquires`;
    /// now the same burst fits a small number of blocks.
    ///
    /// The bound is on the *count*, not on the identity of the blocks, because
    /// the split between size classes is a tuning decision and the point of the
    /// test is that the count does not track the acquire count.
    #[test]
    fn a_cold_staging_burst_allocates_blocks_not_buffers() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP staging burst: no device ({e})");
                return;
            }
        };
        let counters = EngineCounters::default();
        let mut pools = ResourcePools::new();
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER;

        // Two passes over eleven distinct buckets from 64 B to 64 KiB. The
        // second pass re-requests each bucket while the first pass's slot is
        // still live, so nothing can be served from the free list and all 22
        // are genuine misses.
        let mut acquires = 0usize;
        for _ in 0..2 {
            let mut size = 64u64;
            while size <= 64 << 10 {
                unsafe { pools.acquire_staging(&ctx, size, usage, &counters) }
                    .expect("staging slot");
                acquires += 1;
                size <<= 1;
            }
        }
        assert_eq!(acquires, 22, "the burst must be the size this test claims");

        let allocs = counters.snapshot().allocs;
        assert!(
            allocs < acquires as u64,
            "a cold staging burst allocated {allocs} memory objects for \
             {acquires} acquires — one per acquire means the suballocator is \
             not in the path"
        );
        assert!(
            allocs <= 4,
            "{acquires} acquires totalling under 256 KiB took {allocs} blocks; \
             the size classes are fragmenting rather than packing"
        );

        pools.recycle_staging();
        unsafe { pools.destroy_all(&ctx.device) };
        unsafe { ctx.destroy() };
    }

    /// Two staging slots carved from one block address disjoint bytes.
    ///
    /// The failure this guards is the one suballocation introduces and nothing
    /// else in the suite would see: two slots that share a `VkDeviceMemory` but
    /// whose host pointers or bind offsets collide read back each other's
    /// payload, which downstream looks like corrupted geometry rather than like
    /// an allocator bug. Asserted on both halves — the mapping the CPU writes
    /// through, and the `VkBuffer` the GPU reads through — because a correct
    /// `mapped` with a wrong bind offset aliases only on the GPU side.
    #[test]
    fn two_carves_from_one_block_do_not_alias() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP staging aliasing: no device ({e})");
                return;
            }
        };
        let counters = EngineCounters::default();
        let mut pools = ResourcePools::new();
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER;

        let a = unsafe { pools.acquire_staging(&ctx, 4096, usage, &counters) }.expect("slot a");
        let b = unsafe { pools.acquire_staging(&ctx, 4096, usage, &counters) }.expect("slot b");
        assert_ne!(a.buffer, b.buffer, "two live acquires must be two buffers");

        let pa: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let pb: Vec<u8> = (0..4096u32).map(|i| ((i % 251) ^ 0xff) as u8).collect();
        unsafe { pools.write_staging(&ctx, &a, &pa) }.expect("write a");
        unsafe { pools.write_staging(&ctx, &b, &pb) }.expect("write b");

        let seen_a = unsafe { std::slice::from_raw_parts(a.mapped as *const u8, pa.len()) };
        let seen_b = unsafe { std::slice::from_raw_parts(b.mapped as *const u8, pb.len()) };
        assert_eq!(
            seen_a,
            &pa[..],
            "slot a's mapping was overwritten by slot b"
        );
        assert_eq!(
            seen_b,
            &pb[..],
            "slot b's mapping was overwritten by slot a"
        );

        // Same memory object is expected and fine; the same *bytes* are not.
        if a.memory == b.memory {
            let (oa, ob) = (a.mapped, b.mapped);
            assert!(
                oa + pa.len() <= ob || ob + pb.len() <= oa,
                "two carves of one block overlap: {oa:#x}+{} vs {ob:#x}+{}",
                pa.len(),
                pb.len()
            );
        }

        pools.recycle_staging();
        unsafe { pools.destroy_all(&ctx.device) };
        unsafe { ctx.destroy() };
    }

    /// A leased readback slot is reachable by nothing else until it is returned.
    ///
    /// The lease exists so the deferred render flush can scatter straight out of
    /// the staging buffer instead of copying it into a `Vec` first, and the only
    /// thing that makes that sound is exclusivity: while the borrow is live the
    /// slot must be in no free list, so no second acquire can hand it to a GPU
    /// copy that overwrites the bytes being read.
    ///
    /// Both halves are asserted, and the second is the one that would rot
    /// silently. A lease that is never given back is not a correctness bug on
    /// the read — it strands the slot for the process lifetime and makes every
    /// later teardown wait out its whole quiesce budget, which reads as a hang
    /// rather than as a leak.
    #[test]
    fn a_leased_readback_slot_leaves_the_pool_and_comes_back() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP readback lease: no device ({e})");
                return;
            }
        };
        let counters = EngineCounters::default();
        let mut pools = ResourcePools::new();

        let slot = unsafe { pools.acquire_readback(&ctx, 4096, &counters) }
            .expect("a 4 KiB readback slot must be available");
        assert_ne!(
            slot.mapped, 0,
            "a readback slot maps for life, or a lease could not return \
             without the device"
        );
        if !slot.cached {
            // The gate is the point of the field: where the cached type was
            // unavailable the lease declines and the copy is the faster shape.
            assert!(
                pools.lease_readback().is_none(),
                "an uncached slot must not be leased"
            );
            unsafe { pools.destroy_all(&ctx.device) };
            unsafe { ctx.destroy() };
            return;
        }

        let lease = pools.lease_readback().expect("a mapped cached slot leases");
        let token = lease.token;
        assert_eq!(
            lease.ptr, slot.mapped,
            "the lease must lend the slot's mapping"
        );
        // The extent travels with the pointer, and it is the slot's own — not
        // whatever the acquirer asked for. A lease that reported the request
        // would certify a span it had not measured.
        assert_eq!(
            lease.slot_size, slot.size,
            "a lease must report the extent of what it lends"
        );
        assert!(
            lease.slot_size >= 4096,
            "a slot acquired for 4 KiB lends at least 4 KiB"
        );
        assert_eq!(
            readback_leases_outstanding(),
            1,
            "a teardown reads this to decide whether a borrow is live"
        );
        // The exclusivity claim, stated as the thing that would break it: a
        // second acquire must not be able to reach the leased slot.
        let other = unsafe { pools.acquire_readback(&ctx, 4096, &counters) }
            .expect("a second readback slot must be available");
        assert_ne!(
            other.buffer, slot.buffer,
            "the leased slot was handed out again under a live borrow"
        );

        return_readback_lease(token);
        assert_eq!(
            readback_leases_outstanding(),
            0,
            "the borrow is over the moment the holder says so"
        );
        // Returned, then collected: the two are deliberately separate, because
        // the return may not take the engine lock and the collection needs it.
        pools.reclaim_returned_readback_leases();
        let back = unsafe { pools.acquire_readback(&ctx, 4096, &counters) }
            .expect("the returned slot must be reusable");
        assert!(
            back.buffer == slot.buffer || back.buffer == other.buffer,
            "a returned lease must rejoin the free list rather than leak"
        );

        pools.recycle_readback();
        unsafe { pools.destroy_all(&ctx.device) };
        unsafe { ctx.destroy() };
    }
}

#[cfg(test)]
mod resident_reuse_tests {
    use super::{ResidentAccess, ResidentTargetSlot};
    use crate::backend::vulkan::translate;
    use ash::vk;

    /// A slot with nothing in it but the words reuse turns on.
    fn slot(width: u32, height: u32, generation: u64, format: vk::Format) -> ResidentTargetSlot {
        ResidentTargetSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            render_pass: vk::RenderPass::null(),
            width,
            height,
            generation,
            content_ready: false,
            content_epoch: None,
            access: ResidentAccess::Untouched,
            color_format: format,
            pin_count: 0,
            gpu_only_content: false,
            last_touch_ms: 0,
        }
    }

    /// The MRT secondary path's slot is not re-usable as a primary attachment,
    /// however well its geometry lines up.
    ///
    /// Both `registry_ensure*` arms write one `registry` keyed by identity, and
    /// an identity does cross between them. The primary arm used to test a
    /// stored `bgra` bool, and a secondary slot created for an `RG16Float`
    /// vibrancy mask reads `false` there because its format is not the scanout
    /// one — so it matched a primary request for RGBA8, whose format is *also*
    /// not the scanout one, and the arm went on to build a framebuffer with an
    /// RGBA8 render pass over an `RG16Float` view. Vulkan requires those to
    /// agree. The bool is gone; [`ResidentTargetSlot::scanout_order`] is now
    /// derived from `color_format`, so it cannot answer for a slot the format
    /// test would refuse.
    #[test]
    fn a_secondary_format_slot_is_not_reused_as_a_primary_attachment() {
        let rgba = translate::pixel::resident_color(false);
        let bgra = translate::pixel::resident_color(true);
        assert_ne!(rgba, bgra, "the two bgra values must name two formats");

        let secondary = slot(64, 32, 7, vk::Format::R16G16_SFLOAT);
        assert!(
            !secondary.scanout_order(),
            "a non-scanout secondary format reads scanout_order()=false, which \
             is what made the one-bit test match"
        );
        assert!(
            !secondary.reusable_for(64, 32, 7, rgba),
            "an RG16Float image must not be handed to an RGBA8 attachment"
        );
        assert!(!secondary.reusable_for(64, 32, 7, bgra));
        assert!(
            secondary.reusable_for(64, 32, 7, vk::Format::R16G16_SFLOAT),
            "the secondary path must still get its own slot back"
        );
    }

    /// The format test is a strengthening, not a replacement: everything the
    /// geometry and generation tests rejected is still rejected, and a primary
    /// slot still matches its own request.
    #[test]
    fn geometry_generation_and_format_all_still_decide_reuse() {
        let rgba = translate::pixel::resident_color(false);
        let s = slot(64, 32, 7, rgba);
        assert!(s.reusable_for(64, 32, 7, rgba));
        assert!(!s.reusable_for(65, 32, 7, rgba), "width");
        assert!(!s.reusable_for(64, 33, 7, rgba), "height");
        assert!(!s.reusable_for(64, 32, 8, rgba), "generation");
        assert!(
            !s.reusable_for(64, 32, 7, translate::pixel::resident_color(true)),
            "format still separates the two bgra orders"
        );
    }
}
