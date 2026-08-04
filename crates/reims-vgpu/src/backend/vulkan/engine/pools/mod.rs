//! Staging / target / readback / command / descriptor pools for warm-path reuse.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use ash::vk::Handle;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::compute_execution::ComputeExecutionDecline;
use super::context::{DeviceContext, FENCE_TIMEOUT_NS};
use super::counters::EngineCounters;
use super::desc_arena::{DescriptorArena, DESC_BLOCK_MAX_SETS};
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::host_slab::{HostSlabToken, HOST_SLAB_IDLE_KEEP_EMPTY};
use super::types::{DrawError, StorageImageFormat, TargetIdentity};
use super::vk_call::{VkCall, VkOp};
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
    /// returns the range to [`super::host_slab::HostSlabPool`] and must never
    /// call `vkFreeMemory`.
    HostSlab(HostSlabToken),
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
    host_slab: &mut super::host_slab::HostSlabPool,
    slot: BufferSlot,
) {
    device.destroy_buffer(slot.buffer, None);
    match slot.backing {
        BufferBacking::Dedicated => device.free_memory(slot.memory, None),
        BufferBacking::HostSlab(token) => host_slab.release(device, token),
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

/// Geometry and stencil-aspect of the kept depth-stencil attachment.
pub(crate) type DepthStencilKey = (u32, u32, bool);
/// The image, its memory and its view, as [`ResourcePools::acquire_depth_stencil`]
/// hands them out.
pub(crate) type DepthStencilParts = (vk::Image, vk::DeviceMemory, vk::ImageView);

pub(crate) struct ResourcePools {
    /// Size-bucketed free host-visible buffers (TRANSFER_SRC | VERTEX | INDEX | STORAGE).
    staging_free: HashMap<u64, Vec<BufferSlot>>,
    /// In-use staging slots returned after submit/wait.
    staging_live: Vec<BufferSlot>,
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
    /// Readback slots handed to a reader that is consuming their mapping with
    /// the engine unlocked; see [`ResourcePools::lease_readback`].
    ///
    /// Deliberately in none of the three lists above. A leased slot must not
    /// reach a ring entry's `PendingGpuCleanup` (which would return it to
    /// `readback_free` when that entry retires) and must not be handed to a
    /// second acquire, because either one lets a GPU copy overwrite bytes a
    /// live borrow is still reading.
    readback_leased: Vec<LeasedReadback>,
    /// The depth-stencil attachment, kept across draws rather than built and
    /// dropped per draw.
    ///
    /// A Metal render pass clears its stencil once and then relies on it: one
    /// draw writes a mask, the next tests against it. Building a fresh image
    /// for every draw makes that impossible — the second draw loads an image
    /// that was never written, every fragment fails the test, and what reaches
    /// the surface is an outline with no fill. Keyed by geometry and by whether
    /// a stencil aspect is wanted; a change in either disposes the old one.
    depth_stencil_keep: Option<(DepthStencilKey, DepthStencilParts)>,
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
    /// LRU order for [`Self::compute_storage_registry`], oldest at the front.
    /// A `VecDeque` for the same reason as [`Self::registry_order`]: the cap
    /// sweep's front pop / rotate-to-back is O(1), keeping the sweep O(n).
    compute_storage_order: VecDeque<ComputeStorageResidencyKey>,
    /// Identity-keyed resident target registry (workstream D).
    registry: HashMap<TargetIdentity, ResidentTargetSlot>,
    /// LRU order for [`Self::registry`], oldest at the front. A `VecDeque` so the
    /// cap-eviction sweep's front pop / rotate-to-back is O(1) — the sweep is
    /// then O(n), not the O(n²) a `Vec` front-`remove(0)` would make it under a
    /// large pinned population (measured `reg=512` under multi-4K load).
    registry_order: VecDeque<TargetIdentity>,
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
    slab: super::slab::SlabPool,
    /// Offset suballocator for the HOST_VISIBLE upload blocks every staging
    /// buffer is carved from. Same reason as `slab` one field up, against a
    /// different measurement: a staging miss cost ~0.4-0.6 ms of
    /// `vkAllocateMemory` whatever its size, and the pool takes ~1 500 of them
    /// a boot, clustered on the first composite after idle.
    host_slab: super::host_slab::HostSlabPool,
    initialized: bool,
}

/// State of the deferred-submit draw batch (draw-batching increment 1): the
/// opener's ring slot CB stays in recording state across joinable same-target
/// draws; per-draw descriptor sets and sampled-cache admissions accumulate
/// here and seal as ONE entry at flush.
pub(crate) struct OpenBatch {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    identity: TargetIdentity,
    width: u32,
    height: u32,
    bgra: bool,
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
const RING_DEPTH: usize = 8;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct SampledKey {
    width: u32,
    height: u32,
    layers: u32,
    volume: bool,
    cube: bool,
    arrayed: bool,
    one_dim: bool,
    format: ash::vk::Format,
    swizzle: crate::contract::pixel_format::SwizzlePlan,
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
    /// This *is* the match key on the content-fallback path — no byte copy is
    /// kept, so a hit binds the retained image without a full-frame `memcmp`.
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
    /// Byte length of the content this slot was admitted with, kept only for the
    /// LRU byte-cap accounting (the bytes themselves are not retained).
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
    pub layout: vk::ImageLayout,
    pub generation_match: bool,
}

struct ResidentStorageImageSlot {
    slot: StorageImageSlot,
    generation: u32,
    layout: vk::ImageLayout,
    /// Deferred-writeback pin: the resident is the only copy of this content
    /// (guest pages are stale) — LRU eviction must skip it until the caller
    /// flushes and unpins.
    pinned: bool,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this resident's
    /// last use (admit or `acquire_resident_storage_image` hit). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the clock — so a
    /// compute-heavy burst's stale residents (a settled page's blur/decode storage
    /// images) are returned to the driver instead of pinning up to
    /// `COMPUTE_STORAGE_REGISTRY_CAP` standalone VkDeviceMemory allocations for the
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
pub(crate) fn slot_presentable(slot: &ResidentTargetSlot, width: u32, height: u32) -> bool {
    slot.content_ready && slot.bgra && slot.width == width && slot.height == height
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
    /// Last known layout (tracked for correct barriers).
    pub layout: vk::ImageLayout,
    /// Attachment format: true = B8G8R8A8_UNORM (guest scanout order), false =
    /// R8G8B8A8_UNORM. A format change forces image recreate (not just FB).
    pub bgra: bool,
    /// Concrete Vulkan attachment format. For the primary single-RT path this
    /// is derived from `bgra`; MRT secondary residents (e.g. the RG16Float
    /// vibrancy mask) carry a format `bgra` cannot express, so reuse is keyed
    /// on this exact format.
    pub color_format: vk::Format,
    /// Deferred render-Store pin count: this target's content exists only on
    /// the GPU (guest pages stale). The registry LRU sweep skips slots with a
    /// nonzero count. A count (not a bool) because a surface with several
    /// identity is pinned independently by each member's deferred window —
    /// the first member's flush must not expose the image to eviction while
    /// a peer's window is still armed.
    pub pin_count: u32,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this target's
    /// last use (admit, `registry_ensure` hit, or present touch). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the current
    /// clock — so a burst's stale targets (a settled YouTube page's thumbnail RTs)
    /// are reclaimed instead of pinning VRAM at the high `REGISTRY_CAP` for the
    /// guest lifetime, while an actively-drawn target (touched every frame) never
    /// ages out.
    pub last_touch_ms: u64,
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

/// Cap on the **non-pinned** (LRU-evictable) resident-target population — the
/// active render working set. Pinned slots (deferred-write windows, each holding
/// content only on the GPU, bounded separately by the arming rail's own window
/// cap — `metal_draw::vulkan::GVA_DEFERRED_WINDOW_CAP` for the GVA Store rail)
/// are **excluded** from this count
/// (see the eviction loops): counting them would force the still-in-use active
/// targets out whenever a compositing burst pins hundreds, thrashing exactly the
/// targets a draw is about to reuse (measured `reg=512/512 evicts=168` under a
/// YouTube page-load, ~320 pinned). Excluding them lets the active set keep its
/// full cap regardless of the pinned burst, so a burst is *absorbed* (evicts≈0)
/// instead of thrashing. Total registry is bounded by construction —
/// `REGISTRY_CAP` non-pinned plus the pinned windows, which each arming rail caps
/// itself. VRAM does not stay pinned at this ceiling: the idle drain
/// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a burst's stale
/// leftovers ~2 s after last use, returning the resident set to the ~56 idle
/// working set once the burst ends. So this is sized to absorb the burst's *live*
/// working set (measured non-pinned peak ~260 during a YouTube page-load), not to
/// hold it forever. Slots are cheap; the real VRAM guard is per-image bytes.
pub(crate) const REGISTRY_CAP: usize = 320;
/// Wall-clock milliseconds a non-pinned resident may go untouched before the
/// idle drain reclaims it. An actively-drawn target is touched every frame (and
/// the presented target is touched every poll) so it never ages out, while a
/// burst's stale targets (a settled page's thumbnail RTs) are reclaimed ~2 s
/// after last use — so `REGISTRY_CAP` can be high enough to absorb a burst (no
/// eviction thrash) without pinning that VRAM for the guest lifetime.
///
/// **Wall-clock, not publish-count:** the drain clock is fed from the poll
/// heartbeat (`device_poll`, ~244 Hz), which ticks even when the guest stops
/// compositing and issuing present publishes. A publish-count clock froze on a
/// static page — measured at zero publishes per second — so a burst's ~260 stale
/// residents (~516 MiB) never aged out and VRAM never returned to the ~1005 MiB
/// idle baseline. Real time keeps advancing regardless of guest activity.
const IDLE_TARGET_AGE_MS: u64 = 2000;
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
/// Cap for the separate compute-storage resident registry. Kept at its own value
/// (independent of the target `REGISTRY_CAP` retune) — compute storage residents
/// have their own pin lifecycle and working-set profile, and were never part of
/// the deferred-present pin-burst class that motivated the target-cap change.
const COMPUTE_STORAGE_REGISTRY_CAP: usize = 64;
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
/// ~593 images (measured `vram sfree=593`), each pinning a slab sub-allocation so
/// no block could ever empty (`block_frees=0`) — the VRAM-return stall. This
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
/// its own well under `per_key`, filled `sampled_free` to ~593 images (measured
/// `vram sfree=593`), every one pinning a slab sub-allocation so no block could
/// ever empty. `per_key` is the second bound, for one geometry churning.
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
    TransientDepth,
    /// A HOST_VISIBLE upload block, not one staging buffer.
    ///
    /// Every staging bind is a sub-allocation out of one of these
    /// ([`super::host_slab`]), so this counts blocks: a boot that once read
    /// `staging=242:134:273` (count:MiB:ms) should read a single-digit count
    /// here. The name changed with the meaning deliberately — a reader
    /// comparing a new `staging_block` figure against an old `staging` one
    /// would be comparing block allocations against buffer allocations.
    StagingBlock,
    Readback,
    ReadbackMulti,
    SlabBlock,
}

const ALLOC_SITE_N: usize = 7;

impl AllocSite {
    const fn idx(self) -> usize {
        match self {
            AllocSite::StorageImage => 0,
            AllocSite::MrtSecondary => 1,
            AllocSite::TransientDepth => 2,
            AllocSite::StagingBlock => 3,
            AllocSite::Readback => 4,
            AllocSite::ReadbackMulti => 5,
            AllocSite::SlabBlock => 6,
        }
    }
}

const ALLOC_SITE_NAMES: [&str; ALLOC_SITE_N] = [
    "storage_image",
    "mrt_secondary",
    "transient_depth",
    "staging_block",
    "readback",
    "readback_multi",
    "slab_block",
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

include!("submission_and_buffers.rs");
include!("images_and_registry.rs");
include!("teardown.rs");

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

/// Host write pointer for a staging slot's first `size` bytes.
///
/// Staging slots are mapped for their lifetime at allocation, so this is a field
/// read. The fallback map exists for a slot that predates the persistent
/// mapping or was built by a path that does not map — it is the same
/// map-per-write the pools used to do everywhere, and it leaks nothing because
/// `vkFreeMemory` unmaps implicitly.
unsafe fn staging_write_ptr(
    ctx: &DeviceContext,
    slot: &BufferSlot,
    size: u64,
) -> Result<*mut u8, DrawError> {
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
pub(super) unsafe fn read_back_slot(
    ctx: &DeviceContext,
    slot: &BufferSlot,
    len: u64,
    map_op: VkOp,
    invalidate_op: VkOp,
) -> Result<Vec<u8>, DrawError> {
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
    /// This is the whole claim of [`super::super::host_slab`]. The staging pool
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

        let (token, ptr) = pools.lease_readback().expect("a mapped cached slot leases");
        assert_eq!(ptr, slot.mapped, "the lease must lend the slot's mapping");
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
