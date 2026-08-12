//! Offset free-list core for a GPU device-memory slab suballocator.
//!
//! `BlockPlan` tracks the free byte ranges inside one `VkDeviceMemory` block.
//! The Vulkan-facing pool (added on top of this core) owns the `DeviceMemory`
//! handle plus one `BlockPlan` per block and sub-allocates many image binds
//! from each block via [`BlockPlan::carve`] / [`BlockPlan::release`]. This
//! module is the pure allocator logic — no Vulkan dependency — so the
//! alignment, splitting, and coalescing are exhaustively unit-testable in
//! isolation, where an off-by-one turns into image aliasing (a/b corruption)
//! rather than a caught assertion.
//!
//! Why it exists: a layer-tree reflow recreates ~113 all-new-geometry images in
//! a single drain tranche (kb present-thrash-proxies bug #2). One
//! `vkAllocateMemory` per image costs ~198 us = ~13.5 ms of stall the geometry
//! recycle pools cannot absorb — on a *first* burst nothing has been freed yet,
//! so there is nothing to recycle. Sub-allocating those binds from a few large
//! blocks collapses 113 allocations into a handful (ceil(total_bytes / slab)).
//!
//! Restricted to DEVICE_LOCAL *images* only. Never mixing linear buffers and
//! optimal-tiled images in one block sidesteps `bufferImageGranularity` (the
//! spec's padding rule for adjacent linear/non-linear resources), so a block
//! only ever holds mutually-granularity-safe optimal images.

/// A half-open free byte range `[start, start + len)` inside a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeRange {
    start: u64,
    len: u64,
}

/// Free-list for one memory block of `size` bytes.
///
/// Invariant: `free` is sorted by `start` and every pair of neighbours is
/// non-adjacent (fully coalesced), so `is_empty` is a single-range check and a
/// released range always merges with any touching neighbour.
pub(crate) struct BlockPlan {
    size: u64,
    free: Vec<FreeRange>,
}

impl BlockPlan {
    /// A block with the whole `size` free. A zero-size block holds nothing.
    pub(crate) fn new(size: u64) -> Self {
        let free = if size == 0 {
            Vec::new()
        } else {
            vec![FreeRange {
                start: 0,
                len: size,
            }]
        };
        Self { size, free }
    }

    /// Total capacity of the block in bytes.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// Sum of the currently-free byte ranges.
    pub(crate) fn free_bytes(&self) -> u64 {
        self.free.iter().map(|r| r.len).sum()
    }

    /// True when nothing is carved out — the block is one whole free range and
    /// can be handed back to `vkFreeMemory` without leaking a live binding.
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self.free.as_slice(), [r] if r.start == 0 && r.len == self.size)
    }

    /// Structural invariant of the free list: every range is non-empty, in
    /// bounds, strictly sorted by start, and separated from its successor by at
    /// least one live byte (fully coalesced — no two free ranges touch or
    /// overlap). This is the load-bearing safety property: while it holds, every
    /// range `carve` hands out is provably disjoint from all free ranges and
    /// from every other carved range (each carve removes its range from `free`
    /// before the next reads it), so two live images can never alias. The pool
    /// checks this after every mutation and fail-logs + poisons the block on a
    /// violation, converting an allocator logic bug into a loud, safe fallback
    /// instead of silent image aliasing (a/b corruption).
    pub(crate) fn well_formed(&self) -> bool {
        let mut prev_end: Option<u64> = None;
        for r in &self.free {
            if r.len == 0 {
                return false;
            }
            let Some(end) = r.start.checked_add(r.len) else {
                return false;
            };
            if end > self.size {
                return false;
            }
            if let Some(pe) = prev_end {
                // Strictly greater: a successor that merely touches (`start ==
                // prev_end`) should have coalesced, so equality is a bug too.
                if r.start <= pe {
                    return false;
                }
            }
            prev_end = Some(end);
        }
        true
    }

    /// Carve `size` bytes aligned to `align` (bytes) out of the block.
    ///
    /// First-fit over the free ranges. Returns the aligned start offset on
    /// success; the leading alignment pad and any trailing remainder stay free
    /// (so `release` of exactly `[offset, offset + size)` fully reverses this).
    /// Returns `None` when no free range can host the aligned request.
    pub(crate) fn carve(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let align = align.max(1);
        for i in 0..self.free.len() {
            let r = self.free[i];
            let aligned = align_up(r.start, align)?;
            // Distance from the range start to the aligned position.
            let pad = aligned.checked_sub(r.start)?;
            let need = pad.checked_add(size)?;
            if need > r.len {
                continue;
            }
            let range_end = r.start.checked_add(r.len)?;
            let carved_end = aligned.checked_add(size)?;
            let tail_len = range_end - carved_end;
            // Replace the consumed range with its (≤2) leftover fragments,
            // preserving sort order (pad precedes tail).
            self.free.remove(i);
            let mut at = i;
            if pad > 0 {
                self.free.insert(
                    at,
                    FreeRange {
                        start: r.start,
                        len: pad,
                    },
                );
                at += 1;
            }
            if tail_len > 0 {
                self.free.insert(
                    at,
                    FreeRange {
                        start: carved_end,
                        len: tail_len,
                    },
                );
            }
            return Some(aligned);
        }
        None
    }

    /// Return `[start, start + size)` to the free list, coalescing neighbours.
    ///
    /// The caller must pass the exact `size` handed to the matching `carve`
    /// (not the padded extent) — the pad was returned to the free list at carve
    /// time and coalesces back here.
    pub(crate) fn release(&mut self, start: u64, size: u64) {
        if size == 0 {
            return;
        }
        let idx = self.free.partition_point(|r| r.start < start);
        self.free.insert(idx, FreeRange { start, len: size });
        self.coalesce_around(idx);
    }

    /// Prove a token describes a currently-live range before mutating the free
    /// list. This turns a double release or corrupt token into a typed leak
    /// instead of inserting an overlapping range and only noticing after the
    /// allocator has already poisoned itself.
    pub(crate) fn release_preflight(
        &self,
        block: u32,
        start: u64,
        size: u64,
    ) -> Result<(), SlabDecline> {
        if size == 0 {
            return Err(SlabDecline::ReleaseZeroSize {
                block,
                offset: start,
            });
        }
        let end = start
            .checked_add(size)
            .ok_or(SlabDecline::ReleaseRangeOverflow {
                block,
                offset: start,
                size,
            })?;
        if end > self.size {
            return Err(SlabDecline::ReleaseRangeOutOfBounds {
                block,
                offset: start,
                size,
                block_size: self.size,
            });
        }
        if self.free.iter().any(|range| {
            let range_end = range.start.saturating_add(range.len);
            start < range_end && end > range.start
        }) {
            return Err(SlabDecline::ReleaseRangeAlreadyFree {
                block,
                offset: start,
                size,
            });
        }
        Ok(())
    }

    /// Merge the range at `idx` with an adjacent predecessor/successor. Only the
    /// two neighbours can newly touch, so this is O(1), not a full re-scan.
    fn coalesce_around(&mut self, idx: usize) {
        // Merge with successor first so index math for the predecessor stays put.
        if idx + 1 < self.free.len() {
            let cur_end = self.free[idx].start + self.free[idx].len;
            if cur_end == self.free[idx + 1].start {
                self.free[idx].len += self.free[idx + 1].len;
                self.free.remove(idx + 1);
            }
        }
        if idx > 0 {
            let prev_end = self.free[idx - 1].start + self.free[idx - 1].len;
            if prev_end == self.free[idx].start {
                self.free[idx - 1].len += self.free[idx].len;
                self.free.remove(idx);
            }
        }
    }
}

/// Round `v` up to the next multiple of `align` (any `align >= 1`, not only
/// powers of two). Returns `None` on overflow.
fn align_up(v: u64, align: u64) -> Option<u64> {
    let rem = v % align;
    if rem == 0 {
        Some(v)
    } else {
        v.checked_add(align - rem)
    }
}

use super::context::DeviceContext;
use super::counters::EngineCounters;
use ash::vk;
use ash::vk::Handle as _;
use std::collections::HashMap;

/// Slab size for shared blocks. 64 MiB holds ~7 full-HD BGRA8 targets, so a
/// reflow burst of ~113 images (~450 MiB) collapses to ~7-8 `vkAllocateMemory`
/// calls instead of 113. Kept iGPU-friendly (Intel/AMD portability directive) —
/// a larger slab over-reserves shared system RAM on integrated GPUs where every
/// resident block is real RAM held from the guest.
const SLAB_SIZE: u64 = 64 << 20;

/// Images strictly smaller than this bind from the **small** size class
/// (`SMALL_SLAB_SIZE` blocks); everything else from the large class
/// (`SLAB_SIZE`). Measured live, the working set is strongly bimodal: ~75 small
/// (<256 KiB) sampled-cache textures + glyphs, plus a handful of multi-MiB
/// render/video-frame images. Mixed first-fit interleaved the stable small
/// textures through the large churny blocks, so a settled page left 128 MiB of
/// large blocks pinned by a few tiny survivors (`max_block_free_mb=63`, one live
/// sub pinning a whole 64 MiB block). Segregating the classes lets the large
/// blocks vacate and free when their content drains, while the small textures
/// pack into a cheap 8 MiB block. 256 KiB is above the small-texture cluster
/// (bucket edge) and below every render target.
const SMALL_CLASS_MAX: u64 = 256 << 10;

/// Slab size for the small class. 8 MiB holds ~128 max-size (256 KiB) small
/// images, far more than the measured ~75 live — the whole small working set
/// fits one block. Small so a small-class block is cheap to hold and to
/// re-allocate, and iGPU-friendly (every resident block is real RAM).
const SMALL_SLAB_SIZE: u64 = 8 << 20;

/// Keep at most this many fully-empty shared blocks **of each size class**
/// resident before freeing the next emptied block back to the driver. Absorbs
/// steady-state churn without letting the reverse of a fullscreen toggle re-pay
/// the block allocation.
///
/// Per class, because a carve only reuses a block of its own class
/// ([`MemBlock::small`]). A budget counted across both let two empty 64 MiB
/// large blocks satisfy it while the next emptied small block was handed
/// straight back — so the small working set, which
/// [`MemBlock::small`] describes as the *stable* one, re-paid an 8 MiB
/// `vkAllocateMemory` on its next carve. That is the allocation this budget
/// exists to prevent. [`super::buffer_slab`] reached the same conclusion first and
/// states it at its own `empty_block_victims`.
const SLAB_KEEP_EMPTY: usize = 2;

/// A live sub-allocation handed out by [`SlabPool::acquire`]: which block, at
/// what offset, of what size, backed by which `VkDeviceMemory`. The caller
/// binds its image to `(memory, offset)`, then either registers the token
/// against the image ([`SlabPool::register`]) or releases it on bind failure
/// ([`SlabPool::release_token`]).
#[derive(Clone, Copy)]
pub(crate) struct SlabToken {
    block: u32,
    offset: u64,
    size: u64,
    pub memory: vk::DeviceMemory,
}

impl SlabToken {
    /// Bind offset for `vkBindImageMemory` (0 for a dedicated whole-block image,
    /// else the carved sub-range start).
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }
}

struct MemBlock {
    memory: vk::DeviceMemory,
    plan: BlockPlan,
    mem_type: u32,
    /// A whole-block image (its `MemoryRequirements.size` exceeded `SLAB_SIZE`);
    /// never shared and always freed the moment it empties.
    dedicated: bool,
    /// Small-size-class block (`SMALL_SLAB_SIZE`, holds only images `<
    /// SMALL_CLASS_MAX`). Kept apart from large-class blocks so the stable small
    /// working set (the sampled/glyph cache) never interleaves with — and pins —
    /// the large churny render/video-frame blocks. A carve only reuses a block of
    /// its own class.
    small: bool,
    /// Set once the free-list invariant is seen violated: the block is leaked
    /// (never carved from, never freed) so a logic bug cannot corrupt memory.
    poisoned: bool,
}

/// Offset suballocator for the engine's DEVICE_LOCAL optimal images.
///
/// Sub-allocates many image binds from a few large `VkDeviceMemory` blocks to
/// collapse the per-image `vkAllocateMemory` storm of a layer-tree reflow burst
/// (kb present-thrash-proxies bug #2). Live sub-allocations are tracked by
/// `vk::Image` handle so the free path needs only the image (no per-image field
/// threaded through every pool struct); the handle travels wherever the memory
/// does.
///
/// Safety net (no Vulkan validation layer on this host): every mutation is
/// followed by a [`BlockPlan::well_formed`] check. A violation fail-logs
/// `slab_invariant` and poisons the block, so an allocator bug degrades to a
/// leak + fresh dedicated allocation rather than silent image aliasing.
pub(crate) struct SlabPool {
    blocks: Vec<Option<MemBlock>>,
    live: HashMap<vk::Image, SlabToken>,
    /// `buffer_image_granularity` from the device; every carve is aligned to at
    /// least this so no two sub-allocations share a granularity window (the
    /// fully-safe rule regardless of resource tiling). 0 until first `acquire`.
    granularity: u64,
    block_allocs: u64,
    block_frees: u64,
    sub_allocs: u64,
    invariant_violations: u64,
}

impl SlabPool {
    /// `VkDeviceMemory` bytes this pool is holding right now, and how many of
    /// them are carved into live sub-allocations.
    ///
    /// The device-side figure this crate did not have. Every other memory
    /// reading here is either cumulative-allocated (`vk_alloc_sites`, which only
    /// ever grows and so cannot say what is held) or an *attachment* footprint
    /// computed from geometry (`registry_non_pinned_peak_bytes`, which knows
    /// nothing of tiling padding, slab rounding, or the empty blocks the pool
    /// deliberately retains). Neither can answer "did that policy change cost
    /// VRAM", which is the question every reclaim-policy decision here runs into.
    ///
    /// Exact for what it covers and honest about what it does not: this is the
    /// DEVICE_LOCAL *image* slab only. HOST_VISIBLE staging (`buffer_slab`),
    /// standalone compute-storage allocations, imported guest RAM and the present
    /// path's own allocations are outside it. It is the pool the render-target
    /// population actually lands in, which is why it is the one worth having
    /// first.
    ///
    /// `held` counts whole blocks because that is what the driver has given
    /// away — an empty block retained by `SLAB_KEEP_EMPTY` still occupies VRAM,
    /// and a reader comparing `held` against `carved` is reading exactly the
    /// retention this pool trades allocation cost for.
    pub(crate) fn held_bytes(&self) -> (u64, u64) {
        let mut held = 0u64;
        let mut carved = 0u64;
        for b in self.blocks.iter().flatten() {
            held = held.saturating_add(b.plan.size());
            carved = carved.saturating_add(b.plan.size().saturating_sub(b.plan.free_bytes()));
        }
        (held, carved)
    }

    pub(crate) fn new() -> Self {
        Self {
            blocks: Vec::new(),
            live: HashMap::new(),
            granularity: 0,
            block_allocs: 0,
            block_frees: 0,
            sub_allocs: 0,
            invariant_violations: 0,
        }
    }

    /// Acquire an offset-bound sub-allocation for a DEVICE_LOCAL optimal image.
    ///
    /// Reuses an existing shared block of the same memory type when the request
    /// fits; else allocates a new block (`SLAB_SIZE`, or a dedicated block for an
    /// image larger than a slab). The returned token's `memory`/`offset` are the
    /// bind target. `note_alloc`/timing is charged only on a real block
    /// allocation, so `engine_allocs` measures true `vkAllocateMemory` count and
    /// collapses during a burst.
    pub(crate) unsafe fn acquire(
        &mut self,
        ctx: &DeviceContext,
        ireq: &vk::MemoryRequirements,
        counters: &EngineCounters,
    ) -> Result<SlabToken, super::types::DrawError> {
        if self.granularity == 0 {
            let props = ctx.instance.get_physical_device_properties(ctx.pd);
            self.granularity = props.limits.buffer_image_granularity.max(1);
        }
        let mem_type = ctx
            .memory_type_for(ireq.memory_type_bits, ireq.size, MemoryClass::DeviceLocal)
            .ok_or({
                super::types::DrawError::Unsupported(
                    super::reason::DrawReason::NoDeviceLocalMemoryForSlab {
                        memory_type_bits: ireq.memory_type_bits,
                    },
                )
            })?;
        let align = ireq.alignment.max(1).max(self.granularity);
        let size = ireq.size;
        if size == 0 {
            return Err(super::types::DrawError::Slab(SlabDecline::ZeroSize {
                memory_type_bits: ireq.memory_type_bits,
            }));
        }

        // Oversized image: a dedicated block whose whole extent is this image.
        if size > SLAB_SIZE {
            return self.new_block(ctx, size, mem_type, align, size, true, false, counters);
        }

        // Size class routes small images to their own (small) blocks so the
        // stable small working set never pins the large churny blocks.
        let want_small = size < SMALL_CLASS_MAX;

        // Reuse a shared block of this memory type AND size class that fits.
        for i in 0..self.blocks.len() {
            let hit = match &mut self.blocks[i] {
                Some(b)
                    if !b.poisoned
                        && !b.dedicated
                        && b.small == want_small
                        && b.mem_type == mem_type =>
                {
                    b.plan.carve(size, align)
                }
                _ => None,
            };
            if let Some(offset) = hit {
                if !self.check_block(i) {
                    // The carve corrupted the free list — poison + retry fresh.
                    continue;
                }
                let memory = self.blocks[i].as_ref().expect("just carved").memory;
                self.sub_allocs += 1;
                return Ok(SlabToken {
                    block: i as u32,
                    offset,
                    size,
                    memory,
                });
            }
        }

        // No shared block of this class fits: allocate a new one, sized to the
        // request's class so small images never reserve a 64 MiB block.
        let block_size = if want_small {
            SMALL_SLAB_SIZE
        } else {
            SLAB_SIZE
        };
        self.new_block(
            ctx, block_size, mem_type, align, size, false, want_small, counters,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_block(
        &mut self,
        ctx: &DeviceContext,
        block_size: u64,
        mem_type: u32,
        align: u64,
        carve: u64,
        dedicated: bool,
        small: bool,
        counters: &EngineCounters,
    ) -> Result<SlabToken, super::types::DrawError> {
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(block_size)
                .memory_type_index(mem_type),
            AllocSite::SlabBlock,
        )
        .map_err(|result| {
            super::types::DrawError::VkCall(super::vk_call::VkCall::new(
                super::vk_call::VkOp::SlabAllocateMemory,
                result,
            ))
        })?;
        counters.note_alloc();
        self.block_allocs += 1;
        let mut plan = BlockPlan::new(block_size);
        let offset = match plan.carve(carve, align) {
            Some(o) => o,
            None => {
                // A fresh block that cannot host its own reason-for-being is a
                // logic error (align/size math); free it, do not leak.
                ctx.device.free_memory(memory, None);
                self.block_frees += 1;
                return Err(super::types::DrawError::Slab(
                    SlabDecline::FreshBlockCarve {
                        block_size,
                        carve,
                        alignment: align,
                    },
                ));
            }
        };
        let block = MemBlock {
            memory,
            plan,
            mem_type,
            dedicated,
            small,
            poisoned: false,
        };
        let idx = self.insert_block(block);
        // Always-on, low-frequency census: block events fire a handful of times
        // per reflow burst (not per image), so this proves the collapse without
        // flooding — `sub_allocs` climbing while `block_allocs` stays flat is the
        // suballocation win. Off-main-core (drain worker).
        crate::observe::off(format!(
            "slab_block ev=alloc size={block_size} dedicated={} small={} block_allocs={} \
             block_frees={} sub_allocs={} live={} violations={}",
            dedicated as u8,
            small as u8,
            self.block_allocs,
            self.block_frees,
            self.sub_allocs,
            self.live.len(),
            self.invariant_violations,
        ));
        Ok(SlabToken {
            block: idx,
            offset,
            size: carve,
            memory,
        })
    }

    fn insert_block(&mut self, block: MemBlock) -> u32 {
        if let Some(i) = self.blocks.iter().position(Option::is_none) {
            self.blocks[i] = Some(block);
            i as u32
        } else {
            self.blocks.push(Some(block));
            (self.blocks.len() - 1) as u32
        }
    }

    /// The engine may bind each `VkImage` exactly once. Checking before
    /// `vkBindImageMemory` means a stale live-map entry is rejected while the
    /// new slab token is still unallocated and the image is still unbound.
    pub(crate) fn ensure_image_unregistered(&self, image: vk::Image) -> Result<(), SlabDecline> {
        if self.live.contains_key(&image) {
            Err(SlabDecline::ImageAlreadyRegistered {
                image: image.as_raw(),
            })
        } else {
            Ok(())
        }
    }

    /// Record a successfully-bound image's sub-allocation.
    pub(crate) fn register(&mut self, image: vk::Image, token: SlabToken) {
        let previous = self.live.insert(image, token);
        debug_assert!(
            previous.is_none(),
            "ensure_image_unregistered must precede slab registration"
        );
    }

    /// Release a token whose image bind failed (never registered).
    pub(crate) unsafe fn release_token(&mut self, device: &ash::Device, token: SlabToken) {
        self.release_range(device, token.block, token.offset, token.size);
    }

    /// Release the sub-allocation held by `image`; the caller destroys the image
    /// handle itself. Returns true when the image was slab-backed (so the caller
    /// knows not to also `vkFreeMemory`). A non-slab image (e.g. an import) is a
    /// no-op returning false.
    pub(crate) unsafe fn free_image(&mut self, device: &ash::Device, image: vk::Image) -> bool {
        match self.live.remove(&image) {
            Some(token) => {
                self.release_range(device, token.block, token.offset, token.size);
                true
            }
            None => false,
        }
    }

    /// Validate a token's block and range without touching Vulkan state.
    /// `Ok(false)` is the deliberate leak policy for an already-poisoned block;
    /// the poisoning event itself was logged when it happened.
    fn release_preflight(&self, block: u32, offset: u64, size: u64) -> Result<bool, SlabDecline> {
        match self.blocks.get(block as usize).and_then(Option::as_ref) {
            Some(mem_block) if mem_block.poisoned => Ok(false),
            Some(mem_block) => {
                mem_block.plan.release_preflight(block, offset, size)?;
                Ok(true)
            }
            None => Err(SlabDecline::ReleaseBlockMissing {
                block,
                block_slots: self.blocks.len(),
            }),
        }
    }

    unsafe fn release_range(&mut self, device: &ash::Device, block: u32, offset: u64, size: u64) {
        let idx = block as usize;
        match self.release_preflight(block, offset, size) {
            Ok(true) => {}
            Ok(false) => return,
            Err(decline) => {
                crate::observe::Emit::decline("slab", &decline).fail_once(u64::from(block));
                return;
            }
        }
        let b = self.blocks[idx]
            .as_mut()
            .expect("release preflight proved the slab block exists");
        b.plan.release(offset, size);
        let (empty, dedicated, small) = (b.plan.is_empty(), b.dedicated, b.small);
        if !self.check_block(idx) {
            // A corrupt release poisoned the block; leak it (already logged).
            return;
        }
        if empty && (dedicated || self.empty_spares(small) > SLAB_KEEP_EMPTY) {
            if let Some(b) = self.blocks[idx].take() {
                device.free_memory(b.memory, None);
                self.block_frees += 1;
                crate::observe::off(format!(
                    "slab_block ev=free block_allocs={} block_frees={} sub_allocs={} live={}",
                    self.block_allocs,
                    self.block_frees,
                    self.sub_allocs,
                    self.live.len(),
                ));
            }
        }
    }

    /// Free fully-empty shared blocks beyond `keep`, returning the count freed.
    ///
    /// `keep` is per size class, matching [`Self::empty_block_victims`].
    ///
    /// [`release_range`] retains `SLAB_KEEP_EMPTY` empty blocks per class to
    /// absorb steady-state churn without re-paying a `vkAllocateMemory` on the
    /// reverse of a fullscreen toggle — but it only runs on an image *release*,
    /// so at settled idle (no releases) those empty blocks sit resident forever
    /// (each a whole `SLAB_SIZE` of held VRAM). The idle drain calls this to
    /// release them. Keeping one empty per class means a workload that cycles a
    /// single block empty↔full (one fullscreen toggle) never re-allocates; only
    /// genuinely surplus empties are returned to the driver. The engine passes
    /// `IDLE_SLAB_KEEP_EMPTY`, which is 0 — at that value every empty block goes
    /// and the class split decides nothing, so the split is load-bearing on the
    /// release path rather than here.
    ///
    /// Never touches dedicated or poisoned blocks (a dedicated block already
    /// frees itself the moment it empties; a poisoned one is deliberately leaked).
    pub(crate) unsafe fn trim_empty_blocks(&mut self, device: &ash::Device, keep: usize) -> usize {
        let victims = self.empty_block_victims(keep);
        let mut freed = 0;
        for idx in victims {
            if let Some(b) = self.blocks[idx].take() {
                device.free_memory(b.memory, None);
                self.block_frees += 1;
                freed += 1;
            }
        }
        if freed > 0 {
            crate::observe::off(format!(
                "slab_block ev=idle_trim freed={freed} block_allocs={} block_frees={} \
                 sub_allocs={} live={}",
                self.block_allocs,
                self.block_frees,
                self.sub_allocs,
                self.live.len(),
            ));
        }
        freed
    }

    /// How many fully-empty shared blocks of one size class are resident.
    ///
    /// The class is the point. [`MemBlock::small`]'s own doc records that a
    /// carve only reuses a block of its own class, so an empty block of the
    /// other class is not a spare for this one — counting it lets the budget be
    /// satisfied by memory the next carve cannot touch. The classes are also
    /// nothing alike in cost: [`SLAB_SIZE`] is 64 MiB against
    /// [`SMALL_SLAB_SIZE`]'s 8 MiB.
    fn empty_spares(&self, small: bool) -> usize {
        self.blocks
            .iter()
            .filter(|s| {
                matches!(s, Some(b)
                    if !b.dedicated && !b.poisoned && b.small == small && b.plan.is_empty())
            })
            .count()
    }

    /// Indices of surplus empty shared blocks to free — `keep` spares **of each
    /// size class**, not `keep` in total.
    ///
    /// Same rule and same reason as
    /// [`super::buffer_slab::BufferSlabPool::empty_block_victims`], which is where
    /// it was written down first. Pure — split out so the selection is
    /// unit-testable without a device.
    fn empty_block_victims(&self, keep: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        for class in [true, false] {
            let empties = self.blocks.iter().enumerate().filter_map(|(i, s)| match s {
                Some(b) if !b.dedicated && !b.poisoned && b.small == class && b.plan.is_empty() => {
                    Some(i)
                }
                _ => None,
            });
            victims.extend(empties.skip(keep));
        }
        victims.sort_unstable();
        victims
    }

    /// Verify the block's free-list invariant after a mutation; on violation,
    /// fail-log once and poison the block (leak it) so a logic bug degrades to a
    /// leak, never to image aliasing. Returns whether the block is usable.
    fn check_block(&mut self, idx: usize) -> bool {
        if let Some(Some(b)) = self.blocks.get_mut(idx) {
            if b.plan.well_formed() {
                return true;
            }
            b.poisoned = true;
            self.invariant_violations += 1;
            let decline = SlabDecline::FreeListInvariant {
                block: idx,
                size: b.plan.size(),
                free_bytes: b.plan.free_bytes(),
            };
            crate::observe::Emit::decline("slab", &decline).fail_once(idx as u64);
            return false;
        }
        false
    }

    /// Destroy every remaining block (device teardown / recreate). The caller
    /// has already destroyed all images bound into these blocks.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for b in self.blocks.drain(..).flatten() {
            device.free_memory(b.memory, None);
        }
        self.live.clear();
        self.granularity = 0;
    }
}

use super::pools::{allocate_memory_timed, AllocSite};
use crate::backend::vulkan::caps::MemoryClass;

/// A slab allocation/free-list invariant that cannot honestly masquerade as a
/// driver OOM or image-bind failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabDecline {
    FreeListInvariant {
        block: usize,
        size: u64,
        free_bytes: u64,
    },
    ZeroSize {
        memory_type_bits: u32,
    },
    FreshBlockCarve {
        block_size: u64,
        carve: u64,
        alignment: u64,
    },
    ImageAlreadyRegistered {
        image: u64,
    },
    ReleaseBlockMissing {
        block: u32,
        block_slots: usize,
    },
    ReleaseZeroSize {
        block: u32,
        offset: u64,
    },
    ReleaseRangeOverflow {
        block: u32,
        offset: u64,
        size: u64,
    },
    ReleaseRangeOutOfBounds {
        block: u32,
        offset: u64,
        size: u64,
        block_size: u64,
    },
    ReleaseRangeAlreadyFree {
        block: u32,
        offset: u64,
        size: u64,
    },
    /// A [`super::buffer_slab::BufferSlabToken`] was handed to a pool of a
    /// different [`super::buffer_slab::SlabKind`] than the one that carved it.
    /// Block indices are per pool, so accepting it would insert an overlap into
    /// a live block's free list.
    ReleaseWrongPool {
        token_kind: &'static str,
        pool_kind: &'static str,
        block: u32,
    },
}

impl crate::observe::Decline for SlabDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::FreeListInvariant { .. } => "vk_slab_free_list_invariant",
            Self::ZeroSize { .. } => "vk_slab_zero_size",
            Self::FreshBlockCarve { .. } => "vk_slab_fresh_block_carve",
            Self::ImageAlreadyRegistered { .. } => "vk_slab_image_already_registered",
            Self::ReleaseBlockMissing { .. } => "vk_slab_release_block_missing",
            Self::ReleaseZeroSize { .. } => "vk_slab_release_zero_size",
            Self::ReleaseRangeOverflow { .. } => "vk_slab_release_range_overflow",
            Self::ReleaseRangeOutOfBounds { .. } => "vk_slab_release_range_out_of_bounds",
            Self::ReleaseRangeAlreadyFree { .. } => "vk_slab_release_range_already_free",
            Self::ReleaseWrongPool { .. } => "vk_slab_release_wrong_pool",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::FreeListInvariant {
                block,
                size,
                free_bytes,
            } => vec![
                ("block", block.to_string()),
                ("size", size.to_string()),
                ("free_bytes", free_bytes.to_string()),
            ],
            Self::ZeroSize { memory_type_bits } => {
                vec![("memory_type_bits", format!("{memory_type_bits:#x}"))]
            }
            Self::FreshBlockCarve {
                block_size,
                carve,
                alignment,
            } => vec![
                ("block_size", block_size.to_string()),
                ("carve", carve.to_string()),
                ("alignment", alignment.to_string()),
            ],
            Self::ImageAlreadyRegistered { image } => {
                vec![("image", format!("{image:#x}"))]
            }
            Self::ReleaseBlockMissing { block, block_slots } => vec![
                ("block", block.to_string()),
                ("block_slots", block_slots.to_string()),
            ],
            Self::ReleaseZeroSize { block, offset } => {
                vec![("block", block.to_string()), ("offset", offset.to_string())]
            }
            Self::ReleaseRangeOverflow {
                block,
                offset,
                size,
            }
            | Self::ReleaseRangeAlreadyFree {
                block,
                offset,
                size,
            } => vec![
                ("block", block.to_string()),
                ("offset", offset.to_string()),
                ("size", size.to_string()),
            ],
            Self::ReleaseRangeOutOfBounds {
                block,
                offset,
                size,
                block_size,
            } => vec![
                ("block", block.to_string()),
                ("offset", offset.to_string()),
                ("size", size.to_string()),
                ("block_size", block_size.to_string()),
            ],
            Self::ReleaseWrongPool {
                token_kind,
                pool_kind,
                block,
            } => vec![
                ("token_kind", (*token_kind).to_string()),
                ("pool_kind", (*pool_kind).to_string()),
                ("block", block.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(SlabDecline);

#[cfg(test)]
mod tests {
    use super::*;

    /// A spare of one size class is not a spare for the other.
    ///
    /// `MemBlock::small` says a carve only reuses a block of its own class, so
    /// an empty large block does nothing for a small carve. The budget used to
    /// be counted across both: two empty 64 MiB large blocks satisfied
    /// `SLAB_KEEP_EMPTY`, and the next emptied 8 MiB small block was handed back
    /// to the driver — so the small working set, which `MemBlock::small` calls
    /// the stable one, re-paid a `vkAllocateMemory` on its next carve. That is
    /// the allocation the budget exists to prevent.
    ///
    /// The same assertions `buffer_slab`'s `the_empty_spare_budget_is_per_size_class`
    /// makes, on the allocator that did not have them.
    #[test]
    fn the_empty_spare_budget_is_per_size_class() {
        let block = |small: bool| {
            Some(MemBlock {
                memory: vk::DeviceMemory::null(),
                plan: BlockPlan::new(1024),
                mem_type: 0,
                dedicated: false,
                small,
                poisoned: false,
            })
        };
        let pool = SlabPool {
            blocks: vec![block(true), block(false), block(true), block(false)],
            live: HashMap::new(),
            granularity: 0,
            block_allocs: 4,
            block_frees: 0,
            sub_allocs: 0,
            invariant_violations: 0,
        };

        // Two of each class, all empty: one of each survives, the other two go.
        assert_eq!(pool.empty_block_victims(1), vec![2, 3]);
        assert_eq!(pool.empty_block_victims(0), vec![0, 1, 2, 3]);
        assert!(pool.empty_block_victims(2).is_empty());

        // And the release path counts the same way. Four empties across two
        // classes is two per class, so neither class is over a budget of two —
        // a total count would read four and free the block being released.
        assert_eq!(pool.empty_spares(true), 2);
        assert_eq!(pool.empty_spares(false), 2);
        assert!(pool.empty_spares(true) <= SLAB_KEEP_EMPTY);
    }

    /// A dedicated or poisoned block is not a spare, on either path.
    #[test]
    fn dedicated_and_poisoned_blocks_are_not_spares() {
        let block = |dedicated: bool, poisoned: bool| {
            Some(MemBlock {
                memory: vk::DeviceMemory::null(),
                plan: BlockPlan::new(1024),
                mem_type: 0,
                dedicated,
                small: true,
                poisoned,
            })
        };
        let pool = SlabPool {
            blocks: vec![block(true, false), block(false, true), block(false, false)],
            live: HashMap::new(),
            granularity: 0,
            block_allocs: 3,
            block_frees: 0,
            sub_allocs: 0,
            invariant_violations: 0,
        };
        assert_eq!(
            pool.empty_spares(true),
            1,
            "only the plain shared block counts"
        );
        assert_eq!(pool.empty_block_victims(0), vec![2]);
    }

    #[test]
    fn new_block_is_empty_and_full() {
        let p = BlockPlan::new(4096);
        assert!(p.is_empty());
        assert_eq!(p.free_bytes(), 4096);
        assert_eq!(p.size(), 4096);
    }

    #[test]
    fn slab_invariant_decline_names_the_poisoned_block() {
        use crate::observe::Decline as _;
        let decline = SlabDecline::FreeListInvariant {
            block: 7,
            size: 64 << 20,
            free_bytes: 13,
        };
        assert_eq!(decline.slug(), "vk_slab_free_list_invariant");
        assert_eq!(
            crate::observe::Emit::decline("slab", &decline).render(),
            "slab reason=vk_slab_free_list_invariant block=7 size=67108864 free_bytes=13"
        );
    }

    #[test]
    fn impossible_slab_allocations_do_not_masquerade_as_driver_oom() {
        use crate::observe::Decline as _;
        let zero = SlabDecline::ZeroSize {
            memory_type_bits: 0x81,
        };
        assert_eq!(zero.slug(), "vk_slab_zero_size");
        assert_eq!(zero.fields(), vec![("memory_type_bits", "0x81".into())]);

        let carve = SlabDecline::FreshBlockCarve {
            block_size: 4096,
            carve: 8192,
            alignment: 256,
        };
        assert_eq!(carve.slug(), "vk_slab_fresh_block_carve");
        assert_eq!(
            carve.fields(),
            vec![
                ("block_size", "4096".into()),
                ("carve", "8192".into()),
                ("alignment", "256".into()),
            ]
        );
    }

    #[test]
    fn slab_release_preflight_refuses_every_corrupt_token_class_before_mutation() {
        use crate::observe::Decline as _;
        let mut plan = BlockPlan::new(4096);
        let live = plan.carve(1024, 1).unwrap();
        assert_eq!(live, 0);
        assert_eq!(plan.release_preflight(7, live, 1024), Ok(()));

        for (decline, slug) in [
            (
                plan.release_preflight(7, live, 0).unwrap_err(),
                "vk_slab_release_zero_size",
            ),
            (
                plan.release_preflight(7, u64::MAX, 2).unwrap_err(),
                "vk_slab_release_range_overflow",
            ),
            (
                plan.release_preflight(7, 4000, 200).unwrap_err(),
                "vk_slab_release_range_out_of_bounds",
            ),
            (
                plan.release_preflight(7, 1024, 1).unwrap_err(),
                "vk_slab_release_range_already_free",
            ),
        ] {
            assert_eq!(decline.slug(), slug);
        }
        assert_eq!(
            plan.free_bytes(),
            3072,
            "preflight refusals must not mutate the allocator"
        );
    }

    #[test]
    fn slab_registration_rejects_a_live_image_before_a_second_bind() {
        use crate::observe::Decline as _;
        let mut pool = SlabPool::new();
        let image = vk::Image::from_raw(0x1234);
        pool.live.insert(
            image,
            SlabToken {
                block: 0,
                offset: 0,
                size: 4096,
                memory: vk::DeviceMemory::null(),
            },
        );
        let decline = pool.ensure_image_unregistered(image).unwrap_err();
        assert_eq!(decline.slug(), "vk_slab_image_already_registered");
        assert_eq!(decline.fields(), vec![("image", "0x1234".into())]);
        assert!(pool
            .ensure_image_unregistered(vk::Image::from_raw(0x5678))
            .is_ok());
    }

    #[test]
    fn missing_slab_block_has_an_exact_release_reason() {
        use crate::observe::Decline as _;
        let pool = SlabPool::new();
        let decline = pool.release_preflight(7, 0, 4096).unwrap_err();
        assert_eq!(decline.slug(), "vk_slab_release_block_missing");
        assert_eq!(
            decline.fields(),
            vec![("block", "7".into()), ("block_slots", "0".into())]
        );
    }

    #[test]
    fn well_formed_holds_through_carve_and_release_churn() {
        let mut p = BlockPlan::new(1 << 20);
        assert!(p.well_formed());
        let mut live = Vec::new();
        // Interleave carves and releases; the invariant must hold at every step.
        for i in 0..64u64 {
            let size = 1000 + (i % 7) * 333;
            let align = 1u64 << (4 + (i % 5));
            if let Some(off) = p.carve(size, align) {
                live.push((off, size));
            }
            assert!(p.well_formed(), "carve broke invariant at i={i}");
            if i % 3 == 0 && !live.is_empty() {
                let (off, size) = live.remove(live.len() / 2);
                p.release(off, size);
                assert!(p.well_formed(), "release broke invariant at i={i}");
            }
        }
        // Drain the rest; must coalesce fully back.
        for (off, size) in live {
            p.release(off, size);
            assert!(p.well_formed());
        }
        assert!(p.is_empty());
    }

    #[test]
    fn well_formed_rejects_a_hand_corrupted_free_list() {
        // A well-formed block, then inject overlapping / touching free ranges to
        // prove the check catches the exact aliasing-precursor states.
        let mut p = BlockPlan::new(1000);
        assert!(p.well_formed());
        // Two ranges that touch (should have coalesced) — a double-free smell.
        p.free = vec![
            FreeRange { start: 0, len: 500 },
            FreeRange {
                start: 500,
                len: 500,
            },
        ];
        assert!(
            !p.well_formed(),
            "touching ranges must fail (not coalesced)"
        );
        // Overlapping ranges — the aliasing precursor.
        p.free = vec![
            FreeRange { start: 0, len: 600 },
            FreeRange {
                start: 500,
                len: 400,
            },
        ];
        assert!(!p.well_formed(), "overlapping ranges must fail");
        // Out of bounds.
        p.free = vec![FreeRange {
            start: 900,
            len: 200,
        }];
        assert!(!p.well_formed(), "out-of-bounds range must fail");
        // Zero-length range.
        p.free = vec![FreeRange { start: 0, len: 0 }];
        assert!(!p.well_formed(), "zero-length range must fail");
    }

    #[test]
    fn zero_size_block_carves_nothing() {
        let mut p = BlockPlan::new(0);
        assert_eq!(p.carve(1, 1), None);
        assert_eq!(p.free_bytes(), 0);
    }

    #[test]
    fn carve_zero_is_rejected() {
        let mut p = BlockPlan::new(4096);
        assert_eq!(p.carve(0, 256), None);
    }

    #[test]
    fn sequential_carves_are_non_overlapping_and_aligned() {
        let mut p = BlockPlan::new(1 << 20);
        let a = p.carve(1000, 256).unwrap();
        let b = p.carve(1000, 256).unwrap();
        let c = p.carve(1000, 256).unwrap();
        assert_eq!(a % 256, 0);
        assert_eq!(b % 256, 0);
        assert_eq!(c % 256, 0);
        // Each carve consumes 1000 rounded up to its own alignment start; ranges
        // must not overlap.
        for (lo, hi) in [(a, b), (b, c), (a, c)] {
            let (lo, hi) = if lo < hi { (lo, hi) } else { (hi, lo) };
            assert!(lo + 1000 <= hi, "overlap: {lo}+1000 > {hi}");
        }
    }

    #[test]
    fn alignment_pad_stays_free_and_reusable() {
        let mut p = BlockPlan::new(4096);
        // Carve a tiny odd-sized block, then a large-aligned one; the pad the
        // second carve skips must be reusable by a small third carve.
        let a = p.carve(10, 1).unwrap();
        assert_eq!(a, 0);
        let b = p.carve(100, 256).unwrap();
        assert_eq!(b, 256); // aligned past the 10-byte head
                            // The [10, 256) gap (246 bytes) is free — a small carve fits it.
        let c = p.carve(200, 1).unwrap();
        assert_eq!(c, 10);
    }

    #[test]
    fn full_reverse_returns_to_empty() {
        let mut p = BlockPlan::new(1 << 16);
        let a = p.carve(4096, 4096).unwrap();
        let b = p.carve(8192, 4096).unwrap();
        let c = p.carve(4096, 4096).unwrap();
        assert!(!p.is_empty());
        p.release(b, 8192);
        p.release(a, 4096);
        p.release(c, 4096);
        assert!(
            p.is_empty(),
            "free list did not coalesce back to whole block"
        );
        assert_eq!(p.free_bytes(), 1 << 16);
    }

    #[test]
    fn release_coalesces_both_neighbours() {
        let mut p = BlockPlan::new(3000);
        let a = p.carve(1000, 1).unwrap(); // [0,1000)
        let b = p.carve(1000, 1).unwrap(); // [1000,2000)
        let c = p.carve(1000, 1).unwrap(); // [2000,3000)
        assert_eq!((a, b, c), (0, 1000, 2000));
        p.release(a, 1000);
        p.release(c, 1000);
        // Now [0,1000) and [2000,3000) free, [1000,2000) live. Releasing the
        // middle must merge all three into one [0,3000) range.
        p.release(b, 1000);
        assert!(p.is_empty());
    }

    #[test]
    fn carve_fails_when_no_range_fits() {
        let mut p = BlockPlan::new(4096);
        let _ = p.carve(3000, 1).unwrap();
        // 1096 left, request 2000 → fails, leaves the block untouched.
        assert_eq!(p.carve(2000, 1), None);
        assert_eq!(p.free_bytes(), 1096);
    }

    #[test]
    fn carve_fails_when_alignment_pad_overflows_range() {
        let mut p = BlockPlan::new(4096);
        // Consume [0, 4000); the last 96 bytes free but a 4096-aligned carve
        // there is impossible (next 4096 boundary is 4096, past the block).
        let _ = p.carve(4000, 1).unwrap();
        assert_eq!(p.carve(1, 4096), None);
    }

    #[test]
    fn reuse_after_partial_free_prefers_the_hole() {
        let mut p = BlockPlan::new(10_000);
        let a = p.carve(2000, 1).unwrap();
        let _b = p.carve(2000, 1).unwrap();
        p.release(a, 2000); // hole [0,2000)
                            // First-fit lands the next same-size carve back in the freed hole.
        let d = p.carve(2000, 1).unwrap();
        assert_eq!(d, a);
    }

    #[test]
    fn large_alignment_within_block() {
        let mut p = BlockPlan::new(1 << 20);
        let a = p.carve(1, 1).unwrap();
        assert_eq!(a, 0);
        let b = p.carve(4096, 1 << 16).unwrap();
        assert_eq!(b % (1 << 16), 0);
        assert_eq!(b, 1 << 16);
    }

    /// The idle empty-block trim keeps exactly `keep` empty shared blocks and
    /// names the rest as victims — while never selecting a dedicated block (frees
    /// itself), a poisoned block (deliberately leaked), or a block still holding a
    /// live sub-allocation (not empty). This is the selection the idle drain runs
    /// to return surplus empty `SLAB_SIZE` blocks to the driver.
    #[test]
    fn empty_block_victims_keeps_spare_and_skips_non_empty() {
        fn block(empty: bool, dedicated: bool, poisoned: bool) -> Option<MemBlock> {
            let mut plan = BlockPlan::new(SLAB_SIZE);
            if !empty {
                plan.carve(4096, 256).expect("carve for a non-empty block");
            }
            Some(MemBlock {
                memory: vk::DeviceMemory::null(),
                plan,
                mem_type: 0,
                dedicated,
                small: false,
                poisoned,
            })
        }
        let mut pool = SlabPool::new();
        pool.blocks = vec![
            block(true, false, false),  // 0: empty shared
            block(false, false, false), // 1: live — never a victim
            block(true, false, false),  // 2: empty shared
            block(true, true, false),   // 3: empty dedicated — never a victim
            block(true, false, true),   // 4: empty poisoned — never a victim
            block(true, false, false),  // 5: empty shared
        ];
        // keep=1: three empty shared blocks (0,2,5) → keep the first, free 2 & 5.
        assert_eq!(pool.empty_block_victims(1), vec![2, 5]);
        // keep=0: free all three empty shared blocks.
        assert_eq!(pool.empty_block_victims(0), vec![0, 2, 5]);
        // keep >= empties: nothing to free.
        assert!(pool.empty_block_victims(3).is_empty());
        assert!(pool.empty_block_victims(9).is_empty());
    }

    #[test]
    fn align_up_handles_non_power_of_two() {
        assert_eq!(align_up(0, 3), Some(0));
        assert_eq!(align_up(1, 3), Some(3));
        assert_eq!(align_up(3, 3), Some(3));
        assert_eq!(align_up(4, 3), Some(6));
        assert_eq!(align_up(10, 256), Some(256));
    }
}
