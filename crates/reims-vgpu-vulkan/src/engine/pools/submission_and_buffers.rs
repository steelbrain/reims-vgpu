//! The submission ring, the buffer pools it refills, and the sampled cache —
//! the [`ResourcePools`] methods whose unit is a *slot*.
//!
//! These three read as separate concerns and are one, because the ring's clock
//! is what recycles the pools. A staging, readback, sampled or storage-image
//! slot handed to a batch is not free when its caller is done with it; it rides
//! `PendingGpuCleanup` until `retire_slot` has waited that batch's fence, and
//! only then does `drain_cleanup` push it back onto a free list. Periodic
//! maintenance decides when those already-free lists may shrink.
//!
//! The image *registry* is keyed by identity rather than by slot and lives in
//! [`super::images_and_registry`]; destroying everything here is
//! [`super::teardown`]'s job.
//!
//! `use super::*` is the seam. This is an `impl` chapter of the module that
//! declares `ResourcePools` and owns its fields, not a layer beneath it.

use super::*;

/// A readback slot lent out to be read where it lies.
///
/// The three values travel together because reading the mapping needs all
/// three: `token` to give the slot back, `ptr` to read from, and `slot_size` to
/// know how far. The pointer used to travel with only the token, and the span
/// the holder read came from its own request instead — correct, since
/// `acquire_readback` rounds a request up to a bucket and records the bucket,
/// but stated nowhere the holder could check. Carrying the extent beside the
/// pointer is the same rule `staging_write_ptr` and `read_back_slot` answer
/// through `slot_span_fits`.
pub(crate) struct ReadbackLease {
    pub token: u64,
    pub ptr: usize,
    pub slot_size: u64,
    pub(super) returns: Arc<ReadbackLeaseReturns>,
}

#[cfg(test)]
mod pass_echo_delta_order {
    use super::super::{PassEcho, ResourcePools};
    use crate::engine::caches::{ColorLoadKey, PassKey};
    use crate::engine::pools::PassEchoField;
    use ash::vk;
    use ash::vk::Handle as _;

    /// One echo. `fb` is derived from the pair the way the real framebuffer
    /// cache derives it — `AdHocFramebufferKey` is `(render pass, views,
    /// extent)` — so a shape change brings a new handle with it exactly as it
    /// does on the draw path, which is the whole condition under test.
    fn echo(image: u64, variation: bool) -> PassEcho {
        let mut key = PassKey::single(ColorLoadKey::Load, vk::Format::B8G8R8A8_UNORM);
        key.color_input = variation;
        PassEcho {
            cb: vk::CommandBuffer::null(),
            compatibility: key.compatibility(),
            fb: vk::Framebuffer::from_raw(image * 2 + variation as u64),
            target_image: vk::Image::from_raw(image),
            area: (1920, 1080),
        }
    }

    /// `passdiff_compat` must mean **one image described two ways** and nothing
    /// else, which takes both of these arms to pin.
    ///
    /// The target has to subsume, or a break where the guest also moved lands in
    /// the defect bucket; and the framebuffer cannot be asked before either,
    /// because it is a function of the render pass as well as the views, so a
    /// shape flip on one image brings a new handle with it and would empty the
    /// defect bucket into the guest-action one. At ~63 µs of GPU per pass
    /// boundary — measured causally, `REIMS_VGPU_PASS_CHURN=on` moved GPU per
    /// draw from 9.25 to 67.64 µs — this is the ranking question for the largest
    /// cost in the device.
    #[test]
    fn passdiff_compat_means_one_image_described_two_ways() {
        let mut pools = ResourcePools::new();
        pools.note_pass_opened(echo(1, false));
        assert!(
            matches!(
                pools.pass_echo_delta(&echo(1, true)),
                Some(PassEchoField::Compatibility(_))
            ),
            "the same image with a different declared shape is the defect this \
             bucket exists to count, despite the framebuffer the shape brings"
        );
        assert_eq!(
            pools.pass_echo_delta(&echo(2, true)),
            Some(PassEchoField::Target),
            "a break where the image also moved is the guest changing target, \
             however the shape differs alongside it"
        );
        assert_eq!(
            pools.pass_echo_delta(&echo(2, false)),
            Some(PassEchoField::Target),
            "and at an identical shape too"
        );
        assert_eq!(
            pools.pass_echo_delta(&echo(1, false)),
            None,
            "an identical echo continues the standing pass"
        );
    }
}

impl ResourcePools {
    /// End one guest allocation's backend lifetime after open submissions.
    pub(crate) unsafe fn retire_guest_import(
        &mut self,
        device: &ash::Device,
        import_id: reims_vgpu_memory::ImportId,
    ) {
        match self.host_ram_imports.retire(import_id) {
            host_ram::ParentRetire::Ready(parent) => {
                crate::telemetry::note_route("guest_import_retired_now");
                self.dispose(device, DeferredHandle::GuestAllocation(parent));
            }
            host_ram::ParentRetire::WaitingForChildren => {
                crate::telemetry::note_route("guest_import_retired_waiting_child");
                self.dispose(device, DeferredHandle::GuestAllocationBarrier(import_id));
            }
            host_ram::ParentRetire::NotImported => {
                crate::telemetry::note_route("guest_import_retired_unimported");
                self.completed_guest_imports.push(import_id);
            }
        }
    }

    /// Take allocation identities whose backend access has ended.
    pub(crate) fn take_completed_guest_imports(&mut self) -> Vec<reims_vgpu_memory::ImportId> {
        std::mem::take(&mut self.completed_guest_imports)
    }

    pub(crate) fn guest_reset_counts(&self) -> (usize, usize, usize, usize) {
        let sampled = self.sampled_live.len()
            + self.sampled_free.len()
            + self.sampled_cache.len()
            + self.attachment_snapshot_live.len()
            + self.attachment_snapshot_free.len();
        let storage = self.storage_image_live.len()
            + self.storage_image_free.len()
            + self.compute_storage_registry.len();
        (self.registry.len(), self.targets.len(), sampled, storage)
    }

    /// The bindable range `guest_ref` names, importing its RAMBlock if this is
    /// the first reference into it.
    ///
    /// Unlike a per-window import there is nothing to displace: an
    /// import is per RAMBlock and lives as long as the device, so no caller can
    /// find the buffer it just bound freed underneath a submission in flight.
    ///
    /// # Safety
    ///
    /// `ctx` must own the device every live import was made against.
    pub(crate) unsafe fn bind_guest_ram(
        &mut self,
        ctx: &DeviceContext,
        guest_ref: &reims_vgpu_memory::GuestRef,
    ) -> Result<host_ram::BoundGuestRam, host_ram::HostRamDecline> {
        unsafe { self.host_ram_imports.bind(ctx, guest_ref) }
    }

    /// Import a RAMBlock ahead of any reference into it.
    ///
    /// # Safety
    ///
    /// `ctx` must own the device every live import was made against.
    pub(crate) unsafe fn warm_guest_ram(
        &mut self,
        ctx: &DeviceContext,
        import: &reims_vgpu_memory::GuestRamImport,
    ) -> Result<bool, host_ram::HostRamDecline> {
        unsafe { self.host_ram_imports.warm(ctx, import) }
    }

    /// How many RAMBlocks are imported, and how many bytes they cover.
    ///
    /// The count is the reading that says whether the model held: one or two
    /// for a whole boot. A count that tracks the workload is a per-resource
    /// import, which the extension does not guarantee works and which pays the
    /// driver's page pinning for an answer that never changes.
    pub(crate) fn host_ram_import_census(&self) -> (usize, usize, u64, usize, usize, usize) {
        let (ramblocks, aliases) = self.host_ram_imports.counts();
        let recycled_images = self.sampled_free.len()
            + self.attachment_snapshot_free.len()
            + self.target_free.len()
            + self.storage_image_free.len();
        (
            ramblocks,
            aliases,
            self.host_ram_imports.imported_bytes(),
            0,
            0,
            recycled_images,
        )
    }

    pub(crate) fn new() -> Self {
        Self {
            readback_lease_returns: Arc::new(ReadbackLeaseReturns::default()),
            staging_free: HashMap::new(),
            staging_live: Vec::new(),
            gather_free: HashMap::new(),
            gather_live: Vec::new(),
            cb_bound_buffers: super::CbBufferMemo::default(),
            cb_gather_owed: Vec::new(),
            cb_sampled_guest: std::collections::HashMap::new(),
            cb_graphics: super::CbGraphicsState::default(),
            cb_guest_visibility: super::CbGuestVisibility::default(),
            staging_hits: 0,
            staging_misses: 0,
            staging_miss_bins: [0; STAGING_BUCKET_BINS],
            staging_miss_us_bins: [0; STAGING_BUCKET_BINS],
            settled_staging_mark: 0,
            targets: HashMap::new(),
            target_order: Vec::new(),
            multisample_target: None,
            ad_hoc_framebuffers: HashMap::new(),
            readback_free: HashMap::new(),
            readback_live: None,
            readback_multi_live: Vec::new(),
            readback_leased: Vec::new(),
            sampled_free: FreePool::new(),
            sampled_live: Vec::new(),
            attachment_snapshot_free: FreePool::new(),
            attachment_snapshot_live: Vec::new(),
            sampled_cache: Vec::new(),
            sampled_cache_bytes: 0,
            storage_image_free: FreePool::new(),
            storage_image_live: Vec::new(),
            compute_storage_registry: HashMap::new(),
            compute_storage_order: VecDeque::new(),
            registry: HashMap::new(),
            guest_resident_authority: HashMap::new(),
            registry_order: VecDeque::new(),
            reclaimed_recent: VecDeque::new(),
            registry_non_pinned_peak: 0,
            registry_non_pinned: NonPinnedTotals::default(),
            registry_non_pinned_peak_bytes: 0,
            registry_sole_copy: NonPinnedTotals::default(),
            registry_sole_copy_peak: NonPinnedTotals::default(),
            resident_resample_peak_ms: 0,
            compute_storage_sole_copy: NonPinnedTotals::default(),
            compute_storage_sole_copy_peak: NonPinnedTotals::default(),
            idle_clock_ms: 0,
            last_maintenance_ms: 0,
            settled_maintenance_passes: 0,
            cmd_pool: vk::CommandPool::null(),
            desc_arena: DescriptorArena::empty(),
            scatter: None,
            scatter_refused: false,
            scatter_dsets: Vec::new(),
            scatter_dset_free: Vec::new(),
            slots: Vec::new(),
            cur: 0,
            in_flight: 0,
            graveyard: Vec::new(),
            completed_guest_imports: Vec::new(),
            target_free: FreePool::new(),
            open_batch: None,
            batch_max_draws: BATCH_MAX_DRAWS,
            last_pass: None,
            open_pass: None,
            slab: slab::SlabPool::new(),
            slabs: buffer_slab::BufferSlabs::new(),
            host_ram_imports: host_ram::HostRamImports::default(),
            guest_reads_in_flight: false,
            guest_write_tokens_live: std::collections::HashMap::new(),
            resident_pins_live: Vec::new(),
            compute_write_pins_live: Vec::new(),
            initialized: false,
        }
    }
    /// Advance the registry clock and report whether a bounded maintenance pass
    /// is due. The pass may release objects already outside every live resource
    /// (dead direct images and free-pool entries), but elapsed time is never an
    /// authority to destroy a live resident.
    pub(super) fn plan_idle_maintenance(&mut self, now_ms: u64) -> bool {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let now = self.idle_clock_ms;
        if now < IDLE_MAINTENANCE_START_MS
            || now.saturating_sub(self.last_maintenance_ms) < MAINTENANCE_INTERVAL_MS
        {
            return false;
        }
        self.last_maintenance_ms = now;
        true
    }

    /// Destroy up to `max` images from the image recycle pools (`sampled_free`
    /// then `target_free`) — freeing their slab sub-allocations — and up to `max`
    /// buffers from the HOST_VISIBLE `staging_free`/`readback_free` pools. Called
    /// only on a fired idle pass. Returns the total count destroyed. Terminal
    /// destroy (not re-recycle): at idle these are pure retained memory.
    ///
    /// The buffer pools matter as much as the image pools for "least host memory":
    /// they are never re-evaluated on the hot path, so at idle they hold a whole
    /// video session's upload working set forever (measured `staging_mb=177` +
    /// `readback_mb=61` frozen for 30 s+ after the tab closed). On a discrete GPU
    /// that is system RAM; on an iGPU it is shared guest RAM (portability target).
    /// Trimming here (gradual, like the image pools; they refill a-few-per-frame
    /// when uploads resume) returns it once the session ends without any hot-path
    /// re-alloc churn.
    ///
    /// `trim_buffers` gates ONLY the HOST_VISIBLE buffer pools: they re-alloc via
    /// full `vkAllocateMemory` on the upload hot path, so we only free them once
    /// idle has *settled* (see [`SETTLED_PASSES_FOR_BUFFER_TRIM`]). The image
    /// pools always trim — they refill via cheap slab suballocation.
    pub(super) unsafe fn trim_recycle_pools(
        &mut self,
        device: &ash::Device,
        max: usize,
        trim_buffers: bool,
    ) -> usize {
        let mut trimmed = 0;
        while trimmed < max {
            let Some(slot) = self.attachment_snapshot_free.pop_any() else {
                break;
            };
            self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
            trimmed += 1;
        }
        while trimmed < max {
            let Some(slot) = self.sampled_free.pop_any() else {
                break;
            };
            self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
            trimmed += 1;
        }
        while trimmed < max {
            let Some(img) = self.target_free.pop_any() else {
                break;
            };
            self.destroy_deferred_handle(
                device,
                DeferredHandle::Image {
                    image: img.image,
                    view: img.view,
                    memory: img.memory,
                },
            );
            trimmed += 1;
        }
        // Buffer pools get their own budget so image trimming never starves them,
        // and only trim once idle has settled (they re-alloc via full
        // vkAllocateMemory on the upload hot path — trimming mid-video hitches it).
        let mut buf_trimmed = 0;
        if trim_buffers {
            while buf_trimmed < max {
                let Some(slot) = pop_largest_pool_entry(&mut self.staging_free) else {
                    break;
                };
                release_buffer_slot(device, &mut self.slabs, slot);
                buf_trimmed += 1;
            }
            while buf_trimmed < max {
                let Some(slot) = pop_largest_pool_entry(&mut self.readback_free) else {
                    break;
                };
                release_buffer_slot(device, &mut self.slabs, slot);
                buf_trimmed += 1;
            }
            // The gather pool is under the same budget and the same settled-idle
            // gate as the two above, for the same reason: its refill is a slab
            // carve that only costs anything when it lands on a new block, and
            // trimming it mid-workload would buy back VRAM at the price of that
            // block allocation under the engine lock.
            while buf_trimmed < max {
                let Some(slot) = pop_largest_pool_entry(&mut self.gather_free) else {
                    break;
                };
                release_buffer_slot(device, &mut self.slabs, slot);
                buf_trimmed += 1;
            }
            // Returning the carves above is what lets an upload block empty;
            // this is where the emptied blocks go back. It belongs here and not
            // in the release that empties a block, because a live workload
            // crosses a block boundary and back several times a minute and
            // re-crossing costs a whole block allocation (20-30 ms under this
            // lock). The settled gate is the same one the buffer pools are
            // under, so what is retained mid-session is bounded by the live
            // working set, and what a quiet desktop keeps is one spare block per
            // size class — tens of MiB against the ~177 MiB of frozen staging
            // this trim was written for.
            self.slabs
                .trim_empty_blocks(device, BUFFER_SLAB_IDLE_KEEP_EMPTY);
            // The standalone (non-slab) compute-storage recycle pool re-allocs via
            // a full `vkAllocateMemory` on the next dispatch — like the buffer
            // pools above and unlike the slab-backed image pools (cheap
            // suballocation refill) — so gate its trim on settled idle too: a brief
            // pause between compute dispatches must not steal a pooled storage
            // image and spike the next dispatch with a fresh allocation. Its own
            // budget so it never starves the buffer trim.
            let mut storage_trimmed = 0;
            while storage_trimmed < max {
                let Some(slot) = self.storage_image_free.pop_any() else {
                    break;
                };
                self.destroy_deferred_handle(
                    device,
                    DeferredHandle::Image {
                        image: slot.image,
                        view: slot.view,
                        memory: slot.memory,
                    },
                );
                storage_trimmed += 1;
            }
            buf_trimmed += storage_trimmed;
        }
        // The DEVICE_LOCAL image slab, under the same settled state as the
        // pools above rather than at the caller — which is how it came to run on
        // *every* fired pass instead. The pass fires every
        // `MAINTENANCE_INTERVAL_MS` whenever the poll heartbeat ticks, which is
        // most of the time on any workload that does not saturate the drain
        // worker, so trimming to zero there handed back the block the next frame
        // re-allocated. [`IDLE_SLAB_KEEP_EMPTY`] carries the boot that read 257
        // allocations against 162 trims of a 64 MiB block in 25 seconds.
        //
        // Not inside the `if` above, because the switch that restores the old
        // behaviour has to be able to reach this on an unsettled pass; the
        // policy is [`idle_slab_trim_keep`] and this is its only caller.
        if let Some(keep) = idle_slab_trim_keep(trim_buffers) {
            self.slab.trim_empty_blocks(device, keep);
        }
        trimmed + buf_trimmed
    }

    /// Take entry `index` out of the sampled cache and update its byte gauge.
    fn remove_sampled_entry(&mut self, index: usize) -> SampledSlot {
        let evicted = self.sampled_cache.swap_remove(index);
        self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_sub(evicted.content_len);
        evicted.slot
    }

    fn push_sampled_entry(&mut self, entry: ResidentSampledSlot) {
        self.sampled_cache.push(entry);
    }

    /// Remove entries whose serialized texture owners are all gone. The
    /// returned slots still follow the ordinary deferred recycle path, so a
    /// submission recorded before deletion may finish sampling them safely.
    fn take_released_sampled_entries(&mut self) -> Vec<SampledSlot> {
        let mut released = Vec::new();
        while let Some(index) = self
            .sampled_cache
            .iter()
            .position(|entry| !entry.has_live_owner())
        {
            released.push(self.remove_sampled_entry(index));
        }
        released
    }

    /// Update the consecutive-settled-pass counter for maintenance and return
    /// whether the HOST_VISIBLE buffer pools may be trimmed this pass.
    ///
    /// A pass is settled only if no staging buffer was acquired since the
    /// previous pass. The trim then needs
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive settled passes.
    ///
    /// Upload traffic is the direct signal for the failure this gate prevents:
    /// "a single quiet pass mid-playback
    /// cannot steal a staging buffer and spike the next upload's latency with a
    /// full `vkAllocateMemory`". Registry victims go to zero when the session is
    /// idle *and* when it is busy with a stable working set — a steady animation
    /// re-uses the same render targets forever, so nothing ages out and every pass
    /// reads as quiet. Measured under testufo: `idle_target_drain` fired 169 times
    /// in one boot, roughly once a second throughout the load, and the staging pool
    /// re-allocated the 8 MiB full-frame bucket 607 times at **12.6 ms each**.
    ///
    /// So ask the pool that is about to be trimmed. `staging_hits + misses` is
    /// every acquire, so an unchanged total is the upload path genuinely doing
    /// nothing — the quantity the victim count was standing in for, measured
    /// directly instead of inferred. At true idle the guest stops publishing, no
    /// draw acquires staging, and the trim still fires and still returns the
    /// memory.
    pub(super) fn note_maintenance_settled(&mut self) -> bool {
        let acquires = self.staging_hits.wrapping_add(self.staging_misses);
        let uploads_ran = acquires != self.settled_staging_mark;
        self.settled_staging_mark = acquires;
        if !uploads_ran {
            self.settled_maintenance_passes = self.settled_maintenance_passes.saturating_add(1);
        } else {
            self.settled_maintenance_passes = 0;
        }
        self.settled_maintenance_passes >= SETTLED_PASSES_FOR_BUFFER_TRIM
    }

    /// Run periodic maintenance for objects that are already outside live
    /// resource ownership. Live render, sampled, and compute residents leave
    /// only through explicit lifetime changes or allocation-pressure recovery;
    /// an idle interval is not a resource-state transition.
    pub(crate) unsafe fn advance_registry_maintenance(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        now_ms: u64,
    ) {
        if !self.plan_idle_maintenance(now_ms) {
            return;
        }
        for slot in self.take_released_sampled_entries() {
            self.dispose(&ctx.device, DeferredHandle::RecycleSampled(slot));
        }
        self.retire_released_residents(ctx, counters, IDLE_RECYCLE_TRIM_PER_PASS);
        let trim_buffers = self.note_maintenance_settled();
        self.trim_recycle_pools(&ctx.device, IDLE_RECYCLE_TRIM_PER_PASS, trim_buffers);
    }

    /// Cumulative transient sampled/snapshot pool recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    /// Merged into `CounterSnapshot` by `engine::counter_snapshot`.
    pub(crate) fn recycle_stats(&self) -> (u64, u64, u64, u64) {
        let sampled = self.sampled_free.stats();
        let snapshots = self.attachment_snapshot_free.stats();
        (
            sampled.0 + snapshots.0,
            sampled.1 + snapshots.1,
            sampled.2 + snapshots.2,
            sampled.3 + snapshots.3,
        )
    }

    pub(crate) unsafe fn ensure_init(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        if self.initialized {
            return Ok(());
        }
        self.configure_batch_capacity(super::batch_max_draws(ctx.caps.memory.topology));
        let cmd_pool = ctx
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.gq)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateCommandPool, e)))?;
        counters.note_create(CreateSite::CommandPool);
        let cmd_bufs = ctx
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING_DEPTH as u32),
            )
            .map_err(|e| {
                ctx.device.destroy_command_pool(cmd_pool, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsAllocCommandBuffers, e))
            })?;
        // Growable descriptor arena: block 0 up front, more blocks on demand.
        // Free sets after each draw/dispatch; exhaustion grows rather than drops.
        let mut desc_arena = DescriptorArena::empty();
        if let Err(e) = desc_arena.create_first_block(&ctx.device) {
            ctx.device.destroy_command_pool(cmd_pool, None);
            return Err(e);
        }
        counters.note_create(CreateSite::DescriptorPool);
        let mut slots = Vec::with_capacity(RING_DEPTH);
        for cmd_buf in cmd_bufs.into_iter() {
            // Fences start unsignaled: a slot with no pending cleanup is never
            // waited on, and a submit requires an unsignaled fence.
            match ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(fence) => {
                    counters.note_create(CreateSite::Fence);
                    slots.push(CmdSlot {
                        cmd_buf,
                        fence,
                        pending: None,
                        submission: SlotSubmission::HostOwned,
                        span: super::gpu_span::SlotSpan::Idle,
                        readback_span_armed: false,
                    });
                }
                Err(e) => {
                    for slot in &slots {
                        ctx.device.destroy_fence(slot.fence, None);
                    }
                    desc_arena.destroy(&ctx.device);
                    ctx.device.destroy_command_pool(cmd_pool, None);
                    return Err(DrawError::VkCall(VkCall::new(VkOp::PoolsCreateFence, e)));
                }
            }
        }
        self.cmd_pool = cmd_pool;
        self.desc_arena = desc_arena;
        self.slots = slots;
        self.cur = 0;
        self.in_flight = 0;
        self.initialized = true;
        Ok(())
    }

    /// Install one device's submission capacity before the Vulkan-owned pool
    /// objects are created. Attachment snapshots have the same command-buffer
    /// lifetime, so their recycle budget is derived here rather than retaining
    /// the largest topology's population on every host.
    fn configure_batch_capacity(&mut self, draws: u64) {
        debug_assert!(self.attachment_snapshot_live.is_empty());
        debug_assert_eq!(self.attachment_snapshot_free.len(), 0);
        self.batch_max_draws = draws;
        self.attachment_snapshot_free = FreePool::new();
    }

    /// The capacity installed for this pool's physical device.
    pub(crate) fn batch_capacity(&self) -> u64 {
        self.batch_max_draws
    }

    /// Allocate one descriptor set for `dsl` from the arena, growing a new pool
    /// block on exhaustion (rather than dropping the draw). Returns the set and
    /// its owning pool — pair the pool with the set so a later free routes to
    /// the allocating block. Emits the always-on `desc_arena_grow` cap-pressure
    /// proxy on growth (a rare event; zero under normal load).
    pub(crate) unsafe fn alloc_descriptor_set(
        &mut self,
        device: &ash::Device,
        dsl: vk::DescriptorSetLayout,
        counters: &EngineCounters,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool), DrawError> {
        let (set, pool, grew) = self.desc_arena.allocate(device, dsl)?;
        if grew {
            counters.desc_pool_grow.fetch_add(1, Ordering::Relaxed);
            reims_vgpu_observe::off(format!(
                "desc_arena_grow blocks={} block_max_sets={DESC_BLOCK_MAX_SETS} cause=pool_exhausted",
                self.desc_arena.block_count()
            ));
        }
        Ok((set, pool))
    }

    /// The device's guest-scatter pipeline, built on the first writeback that
    /// wants it and `None` on a device whose driver refused to build it.
    ///
    /// The refusal latches: a host that cannot compile our kernel is not going
    /// to compile it on the next frame either, and retrying would put a pipeline
    /// creation on the hot path several hundred times a second. It is emitted
    /// once, on the fail channel, because the fallback it selects is the
    /// expensive path this rail exists to leave.
    ///
    /// # Safety
    ///
    /// `ctx` must be the device these pools belong to.
    pub(crate) unsafe fn scatter_pipeline(
        &mut self,
        ctx: &DeviceContext,
    ) -> Option<super::super::guest_scatter::ScatterPipeline> {
        if let Some(p) = self.scatter {
            return Some(p);
        }
        if self.scatter_refused {
            return None;
        }
        match unsafe { super::super::guest_scatter::ScatterPipeline::create(ctx) } {
            Ok(p) => {
                self.scatter = Some(p);
                Some(p)
            }
            Err(e) => {
                self.scatter_refused = true;
                reims_vgpu_observe::Emit::decline("scatter_pipeline", &e).fail();
                None
            }
        }
    }

    /// A descriptor set for the guest-scatter kernel, recycled if one has
    /// retired and allocated if not, parked on the list [`Self::seal_entry`]
    /// drains so the caller owes it nothing.
    ///
    /// The caller must write it before dispatching: a recycled set still names
    /// the previous dispatch's three buffers, and `ScatterPipeline::write_set`
    /// is the only writer. Every caller does, because writing is how it names
    /// its own buffers at all — there is no path that binds a set it did not
    /// just write.
    ///
    /// # Safety
    ///
    /// `device` must be the device these pools belong to, and `dsl` must be the
    /// guest-scatter layout every set on the free list was allocated against.
    pub(crate) unsafe fn alloc_scatter_descriptor_set(
        &mut self,
        device: &ash::Device,
        dsl: vk::DescriptorSetLayout,
        counters: &EngineCounters,
    ) -> Result<vk::DescriptorSet, DrawError> {
        let (set, pool) = match self.take_free_scatter_dset() {
            Some(recycled) => recycled,
            None => unsafe { self.alloc_descriptor_set(device, dsl, counters) }?,
        };
        self.scatter_dsets.push((set, pool));
        Ok(set)
    }

    /// Take a retired guest-scatter set off the free list, or `None` when there
    /// is none.
    ///
    /// Split out of [`Self::alloc_scatter_descriptor_set`] for the same reason
    /// [`Self::take_free_gather`] is split out of `acquire_guest_gather`: it is
    /// the whole of the no-aliasing property and it is the half that needs no
    /// device, so a test can exercise it. A set **leaves** the free list when it
    /// is handed out and only `drain_cleanup` puts it back, which runs after the
    /// fence of the submission that named it — so two dispatches inside one
    /// command buffer cannot be handed one set and have the second's
    /// `write_set` overwrite the first's bindings.
    fn take_free_scatter_dset(&mut self) -> Option<(vk::DescriptorSet, vk::DescriptorPool)> {
        self.scatter_dset_free.pop()
    }

    /// Return a retired entry's guest-scatter sets to the free list.
    ///
    /// The other half of [`Self::take_free_scatter_dset`], named for the same
    /// reason: the fence is what makes a rewrite safe, and this is the only
    /// caller that has one. `drain_cleanup` is the only call site and it runs
    /// after the wait.
    fn recycle_scatter_dsets(&mut self, sets: &mut Vec<(vk::DescriptorSet, vk::DescriptorPool)>) {
        self.scatter_dset_free.append(sets);
    }

    /// Free `(set, owning_pool)` pairs back to their allocating blocks.
    pub(crate) unsafe fn free_descriptor_sets(
        &self,
        device: &ash::Device,
        sets: &[(vk::DescriptorSet, vk::DescriptorPool)],
    ) {
        self.desc_arena.free(device, sets);
    }

    /// The ring slots whose recorded GPU work may still reference pool objects:
    /// every submitted-but-unretired CB, plus the open draw batch's slot (its
    /// CB is still recording, so unsubmitted, but it already references
    /// images/buffers — destroying them before its flush would be a
    /// use-after-free at submit). A zero mask means nothing can be reading
    /// anything.
    ///
    /// # The one recording state this does *not* cover
    ///
    /// A slot claimed by `begin_entry` and being recorded into, which is neither
    /// `pending` yet nor an `open_batch`, is in neither set — so a
    /// [`Self::dispose`] while it records destroys immediately, on an object the
    /// open command buffer already names.
    ///
    /// That is sound today only because **nothing runs in that gap**: every
    /// non-batch caller claims the slot, records, submits and seals inside one
    /// call, so no host code can reach a dispose in between. It is a property of
    /// those callers and not of this function, and the compiler cannot see it.
    ///
    /// A caller that wants to record several independent units of work into one
    /// submission — the shape the guest-page writeback would take to stop paying
    /// a fence per window — reopens exactly this gap, because its per-unit
    /// bookkeeping (`unpin_resident_target`, which is what permits eviction)
    /// would run while later units are still recording. Such a caller must
    /// either do all of its bookkeeping after the submit, or make its recording
    /// slot visible here the way `open_batch` is. Do not assume the graveyard
    /// covers it: it covers the two states above, and this is a third.
    fn open_slot_mask(&self) -> SlotMask {
        let mut mask: SlotMask = 0;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.pending.is_some() {
                mask |= 1 << index;
            }
        }
        if self.open_batch.is_some() {
            // The batch records into `cur` and seals onto that same slot at
            // flush, and every path that retires a slot flushes the batch
            // first, so this bit cannot clear before the batch's work retires.
            mask |= 1 << self.cur;
        }
        mask
    }

    /// Destroy `handle` now if nothing can be reading it, else park it in the
    /// graveyard until each slot open at this instant has retired.
    pub(crate) unsafe fn dispose(&mut self, device: &ash::Device, handle: DeferredHandle) {
        let waiting = self.open_slot_mask();
        if waiting == 0 {
            self.destroy_or_recycle(device, handle);
        } else {
            self.graveyard.push((waiting, handle));
        }
    }

    /// Acquire + bind DEVICE_LOCAL image memory from the slab suballocator,
    /// registering the sub-allocation against `image` on success. On bind
    /// failure the slab range is released (the caller destroys the `image`).
    /// Returns the backing `VkDeviceMemory` (shared across the block's other
    /// images) for the pool's image struct; the image was bound at the slab
    /// offset, not offset 0.
    ///
    /// Out of memory gets one retry here, for every caller, after emptying the
    /// recycle pools. Only the pools: this runs at four sites, one of them
    /// part-way through recording a draw, and
    /// [`ResourcePools::reclaim_pools_for_allocation_retry`] is the half of the
    /// recovery that is safe there — a free-list entry is by construction one no
    /// command buffer holds. Retiring live residents is not safe here, and is
    /// done only by `registry_ensure_attachment`, which calls the fuller
    /// [`ResourcePools::reclaim_for_allocation_retry`] itself; that function
    /// records the segfault which established the difference.
    pub(super) unsafe fn bind_image_slab(
        &mut self,
        ctx: &DeviceContext,
        image: vk::Image,
        ireq: &vk::MemoryRequirements,
        bind_op: VkOp,
        counters: &EngineCounters,
    ) -> Result<vk::DeviceMemory, DrawError> {
        match self.bind_image_slab_once(ctx, image, ireq, bind_op, counters) {
            // Once, and only for this result. A reclaim that frees nothing means
            // there was nothing to free, so the original error is the honest one
            // and a second attempt would only repeat it.
            Err(error)
                if error.out_of_memory() && self.reclaim_pools_for_allocation_retry(ctx) > 0 =>
            {
                self.bind_image_slab_once(ctx, image, ireq, bind_op, counters)
                    .map_err(|_| error)
            }
            other => other,
        }
    }

    unsafe fn bind_image_slab_once(
        &mut self,
        ctx: &DeviceContext,
        image: vk::Image,
        ireq: &vk::MemoryRequirements,
        bind_op: VkOp,
        counters: &EngineCounters,
    ) -> Result<vk::DeviceMemory, DrawError> {
        self.slab
            .ensure_image_unregistered(image)
            .map_err(DrawError::Slab)?;
        let token = self.slab.acquire(ctx, ireq, counters)?;
        match ctx
            .device
            .bind_image_memory(image, token.memory, token.offset())
        {
            Ok(()) => {
                self.slab.register(image, token);
                Ok(token.memory)
            }
            Err(e) => {
                self.slab.release_token(&ctx.device, token);
                Err(DrawError::VkCall(VkCall::new(bind_op, e)))
            }
        }
    }

    /// Release the slab sub-allocation backing `image` (the caller destroys the
    /// `image`/view handles). No-op for a non-slab image. This replaces the raw
    /// `vkFreeMemory` at every DEVICE_LOCAL-image free site: the memory belongs
    /// to a shared block, not the image.
    pub(super) unsafe fn free_image_slab(&mut self, device: &ash::Device, image: vk::Image) {
        self.slab.free_image(device, image);
    }

    /// Terminal handling for a deferred handle once it is safe (in_flight == 0):
    /// recyclable slots rejoin their geometry-keyed pools; every other handle
    /// is destroyed.
    unsafe fn destroy_or_recycle(&mut self, device: &ash::Device, handle: DeferredHandle) {
        match handle {
            DeferredHandle::RecycleSampled(slot) => {
                self.admit_sampled(slot);
            }
            DeferredHandle::RecycleTarget(img) => {
                self.admit_target(img);
            }
            other => self.destroy_deferred_handle(device, other),
        }
    }

    /// Return an evicted sampled slot to `sampled_free` for reuse by a later
    /// same-geometry `acquire_sampled`.
    fn admit_sampled(&mut self, slot: SampledSlot) {
        self.sampled_free.admit(slot.key(), slot)
    }

    /// Return a displaced resident-target image to `target_free` for reuse by a
    /// later same-(geometry, format) `registry_ensure`/`registry_ensure_attachment`
    /// create.
    fn admit_target(&mut self, img: FreeTargetImage) {
        self.target_free.admit(img.key(), img)
    }

    /// Return a retired transient compute-storage image to `storage_image_free`
    /// for reuse by a later same-geometry `acquire_storage_image`.
    fn admit_storage_image(&mut self, slot: StorageImageSlot) {
        self.storage_image_free.admit(slot.key, slot)
    }

    /// Pop a recycled resident-target image for `(width, height, format)` if one
    /// is available, else `None`. Splits the reuse (`target_free_hits`) vs
    /// fresh-alloc (`target_free_allocs`) census so a boot can prove the
    /// per-frame realloc storm collapsed (allocs ≈ 0 under video).
    pub(super) fn take_free_target(
        &mut self,
        width: u32,
        height: u32,
        sample_count: u32,
        format: vk::Format,
    ) -> Option<FreeTargetImage> {
        let key = TargetRecycleKey {
            width,
            height,
            sample_count,
            format,
        };
        self.target_free.take(&key)
    }

    /// Cumulative resident-target recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    pub(crate) fn target_recycle_stats(&self) -> (u64, u64, u64, u64) {
        self.target_free.stats()
    }

    /// `(non_pinned_peak, non_pinned_peak_bytes)` for the resident registry —
    /// the reach, and what the reach costs. Neither answers alone.
    ///
    /// Both are cumulative for the life of the pools. The bytes are what said a
    /// slot count was never measuring the resource it claimed to protect, which
    /// is the reading that retired that count — see
    /// [`ResourcePools::non_pinned_registry_bytes`] and
    /// [`ResourcePools::recoverable_residents`].
    pub(crate) fn registry_pressure_stats(&self) -> (u64, u64) {
        (
            self.registry_non_pinned_peak,
            self.registry_non_pinned_peak_bytes,
        )
    }

    /// `(sole_copy_peak_slots, sole_copy_peak_bytes)` — what protecting
    /// unreproducible content costs, in the two quantities that have to be read
    /// together.
    ///
    /// This is the population the allocation-failure retry cannot give back, so
    /// its peak against `registry_non_pinned_peak` is what says whether the
    /// retry still has anything to work with. See
    /// [`ResourcePools::registry_sole_copy`].
    pub(crate) fn registry_sole_copy_stats(&self) -> (u64, u64) {
        (
            self.registry_sole_copy_peak.count as u64,
            self.registry_sole_copy_peak.bytes,
        )
    }

    /// `(sole_copy_peak_slots, sole_copy_peak_bytes)` for the compute-storage
    /// registry — the sibling of [`Self::registry_sole_copy_stats`] over the
    /// other population.
    pub(crate) fn compute_storage_sole_copy_stats(&self) -> (u64, u64) {
        (
            self.compute_storage_sole_copy_peak.count as u64,
            self.compute_storage_sole_copy_peak.bytes,
        )
    }

    /// `VkDeviceMemory` the image slab holds right now, and the carved half of
    /// it, in bytes. See [`slab::SlabPool::held_bytes`] for what it covers.
    pub(crate) fn slab_held_bytes(&self) -> (u64, u64) {
        self.slab.held_bytes()
    }

    /// The worst gap between a resident being touched and being read again, for
    /// the life of the pools. See
    /// [`ResourcePools::resident_resample_peak_ms`] for what it is the margin
    /// against.
    pub(crate) fn resident_resample_peak_ms(&self) -> u64 {
        self.resident_resample_peak_ms
    }

    /// Clear `retired` from every parked handle's wait mask and take out the
    /// ones that now wait on nothing. `retired` is the bit of the slot that
    /// just retired; teardown passes every bit.
    ///
    /// Taking rather than destroying in place is what lets `destroy_or_recycle`
    /// borrow `self.sampled_free`/`self.target_free` while the graveyard is
    /// being walked, and it is the whole decision the mask exists to make, so
    /// it is also the seam the tests drive without a device.
    fn take_released_graveyard(&mut self, retired: SlotMask) -> Vec<DeferredHandle> {
        let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.graveyard)
            .into_iter()
            .map(|(mask, handle)| (mask & !retired, handle))
            .partition(|(mask, _)| *mask == 0);
        self.graveyard = waiting;
        ready.into_iter().map(|(_, handle)| handle).collect()
    }

    /// Terminally handle every graveyard entry released by `retired` retiring.
    pub(super) unsafe fn release_graveyard(&mut self, device: &ash::Device, retired: SlotMask) {
        for handle in self.take_released_graveyard(retired) {
            self.destroy_or_recycle(device, handle);
        }
    }

    fn wait_error(counters: &EngineCounters, e: vk::Result, op: DeviceLostOp) -> DrawError {
        if e == vk::Result::TIMEOUT {
            counters.fence_timeouts.fetch_add(1, Ordering::Relaxed);
            DrawError::FenceTimeout
        } else if e == vk::Result::ERROR_DEVICE_LOST {
            DrawError::DeviceLost(DeviceLostDecline::Driver { op, result: e })
        } else {
            DrawError::VkCall(VkCall::new(op.vk_op(), e))
        }
    }

    /// Reset a ring-slot command buffer and begin recording it, arming its GPU
    /// timestamp pair in the same call.
    ///
    /// **This is the only way a ring-slot command buffer may be begun.** All five
    /// submission kinds used to spell the reset-then-begin pair out by hand, and
    /// the probe was wired into exactly one of them — which reported 51 % GPU
    /// occupancy for a device whose writeback submissions carried no stamps at
    /// all. Folding the arm into the begin is what makes a sixth kind unable to
    /// join without one: there is no longer a shorter path to a recording slot CB.
    ///
    /// The `VkOp`s are parameters because each caller reports its own failure name
    /// and those names are load-bearing in the fail log.
    ///
    /// # Safety
    ///
    /// `cb` must be the current slot's command buffer and must not be recording.
    pub(crate) unsafe fn begin_slot_recording(
        &mut self,
        ctx: &DeviceContext,
        cb: vk::CommandBuffer,
        kind: gpu_span::Kind,
        reset_op: VkOp,
        begin_op: VkOp,
    ) -> Result<(), DrawError> {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(reset_op, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(begin_op, e)))?;
        unsafe { self.gpu_span_arm(ctx, cb, kind) };
        Ok(())
    }

    /// Reset the current slot's timestamp pair and write the top one, so the
    /// submission about to be recorded reports its own GPU execution time.
    ///
    /// Private, and reached only through [`Self::begin_slot_recording`]: a caller
    /// that could arm without beginning could also begin without arming, which is
    /// the failure this pairing exists to prevent. A batch joiner reaches neither
    /// — it appends to a CB already armed, and arming again would move the top
    /// stamp forward past work the batch has already recorded, reading as a fast
    /// submission rather than as a broken one.
    ///
    /// Both `vkCmdResetQueryPool` and the write must be outside a render pass
    /// instance, which the caller satisfies by sitting immediately after
    /// `vkBeginCommandBuffer`.
    ///
    /// # Safety
    ///
    /// `cb` must be the current slot's command buffer, recording, and outside any
    /// render pass.
    unsafe fn gpu_span_arm(
        &mut self,
        ctx: &DeviceContext,
        cb: vk::CommandBuffer,
        kind: gpu_span::Kind,
    ) {
        let Some(probe) = ctx.draw_spans.as_ref() else {
            return;
        };
        let slot = self.cur;
        // A slot armed twice without a read between means the ring reused it
        // without retiring it, which would also mean its cleanup was never
        // drained. Report rather than silently overwrite: the sample is lost
        // either way and only the counter says so.
        if self.slots[slot].span != gpu_span::SlotSpan::Idle {
            gpu_span::note_unread();
        }
        let base = DrawSpanProbe::base(slot);
        ctx.device
            .cmd_reset_query_pool(cb, probe.pool, base, DrawSpanProbe::PER_SLOT);
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, probe.pool, base);
        self.slots[slot].span = gpu_span::SlotSpan::Armed(kind);
        gpu_span::note_armed();
    }

    /// Write the bottom timestamp of the slot's command buffer, immediately
    /// before it ends.
    ///
    /// `slot` is passed rather than read from `self.cur` because the batch flush
    /// path seals the slot the batch was opened on, and a caller that guessed
    /// would attribute one submission's span to another slot's queries.
    ///
    /// # Safety
    ///
    /// `cb` must be `slot`'s command buffer, still recording, and outside any
    /// render pass.
    unsafe fn gpu_span_seal(&mut self, ctx: &DeviceContext, cb: vk::CommandBuffer, slot: usize) {
        let Some(probe) = ctx.draw_spans.as_ref() else {
            return;
        };
        let gpu_span::SlotSpan::Armed(kind) = self.slots[slot].span else {
            return;
        };
        ctx.device.cmd_write_timestamp(
            cb,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            probe.pool,
            DrawSpanProbe::base(slot) + 1,
        );
        self.slots[slot].span = gpu_span::SlotSpan::Sealed(kind);
        gpu_span::note_sealed();
    }

    /// [`Self::gpu_span_seal`] for the slot the caller is about to submit on its
    /// own, which is always the current one.
    ///
    /// # Safety
    ///
    /// As [`Self::gpu_span_seal`].
    pub(crate) unsafe fn gpu_span_seal_current(
        &mut self,
        ctx: &DeviceContext,
        cb: vk::CommandBuffer,
    ) {
        unsafe { self.gpu_span_seal(ctx, cb, self.cur) };
    }

    /// Reset this slot's readback timestamp region and write its start stamp.
    ///
    /// The reset must be recorded into the same command buffer that writes the
    /// stamps: a query's results are undefined until it is reset, and resetting
    /// on the host needs `hostQueryReset`, a Vulkan 1.2 feature this device does
    /// not ask for.
    ///
    /// The region belongs to the current ring slot rather than being shared —
    /// see [`TimestampProbe`], which used to be shared between the
    /// readback (which waits its fence) and the guest-page writeback (which does
    /// not), so two writebacks in flight reset each other's queries.
    ///
    /// # Safety
    ///
    /// `cb` must be the current slot's command buffer, recording, and outside
    /// any render pass.
    pub(crate) unsafe fn readback_span_arm(&mut self, ctx: &DeviceContext, cb: vk::CommandBuffer) {
        let Some(probe) = ctx.timestamps.as_ref() else {
            return;
        };
        let slot = self.cur;
        let base = TimestampProbe::base(slot);
        ctx.device
            .cmd_reset_query_pool(cb, probe.pool, base, TimestampProbe::PER_SLOT);
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, probe.pool, base);
        self.slots[slot].readback_span_armed = true;
    }

    /// Write one of the two later stamps of the current slot's readback region.
    ///
    /// `mark` is 1 for the point after the barrier — where the draws ahead are
    /// known done — and 2 for the end of the copy. Silently does nothing if the
    /// slot was never armed, so a caller that stamps without a start cannot
    /// publish a delta against a query holding another submission's ticks.
    ///
    /// # Safety
    ///
    /// As [`Self::readback_span_arm`].
    pub(crate) unsafe fn readback_span_mark(
        &mut self,
        ctx: &DeviceContext,
        cb: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        mark: u32,
    ) {
        debug_assert!(mark < TimestampProbe::PER_SLOT);
        let Some(probe) = ctx.timestamps.as_ref() else {
            return;
        };
        if !self.slots[self.cur].readback_span_armed {
            return;
        }
        ctx.device.cmd_write_timestamp(
            cb,
            stage,
            probe.pool,
            TimestampProbe::base(self.cur) + mark,
        );
    }

    /// Read a retiring slot's readback region and charge the two spans it holds.
    ///
    /// Called only with the slot's fence already signalled, which is what makes
    /// the three queries available — so `vkGetQueryPoolResults` is asked without
    /// `WAIT` and cannot block. This replaces a read that ran *before* the next
    /// copy was recorded and argued that the previous copy's results were still
    /// there because its reset had not executed yet. That argument assumed
    /// submissions complete in submission order, which Vulkan does not grant.
    unsafe fn readback_span_read(&mut self, ctx: &DeviceContext, slot: usize) {
        let Some(probe) = ctx.timestamps.as_ref() else {
            return;
        };
        if !std::mem::replace(&mut self.slots[slot].readback_span_armed, false) {
            return;
        }
        let mut ticks = [0u64; TimestampProbe::PER_SLOT as usize];
        if ctx
            .device
            .get_query_pool_results(
                probe.pool,
                TimestampProbe::base(slot),
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
            .is_ok()
        {
            let us =
                |from: usize, to: usize| probe.scale.elapsed_ns(ticks[from], ticks[to]) / 1_000;
            crate::telemetry::note_readback_gpu_us(us(0, 1), us(1, 2));
        }
    }

    /// Read a retiring slot's timestamp pair and charge the delta.
    ///
    /// Only ever called with the slot's fence already signaled, which is what
    /// makes both queries available — so `vkGetQueryPoolResults` is asked without
    /// `WAIT` and a `NOT_READY` is a real defect in that ordering rather than
    /// something to spin on. It is dropped rather than retried: a lost sample is
    /// visible as `armed - read` and retrying inside the retire path would put an
    /// unbounded wait on the drain worker to fix an instrument.
    unsafe fn gpu_span_read(&mut self, ctx: &DeviceContext, slot: usize) {
        let Some(probe) = ctx.draw_spans.as_ref() else {
            return;
        };
        let gpu_span::SlotSpan::Sealed(kind) =
            std::mem::replace(&mut self.slots[slot].span, gpu_span::SlotSpan::Idle)
        else {
            return;
        };
        let mut ticks = [0u64; DrawSpanProbe::PER_SLOT as usize];
        if ctx
            .device
            .get_query_pool_results(
                probe.pool,
                DrawSpanProbe::base(slot),
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
            .is_ok()
        {
            gpu_span::note_busy_ns(kind, probe.scale.elapsed_ns(ticks[0], ticks[1]));
        }
    }

    /// Retire one slot: wait its fence, reset it, and drain the cleanup it
    /// owes. No-op for a slot with nothing pending.
    unsafe fn retire_slot(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        index: usize,
    ) -> Result<(), DrawError> {
        if self.slots[index].pending.is_none() {
            return Ok(());
        }
        self.await_submit_return(counters, index, DeviceLostOp::PoolsWaitFencesRetire)?;
        if let Some(error) = ctx.queue_failure() {
            return Err(Self::wait_error(
                counters,
                error,
                DeviceLostOp::PoolsWaitFencesRetire,
            ));
        }
        let fence = self.slots[index].fence;
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| {
                // The wait that every macos-11 freeze lands in. Until now the
                // failure said only that *a* wait timed out; this names the
                // submission it timed out on, which is the question two
                // sessions of switch-bisecting could not reach. Emitted before
                // the error is mapped, because `wait_error` may turn it into a
                // device loss and the teardown that follows clears the ring.
                let held = match crate::gpu_hang_trail::submission(index) {
                    Some(note) => format!("{note}"),
                    None => "none (this slot's work was never recorded)".to_string(),
                };
                reims_vgpu_observe::fail(format!(
                    "vk_engine_fence_wedged slot={index} result={e:?} held={held}"
                ));
                if let Some(rest) = crate::gpu_hang_trail::outstanding() {
                    reims_vgpu_observe::fail(format!("vk_engine_fence_wedged_queue {rest}"));
                }
                Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesRetire)
            })?;
        ctx.device
            .reset_fences(&[fence])
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsResetFencesRetire, e)))?;
        // After the wait and before anything else: the fence signalling is
        // precisely what makes this slot's two queries available, and the read is
        // the only thing that returns the slot's span state to `Idle` so the next
        // arming of it is not reported as a lost sample.
        unsafe { self.gpu_span_read(ctx, index) };
        // The same argument, for the other probe: this slot's three readback
        // queries are readable exactly now and never before.
        unsafe { self.readback_span_read(ctx, index) };
        let pending = self.slots[index].pending.take().expect("checked above");
        self.in_flight = self.in_flight.saturating_sub(1);
        // Its fence has signalled, so this submission is no longer a candidate
        // for a wedge. Paired with the `note_submit` in `finish_entry_async`.
        crate::gpu_hang_trail::note_retired(index);
        self.drain_cleanup(&ctx.device, pending);
        self.release_graveyard(&ctx.device, 1 << index);
        Ok(())
    }

    /// Recover host ownership of a slot's fence after an asynchronous queue
    /// handoff. This waits only for the CPU-side `vkQueueSubmit` call to return;
    /// the fence wait below remains the separate GPU-completion boundary.
    fn await_submit_return(
        &mut self,
        counters: &EngineCounters,
        index: usize,
        op: DeviceLostOp,
    ) -> Result<(), DrawError> {
        let state = std::mem::replace(&mut self.slots[index].submission, SlotSubmission::HostOwned);
        let result = match state {
            SlotSubmission::HostOwned => Ok(()),
            SlotSubmission::QueueOwned(receipt) => receipt.wait(),
            SlotSubmission::Failed(error) => Err(error),
        };
        if let Err(error) = result {
            self.slots[index].submission = SlotSubmission::Failed(error);
        }
        result.map_err(|error| Self::wait_error(counters, error, op))
    }

    /// Non-blocking ownership probe for opportunistic retirement. `false`
    /// means the queue thread still owns the fence, so Vulkan must not be asked
    /// for its status yet.
    fn try_take_submit_return(
        &mut self,
        counters: &EngineCounters,
        index: usize,
    ) -> Result<bool, DrawError> {
        let result = match &self.slots[index].submission {
            SlotSubmission::HostOwned => return Ok(true),
            SlotSubmission::QueueOwned(receipt) => {
                let Some(result) = receipt.try_complete() else {
                    return Ok(false);
                };
                result
            }
            SlotSubmission::Failed(error) => Err(*error),
        };
        self.slots[index].submission = match result {
            Ok(()) => SlotSubmission::HostOwned,
            Err(error) => SlotSubmission::Failed(error),
        };
        result.map(|()| true).map_err(|error| {
            Self::wait_error(counters, error, DeviceLostOp::PoolsFenceStatusBeginEntry)
        })
    }

    /// The open batch's command buffer and the fence [`Self::batch_flush`] will
    /// submit it with, for a caller that wants to append to the run rather than
    /// end it.
    ///
    /// The command buffer is **already recording** — the caller must not begin
    /// or reset it. It may carry the preceding draw's still-open render pass;
    /// every command that cannot live there closes it at its recording site.
    /// The fence is the one returned here precisely so the caller can wait it
    /// after `batch_flush` without having to reach back into a batch that no
    /// longer exists by then.
    pub(crate) fn batch_open_recording(&self) -> Option<(vk::CommandBuffer, vk::Fence)> {
        self.open_batch.as_ref().map(|b| (b.cb, b.fence))
    }

    /// Start a new entry (draw / dispatch / sync helper): advance to the next
    /// ring slot, retiring it first if its CB is still in flight (this is the
    /// only place a full ring blocks). Returns the slot's CB + fence; the CB
    /// is ready to reset/record and the fence is unsignaled, ready to submit.
    pub(crate) unsafe fn begin_entry(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(vk::CommandBuffer, vk::Fence), DrawError> {
        // Any path that claims a slot first submits the deferred batch — the
        // single choke point that keeps queue order = record order for every
        // reader/compute/prefetch path (and prevents ring wrap from resetting
        // the still-recording batch CB).
        self.batch_flush(ctx, counters)?;
        // Whatever pass was standing, the caller is about to record into a
        // different command buffer than the one that opened it. Unconditional
        // and above the early exits below, because every claim of a slot ends
        // the echoed pass's recording session whether a batch was open or not.
        self.forget_pass_echo();
        // Reap the oldest contiguous run of already-signaled slots before
        // claiming one, rather than only the slot about to be reused. The
        // readback path deliberately waits a fence without retiring (see
        // `wait_entry_fence`), so a signaled slot can sit unreaped for a whole
        // ring, holding its staging, gather and readback buffers out of the free
        // lists and its descriptor sets out of the arena — every one of which
        // the next draw then allocates fresh.
        //
        // This used to be load-bearing for the sampled content cache too, whose
        // admissions travelled in the same cleanup: a texture the guest had not
        // changed re-uploaded and re-allocated for RING_DEPTH - 1 draws before
        // the cache could serve it. It no longer is — admission happens at
        // submit, in `finish_entry_async`, which is the whole of why.
        //
        // `break` on the first unsignaled slot is load-bearing: reaping out of
        // order can drop `in_flight` to 0 while later slots still run, which
        // would let `gpu_work_open()` admit a graveyard drain under live work.
        let n = self.slots.len();
        for step in 1..=n {
            let index = (self.cur + step) % n;
            if self.slots[index].pending.is_none() {
                continue;
            }
            if !self.try_take_submit_return(counters, index)? {
                break;
            }
            let signaled = ctx
                .device
                .get_fence_status(self.slots[index].fence)
                .map_err(|e| {
                    Self::wait_error(counters, e, DeviceLostOp::PoolsFenceStatusBeginEntry)
                })?;
            if !signaled {
                break;
            }
            self.retire_slot(ctx, counters, index)?;
        }
        let next = (self.cur + 1) % self.slots.len();
        if self.slots[next].pending.is_some() {
            self.await_submit_return(counters, next, DeviceLostOp::PoolsWaitFencesRetire)?;
            // Count as a "block" only when the fence is genuinely unsignaled
            // (the GPU still owns the slot); reclaiming a finished slot on
            // advance is bookkeeping, not a stall.
            let still_running = !ctx
                .device
                .get_fence_status(self.slots[next].fence)
                .map_err(|e| {
                    Self::wait_error(counters, e, DeviceLostOp::PoolsFenceStatusBeginEntry)
                })?;
            self.retire_slot(ctx, counters, next)?;
            if still_running {
                counters.ring_retire_blocks.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.cur = next;
        Ok((self.slots[next].cmd_buf, self.slots[next].fence))
    }

    /// Seal the current entry's transient resources: move every live pool slot
    /// (which the just-recorded CB references) out of the shared live lists so
    /// a concurrent entry cannot recycle them, bundled with the descriptor set.
    ///
    /// The images named by `sampled_retains` are lifted straight back out of the
    /// cleanup into [`SealedEntry::admissions`]: they are not transient, they
    /// are about to become cache entries, and leaving them in the bag is what
    /// made them wait for a fence.
    pub(crate) fn seal_entry(
        &mut self,
        dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<SampledRetain>,
    ) -> SealedEntry {
        // The slots the map names are about to be handed to the cleanup, so
        // nothing recorded after this may bind one.
        self.forget_cb_bound_buffers("bindmap_clear_seal", "bindmap_clear_seal_entries");
        self.forget_cb_sampled_guest();
        let mut readback: Vec<BufferSlot> = self.readback_live.take().into_iter().collect();
        readback.append(&mut self.readback_multi_live);
        let mut sampled = std::mem::take(&mut self.sampled_live);
        let admissions = take_retained_slots(&mut sampled, sampled_retains);
        // Both seal points reach here, which is the whole reason the scatter's
        // sets are parked on `self` instead of travelling with its plan. They
        // travel in their own field rather than joining `dsets`, because they
        // recycle at retire and a draw's do not.
        SealedEntry {
            cleanup: PendingGpuCleanup {
                dsets,
                scatter_dsets: std::mem::take(&mut self.scatter_dsets),
                staging: std::mem::take(&mut self.staging_live),
                gather: std::mem::take(&mut self.gather_live),
                readback,
                sampled,
                attachment_snapshots: std::mem::take(&mut self.attachment_snapshot_live),
                storage_images: std::mem::take(&mut self.storage_image_live),
                unpin_residents: std::mem::take(&mut self.resident_pins_live),
                unpin_compute_residents: std::mem::take(&mut self.compute_write_pins_live),
                guest_write_tokens: {
                    let mut tokens: Vec<_> = std::mem::take(&mut self.guest_write_tokens_live)
                        .into_values()
                        .collect();
                    tokens.sort_unstable();
                    tokens
                },
            },
            admissions,
        }
    }

    /// Park the sealed cleanup on the current slot and give the content cache
    /// the images this entry's CB fills. The CB was submitted with the slot
    /// fence and the entry returns without waiting.
    ///
    /// # Why the cache takes them now and not at retire
    ///
    /// A cache entry is a CPU-side name for a GPU image, and the only thing that
    /// has to have happened before a *consumer* may bind it is that the fill was
    /// recorded and submitted first. Every consumer is itself a recorded
    /// command, `begin_entry` flushes the open batch before any path claims a
    /// slot so queue order is record order, and the fill's own
    /// `TRANSFER_WRITE → SHADER_READ` barrier (see `upload_buffer_to_sampled_image`)
    /// has every later-submitted command in its second scope. So the fence adds
    /// nothing to the consumer's correctness — it only delays the name.
    ///
    /// Admitting at retire delayed it by a whole ring: a window bound N times
    /// while the first bind's slot was still in flight missed N times, gathered
    /// N times, and threw away N-1 of the images on arrival. Measured on the
    /// macos-26 rail at `a6ed11b9`: `sampled_admit_duplicate` 5533 against
    /// `sampled_admit_kept` 3876, and 58.9 GB of guest texels imported in one
    /// driven boot against 219 MB on macos-13.
    ///
    /// The admissions run **after** the slot is marked pending, not before.
    /// Entries that cannot be retained return their images to `sampled_live`;
    /// making this slot visible first ensures the next seal parks them behind
    /// the command buffer that filled or sampled them.
    pub(crate) unsafe fn finish_entry_async(
        &mut self,
        sealed: SealedEntry,
        timeline: Option<u64>,
        submit_return: Option<super::super::queue_owner::PendingQueueSubmit>,
    ) {
        let SealedEntry {
            cleanup,
            admissions,
        } = sealed;
        debug_assert!(
            self.slots[self.cur].pending.is_none(),
            "current slot already owes cleanup"
        );
        self.slots[self.cur].pending = Some(cleanup);
        self.slots[self.cur].submission = submit_return
            .map(SlotSubmission::QueueOwned)
            .unwrap_or(SlotSubmission::HostOwned);
        self.in_flight += 1;
        // The submission is now outstanding, and this is the one point both
        // submit paths reach — a batch flush and a lone draw's own submit. The
        // trail's per-slot record is cleared again in `retire_slot`, so a slot
        // holding one is a submission whose fence has not signalled.
        crate::gpu_hang_trail::note_submit(self.cur, timeline);
        self.admit_recorded_sampled(admissions);
    }

    /// Give the content cache the images a recorded command buffer fills.
    ///
    /// The one admission point, reached from the two places that can be the
    /// earliest safe moment for their caller — [`Self::finish_entry_async`] for
    /// a draw that submits on its own, and [`Self::batch_append`] for one that
    /// defers into an open batch. Earliest matters: a window not yet in the
    /// cache is a window the next bind re-imports across PCIe in full.
    ///
    /// # The precondition, and why it is not the same instant for both callers
    ///
    /// An admission that cannot be retained returns its image to
    /// `sampled_live`. The command buffer that fills or samples that image must
    /// already be represented by [`Self::open_slot_mask`] so the next seal
    /// parks the image behind the right fence. A batch enters the mask when
    /// `open_batch` is set; a non-batch slot enters it when its cleanup is
    /// parked. The assertion keeps a third caller from admitting too early.
    unsafe fn admit_recorded_sampled(&mut self, admissions: Vec<(SampledSlot, SampledRetain)>) {
        debug_assert!(
            admissions.is_empty() || self.open_slot_mask() & (1 << self.cur) != 0,
            "admitting while the filling command buffer's slot is invisible to dispose()"
        );
        for (slot, retain) in admissions {
            self.admit_sampled_slot(slot, &retain.content, retain.resource_lifetime.as_ref());
        }
    }

    /// Discard every content-cache entry, in-flight-safely.
    ///
    /// The answer to a submission that published entries and then failed to
    /// reach the queue: those images hold undefined content and the cache would
    /// hand them to a later bind as if they held a guest window. Nothing records
    /// which entries came from which submission and nothing should — the cache
    /// is a pure optimisation, so "some of this is unfilled" has a sound answer
    /// that needs no bookkeeping at all.
    ///
    /// Discard records failed-publication victims, and each slot is disposed
    /// rather than dropped because an earlier in-flight CB may still sample it.
    pub(crate) unsafe fn discard_sampled_cache(&mut self, device: &ash::Device) {
        for slot in self.take_whole_sampled_cache() {
            self.dispose(device, DeferredHandle::RecycleSampled(slot));
        }
    }

    /// Device-free half of [`Self::discard_sampled_cache`]: empty the cache
    /// through the one removal site and return the slots for the caller to
    /// dispose.
    ///
    /// Split out so the accounting is testable without a device.
    fn take_whole_sampled_cache(&mut self) -> Vec<SampledSlot> {
        if self.sampled_cache.is_empty() {
            return Vec::new();
        }
        crate::telemetry::note_route("sampled_cache_discarded");
        let mut taken = Vec::with_capacity(self.sampled_cache.len());
        while !self.sampled_cache.is_empty() {
            taken.push(self.remove_sampled_entry(0));
        }
        taken
    }

    /// Whether a draw at `target` can append to the open batch, and when it
    /// cannot, which of the three reasons it is. Anything but
    /// [`BatchFit::Open`] means the caller must claim its own slot (and
    /// `begin_entry` will flush the batch first).
    ///
    /// A full batch (BATCH_MAX_DRAWS) refuses, turning the next draw into a
    /// flush-then-reopen. Unbounded batches destroyed the pipeline (live A/B
    /// 2026-07-19): the GPU idled while the CPU recorded the whole run — the
    /// present then blocked behind the entire batch executing from scratch
    /// (presents 38.7 -> 27/s) — and every draw's staging slots stayed hoarded
    /// in ONE pending ring entry until its fence retired, starving the free
    /// lists into per-bind vkCreateBuffer/vkAllocateMemory churn (setup_bufs 50
    /// -> 108 us/draw). The cap keeps the GPU fed every ~N draws while still
    /// amortizing the per-draw submit+fence cost N-fold.
    ///
    /// `narrow_to_target` is [`reims_vgpu_config::BATCH_MIXED_TARGETS`] switched off.
    /// The default is that the batch's own target does not decide this: a draw
    /// own, while the flush itself reads only the CB, fence, and accumulated
    /// descriptor sets. The readback rail likewise appends a copy of *some
    /// other* target's image to whatever batch is recording. Passing the
    /// parameter rather than reading the environment here keeps this function
    /// pure and testable.
    pub(crate) fn batch_fit(&self, target: &BatchTarget, narrow_to_target: bool) -> BatchFit {
        let Some(b) = self.open_batch.as_ref() else {
            return BatchFit::None;
        };
        if b.draws >= self.batch_max_draws {
            return BatchFit::Full;
        }
        if narrow_to_target && b.target != *target {
            return BatchFit::OtherTarget;
        }
        BatchFit::Open(b.cb, b.fence)
    }

    /// Whether the open batch is rendering into the same target this draw wants,
    /// or `None` when nothing is recording.
    ///
    /// **This decides nothing.** It exists because every batched draw begins and
    /// ends its own render pass — see [`Self::batch_fit`]'s doc — and a pass can
    /// only ever be shared between draws whose target agrees, so this is the
    /// ceiling on merging them and there was no number for it. Separate from
    /// `batch_fit` because that function is deliberately pure and testable
    /// without a device, and a counter in it would not be.
    pub(crate) fn batch_target_is(&self, target: &BatchTarget) -> Option<bool> {
        self.open_batch.as_ref().map(|b| b.target == *target)
    }

    /// Whether the pass a draw is about to open is the one already standing in
    /// the same command buffer — see [`PassEcho`].
    ///
    /// False whenever nothing is echoed, so the first draw of a command buffer
    /// answers "no" without a special case, which is correct: it has no
    /// predecessor to continue.
    pub(crate) fn pass_echoes(&self, echo: &PassEcho) -> bool {
        self.last_pass.as_ref() == Some(echo)
    }

    /// Which field of the echo stopped this draw continuing, or `None` when it
    /// continues.
    ///
    /// Derived from the same `last_pass` [`Self::pass_echoes`] compares, so the
    /// two cannot disagree: this answers `None` on exactly the inputs that answer
    /// `true` there, which is what makes the split a partition of
    /// `passmerge_pass_differs` rather than a second opinion about it.
    ///
    /// # The framebuffer cannot be the first question, because it answers three
    ///
    /// Every rung here forces a new pass instance, so the order carries no
    /// correctness weight and decides only which bucket a break is charged to.
    /// That makes it an instrument decision, and it has a right answer: **the
    /// guest rendering somewhere else is a guest action nothing here can remove,
    /// and this device describing one target two ways is a defect.** They must
    /// not share a bucket.
    ///
    /// A framebuffer handle cannot separate them. [`super::AdHocFramebufferKey`]
    /// is `(render pass, views, extent)`, so a shape change *implies* a new
    /// framebuffer over the very same attachments — asking `fb` first would put
    /// every shape flip in `passdiff_fb` and leave `passdiff_compat` reading
    /// zero, which is the informative bucket emptied into the uninformative one.
    /// [`PassEcho::target_image`] has no such coupling: it is what the guest
    /// named, not what this device built from it. So the target is asked
    /// **first**, and it subsumes — a break where the image also moved is a
    /// target switch however the shape differs. What reaches
    /// `Compatibility` is then one image this device described two ways, which
    /// is the defect and nothing else.
    ///
    /// Asking the shape first does not do this, and the difference is not
    /// academic: it leaves `passcompat_host_accessible` as the largest bucket in
    /// the census (106 460 of 301 498 pass begins on /tmp/wb-out80) while saying
    /// nothing about whether the image moved with it.
    ///
    /// Why it matters: a per-window regression of `gpu_span busy_us` on
    /// `(draws, pass begins)` over three driven Maps boots puts a **pass
    /// instance at ~100 µs of GPU against ~2 µs for a draw**, which is two
    /// thirds of this iGPU's whole GPU time and the largest single cost in the
    /// device. Which of these five reasons the 169 345 pass begins of a boot
    /// belong to is therefore the ranking question, and each one names a
    /// different repair.
    pub(crate) fn pass_echo_delta(&self, echo: &PassEcho) -> Option<PassEchoField> {
        let Some(last) = self.last_pass.as_ref() else {
            return Some(PassEchoField::Nothing);
        };
        if last.cb != echo.cb {
            return Some(PassEchoField::Cb);
        }
        if last.target_image != echo.target_image {
            return Some(PassEchoField::Target);
        }
        if let Some(field) = last.compatibility.first_difference(echo.compatibility) {
            return Some(PassEchoField::Compatibility(field));
        }
        if last.fb != echo.fb {
            return Some(PassEchoField::Framebuffer);
        }
        if last.area != echo.area {
            return Some(PassEchoField::Area);
        }
        None
    }

    /// Record the pass a draw just opened. Called at the `vkCmdBeginRenderPass`
    /// and nowhere else, so the echo always names a pass that is standing.
    pub(crate) fn note_pass_opened(&mut self, echo: PassEcho) {
        self.last_pass = Some(echo);
        self.open_pass = Some(echo);
    }

    /// Whether `echo` is the render pass that is actually still open.
    pub(crate) fn open_pass_echoes(&self, echo: &PassEcho) -> bool {
        self.open_pass.as_ref() == Some(echo)
    }

    /// End the pass in `cb` before recording a command that cannot live inside
    /// it or ending the command buffer. No-op when the preceding draw ended its
    /// pass normally.
    pub(crate) unsafe fn close_open_pass(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        let Some(open) = self.open_pass.take() else {
            return;
        };
        debug_assert_eq!(
            open.cb, cb,
            "open render pass belongs to another command buffer"
        );
        unsafe { device.cmd_end_render_pass(open.cb) };
    }

    /// Forget everything remembered about the command buffer that was
    /// recording: the echoed pass and the graphics state it carried.
    ///
    /// Called wherever the command buffer holding them stops being the one a
    /// draw would record into — a reset at `begin_entry`, a submit at the batch
    /// flush, and teardown. Missing one would let a joiner believe a pass ended
    /// in a previous submission is still open, which is why this is a method
    /// rather than an assignment at each site.
    ///
    /// Both halves are dropped by one call because they are one fact, and
    /// because the second is the more dangerous to get wrong. A stale echo makes
    /// a draw *skip a `vkCmdBeginRenderPass`* only in an instrument that decides
    /// nothing; a stale [`CbGraphicsState`] makes it skip a real
    /// `vkCmdSetViewport`, and a recycled handle is exactly the case where that
    /// state was made undefined by a `vkBeginCommandBuffer` the cache never saw.
    /// The handle comparison inside that struct is the second lock on the same
    /// door, not the first.
    pub(crate) fn forget_pass_echo(&mut self) {
        debug_assert!(self.open_pass.is_none(), "forgetting an open render pass");
        self.last_pass = None;
        self.cb_graphics.cb = None;
        self.cb_graphics.pipeline = None;
        self.cb_graphics.pipeline_layout = None;
        self.cb_graphics.viewports.clear();
        self.cb_graphics.scissors.clear();
        self.cb_graphics.stencil = None;
        self.cb_graphics.push_layout = None;
        self.cb_graphics.push_bindings.clear();
        self.cb_guest_visibility = super::CbGuestVisibility::default();
    }

    /// Classify and return the imported-memory dependency `cb` still owes.
    ///
    /// Within one decoded guest operation, the first host dependency covers its
    /// imported consumers. [`Self::begin_guest_operation`] re-arms that half
    /// before a later operation can rely on the same command buffer. A GPU
    /// write recorded later dirties the GPU half; the caller's exact
    /// page-overlap classification decides whether the next read consumes it.
    pub(in crate::engine) fn imported_guest_barrier(
        &mut self,
        cb: vk::CommandBuffer,
        classify: impl FnOnce() -> super::super::exec::ImportedGuestVisibility,
    ) -> Option<super::super::exec::ImportedGuestVisibility> {
        if self.cb_guest_visibility.cb != Some(cb) {
            self.cb_guest_visibility = super::CbGuestVisibility {
                cb: Some(cb),
                host_visible: false,
                gpu_write_since_barrier: false,
            };
        }
        if self.cb_guest_visibility.host_visible
            && !self.cb_guest_visibility.gpu_write_since_barrier
        {
            return None;
        }
        let visibility = classify();
        let includes_gpu_writes = visibility.includes_gpu_writes();
        let needed = !self.cb_guest_visibility.host_visible
            || (includes_gpu_writes && self.cb_guest_visibility.gpu_write_since_barrier);
        if needed {
            self.cb_guest_visibility.host_visible = true;
            if includes_gpu_writes {
                self.cb_guest_visibility.gpu_write_since_barrier = false;
            }
        }
        needed.then_some(visibility)
    }

    /// Begin one decoded guest operation recorded into `cb`.
    ///
    /// Guest CPU writes are external to this backend, so no command-buffer
    /// state can observe that imported pages changed between two decoded
    /// operations. A host-write dependency recorded for the earlier operation
    /// therefore cannot satisfy the later one. Re-arm host visibility at every
    /// operation boundary while preserving GPU-write debt recorded in this
    /// command buffer; [`Self::imported_guest_barrier`] may still deduplicate
    /// multiple imported-memory consumers inside the operation.
    pub(in crate::engine) fn begin_guest_operation(&mut self, cb: vk::CommandBuffer) {
        if self.cb_guest_visibility.cb != Some(cb) {
            self.cb_guest_visibility = super::CbGuestVisibility {
                cb: Some(cb),
                host_visible: false,
                gpu_write_since_barrier: false,
            };
        } else {
            self.cb_guest_visibility.host_visible = false;
        }
        self.forget_cb_guest_cpu_snapshots();
    }

    /// Enter one draw record of a decoded render encoder.
    ///
    /// The first record establishes host-write visibility for the resources the
    /// encoder will reference. Continuation records in the same command buffer
    /// do not invent a CPU publication point between Metal draw calls. If an
    /// explicit guest barrier or another consumer split the encoder across
    /// Vulkan command buffers, the new handle still starts a fresh dependency.
    pub(in crate::engine) fn enter_render_encoder_record(
        &mut self,
        cb: vk::CommandBuffer,
        continues_render_encoder: bool,
    ) {
        if !continues_render_encoder || self.cb_guest_visibility.cb != Some(cb) {
            self.begin_guest_operation(cb);
        }
    }

    fn note_cb_guest_write(&mut self) {
        self.cb_guest_visibility.gpu_write_since_barrier = true;
    }

    /// Retain one target resident through the fence of the command buffer now
    /// being recorded. Every successful pin enters the one list transferred by
    /// [`Self::seal_entry`], so sampled aliases and guest-page copies cannot
    /// acquire a pin without also declaring its release point.
    pub(in crate::engine) fn pin_resident_for_entry(&mut self, identity: &TargetIdentity) -> bool {
        if !self.pin_resident_target(identity, true) {
            return false;
        }
        self.resident_pins_live.push(identity.clone());
        true
    }

    fn install_open_batch(&mut self, batch: OpenBatch) {
        debug_assert!(self.open_batch.is_none(), "replacing an open batch");
        self.open_batch = Some(batch);
        super::super::publish_batch_open(true);
    }

    fn take_open_batch(&mut self) -> Option<OpenBatch> {
        let batch = self.open_batch.take();
        super::super::publish_batch_open(false);
        batch
    }

    pub(super) fn discard_open_batch(&mut self) {
        self.open_batch = None;
        super::super::publish_batch_open(false);
    }

    /// Record a batch-deferred draw's completion: open the batch on its ring
    /// slot (opener) or extend it (joiner), accumulating the per-draw descriptor
    /// set for the single flush-time seal. The CB stays in recording state;
    /// submit happens at [`Self::batch_flush`].
    ///
    /// # Exact-content sampled images go to the cache here, not at the flush
    ///
    /// A batch is several draws sharing one command buffer, and the next draw's
    /// `find_cached_sampled` runs before that buffer is submitted. Publishing at
    /// record completion lets later draws reuse byte-equal CPU-sourced uploads.
    /// Guest-page gathers remain outside that cross-command-buffer cache, but
    /// their exact source identity can reuse the copied image within this same
    /// command buffer until a recorded guest write overlaps its pages.
    ///
    /// Publishing now is sound because the fill is *recorded* now, into the same
    /// command buffer and ahead of any consumer, and because setting
    /// `open_batch` is exactly what puts this slot in [`Self::open_slot_mask`] —
    /// see [`Self::admit_recorded_sampled`] for why that is the precondition.
    /// The opener sets it first, below, so the mask is right for its own
    /// admissions too.
    ///
    /// What it costs is a promise: these entries claim content a command buffer
    /// has not yet delivered. [`Self::batch_flush`] keeps it or, if the submit
    /// fails, calls [`Self::discard_sampled_cache`].
    ///
    pub(crate) unsafe fn batch_append(
        &mut self,
        // The pair [`Self::batch_slot`] and [`Self::begin_entry`] both hand
        // back, passed through as one value because it only ever travels as one.
        slot: (vk::CommandBuffer, vk::Fence),
        target: BatchTarget,
        dset: Option<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<SampledRetain>,
        counters: &EngineCounters,
    ) {
        let (cb, fence) = slot;
        match self.open_batch.as_mut() {
            Some(b) => {
                debug_assert!(b.cb == cb, "joiner recorded into a foreign CB");
                b.draws += 1;
                b.dsets.extend(dset);
                counters.batch_joins.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                super::super::note_batch_open_after_tail(counters);
                debug_assert!(
                    self.slots[self.cur].pending.is_none(),
                    "batch opener's slot already owes cleanup"
                );
                self.install_open_batch(OpenBatch {
                    cb,
                    fence,
                    target,
                    draws: 1,
                    dsets: dset.into_iter().collect(),
                });
                counters.batch_opens.fetch_add(1, Ordering::Relaxed);
            }
        }
        let admissions = take_retained_slots(&mut self.sampled_live, sampled_retains);
        self.admit_recorded_sampled(admissions);
    }

    /// Submit the open batch (if any): end its CB, queue it on the batch
    /// fence, and park the accumulated cleanup on its ring slot. No-op when no
    /// batch is open. On submit failure the descriptor sets are freed
    /// immediately (the CB never reached the queue) and the pool-slot lives
    /// stay for the next seal, matching the per-draw submit-error path.
    pub(crate) unsafe fn batch_flush(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        unsafe { self.batch_flush_inner(ctx, counters) }
    }

    unsafe fn batch_flush_inner(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        let Some(mut batch) = self.take_open_batch() else {
            return Ok(());
        };
        let close_started = std::time::Instant::now();
        // The CB is about to be ended and submitted, so no pass inside it is
        // still open to continue.
        unsafe { self.close_open_pass(&ctx.device, batch.cb) };
        self.forget_pass_echo();
        counters.batch_flushes.fetch_add(1, Ordering::Relaxed);
        counters
            .batch_flush_draws
            .fetch_add(batch.draws, Ordering::Relaxed);
        // `self.cur` is still the slot the batch was opened on: `begin_entry`
        // flushes the open batch *before* it advances, and every other flush
        // caller reaches here without claiming a slot of its own. Sealing against
        // any other index would charge this submission's GPU span to a slot whose
        // queries a different command buffer wrote.
        let slot = self.cur;
        unsafe { self.gpu_span_seal(ctx, batch.cb, slot) };
        counters.batch_flush_close_us.fetch_add(
            close_started.elapsed().as_micros() as u64,
            Ordering::Relaxed,
        );
        let submit = (|| -> Result<super::super::context::AsyncGuestSubmission, DrawError> {
            let end_started = std::time::Instant::now();
            let end_result = ctx.device.end_command_buffer(batch.cb);
            counters
                .batch_flush_end_us
                .fetch_add(end_started.elapsed().as_micros() as u64, Ordering::Relaxed);
            end_result.map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsEndCbBatch, e)))?;
            let cbs = [batch.cb];
            let submit_started = std::time::Instant::now();
            let result = ctx.submit_guest_work_async(&cbs, batch.fence);
            counters.batch_flush_submit_us.fetch_add(
                submit_started.elapsed().as_micros() as u64,
                Ordering::Relaxed,
            );
            match result {
                Ok(submission) => Ok(submission),
                Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
                    Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                        op: DeviceLostOp::PoolsSubmitBatch,
                        result: e,
                    }))
                }
                Err(e) => Err(DrawError::VkCall(VkCall::new(VkOp::PoolsSubmitBatch, e))),
            }
        })();
        match submit {
            Ok(submission) => {
                let finish_started = std::time::Instant::now();
                let sealed = self.seal_entry(std::mem::take(&mut batch.dsets), Vec::new());
                self.finish_entry_async(sealed, submission.timeline, submission.driver_return);
                counters.batch_flush_finish_us.fetch_add(
                    finish_started.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                Ok(())
            }
            Err(e) => {
                self.desc_arena.free(&ctx.device, &batch.dsets);
                self.abort_recorded_guest_work();
                // This batch's draws published sampled images to the content
                // cache on the promise that this command buffer would fill
                // them. It never reached the queue, so their contents are
                // undefined and a later bind would sample them as if they held
                // a guest window. Nothing tracks which entries were this
                // batch's, so the whole cache goes — see
                // `discard_sampled_cache` for why that is the right shape and
                // not a shortcut.
                self.discard_sampled_cache(&ctx.device);
                Err(e)
            }
        }
    }

    /// Wait a single already-submitted entry fence WITHOUT retiring the ring.
    ///
    /// A synchronous reader (e.g. [`super::super::read_target_inner`]) that submitted
    /// its own copy CB with `fence` only needs *that* copy to finish before it
    /// maps the readback — it does not need to quiesce unrelated in-flight
    /// draws. The copy's `ALL_COMMANDS → TRANSFER` barrier plus single-queue
    /// submission order already guarantee it observes every prior-submitted
    /// draw's writes (the same argument the async prefetch path relies on), so
    /// waiting the whole ring here would just serialize the guest-blocking
    /// readback behind an unrelated heavy draw — the `finish_us` tail. The
    /// caller must have parked its cleanup with [`Self::finish_entry_async`]
    /// first; the slot stays pending and the ring retires it later (its fence
    /// is already signaled, so that retire is a no-wait drain).
    pub(crate) unsafe fn wait_entry_fence(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        fence: vk::Fence,
    ) -> Result<(), DrawError> {
        if let Some(index) = self
            .slots
            .iter()
            .position(|slot| slot.fence == fence && slot.pending.is_some())
        {
            self.await_submit_return(counters, index, DeviceLostOp::PoolsWaitFencesEntry)?;
        }
        if let Some(error) = ctx.queue_failure() {
            return Err(Self::wait_error(
                counters,
                error,
                DeviceLostOp::PoolsWaitFencesEntry,
            ));
        }
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesEntry))
    }

    /// Record that the command buffer being built reads guest RAM when it
    /// executes, so the next completion stamp waits for it.
    pub(crate) fn note_guest_read_recorded(&mut self) {
        self.guest_reads_in_flight = true;
        super::super::publish_guest_read_debt(true);
    }

    /// The buffer this command buffer already staged or gathered for the bind
    /// `key` identifies, if it still holds one. See
    /// `ResourcePools::cb_bound_buffers`.
    ///
    /// Takes the key and **not** a [`super::CbBind`], because a lookup needs the
    /// identity and not the ownership. Building the `CbBind` first costs an `Arc`
    /// clone — two atomics here and two more when it drops — and this is the hit
    /// path: a Metal argument table is sticky across the draws of an encoder, so
    /// the same buffers are re-presented on every draw and almost every probe
    /// hits. The `Arc` is what keeps a recorded key's address from being reissued
    /// under a live entry, and that guarantee belongs to
    /// [`Self::note_cb_bound_buffer`], which still takes the `CbBind` by value.
    /// So the invariant is unchanged and only the miss path pays for it.
    pub(in crate::engine) fn cb_bound_buffer(
        &self,
        key: (usize, u64, u64),
    ) -> Option<super::super::exec::BoundBuffer> {
        self.cb_bound_buffers
            .aliases
            .get(&key)
            .or_else(|| self.cb_bound_buffers.immutable.get(&key))
            .or_else(|| self.cb_bound_buffers.guest_snapshots.get(&key))
            .map(|(b, _)| *b)
    }

    /// Remember that `bind`'s bytes are in `bound` for the rest of this command
    /// buffer.
    ///
    /// Takes the [`super::CbBind`] by value because the entry has to keep the
    /// `Arc` inside it: the map is keyed on that allocation's address, and an
    /// address whose allocation has been freed is one the next unrelated bind of
    /// the same length can be handed. Holding it is what makes the key unique
    /// for as long as it is answerable.
    pub(in crate::engine) fn note_cb_bound_buffer(
        &mut self,
        bind: super::CbBind,
        bound: super::super::exec::BoundBuffer,
    ) {
        let (key, owner) = bind.into_parts();
        self.cb_bound_buffers.aliases.remove(&key);
        self.cb_bound_buffers.immutable.remove(&key);
        self.cb_bound_buffers.guest_snapshots.remove(&key);
        if bound.guest_import {
            self.cb_bound_buffers.aliases.insert(key, (bound, owner));
        } else if matches!(&owner.allocation, super::CbBindAllocation::Bytes(_)) {
            self.cb_bound_buffers.immutable.insert(key, (bound, owner));
        } else {
            self.cb_bound_buffers
                .guest_snapshots
                .insert(key, (bound, owner));
        }
    }

    pub(in crate::engine) fn cb_sampled_guest(
        &self,
        source: &super::CbSampledGuest,
    ) -> Option<super::SampledSlot> {
        let found = self
            .cb_sampled_guest
            .get(&source.key)
            .map(|(slot, _)| slot.handles());
        if found.is_some() {
            crate::telemetry::note_route("sampled_guest_cb_reuse");
        }
        found
    }

    pub(in crate::engine) fn note_cb_sampled_guest(
        &mut self,
        source: super::CbSampledGuest,
        image: &super::SampledSlot,
    ) {
        self.cb_sampled_guest
            .insert(source.key, (image.handles(), source.owner));
    }

    /// Record that the bind just published is backed by a **recycled slot whose
    /// gather has not been recorded yet** — see [`super::ResourcePools::cb_gather_owed`].
    ///
    /// Called from the one arm of `stage_buffer_content` that hands back a slot
    /// it has not filled. Everything published without this call is answerable
    /// the moment it is published.
    pub(crate) fn note_cb_bind_owes_gather(&mut self, key: (usize, u64, u64)) {
        self.cb_gather_owed.push(key);
    }

    /// The owed gathers have been recorded into the command buffer, so every
    /// bind that was waiting on one is now answerable.
    ///
    /// Called at the single point in `execute_draw_inner` that records them,
    /// after both forms — the compute dispatches and the transfer copies — and
    /// after the barrier that orders them before the draw.
    pub(crate) fn note_cb_gathers_recorded(&mut self) {
        self.cb_gather_owed.clear();
    }

    /// A draw abandoned before its gathers were recorded: forget exactly the
    /// binds those gathers were going to fill.
    ///
    /// The rest of the memo is untouched, because a bind published by a draw
    /// that completed is still correct and this rail carries ~4.8 binds a draw.
    /// Returns how many were forgotten so a boot can say whether the window this
    /// closes was ever open.
    pub(crate) fn discard_cb_binds_owed_a_gather(&mut self) -> usize {
        let n = self.cb_gather_owed.len();
        let keys = std::mem::take(&mut self.cb_gather_owed);
        for key in keys {
            self.cb_bound_buffers.aliases.remove(&key);
            self.cb_bound_buffers.immutable.remove(&key);
            self.cb_bound_buffers.guest_snapshots.remove(&key);
        }
        n
    }

    /// Remove copied guest snapshots while retaining direct aliases and
    /// immutable byte allocations.
    ///
    /// `guest_import` means the descriptor names the guest allocation itself.
    /// A Store changes what a later shader reads through it, not which Vulkan
    /// buffer and offset name those bytes. Staging and gather entries instead
    /// name copies taken before the Store and are all invalid after it.
    fn forget_cb_bound_buffer_copies(&mut self) {
        let invalidated = self.cb_bound_buffers.guest_snapshots.len() as u64;
        let retained = self.cb_bound_buffers.aliases.len() as u64;
        self.cb_bound_buffers.guest_snapshots.clear();
        crate::telemetry::note_route_n("bindmap_write_invalidated", invalidated);
        crate::telemetry::note_route_n("bindmap_write_alias_retained", retained);
    }

    /// Invalidate every snapshot whose source is guest-mutable at a decoded
    /// operation boundary.
    ///
    /// The guest CPU writes outside this backend, so allocation identity is not
    /// a content version. Direct aliases remain valid because they still name
    /// the live bytes; `BufferContent::Bytes` remains valid because its `Arc`
    /// owns immutable content. Gathered buffers and uploaded sampled images are
    /// snapshots and must be rebuilt for the next operation.
    fn forget_cb_guest_cpu_snapshots(&mut self) {
        self.forget_cb_bound_buffer_copies();
        self.forget_cb_sampled_guest();
    }

    /// Drop every cached buffer bind, and say how many were dropped and why.
    ///
    /// Both callers are about the slots: `seal_entry` and `recycle_staging` are
    /// handing the staging slots the map names to a cleanup or a free list, so
    /// nothing recorded after them may name one. Guest writes instead clear
    /// only copied snapshots and retain direct aliases.
    fn forget_cb_bound_buffers(&mut self, why: &'static str, entries_slug: &'static str) {
        let n = (self.cb_bound_buffers.aliases.len()
            + self.cb_bound_buffers.immutable.len()
            + self.cb_bound_buffers.guest_snapshots.len()) as u64;
        crate::telemetry::note_route(why);
        crate::telemetry::note_route_n(entries_slug, n);
        self.cb_bound_buffers = super::CbBufferMemo::default();
        // The owed list names keys in the map that just went, so it cannot
        // outlive it — a surviving key would later remove whatever unrelated
        // bind the next command buffer published under the same address.
        self.cb_gather_owed.clear();
    }

    fn forget_cb_sampled_guest(&mut self) {
        crate::telemetry::note_route_n(
            "sampled_bindmap_clear_entries",
            self.cb_sampled_guest.len() as u64,
        );
        self.cb_sampled_guest.clear();
    }

    /// Invalidate only sampled copies whose source overlaps the pages a Store
    /// will change. An entry without canonical page identity cannot prove
    /// disjointness and is removed conservatively.
    fn forget_cb_sampled_guest_pages(&mut self, written: &reims_vgpu_memory::GuestWritePages) {
        let mut overlap = 0u64;
        let mut unknown = 0u64;
        let mut disjoint = 0u64;
        self.cb_sampled_guest
            .retain(|_, (_, owner)| match owner.physical_pages.as_ref() {
                Some(held) if held.overlaps(written) => {
                    overlap += 1;
                    false
                }
                Some(_) => {
                    disjoint += 1;
                    true
                }
                None => {
                    unknown += 1;
                    false
                }
            });
        crate::telemetry::note_route_n("sampled_bindmap_write_invalidated", overlap + unknown);
        crate::telemetry::note_route_n("sampled_bindmap_write_overlap", overlap);
        crate::telemetry::note_route_n("sampled_bindmap_write_unknown", unknown);
        crate::telemetry::note_route_n("sampled_bindmap_write_disjoint", disjoint);
    }

    /// Bind the graphics pipeline unless this command buffer already carries it.
    ///
    /// A pipeline change clears the dynamic half of [`CbGraphicsState`], which is
    /// the rule that makes the three skips below sound whatever dynamic-state
    /// list each pipeline was built with — see that type's doc.
    ///
    /// # Safety
    ///
    /// `cb` must be a command buffer in the recording state, and `pipeline` a
    /// live graphics pipeline compatible with what the draw is about to record.
    pub(crate) unsafe fn bind_graphics_pipeline(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        counters: &EngineCounters,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let g = &mut self.cb_graphics;
        if g.cb != Some(cb) {
            // A recycled handle: everything the previous user of it bound was
            // made undefined by the `vkBeginCommandBuffer` in between.
            g.cb = Some(cb);
            g.pipeline = None;
            g.pipeline_layout = None;
            g.viewports.clear();
            g.scissors.clear();
            g.stencil = None;
            g.push_layout = None;
            g.push_bindings.clear();
        }
        if g.pipeline == Some(pipeline) {
            counters
                .dynstate_pipeline_held
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        g.pipeline = Some(pipeline);
        if g.pipeline_layout != Some(pipeline_layout) {
            g.pipeline_layout = Some(pipeline_layout);
            g.push_layout = None;
            g.push_bindings.clear();
        }
        // Static state on the incoming pipeline may have replaced any of these,
        // so none of them is known any more.
        g.viewports.clear();
        g.scissors.clear();
        g.stencil = None;
        unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline) };
    }

    /// The scratch arrays a draw builds its viewport and scissor lists into,
    /// cleared and ready.
    ///
    /// Handed out rather than allocated per draw: the lists are rebuilt every
    /// draw and were two `Vec` allocations on the recording path, and the
    /// comparison in [`Self::set_dynamic_viewport_scissor`] needs them in a
    /// buffer it can swap rather than copy.
    pub(crate) fn dynamic_scratch(&mut self) -> (&mut Vec<vk::Viewport>, &mut Vec<vk::Rect2D>) {
        let g = &mut self.cb_graphics;
        g.vp_scratch.clear();
        g.sc_scratch.clear();
        (&mut g.vp_scratch, &mut g.sc_scratch)
    }

    /// Record `vkCmdSetViewport` / `vkCmdSetScissor` from the scratch arrays,
    /// each only if this command buffer is not already carrying exactly it.
    ///
    /// # Safety
    ///
    /// `cb` must be the recording command buffer most recently passed to
    /// [`Self::bind_graphics_pipeline`], so the pipeline-change rule that
    /// authorises the skip has been applied.
    pub(crate) unsafe fn set_dynamic_viewport_scissor(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        counters: &EngineCounters,
    ) {
        let g = &mut self.cb_graphics;
        if super::viewports_match(&g.vp_scratch, &g.viewports) {
            counters
                .dynstate_viewport_held
                .fetch_add(1, Ordering::Relaxed);
        } else {
            std::mem::swap(&mut g.viewports, &mut g.vp_scratch);
            unsafe { device.cmd_set_viewport(cb, 0, &g.viewports) };
        }
        if super::scissors_match(&g.sc_scratch, &g.scissors) {
            counters
                .dynstate_scissor_held
                .fetch_add(1, Ordering::Relaxed);
        } else {
            std::mem::swap(&mut g.scissors, &mut g.sc_scratch);
            unsafe { device.cmd_set_scissor(cb, 0, &g.scissors) };
        }
    }

    /// The scissor rectangles this command buffer is carrying, which are the
    /// recording draw's.
    ///
    /// True on both arms of the skip and that is the point: a draw that set them
    /// swapped its own array in, and a draw that skipped did so *because* the
    /// array already there is bit-for-bit what it would have sent. A consumer
    /// that kept its own copy would be a second spelling of the same rects.
    pub(crate) fn bound_scissors(&self) -> &[vk::Rect2D] {
        &self.cb_graphics.scissors
    }

    /// Record the draw's vertex buffers as maximal consecutive binding runs,
    /// preserving every exact `(buffer, offset)` value in the request.
    ///
    /// The guest bulk operation is a pair of parallel buffer/offset arrays.
    /// Vulkan has the same operation but requires consecutive binding numbers,
    /// so gaps split the request into runs. All arrays live in the command
    /// buffer's reusable graphics scratch; the hot path allocates nothing after
    /// their first high-water.
    ///
    /// # Safety
    ///
    /// `cb` must be the recording command buffer most recently passed to
    /// [`Self::bind_graphics_pipeline`]. Every buffer must remain alive through
    /// submission, as required by `vkCmdBindVertexBuffers`.
    pub(in crate::engine) unsafe fn bind_vertex_buffers(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        counters: &EngineCounters,
        requested: &[(u32, super::super::exec::BoundBuffer)],
    ) {
        let g = &mut self.cb_graphics;
        counters
            .vertex_buffer_bind_slots
            .fetch_add(requested.len() as u64, Ordering::Relaxed);

        g.vertex_scratch.clear();
        g.vertex_scratch
            .extend(
                requested
                    .iter()
                    .map(|(binding, bound)| super::VertexBufferBinding {
                        binding: *binding,
                        buffer: bound.buffer,
                        offset: bound.offset,
                    }),
            );
        super::normalize_vertex_bindings(&mut g.vertex_scratch);
        counters
            .vertex_buffer_bind_emitted
            .fetch_add(g.vertex_scratch.len() as u64, Ordering::Relaxed);

        g.vertex_buffers.clear();
        g.vertex_offsets.clear();
        g.vertex_buffers
            .extend(g.vertex_scratch.iter().map(|entry| entry.buffer));
        g.vertex_offsets
            .extend(g.vertex_scratch.iter().map(|entry| entry.offset));

        let mut start = 0;
        while start < g.vertex_scratch.len() {
            let end = super::vertex_binding_run_end(&g.vertex_scratch, start);
            unsafe {
                device.cmd_bind_vertex_buffers(
                    cb,
                    g.vertex_scratch[start].binding,
                    &g.vertex_buffers[start..end],
                    &g.vertex_offsets[start..end],
                )
            };
            counters
                .vertex_buffer_bind_calls
                .fetch_add(1, Ordering::Relaxed);
            start = end;
        }
    }

    /// Scratch in which the next draw normalizes its push-descriptor state.
    pub(crate) fn push_descriptor_scratch(&mut self) -> &mut Vec<super::PushDescriptorBinding> {
        self.cb_graphics.push_scratch.clear();
        &mut self.cb_graphics.push_scratch
    }

    /// Return whether this draw must record its push descriptors, retaining a
    /// byte-exact echo when it does.
    pub(crate) fn push_descriptors_changed(
        &mut self,
        layout: vk::PipelineLayout,
        counters: &EngineCounters,
    ) -> bool {
        let g = &mut self.cb_graphics;
        if super::push_descriptors_match(g.push_layout, &g.push_bindings, layout, &g.push_scratch) {
            counters
                .descriptor_push_held
                .fetch_add(1, Ordering::Relaxed);
            false
        } else {
            g.push_layout = Some(layout);
            std::mem::swap(&mut g.push_bindings, &mut g.push_scratch);
            true
        }
    }

    /// Record both `vkCmdSetStencilReference` faces unless this command buffer
    /// already carries exactly this pair.
    ///
    /// # Safety
    ///
    /// As [`Self::set_dynamic_viewport_scissor`].
    pub(crate) unsafe fn set_dynamic_stencil_reference(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        counters: &EngineCounters,
        front: u32,
        back: u32,
    ) {
        let g = &mut self.cb_graphics;
        if g.stencil == Some((front, back)) {
            counters
                .dynstate_stencil_held
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        g.stencil = Some((front, back));
        unsafe {
            device.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT, front);
            device.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::BACK, back);
        }
    }

    /// Whether a recorded command buffer still owes an ordering point for its
    /// guest-memory reads.
    pub(crate) fn has_guest_read_debt(&self) -> bool {
        self.guest_reads_in_flight
    }

    /// Mark every guest read before the caller's established ordering point as
    /// settled.
    ///
    /// Checking and settling are separate operations because a failed fence
    /// wait establishes no ordering point. Clearing before that wait would let
    /// the next completion treat still-running GPU reads as finished.
    pub(crate) fn settle_guest_read_debt(&mut self) {
        self.guest_reads_in_flight = false;
        super::super::publish_guest_read_debt(false);
    }

    /// Record that a submitted command buffer **writes** guest RAM when it
    /// executes, and hold `identity`'s image against reclaim until it has.
    ///
    /// Both halves are one call because they are one fact: this copy has not
    /// happened yet. Whoever asked for it may not read those guest bytes and the
    /// registry may not reclaim that image until the fence says otherwise, and a
    /// site that recorded the debt without taking the pin would satisfy the first
    /// while breaking the second.
    ///
    /// # The pin is taken here, not handed in
    ///
    /// It used to be handed in: the deferred-Store rail pinned the resident,
    /// then gave that pin to this ledger to release after execution. That rail was
    /// removed with `runtime::storage_flush`, and it held every product caller
    /// of [`ResourcePools::pin_resident_target`]`(_, true)` — so this ledger went
    /// on releasing a pin nobody had taken, once per GPU-direct writeback, and
    /// the guard in `pin_resident_target` reported each one.
    ///
    /// Nothing else covered the window it left. `gpu_only_content` is what both
    /// reclaim paths actually skip on, and the render writeback clears it the
    /// moment `copy_target_to_guest_pages` returns — while the copy is submitted
    /// and not yet executed. The pin is the only thing holding the image from
    /// there to the settle.
    ///
    /// Taking it here rather than trusting a caller makes that unrepresentable:
    /// the ledger releases exactly the pins it took, so a handoff cannot be
    /// mismatched by a caller that forgets, and a second holder (the host
    /// window's present pin is the live one) can no longer have its pin released
    /// on its behalf. A pin that cannot be taken — no slot, or content not ready
    /// — records the debt and nothing else, because there is then no image for
    /// the reclaim to take.
    pub(crate) fn note_guest_write_pages_recorded(
        &mut self,
        source: super::super::GuestWriteSource<'_>,
        pages: &reims_vgpu_memory::GuestWritePages,
    ) {
        self.note_cb_guest_write();
        // A copied snapshot predating this Store is stale. A direct guest-memory
        // alias remains the same binding and observes the Store through the
        // command buffer's imported-memory visibility dependency.
        self.forget_cb_bound_buffer_copies();
        self.forget_cb_sampled_guest_pages(pages);
        if !self.guest_write_tokens_live.contains_key(pages) {
            if let Some(token) = super::super::arm_guest_write_pages(pages.clone()) {
                self.guest_write_tokens_live.insert(pages.clone(), token);
            }
        }
        self.note_guest_write_source(source);
    }

    fn note_guest_write_source(&mut self, source: super::super::GuestWriteSource<'_>) {
        match source {
            super::super::GuestWriteSource::ResidentTarget(identity) => {
                self.pin_resident_for_entry(identity);
            }
            // Same ledger discipline, the other registry. Recorded separately
            // because the two are keyed differently and the release has to reach
            // the registry that holds the image.
            super::super::GuestWriteSource::ResidentStorage(identity) => {
                if self.pin_resident_storage(identity, true) {
                    self.compute_write_pins_live.push(*identity);
                }
            }
            // The parent import is device-owned and its retirement is deferred
            // against this submission's ring slot. There is no separate
            // resident object to pin.
            super::super::GuestWriteSource::ImportedBuffer => {}
            // Nothing to pin: the ring entry this submission sealed already owns
            // the image and will not recycle it before the fence retires, so a
            // pin here would be a second owner of one lifetime and the ledger
            // would have a release to get right for no gain.
            super::super::GuestWriteSource::RingEntry => {}
        }
    }

    #[cfg(test)]
    fn note_guest_write_recorded(
        &mut self,
        source: super::super::GuestWriteSource<'_>,
        token: Option<super::super::GuestWriteToken>,
    ) {
        // Unit tests drive ownership transfer without installing a global
        // session. Give each supplied token its own exact synthetic page set;
        // production tokens are created only by `note_guest_write_pages_recorded`.
        if let Some(token) = token {
            let pages = reims_vgpu_memory::GuestWritePages::new(&[token.serial])
                .expect("the synthetic page set is non-empty");
            self.guest_write_tokens_live.insert(pages, token);
        }
        self.forget_cb_bound_buffers(
            "bindmap_clear_guestwrite",
            "bindmap_clear_guestwrite_entries",
        );
        self.forget_cb_sampled_guest();
        self.note_guest_write_source(source);
    }

    /// Withdraw guest-memory state recorded into a command buffer that failed
    /// before submission. No fence owns these tokens or resident pins, so
    /// carrying them into the next entry would invent a lifetime the GPU never
    /// began.
    pub(crate) fn abort_recorded_guest_work(&mut self) {
        let tokens: Vec<_> = self.guest_write_tokens_live.values().copied().collect();
        super::super::retire_guest_write_pages(&tokens);
        self.guest_write_tokens_live.clear();
        for identity in std::mem::take(&mut self.resident_pins_live) {
            self.pin_resident_target(&identity, false);
        }
        for identity in std::mem::take(&mut self.compute_write_pins_live) {
            self.pin_resident_storage(&identity, false);
        }
    }

    /// Wait until every guest-page writeback this device has recorded has
    /// landed.
    ///
    /// The obligation this settles is the one `flush_all_windows_before_fence`
    /// used to settle inline, one blocking fence per window. It is the same
    /// wait; what changed is that it happens once for all of them and after
    /// they were all submitted, so the queue runs them back to back instead of
    /// stopping between each.
    ///
    /// Every caller is a point where the copy becoming visible stops being
    /// this device's own business: the completion stamp, and the flush-on-access
    /// choke points where a host reader or writer is about to touch the same
    /// guest bytes. A new one of either is a new caller.
    pub(crate) unsafe fn quiesce_guest_writes(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        unsafe { self.retire_all(ctx, counters) }
    }

    /// Wait until nothing this device has recorded will read guest RAM again.
    ///
    /// The completion stamp is the only thing this device says to the guest
    /// about whether work is finished, and once it moves the guest may repaint
    /// or free everything that work named. `flush_all_windows_before_fence`
    /// settles what the device still owes guest RAM; this settles what it is
    /// still *reading* from guest RAM, which is the other half of the same
    /// sentence and the one that only exists because a draw can bind guest pages
    /// directly.
    ///
    /// Retiring the whole ring is deliberately blunter than waiting the fences
    /// that actually carry a guest read. The open batch has to be flushed
    /// regardless — an unsubmitted CB's read has not happened yet, so waiting
    /// fences alone would let it happen *after* the stamp — and once the batch
    /// is submitted, the slots ahead of it are the ones about to be reclaimed
    /// anyway. Cheap where it matters: this runs at a stamp, and a stamp that
    /// flushed a window has already waited that flush's own fence, which is
    /// ordered behind every draw the flush read.
    ///
    /// A no-op when nothing recorded a guest read, which is every packet on a
    /// host with no host-pointer import.
    pub(crate) unsafe fn quiesce_guest_reads(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        if !self.has_guest_read_debt() {
            return Ok(());
        }
        unsafe { self.retire_all(ctx, counters)? };
        self.settle_guest_read_debt();
        Ok(())
    }

    /// Wait + retire every in-flight slot and drain the graveyard. Callers
    /// park their own cleanup with [`Self::finish_entry_async`] right after
    /// submit, then call this for a synchronous result — a failed wait leaves
    /// the slot pending, and whichever entry claims it next re-waits, so no
    /// path ever submits on an unretired fence.
    pub(crate) unsafe fn retire_all(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        // Quiesce includes deferred batched work: submit it so the retire
        // below actually waits it out.
        self.batch_flush(ctx, counters)?;
        for index in 0..self.slots.len() {
            self.retire_slot(ctx, counters, index)?;
        }
        // A slot that owed nothing retires without clearing its bit, and a
        // batch whose submit failed leaves its opener's bit set with no fence
        // behind it. Sweep every bit that is no longer open so a quiesce always
        // leaves the graveyard holding only genuinely-blocked handles.
        let still_open = self.open_slot_mask();
        self.release_graveyard(&ctx.device, !still_open);
        Ok(())
    }

    /// Free/recycle the resources owed by a retired entry.
    ///
    /// # Safety
    /// The CB that referenced these resources must have retired.
    unsafe fn drain_cleanup(&mut self, device: &ash::Device, mut pending: PendingGpuCleanup) {
        super::super::retire_guest_write_pages(&pending.guest_write_tokens);
        for identity in pending.unpin_residents.drain(..) {
            self.pin_resident_target(&identity, false);
        }
        for identity in pending.unpin_compute_residents.drain(..) {
            self.pin_resident_storage(&identity, false);
        }
        self.desc_arena.free(device, &pending.dsets);
        // The fence this entry waited on is exactly what makes a rewrite of
        // these safe, so the free list is fed from here and nowhere else.
        self.recycle_scatter_dsets(&mut pending.scatter_dsets);
        // No cache admissions here: `seal_entry` lifted them out and
        // `finish_entry_async` gave them to the cache at submit. What is left in
        // `pending.sampled` is every slot nothing retained, which recycles.
        for slot in pending.staging.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.staging_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.gather.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.gather_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.readback.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.sampled.drain(..) {
            self.admit_sampled(slot);
        }
        for slot in pending.attachment_snapshots.drain(..) {
            self.attachment_snapshot_free.admit(slot.key(), slot);
        }
        for slot in pending.storage_images.drain(..) {
            self.admit_storage_image(slot);
        }
    }

    fn bucket(size: u64) -> u64 {
        // Power-of-two bucket, min 64.
        let mut b = 64u64;
        while b < size {
            b = b.saturating_mul(2);
            if b == 0 {
                return u64::MAX;
            }
        }
        b
    }

    fn note_staging_hit(&mut self) {
        self.staging_hits = self.staging_hits.saturating_add(1);
    }

    /// A staging acquire that found no free slot in its bucket and must pay a
    /// full `vkAllocateMemory`.
    ///
    /// `vk_alloc_sites` puts 99.4 % of all allocation wall-clock in this pool —
    /// 9 725 allocations at ~817 µs each over one 260 s boot — while a `vram`
    /// census line, since deleted, showed the free pool holding up to 133 MiB at
    /// the same time. Only the first half is reproducible today. A pool
    /// that is simultaneously full and missing is either holding the wrong
    /// buckets or being emptied behind the hot path, and the aggregate cannot
    /// tell those apart. The miss's own bucket plus the free pool's bucket
    /// histogram at the moment of the miss can.
    fn note_staging_miss(&mut self, bucket: u64, us: u64) {
        self.staging_misses = self.staging_misses.saturating_add(1);
        let log2 = bucket.trailing_zeros().min(STAGING_BUCKET_BINS as u32 - 1) as usize;
        self.staging_miss_bins[log2] = self.staging_miss_bins[log2].saturating_add(1);
        self.staging_miss_us_bins[log2] = self.staging_miss_us_bins[log2].saturating_add(us);
        if !self.staging_misses.is_multiple_of(STAGING_MISS_EMIT_EVERY) {
            return;
        }
        use std::fmt::Write as _;
        let mut free_slots = 0usize;
        let mut free_bytes = 0u64;
        let mut free_bins = [0usize; STAGING_BUCKET_BINS];
        for (&b, list) in &self.staging_free {
            free_slots += list.len();
            free_bytes += b.saturating_mul(list.len() as u64);
            let i = (b.trailing_zeros() as usize).min(STAGING_BUCKET_BINS - 1);
            free_bins[i] += list.len();
        }
        let bins = |v: &[usize; STAGING_BUCKET_BINS]| {
            let mut s = String::new();
            for (i, n) in v.iter().enumerate() {
                if *n != 0 {
                    let _ = write!(
                        s,
                        "{}{}:{n}",
                        if s.is_empty() { "" } else { "," },
                        1u64 << i
                    );
                }
            }
            if s.is_empty() {
                s.push('-');
            }
            s
        };
        // Mean microseconds per miss in each bucket. Size does not predict
        // allocation cost across the seven sites (a 1.6 MiB DEVICE_LOCAL image is
        // 275 us, a 3.6 MiB HOST_VISIBLE readback is 1313 us), so whether a
        // 64-byte staging miss costs the same as a 4 MiB one decides whether the
        // fix is fewer misses or fewer VkDeviceMemory objects.
        let mut us_bins = String::new();
        for (i, n) in self.staging_miss_bins.iter().enumerate() {
            if *n != 0 {
                let _ = write!(
                    us_bins,
                    "{}{}:{}",
                    if us_bins.is_empty() { "" } else { "," },
                    1u64 << i,
                    self.staging_miss_us_bins[i] / *n as u64
                );
            }
        }
        reims_vgpu_observe::off(format!(
            "staging_pool hits={} misses={} live={} free_slots={free_slots} free_mb={} miss_bins={} miss_us_bins={us_bins} free_bins={}",
            self.staging_hits,
            self.staging_misses,
            self.staging_live.len(),
            free_bytes >> 20,
            bins(&self.staging_miss_bins),
            bins(&free_bins),
        ));
    }

    pub(crate) unsafe fn acquire_staging(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let need = size.max(4);
        let bucket = Self::bucket(need);
        // Prefer exact-usage free slots in this bucket; usage is OR'd broadly so reuse is fine.
        if let Some(list) = self.staging_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.note_staging_hit();
                self.staging_live.push(slot);
                return Ok(slot);
            }
        }
        let miss_started = Instant::now();
        let _slow = SlowStagingWrite::watch("acquire", need, 0);
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(
                        usage
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStaging, e)))?;
        counters.note_create(CreateSite::StagingBuffer);
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, req.size, MemoryClass::Upload)
            .ok_or({
                DrawError::Unsupported(reason::DrawReason::NoHostVisibleMemoryForStaging {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        // Carve out of a shared HOST_VISIBLE block rather than allocating one
        // memory object per buffer. The block is allocated and mapped once; a
        // miss here is a create + carve + bind, which is what turns the ~0.4 ms
        // floor every miss used to pay into a handful of block allocations for
        // the whole boot. The slab picks the same `MemoryClass::Upload` type
        // `mt` resolves to and records it on the block, so a carve only ever
        // lands in a block this bind can legally use; `mt` stays here because
        // the slot still has to report that type's caching.
        let token = self
            .slabs
            .upload()
            .acquire(ctx, &req, counters)
            .inspect_err(|_| ctx.device.destroy_buffer(buffer, None))?;
        ctx.device
            .bind_buffer_memory(buffer, token.memory, token.offset())
            .map_err(|e| {
                self.slabs.release(&ctx.device, token);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindStaging, e))
            })?;
        let slot = BufferSlot {
            buffer,
            memory: token.memory,
            size: bucket,
            // The block's mapping covers every carve in it, so a slot's host
            // address is a pointer into it. Nothing maps or unmaps per slot;
            // the pool's whole point is that the allocation outlives the bind,
            // and now so does the mapping of the block behind it.
            mapped: token.mapped,
            backing: BufferBacking::Slab(token),
            // `MemoryClass::Upload` requires HOST_COHERENT, so a staging write
            // needs no flush and the persistent mapping above is sound.
            coherent: true,
            // Read rather than asserted: `MemoryClass::Upload` says nothing
            // about caching, and nothing on the staging path reads this field —
            // it is the readback slots that decide anything on it.
            cached: ctx.mapped_memory_kind(mt).cached,
        };
        self.staging_live.push(slot);
        self.note_staging_miss(bucket, miss_started.elapsed().as_micros() as u64);
        Ok(slot)
    }

    /// Take a recycled gather slot of exactly `bucket` bytes out of the free
    /// list and record it live, or `None` when the list has none.
    ///
    /// Split out of [`Self::acquire_guest_gather`] because it is the whole of
    /// the pool's no-aliasing property and it is the half that needs no device,
    /// so a test can exercise it: a slot **leaves** the free list when it is
    /// handed out, and only `drain_cleanup` puts it back, which happens after
    /// the fence of the submission that named it. Two acquires with no fence
    /// between them therefore cannot resolve to one buffer.
    ///
    /// That property is why the guest-page writeback's detiling buffer comes
    /// from here rather than from a slot of its own. It used to be a singleton
    /// grown in place, whose safety rested on the writeback rail waiting its
    /// fence before returning — and that wait was removed when the rail learned
    /// to record the obligation instead. A grow then freed the buffer while a
    /// submitted copy was still reading it.
    fn take_free_gather(&mut self, bucket: u64) -> Option<BufferSlot> {
        let slot = self.gather_free.get_mut(&bucket)?.pop()?;
        self.gather_live.push(slot);
        Some(slot)
    }

    /// A DEVICE_LOCAL buffer of at least `size` bytes for the draw-time guest
    /// gather to assemble a scattered window into.
    ///
    /// Same shape as [`Self::acquire_staging`] — power-of-two buckets, a free
    /// list, a slab carve on a miss, and the ring returns it when the fence
    /// retires — because it has the same lifetime and the same size
    /// distribution. What differs is the memory: these are read by the draw and
    /// never by the CPU, so they carry no mapping and come from the
    /// device-local slab.
    ///
    /// # Why the usage superset matches staging's
    ///
    /// A gather destination stands in for exactly the slot the CPU gather would
    /// have taken, and a draw deduplicates its binds by content — the same
    /// window bound as a vertex stream and as a storage buffer resolves to one
    /// buffer. So a slot must be legal for either, or the free list would have
    /// to be keyed by usage and would miss on the crossover.
    pub(crate) unsafe fn acquire_guest_gather(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let bucket = Self::bucket(size.max(4));
        if let Some(slot) = self.take_free_gather(bucket) {
            return Ok(slot);
        }
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(
                        usage
                            | vk::BufferUsageFlags::TRANSFER_DST
                            // TRANSFER_SRC unconditionally, because these slots
                            // are recycled by size bucket and not by usage: the
                            // sampled rail gathers into one and then copies it
                            // into an image, so a slot first created for a
                            // vertex window would be an invalid copy source the
                            // second time it came out of `gather_free`.
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateGuestGather, e)))?;
        counters.note_create(CreateSite::GatherBuffer);
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let token = self
            .slabs
            .gather()
            .acquire(ctx, &req, counters)
            .inspect_err(|_| ctx.device.destroy_buffer(buffer, None))?;
        ctx.device
            .bind_buffer_memory(buffer, token.memory, token.offset())
            .map_err(|e| {
                self.slabs.release(&ctx.device, token);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindGuestGather, e))
            })?;
        let slot = BufferSlot {
            buffer,
            memory: token.memory,
            size: bucket,
            // Device-local blocks are not mapped, and nothing on this rail reads
            // these bytes with the CPU. A non-zero value here would be a host
            // address into memory that may not be host-visible at all.
            mapped: 0,
            backing: BufferBacking::Slab(token),
            // Neither field is consulted for a slot the CPU never touches: both
            // decide flush and invalidate behaviour on the mapped paths.
            coherent: false,
            cached: false,
        };
        self.gather_live.push(slot);
        Ok(slot)
    }

    pub(crate) unsafe fn write_staging(
        &self,
        ctx: &DeviceContext,
        slot: &BufferSlot,
        bytes: &[u8],
    ) -> Result<(), DrawError> {
        let _slow = SlowStagingWrite::watch("bytes", bytes.len() as u64, 0);
        let size = bytes.len().max(4) as u64;
        let ptr = staging_write_ptr(ctx, slot, size)?;
        unsafe {
            if bytes.is_empty() {
                // Nothing to copy — the mapped span is the 4-byte minimum; zero it
                // so the bind reads defined memory.
                std::ptr::write_bytes(ptr, 0, size as usize);
            } else {
                // The mapped span is exactly `bytes.len()` (`size == bytes.len()`
                // here), so the copy overwrites every mapped byte — a preceding
                // full-span zeroing would just be overwritten. Copy only.
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            }
        }
        Ok(())
    }

    /// Copy into a mapped staging slot with R and B exchanged.
    ///
    /// The exchange is an involution, so this serves both directions: a
    /// semantic-RGBA seed into a BGRA attachment, and a guest-scanout-order seed
    /// (what `surface_cache` holds) into the RGBA pooled target. Which one is
    /// wanted is decided by the caller from `target_seed_order` against
    /// `output_bgra`; the transformation is the same either way.
    ///
    /// It used to run over a heap copy of the seed, which cost a full-frame
    /// allocation, a memcpy to fill it, a second pass to swizzle it and a third
    /// to write it into the mapped span. Being per-pixel and order-independent
    /// it folds into the copy: read from the caller's borrow, write the exchanged
    /// bytes straight into mapped memory, one pass, no allocation.
    ///
    /// Mapped staging is host-visible and written only, never read back, so
    /// this writes each destination byte exactly once and never loads from
    /// `ptr` — a read-modify-write in place would fault the write-combined
    /// case into a far slower path.
    ///
    /// One store per pixel, not four. The exchange used to be four separate
    /// one-byte stores into the mapped span; reading and writing whole
    /// `[u8; 4]` pixels keeps the "never load from `ptr`" property — a store of
    /// a whole element is not a read-modify-write — while giving the compiler
    /// an aligned 32-bit move it can widen.
    ///
    /// **That was done on a hypothesis and the hypothesis was wrong, so do not
    /// read a speedup into it.** `draw_phase`'s `stage_us` divided by
    /// `engine_delta`'s `seed_upload_bytes` reads ~1.1 GB/s against 5.2-8.8 GB/s
    /// for `write_staging_from_runs`, which does the same job into the same
    /// memory class with a plain `copy_nonoverlapping`, and this loop looked
    /// like the difference. It is not: `stage_us` also carries vertex, index and
    /// storage staging, so that ratio is a rate for a mixture.
    ///
    /// `swap_rb_us` / `swap_rb_kb` time this write and nothing else, and they
    /// acquit it. Live x86/PCI, six windows: **6.1, 6.4, 7.1, 8.0, 8.6 and
    /// 10.5 GB/s**, against `stage_us` shares of **0 % in every driven window
    /// and 5-15 % in the sparse ones** — including a window whose worst draw was
    /// 35.7 ms and whose swizzle was 5 % of its staging. Whatever holds
    /// `stage_us`, it is not this.
    ///
    /// What the rewrite is worth is the test below: the exchange is the part
    /// that can be silently wrong (every seeded LOAD composites with red and
    /// blue swapped), and before this it lived in pointer arithmetic no test
    /// could reach.
    ///
    /// A trailing partial pixel (`len % 4 != 0`) is copied through unswizzled;
    /// a seed is always whole RGBA8 pixels, so the remainder is empty in
    /// practice and this only keeps the mapped span fully defined.
    pub(crate) unsafe fn write_staging_swap_rb(
        &self,
        ctx: &DeviceContext,
        slot: &BufferSlot,
        rgba: &[u8],
    ) -> Result<(), DrawError> {
        let size = rgba.len().max(4) as u64;
        let ptr = staging_write_ptr(ctx, slot, size)?;
        unsafe {
            if rgba.is_empty() {
                std::ptr::write_bytes(ptr, 0, size as usize);
                return Ok(());
            }
            // The mapped span is at least `rgba.len()` and is exclusively ours
            // for the duration of this call, so a slice over it is sound. It
            // exists so the transformation can be a plain function with a test
            // rather than pointer arithmetic no test can reach.
            //
            // Timed on its own, because `draw_phase`'s `stage_us` also carries
            // vertex, index and storage staging: dividing that by
            // `seed_upload_bytes` gives a rate contaminated by whatever else the
            // draw staged, which is enough to see the seed path is slow and not
            // enough to say what limits it. `swap_rb_us` against `swap_rb_kb` is
            // this write and nothing else, so it can be read against the memcpy
            // rate `write_staging_from_runs` gets into the same memory class and
            // convict either the loop or the memory.
            let started = std::time::Instant::now();
            exchange_rb_into(rgba, std::slice::from_raw_parts_mut(ptr, rgba.len()));
            crate::telemetry::note_route_us("swap_rb_us", started.elapsed().as_micros() as u64);
            crate::telemetry::note_route_n("swap_rb_kb", (rgba.len() / 1024) as u64);
        }
        Ok(())
    }

    /// Snapshot guest-run spans directly into a mapped staging slot.
    ///
    /// The deferred-submit snapshot path used to `cpu_bytes()` the runs into a
    /// heap `Vec` and then `write_staging` that `Vec` into the mapped buffer —
    /// two full copies plus an allocation per bind, ~4.8 binds/draw under
    /// compositing. This copies each run's guest RAM straight into the mapped
    /// staging span at its running offset (one copy, no intermediate `Vec`), and
    /// zeroes only the tail if the runs underfill `total_len` (short read). The
    /// freshness contract is identical to `cpu_bytes` — the read races guest CPU
    /// writes exactly as the encode-time staging read does.
    pub(crate) unsafe fn write_staging_from_runs(
        &self,
        ctx: &DeviceContext,
        slot: &BufferSlot,
        runs: &[types::GuestRun],
        source_offset: u64,
        total_len: u64,
    ) -> Result<(), DrawError> {
        let _slow = SlowStagingWrite::watch("guest_runs", total_len, runs.len());
        let size = total_len.max(4);
        let ptr = staging_write_ptr(ctx, slot, size)?;
        let total = total_len as usize;
        let mut off = 0usize;
        let mut skip = source_offset;
        unsafe {
            for run in runs {
                if off >= total {
                    break;
                }
                if skip >= run.len {
                    skip -= run.len;
                    continue;
                }
                let within = skip as usize;
                skip = 0;
                let available = (run.len as usize).saturating_sub(within);
                let n = available.min(total - off);
                // SAFETY: `host_ptr` is a stable RAMBlock alias from
                // `HostOps::map_pages`, valid for the VM lifetime; `ptr` is the
                // mapped staging span of `size >= total` bytes and `off + n <=
                // total`, so the destination stays in bounds.
                std::ptr::copy_nonoverlapping(
                    (run.host_ptr as *const u8).add(within),
                    ptr.add(off),
                    n,
                );
                off += n;
            }
            // Runs underfilled the span (short read) or the 4-byte minimum tail:
            // zero the remainder so the bind reads defined memory.
            if off < size as usize {
                std::ptr::write_bytes(ptr.add(off), 0, size as usize - off);
            }
        }
        Ok(())
    }

    pub(crate) fn recycle_staging(&mut self) {
        self.forget_cb_bound_buffers("bindmap_clear_recycle", "bindmap_clear_recycle_entries");
        for slot in self.staging_live.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.staging_free.entry(bucket).or_default().push(slot);
        }
    }

    /// Create one host-visible buffer of exactly `bucket` bytes, usable as a
    /// transfer destination and as a storage buffer.
    ///
    /// `TRANSFER_DST` is what every acquirer of a readback slot uses: the
    /// graphics and compute paths both fill one with `vkCmdCopyImageToBuffer`
    /// and then read the persistent mapping. No caller binds a readback slot as
    /// a shader-visible buffer today, so `STORAGE_BUFFER` is reach rather than a
    /// requirement, and it is unconditional because the pool is bucketed by size
    /// and shared by every rail — two usages would have to be bucketed
    /// separately, and a slot acquired for a copy could not then be handed to a
    /// dispatch. The flag costs a `memory_type_bits` that may be narrower; the
    /// type is chosen from the bits this buffer actually reports, so a device
    /// that excluded a type here picks another rather than mis-binds.
    ///
    /// The two readback acquires differ only in where they stash the slot and in
    /// which `VkOp`/`AllocSite` names each step, so the Vulkan sequence lives
    /// here once. The names stay per-caller: `vk_pools_alloc_readback` and
    /// `vk_pools_alloc_readback_extra` are different exhaustion sites, and a
    /// shared slug could not say which pool ran out.
    #[allow(
        clippy::too_many_arguments,
        reason = "one VkOp per fallible Vulkan call, so an exhaustion names its own site"
    )]
    unsafe fn create_readback_buffer(
        ctx: &DeviceContext,
        bucket: u64,
        counters: &EngineCounters,
        site: AllocSite,
        create_op: VkOp,
        alloc_op: VkOp,
        bind_op: VkOp,
        map_op: VkOp,
    ) -> Result<BufferSlot, DrawError> {
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(create_op, e)))?;
        counters.note_create(CreateSite::ReadbackBuffer);
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, req.size, MemoryClass::Readback)
            .ok_or({
                DrawError::Unsupported(reason::DrawReason::NoHostVisibleMemoryForReadback {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            site,
        )
        .map_err(|e| {
            ctx.device.destroy_buffer(buffer, None);
            DrawError::VkCall(VkCall::new(alloc_op, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(bind_op, e))
            })?;
        let kind = ctx.mapped_memory_kind(mt);
        note_readback_memory(ctx, mt, req.memory_type_bits, kind);
        // Map once, for the slot's lifetime, exactly as the staging pool does
        // and for one extra reason beyond saving the map/unmap round-trip pair.
        //
        // A readback slot can be *leased*: handed to a reader that consumes the
        // mapping after the engine lock is dropped (`lease_readback`). Returning
        // such a lease must not need the device, because the thread that ends it
        // may be racing a teardown that holds the engine lock and is waiting for
        // exactly this lease to come back. A slot that owns its mapping for life
        // makes the return pure bookkeeping — no `vkUnmapMemory`, so no lock, so
        // no cycle.
        //
        // Non-coherent readback memory is still invalidated per read
        // (`read_back_slot`); a persistent mapping does not change what a read
        // owes, only how many times the mapping is established.
        let mapped = ctx
            .device
            .map_memory(memory, 0, bucket, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                ctx.device.destroy_buffer(buffer, None);
                ctx.device.free_memory(memory, None);
                DrawError::VkCall(VkCall::new(map_op, e))
            })? as usize;
        Ok(BufferSlot {
            buffer,
            memory,
            size: bucket,
            mapped,
            // Readback slots keep their own memory object. They are few (tens
            // a boot against the staging pool's ~1 500), individually large,
            // and of a different memory class — so the block-suballocation
            // argument that carries the staging pool does not carry here, while
            // the lease path's "this memory is mine alone" is easier to hold
            // when it is true.
            backing: BufferBacking::Dedicated,
            coherent: kind.coherent,
            cached: kind.cached,
        })
    }

    pub(crate) unsafe fn acquire_readback(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        // Every acquire, so a returned lease is back in circulation by the next
        // readback rather than after an arbitrary delay. This is the only
        // engine-locked point the return path can rely on running.
        self.reclaim_returned_readback_leases();
        let bucket = Self::bucket(size.max(4));
        if let Some(list) = self.readback_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.readback_live = Some(slot);
                return Ok(slot);
            }
        }
        let slot = Self::create_readback_buffer(
            ctx,
            bucket,
            counters,
            AllocSite::Readback,
            VkOp::PoolsCreateReadback,
            VkOp::PoolsAllocReadback,
            VkOp::PoolsBindReadback,
            VkOp::PoolsMapReadback,
        )?;
        self.readback_live = Some(slot);
        Ok(slot)
    }

    /// Check the live readback slot out of the pool, for a reader that will go
    /// on consuming its mapping after the engine lock is dropped.
    ///
    /// Returns the lease token and the slot's host address, or `None` when
    /// there is no live slot to lease. The caller owes exactly one
    /// [`return_readback_lease`] for a token it is handed; until that
    /// arrives the slot is in no free list, no live list and no ring entry's
    /// pending cleanup, so nothing can hand it to a GPU copy underneath the
    /// borrow.
    ///
    /// Must be called before [`Self::seal_entry`], which is what would
    /// otherwise move the slot into the submitted entry's cleanup.
    pub(crate) fn lease_readback(&mut self) -> Option<ReadbackLease> {
        let slot = self.readback_live.take()?;
        // Two refusals, and both send the caller to the copying path rather
        // than to a failure.
        //
        // A slot maps for life, so `mapped == 0` should not occur; leasing one
        // anyway would mean establishing a mapping here, which is a device call
        // the return path could not undo without the engine lock.
        //
        // Uncached memory is the one that matters in practice. The lease exists
        // to let a consumer read the mapping in place, and an uncached mapping
        // reads at roughly a tenth of memcpy speed
        // (`readback_memory_not_cached`). Paying that rate once on a linear
        // memcpy and then consuming a cached `Vec` beats paying it on every row
        // of a scattered walk, so where the cached type was unavailable the
        // copy is genuinely the faster shape and the lease declines.
        if slot.mapped == 0 || !slot.cached {
            self.readback_live = Some(slot);
            return None;
        }
        let token = NEXT_READBACK_LEASE_TOKEN.fetch_add(1, Ordering::Relaxed);
        // Before the slot leaves the pool: the counter is what a teardown reads
        // to decide whether a borrow is live, and it must never see the slot
        // gone while the count still says nobody has it.
        self.readback_lease_returns
            .outstanding
            .fetch_add(1, Ordering::AcqRel);
        let lease = ReadbackLease {
            token,
            ptr: slot.mapped,
            slot_size: slot.size,
            returns: Arc::clone(&self.readback_lease_returns),
        };
        self.readback_leased.push(LeasedReadback { token, slot });
        Some(lease)
    }

    /// Take back every lease whose holder has finished and return its slot to
    /// the free list.
    ///
    /// Cheap and unconditional: the returned-token channel is empty on all but
    /// the calls that follow a lease, and a lease that is still out is simply
    /// not in the drained set.
    pub(crate) fn reclaim_returned_readback_leases(&mut self) {
        let returned = std::mem::take(&mut *self.readback_lease_returns.returned.lock());
        for token in returned {
            let Some(index) = self.readback_leased.iter().position(|l| l.token == token) else {
                // A teardown collected the leases while this token was in
                // flight, so there is no slot left to give back. The handles
                // died with the device; dropping the token is the whole of it.
                continue;
            };
            let slot = self.readback_leased.remove(index).slot;
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
    }

    pub(crate) fn recycle_readback(&mut self) {
        if let Some(slot) = self.readback_live.take() {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
        for slot in self.readback_multi_live.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
    }

    /// Acquire an additional readback buffer without replacing the primary live slot.
    pub(crate) unsafe fn acquire_readback_extra(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let bucket = Self::bucket(size.max(4));
        if let Some(list) = self.readback_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.readback_multi_live.push(slot);
                return Ok(slot);
            }
        }
        let slot = Self::create_readback_buffer(
            ctx,
            bucket,
            counters,
            AllocSite::ReadbackMulti,
            VkOp::PoolsCreateReadbackExtra,
            VkOp::PoolsAllocReadbackExtra,
            VkOp::PoolsBindReadbackExtra,
            VkOp::PoolsMapReadbackExtra,
        )?;
        self.readback_multi_live.push(slot);
        Ok(slot)
    }

    pub(crate) unsafe fn acquire_target(
        &mut self,
        ctx: &DeviceContext,
        key: TargetKey,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
    ) -> Result<&TargetSlot, DrawError> {
        // Band the pool's occupancy on **every** call, hit or miss, and before
        // anything can return. Sited here rather than beside the cap because a
        // band taken after the hit early-return counts only misses, and a zero
        // from it then means either "never called" or "always hit" — two states
        // a reader cannot separate, which is the sampling-point trap this exists
        // to escape. Taken here, a zero means the function did not run.
        crate::telemetry::note_route(target_pool_depth_band(self.target_order.len()));
        let map_key = (key, render_pass.as_raw());
        if self.targets.contains_key(&map_key) {
            return Ok(self.targets.get(&map_key).unwrap());
        }
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | if key.with_transfer_dst {
                vk::ImageUsageFlags::TRANSFER_DST
            } else {
                vk::ImageUsageFlags::empty()
            };
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(translate::pixel::RESIDENT_RGBA_FORMAT)
                    .extent(vk::Extent3D {
                        width: key.width,
                        height: key.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateTargetImage, e)))?;
        counters.note_create(CreateSite::TargetImage);
        let ireq = ctx.device.get_image_memory_requirements(image);
        let memory = match self.bind_image_slab(ctx, image, &ireq, VkOp::PoolsBindTarget, counters)
        {
            Ok(m) => m,
            Err(error) => {
                ctx.device.destroy_image(image, None);
                return Err(error);
            }
        };
        let view = match ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(translate::pixel::RESIDENT_RGBA_FORMAT)
                .subresource_range(color_subresource_range()),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                ctx.device.destroy_image(image, None);
                self.free_image_slab(&ctx.device, image);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateTargetView,
                    e,
                )));
            }
        };
        counters.note_create(CreateSite::TargetImageView);
        let attachments = [view];
        let framebuffer = match ctx.device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(key.width)
                .height(key.height)
                .layers(1),
            None,
        ) {
            Ok(fb) => fb,
            Err(e) => {
                ctx.device.destroy_image_view(view, None);
                ctx.device.destroy_image(image, None);
                self.free_image_slab(&ctx.device, image);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create(CreateSite::TargetFramebuffer);
        if self.target_order.len() >= TARGET_POOL_MAX_ENTRIES {
            if let Some(old_k) = self.target_order.first().cloned() {
                if let Some(old) = self.targets.remove(&old_k) {
                    // Counted, not logged. The eviction costs a re-create and
                    // nothing else — the key is geometry plus render pass, so
                    // this slot is scratch and holds no guest resource's content
                    // — but an uncounted eviction is one nobody can rank against
                    // the re-creates it causes.
                    crate::telemetry::note_route("target_pool_evict");
                    self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                    self.dispose(
                        &ctx.device,
                        DeferredHandle::Image {
                            image: old.image,
                            view: old.view,
                            memory: old.memory,
                        },
                    );
                }
                self.target_order.remove(0);
            }
        }
        self.targets.insert(
            map_key,
            TargetSlot {
                image,
                memory,
                view,
                framebuffer,
            },
        );
        self.target_order.push(map_key);
        Ok(self.targets.get(&map_key).unwrap())
    }

    /// Acquire the discard-only source of a multisample resolve pass.
    ///
    /// One slot is sufficient: Metal's resolve-only store action makes the
    /// source unobservable after the encoder ends. Replacing its shape does not
    /// evict guest data; the displaced handles remain alive behind every ring
    /// slot that could still reference them.
    pub(crate) unsafe fn acquire_multisample_target(
        &mut self,
        ctx: &DeviceContext,
        key: MultisampleTargetKey,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView, vk::Framebuffer), DrawError> {
        if let Some(slot) = self.multisample_target.as_ref() {
            if slot.key == key && !key.transient_depth {
                return Ok((slot.image, slot.view, slot.framebuffer));
            }
        }
        if let Some(old) = self.multisample_target.take() {
            self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
            self.dispose(
                &ctx.device,
                DeferredHandle::Image {
                    image: old.image,
                    view: old.view,
                    memory: old.memory,
                },
            );
        }
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(key.format)
                    .extent(vk::Extent3D {
                        width: key.width,
                        height: key.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(super::super::caches::vk_sample_count(key.samples))
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateTargetImage, e)))?;
        counters.note_create(CreateSite::TargetImage);
        let requirements = ctx.device.get_image_memory_requirements(image);
        let memory = match self.bind_image_slab(
            ctx,
            image,
            &requirements,
            VkOp::PoolsBindTarget,
            counters,
        ) {
            Ok(memory) => memory,
            Err(error) => {
                ctx.device.destroy_image(image, None);
                return Err(error);
            }
        };
        let view = ctx
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(key.format)
                    .subresource_range(super::super::color_subresource_range()),
                None,
            )
            .map_err(|e| {
                ctx.device.destroy_image(image, None);
                self.free_image_slab(&ctx.device, image);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateTargetView, e))
            })?;
        counters.note_create(CreateSite::TargetImageView);
        let mut attachments = vec![view, key.resolve_view];
        attachments.extend(key.depth_view);
        let framebuffer = ctx
            .device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(key.width)
                    .height(key.height)
                    .layers(1),
                None,
            )
            .map_err(|e| {
                ctx.device.destroy_image_view(view, None);
                ctx.device.destroy_image(image, None);
                self.free_image_slab(&ctx.device, image);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateFramebuffer, e))
            })?;
        counters.note_create(CreateSite::TargetFramebuffer);
        self.multisample_target = Some(MultisampleTargetSlot {
            image,
            memory,
            view,
            framebuffer,
            key,
        });
        Ok((image, view, framebuffer))
    }

    pub(crate) unsafe fn acquire_sampled(
        &mut self,
        ctx: &DeviceContext,
        sk: SampledKey,
        counters: &EngineCounters,
    ) -> Result<SampledSlot, DrawError> {
        unsafe { self.acquire_sampled_for(ctx, sk, counters, SampledTransientUse::Upload) }
    }

    pub(crate) unsafe fn acquire_attachment_snapshot(
        &mut self,
        ctx: &DeviceContext,
        sk: SampledKey,
        counters: &EngineCounters,
    ) -> Result<SampledSlot, DrawError> {
        unsafe {
            self.acquire_sampled_for(ctx, sk, counters, SampledTransientUse::AttachmentSnapshot)
        }
    }

    unsafe fn acquire_sampled_for(
        &mut self,
        ctx: &DeviceContext,
        sk: SampledKey,
        counters: &EngineCounters,
        use_: SampledTransientUse,
    ) -> Result<SampledSlot, DrawError> {
        let SampledKey {
            width,
            height,
            mip_levels,
            layers,
            volume,
            cube,
            arrayed,
            one_dim,
            format,
            swizzle,
        } = sk;
        // A hit reuses a recycled slot — no vkAllocateMemory this acquire. A miss
        // is counted here, at the empty free list, rather than after the create
        // succeeds: the census question is whether the pool had one, not whether
        // the fallback worked.
        let recycled = match use_ {
            SampledTransientUse::Upload => self.sampled_free.take(&sk),
            SampledTransientUse::AttachmentSnapshot => self.attachment_snapshot_free.take(&sk),
        };
        if let Some(slot) = recycled {
            let handles = slot.handles();
            match use_ {
                SampledTransientUse::Upload => self.sampled_live.push(slot),
                SampledTransientUse::AttachmentSnapshot => self.attachment_snapshot_live.push(slot),
            }
            return Ok(handles);
        }
        let image_type = if one_dim {
            vk::ImageType::TYPE_1D
        } else if volume {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };
        let view_type = if one_dim && arrayed {
            vk::ImageViewType::TYPE_1D_ARRAY
        } else if one_dim {
            vk::ImageViewType::TYPE_1D
        } else if volume {
            vk::ImageViewType::TYPE_3D
        } else if cube {
            vk::ImageViewType::CUBE
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let extent_depth = if volume { layers } else { 1 };
        let array_layers = if volume { 1 } else { layers };
        let flags = if cube {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let vk_format = format;
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(image_type)
                    .format(vk_format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: extent_depth,
                    })
                    .mip_levels(mip_levels)
                    .array_layers(array_layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateSampledImage, e)))?;
        counters.note_create(CreateSite::SampledImage);
        let req = ctx.device.get_image_memory_requirements(image);
        let memory = match self.bind_image_slab(ctx, image, &req, VkOp::PoolsBindSampled, counters)
        {
            Ok(m) => m,
            Err(error) => {
                ctx.device.destroy_image(image, None);
                return Err(error);
            }
        };
        let view = match ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(view_type)
                .format(vk_format)
                // The decoded type-8 view swizzle, performed by the hardware at
                // sample time. Identity for every ordinary bind. The format
                // contributes no mapping of its own: `translate::pixel`'s
                // sampled rail admits only formats whose Metal channels sit
                // identically on their Vulkan ones, and declines the rest by
                // name rather than binding a plan it cannot carry.
                .components(translate::pixel::vk_component_mapping(&swizzle))
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: array_layers,
                }),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                ctx.device.destroy_image(image, None);
                self.free_image_slab(&ctx.device, image);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateSampledView,
                    e,
                )));
            }
        };
        counters.note_create(CreateSite::SampledImageView);
        let slot = SampledSlot {
            image,
            memory,
            view,
            width,
            height,
            mip_levels,
            layers,
            volume,
            cube,
            arrayed,
            one_dim,
            format,
            swizzle,
        };
        let handles = slot.handles();
        match use_ {
            SampledTransientUse::Upload => self.sampled_live.push(slot),
            SampledTransientUse::AttachmentSnapshot => self.attachment_snapshot_live.push(slot),
        }
        Ok(handles)
    }

    pub(crate) fn find_cached_sampled(
        &mut self,
        key: SampledKey,
        content: &[u8],
        identity: Option<crate::engine::SampledContentIdentity>,
        resource_lifetime: Option<&reims_vgpu_core::ResourceLifetimeRef>,
        counters: &EngineCounters,
    ) -> Option<SampledSlot> {
        // Before any cache is consulted, so the set is what the workload *asked
        // for* rather than what this cache happened to keep. That is the whole
        // distinction from the victim ledger, whose reach is censored at
        // `SAMPLED_VICTIM_LEDGER` and reads `beyond_ledger` 6 704 times on a
        // driven macos-26 boot.
        super::sampled_working_set::note_wanted(key, identity, content.len());
        let content_hash = sampled_content_hash(content);
        // The digest narrows the walk to one candidate; the retained bytes are
        // what decide the hit. `ResidentSampledSlot::content` carries why the
        // digest is not allowed to answer on its own.
        let found = self.sampled_cache.iter().position(|entry| {
            entry.has_live_owner()
                && entry.slot.key() == key
                && entry.fingerprint == SampledFingerprint::Content(content_hash)
                && entry.content.as_deref().is_some_and(|b| b == content)
        });
        let Some(index) = found else {
            counters
                .sampled_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let Some(owner) = resource_lifetime.filter(|owner| owner.is_live()) else {
            counters
                .sampled_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let entry = &mut self.sampled_cache[index];
        entry.retain_owner(owner);
        entry.last_touch_ms = self.idle_clock_ms;
        let handles = entry.slot.handles();
        counters.sampled_cache_hits.fetch_add(1, Ordering::Relaxed);
        counters
            .sampled_cache_hit_bytes
            .fetch_add(content.len() as u64, Ordering::Relaxed);
        Some(handles)
    }

    /// Admit one detached sampled slot into the exact-content cache. A slot
    /// whose content duplicates an existing entry, or has no serialized owner,
    /// returns to the live list for fence-safe recycling.
    ///
    /// # Every exit is counted, because a cache that never fills looks the same
    ///
    /// `sampled_admit_kept`, `_duplicate`, `_no_identity` and `_no_lifetime`
    /// partition the calls.
    fn admit_sampled_slot(
        &mut self,
        slot: SampledSlot,
        content: &SampledRetainContent,
        resource_lifetime: Option<&reims_vgpu_core::ResourceLifetimeRef>,
    ) {
        self.admit_sampled_entry(slot, content, resource_lifetime);
    }

    /// Device-free half of [`Self::admit_sampled_slot`], split out so admission
    /// is reachable from a test with no GPU.
    ///
    /// The three arms that decline hand the slot back to `sampled_live`, which
    /// this entry's own [`Self::seal_entry`] has just emptied — so it is swept
    /// into the *next* entry's cleanup and recycled when that entry's fence
    /// signals. On one queue that fence is behind this entry's, so a CB still
    /// reading the image cannot be racing the recycle.
    fn admit_sampled_entry(
        &mut self,
        slot: SampledSlot,
        content: &SampledRetainContent,
        resource_lifetime: Option<&reims_vgpu_core::ResourceLifetimeRef>,
    ) {
        let Some(owner) = resource_lifetime.filter(|owner| owner.is_live()) else {
            crate::telemetry::note_route("sampled_admit_no_lifetime");
            self.sampled_live.push(slot);
            return;
        };
        let (fingerprint, retained, content_len) = match content {
            // The `Arc` is cloned rather than the bytes copied: the retire path
            // already holds one, so recognising this entry by content later
            // costs a refcount here and no new allocation.
            SampledRetainContent::Bytes(bytes) => (
                SampledFingerprint::Content(sampled_content_hash(bytes)),
                Some(bytes.clone()),
                bytes.len(),
            ),
        };
        // Deduplication asks the same question a lookup does, so it has to
        // answer it the same way: a fingerprint match proposes a duplicate and
        // something else confirms it. Collapsing two entries here is exactly as
        // wrong as returning the wrong one from `find_cached_sampled` — the
        // survivor then answers for content it was not built from.
        let duplicate = self.sampled_cache.iter().position(|entry| {
            entry.has_live_owner()
                && entry.slot.key() == slot.key()
                && entry.fingerprint == fingerprint
                && entry.content.as_deref() == retained.as_deref()
        });
        if let Some(index) = duplicate {
            crate::telemetry::note_route("sampled_admit_duplicate");
            let existing = &mut self.sampled_cache[index];
            existing.retain_owner(owner);
            self.sampled_live.push(slot);
            return;
        }
        crate::telemetry::note_route("sampled_admit_kept");
        self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_add(content_len);
        let touch = self.idle_clock_ms;
        self.push_sampled_entry(ResidentSampledSlot {
            slot,
            fingerprint,
            content: retained,
            content_len,
            owners: vec![owner.clone()],
            last_touch_ms: touch,
        });
    }

    pub(crate) fn recycle_sampled(&mut self) {
        // This method owns sampled-slot recycling even when a future caller
        // does not also recycle staging. No memo may survive the slots it
        // names entering the free pool.
        self.forget_cb_sampled_guest();
        for slot in self.sampled_live.drain(..) {
            let sk = slot.key();
            self.sampled_free.push_uncapped(sk, slot);
        }
        for slot in self.attachment_snapshot_live.drain(..) {
            let sk = slot.key();
            self.attachment_snapshot_free.push_uncapped(sk, slot);
        }
    }
}

/// Lift the images named by `retains` out of `slots`, pairing each with what
/// names it.
///
/// A retained image is not a transient: it is about to become a cache entry, and
/// leaving it in the list that recycles is how one image ends up both a cache
/// entry and a free-list slot, handed to a later draw that overwrites content
/// another draw is sampling.
///
/// A retain naming an image the list does not hold is not an admission — the
/// slot was lifted by an earlier caller, which is the ordinary case for a batch
/// joiner whose retains were admitted at `batch_append`.
fn take_retained_slots(
    slots: &mut Vec<SampledSlot>,
    retains: Vec<SampledRetain>,
) -> Vec<(SampledSlot, SampledRetain)> {
    let mut taken = Vec::with_capacity(retains.len());
    for retain in retains {
        if let Some(index) = slots.iter().position(|slot| slot.image == retain.image) {
            taken.push((slots.remove(index), retain));
        }
    }
    taken
}

/// Why a readback allocation is slower than the class asked for.
///
/// `MemoryClass::Readback` ranks `HOST_CACHED` first because the CPU read that
/// follows is the whole reason the buffer exists. When no type in the buffer's
/// `memoryTypeBits` carries it, the fallback still *works* — every readback is
/// correct — and every one of them reads uncached at roughly a tenth of memcpy
/// speed. Measured on an Intel ARL iGPU before the class stopped requiring
/// `HOST_COHERENT`: 460 MB/s, 7-11 ms per 3.2 MB frame, 70-86 % of all draw time
/// and a device pinned at duty 1.000.
///
/// That is exactly the shape the ground rules forbid going unreported: no guest
/// work is lost, so nothing else in the device has any reason to say a word.
pub(crate) struct ReadbackMemoryDegrade {
    memory_type: u32,
    type_bits: u32,
}

impl reims_vgpu_observe::Decline for ReadbackMemoryDegrade {
    fn slug(&self) -> &'static str {
        "readback_memory_not_cached"
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("memory_type", self.memory_type.to_string()),
            ("type_bits", format!("{:#x}", self.type_bits)),
        ]
    }
}

/// Report which memory type a readback buffer landed in, once per type per boot.
///
/// Both outcomes are reported. A zero on the degraded arm has to be readable as
/// "the cached type was available and taken", which it cannot be if the healthy
/// case says nothing — so the healthy case emits an always-on notice naming the
/// type, and only the uncached case is a typed decline.
fn note_readback_memory(
    ctx: &DeviceContext,
    memory_type: u32,
    type_bits: u32,
    kind: MappedMemoryKind,
) {
    if !reims_vgpu_observe::first_sight("readback_memory", u64::from(memory_type)) {
        return;
    }
    let topology = ctx.caps.memory.topology.slug();
    if kind.cached {
        reims_vgpu_observe::off(format!(
            "readback_memory type={memory_type} cached=1 coherent={} topology={topology}",
            u8::from(kind.coherent),
        ));
    } else {
        reims_vgpu_observe::Emit::decline(
            "readback_memory",
            &ReadbackMemoryDegrade {
                memory_type,
                type_bits,
            },
        )
        .field("coherent", u8::from(kind.coherent))
        .field("topology", topology)
        .off();
    }
}

/// Which quarter of [`TARGET_POOL_MAX_ENTRIES`] the scratch target pool is
/// occupying, as a census route name.
///
/// The bands are quarters of the cap rather than fixed sizes, so a change to the
/// cap moves them with it and a series taken before the change stays comparable
/// in the only terms that matter — how close the pool came to its bound. Split
/// out from `acquire_target` so the naming is testable without a device: that
/// function is `unsafe` and takes a live `DeviceContext`, and an instrument
/// nobody can test is one whose zero nobody should believe.
fn target_pool_depth_band(len: usize) -> &'static str {
    let quarter = TARGET_POOL_MAX_ENTRIES / 4;
    if len < quarter {
        "target_pool_depth_q1"
    } else if len < quarter * 2 {
        "target_pool_depth_q2"
    } else if len < quarter * 3 {
        "target_pool_depth_q3"
    } else {
        "target_pool_depth_q4"
    }
}

#[cfg(test)]
mod target_pool_band_tests {
    use super::*;

    /// Each quarter, at both of its ends. A band that is right in the middle and
    /// wrong at a boundary reads as working for as long as the pool stays shallow.
    #[test]
    fn each_band_covers_its_quarter_of_the_cap() {
        let q = TARGET_POOL_MAX_ENTRIES / 4;
        assert_eq!(target_pool_depth_band(0), "target_pool_depth_q1");
        assert_eq!(target_pool_depth_band(q - 1), "target_pool_depth_q1");
        assert_eq!(target_pool_depth_band(q), "target_pool_depth_q2");
        assert_eq!(target_pool_depth_band(q * 2 - 1), "target_pool_depth_q2");
        assert_eq!(target_pool_depth_band(q * 2), "target_pool_depth_q3");
        assert_eq!(target_pool_depth_band(q * 3 - 1), "target_pool_depth_q3");
        assert_eq!(target_pool_depth_band(q * 3), "target_pool_depth_q4");
    }

    /// The band the cap acts in must be the top one, or a boot could sit at the
    /// bound and report headroom.
    #[test]
    fn the_band_at_the_cap_is_the_top_one() {
        assert_eq!(
            target_pool_depth_band(TARGET_POOL_MAX_ENTRIES),
            "target_pool_depth_q4"
        );
        assert_eq!(
            target_pool_depth_band(TARGET_POOL_MAX_ENTRIES - 1),
            "target_pool_depth_q4"
        );
        // Past the cap is not reachable — `acquire_target` evicts down to it —
        // but a band that stopped naming the top quarter there would be a silent
        // hole if it ever were.
        assert_eq!(
            target_pool_depth_band(TARGET_POOL_MAX_ENTRIES * 4),
            "target_pool_depth_q4"
        );
    }
}

#[cfg(test)]
mod recycle_tests {
    use super::*;
    use crate::engine::{GuestWriteSource, GuestWriteToken, SessionId};

    fn write_token(serial: u64) -> GuestWriteToken {
        GuestWriteToken {
            session: SessionId(0),
            serial,
        }
    }

    fn null_slot(w: u32, h: u32) -> SampledSlot {
        SampledSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            mip_levels: 1,
            layers: 1,
            volume: false,
            cube: false,
            arrayed: false,
            one_dim: false,
            format: crate::translate::pixel::vk_texel_layout(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
            swizzle: Default::default(),
        }
    }

    fn guest_runs(
        runs: Arc<Vec<reims_vgpu_memory::GuestRun>>,
    ) -> reims_vgpu_memory::GuestRunSource {
        reims_vgpu_memory::GuestRunSource {
            runs,
            source_offset: 0,
            total_len: 64,
            row_length_texels: 4,
            pages: None,
            physical_pages: None,
        }
    }

    #[test]
    fn sampled_guest_reuse_is_exact_and_ends_at_the_next_guest_operation() {
        let mut pools = ResourcePools::new();
        let slot = null_slot(4, 4);
        let runs = Arc::new(vec![reims_vgpu_memory::GuestRun {
            host_ptr: 0x1000,
            len: 64,
        }]);
        let source = guest_runs(Arc::clone(&runs));
        let bind = CbSampledGuest::runs(slot.key(), &source);

        assert!(pools.cb_sampled_guest(&bind).is_none());
        pools.note_cb_sampled_guest(bind.clone(), &slot);
        let rebuilt_source = guest_runs(Arc::new(vec![reims_vgpu_memory::GuestRun {
            host_ptr: 0x1000,
            len: 64,
        }]));
        let rebuilt = CbSampledGuest::runs(slot.key(), &rebuilt_source);
        assert!(
            pools.cb_sampled_guest(&rebuilt).is_some(),
            "the checked storage sequence, not a per-draw Vec allocation, is the identity"
        );

        let mut shifted = guest_runs(Arc::clone(&runs));
        shifted.source_offset = 4;
        assert!(
            pools
                .cb_sampled_guest(&CbSampledGuest::runs(slot.key(), &shifted))
                .is_none(),
            "a different guest byte window must record its own upload"
        );

        pools.begin_guest_operation(vk::CommandBuffer::from_raw(1));
        assert!(
            pools.cb_sampled_guest(&bind).is_none(),
            "guest CPU writes are unobservable, so reuse ends at the operation boundary"
        );
    }

    #[test]
    fn sampled_guest_write_invalidation_is_page_exact() {
        let mut pools = ResourcePools::new();
        let slot = null_slot(4, 4);
        let make = |host_ptr, page| {
            let mut source = guest_runs(Arc::new(vec![reims_vgpu_memory::GuestRun {
                host_ptr,
                len: 64,
            }]));
            source.physical_pages = reims_vgpu_memory::GuestPageSet::new(&[page]);
            CbSampledGuest::runs(slot.key(), &source)
        };
        let overlap = make(0x1000, 11);
        let disjoint = make(0x2000, 22);
        pools.note_cb_sampled_guest(overlap.clone(), &slot);
        pools.note_cb_sampled_guest(disjoint.clone(), &slot);

        let written = reims_vgpu_memory::GuestWritePages::new(&[11]).unwrap();
        pools.forget_cb_sampled_guest_pages(&written);

        assert!(pools.cb_sampled_guest(&overlap).is_none());
        assert!(pools.cb_sampled_guest(&disjoint).is_some());
    }

    #[test]
    fn a_guest_store_keeps_alias_bindings_and_discards_copied_snapshots() {
        use crate::engine::exec::BoundBuffer;
        use crate::engine::types::BufferContent;

        let mut pools = ResourcePools::new();
        let content = |host_ptr| {
            BufferContent::GuestRuns(reims_vgpu_memory::GuestRunSource {
                runs: Arc::new(vec![reims_vgpu_memory::GuestRun { host_ptr, len: 64 }]),
                source_offset: 0,
                total_len: 64,
                row_length_texels: 0,
                pages: None,
                physical_pages: None,
            })
        };
        let alias_content = content(0x1000);
        let copy_content = content(0x2000);
        let alias = super::CbBind::of(&alias_content);
        let copy = super::CbBind::of(&copy_content);
        let alias_bound = BoundBuffer {
            buffer: vk::Buffer::null(),
            offset: 0,
            guest_import: true,
        };
        let copy_bound = BoundBuffer {
            guest_import: false,
            ..alias_bound
        };
        pools.note_cb_bound_buffer(alias.clone(), alias_bound);
        pools.note_cb_bound_buffer(copy.clone(), copy_bound);

        pools.forget_cb_bound_buffer_copies();

        assert!(
            pools.cb_bound_buffer(alias.key()).is_some(),
            "the binding still names the same live guest allocation after its bytes change"
        );
        assert!(
            pools.cb_bound_buffer(copy.key()).is_none(),
            "a copied snapshot predating the Store is stale"
        );
    }

    #[test]
    fn a_guest_operation_keeps_live_aliases_and_immutable_copies_only() {
        use crate::engine::exec::BoundBuffer;
        use crate::engine::types::BufferContent;

        let mut pools = ResourcePools::new();
        let guest = BufferContent::GuestRuns(reims_vgpu_memory::GuestRunSource {
            runs: Arc::new(vec![reims_vgpu_memory::GuestRun {
                host_ptr: 0x1000,
                len: 64,
            }]),
            source_offset: 0,
            total_len: 64,
            row_length_texels: 0,
            pages: None,
            physical_pages: None,
        });
        let guest_alias = BufferContent::GuestRuns(reims_vgpu_memory::GuestRunSource {
            runs: Arc::new(vec![reims_vgpu_memory::GuestRun {
                host_ptr: 0x2000,
                len: 64,
            }]),
            source_offset: 0,
            total_len: 64,
            row_length_texels: 0,
            pages: None,
            physical_pages: None,
        });
        let immutable = BufferContent::Bytes(Arc::new(vec![7; 64]));
        let alias = super::CbBind::of(&guest_alias);
        let snapshot = super::CbBind::of(&guest);
        let stable = super::CbBind::of(&immutable);
        let copied = BoundBuffer {
            buffer: vk::Buffer::null(),
            offset: 0,
            guest_import: false,
        };
        pools.note_cb_bound_buffer(snapshot.clone(), copied);
        pools.note_cb_bound_buffer(stable.clone(), copied);
        pools.note_cb_bound_buffer(
            alias.clone(),
            BoundBuffer {
                guest_import: true,
                ..copied
            },
        );

        pools.begin_guest_operation(vk::CommandBuffer::from_raw(1));

        assert!(pools.cb_bound_buffer(alias.key()).is_some());
        assert!(pools.cb_bound_buffer(stable.key()).is_some());
        assert!(
            pools.cb_bound_buffer(snapshot.key()).is_none(),
            "a copied guest window is not a versioned source"
        );
    }

    /// A cache holding entries a failed submission promised to fill is emptied,
    /// and emptied completely — the entries, the bytes, and the answers.
    ///
    /// Publishing an admission while its command buffer is still recording is
    /// what lets the next draw of a batch find the window, and the price is that
    /// an entry can outlive the promise that filled it. This is the whole of the
    /// undo, so each of the three things it must leave behind is asserted
    /// separately:
    ///
    /// - **The bytes go back to zero.** The gauge follows the retained images.
    #[test]
    fn a_discarded_cache_leaves_no_entry_and_no_bytes() {
        let mut pools = ResourcePools::new();
        let resource = reims_vgpu_core::ResourceLifetime::new();
        let owner = resource.reference();

        for i in 0..3u32 {
            let slot = null_slot(16 + i, 16);
            pools.push_sampled_entry(ResidentSampledSlot {
                slot,
                fingerprint: SampledFingerprint::Content(i as u128),
                content: Some(std::sync::Arc::new(vec![i as u8; 4096])),
                content_len: 4096,
                owners: vec![owner.clone()],
                last_touch_ms: 0,
            });
            pools.sampled_cache_bytes += 4096;
        }

        let taken = pools.take_whole_sampled_cache();

        assert_eq!(taken.len(), 3, "every entry is handed back for disposal");
        assert_eq!(
            pools.sampled_cache_bytes, 0,
            "the byte accounting follows the entries out"
        );
        assert_eq!(
            pools.take_whole_sampled_cache().len(),
            0,
            "discarding an empty cache is a no-op, not a second sweep"
        );
    }

    /// A bind whose gather was never recorded must not survive the draw that
    /// abandoned, and the binds around it must.
    ///
    /// The gather arm of `stage_buffer_content` is the only one that publishes a
    /// memo entry naming a slot it has not filled: the slot comes off the free
    /// list holding the previous tenant's bytes, and what fills it is recorded
    /// hundreds of lines later, past three recoverable sampled refusals. A draw
    /// that takes one of those exits drops the owed copy with its stack frame.
    /// If the memo entry outlives it, the next draw of the same command buffer
    /// hits the memo, records no copy, and binds the previous tenant as its
    /// constant buffer or vertex stream — wrong pixels, or a loop bound that
    /// does not terminate.
    ///
    /// The second half is why this is a list and not a clear: entries published
    /// by draws that completed are still correct, and this rail carries ~4.8
    /// binds a draw.
    #[test]
    fn a_bind_whose_gather_was_never_recorded_does_not_outlive_the_draw() {
        use crate::engine::exec::BoundBuffer;
        use crate::engine::types::BufferContent;

        let mut pools = ResourcePools::new();
        let filled = BufferContent::Bytes(std::sync::Arc::new(vec![1u8; 64]));
        let owed = BufferContent::Bytes(std::sync::Arc::new(vec![2u8; 128]));
        let bound = |n: u64| BoundBuffer {
            buffer: vk::Buffer::null(),
            offset: n,
            guest_import: false,
        };

        // A bind whose bytes are already where the descriptor points — a CPU
        // write into mapped staging — and one that owes a gather.
        let filled_bind = super::CbBind::of(&filled);
        pools.note_cb_bound_buffer(filled_bind.clone(), bound(1));
        let owed_bind = super::CbBind::of(&owed);
        let owed_key = owed_bind.key();
        pools.note_cb_bound_buffer(owed_bind.clone(), bound(2));
        pools.note_cb_bind_owes_gather(owed_key);

        assert_eq!(
            pools.discard_cb_binds_owed_a_gather(),
            1,
            "exactly the unfilled bind is forgotten"
        );
        assert!(
            pools.cb_bound_buffer(owed_bind.key()).is_none(),
            "binding this would hand the shader the recycled slot's previous tenant"
        );
        assert!(
            pools.cb_bound_buffer(filled_bind.key()).is_some(),
            "a bind published by a draw that completed is still correct, and \
             re-staging it is a cost this rail pays 4.8 times a draw"
        );
        assert_eq!(
            pools.discard_cb_binds_owed_a_gather(),
            0,
            "the list is drained, so a second abandon cannot remove an unrelated \
             bind the next command buffer published at the same address"
        );

        // Recording the gathers is what makes the entry answerable. After it, an
        // abandoning draw has nothing to forget.
        let owed_bind = super::CbBind::of(&owed);
        pools.note_cb_bound_buffer(owed_bind.clone(), bound(3));
        pools.note_cb_bind_owes_gather(owed_bind.key());
        pools.note_cb_gathers_recorded();
        assert_eq!(pools.discard_cb_binds_owed_a_gather(), 0);
        assert!(
            pools.cb_bound_buffer(owed_bind.key()).is_some(),
            "a gather that reached the command buffer leaves a bind the rest of \
             that command buffer may reuse"
        );
    }

    /// Two different textures filed under one digest stay two textures.
    ///
    /// A natural 128-bit collision is not something a test can produce, so this
    /// builds the state one would produce — an entry whose `Content` digest
    /// equals the incoming blob's while its retained bytes differ — and asks the
    /// lookup. That is the whole of what `ResidentSampledSlot::content` buys,
    /// and without it this bind returns the wrong image with nothing to log.
    #[test]
    fn a_collided_digest_over_different_bytes_is_not_a_sampled_hit() {
        let mut pools = ResourcePools::new();
        let counters = EngineCounters::default();

        let retained: Vec<u8> = (0..64u8).collect();
        let incoming: Vec<u8> = (0..64u8).map(|b| b ^ 0x5a).collect();
        assert_ne!(retained, incoming, "the two blobs must differ");

        let slot = null_slot(8, 8);
        let key = slot.key();
        let resource = reims_vgpu_core::ResourceLifetime::new();
        let owner = resource.reference();
        // File the retained blob under the *incoming* blob's digest. This is the
        // collision, forced rather than found.
        pools.sampled_cache.push(ResidentSampledSlot {
            slot,
            fingerprint: SampledFingerprint::Content(sampled_content_hash(&incoming)),
            content: Some(std::sync::Arc::new(retained.clone())),
            content_len: retained.len(),
            owners: vec![owner.clone()],
            last_touch_ms: 0,
        });

        assert!(
            pools
                .find_cached_sampled(key, &incoming, None, Some(&owner), &counters)
                .is_none(),
            "equal digests over different bytes must not bind the retained image"
        );
        assert!(
            pools
                .find_cached_sampled(key, &retained, None, Some(&owner), &counters)
                .is_none(),
            "the retained bytes do not hash to the digest they were filed under, \
             so they are not a hit either — the digest still gates the walk"
        );
    }

    /// The bytes decide a hit, and equal bytes are one entry however the blob
    /// reached the cache. This is the half the compare must not cost: a genuine
    /// re-present of identical content still hits.
    #[test]
    fn the_same_bytes_under_their_own_digest_are_a_sampled_hit() {
        let mut pools = ResourcePools::new();
        let counters = EngineCounters::default();

        let content: Vec<u8> = (0..64u8).map(|b| b.wrapping_mul(7)).collect();
        let slot = null_slot(8, 8);
        let key = slot.key();
        let resource = reims_vgpu_core::ResourceLifetime::new();
        let owner = resource.reference();
        pools.sampled_cache.push(ResidentSampledSlot {
            slot,
            fingerprint: SampledFingerprint::Content(sampled_content_hash(&content)),
            content: Some(std::sync::Arc::new(content.clone())),
            content_len: content.len(),
            owners: vec![owner.clone()],
            last_touch_ms: 0,
        });

        let copy = content.clone();
        assert!(
            pools
                .find_cached_sampled(key, &copy, None, Some(&owner), &counters)
                .is_some(),
            "identical content must still hit, or the compare has cost a real reuse"
        );
    }

    fn null_storage_slot(w: u32, h: u32) -> StorageImageSlot {
        StorageImageSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            key: StorageImageKey {
                width: w,
                height: h,
                format: StorageImageFormat::default(),
                sampled_only: false,
            },
        }
    }

    #[test]
    fn every_recycle_pool_retains_the_active_working_set() {
        let mut pools = ResourcePools::new();
        for i in 0..104 {
            pools.admit_sampled(null_slot(16, 16 + i));
            pools.admit_target(null_target(16, 16 + i, translate::pixel::SCANOUT_FORMAT));
            pools.admit_storage_image(null_storage_slot(16, 16 + i));
        }
        assert_eq!(pools.sampled_free.len(), 104);
        assert_eq!(pools.target_free.len(), 104);
        assert_eq!(pools.storage_image_free.len(), 104);
    }

    /// `pop_any_pool_entry` drains a keyed pool one entry at a time across all
    /// buckets and removes empties, so the idle trim can empty the whole pool.
    #[test]
    fn pop_any_pool_entry_drains_all_buckets() {
        let mut pool: HashMap<u32, Vec<u32>> = HashMap::new();
        pool.insert(1, vec![10, 11]);
        pool.insert(2, vec![20]);
        let mut popped = Vec::new();
        while let Some(v) = pop_any_pool_entry(&mut pool) {
            popped.push(v);
        }
        popped.sort_unstable();
        assert_eq!(popped, vec![10, 11, 20]);
        assert!(pool.is_empty(), "emptied buckets are removed");
    }

    /// Evicted sampled-cache slots rejoin `sampled_free` for reuse (no fresh
    /// image creation next frame), including a working set wider than the old
    /// per-key capacity.
    #[test]
    fn evicted_sampled_slots_recycle_for_the_whole_active_working_set() {
        let mut pools = ResourcePools::new();
        let hd = null_slot(1920, 1080).key();
        for i in 0..100 {
            pools.admit_sampled(null_slot(1920, 1080));
            assert_eq!(pools.sampled_free.count_for(&hd), i + 1);
        }
        assert_eq!(pools.sampled_free.count_for(&hd), 100);
        let small = null_slot(64, 64).key();
        pools.admit_sampled(null_slot(64, 64));
        assert_eq!(pools.sampled_free.count_for(&small), 1);
    }

    /// Elapsed time leaves live sampled entries alone, while deleting their
    /// serialized owner makes them reclaimable.
    #[test]
    fn sampled_cache_entries_follow_resource_lifetime_not_age() {
        let mut pools = ResourcePools::new();
        let resource = reims_vgpu_core::ResourceLifetime::new();
        let owner = resource.reference();
        let push = |pools: &mut ResourcePools, w: u32, h: u32, touch: u64, len: usize| {
            pools.sampled_cache_bytes = pools.sampled_cache_bytes.saturating_add(len);
            pools.sampled_cache.push(ResidentSampledSlot {
                slot: null_slot(w, h),
                fingerprint: SampledFingerprint::Content(((w as u128) << 64) | h as u128),
                // This test only ages entries out; nothing here looks one up by
                // content, so the geometry stands in for a digest and there are
                // no bytes to retain beside it.
                content: None,
                content_len: len,
                owners: vec![owner.clone()],
                last_touch_ms: touch,
            });
        };
        push(&mut pools, 1920, 1080, 1_000, 8_000_000);
        push(&mut pools, 1280, 720, 1_500, 4_000_000);
        push(&mut pools, 640, 480, 9_000, 1_000_000);
        assert_eq!(pools.sampled_cache.len(), 3);
        assert_eq!(pools.sampled_cache_bytes, 13_000_000);

        assert!(pools.plan_idle_maintenance(10_000));
        assert_eq!(pools.sampled_cache.len(), 3);
        assert_eq!(pools.sampled_cache_bytes, 13_000_000);

        drop(resource);
        assert_eq!(
            pools.take_released_sampled_entries().len(),
            3,
            "resource deletion, not age or a chosen capacity, ends retention"
        );
        assert!(pools.sampled_cache.is_empty());
        assert_eq!(pools.sampled_cache_bytes, 0);
    }

    /// Compute-storage residents are standalone allocations, but elapsed time
    /// still cannot end their guest-visible lifetime.
    #[test]
    fn maintenance_does_not_reclaim_compute_storage_by_age() {
        let mut pools = ResourcePools::new();
        let admit = |pools: &mut ResourcePools, tex: u32, touch: u64, pinned: bool| {
            let id = ComputeStorageResidencyKey::linear(
                reims_vgpu_protocol::ResourceId::new(tex, 1),
                0,
                0,
                0,
                8,
                8,
                0,
            );
            pools.compute_storage_registry.insert(
                id,
                ResidentStorageImageSlot {
                    slot: null_storage_slot(8, 8),
                    generation: 0,
                    access: ResidentAccess::Untouched,
                    pinned,
                    gpu_only_content: false,
                    last_touch_ms: touch,
                },
            );
            pools.compute_storage_order.push_back(id);
        };
        admit(&mut pools, 1, 1_000, false); // aged, evictable
        admit(&mut pools, 2, 1_500, false); // aged, evictable
        admit(&mut pools, 3, 1_500, true); // aged but PINNED — must survive
        admit(&mut pools, 4, 9_000, false); // freshly touched — must survive
        assert_eq!(pools.compute_storage_registry.len(), 4);

        assert!(pools.plan_idle_maintenance(10_000));
        assert_eq!(pools.compute_storage_registry.len(), 4);
        assert_eq!(pools.compute_storage_order.len(), 4);
    }

    /// A compute-storage resident holding dispatch output nothing has copied out
    /// is never aged out, however long it sits — and becomes reclaimable the
    /// moment a readback lands.
    ///
    /// This is the produce-once/sample-many shape: a dispatch writes the image,
    /// later chains only *read* it, and no deferred writeback is ever armed — so
    /// `pinned` stays false for its whole life while the image is the only place
    /// its content exists. Destroying it is not a re-upload here; the next
    /// dispatch naming the identity refuses with `ResidentSampleAbsent`.
    ///
    /// Fails without the gate: the resident is taken on the first call.
    #[test]
    fn a_compute_resident_that_is_the_only_copy_of_its_output_is_never_aged_out() {
        let mut pools = ResourcePools::new();
        let id = admit_compute_resident(&mut pools, 1, 1_000, false);
        // The dispatch wrote it. Nothing else holds the result.
        pools.mark_resident_storage_image(&id, 7);

        assert!(pools.plan_idle_maintenance(10_000));
        assert!(pools.compute_storage_registry.contains_key(&id));

        // A readback lands and the same resident is reclaimable like any other.
        assert!(pools.note_compute_storage_copied_out(&id));
        assert!(pools.plan_idle_maintenance(100_000));
        assert!(
            pools.compute_storage_registry.contains_key(&id),
            "a current backing permits pressure reclaim, not time-based removal"
        );
    }

    /// The allocation-failure reclaim offers up no compute-storage resident whose
    /// only copy is on the GPU, and offers nothing at all when that is all there
    /// is — so the allocation refuses rather than a later dispatch.
    ///
    /// This registry's losses are worse than the target registry's: nothing
    /// recreates a compute-storage resident's contents, so one taken here costs
    /// a refused dispatch rather than a re-upload.
    ///
    /// Fails without the gate: both residents stay in the list once marked.
    #[test]
    fn the_compute_allocation_reclaim_offers_up_nothing_that_is_the_only_copy() {
        let mut pools = ResourcePools::new();
        let a = admit_compute_resident(&mut pools, 1, 1_000, false);
        let b = admit_compute_resident(&mut pools, 2, 2_000, false);
        assert_eq!(
            pools.recoverable_compute_storage_residents(),
            vec![a, b],
            "both are re-servable, so both could be given back"
        );

        pools.mark_resident_storage_image(&a, 1);
        assert_eq!(
            pools.recoverable_compute_storage_residents(),
            vec![b],
            "the sole copy drops out; its peer is still offered"
        );

        pools.mark_resident_storage_image(&b, 1);
        assert!(
            pools.recoverable_compute_storage_residents().is_empty(),
            "nothing left that can be destroyed without refusing a later dispatch"
        );
    }

    /// The maintained compute-storage sole-copy totals agree with a walk at
    /// every transition, including the ones that move nothing and the removal
    /// that is the only way such a resident leaves.
    #[test]
    fn the_maintained_compute_sole_copy_totals_track_the_walk() {
        let mut pools = ResourcePools::new();
        let check = |pools: &ResourcePools, what: &str| {
            let walk = {
                let sole = || {
                    pools
                        .compute_storage_registry
                        .values()
                        .filter(|r| r.gpu_only_content)
                };
                NonPinnedTotals {
                    count: sole().count(),
                    bytes: sole().map(ResourcePools::storage_slot_bytes).sum(),
                }
            };
            assert_eq!(
                pools.compute_storage_sole_copy, walk,
                "maintained compute sole-copy totals disagree with the walk after {what}"
            );
        };
        check(&pools, "construction");

        let a = admit_compute_resident(&mut pools, 1, 0, false);
        let b = admit_compute_resident(&mut pools, 2, 0, false);
        check(&pools, "two admits");
        assert_eq!(
            pools.compute_storage_sole_copy.count, 0,
            "a resident no dispatch has written holds no guest work"
        );

        pools.mark_resident_storage_image(&a, 1);
        check(&pools, "a dispatch wrote the first");
        assert_eq!(pools.compute_storage_sole_copy.count, 1);
        // A second dispatch into the same resident is still one resident.
        pools.mark_resident_storage_image(&a, 2);
        check(&pools, "a second dispatch into the same resident");
        assert_eq!(pools.compute_storage_sole_copy.count, 1);

        pools.mark_resident_storage_image(&b, 1);
        check(&pools, "a dispatch wrote the second");
        assert_eq!(pools.compute_storage_sole_copy.count, 2);

        assert!(pools.note_compute_storage_copied_out(&a));
        check(&pools, "a landed readback");
        assert_eq!(pools.compute_storage_sole_copy.count, 1);
        assert!(pools.note_compute_storage_copied_out(&a));
        check(&pools, "a redundant copy-out");
        assert_eq!(pools.compute_storage_sole_copy.count, 1);

        // The guest deleting the object is the other way the flag clears, and
        // without it `retire_linear_residents` would strand the image.
        assert!(pools.note_compute_storage_content_retired(&b));
        check(&pools, "the guest retiring the object");
        assert_eq!(pools.compute_storage_sole_copy.count, 0);

        // Removal folds a still-sole-copy resident out — the re-key path.
        pools.mark_resident_storage_image(&b, 3);
        check(&pools, "the second written again");
        assert_eq!(pools.compute_storage_sole_copy.count, 1);
        assert!(pools.remove_compute_storage_resident(&b).is_some());
        check(&pools, "removing a sole-copy resident");
        assert_eq!(pools.compute_storage_sole_copy, NonPinnedTotals::default());

        assert!(
            !pools.note_compute_storage_copied_out(&b),
            "an identity holding no resident reports the miss rather than inventing a subtraction"
        );
        check(&pools, "a copy-out for an absent identity");
    }

    fn admit_compute_resident(
        pools: &mut ResourcePools,
        tex: u32,
        touch: u64,
        pinned: bool,
    ) -> ComputeStorageResidencyKey {
        let id = ComputeStorageResidencyKey::linear(
            reims_vgpu_protocol::ResourceId::new(tex, 1),
            0,
            0,
            0,
            8,
            8,
            0,
        );
        pools.compute_storage_registry.insert(
            id,
            ResidentStorageImageSlot {
                slot: null_storage_slot(8, 8),
                generation: 0,
                access: ResidentAccess::Untouched,
                pinned,
                gpu_only_content: false,
                last_touch_ms: touch,
            },
        );
        pools.compute_storage_order.push_back(id);
        id
    }

    /// `compute_resident_snapshot` records a use without changing lifetime.
    ///
    /// The sibling of `a_read_only_compute_storage_resident_is_not_aged_out`
    /// over the other read-only accessor. All three of them took `&self` and
    /// refreshed nothing, so a produce-once/sample-many resident looked
    /// stone-cold to the drain — and its loss is a refused dispatch, not a
    /// re-upload. This one asserted the same property through the capacity
    /// walk's victim choice while there was one; the drain is now the only
    /// consumer of `last_touch_ms`, so it asserts it there.
    ///
    /// Fails without the fix: `tex 1` is reclaimed alongside its untouched peers.
    #[test]
    fn a_compute_resident_read_through_snapshot_is_not_aged_out() {
        let mut pools = ResourcePools::new();
        let read = admit_compute_resident(&mut pools, 1, 1_000, false);
        let untouched = admit_compute_resident(&mut pools, 2, 1_000, false);
        pools.idle_clock_ms = 10_000;
        // A read through the product accessor, which is how a resident sample
        // consumer touches a resident it never dispatches into again.
        assert!(pools.compute_resident_snapshot(&read).is_some());

        pools.plan_idle_maintenance(20_000);

        assert!(
            pools.compute_storage_registry.contains_key(&read),
            "a resident a chain is reading is in use and must not be destroyed"
        );
        assert!(
            pools.compute_storage_registry.contains_key(&untouched),
            "elapsed time does not end the untouched resource's lifetime"
        );
    }

    /// A resident that is only ever read remains live.
    ///
    /// The produce-once/sample-many case: a compute chain writes an image and
    /// later chains sample it without dispatching into it again. All three
    /// read-only accessors took `&self` and refreshed nothing, so such a
    /// resident looked stone-cold to both reclaim rules — the cap sweep takes
    /// the minimum `last_touch_ms` and the drain compares it against a cutoff —
    /// and its loss is a refused dispatch, not a re-upload.
    ///
    /// Fails without the fix: `tex 1` is reclaimed alongside its untouched peer.
    #[test]
    fn a_read_only_compute_storage_resident_is_not_aged_out() {
        let mut pools = ResourcePools::new();
        let read = admit_compute_resident(&mut pools, 1, 1_000, false);
        let untouched = admit_compute_resident(&mut pools, 2, 1_000, false);
        // The drain's clock has to be current before a read can be recorded
        // against it; the real caller advances it from the poll heartbeat.
        pools.idle_clock_ms = 10_000;
        assert!(pools.compute_resident_sample_source(&read).is_some());

        pools.plan_idle_maintenance(20_000);

        assert!(
            pools.compute_storage_registry.contains_key(&read),
            "a resident a chain is reading is in use and must not be destroyed"
        );
        assert!(
            pools.compute_storage_registry.contains_key(&untouched),
            "elapsed time does not end the untouched resource's lifetime"
        );
    }

    /// Re-keying an identity whose resident still owes a deferred writeback is
    /// refused, not performed.
    ///
    /// One identity holds one slot, so a shape change destroys the old image —
    /// and a pinned resident's pixels exist only there, having been accepted
    /// from the guest and not yet landed in its pages. Every other removal in
    /// this registry skips a pinned entry; this path did not, and the loss
    /// surfaced later and elsewhere as `StorageReadResidentAbsent`.
    ///
    /// The unpinned half must keep working: a re-key of a resident that owes
    /// nothing is an ordinary recreate.
    ///
    /// Fails without the fix: `compute_rekey_refusal` does not exist and the
    /// destroy is unconditional.
    #[test]
    fn rekeying_a_pinned_compute_resident_is_refused_rather_than_dropped() {
        use reims_vgpu_observe::decline::Decline;
        let mut pools = ResourcePools::new();
        let pinned = admit_compute_resident(&mut pools, 1, 0, true);
        let unpinned = admit_compute_resident(&mut pools, 2, 0, false);
        let same = StorageImageKey {
            width: 8,
            height: 8,
            format: StorageImageFormat::default(),
            sampled_only: false,
        };
        let reshaped = StorageImageKey { width: 16, ..same };

        assert!(
            pools.compute_rekey_refusal(&pinned, same).is_none(),
            "the same shape is not a re-key, pinned or not"
        );
        assert!(
            pools.compute_rekey_refusal(&unpinned, reshaped).is_none(),
            "re-keying a resident that owes no writeback is an ordinary recreate"
        );
        let decline = pools
            .compute_rekey_refusal(&pinned, reshaped)
            .expect("re-keying a pinned resident must be refused");
        assert_eq!(
            decline.slug(),
            "vk_compute_exec_resident_rekey_would_drop_pinned"
        );
        let fields = decline.fields();
        let field = |name: &str| {
            fields
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            (field("held_width"), field("wanted_width")),
            (Some("8".to_string()), Some("16".to_string())),
            "the refusal names both shapes, so a reader can tell which side moved"
        );
        assert!(
            pools.compute_storage_registry.contains_key(&pinned),
            "the refusal must leave the unflushed content in place"
        );
    }

    /// With every remaining resident pinned there is no victim, so the caller's
    /// sweep loop terminates and the registry soft-exceeds its cap rather than
    /// destroying content whose only copy is on the GPU. Same trade the sibling
    /// target registry's walk makes.
    #[test]
    fn an_all_pinned_compute_storage_registry_offers_no_victim() {
        let mut pools = ResourcePools::new();
        admit_compute_resident(&mut pools, 1, 0, true);
        admit_compute_resident(&mut pools, 2, 0, true);
        assert!(
            pools.recoverable_compute_storage_residents().is_empty(),
            "every entry is pinned, so there is nothing to give back"
        );
        let recoverable = admit_compute_resident(&mut pools, 3, 0, false);
        assert_eq!(
            pools.recoverable_compute_storage_residents(),
            vec![recoverable],
            "the one unpinned resident is the only candidate"
        );
    }

    /// The compatibility counter remains zero after removing capacity-based
    /// drops, while admissions still count the whole active working set.
    #[test]
    fn recycle_stats_count_every_admission_and_no_capacity_drops() {
        let mut pools = ResourcePools::new();
        assert_eq!(pools.recycle_stats(), (0, 0, 0, 0));

        for _ in 0..100 {
            pools.admit_sampled(null_slot(1920, 1080));
        }
        pools.admit_sampled(null_slot(64, 64));

        let (free_hits, free_allocs, admits, cap_drops) = pools.recycle_stats();
        assert_eq!(free_hits, 0, "no acquires happened");
        assert_eq!(free_allocs, 0, "no acquires happened");
        assert_eq!(admits, 101);
        assert_eq!(cap_drops, 0);
    }

    fn null_target(w: u32, h: u32, format: vk::Format) -> FreeTargetImage {
        FreeTargetImage {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            sample_count: 1,
            format,
        }
    }

    /// Resident-target images displaced from the identity registry rejoin the
    /// geometry-keyed pool for the complete active working set.
    #[test]
    fn displaced_targets_recycle_for_the_whole_active_working_set() {
        let mut pools = ResourcePools::new();
        let fmt = translate::pixel::SCANOUT_FORMAT;
        let key = null_target(1920, 1080, fmt).key();

        for i in 0..100 {
            pools.admit_target(null_target(1920, 1080, fmt));
            assert_eq!(pools.target_free.count_for(&key), i + 1);
        }
        assert_eq!(pools.target_free.count_for(&key), 100);

        // A different format is an independent bucket (an RGBA image cannot back
        // a BGRA attachment).
        let rgba = null_target(1920, 1080, translate::pixel::RESIDENT_RGBA_FORMAT).key();
        pools.admit_target(null_target(
            1920,
            1080,
            translate::pixel::RESIDENT_RGBA_FORMAT,
        ));
        assert_eq!(pools.target_free.count_for(&rgba), 1);
    }

    /// The exact video regression this pool fixes: a per-frame *generation* bump
    /// makes every frame a new `TargetIdentity` (registry miss) — but the
    /// displaced image, recycled by (geometry, format), is popped back on the
    /// next frame's create instead of a fresh `vkCreateImage`/`vkAllocateMemory`.
    /// The reuse split (`target_free_hits` vs `target_free_allocs`) proves it: a
    /// steady-geometry stream after the first fill is all hits, zero allocs.
    #[test]
    fn recycled_target_is_reused_across_generation_bumps() {
        let mut pools = ResourcePools::new();
        let fmt = translate::pixel::SCANOUT_FORMAT;

        // Frame 0: cold — no free image, so it counts as an alloc (miss).
        assert!(pools.take_free_target(1920, 1080, 1, fmt).is_none());
        // Its predecessor image is displaced (a new generation replaced it) and
        // recycled.
        pools.admit_target(null_target(1920, 1080, fmt));

        // Frames 1..N: each pops the recycled image (hit) and recycles the one
        // it replaces — steady state is hit-per-frame, alloc-once.
        for f in 0..8 {
            assert!(
                pools.take_free_target(1920, 1080, 1, fmt).is_some(),
                "frame {f} must reuse the recycled image"
            );
            pools.admit_target(null_target(1920, 1080, fmt));
        }

        let (hits, allocs, _admits, _drops) = pools.target_recycle_stats();
        assert_eq!(hits, 8, "8 steady frames each reused a recycled image");
        assert_eq!(allocs, 1, "only the cold first frame allocated");
    }

    /// A ring slot that owes cleanup, with null handles — enough for the
    /// graveyard's mask bookkeeping, which reads only `pending.is_some()`.
    fn pending_slot() -> CmdSlot {
        CmdSlot {
            cmd_buf: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            pending: Some(PendingGpuCleanup {
                dsets: Vec::new(),
                scatter_dsets: Vec::new(),
                staging: Vec::new(),
                gather: Vec::new(),
                readback: Vec::new(),
                sampled: Vec::new(),
                attachment_snapshots: Vec::new(),
                storage_images: Vec::new(),
                unpin_residents: Vec::new(),
                unpin_compute_residents: Vec::new(),
                guest_write_tokens: Vec::new(),
            }),
            submission: SlotSubmission::HostOwned,
            span: super::gpu_span::SlotSpan::Idle,
            readback_span_armed: false,
        }
    }

    fn idle_slot() -> CmdSlot {
        CmdSlot {
            cmd_buf: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            pending: None,
            submission: SlotSubmission::HostOwned,
            span: super::gpu_span::SlotSpan::Idle,
            readback_span_armed: false,
        }
    }

    #[test]
    fn opportunistic_retirement_never_polls_a_queue_owned_fence() {
        let counters = EngineCounters::default();
        let (receipt, returned) = crate::engine::queue_owner::PendingQueueSubmit::test_pair();
        let mut pools = ResourcePools::new();
        let mut slot = pending_slot();
        slot.submission = SlotSubmission::QueueOwned(receipt);
        pools.slots.push(slot);

        assert!(!pools.try_take_submit_return(&counters, 0).unwrap());
        assert!(matches!(
            pools.slots[0].submission,
            SlotSubmission::QueueOwned(_)
        ));

        returned.send(Ok(())).unwrap();
        assert!(pools.try_take_submit_return(&counters, 0).unwrap());
        assert!(matches!(
            pools.slots[0].submission,
            SlotSubmission::HostOwned
        ));
    }

    #[test]
    fn rejected_async_submit_never_becomes_a_waitable_fence() {
        let counters = EngineCounters::default();
        let (receipt, returned) = crate::engine::queue_owner::PendingQueueSubmit::test_pair();
        let mut pools = ResourcePools::new();
        let mut slot = pending_slot();
        slot.submission = SlotSubmission::QueueOwned(receipt);
        pools.slots.push(slot);

        returned
            .send(Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
            .unwrap();
        assert!(pools.try_take_submit_return(&counters, 0).is_err());
        assert!(matches!(
            pools.slots[0].submission,
            SlotSubmission::Failed(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ));
    }

    /// The completion stamp pays for the guest-read rail exactly when the rail
    /// ran, and never otherwise.
    ///
    /// Both directions matter. A debt that survived its quiesce would make every
    /// stamp for the rest of the boot wait out the whole ring, which is a
    /// serialization of the guest against the host GPU that no counter would
    /// name. A debt that was never recorded would let a `vkCmdCopyBufferToImage`
    /// read guest pages after the guest was told it could repaint them, which is
    /// the corruption the rail is allowed to exist only because this prevents.
    #[test]
    fn a_stamp_waits_for_guest_reads_only_when_one_was_recorded() {
        let mut pools = ResourcePools::new();
        assert!(!super::super::super::guest_access_outstanding());
        assert!(
            !pools.has_guest_read_debt(),
            "a device that has recorded nothing owes no wait; this is every \
             packet on a host with no host-pointer import"
        );

        pools.note_guest_read_recorded();
        assert!(
            super::super::super::guest_access_outstanding(),
            "a read-only packet still needs its completion stamp queued behind the GPU read"
        );
        assert!(pools.has_guest_read_debt());
        pools.settle_guest_read_debt();
        assert!(!super::super::super::guest_access_outstanding());
        assert!(
            !pools.has_guest_read_debt(),
            "one recorded read is one wait, not a wait at every stamp after it"
        );

        // Several reads inside one packet still settle in one quiesce: the
        // wait retires the whole ring, so there is nothing left for a second.
        pools.note_guest_read_recorded();
        pools.note_guest_read_recorded();
        assert!(pools.has_guest_read_debt());
        pools.settle_guest_read_debt();
        assert!(!pools.has_guest_read_debt());
    }

    /// A remembered bind names a pool slot, and it must not outlive either
    /// event that ends that slot's usefulness.
    ///
    /// The seal and the recycle are lifetime: both hand the slot away, and a
    /// later bind of the same content would be given a buffer the ring is free
    /// to reissue to somebody else. Guest writes do not end the slot lifetime;
    /// the inverse-page-index test above covers their exact content lifetime.
    #[test]
    fn a_remembered_bind_does_not_survive_either_slot_lifetime_end() {
        let content = crate::engine::types::BufferContent::from(vec![7u8; 4096]);
        let bind = super::super::CbBind::of(&content);
        let bound = crate::engine::exec::BoundBuffer {
            buffer: vk::Buffer::null(),
            offset: 0,
            guest_import: false,
        };
        /// One thing that ends a remembered bind's life, named for the failure
        /// message.
        type EndOfLife<'a> = (&'a str, &'a dyn Fn(&mut ResourcePools));
        let ends: [EndOfLife<'_>; 2] = [
            ("a seal", &|p| {
                p.seal_entry(Vec::new(), Vec::new());
            }),
            ("a staging recycle", &|p| p.recycle_staging()),
        ];
        for (what, end) in ends {
            let mut pools = ResourcePools::new();
            pools.note_cb_bound_buffer(bind.clone(), bound);
            assert!(
                pools.cb_bound_buffer(bind.key()).is_some(),
                "a bind must be reusable before {what}"
            );
            end(&mut pools);
            assert!(
                pools.cb_bound_buffer(bind.key()).is_none(),
                "a bind remembered across {what} names a slot no longer this command buffer's"
            );
        }
    }

    /// A remembered bind keeps the allocation its key names alive.
    ///
    /// This is the identity half of the map's contract, and the test above is
    /// only the lifetime half — a bind can be dropped at all the right moments
    /// and still be *answered by the wrong content* in between. The key is an
    /// `Arc`'s address, and a `BufferContent::Bytes` is `Arc::new`-ed per bind
    /// from a freshly read `Vec` and dropped with the `DrawRequest`. If the
    /// entry does not hold that `Arc`, the allocator is free to hand the same
    /// address to the next draw's unrelated read — `ArcInner<Vec<u8>>` is a
    /// fixed 40-byte allocation whatever the payload, so it does — and a draw
    /// renders through another draw's bytes with the reuse counted as a win.
    ///
    /// Asserted through a `Weak` rather than by trying to provoke a collision:
    /// a test that frees an address and allocates again is at the allocator's
    /// discretion and would be flaky in the direction that reads as passing.
    /// "The map still holds it" is the invariant itself, and it is exact.
    #[test]
    fn a_remembered_bind_holds_the_allocation_its_key_names() {
        let mut pools = ResourcePools::new();
        let bound = crate::engine::exec::BoundBuffer {
            buffer: vk::Buffer::null(),
            offset: 0,
            guest_import: false,
        };

        let content = crate::engine::types::BufferContent::from(vec![3u8; 256]);
        let crate::engine::types::BufferContent::Bytes(strong) = &content else {
            unreachable!("BufferContent::from(Vec<u8>) is the Bytes arm")
        };
        let watch = std::sync::Arc::downgrade(strong);

        pools.note_cb_bound_buffer(super::super::CbBind::of(&content), bound);
        drop(content);

        assert!(
            watch.upgrade().is_some(),
            "the map dropped the allocation its key is the address of — that address \
             is now free for the next bind of the same length to be given, and this \
             entry would answer for it"
        );

        // And the holding ends with the entry, or the map is a leak instead.
        pools.recycle_staging();
        assert!(
            watch.upgrade().is_none(),
            "a cleared map still holds a bind's bytes"
        );
    }

    /// The write ledger's own half, and the reason it is a ledger rather than a
    /// blocking call: several windows landed in one fence pass settle together.
    /// A rail that took the debt per window would be the per-window fence this
    /// change removed, wearing a different name.
    #[test]
    fn a_submission_seal_takes_exactly_the_guest_write_tokens_recorded_into_it() {
        let mut pools = ResourcePools::new();
        let identity = TargetIdentity::Surface {
            id: 1,
            width: 16,
            height: 16,
            generation: 0,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        pools.note_guest_write_recorded(
            GuestWriteSource::ResidentTarget(&identity),
            Some(write_token(11)),
        );
        pools.note_guest_write_recorded(
            GuestWriteSource::ResidentTarget(&identity),
            Some(write_token(12)),
        );
        let cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        assert!(pools.guest_write_tokens_live.is_empty());
        assert_eq!(
            cleanup.guest_write_tokens,
            vec![write_token(11), write_token(12)]
        );
    }

    #[test]
    fn repeated_writes_to_one_page_set_share_the_command_buffers_retirement() {
        let session = crate::engine::SessionHandle::new(SessionId(90_001));
        let _scope = crate::engine::enter_session(&session);
        crate::engine::clear_guest_write_pages();
        let pages =
            reims_vgpu_memory::GuestWritePages::new(&[0x1000, 0x2000]).expect("non-empty page set");
        let mut pools = ResourcePools::new();

        pools.note_guest_write_pages_recorded(GuestWriteSource::ImportedBuffer, &pages);
        pools.note_guest_write_pages_recorded(GuestWriteSource::ImportedBuffer, &pages);

        assert_eq!(
            pools.guest_write_tokens_live.len(),
            1,
            "one command buffer has one fence and therefore one lifetime for this page set"
        );
        let cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        assert_eq!(cleanup.guest_write_tokens.len(), 1);
        crate::engine::retire_guest_write_pages(&cleanup.guest_write_tokens);
        assert!(!crate::engine::guest_writes_outstanding());
    }

    /// A submitted-but-unsettled copy reads its resident's image, so the ledger
    /// takes a pin that outlives the call which issued the copy.
    ///
    /// The pin is taken by the ledger rather than handed to it. This test used
    /// to pin first and assert the count stayed at one across the handoff, which
    /// passed just as well when no caller pinned at all — and that is what
    /// happened: `storage_flush` held every product pinner, its removal left the
    /// writeback recording the debt with no pin, and the settle then released
    /// one that was never taken.
    ///
    /// Device-free: `quiesce_guest_writes` needs a `DeviceContext` for the wait,
    /// so this drives the two halves it composes — recording the copy's pin and
    /// transferring that pin to the submission that releases it.
    #[test]
    fn a_submitted_writeback_holds_its_residents_pin_until_its_slot_retires() {
        let mut pools = ResourcePools::new();
        let identity = TargetIdentity::Surface {
            id: 7,
            width: 16,
            height: 16,
            generation: 3,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        pools.registry.insert(
            identity.clone(),
            crate::engine::pools::images_and_registry::pin_count_tests::ready_slot(),
        );
        assert_eq!(pools.registry[&identity].pin_count, 0, "nobody has pinned");

        // Recording the copy is what pins: the caller holds nothing.
        pools.note_guest_write_recorded(GuestWriteSource::ResidentTarget(&identity), None);
        assert_eq!(
            pools.registry[&identity].pin_count, 1,
            "the submitted copy must hold the image against reclaim"
        );

        // A second window on the same resident inside one pass takes its own.
        pools.note_guest_write_recorded(GuestWriteSource::ResidentTarget(&identity), None);
        assert_eq!(pools.registry[&identity].pin_count, 2);

        // Sealing transfers the pins to the exact submission that references
        // them. Model that slot's fence retirement without a Vulkan device.
        let mut cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        assert!(pools.resident_pins_live.is_empty());
        for held in cleanup.unpin_residents.drain(..) {
            pools.pin_resident_target(&held, false);
        }
        assert_eq!(
            pools.registry[&identity].pin_count, 0,
            "every pin the ledger took is released exactly once"
        );
        assert!(
            cleanup.unpin_residents.is_empty(),
            "a retired pin must not be released a second time"
        );
    }

    /// A ring-owned source records the same debt and takes no pin.
    ///
    /// This is the compute storage-image arm. Its image is a transient slot
    /// sealed into the submission's own ring entry, which cannot be recycled
    /// until the fence retires — so the lifetime is already held and a pin here
    /// would be a second owner of it, with a release to get right for no gain.
    ///
    /// Both halves matter and they fail in opposite directions. Pinning would
    /// leak: nothing releases a pin the ledger did not record in
    /// `resident_pins_live`, so the image would never be reclaimable again.
    /// Skipping the *debt* would be the correctness bug — the whole ordering
    /// argument for a submitted-not-waited copy is that the session's guest-write debt
    /// removes `StampOrder::CpuReady` from the stamp's answers, so a guest told
    /// the dispatch finished would read its pages before the copy landed.
    #[test]
    fn a_ring_owned_writeback_records_the_debt_without_pinning_anything() {
        let mut pools = ResourcePools::new();
        let identity = TargetIdentity::Surface {
            id: 9,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        pools.registry.insert(
            identity.clone(),
            crate::engine::pools::images_and_registry::pin_count_tests::ready_slot(),
        );

        pools.note_guest_write_recorded(GuestWriteSource::RingEntry, Some(write_token(21)));
        assert_eq!(
            pools.registry[&identity].pin_count, 0,
            "a ring-owned source pins no resident"
        );
        assert!(
            pools.resident_pins_live.is_empty(),
            "and leaves the ledger nothing to release"
        );
        let cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        assert_eq!(
            cleanup.guest_write_tokens,
            vec![write_token(21)],
            "the write's visibility token follows the ring-owned image to its fence"
        );
        assert!(
            cleanup.unpin_residents.is_empty(),
            "so its submission carries no unpin either"
        );
    }

    /// A compute-storage resident's writeback pins in the *other* registry, and
    /// the ring slot's cleanup releases it.
    ///
    /// The arm that lets a registered resident copy straight into guest pages. It
    /// is not the ring-owned case above: a resident is popped out of the ring's
    /// live set when it is acquired, so the submission's own cleanup does not own
    /// its image and the pin is the only thing holding it.
    ///
    /// What the pin closes is narrower than it looks, and it is worth being exact
    /// about because the wrong reading is what made this arm read back for a whole
    /// boot. Both reclaim paths skip a resident whose `gpu_only_content` holds,
    /// and every executed dispatch sets it — so the *reclaim* was never the
    /// hazard. `acquire_resident_storage_image` is: it destroys the held image
    /// when the same identity arrives at a different shape, and
    /// `compute_rekey_refusal` is what stops it, reading `pinned` and nothing
    /// else. Between this submit and its fence, that refusal is the whole defence.
    ///
    /// Fails without the ledger in either direction: no pin and a re-keying
    /// dispatch frees an image the queue is still reading; no release and the
    /// identity refuses every later re-shape for the life of the guest.
    #[test]
    fn a_compute_resident_writeback_pins_in_its_own_registry_until_the_slot_retires() {
        let mut pools = ResourcePools::new();
        let id = admit_compute_resident(&mut pools, 1, 1_000, false);

        pools.note_guest_write_recorded(GuestWriteSource::ResidentStorage(&id), None);
        assert!(
            pools.compute_storage_registry[&id].pinned,
            "the copy is submitted and not waited, so the image must be held"
        );
        assert_eq!(
            pools.compute_write_pins_live,
            vec![id],
            "and the ledger records the pin it actually took"
        );

        let mut cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        assert!(
            pools.compute_write_pins_live.is_empty(),
            "sealing hands the pin to the slot rather than copying it"
        );
        assert_eq!(cleanup.unpin_compute_residents, vec![id]);

        // What `drain_cleanup` does with them, once the fence has signalled.
        for identity in cleanup.unpin_compute_residents.drain(..) {
            pools.pin_resident_storage(&identity, false);
        }
        assert!(
            !pools.compute_storage_registry[&id].pinned,
            "and the fence is what ends the hold"
        );
    }

    /// An identity with no slot records the debt and no pin.
    ///
    /// The pin is taken in the registry that owns the image, so an identity that
    /// is not registered has no image for anything to remove and nothing to hold.
    /// The ledger must not record a release for a pin it could not take: a stray
    /// unpin would land on a future resident admitted under the same key.
    #[test]
    fn a_compute_writeback_against_an_unregistered_identity_records_no_pin() {
        let mut pools = ResourcePools::new();
        let absent = ComputeStorageResidencyKey::linear(
            reims_vgpu_protocol::ResourceId::new(77, 1),
            0,
            0,
            0,
            8,
            8,
            0,
        );

        pools.note_guest_write_recorded(GuestWriteSource::ResidentStorage(&absent), None);
        assert!(pools.compute_write_pins_live.is_empty());
        assert!(pools
            .seal_entry(Vec::new(), Vec::new())
            .cleanup
            .unpin_compute_residents
            .is_empty());
    }

    /// Another holder's pin survives a writeback submission's retirement.
    ///
    /// The hazard the ledger's own pin closes, and the one its guard could not
    /// report: an unpin at zero is logged, but an unpin that lands on *someone
    /// else's* count is silent and leaves them reading an image the reclaim may
    /// now take. The host window's present pin is the live second holder.
    #[test]
    fn a_writeback_retirement_does_not_release_another_holders_pin() {
        let mut pools = ResourcePools::new();
        let identity = TargetIdentity::Surface {
            id: 9,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        pools.registry.insert(
            identity.clone(),
            crate::engine::pools::images_and_registry::pin_count_tests::ready_slot(),
        );
        // The window pins for its present blit and still holds it.
        assert!(
            pools.pin_resident_target(&identity, true),
            "the window's pin"
        );

        pools.note_guest_write_recorded(GuestWriteSource::ResidentTarget(&identity), None);
        assert_eq!(pools.registry[&identity].pin_count, 2);

        let mut cleanup = pools.seal_entry(Vec::new(), Vec::new()).cleanup;
        for held in cleanup.unpin_residents.drain(..) {
            pools.pin_resident_target(&held, false);
        }
        assert_eq!(
            pools.registry[&identity].pin_count, 1,
            "the window is still reading this image and must still hold it"
        );
    }

    /// A resident the ledger cannot pin records the debt and queues no release.
    ///
    /// The other half of the same rule: pushing an identity the pin refused is
    /// how the settle came to release pins that were never taken.
    #[test]
    fn a_writeback_on_an_unpinnable_resident_queues_no_release() {
        let mut pools = ResourcePools::new();
        let identity = TargetIdentity::Surface {
            id: 11,
            width: 16,
            height: 16,
            generation: 0,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        // No slot at all: nothing to pin, and nothing for a reclaim to take.
        pools.note_guest_write_recorded(
            GuestWriteSource::ResidentTarget(&identity),
            Some(write_token(31)),
        );
        assert!(
            pools.resident_pins_live.is_empty(),
            "no pin was taken, so none may be released"
        );
        assert_eq!(pools.guest_write_tokens_live.len(), 1);
    }

    /// A dispose site has already unlinked the handle, so only the entries
    /// recording or in flight *at that instant* can still reference it. The
    /// handle is therefore released when those slots retire — not when the
    /// whole ring goes idle. Device-free: drives the mask decision only.
    #[test]
    fn a_disposed_handle_waits_on_the_slots_open_when_it_was_disposed_and_no_others() {
        let mut pools = ResourcePools::new();
        pools.slots = (0..4).map(|_| idle_slot()).collect();
        pools.slots[0] = pending_slot();
        pools.slots[1] = pending_slot();

        let waiting = pools.open_slot_mask();
        assert_eq!(waiting, 0b0011, "slots 0 and 1 are in flight");
        pools.graveyard.push((
            waiting,
            DeferredHandle::Framebuffer(vk::Framebuffer::null()),
        ));

        // A later entry claims slot 2. It began after the dispose, so it cannot
        // have recorded a reference to the handle, and the handle must not
        // start waiting on it.
        pools.slots[2] = pending_slot();
        assert!(
            pools.take_released_graveyard(1 << 2).is_empty(),
            "slot 2 retiring says nothing about a handle disposed before it began"
        );
        assert!(
            pools.take_released_graveyard(1 << 0).is_empty(),
            "slot 1 could still be reading it"
        );
        assert_eq!(
            pools.take_released_graveyard(1 << 1).len(),
            1,
            "both witnesses retired: the handle is free"
        );
        assert!(pools.graveyard.is_empty());
    }

    /// Installing and taking the owned batch publish the exact state the drain
    /// tail uses to decide whether it has submission work.
    #[test]
    fn batch_ownership_publishes_and_clears_the_tail_gate() {
        let mut pools = ResourcePools::new();
        assert!(!crate::engine::batch_open_for_current_session());
        pools.install_open_batch(OpenBatch {
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            target: BatchTarget {
                identity: TargetIdentity::Anonymous { slot: 0 },
                width: 16,
                height: 16,
                bgra: false,
            },
            draws: 1,
            dsets: Vec::new(),
        });
        assert!(crate::engine::batch_open_for_current_session());
        assert!(pools.take_open_batch().is_some());
        assert!(!crate::engine::batch_open_for_current_session());
    }

    /// The slot mask and the tail publication describe the same owned batch,
    /// but answer different readers: reclamation under the engine lock and the
    /// drain boundary before it takes that lock.
    #[test]
    fn the_open_batch_slot_counts_as_open_even_though_it_owes_no_cleanup() {
        let mut pools = ResourcePools::new();
        pools.slots = (0..4).map(|_| idle_slot()).collect();
        pools.cur = 3;
        assert_eq!(pools.open_slot_mask(), 0, "idle ring, no batch");

        pools.open_batch = Some(OpenBatch {
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            target: BatchTarget {
                identity: TargetIdentity::Anonymous { slot: 0 },
                width: 16,
                height: 16,
                bgra: false,
            },
            draws: 1,
            dsets: Vec::new(),
        });
        assert_eq!(pools.open_slot_mask(), 1 << 3, "the batch's own slot");
    }

    /// The target decides a join only on the narrowed arm, and each of the three
    /// refusals answers with its own name.
    ///
    /// The wide arm is the one a workload lives on: a driven macos-13 hammer
    /// boot refused 26.1 % of all draws for a target switch alone, against a
    /// batch that was recording and had room. Without this the two arms are
    /// indistinguishable except by a live boot, and `BatchFit::OtherTarget`
    /// would be reachable only through the environment.
    #[test]
    fn only_the_narrowed_arm_asks_what_the_open_batch_was_drawing_into() {
        let target = |slot: u64| BatchTarget {
            identity: TargetIdentity::Anonymous { slot },
            width: 16,
            height: 16,
            bgra: false,
        };
        let mut pools = ResourcePools::new();
        pools.slots = (0..4).map(|_| idle_slot()).collect();

        assert!(
            matches!(pools.batch_fit(&target(0), false), BatchFit::None),
            "nothing recording"
        );

        pools.open_batch = Some(OpenBatch {
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            target: target(0),
            draws: 1,
            dsets: Vec::new(),
        });
        assert!(
            matches!(pools.batch_fit(&target(0), true), BatchFit::Open(..)),
            "its own target fits on either arm"
        );
        assert!(
            matches!(pools.batch_fit(&target(1), true), BatchFit::OtherTarget),
            "the narrowed arm refuses a second surface"
        );
        assert!(
            matches!(pools.batch_fit(&target(1), false), BatchFit::Open(..)),
            "the default arm admits it — every draw opens and ends its own pass"
        );

        // Fullness outranks the target on both arms: the cap is what keeps the
        // GPU fed, and a full batch has to be flushed whoever asks.
        pools.open_batch.as_mut().expect("open").draws = BATCH_MAX_DRAWS;
        for narrow in [false, true] {
            assert!(
                matches!(pools.batch_fit(&target(0), narrow), BatchFit::Full),
                "narrow={narrow}"
            );
        }
    }

    /// What `GRAVEYARD_FORCE_DRAIN` used to backstop: a pure-async streak never
    /// lets the ring reach idle, so under a global drain the graveyard grew
    /// until a forced full quiesce cut it down. Per-slot masks make that
    /// structural — every slot a handle waits on retires within one ring wrap,
    /// so the population is bounded by the disposes of one wrap with no valve.
    #[test]
    fn a_ring_that_never_goes_idle_still_drains_the_graveyard() {
        let mut pools = ResourcePools::new();
        let depth = 4;
        pools.slots = (0..depth).map(|_| pending_slot()).collect();

        let mut released = 0;
        let mut peak = 0;
        for step in 0..64 {
            // Every slot but the one about to retire stays in flight, so the
            // ring is never idle and `open_slot_mask()` is never zero.
            let waiting = pools.open_slot_mask();
            assert_ne!(waiting, 0, "step {step}: ring must stay busy");
            pools.graveyard.push((
                waiting,
                DeferredHandle::Framebuffer(vk::Framebuffer::null()),
            ));
            peak = peak.max(pools.graveyard.len());

            let index = step % depth;
            pools.slots[index].pending = None;
            released += pools.take_released_graveyard(1 << index).len();
            pools.slots[index] = pending_slot();
        }

        // A handle disposed at step `s` waits on all `depth` slots and the
        // retire at step `s` clears the first of them, so it frees at step
        // `s + depth - 1`. Only the final `depth - 1` disposes are still parked.
        assert_eq!(
            released,
            64 - (depth - 1),
            "everything older than one ring wrap freed without a forced quiesce"
        );
        assert!(
            peak <= depth,
            "outstanding population is bounded by the ring depth, got {peak}"
        );
    }
}

/// Copy `src` into `dst` with the R and B channels of every whole RGBA8 pixel
/// exchanged, writing each destination byte exactly once.
///
/// Split out of [`ResourcePools::write_staging_swap_rb`] so the transformation has a
/// test: that method's destination is a mapped Vulkan allocation, so nothing
/// about it is reachable from a unit test, and the exchange is the part that can
/// be wrong. `dst` is written and never read, which is what keeps the
/// write-combined case off a read-modify-write path — see the caller for the
/// measurement that motivated writing whole pixels instead of single bytes.
///
/// A trailing partial pixel is copied through unexchanged. Bytes of `dst` past
/// `src.len()` are left alone; the caller owns whatever the mapped span needs
/// beyond the source.
fn exchange_rb_into(src: &[u8], dst: &mut [u8]) {
    let n = src.len().min(dst.len());
    let whole = n / 4 * 4;
    for (s, d) in src[..whole]
        .chunks_exact(4)
        .zip(dst[..whole].chunks_exact_mut(4))
    {
        d.copy_from_slice(&[s[2], s[1], s[0], s[3]]);
    }
    dst[whole..n].copy_from_slice(&src[whole..n]);
}

#[cfg(test)]
mod exchange_rb_tests {
    use super::exchange_rb_into;

    /// The exchange is what the caller's correctness rests on: get it wrong and
    /// every seeded LOAD composites with red and blue swapped.
    #[test]
    fn whole_pixels_swap_red_and_blue_and_leave_green_and_alpha() {
        let src = [1u8, 2, 3, 4, 250, 251, 252, 253];
        let mut dst = [0u8; 8];
        exchange_rb_into(&src, &mut dst);
        assert_eq!(dst, [3, 2, 1, 4, 252, 251, 250, 253]);
    }

    /// It is an involution, which is why one function serves both directions:
    /// a semantic-RGBA seed into a BGRA attachment and a guest-scanout-order
    /// seed into an RGBA target.
    #[test]
    fn exchanging_twice_is_the_identity() {
        let src: Vec<u8> = (0u8..=255).collect();
        let mut once = vec![0u8; src.len()];
        let mut twice = vec![0u8; src.len()];
        exchange_rb_into(&src, &mut once);
        exchange_rb_into(&once, &mut twice);
        assert_eq!(twice, src);
    }

    /// A trailing partial pixel goes through unexchanged rather than being
    /// dropped, so the mapped span is fully defined even for a length the
    /// caller says cannot occur.
    #[test]
    fn a_trailing_partial_pixel_is_copied_through() {
        let src = [9u8, 8, 7, 6, 5, 4];
        let mut dst = [0u8; 6];
        exchange_rb_into(&src, &mut dst);
        assert_eq!(dst, [7, 8, 9, 6, 5, 4]);
    }

    /// The copy is bounded by the shorter of the two, so a destination shorter
    /// than the source cannot walk off the end of the mapped span.
    #[test]
    fn a_short_destination_bounds_the_copy() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = [0u8; 4];
        exchange_rb_into(&src, &mut dst);
        assert_eq!(dst, [3, 2, 1, 4]);
    }
}

/// The gather pool's no-aliasing property, which is what the guest-page
/// writeback's detiling buffer now rests on.
///
/// These exercise [`ResourcePools::take_free_gather`] rather than
/// `acquire_guest_gather`, because the miss path allocates and needs a device
/// while the property under test does not: a slot is *removed* from the free
/// list when it is handed out and is recorded live, so nothing can hand the
/// same buffer to two submissions that have not been separated by a fence.
///
/// # What these do not prove
///
/// **They would not have caught the bug they were written for**, and saying so
/// is the point of this paragraph. The fault was in the singleton the writeback
/// used *instead* of this pool; the pool itself was always correct. What they
/// pin is the property the writeback now depends on, so a later change that
/// starts recycling gather slots without a fence — the same mistake one level
/// down — fails here rather than on a guest's desktop.
///
/// Reproducing the original fault needs a GPU, a guest, and a writeback of a
/// larger window landing before an earlier one's fence retires. Nothing in this
/// crate reaches that. The evidence for it is an `NVRM: Xid 31` MMU fault on
/// the copy engine, and the argument is the pair of comments quoted on
/// [`ResourcePools::take_free_gather`].
#[cfg(test)]
mod gather_slots_do_not_alias {
    use super::*;
    use ash::vk::Handle;

    fn slot(raw: u64, size: u64) -> BufferSlot {
        BufferSlot {
            buffer: vk::Buffer::from_raw(raw),
            memory: vk::DeviceMemory::from_raw(raw),
            size,
            mapped: 0,
            backing: BufferBacking::Dedicated,
            coherent: false,
            cached: false,
        }
    }

    /// Two acquires with no fence between them resolve to two different
    /// buffers.
    ///
    /// This is the regression. The writeback rail used to detile through a
    /// singleton (`guest_scratch`) that answered the second acquire with the
    /// *same* buffer whenever it was already large enough, so a second frame's
    /// detile wrote the buffer the first frame's scatter was still reading.
    #[test]
    fn a_second_acquire_cannot_name_the_first_ones_buffer() {
        let mut pools = ResourcePools::new();
        let bucket = ResourcePools::bucket(4096);
        pools
            .gather_free
            .insert(bucket, vec![slot(1, bucket), slot(2, bucket)]);

        let first = pools.take_free_gather(bucket).expect("a free slot");
        let second = pools.take_free_gather(bucket).expect("a second free slot");

        assert_ne!(
            first.buffer, second.buffer,
            "two live gather slots must not be one buffer"
        );
        assert_eq!(pools.gather_live.len(), 2, "both must be recorded live");
    }

    /// A slot that has been handed out is gone from the free list, so it cannot
    /// be found again until the ring puts it back after its fence.
    ///
    /// The singleton had no such list: it was grown in place and destroyed
    /// outright on a grow, which is the shape that freed memory underneath a
    /// submitted copy.
    #[test]
    fn an_acquired_slot_is_no_longer_free() {
        let mut pools = ResourcePools::new();
        let bucket = ResourcePools::bucket(4096);
        pools.gather_free.insert(bucket, vec![slot(7, bucket)]);

        let taken = pools.take_free_gather(bucket).expect("a free slot");
        assert_eq!(taken.buffer, vk::Buffer::from_raw(7));
        assert!(
            pools.take_free_gather(bucket).is_none(),
            "the only slot was handed out; the list must be empty"
        );
        assert!(
            !pools.gather_live.is_empty(),
            "and the live list is what keeps it alive until its fence retires"
        );
    }
}

/// The guest-scatter descriptor pool's no-aliasing property, which is what makes
/// recycling a set cheaper than allocating one *and* correct.
///
/// These exercise [`ResourcePools::take_free_scatter_dset`] and
/// [`ResourcePools::recycle_scatter_dsets`] rather than
/// `alloc_scatter_descriptor_set`, because the miss path allocates and needs a
/// device while the property under test does not — the same split, for the same
/// reason, as [`gather_slots_do_not_alias`].
///
/// The property: a set is *removed* from the free list when it is handed out,
/// and only a retired entry puts it back. A dispatch's bindings are written into
/// its set immediately before it is recorded, so two dispatches handed one set
/// would have the second's `write_set` silently retarget the first — a gather
/// reading another window's runs, which is wrong pixels rather than slow ones.
#[cfg(test)]
mod scatter_descriptor_sets_do_not_alias {
    use super::*;
    use ash::vk::Handle;

    fn pair(raw: u64) -> (vk::DescriptorSet, vk::DescriptorPool) {
        (
            vk::DescriptorSet::from_raw(raw),
            vk::DescriptorPool::from_raw(1),
        )
    }

    /// Two takes with no retire between them cannot resolve to one set.
    #[test]
    fn two_takes_with_no_retire_between_them_give_two_sets() {
        let mut pools = ResourcePools::new();
        pools.recycle_scatter_dsets(&mut vec![pair(1), pair(2)]);
        let first = pools.take_free_scatter_dset().expect("two were recycled");
        let second = pools.take_free_scatter_dset().expect("two were recycled");
        assert_ne!(first.0, second.0);
        assert!(
            pools.take_free_scatter_dset().is_none(),
            "the list held exactly what was recycled into it"
        );
    }

    /// A retired set comes back, which is the whole point: the steady state must
    /// stop calling `vkAllocateDescriptorSets`.
    #[test]
    fn a_retired_set_is_handed_out_again() {
        let mut pools = ResourcePools::new();
        pools.recycle_scatter_dsets(&mut vec![pair(7)]);
        let taken = pools.take_free_scatter_dset().expect("one was recycled");
        assert_eq!(taken.0.as_raw(), 7);
        assert!(pools.take_free_scatter_dset().is_none());
        pools.recycle_scatter_dsets(&mut vec![taken]);
        assert_eq!(
            pools
                .take_free_scatter_dset()
                .expect("it was recycled again")
                .0
                .as_raw(),
            7
        );
    }

    /// A fresh device has nothing to recycle, so the first dispatch of a boot
    /// allocates rather than reading an empty `pop` as a usable handle.
    #[test]
    fn a_fresh_pool_has_nothing_to_hand_out() {
        let mut pools = ResourcePools::new();
        assert!(pools.take_free_scatter_dset().is_none());
    }

    /// Recycling drains the caller's vector, so a `PendingGpuCleanup` cannot be
    /// drained twice into the free list and hand the same set to two live
    /// dispatches.
    #[test]
    fn recycling_takes_the_sets_out_of_the_entry_that_owed_them() {
        let mut pools = ResourcePools::new();
        let mut owed = vec![pair(3), pair(4)];
        pools.recycle_scatter_dsets(&mut owed);
        assert!(owed.is_empty());
        pools.recycle_scatter_dsets(&mut owed);
        assert!(pools.take_free_scatter_dset().is_some());
        assert!(pools.take_free_scatter_dset().is_some());
        assert!(pools.take_free_scatter_dset().is_none());
    }

    fn batch_target() -> super::BatchTarget {
        super::BatchTarget {
            identity: crate::engine::types::TargetIdentity::Surface {
                id: 56,
                width: 1024,
                height: 768,
                generation: 1,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            },
            width: 1024,
            height: 768,
            bgra: true,
        }
    }

    /// A batch fills at the pool's own cap, not at the compiled constant.
    ///
    /// This is the wiring, and it is the half that a test of the environment
    /// parse alone cannot reach: [`super::BATCH_MAX_DRAWS`] was read directly by
    /// `batch_fit`, so a narrowed cap that never arrived here would leave the
    /// device batching thirty-two draws while the boot line reported one. That
    /// failure is silent in exactly the direction that matters — an arm of a
    /// bisect that did not take still produces a well-formed driven boot.
    #[test]
    fn a_batch_fills_at_the_pools_cap_and_not_at_the_compiled_constant() {
        let target = batch_target();
        let mut pools = ResourcePools::new();
        pools.batch_max_draws = 4;
        pools.open_batch = Some(super::OpenBatch {
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            target: target.clone(),
            draws: 3,
            dsets: Vec::new(),
        });
        assert!(matches!(
            pools.batch_fit(&target, false),
            super::BatchFit::Open(..)
        ));
        if let Some(b) = pools.open_batch.as_mut() {
            b.draws = 4;
        }
        assert!(matches!(
            pools.batch_fit(&target, false),
            super::BatchFit::Full
        ));
    }

    /// Before a pool sees a device it carries the largest supported capacity;
    /// `ensure_init` replaces this with that device's topology policy.
    #[test]
    fn a_pool_that_was_told_nothing_carries_the_compiled_cap() {
        assert_eq!(
            ResourcePools::new().batch_max_draws,
            super::BATCH_MAX_DRAWS,
            "a device-free pool carries enough capacity for either topology"
        );
    }

    /// Topology changes only submission granularity: both arms execute the
    /// same draws, while a unified host preserves more of the serialized
    /// command-buffer unit before the Vulkan scheduling bound cuts it.
    #[test]
    fn batch_defaults_follow_structural_memory_topology() {
        use crate::memory::MemoryTopology;

        assert_eq!(
            super::batch_default_draws(MemoryTopology::Discrete),
            super::DISCRETE_BATCH_MAX_DRAWS
        );
        assert_eq!(
            super::batch_default_draws(MemoryTopology::Unified),
            super::BATCH_MAX_DRAWS
        );
        assert!(
            super::batch_default_draws(MemoryTopology::Discrete)
                < super::batch_default_draws(MemoryTopology::Unified)
        );
    }

    /// Installing device policy resets the snapshot pool before any Vulkan
    /// object can enter it, without installing a second capacity policy.
    #[test]
    fn configuring_batch_capacity_resets_the_snapshot_pool() {
        let mut pools = ResourcePools::new();
        pools.configure_batch_capacity(super::DISCRETE_BATCH_MAX_DRAWS);

        assert_eq!(pools.batch_max_draws, super::DISCRETE_BATCH_MAX_DRAWS);
        assert_eq!(pools.attachment_snapshot_free.len(), 0);
    }
}

#[cfg(test)]
mod imported_guest_visibility_tests {
    use super::*;
    use crate::engine::exec::ImportedGuestVisibility;

    #[test]
    fn one_host_dependency_covers_one_guest_operation_until_a_gpu_write() {
        let mut pools = ResourcePools::new();
        let cb = vk::CommandBuffer::from_raw(1);

        pools.begin_guest_operation(cb);
        assert_eq!(
            pools.imported_guest_barrier(cb, || ImportedGuestVisibility::HostOnly),
            Some(ImportedGuestVisibility::HostOnly)
        );
        assert_eq!(
            pools.imported_guest_barrier(cb, || { panic!("visibility is already established") }),
            None
        );

        pools.note_cb_guest_write();
        assert_eq!(
            pools.imported_guest_barrier(cb, || ImportedGuestVisibility::HostOnly),
            None,
            "a disjoint GPU write does not invalidate host visibility"
        );
        assert_eq!(
            pools.imported_guest_barrier(cb, || ImportedGuestVisibility::GpuOverlap),
            Some(ImportedGuestVisibility::GpuOverlap)
        );
        assert_eq!(
            pools
                .imported_guest_barrier(cb, || { panic!("GPU visibility is already established") }),
            None
        );

        pools.begin_guest_operation(cb);
        assert_eq!(
            pools.imported_guest_barrier(cb, || ImportedGuestVisibility::HostOnly),
            Some(ImportedGuestVisibility::HostOnly),
            "a later guest operation may follow an unobserved guest CPU write"
        );
    }

    #[test]
    fn an_operation_boundary_preserves_unconsumed_gpu_write_debt() {
        let mut pools = ResourcePools::new();
        let cb = vk::CommandBuffer::from_raw(1);

        pools.begin_guest_operation(cb);
        pools.note_cb_guest_write();
        pools.begin_guest_operation(cb);
        assert_eq!(
            pools.imported_guest_barrier(cb, || ImportedGuestVisibility::GpuOverlap),
            Some(ImportedGuestVisibility::GpuOverlap)
        );
    }

    #[test]
    fn a_new_command_buffer_owes_its_own_host_dependency() {
        let mut pools = ResourcePools::new();
        let first = vk::CommandBuffer::from_raw(1);
        let second = vk::CommandBuffer::from_raw(2);

        pools.begin_guest_operation(first);
        assert!(pools
            .imported_guest_barrier(first, || ImportedGuestVisibility::HostOnly)
            .is_some());
        assert!(pools
            .imported_guest_barrier(first, || { panic!("visibility is already established") })
            .is_none());
        pools.begin_guest_operation(second);
        assert!(pools
            .imported_guest_barrier(second, || ImportedGuestVisibility::HostOnly)
            .is_some());
    }

    #[test]
    fn render_encoder_continuations_share_visibility_only_in_one_command_buffer() {
        let mut pools = ResourcePools::new();
        let first = vk::CommandBuffer::from_raw(1);
        let second = vk::CommandBuffer::from_raw(2);

        pools.enter_render_encoder_record(first, false);
        assert_eq!(
            pools.imported_guest_barrier(first, || ImportedGuestVisibility::HostOnly),
            Some(ImportedGuestVisibility::HostOnly)
        );

        pools.enter_render_encoder_record(first, true);
        assert_eq!(
            pools.imported_guest_barrier(first, || {
                panic!("a draw call is not a host publication boundary")
            }),
            None
        );

        pools.enter_render_encoder_record(second, true);
        assert_eq!(
            pools.imported_guest_barrier(second, || ImportedGuestVisibility::HostOnly),
            Some(ImportedGuestVisibility::HostOnly),
            "a continuation moved by an explicit boundary owes visibility in its new CB"
        );
    }
}
