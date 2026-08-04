impl ResourcePools {
    pub(crate) fn guest_reset_counts(&self) -> (usize, usize, usize, usize) {
        let sampled = self.sampled_live.len() + self.sampled_free.len() + self.sampled_cache.len();
        let storage =
            self.storage_image_live.len() + self.storage_image_free.len() + self.compute_storage_registry.len();
        (self.registry.len(), self.targets.len(), sampled, storage)
    }

    pub(crate) fn new() -> Self {
        Self {
            depth_stencil_keep: None,
            staging_free: HashMap::new(),
            staging_live: Vec::new(),
            staging_hits: 0,
            staging_misses: 0,
            staging_miss_bins: [0; STAGING_BUCKET_BINS],
            staging_miss_us_bins: [0; STAGING_BUCKET_BINS],
            settled_staging_mark: 0,
            targets: HashMap::new(),
            target_order: Vec::new(),
            readback_free: HashMap::new(),
            readback_live: None,
            readback_multi_live: Vec::new(),
            readback_leased: Vec::new(),
            sampled_free: FreePool::new(SAMPLED_FREE_CAP_PER_KEY, SAMPLED_FREE_CAP_TOTAL),
            sampled_live: Vec::new(),
            sampled_cache: Vec::new(),
            sampled_cache_bytes: 0,
            storage_image_free: FreePool::new(
                STORAGE_IMAGE_FREE_CAP_PER_KEY,
                STORAGE_IMAGE_FREE_CAP_TOTAL,
            ),
            storage_image_live: Vec::new(),
            compute_storage_registry: HashMap::new(),
            compute_storage_order: VecDeque::new(),
            registry: HashMap::new(),
            registry_order: VecDeque::new(),
            idle_clock_ms: 0,
            last_drain_ms: 0,
            settled_drain_passes: 0,
            cmd_pool: vk::CommandPool::null(),
            desc_arena: DescriptorArena::empty(),
            slots: Vec::new(),
            cur: 0,
            in_flight: 0,
            graveyard: Vec::new(),
            target_free: FreePool::new(TARGET_FREE_CAP_PER_KEY, TARGET_FREE_CAP_TOTAL),
            open_batch: None,
            slab: super::slab::SlabPool::new(),
            host_slab: super::host_slab::HostSlabPool::new(),
            initialized: false,
        }
    }
    /// Advance the wall-clock idle-drain clock to `now_ms`, keep the presented
    /// target alive (`display`), and — if an idle pass is due — select a bounded
    /// set of aged non-pinned residents to reclaim. Pure — no GPU work — so the
    /// aging + throttle logic is unit-testable without a device.
    ///
    /// Returns `None` when no pass is due (before the first age window, or inside
    /// the throttle interval); `Some(victims)` (possibly empty) when a pass fired.
    /// The `Some`/`None` distinction is load-bearing: the caller also trims the
    /// recycle pools on a fired pass, and an empty registry with a full recycle
    /// pool must still trim.
    ///
    /// `display` is the currently-presented target's identity: it is resolved via
    /// `registry_get` (a read that does not stamp), so at true idle — where the
    /// poll heartbeat still ticks this clock but no publish re-touches the frame —
    /// it would otherwise age out from under the display. Stamping it here every
    /// call makes it un-ageable while it is on screen.
    fn plan_idle_drain(
        &mut self,
        now_ms: u64,
        display: Option<&TargetIdentity>,
    ) -> Option<Vec<TargetIdentity>> {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let now = self.idle_clock_ms;
        if let Some(id) = display {
            if let Some(slot) = self.registry.get_mut(id) {
                slot.last_touch_ms = now;
            }
        }
        if now < IDLE_TARGET_AGE_MS
            || now.saturating_sub(self.last_drain_ms) < IDLE_DRAIN_INTERVAL_MS
        {
            return None;
        }
        self.last_drain_ms = now;
        let cutoff = now - IDLE_TARGET_AGE_MS;
        let mut victims = Vec::new();
        for k in &self.registry_order {
            if victims.len() >= IDLE_TARGET_DRAIN_MAX_PER_CALL {
                break;
            }
            if self
                .registry
                .get(k)
                .is_some_and(|s| s.pin_count == 0 && s.last_touch_ms <= cutoff)
            {
                victims.push(k.clone());
            }
        }
        Some(victims)
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
    unsafe fn trim_recycle_pools(
        &mut self,
        device: &ash::Device,
        max: usize,
        trim_buffers: bool,
    ) -> usize {
        let mut trimmed = 0;
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
                release_buffer_slot(device, &mut self.host_slab, slot);
                buf_trimmed += 1;
            }
            while buf_trimmed < max {
                let Some(slot) = pop_largest_pool_entry(&mut self.readback_free) else {
                    break;
                };
                release_buffer_slot(device, &mut self.host_slab, slot);
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
            self.host_slab
                .trim_empty_blocks(device, HOST_SLAB_IDLE_KEEP_EMPTY);
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
        trimmed + buf_trimmed
    }

    /// Reclaim up to `max` sampled-cache entries whose last use is at least
    /// `IDLE_TARGET_AGE_MS` behind the idle clock (`cutoff`) — the upload-side
    /// analogue of the resident-target idle drain. At idle a video session's
    /// retained frame textures (the ≤128 MiB `sampled_cache`) are otherwise
    /// pinned until new content evicts them by LRU, which never happens on a
    /// static desktop, so they hold VRAM for the guest lifetime (measured
    /// `sampled=64` / resident +~96 MiB frozen after a video tab closed).
    /// Age-gating means an actively-sampled entry (hit every frame, so
    /// `last_touch_ms` fresh) is never trimmed — only entries idle past the age
    /// fall out, so a live session is undisturbed (no re-upload hitch). Terminal
    /// destroy via `dispose(Image)` (not `RecycleSampled`): an idle-stale frame
    /// will not be re-sampled, so caching it in `sampled_free` would defeat the
    /// drain; freeing the image releases its slab sub-range exactly like the
    /// target drain. In-flight-safe (deferred until the referencing CB retires).
    /// Returns the count trimmed, reflected in the always-on `sampled=` census.
    unsafe fn trim_aged_sampled_cache(
        &mut self,
        device: &ash::Device,
        cutoff: u64,
        max: usize,
    ) -> usize {
        let aged = self.take_aged_sampled_slots(cutoff, max);
        let trimmed = aged.len();
        for slot in aged {
            self.dispose(
                device,
                DeferredHandle::Image {
                    image: slot.image,
                    view: slot.view,
                    memory: slot.memory,
                },
            );
        }
        trimmed
    }

    /// Device-free half of [`Self::trim_aged_sampled_cache`]: remove up to `max`
    /// sampled-cache entries whose `last_touch_ms <= cutoff`, decrement the byte
    /// accounting, and return their slots for the caller to dispose. Split out so
    /// the age selection + byte bookkeeping is unit-testable without a GPU
    /// (mirrors [`Self::plan_idle_drain`] returning victims for the caller to
    /// dispose).
    fn take_aged_sampled_slots(&mut self, cutoff: u64, max: usize) -> Vec<SampledSlot> {
        let mut taken = Vec::new();
        let mut i = 0;
        while i < self.sampled_cache.len() && taken.len() < max {
            if self.sampled_cache[i].last_touch_ms <= cutoff {
                let evicted = self.sampled_cache.remove(i);
                self.sampled_cache_bytes =
                    self.sampled_cache_bytes.saturating_sub(evicted.content_len);
                taken.push(evicted.slot);
                // `remove(i)` shifted the next entry into slot `i`; do not advance.
            } else {
                i += 1;
            }
        }
        taken
    }

    /// Reclaim up to `max` compute-storage residents whose last use is at least
    /// `IDLE_TARGET_AGE_MS` behind the idle clock (`cutoff`) — the compute analogue
    /// of the resident-target idle drain. Each resident is a standalone (non-slab)
    /// `VkDeviceMemory`, so a settled compute-heavy session (blur passes, a decode
    /// storage image) otherwise pins up to `COMPUTE_STORAGE_REGISTRY_CAP` whole
    /// allocations until an LRU eviction that never comes on a static desktop.
    /// Age-gating leaves an actively-dispatched resident (touched every pass, so
    /// `last_touch_ms` fresh) untouched and skips pinned residents (deferred
    /// writeback, only-copy-on-GPU) exactly like the LRU sweep. Terminal destroy
    /// via `dispose(Image)`; in-flight-safe (deferred until the referencing CB
    /// retires). Returns the count trimmed, reflected in the `st_res` census.
    unsafe fn trim_aged_compute_storage(
        &mut self,
        device: &ash::Device,
        cutoff: u64,
        max: usize,
    ) -> usize {
        let aged = self.take_aged_storage_residents(cutoff, max);
        let trimmed = aged.len();
        for slot in aged {
            self.dispose(
                device,
                DeferredHandle::Image {
                    image: slot.image,
                    view: slot.view,
                    memory: slot.memory,
                },
            );
        }
        trimmed
    }

    /// Device-free half of [`Self::trim_aged_compute_storage`]: remove up to `max`
    /// non-pinned compute-storage residents whose `last_touch_ms <= cutoff` from
    /// the registry and its LRU order, and return their slots for the caller to
    /// dispose. Split out so the age/pin selection is unit-testable without a GPU
    /// (mirrors [`Self::take_aged_sampled_slots`]).
    fn take_aged_storage_residents(&mut self, cutoff: u64, max: usize) -> Vec<StorageImageSlot> {
        let victims: Vec<ComputeStorageResidencyKey> = self
            .compute_storage_order
            .iter()
            .filter(|k| {
                self.compute_storage_registry
                    .get(*k)
                    .is_some_and(|r| !r.pinned && r.last_touch_ms <= cutoff)
            })
            .take(max)
            .copied()
            .collect();
        let mut taken = Vec::with_capacity(victims.len());
        for k in victims {
            self.compute_storage_order.retain(|entry| entry != &k);
            if let Some(resident) = self.compute_storage_registry.remove(&k) {
                taken.push(resident.slot);
            }
        }
        taken
    }

    /// Update the consecutive-settled-pass counter for a fired idle-drain pass
    /// that reclaimed `drained` registry victims, and return whether the
    /// HOST_VISIBLE buffer pools may be trimmed this pass.
    ///
    /// A pass is settled only if it drained no registry victim AND no staging
    /// buffer was acquired since the previous pass. The trim then needs
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive settled passes.
    ///
    /// The victim count alone was the wrong signal, and its own doc comment named
    /// the failure it was meant to prevent: "a single quiet pass mid-playback
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
    fn note_drain_settled(&mut self, drained: usize) -> bool {
        let acquires = self.staging_hits.wrapping_add(self.staging_misses);
        let uploads_ran = acquires != self.settled_staging_mark;
        self.settled_staging_mark = acquires;
        if drained == 0 && !uploads_ran {
            self.settled_drain_passes = self.settled_drain_passes.saturating_add(1);
        } else {
            self.settled_drain_passes = 0;
        }
        self.settled_drain_passes >= SETTLED_PASSES_FOR_BUFFER_TRIM
    }

    /// Advance the wall-clock idle-drain clock and reclaim a bounded number of
    /// non-pinned residents untouched for `IDLE_TARGET_AGE_MS`. Called from the
    /// poll heartbeat (ticks even when the guest stops publishing) and each
    /// publish. This is the mechanism that lets `REGISTRY_CAP` sit high enough to
    /// *absorb* a compositing burst (no eviction thrash) while still returning
    /// VRAM to the baseline working set once the burst ends — even on a static
    /// page where no further publishes occur: a burst's targets are all
    /// recently-touched (kept), and its stale leftovers age out ~2 s later. Pinned
    /// slots (deferred-write windows) are never drained — they leave via their own
    /// window lifecycle. Reclaimed images route through the same in-flight-safe
    /// recycle/dispose path as an LRU eviction; they are NOT counted as
    /// `target_evicts` (that counter is the thrash signal, and idle reclamation is
    /// not thrash). On a fired pass it also trims the recycle pools
    /// ([`Self::trim_recycle_pools`]) — at idle those are pure retained VRAM.
    /// Returns the count of registry residents drained this call, for the
    /// always-on census.
    pub(crate) unsafe fn advance_registry_touch_and_drain(
        &mut self,
        ctx: &DeviceContext,
        now_ms: u64,
        display: Option<&TargetIdentity>,
    ) {
        let Some(victims) = self.plan_idle_drain(now_ms, display) else {
            return;
        };
        let drained = victims.len();
        for k in victims {
            if let Some(pos) = self.registry_order.iter().position(|x| x == &k) {
                self.registry_order.remove(pos);
            }
            if let Some(old) = self.registry.remove(&k) {
                self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                // Terminal DESTROY, not RecycleTarget: an idle-drained resident is
                // stale by `IDLE_TARGET_AGE_MS` — it is not being actively recycled
                // (that is the capacity-eviction path's job for a per-frame video
                // output), so caching it in `target_free` would defeat the whole
                // point of the drain. `RecycleTarget` keeps the image's slab
                // sub-allocation live, so a diverse burst (hundreds of distinct
                // geometries, ≤ `TARGET_FREE_CAP_PER_KEY` each) would cache every
                // image and no slab block could ever empty — measured VRAM stuck at
                // 1532 MiB after the registry drained to ~22. Freeing the image
                // releases its slab sub-range; when a block's last sub-allocation
                // leaves, the slab returns it to the driver (`SLAB_KEEP_EMPTY`), so
                // VRAM returns to the idle baseline.
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.image,
                        view: old.view,
                        memory: old.memory,
                    },
                );
            }
        }
        // Track settled-ness and decide whether this pass may trim the expensive
        // HOST_VISIBLE buffer pools.
        let trim_buffers = self.note_drain_settled(drained);
        // A fired pass also returns the recycle pools' idle VRAM to the driver.
        self.trim_recycle_pools(&ctx.device, IDLE_RECYCLE_TRIM_PER_PASS, trim_buffers);
        // …and ages out the sampled-content cache, the upload-side pool the
        // recycle/buffer trims above do not cover. A settled video session's
        // frame textures (≤128 MiB) are pinned until LRU eviction that never
        // comes on a static desktop; age-gating on the same `IDLE_TARGET_AGE_MS`
        // cutoff frees them once idle without touching an actively-sampled entry.
        let sampled_cutoff = self.idle_clock_ms.saturating_sub(IDLE_TARGET_AGE_MS);
        self.trim_aged_sampled_cache(&ctx.device, sampled_cutoff, IDLE_RECYCLE_TRIM_PER_PASS);
        // …and the compute-storage residents, the standalone-VkDeviceMemory pool the
        // render-registry / sampled-cache drains above do not cover. A settled
        // compute-heavy session's blur/decode storage images are pinned until an
        // LRU eviction that never comes on a static desktop; the same age cutoff
        // frees them once idle without touching an actively-dispatched resident.
        self.trim_aged_compute_storage(&ctx.device, sampled_cutoff, IDLE_RECYCLE_TRIM_PER_PASS);
        // …and releases the empty slab blocks the hot release path retains as a
        // churn buffer, which otherwise sit resident forever at idle (no image
        // release fires to trigger their free). Down to one spare.
        self.slab
            .trim_empty_blocks(&ctx.device, IDLE_SLAB_KEEP_EMPTY);
    }

    /// Count of registry residents NOT held by a deferred-write pin — the
    /// LRU-evictable (active) working set the `REGISTRY_CAP` bounds. Pinned slots
    /// are bounded separately (by the arming rail's own window cap) and excluded
    /// so a pinned burst cannot force the active set into eviction thrash.
    fn non_pinned_registry_len(&self) -> usize {
        let pinned = self
            .registry
            .values()
            .filter(|slot| slot.pin_count > 0)
            .count();
        self.registry_order.len().saturating_sub(pinned)
    }

    /// Evict non-pinned resident targets (LRU, oldest at the front of
    /// `registry_order`) until the non-pinned population is at or below
    /// [`REGISTRY_CAP`]. Pinned slots (deferred render Stores whose only copy is
    /// on the GPU) and an optional `protect`ed identity rotate to the back
    /// instead of evicting — they are bounded separately and must not count
    /// toward the cap, or a pinned burst would force the active set out (thrash).
    /// One full rotation is the budget. `registry_order` is a `VecDeque`, so each
    /// front pop / rotate-to-back is O(1) and the whole sweep is O(n) — not the
    /// O(n²) a `Vec` front-`remove(0)` per rotation would cost under a large
    /// pinned population (`reg=512` measured under multi-4K load).
    ///
    /// Shared by both admit paths (`registry_ensure` passes the just-resolved
    /// identity as `protect`; `registry_ensure_color` passes `None`).
    unsafe fn evict_registry_to_cap(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        protect: Option<&TargetIdentity>,
    ) {
        let mut non_pinned = self.non_pinned_registry_len();
        let mut rotations = self.registry_order.len();
        while non_pinned > REGISTRY_CAP && rotations > 0 {
            rotations -= 1;
            let Some(old_k) = self.registry_order.front().cloned() else {
                break;
            };
            if self
                .registry
                .get(&old_k)
                .is_some_and(|slot| slot.pin_count > 0)
                || protect == Some(&old_k)
            {
                self.registry_order.pop_front();
                self.registry_order.push_back(old_k);
                continue;
            }
            if let Some(old) = self.registry.remove(&old_k) {
                if old.framebuffer != vk::Framebuffer::null() {
                    self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                }
                self.dispose(
                    &ctx.device,
                    DeferredHandle::RecycleTarget(FreeTargetImage {
                        image: old.image,
                        memory: old.memory,
                        view: old.view,
                        width: old.width,
                        height: old.height,
                        format: old.color_format,
                    }),
                );
                counters
                    .target_evicts
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.registry_order.pop_front();
            non_pinned = non_pinned.saturating_sub(1);
        }
    }

    /// Refresh a resident's idle-drain timestamp to at least `now_ms`. Used by
    /// host-window direct present before the export attempt so offscreen
    /// compositor peers needed for route-B tile compositing do not age out while
    /// the displayed member remains active.
    pub(crate) fn registry_touch_at(&mut self, identity: &TargetIdentity, now_ms: u64) {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let touch = self.idle_clock_ms;
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.last_touch_ms = touch;
        }
    }

    /// Cumulative sampled-cache pool recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    /// Merged into `CounterSnapshot` by `engine::counter_snapshot`.
    pub(crate) fn recycle_stats(&self) -> (u64, u64, u64, u64) {
        self.sampled_free.stats()
    }

    pub(crate) unsafe fn ensure_init(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        if self.initialized {
            return Ok(());
        }
        let cmd_pool = ctx
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.gq)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateCommandPool, e)))?;
        counters.note_create();
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
        counters.note_create();
        let mut slots = Vec::with_capacity(RING_DEPTH);
        for cmd_buf in cmd_bufs.into_iter() {
            // Fences start unsignaled: a slot with no pending cleanup is never
            // waited on, and a submit requires an unsignaled fence.
            match ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(fence) => {
                    counters.note_create();
                    slots.push(CmdSlot {
                        cmd_buf,
                        fence,
                        pending: None,
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
            crate::observe::off(format!(
                "desc_arena_grow blocks={} block_max_sets={DESC_BLOCK_MAX_SETS} cause=pool_exhausted",
                self.desc_arena.block_count()
            ));
        }
        Ok((set, pool))
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
    unsafe fn bind_image_slab(
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
    unsafe fn free_image_slab(&mut self, device: &ash::Device, image: vk::Image) {
        self.slab.free_image(device, image);
    }

    /// Terminal handling for a deferred handle once it is safe (in_flight == 0):
    /// a `RecycleSampled` slot rejoins `sampled_free` (bounded per key) for reuse
    /// instead of being destroyed; every other handle is destroyed.
    unsafe fn destroy_or_recycle(&mut self, device: &ash::Device, handle: DeferredHandle) {
        match handle {
            DeferredHandle::RecycleSampled(slot) => {
                if let Some(slot) = self.try_recycle_sampled(slot) {
                    self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
                }
            }
            DeferredHandle::RecycleTarget(img) => {
                if let Some(img) = self.try_recycle_target(img) {
                    self.destroy_deferred_handle(device, DeferredHandle::RecycleTarget(img));
                }
            }
            other => self.destroy_deferred_handle(device, other),
        }
    }

    /// Return an evicted sampled slot to `sampled_free` for reuse by a later
    /// same-geometry `acquire_sampled`. `None` means it was recycled; `Some(slot)`
    /// means a cap was full and the caller must destroy it.
    fn try_recycle_sampled(&mut self, slot: SampledSlot) -> Option<SampledSlot> {
        self.sampled_free.admit(slot.key(), slot)
    }

    /// Return a displaced resident-target image to `target_free` for reuse by a
    /// later same-(geometry, format) `registry_ensure`/`registry_ensure_color`
    /// create. `None` means it was recycled; `Some(img)` means a cap was full and
    /// the caller must destroy it.
    fn try_recycle_target(&mut self, img: FreeTargetImage) -> Option<FreeTargetImage> {
        self.target_free.admit(img.key(), img)
    }

    /// Return a retired transient compute-storage image to `storage_image_free`
    /// for reuse by a later same-geometry `acquire_storage_image`. `None` means it
    /// was recycled; `Some(slot)` means a cap was full and the caller must destroy
    /// it, freeing its standalone `VkDeviceMemory` (these are not slab-backed).
    fn try_recycle_storage_image(&mut self, slot: StorageImageSlot) -> Option<StorageImageSlot> {
        self.storage_image_free.admit(slot.key, slot)
    }

    /// Pop a recycled resident-target image for `(width, height, format)` if one
    /// is available, else `None`. Splits the reuse (`target_free_hits`) vs
    /// fresh-alloc (`target_free_allocs`) census so a boot can prove the
    /// per-frame realloc storm collapsed (allocs ≈ 0 under video).
    fn take_free_target(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Option<FreeTargetImage> {
        let key = TargetRecycleKey {
            width,
            height,
            format,
        };
        self.target_free.take(&key)
    }

    /// Cumulative resident-target recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    pub(crate) fn target_recycle_stats(&self) -> (u64, u64, u64, u64) {
        self.target_free.stats()
    }

    /// Cumulative compute-storage recycle diagnostics: `(admits, cap_drops)`.
    /// A nonzero `cap_drops` means a per-key or global cap actively bounded the
    /// pool (an all-new-geometry compute burst) — the "the cap is biting" signal
    /// surfaced as `st_drop` on the `vram` census.
    #[cfg(test)]
    pub(crate) fn storage_recycle_stats(&self) -> (u64, u64) {
        let (_, _, admits, cap_drops) = self.storage_image_free.stats();
        (admits, cap_drops)
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
    unsafe fn release_graveyard(&mut self, device: &ash::Device, retired: SlotMask) {
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
        let fence = self.slots[index].fence;
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesRetire))?;
        ctx.device
            .reset_fences(&[fence])
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsResetFencesRetire, e)))?;
        let pending = self.slots[index].pending.take().expect("checked above");
        self.in_flight = self.in_flight.saturating_sub(1);
        self.drain_cleanup(&ctx.device, pending);
        self.release_graveyard(&ctx.device, 1 << index);
        Ok(())
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
        // Reap the oldest contiguous run of already-signaled slots before
        // claiming one. A slot's cleanup carries the sampled-cache admissions
        // it owes (`drain_cleanup` -> `admit_sampled_slot`), and the readback
        // path deliberately waits the fence without retiring (see
        // `wait_entry_fence`). Retiring only the slot about to be reused
        // therefore parked every admission until the ring wrapped, so an
        // unchanged texture re-uploaded and re-allocated for RING_DEPTH - 1
        // draws before the content cache could serve it.
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
    /// a concurrent entry cannot recycle them, bundled with the descriptor set
    /// and deferred sampled-cache admissions.
    pub(crate) fn seal_entry(
        &mut self,
        dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<SampledRetain>,
    ) -> PendingGpuCleanup {
        let mut readback: Vec<BufferSlot> = self.readback_live.take().into_iter().collect();
        readback.append(&mut self.readback_multi_live);
        PendingGpuCleanup {
            dsets,
            staging: std::mem::take(&mut self.staging_live),
            readback,
            sampled: std::mem::take(&mut self.sampled_live),
            storage_images: std::mem::take(&mut self.storage_image_live),
            sampled_retains,
        }
    }

    /// Park the sealed cleanup on the current slot: its CB was submitted with
    /// the slot fence and the entry returns without waiting.
    pub(crate) fn finish_entry_async(&mut self, cleanup: PendingGpuCleanup) {
        debug_assert!(
            self.slots[self.cur].pending.is_none(),
            "current slot already owes cleanup"
        );
        self.slots[self.cur].pending = Some(cleanup);
        self.in_flight += 1;
    }

    /// The open batch's (CB, fence) when a draw at (identity, geometry, bgra)
    /// can append to it. `None` means the caller must claim its own slot (and
    /// `begin_entry` will flush the batch first).
    ///
    /// A full batch (BATCH_MAX_DRAWS) also answers `None`, turning the next
    /// same-target draw into a flush-then-reopen. Unbounded batches destroyed
    /// the pipeline (live A/B 2026-07-19): the GPU idled while the CPU
    /// recorded the whole run — the present then blocked behind the entire
    /// batch executing from scratch (presents 38.7 -> 27/s) — and every
    /// draw's staging slots stayed hoarded in ONE pending ring entry until
    /// its fence retired, starving the free lists into per-bind
    /// vkCreateBuffer/vkAllocateMemory churn (setup_bufs 50 -> 108 us/draw).
    /// The cap keeps the GPU fed every ~N draws while still amortizing the
    /// per-draw submit+fence cost N-fold.
    pub(crate) fn batch_slot(
        &self,
        identity: &TargetIdentity,
        width: u32,
        height: u32,
        bgra: bool,
    ) -> Option<(vk::CommandBuffer, vk::Fence)> {
        let b = self.open_batch.as_ref()?;
        (b.draws < BATCH_MAX_DRAWS
            && b.identity == *identity
            && b.width == width
            && b.height == height
            && b.bgra == bgra)
            .then_some((b.cb, b.fence))
    }

    /// Record a batch-deferred draw's completion: open the batch on its ring
    /// slot (opener) or extend it (joiner), accumulating the per-draw
    /// descriptor set and sampled-cache admissions for the single flush-time
    /// seal. The CB stays in recording state; submit happens at
    /// [`Self::batch_flush`].
    #[allow(
        clippy::too_many_arguments,
        reason = "batch ownership tracks every Vulkan object and resident pin explicitly"
    )]
    pub(crate) fn batch_append(
        &mut self,
        cb: vk::CommandBuffer,
        fence: vk::Fence,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        bgra: bool,
        dset: Option<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<SampledRetain>,
        counters: &EngineCounters,
    ) {
        match self.open_batch.as_mut() {
            Some(b) => {
                debug_assert!(b.cb == cb, "joiner recorded into a foreign CB");
                b.draws += 1;
                b.dsets.extend(dset);
                b.sampled_retains.extend(sampled_retains);
                counters.batch_joins.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                debug_assert!(
                    self.slots[self.cur].pending.is_none(),
                    "batch opener's slot already owes cleanup"
                );
                self.open_batch = Some(OpenBatch {
                    cb,
                    fence,
                    identity,
                    width,
                    height,
                    bgra,
                    draws: 1,
                    dsets: dset.into_iter().collect(),
                    sampled_retains,
                });
                counters.batch_opens.fetch_add(1, Ordering::Relaxed);
            }
        }
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
        let Some(mut batch) = self.open_batch.take() else {
            return Ok(());
        };
        counters.batch_flushes.fetch_add(1, Ordering::Relaxed);
        counters
            .batch_flush_draws
            .fetch_add(batch.draws, Ordering::Relaxed);
        let submit = (|| -> Result<(), DrawError> {
            ctx.device
                .end_command_buffer(batch.cb)
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsEndCbBatch, e)))?;
            let cbs = [batch.cb];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            match ctx.device.queue_submit(ctx.queue(), &[si], batch.fence) {
                Ok(()) => Ok(()),
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
            Ok(()) => {
                let cleanup = self.seal_entry(
                    std::mem::take(&mut batch.dsets),
                    std::mem::take(&mut batch.sampled_retains),
                );
                self.finish_entry_async(cleanup);
                Ok(())
            }
            Err(e) => {
                self.desc_arena.free(&ctx.device, &batch.dsets);
                Err(e)
            }
        }
    }

    /// Wait a single already-submitted entry fence WITHOUT retiring the ring.
    ///
    /// A synchronous reader (e.g. [`super::read_target_inner`]) that submitted
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
        &self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        fence: vk::Fence,
    ) -> Result<(), DrawError> {
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesEntry))
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
        self.desc_arena.free(device, &pending.dsets);
        // Cache admissions first (they move slots into the cache and may evict
        // images — deferred via dispose() while others are in flight), then
        // recycle what remains.
        for retain in pending.sampled_retains.drain(..) {
            if let Some(index) = pending
                .sampled
                .iter()
                .position(|slot| slot.image == retain.image)
            {
                let slot = pending.sampled.remove(index);
                self.admit_sampled_slot(device, slot, &retain.content, retain.identity);
            }
        }
        for slot in pending.staging.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.staging_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.readback.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.sampled.drain(..) {
            // Respect the same global + per-key cap as the eviction-recycle path
            // (`destroy_or_recycle`). This per-frame retire path previously pushed
            // unconditionally, so a diverse-content workload (many distinct sampled
            // keys — measured 4×4K video) grew `sampled_free` past
            // `SAMPLED_FREE_CAP_TOTAL` (live `sfree=203`), each slot pinning a slab
            // sub-allocation. Past the cap the slot is destroyed, not recycled.
            if let Some(slot) = self.try_recycle_sampled(slot) {
                self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
            }
        }
        for slot in pending.storage_images.drain(..) {
            // Respect the per-key + global cap (mirrors the sampled retire path
            // above). This path previously pushed unconditionally, so an all-new-
            // geometry compute workload (diff-heavy / CoreImage / blur burst) grew
            // `storage_image_free` without bound — and because storage images are
            // standalone (non-slab) allocations, each hoarded slot pins a whole
            // VkDeviceMemory (invisible to the slab `resident_mb`/`live_subs`
            // census). Past the cap the slot is destroyed, freeing its device
            // memory.
            if let Some(slot) = self.try_recycle_storage_image(slot) {
                self.destroy_deferred_handle(
                    device,
                    DeferredHandle::Image {
                        image: slot.image,
                        view: slot.view,
                        memory: slot.memory,
                    },
                );
            }
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
    /// 9 725 allocations at ~817 µs each over one 260 s boot — while the `vram`
    /// census showed the free pool holding up to 133 MiB at the same time. A pool
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
        crate::observe::off(format!(
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
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Upload)
            .ok_or({
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForStaging {
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
            .host_slab
            .acquire(ctx, &req, counters)
            .inspect_err(|_| ctx.device.destroy_buffer(buffer, None))?;
        ctx.device
            .bind_buffer_memory(buffer, token.memory, token.offset())
            .map_err(|e| {
                self.host_slab.release(&ctx.device, token);
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
            backing: BufferBacking::HostSlab(token),
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
            crate::runtime::drain::note_store_route_us(
                "swap_rb_us",
                started.elapsed().as_micros() as u64,
            );
            crate::runtime::drain::note_store_route_n("swap_rb_kb", (rgba.len() / 1024) as u64);
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
        runs: &[super::types::GuestRun],
        total_len: u64,
    ) -> Result<(), DrawError> {
        let _slow = SlowStagingWrite::watch("guest_runs", total_len, runs.len());
        let size = total_len.max(4);
        let ptr = staging_write_ptr(ctx, slot, size)?;
        let total = total_len as usize;
        let mut off = 0usize;
        unsafe {
            for run in runs {
                if off >= total {
                    break;
                }
                let n = (run.len as usize).min(total - off);
                // SAFETY: `host_ptr` is a stable RAMBlock alias from
                // `HostOps::map_pages`, valid for the VM lifetime; `ptr` is the
                // mapped staging span of `size >= total` bytes and `off + n <=
                // total`, so the destination stays in bounds.
                std::ptr::copy_nonoverlapping(run.host_ptr as *const u8, ptr.add(off), n);
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
                        vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(create_op, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Readback)
            .ok_or({
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForReadback {
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
    pub(crate) fn lease_readback(&mut self) -> Option<(u64, usize)> {
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
        READBACK_LEASES_OUT.fetch_add(1, Ordering::AcqRel);
        let mapped = slot.mapped;
        self.readback_leased.push(LeasedReadback { token, slot });
        Some((token, mapped))
    }

    /// Take back every lease whose holder has finished and return its slot to
    /// the free list.
    ///
    /// Cheap and unconditional: the returned-token channel is empty on all but
    /// the calls that follow a lease, and a lease that is still out is simply
    /// not in the drained set.
    pub(crate) fn reclaim_returned_readback_leases(&mut self) {
        let returned = std::mem::take(&mut *RETURNED_READBACK_LEASES.lock());
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
        counters.note_create();
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
                .subresource_range(super::color_subresource_range()),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateTargetView,
                    e,
                )));
            }
        };
        counters.note_create();
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
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create();
        // Cap target pool
        if self.target_order.len() >= 32 {
            if let Some(old_k) = self.target_order.first().cloned() {
                if let Some(old) = self.targets.remove(&old_k) {
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

    #[allow(
        clippy::too_many_arguments,
        reason = "sampled-image acquisition mirrors its Vulkan format and content identity"
    )]
    pub(crate) unsafe fn acquire_sampled(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        one_dim: bool,
        format: ash::vk::Format,
        swizzle: crate::contract::pixel_format::SwizzlePlan,
        counters: &EngineCounters,
    ) -> Result<SampledSlot, DrawError> {
        let sk = SampledKey {
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            one_dim,
            format,
            swizzle,
        };
        // A hit reuses a recycled slot — no vkAllocateMemory this acquire. A miss
        // is counted here, at the empty free list, rather than after the create
        // succeeds: the census question is whether the pool had one, not whether
        // the fallback worked.
        if let Some(slot) = self.sampled_free.take(&sk) {
            let handles = slot.handles();
            self.sampled_live.push(slot);
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
                    .mip_levels(1)
                    .array_layers(array_layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateSampledImage, e)))?;
        counters.note_create();
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
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: array_layers,
                }),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateSampledView,
                    e,
                )));
            }
        };
        counters.note_create();
        let slot = SampledSlot {
            image,
            memory,
            view,
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            one_dim,
            format,
            swizzle,
        };
        let handles = slot.handles();
        self.sampled_live.push(slot);
        Ok(handles)
    }

    /// Bind a retained image on producer identity alone — same key, same
    /// generation means the same bytes under the producer's coherence model, so
    /// nothing is hashed and nothing compared.
    ///
    /// The only way to reach a `Gathered` entry, whose bytes were never on the
    /// CPU to be digested.
    fn find_sampled_by_identity(
        &mut self,
        key: SampledKey,
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    ) -> Option<SampledSlot> {
        let id = identity?;
        let index = self
            .sampled_cache
            .iter()
            .position(|entry| entry.slot.key() == key && entry.identity == Some(id))?;
        let mut entry = self.sampled_cache.remove(index);
        entry.last_touch_ms = self.idle_clock_ms;
        let handles = entry.slot.handles();
        self.sampled_cache.push(entry);
        Some(handles)
    }

    /// The guest-gather rail's lookup: the complete image key, and identity as
    /// the only content evidence there is.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the complete image key, as its content-bearing sibling does"
    )]
    pub(crate) fn find_gathered_sampled(
        &mut self,
        width: u32,
        height: u32,
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        one_dim: bool,
        format: ash::vk::Format,
        swizzle: crate::contract::pixel_format::SwizzlePlan,
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
        counters: &EngineCounters,
    ) -> Option<SampledSlot> {
        let handles = self.find_sampled_by_identity(
            SampledKey {
                width,
                height,
                layers,
                volume,
                cube,
                arrayed,
                one_dim,
                format,
                swizzle,
            },
            identity,
        )?;
        counters
            .sampled_identity_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(handles)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "sampled-cache lookup mirrors the complete image and content key"
    )]
    pub(crate) fn find_cached_sampled(
        &mut self,
        width: u32,
        height: u32,
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        one_dim: bool,
        format: ash::vk::Format,
        swizzle: crate::contract::pixel_format::SwizzlePlan,
        content: &[u8],
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
        counters: &EngineCounters,
    ) -> Option<SampledSlot> {
        let key = SampledKey {
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            one_dim,
            format,
            swizzle,
        };
        if let Some(handles) = self.find_sampled_by_identity(key, identity) {
            counters
                .sampled_identity_hits
                .fetch_add(1, Ordering::Relaxed);
            return Some(handles);
        }
        let content_hash = sampled_content_hash(content);
        // 128-bit fingerprint match binds the retained image directly — the
        // former full-frame `entry.content == content` compare (a cold DRAM
        // read of the retained copy on every hit) is dropped in favour of the
        // wider digest.
        let found = self.sampled_cache.iter().position(|entry| {
            entry.slot.key() == key
                && entry.fingerprint == SampledFingerprint::Content(content_hash)
        });
        let Some(index) = found else {
            counters
                .sampled_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let mut entry = self.sampled_cache.remove(index);
        // Same content re-presented under a new identity/generation: adopt it
        // so the fast path serves the follow-up draws.
        if identity.is_some() {
            entry.identity = identity;
        }
        entry.last_touch_ms = self.idle_clock_ms;
        let handles = entry.slot.handles();
        self.sampled_cache.push(entry);
        counters.sampled_cache_hits.fetch_add(1, Ordering::Relaxed);
        counters
            .sampled_cache_hit_bytes
            .fetch_add(content.len() as u64, Ordering::Relaxed);
        Some(handles)
    }

    /// Admit one detached sampled slot into the exact-content cache. A slot
    /// whose content duplicates an existing entry returns to the live list
    /// (recycled later); cap evictions go through dispose() so an image a
    /// concurrent in-flight CB samples is never destroyed under it.
    unsafe fn admit_sampled_slot(
        &mut self,
        device: &ash::Device,
        slot: SampledSlot,
        content: &SampledRetainContent,
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    ) {
        let (fingerprint, content_len) = match content {
            SampledRetainContent::Bytes(bytes) => (
                SampledFingerprint::Content(sampled_content_hash(bytes)),
                bytes.len(),
            ),
            // Nothing hashed the bytes, so nothing can recognise this image by
            // them. An entry with no identity to be found under would be
            // unreachable dead weight in a capped cache, so it is not admitted.
            SampledRetainContent::Gathered { len } => {
                if identity.is_none() {
                    self.sampled_live.push(slot);
                    return;
                }
                (SampledFingerprint::Gathered, *len)
            }
        };
        if content_len > SAMPLED_CACHE_BYTE_CAP {
            self.sampled_live.push(slot);
            return;
        }
        // Deduplication is by fingerprint, and `Gathered` entries are equal to
        // each other under it, so they dedupe by identity instead — two windows
        // that gathered different pixels must not collapse into one image.
        let duplicate = self.sampled_cache.iter_mut().find(|entry| {
            entry.slot.key() == slot.key()
                && entry.fingerprint == fingerprint
                && (fingerprint != SampledFingerprint::Gathered
                    || entry.identity == identity)
        });
        if let Some(existing) = duplicate {
            if identity.is_some() {
                existing.identity = identity;
            }
            self.sampled_live.push(slot);
            return;
        }
        self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_add(content_len);
        let touch = self.idle_clock_ms;
        self.sampled_cache.push(ResidentSampledSlot {
            slot,
            fingerprint,
            content_len,
            identity,
            last_touch_ms: touch,
        });
        // Which of the two caps is doing the evicting decides whether a later
        // miss is a capacity eviction or content the cache has never held.
        // `sampled_cache_misses` cannot tell them apart — a miss is only the
        // absence of a (key, fingerprint) entry, and the two causes look
        // identical from there. Both routes reading zero says every miss is
        // content, and then raising either cap buys nothing.
        while self.sampled_cache.len() > SAMPLED_CACHE_CAP
            || self.sampled_cache_bytes > SAMPLED_CACHE_BYTE_CAP
        {
            if let Some(route) = sampled_evict_route(self.sampled_cache.len(), self.sampled_cache_bytes)
            {
                crate::runtime::drain::note_store_route(route);
            }
            let evicted = self.sampled_cache.remove(0);
            self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_sub(evicted.content_len);
            // Recycle rather than destroy: a content-changing sampled input
            // (live tile / video frame) re-uploads into this same-geometry image
            // next frame instead of a fresh vkAllocateMemory. Routed through the
            // in-flight-safe deferral (an in-flight CB may still sample it).
            self.dispose(device, DeferredHandle::RecycleSampled(evicted.slot));
        }
    }

    pub(crate) fn recycle_sampled(&mut self) {
        for slot in self.sampled_live.drain(..) {
            let sk = slot.key();
            self.sampled_free.push_uncapped(sk, slot);
        }
    }
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

impl crate::observe::Decline for ReadbackMemoryDegrade {
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
    if !crate::observe::first_sight("readback_memory", u64::from(memory_type)) {
        return;
    }
    let topology = ctx.caps.memory.topology.slug();
    if kind.cached {
        crate::observe::off(format!(
            "readback_memory type={memory_type} cached=1 coherent={} topology={topology}",
            u8::from(kind.coherent),
        ));
    } else {
        crate::observe::Emit::decline(
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

/// Which cap is over budget, given the exact-content cache's size after an
/// admission. `None` when neither is, which is the caller's loop condition and
/// so unreachable from it — it exists so this decision is testable without a
/// device.
///
/// The count cap is reported in preference to the byte cap when both are over,
/// because a count-capped cache at three orders of magnitude under its byte cap
/// is the shape the sampled rail was measured in and the one that would make
/// raising `SAMPLED_CACHE_CAP` worth anything.
fn sampled_evict_route(len: usize, bytes: usize) -> Option<&'static str> {
    if len > SAMPLED_CACHE_CAP {
        Some("sampled_evict_count_cap")
    } else if bytes > SAMPLED_CACHE_BYTE_CAP {
        Some("sampled_evict_byte_cap")
    } else {
        None
    }
}

#[cfg(test)]
mod evict_route_tests {
    use super::*;

    #[test]
    fn a_cache_inside_both_caps_evicts_for_neither_reason() {
        assert_eq!(sampled_evict_route(SAMPLED_CACHE_CAP, 0), None);
        assert_eq!(
            sampled_evict_route(SAMPLED_CACHE_CAP, SAMPLED_CACHE_BYTE_CAP),
            None
        );
    }

    #[test]
    fn one_entry_over_the_count_cap_names_the_count_cap() {
        assert_eq!(
            sampled_evict_route(SAMPLED_CACHE_CAP + 1, 0),
            Some("sampled_evict_count_cap")
        );
    }

    #[test]
    fn one_byte_over_the_byte_cap_names_the_byte_cap() {
        assert_eq!(
            sampled_evict_route(1, SAMPLED_CACHE_BYTE_CAP + 1),
            Some("sampled_evict_byte_cap")
        );
    }

    /// Both over is the count cap, so the two routes partition the evictions
    /// and their sum is the eviction count rather than exceeding it.
    #[test]
    fn over_both_caps_is_charged_once_to_the_count_cap() {
        assert_eq!(
            sampled_evict_route(SAMPLED_CACHE_CAP + 1, SAMPLED_CACHE_BYTE_CAP + 1),
            Some("sampled_evict_count_cap")
        );
    }
}

#[cfg(test)]
mod recycle_tests {
    use super::*;

    fn null_slot(w: u32, h: u32) -> SampledSlot {
        SampledSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            layers: 1,
            volume: false,
            cube: false,
            arrayed: false,
            one_dim: false,
            format: crate::backend::vulkan::translate::pixel::vk_texel_layout(
                crate::contract::pixel_format::TexelLayout::Bgra8,
            ),
            swizzle: Default::default(),
        }
    }

    /// A *diverse* burst — many distinct geometries, each ≤ the per-key cap —
    /// must not grow `sampled_free` past the GLOBAL cap: this is the measured
    /// VRAM-return stall (`sfree=593` pinning every slab block). Each distinct
    /// key admits until the pool total hits `SAMPLED_FREE_CAP_TOTAL`, then every
    /// further eviction is destroyed (returns Some) regardless of its key.
    #[test]
    fn sampled_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        // One eviction per distinct 1-pixel-taller geometry, more than the global
        // cap. Each key is fresh so the per-key cap never bites — only the global
        // cap can bound this.
        let mut admitted = 0;
        for i in 0..(SAMPLED_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_sampled(null_slot(16, 16 + i as u32))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, SAMPLED_FREE_CAP_TOTAL,
            "global cap bounds the diverse burst"
        );
        assert_eq!(
            pools.sampled_free.len(),
            SAMPLED_FREE_CAP_TOTAL,
            "pool total pinned at the global cap"
        );
    }

    /// `target_free` has the same global cap for the same reason.
    #[test]
    fn target_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        let mut admitted = 0;
        for i in 0..(TARGET_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_target(null_target(
                    16,
                    16 + i as u32,
                    translate::pixel::SCANOUT_FORMAT,
                ))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, TARGET_FREE_CAP_TOTAL,
            "global cap bounds the burst"
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

    /// The compute-storage recycle pool (`storage_image_free`) had NO cap before
    /// this fix, so an all-new-geometry compute burst (each a standalone, non-slab
    /// `vkAllocateMemory`) grew it without bound. A diverse burst — many distinct
    /// geometries, each ≤ the per-key cap — must now stop admitting at the GLOBAL
    /// cap; past it every slot is returned (Some) for the caller to destroy.
    #[test]
    fn storage_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        let mut admitted = 0;
        for i in 0..(STORAGE_IMAGE_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_storage_image(null_storage_slot(16, 16 + i as u32))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, STORAGE_IMAGE_FREE_CAP_TOTAL,
            "global cap bounds the diverse burst"
        );
        assert_eq!(
            pools.storage_image_free.len(),
            STORAGE_IMAGE_FREE_CAP_TOTAL,
            "pool total pinned at the global cap"
        );
    }

    /// Within one geometry the pool recycles up to the per-key cap (reuse instead
    /// of a fresh allocation); beyond the cap the slot is returned for the caller
    /// to destroy, and the admits/cap-drops counters split the two so a leak is
    /// diagnosable (`st_drop` on the census).
    #[test]
    fn storage_free_recycle_up_to_per_key_cap_then_drops() {
        let mut pools = ResourcePools::new();
        let key = null_storage_slot(512, 512).key;
        for i in 0..STORAGE_IMAGE_FREE_CAP_PER_KEY {
            assert!(
                pools
                    .try_recycle_storage_image(null_storage_slot(512, 512))
                    .is_none(),
                "recycle {i} within the per-key cap must be admitted"
            );
        }
        assert_eq!(
            pools.storage_image_free.count_for(&key),
            STORAGE_IMAGE_FREE_CAP_PER_KEY
        );
        assert!(
            pools
                .try_recycle_storage_image(null_storage_slot(512, 512))
                .is_some(),
            "over the per-key cap the slot is returned for destroy"
        );
        let (admits, cap_drops) = pools.storage_recycle_stats();
        assert_eq!(admits, STORAGE_IMAGE_FREE_CAP_PER_KEY as u64);
        assert_eq!(cap_drops, 1);
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
    /// `vkAllocateMemory` next frame) up to a per-key cap; beyond the cap the
    /// caller must destroy so a one-off geometry cannot pin memory for the whole
    /// guest lifetime. Device-free: exercises only the routing/cap decision.
    #[test]
    fn evicted_sampled_slots_recycle_into_free_list_up_to_cap() {
        let mut pools = ResourcePools::new();
        let hd = null_slot(1920, 1080).key();

        // The first CAP evictions of one geometry recycle (return None) and are
        // available for a later same-geometry acquire.
        for i in 0..SAMPLED_FREE_CAP_PER_KEY {
            assert!(
                pools.try_recycle_sampled(null_slot(1920, 1080)).is_none(),
                "eviction {i} within cap must recycle"
            );
        }
        assert_eq!(
            pools.sampled_free.count_for(&hd),
            SAMPLED_FREE_CAP_PER_KEY
        );

        // Over the cap: caller must destroy (returns the slot); free list is
        // bounded, not grown.
        assert!(
            pools.try_recycle_sampled(null_slot(1920, 1080)).is_some(),
            "over-cap eviction must not recycle"
        );
        assert_eq!(
            pools.sampled_free.count_for(&hd),
            SAMPLED_FREE_CAP_PER_KEY
        );

        // A different geometry has an independent cap.
        let small = null_slot(64, 64).key();
        assert!(pools.try_recycle_sampled(null_slot(64, 64)).is_none());
        assert_eq!(pools.sampled_free.count_for(&small), 1);
    }

    /// The idle sampled-cache trim reclaims only entries idle past
    /// `IDLE_TARGET_AGE_MS` (an actively-sampled entry is touched every frame and
    /// must survive), decrements the byte accounting by exactly the trimmed
    /// entries, and is bounded per pass — so a live video session is never
    /// disturbed while a settled one's ≤128 MiB of frame textures return to the
    /// driver.
    #[test]
    fn idle_trim_reclaims_only_aged_sampled_cache_entries() {
        let mut pools = ResourcePools::new();
        let push = |pools: &mut ResourcePools, w: u32, h: u32, touch: u64, len: usize| {
            pools.sampled_cache_bytes = pools.sampled_cache_bytes.saturating_add(len);
            pools.sampled_cache.push(ResidentSampledSlot {
                slot: null_slot(w, h),
                fingerprint: SampledFingerprint::Content(((w as u128) << 64) | h as u128),
                content_len: len,
                identity: None,
                last_touch_ms: touch,
            });
        };
        push(&mut pools, 1920, 1080, 1_000, 8_000_000); // aged
        push(&mut pools, 1280, 720, 1_500, 4_000_000); // aged
        push(&mut pools, 640, 480, 9_000, 1_000_000); // freshly touched
        assert_eq!(pools.sampled_cache.len(), 3);
        assert_eq!(pools.sampled_cache_bytes, 13_000_000);

        // Idle clock at 10_000 ms → cutoff = 10_000 - IDLE_TARGET_AGE_MS.
        let cutoff = 10_000u64.saturating_sub(IDLE_TARGET_AGE_MS);
        let taken = pools.take_aged_sampled_slots(cutoff, IDLE_RECYCLE_TRIM_PER_PASS);

        assert_eq!(taken.len(), 2, "only the two aged entries are taken");
        assert_eq!(pools.sampled_cache.len(), 1, "the fresh entry stays cached");
        assert_eq!(
            pools.sampled_cache[0].slot.width, 640,
            "the surviving entry is the freshly-touched one"
        );
        assert_eq!(
            pools.sampled_cache_bytes, 1_000_000,
            "byte accounting drops exactly the two aged entries"
        );

        // Per-pass bound: three more aged entries, max=2 → only two taken.
        push(&mut pools, 100, 100, 0, 500_000);
        push(&mut pools, 101, 101, 0, 500_000);
        push(&mut pools, 102, 102, 0, 500_000);
        assert_eq!(pools.take_aged_sampled_slots(cutoff, 2).len(), 2);
    }

    /// The compute-storage residents are standalone (non-slab) VkDeviceMemory, so a
    /// settled compute session that leaves stale residents pins whole allocations
    /// until an LRU eviction that never comes on a static desktop. The idle drain
    /// must reclaim exactly the non-pinned residents idle past the age cutoff, leave
    /// a freshly-touched or pinned resident alone, and bound the batch per pass.
    #[test]
    fn idle_trim_reclaims_only_aged_non_pinned_compute_storage() {
        let mut pools = ResourcePools::new();
        let admit = |pools: &mut ResourcePools, tex: u32, touch: u64, pinned: bool| {
            let id = ComputeStorageResidencyKey::linear(0, tex, 0, 0, 0, 8, 8, 0);
            pools.compute_storage_registry.insert(
                id,
                ResidentStorageImageSlot {
                    slot: null_storage_slot(8, 8),
                    generation: 0,
                    layout: vk::ImageLayout::UNDEFINED,
                    pinned,
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

        // Idle clock at 10_000 ms → cutoff = 10_000 - IDLE_TARGET_AGE_MS.
        let cutoff = 10_000u64.saturating_sub(IDLE_TARGET_AGE_MS);
        let taken = pools.take_aged_storage_residents(cutoff, IDLE_RECYCLE_TRIM_PER_PASS);

        assert_eq!(
            taken.len(),
            2,
            "only the two aged non-pinned residents are taken"
        );
        assert_eq!(
            pools.compute_storage_registry.len(),
            2,
            "pinned + fresh survive"
        );
        assert_eq!(
            pools.compute_storage_order.len(),
            2,
            "the LRU order drops exactly the reclaimed residents"
        );
        assert!(
            pools
                .compute_storage_registry
                .keys()
                .all(|k| k.texture_ref == 3 || k.texture_ref == 4),
            "the survivors are the pinned (3) and the freshly-touched (4) residents"
        );

        // Per-pass bound: three more aged residents, max=2 → only two taken.
        admit(&mut pools, 5, 0, false);
        admit(&mut pools, 6, 0, false);
        admit(&mut pools, 7, 0, false);
        assert_eq!(pools.take_aged_storage_residents(cutoff, 2).len(), 2);
    }

    /// The recycle diagnostics (`recycle_stats`) count admits vs cap-drops so a
    /// later boot can tell whether the per-key cap or the drain timing is the
    /// lag-tail limiter. `try_recycle_sampled` is the only mutator of the two
    /// recycle counters; the acquire-side hit/alloc counters need a device so
    /// they are exercised on the live path, not here.
    #[test]
    fn recycle_stats_count_admits_and_cap_drops() {
        let mut pools = ResourcePools::new();
        assert_eq!(pools.recycle_stats(), (0, 0, 0, 0));

        // CAP admits, then one over-cap drop, on one geometry.
        for _ in 0..SAMPLED_FREE_CAP_PER_KEY {
            pools.try_recycle_sampled(null_slot(1920, 1080));
        }
        pools.try_recycle_sampled(null_slot(1920, 1080));
        // One admit on an independent geometry.
        pools.try_recycle_sampled(null_slot(64, 64));

        let (free_hits, free_allocs, admits, cap_drops) = pools.recycle_stats();
        assert_eq!(free_hits, 0, "no acquires happened");
        assert_eq!(free_allocs, 0, "no acquires happened");
        assert_eq!(
            admits,
            SAMPLED_FREE_CAP_PER_KEY as u64 + 1,
            "CAP big-geometry admits + 1 small-geometry admit"
        );
        assert_eq!(
            cap_drops, 1,
            "exactly the one over-cap eviction was dropped"
        );
    }

    fn null_target(w: u32, h: u32, format: vk::Format) -> FreeTargetImage {
        FreeTargetImage {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            format,
        }
    }

    /// Resident-target images displaced from the identity registry (generation
    /// bump / geometry change / LRU) rejoin `target_free` for reuse up to a
    /// per-(geometry, format) cap; beyond it the caller must destroy so a
    /// one-off geometry cannot pin VRAM for the guest lifetime. Device-free:
    /// exercises only the routing/cap decision (mirrors the sampled recycle).
    #[test]
    fn displaced_targets_recycle_into_free_list_up_to_cap() {
        let mut pools = ResourcePools::new();
        let fmt = translate::pixel::SCANOUT_FORMAT;
        let key = null_target(1920, 1080, fmt).key();

        for i in 0..TARGET_FREE_CAP_PER_KEY {
            assert!(
                pools
                    .try_recycle_target(null_target(1920, 1080, fmt))
                    .is_none(),
                "displacement {i} within cap must recycle"
            );
        }
        assert_eq!(
            pools.target_free.count_for(&key),
            TARGET_FREE_CAP_PER_KEY
        );

        assert!(
            pools
                .try_recycle_target(null_target(1920, 1080, fmt))
                .is_some(),
            "over-cap displacement must not recycle"
        );
        assert_eq!(
            pools.target_free.count_for(&key),
            TARGET_FREE_CAP_PER_KEY
        );

        // A different format is an independent bucket (an RGBA image cannot back
        // a BGRA attachment).
        let rgba = null_target(1920, 1080, translate::pixel::RESIDENT_RGBA_FORMAT).key();
        assert!(pools
            .try_recycle_target(null_target(
                1920,
                1080,
                translate::pixel::RESIDENT_RGBA_FORMAT
            ))
            .is_none());
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
        assert!(pools.take_free_target(1920, 1080, fmt).is_none());
        // Its predecessor image is displaced (a new generation replaced it) and
        // recycled.
        assert!(pools
            .try_recycle_target(null_target(1920, 1080, fmt))
            .is_none());

        // Frames 1..N: each pops the recycled image (hit) and recycles the one
        // it replaces — steady state is hit-per-frame, alloc-once.
        for f in 0..8 {
            assert!(
                pools.take_free_target(1920, 1080, fmt).is_some(),
                "frame {f} must reuse the recycled image"
            );
            assert!(pools
                .try_recycle_target(null_target(1920, 1080, fmt))
                .is_none());
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
                staging: Vec::new(),
                readback: Vec::new(),
                sampled: Vec::new(),
                storage_images: Vec::new(),
                sampled_retains: Vec::new(),
            }),
        }
    }

    fn idle_slot() -> CmdSlot {
        CmdSlot {
            cmd_buf: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            pending: None,
        }
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

    /// The open draw batch records into `cur` without being submitted, so its
    /// slot has no pending cleanup yet — but its CB already references pool
    /// objects. Its bit must be in the mask, or a dispose during an open batch
    /// would free something the flush is about to submit against.
    #[test]
    fn the_open_batch_slot_counts_as_open_even_though_it_owes_no_cleanup() {
        let mut pools = ResourcePools::new();
        pools.slots = (0..4).map(|_| idle_slot()).collect();
        pools.cur = 3;
        assert_eq!(pools.open_slot_mask(), 0, "idle ring, no batch");

        pools.open_batch = Some(OpenBatch {
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            identity: TargetIdentity::Anonymous { slot: 0 },
            width: 16,
            height: 16,
            bgra: false,
            draws: 1,
            dsets: Vec::new(),
            sampled_retains: Vec::new(),
        });
        assert_eq!(pools.open_slot_mask(), 1 << 3, "the batch's own slot");
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
/// Split out of [`Pools::write_staging_swap_rb`] so the transformation has a
/// test: that method's destination is a mapped Vulkan allocation, so nothing
/// about it is reachable from a unit test, and the exchange is the part that can
/// be wrong. `dst` is written and never read, which is what keeps the
/// write-combined case off a read-modify-write path — see the caller for the
/// measurement that motivated writing whole pixels instead of single bytes.
///
/// A trailing partial pixel is copied through unexchanged. Bytes of `dst` past
/// `src.len()` are left alone; the caller owns whatever the mapped span needs
/// beyond the source.
pub(crate) fn exchange_rb_into(src: &[u8], dst: &mut [u8]) {
    let n = src.len().min(dst.len());
    let whole = n / 4 * 4;
    for (s, d) in src[..whole].chunks_exact(4).zip(dst[..whole].chunks_exact_mut(4)) {
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
