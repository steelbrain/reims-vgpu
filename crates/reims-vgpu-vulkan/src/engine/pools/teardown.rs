//! `destroy_all` — the one ordered driver that takes every pool down.
//!
//! Kept apart from the two chapters that fill the pools because the order is
//! the contract: quiesce the in-flight fences, fold each slot's owed transients
//! back into the live lists, then destroy in an order no acquire path has any
//! say in. A second teardown path would be a second order, so there is one.
//!
//! `use super::*` is the seam. This is an `impl` chapter of the module that
//! declares `ResourcePools` and owns its fields, not a layer beneath it.

use super::*;

impl ResourcePools {
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        // An open (never-submitted) batch dies with the pool: its CB belongs
        // to cmd_pool (destroyed below) and its dsets to desc_pool; the
        // accumulated transients are already in the live lists.
        self.discard_open_batch();
        self.abort_recorded_guest_work();
        // No command from this CB will be submitted now. Destroying its command
        // pool discards the unfinished recording, including an open pass, so
        // there is neither a legal nor a useful cmd_end_render_pass to emit.
        self.encoder.open_pass = None;
        self.encoder.forget_pass_echo();
        // A queue handoff transaction is installed on its exact slot before
        // the driver can accept it. Current callers hold the pool exclusively
        // across that call; keeping teardown total over this state also makes
        // the ownership contract valid when recorders become independent.
        for index in 0..self.encoder.slots.len() {
            let state = std::mem::replace(
                &mut self.encoder.slots[index].submission,
                SlotSubmission::HostOwned,
            );
            match state {
                SlotSubmission::SealedWaitingCommit(sealed) => {
                    self.abort_sealed_entry(device, index, sealed);
                }
                state => self.encoder.slots[index].submission = state,
            }
        }
        // A batched submit transfers host ownership of its fence to the queue
        // thread until the driver's submit call returns. Teardown can be
        // reached through session release without a queue-wide barrier, so
        // reclaim each exact fence before the waits below touch it.
        for slot in &mut self.encoder.slots {
            let state = std::mem::replace(&mut slot.submission, SlotSubmission::HostOwned);
            let result = match state {
                SlotSubmission::HostOwned => Ok(()),
                SlotSubmission::SealedWaitingCommit(_) => {
                    unreachable!("sealed entries were recovered before queue returns")
                }
                SlotSubmission::QueueOwned(receipt) => receipt.wait(),
                SlotSubmission::Failed(result) => Err(result),
            };
            if let Err(result) = result {
                slot.submission = SlotSubmission::Failed(result);
                let decline = DrawError::VkCall(VkCall::new(VkOp::PoolsWaitFencesDestroy, result));
                reims_vgpu_observe::Emit::decline("vk_pools_destroy", &decline).fail_once(0);
            }
        }
        // Best-effort quiesce: wait every in-flight fence so no CB references
        // what we are about to destroy. On device loss the waits fail — the
        // teardown proceeds regardless, matching the recreate path.
        for slot in &self.encoder.slots {
            if slot.pending.is_some() && !matches!(slot.submission, SlotSubmission::Failed(_)) {
                if let Err(result) = device.wait_for_fences(&[slot.fence], true, FENCE_TIMEOUT_NS) {
                    let decline =
                        DrawError::VkCall(VkCall::new(VkOp::PoolsWaitFencesDestroy, result));
                    reims_vgpu_observe::Emit::decline("vk_pools_destroy", &decline).fail_once(0);
                }
            }
        }
        for slot in &mut self.encoder.slots {
            if let Some(pending) = slot.pending.take() {
                super::super::retire_guest_write_pages(&pending.visibility.guest_write_tokens);
                // The descriptor pool is destroyed below (frees every set);
                // move the owed transients into the live lists so the drains
                // below destroy them.
                self.encoder.staging_live.extend(pending.encoder.staging);
                self.encoder.gather_live.extend(pending.encoder.gather);
                self.encoder
                    .readback_multi_live
                    .extend(pending.encoder.readback);
                self.encoder.sampled_live.extend(pending.shared.sampled);
                self.encoder
                    .attachment_snapshot_live
                    .extend(pending.shared.attachment_snapshots);
                self.encoder
                    .storage_image_live
                    .extend(pending.shared.storage_images);
            }
        }
        self.encoder.in_flight = 0;
        // Every fence above was waited (or failed on a lost device, where the
        // handles die with the device anyway), so no slot can still be reading:
        // release the whole graveyard regardless of what each handle waits on.
        self.release_all_graveyard(device);
        for list in self.encoder.staging_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.shared.slabs, s);
            }
        }
        for s in self.encoder.staging_live.drain(..) {
            release_buffer_slot(device, &mut self.shared.slabs, s);
        }
        for list in self.encoder.gather_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.shared.slabs, s);
            }
        }
        for s in self.encoder.gather_live.drain(..) {
            release_buffer_slot(device, &mut self.shared.slabs, s);
        }
        for list in self.encoder.readback_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.shared.slabs, s);
            }
        }
        if let Some(s) = self.encoder.readback_live.take() {
            release_buffer_slot(device, &mut self.shared.slabs, s);
        }
        for s in self.encoder.readback_multi_live.drain(..) {
            release_buffer_slot(device, &mut self.shared.slabs, s);
        }
        // Leased slots are the one class here whose memory a live borrow may
        // still be reading, and freeing it unmaps that borrow's pointer — a
        // read after this line is a fault, not a stale pixel. So wait for the
        // holders rather than for a fence: the pool-owned outstanding count moves
        // with the holder, and a holder never blocks on the engine lock while
        // it holds a lease, so it always makes progress and this always
        // terminates.
        //
        // The bound exists because a teardown that hangs is worse than one that
        // races: it is generous against a scatter that takes ~1 ms, and expiry
        // is reported rather than assumed away.
        let lease_deadline = std::time::Instant::now() + LEASE_QUIESCE;
        while self
            .encoder
            .readback_lease_returns
            .outstanding
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
        {
            if std::time::Instant::now() >= lease_deadline {
                reims_vgpu_observe::Emit::decline(
                    "vk_pools_destroy",
                    &ReadbackLeaseQuiesceExpired {
                        outstanding: self
                            .encoder
                            .readback_lease_returns
                            .outstanding
                            .load(std::sync::atomic::Ordering::Acquire),
                        waited_ms: LEASE_QUIESCE.as_millis() as u64,
                    },
                )
                .fail_once(0);
                break;
            }
            std::thread::yield_now();
        }
        self.encoder.reclaim_returned_readback_leases();
        for l in self.encoder.readback_leased.drain(..) {
            release_buffer_slot(device, &mut self.shared.slabs, l.slot);
        }
        // Ad-hoc framebuffers name views owned by the targets and residents
        // destroyed below, and a framebuffer may not outlive its attachments —
        // so they go first, before anything drains a view out from under one.
        for (_, fb) in self.shared.ad_hoc_framebuffers.drain() {
            device.destroy_framebuffer(fb, None);
        }
        // Sampled / target / registry images are slab-backed: destroy the image
        // + view handles here, but their memory belongs to shared blocks freed
        // once by `self.shared.slab.destroy_all(device)` at the end — never a per-image
        // `vkFreeMemory` (that would double-free a block many images share).
        for s in self.shared.sampled_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.shared.attachment_snapshot_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for img in self.shared.target_free.drain() {
            device.destroy_image_view(img.view, None);
            device.destroy_image(img.image, None);
        }
        for s in self.encoder.sampled_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.encoder.attachment_snapshot_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.shared.sampled_cache.drain(..) {
            device.destroy_image_view(s.slot.view, None);
            device.destroy_image(s.slot.image, None);
        }
        self.shared.sampled_cache_bytes = 0;
        for s in self.shared.storage_image_free.drain() {
            device.destroy_image_view(s.view, None);
            match s.backing {
                StorageImageBacking::Dedicated(memory) => {
                    device.destroy_image(s.image, None);
                    device.free_memory(memory, None);
                }
                StorageImageBacking::HeapPlacement { .. } => {}
            }
        }
        for s in self.encoder.storage_image_live.drain(..) {
            device.destroy_image_view(s.view, None);
            match s.backing {
                StorageImageBacking::Dedicated(memory) => {
                    device.destroy_image(s.image, None);
                    device.free_memory(memory, None);
                }
                StorageImageBacking::HeapPlacement { .. } => {}
            }
        }
        for (_, resident) in self.shared.compute_storage_registry.drain() {
            device.destroy_image_view(resident.slot.view, None);
            match resident.slot.backing {
                StorageImageBacking::Dedicated(memory) => {
                    device.destroy_image(resident.slot.image, None);
                    device.free_memory(memory, None);
                }
                StorageImageBacking::HeapPlacement { .. } => {}
            }
        }
        for (_, placement) in self.shared.heap_placement_memory.drain() {
            device.destroy_image(placement.image, None);
            device.free_memory(placement.memory, None);
        }
        self.shared.compute_storage_order.clear();
        for (_, t) in self.shared.targets.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.shared.target_order.clear();
        if let Some(t) = self.shared.multisample_target.take() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        for (_, t) in self.shared.registry.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            for (_, view) in t.alternate_views {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.shared.guest_resident_authority.clear();
        self.shared.registry_order.clear();
        // Every fence above was waited, so nothing can still be reading or
        // writing an imported RAMBlock. Freeing the memory is what ends the
        // GPU's access to guest RAM, so it runs on every teardown path
        // including the ones that are otherwise giving up.
        self.shared.host_ram_imports.destroy_all(device);
        // Free every slab block now that all slab-backed images are destroyed.
        self.shared.slab.destroy_all(device);
        // Same for the HOST_VISIBLE upload blocks: every staging buffer bound
        // into one was destroyed above, so nothing can still reference the
        // block mappings this drops.
        self.shared.slabs.destroy_all(device);
        for slot in self.encoder.slots.drain(..) {
            device.destroy_fence(slot.fence, None);
        }
        self.encoder.cur = 0;
        // After the fences above, so nothing submitted can still name it, and
        // before the arena because its sets were allocated against this layout.
        // Freed before the arena that owns their blocks. Anything still here was
        // never submitted, or its fence has already retired above.
        let mut owed = std::mem::take(&mut self.encoder.scatter_dsets);
        // The recycle list holds only sets from entries whose fence retired,
        // which is the same "nothing can still name it" state this relies on.
        owed.append(&mut self.encoder.scatter_dset_free);
        self.encoder.desc_arena.free(device, &owed);
        if let Some(scatter) = self.shared.scatter.take() {
            scatter.destroy(device);
        }
        self.shared.scatter_refused = false;
        self.encoder.desc_arena.destroy(device);
        if self.encoder.cmd_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.encoder.cmd_pool, None);
            self.encoder.cmd_pool = vk::CommandPool::null();
        }
        self.encoder.initialized = false;
    }
}

/// How long [`ResourcePools::destroy_all`] waits for readback-lease holders.
///
/// A lease spans one scatter of one frame into guest pages — measured at
/// ~1 ms for a 1920x1080 composite — so this is three orders of magnitude of
/// headroom, not a tuned number. It is a liveness bound on a wait that should
/// never actually reach it, and reaching it is reported.
const LEASE_QUIESCE: std::time::Duration = std::time::Duration::from_millis(500);

/// A teardown gave up waiting for a readback lease to come back.
///
/// Nothing here is recoverable and nothing is retried: the destroy proceeds, so
/// a holder that wakes afterwards reads memory the device no longer owns. That
/// makes this the report of a broken invariant rather than of a degraded
/// result — a lease is only ever held across code that takes no engine lock, so
/// a holder that has not returned in half a second is one that acquired the
/// lock anyway or died holding it.
struct ReadbackLeaseQuiesceExpired {
    outstanding: usize,
    waited_ms: u64,
}

impl reims_vgpu_observe::Decline for ReadbackLeaseQuiesceExpired {
    fn slug(&self) -> &'static str {
        "readback_lease_quiesce_expired"
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("outstanding", self.outstanding.to_string()),
            ("waited_ms", self.waited_ms.to_string()),
        ]
    }
}
