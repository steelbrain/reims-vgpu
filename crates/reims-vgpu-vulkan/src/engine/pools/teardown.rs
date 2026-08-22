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
        self.open_pass = None;
        self.forget_pass_echo();
        // A batched submit transfers host ownership of its fence to the queue
        // thread until the driver's submit call returns. Teardown can be
        // reached through session release without a queue-wide barrier, so
        // reclaim each exact fence before the waits below touch it.
        for slot in &mut self.slots {
            let state = std::mem::replace(&mut slot.submission, SlotSubmission::HostOwned);
            let result = match state {
                SlotSubmission::HostOwned => Ok(()),
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
        for slot in &self.slots {
            if slot.pending.is_some() && !matches!(slot.submission, SlotSubmission::Failed(_)) {
                if let Err(result) = device.wait_for_fences(&[slot.fence], true, FENCE_TIMEOUT_NS) {
                    let decline =
                        DrawError::VkCall(VkCall::new(VkOp::PoolsWaitFencesDestroy, result));
                    reims_vgpu_observe::Emit::decline("vk_pools_destroy", &decline).fail_once(0);
                }
            }
        }
        for slot in &mut self.slots {
            if let Some(pending) = slot.pending.take() {
                super::super::retire_guest_write_pages(&pending.guest_write_tokens);
                // The descriptor pool is destroyed below (frees every set);
                // move the owed transients into the live lists so the drains
                // below destroy them.
                self.staging_live.extend(pending.staging);
                self.gather_live.extend(pending.gather);
                self.readback_multi_live.extend(pending.readback);
                self.sampled_live.extend(pending.sampled);
                self.attachment_snapshot_live
                    .extend(pending.attachment_snapshots);
                self.storage_image_live.extend(pending.storage_images);
            }
        }
        self.in_flight = 0;
        // Every fence above was waited (or failed on a lost device, where the
        // handles die with the device anyway), so no slot can still be reading:
        // release the whole graveyard regardless of what each handle waits on.
        self.release_graveyard(device, SlotMask::MAX);
        for list in self.staging_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.slabs, s);
            }
        }
        for s in self.staging_live.drain(..) {
            release_buffer_slot(device, &mut self.slabs, s);
        }
        for list in self.gather_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.slabs, s);
            }
        }
        for s in self.gather_live.drain(..) {
            release_buffer_slot(device, &mut self.slabs, s);
        }
        for list in self.readback_free.values_mut() {
            for s in list.drain(..) {
                release_buffer_slot(device, &mut self.slabs, s);
            }
        }
        if let Some(s) = self.readback_live.take() {
            release_buffer_slot(device, &mut self.slabs, s);
        }
        for s in self.readback_multi_live.drain(..) {
            release_buffer_slot(device, &mut self.slabs, s);
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
        self.reclaim_returned_readback_leases();
        for l in self.readback_leased.drain(..) {
            release_buffer_slot(device, &mut self.slabs, l.slot);
        }
        // Ad-hoc framebuffers name views owned by the targets and residents
        // destroyed below, and a framebuffer may not outlive its attachments —
        // so they go first, before anything drains a view out from under one.
        for (_, fb) in self.ad_hoc_framebuffers.drain() {
            device.destroy_framebuffer(fb, None);
        }
        // Sampled / target / registry images are slab-backed: destroy the image
        // + view handles here, but their memory belongs to shared blocks freed
        // once by `self.slab.destroy_all(device)` at the end — never a per-image
        // `vkFreeMemory` (that would double-free a block many images share).
        for s in self.sampled_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.attachment_snapshot_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for img in self.target_free.drain() {
            device.destroy_image_view(img.view, None);
            device.destroy_image(img.image, None);
        }
        for s in self.sampled_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.attachment_snapshot_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.sampled_cache.drain(..) {
            device.destroy_image_view(s.slot.view, None);
            device.destroy_image(s.slot.image, None);
        }
        self.sampled_cache_bytes = 0;
        for s in self.storage_image_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
            device.free_memory(s.memory, None);
        }
        for s in self.storage_image_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
            device.free_memory(s.memory, None);
        }
        for (_, resident) in self.compute_storage_registry.drain() {
            device.destroy_image_view(resident.slot.view, None);
            device.destroy_image(resident.slot.image, None);
            device.free_memory(resident.slot.memory, None);
        }
        self.compute_storage_order.clear();
        for (_, t) in self.targets.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.target_order.clear();
        if let Some(t) = self.multisample_target.take() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        for (_, t) in self.registry.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            for (_, view) in t.alternate_views {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.guest_resident_authority.clear();
        self.registry_order.clear();
        // Every fence above was waited, so nothing can still be reading or
        // writing an imported RAMBlock. Freeing the memory is what ends the
        // GPU's access to guest RAM, so it runs on every teardown path
        // including the ones that are otherwise giving up.
        self.host_ram_imports.destroy_all(device);
        // Free every slab block now that all slab-backed images are destroyed.
        self.slab.destroy_all(device);
        // Same for the HOST_VISIBLE upload blocks: every staging buffer bound
        // into one was destroyed above, so nothing can still reference the
        // block mappings this drops.
        self.slabs.destroy_all(device);
        for slot in self.slots.drain(..) {
            device.destroy_fence(slot.fence, None);
        }
        self.cur = 0;
        // After the fences above, so nothing submitted can still name it, and
        // before the arena because its sets were allocated against this layout.
        // Freed before the arena that owns their blocks. Anything still here was
        // never submitted, or its fence has already retired above.
        let mut owed = std::mem::take(&mut self.scatter_dsets);
        // The recycle list holds only sets from entries whose fence retired,
        // which is the same "nothing can still name it" state this relies on.
        owed.append(&mut self.scatter_dset_free);
        self.desc_arena.free(device, &owed);
        if let Some(scatter) = self.scatter.take() {
            scatter.destroy(device);
        }
        self.scatter_refused = false;
        self.desc_arena.destroy(device);
        if self.cmd_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.cmd_pool, None);
            self.cmd_pool = vk::CommandPool::null();
        }
        self.initialized = false;
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
