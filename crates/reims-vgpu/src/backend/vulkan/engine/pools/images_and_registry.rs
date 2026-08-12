//! The resident-image registry and the storage images beside it — the
//! [`ResourcePools`] methods whose unit is a *guest identity* rather than a
//! slot.
//!
//! A registry entry outlives any one submission. The guest names the same
//! surface across frames, so an entry is keyed by `TargetIdentity`, held alive
//! by a pin count, and reclaimed by an idle drain or an allocation failure
//! rather than by a fence — which is why none of this sits with the ring in
//! [`super::submission_and_buffers`].
//!
//! The two meet in one place: the idle-drain planner over there picks its
//! victims out of the registry order maintained here.
//!
//! `use super::*` is the seam. This is an `impl` chapter of the module that
//! declares `ResourcePools` and owns its fields, not a layer beneath it.

use super::*;

/// Band how long a resident had gone untouched before something read it,
/// against the cutoff that would have destroyed it.
///
/// The idle drain terminally destroys any non-pinned resident untouched for
/// `IDLE_TARGET_AGE_MS`, and nothing recreates a resident's content — so a draw
/// that samples one afterwards refuses permanently. What stops that today is
/// that a resident being read is touched by the read
/// ([`ResourcePools::registry_note_sampled_use`]), which only helps while the
/// gaps between reads stay under the cutoff. A guest that renders a layer, has
/// it occluded for longer than that, and then reveals it is the shape that loses
/// it.
///
/// `sampled_resident_missing` reading zero cannot say whether that is far away
/// or one slow frame away — it is a drop counter, and this project's own rule is
/// that a drop counter reading zero is not a measurement: a gap that peaks at
/// 50 ms and one that peaks at 1900 ms both report zero, and only one of them
/// says the cutoff has headroom. This is the reach that separates them, in time
/// rather than in slots, and it is the same instrument the bind tables already
/// have as their `reach_route` bands.
///
/// Read `resident_resample_past_cutoff` as the alarm: a resident that was read
/// after sitting longer than its own drain cutoff survived only because the
/// drain is throttled to `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass and had not
/// reached it yet. A non-zero reading there is the argument for changing the
/// drain — it does not mean content was lost, it means the margin is gone.
///
/// Bands are fractions of `IDLE_TARGET_AGE_MS` rather than absolute
/// milliseconds, so retuning the cutoff moves them with it and no reading is
/// ever quoted against a bound it did not come from. Division rather than
/// multiplication so a large gap cannot overflow the comparison.
///
/// # First reading, and it is not the comfortable one
///
/// Driven x86/PCI boot, `web-content-probe --churn 1`, whole run, against
/// `IDLE_TARGET_AGE_MS` = 2000 ms:
///
/// ```text
///   lt_eighth_cutoff   (<250 ms)      24643
///   lt_quarter_cutoff  (250-499 ms)     413
///   lt_half_cutoff     (500-999 ms)       5
///   under_cutoff       (1000-1999 ms)     1
///   past_cutoff        (>=2000 ms)        0
/// ```
///
/// The distribution is overwhelmingly under an eighth of the cutoff, which is
/// the answer that would have been assumed. The reading that matters is the
/// **1**: one resample arrived after its resident had sat between 1000 and
/// 1999 ms, so somewhere between half the budget and none of it was left. The
/// drain destroys at `last_touch_ms <= now - cutoff` and is throttled to
/// `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass, so that resident was not yet a
/// victim — but it is not the case that this workload stays an order of
/// magnitude clear of the cutoff, which is what `sampled_resident_missing=0`
/// on its own invites you to conclude.
///
/// Finer bands past the half mark are what would say whether that one sample sat
/// at 1.0 s or at 1.9 s, and those are very different margins. This does not
/// resolve it; it establishes that the question is live.
///
/// # `past_cutoff` has since fired, and it does not mean what it was written to
///
/// Two later driven x86/PCI boots, window-drag probe, quiesced, the second on a
/// guest running a page animation:
///
/// ```text
///                        boot A     boot B
///   resamples, all bands   48738     392534
///   past_cutoff                1          4
///   resample_peak_ms        2007       2005
/// ```
///
/// So both signals the top of this doc and [`IDLE_TARGET_AGE_MS`] name as the
/// argument for changing the drain are now non-zero, routinely, on ordinary
/// desktop work. Read that as the margin being gone, which it is — and not as
/// work being at risk, which it no longer is. Two things changed underneath the
/// alarm since it was written:
///
/// - **The class it guarded is closed.** Every draw that stores into a target
///   marks it sole-copy ([`ResourcePools::registry_mark_ready`]), and only a
///   writeback clears it, so a resident the drain is *allowed* to take has
///   demonstrably been copied out to the guest's own pages. What a reclaim costs
///   is the re-fetch, not the pixels — and `sampled_resident_missing` was 0 on
///   both boots above, as it must be.
/// - **The re-fetch is the reading worth having**, and it is much larger than
///   these four: `t11sample_reclaimed_from_pages` was 2085 and 1937 on the same
///   two boots, with 88.5 % of boot A's coming back more than 8 s after the
///   destroy. The drain is churning residents the workload returns to, while
///   `peak_mib=81` and `slab_mib=39/136` say VRAM was under no pressure at all.
///
/// Neither of those is fixed by moving this cutoff, and
/// [`IDLE_TARGET_AGE_MS`]'s own history says why a population gate is not the
/// answer either. What they change is what a `past_cutoff` reading is evidence
/// *of*: a cost, measurable in re-fetches, rather than an imminent loss.
///
/// # Reading the count against `resident_samples`
///
/// The band total is about **twice** `sampled_gpu_binds` (25062 against 12531 on
/// the run above, exactly 2x). That is not a discrepancy:
/// [`ResourcePools::registry_note_sampled_use`] is called from two sites per
/// draw — the pre-pass loop that marks every sampled target before the render
/// target is ensured, and the resolve loop that binds them — while
/// `sampled_gpu_binds` increments once per bind. So these bands count *touches*
/// and `resident_samples` counts *binds*. Do not divide one by the other and
/// call the result a rate.
fn resident_resample_band(idle_ms: u64) -> &'static str {
    let cutoff = IDLE_TARGET_AGE_MS;
    if idle_ms < cutoff / 8 {
        "resident_resample_lt_eighth_cutoff"
    } else if idle_ms < cutoff / 4 {
        "resident_resample_lt_quarter_cutoff"
    } else if idle_ms < cutoff / 2 {
        "resident_resample_lt_half_cutoff"
    } else if idle_ms < cutoff {
        "resident_resample_under_cutoff"
    } else {
        "resident_resample_past_cutoff"
    }
}

/// Everything a creation site knows about a resident it has just built.
///
/// The stored [`ResidentTargetSlot`] is this plus what the registry owns and a
/// creation site does not: the birth state and the two LRU clocks. Handing over
/// this rather than a finished slot is what stops an arm getting either wrong,
/// and it is why [`ResourcePools::register_resident`] takes no `&mut` slot to
/// patch afterwards.
struct NewResident {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// `vk::Framebuffer::null()` for a resident that is never bound as a
    /// standalone single-RT target — see
    /// [`ResidentTargetSlot::owed_framebuffer`], which is what every destroy
    /// path asks instead of testing this field itself.
    framebuffer: vk::Framebuffer,
    /// Null exactly when `framebuffer` is: the pass a per-slot framebuffer was
    /// built against, and the thing `registry_ensure` compares to decide
    /// whether a reused image needs its framebuffer rebuilt.
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
    generation: u64,
    color_format: vk::Format,
}

impl ResourcePools {
    pub(crate) unsafe fn acquire_storage_image(
        &mut self,
        ctx: &DeviceContext,
        key: StorageImageKey,
        counters: &EngineCounters,
    ) -> Result<StorageImageSlot, DrawError> {
        if let Some(slot) = self.storage_image_free.take(&key) {
            self.storage_image_live.push(StorageImageSlot {
                image: slot.image,
                memory: slot.memory,
                view: slot.view,
                key: slot.key,
            });
            return Ok(slot);
        }
        let format = key.format.vk_format();
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: key.width.max(1),
                        height: key.height.max(1),
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(if key.sampled_only {
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST
                    } else {
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC
                    })
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStorageImage, e)))?;
        counters.note_create();
        let req = ctx.device.get_image_memory_requirements(image);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, req.size, MemoryClass::DeviceLocal)
            .ok_or({
                DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForStorageImage {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        // A compute storage image takes a dedicated `vkAllocateMemory` rather
        // than a slab suballocation, so it does not pass through
        // `bind_image_slab` and does not inherit its out-of-memory retry. It
        // gets the same one here, and for the same reason: a refusal costs the
        // guest a dispatch, and this device is usually still holding recycle
        // pools it was entitled to give back.
        let alloc = |ctx: &DeviceContext| {
            allocate_memory_timed(
                ctx,
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                AllocSite::StorageImage,
            )
        };
        let memory = match alloc(ctx) {
            Ok(m) => m,
            Err(first)
                if (first == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    || first == vk::Result::ERROR_OUT_OF_HOST_MEMORY)
                    && self.reclaim_pools_for_allocation_retry(ctx) > 0 =>
            {
                alloc(ctx).map_err(|_| {
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsAllocStorageImage, first))
                })?
            }
            Err(e) => {
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsAllocStorageImage,
                    e,
                )));
            }
        };
        counters.note_alloc();
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindStorageImage, e))
            })?;
        let view = ctx
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_subresource_range()),
                None,
            )
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStorageImageView, e))
            })?;
        counters.note_create();
        let slot = StorageImageSlot {
            image,
            memory,
            view,
            key,
        };
        self.storage_image_live.push(slot);
        Ok(slot)
    }

    pub(crate) fn recycle_storage_images(&mut self) {
        for slot in self.storage_image_live.drain(..) {
            self.storage_image_free.push_uncapped(slot.key, slot);
        }
    }

    pub(crate) unsafe fn acquire_resident_storage_image(
        &mut self,
        ctx: &DeviceContext,
        identity: ComputeStorageResidencyKey,
        key: StorageImageKey,
        seed_generation: u32,
        counters: &EngineCounters,
    ) -> Result<ResidentStorageImageUse, DrawError> {
        // A shape change re-keys the identity, and one identity holds one slot,
        // so the old image is destroyed. Every other removal in this registry
        // skips a pinned resident — the cap sweep and the age drain both do —
        // because a pin means the content owes a deferred writeback and exists
        // nowhere but that image. This path did not, so a re-shape between a
        // Store and its flush discarded accepted guest output with nothing said,
        // surfacing later and elsewhere as `StorageReadResidentAbsent`. Refuse
        // instead: the pin clears when the writeback lands and the next dispatch
        // re-keys normally, so this holds the request rather than ending it.
        if let Some(decline) = self.compute_rekey_refusal(&identity, key) {
            return Err(DrawError::ComputeExecution(decline));
        }
        if self
            .compute_storage_registry
            .get(&identity)
            .is_some_and(|resident| resident.slot.key != key)
        {
            if let Some(old) = self.remove_compute_storage_resident(&identity) {
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.slot.image,
                        view: old.slot.view,
                        memory: old.slot.memory,
                    },
                );
            }
        }
        let now = self.idle_clock_ms;
        if let Some(resident) = self.compute_storage_registry.get_mut(&identity) {
            resident.last_touch_ms = now;
            return Ok(ResidentStorageImageUse {
                slot: resident.slot,
                access: resident.access,
                generation_match: resident.generation == seed_generation,
            });
        }

        // Census only. A slot count used to trim this population here and its
        // losses were terminal — see `recoverable_compute_storage_residents`.
        self.note_compute_storage_reach();

        // Reuse the common allocator, then detach its bookkeeping copy from
        // the transient live list: the registry now owns this allocation.
        //
        // Out of memory is the one refusal worth a second attempt, exactly as at
        // the sibling target registry's admission: this device is usually still
        // holding residents it was entitled to give back. This is also the point
        // the retired cap sweep destroyed residents at, which is what makes
        // retiring live ones safe here — the caller holds no handle from this
        // acquire yet, and no sampled source has been resolved.
        let slot = match self.acquire_storage_image(ctx, key, counters) {
            Ok(slot) => slot,
            Err(error) if error.out_of_memory() => {
                if self.reclaim_compute_storage_for_allocation_retry(ctx) == 0 {
                    return Err(error);
                }
                self.acquire_storage_image(ctx, key, counters)
                    .map_err(|_| error)?
            }
            Err(error) => return Err(error),
        };
        let live = self.storage_image_live.pop().ok_or({
            DrawError::ComputeExecution(ComputeExecutionDecline::ResidentAllocatorLiveSlotMissing {
                identity,
                width: key.width,
                height: key.height,
                format: key.format,
            })
        })?;
        debug_assert_eq!(live.image, slot.image);
        self.compute_storage_registry.insert(
            identity,
            ResidentStorageImageSlot {
                slot,
                generation: 0,
                access: ResidentAccess::Untouched,
                pinned: false,
                // No dispatch has written it, so it holds no guest work to lose.
                gpu_only_content: false,
                last_touch_ms: now,
            },
        );
        self.compute_storage_order.push_back(identity);
        Ok(ResidentStorageImageUse {
            slot,
            access: ResidentAccess::Untouched,
            generation_match: false,
        })
    }

    /// The refusal owed when `identity` is already held at a different image
    /// shape by a resident that still owes a deferred writeback, or `None` when
    /// the re-key is safe to perform.
    ///
    /// Split out from [`ResourcePools::acquire_resident_storage_image`] so the
    /// pin check is unit-testable without a device, the same reason
    /// `take_aged_storage_residents` is split from its disposal half.
    pub(crate) fn compute_rekey_refusal(
        &self,
        identity: &ComputeStorageResidencyKey,
        key: StorageImageKey,
    ) -> Option<ComputeExecutionDecline> {
        let held = self
            .compute_storage_registry
            .get(identity)
            .filter(|resident| resident.pinned && resident.slot.key != key)?;
        Some(ComputeExecutionDecline::ResidentRekeyWouldDropPinned {
            identity: *identity,
            held_width: held.slot.key.width,
            held_height: held.slot.key.height,
            held_format: held.slot.key.format,
            wanted_width: key.width,
            wanted_height: key.height,
            wanted_format: key.format,
        })
    }

    /// Every compute-storage resident the device may destroy without losing
    /// guest work: not pinned, and not the only copy of its own contents.
    ///
    /// The sibling of [`Self::recoverable_residents`] over the other registry,
    /// and it answers the same question — what could be given back if it had to
    /// be, which is what an allocation failure asks.
    ///
    /// # This population used to be trimmed by a slot count, and that lost work
    ///
    /// `COMPUTE_STORAGE_REGISTRY_CAP = 64` swept the least-recently-used entry
    /// on every admission past the count. Its losses were **terminal and worse
    /// than the target registry's**: nothing recreates a compute-storage
    /// resident's contents, so a dispatch that later reads a destroyed identity
    /// refuses with `ResidentSampleAbsent` or `ResidentSeedGenerationLost`
    /// rather than paying a re-upload. `compute_storage_cap_evictions` counted
    /// exactly that, and it counted destroyed guest work.
    ///
    /// The count is gone and the allocation is the bound, as on the sibling. The
    /// prerequisite that was missing — and the reason the two registries did not
    /// change together — is that `reclaim_for_allocation_retry` gave back target
    /// residents and recycle pools and nothing from here, so removing the count
    /// alone would have left this population with the age drain trimming it and
    /// nothing to hand back when an allocation failed.
    /// [`Self::reclaim_compute_storage_for_allocation_retry`] is that half.
    ///
    /// Ordering is `compute_storage_order`, so the result is deterministic and
    /// oldest-created first. Nothing selects a single victim any more: the
    /// reclaim takes the whole set at once, because by the time it runs the
    /// question is whether the dispatch survives at all.
    pub(super) fn recoverable_compute_storage_residents(&self) -> Vec<ComputeStorageResidencyKey> {
        self.compute_storage_order
            .iter()
            .filter(|identity| {
                self.compute_storage_registry
                    .get(*identity)
                    .is_some_and(|resident| !resident.pinned && !resident.gpu_only_content)
            })
            .copied()
            .collect()
    }

    /// Give back every compute-storage resident that is neither pinned nor the
    /// only copy of its contents, plus the recycle pools, for a retry after the
    /// device refused an allocation. Returns how many were released.
    ///
    /// Deliberately *not* folded into [`Self::reclaim_for_allocation_retry`],
    /// and not the reverse either. Each registry's reclaim runs only at its own
    /// admission, which is the one point where the caller provably holds no
    /// handle it is about to use — the property `333c126e` established the hard
    /// way when hoisting the fuller reclaim to every allocation site segfaulted
    /// QEMU. A draw can bind a compute-storage resident as a sampled source, so
    /// a target admission reclaiming this registry would have exactly that
    /// defect.
    ///
    /// # It works, and it was measured by breaking the allocator on purpose
    ///
    /// No boot has produced a real allocation failure here, and the driven
    /// web-content boot this device is usually measured on reports a *single*
    /// compute-storage resident — so that workload cannot reach this at all.
    /// Driven instead with temporary fault injection (not in the tree): every
    /// 10th `create_storage_image` failing its `vkAllocateMemory` for the whole
    /// of that invocation, including the pools-only retry inside it, so the
    /// refusal reaches this handler. Run against
    /// `every_admitted_compute_storage_resident_survives_past_the_retired_slot_cap`,
    /// whose 80 admissions are enough population for a reclaim to have something
    /// to give:
    ///
    /// ```text
    ///                     injected failures   dispatches lost
    ///   with this retry                  16                 0
    ///   without it                       16    the first one
    /// ```
    ///
    /// Every one of the sixteen was absorbed. The same run's *seed-upload*
    /// assertion does fail with the retry in place, and that is the reclaim
    /// working rather than a defect: residents really were given back, so their
    /// next dispatch re-uploaded a seed. Under real pressure that is the trade —
    /// a re-upload instead of a refused dispatch.
    ///
    /// One refusal is **not** absorbed and should not be: an injected failure on
    /// the very first dispatch of a boot, where the registry and the pools are
    /// both empty. This returns 0, the caller refuses with the driver's own
    /// error, and that is a GPU with nothing left to give doing the only correct
    /// thing.
    unsafe fn reclaim_compute_storage_for_allocation_retry(
        &mut self,
        ctx: &DeviceContext,
    ) -> usize {
        let trimmed = self.reclaim_pools_for_allocation_retry(ctx);
        let victims = self.recoverable_compute_storage_residents();
        let mut freed = 0;
        for victim in &victims {
            if let Some(old) = self.remove_compute_storage_resident(victim) {
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.slot.image,
                        view: old.slot.view,
                        memory: old.slot.memory,
                    },
                );
                freed += 1;
            }
        }
        // Always visible, for the reason the sibling's line is: a device that has
        // had to do this is one whose next allocation is also likely to fail.
        crate::observe::fail(format!(
            "vram_compute_storage_reclaim_retry residents={freed} recycled={trimmed} \
             held_bytes={} sole_copy={} live={} (an allocation was refused; gave \
             back everything that is neither pinned nor the only copy of its contents)",
            self.slab.held_bytes().0,
            self.compute_storage_sole_copy.count,
            self.compute_storage_registry.len(),
        ));
        freed + trimmed
    }

    /// Drop the compute-storage resident under `identity` from both halves of
    /// the registry, folding it out of the maintained totals, and hand the slot
    /// back so the caller can dispose or recycle it.
    ///
    /// The one removal path, for the reason [`Self::unregister_resident`] is the
    /// one on the sibling registry: the map and the order are a single structure
    /// split for lookup and for order, and three call sites each doing their own
    /// pair is how one of them comes to forget the totals. Removal is the only
    /// way a sole-copy resident leaves — both reclaim paths refuse to select
    /// one, so what arrives here carrying the flag is a re-key or a guest-side
    /// delete.
    pub(super) fn remove_compute_storage_resident(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<ResidentStorageImageSlot> {
        let old = self.compute_storage_registry.remove(identity);
        self.compute_storage_order.retain(|entry| entry != identity);
        let old = old?;
        if old.gpu_only_content {
            let bytes = Self::storage_slot_bytes(&old);
            Self::fold_totals(&mut self.compute_storage_sole_copy, bytes, false);
        }
        Some(old)
    }

    /// Fold the current compute-storage sole-copy population into its high-water
    /// band. Called at the top of the capacity walk — the one point every
    /// admission passes — for the reason [`Self::note_registry_reach`] is.
    fn note_compute_storage_reach(&mut self) {
        self.compute_storage_sole_copy_peak = Self::high_water(
            self.compute_storage_sole_copy_peak,
            self.compute_storage_sole_copy,
        );
    }

    /// Record that something read this resident, refreshing its reclaim stamp.
    ///
    /// Reading a resident is using it. The three read-only accessors below all
    /// mean "a guest chain is about to consume this image" — the stage-time
    /// guest-read skip, the copy-on-sample gate, and the flush/sample snapshot —
    /// so a produce-once/sample-many resident that is never dispatched into
    /// again is in continuous use while looking stone-cold to both reclaim
    /// rules, which read `last_touch_ms`: the cap sweep takes the minimum and
    /// the age drain compares it against its cutoff.
    ///
    /// The sibling target registry had exactly this defect and names it at its
    /// own call site: "aging it out between two attempts is how a recoverable
    /// not-ready became a permanent missing." Here the loss is a refused
    /// dispatch — `ResidentSampleAbsent` or `ResidentSeedGenerationLost` — so
    /// the stamp is written by the accessors themselves rather than by their
    /// callers, because a caller that forgets is indistinguishable from this
    /// bug.
    fn note_compute_resident_use(&mut self, identity: &ComputeStorageResidencyKey) {
        let touch = self.idle_clock_ms;
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.last_touch_ms = touch;
        }
    }

    /// Record the dispatch that just wrote this resident storage image.
    ///
    /// The access is not a parameter because there is only one: every executed
    /// dispatch ends by copying the image out to its readback buffer, and the
    /// barrier that arms that copy chains off the dispatch's own `SHADER_WRITE`.
    /// So what a later reader must wait for is the transfer read, and the layout
    /// it must name is that read's `TRANSFER_SRC_OPTIMAL` — both carried by
    /// [`ResidentAccess::TransferRead`].
    pub(crate) fn mark_resident_storage_image(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        generation: u32,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.generation = generation;
            resident.access = ResidentAccess::TransferRead;
        }
        // The dispatch just wrote this image, so nothing outside it holds the
        // result yet.
        self.set_compute_sole_copy(identity, true);
    }

    /// Set a compute-storage resident's
    /// [`ResidentStorageImageSlot::gpu_only_content`] and keep the maintained
    /// totals in step. The single writer of that field on a live slot, for the
    /// reason [`Self::set_sole_copy`] is on the sibling registry.
    fn set_compute_sole_copy(&mut self, identity: &ComputeStorageResidencyKey, sole: bool) -> bool {
        let Some(resident) = self.compute_storage_registry.get_mut(identity) else {
            return false;
        };
        if resident.gpu_only_content == sole {
            return true;
        }
        resident.gpu_only_content = sole;
        let bytes = Self::storage_slot_bytes(resident);
        Self::fold_totals(&mut self.compute_storage_sole_copy, bytes, sole);
        // Same reason as the sibling's: this population grows on
        // `mark_resident_storage_image` and the capacity walk that also folds it
        // runs on admission, so sampling only there lags by a dispatch.
        if sole {
            self.compute_storage_sole_copy_peak = Self::high_water(
                self.compute_storage_sole_copy_peak,
                self.compute_storage_sole_copy,
            );
        }
        true
    }

    /// Level-0 bytes of a compute-storage resident's image.
    ///
    /// The same lower bound `slot_attachment_bytes` is on the sibling registry —
    /// texel footprint from the decoded geometry, blind to tiling padding and
    /// allocator rounding — and quoted for the same purpose, which is deciding
    /// whether a bound is too loose. A lower bound is the safe direction there.
    pub(super) fn storage_slot_bytes(resident: &ResidentStorageImageSlot) -> u64 {
        let key = resident.slot.key;
        u64::from(key.width) * u64::from(key.height) * key.format.bytes_per_texel() as u64
    }

    /// Record that a compute-storage resident's pixels have been copied out to
    /// the host, so the reclaim paths may take it.
    ///
    /// Only for a readback that actually landed. Both unpin paths that abort —
    /// `flush_storage_one`'s failure arm and `lifecycle`'s window-cleared arm —
    /// leave the flag set, because nothing wrote anything.
    pub(crate) fn note_compute_storage_copied_out(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> bool {
        self.set_compute_sole_copy(identity, false)
    }

    /// Record that the guest deleted the object this resident's content belonged
    /// to, so there is no longer any guest work here to protect.
    ///
    /// Distinct from [`Self::note_compute_storage_copied_out`] because the reason
    /// is opposite — nothing was written anywhere, the content simply stopped
    /// mattering — and because without it `retire_linear_residents` would unpin a
    /// resident that the reclaim paths then refuse forever, turning a fix for a
    /// pinned-VRAM leak into a sole-copy one.
    pub(crate) fn note_compute_storage_content_retired(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> bool {
        self.set_compute_sole_copy(identity, false)
    }

    /// Pin/unpin a resident against LRU eviction (deferred-writeback content
    /// whose only copy is the GPU image). No-op for an absent identity.
    pub(crate) fn pin_resident_storage(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        pinned: bool,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.pinned = pinned;
        }
    }

    /// Generation of a resident compute storage image, if one is registered.
    /// Used by the runtime to decide a stage-time guest-read skip.
    ///
    /// Takes `&mut self` to record the read — see
    /// [`ResourcePools::note_compute_resident_use`]. A skip taken against this
    /// answer means the dispatch is about to consume the resident, which is the
    /// definition of using it.
    pub(crate) fn compute_resident_generation(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<u32> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry
            .get(identity)
            .map(|resident| resident.generation)
    }

    /// Generation + engine format of a resident compute storage image, if one
    /// is registered. Read-only — used by the runtime to decide a stage-time
    /// copy-on-sample skip (the format must match what the sampled view will
    /// bind, or the engine's resident-bind shape guard would fail every run).
    pub(crate) fn compute_resident_sample_source(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry
            .get(identity)
            .map(|resident| (resident.generation, resident.slot.key.format))
    }

    /// Snapshot of a resident storage image for a copy-on-sample source:
    /// `(image, key, generation, what last touched it)`.
    pub(crate) fn compute_resident_snapshot(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(vk::Image, StorageImageKey, u32, ResidentAccess)> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry.get(identity).map(|resident| {
            (
                resident.slot.image,
                resident.slot.key,
                resident.generation,
                resident.access,
            )
        })
    }

    // --- Target registry (workstream D) ------------------------------------

    pub(crate) fn registry_get(&self, identity: &TargetIdentity) -> Option<&ResidentTargetSlot> {
        self.registry.get(identity)
    }

    /// Forget the resident registered under `identity`, recording `why`, and
    /// hand back the slot that was removed.
    ///
    /// Split out from [`Self::retire_resident`] for the same reason
    /// [`Self::recoverable_residents`] is split out of
    /// [`Self::reclaim_for_allocation_retry`]: retiring needs a live
    /// `DeviceContext` to dispose what it removes, and the bookkeeping — which
    /// is the part that was diverging — is worth testing without a GPU.
    ///
    /// Every path that removes a live entry comes through here, including the
    /// idle drain in [`super::submission_and_buffers`], which disposes on its
    /// own terms and so is not a [`Self::retire_resident`] caller. It is the
    /// death counterpart of [`Self::register_resident`], and the pair is why
    /// `registry` and `registry_order` cannot fall out of step.
    ///
    /// `registry_order` is pruned whether or not the map held the entry, which
    /// is what every caller did around its own copy. Nothing is recorded for an
    /// identity that held no resident: [`Self::prior_reclaim`] deliberately does
    /// not guess between "never held one" and "reclaimed too long ago", and a
    /// record for a removal that did not happen would make it guess wrong.
    pub(super) fn unregister_resident(
        &mut self,
        identity: &TargetIdentity,
        why: ResidentReclaim,
    ) -> Option<ResidentTargetSlot> {
        let old = self.registry.remove(identity);
        self.registry_order.retain(|k| k != identity);
        let old = old?;
        if old.pin_count == 0 {
            self.registry_non_pinned_adjust(Self::slot_attachment_bytes(&old), false);
        }
        // A sole-copy slot never leaves here through a *reclaim* — both reclaim
        // paths skip it — but it does leave through the recreate arms, where the
        // guest itself replaced the resource under this identity. Not folding it
        // out there would leak the population upward until it read as a ceiling
        // that was never reached.
        if old.gpu_only_content {
            self.registry_sole_copy_adjust(Self::slot_attachment_bytes(&old), false);
        }
        self.note_resident_reclaimed(identity, why);
        Some(old)
    }

    /// Hand a resident's framebuffer to the deferred-destroy path, if it has
    /// one. [`ResidentTargetSlot::owed_framebuffer`] is where "if" is decided,
    /// and why.
    ///
    /// # Safety
    /// The caller must already have taken this framebuffer out of the registry,
    /// or be about to overwrite the field it came from. The graveyard frees it
    /// once the ring says no command buffer still references it.
    pub(super) unsafe fn dispose_owed_framebuffer(
        &mut self,
        device: &ash::Device,
        owed: Option<vk::Framebuffer>,
    ) {
        if let Some(fb) = owed {
            self.dispose(device, DeferredHandle::Framebuffer(fb));
        }
    }

    /// Store a newly created resident under `identity` and put it at the back
    /// of the LRU order.
    ///
    /// One home for a resident's *birth*, as [`Self::unregister_resident`] is
    /// one home for its death. Both `registry_ensure*` arms wrote all of this
    /// out, and it is three rules rather than one:
    ///
    /// - **The birth state.** Nothing has drawn into an image created a line
    ///   ago, so it carries no content stamp and no epoch; nothing has
    ///   transitioned it, so its layout is `UNDEFINED`; and no window holds it,
    ///   so it is unpinned. These are not defaults a creation site may pick —
    ///   `registry_mark_ready`, the type-11 LOAD gate and the idle drain each
    ///   read one of them, and an arm that guessed differently would be
    ///   answering a question the others think they already asked.
    /// - **The idle-drain clock belongs to the registry.** `last_touch_ms` comes
    ///   from `idle_clock_ms`, which the poll heartbeat advances. A creation site
    ///   that stamped its own value would age against a different clock than the
    ///   drain reads.
    /// - **`registry` and `registry_order` are written together.** They are one
    ///   structure split for lookup and for order. An entry in the map but not
    ///   the order is a resident no sweep can ever choose; one in the order but
    ///   not the map is a victim that frees nothing.
    fn register_resident(&mut self, identity: &TargetIdentity, new: NewResident) {
        let last_touch_ms = self.idle_clock_ms;
        self.registry.insert(
            identity.clone(),
            ResidentTargetSlot {
                image: new.image,
                memory: new.memory,
                view: new.view,
                framebuffer: new.framebuffer,
                render_pass: new.render_pass,
                width: new.width,
                height: new.height,
                generation: new.generation,
                content_ready: false,
                content_epoch: None,
                access: ResidentAccess::Untouched,
                color_format: new.color_format,
                pin_count: 0,
                // Nothing has drawn into it, so it holds no guest work to lose.
                // A recycled image arrives here too, and its stale contents are
                // not this identity's content.
                gpu_only_content: false,
                last_touch_ms,
            },
        );
        self.registry_order.push_back(identity.clone());
        // Born unpinned (see the birth-state rule above), so it joins the
        // non-pinned totals unconditionally.
        let bytes = self
            .registry
            .get(identity)
            .map(Self::slot_attachment_bytes)
            .unwrap_or(0);
        self.registry_non_pinned_adjust(bytes, true);
    }

    /// Drop the resident registered under `identity`, recording `why`, returning
    /// its image/memory/view to `target_free` and its framebuffer to the
    /// graveyard. Returns the slot that was removed, or `None` when nothing was
    /// registered.
    ///
    /// The recycling exit for a live registry entry: the two `registry_ensure*`
    /// recreate arms and [`Self::reclaim_for_allocation_retry`] all take it, and
    /// were copies of one another before they did. The MRT-secondary path recorded
    /// no reclaim reason at all, so a later draw whose sampled source that path
    /// had recreated could not be told "taken from under you" from "never
    /// existed", which is the whole point of
    /// [`Self::note_resident_reclaimed`]. The primary path was the one that
    /// disposed `old.framebuffer` without asking whether the slot had one.
    ///
    /// It is not the *only* exit, and reading it as one is how the fourth stayed
    /// out of step. The idle drain destroys rather than recycles and does not
    /// count a `target_evict`, both deliberately, so it cannot come through
    /// here — what it shares is [`Self::unregister_resident`] and
    /// [`ResidentTargetSlot::owed_framebuffer`], which is why the bookkeeping
    /// and the null question are each their own function rather than lines in
    /// this body.
    unsafe fn retire_resident(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        why: ResidentReclaim,
        counters: &EngineCounters,
    ) -> Option<ResidentTargetSlot> {
        let old = self.unregister_resident(identity, why)?;
        self.dispose_owed_framebuffer(&ctx.device, old.owed_framebuffer());
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
        Some(old)
    }

    /// Ensure a resident target exists for `identity` with the given geometry + pass.
    /// Image/memory persist across Load vs Clear render-pass changes; only the
    /// framebuffer is rebuilt when the pass handle differs.
    ///
    /// This used to take a `protect` identity — a same-draw GPU seed source to
    /// shield from the capacity sweep the admission ran. Nothing is swept on
    /// admission any more, so there is nothing to shield from and the parameter
    /// is gone rather than ignored. See [`Self::recoverable_residents`].
    #[allow(
        clippy::too_many_arguments,
        reason = "resident creation mirrors the target identity, geometry, pass and format"
    )]
    pub(crate) unsafe fn registry_ensure(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
        generation: u64,
        format: vk::Format,
        counters: &EngineCounters,
    ) -> Result<&ResidentTargetSlot, DrawError> {
        // The format arrives resolved rather than as a channel-order flag, and
        // from the same variable that built `render_pass`'s key — an image and
        // the pass it is attached to must name one format, and deriving it
        // twice from a shared input is how they drift apart.
        //
        // Compatible geometry + gen + format: reuse image; rebuild FB if pass
        // changed. A format change must recreate the image, not just the
        // framebuffer — an RGBA image under a BGRA pass is invalid.
        if let Some(slot) = self.registry.get(&identity) {
            if slot.reusable_for(width, height, generation, format) {
                if slot.render_pass == render_pass {
                    counters
                        .gpu_load_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let touch = self.idle_clock_ms;
                    let slot = self.registry.get_mut(&identity).unwrap();
                    slot.last_touch_ms = touch;
                    return Ok(slot);
                }
                // Same image, new pass → recreate framebuffer only.
                let view = slot.view;
                let old_fb = slot.owed_framebuffer();
                let attachments = [view];
                let framebuffer = ctx
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .map_err(|e| {
                        DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryFramebuffer, e))
                    })?;
                counters.note_create();
                self.dispose_owed_framebuffer(&ctx.device, old_fb);
                let slot = self.registry.get_mut(&identity).unwrap();
                slot.framebuffer = framebuffer;
                slot.render_pass = render_pass;
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(self.registry.get(&identity).unwrap());
            }
            // Geometry/gen mismatch → destroy and recreate.
            if let Some(old) =
                self.retire_resident(ctx, &identity, ResidentReclaim::Recreated, counters)
            {
                if old.generation != generation {
                    counters
                        .gen_mismatch
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        // Census only: this population is bounded by the allocation below, which
        // reclaims and retries on out-of-memory rather than trimming ahead of
        // one. See `recoverable_residents`.
        self.note_registry_reach();
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating a fresh one — the usage set is identical across all
        // registry targets, so a recycled image of the same geometry/format is
        // bind-compatible. This is what collapses the per-frame realloc storm a
        // per-generation target (video) would otherwise pay: skips vkCreateImage
        // + vkAllocateMemory + bind + view (and their note_create/note_alloc).
        // The recycled contents are stale — the slot is inserted with
        // layout=UNDEFINED / content_ready=false, and a fresh framebuffer is
        // always built below (it binds this specific render_pass).
        let (image, memory, view) = if let Some(free) = self.take_free_target(width, height, format)
        {
            (free.image, free.memory, free.view)
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
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
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryTarget, e)))?;
            counters.note_create();
            let ireq = ctx.device.get_image_memory_requirements(image);
            let memory = match self.bind_image_slab(
                ctx,
                image,
                &ireq,
                VkOp::PoolsBindRegistryTarget,
                counters,
            ) {
                Ok(m) => m,
                // Out of memory is the one refusal worth a second attempt: the
                // registry and the recycle pools are usually holding VRAM this
                // device was already entitled to give back, and refusing the
                // draw while holding it is not the same thing as the heap being
                // full. Once only, and only for this result — a retry on a
                // driver error that is not about memory would just fail twice.
                //
                // Here rather than inside `bind_image_slab`, where one wrapper
                // would cover every allocation site. That was tried and it
                // segfaults — see `reclaim_for_allocation_retry`.
                Err(error) if error.out_of_memory() => {
                    let given_back = self.reclaim_for_allocation_retry(ctx, counters);
                    match (given_back > 0)
                        .then(|| {
                            self.bind_image_slab(
                                ctx,
                                image,
                                &ireq,
                                VkOp::PoolsBindRegistryTarget,
                                counters,
                            )
                        })
                        .transpose()
                    {
                        Ok(Some(m)) => m,
                        // Nothing to give back, or the heap really is full. The
                        // draw refuses with the original error, which names the
                        // allocation that failed.
                        Ok(None) | Err(_) => {
                            ctx.device.destroy_image(image, None);
                            return Err(error);
                        }
                    }
                }
                Err(error) => {
                    ctx.device.destroy_image(image, None);
                    return Err(error);
                }
            };
            let view = match ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_subresource_range()),
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.free_image_slab(&ctx.device, image);
                    ctx.device.destroy_image(image, None);
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsCreateRegistryView,
                        e,
                    )));
                }
            };
            counters.note_create();
            (image, memory, view)
        };
        let attachments = [view];
        let framebuffer = match ctx.device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1),
            None,
        ) {
            Ok(fb) => fb,
            Err(e) => {
                ctx.device.destroy_image_view(view, None);
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateRegistryFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create();
        self.register_resident(
            &identity,
            NewResident {
                image,
                memory,
                view,
                framebuffer,
                render_pass,
                width,
                height,
                generation,
                color_format: format,
            },
        );
        Ok(self.registry.get(&identity).unwrap())
    }

    /// Ensure a resident attachment of an arbitrary Vulkan format — an MRT
    /// secondary colour target, or a depth-stencil buffer.
    ///
    /// The primary single-RT [`Self::registry_ensure`] only speaks `bgra` and
    /// owns a framebuffer; this one builds none, because its residents are only
    /// ever attachment N of an ad-hoc framebuffer or sampled through the view,
    /// never a standalone single-RT target. Reuse requires an exact (geometry,
    /// generation, format) match. Returns (image, view).
    ///
    /// **Colour and depth share this body rather than having one each.** The two
    /// differ only in image usage and view aspect, and both of those are
    /// functions of the format — see
    /// [`crate::backend::vulkan::engine::registry_target_usage`]. A second copy
    /// specialised for depth would be a copied arm over one wire form, which is
    /// how the recycle bucket's usage invariant would drift out of step with the
    /// creation site that has to honour it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn registry_ensure_attachment(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        generation: u64,
        format: vk::Format,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView), DrawError> {
        if let Some(slot) = self.registry.get(&identity) {
            if slot.reusable_for(width, height, generation, format) {
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok((slot.image, slot.view));
            }
            // Geometry / gen / format mismatch → destroy and recreate.
            self.retire_resident(ctx, &identity, ResidentReclaim::Recreated, counters);
        }
        // Census only, as in the primary `registry_ensure`.
        self.note_registry_reach();
        let usage = super::super::registry_target_usage(format);
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating — same recycle discipline as the primary
        // `registry_ensure`. Usage is a function of the format
        // (`registry_target_usage`), so images cross-flow between these paths by
        // geometry+format without a bucket ever mixing usages. Skips the
        // create/alloc/bind/view + their note_create/note_alloc; recycled
        // contents are stale, so the slot below is inserted layout=UNDEFINED /
        // content_ready=false.
        let (image, memory, view) = if let Some(free) = self.take_free_target(width, height, format)
        {
            (free.image, free.memory, free.view)
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
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
                .map_err(|e| {
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryTarget, e))
                })?;
            counters.note_create();
            let ireq = ctx.device.get_image_memory_requirements(image);
            let imt = ctx
                .memory_type_for(ireq.memory_type_bits, ireq.size, MemoryClass::DeviceLocal)
                .ok_or_else(|| {
                    ctx.device.destroy_image(image, None);
                    DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForMrtSecondary {
                        memory_type_bits: ireq.memory_type_bits,
                    })
                })?;
            let memory = allocate_memory_timed(
                ctx,
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(ireq.size)
                    .memory_type_index(imt),
                if super::super::format_is_depth(format) {
                    AllocSite::DepthResident
                } else {
                    AllocSite::MrtSecondary
                },
            )
            .map_err(|e| {
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsAllocMrtSecondary, e))
            })?;
            counters.note_alloc();
            ctx.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsBindMrtSecondary, e))
                })?;
            let view = ctx
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(super::super::registry_subresource_range(format)),
                    None,
                )
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryView, e))
                })?;
            counters.note_create();
            (image, memory, view)
        };
        self.register_resident(
            &identity,
            NewResident {
                image,
                memory,
                view,
                // No per-slot framebuffer and so no pass it was built against:
                // this arm's residents are bound as attachment N of an ad-hoc
                // MRT framebuffer, or sampled through the view.
                framebuffer: vk::Framebuffer::null(),
                render_pass: vk::RenderPass::null(),
                width,
                height,
                generation,
                color_format: format,
            },
        );
        Ok((image, view))
    }

    /// Allocate a transient D32_SFLOAT depth attachment (image + memory + view)
    /// sized to `width`x`height`. The caller owns it for exactly one draw and
    /// must dispose it deferred (`DeferredHandle::Image`) after submit — the CB
    /// still references it until its fence signals. Depth is never read back, so
    /// no TRANSFER_SRC usage.
    ///
    /// # It does not recycle, and on this pathway it never runs
    ///
    /// Every sibling allocator here reuses before allocating; this one does not.
    /// `exec` creates one per draw that carries depth state and disposes it at
    /// the end of that same draw. That was once by far the largest allocator in
    /// the engine: driven boots read `vk_alloc_sites transient_depth=5374:21225`
    /// and later `4623:18257` — thousands of `vkAllocateMemory` calls totalling
    /// ~18-21 GiB, against `slab_block=41:2568` for every resident colour target
    /// in the same boot. A depth [`FreePool`] was built against exactly that,
    /// measured at a 4 % improvement, and reverted as more code for no benefit.
    ///
    /// **A driven boot on the current build allocates zero.** x86/Vulkan,
    /// measured across all 126 `vk_alloc_sites` census windows spanning the boot
    /// (Safari page loads and page-downs, a title-bar drag, a wallpaper drag,
    /// Chess, the WebGL aquarium, then `killall` teardown): every window read
    /// `transient_depth=0:0`. The zero is not a broken probe — the counter is the
    /// `allocate_memory` wrapper's, keyed [`AllocSite::TransientDepth`], and
    /// `slab_block` and `staging` moved by thousands in the same lines.
    ///
    /// Nothing was refused on the way, either: zero `shader_state_degraded`, zero
    /// `depth_compare_unmapped`, zero `depth_load_unsupported_transient` in the
    /// whole log. So the guest is not asking for depth and being turned away — it
    /// is not asking. `resources.depth` is set only where the guest's
    /// depth-stencil descriptor decodes with a mapped compare, and no draw we
    /// executed carried one. The likely reading, NOT measured, is that a 3D
    /// application's depth work happens before its surface reaches our stream and
    /// what we execute is the compositor's 2D layer work.
    ///
    /// So do not build the depth pool. The premise it was queued against — a
    /// per-draw allocation storm — does not reproduce, and a fourth recycle pool
    /// would be more mechanism guarding nothing. `vk_alloc_sites transient_depth`
    /// is the number that would say a future workload changed this; until it is
    /// nonzero there is nothing here to recycle.
    ///
    /// The depth resident the guest's bound depth texture names, created on
    /// first use and held until the guest stops touching it.
    ///
    /// This is the rail that replaced a per-draw allocation. A depth buffer is a
    /// guest resource: the pass descriptor binds one, so it has an identity and
    /// a lifetime, and the device's job is to resolve it rather than to
    /// manufacture a private one per draw and throw it away. Under a drag the
    /// guest re-binds the same texture every frame, the registry touches it every
    /// frame, and the idle reclaim — which fires on age, not on population —
    /// never reaches it. Steady state is therefore zero allocations for any
    /// amount of traffic, which a pool keyed by geometry could not promise: that
    /// would have a hit rate, and hit rates fall off when a second window with a
    /// different size appears.
    ///
    /// Contents are *not* preserved by this: the pass still CLEARs, and
    /// `DepthState::load` is still false. What became persistent is the
    /// allocation, not the pixels. See
    /// [`crate::runtime::draw::vulkan::depth_chain_identity`] for why that
    /// distinction is what lets the identity carry generation zero.
    pub(crate) unsafe fn registry_ensure_depth(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        with_stencil: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView), DrawError> {
        let format = Self::depth_format(ctx, with_stencil);
        self.registry_ensure_attachment(ctx, identity, width, height, 0, format, counters)
    }

    /// The attachment format a depth buffer of this device is created with.
    ///
    /// Device-queried combined depth-stencil format when the bound state runs
    /// the stencil test (D32_S8 preferred, D24_S8 fallback — see
    /// `DeviceContext::depth_stencil_format`); plain D32_SFLOAT (no stencil
    /// aspect) otherwise, which is spec-mandatory. Depth is 32-bit float in the
    /// preferred case; the D24_S8 fallback is 24-bit UNORM depth, which the
    /// stencil-test path tolerates (it asserts stencil, not depth bits).
    ///
    /// One function because the resident rail and the transient fallback must
    /// pick the same format for the same draw — they feed the same render pass,
    /// whose attachment format is fixed by `PassKey`.
    fn depth_format(ctx: &DeviceContext, with_stencil: bool) -> vk::Format {
        if with_stencil {
            ctx.depth_stencil_format
        } else {
            translate::pixel::TRANSIENT_DEPTH_FORMAT
        }
    }

    pub(crate) unsafe fn create_transient_depth(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
        with_stencil: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), DrawError> {
        // Device-queried combined depth-stencil format when the bound state runs
        // the stencil test (D32_S8 preferred, D24_S8 fallback — see
        // DeviceContext::depth_stencil_format); plain D32_SFLOAT (no stencil
        // aspect) otherwise, which is spec-mandatory. Depth is 32-bit float in
        // the preferred case; the D24_S8 fallback is 24-bit UNORM depth, which
        // the stencil-test path tolerates (it asserts stencil, not depth bits).
        let (format, aspect) = if with_stencil {
            (
                ctx.depth_stencil_format,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            )
        } else {
            (
                translate::pixel::TRANSIENT_DEPTH_FORMAT,
                vk::ImageAspectFlags::DEPTH,
            )
        };
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDepthImage, e)))?;
        counters.note_create();
        let ireq = ctx.device.get_image_memory_requirements(image);
        let imt = ctx
            .memory_type_for(ireq.memory_type_bits, ireq.size, MemoryClass::DeviceLocal)
            .ok_or_else(|| {
                ctx.device.destroy_image(image, None);
                DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForDepth {
                    memory_type_bits: ireq.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(ireq.size)
                .memory_type_index(imt),
            AllocSite::TransientDepth,
        )
        .map_err(|e| {
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocDepth, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindDepth, e))
            })?;
        let view = ctx
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDepthView, e))
            })?;
        counters.note_create();
        Ok((image, memory, view))
    }

    /// Build an ad-hoc MRT framebuffer over `views` (primary slot 0 + secondary
    /// slots 1..) under `render_pass`. Not cached — the caller disposes it via
    /// `dispose(Framebuffer)` after the draw is sealed onto the ring slot.
    pub(crate) unsafe fn create_mrt_framebuffer(
        &mut self,
        ctx: &DeviceContext,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        width: u32,
        height: u32,
        counters: &EngineCounters,
    ) -> Result<vk::Framebuffer, DrawError> {
        let fb = ctx
            .device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(views)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtFramebuffer, e)))?;
        counters.note_create();
        Ok(fb)
    }

    /// Mark a resident ready after a render pass wrote it as a colour
    /// attachment and left it at an explicit `final_layout` — the MRT secondary
    /// arm, which settles at `COLOR_ATTACHMENT_OPTIMAL` where
    /// [`Self::registry_mark_ready`]'s primary settles at
    /// `TRANSFER_SRC_OPTIMAL`. Both record the same *access*; only the layout
    /// differs, which is the distinction [`ResidentAccess::ColorWrite`] carries.
    pub(crate) fn registry_mark_ready_at(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            slot.access = ResidentAccess::ColorWrite(layout);
        }
        self.set_sole_copy(identity, true);
    }

    /// Mark a depth resident as holding rendered contents, after a pass that
    /// stored them.
    ///
    /// The depth sibling of [`Self::registry_mark_ready_at`], and it exists
    /// rather than reusing that one because of the line that one ends on:
    /// `set_sole_copy(identity, true)`. Sole-copy means "these pixels exist
    /// nowhere else, so reclaiming destroys guest work", and **both reclaim
    /// paths skip such a slot at any age and any population**. That is right for
    /// a colour target whose pixels the guest is waiting for and wrong for a
    /// depth buffer, which no rail ever writes back to guest pages: marking one
    /// sole-copy would make every depth resident permanently unreclaimable, and
    /// VRAM would grow with the number of depth textures a guest has ever bound
    /// rather than with the number it is using. That is the cliff this rail was
    /// built to avoid, arrived at from the other side.
    ///
    /// So a depth resident stays reclaimable. The cost of being reclaimed is one
    /// pass that wanted `MTLLoadActionLoad` getting a CLEAR instead, which
    /// `note_depth_load_without_content` names — it is a real artifact, it is
    /// bounded by the idle age, and it is visible. Read that counter before
    /// deciding depth needs its own age.
    pub(crate) fn registry_mark_depth_ready(&mut self, identity: &TargetIdentity) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            // The depth pass declares `final_layout` DEPTH_STENCIL_ATTACHMENT_
            // OPTIMAL unconditionally, so this is where the image is left and it
            // is the `initial_layout` the next LOAD pass names. The two agreeing
            // is what makes a LOAD valid without a barrier between the passes.
            slot.access = ResidentAccess::ColorWrite(
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            );
        }
    }

    /// Whether this resident already holds rendered contents — the question a
    /// depth LOAD has to ask before it can be honoured.
    pub(crate) fn registry_content_ready(&self, identity: &TargetIdentity) -> bool {
        self.registry
            .get(identity)
            .is_some_and(|slot| slot.content_ready)
    }

    /// Pin/unpin a resident render target against LRU eviction (deferred
    /// render Stores). Pins are counted, not boolean: a surface can have several
    /// deferred windows armed at once and each holds one count, so the slot
    /// stays protected until every holder unpins. Returns false when the identity is absent or (for pinning) its
    /// content is not ready — callers must fall back to the synchronous
    /// Store. Unpin saturates at zero (a spurious unpin never underflows).
    ///
    /// # An unpin at zero is reported, not absorbed
    ///
    /// The saturation keeps the count sane; it does not make the call correct.
    /// Two holders and one unpin too many is a resident left reclaimable while
    /// somebody still reads it, and the deferred writeback rail made that
    /// reachable from a single line: it hands its pin to
    /// `note_guest_write_recorded` and a caller that also unpins would release
    /// one holder's pin on another's behalf. Silent saturation is what would let
    /// that land as a rare wrong frame instead of a log line.
    pub(crate) fn pin_resident_target(&mut self, identity: &TargetIdentity, pinned: bool) -> bool {
        let Some(slot) = self.registry.get_mut(identity) else {
            return false;
        };
        if pinned && !slot.content_ready {
            return false;
        }
        if !pinned && slot.pin_count == 0 {
            crate::observe::fail(format!(
                "resident_unpin_unbalanced identity={identity:?} \
                 (an unpin with no pin outstanding — some other holder's pin has \
                 already been released on its behalf, or this one was never taken)"
            ));
            return true;
        }
        // Counted pins, so only the 0 <-> 1 crossings change whether this slot is
        // in the non-pinned totals. A second pin, or an unpin that leaves one
        // holder, moves nothing.
        let before_non_pinned = slot.pin_count == 0;
        if pinned {
            slot.pin_count += 1;
        } else {
            slot.pin_count -= 1;
        }
        let after_non_pinned = slot.pin_count == 0;
        if before_non_pinned != after_non_pinned {
            let bytes = Self::slot_attachment_bytes(slot);
            self.registry_non_pinned_adjust(bytes, after_non_pinned);
        }
        true
    }

    /// Mark a resident ready after a draw stored into it.
    ///
    /// Clears `content_epoch`: this image's pixels just changed, and until
    /// something publishes them as the mapping's content and stamps the slot,
    /// nothing may claim they match a mapping epoch. Every path that ends in a
    /// resident holding new pixels comes through here or
    /// [`Self::registry_mark_ready_at`], which is what keeps the reset total
    /// rather than a list of the writers somebody remembered.
    pub(crate) fn registry_mark_ready(&mut self, identity: &TargetIdentity) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            // Draw pass final_layout is TRANSFER_SRC_OPTIMAL, and the access
            // that left it there is the pass's colour attachment write — not a
            // transfer, despite the layout's name.
            slot.access = ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }
        self.set_sole_copy(identity, true);
    }

    /// Record that this resident's current pixels have been copied somewhere
    /// that outlives the image — the guest's own pages — so reclaiming it now
    /// costs redundant work rather than the frame.
    ///
    /// Returns whether a slot was found and cleared, so a caller that believes
    /// it just wrote a resident back can say so; a silent no-op here would let
    /// the belief and the registry drift apart with nothing able to read it.
    ///
    /// Clears exactly the flag. It does not touch `content_epoch`, which answers
    /// the different question of whether these pixels are *current* for the
    /// mapping — a resident can be faithfully written back and then superseded
    /// by a guest CPU write, which makes it stale but does not make destroying
    /// it a loss.
    pub(crate) fn registry_note_content_copied_out(&mut self, identity: &TargetIdentity) -> bool {
        self.set_sole_copy(identity, false)
    }

    /// Record that this resident's pixels are the mapping's content as of
    /// `epoch`. Refuses (returns false) unless the slot exists and is
    /// content_ready — stamping an image no draw has stored into would vouch
    /// for undefined memory.
    pub(crate) fn registry_stamp_content_epoch(
        &mut self,
        identity: &TargetIdentity,
        epoch: u32,
    ) -> bool {
        match self.registry.get_mut(identity) {
            Some(slot) if slot.content_ready => {
                slot.content_epoch = Some(epoch);
                true
            }
            _ => false,
        }
    }

    /// Record a non-writing touch of a resident: a draw sampled it, or a
    /// transfer read it out (present blit, guest-page readback, GPU seed
    /// source). The writing touches go through [`Self::registry_mark_ready`]
    /// and [`Self::registry_mark_ready_at`], which also vouch for the pixels.
    ///
    /// Every rail that touches a resident has to land in one of the three,
    /// because the next barrier over that image derives its source scope from
    /// what it finds here — see [`ResidentAccess`].
    pub(crate) fn registry_note_access(&mut self, identity: &TargetIdentity, access: ResidentAccess) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.access = access;
        }
    }

    /// Count of registry residents NOT held by a deferred-write pin — the active
    /// working set the reclaim paths may draw on. Pinned slots are bounded
    /// separately (by the arming rail's own window cap) and excluded, so a pinned
    /// burst does not read as reclaimable population.
    ///
    /// O(1). This and [`Self::non_pinned_registry_bytes`] each walked the whole
    /// registry, on every admit — affordable only while something held the
    /// population near a few hundred, which nothing does now that the slot count
    /// is gone. [`Self::registry_non_pinned`] is maintained instead at the sites
    /// that can change either total, and
    /// `non_pinned_registry_totals_by_walk` is the walk kept as the thing to
    /// check it against (test-only, so not linkable from here).
    pub(super) fn non_pinned_registry_len(&self) -> usize {
        self.registry_non_pinned.count
    }

    /// The drain's wall clock, for a caller measuring an age against it.
    pub(crate) fn idle_clock_ms(&self) -> u64 {
        self.idle_clock_ms
    }

    /// Attachment bytes the same non-pinned set occupies: `w x h x texel` summed
    /// over every slot [`Self::non_pinned_registry_len`] counts.
    ///
    /// The number that retired the slot count: that count claimed slots were
    /// cheap and the real guard was per-image bytes, and then bounded the slots.
    /// Its doc also quoted ~516 MiB for a burst and a ~1005 MiB idle baseline, both from
    /// a `vram` census line that no longer exists anywhere in this crate, so
    /// until this counter there was nothing in the device that could say what a
    /// count of 320 costs. 320 slots is 5 MiB of 16x16 scratch or 10 GiB of 4K.
    fn non_pinned_registry_bytes(&self) -> u64 {
        self.registry_non_pinned.bytes
    }

    /// One slot's contribution to [`Self::non_pinned_registry_bytes`].
    ///
    /// Attachment footprint, not allocation footprint: it does not know tiling
    /// padding or the slab's rounding, and a format
    /// [`crate::backend::vulkan::translate::pixel::bytes_per_texel`] declines
    /// (block-compressed, multi-planar — neither of which a colour attachment
    /// uses) contributes nothing rather than a guessed size. So the total is a
    /// lower bound on VRAM, which is the safe direction for a figure that exists
    /// to decide whether a bound is too loose.
    fn slot_attachment_bytes(slot: &ResidentTargetSlot) -> u64 {
        crate::backend::vulkan::translate::pixel::bytes_per_texel(slot.color_format)
            .map(|texel| u64::from(slot.width) * u64::from(slot.height) * u64::from(texel))
            .unwrap_or(0)
    }

    /// The same two totals recomputed from the registry, for the test that says
    /// the maintained pair still agrees with it.
    ///
    /// Kept because the maintained pair has three writers and a fourth mutation
    /// site would desync it in silence — a resident that stopped being counted
    /// makes the population read smaller than it is, which is the direction that
    /// lets the cap sit above its own bound. This is what a desync is diffed
    /// against.
    #[cfg(test)]
    fn non_pinned_registry_totals_by_walk(&self) -> NonPinnedTotals {
        let non_pinned = || self.registry.values().filter(|slot| slot.pin_count == 0);
        NonPinnedTotals {
            count: non_pinned().count(),
            bytes: non_pinned().map(Self::slot_attachment_bytes).sum(),
        }
    }

    /// Fold one slot into or out of the maintained non-pinned totals.
    ///
    /// Every change of "is this slot non-pinned" goes through here, so the count
    /// and the bytes cannot move apart from each other, or be updated at two
    /// sites and forgotten at a third.
    fn registry_non_pinned_adjust(&mut self, slot_bytes: u64, joined: bool) {
        Self::fold_totals(&mut self.registry_non_pinned, slot_bytes, joined);
    }

    /// Fold one slot into or out of the maintained sole-copy totals.
    ///
    /// Same shape and same reason as [`Self::registry_non_pinned_adjust`], over
    /// the population [`ResidentTargetSlot::gpu_only_content`] describes. Kept as
    /// its own call rather than a flag on that one so a site cannot update the
    /// wrong population by passing a bool.
    fn registry_sole_copy_adjust(&mut self, slot_bytes: u64, joined: bool) {
        Self::fold_totals(&mut self.registry_sole_copy, slot_bytes, joined);
    }

    /// The arithmetic both maintained populations share: a count and its bytes
    /// move together or the pair stops describing one population.
    fn fold_totals(totals: &mut NonPinnedTotals, slot_bytes: u64, joined: bool) {
        if joined {
            totals.count += 1;
            totals.bytes += slot_bytes;
        } else {
            totals.count = totals.count.saturating_sub(1);
            totals.bytes = totals.bytes.saturating_sub(slot_bytes);
        }
    }

    /// Set a slot's [`ResidentTargetSlot::gpu_only_content`] and keep the
    /// maintained totals in step.
    ///
    /// The single writer of that field on a live slot, so the flag and the
    /// population that reports it cannot be updated at two sites and forgotten
    /// at a third — the defect `registry_non_pinned_adjust` exists to prevent on
    /// the other population. Registration sets the field directly because there
    /// is no slot to read yet, and it sets it to the default that counts for
    /// nothing.
    fn set_sole_copy(&mut self, identity: &TargetIdentity, sole: bool) -> bool {
        let Some(slot) = self.registry.get_mut(identity) else {
            return false;
        };
        // Returning early on a no-op is what keeps the totals a population
        // rather than a transition count: `registry_mark_ready` fires on every
        // draw into an already-sole-copy slot.
        if slot.gpu_only_content == sole {
            return true;
        }
        slot.gpu_only_content = sole;
        let bytes = Self::slot_attachment_bytes(slot);
        self.registry_sole_copy_adjust(bytes, sole);
        // Folded here rather than only at the capacity walk, because that walk
        // runs on admission and this population grows on `registry_mark_ready` —
        // a peak sampled only at the other event lags by a draw, and can miss
        // one entirely for a burst that marks more than it admits.
        if sole {
            self.registry_sole_copy_peak =
                Self::high_water(self.registry_sole_copy_peak, self.registry_sole_copy);
        }
        true
    }

    /// Per-field maximum of two population readings. Both fields advance from the
    /// same sample, so the pair keeps describing one population rather than two
    /// moments.
    fn high_water(peak: NonPinnedTotals, now: NonPinnedTotals) -> NonPinnedTotals {
        NonPinnedTotals {
            count: peak.count.max(now.count),
            bytes: peak.bytes.max(now.bytes),
        }
    }

    /// Every resident the device may destroy without losing guest work: not
    /// pinned, and not the only copy of its own pixels.
    ///
    /// This answers "what could be given back if it had to be", which is the
    /// question an allocation failure asks — and, since the slot cap was retired,
    /// it is the only question anything asks about this population. Split out
    /// from the reclaim so the selection is testable without a device.
    ///
    /// # The resident-target population is bounded by the allocator, not a count
    ///
    /// A slot count used to bound it (`REGISTRY_CAP`, 320): crossing it retired
    /// the least-recently-used resident this predicate admits. Two things retired
    /// the count.
    ///
    /// **The quantity it bounded is not the quantity that runs out.** Both
    /// readings, driven x86/PCI Vulkan, `registry_pressure`:
    ///
    /// ```text
    ///                                   peak slots   peak_mib   MiB/slot
    ///   idle + Safari window drag               41         74       1.80
    ///   web-content-probe --churn 1            194        211       1.09
    /// ```
    ///
    /// A burst quadruples the population and its residents are *smaller* than the
    /// idle set's, so 320 slots was reached at roughly 350 MiB at the burst's mix
    /// and would be 10 GiB at 4K, on a `DEVICE_LOCAL` heap
    /// ([`crate::backend::vulkan::caps::memory_topology::MemoryProfile::device_local_bytes`])
    /// measured in gigabytes. One constant could not be both.
    ///
    /// **And it could only ever have fired on residents in active use.** The idle
    /// drain ([`ResourcePools::advance_registry_touch_and_drain`]) already
    /// reclaims anything untouched for `IDLE_TARGET_AGE_MS`, at up to
    /// `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass every `IDLE_DRAIN_INTERVAL_MS` —
    /// about forty a second. So the standing population *is* the live
    /// two-second working set, and a count crossing it means the guest is using
    /// more targets than the count allowed, which is the worst moment to take one
    /// away. Measured, it never came close: peak 194 of 320 under
    /// `web-content-probe --churn 1`, `evicts=0` on every boot ever taken.
    ///
    /// # What bounds it now
    ///
    /// The allocation. [`ResourcePools::registry_ensure`] tries the recycle pool,
    /// then `vkCreateImage` + [`ResourcePools::bind_image_slab`]; on an
    /// out-of-memory result it calls [`Self::reclaim_for_allocation_retry`],
    /// which gives back every recycle pool plus everything this function returns,
    /// and retries once. If that still fails the draw refuses with the driver's
    /// own error. That is a GPU refusing because its memory is full — current,
    /// attributable, and self-healing, since the idle drain frees the space for
    /// the next attempt — rather than a count destroying an earlier accepted
    /// result in order to break a future draw.
    ///
    /// The sole-copy population was already exempt from the count and already
    /// grew past it unbounded (the walk soft-exceeded rather than take one), so
    /// the allocator was already the only bound on the half that *cannot* be
    /// given back. This puts the half that can behind the same bound.
    ///
    /// [`Self::note_registry_reach`] still samples the population and its bytes
    /// at every admission and `registry_pressure` still publishes both. They are
    /// worth more without the count than with it: `peak` and `peak_mib` now
    /// describe what the guest asked for rather than what a constant permitted.
    ///
    /// # The boot after the count came out
    ///
    /// Driven x86/PCI, `web-content-probe -n 10 --churn 1`, QEMU relinked,
    /// nothing else running:
    ///
    /// ```text
    ///   registry_pressure peak=223 peak_mib=203 resident_samples=12078
    ///                     resample_peak_ms=1590/2000 slab_mib=45/200
    ///                     sole_copy=108/44mib cs_sole_copy=2/1mib
    /// ```
    ///
    /// Visual gate 10/10 regions on colour, `sampled_resident_missing=0`, and
    /// `vram_reclaim_retry` absent — no allocation was refused, so the path that
    /// now bounds this population was never asked to run. That is the expected
    /// shape: the bound is a *pressure* mechanism and this workload applies none.
    ///
    /// **`sole_copy=108` of `peak=223` is the reading to keep.** Roughly half the
    /// population is protected, so a reclaim arriving under real pressure would
    /// still have ~115 slots and ~159 MiB to give back. It is that ratio, not the
    /// peak, that says whether the retry has anything to work with — near 1 and
    /// the copy-out sites are what needs attention.
    ///
    /// `peak` was 194 on the same probe while the count was still in place, and
    /// `evicts` was 0 then, so 194 -> 223 is workload variance rather than the
    /// count's absence: a walk that never removed anything cannot have been
    /// holding the population down.
    fn recoverable_residents(&self) -> Vec<TargetIdentity> {
        self.registry_order
            .iter()
            .filter(|k| {
                self.registry
                    .get(*k)
                    .is_some_and(|slot| slot.pin_count == 0 && !slot.gpu_only_content)
            })
            .cloned()
            .collect()
    }

    /// Empty the image and buffer recycle pools, for a retry after the device
    /// refused an allocation. Returns how many entries were released.
    ///
    /// The half of [`Self::reclaim_for_allocation_retry`] that is safe to call
    /// from **any** allocation site, including one reached part-way through
    /// recording a draw. A free-list entry is one that already went through
    /// [`ResourcePools::dispose`] and came out the recycling side, which happens
    /// only when no command-buffer slot was open or when the graveyard released
    /// it after its slots retired. Nothing submitted references it, and the draw
    /// being recorded does not hold it either — taking an entry removes it from
    /// the pool, so anything still in one was never handed out.
    ///
    /// That is exactly the property retiring a live *resident* does not have,
    /// which is what makes the sibling site-specific and this one not.
    ///
    /// `trim_buffers` is forced on, so the HOST_VISIBLE staging and readback
    /// rings go too. They are normally held back behind
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` because refilling them costs a full
    /// `vkAllocateMemory` on the upload path — a real cost, and a smaller one
    /// than the draw this is trying to save.
    ///
    /// # Measured against the two alternatives it sits between
    ///
    /// Same driven x86/PCI boot, same injected failure (every 200th
    /// `bind_image_slab` returning `ERROR_OUT_OF_DEVICE_MEMORY`), same probe:
    ///
    /// ```text
    ///                              draws lost   segfault
    ///   no retry at these sites             7         no
    ///   full reclaim, all sites             0        YES
    ///   pools only, all sites               0         no
    /// ```
    ///
    /// The last row is this function: 16 injected failures over the boot, every
    /// one absorbed, 0 regions off their declared colour, and
    /// `vram_reclaim_retry` never needed — the pools alone were always enough, so
    /// the registry site's fuller reclaim is now a second line rather than the
    /// first.
    ///
    /// The `released=` field is why that last claim is checkable. Without it a
    /// boot where this absorbed every failure and one where no allocation ever
    /// failed read identically — both silent, both zero draws lost. The first
    /// instrumented run reported exactly that pair of zeros and could not tell
    /// them apart.
    pub(super) unsafe fn reclaim_pools_for_allocation_retry(
        &mut self,
        ctx: &DeviceContext,
    ) -> usize {
        let released = self.trim_recycle_pools(&ctx.device, usize::MAX, true);
        // Always visible, and not only for the usual "a degradation must be
        // readable" reason. Without a line here a boot where this absorbed every
        // allocation failure and one where no allocation ever failed report
        // identically — both silent, both zero draws lost — so the recovery
        // could not be told from never having been needed.
        crate::observe::fail(format!(
            "vram_pool_reclaim_retry released={released} held_bytes={}              (an allocation was refused; emptied the recycle pools, which hold              nothing any command buffer references, and retried)",
            self.slab.held_bytes().0,
        ));
        released
    }

    /// Give back everything the registry can spare, for a retry after the device
    /// refused an allocation. Returns how many residents and pooled images were
    /// released.
    ///
    /// Out of device memory is the one refusal this device can still do
    /// something about, and since the slot count was retired it is the only thing
    /// bounding this population. The idle drain returns VRAM on a 2 s timer, so
    /// at the moment an allocation fails the registry is usually holding
    /// residents it was entitled to drop and had simply not got to yet. Refusing the guest's draw
    /// while still holding them is not "the GPU is out of memory"; it is this
    /// device declining to tidy up first, which is not what the hardware being
    /// emulated does.
    ///
    /// Age is deliberately ignored. The drain cutoff is a throughput compromise,
    /// and by this point throughput is already lost — what is left is whether the
    /// draw survives at all. `pin_count` and `gpu_only_content` are still
    /// honoured, so this can only ever cost re-reads, never a frame.
    ///
    /// The recycle pools go first and go completely, including the HOST_VISIBLE
    /// buffer pools that `trim_recycle_pools` otherwise holds back behind
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM`: free-list entries hold no guest content
    /// at all, so they are strictly cheaper to give back than any resident.
    ///
    /// # It works, and it was measured by breaking the allocator on purpose
    ///
    /// No boot has ever produced a real allocation failure, so this was driven
    /// with temporary fault injection — every 200th `bind_image_slab` returning
    /// `ERROR_OUT_OF_DEVICE_MEMORY` before doing anything — on a driven x86/PCI
    /// boot under `web-content-probe -n 10 --churn 1`. The injection is not in
    /// the tree; what it established is:
    ///
    /// ```text
    ///   vram_reclaim_retry residents=13 recycled=65   sole_copy=25  live=26
    ///                      residents=28 recycled=146  sole_copy=84  live=85
    ///                      residents=66 recycled=106  sole_copy=146 live=147
    ///                      residents=74 recycled=53   sole_copy=147 live=148
    ///                      residents=82 recycled=133  sole_copy=172 live=173
    /// ```
    ///
    /// Five forced failures, five recoveries, and the probe still reported 0
    /// regions off their declared colour — so every draw that would have been
    /// dropped was served instead. The `sole_copy`/`live` pair is the invariant
    /// visible from outside: after each reclaim the registry holds its protected
    /// set and one more, which is exactly what "give back everything that is
    /// neither pinned nor the only copy" should leave behind.
    ///
    /// The same run measured the cost of *not* having this. The injection hit
    /// every `bind_image_slab` caller and the sampled-image bind has no retry, so
    /// seven draws died there as `linux_m2v_draw reason=vk_pools_bind_sampled`
    /// `vk_result=A_device_memory_allocation_has_failed`, at geometries up to
    /// 1920x1080.
    ///
    /// # It cannot be hoisted into `bind_image_slab`, and that is measured too
    ///
    /// Those seven look like an argument for one wrapper around
    /// `bind_image_slab`, covering every allocation site at once. **That was
    /// implemented and it segfaults QEMU**, on the same injected boot, ~27 s in.
    /// It does remove the class it was aimed at — zero draws lost to the injected
    /// failure, against seven — and then the process dies, so the fix is worse
    /// than the bug.
    ///
    /// The reasoning that said it was safe is wrong in one specific way, and it
    /// is worth stating because it is convincing: [`ResourcePools::dispose`]
    /// parks a retired resident in the graveyard whenever `open_slot_mask` is
    /// non-zero, and that mask counts the *recording* batch's slot, so a resident
    /// released mid-draw cannot be handed back to a later allocation in the same
    /// draw. True — but it only holds once a batch is open. The sampled binds run
    /// early in `execute_draw_inner`, before the batch exists, so the mask can be
    /// **0** there and `dispose` destroys immediately. The caller is by then
    /// holding raw `vk::Image`/`vk::ImageView` handles for the target it resolved
    /// and the residents it is about to sample, and the reclaim frees them under
    /// it.
    ///
    /// This site is safe for a reason that does not generalise: it runs inside
    /// `registry_ensure_attachment`, at the one point in a draw where retiring a
    /// resident has always been safe — before the caller holds anything and
    /// before any sampled source has been resolved. The retired slot cap ran
    /// here for the same reason.
    ///
    /// What generalises is the *pools* half — see
    /// [`Self::reclaim_pools_for_allocation_retry`]. Retiring live residents is
    /// the part that needs the reclaim to know what the in-progress draw is
    /// holding, either by shielding those identities the way `protect` already
    /// shields one, or by opening the batch before the first allocation so the
    /// graveyard gate is armed for the whole draw. Neither is attempted here.
    pub(super) unsafe fn reclaim_for_allocation_retry(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> usize {
        let trimmed = self.reclaim_pools_for_allocation_retry(ctx);
        let victims = self.recoverable_residents();
        let mut freed = 0;
        for victim in &victims {
            if self
                .retire_resident(ctx, victim, ResidentReclaim::AllocationReclaimed, counters)
                .is_some()
            {
                freed += 1;
            }
        }
        // Always visible. A device that has had to do this is one whose next
        // allocation is also likely to fail, and a silent recovery would leave
        // the run before that failure looking healthy.
        crate::observe::fail(format!(
            "vram_reclaim_retry residents={freed} recycled={trimmed} held_bytes={} \
             sole_copy={} live={} (an allocation was refused; gave back everything \
             that is neither pinned nor the only copy of its pixels)",
            self.slab.held_bytes().0,
            self.registry_sole_copy.count,
            self.registry.len(),
        ));
        freed + trimmed
    }

    /// Fold the current non-pinned and sole-copy populations into their
    /// high-water bands.
    ///
    /// Called from both admit paths, immediately before the allocation: that is
    /// the one point every admission passes through, and it is where the
    /// population is at its highest. It is a pure census — nothing here selects,
    /// retires or refuses anything, and it takes neither a `DeviceContext` nor
    /// `unsafe`, so it is exercisable without a GPU. An instrument that cannot be
    /// tested is one that can silently read zero forever, which is the failure it
    /// exists to prevent.
    ///
    /// Both bands are folded from the same sample, so `peak` and `peak_bytes`
    /// describe one population rather than two moments — the two together are
    /// what said that a slot count was never a sane proxy for VRAM, which is the
    /// reading that retired the count. See [`Self::recoverable_residents`].
    fn note_registry_reach(&mut self) {
        self.registry_non_pinned_peak = self
            .registry_non_pinned_peak
            .max(self.non_pinned_registry_len() as u64);
        self.registry_non_pinned_peak_bytes = self
            .registry_non_pinned_peak_bytes
            .max(self.non_pinned_registry_bytes());
        // Sampled from the same instant as the two above, so the ratio between
        // them describes one population and not two moments.
        self.registry_sole_copy_peak =
            Self::high_water(self.registry_sole_copy_peak, self.registry_sole_copy);
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

    /// Remember that `identity`'s resident was reclaimed, and by which path.
    ///
    /// Called from every site that removes a live registry entry, so a later
    /// draw sampling it can distinguish "taken from under you" from "never
    /// existed". Bounded FIFO; the oldest record is dropped rather than letting
    /// a diagnostic grow without limit.
    pub(crate) fn note_resident_reclaimed(
        &mut self,
        identity: &TargetIdentity,
        why: ResidentReclaim,
    ) {
        if self.reclaimed_recent.len() >= RECLAIM_HISTORY {
            self.reclaimed_recent.pop_front();
        }
        let at = self.idle_clock_ms;
        self.reclaimed_recent.push_back((identity.clone(), why, at));
    }

    /// [`ResourcePools::prior_reclaim`] with the idle-clock time the reclaim
    /// happened, for a caller that needs to know how long ago rather than only
    /// what.
    ///
    /// This is what uncensors `resident_resample_peak_ms`. That peak can only
    /// observe gaps for residents that *survived* to be read, so a reclaim
    /// policy tuned from it is tuned from data it destroyed the tail of — every
    /// gap longer than the cutoff shows up as an absence, not as a longer gap.
    /// Pairing this with `IDLE_TARGET_AGE_MS` recovers the missing side: a
    /// resident read `since` ms after being reclaimed had gone at least
    /// `IDLE_TARGET_AGE_MS + since` ms between uses.
    pub(crate) fn prior_reclaim_at(
        &self,
        identity: &TargetIdentity,
    ) -> Option<(ResidentReclaim, u64)> {
        self.reclaimed_recent
            .iter()
            .rev()
            .find(|(k, _, _)| k == identity)
            .map(|(_, why, at)| (*why, *at))
    }

    /// The most recent thing this device did with `identity`'s resident, if it
    /// is still inside the history window. `None` means no record — which covers
    /// both "never held one" and "reclaimed longer ago than the window reaches",
    /// two cases this deliberately does not guess between.
    pub(crate) fn prior_reclaim(&self, identity: &TargetIdentity) -> Option<ResidentReclaim> {
        self.reclaimed_recent
            .iter()
            .rev()
            .find(|(k, _, _)| k == identity)
            .map(|(_, why, _)| *why)
    }

    /// Record that a draw is reading this resident as a **sampled source**, so
    /// both reclaim paths count it as in use.
    ///
    /// See [`resident_resample_band`] for why this also bands how long the
    /// resident had been sitting untouched before the read.
    ///
    /// Reading a resident was not a use. `last_touch_ms` was refreshed by
    /// `registry_ensure` (a draw rendering *into* the target), by the present
    /// touch, and by nothing else — while the sampled-source resolve in
    /// `execute_draw_inner` goes through `registry_get`, which takes `&self` and
    /// therefore cannot mark anything. A resident that every frame samples but
    /// no frame draws into consequently aged as if it were abandoned.
    ///
    /// That is the shape of a compositor backdrop: the desktop behind a
    /// translucent panel is rendered once and then read by every vibrancy draw
    /// over it. After `IDLE_TARGET_AGE_MS` the idle drain took it, and the drain
    /// is a terminal destroy rather than a recycle, so the pixels were gone. The
    /// next draw to sample it refuses with
    /// `vk_draw_exec_sampled_resident_missing`, and because the exec loop
    /// abandons the remaining records of a packet once a record cannot encode,
    /// one missing backdrop drops a whole packet of draws.
    ///
    /// Nothing recreates a resident except a draw rendering into that identity,
    /// so a backdrop the guest considers still valid is never rebuilt: the
    /// refusal repeats for the life of the boot. That is why this class survives
    /// closing the application that caused the pressure and why only a reboot
    /// clears it.
    ///
    /// The stamp this writes is the one the idle drain reads — it compares
    /// `last_touch_ms` against `IDLE_TARGET_AGE_MS` — so recording a read here is
    /// what keeps a resident nothing draws into from aging out under one that
    /// every frame samples.
    pub(crate) fn registry_note_sampled_use(&mut self, identity: &TargetIdentity) {
        let touch = self.idle_clock_ms;
        if let Some(slot) = self.registry.get_mut(identity) {
            let idle_ms = touch.saturating_sub(slot.last_touch_ms);
            slot.last_touch_ms = touch;
            crate::runtime::drain::note_store_route(resident_resample_band(idle_ms));
            // The bands give the distribution; this gives the margin. They
            // answer different questions, and the bands alone could not say
            // whether their one sample above the half mark sat at 1.0 s or at
            // 1.9 s against a 2 s cutoff — which is the difference between
            // comfortable and one slow frame from a permanent loss.
            self.resident_resample_peak_ms = self.resident_resample_peak_ms.max(idle_ms);
        }
    }
}

#[cfg(test)]
pub(super) mod pin_count_tests {
    use super::*;

    /// A registry slot a pin will be granted on, for tests in sibling modules
    /// that need a resident to hold rather than a resident to inspect.
    pub(in crate::backend::vulkan::engine::pools) fn ready_slot() -> ResidentTargetSlot {
        dummy_slot(true)
    }

    fn dummy_slot(content_ready: bool) -> ResidentTargetSlot {
        ResidentTargetSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            render_pass: vk::RenderPass::null(),
            width: 16,
            height: 16,
            generation: 1,
            content_ready,
            content_epoch: None,
            access: ResidentAccess::ColorWrite(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            color_format: translate::pixel::SCANOUT_FORMAT,
            pin_count: 0,
            gpu_only_content: false,
            last_touch_ms: 0,
        }
    }

    fn pinned_identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 1,
            width: 16,
            height: 16,
            generation: 0,
            format: translate::pixel::SCANOUT_FORMAT,
        }
    }

    /// The window presenter blits a resident with no format conversion and no
    /// source scaling, so every one of these four conditions is load-bearing.
    ///
    /// It matters that this is ONE function: the device's publish path asks it
    /// a frame ahead of the presenter to decide whether to read the frame back
    /// into host memory. A looser predicate there elides the readback for a
    /// frame the presenter then refuses, and the window goes blank with no CPU
    /// pixels behind it — a disagreement neither call site can see on its own.
    #[test]
    fn only_a_ready_bgra_resident_at_the_exact_geometry_is_presentable() {
        let ready = dummy_slot(true);
        assert!(slot_presentable(&ready, 16, 16));

        assert!(
            !slot_presentable(&dummy_slot(false), 16, 16),
            "content that has not landed would present the previous frame"
        );

        let mut rgba = dummy_slot(true);
        rgba.color_format = translate::pixel::RESIDENT_RGBA_FORMAT;
        assert!(
            !slot_presentable(&rgba, 16, 16),
            "the blit does no channel swap; RGBA would present with red and blue exchanged"
        );

        assert!(
            !slot_presentable(&ready, 32, 16),
            "a wider present than the resident holds would blit a stretched frame"
        );
        assert!(
            !slot_presentable(&ready, 16, 32),
            "a taller present than the resident holds would blit a stretched frame"
        );
    }

    /// A draw into this identity invalidates any stamp on it. The image's
    /// pixels just changed, and until something publishes them as the mapping's
    /// content the type-11 LOAD gate must not treat them as current — otherwise
    /// an intermediate record's output is loaded as though it were the guest's
    /// prior frame.
    ///
    /// Placed on `registry_mark_ready` rather than on the individual writers on
    /// purpose: every path that leaves a resident holding new pixels goes
    /// through here or `registry_mark_ready_at`, so the invalidation is total
    /// rather than a list of the writers somebody remembered.
    #[test]
    fn a_draw_into_a_resident_clears_its_content_stamp() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));

        assert!(pools.registry_stamp_content_epoch(&id, 9));
        assert_eq!(pools.registry.get(&id).unwrap().content_epoch, Some(9));

        pools.registry_mark_ready(&id);
        assert_eq!(
            pools.registry.get(&id).unwrap().content_epoch,
            None,
            "a draw stored new pixels; the old stamp cannot vouch for them"
        );

        assert!(pools.registry_stamp_content_epoch(&id, 10));
        pools.registry_mark_ready_at(&id, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(
            pools.registry.get(&id).unwrap().content_epoch,
            None,
            "the MRT-secondary ready arm must invalidate identically"
        );
    }

    /// Stamping an image no draw has stored into would vouch for undefined
    /// memory, and an absent identity has no image at all. Both refuse, and the
    /// caller reads the `false` as "the elision is off for this surface".
    #[test]
    fn a_stamp_refuses_an_image_no_draw_has_written() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();

        assert!(
            !pools.registry_stamp_content_epoch(&id, 1),
            "an absent identity cannot be stamped"
        );

        pools.registry.insert(id.clone(), dummy_slot(false));
        assert!(
            !pools.registry_stamp_content_epoch(&id, 1),
            "a resident that is not content_ready holds undefined pixels"
        );
        assert_eq!(pools.registry.get(&id).unwrap().content_epoch, None);
    }

    /// Two deferred windows on one surface pin the SAME identity; the first
    /// window's flush-unpin must NOT expose the image to the LRU sweep while the
    /// second is still armed. This is the eviction window a boolean pin had.
    #[test]
    fn shared_identity_pin_is_counted_not_boolean() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));

        assert!(pools.pin_resident_target(&id, true), "window A pin");
        assert!(pools.pin_resident_target(&id, true), "window B pin");
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 2);

        // Window A flushes: one unpin — the slot must stay sweep-protected.
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(
            pools.registry.get(&id).unwrap().pin_count,
            1,
            "the second window is still armed: slot must remain pinned"
        );

        // Window B flushes: fully released.
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
    }

    /// A spurious unpin (double-release) saturates at zero instead of
    /// underflowing into a forever-pin.
    #[test]
    fn unpin_saturates_at_zero() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
        assert!(pools.pin_resident_target(&id, true));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 1);
    }

    /// Pin still refuses a not-ready slot (callers fall back to the sync
    /// Store) and an absent identity.
    #[test]
    fn pin_refuses_not_ready_and_absent() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        assert!(!pools.pin_resident_target(&id, true), "absent identity");
        pools.registry.insert(id.clone(), dummy_slot(false));
        assert!(!pools.pin_resident_target(&id, true), "not-ready slot");
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
    }

    fn surf(id: u32) -> TargetIdentity {
        TargetIdentity::Surface {
            id,
            width: 16,
            height: 16,
            generation: 1,
            format: translate::pixel::SCANOUT_FORMAT,
        }
    }

    /// A resident shaped like the one an arm builds, for the registration tests.
    ///
    /// The two arms differ in exactly these two handles, so they are the
    /// parameters: `registry_ensure` passes a real framebuffer and the pass it
    /// was built against, `registry_ensure_attachment` passes neither.
    fn new_resident(framebuffer: vk::Framebuffer, render_pass: vk::RenderPass) -> NewResident {
        NewResident {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer,
            render_pass,
            width: 16,
            height: 16,
            generation: 1,
            color_format: translate::pixel::SCANOUT_FORMAT,
        }
    }

    /// A framebuffer handle that is merely non-null. Nothing dereferences it —
    /// every test here asks only whether the slot has one.
    fn some_framebuffer() -> vk::Framebuffer {
        vk::Framebuffer::from_raw(1)
    }

    /// Admit a resident with an explicit last-touch stamp and pin count.
    ///
    /// Registers through the product path rather than writing the map and the
    /// order itself, so this helper cannot be the copy that keeps them in step
    /// by accident while the product one stops.
    fn admit(pools: &mut ResourcePools, id: TargetIdentity, last_touch_ms: u64, pin: u32) {
        pools.register_resident(
            &id,
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );
        let slot = pools.registry.get_mut(&id).expect("just registered");
        slot.content_ready = true;
        slot.last_touch_ms = last_touch_ms;
        // Through the product path, not `slot.pin_count = pin`: pinning is what
        // takes a resident out of the maintained non-pinned totals, and a helper
        // that wrote the field itself would be the one mutation site the totals
        // cannot see — which is the desync
        // `the_maintained_non_pinned_totals_track_the_walk` exists to catch.
        for _ in 0..pin {
            assert!(pools.pin_resident_target(&id, true), "content is ready");
        }
    }

    /// [`admit`] at an explicit geometry, so a test can build populations the
    /// slot count cannot tell apart. Geometry is fixed at registration because
    /// nothing in the product mutates a live slot's — a geometry change goes
    /// through unregister + register — and the byte total relies on that.
    fn admit_sized(
        pools: &mut ResourcePools,
        id: TargetIdentity,
        last_touch_ms: u64,
        pin: u32,
        (width, height): (u32, u32),
    ) {
        let mut resident = new_resident(some_framebuffer(), vk::RenderPass::null());
        resident.width = width;
        resident.height = height;
        pools.register_resident(&id, resident);
        let slot = pools.registry.get_mut(&id).expect("just registered");
        slot.content_ready = true;
        slot.last_touch_ms = last_touch_ms;
        for _ in 0..pin {
            assert!(pools.pin_resident_target(&id, true), "content is ready");
        }
    }

    /// The MRT-secondary arm builds no per-slot framebuffer, so the residents it
    /// creates owe the graveyard nothing — and the deferred-handle ring is
    /// bounded, so a destroy path that enqueued their null handle would be
    /// spending a slot every real destroy has to wait behind.
    ///
    /// `vkDestroyFramebuffer` accepts `VK_NULL_HANDLE` and does nothing with it.
    /// That is why the two paths which asked this question wrong produced no
    /// crash, no validation error and no log line, and why the answer is worth
    /// a test rather than an assertion at each site.
    #[test]
    fn a_resident_built_without_a_framebuffer_owes_the_graveyard_none() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(vk::Framebuffer::null(), vk::RenderPass::null()),
        );
        pools.register_resident(
            &surf(2),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        assert_eq!(
            pools.registry.get(&surf(1)).unwrap().owed_framebuffer(),
            None,
            "an MRT-secondary resident has no framebuffer to destroy"
        );
        assert_eq!(
            pools.registry.get(&surf(2)).unwrap().owed_framebuffer(),
            Some(some_framebuffer()),
            "a single-RT resident owes the one it was built with"
        );
    }

    /// A resident is born with nothing drawn into it, nothing vouching for its
    /// pixels, no layout transition behind it and no window holding it.
    ///
    /// Each of these four is read by a different rail — `registry_mark_ready`,
    /// the type-11 LOAD gate's epoch check, the barrier tracker and the idle
    /// drain — so an arm that registered a slot with any of them set differently
    /// would be answering a question the other rails believe they already asked.
    #[test]
    fn a_registered_resident_is_born_undrawn_unvouched_untransitioned_and_unpinned() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        let slot = pools.registry.get(&surf(1)).expect("registered");
        assert!(!slot.content_ready, "nothing has drawn into it yet");
        assert_eq!(
            slot.content_epoch, None,
            "nothing has vouched for its pixels"
        );
        assert_eq!(
            slot.access,
            ResidentAccess::Untouched,
            "nothing has touched it yet"
        );
        assert_eq!(slot.pin_count, 0, "no deferred window holds it yet");
    }

    /// Registration writes the map and the order together.
    ///
    /// `registry` and `registry_order` are one structure split for lookup and
    /// for order: an entry in the map alone is a resident no reclaim can choose,
    /// and one in the order alone is a victim that frees nothing.
    #[test]
    fn registration_writes_both_halves() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );
        pools.register_resident(
            &surf(2),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        assert_eq!(
            pools.registry_order.iter().cloned().collect::<Vec<_>>(),
            vec![surf(1), surf(2)],
            "the order holds both, in registration order"
        );
        assert!(
            pools.registry.contains_key(&surf(1)) && pools.registry.contains_key(&surf(2)),
            "the map holds both"
        );
    }

    /// A resident whose pixels exist nowhere else is never aged out, however
    /// long it sits — and becomes reclaimable the moment something copies them
    /// out.
    ///
    /// This is the MRT-secondary shape exactly: `registry_mark_ready_at` is the
    /// call the secondary attachments take, it is the only thing that ever
    /// happens to them, and they are never pinned and never written back. Aged
    /// out, such a resident does not refuse — `resolve_sampled_source` finds no
    /// resident and falls through to the mapping's guest pages, which hold a
    /// different frame — so the destroy is silent guest-work loss.
    ///
    /// Fails without the gate: `surf(1)` is selected on the first pass, because
    /// `pin_count == 0` is true of a resident that was written back *and* of one
    /// that never was.
    #[test]
    fn a_resident_that_is_the_only_copy_of_its_pixels_is_never_aged_out() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 10, 0);
        // The MRT-secondary path: rendered into, marked ready at the pass's
        // final layout, never pinned, never stamped, never flushed.
        pools.registry_mark_ready_at(&surf(1), vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        pools.registry_touch_at(&surf(1), 10);

        let now = 10 + IDLE_TARGET_AGE_MS + 1;
        assert_eq!(
            pools.plan_idle_drain(now, None),
            Some(Vec::new()),
            "aged past the cutoff and unpinned, but destroying it destroys the frame"
        );

        // Ten more cutoffs' worth of idleness changes nothing: this is not a
        // longer timer, it is a different question.
        let much_later = now + IDLE_TARGET_AGE_MS * 10;
        assert_eq!(
            pools.plan_idle_drain(much_later, None),
            Some(Vec::new()),
            "no age makes destroying the only copy anything but a loss"
        );

        // Something copies the pixels out — a landed flush, a writeback Store —
        // and the same resident is now exactly as reclaimable as any other.
        assert!(
            pools.registry_note_content_copied_out(&surf(1)),
            "the slot is there to be cleared"
        );
        let later_still = much_later + IDLE_DRAIN_INTERVAL_MS + 1;
        assert_eq!(
            pools.plan_idle_drain(later_still, None),
            Some(vec![surf(1)]),
            "with the pixels held elsewhere, reclaiming costs redundant work only"
        );
    }

    /// The allocation-failure reclaim offers up nothing whose only copy is on
    /// the GPU, so a device out of memory refuses the next allocation rather
    /// than silently dropping a client's render target.
    ///
    /// Asserted on [`ResourcePools::recoverable_residents`] rather than on
    /// `reclaim_for_allocation_retry`, because that function is `unsafe` and
    /// needs a live `DeviceContext` to dispose what it chooses; the selection is
    /// the part with the policy in it.
    ///
    /// Fails without the gate: `surf(1)` and `surf(2)` stay in the list after
    /// they are marked ready.
    #[test]
    fn the_allocation_reclaim_offers_up_nothing_that_is_the_only_copy() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        admit(&mut pools, surf(2), 0, 0);
        assert_eq!(
            pools.recoverable_residents(),
            vec![surf(1), surf(2)],
            "both are re-servable from the guest's own pages"
        );

        pools.registry_mark_ready(&surf(1));
        assert_eq!(
            pools.recoverable_residents(),
            vec![surf(2)],
            "the sole copy drops out; the one behind it is still offered"
        );

        pools.registry_mark_ready(&surf(2));
        assert!(
            pools.recoverable_residents().is_empty(),
            "nothing left that can be destroyed without losing a frame"
        );
    }

    /// Every way this device can stop holding a resident answers whether it may
    /// take the only copy of a frame — and the two that may not, do not.
    ///
    /// The two tests above each hold one selector. Neither notices a **third**
    /// way to lose a resident being added, which is the shape the original
    /// defect had: `pin_count == 0` was the predicate at two sites and both were
    /// wrong for the same reason, and nothing compared them. The `match` here is
    /// exhaustive over [`ResidentReclaim`], so a fourth variant does not compile
    /// until its author has said which side of this it falls on.
    ///
    /// `Recreated` is the one exemption and it is not a weakening: the guest
    /// asked `registry_ensure` for this identity at a different geometry,
    /// generation or format, so the pixels being replaced are ones it has
    /// declared it no longer wants. Losing those is the guest's own instruction,
    /// not a policy of ours — and there is nothing to skip, because the
    /// replacement is the point of the call.
    #[test]
    fn no_reclaim_cause_may_take_the_only_copy_of_a_frame() {
        for cause in [
            ResidentReclaim::IdleDrained,
            ResidentReclaim::AllocationReclaimed,
            ResidentReclaim::Recreated,
        ] {
            let mut pools = ResourcePools::new();
            admit(&mut pools, surf(1), 10, 0);
            // The sole-copy shape: rendered into and marked ready, never pinned,
            // never stamped, never written back.
            pools.registry_mark_ready(&surf(1));
            pools.registry_touch_at(&surf(1), 10);
            let aged = 10 + IDLE_TARGET_AGE_MS + 1;

            match cause {
                ResidentReclaim::IdleDrained => assert_eq!(
                    pools.plan_idle_drain(aged, None),
                    Some(Vec::new()),
                    "the idle drain must not offer a sole copy at any age"
                ),
                ResidentReclaim::AllocationReclaimed => assert!(
                    pools.recoverable_residents().is_empty(),
                    "a device out of memory refuses the next allocation rather \
                     than destroying a client's only render target"
                ),
                // Exempt, and the assertion says why rather than skipping: the
                // slot is still the sole copy, and that is not what stops this
                // cause — the guest asking for a different target is.
                ResidentReclaim::Recreated => assert!(
                    pools
                        .registry
                        .get(&surf(1))
                        .is_some_and(|s| s.gpu_only_content),
                    "the exemption is about who asked, not about the flag: the \
                     slot this cause replaces is still marked as the only copy"
                ),
            }
        }
    }

    /// The maintained sole-copy totals agree with a walk of the registry at
    /// every transition, including the ones that move nothing.
    ///
    /// The same defect class as `the_maintained_non_pinned_totals_track_the_walk`
    /// and it matters more here: this population is what says whether protecting
    /// unreproducible content is affordable, so a total that drifts high reads as
    /// a VRAM ceiling that was never approached, and one that drifts low hides
    /// one that was.
    #[test]
    fn the_maintained_sole_copy_totals_track_the_walk() {
        let mut pools = ResourcePools::new();
        let check = |pools: &ResourcePools, what: &str| {
            let walk = {
                let sole = || pools.registry.values().filter(|s| s.gpu_only_content);
                NonPinnedTotals {
                    count: sole().count(),
                    bytes: sole().map(ResourcePools::slot_attachment_bytes).sum(),
                }
            };
            assert_eq!(
                pools.registry_sole_copy, walk,
                "maintained sole-copy totals disagree with the walk after {what}"
            );
        };
        check(&pools, "construction");

        admit_sized(&mut pools, surf(1), 0, 0, (16, 16));
        admit_sized(&mut pools, surf(2), 0, 0, (64, 32));
        check(&pools, "two admits");
        assert_eq!(
            pools.registry_sole_copy,
            NonPinnedTotals::default(),
            "a registered slot nothing has drawn into holds no guest work"
        );

        pools.registry_mark_ready(&surf(1));
        check(&pools, "a draw stored into the first");
        assert_eq!(pools.registry_sole_copy.count, 1);
        // A second store into the same slot is still one slot.
        pools.registry_mark_ready(&surf(1));
        check(&pools, "a second store into the same slot");
        assert_eq!(pools.registry_sole_copy.count, 1);

        pools.registry_mark_ready_at(&surf(2), vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        check(&pools, "the MRT-secondary arm on the second");
        assert_eq!(pools.registry_sole_copy.count, 2);

        assert!(pools.registry_note_content_copied_out(&surf(1)));
        check(&pools, "a copy-out");
        assert_eq!(pools.registry_sole_copy.count, 1);
        // A second copy-out of an already-copied slot must not double-subtract.
        assert!(pools.registry_note_content_copied_out(&surf(1)));
        check(&pools, "a redundant copy-out");
        assert_eq!(pools.registry_sole_copy.count, 1);

        // Death of a sole-copy slot and of a copied-out one. Only the first was
        // ever in the totals, and the guest replacing a resource is the one way
        // a sole-copy slot leaves the registry.
        pools.unregister_resident(&surf(2), ResidentReclaim::AllocationReclaimed);
        check(&pools, "unregistering the sole-copy slot");
        pools.unregister_resident(&surf(1), ResidentReclaim::AllocationReclaimed);
        check(&pools, "unregistering the copied-out slot");
        assert_eq!(pools.registry_sole_copy, NonPinnedTotals::default());

        // A copy-out against an identity holding no resident reports the miss
        // rather than inventing a subtraction.
        assert!(!pools.registry_note_content_copied_out(&surf(1)));
        check(&pools, "a copy-out for an absent identity");
    }

    /// The sole-copy high-water rises on the mark that grows the population, not
    /// only at the capacity walk that also samples it.
    ///
    /// That walk runs on *admission* while this population grows on
    /// `registry_mark_ready`, so a peak folded only there lags by a draw — and a
    /// burst that marks more residents than it admits can retreat below its own
    /// maximum before the next sample, which a high-water would then never
    /// report. This reading is what says whether protecting the class is
    /// affordable, so under-reporting is the dangerous direction.
    ///
    /// Fails without the fold in `set_sole_copy`: the peak stays at zero.
    #[test]
    fn the_sole_copy_high_water_rises_when_the_population_does() {
        let mut pools = ResourcePools::new();
        admit_sized(&mut pools, surf(1), 0, 0, (64, 64));
        admit_sized(&mut pools, surf(2), 0, 0, (64, 64));
        assert_eq!(pools.registry_sole_copy_peak, NonPinnedTotals::default());

        pools.registry_mark_ready(&surf(1));
        pools.registry_mark_ready(&surf(2));
        let at_peak = pools.registry_sole_copy;
        assert_eq!(at_peak.count, 2);
        assert_eq!(
            pools.registry_sole_copy_peak, at_peak,
            "the high-water saw the population at its maximum"
        );

        // The population retreats with no admission in between, which is exactly
        // the shape a walk-sampled peak misses.
        assert!(pools.registry_note_content_copied_out(&surf(1)));
        assert!(pools.registry_note_content_copied_out(&surf(2)));
        assert_eq!(pools.registry_sole_copy, NonPinnedTotals::default());
        assert_eq!(
            pools.registry_sole_copy_peak, at_peak,
            "a high-water does not fall back with the population"
        );
    }

    /// The set an allocation failure may give back is every resident that is
    /// neither pinned nor the only copy of its pixels — with no reference to age.
    ///
    /// The drain's cutoff is a throughput compromise, and by the time the device
    /// has refused an allocation the throughput is already lost; what is left is
    /// whether the draw survives. So this selection deliberately takes residents
    /// the drain would not have touched yet, while still honouring the two
    /// conditions that would make a destroy a lost frame.
    ///
    /// The order is the registry's own, so a caller reclaiming a prefix of it
    /// takes the oldest-created first.
    #[test]
    fn the_allocation_retry_may_reclaim_everything_that_is_not_a_sole_copy_or_pinned() {
        let mut pools = ResourcePools::new();
        let now = 10_000;
        pools.idle_clock_ms = now;
        admit(&mut pools, surf(1), now, 0); // fresh, recoverable -> yes
        admit(&mut pools, surf(2), now, 1); // pinned             -> no
        admit(&mut pools, surf(3), now, 0); // sole copy          -> no
        admit(&mut pools, surf(4), 0, 0); // aged, recoverable  -> yes
        pools.registry_mark_ready(&surf(3));

        assert_eq!(
            pools.recoverable_residents(),
            vec![surf(1), surf(4)],
            "freshness is not a reason to keep one when the alternative is refusing the draw"
        );

        // Copy the sole copy out and it joins the set; pin one of the two and it
        // leaves. Both directions, so this is the predicate and not the order.
        assert!(pools.registry_note_content_copied_out(&surf(3)));
        assert!(pools.pin_resident_target(&surf(1), true));
        assert_eq!(pools.recoverable_residents(), vec![surf(3), surf(4)]);
    }

    /// Only an out-of-memory result is worth a reclaim and a retry.
    ///
    /// The retry exists because out-of-memory describes how much is in use at an
    /// instant rather than anything about the request, so giving memory back can
    /// change the answer. Every other refusal would meet the same answer twice,
    /// and device-lost is answered by recreating the context instead — retrying
    /// an allocation against a lost device only fails again.
    #[test]
    fn only_an_out_of_memory_refusal_is_worth_retrying() {
        use crate::backend::vulkan::engine::vk_call::{VkCall, VkOp};
        let oom = |r| DrawError::VkCall(VkCall::new(VkOp::PoolsBindRegistryTarget, r));
        assert!(oom(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).out_of_memory());
        assert!(
            oom(vk::Result::ERROR_OUT_OF_HOST_MEMORY).out_of_memory(),
            "this device's pools hold host allocations too"
        );
        assert!(!oom(vk::Result::ERROR_DEVICE_LOST).out_of_memory());
        assert!(!oom(vk::Result::ERROR_INITIALIZATION_FAILED).out_of_memory());
        assert!(
            !DrawError::FenceTimeout.out_of_memory(),
            "a non-Vulkan-call refusal is never an allocation failure"
        );
    }

    /// A non-pinned resident untouched for `IDLE_TARGET_AGE_MS` is selected; a
    /// freshly-touched peer and a pinned peer are not. The wall clock advances to
    /// the passed `now_ms` (not a per-call increment), so a static guest that
    /// keeps ticking the poll heartbeat still reclaims stale VRAM.
    #[test]
    fn plan_idle_drain_selects_only_aged_non_pinned() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 10, 0); // aged, non-pinned  -> victim
        admit(&mut pools, surf(2), 10, 1); // aged but PINNED   -> kept
                                           // now = 10 + AGE + 1 so slot 1's cutoff is crossed; a fresh slot is not.
        let now = 10 + IDLE_TARGET_AGE_MS + 1;
        admit(&mut pools, surf(3), now, 0); // fresh            -> kept
        let victims = pools.plan_idle_drain(now, None).expect("pass due");
        assert_eq!(victims, vec![surf(1)], "only the aged non-pinned resident");
        assert_eq!(pools.idle_clock_ms, now, "clock advanced to wall time");
    }

    /// A reclaim records *when*, so the gap the drain censored can be recovered.
    ///
    /// `resident_resample_peak_ms` measures the interval between two reads of a
    /// resident that survived both, so it structurally cannot observe a gap
    /// longer than `IDLE_TARGET_AGE_MS`: past that the resident is gone and the
    /// closing read falls through to the guest's pages, recording an absence
    /// rather than a longer interval. Tuning the cutoff from that peak means
    /// tuning it from a distribution the policy itself truncates.
    ///
    /// The reclaim stamp is what closes the interval from the other side — a
    /// resident read `since` ms after being destroyed had gone at least
    /// `IDLE_TARGET_AGE_MS + since` between uses.
    ///
    /// Fails without the stamp: `reclaimed_recent` carries no time at all.
    #[test]
    fn a_reclaim_records_when_so_the_censored_gap_can_be_recovered() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        pools.idle_clock_ms = 5_000;
        pools.unregister_resident(&surf(1), ResidentReclaim::IdleDrained);

        let (why, at) = pools
            .prior_reclaim_at(&surf(1))
            .expect("a reclaim is recorded with its time");
        assert_eq!(why, ResidentReclaim::IdleDrained);
        assert_eq!(at, 5_000, "stamped with the drain's own clock");

        // The guest comes back 3 s after the destroy, so it had gone at least
        // IDLE_TARGET_AGE_MS + 3000 between uses — a gap the surviving-resident
        // peak could never have reported.
        pools.idle_clock_ms = 8_000;
        assert_eq!(
            pools.idle_clock_ms().saturating_sub(at),
            3_000,
            "the elapsed side of the pair the facade returns"
        );

        assert_eq!(
            pools.prior_reclaim_at(&surf(2)),
            None,
            "an identity never reclaimed has no stamp, and is not guessed at"
        );
    }

    /// The resample bands are fractions of the drain cutoff, so retuning the
    /// cutoff moves them with it and no reading is ever quoted against a bound
    /// it did not come from.
    ///
    /// The boundary that matters is the last one. `past_cutoff` is exactly the
    /// case where a resident was read after sitting longer than the age at which
    /// the drain would have destroyed it, so it survived only because the drain
    /// is throttled to `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass and had not
    /// reached it. `IDLE_TARGET_AGE_MS` itself must therefore land in that band
    /// and not under it: the drain's own comparison is `last_touch_ms <= cutoff`,
    /// so a resident exactly at the cutoff is already a victim.
    #[test]
    fn the_resident_resample_bands_are_fractions_of_the_drain_cutoff() {
        let c = IDLE_TARGET_AGE_MS;
        for (idle, expected) in [
            (0, "resident_resample_lt_eighth_cutoff"),
            (c / 8 - 1, "resident_resample_lt_eighth_cutoff"),
            (c / 8, "resident_resample_lt_quarter_cutoff"),
            (c / 4, "resident_resample_lt_half_cutoff"),
            (c / 2, "resident_resample_under_cutoff"),
            (c - 1, "resident_resample_under_cutoff"),
            (c, "resident_resample_past_cutoff"),
            (u64::MAX, "resident_resample_past_cutoff"),
        ] {
            assert_eq!(
                resident_resample_band(idle),
                expected,
                "idle_ms={idle} against cutoff={c}"
            );
        }
    }

    /// "This device destroyed a resident for this identity" and "this device
    /// never held one" must be distinguishable, and a re-created resident must
    /// report neither.
    ///
    /// These are the three states behind `resident_absent_after_reclaim`, whose
    /// body is `if present { None } else { prior_reclaim(..) }`. The composition
    /// is what matters: `prior_reclaim` alone keeps answering for the life of
    /// `RECLAIM_HISTORY`, so consulting it without the presence check would make
    /// a resident that was reclaimed and then re-created keep reporting itself
    /// as destroyed — and the caller uses that answer to decide whether falling
    /// through to the guest's pages is sound.
    ///
    /// The facade itself needs the engine lock and so cannot be unit-tested; the
    /// logic it composes is here.
    #[test]
    fn a_recreated_resident_no_longer_reports_the_reclaim_that_took_it() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);

        // Never held: no record, and nothing to mistake for one.
        assert_eq!(pools.prior_reclaim(&surf(2)), None);
        assert!(!pools.registry.contains_key(&surf(2)));

        // Held: present, so the facade short-circuits before prior_reclaim.
        assert!(pools.registry.contains_key(&surf(1)));

        // Destroyed: absent, and the cause survives.
        pools.unregister_resident(&surf(1), ResidentReclaim::IdleDrained);
        assert!(!pools.registry.contains_key(&surf(1)));
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained)
        );

        // Re-created: the record is still in history, so only the presence
        // check keeps this from reading as destroyed.
        admit(&mut pools, surf(1), 0, 0);
        assert!(
            pools.registry.contains_key(&surf(1)),
            "the presence check is the only thing separating this from the case above"
        );
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained),
            "history is deliberately not cleared on re-admit, which is why the \
             presence check cannot be dropped"
        );
    }

    /// The resample peak is the worst gap the boot ever saw, not the last one.
    ///
    /// A high-water, so a large gap early is not erased by a run of small ones
    /// after it — which is the whole reason it is not a windowed reading. The
    /// margin question is "how close did this boot ever come to
    /// `IDLE_TARGET_AGE_MS`", and a gap that peaks between two census samples is
    /// exactly what an instantaneous value misses.
    ///
    /// Fails without the fix: nothing records the gap at all.
    #[test]
    fn the_resample_peak_holds_the_worst_gap_not_the_latest() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        assert_eq!(pools.resident_resample_peak_ms(), 0);

        // A 900 ms gap, then a 100 ms one. The peak must keep the 900.
        pools.idle_clock_ms = 900;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(pools.resident_resample_peak_ms(), 900);
        pools.idle_clock_ms = 1_000;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(
            pools.resident_resample_peak_ms(),
            900,
            "a smaller later gap must not lower the high-water"
        );

        // And a larger one does raise it.
        pools.idle_clock_ms = 1_000 + IDLE_TARGET_AGE_MS;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_TARGET_AGE_MS);

        // A read of an identity the registry does not hold records nothing —
        // there is no gap to measure, and counting it as one would report a
        // margin that no resident ever spent.
        pools.idle_clock_ms = u64::MAX;
        pools.registry_note_sampled_use(&surf(99));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_TARGET_AGE_MS);
    }

    /// A resident that a draw only ever *samples* survives the idle drain.
    ///
    /// This is the compositor-backdrop case: the desktop behind a translucent
    /// panel is rendered once and then read by every vibrancy draw over it.
    /// Before [`ResourcePools::registry_note_sampled_use`] existed, reading a
    /// resident refreshed nothing — `registry_ensure` (render *into*) and the
    /// present touch were the only writers of `last_touch_ms` — so the backdrop
    /// aged exactly as if it had been abandoned and the drain terminally
    /// destroyed it. Every later draw sampling it then refused with
    /// `vk_draw_exec_sampled_resident_missing`, and nothing recreates a resident
    /// except a draw rendering into that identity, so the refusal held for the
    /// rest of the boot.
    ///
    /// Fails without the fix: with the body of `registry_note_sampled_use`
    /// removed, `surf(1)` is selected and the assertion reports it as a victim.
    #[test]
    fn a_sampled_only_resident_is_not_aged_out() {
        let mut pools = ResourcePools::new();
        // Aged past the cutoff by every measure the drain has, and never drawn
        // into again — only read.
        admit(&mut pools, surf(1), 10, 0);
        admit(&mut pools, surf(2), 10, 0);
        let now = 10 + IDLE_TARGET_AGE_MS + 1;
        // The drain's clock has to be current before a use can be recorded
        // against it; the real caller advances it from the poll heartbeat.
        pools.plan_idle_drain(now, None);
        pools.registry_note_sampled_use(&surf(1));
        // A second pass, far enough after the first to clear the throttle.
        let later = now + IDLE_DRAIN_INTERVAL_MS + 1;
        let victims = pools.plan_idle_drain(later, None).expect("pass due");
        assert!(
            !victims.contains(&surf(1)),
            "a resident being sampled is in use and must not be destroyed"
        );
        assert!(
            victims.contains(&surf(2)),
            "its untouched peer is still reclaimed — the fix must not disable the drain"
        );
    }

    /// A pinned resident is never offered to the allocation-failure reclaim, and
    /// a registry of nothing else offers an empty list — so the allocation
    /// refuses rather than the device dropping content whose only copy is on the
    /// GPU.
    #[test]
    fn the_allocation_reclaim_never_offers_a_pinned_resident() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 1);
        admit(&mut pools, surf(2), 0, 2);
        assert!(
            pools.recoverable_residents().is_empty(),
            "every entry is pinned, so there is nothing to give back"
        );
        admit(&mut pools, surf(3), 10, 0);
        assert_eq!(
            pools.recoverable_residents(),
            vec![surf(3)],
            "the one unpinned resident is the only candidate"
        );
    }

    /// Every removal of a live registry entry leaves a record naming the path.
    ///
    /// This is the invariant `note_resident_reclaimed` claims ("called from
    /// every site that removes a live registry entry") and that the MRT
    /// secondary recreate arm broke while it was a copy: it removed the entry
    /// and recorded nothing, so `prior_reclaim` answered `None` — which
    /// `exec` reports as "never existed" for a resident this device had just
    /// taken. Routing all three sites through `unregister_resident` is what
    /// makes the record unconditional.
    #[test]
    fn unregistering_a_resident_always_names_why_and_leaves_the_order_clean() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        admit(&mut pools, surf(2), 0, 0);

        assert!(pools
            .unregister_resident(&surf(1), ResidentReclaim::Recreated)
            .is_some());
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::Recreated),
            "a removed resident must say which path took it"
        );
        assert!(!pools.registry.contains_key(&surf(1)));
        assert!(
            !pools.registry_order.contains(&surf(1)),
            "the order list must not keep a key the map no longer holds"
        );
        assert!(
            pools.registry_order.contains(&surf(2)),
            "an untouched resident keeps its place"
        );

        // An identity that held nothing is not a removal, so it gets no record.
        // Writing one would make `prior_reclaim` claim this device took a
        // resident that never existed — the exact confusion it exists to avoid.
        assert!(pools
            .unregister_resident(&surf(3), ResidentReclaim::AllocationReclaimed)
            .is_none());
        assert_eq!(pools.prior_reclaim(&surf(3)), None);
    }

    /// The reclaim history answers which path took a resident, and says "no
    /// record" rather than guessing when it cannot.
    ///
    /// This is what lets `vk_draw_exec_sampled_resident_missing` distinguish a
    /// resident reclaimed out from under an active reader from one the guest
    /// never rendered into. Both present as an absent registry entry, and the
    /// two have different repairs.
    #[test]
    fn the_reclaim_history_names_the_path_and_is_bounded() {
        let mut pools = ResourcePools::new();
        pools.note_resident_reclaimed(&surf(1), ResidentReclaim::IdleDrained);
        pools.note_resident_reclaimed(&surf(2), ResidentReclaim::AllocationReclaimed);
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained)
        );
        assert_eq!(
            pools.prior_reclaim(&surf(2)),
            Some(ResidentReclaim::AllocationReclaimed)
        );
        assert_eq!(
            pools.prior_reclaim(&surf(3)),
            None,
            "an identity never reclaimed has no record, and is not guessed at"
        );
        // The most recent verdict wins: an identity recreated after being
        // evicted is not still reported as evicted.
        pools.note_resident_reclaimed(&surf(1), ResidentReclaim::Recreated);
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::Recreated)
        );
        // Bounded: the oldest record falls out rather than the history growing
        // without limit, and falling out reads as no record.
        for i in 0..RECLAIM_HISTORY as u32 {
            pools.note_resident_reclaimed(&surf(1000 + i), ResidentReclaim::AllocationReclaimed);
        }
        assert!(pools.reclaimed_recent.len() <= RECLAIM_HISTORY);
        assert_eq!(
            pools.prior_reclaim(&surf(2)),
            None,
            "aged out of the window"
        );
    }

    /// The reclaim pass is throttled to `IDLE_DRAIN_INTERVAL_MS`: a second call
    /// inside the interval selects nothing even though a resident is aged, so the
    /// ~244 Hz poll cadence cannot empty the registry at once. The clock still
    /// advances (admits stay fresh).
    #[test]
    fn plan_idle_drain_throttles_between_passes() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        let t0 = IDLE_TARGET_AGE_MS + 1;
        assert_eq!(pools.plan_idle_drain(t0, None), Some(vec![surf(1)]));
        // Simulate the dispose the real caller (advance_registry_touch_and_drain)
        // performs for each selected victim.
        pools.registry.remove(&surf(1));
        pools.registry_order.retain(|k| k != &surf(1));
        admit(&mut pools, surf(2), 0, 0);
        // A call one ms later is inside the interval → no pass (None), despite
        // surf(2) being aged.
        assert_eq!(
            pools.plan_idle_drain(t0 + 1, None),
            None,
            "throttled: no pass"
        );
        assert_eq!(
            pools.idle_clock_ms,
            t0 + 1,
            "clock still advances when throttled"
        );
        // Past the interval → the next aged resident is selected.
        assert_eq!(
            pools.plan_idle_drain(t0 + IDLE_DRAIN_INTERVAL_MS, None),
            Some(vec![surf(2)])
        );
    }

    /// Each pass selects at most `IDLE_TARGET_DRAIN_MAX_PER_CALL` so a huge stale
    /// set drains gradually (no dispose storm that would be a P3 hitch itself).
    #[test]
    fn plan_idle_drain_bounds_batch_per_pass() {
        let mut pools = ResourcePools::new();
        for i in 0..(IDLE_TARGET_DRAIN_MAX_PER_CALL as u32 + 5) {
            admit(&mut pools, surf(100 + i), 0, 0);
        }
        let victims = pools
            .plan_idle_drain(IDLE_TARGET_AGE_MS + 1, None)
            .expect("pass due");
        assert_eq!(victims.len(), IDLE_TARGET_DRAIN_MAX_PER_CALL);
    }

    /// A pass with no registry victim but live staging traffic is NOT settled.
    ///
    /// This is the case the victim count alone cannot see and the one that
    /// actually happens: a steady animation re-uses the same render targets, so
    /// nothing ages out and every pass reads as quiet, while the upload path runs
    /// flat out. Measured under testufo the trim fired about once a second
    /// throughout the load and cost 607 re-allocations of the 8 MiB full-frame
    /// staging bucket at 12.6 ms each.
    #[test]
    fn a_pass_with_no_victims_but_live_uploads_is_not_settled() {
        let mut pools = ResourcePools::new();
        // Quiet the gate first, so the assertion below is about uploads and not
        // about the counter still warming up.
        for _ in 0..SETTLED_PASSES_FOR_BUFFER_TRIM {
            pools.note_drain_settled(0);
        }
        assert!(
            pools.note_drain_settled(0),
            "no victims, no uploads → settled"
        );

        // One staging acquire between passes — no victim, still not settled.
        pools.staging_hits += 1;
        assert!(
            !pools.note_drain_settled(0),
            "uploads ran between passes; the buffer pools must not be trimmed"
        );
        // …and the gate stays shut while uploads keep flowing, however many
        // zero-victim passes go by.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM * 3) {
            pools.staging_misses += 1;
            assert!(!pools.note_drain_settled(0), "still uploading");
        }
        // Uploads stop: the gate reopens after the usual consecutive passes.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "counter restarted from zero");
        }
        assert!(pools.note_drain_settled(0), "settled once uploads stopped");
    }

    /// The HOST_VISIBLE buffer trim gate: only permitted after
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive zero-victim passes, and any
    /// pass that drains ≥1 victim (active churn) resets the counter — so a
    /// staging buffer cannot be freed and re-alloc'd mid-video.
    #[test]
    fn note_drain_settled_gates_buffer_trim_on_consecutive_idle() {
        let mut pools = ResourcePools::new();
        // Fewer than the threshold of quiet passes: no buffer trim yet.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "not settled enough yet");
        }
        // The Nth consecutive zero-victim pass crosses the threshold.
        assert!(
            pools.note_drain_settled(0),
            "N consecutive settled passes → trim allowed"
        );
        // A subsequent quiet pass stays allowed.
        assert!(pools.note_drain_settled(0), "stays settled");
        // A pass that drains a victim (active churn) resets the counter…
        assert!(
            !pools.note_drain_settled(1),
            "any drained victim resets settled state"
        );
        // …and the gate stays closed until the run rebuilds.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "counter restarted from zero");
        }
        assert!(pools.note_drain_settled(0), "settled again after rebuild");
    }

    /// The presented target passed as `display` is stamped to the current clock
    /// every call, so even though it is only resolved via `registry_get` (never
    /// re-drawn on a static page) it never ages out from under the display.
    #[test]
    fn plan_idle_drain_keeps_display_target_alive() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // would be aged...
        let now = IDLE_TARGET_AGE_MS + 500;
        // ...but it is the presented target this frame.
        let victims = pools
            .plan_idle_drain(now, Some(&surf(1)))
            .expect("pass due");
        assert!(victims.is_empty(), "display target must not be reclaimed");
        assert_eq!(
            pools.registry.get(&surf(1)).unwrap().last_touch_ms,
            now,
            "display target stamped fresh"
        );
    }

    /// `registry_touch_at` refreshes a target against the idle-drain cutoff
    /// without going through the draw path, so a target that is registered but
    /// not being drawn survives a static desktop interval when a caller still
    /// needs it.
    #[test]
    fn registry_touch_at_defers_the_idle_drain_for_an_untouched_target() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // displayed target
        admit(&mut pools, surf(4), 0, 0); // registered but undrawn, otherwise aged
        let now = IDLE_TARGET_AGE_MS + 500;

        pools.registry_touch_at(&surf(4), now);
        let victims = pools
            .plan_idle_drain(now, Some(&surf(1)))
            .expect("pass due");
        assert_eq!(
            victims,
            Vec::<TargetIdentity>::new(),
            "the display target and the touched target both survive"
        );
        assert_eq!(
            pools.registry.get(&surf(4)).unwrap().last_touch_ms,
            now,
            "the touched target is stamped at the touch time"
        );
    }

    /// The byte band is not the slot band scaled, and a boot needs both.
    ///
    /// A slot count once bounded this population while the resource it protected
    /// was bytes. This drives the difference directly: two populations of the
    /// same size, one of 16x16 scratch and one of 4K attachments, are
    /// indistinguishable to the slot band and four orders of magnitude apart in
    /// VRAM. That gap is why this counter exists, and it is what retired the
    /// count — see [`ResourcePools::recoverable_residents`].
    #[test]
    fn the_registry_byte_band_separates_populations_the_slot_band_cannot() {
        const TEXEL: u64 = 4; // SCANOUT_FORMAT, the shape `new_resident` builds
        const SMALL: (u32, u32) = (16, 16);
        const UHD: (u32, u32) = (3840, 2160);
        let mut pools = ResourcePools::new();
        for i in 1..=3u32 {
            admit_sized(&mut pools, surf(i), 10, 0, SMALL);
        }
        pools.note_registry_reach();
        let (slots_small, bytes_small) = pools.registry_pressure_stats();
        assert_eq!(slots_small, 3);
        assert_eq!(bytes_small, 3 * 16 * 16 * TEXEL);

        // The same slot count at 4K geometry. Pinned peers stay out of both
        // bands, so the two readings describe one population.
        let mut big = ResourcePools::new();
        for i in 1..=3u32 {
            admit_sized(&mut big, surf(i), 10, 0, UHD);
        }
        admit_sized(&mut big, surf(9), 10, 1, UHD);
        big.note_registry_reach();
        let (slots_big, bytes_big) = big.registry_pressure_stats();
        assert_eq!(
            slots_big, slots_small,
            "the slot band cannot tell the two populations apart"
        );
        assert_eq!(
            bytes_big,
            3 * 3840 * 2160 * TEXEL,
            "the byte band can, and the pinned 4K peer is not in it"
        );

        // A high-water mark, like its sibling: the population falling does not
        // lower it, or a burst that drains between two census samples reads as
        // if it never happened.
        for i in 1..=3u32 {
            big.unregister_resident(&surf(i), ResidentReclaim::AllocationReclaimed);
        }
        big.note_registry_reach();
        assert_eq!(
            big.registry_pressure_stats().1,
            bytes_big,
            "the byte band holds its peak"
        );
    }

    /// The maintained non-pinned totals still say what a full walk would.
    ///
    /// They stopped being a walk so the population can grow without a per-admit
    /// O(n) scan, and the cost of that is three writers that
    /// can fall out of step with the registry in silence. A slot that stopped
    /// being counted makes the population read smaller than it is, which is the
    /// direction that lets a bound sit above itself — so every transition is
    /// driven here and diffed against the walk after each one.
    ///
    /// Counted pins are the part worth driving twice: only the 0 <-> 1 crossings
    /// may move the totals, a second pin must not remove the slot again, and an
    /// unpin that saturates at zero must not add a slot that is already there.
    #[test]
    fn the_maintained_non_pinned_totals_track_the_walk() {
        let mut pools = ResourcePools::new();
        let check = |pools: &ResourcePools, what: &str| {
            assert_eq!(
                pools.registry_non_pinned,
                pools.non_pinned_registry_totals_by_walk(),
                "maintained totals disagree with the walk after {what}"
            );
        };
        check(&pools, "construction");

        admit_sized(&mut pools, surf(1), 0, 0, (16, 16));
        admit_sized(&mut pools, surf(2), 0, 0, (64, 32));
        check(&pools, "two admits");
        assert_eq!(pools.registry_non_pinned.count, 2);

        // First pin removes it; the second must not remove it twice.
        assert!(pools.pin_resident_target(&surf(1), true));
        check(&pools, "first pin");
        assert_eq!(pools.registry_non_pinned.count, 1);
        assert!(pools.pin_resident_target(&surf(1), true));
        check(&pools, "second pin");
        assert_eq!(
            pools.registry_non_pinned.count, 1,
            "a second pin moves nothing"
        );

        // First unpin leaves a holder, so it stays out; the second returns it.
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "first unpin");
        assert_eq!(pools.registry_non_pinned.count, 1);
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "second unpin");
        assert_eq!(
            pools.registry_non_pinned.count, 2,
            "the last unpin returns it"
        );

        // A spurious unpin saturates at zero and must not add it again.
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "unpin below zero");
        assert_eq!(pools.registry_non_pinned.count, 2);

        // Death, of a pinned slot and an unpinned one — only the unpinned one
        // was ever in the totals.
        assert!(pools.pin_resident_target(&surf(2), true));
        check(&pools, "pinning the second");
        pools.unregister_resident(&surf(2), ResidentReclaim::AllocationReclaimed);
        check(&pools, "unregistering a pinned resident");
        pools.unregister_resident(&surf(1), ResidentReclaim::AllocationReclaimed);
        check(&pools, "unregistering an unpinned resident");
        assert_eq!(pools.registry_non_pinned, NonPinnedTotals::default());

        // And an unregister of something that was never there.
        pools.unregister_resident(&surf(7), ResidentReclaim::AllocationReclaimed);
        check(&pools, "unregistering an absent identity");
    }

    /// The registry reach band records the highest population, and does not
    /// fall back when residents go away.
    ///
    /// This is the whole instrument now that no count bounds the population: the
    /// peak is what says how far a workload reached, and `peak_bytes` beside it
    /// is what says whether a slot count could ever have stood for VRAM. It
    /// answered no, which is what retired that count. AGENTS.md states the rule
    /// this implements: band the requested reach before widening or narrowing any
    /// table.
    ///
    /// Two properties, because only the pair is a high-water mark. It has to
    /// rise with the population, and it has to *stay* when the population drops
    /// — an instrument that tracked the current value would report whatever the
    /// registry happened to hold at census time and miss every burst, which is
    /// the only thing the cap exists for.
    ///
    /// Pinned residents are excluded — they are the population the reclaim paths
    /// refuse to take — so a pinned peer must not inflate the reading.
    #[test]
    fn the_registry_reach_band_holds_the_peak_and_ignores_pinned_residents() {
        let mut pools = ResourcePools::new();
        assert_eq!(
            pools.registry_pressure_stats(),
            (0, 0),
            "a fresh pools has neither reach nor footprint"
        );

        admit(&mut pools, surf(1), 10, 0);
        admit(&mut pools, surf(2), 10, 0);
        admit(&mut pools, surf(3), 10, 1); // pinned -- excluded from the band
        pools.note_registry_reach();
        assert_eq!(
            pools.registry_pressure_stats().0,
            2,
            "the band took the non-pinned population, excluding the pinned peer"
        );

        admit(&mut pools, surf(4), 10, 0);
        pools.note_registry_reach();
        assert_eq!(pools.registry_pressure_stats().0, 3, "the band rose");

        // Every non-pinned resident goes away, leaving only the pinned peer, so
        // the non-pinned population returns to zero. A current-value reading
        // would now report nothing at all and the burst above would be
        // invisible — which is exactly the failure this band prevents.
        // Through `unregister_resident`, not a hand-written map+order removal:
        // that pair is what the maintained non-pinned totals hang off, and a
        // test that wrote both itself would leave them counting residents that
        // are gone.
        for id in [surf(1), surf(2), surf(4)] {
            pools.unregister_resident(&id, ResidentReclaim::AllocationReclaimed);
        }
        assert_eq!(
            pools.non_pinned_registry_len(),
            0,
            "the population really fell, and the pinned peer never counted"
        );
        pools.note_registry_reach();
        assert_eq!(
            pools.registry_pressure_stats().0,
            3,
            "the peak is a high-water mark, not the current population"
        );
    }
}
