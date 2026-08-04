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
            .memory_type_for(req.memory_type_bits, MemoryClass::DeviceLocal)
            .ok_or({
                DrawError::Unsupported(
                    super::reason::DrawReason::NoDeviceLocalMemoryForStorageImage {
                        memory_type_bits: req.memory_type_bits,
                    },
                )
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            AllocSite::StorageImage,
        )
        .map_err(|e| {
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocStorageImage, e))
        })?;
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
                    .subresource_range(super::color_subresource_range()),
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
        if self
            .compute_storage_registry
            .get(&identity)
            .map(|resident| resident.slot.key != key)
            .unwrap_or(false)
        {
            if let Some(old) = self.compute_storage_registry.remove(&identity) {
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.slot.image,
                        view: old.slot.view,
                        memory: old.slot.memory,
                    },
                );
            }
            self.compute_storage_order
                .retain(|entry| entry != &identity);
        }
        let now = self.idle_clock_ms;
        if let Some(resident) = self.compute_storage_registry.get_mut(&identity) {
            resident.last_touch_ms = now;
            return Ok(ResidentStorageImageUse {
                slot: resident.slot,
                layout: resident.layout,
                generation_match: resident.generation == seed_generation,
            });
        }

        // LRU sweep skips pinned residents (deferred-writeback content whose
        // only copy is on the GPU). Bounded by one full rotation; if every
        // entry is pinned the registry soft-exceeds the cap rather than lose
        // unflushed content.
        let mut rotations = self.compute_storage_order.len();
        while self.compute_storage_registry.len() >= COMPUTE_STORAGE_REGISTRY_CAP && rotations > 0 {
            rotations -= 1;
            let Some(oldest) = self.compute_storage_order.front().copied() else {
                break;
            };
            if self
                .compute_storage_registry
                .get(&oldest)
                .is_some_and(|resident| resident.pinned)
            {
                self.compute_storage_order.pop_front();
                self.compute_storage_order.push_back(oldest);
                continue;
            }
            self.compute_storage_order.pop_front();
            if let Some(old) = self.compute_storage_registry.remove(&oldest) {
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

        // Reuse the common allocator, then detach its bookkeeping copy from
        // the transient live list: the registry now owns this allocation.
        let slot = self.acquire_storage_image(ctx, key, counters)?;
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
                layout: vk::ImageLayout::UNDEFINED,
                pinned: false,
                last_touch_ms: now,
            },
        );
        self.compute_storage_order.push_back(identity);
        Ok(ResidentStorageImageUse {
            slot,
            layout: vk::ImageLayout::UNDEFINED,
            generation_match: false,
        })
    }

    pub(crate) fn mark_resident_storage_image(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        generation: u32,
        layout: vk::ImageLayout,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.generation = generation;
            resident.layout = layout;
        }
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

    /// Record the post-flush layout of a resident (the flush read transitions
    /// it to TRANSFER_SRC_OPTIMAL).
    pub(crate) fn set_resident_storage_layout(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        layout: vk::ImageLayout,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.layout = layout;
        }
    }

    /// Generation of a resident compute storage image, if one is registered.
    /// Read-only — used by the runtime to decide a stage-time guest-read skip.
    pub(crate) fn compute_resident_generation(
        &self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<u32> {
        self.compute_storage_registry
            .get(identity)
            .map(|resident| resident.generation)
    }

    /// Generation + engine format of a resident compute storage image, if one
    /// is registered. Read-only — used by the runtime to decide a stage-time
    /// copy-on-sample skip (the format must match what the sampled view will
    /// bind, or the engine's resident-bind shape guard would fail every run).
    pub(crate) fn compute_resident_sample_source(
        &self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        self.compute_storage_registry
            .get(identity)
            .map(|resident| (resident.generation, resident.slot.key.format))
    }

    /// Snapshot of a resident storage image for a copy-on-sample source:
    /// `(image, key, generation, current layout)`. Read-only.
    pub(crate) fn compute_resident_snapshot(
        &self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(vk::Image, StorageImageKey, u32, vk::ImageLayout)> {
        self.compute_storage_registry.get(identity).map(|resident| {
            (
                resident.slot.image,
                resident.slot.key,
                resident.generation,
                resident.layout,
            )
        })
    }

    // --- Target registry (workstream D) ------------------------------------

    pub(crate) fn registry_get(&self, identity: &TargetIdentity) -> Option<&ResidentTargetSlot> {
        self.registry.get(identity)
    }

    /// Ensure a resident target exists for `identity` with the given geometry + pass.
    /// Image/memory persist across Load vs Clear render-pass changes; only the
    /// framebuffer is rebuilt when the pass handle differs.
    /// `protect` shields one additional identity (a same-draw GPU seed
    /// source) from the capacity sweep, exactly like a pinned slot.
    #[allow(
        clippy::too_many_arguments,
        reason = "resident creation mirrors the target identity, format, seed, and protection set"
    )]
    pub(crate) unsafe fn registry_ensure(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
        generation: u64,
        bgra: bool,
        protect: Option<&TargetIdentity>,
        counters: &EngineCounters,
    ) -> Result<&ResidentTargetSlot, DrawError> {
        // Compatible geometry + gen + format: reuse image; rebuild FB if pass
        // changed. A format (bgra) change must recreate the image, not just the
        // framebuffer — an RGBA image under a BGRA pass is invalid.
        if let Some(slot) = self.registry.get(&identity) {
            if slot.width == width
                && slot.height == height
                && slot.generation == generation
                && slot.bgra == bgra
            {
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
                let old_fb = slot.framebuffer;
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
                self.dispose(&ctx.device, DeferredHandle::Framebuffer(old_fb));
                let slot = self.registry.get_mut(&identity).unwrap();
                slot.framebuffer = framebuffer;
                slot.render_pass = render_pass;
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(self.registry.get(&identity).unwrap());
            }
            // Geometry/gen mismatch → destroy and recreate.
            if let Some(old) = self.registry.remove(&identity) {
                self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
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
                if old.generation != generation {
                    counters
                        .gen_mismatch
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            self.registry_order.retain(|k| k != &identity);
        }
        // Cap the *non-pinned* (evictable) population at REGISTRY_CAP, shielding
        // the just-resolved `protect` identity from its own eviction.
        self.evict_registry_to_cap(ctx, counters, protect);
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        let format = translate::pixel::resident_color(bgra);
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
                    .subresource_range(super::color_subresource_range()),
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
        let touch_ms = self.idle_clock_ms;
        self.registry.insert(
            identity.clone(),
            ResidentTargetSlot {
                image,
                memory,
                view,
                framebuffer,
                render_pass,
                width,
                height,
                generation,
                content_ready: false,
                content_epoch: None,
                layout: vk::ImageLayout::UNDEFINED,
                bgra,
                color_format: format,
                pin_count: 0,
                last_touch_ms: touch_ms,
            },
        );
        self.registry_order.push_back(identity.clone());
        Ok(self.registry.get(&identity).unwrap())
    }

    /// Ensure a resident color attachment of an arbitrary Vulkan format (MRT
    /// secondary path — the primary single-RT `registry_ensure` only speaks
    /// `bgra`). No per-slot framebuffer is built: a secondary attachment is
    /// only ever bound as attachment N of an ad-hoc MRT framebuffer or sampled
    /// via its view, never as a standalone single-RT target. Reuse requires an
    /// exact (geometry, generation, format) match. Returns (image, view).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn registry_ensure_color(
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
            if slot.width == width
                && slot.height == height
                && slot.generation == generation
                && slot.color_format == format
            {
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok((slot.image, slot.view));
            }
            // Geometry / gen / format mismatch → destroy and recreate.
            if let Some(old) = self.registry.remove(&identity) {
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
            self.registry_order.retain(|k| k != &identity);
        }
        // Cap the *non-pinned* population (skip pinned slots), same LRU
        // discipline as the primary `registry_ensure` — pinned deferred windows
        // are bounded separately and excluded from the cap count. No `protect`
        // here: this color path has no just-resolved identity to shield.
        self.evict_registry_to_cap(ctx, counters, None);
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating — same recycle discipline as the primary
        // `registry_ensure` (the usage set is identical, so images cross-flow
        // between the two paths by geometry+format). Skips the create/alloc/bind/
        // view + their note_create/note_alloc; recycled contents are stale, so
        // the slot below is inserted layout=UNDEFINED / content_ready=false.
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
                .memory_type_for(ireq.memory_type_bits, MemoryClass::DeviceLocal)
                .ok_or_else(|| {
                    ctx.device.destroy_image(image, None);
                    DrawError::Unsupported(
                        super::reason::DrawReason::NoDeviceLocalMemoryForMrtSecondary {
                            memory_type_bits: ireq.memory_type_bits,
                        },
                    )
                })?;
            let memory = allocate_memory_timed(
                ctx,
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(ireq.size)
                    .memory_type_index(imt),
                AllocSite::MrtSecondary,
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
                        .subresource_range(super::color_subresource_range()),
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
        let touch_ms = self.idle_clock_ms;
        self.registry.insert(
            identity.clone(),
            ResidentTargetSlot {
                image,
                memory,
                view,
                framebuffer: vk::Framebuffer::null(),
                render_pass: vk::RenderPass::null(),
                width,
                height,
                generation,
                content_ready: false,
                content_epoch: None,
                layout: vk::ImageLayout::UNDEFINED,
                bgra: format == translate::pixel::SCANOUT_FORMAT,
                color_format: format,
                pin_count: 0,
                last_touch_ms: touch_ms,
            },
        );
        self.registry_order.push_back(identity.clone());
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
    /// `exec.rs` creates one per draw that carries depth state and disposes it at
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
            .memory_type_for(ireq.memory_type_bits, MemoryClass::DeviceLocal)
            .ok_or_else(|| {
                ctx.device.destroy_image(image, None);
                DrawError::Unsupported(super::reason::DrawReason::NoDeviceLocalMemoryForDepth {
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

    /// Mark a resident ready with an explicit post-pass layout (MRT secondary
    /// resolves to SHADER_READ_ONLY_OPTIMAL; the primary uses
    /// `registry_mark_ready`'s TRANSFER_SRC_OPTIMAL).
    pub(crate) fn registry_mark_ready_at(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            slot.layout = layout;
        }
    }

    /// Pin/unpin a resident render target against LRU eviction (deferred
    /// render Stores). Pins are counted, not boolean: a surface can have several
    /// deferred windows armed at once and each holds one count, so the slot
    /// stays protected until every holder unpins. Returns false when the identity is absent or (for pinning) its
    /// content is not ready — callers must fall back to the synchronous
    /// Store. Unpin saturates at zero (a spurious unpin never underflows).
    pub(crate) fn pin_resident_target(&mut self, identity: &TargetIdentity, pinned: bool) -> bool {
        if let Some(slot) = self.registry.get_mut(identity) {
            if pinned {
                if !slot.content_ready {
                    return false;
                }
                slot.pin_count += 1;
            } else {
                slot.pin_count = slot.pin_count.saturating_sub(1);
            }
            return true;
        }
        false
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
            // Draw pass final_layout is TRANSFER_SRC_OPTIMAL.
            slot.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
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

    pub(crate) fn registry_set_layout(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.layout = layout;
        }
    }
}

#[cfg(test)]
mod pin_count_tests {
    use super::*;

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
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            bgra: true,
            color_format: translate::pixel::SCANOUT_FORMAT,
            pin_count: 0,
            last_touch_ms: 0,
        }
    }

    fn pinned_identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 1,
            width: 16,
            height: 16,
            generation: 0,
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
        rgba.bgra = false;
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
        }
    }

    /// Admit a resident with an explicit last-touch stamp and pin count.
    fn admit(pools: &mut ResourcePools, id: TargetIdentity, last_touch_ms: u64, pin: u32) {
        let mut slot = dummy_slot(true);
        slot.last_touch_ms = last_touch_ms;
        slot.pin_count = pin;
        pools.registry_order.push_back(id.clone());
        pools.registry.insert(id, slot);
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
}

impl ResourcePools {
    /// The depth-stencil attachment for this geometry, built once and kept.
    ///
    /// [`Self::create_transient_depth`] builds a fresh image every call, which
    /// is right for a depth buffer a single draw clears and forgets and wrong
    /// for a stencil: a Metal render pass clears its stencil once and then has
    /// one draw write a mask and the next test against it. A per-draw image
    /// cannot carry that, and the fill draw ends up testing against contents
    /// nothing wrote.
    ///
    /// Keyed by geometry and by whether a stencil aspect is wanted, because
    /// those are what change the image. A different key disposes the old one
    /// through the deferred queue, so a submission still referencing it is not
    /// pulled out from under the GPU.
    pub(crate) unsafe fn acquire_depth_stencil(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
        with_stencil: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), DrawError> {
        let key = (width, height, with_stencil);
        if let Some((kept_key, parts)) = self.depth_stencil_keep {
            if kept_key == key {
                return Ok(parts);
            }
            let (image, memory, view) = parts;
            self.dispose(
                &ctx.device,
                DeferredHandle::Image {
                    image,
                    view,
                    memory,
                },
            );
            self.depth_stencil_keep = None;
        }
        let parts = self.create_transient_depth(ctx, width, height, with_stencil, counters)?;
        self.depth_stencil_keep = Some((key, parts));
        Ok(parts)
    }
}
