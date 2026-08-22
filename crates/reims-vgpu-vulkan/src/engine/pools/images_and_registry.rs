//! Guest-identity keyed render and storage residents.
//!
//! A live entry outlives submissions and remains resident until its serialized
//! resource ends, its identity is explicitly replaced, or allocation pressure
//! requires recovery. Fences govern in-flight safety; elapsed time does not end
//! resource lifetime.

use super::*;
use crate::engine::types::TargetKeyDivergence;
use reims_vgpu_core::pixel_format::SwizzlePlan;

/// Band observed intervals between sampled uses. This is a reuse-distance
/// diagnostic only; no residency decision reads it.
fn resident_resample_band(idle_ms: u64) -> &'static str {
    let cutoff = IDLE_MAINTENANCE_START_MS;
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

/// Whether a sampled declaration names the exact imported allocation already
/// owned by a target resident.
///
/// Reusing the resident is correctness, not a cache hit. Two Vulkan images
/// bound over the same external bytes have independent layout state, and image
/// aliasing only preserves a common interpretation under stricter identical-
/// creation rules. One backing therefore has one image whenever the semantic
/// view fits it; format interpretation remains an image-view concern.
fn guest_sampled_matches_resident(
    slot: &ResidentTargetSlot,
    memory: &reims_vgpu_memory::GuestTargetMemory,
    sampled: SampledKey,
) -> bool {
    let Some(held) = slot.memory.guest_memory() else {
        return false;
    };
    held.import.id() == memory.import.id()
        && held.backing == memory.backing
        && slot.width == sampled.width
        && slot.height == sampled.height
        && slot.sample_count == 1
        && slot.mip_levels == sampled.mip_levels
        && slot.format.allocation() == translate::pixel::storage_format(sampled.format)
        && slot.content_ready
}

fn guest_resident_backing_key(slot: &ResidentTargetSlot) -> Option<GuestResidentBackingKey> {
    let memory = slot.memory.guest_memory()?;
    Some(GuestResidentBackingKey {
        import: memory.import.id(),
        backing: memory.backing,
        width: slot.width,
        height: slot.height,
        sample_count: slot.sample_count,
        mip_levels: slot.mip_levels,
        format: slot.format.allocation(),
    })
}

fn guest_sampled_resident_match(
    registry: &HashMap<TargetIdentity, ResidentTargetSlot>,
    authority: &HashMap<GuestResidentBackingKey, TargetIdentity>,
    memory: &reims_vgpu_memory::GuestTargetMemory,
    sampled: SampledKey,
    excludes: impl Fn(&TargetIdentity) -> bool,
) -> Result<Option<TargetIdentity>, usize> {
    let key = GuestResidentBackingKey {
        import: memory.import.id(),
        backing: memory.backing,
        width: sampled.width,
        height: sampled.height,
        sample_count: 1,
        mip_levels: sampled.mip_levels,
        format: translate::pixel::storage_format(sampled.format),
    };
    if let Some(identity) = authority.get(&key) {
        crate::telemetry::note_route("sampled_guest_uses_backing_authority");
        return Ok((!excludes(identity)
            && registry
                .get(identity)
                .is_some_and(|slot| guest_sampled_matches_resident(slot, memory, sampled)))
        .then(|| identity.clone()));
    }
    let mut matches = registry.iter().filter_map(|(identity, slot)| {
        (!excludes(identity) && guest_sampled_matches_resident(slot, memory, sampled))
            .then_some(identity.clone())
    });
    let first = matches.next();
    let Some(second) = matches.next() else {
        return Ok(first);
    };
    let count = 2 + matches.count();
    let _ = second;
    Err(count)
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
    memory: ResidentMemory,
    guest_backing: Option<reims_vgpu_memory::GuestTargetBacking>,
    /// Set only by a creation site that aliased already-populated guest pages,
    /// which is the one case where a resident is registered over bytes no
    /// Vulkan operation has yet written into its image.
    guest_materialization: Option<AliasMaterialization>,
    view: vk::ImageView,
    /// `vk::Framebuffer::null()` for a resident that is never bound as a
    /// standalone single-RT target — see
    /// [`ResidentTargetSlot::owed_framebuffer`], which is what every destroy
    /// path asks instead of testing this field itself.
    framebuffer: vk::Framebuffer,
    /// Null exactly when `framebuffer` is: the pass handle used to create the
    /// per-slot framebuffer. Compatibility, not handle identity, decides reuse.
    render_pass: vk::RenderPass,
    /// `None` exactly when `framebuffer` is null. Unlike the creation handle,
    /// this is the Vulkan compatibility identity: actions, layouts, and
    /// dependencies can change without replacing it between encoder records.
    framebuffer_compatibility: Option<FramebufferCompatibilityKey>,
    width: u32,
    height: u32,
    sample_count: u32,
    /// Levels the created image carries. One on every creation site but the
    /// guest-alias rail, where the guest's own declaration decides.
    mip_levels: u32,
    generation: u64,
    format: translate::pixel::ResidentFormat,
    /// The view in `format.declared()`, when the declaration needs one of its
    /// own. `None` says the declaration and the allocation are one format and
    /// `view` serves both, which is every format but the two sRGB spellings.
    attachment_view: Option<vk::ImageView>,
}

impl ResourcePools {
    /// Resolve a sampled declaration to a resident whose image *is* the guest's
    /// allocation, so the GPU samples the pages the guest CPU writes.
    ///
    /// Three outcomes, and only the shape of the declaration chooses between
    /// them:
    ///
    /// - a resident already aliases these exact bytes — the registry's own
    ///   backing authority answers, and the bind costs nothing;
    /// - nothing does, and the declaration is the exact allocation shape — one
    ///   image is created over the guest pages and registered, so every later
    ///   bind takes the first arm;
    /// - anything else — a mip chain, a subresource, an array, a volume, a
    ///   cube, a swizzled view — returns a named refusal to the imported
    ///   buffer-to-image copy rail, which is also the only rail on a host with
    ///   no host-pointer import at all.
    ///
    /// A newly created alias owes exactly one copy before anything reads it:
    /// an external image is born `UNDEFINED` and the first transition out of
    /// that layout may discard what the aliased memory holds, so the guest's
    /// existing bytes have to be written back into the image. That copy is
    /// described by [`AliasMaterialization`] and is the caller's to record; the
    /// resident keeps owing it until the caller says it went into a command
    /// buffer.
    pub(crate) unsafe fn acquire_guest_sampled(
        &mut self,
        ctx: &DeviceContext,
        source: &reims_vgpu_memory::GuestImageSource,
        resource: &super::super::types::SampledImageResource,
        counters: &EngineCounters,
        excludes: impl Fn(&TargetIdentity) -> bool,
    ) -> Result<GuestSampledUse, super::super::linear_image_import::WindowRefusal> {
        use super::super::linear_image_import::{SampledCopyCause, WindowRefusal};

        let memory = source
            .direct
            .as_ref()
            .ok_or(WindowRefusal::HostImportUnavailable)?;
        let sampled = SampledKey {
            mip_levels: source.view.mip_level_count,
            ..SampledKey::of(resource)
        };
        if !source.view.fits(&source.allocation) {
            return Err(WindowRefusal::UnsupportedImageShape {
                layers: sampled.layers,
                volume: sampled.volume,
                cube: sampled.cube,
                arrayed: sampled.arrayed,
                one_dim: sampled.one_dim,
            });
        }
        let layout = source.allocation.mips[source.view.base_mip_level as usize].layout;
        let allocation_is_array = layout.is_arrayed();
        let shape_matches = match layout {
            reims_vgpu_memory::GuestImageLayout::D1 { .. }
            | reims_vgpu_memory::GuestImageLayout::D1Array { .. } => {
                sampled.one_dim
                    && !sampled.volume
                    && if sampled.arrayed {
                        allocation_is_array
                    } else {
                        source.view.array_layer_count == 1
                    }
                    && sampled.layers == source.view.array_layer_count
            }
            reims_vgpu_memory::GuestImageLayout::D2 { .. }
            | reims_vgpu_memory::GuestImageLayout::D2Array { .. } => {
                !sampled.one_dim
                    && !sampled.volume
                    && if sampled.arrayed {
                        allocation_is_array
                    } else {
                        source.view.array_layer_count == 1
                    }
                    && sampled.layers == source.view.array_layer_count
            }
            reims_vgpu_memory::GuestImageLayout::D3 { depth, .. } => {
                !sampled.one_dim && !sampled.arrayed && sampled.volume && sampled.layers == depth
            }
        };
        if sampled.cube || !shape_matches {
            return Err(WindowRefusal::UnsupportedImageShape {
                layers: sampled.layers,
                volume: sampled.volume,
                cube: sampled.cube,
                arrayed: sampled.arrayed,
                one_dim: sampled.one_dim,
            });
        }
        // Only the target registry's exact allocation shape may converge here.
        // A subresource, array, mip-chain, or volume declaration is a different
        // view contract and stays on the general guest-image rail. A component
        // swizzle is not one of them: it rides on the view this rail creates.
        // Current attachments are excluded by the caller because sampling an
        // image while writing it requires the snapshot/feedback machinery that
        // is selected from a typed Target source, not an opportunistic match.
        //
        // The rule itself belongs to `SampledCopyCause` and is asked here rather
        // than restated, so this arm and the length arm in `linear_image_import`
        // cannot drift apart, and so the fall-through below can say which term
        // refused instead of only that a copy is required.
        let Some(allocation_mips) = source.allocation.mip_level_count() else {
            return Err(WindowRefusal::ResourceWindowTooShort);
        };
        let requires_copy = SampledCopyCause::of_allocation(layout).or_else(|| {
            SampledCopyCause::of_view(
                source.view.base_mip_level,
                source.view.mip_level_count,
                allocation_mips,
                source.view.base_array_layer,
                source.view.array_layer_count,
            )
        });
        if requires_copy.is_none() {
            match guest_sampled_resident_match(
                &self.registry,
                &self.guest_resident_authority,
                memory,
                sampled,
                excludes,
            ) {
                Ok(Some(identity)) => {
                    if let Some(use_) = unsafe {
                        self.bind_guest_sampled_resident(ctx, &identity, sampled, counters)
                    }? {
                        crate::telemetry::note_route("sampled_guest_reuses_target_resident");
                        return Ok(use_);
                    }
                }
                Ok(None) => {
                    let identity = unsafe {
                        self.create_guest_sampled_alias(ctx, memory, source, sampled, counters)
                    }?;
                    if let Some(use_) = unsafe {
                        self.bind_guest_sampled_resident(ctx, &identity, sampled, counters)
                    }? {
                        crate::telemetry::note_route("sampled_guest_alias_created");
                        return Ok(use_);
                    }
                }
                Err(matches) => {
                    return Err(WindowRefusal::AmbiguousResidentBacking { matches });
                }
            }
        }
        // Everything the aliasing rail above does not admit. The imported
        // buffer stays zero-copy for guest RAM and feeds an ordinary sampled
        // image with one GPU-side copy, which is also the only rail a host
        // without `VK_EXT_external_memory_host` has.
        Err(WindowRefusal::SampledContentRequiresCopy(
            // Either the declaration was inadmissible above, or it was
            // admissible and the registry declined to pin a resident for it.
            // The second is not a shape refusal, and reporting it as the shape
            // this bind happens to have would be a false cause.
            requires_copy.unwrap_or(SampledCopyCause::ResidentUnavailable),
        ))
    }

    /// Take a sampled bind on a resident that already aliases the guest bytes.
    ///
    /// `Ok(None)` means the pin was refused, which is the registry saying this
    /// resident is not one a draw may hold — the caller falls through to the
    /// copy rail rather than binding it anyway.
    unsafe fn bind_guest_sampled_resident(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        sampled: SampledKey,
        counters: &EngineCounters,
    ) -> Result<Option<GuestSampledUse>, super::super::linear_image_import::WindowRefusal> {
        use super::super::linear_image_import::WindowRefusal;

        let (image, slot_access, levels, materialize) = self
            .registry
            .get(identity)
            .map(|slot| {
                (
                    slot.image,
                    slot.access,
                    slot.mip_levels,
                    slot.guest_materialization.clone(),
                )
            })
            .expect("matched resident remains registered");
        let view = unsafe {
            self.registry_sample_view_raw(ctx, identity, sampled.format, sampled.swizzle, counters)
        }
        .map_err(WindowRefusal::CreateView)?
        .expect("matched resident remains registered");
        if !self.pin_resident_for_entry(identity) {
            return Ok(None);
        }
        self.registry_note_sampled_use(identity);
        // An owed materialization ends in the layout a sampled read wants, so
        // the access reported here is the one the caller's own recording will
        // leave behind. Reporting the image's current `Untouched` state instead
        // would make the caller place a second, discarding transition over the
        // copy it is about to record.
        let access = if materialize.is_some() {
            ResidentAccess::shader_read()
        } else {
            slot_access
        };
        Ok(Some(GuestSampledUse::Resident {
            identity: identity.clone(),
            image,
            view,
            access,
            levels,
            materialize,
        }))
    }

    /// Create one Vulkan image over the guest's own texel bytes and register it.
    ///
    /// The image is the guest allocation: no copy happens here, and none
    /// happens on any later bind. What the registration owes is the single
    /// birth copy described by [`AliasMaterialization`], recorded on the
    /// resident so that an abandoned draw leaves it owed rather than lost.
    ///
    /// Lifetime is the registry's, exactly as for an imported colour target: a
    /// guest-backed resident is not sole-copy, so ordinary allocation pressure
    /// may reclaim it and the next bind rebuilds it. Nothing here holds a
    /// second table keyed by anything else.
    unsafe fn create_guest_sampled_alias(
        &mut self,
        ctx: &DeviceContext,
        memory: &reims_vgpu_memory::GuestTargetMemory,
        source: &reims_vgpu_memory::GuestImageSource,
        sampled: SampledKey,
        counters: &EngineCounters,
    ) -> Result<TargetIdentity, super::super::linear_image_import::WindowRefusal> {
        use super::super::linear_image_import::WindowRefusal;

        let format =
            translate::pixel::ResidentFormat::of(translate::pixel::storage_format(sampled.format));
        let bytes_per_texel = translate::pixel::texel_layout_of(format.allocation())
            .map(|layout| u64::from(layout.bytes_per_texel()))
            .ok_or(WindowRefusal::ResourceWindowTooShort)?;
        // The image is the guest's whole allocation, so the chain it carries is
        // the one the guest declared — not the range this bind's view named,
        // which admission has already required to be all of it.
        let mip_levels = source
            .allocation
            .mip_level_count()
            .ok_or(WindowRefusal::ResourceWindowTooShort)?;
        // The window every level of the chain occupies, and each level's copy
        // into it. One launder covers the whole window, so it is both the
        // staging span and the origin the per-level offsets count from.
        let (window, levels) =
            AliasMipCopy::chain(&source.allocation, memory.backing, bytes_per_texel)?;
        // The exact guest pages a host write into this image may land on. Page
        // ownership is asked before any Vulkan object exists, so a declaration
        // whose window this device may not write to never becomes an image, and
        // it is asked of the whole chain because the birth copy writes every
        // level of it.
        let write_pages = memory
            .chain_write_pages(&source.allocation, bytes_per_texel)
            .ok_or(WindowRefusal::ResourceWindowTooShort)?;

        // The guest writes these pages with its own CPU and this device does
        // not own the mapping, so the only thing that can make those writes
        // visible to the device is the memory type carrying HOST_COHERENT.
        // Without it there is no flush anyone here could issue.
        let allocation = unsafe { self.host_ram_imports.allocation(ctx, &memory.import) }
            .map_err(WindowRefusal::ParentImport)?;
        if !ctx
            .mapped_memory_kind(allocation.memory_type_index)
            .coherent
        {
            return Err(WindowRefusal::SampledAliasHostWritesNotCoherent {
                memory_type_index: allocation.memory_type_index,
            });
        }

        let imported = unsafe {
            super::super::linear_image_import::create_allocation(
                ctx,
                &mut self.host_ram_imports,
                &memory.import,
                memory.backing,
                &source.allocation,
                format.allocation(),
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )
        }?;
        counters.note_create(CreateSite::RegistryImage);
        let resident_memory = ResidentMemory::GuestImported {
            memory: memory.clone(),
            write_pages,
        };
        let view = match unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(imported.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.allocation())
                    // The whole chain: a `TYPE_2D` view chooses the
                    // dimensionality and `subresourceRange` chooses the levels,
                    // so naming only level zero here would leave a mipmapped
                    // texture sampling nothing above it.
                    .subresource_range(vk::ImageSubresourceRange {
                        level_count: mip_levels,
                        ..color_subresource_range()
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(result) => {
                unsafe {
                    self.abandon_new_resident_image(ctx, imported.image, &resident_memory);
                }
                return Err(WindowRefusal::CreateView(result));
            }
        };
        counters.note_create(CreateSite::RegistryImageView);

        let identity = TargetIdentity::Anonymous {
            slot: Self::allocate_alias_slot(),
        };
        self.register_resident(
            &identity,
            NewResident {
                image: imported.image,
                memory: resident_memory,
                guest_backing: Some(memory.backing),
                guest_materialization: Some(AliasMaterialization {
                    source: allocation.buffer,
                    source_offset: window.start,
                    bytes: window.end - window.start,
                    levels: std::sync::Arc::from(levels),
                }),
                view,
                // Never a standalone colour target: this resident is reached
                // through the sampled rail and has no pass of its own.
                framebuffer: vk::Framebuffer::null(),
                render_pass: vk::RenderPass::null(),
                framebuffer_compatibility: None,
                width: sampled.width,
                height: sampled.height,
                sample_count: 1,
                mip_levels,
                generation: 0,
                format,
                attachment_view: None,
            },
        );
        // The guest's bytes are this image's contents from the moment it exists
        // — that is what aliasing means — so the slot is ready and may be
        // pinned. `Untouched` is the truth about the *image*: no Vulkan
        // operation has placed a layout on it, which is what the owed
        // materialization exists to fix. Publishing also enters the backing
        // authority, so every later bind of these bytes finds this one image.
        self.registry_mark_ready_with_access(&identity, ResidentAccess::Untouched);
        Ok(identity)
    }

    /// A fresh anonymous slot number for one aliasing resident.
    ///
    /// The guest never names this storage — it names the allocation, and the
    /// registry's backing authority is what turns an allocation back into this
    /// identity. So the identity itself only has to be unique and never reused,
    /// which is what a monotonic counter is.
    fn allocate_alias_slot() -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// How many mip levels the resident registered under `identity` carries,
    /// or `None` if nothing is registered there.
    ///
    /// Asked by a caller placing a barrier over the image, which has to name
    /// every level the image has. The slot is the authority because it is where
    /// the creating site recorded what it built.
    pub(crate) fn resident_mip_levels(&self, identity: &TargetIdentity) -> Option<u32> {
        self.registry.get(identity).map(|slot| slot.mip_levels)
    }

    /// Report that the copy [`GuestSampledUse::Resident`] handed back is in a
    /// command buffer, and that it left the image in `access`.
    ///
    /// Called only once the recording exists. Until then the resident keeps
    /// owing the copy, so a draw abandoned part-way through leaves the next
    /// bind to record it again rather than sampling an image whose layout was
    /// placed but whose texels were never written.
    pub(crate) fn registry_note_materialized(
        &mut self,
        identity: &TargetIdentity,
        access: ResidentAccess,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.guest_materialization = None;
            slot.access = access;
        }
    }

    /// Whether this resident's image aliases imported guest memory, so a draw
    /// binding it reads bytes another Vulkan alias — or the guest's own CPU —
    /// may have written.
    pub(crate) fn resident_reads_imported_guest(&self, identity: &TargetIdentity) -> bool {
        self.registry
            .get(identity)
            .is_some_and(|slot| slot.memory.guest_memory().is_some())
    }

    /// Apply the one exact-layout admission rule used by every colour
    /// attachment. Slot number and framebuffer ownership are intentionally not
    /// inputs: they do not change whether a Vulkan image aliases these bytes.
    #[allow(
        clippy::too_many_arguments,
        reason = "the admission equation needs the declared image shape and Vulkan usage"
    )]
    unsafe fn try_import_guest_target(
        &mut self,
        ctx: &DeviceContext,
        target: Option<&reims_vgpu_memory::GuestTargetMemory>,
        width: u32,
        height: u32,
        sample_count: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Option<(
        vk::Image,
        reims_vgpu_memory::GuestTargetMemory,
        reims_vgpu_memory::GuestWritePages,
    )> {
        if sample_count != 1 {
            return None;
        }
        // Narrowing switch, asked before the host is. A target withheld here
        // takes the device-local allocation every host without the import
        // takes, and reaches guest pages through `render_writeback` instead of
        // being written in place.
        if matches!(
            reims_vgpu_config::switch(reims_vgpu_config::TARGET_IMPORT),
            reims_vgpu_config::Switch::Off
        ) {
            crate::telemetry::note_route("target_guest_import_disabled_by_env");
            return None;
        }
        let target = target?;
        let bytes_per_texel = crate::translate::pixel::texel_layout_of(format)
            .map(|layout| u64::from(layout.bytes_per_texel()))?;
        let write_pages = target.visible_write_pages(width, height, bytes_per_texel)?;
        match unsafe {
            super::super::linear_image_import::create(
                ctx,
                &mut self.host_ram_imports,
                &target.import,
                target.backing,
                reims_vgpu_memory::GuestImageLayout::D2 { width, height },
                format,
                usage,
            )
        } {
            Ok(imported) => {
                crate::telemetry::note_route("target_guest_imported");
                Some((imported.image, target.clone(), write_pages))
            }
            Err(decline) => {
                reims_vgpu_observe::Emit::decline("target_guest_import_declined", &decline).off();
                None
            }
        }
    }

    unsafe fn abandon_new_resident_image(
        &mut self,
        ctx: &DeviceContext,
        image: vk::Image,
        memory: &ResidentMemory,
    ) {
        ctx.device.destroy_image(image, None);
        match memory {
            ResidentMemory::Recyclable(_) => self.free_image_slab(&ctx.device, image),
            ResidentMemory::GuestImported { memory, .. } => {
                if let Some(parent) = self.host_ram_imports.release_child(&memory.import) {
                    let completed = parent.destroy(&ctx.device);
                    self.completed_guest_imports.push(completed);
                }
            }
        }
    }

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
        let format = crate::format::vk_storage_image(key.format);
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
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC
                    })
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStorageImage, e)))?;
        counters.note_create(CreateSite::StorageImage);
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
        counters.note_create(CreateSite::StorageImageView);
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
        // skips a pinned resident, as allocation-pressure recovery does,
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
    /// pin check is unit-testable without a device.
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
    /// alone would have left this population with nothing to hand back when an
    /// allocation failed.
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
        reims_vgpu_observe::fail(format!(
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

    /// Record that something read this resident for reuse-distance diagnostics.
    ///
    /// Reading a resident is using it. The three read-only accessors below all
    /// mean "a guest chain is about to consume this image" — the stage-time
    /// guest-read skip, the resident-sample gate, and the flush/sample snapshot —
    /// so a produce-once/sample-many resident that is never dispatched into
    /// again remains observable even when it is never dispatched into. No
    /// reclaim decision reads this timestamp.
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
            resident.access = ResidentAccess::transfer_read();
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

    /// Record that a compute-storage resident's pixels have reached a durable
    /// replica outside the image, so the reclaim paths may take it.
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

    /// Pin/unpin a resident against removal while its content exists nowhere but
    /// the GPU image. Answers whether a slot was there to pin.
    ///
    /// The bool exists for the same reason [`Self::pin_resident_target`]'s does:
    /// the guest-write ledger records the pins it actually took, so a release can
    /// never be handed out for a pin that was never taken. An absent identity is
    /// not an error — there is then no image for anything to remove.
    ///
    /// # What this holds that `gpu_only_content` does not
    ///
    /// Both reclaim paths already skip a sole-copy resident, so a dispatch's own
    /// output is safe from *allocation-pressure recovery* without any pin. The
    /// window this closes is the other removal: a re-key.
    /// [`Self::acquire_resident_storage_image`] destroys the held image when the
    /// same identity arrives at a different shape, and
    /// [`Self::compute_rekey_refusal`] is what stops it — and that refusal reads
    /// `pinned`, nothing else. A compute writeback copying straight into guest
    /// pages is submitted and not waited, so between the submit and the fence a
    /// re-shaped dispatch would hand the queue a destroyed image.
    pub(crate) fn pin_resident_storage(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        pinned: bool,
    ) -> bool {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.pinned = pinned;
            return true;
        }
        false
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
    /// resident-sample skip (the format must match what the sampled view will
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

    /// Snapshot of a resident storage image for a sampled bind. Creation
    /// declared both storage and sampled use, so the view names this same image
    /// rather than a copied surrogate.
    pub(crate) fn compute_resident_snapshot(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(
        vk::Image,
        vk::ImageView,
        StorageImageKey,
        u32,
        ResidentAccess,
    )> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry.get(identity).map(|resident| {
            (
                resident.slot.image,
                resident.slot.view,
                resident.slot.key,
                resident.generation,
                resident.access,
            )
        })
    }

    /// Record the resting state of a resident that this dispatch only sampled.
    pub(crate) fn mark_compute_resident_sampled(&mut self, identity: &ComputeStorageResidencyKey) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.access = ResidentAccess::ShaderRead(vk::ImageLayout::GENERAL);
        }
    }

    // --- Target registry (workstream D) ------------------------------------

    pub(crate) fn registry_get(&self, identity: &TargetIdentity) -> Option<&ResidentTargetSlot> {
        self.registry.get(identity)
    }

    /// The generation the registry holds for the *same target* as `identity`,
    /// when the exact key is absent.
    ///
    /// This separates the two things an absent registry entry can mean, which
    /// otherwise arrive as one word. `None` says nothing in the registry names
    /// this surface at all — the guest never rendered into it, or its resident
    /// was reclaimed. `Some(g)` says the target is there and the caller asked
    /// under a different generation, which is a **key** fault rather than a
    /// missing target and has an entirely different repair.
    ///
    /// A linear scan, on a refusal path only: the registry's own
    /// `registry_pressure` census reads a peak of about thirty slots.
    ///
    /// It also names *how* the closest key differs, because "the generation
    /// moved" is only one of four ways it can and a boot reading `held=none`
    /// cannot tell "nothing names this object" from "this object is registered
    /// under a different extent or namespace". See
    /// [`crate::engine::TargetKeyDivergence`].
    pub(crate) fn registry_key_divergence(
        &self,
        identity: &TargetIdentity,
    ) -> (TargetKeyDivergence, Option<u64>) {
        // The finest difference is the closest key, so the ladder is ranked and
        // minimised over rather than short-circuited: a registry holding both a
        // resized entry and a re-generated one should report the re-generated
        // one, whichever the hash map yields first.
        let rank = |d: TargetKeyDivergence| match d {
            TargetKeyDivergence::Generation => 0,
            TargetKeyDivergence::Other => 1,
            TargetKeyDivergence::Geometry => 2,
            TargetKeyDivergence::Namespace => 3,
            TargetKeyDivergence::Absent => 4,
        };
        self.registry
            .keys()
            .map(|held| (identity.diverges_from(held), Some(held.generation())))
            // A key in another namespace is not about this object at all, so it
            // is not a near miss — reporting one would claim the target exists.
            .filter(|(d, _)| *d != TargetKeyDivergence::Namespace)
            .min_by_key(|(d, _)| rank(*d))
            .unwrap_or((TargetKeyDivergence::Absent, None))
    }

    /// Return the sampled view for a resident image, at the format the guest's
    /// declaration reaches rather than at the one the bind could spell.
    ///
    /// The image allocation is shared by every compatible texture view. Views
    /// are retained with that allocation because their lifetime is the resource
    /// lifetime, not one draw, and the finite translated format vocabulary is
    /// the natural bound on this collection.
    ///
    /// **The transfer function comes from the resident and the channel order
    /// from the bind**, which is [`translate::pixel::sample_view_format`]'s whole
    /// job — read its doc before changing what is passed in. A sampled bind
    /// names its format through a `TexelLayout`, which describes stored bytes and
    /// cannot carry a transfer function, so asking it alone drops the sRGB
    /// qualifier off every resident that has one. This is the single place that
    /// decides which view a bind gets, so it is the single place the fold
    /// belongs; a caller doing it for itself would be the second spelling.
    /// **The channel mapping comes from the bind too**, and for the same reason
    /// the channel order does: a Metal texture view carries a swizzle, and
    /// Vulkan applies a component mapping at sample time, so the two express one
    /// concept and the mapping belongs in the view rather than in the bytes.
    /// Baking it here is what keeps a swizzled sampled declaration on the
    /// aliasing rail — rewriting texels to move a single channel into place
    /// would force the whole resource onto the CPU upload path and cost it the
    /// zero-copy property, which is the trade
    /// [`translate::pixel::vk_component_mapping`] exists to avoid.
    pub(crate) unsafe fn registry_sample_view(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        format: vk::Format,
        swizzle: SwizzlePlan,
        counters: &EngineCounters,
    ) -> Result<Option<vk::ImageView>, DrawError> {
        unsafe { self.registry_sample_view_raw(ctx, identity, format, swizzle, counters) }
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryView, error)))
    }

    unsafe fn registry_sample_view_raw(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        format: vk::Format,
        swizzle: SwizzlePlan,
        counters: &EngineCounters,
    ) -> Result<Option<vk::ImageView>, vk::Result> {
        let Some(slot) = self.registry.get(identity) else {
            return Ok(None);
        };
        let format = translate::pixel::sample_view_format(format, slot.format.declared());
        unsafe {
            self.registry_view_raw(ctx, identity, ResidentViewKey { format, swizzle }, counters)
        }
    }

    /// The view over this resident's allocation in exactly `format`, created and
    /// retained on first ask.
    ///
    /// One home for every interpretation of one image, which is what lets
    /// `slot.view` be the *allocation*-format view and nothing else. That
    /// choice is load-bearing rather than arbitrary: the recycle pool is keyed
    /// on the allocation format, so the view that travels with a retired image
    /// must be the one in that format, and a slot whose stored view was the
    /// guest's declaration instead would put an sRGB view into a bucket only
    /// UNORM requests ever take from. Every other interpretation — the render
    /// pass's attachment view included — is reached through here.
    ///
    /// A caller asking for the format the slot was allocated in gets
    /// `slot.view` and no allocation happens, which is every format this device
    /// renders to except the two sRGB spellings.
    unsafe fn registry_view(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        format: vk::Format,
        counters: &EngineCounters,
    ) -> Result<Option<vk::ImageView>, DrawError> {
        unsafe { self.registry_view_raw(ctx, identity, ResidentViewKey::plain(format), counters) }
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryView, error)))
    }

    unsafe fn registry_view_raw(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        key: ResidentViewKey,
        counters: &EngineCounters,
    ) -> Result<Option<vk::ImageView>, vk::Result> {
        let Some(slot) = self.registry.get_mut(identity) else {
            return Ok(None);
        };
        if let Some(view) = slot.held_view(key) {
            return Ok(Some(view));
        }
        let view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(slot.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(key.format)
                    .components(translate::pixel::vk_component_mapping(&key.swizzle))
                    .subresource_range(slot.subresource_range(key.format)),
                None,
            )
        }?;
        counters.note_create(CreateSite::RegistryImageView);
        slot.alternate_views.push((key, view));
        Ok(Some(view))
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
    /// Every path that removes a live entry comes through here. It is the death
    /// counterpart of [`Self::register_resident`], and the pair is why
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
        if let Some(key) = guest_resident_backing_key(&old) {
            if self.guest_resident_authority.get(&key) == Some(identity) {
                self.guest_resident_authority.remove(&key);
            }
        }
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
    ///   `registry_mark_ready_at` and the IOSurface texture LOAD gate read them, and an arm
    ///   that guessed differently would be
    ///   answering a question the others think they already asked.
    /// - **The diagnostic clock belongs to the registry.** `last_touch_ms` comes
    ///   from `idle_clock_ms`, which the poll heartbeat advances.
    /// - **`registry` and `registry_order` are written together.** They are one
    ///   structure split for lookup and for order. An entry in the map but not
    ///   the order is a resident no sweep can ever choose; one in the order but
    ///   not the map is a victim that frees nothing.
    fn register_resident(&mut self, identity: &TargetIdentity, new: NewResident) {
        let last_touch_ms = self.idle_clock_ms;
        self.registry.insert(
            identity.clone(),
            ResidentTargetSlot {
                incarnation: super::ResidentIncarnation::allocate(),
                image: new.image,
                memory: new.memory,
                guest_backing: new.guest_backing,
                guest_materialization: new.guest_materialization,
                view: new.view,
                // Seeded rather than left empty: the declaration's view is one
                // more interpretation of this allocation, and `alternate_views`
                // is where every interpretation but the allocation's own lives,
                // so putting it anywhere else would give `registry_view` a
                // second place to look and `retire_resident` a view it does not
                // dispose.
                alternate_views: new
                    .attachment_view
                    .map(|v| vec![(ResidentViewKey::plain(new.format.declared()), v)])
                    .unwrap_or_default(),
                framebuffer: new.framebuffer,
                render_pass: new.render_pass,
                framebuffer_compatibility: new.framebuffer_compatibility,
                width: new.width,
                height: new.height,
                sample_count: new.sample_count,
                mip_levels: new.mip_levels,
                generation: new.generation,
                content_ready: false,
                content_epoch: None,
                access: ResidentAccess::Untouched,
                format: new.format,
                pin_count: 0,
                resource_released: false,
                resource_owner_count: 0,
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
    /// Resource release also comes through here but is not counted as an
    /// eviction: the guest ended that lifetime. Bookkeeping and the framebuffer
    /// null question remain centralized so removal paths cannot diverge.
    unsafe fn retire_resident(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        why: ResidentReclaim,
        counters: &EngineCounters,
    ) -> Option<ResidentTargetSlot> {
        let old = self.unregister_resident(identity, why)?;
        self.dispose_owed_framebuffer(&ctx.device, old.owed_framebuffer());
        for (_, view) in &old.alternate_views {
            self.dispose(&ctx.device, DeferredHandle::ImageView(*view));
        }
        let retired = match &old.memory {
            ResidentMemory::Recyclable(memory) => DeferredHandle::RecycleTarget(FreeTargetImage {
                image: old.image,
                memory: *memory,
                view: old.view,
                width: old.width,
                height: old.height,
                sample_count: old.sample_count,
                format: old.format.allocation(),
            }),
            ResidentMemory::GuestImported { memory, .. } => DeferredHandle::GuestImage {
                image: old.image,
                view: old.view,
                import: std::sync::Arc::clone(&memory.import),
            },
        };
        self.dispose(&ctx.device, retired);
        if why != ResidentReclaim::ResourceReleased {
            counters
                .target_evicts
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Some(old)
    }

    /// Ensure a resident target exists for `identity` with the given geometry + pass.
    /// Image, memory, and framebuffer persist across compatible render-pass
    /// changes. In particular, LOAD versus CLEAR does not rebuild a framebuffer;
    /// formats and subpass attachment shape still partition it.
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
        sample_count: u32,
        render_pass: vk::RenderPass,
        framebuffer_compatibility: FramebufferCompatibilityKey,
        generation: u64,
        format: vk::Format,
        guest_target: Option<&reims_vgpu_memory::GuestTargetMemory>,
        preserve_existing_content: bool,
        counters: &EngineCounters,
    ) -> Result<(&ResidentTargetSlot, vk::ImageView), DrawError> {
        // The format arrives resolved rather than as a channel-order flag, and
        // from the same variable that built `render_pass`'s key — an image and
        // the pass it is attached to must name one format, and deriving it
        // twice from a shared input is how they drift apart.
        //
        // It is the guest's *declaration*, so it carries the transfer function
        // and the allocation behind it does not. The view handed back beside the
        // slot is the attachment view, in that declaration; `slot.view` and the
        // recycle bucket stay in the allocation format. See
        // [`translate::pixel::ResidentFormat`].
        let format = translate::pixel::ResidentFormat::of(format);
        let guest_backing = guest_target.map(|target| target.backing);
        // Compatible geometry + gen + allocation: reuse image; rebuild the FB
        // only if Vulkan render-pass compatibility changed. A change of
        // allocation must recreate the image, not just the framebuffer — an RGBA
        // image under a BGRA pass is invalid.
        if let Some(slot) = self.registry.get(&identity) {
            let reusable = slot.reusable_for_materialization(
                width,
                height,
                sample_count,
                generation,
                format,
                super::RequestedMaterialization::new(guest_backing, preserve_existing_content),
            );
            if reusable {
                // Resolved before the fast path rather than inside the rebuild,
                // because a declaration that changed while the allocation did
                // not is exactly the case the fast path must *not* take: the
                // framebuffer it would hand back was built over the previous
                // interpretation's view.
                let attachment =
                    unsafe { self.registry_view(ctx, &identity, format.declared(), counters)? }
                        .expect("the slot reused on the line above is still registered");
                let slot = self.registry.get(&identity).unwrap();
                if slot.framebuffer_compatibility == Some(framebuffer_compatibility)
                    && slot.format == format
                {
                    counters
                        .gpu_load_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let touch = self.idle_clock_ms;
                    let slot = self.registry.get_mut(&identity).unwrap();
                    slot.last_touch_ms = touch;
                    return Ok((slot, attachment));
                }
                // Same image, new pass or new interpretation → recreate
                // framebuffer only.
                let view = attachment;
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
                counters.note_create(CreateSite::RegistryFramebuffer);
                self.dispose_owed_framebuffer(&ctx.device, old_fb);
                let slot = self.registry.get_mut(&identity).unwrap();
                slot.framebuffer = framebuffer;
                slot.render_pass = render_pass;
                slot.framebuffer_compatibility = Some(framebuffer_compatibility);
                // The interpretation this framebuffer was just built over. The
                // allocation is unchanged — `reusable_for` matched on it — so
                // this is the one field a second texture view of one surface
                // moves.
                slot.format = format;
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok((self.registry.get(&identity).unwrap(), attachment));
            }
            // Geometry/gen/allocation mismatch → destroy and recreate.
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
        let mut usage = super::super::registry_target_usage(
            format.allocation(),
            ctx.depth_format_transfer_dst(format.allocation()),
        );
        if ctx.features.attachment_feedback_loop_layout {
            usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        }
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating a fresh one — the usage set is identical across all
        // registry targets, so a recycled image of the same geometry/format is
        // bind-compatible. This is what collapses the per-frame realloc storm a
        // per-generation target (video) would otherwise pay: skips vkCreateImage
        // + vkAllocateMemory + bind + view (and their note_create/note_alloc).
        // The recycled contents are stale — the slot is inserted with
        // layout=UNDEFINED / content_ready=false, and a fresh framebuffer is
        // always built below (it binds this specific render_pass).
        let imported = unsafe {
            self.try_import_guest_target(
                ctx,
                guest_target,
                width,
                height,
                sample_count,
                format.allocation(),
                usage,
            )
        };
        let (image, memory, view) = if let Some((image, target, write_pages)) = imported {
            let view = match ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.allocation())
                    .subresource_range(color_subresource_range()),
                None,
            ) {
                Ok(view) => view,
                Err(result) => {
                    ctx.device.destroy_image(image, None);
                    if let Some(parent) = self.host_ram_imports.release_child(&target.import) {
                        let completed = parent.destroy(&ctx.device);
                        self.completed_guest_imports.push(completed);
                    }
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsCreateRegistryView,
                        result,
                    )));
                }
            };
            counters.note_create(CreateSite::RegistryImage);
            counters.note_create(CreateSite::RegistryImageView);
            (
                image,
                ResidentMemory::GuestImported {
                    memory: target,
                    write_pages,
                },
                view,
            )
        } else if let Some(free) =
            self.take_free_target(width, height, sample_count, format.allocation())
        {
            (
                free.image,
                ResidentMemory::Recyclable(free.memory),
                free.view,
            )
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format.allocation())
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(super::super::caches::vk_sample_count(sample_count))
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryTarget, e)))?;
            counters.note_create(CreateSite::RegistryImage);
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
                    .format(format.allocation())
                    .subresource_range(color_subresource_range()),
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    ctx.device.destroy_image(image, None);
                    self.free_image_slab(&ctx.device, image);
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsCreateRegistryView,
                        e,
                    )));
                }
            };
            counters.note_create(CreateSite::RegistryImageView);
            (image, ResidentMemory::Recyclable(memory), view)
        };
        // `view` is the allocation's; the render pass was built against the
        // declaration, and Vulkan requires the framebuffer attachment to name
        // that same format. They are one handle on every format that carries no
        // transfer function, which is every format but the two sRGB spellings.
        let attachment_view = if format.needs_own_view() {
            match ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.declared())
                    .subresource_range(color_subresource_range()),
                None,
            ) {
                Ok(v) => {
                    counters.note_create(CreateSite::RegistryImageView);
                    Some(v)
                }
                Err(e) => {
                    ctx.device.destroy_image_view(view, None);
                    self.abandon_new_resident_image(ctx, image, &memory);
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsCreateRegistryView,
                        e,
                    )));
                }
            }
        } else {
            None
        };
        let attachments = [attachment_view.unwrap_or(view)];
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
                if let Some(extra) = attachment_view {
                    ctx.device.destroy_image_view(extra, None);
                }
                self.abandon_new_resident_image(ctx, image, &memory);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateRegistryFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create(CreateSite::RegistryFramebuffer);
        self.register_resident(
            &identity,
            NewResident {
                image,
                memory,
                guest_backing,
                guest_materialization: None,
                view,
                framebuffer,
                render_pass,
                framebuffer_compatibility: Some(framebuffer_compatibility),
                width,
                height,
                sample_count,
                // A colour or depth attachment, whichever arm above built it:
                // both create the image with one level, and an imported one
                // imports a single `D2` layout.
                mip_levels: 1,
                generation,
                format,
                attachment_view,
            },
        );
        Ok((
            self.registry.get(&identity).unwrap(),
            attachment_view.unwrap_or(view),
        ))
    }

    /// Ensure a resident attachment of an arbitrary Vulkan format — an MRT
    /// secondary colour target, or a depth-stencil buffer.
    ///
    /// The primary single-RT [`Self::registry_ensure`] only speaks `bgra` and
    /// owns a framebuffer; this one builds none, because its residents are only
    /// ever attachment N of an ad-hoc framebuffer or sampled through the view,
    /// never a standalone single-RT target. Reuse requires an exact (geometry,
    /// generation, storage format) match. Returns (image, view).
    ///
    /// **The allocation is keyed on [`translate::pixel::storage_format`], not on
    /// the requested view format.** A guest texture view over a surface is a
    /// second interpretation of one allocation, so a surface bound once as
    /// `BGRA8Unorm` and once as `BGRA8Unorm_sRGB` must resolve to one image with
    /// two views. Keying the image on the view format instead makes the second
    /// spelling miss `reusable_for`, retire the resident and recreate it empty,
    /// so the two interpretations alternate frame to frame and each shows half
    /// the content. The requested format is served by
    /// [`Self::registry_view`], which keeps `slot.view` the view in
    /// `format.allocation()` and caches every other interpretation alongside the
    /// allocation.
    ///
    /// **Colour and depth share this body rather than having one each.** The two
    /// differ only in image usage and view aspect, and both of those are
    /// functions of the format — see
    /// [`crate::engine::registry_target_usage`]. A second copy
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
        sample_count: u32,
        generation: u64,
        format: vk::Format,
        guest_target: Option<&reims_vgpu_memory::GuestTargetMemory>,
        preserve_existing_content: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView), DrawError> {
        // The guest's declaration; `format.allocation()` is the family it
        // belongs to and is what the image, the reuse test and the recycle
        // bucket are keyed on.
        let format = translate::pixel::ResidentFormat::of(format);
        let guest_backing = guest_target.map(|target| target.backing);
        if let Some(slot) = self.registry.get(&identity) {
            let reusable = slot.reusable_for_materialization(
                width,
                height,
                sample_count,
                generation,
                format,
                super::RequestedMaterialization::new(guest_backing, preserve_existing_content),
            );
            if reusable {
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let image = slot.image;
                let view =
                    unsafe { self.registry_view(ctx, &identity, format.declared(), counters)? }
                        .expect("the slot reused on the line above is still registered");
                return Ok((image, view));
            }
            // Geometry / gen / allocation mismatch → destroy and recreate.
            self.retire_resident(ctx, &identity, ResidentReclaim::Recreated, counters);
        }
        // Census only, as in the primary `registry_ensure`.
        self.note_registry_reach();
        let usage = super::super::registry_target_usage(
            format.allocation(),
            ctx.depth_format_transfer_dst(format.allocation()),
        );
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating — same recycle discipline as the primary
        // `registry_ensure`. Usage is a function of the format
        // (`registry_target_usage`), so images cross-flow between these paths by
        // geometry+format within this device without a bucket ever mixing
        // usages. Skips the
        // create/alloc/bind/view + their note_create/note_alloc; recycled
        // contents are stale, so the slot below is inserted layout=UNDEFINED /
        // content_ready=false.
        let imported = if super::super::format_is_depth(format.allocation()) {
            None
        } else {
            unsafe {
                self.try_import_guest_target(
                    ctx,
                    guest_target,
                    width,
                    height,
                    sample_count,
                    format.allocation(),
                    usage,
                )
            }
        };
        let (image, memory, view) = if let Some((image, target, write_pages)) = imported {
            let view = ctx
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format.allocation())
                        .subresource_range(super::super::registry_subresource_range(
                            format.allocation(),
                        )),
                    None,
                )
                .map_err(|e| {
                    ctx.device.destroy_image(image, None);
                    if let Some(parent) = self.host_ram_imports.release_child(&target.import) {
                        let completed = parent.destroy(&ctx.device);
                        self.completed_guest_imports.push(completed);
                    }
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryView, e))
                })?;
            counters.note_create(CreateSite::MrtImage);
            counters.note_create(CreateSite::MrtImageView);
            (
                image,
                ResidentMemory::GuestImported {
                    memory: target,
                    write_pages,
                },
                view,
            )
        } else if let Some(free) =
            self.take_free_target(width, height, sample_count, format.allocation())
        {
            (
                free.image,
                ResidentMemory::Recyclable(free.memory),
                free.view,
            )
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format.allocation())
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(super::super::caches::vk_sample_count(sample_count))
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| {
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryTarget, e))
                })?;
            counters.note_create(CreateSite::MrtImage);
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
                if super::super::format_is_depth(format.allocation()) {
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
                        .format(format.allocation())
                        .subresource_range(super::super::registry_subresource_range(
                            format.allocation(),
                        )),
                    None,
                )
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryView, e))
                })?;
            counters.note_create(CreateSite::MrtImageView);
            (image, ResidentMemory::Recyclable(memory), view)
        };
        self.register_resident(
            &identity,
            NewResident {
                image,
                // Record the requested allocation even when Vulkan declined
                // the import. The materialization decision is stable for this
                // declaration and must not become a recreate/retry loop.
                guest_backing,
                guest_materialization: None,
                memory,
                view,
                // No per-slot framebuffer and so no pass it was built against:
                // this arm's residents are bound as attachment N of an ad-hoc
                // MRT framebuffer, or sampled through the view.
                framebuffer: vk::Framebuffer::null(),
                render_pass: vk::RenderPass::null(),
                framebuffer_compatibility: None,
                width,
                height,
                sample_count,
                // An MRT secondary attachment: created, recycled or imported,
                // all three arms above are one level.
                mip_levels: 1,
                generation,
                format,
                // Nothing here needs the declaration's view yet — this arm owns
                // no framebuffer — so it is created on the first ask below,
                // which is the same path a later sampled bind takes.
                attachment_view: None,
            },
        );
        let view = unsafe { self.registry_view(ctx, &identity, format.declared(), counters)? }
            .expect("the resident registered on the line above is still there");
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
    /// `reims-vgpu::runtime::draw::depth_chain_identity` for why that
    /// distinction is what lets the identity carry generation zero.
    // The arguments are the depth attachment's decoded geometry; a struct here
    // would only rename the same fields at every call site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn registry_ensure_depth(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        sample_count: u32,
        with_stencil: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView), DrawError> {
        let format = Self::depth_format(ctx, with_stencil);
        self.registry_ensure_attachment(
            ctx,
            identity,
            width,
            height,
            sample_count,
            0,
            format,
            None,
            false,
            counters,
        )
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
        sample_count: u32,
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
                    .samples(super::super::caches::vk_sample_count(sample_count))
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            | if ctx.depth_format_transfer_dst(format) {
                                vk::ImageUsageFlags::TRANSFER_DST
                            } else {
                                vk::ImageUsageFlags::empty()
                            },
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDepthImage, e)))?;
        counters.note_create(CreateSite::DepthImage);
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
        counters.note_create(CreateSite::DepthImageView);
        Ok((image, memory, view))
    }

    /// The framebuffer for an attachment shape the target slot does not cache,
    /// created once and handed back to every later draw naming the same render
    /// pass, views and extent.
    ///
    /// This is the whole repair for `passdiff_fb`. The previous spelling built one
    /// per draw, and because a framebuffer handle is part of [`super::PassEcho`],
    /// two consecutive draws of one serialized render encoder — same target, same
    /// depth resident, same extent — were read as wanting different render pass
    /// instances. Reuse is not an optimisation Vulkan merely tolerates: the two
    /// framebuffers agreed on every input `vkCreateFramebuffer` reads, so the
    /// second was a distinct handle for an identical object.
    ///
    /// Not bounded by a count, because the population is bounded by what it is
    /// keyed on — the views belong to registry residents and an entry dies with
    /// the first of its views. A count bound could only evict entries whose views
    /// are still live, which is the one case reuse exists for.
    pub(crate) unsafe fn ensure_ad_hoc_framebuffer(
        &mut self,
        ctx: &DeviceContext,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        width: u32,
        height: u32,
        counters: &EngineCounters,
    ) -> Result<vk::Framebuffer, DrawError> {
        use ash::vk::Handle;
        let key = super::AdHocFramebufferKey {
            render_pass: render_pass.as_raw(),
            views: views.iter().map(|v| v.as_raw()).collect(),
            width,
            height,
        };
        if let Some(fb) = self.ad_hoc_framebuffers.get(&key) {
            crate::telemetry::note_route("adhoc_fb_hit");
            return Ok(*fb);
        }
        let fb = unsafe {
            self.create_mrt_framebuffer(ctx, render_pass, views, width, height, counters)
        }?;
        crate::telemetry::note_route("adhoc_fb_miss");
        self.ad_hoc_framebuffers.insert(key, fb);
        Ok(fb)
    }

    /// Destroy every cached ad-hoc framebuffer naming `view`, because the view is
    /// about to be destroyed and a framebuffer may not outlive its attachments.
    ///
    /// Called from the terminal destroy rather than from `dispose`, so a
    /// framebuffer goes at the same moment its view does and under the same
    /// already-established guarantee that no command buffer still names either.
    pub(crate) unsafe fn purge_ad_hoc_framebuffers_for_view(
        &mut self,
        device: &ash::Device,
        view: vk::ImageView,
    ) {
        use ash::vk::Handle;
        let raw = view.as_raw();
        let doomed: Vec<super::AdHocFramebufferKey> = self
            .ad_hoc_framebuffers
            .keys()
            .filter(|k| k.views.contains(&raw))
            .cloned()
            .collect();
        for key in doomed {
            if let Some(fb) = self.ad_hoc_framebuffers.remove(&key) {
                unsafe { device.destroy_framebuffer(fb, None) };
                crate::telemetry::note_route("adhoc_fb_purged");
            }
        }
    }

    /// Build an ad-hoc MRT framebuffer over `views` (primary slot 0 + secondary
    /// slots 1..) under `render_pass`. The caching entry point is
    /// [`Self::ensure_ad_hoc_framebuffer`]; this is its miss arm.
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
        counters.note_create(CreateSite::MrtFramebuffer);
        Ok(fb)
    }

    #[cfg(test)]
    pub(crate) fn registry_mark_ready(&mut self, identity: &TargetIdentity) {
        self.registry_mark_ready_at(identity, crate::engine::caches::color0_pass_exit_layout());
    }

    /// Mark a resident ready after a render pass wrote it as a colour
    /// attachment and left it at an explicit `final_layout` — the MRT secondary
    /// arm, which settles at `COLOR_ATTACHMENT_OPTIMAL` where
    /// The layout is the pass's exact `finalLayout`, including `GENERAL` for a
    /// host-accessible primary. Recording it beside the access is what keeps a
    /// later barrier from naming an optimized layout the imported image is not
    /// in.
    pub(crate) fn registry_mark_ready_at(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        self.registry_mark_ready_with_access(identity, ResidentAccess::ColorWrite(layout));
    }

    /// Publish newly rendered colour contents together with the exact access
    /// contract the pass left behind. Feedback-loop passes are both shader
    /// reads and colour writes, so reducing them to a plain colour write would
    /// lose half of the next barrier's source scope.
    pub(crate) fn registry_mark_ready_with_access(
        &mut self,
        identity: &TargetIdentity,
        access: ResidentAccess,
    ) {
        let mut guest_backed = false;
        let mut authority = None;
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            slot.access = access;
            guest_backed = slot.memory.guest_memory().is_some();
            authority = guest_resident_backing_key(slot);
        }
        if let Some(key) = authority {
            self.guest_resident_authority.insert(key, identity.clone());
        }
        self.set_sole_copy(identity, !guest_backed);
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
            slot.access =
                ResidentAccess::ColorWrite(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
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
        if pinned && (!slot.content_ready || slot.resource_released) {
            return false;
        }
        if !pinned && slot.pin_count == 0 {
            reims_vgpu_observe::fail(format!(
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

    /// Retain a live resource's engine-owned resident.
    pub(crate) fn retain_resident_target(
        &mut self,
        identity: &TargetIdentity,
    ) -> Option<super::ResidentIncarnation> {
        let incarnation = self
            .registry
            .get(identity)
            .filter(|slot| slot.content_ready)
            .map(|slot| slot.incarnation)?;
        // A new serialized owner may legitimately arrive while a transient GPU
        // holder is finishing the previous owner's use of the same allocation.
        // Revive the ownership before pinning; maintenance cannot retire the
        // slot while this engine transaction holds the registry lock.
        self.registry.get_mut(identity)?.resource_released = false;
        if !self.pin_resident_target(identity, true) {
            return None;
        }
        let slot = self.registry.get_mut(identity)?;
        let Some(next) = slot.resource_owner_count.checked_add(1) else {
            reims_vgpu_observe::fail(format!(
                "resident_resource_owner_overflow identity={identity:?}"
            ));
            // Undo the pin acquired above; the caller receives no lease.
            self.pin_resident_target(identity, false);
            return None;
        };
        slot.resource_owner_count = next;
        Some(incarnation)
    }

    /// End one serialized resource's ownership of its resident. The ownership
    /// pin is released exactly once. If an in-flight holder still has the target
    /// pinned, retirement waits for maintenance after that holder finishes; no
    /// new holder may retain the released resource.
    pub(crate) unsafe fn release_resident_resource(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        incarnation: super::ResidentIncarnation,
        counters: &EngineCounters,
    ) -> bool {
        let Some(unpinned) = self.release_resident_ownership_for(identity, incarnation) else {
            return false;
        };
        if unpinned {
            self.retire_resident(ctx, identity, ResidentReclaim::ResourceReleased, counters);
        }
        true
    }

    /// Release ownership only from the concrete slot the lease retained.
    ///
    /// A missing or replaced slot has already left the registry. Its deferred
    /// handles own its destruction; the old lease has no authority over the
    /// replacement under the same protocol identity.
    pub(crate) fn release_resident_ownership_for(
        &mut self,
        identity: &TargetIdentity,
        incarnation: super::ResidentIncarnation,
    ) -> Option<bool> {
        if self.registry.get(identity)?.incarnation != incarnation {
            crate::telemetry::note_route("resident_resource_release_stale_incarnation");
            return None;
        }
        self.release_resident_ownership(identity)
    }

    /// Device-free ownership transition behind resource release. Returns
    /// whether the ownership pin was the last pin and the resident may retire
    /// immediately, or `None` when the identity was already absent.
    fn release_resident_ownership(&mut self, identity: &TargetIdentity) -> Option<bool> {
        let slot = self.registry.get_mut(identity)?;
        if slot.resource_owner_count == 0 {
            reims_vgpu_observe::fail(format!(
                "resident_resource_release_unbalanced identity={identity:?}"
            ));
            return Some(false);
        }
        slot.resource_owner_count -= 1;
        slot.resource_released = slot.resource_owner_count == 0;
        self.pin_resident_target(identity, false);
        Some(
            self.registry
                .get(identity)
                .is_some_and(ResidentTargetSlot::released_and_collectable),
        )
    }

    fn released_resident_keys(&self, max: usize) -> Vec<TargetIdentity> {
        self.registry_order
            .iter()
            .filter(|identity| {
                self.registry
                    .get(*identity)
                    .is_some_and(ResidentTargetSlot::released_and_collectable)
            })
            .take(max)
            .cloned()
            .collect()
    }

    pub(super) unsafe fn retire_released_residents(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        max: usize,
    ) -> usize {
        let victims = self.released_resident_keys(max);
        for identity in &victims {
            self.retire_resident(ctx, identity, ResidentReclaim::ResourceReleased, counters);
        }
        victims.len()
    }

    /// Mark a resident ready after a draw stored into it.
    ///
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

    /// Record a guest CPU write into the canonical allocation imported by this
    /// resident.
    ///
    /// Host access changes neither the image identity nor its Vulkan layout.
    /// It does change the source scope for the next GPU access, so the access
    /// tracker must name `HOST_WRITE` while retaining the layout established by
    /// the preceding GPU transition. Refuse device-local residents: their guest
    /// pages are a different allocation and need a copy, not this transition.
    pub(crate) fn registry_note_guest_write(
        &mut self,
        identity: &TargetIdentity,
        epoch: u32,
    ) -> bool {
        let Some(slot) = self.registry.get_mut(identity) else {
            return false;
        };
        if !slot.content_ready || slot.memory.guest_memory().is_none() {
            return false;
        }
        slot.content_epoch = Some(epoch);
        slot.access = ResidentAccess::HostWrite(slot.access.layout());
        let authority = guest_resident_backing_key(slot);
        if let Some(key) = authority {
            self.guest_resident_authority.insert(key, identity.clone());
        }
        true
    }

    /// Record a non-writing touch of a resident: a draw sampled it, or a
    /// transfer read it out (present blit, guest-page readback, GPU seed
    /// source). The writing touches go through [`Self::registry_mark_ready_at`]
    /// or [`Self::registry_mark_ready_with_access`], which also vouch for the pixels.
    ///
    /// Every rail that touches a resident has to land in one of the three,
    /// because the next barrier over that image derives its source scope from
    /// what it finds here — see [`ResidentAccess`].
    pub(crate) fn registry_note_access(
        &mut self,
        identity: &TargetIdentity,
        access: ResidentAccess,
    ) {
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
    /// [`crate::translate::pixel::bytes_per_texel`] declines
    /// (block-compressed, multi-planar — neither of which a colour attachment
    /// uses) contributes nothing rather than a guessed size. So the total is a
    /// lower bound on VRAM, which is the safe direction for a figure that exists
    /// to decide whether a bound is too loose.
    fn slot_attachment_bytes(slot: &ResidentTargetSlot) -> u64 {
        crate::translate::pixel::bytes_per_texel(slot.format.declared())
            .map(|texel| {
                u64::from(slot.width)
                    * u64::from(slot.height)
                    * u64::from(slot.sample_count)
                    * u64::from(texel)
            })
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
    /// ([`crate::memory::MemoryProfile::device_local_bytes`])
    /// measured in gigabytes. One constant could not be both.
    ///
    /// A count crossing under load means the guest is using more targets than
    /// the count allowed, which is the worst moment to take one away. Measured,
    /// it never came close: peak 194 of 320 under
    /// `web-content-probe --churn 1`, `evicts=0` on every boot ever taken.
    ///
    /// # What bounds it now
    ///
    /// The allocation. [`ResourcePools::registry_ensure`] tries the recycle pool,
    /// then `vkCreateImage` + [`ResourcePools::bind_image_slab`]; on an
    /// out-of-memory result it calls [`Self::reclaim_for_allocation_retry`],
    /// which gives back every recycle pool plus everything this function returns,
    /// and retries once. If that still fails the draw refuses with the driver's
    /// own error. That is a GPU refusing because its memory is full — current
    /// and attributable — rather than a count destroying an earlier accepted
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
        reims_vgpu_observe::fail(format!(
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
    /// bounding this population. At allocation failure the registry may hold
    /// reproducible residents it can safely give back. `pin_count` and
    /// `gpu_only_content` are honoured, so this can only cost re-reads, never a
    /// frame.
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
        reims_vgpu_observe::fail(format!(
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

    /// Test seam for advancing one resident's idle age without running the
    /// device-touching reclaim operation.
    #[cfg(test)]
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
    /// This distinguishes a missing identity that was explicitly released or
    /// pressure-reclaimed from one that this device never held. The timestamp
    /// reports how quickly it was requested again; it does not infer an idle
    /// lifetime or authorize another reclaim.
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
    /// Sampling is a real use even when no draw renders back into the target.
    /// Refresh the timestamp so reuse-distance diagnostics describe all GPU
    /// access, including compositor backdrops. No residency decision reads this
    /// timestamp: live targets end through resource lifetime or, when safe,
    /// allocation-pressure recovery.
    pub(crate) fn registry_note_sampled_use(&mut self, identity: &TargetIdentity) {
        let touch = self.idle_clock_ms;
        if let Some(slot) = self.registry.get_mut(identity) {
            let idle_ms = touch.saturating_sub(slot.last_touch_ms);
            slot.last_touch_ms = touch;
            crate::telemetry::note_route(resident_resample_band(idle_ms));
            // The bands give the distribution; this gives the margin. They
            // answer different questions: the peak preserves the longest exact
            // reuse interval while the bands keep a cheap distribution.
            self.resident_resample_peak_ms = self.resident_resample_peak_ms.max(idle_ms);
        }
    }
}

#[cfg(test)]
pub(super) mod pin_count_tests {
    use super::*;

    /// A registry slot a pin will be granted on, for tests in sibling modules
    /// that need a resident to hold rather than a resident to inspect.
    pub(in crate::engine::pools) fn ready_slot() -> ResidentTargetSlot {
        dummy_slot(true)
    }

    fn dummy_slot(content_ready: bool) -> ResidentTargetSlot {
        ResidentTargetSlot {
            incarnation: super::ResidentIncarnation::allocate(),
            image: vk::Image::null(),
            memory: ResidentMemory::Recyclable(vk::DeviceMemory::null()),
            guest_backing: None,
            guest_materialization: None,
            view: vk::ImageView::null(),
            alternate_views: Vec::new(),
            framebuffer: vk::Framebuffer::null(),
            render_pass: vk::RenderPass::null(),
            framebuffer_compatibility: None,
            width: 16,
            height: 16,
            sample_count: 1,
            mip_levels: 1,
            generation: 1,
            content_ready,
            content_epoch: None,
            // What `registry_mark_ready` actually records, read from the same
            // constant it reads, so this fixture cannot drift into describing a
            // resident no pass produces.
            access: ResidentAccess::ColorWrite(crate::engine::caches::color0_pass_exit_layout()),
            format: translate::pixel::ResidentFormat::of(translate::pixel::SCANOUT_FORMAT),
            pin_count: 0,
            resource_released: false,
            resource_owner_count: 0,
            gpu_only_content: false,
            last_touch_ms: 0,
        }
    }

    fn retain(pools: &mut ResourcePools, identity: &TargetIdentity) -> bool {
        pools.retain_resident_target(identity).is_some()
    }

    fn pinned_identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 1,
            width: 16,
            height: 16,
            generation: 0,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }
    }

    /// A swizzled ask never reaches the allocation's own view, and two
    /// differently-swizzled asks are two views.
    ///
    /// This is the whole safety argument for keying a retained view on the
    /// format *and* the mapping. The alias rail used to refuse a non-identity
    /// mapping outright and copy the texture instead — 99.3 % of every refusal
    /// it produced on a driven Maps boot, one per single-channel glyph coverage
    /// texture. Admitting it means the cache key has to be able to tell the
    /// mappings apart, and a key that could not would hand a `.rrrr` bind the
    /// unmapped view: correct-looking, silently wrong channels, and counted
    /// nowhere.
    #[test]
    fn a_retained_view_is_keyed_on_the_mapping_as_well_as_the_format() {
        let alpha_in_red = reims_vgpu_protocol::SwizzlePlan {
            source: [reims_vgpu_protocol::SwizzleSource::A; 4],
        };
        let red_everywhere = reims_vgpu_protocol::SwizzlePlan {
            source: [reims_vgpu_protocol::SwizzleSource::R; 4],
        };
        let mut slot = dummy_slot(true);
        slot.view = vk::ImageView::from_raw(0x11);
        let allocation = slot.format.allocation();

        // The allocation's own view answers only the unmapped ask.
        assert_eq!(
            slot.held_view(ResidentViewKey::plain(allocation)),
            Some(vk::ImageView::from_raw(0x11))
        );
        for plan in [alpha_in_red, red_everywhere] {
            assert_eq!(
                slot.held_view(ResidentViewKey {
                    format: allocation,
                    swizzle: plan,
                }),
                None,
                "a mapped ask was served the allocation's unmapped view"
            );
        }

        // Each mapping retains its own, and neither answers for the other.
        slot.alternate_views.push((
            ResidentViewKey {
                format: allocation,
                swizzle: alpha_in_red,
            },
            vk::ImageView::from_raw(0x22),
        ));
        assert_eq!(
            slot.held_view(ResidentViewKey {
                format: allocation,
                swizzle: alpha_in_red,
            }),
            Some(vk::ImageView::from_raw(0x22))
        );
        assert_eq!(
            slot.held_view(ResidentViewKey {
                format: allocation,
                swizzle: red_everywhere,
            }),
            None,
            "one mapping's view answered for another's"
        );
        assert_eq!(
            slot.held_view(ResidentViewKey::plain(allocation)),
            Some(vk::ImageView::from_raw(0x11)),
            "a mapped view displaced the allocation's own"
        );
    }

    #[test]
    fn guest_sampling_reuses_only_the_same_ready_imported_allocation() {
        let import = std::sync::Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(0x1000, 0x4000, 0x1000).unwrap(),
        );
        let backing = reims_vgpu_memory::GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000,
            resource_offset: 0,
            resource_len: 0x4000,
            plane_offset: 0,
            row_pitch: 64,
        };
        let footprint =
            reims_vgpu_memory::GuestPageFootprint::new(std::sync::Arc::from([0x1000]), 0x1000)
                .unwrap();
        let memory = reims_vgpu_memory::GuestTargetMemory {
            backing,
            import: import.clone(),
            footprint,
        };
        let imported_slot = || {
            let mut slot = dummy_slot(true);
            slot.memory = ResidentMemory::GuestImported {
                memory: memory.clone(),
                write_pages: reims_vgpu_memory::GuestWritePages::new(&[0x1000]).unwrap(),
            };
            slot
        };
        let mut slot = imported_slot();
        let sampled = SampledKey {
            width: 16,
            height: 16,
            mip_levels: 1,
            layers: 1,
            volume: false,
            cube: false,
            arrayed: false,
            one_dim: false,
            format: slot.format.allocation(),
            swizzle: Default::default(),
        };
        assert!(guest_sampled_matches_resident(&slot, &memory, sampled));

        let mut other = memory.clone();
        other.backing.plane_offset = 0x1000;
        assert!(!guest_sampled_matches_resident(&slot, &other, sampled));

        slot.content_ready = false;
        assert!(!guest_sampled_matches_resident(&slot, &memory, sampled));

        let first = pinned_identity();
        let second = TargetIdentity::Gva {
            gva: 0x8000,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
        };
        let mut registry = HashMap::new();
        let mut authority = HashMap::new();
        registry.insert(first.clone(), imported_slot());
        assert_eq!(
            guest_sampled_resident_match(&registry, &authority, &memory, sampled, |_| false),
            Ok(Some(first.clone()))
        );
        assert_eq!(
            guest_sampled_resident_match(&registry, &authority, &memory, sampled, |id| {
                id == &first
            }),
            Ok(None),
            "an attachment the draw writes cannot be selected opportunistically"
        );
        registry.insert(second.clone(), imported_slot());
        assert_eq!(
            guest_sampled_resident_match(&registry, &authority, &memory, sampled, |_| false),
            Err(2),
            "two image identities over one backing must not be selected by map order"
        );
        let key = guest_resident_backing_key(&registry[&second]).unwrap();
        authority.insert(key, second.clone());
        assert_eq!(
            guest_sampled_resident_match(&registry, &authority, &memory, sampled, |_| false),
            Ok(Some(second)),
            "the backing's last ordered writer is its content authority"
        );
    }

    #[test]
    fn a_sampled_resident_pin_moves_into_the_submission_cleanup() {
        let mut pools = ResourcePools::new();
        let identity = pinned_identity();
        pools.registry.insert(identity.clone(), dummy_slot(true));

        assert!(pools.pin_resident_for_entry(&identity));
        assert_eq!(pools.registry.get(&identity).unwrap().pin_count, 1);
        let sealed = pools.seal_entry(Vec::new(), Vec::new());
        assert!(pools.resident_pins_live.is_empty());
        assert_eq!(sealed.cleanup.unpin_residents, vec![identity]);
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
        rgba.format = translate::pixel::ResidentFormat::of(translate::pixel::RESIDENT_RGBA_FORMAT);
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
    /// content the IOSurface texture LOAD gate must not treat them as current — otherwise
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

    #[test]
    fn a_guest_write_updates_only_a_resident_over_the_guest_allocation() {
        let mut pools = ResourcePools::new();
        let imported_id = pinned_identity();
        let device_id = TargetIdentity::Gva {
            gva: 0x8000,
            width: 16,
            height: 16,
            generation: 2,
            format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
        };
        let mut imported = dummy_slot(true);
        let import = std::sync::Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(0x1000, 0x4000, 0x1000).unwrap(),
        );
        let footprint =
            reims_vgpu_memory::GuestPageFootprint::new(std::sync::Arc::from([0x1000]), 0x1000)
                .unwrap();
        imported.memory = ResidentMemory::GuestImported {
            memory: reims_vgpu_memory::GuestTargetMemory {
                backing: reims_vgpu_memory::GuestTargetBacking {
                    allocation_host_ptr: 0x1000,
                    allocation_len: 0x4000,
                    resource_offset: 0,
                    resource_len: 0x4000,
                    plane_offset: 0,
                    row_pitch: 64,
                },
                import,
                footprint,
            },
            write_pages: reims_vgpu_memory::GuestWritePages::new(&[0x1000]).unwrap(),
        };
        imported.access = ResidentAccess::ColorWrite(vk::ImageLayout::GENERAL);
        pools.registry.insert(imported_id.clone(), imported);
        pools.registry.insert(device_id.clone(), dummy_slot(true));

        assert!(pools.registry_note_guest_write(&imported_id, 11));
        let imported = &pools.registry[&imported_id];
        assert_eq!(imported.content_epoch, Some(11));
        assert_eq!(
            imported.access,
            ResidentAccess::HostWrite(vk::ImageLayout::GENERAL),
            "host access keeps the established layout and changes the source scope"
        );

        assert!(!pools.registry_note_guest_write(&device_id, 12));
        assert_eq!(pools.registry[&device_id].content_epoch, None);
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

    #[test]
    fn retaining_a_resource_pins_its_engine_owned_resident() {
        let mut pools = ResourcePools::new();
        let recyclable_id = pinned_identity();
        pools
            .registry
            .insert(recyclable_id.clone(), dummy_slot(true));
        assert!(retain(&mut pools, &recyclable_id));
        assert_eq!(pools.registry[&recyclable_id].pin_count, 1);
    }

    #[test]
    fn ending_the_last_resource_ownership_makes_the_slot_retirable() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        pools.registry_order.push_back(id.clone());

        assert!(retain(&mut pools, &id));
        assert_eq!(pools.release_resident_ownership(&id), Some(true));
        assert_eq!(pools.released_resident_keys(1), vec![id]);
    }

    #[test]
    fn an_old_lease_cannot_release_the_replacement_under_the_same_identity() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        let old_incarnation = pools
            .retain_resident_target(&id)
            .expect("the old allocation is retained");

        pools.registry.insert(id.clone(), dummy_slot(true));
        let replacement = pools.registry[&id].incarnation;
        assert_ne!(replacement, old_incarnation);
        assert_eq!(
            pools.release_resident_ownership_for(&id, old_incarnation),
            None,
            "the old lease names an allocation no longer in the registry"
        );
        assert_eq!(
            pools.registry[&id].resource_owner_count, 0,
            "the replacement acquired no owner and must not lose one"
        );
        assert_eq!(pools.registry[&id].pin_count, 0);
    }

    #[test]
    fn one_alias_release_does_not_end_another_resources_ownership() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        pools.registry_order.push_back(id.clone());

        assert!(retain(&mut pools, &id));
        assert!(retain(&mut pools, &id));
        assert_eq!(pools.registry[&id].resource_owner_count, 2);
        assert_eq!(pools.release_resident_ownership(&id), Some(false));
        assert_eq!(pools.registry[&id].resource_owner_count, 1);
        assert!(!pools.registry[&id].resource_released);
        assert!(pools.released_resident_keys(1).is_empty());

        assert!(
            retain(&mut pools, &id),
            "the surviving alias keeps the shared allocation retainable"
        );
    }

    #[test]
    fn released_resource_waits_for_existing_holders_but_not_for_time() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        pools.registry_order.push_back(id.clone());

        assert!(retain(&mut pools, &id));
        assert!(pools.pin_resident_target(&id, true), "in-flight holder");
        assert_eq!(pools.release_resident_ownership(&id), Some(false));
        assert!(pools.released_resident_keys(1).is_empty());

        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(pools.released_resident_keys(1), vec![id]);
    }

    /// A released resource whose resident still holds the only copy of a frame
    /// survives the release, and is collected once that frame has been copied
    /// out.
    ///
    /// This is [`ResidentTargetSlot::released_and_collectable`]'s third term.
    /// Without it, a render Store that deferred its writeback into
    /// `writeback_debt` and a guest that released the serialized resource
    /// before the debt was paid destroy the frame between them — which is what
    /// **135 of 135** `read_target_unknown_identity diverges=absent
    /// prior=resource_released` refusals were, on one driven macos-13 boot of
    /// the copying rail.
    ///
    /// The wait is on the copy-out and not on time: the second half asserts the
    /// slot does not become uncollectable, only late.
    #[test]
    fn a_released_resource_holding_the_only_copy_of_a_frame_waits_for_the_copy_out() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        pools.registry_order.push_back(id.clone());

        assert!(retain(&mut pools, &id));
        pools.registry_mark_ready(&id);
        assert!(
            pools.registry[&id].gpu_only_content,
            "a Store with no copy-out leaves the resident sole-copy"
        );

        assert_eq!(
            pools.release_resident_ownership(&id),
            Some(false),
            "the guest ended the lifetime, but the frame is still only here"
        );
        assert!(pools.released_resident_keys(1).is_empty());
        assert!(
            pools.registry.contains_key(&id),
            "the resident the owed writeback names is still findable"
        );

        assert!(pools.registry_note_content_copied_out(&id));
        assert_eq!(
            pools.released_resident_keys(1),
            vec![id],
            "once the frame is in the guest's pages the slot is collectable"
        );
    }

    fn surf(id: u32) -> TargetIdentity {
        TargetIdentity::Surface {
            id,
            width: 16,
            height: 16,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }
    }

    /// A resident shaped like the one an arm builds, for the registration tests.
    ///
    /// The two arms differ in exactly these two handles, so they are the
    /// parameters: `registry_ensure` passes a real framebuffer and the pass it
    /// was built against, `registry_ensure_attachment` passes neither.
    fn new_resident(framebuffer: vk::Framebuffer, render_pass: vk::RenderPass) -> NewResident {
        let framebuffer_compatibility = (framebuffer != vk::Framebuffer::null()).then(|| {
            crate::engine::caches::PassKey::single(
                crate::engine::caches::ColorLoadKey::Clear,
                translate::pixel::SCANOUT_FORMAT,
            )
            .framebuffer_compatibility()
        });
        NewResident {
            image: vk::Image::null(),
            memory: ResidentMemory::Recyclable(vk::DeviceMemory::null()),
            guest_backing: None,
            guest_materialization: None,
            view: vk::ImageView::null(),
            framebuffer,
            render_pass,
            framebuffer_compatibility,
            width: 16,
            height: 16,
            sample_count: 1,
            mip_levels: 1,
            generation: 1,
            format: translate::pixel::ResidentFormat::of(translate::pixel::SCANOUT_FORMAT),
            attachment_view: None,
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
    /// the IOSurface texture LOAD gate's epoch check, the barrier tracker and the idle
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
    fn elapsed_time_never_reclaims_a_live_resident() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 10, 0);
        // The MRT-secondary path: rendered into, marked ready at the pass's
        // final layout, never pinned, never stamped, never flushed.
        pools.registry_mark_ready_at(&surf(1), vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        pools.registry_touch_at(&surf(1), 10);

        let now = 10 + IDLE_MAINTENANCE_START_MS + 1;
        assert!(pools.plan_idle_maintenance(now));
        assert!(pools.registry.contains_key(&surf(1)));

        // Ten more cutoffs' worth of idleness changes nothing: this is not a
        // longer timer, it is a different question.
        let much_later = now + IDLE_MAINTENANCE_START_MS * 10;
        assert!(pools.plan_idle_maintenance(much_later));
        assert!(pools.registry.contains_key(&surf(1)));

        // Something copies the pixels out — a landed flush, a writeback Store —
        // and the same resident is now exactly as reclaimable as any other.
        assert!(
            pools.registry_note_content_copied_out(&surf(1)),
            "the slot is there to be cleared"
        );
        let later_still = much_later + MAINTENANCE_INTERVAL_MS + 1;
        assert!(pools.plan_idle_maintenance(later_still));
        assert!(
            pools.registry.contains_key(&surf(1)),
            "a current guest copy makes pressure reclaim safe, not time authoritative"
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
            ResidentReclaim::AllocationReclaimed,
            ResidentReclaim::Recreated,
            ResidentReclaim::ResourceReleased,
        ] {
            let mut pools = ResourcePools::new();
            admit(&mut pools, surf(1), 10, 0);
            // The sole-copy shape: rendered into and marked ready, never pinned,
            // never stamped, never written back.
            pools.registry_mark_ready(&surf(1));
            pools.registry_touch_at(&surf(1), 10);
            match cause {
                ResidentReclaim::AllocationReclaimed => assert!(
                    pools.recoverable_residents().is_empty(),
                    "a device out of memory refuses the next allocation rather \
                     than destroying a client's only render target"
                ),
                // Exempt, and the assertion says why rather than skipping: the
                // slot is still the sole copy, and that is not what stops this
                // cause — the guest asking for a different target is.
                ResidentReclaim::Recreated | ResidentReclaim::ResourceReleased => assert!(
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
        use crate::engine::vk_call::{VkCall, VkOp};
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

    /// A non-pinned resident untouched for `IDLE_MAINTENANCE_START_MS` is selected; a
    /// freshly-touched peer and a pinned peer are not. The wall clock advances to
    /// the passed `now_ms` (not a per-call increment), so a static guest that
    /// keeps ticking the poll heartbeat still reclaims stale VRAM.
    #[test]
    fn maintenance_never_selects_live_residents_by_age() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 10, 0); // aged, non-pinned  -> victim
        admit(&mut pools, surf(2), 10, 1); // aged but PINNED   -> kept
                                           // now = 10 + AGE + 1 so slot 1's cutoff is crossed; a fresh slot is not.
        let now = 10 + IDLE_MAINTENANCE_START_MS + 1;
        admit(&mut pools, surf(3), now, 0); // fresh            -> kept
        assert!(pools.plan_idle_maintenance(now), "maintenance pass is due");
        assert_eq!(pools.registry.len(), 3, "time changes no live residency");
        assert_eq!(pools.idle_clock_ms, now, "clock advanced to wall time");
    }

    /// A reclaim records *when*, so the gap the drain censored can be recovered.
    ///
    /// A reclaim records when it happened so diagnostics can report how soon an
    /// explicitly released or pressure-reclaimed identity was requested again.
    #[test]
    fn a_reclaim_records_when_so_the_censored_gap_can_be_recovered() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        pools.idle_clock_ms = 5_000;
        pools.unregister_resident(&surf(1), ResidentReclaim::AllocationReclaimed);

        let (why, at) = pools
            .prior_reclaim_at(&surf(1))
            .expect("a reclaim is recorded with its time");
        assert_eq!(why, ResidentReclaim::AllocationReclaimed);
        assert_eq!(at, 5_000, "stamped with the maintenance clock");

        // The guest comes back 3 s after the destroy.
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

    /// The resample bands use the maintenance start interval as a stable scale.
    /// They are diagnostics only: crossing any boundary has no effect on a live
    /// resident's lifetime.
    #[test]
    fn the_resident_resample_bands_are_fractions_of_the_drain_cutoff() {
        let c = IDLE_MAINTENANCE_START_MS;
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
        pools.unregister_resident(&surf(1), ResidentReclaim::AllocationReclaimed);
        assert!(!pools.registry.contains_key(&surf(1)));
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::AllocationReclaimed)
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
            Some(ResidentReclaim::AllocationReclaimed),
            "history is deliberately not cleared on re-admit, which is why the \
             presence check cannot be dropped"
        );
    }

    /// The resample peak is the worst gap the boot ever saw, not the last one.
    ///
    /// A high-water, so a large gap early is not erased by a run of small ones
    /// after it — which is the whole reason it is not a windowed reading. The
    /// margin question is "how close did this boot ever come to
    /// `IDLE_MAINTENANCE_START_MS`", and a gap that peaks between two census samples is
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
        pools.idle_clock_ms = 1_000 + IDLE_MAINTENANCE_START_MS;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_MAINTENANCE_START_MS);

        // A read of an identity the registry does not hold records nothing —
        // there is no gap to measure, and counting it as one would report a
        // margin that no resident ever spent.
        pools.idle_clock_ms = u64::MAX;
        pools.registry_note_sampled_use(&surf(99));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_MAINTENANCE_START_MS);
    }

    /// A resident that a draw only ever samples remains live, and its diagnostic
    /// timestamp reflects that use.
    #[test]
    fn a_sampled_only_resident_is_not_aged_out() {
        let mut pools = ResourcePools::new();
        // Aged past the cutoff by every measure the drain has, and never drawn
        // into again — only read.
        admit(&mut pools, surf(1), 10, 0);
        admit(&mut pools, surf(2), 10, 0);
        let now = 10 + IDLE_MAINTENANCE_START_MS + 1;
        // The drain's clock has to be current before a use can be recorded
        // against it; the real caller advances it from the poll heartbeat.
        pools.plan_idle_maintenance(now);
        pools.registry_note_sampled_use(&surf(1));
        // A second pass, far enough after the first to clear the throttle.
        let later = now + MAINTENANCE_INTERVAL_MS + 1;
        assert!(pools.plan_idle_maintenance(later), "second pass is due");
        assert!(pools.registry.contains_key(&surf(1)));
        assert!(pools.registry.contains_key(&surf(2)));
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
        pools.note_resident_reclaimed(&surf(1), ResidentReclaim::AllocationReclaimed);
        pools.note_resident_reclaimed(&surf(2), ResidentReclaim::AllocationReclaimed);
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::AllocationReclaimed)
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

    /// The reclaim pass is throttled to `MAINTENANCE_INTERVAL_MS`: a second call
    /// inside the interval selects nothing even though a resident is aged, so the
    /// ~244 Hz poll cadence cannot empty the registry at once. The clock still
    /// advances (admits stay fresh).
    #[test]
    fn maintenance_is_throttled_between_passes() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        let t0 = IDLE_MAINTENANCE_START_MS + 1;
        assert!(pools.plan_idle_maintenance(t0));
        admit(&mut pools, surf(2), 0, 0);
        assert!(!pools.plan_idle_maintenance(t0 + 1), "throttled: no pass");
        assert_eq!(
            pools.idle_clock_ms,
            t0 + 1,
            "clock still advances when throttled"
        );
        assert!(pools.plan_idle_maintenance(t0 + MAINTENANCE_INTERVAL_MS));
        assert_eq!(
            pools.registry.len(),
            2,
            "maintenance owns no live residents"
        );
    }

    /// A maintenance pass cannot shrink a live registry, regardless of its size.
    #[test]
    fn maintenance_does_not_shrink_a_large_live_registry() {
        let mut pools = ResourcePools::new();
        const LIVE_RESIDENTS: usize = 9;
        for i in 0..LIVE_RESIDENTS as u32 {
            admit(&mut pools, surf(100 + i), 0, 0);
        }
        assert!(pools.plan_idle_maintenance(IDLE_MAINTENANCE_START_MS + 1));
        assert_eq!(pools.registry.len(), LIVE_RESIDENTS);
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
            pools.note_maintenance_settled();
        }
        assert!(
            pools.note_maintenance_settled(),
            "no victims, no uploads → settled"
        );

        // One staging acquire between passes — no victim, still not settled.
        pools.staging_hits += 1;
        assert!(
            !pools.note_maintenance_settled(),
            "uploads ran between passes; the buffer pools must not be trimmed"
        );
        // …and the gate stays shut while uploads keep flowing, however many
        // zero-victim passes go by.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM * 3) {
            pools.staging_misses += 1;
            assert!(!pools.note_maintenance_settled(), "still uploading");
        }
        // Uploads stop: the gate reopens after the usual consecutive passes.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(
                !pools.note_maintenance_settled(),
                "counter restarted from zero"
            );
        }
        assert!(
            pools.note_maintenance_settled(),
            "settled once uploads stopped"
        );
    }

    /// The HOST_VISIBLE buffer trim gate: only permitted after
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive passes without upload
    /// activity, so a staging buffer cannot be freed and re-allocated mid-video.
    #[test]
    fn note_maintenance_settled_gates_buffer_trim_on_consecutive_idle() {
        let mut pools = ResourcePools::new();
        // Fewer than the threshold of quiet passes: no buffer trim yet.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_maintenance_settled(), "not settled enough yet");
        }
        // The Nth consecutive zero-victim pass crosses the threshold.
        assert!(
            pools.note_maintenance_settled(),
            "N consecutive settled passes → trim allowed"
        );
        // A subsequent quiet pass stays allowed.
        assert!(pools.note_maintenance_settled(), "stays settled");
        // Upload activity resets the counter.
        pools.staging_hits += 1;
        assert!(
            !pools.note_maintenance_settled(),
            "uploads reset settled state"
        );
        // …and the gate stays closed until the run rebuilds.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(
                !pools.note_maintenance_settled(),
                "counter restarted from zero"
            );
        }
        assert!(
            pools.note_maintenance_settled(),
            "settled again after rebuild"
        );
    }

    /// The presented target passed as `display` is stamped to the current clock
    /// every call, so reuse-distance diagnostics include presentation reads.
    #[test]
    fn maintenance_keeps_the_display_target_alive_without_a_special_case() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // would be aged...
        let now = IDLE_MAINTENANCE_START_MS + 500;
        // ...but it is the presented target this frame.
        assert!(pools.plan_idle_maintenance(now));
        assert!(pools.registry.contains_key(&surf(1)));
    }

    /// Touching changes diagnostics, not lifetime: both touched and untouched
    /// live targets survive a static desktop interval.
    #[test]
    fn maintenance_does_not_require_touching_a_live_target() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // displayed target
        admit(&mut pools, surf(4), 0, 0); // registered but undrawn, otherwise aged
        let now = IDLE_MAINTENANCE_START_MS + 500;

        pools.registry_touch_at(&surf(4), now);
        assert!(pools.plan_idle_maintenance(now));
        assert!(pools.registry.contains_key(&surf(1)));
        assert!(pools.registry.contains_key(&surf(4)));
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
