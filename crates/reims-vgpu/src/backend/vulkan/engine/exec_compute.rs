//! Record / submit (bounded fence) / readback for one compute dispatch.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use super::caches::{
    canonicalize_layout_bindings, BindingSig, ComputePipelineKey, LayoutKey, ObjectCaches,
};
use super::compute_execution::ComputeExecutionDecline;
use super::compute_validation::ComputeValidationDecline;
use super::context::ContextOwner;
use super::counters::EngineCounters;
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::pools::{BufferSlot, ResourcePools, StorageImageKey, StorageImageSlot};
use super::types::{
    ComputeBufferOutput, ComputeDispatch, ComputeDispatchPayload, ComputeOutput, ComputeRequest,
    ComputeResidentSampleBind, ComputeSampledImageResource, ComputeSampledSource,
    ComputeStorageResidency, DrawError, TargetIdentity,
};
use super::vk_call::{VkCall, VkOp};

/// One recorded `vkCmdDispatch`, with the pipeline its workgroup size selected
/// and the payload that names its place in the logical grid.
struct DispatchStep {
    pipeline: vk::Pipeline,
    group_count: [u32; 3],
    push: Option<(u32, ComputeDispatchPayload)>,
}

struct PreparedStorageImage {
    binding: u32,
    array_element: u32,
    slot: StorageImageSlot,
    seed: Option<BufferSlot>,
    dst: ComputeImageDst,
    len: usize,
    width: u32,
    height: u32,
    /// What last touched the image this slot holds, before this dispatch —
    /// [`super::pools::ResidentAccess::Untouched`] for a freshly acquired pooled
    /// slot, and whatever the compute-storage registry recorded for a resident.
    initial_access: super::pools::ResidentAccess,
    residency: Option<ComputeStorageResidency>,
}

/// One prepared sampled input: a transient sampled-only image seeded either
/// from a host staging upload or from a device-local resident copy.
enum PreparedSampledImage {
    /// A pooled transient this dispatch fills before it reads: from a staging
    /// upload, or by a device-local copy out of a resident storage image.
    Staged {
        binding: u32,
        array_element: u32,
        img: StorageImageSlot,
        upload: Option<BufferSlot>,
        /// Copy-on-sample source `(resident image, what last touched it)`.
        resident_src: Option<(vk::Image, super::pools::ResidentAccess)>,
        width: u32,
        height: u32,
        mip_levels: u32,
    },
    /// A retained multisample render target, bound through its own registry
    /// view. Nothing is acquired, uploaded or copied — see
    /// [`ComputeSampledSource::MultisampleTarget`] for why there is no
    /// alternative for this shape.
    MultisampleTarget {
        binding: u32,
        array_element: u32,
        identity: TargetIdentity,
        image: vk::Image,
        view: vk::ImageView,
        access: super::pools::ResidentAccess,
        next_access: super::pools::ResidentAccess,
    },
}

impl PreparedSampledImage {
    fn binding(&self) -> u32 {
        match self {
            Self::Staged { binding, .. } | Self::MultisampleTarget { binding, .. } => *binding,
        }
    }

    fn array_element(&self) -> u32 {
        match self {
            Self::Staged { array_element, .. } | Self::MultisampleTarget { array_element, .. } => {
                *array_element
            }
        }
    }

    /// The view the descriptor binds, and the layout it will be in when the
    /// dispatch reads it.
    fn descriptor_view(&self) -> (vk::ImageView, vk::ImageLayout) {
        match self {
            Self::Staged { img, .. } => (img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            Self::MultisampleTarget {
                view, next_access, ..
            } => (*view, next_access.layout()),
        }
    }
}

/// Post-dispatch copy destination for one storage image.
enum ComputeImageDst {
    /// Pooled host-visible buffer; the CPU reads it back and the runtime
    /// writes guest pages itself.
    Readback(BufferSlot),
    /// The dispatch's own image→buffer copy lands in the guest's pages and no
    /// pixels cross device→host.
    ///
    /// An older variant of this name bound a transfer-dst buffer over an
    /// imported view of the caller's guest window and was removed, because a
    /// raw buffer the GPU can write backed by guest pages is an unbounded
    /// reach into this process's address space. This one is not that: it
    /// carries a [`super::GuestCopyPlan`], built by the same `plan_guest_copy`
    /// the render rail uses, whose every destination range is a `GuestSlice`
    /// bounds-checked against the one RAMBlock import that produced it. The
    /// bound is the type, which is the whole argument in
    /// `runtime/guest_ram.rs`.
    Direct(super::GuestCopyPlan),
}

pub(crate) fn validate_compute(req: &ComputeRequest) -> Result<(), DrawError> {
    if req.spirv.is_empty() {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EmptySpirv,
        ));
    }
    if req.entry.is_empty() {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EmptyEntry,
        ));
    }
    if req.entry.as_bytes().contains(&0) {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EntryInteriorNul,
        ));
    }
    let threadgroups_per_grid = req.dispatch.threadgroups_per_grid();
    if threadgroups_per_grid.contains(&0) {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::ZeroGrid {
                grid: threadgroups_per_grid,
            },
        ));
    }
    if let ComputeDispatch::Regions { regions, .. } = &req.dispatch {
        if regions.is_empty() {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::NoDispatchRegions,
            ));
        }
        for (index, region) in regions.iter().enumerate() {
            if region.local_size.contains(&0) || region.group_count.contains(&0) {
                return Err(DrawError::ComputeValidation(
                    ComputeValidationDecline::ZeroDispatchRegion {
                        region: index,
                        local_size: region.local_size,
                        group_count: region.group_count,
                    },
                ));
            }
        }
    }
    let mut bindings = BTreeSet::new();
    for b in &req.storage_buffers {
        if !bindings.insert((b.binding, 0)) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateStorageBufferBinding { binding: b.binding },
            ));
        }
        if b.bytes.is_empty() {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::EmptyStorageBuffer { binding: b.binding },
            ));
        }
    }
    for img in &req.sampled_images {
        if img.descriptor_count == 0 || img.array_element >= img.descriptor_count {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledArrayElementOutOfRange {
                    binding: img.binding,
                    element: img.array_element,
                    count: img.descriptor_count,
                },
            ));
        }
        if !bindings.insert((img.binding, img.array_element)) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateSampledImageBinding {
                    binding: img.binding,
                },
            ));
        }
        if img.width == 0 || img.height == 0 || img.mip_levels == 0 {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledZeroGeometry {
                    binding: img.binding,
                    width: img.width,
                    height: img.height,
                },
            ));
        }
        // Only the source that actually carries bytes owes a length. The other
        // two derive theirs from this same geometry, so there is nothing left
        // for them to disagree with.
        //
        // The whole pyramid, not the base: `bytes` carries every level the
        // binding declares, and checking only the base would let a request
        // through whose upper levels the copy then reads past the end of.
        if let ComputeSampledSource::Bytes(bytes) = &img.source {
            let expected = crate::contract::extent::tight_pyramid_bytes(
                img.width,
                img.height,
                img.mip_levels,
                img.format.bytes_per_texel(),
            )
            .unwrap_or(usize::MAX);
            if bytes.len() != expected {
                return Err(DrawError::ComputeValidation(
                    ComputeValidationDecline::SampledBytesLength {
                        binding: img.binding,
                        actual: bytes.len(),
                        expected,
                    },
                ));
            }
        }
    }
    for sampler in &req.samplers {
        let lod_min = sampler.lod_min_f32();
        let lod_max = sampler.lod_max_f32();
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::InvalidSamplerLod {
                    binding: sampler.binding,
                    lod_min_bits: sampler.lod_min,
                    lod_max_bits: sampler.lod_max,
                },
            ));
        }
        if !bindings.insert((sampler.binding, 0)) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateSamplerBinding {
                    binding: sampler.binding,
                },
            ));
        }
    }
    for img in &req.storage_images {
        if img.descriptor_count == 0 || img.array_element >= img.descriptor_count {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageArrayElementOutOfRange {
                    binding: img.binding,
                    element: img.array_element,
                    count: img.descriptor_count,
                },
            ));
        }
        if !bindings.insert((img.binding, img.array_element)) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateStorageImageBinding {
                    binding: img.binding,
                },
            ));
        }
        if img.width == 0 || img.height == 0 {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageZeroGeometry {
                    binding: img.binding,
                    width: img.width,
                    height: img.height,
                },
            ));
        }
        let expected = (img.width as usize)
            .saturating_mul(img.height as usize)
            .saturating_mul(img.format.bytes_per_texel());
        if img.bytes.len() != expected {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageBytesLength {
                    binding: img.binding,
                    actual: img.bytes.len(),
                    expected,
                },
            ));
        }
    }
    Ok(())
}

/// A copy-on-sample bind must name a resident whose image is byte-for-byte the
/// image the view will sample: same vk format, same width, same height. The
/// runtime only ever binds the key it looked the mirror up under, so an
/// inexact source means the registry and the mirror disagree — refuse by name
/// rather than reinterpreting bytes the shader did not ask us to reinterpret.
fn resident_sample_exact(
    resource: &ComputeSampledImageResource,
    bind: ComputeResidentSampleBind,
    src_key: StorageImageKey,
) -> Result<(), DrawError> {
    let exact = src_key.format.vk_format() == resource.format.vk_format()
        && src_key.width == resource.width
        && src_key.height == resource.height;
    if !exact {
        let source_row_bytes = src_key.width as u64 * src_key.format.bytes_per_texel() as u64;
        let resource_row_bytes = resource.width as u64 * resource.format.bytes_per_texel() as u64;
        return Err(DrawError::ComputeExecution(
            ComputeExecutionDecline::ResidentSampleByteShapeMismatch {
                binding: resource.binding,
                identity: bind.identity,
                source_width: src_key.width,
                source_height: src_key.height,
                source_format: src_key.format,
                source_row_bytes,
                resource_width: resource.width,
                resource_height: resource.height,
                resource_format: resource.format,
                resource_row_bytes,
            },
        ));
    }
    Ok(())
}

/// The lowest `set = 0` binding the module statically uses and `layout` does not
/// describe, if there is one.
///
/// This is the one descriptor-layout defect that cannot be reported from any
/// later point, which is why it is checked before the layout is built rather
/// than inferred from a result code afterwards. Vulkan requires the pipeline
/// layout to contain a descriptor for every resource the shader statically uses.
/// Mesa's Intel driver sizes its own binding array to `max_binding + 1`,
/// zero-fills every number the layout did not declare, and then scores each used
/// binding as `(use_count << 7) / array_size` when it picks binding-table slots.
/// A binding the module uses and the layout omits therefore divides by zero:
/// `vkCreateComputePipelines` does not return an error, the host process dies of
/// `SIGFPE` inside it, and the VM goes with it. There is no validation layer in
/// this tree and no status to inspect, so refusing the dispatch here is the only
/// outcome left that keeps the device alive and says why.
///
/// **Expected to return `None` always.** The two repairable classes are filled
/// before the request is built — `runtime::compute_exec` provisions a neutral
/// sampler and a neutral sampled image for every binding of those classes the
/// guest left empty — so this is the backstop for a class those passes do not
/// cover, and a firing names a real gap rather than noise.
///
/// [`crate::runtime::spirv_bind::descriptor_static_use`] answers `NotDeclared`
/// for anything that is not a `UniformConstant` descriptor, so a storage buffer,
/// whose root this walk cannot resolve, is never refused on a guess.
fn used_binding_absent_from_layout(spirv: &[u32], layout: &[BindingSig]) -> Option<u32> {
    crate::runtime::spirv_bind::declared_binding_numbers(spirv)
        .into_iter()
        .find(|binding| {
            !layout.iter().any(|b| b.binding == *binding)
                && crate::runtime::spirv_bind::descriptor_static_use(spirv, *binding).is_violation()
        })
}

pub(crate) unsafe fn execute_compute_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &ComputeRequest,
) -> Result<ComputeOutput, DrawError> {
    validate_compute(req)?;
    let force_loss = owner.force_device_lost;
    if force_loss {
        owner.force_device_lost = false;
    }
    let ctx = owner.ensure(counters)?;
    if !ctx.compute_capable {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::NoCombinedGraphicsComputeQueue,
        ));
    }
    pools.ensure_init(ctx, counters)?;

    // Claim the next ring slot — BEFORE any pool acquire, so a recycled slot
    // can never alias a still-in-flight CB. Blocks (retire) only when every
    // slot is still in flight; the wait lands in retire_wait_us.
    let (cb, fence) = pools.begin_entry(ctx, counters)?;

    let mut layout_bindings = Vec::new();
    for b in &req.storage_buffers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        });
    }
    for img in &req.sampled_images {
        layout_bindings.push(BindingSig {
            binding: img.binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: img.descriptor_count,
        });
    }
    for sampler in &req.samplers {
        layout_bindings.push(BindingSig {
            binding: sampler.binding,
            ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        });
    }
    for img in &req.storage_images {
        layout_bindings.push(BindingSig {
            binding: img.binding,
            ty: vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: img.descriptor_count,
        });
    }
    let layout_bindings = canonicalize_layout_bindings(layout_bindings)?;
    for binding in layout_bindings.iter().filter(|binding| binding.count > 1) {
        let descriptor_type = vk::DescriptorType::from_raw(binding.ty as i32);
        let dynamic_indexing = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => ctx.features.sampled_image_array_dynamic_indexing,
            vk::DescriptorType::STORAGE_IMAGE => ctx.features.storage_image_array_dynamic_indexing,
            _ => true,
        };
        let required_descriptors = layout_bindings
            .iter()
            .filter(|candidate| {
                vk::DescriptorType::from_raw(candidate.ty as i32) == descriptor_type
            })
            .fold(0u32, |total, candidate| {
                total.saturating_add(candidate.count)
            });
        let arrays_supported = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => {
                ctx.features.sampled_descriptor_arrays(required_descriptors)
            }
            vk::DescriptorType::STORAGE_IMAGE => {
                ctx.features.storage_descriptor_arrays(required_descriptors)
            }
            _ => true,
        };
        let descriptor_limit = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => ctx.features.sampled_image_descriptor_limit,
            vk::DescriptorType::STORAGE_IMAGE => ctx.features.storage_image_descriptor_limit,
            _ => u32::MAX,
        };
        if !arrays_supported {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::DescriptorArrayUnsupported {
                    binding: binding.binding,
                    count: binding.count,
                    required_descriptors,
                    descriptor_limit,
                    partially_bound: ctx.features.descriptor_binding_partially_bound,
                    dynamic_indexing,
                },
            ));
        }
    }

    if let Some(binding) = used_binding_absent_from_layout(&req.spirv, &layout_bindings) {
        return Err(DrawError::ComputeExecution(
            ComputeExecutionDecline::UsedBindingAbsentFromLayout { binding },
        ));
    }

    let layout_key = LayoutKey {
        bindings: layout_bindings,
        push_constant: req.dispatch.push_constant_range(),
    };

    let (spirv_digest, module) = caches.get_or_create_shader(ctx, &req.spirv, counters, pools)?;
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let shader_source = super::caches::ShaderModuleSource {
        module,
        spirv: &req.spirv,
    };
    // One pipeline per region workgroup size. A whole-workgroup module baked
    // its local size and needs exactly one; an exact-thread launch needs the
    // interior size plus whichever boundary sizes its grid produced. The cache
    // is content-keyed on that size, so a steady dispatch shape pays no create
    // after its first launch. `get_or_create_compute_pipeline` counts each hit.
    let mut dispatch_steps: Vec<DispatchStep> = Vec::new();
    match &req.dispatch {
        ComputeDispatch::Workgroups(grid) => {
            let cpipe_key = ComputePipelineKey {
                spirv: spirv_digest,
                entry: req.entry.clone(),
                layout: layout_key.clone(),
                local_size: None,
            };
            dispatch_steps.push(DispatchStep {
                pipeline: caches.get_or_create_compute_pipeline(
                    ctx,
                    &cpipe_key,
                    shader_source,
                    pipeline_layout,
                    counters,
                    pools,
                )?,
                group_count: *grid,
                push: None,
            });
        }
        ComputeDispatch::Regions {
            push_offset,
            regions,
            ..
        } => {
            dispatch_steps.reserve(regions.len());
            for region in regions {
                let cpipe_key = ComputePipelineKey {
                    spirv: spirv_digest,
                    entry: req.entry.clone(),
                    layout: layout_key.clone(),
                    local_size: Some(region.local_size),
                };
                dispatch_steps.push(DispatchStep {
                    pipeline: caches.get_or_create_compute_pipeline(
                        ctx,
                        &cpipe_key,
                        shader_source,
                        pipeline_layout,
                        counters,
                        pools,
                    )?,
                    group_count: region.group_count,
                    push: Some((*push_offset, region.push_constants)),
                });
            }
        }
    }

    // Storage buffers: host-visible staging used as SSBOs (same as draw path).
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let slot = pools.acquire_staging(
            ctx,
            resource.bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            counters,
        )?;
        pools.write_staging(ctx, &slot, &resource.bytes)?;
        storage_slots.push((
            resource.binding,
            slot,
            resource.bytes.len(),
            resource.writable,
        ));
    }

    // Sampled images: device-local + staging seed upload — or a device-local
    // copy from a resident storage image (copy-on-sample: the transient never
    // aliases the live resident, so the same dispatch may storage-write it).
    let mut sampled_slots = Vec::new();
    for resource in &req.sampled_images {
        // Ahead of the pooled transient, because this source does not want one.
        // A multisample image cannot be filled by an upload or a copy, so
        // acquiring a transient for it would allocate a single-sample image
        // this dispatch then binds instead of the samples the guest rendered.
        if let ComputeSampledSource::MultisampleTarget(identity) = &resource.source {
            // Reading a resident is using it, and the mark goes ahead of the
            // lookup so the refusals below cannot skip it — the render rail's
            // `SampledSource::Target` arm states the reason: a resident whose
            // content is not ready yet is still one the guest is actively
            // sampling, and aging it out between two attempts turns a
            // recoverable not-ready into a permanent missing.
            pools.registry_note_sampled_use(identity);
            let held = pools.registry_get(identity).map(|slot| {
                (
                    slot.image,
                    slot.access,
                    slot.content_ready,
                    slot.width,
                    slot.height,
                    slot.sample_count,
                    slot.memory.is_guest_imported(),
                )
            });
            let Some((image, access, content_ready, width, height, samples, host_accessible)) =
                held
            else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::MultisampleSampleAbsent {
                        binding: resource.binding,
                        identity: identity.clone(),
                        prior: pools.prior_reclaim(identity),
                    },
                ));
            };
            // Each of the three is a distinct loss and none substitutes for
            // another: nothing has been rendered yet, the samples are not the
            // ones this bind names, or the resident is single-sample and would
            // be a descriptor-type mismatch against the shader's declared
            // multisampled image.
            if !content_ready
                || width != resource.width
                || height != resource.height
                || samples <= 1
            {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::MultisampleSampleUnusable {
                        binding: resource.binding,
                        identity: identity.clone(),
                        content_ready,
                        resident_width: width,
                        resident_height: height,
                        resident_samples: samples,
                        resource_width: resource.width,
                        resource_height: resource.height,
                    },
                ));
            }
            // The view's format comes from the binding's own declared format,
            // reduced to the `VkFormat` the registry stores — the same
            // reduction `StorageImageFormat` already carries, so the descriptor
            // and the image cannot disagree about how the texels are read.
            let Some(view) = (unsafe {
                pools.registry_sample_view(ctx, identity, resource.format.vk_format(), counters)?
            }) else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::MultisampleSampleAbsent {
                        binding: resource.binding,
                        identity: identity.clone(),
                        prior: pools.prior_reclaim(identity),
                    },
                ));
            };
            sampled_slots.push(PreparedSampledImage::MultisampleTarget {
                binding: resource.binding,
                array_element: resource.array_element,
                identity: identity.clone(),
                image,
                view,
                access,
                next_access: super::pools::ResidentAccess::shader_read(host_accessible),
            });
            continue;
        }
        let key = StorageImageKey {
            width: resource.width,
            height: resource.height,
            format: resource.format,
            sampled_only: true,
            mip_levels: resource.mip_levels.max(1),
        };
        let img = pools.acquire_storage_image(ctx, key, counters)?;
        // The byte weight of this binding, derived from its own geometry rather
        // than carried beside it. Both arms below want it, and the upload arm's
        // `bytes` is required to equal it by the request validation above.
        let staged_bytes = crate::contract::extent::tight_pyramid_bytes(
            resource.width,
            resource.height,
            resource.mip_levels.max(1),
            resource.format.bytes_per_texel(),
        )
        .unwrap_or(0) as u64;
        let resident_copy = match &resource.source {
            ComputeSampledSource::ResidentCopy(bind) => Some(*bind),
            ComputeSampledSource::Bytes(_) => None,
            // Returned above.
            ComputeSampledSource::MultisampleTarget(_) => unreachable!(),
        };
        if resident_copy.is_some() && resource.mip_levels > 1 {
            // A resident is one window at one level. Seeding a pyramid's base
            // from it would leave every level above it empty, which reads as a
            // texture whose upper levels were never written — the exact defect
            // the pyramid is here to repair.
            return Err(DrawError::ComputeExecution(
                ComputeExecutionDecline::ResidentSampleIsNotAPyramid {
                    binding: resource.binding,
                    mip_levels: resource.mip_levels,
                },
            ));
        }
        let (upload, resident_src) = if let Some(bind) = resident_copy {
            // The caller skipped the guest read; nothing may be uploaded here.
            // Every mismatch names the check that refused.
            let Some((src_image, src_key, generation, src_access)) =
                pools.compute_resident_snapshot(&bind.identity)
            else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSampleAbsent {
                        binding: resource.binding,
                        identity: bind.identity,
                        width: resource.width,
                        height: resource.height,
                    },
                ));
            };
            if generation != bind.generation {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSampleGenerationMismatch {
                        binding: resource.binding,
                        identity: bind.identity,
                        actual_generation: generation,
                        expected_generation: bind.generation,
                    },
                ));
            }
            // The source must be byte-identical to the view; anything else is
            // a shape loss the runtime cannot have produced.
            resident_sample_exact(resource, bind, src_key)?;
            counters.note_compute_sampled_resident_copy(staged_bytes);
            (None, Some((src_image, src_access)))
        } else {
            let ComputeSampledSource::Bytes(bytes) = &resource.source else {
                unreachable!("the resident and multisample sources are handled above")
            };
            let st = pools.acquire_staging(
                ctx,
                bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?;
            pools.write_staging(ctx, &st, bytes)?;
            counters.note_compute_sampled_upload(bytes.len() as u64);
            (Some(st), None)
        };
        sampled_slots.push(PreparedSampledImage::Staged {
            binding: resource.binding,
            array_element: resource.array_element,
            img,
            upload,
            resident_src,
            width: resource.width,
            height: resource.height,
            mip_levels: resource.mip_levels.max(1),
        });
    }

    let mut sampler_handles = Vec::new();
    for sampler in &req.samplers {
        let handle = caches.get_or_create_sampler(ctx, &sampler.state_key(), counters, pools)?;
        sampler_handles.push((sampler.binding, handle));
    }

    // Storage images: device-local + staging seed upload + readback buffer.
    let mut simg_slots = Vec::new();
    for resource in &req.storage_images {
        let key = StorageImageKey {
            width: resource.width,
            height: resource.height,
            format: resource.format,
            sampled_only: false,
            // A compute write names one level, so a storage image is one level.
            mip_levels: 1,
        };
        let (img, initial_access, generation_match) = if let Some(residency) = resource.residency {
            let resident = pools.acquire_resident_storage_image(
                ctx,
                residency.identity,
                key,
                residency.seed_generation,
                counters,
            )?;
            (resident.slot, resident.access, resident.generation_match)
        } else {
            (
                pools.acquire_storage_image(ctx, key, counters)?,
                super::pools::ResidentAccess::Untouched,
                false,
            )
        };
        if resource.seed_skipped {
            let Some(residency) = resource.residency else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::SeedSkippedWithoutResidency {
                        binding: resource.binding,
                        width: resource.width,
                        height: resource.height,
                    },
                ));
            };
            if !generation_match {
                // The caller verified the resident generation at stage time
                // and skipped the guest read; seeding the zero placeholder now
                // would silently corrupt the chain. Named failure instead.
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSeedGenerationLost {
                        binding: resource.binding,
                        identity: residency.identity,
                        expected_generation: residency.seed_generation,
                    },
                ));
            }
        }
        let st = if generation_match {
            None
        } else {
            let staging = pools.acquire_staging(
                ctx,
                resource.bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?;
            pools.write_staging(ctx, &staging, &resource.bytes)?;
            counters.note_compute_storage_seed_upload(resource.bytes.len() as u64);
            Some(staging)
        };
        // Where the output goes is the caller's decision, not this rail's: a
        // request that named guest pages licensed them first, and one that did
        // not gets the pooled readback and the device→host crossing with it.
        let dst = match &resource.destination {
            super::types::ComputeImageDestination::Host => ComputeImageDst::Readback(
                pools.acquire_readback_extra(ctx, resource.bytes.len() as u64, counters)?,
            ),
            super::types::ComputeImageDestination::GuestPages { target, .. } => {
                ComputeImageDst::Direct(unsafe {
                    super::plan_guest_copy(ctx, pools, counters, target)?
                })
            }
        };
        simg_slots.push(PreparedStorageImage {
            binding: resource.binding,
            array_element: resource.array_element,
            slot: img,
            seed: st,
            dst,
            len: resource.bytes.len(),
            width: resource.width,
            height: resource.height,
            initial_access,
            residency: resource.residency,
        });
    }

    let push_descriptors = layout_key.uses_push_descriptors(ctx.caps.push_descriptor);
    // Owning pool block travels with an allocated set for a correctly-routed
    // free. A push layout records its writes into the command buffer instead.
    let mut dset_pool: Option<vk::DescriptorPool> = None;
    let dset = if dsl != vk::DescriptorSetLayout::null() && !push_descriptors {
        let (dset, pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;
        dset_pool = Some(pool);
        Some(dset)
    } else {
        None
    };
    let buffer_infos: Vec<_> = storage_slots
        .iter()
        .map(|(_, s, len, _)| {
            vk::DescriptorBufferInfo::default()
                .buffer(s.buffer)
                .offset(0)
                .range(super::exec::descriptor_range(*len as u64))
        })
        .collect();
    let sampled_infos: Vec<_> = sampled_slots
        .iter()
        .map(|prepared| {
            let (view, layout) = prepared.descriptor_view();
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout)
        })
        .collect();
    let sampler_infos: Vec<_> = sampler_handles
        .iter()
        .map(|(_, sampler)| vk::DescriptorImageInfo::default().sampler(*sampler))
        .collect();
    let image_infos: Vec<_> = simg_slots
        .iter()
        .map(|prepared| {
            vk::DescriptorImageInfo::default()
                .image_view(prepared.slot.view)
                .image_layout(vk::ImageLayout::GENERAL)
        })
        .collect();
    let dst_set = dset.unwrap_or_default();
    let mut descriptor_writes = Vec::new();
    for (i, (binding, _, _, _)) in storage_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i])),
        );
    }
    for (i, prepared) in sampled_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(prepared.binding())
                .dst_array_element(prepared.array_element())
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&sampled_infos[i])),
        );
    }
    for (i, (binding, _)) in sampler_handles.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&sampler_infos[i])),
        );
    }
    for (i, prepared) in simg_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(prepared.binding)
                .dst_array_element(prepared.array_element)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[i])),
        );
    }
    if dset.is_some() {
        ctx.device.update_descriptor_sets(&descriptor_writes, &[]);
        counters
            .descriptor_set_updates
            .fetch_add(1, Ordering::Relaxed);
    }

    // The ring slot's CB retired at begin_entry and its fence is unsignaled —
    // no pre-record wait remains (pre_record_wait_us stays 0 on this path).
    unsafe {
        pools.begin_slot_recording(
            ctx,
            cb,
            super::gpu_span::Kind::Compute,
            VkOp::ComputeExecResetCb,
            VkOp::ComputeExecBeginCb,
        )?
    };

    // Seed sampled images (staging upload or resident device copy)
    // → SHADER_READ_ONLY_OPTIMAL.
    for prepared in &sampled_slots {
        let PreparedSampledImage::Staged {
            binding,
            img,
            upload,
            resident_src,
            width,
            height,
            mip_levels,
            ..
        } = prepared
        else {
            // A multisample target is bound where it already lives; it is
            // placed for the read below, not seeded here.
            continue;
        };
        let (binding, width, height, mip_levels) = (*binding, *width, *height, *mip_levels);
        // Every level of the pyramid, not just the base: a level left in
        // `UNDEFINED` reads as a level nothing ever wrote.
        let range = super::color_subresource_range_levels(mip_levels);
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(img.image)
            .subresource_range(range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        if let Some(st) = upload {
            // One region per level, at the offset the *producer* packed it at.
            // Both ends read `tight_pyramid_spans`, so neither computes a
            // layout of its own and the two cannot drift.
            let Some(spans) = crate::contract::extent::tight_pyramid_spans(
                width,
                height,
                mip_levels,
                img.key.format.bytes_per_texel(),
            ) else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::SampledPyramidLayout {
                        binding,
                        width,
                        height,
                        mip_levels,
                    },
                ));
            };
            let copy: Vec<_> = spans
                .iter()
                .map(|span| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(span.offset as u64)
                        .image_subresource(super::color_subresource_layers().mip_level(span.level))
                        .image_extent(vk::Extent3D {
                            width: span.width,
                            height: span.height,
                            depth: 1,
                        })
                })
                .collect();
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                st.buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
        } else if let Some((src_image, src_access)) = *resident_src {
            // Copy-on-sample. The resident stays in its registry layout on
            // exit so the storage-acquire's captured initial_access (and the
            // storage pre-dispatch barrier, which syncs on TRANSFER when that
            // layout is TRANSFER_SRC_OPTIMAL) remains truthful.
            // Both halves — that the barrier is unconditional and that its scope
            // names what last *touched* the image rather than where it sits —
            // are `barrier_resident_for_transfer_read`'s to answer, and this
            // site had each of them wrong once.
            let next_access = super::pools::ResidentAccess::transfer_read(false);
            super::exec::barrier_resident_for_transfer_read(
                &ctx.device,
                cb,
                src_image,
                src_access,
                next_access,
            );
            let copy = [vk::ImageCopy::default()
                .src_subresource(super::color_subresource_layers())
                .dst_subresource(super::color_subresource_layers())
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_image(
                cb,
                src_image,
                next_access.layout(),
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
            if src_access.layout() != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                let restore = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(src_access.layout())
                    .image(src_image)
                    // The *source* resident's range, which is one level —
                    // `range` above describes this binding's own pyramid and
                    // naming it here would transition levels `src_image` has
                    // not got.
                    .subresource_range(super::color_subresource_range())];
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &restore,
                );
            }
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(img.image)
            .subresource_range(range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Upload storage-image misses, or transition a generation-matched resident
    // image directly from the prior readback layout into GENERAL.
    for prepared in &simg_slots {
        let img = &prepared.slot;
        let range = super::color_subresource_range();
        let (src_stage, src_access) = prepared.initial_access.source_scope();
        if let Some(st) = &prepared.seed {
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(prepared.initial_access.layout())
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(img.image)
                .subresource_range(range)];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            let copy = [vk::BufferImageCopy::default()
                .image_subresource(super::color_subresource_layers())
                .image_extent(vk::Extent3D {
                    width: prepared.width,
                    height: prepared.height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                st.buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
        }
        let old_layout = if prepared.seed.is_some() {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else {
            prepared.initial_access.layout()
        };
        let old_access = if prepared.seed.is_some() {
            vk::AccessFlags::TRANSFER_WRITE
        } else {
            src_access
        };
        let old_stage = if prepared.seed.is_some() {
            vk::PipelineStageFlags::TRANSFER
        } else {
            src_stage
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(old_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(img.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            old_stage,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Host-written SSBOs visible to compute.
    if !storage_slots.is_empty() {
        let buf_barriers: Vec<_> = storage_slots
            .iter()
            .map(|(_, s, _, writable)| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(if *writable {
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                    } else {
                        vk::AccessFlags::SHADER_READ
                    })
                    .buffer(s.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &buf_barriers,
            &[],
        );
    }

    // Place every multisample target this dispatch reads, and note where it
    // left them. These are not seeded above — the loop skips them — so this is
    // the only transition they get, and it is owed for the same reason the
    // render rail's `PreparedSampled::Resident` arm owes one: the resident's
    // last touch was a colour write by another submission.
    for prepared in &sampled_slots {
        let PreparedSampledImage::MultisampleTarget {
            identity,
            image,
            access,
            next_access,
            ..
        } = prepared
        else {
            continue;
        };
        if access != next_access {
            let (src_stage, src_access) = access.source_scope();
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(access.layout())
                .new_layout(next_access.layout())
                .image(*image)
                .subresource_range(super::color_subresource_range())];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        // After the barrier is recorded, so the registry never claims a layout
        // this command buffer has not been told to place the image in.
        pools.registry_note_access(identity, *next_access);
    }

    if push_descriptors {
        ctx.push_descriptor
            .as_ref()
            .expect("push layout requires enabled entry points")
            .cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &descriptor_writes,
            );
        counters.descriptor_pushes.fetch_add(1, Ordering::Relaxed);
    } else if let Some(dset) = dset {
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[dset],
            &[],
        );
        counters
            .descriptor_set_binds
            .fetch_add(1, Ordering::Relaxed);
    }
    // Descriptors are bound once for the whole launch: every region pipeline
    // shares this exact layout object, so a pipeline bind between them does not
    // disturb the set. No barrier separates the regions either — they are one
    // Metal `dispatchThreads`, whose threads have no ordering among themselves,
    // and consecutive Vulkan dispatches without a barrier carry that same
    // permission to overlap.
    for step in &dispatch_steps {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, step.pipeline);
        if let Some((offset, payload)) = &step.push {
            let bytes =
                std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), size_of_val(payload));
            ctx.device.cmd_push_constants(
                cb,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                *offset,
                bytes,
            );
        }
        ctx.device.cmd_dispatch(
            cb,
            step.group_count[0],
            step.group_count[1],
            step.group_count[2],
        );
    }

    // SSBO → host
    if storage_slots.iter().any(|(_, _, _, writable)| *writable) {
        let buf_barriers: Vec<_> = storage_slots
            .iter()
            .filter(|(_, _, _, writable)| *writable)
            .map(|(_, s, _, _)| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .buffer(s.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &buf_barriers,
            &[],
        );
    }

    // Storage images → readback buffers
    for prepared in &simg_slots {
        let img = &prepared.slot;
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(img.image)
            .subresource_range(super::color_subresource_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        match &prepared.dst {
            ComputeImageDst::Readback(slot) => {
                // The pooled readback is always tightly packed from texel zero.
                // A guest window's own offset and row stride belong to the
                // plan on the direct arm, never to this one.
                let copy = [vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .image_subresource(super::color_subresource_layers())
                    .image_extent(vk::Extent3D {
                        width: prepared.width,
                        height: prepared.height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_image_to_buffer(
                    cb,
                    img.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    slot.buffer,
                    &copy,
                );
            }
            ComputeImageDst::Direct(plan) => {
                // Recorded into the dispatch's own command buffer, so the whole
                // thing is still one submission and one fence. Both calls are
                // the render rail's, unchanged: the plan already describes
                // every guest run, and the release is what makes the bytes
                // visible to the guest's vCPU once the fence signals.
                unsafe {
                    super::record_guest_copy_plan(
                        ctx,
                        pools,
                        cb,
                        img.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        plan,
                    );
                    super::release_guest_copy_to_host(ctx, cb, plan);
                }
            }
        }
    }
    // Only a readback owes the host-visibility barrier below; the direct arm
    // released its own writes to `HOST` per plan, right where it recorded them.
    if simg_slots
        .iter()
        .any(|prepared| matches!(prepared.dst, ComputeImageDst::Readback(_)))
    {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    unsafe { pools.gpu_span_seal_current(ctx, cb) };
    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ComputeExecEndCb, e)))?;

    if force_loss {
        if let (Some(ds), Some(pool)) = (dset, dset_pool) {
            pools.free_descriptor_sets(&ctx.device, &[(ds, pool)]);
        }
        pools.recycle_staging();
        pools.recycle_readback();
        pools.recycle_storage_images();
        return Err(DrawError::DeviceLost(DeviceLostDecline::ForcedCompute));
    }

    let cbs = [cb];
    match ctx.submit_guest_work(&cbs, fence) {
        Ok(()) => {}
        Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
            return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                op: DeviceLostOp::ComputeSubmit,
                result: e,
            }));
        }
        Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::ComputeExecSubmit, e))),
    }

    // The copy into guest pages is on the queue now, so the debt is owed from
    // here — before any fallible step below, because a failure past the submit
    // does not un-submit the copy and the pages are being written either way.
    //
    // Read straight off the request rather than off `simg_slots`, so nothing
    // depends on the two being the same length in the same order — and so the
    // destination and the residency that decide the source are read off one
    // record, which is the only way they cannot disagree.
    //
    // Which source applies is a fact about where the image lives. A transient
    // slot is sealed into this submission's ring entry and cannot be recycled
    // before the fence retires, so the ring is its lifetime. A registered
    // resident was popped out of that live set at acquire and lives in the
    // compute-storage registry instead, so it needs the pin.
    for resource in &req.storage_images {
        if let super::types::ComputeImageDestination::GuestPages { pages, .. } =
            &resource.destination
        {
            let source = match &resource.residency {
                Some(residency) => super::GuestWriteSource::ResidentStorage(&residency.identity),
                None => super::GuestWriteSource::RingEntry,
            };
            super::record_guest_write_debt(pools, source, pages);
        }
    }

    // A dispatch whose every output stays on the GPU (deferred storage-image
    // writebacks, no writable SSBO readbacks, no direct guest-window DMA) has
    // nothing to hand the CPU — skip the post-submit fence wait and return
    // while the GPU still runs. Ordering stays intact everywhere: every user
    // of the shared fence/CB waits it before reuse, the deferred flush
    // (read_resident_storage) waits it before copying, and the owed
    // descriptor-set/pool cleanup is stashed until a later wait proves the CB
    // retired (drain_pending_compute_cleanup).
    // The test is whether anything owes the *CPU* bytes, not whether there were
    // storage images: a direct image lands in the guest's own pages and there is
    // nothing to read back, so it belongs on the deferred side exactly as a
    // read-only SSBO does. Its ordering does not come from this wait — see
    // `ComputeImageDestination::GuestPages` for the stamp chain that carries it.
    let all_writeback_deferred = storage_slots.iter().all(|(_, _, _, writable)| !writable)
        && simg_slots
            .iter()
            .all(|prepared| !matches!(prepared.dst, ComputeImageDst::Readback(_)));
    // Park the owed cleanup (descriptor set + transient pool slots) on this
    // ring slot in every mode; whichever entry retires the slot drains it. A
    // failed wait below leaves the slot pending, so no path ever reuses an
    // unretired fence. The readback maps below stay valid: the BufferSlot
    // handles are held by value and nothing else runs under the engine lock.
    let sealed = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), Vec::new());
    pools.finish_entry_async(&ctx.device, sealed);

    if all_writeback_deferred {
        counters
            .compute_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        pools.retire_all(ctx, counters)?;
    }

    for prepared in &simg_slots {
        if let Some(residency) = prepared.residency {
            pools.mark_resident_storage_image(&residency.identity, residency.output_generation);
        }
    }

    let mut buffers = Vec::with_capacity(
        storage_slots
            .iter()
            .filter(|(_, _, _, writable)| *writable)
            .count(),
    );
    for (binding, slot, len, writable) in &storage_slots {
        if !writable {
            continue;
        }
        let out = crate::backend::vulkan::engine::pools::read_back_slot(
            ctx,
            slot,
            *len as u64,
            VkOp::ComputeExecMapStorageReadback,
            VkOp::ComputeExecInvalidateStorageReadback,
        )?;
        counters.note_readback(*len as u64, super::counters::ReadbackSource::ComputeBuffer);
        buffers.push(ComputeBufferOutput {
            binding: *binding,
            bytes: out,
        });
    }
    let mut images = Vec::with_capacity(simg_slots.len());
    for prepared in &simg_slots {
        match &prepared.dst {
            ComputeImageDst::Readback(readback) => {
                let out = crate::backend::vulkan::engine::pools::read_back_slot(
                    ctx,
                    readback,
                    prepared.len as u64,
                    VkOp::ComputeExecMapImageReadback,
                    VkOp::ComputeExecInvalidateImageReadback,
                )?;
                counters.note_readback(
                    prepared.len as u64,
                    super::counters::ReadbackSource::ComputeImage,
                );
                images.push(super::types::ComputeImageResult::Bytes(out));
            }
            // Nothing was read, so nothing is charged to the readback census —
            // that is the saving this arm exists for, and a bump here would
            // report it as still being paid. `bytes` is what the queued copy
            // lands, for the caller's own census.
            ComputeImageDst::Direct(_) => {
                images.push(super::types::ComputeImageResult::Landed {
                    bytes: prepared.len as u64,
                });
            }
        }
    }

    // Cleanup was parked on the ring slot right after submit; nothing left
    // to free here (cleanup_us stays 0 on this path).

    counters
        .dispatches
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    Ok(ComputeOutput { buffers, images })
}

/// Copy a completely initialized mapped Vulkan output without first touching
/// every destination page with a redundant zero fill. The caller supplies an
/// exact readable `len`-byte mapping and the copy initializes the entire Vec
/// capacity before its length becomes visible.
///
/// # Safety
///
/// `ptr` must reference a readable `len`-byte mapping for the duration of this
/// call.
pub(super) unsafe fn copy_mapped_output(ptr: *const u8, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    out.set_len(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::{
        ComputeResidentSampleBind, ComputeSampledImageResource, ComputeSampledSource,
        ComputeStorageImageResource, SamplerResource, StorageImageFormat,
    };
    use crate::model::ComputeStorageResidencyKey;
    use crate::observe::Decline;

    /// A dispatch whose layout omits a binding its module samples is refused by
    /// number, and one that only omits an unreferenced binding is not.
    ///
    /// This is the shape that killed a host: Mesa's Intel driver divides
    /// `(use_count << 7)` by the absent binding's `array_size`, which its own
    /// zero-fill made zero, so `vkCreateComputePipelines` raises `SIGFPE`
    /// instead of returning. Refusing one dispatch is the only outcome that
    /// keeps the VM alive, and the second half of this test is what keeps the
    /// refusal from swallowing legal dispatches with it.
    #[test]
    fn a_used_binding_the_layout_omits_is_refused_and_a_declared_unused_one_is_not() {
        let spirv = crate::runtime::spirv_bind::test_module_with_two_sampled_images(33, 34);
        let sig = |binding| BindingSig {
            binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        };

        assert_eq!(used_binding_absent_from_layout(&spirv, &[]), Some(33));
        assert_eq!(
            used_binding_absent_from_layout(&spirv, &[sig(34)]),
            Some(33)
        );
        // Covering the used binding is sufficient: 34 is declared and never
        // referenced, so leaving it out is legal and must not be reported.
        assert_eq!(used_binding_absent_from_layout(&spirv, &[sig(33)]), None);
        assert_eq!(
            used_binding_absent_from_layout(&spirv, &[sig(33), sig(34)]),
            None
        );
    }

    fn residency_identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: 8,
            surface_offset: 0,
            surface_bpr: 4,
            span_end: 4,
            width: 1,
            height: 1,
            pixel_format: 80,
            texture_ref: 0,
        }
    }

    fn resident_sample_resource() -> ComputeSampledImageResource {
        ComputeSampledImageResource {
            mip_levels: 1,
            binding: 32,
            array_element: 0,
            descriptor_count: 1,
            format: StorageImageFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            source: ComputeSampledSource::ResidentCopy(ComputeResidentSampleBind {
                identity: residency_identity(),
                generation: 9,
            }),
        }
    }

    /// The bind a `ResidentCopy` resource names, for a test that asks about the
    /// shape check rather than about the source enum.
    fn resident_sample_bind(resource: &ComputeSampledImageResource) -> ComputeResidentSampleBind {
        match &resource.source {
            ComputeSampledSource::ResidentCopy(bind) => *bind,
            other => panic!("expected a resident-copy source, got {other:?}"),
        }
    }
    fn resident_sample_key() -> StorageImageKey {
        StorageImageKey {
            mip_levels: 1,
            width: 1,
            height: 1,
            format: StorageImageFormat::Rgba8Unorm,
            sampled_only: false,
        }
    }

    fn resident_sample_shape_slug(
        resource: &ComputeSampledImageResource,
        source: StorageImageKey,
    ) -> &'static str {
        let bind = resident_sample_bind(resource);
        match resident_sample_exact(resource, bind, source) {
            Err(DrawError::ComputeExecution(decline)) => decline.slug(),
            Err(other) => panic!("expected typed compute execution decline, got {other}"),
            Ok(()) => panic!("expected resident-sample shape refusal"),
        }
    }

    #[test]
    fn mapped_output_copy_initializes_exact_bytes_without_seed_buffer() {
        let source = [0x31, 0x00, 0x7f, 0xff, 0x42];
        let out = unsafe { copy_mapped_output(source.as_ptr(), source.len()) };
        assert_eq!(out, source);
    }

    #[test]
    fn compute_entry_with_interior_nul_is_rejected_before_cache_creation() {
        let req = ComputeRequest {
            spirv: vec![0x0723_0203],
            entry: "ma\0in".into(),
            dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
            ..Default::default()
        };
        let decline = match validate_compute(&req) {
            Err(DrawError::ComputeValidation(decline)) => decline,
            Err(other) => panic!("expected typed compute validation, got {other}"),
            Ok(()) => panic!("expected interior-NUL rejection"),
        };
        assert_eq!(decline.slug(), "vk_compute_validate_entry_interior_nul");
    }

    #[test]
    fn descriptor_array_elements_share_one_binding_without_being_duplicates() {
        let mut first = resident_sample_resource();
        first.descriptor_count = 8;
        let mut second = resident_sample_resource();
        second.array_element = 7;
        second.descriptor_count = 8;
        let request = ComputeRequest {
            spirv: vec![0x0723_0203],
            entry: "main".into(),
            dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
            sampled_images: vec![first, second],
            ..Default::default()
        };
        assert_eq!(validate_compute(&request), Ok(()));

        let mut out_of_range = request;
        out_of_range.sampled_images[1].array_element = 8;
        assert!(matches!(
            validate_compute(&out_of_range),
            Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledArrayElementOutOfRange {
                    binding: 32,
                    element: 8,
                    count: 8,
                }
            ))
        ));
    }

    #[test]
    fn resident_sample_shape_causes_are_not_collapsed() {
        let exact = resident_sample_resource();
        assert_eq!(
            resident_sample_exact(&exact, resident_sample_bind(&exact), resident_sample_key()),
            Ok(())
        );

        // Row-byte-identical but a different format and width. This used to be
        // accepted and reinterpreted through a buffer hop; it is now refused,
        // because only a disagreement between the registry and the mirror can
        // produce it.
        let mut row_compatible = resident_sample_resource();
        row_compatible.width = 2;
        row_compatible.format = StorageImageFormat::Rg8Unorm;
        assert_eq!(
            resident_sample_shape_slug(&row_compatible, resident_sample_key()),
            "vk_compute_exec_resident_sample_byte_shape_mismatch"
        );

        let mut byte_mismatch = resident_sample_resource();
        byte_mismatch.width = 2;
        assert_eq!(
            resident_sample_shape_slug(&byte_mismatch, resident_sample_key()),
            "vk_compute_exec_resident_sample_byte_shape_mismatch"
        );
    }

    #[test]
    fn sampled_and_storage_images_keep_distinct_descriptor_access() {
        let mut req = ComputeRequest {
            spirv: vec![0x0723_0203],
            entry: "main".into(),
            dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
            sampled_images: vec![ComputeSampledImageResource {
                mip_levels: 1,
                binding: 32,
                array_element: 0,
                descriptor_count: 1,
                format: StorageImageFormat::Rgba8Unorm,
                width: 1,
                height: 1,
                source: ComputeSampledSource::Bytes(vec![0; 4]),
            }],
            samplers: vec![SamplerResource::normalized_default(64)],
            storage_images: vec![ComputeStorageImageResource {
                destination: Default::default(),
                binding: 34,
                array_element: 0,
                descriptor_count: 1,
                format: StorageImageFormat::Rgba8Uint,
                width: 1,
                height: 1,
                bytes: vec![0; 4],
                residency: None,
                seed_skipped: false,
            }],
            ..Default::default()
        };
        assert!(validate_compute(&req).is_ok());

        req.storage_images[0].binding = 32;
        let decline = match validate_compute(&req) {
            Err(DrawError::ComputeValidation(decline)) => decline,
            Err(other) => panic!("expected typed compute validation, got {other}"),
            Ok(()) => panic!("expected descriptor collision"),
        };
        assert_eq!(
            decline.slug(),
            "vk_compute_validate_duplicate_storage_image_binding"
        );
    }
}
