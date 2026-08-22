//! Record / submit (bounded fence) / readback for one compute dispatch.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

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
    ComputeBufferOutput, ComputeBufferResult, ComputeOutput, ComputeRequest,
    ComputeResidentSampleBind, ComputeSampledImageResource, ComputeStorageResidency, DrawError,
};
use super::vk_call::{VkCall, VkOp};

struct PreparedStorageImage {
    binding: u32,
    array_element: u32,
    slot: StorageImageSlot,
    seed: Option<PreparedTexelSource>,
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

/// One prepared sampled input. A resident keeps the same image identity the
/// guest bound; only non-resident bytes need a transient upload image.
struct PreparedSampledImage {
    binding: u32,
    array_element: u32,
    image: vk::Image,
    view: vk::ImageView,
    upload: Option<PreparedTexelSource>,
    resident: Option<PreparedResidentSample>,
    width: u32,
    height: u32,
    null: bool,
}

#[derive(Clone, Copy)]
struct PreparedResidentSample {
    identity: reims_vgpu_core::ComputeStorageResidencyKey,
    initial_access: super::pools::ResidentAccess,
    /// The same semantic image is also a writable storage binding in this
    /// dispatch. Its storage preparation owns the one transition to GENERAL.
    also_storage: bool,
}

struct PreparedTexelSource {
    texels: super::exec::GuestTexels,
    row_length_texels: u32,
}

struct PreparedStorageBuffer {
    binding: u32,
    bound: super::exec::BoundBuffer,
    /// Present only when the descriptor is backed by a pool slot the host must
    /// read after dispatch. A direct guest bind has no host readback object.
    readback: Option<BufferSlot>,
    len: usize,
    writable: bool,
    /// Exact guest pages written by a direct imported descriptor.
    direct_write_pages: Option<Vec<u64>>,
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
    if req.program.id.get() == 0 {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::MissingProgram,
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
    if req.dispatch.counts.contains(&0) {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::ZeroGrid {
                grid: req.dispatch.counts,
            },
        ));
    }
    let mut bindings = BTreeSet::new();
    for b in &req.storage_buffers {
        if !bindings.insert((b.binding, 0)) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateStorageBufferBinding { binding: b.binding },
            ));
        }
        if b.backing.is_empty() {
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
        if !matches!(img.source, super::types::ComputeSampledImageSource::Null)
            && (img.width == 0 || img.height == 0)
        {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledZeroGeometry {
                    binding: img.binding,
                    width: img.width,
                    height: img.height,
                },
            ));
        }
        let tight = (img.width as usize)
            .saturating_mul(img.height as usize)
            .saturating_mul(img.format.bytes_per_texel());
        let valid = match &img.source {
            super::types::ComputeSampledImageSource::Null => true,
            super::types::ComputeSampledImageSource::Bytes(bytes) => bytes.len() == tight,
            super::types::ComputeSampledImageSource::Resident(_) => true,
            super::types::ComputeSampledImageSource::GuestPages(source) => {
                guest_image_source_is_exact(
                    source,
                    img.width,
                    img.height,
                    img.format.bytes_per_texel(),
                )
            }
        };
        if !valid {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledBytesLength {
                    binding: img.binding,
                    actual: match &img.source {
                        super::types::ComputeSampledImageSource::Null => 0,
                        super::types::ComputeSampledImageSource::Bytes(bytes) => bytes.len(),
                        super::types::ComputeSampledImageSource::GuestPages(source) => {
                            source.total_len as usize
                        }
                        super::types::ComputeSampledImageSource::Resident(_) => 0,
                    },
                    expected: tight,
                },
            ));
        }
    }
    for sampler in &req.samplers {
        if sampler.source == reims_vgpu_core::SamplerSource::Null {
            if !bindings.insert((sampler.binding, 0)) {
                return Err(DrawError::ComputeValidation(
                    ComputeValidationDecline::DuplicateSamplerBinding {
                        binding: sampler.binding,
                    },
                ));
            }
            continue;
        }
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
        let tight = (img.width as usize)
            .saturating_mul(img.height as usize)
            .saturating_mul(img.format.bytes_per_texel());
        let valid = match &img.seed {
            super::types::ComputeStorageImageSeed::Bytes(bytes) => bytes.len() == tight,
            super::types::ComputeStorageImageSeed::Resident => true,
            super::types::ComputeStorageImageSeed::GuestPages(source) => {
                guest_image_source_is_exact(
                    source,
                    img.width,
                    img.height,
                    img.format.bytes_per_texel(),
                )
            }
        };
        if !valid {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageBytesLength {
                    binding: img.binding,
                    actual: match &img.seed {
                        super::types::ComputeStorageImageSeed::Bytes(bytes) => bytes.len(),
                        super::types::ComputeStorageImageSeed::GuestPages(source) => {
                            source.total_len as usize
                        }
                        super::types::ComputeStorageImageSeed::Resident => 0,
                    },
                    expected: tight,
                },
            ));
        }
    }
    Ok(())
}

fn guest_image_source_is_exact(
    source: &super::types::GuestRunSource,
    width: u32,
    height: u32,
    bytes_per_texel: usize,
) -> bool {
    let tight_row = match (width as usize).checked_mul(bytes_per_texel) {
        Some(value) => value,
        None => return false,
    };
    let stride = if source.row_length_texels == 0 {
        tight_row
    } else {
        match (source.row_length_texels as usize).checked_mul(bytes_per_texel) {
            Some(value) => value,
            None => return false,
        }
    };
    let Some(expected) = (height.saturating_sub(1) as usize)
        .checked_mul(stride)
        .and_then(|prefix| prefix.checked_add(tight_row))
    else {
        return false;
    };
    let covered: u64 = source.runs.iter().map(|run| run.len).sum();
    stride >= tight_row
        && source.total_len == expected as u64
        && source
            .source_offset
            .checked_add(source.total_len)
            .is_some_and(|end| end <= covered)
}

#[derive(Clone, Copy)]
enum ComputeTexelRole {
    Sampled,
    StorageSeed,
}

unsafe fn prepare_compute_guest_texels(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    source: &super::types::GuestRunSource,
    role: ComputeTexelRole,
    gathers: &mut Vec<super::exec::PendingGuestGather>,
) -> Result<PreparedTexelSource, DrawError> {
    if let Some(texels) =
        super::exec::prepare_guest_texel_window(ctx, pools, counters, source, gathers)?
    {
        crate::telemetry::note_route(match role {
            ComputeTexelRole::Sampled => "compute_sampled_guest_pages",
            ComputeTexelRole::StorageSeed => "compute_storage_seed_guest_pages",
        });
        return Ok(PreparedTexelSource {
            texels,
            row_length_texels: source.row_length_texels,
        });
    }

    let slot = pools.acquire_staging(
        ctx,
        source.total_len,
        vk::BufferUsageFlags::TRANSFER_SRC,
        counters,
    )?;
    pools.write_staging_from_runs(
        ctx,
        &slot,
        &source.runs,
        source.source_offset,
        source.total_len,
    )?;
    match role {
        ComputeTexelRole::Sampled => {
            counters.note_compute_sampled_upload(source.total_len);
            crate::telemetry::note_route("compute_sampled_guest_cpu_fallback");
        }
        ComputeTexelRole::StorageSeed => {
            counters.note_compute_storage_seed_upload(source.total_len);
            crate::telemetry::note_route("compute_storage_seed_guest_cpu_fallback");
        }
    }
    Ok(PreparedTexelSource {
        texels: super::exec::GuestTexels::Scratch(slot),
        row_length_texels: source.row_length_texels,
    })
}

/// A resident sampled bind must name an image that is byte-for-byte the
/// image the view will sample: same vk format, same width, same height. The
/// runtime only ever binds the key it looked the mirror up under, so an
/// inexact source means the registry and the mirror disagree — refuse by name
/// rather than reinterpreting bytes the shader did not ask us to reinterpret.
fn resident_sample_exact(
    resource: &ComputeSampledImageResource,
    bind: ComputeResidentSampleBind,
    src_key: StorageImageKey,
) -> Result<(), DrawError> {
    let exact = crate::format::vk_storage_image(src_key.format)
        == crate::format::vk_storage_image(resource.format)
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
/// **Expected to return `None` always.** The two nullable classes are filled
/// before the request is built — `runtime::compute_exec` preserves an explicit
/// null descriptor for every binding of those classes the guest left empty —
/// so this is the backstop for a class those passes do not
/// cover, and a firing names a real gap rather than noise.
///
/// [`crate::spirv_bind::descriptor_static_use`] answers `NotDeclared`
/// for anything that is not a `UniformConstant` descriptor, so a storage buffer,
/// whose root this walk cannot resolve, is never refused on a guess.
fn used_binding_absent_from_layout(used: &[u32], layout: &[BindingSig]) -> Option<u32> {
    used.iter()
        .copied()
        .find(|binding| !layout.iter().any(|candidate| candidate.binding == *binding))
}

pub(crate) struct NativeComputeProgram {
    pub(crate) shader: Arc<crate::m2v_cache::ShaderVariant>,
}

pub(crate) unsafe fn execute_compute_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &ComputeRequest,
    program: &NativeComputeProgram,
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
        let populated = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => req
                .sampled_images
                .iter()
                .filter(|image| image.binding == binding.binding)
                .count() as u32,
            vk::DescriptorType::STORAGE_IMAGE => req
                .storage_images
                .iter()
                .filter(|image| image.binding == binding.binding)
                .count() as u32,
            _ => binding.count,
        };
        let unpopulated = binding.count.saturating_sub(populated);
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
            vk::DescriptorType::SAMPLED_IMAGE => ctx
                .features
                .sampled_descriptor_arrays(required_descriptors, unpopulated),
            vk::DescriptorType::STORAGE_IMAGE => ctx
                .features
                .storage_descriptor_arrays(required_descriptors, unpopulated),
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
                    unpopulated,
                    required_descriptors,
                    descriptor_limit,
                    partially_bound: ctx.features.descriptor_binding_partially_bound,
                    null_descriptor: ctx.features.null_descriptor,
                    dynamic_indexing,
                },
            ));
        }
    }

    if let Some(binding) =
        used_binding_absent_from_layout(&req.program.used_descriptor_bindings, &layout_bindings)
    {
        return Err(DrawError::ComputeExecution(
            ComputeExecutionDecline::UsedBindingAbsentFromLayout { binding },
        ));
    }

    // A module that reads push constants under a layout exposing none culls
    // every invocation and reports nothing. Reflection is the authority on
    // where the grid sits; this only refuses the case where the module wants
    // one and the prepared variant carries none.
    if program.shader.kernel_grid.is_none()
        && crate::spirv_bind::declares_push_constants(&program.shader.words)
    {
        return Err(DrawError::ComputeExecution(
            ComputeExecutionDecline::KernelGridRangeAbsent,
        ));
    }

    let layout_key = LayoutKey {
        bindings: layout_bindings,
        kernel_grid: program.shader.kernel_grid,
    };

    let (spirv_digest, module) =
        caches.get_or_create_shader(ctx, &program.shader.words, counters, pools)?;
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let cpipe_key = ComputePipelineKey {
        spirv: spirv_digest,
        entry: req.entry.clone(),
        layout: layout_key.clone(),
    };
    // One cache, consulted once; `get_or_create_compute_pipeline` counts the hit.
    let pipeline = caches.get_or_create_compute_pipeline(
        ctx,
        &cpipe_key,
        super::caches::ShaderModuleSource {
            module,
            spirv: &program.shader.words,
        },
        pipeline_layout,
        counters,
        pools,
    )?;

    // Storage buffers: bind the retained guest allocation when possible. A
    // host-owned source or an import decline retains the host-visible fallback.
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let len = resource.backing.len();
        let (bound, readback, direct_write_pages) = match &resource.backing {
            super::types::ComputeBufferBacking::Bytes(bytes) => {
                let slot = pools.acquire_staging(
                    ctx,
                    bytes.len() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    counters,
                )?;
                pools.write_staging(ctx, &slot, bytes)?;
                (super::exec::BoundBuffer::from(slot), Some(slot), None)
            }
            super::types::ComputeBufferBacking::GuestPages {
                source,
                write_pages,
            } => {
                let direct =
                    unsafe { super::exec::import_guest_compute_buffer_window(ctx, pools, source) };
                if let Some(bound) = direct {
                    pools.note_guest_read_recorded();
                    counters.note_compute_buffer_guest_import(source.total_len);
                    (bound, None, resource.writable.then(|| write_pages.clone()))
                } else {
                    let slot = pools.acquire_staging(
                        ctx,
                        source.total_len,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        counters,
                    )?;
                    pools.write_staging_from_runs(
                        ctx,
                        &slot,
                        &source.runs,
                        source.source_offset,
                        source.total_len,
                    )?;
                    crate::telemetry::note_route("compute_buffer_guest_cpu_fallback");
                    (super::exec::BoundBuffer::from(slot), Some(slot), None)
                }
            }
        };
        storage_slots.push(PreparedStorageBuffer {
            binding: resource.binding,
            bound,
            readback,
            len,
            writable: resource.writable,
            direct_write_pages,
        });
    }

    // Sampled images: non-resident bytes use a transient upload. A resident is
    // the texture object the guest named, so sampling binds that same image;
    // resource binding never implies an unrequested snapshot copy.
    let mut guest_gathers = Vec::new();
    let mut sampled_slots = Vec::new();
    for resource in &req.sampled_images {
        if matches!(
            resource.source,
            super::types::ComputeSampledImageSource::Null
        ) {
            if !ctx.features.null_descriptor {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::NullSampledImageUnsupported {
                        binding: resource.binding,
                    },
                ));
            }
            sampled_slots.push(PreparedSampledImage {
                binding: resource.binding,
                array_element: resource.array_element,
                image: vk::Image::null(),
                view: vk::ImageView::null(),
                upload: None,
                resident: None,
                width: 0,
                height: 0,
                null: true,
            });
            continue;
        }
        let key = StorageImageKey {
            width: resource.width,
            height: resource.height,
            format: resource.format,
            sampled_only: true,
        };
        let (image, view, upload, resident) =
            if let super::types::ComputeSampledImageSource::Resident(bind) = &resource.source {
                let Some((src_image, src_view, src_key, generation, src_access)) =
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
                resident_sample_exact(resource, *bind, src_key)?;
                let bytes = resource.width as u64
                    * resource.height as u64
                    * resource.format.bytes_per_texel() as u64;
                counters.note_compute_sampled_resident_bind(bytes);
                let also_storage = req.storage_images.iter().any(|storage| {
                    storage
                        .residency
                        .is_some_and(|residency| residency.identity == bind.identity)
                });
                (
                    src_image,
                    src_view,
                    None,
                    Some(PreparedResidentSample {
                        identity: bind.identity,
                        initial_access: src_access,
                        also_storage,
                    }),
                )
            } else {
                let img = pools.acquire_storage_image(ctx, key, counters)?;
                let prepared = match &resource.source {
                    super::types::ComputeSampledImageSource::Bytes(bytes) => {
                        let st = pools.acquire_staging(
                            ctx,
                            bytes.len() as u64,
                            vk::BufferUsageFlags::TRANSFER_SRC,
                            counters,
                        )?;
                        pools.write_staging(ctx, &st, bytes)?;
                        counters.note_compute_sampled_upload(bytes.len() as u64);
                        PreparedTexelSource {
                            texels: super::exec::GuestTexels::Scratch(st),
                            row_length_texels: 0,
                        }
                    }
                    super::types::ComputeSampledImageSource::GuestPages(source) => unsafe {
                        prepare_compute_guest_texels(
                            ctx,
                            pools,
                            counters,
                            source,
                            ComputeTexelRole::Sampled,
                            &mut guest_gathers,
                        )?
                    },
                    super::types::ComputeSampledImageSource::Resident(_) => unreachable!(),
                    super::types::ComputeSampledImageSource::Null => unreachable!(),
                };
                (img.image, img.view, Some(prepared), None)
            };
        sampled_slots.push(PreparedSampledImage {
            binding: resource.binding,
            array_element: resource.array_element,
            image,
            view,
            upload,
            resident,
            width: resource.width,
            height: resource.height,
            null: false,
        });
    }

    let mut sampler_handles = Vec::new();
    for sampler in &req.samplers {
        let handle = if sampler.source == reims_vgpu_core::SamplerSource::Null {
            if !ctx.features.null_descriptor {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::NullSamplerUnsupported {
                        binding: sampler.binding,
                    },
                ));
            }
            vk::Sampler::null()
        } else {
            caches.get_or_create_sampler(
                ctx,
                &super::types::sampler_state_key(sampler),
                counters,
                pools,
            )?
        };
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
        if matches!(
            resource.seed,
            super::types::ComputeStorageImageSeed::Resident
        ) {
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
            Some(match &resource.seed {
                super::types::ComputeStorageImageSeed::Bytes(bytes) => {
                    let staging = pools.acquire_staging(
                        ctx,
                        bytes.len() as u64,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        counters,
                    )?;
                    pools.write_staging(ctx, &staging, bytes)?;
                    counters.note_compute_storage_seed_upload(bytes.len() as u64);
                    PreparedTexelSource {
                        texels: super::exec::GuestTexels::Scratch(staging),
                        row_length_texels: 0,
                    }
                }
                super::types::ComputeStorageImageSeed::GuestPages(source) => unsafe {
                    prepare_compute_guest_texels(
                        ctx,
                        pools,
                        counters,
                        source,
                        ComputeTexelRole::StorageSeed,
                        &mut guest_gathers,
                    )?
                },
                super::types::ComputeStorageImageSeed::Resident => {
                    return Err(DrawError::ComputeExecution(
                        ComputeExecutionDecline::ResidentSeedGenerationLost {
                            binding: resource.binding,
                            identity: resource.residency.expect("validated above").identity,
                            expected_generation: resource
                                .residency
                                .expect("validated above")
                                .seed_generation,
                        },
                    ));
                }
            })
        };
        // Where the output goes is the caller's decision, not this rail's: a
        // request that named guest pages licensed them first, and one that did
        // not gets the pooled readback and the device→host crossing with it.
        let dst = match &resource.destination {
            super::types::ComputeImageDestination::Host => {
                let len = resource.width as u64
                    * resource.height as u64
                    * resource.format.bytes_per_texel() as u64;
                ComputeImageDst::Readback(pools.acquire_readback_extra(ctx, len, counters)?)
            }
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
            len: (resource.width as usize)
                .saturating_mul(resource.height as usize)
                .saturating_mul(resource.format.bytes_per_texel()),
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
        .map(|prepared| {
            vk::DescriptorBufferInfo::default()
                .buffer(prepared.bound.buffer)
                .offset(prepared.bound.offset)
                .range(super::exec::descriptor_range(prepared.len as u64))
        })
        .collect();
    let sampled_infos: Vec<_> = sampled_slots
        .iter()
        .map(|prepared| {
            vk::DescriptorImageInfo::default()
                .image_view(prepared.view)
                .image_layout(if prepared.resident.is_some() {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                })
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
    for (i, prepared) in storage_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(prepared.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i])),
        );
    }
    for (i, prepared) in sampled_slots.iter().enumerate() {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(prepared.binding)
                .dst_array_element(prepared.array_element)
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

    let reads_imported_guest = storage_slots
        .iter()
        .any(|prepared| prepared.bound.guest_import)
        || !guest_gathers.is_empty()
        || sampled_slots.iter().any(|prepared| {
            prepared
                .upload
                .as_ref()
                .is_some_and(|source| source.texels.is_imported())
        })
        || simg_slots.iter().any(|prepared| {
            prepared
                .seed
                .as_ref()
                .is_some_and(|source| source.texels.is_imported())
        });
    if reads_imported_guest {
        let mut read_pages: Vec<Option<reims_vgpu_memory::GuestPageSet>> = Vec::new();
        for storage in &req.storage_buffers {
            if let super::types::ComputeBufferBacking::GuestPages { source, .. } = &storage.backing
            {
                read_pages.push(source.physical_pages.clone());
            }
        }
        for sampled in &req.sampled_images {
            if let super::types::ComputeSampledImageSource::GuestPages(source) = &sampled.source {
                read_pages.push(source.physical_pages.clone());
            }
        }
        for storage in &req.storage_images {
            if let super::types::ComputeStorageImageSeed::GuestPages(source) = &storage.seed {
                read_pages.push(source.physical_pages.clone());
            }
        }
        if let Some(visibility) = pools.imported_guest_barrier(cb, || {
            let visibility = super::exec::imported_guest_visibility(&read_pages);
            super::exec::note_imported_guest_visibility(counters, visibility);
            visibility
        }) {
            let barrier = [super::exec::imported_guest_read_barrier(
                vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
                visibility,
            )];
            ctx.device.cmd_pipeline_barrier(
                cb,
                super::exec::imported_guest_read_stage(visibility),
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    // Scattered guest windows are assembled inside this dispatch's command
    // buffer, before any image seed reads them. The barrier is the one relation
    // both sampled and storage seeds need: transfer writes to the gathered
    // buffer become transfer reads by the buffer-to-image copies below.
    for gather in &guest_gathers {
        for (source, regions) in &gather.sources {
            ctx.device.cmd_copy_buffer(cb, *source, gather.dst, regions);
        }
    }
    if !guest_gathers.is_empty() {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    // Seed transient sampled images → SHADER_READ_ONLY_OPTIMAL. Resident
    // samples keep their identity and enter GENERAL directly; when the same
    // image is also storage-bound, the storage loop owns that one transition.
    for prepared in &sampled_slots {
        if prepared.null {
            continue;
        }
        let range = super::color_subresource_range();
        if let Some(resident) = prepared.resident {
            if resident.also_storage {
                continue;
            }
            let (src_stage, src_access) = resident.initial_access.source_scope();
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(resident.initial_access.layout())
                .new_layout(vk::ImageLayout::GENERAL)
                .image(prepared.image)
                .subresource_range(range)];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            continue;
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(prepared.image)
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
        if let Some(source) = &prepared.upload {
            let copy = [vk::BufferImageCopy::default()
                .buffer_offset(source.texels.offset())
                .buffer_row_length(source.row_length_texels)
                .image_subresource(super::color_subresource_layers())
                .image_extent(vk::Extent3D {
                    width: prepared.width,
                    height: prepared.height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                source.texels.buffer(),
                prepared.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(prepared.image)
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
        if let Some(source) = &prepared.seed {
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
                .buffer_offset(source.texels.offset())
                .buffer_row_length(source.row_length_texels)
                .image_subresource(super::color_subresource_layers())
                .image_extent(vk::Extent3D {
                    width: prepared.width,
                    height: prepared.height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                source.texels.buffer(),
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
            .map(|prepared| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(super::exec::imported_guest_write_access())
                    .dst_access_mask(if prepared.writable {
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                    } else {
                        vk::AccessFlags::SHADER_READ
                    })
                    .buffer(prepared.bound.buffer)
                    .offset(prepared.bound.offset)
                    .size(prepared.len as u64)
            })
            .collect();
        ctx.device.cmd_pipeline_barrier(
            cb,
            super::exec::imported_guest_write_stage(),
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &buf_barriers,
            &[],
        );
    }

    ctx.device
        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
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
    // Metal's exact thread grid, for the cull in the translated entry point.
    //
    // `vkCmdDispatch` takes whole workgroups, so a `dispatchThreads` grid that
    // does not divide its threadgroup is rounded up and the excess invocations
    // launch. The translated kernel returns early for any invocation outside
    // the grid, and this is where it reads that grid — one push of three `u32`
    // immediately before the dispatch it describes, so the two cannot be
    // separated by a later binding. A kernel whose reflection declares no range
    // needs no cull and gets no push.
    if let Some(range) = program.shader.kernel_grid {
        ctx.device.cmd_push_constants(
            cb,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            range.offset,
            &crate::m2v_cache::KernelGridRange::bytes(req.dispatch.threads_per_grid),
        );
        counters.kernel_grid_pushes.fetch_add(1, Ordering::Relaxed);
    }
    let counts = req.dispatch.counts;
    ctx.device.cmd_dispatch(cb, counts[0], counts[1], counts[2]);

    // Writable SSBOs become host-visible. For a direct import this releases the
    // shader's writes to the guest allocation itself; a staging slot is mapped
    // after the fence as before.
    if storage_slots.iter().any(|prepared| prepared.writable) {
        let buf_barriers: Vec<_> = storage_slots
            .iter()
            .filter(|prepared| prepared.writable)
            .map(|prepared| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .buffer(prepared.bound.buffer)
                    .offset(prepared.bound.offset)
                    .size(prepared.len as u64)
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
    let submitted_timeline = match ctx.submit_guest_work(&cbs, fence) {
        Ok(timeline) => timeline,
        Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
            return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                op: DeviceLostOp::ComputeSubmit,
                result: e,
            }));
        }
        Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::ComputeExecSubmit, e))),
    };

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
            if let Some(pages) = reims_vgpu_memory::GuestWritePages::new(pages) {
                super::record_guest_write_debt(pools, source, &pages);
            }
        }
    }
    for prepared in &storage_slots {
        if let Some(pages) = &prepared.direct_write_pages {
            if let Some(pages) = reims_vgpu_memory::GuestWritePages::new(pages) {
                super::record_guest_write_debt(
                    pools,
                    super::GuestWriteSource::ImportedBuffer,
                    &pages,
                );
            }
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
    let all_writeback_deferred = storage_slots
        .iter()
        .all(|prepared| !prepared.writable || prepared.readback.is_none())
        && simg_slots
            .iter()
            .all(|prepared| !matches!(prepared.dst, ComputeImageDst::Readback(_)));
    // Park the owed cleanup (descriptor set + transient pool slots) on this
    // ring slot in every mode; whichever entry retires the slot drains it. A
    // failed wait below leaves the slot pending, so no path ever reuses an
    // unretired fence. The readback maps below stay valid: the BufferSlot
    // handles are held by value and nothing else runs under the engine lock.
    let sealed = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), Vec::new());
    pools.finish_entry_async(sealed, submitted_timeline, None);

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
    for prepared in &sampled_slots {
        if let Some(resident) = prepared.resident.filter(|resident| !resident.also_storage) {
            pools.mark_compute_resident_sampled(&resident.identity);
        }
    }

    let mut buffers = Vec::with_capacity(
        storage_slots
            .iter()
            .filter(|prepared| prepared.writable)
            .count(),
    );
    for prepared in &storage_slots {
        if !prepared.writable {
            continue;
        }
        let result = if let Some(slot) = &prepared.readback {
            let out = crate::engine::pools::read_back_slot(
                ctx,
                slot,
                prepared.len as u64,
                VkOp::ComputeExecMapStorageReadback,
                VkOp::ComputeExecInvalidateStorageReadback,
            )?;
            counters.note_readback(
                prepared.len as u64,
                super::counters::ReadbackSource::ComputeBuffer,
            );
            ComputeBufferResult::Bytes(out)
        } else {
            ComputeBufferResult::Landed {
                bytes: prepared.len as u64,
            }
        };
        buffers.push(ComputeBufferOutput {
            binding: prepared.binding,
            result,
        });
    }
    let mut images = Vec::with_capacity(simg_slots.len());
    for prepared in &simg_slots {
        match &prepared.dst {
            ComputeImageDst::Readback(readback) => {
                let out = crate::engine::pools::read_back_slot(
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
    use crate::engine::{
        ComputeResidentSampleBind, ComputeSampledImageResource, ComputeStorageImageResource,
        SamplerResource, StorageImageFormat,
    };
    use reims_vgpu_core::ComputeStorageResidencyKey;
    use reims_vgpu_observe::Decline;

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
        let sig = |binding| BindingSig {
            binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        };

        let used = [33];
        assert_eq!(used_binding_absent_from_layout(&used, &[]), Some(33));
        assert_eq!(used_binding_absent_from_layout(&used, &[sig(34)]), Some(33));
        // Covering the used binding is sufficient: 34 is declared and never
        // referenced, so leaving it out is legal and must not be reported.
        assert_eq!(used_binding_absent_from_layout(&used, &[sig(33)]), None);
        assert_eq!(
            used_binding_absent_from_layout(&used, &[sig(33), sig(34)]),
            None
        );
    }

    fn test_program() -> reims_vgpu_core::PreparedShaderStage {
        reims_vgpu_core::PreparedShaderStage {
            id: reims_vgpu_protocol::PreparedShaderId::new(1),
            ..Default::default()
        }
    }

    #[test]
    fn a_null_sampler_is_not_validated_as_invented_sampler_state() {
        let mut sampler = SamplerResource::null(64);
        sampler.lod_min = f32::NAN.to_bits();
        sampler.lod_max = f32::NEG_INFINITY.to_bits();
        let req = ComputeRequest {
            program: test_program(),
            entry: "main".into(),
            dispatch: reims_vgpu_protocol::dispatch::workgroup_counts([1, 1, 1], [1, 1, 1], false)
                .expect("a one-by-one dispatch is a valid grid"),
            samplers: vec![sampler],
            ..Default::default()
        };

        assert!(validate_compute(&req).is_ok());
    }

    fn residency_identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey::surface(7, 8, 0, 4, 4, 1, 1, 80)
    }

    fn resident_sample_resource() -> ComputeSampledImageResource {
        ComputeSampledImageResource {
            binding: 32,
            array_element: 0,
            descriptor_count: 1,
            format: StorageImageFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            source: super::super::types::ComputeSampledImageSource::Resident(
                ComputeResidentSampleBind {
                    identity: residency_identity(),
                    generation: 9,
                },
            ),
            content: None,
        }
    }
    fn resident_sample_key() -> StorageImageKey {
        StorageImageKey {
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
        let super::super::types::ComputeSampledImageSource::Resident(bind) = resource.source else {
            panic!("fixture must be resident")
        };
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
            program: test_program(),
            entry: "ma\0in".into(),
            dispatch: reims_vgpu_protocol::dispatch::workgroup_counts([1, 1, 1], [1, 1, 1], false)
                .expect("a one-by-one dispatch is a valid grid"),
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
            program: test_program(),
            entry: "main".into(),
            dispatch: reims_vgpu_protocol::dispatch::workgroup_counts([1, 1, 1], [1, 1, 1], false)
                .expect("a one-by-one dispatch is a valid grid"),
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
            resident_sample_exact(
                &exact,
                match exact.source {
                    super::super::types::ComputeSampledImageSource::Resident(bind) => bind,
                    _ => panic!("fixture must be resident"),
                },
                resident_sample_key(),
            ),
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
            program: test_program(),
            entry: "main".into(),
            dispatch: reims_vgpu_protocol::dispatch::workgroup_counts([1, 1, 1], [1, 1, 1], false)
                .expect("a one-by-one dispatch is a valid grid"),
            sampled_images: vec![ComputeSampledImageResource {
                binding: 32,
                array_element: 0,
                descriptor_count: 1,
                format: StorageImageFormat::Rgba8Unorm,
                width: 1,
                height: 1,
                source: super::super::types::ComputeSampledImageSource::Bytes(vec![0; 4]),
                content: None,
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
                seed: super::super::types::ComputeStorageImageSeed::Bytes(vec![0; 4]),
                residency: None,
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
