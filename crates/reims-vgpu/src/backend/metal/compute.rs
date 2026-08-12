//! Compute encode path: PSO cache, binds, dispatch core, reflection.

use crate::backend::blob::BlobKey;
use crate::backend::hash::hash_bytes;
use crate::backend::metal::abi::*;
use crate::backend::metal::cache::{
    compute_pso_insert, compute_pso_lookup, reflect_insert, reflect_lookup, ComputePsoKey,
};
use crate::backend::metal::constants::*;
use crate::backend::metal::format::storage_image_format;
use crate::backend::metal::function::load_only_function;
use crate::backend::metal::raw_metal::{
    command_buffer_error_description, mtl_size, new_compute_pso_with_function_reflection,
    new_texture_view_swizzled, reflection_bindings, set_buffer_with_attribute_stride,
    set_imageblock_width_height, set_stage_in_region, set_stage_in_region_indirect,
    texture_swizzle_channels, BINDING_ACCESS_READ_ONLY, BINDING_ACCESS_READ_WRITE,
    BINDING_ACCESS_WRITE_ONLY, BINDING_TYPE_TEXTURE,
};
use crate::backend::metal::runtime::{new_buffer_from_host, system_device, thread_queue};
use crate::backend::metal::samplers::make_explicit_sampler;
use crate::backend::metal::stage_input::{
    has_indexed_layout, layout_for_buffer, make_compute_stage_input_descriptor,
};
use crate::backend::metal::util::{
    bytes_of, clear_err, sampler_index, set_err, texture_index, valid_buffer_binding,
    valid_threadgroup_memory_index, ErrOut, Status,
};
use crate::contract::extent::{tight_image_bytes, Extent3};
use metal::*;
use std::ptr;

pub fn hash_compute_stage_input(stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>) -> u64 {
    match stage_input {
        None => 0,
        Some(s) => hash_bytes(bytes_of(s)),
    }
}

fn new_compute_pipeline_state_uncached(
    device: &Device,
    function: &Function,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    err: ErrOut<'_>,
) -> Result<ComputePipelineState, Status> {
    match stage_input {
        None => device
            .new_compute_pipeline_state_with_function(function)
            .map_err(|e| {
                set_err(err, format!("compute PSO failed: {e}"));
                Status::execute("metal_compute_pso_create_failed")
            }),
        Some(si) => {
            let stage_descriptor = make_compute_stage_input_descriptor(si, err)?;
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(function));
            descriptor.set_stage_input_descriptor(Some(&stage_descriptor));
            device.new_compute_pipeline_state(&descriptor).map_err(|e| {
                set_err(err, format!("compute PSO failed: {e}"));
                Status::execute("metal_compute_stage_input_pso_create_failed")
            })
        }
    }
}

pub fn new_compute_pipeline_state(
    device: &Device,
    function: &Function,
    mtlb: &[u8],
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    err: ErrOut<'_>,
) -> Result<ComputePipelineState, Status> {
    let key = ComputePsoKey {
        mtlb: BlobKey::new(mtlb),
        stage_hash: hash_compute_stage_input(stage_input),
        stage_input,
    };
    if let Some(hit) = compute_pso_lookup(&key) {
        return Ok(hit);
    }
    let pso = new_compute_pipeline_state_uncached(device, function, stage_input, err)?;
    Ok(compute_pso_insert(&key, pso))
}

fn compute_buffer_backing(buffer: &ReimsVgpuBuffer) -> Result<(*mut u8, usize, usize), Status> {
    if !buffer.backing_data.is_null() {
        if buffer.backing_len == 0 {
            return Err(
                Status::args("metal_compute_backing_length_zero").field("binding", buffer.binding)
            );
        }
        if buffer.backing_offset > buffer.backing_len {
            return Err(Status::args("metal_compute_backing_offset_out_of_range")
                .field("binding", buffer.binding)
                .field("offset", buffer.backing_offset)
                .field("backing_len", buffer.backing_len));
        }
        if buffer.len > buffer.backing_len - buffer.backing_offset {
            return Err(Status::args("metal_compute_backing_span_out_of_range")
                .field("binding", buffer.binding)
                .field("len", buffer.len)
                .field("offset", buffer.backing_offset)
                .field("backing_len", buffer.backing_len));
        }
        Ok((
            buffer.backing_data,
            buffer.backing_len,
            buffer.backing_offset,
        ))
    } else {
        Ok((buffer.data, buffer.len, 0))
    }
}

fn compute_buffer_backing_matches(a: &ReimsVgpuBuffer, b: &ReimsVgpuBuffer) -> bool {
    match (compute_buffer_backing(a), compute_buffer_backing(b)) {
        (Ok((ad, al, _)), Ok((bd, bl, _))) => ad == bd && al == bl,
        _ => false,
    }
}

fn bind_compute_buffers(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    buffers: &mut [ReimsVgpuBuffer],
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    mtl_buffers: &mut Vec<Buffer>,
    err: ErrOut<'_>,
) -> Status {
    let needs_index = has_indexed_layout(stage_input);
    if buffers.is_empty() {
        if needs_index {
            let idx = stage_input.map(|s| s.index_buffer_index).unwrap_or(0);
            set_err(
                err,
                format!("missing compute stageInputDescriptor index buffer {idx}"),
            );
            return Status::args("metal_compute_index_buffer_missing").field("buffer", idx);
        }
        return Status::OK;
    }

    let mut seen = [false; REIMS_VGPU_METAL_MAX_BUFFERS];
    for i in 0..buffers.len() {
        let buffer = &buffers[i];
        if !valid_buffer_binding(buffer.binding) {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_binding_out_of_range")
                .field("binding", buffer.binding)
                .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS);
        }
        if buffer.data.is_null() {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_data_missing")
                .field("binding", buffer.binding);
        }
        if buffer.len == 0 {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_length_zero")
                .field("binding", buffer.binding);
        }
        if seen[buffer.binding as usize] {
            set_err(
                err,
                format!("duplicate compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_binding_duplicate")
                .field("binding", buffer.binding);
        }
        seen[buffer.binding as usize] = true;

        let (stage_layout, stage_has_attr) = layout_for_buffer(stage_input, buffer.binding);
        if buffer.has_attribute_stride != 0 {
            let ok = stage_input.is_some()
                && stage_has_attr
                && stage_layout
                    .map(|l| l.stride == REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC)
                    .unwrap_or(false);
            if !ok {
                set_err(
                    err,
                    format!(
                        "compute buffer {} has attributeStride but no matching dynamic \
                         compute stageInputDescriptor layout",
                        buffer.binding
                    ),
                );
                return Status::args("metal_compute_attribute_stride_without_dynamic_layout")
                    .field("binding", buffer.binding)
                    .field("stride", buffer.attribute_stride);
            }
        } else if stage_has_attr
            && stage_layout
                .map(|l| l.stride == REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC)
                .unwrap_or(false)
        {
            set_err(
                err,
                format!(
                    "compute buffer {} uses a dynamic compute stageInputDescriptor layout \
                     without attributeStride",
                    buffer.binding
                ),
            );
            return Status::args("metal_compute_dynamic_layout_without_attribute_stride")
                .field("binding", buffer.binding);
        }

        let (backing_data, backing_len, backing_offset) = match compute_buffer_backing(buffer) {
            Ok(v) => v,
            Err(status) => {
                set_err(
                    err,
                    format!("invalid compute buffer backing {}", buffer.binding),
                );
                return status;
            }
        };

        let mut mtl_buffer: Option<Buffer> = None;
        for j in 0..i {
            if compute_buffer_backing_matches(buffer, &buffers[j]) {
                mtl_buffer = Some(mtl_buffers[j].clone());
                break;
            }
        }
        let mtl_buffer = match mtl_buffer {
            Some(b) => b,
            None => match new_buffer_from_host(device, backing_data, backing_len) {
                Some(b) => b,
                None => {
                    set_err(
                        err,
                        format!("failed to create compute buffer {}", buffer.binding),
                    );
                    return Status::execute("metal_compute_buffer_create_failed")
                        .field("binding", buffer.binding)
                        .field("backing_len", backing_len);
                }
            },
        };
        mtl_buffers.push(mtl_buffer.clone());
        if buffer.has_attribute_stride != 0 {
            set_buffer_with_attribute_stride(
                encoder,
                &mtl_buffer,
                backing_offset as u64,
                buffer.attribute_stride,
                buffer.binding as u64,
            );
        } else {
            encoder.set_buffer(
                buffer.binding as u64,
                Some(&mtl_buffer),
                backing_offset as u64,
            );
        }
    }

    if let Some(indexed_stage_input) = stage_input.filter(|_| needs_index) {
        let idx = indexed_stage_input.index_buffer_index as usize;
        // `valid_buffer_binding` first: it is what keeps `seen[idx]` in range.
        if !valid_buffer_binding(indexed_stage_input.index_buffer_index) || !seen[idx] {
            set_err(
                err,
                format!("missing compute stageInputDescriptor index buffer {idx}"),
            );
            return Status::args("metal_compute_stage_input_index_buffer_missing")
                .field("buffer", idx);
        }
    }
    Status::OK
}

pub(crate) fn bind_storage_images(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    images: &mut [ReimsVgpuStorageImage],
    mtl_images: &mut Vec<Texture>,
    err: ErrOut<'_>,
) -> Status {
    if images.is_empty() {
        return Status::OK;
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    for image in images.iter() {
        let Some(texture_index) = texture_index(image.binding) else {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_binding_invalid")
                .field("binding", image.binding);
        };
        let (pixel_format, bpp) = storage_image_format(image.format);
        let Some(expected_len) = tight_image_bytes(image.width, image.height, bpp) else {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_geometry_invalid")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height)
                .field("bpp", bpp);
        };
        if image.data.is_null() {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_data_missing")
                .field("binding", image.binding);
        }
        if image.len < expected_len {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_data_too_short")
                .field("binding", image.binding)
                .field("len", image.len)
                .field("expected", expected_len);
        }
        if seen[texture_index] {
            set_err(
                err,
                format!("duplicate storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_binding_duplicate")
                .field("binding", image.binding);
        }
        seen[texture_index] = true;

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(pixel_format);
        descriptor.set_width(image.width as u64);
        descriptor.set_height(image.height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor)
        else {
            set_err(err, "failed to allocate storage image texture");
            return Status::execute("metal_compute_storage_texture_alloc_failed")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height);
        };
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.replace_region(
            region,
            0,
            image.data as *const _,
            (image.width as u64) * (bpp as u64),
        );
        encoder.set_texture(texture_index as u64, Some(&texture));
        mtl_images.push(texture);
    }
    Status::OK
}

pub(crate) fn bind_compute_sampled_images(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    sampled: &[ReimsVgpuComputeSampledImage],
    mtl_sampled: &mut Vec<Texture>,
    err: ErrOut<'_>,
) -> Status {
    if sampled.is_empty() {
        return Status::OK;
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    for image in sampled {
        let Some(texture_index) = texture_index(image.binding) else {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_binding_invalid")
                .field("binding", image.binding);
        };
        let (pixel_format, bpp) = storage_image_format(image.format);
        let Some(expected_len) = tight_image_bytes(image.width, image.height, bpp) else {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_geometry_invalid")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height)
                .field("bpp", bpp);
        };
        if image.data.is_null() {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_data_missing")
                .field("binding", image.binding);
        }
        if image.len < expected_len {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_data_too_short")
                .field("binding", image.binding)
                .field("len", image.len)
                .field("expected", expected_len);
        }
        let swizzle = if image.has_swizzle != 0 {
            match texture_swizzle_channels(image.swizzle) {
                Some(s) => Some(s),
                None => {
                    set_err(
                        err,
                        format!("invalid sampled compute image swizzle {}", image.binding),
                    );
                    return Status::args("metal_compute_sampled_swizzle_invalid")
                        .field("binding", image.binding)
                        .field("swizzle", u32::from_le_bytes(image.swizzle));
                }
            }
        } else {
            None
        };
        if seen[texture_index] {
            set_err(
                err,
                format!("duplicate sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_binding_duplicate")
                .field("binding", image.binding);
        }
        seen[texture_index] = true;

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(pixel_format);
        descriptor.set_width(image.width as u64);
        descriptor.set_height(image.height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead;
        if swizzle.is_some() {
            usage |= MTLTextureUsage::PixelFormatView;
        }
        descriptor.set_usage(usage);
        let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor)
        else {
            set_err(err, "failed to allocate compute sampled image texture");
            return Status::execute("metal_compute_sampled_texture_alloc_failed")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height);
        };
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.replace_region(
            region,
            0,
            image.data as *const _,
            (image.width as u64) * (bpp as u64),
        );
        mtl_sampled.push(texture.clone());
        let bound = if let Some(sw) = swizzle {
            match new_texture_view_swizzled(&texture, pixel_format, sw) {
                Some(v) => {
                    mtl_sampled.push(v.clone());
                    v
                }
                None => {
                    set_err(
                        err,
                        format!(
                            "failed to create sampled compute swizzle view {}",
                            image.binding
                        ),
                    );
                    return Status::execute("metal_compute_sampled_swizzle_view_create_failed")
                        .field("binding", image.binding)
                        .field("format", image.format as u32);
                }
            }
        } else {
            texture
        };
        encoder.set_texture(texture_index as u64, Some(&bound));
    }
    Status::OK
}

pub(crate) fn bind_compute_samplers(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    samplers: &[ReimsVgpuSampler],
    err: ErrOut<'_>,
) -> Status {
    if samplers.is_empty() {
        return Status::OK;
    }
    if samplers.len() > REIMS_VGPU_METAL_MAX_SAMPLERS {
        set_err(err, "too many compute samplers");
        return Status::args("metal_compute_sampler_count_exceeded")
            .field("count", samplers.len())
            .field("limit", REIMS_VGPU_METAL_MAX_SAMPLERS);
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_SAMPLERS];
    for s in samplers {
        let Some(index) = sampler_index(s.binding) else {
            set_err(
                err,
                format!("invalid compute sampler binding {}", s.binding),
            );
            return Status::args("metal_compute_sampler_binding_invalid")
                .field("binding", s.binding);
        };
        if seen[index] {
            set_err(
                err,
                format!("duplicate compute sampler binding {}", s.binding),
            );
            return Status::args("metal_compute_sampler_binding_duplicate")
                .field("binding", s.binding);
        }
        let sampler = match make_explicit_sampler(device, s, err) {
            Ok(s) => s,
            Err(st) => return st,
        };
        seen[index] = true;
        if s.has_lod_clamp != 0 {
            encoder.set_sampler_state_with_lod(
                index as u64,
                Some(&sampler),
                f32::from_bits(s.clamp_lod_min_bits)..f32::from_bits(s.clamp_lod_max_bits),
            );
        } else {
            encoder.set_sampler_state(index as u64, Some(&sampler));
        }
    }
    Status::OK
}

/// Bind the dispatch's threadgroup-memory allocations.
///
/// The one bind path where an out-of-range index is not a wrong result but a
/// **process abort**: Metal answers an index at or past
/// `maxComputeLocalMemorySizes` by throwing, and there is no status to catch.
/// So this refuses first, by name, and the dispatch is declined rather than the
/// VM taken down — which is also what the accumulator no longer does, having
/// carried an unjustified 16 for the same job.
fn bind_threadgroup_memory(
    encoder: &ComputeCommandEncoderRef,
    tg: &[ReimsVgpuThreadgroupMemory],
) -> Status {
    for entry in tg {
        if !valid_threadgroup_memory_index(entry.index) {
            return Status::args("metal_compute_threadgroup_memory_index_over_table")
                .field("index", entry.index)
                .field("limit", REIMS_VGPU_METAL_MAX_THREADGROUP_MEMORY as u32);
        }
        encoder.set_threadgroup_memory_length(entry.index as u64, entry.length);
    }
    Status::OK
}

fn bind_stage_in_region(
    encoder: &ComputeCommandEncoderRef,
    region: Option<&ReimsVgpuComputeStageInRegion>,
) {
    let Some(region) = region else {
        return;
    };
    let metal_region = MTLRegion {
        origin: MTLOrigin {
            x: region.origin_x,
            y: region.origin_y,
            z: region.origin_z,
        },
        size: MTLSize {
            width: region.size_x,
            height: region.size_y,
            depth: region.size_z,
        },
    };
    set_stage_in_region(encoder, metal_region);
}

/// `Status` rather than `()` because the allocation below can refuse, and a
/// dispatch whose indirect stage-in region never reached the encoder reads its
/// threads from whatever the encoder held before.
fn bind_stage_in_region_indirect(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    retained: &mut Vec<Buffer>,
    arguments: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
) -> Status {
    let Some(arguments) = arguments else {
        return Status::OK;
    };
    let bytes = bytes_of(arguments);
    let indirect = unsafe {
        crate::backend::metal::raw_metal::new_buffer_with_data(
            device,
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let Some(indirect) = indirect else {
        return Status::execute("metal_compute_stage_in_indirect_buffer_alloc_failed")
            .field("len", bytes.len());
    };
    retained.push(indirect.clone());
    set_stage_in_region_indirect(encoder, &indirect, 0);
    Status::OK
}

fn mtl_dispatch_type(raw: u32) -> Option<MTLDispatchType> {
    match raw {
        REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL => Some(MTLDispatchType::Serial),
        REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT => Some(MTLDispatchType::Concurrent),
        _ => None,
    }
}

/// Metal resources retained after encode for deferred writeback (session path).
pub struct ComputeEncodeRetain {
    pub buffers: Vec<Buffer>,
    pub images: Vec<Texture>,
    pub sampled: Vec<Texture>,
    pub indirect: Vec<Buffer>,
}

/// Encode one compute dispatch onto an existing encoder (no end/commit/wait).
///
/// Used by multi-record control-flow sessions so nested dispatches sit inside
/// `encodeStartIf`/`While` SPI regions. Caller must keep `retain` alive until
/// after GPU completion, then call [`compute_writeback_from_mtl`].
// Twenty arguments because they are the compute dispatch's whole input set —
// the eight `ReimsVgpu*` bind arrays plus the grid — and this is the encoder-
// borrowing twin of `compute_core` below. Grouping them into a struct would
// have to be done to both or it splits one contract in two.
#[allow(clippy::too_many_arguments)]
pub fn compute_encode_on_encoder(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    mtlb: &[u8],
    buffers: &mut [ReimsVgpuBuffer],
    images: &mut [ReimsVgpuStorageImage],
    sampled: &[ReimsVgpuComputeSampledImage],
    samplers: &[ReimsVgpuSampler],
    threadgroup_memory: &[ReimsVgpuThreadgroupMemory],
    stage_in_region: Option<&ReimsVgpuComputeStageInRegion>,
    stage_in_region_indirect: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
    imageblock_dimensions: Option<&ReimsVgpuComputeImageblockDimensions>,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    // `dispatchThreads:` when true, `dispatchThreadgroups:` when false.
    //
    // A `bool` rather than the `REIMS_VGPU_COMPUTE_DISPATCH_KIND_*` ordinal the
    // archived C header spells, because the only producer already holds a
    // `bool` — `resolve_dispatch_dims_reported` returns one — and widening it
    // to a `{0, 1}` ordinal to cross this call put it next to `dispatch_type`,
    // which is also `{0, 1}`. Transposing that pair compiled, passed both
    // validators, and changed whether the grid counts threads or threadgroups
    // *and* whether Metal may overlap the segment. Two types cannot be
    // transposed, so the `match` that used to guard the ordinal is gone with
    // it: the state it refused is no longer representable.
    dispatch_threads: bool,
    grid: Extent3,
    threadgroup: Extent3,
    err: ErrOut<'_>,
) -> Result<ComputeEncodeRetain, Status> {
    // Unpacked once, here, so the body below keeps reading the six names it
    // always did. The pair crosses the call as two extents because that is
    // where a transposition stops being a compile error.
    let (grid_x, grid_y, grid_z) = (grid.x, grid.y, grid.z);
    let (tg_x, tg_y, tg_z) = (threadgroup.x, threadgroup.y, threadgroup.z);
    if grid_x == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_x_zero"));
    }
    if grid_y == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_y_zero"));
    }
    if grid_z == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_z_zero"));
    }
    if tg_x == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_x_zero"));
    }
    if tg_y == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_y_zero"));
    }
    if tg_z == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_z_zero"));
    }

    let function = load_only_function(device, mtlb, "compute", err)?;
    let pso = new_compute_pipeline_state(device, &function, mtlb, stage_input, err)?;

    let threadgroup_total = (tg_x as u64) * (tg_y as u64) * (tg_z as u64);
    let max_tg = pso.max_total_threads_per_threadgroup();
    if threadgroup_total > max_tg {
        set_err(
            err,
            format!(
                "compute PSO max threads per threadgroup is {max_tg}, need {threadgroup_total}"
            ),
        );
        return Err(Status::execute("metal_compute_threadgroup_limit_exceeded")
            .field("requested", threadgroup_total)
            .field("limit", max_tg));
    }

    encoder.set_compute_pipeline_state(&pso);

    let mut mtl_buffers = Vec::with_capacity(buffers.len());
    let rc = bind_compute_buffers(device, encoder, buffers, stage_input, &mut mtl_buffers, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let mut mtl_images = Vec::with_capacity(images.len());
    let rc = bind_storage_images(device, encoder, images, &mut mtl_images, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let mut mtl_sampled = Vec::with_capacity(sampled.len());
    let rc = bind_compute_sampled_images(device, encoder, sampled, &mut mtl_sampled, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let rc = bind_compute_samplers(device, encoder, samplers, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let rc = bind_threadgroup_memory(encoder, threadgroup_memory);
    if !rc.is_ok() {
        set_err(
            err,
            "threadgroup memory index past the Metal argument table",
        );
        return Err(rc);
    }
    bind_stage_in_region(encoder, stage_in_region);
    let mut retained_indirect = Vec::new();
    let rc = bind_stage_in_region_indirect(
        device,
        encoder,
        &mut retained_indirect,
        stage_in_region_indirect,
    );
    if !rc.is_ok() {
        return Err(rc);
    }
    if let Some(dims) = imageblock_dimensions {
        set_imageblock_width_height(encoder, dims.width as u64, dims.height as u64);
    }

    let grid = mtl_size(grid_x as u64, grid_y as u64, grid_z as u64);
    let tptg = mtl_size(tg_x as u64, tg_y as u64, tg_z as u64);
    if dispatch_threads {
        encoder.dispatch_threads(grid, tptg);
    } else {
        encoder.dispatch_thread_groups(grid, tptg);
    }
    clear_err(err);
    Ok(ComputeEncodeRetain {
        buffers: mtl_buffers,
        images: mtl_images,
        sampled: mtl_sampled,
        indirect: retained_indirect,
    })
}

/// Copy GPU buffer/image contents back into host `ReimsVgpuBuffer` / `ReimsVgpuStorageImage` pointers.
pub fn compute_writeback_from_mtl(
    buffers: &mut [ReimsVgpuBuffer],
    mtl_buffers: &[Buffer],
    images: &mut [ReimsVgpuStorageImage],
    mtl_images: &[Texture],
    err: ErrOut<'_>,
) -> Status {
    if mtl_buffers.len() != buffers.len() {
        set_err(err, "compute writeback buffer count mismatch");
        return Status::args("metal_compute_writeback_buffer_count_mismatch")
            .field("buffers", buffers.len())
            .field("metal_buffers", mtl_buffers.len());
    }
    if mtl_images.len() != images.len() {
        set_err(err, "compute writeback image count mismatch");
        return Status::args("metal_compute_writeback_image_count_mismatch")
            .field("images", images.len())
            .field("metal_images", mtl_images.len());
    }
    for i in 0..buffers.len() {
        let mut already = false;
        for j in 0..i {
            if compute_buffer_backing_matches(&buffers[i], &buffers[j]) {
                already = true;
                break;
            }
        }
        if already {
            continue;
        }
        let (backing_data, backing_len, _) = match compute_buffer_backing(&buffers[i]) {
            Ok(v) => v,
            Err(status) => {
                set_err(
                    err,
                    format!("invalid compute buffer backing {}", buffers[i].binding),
                );
                return status;
            }
        };
        let mtl_buffer = &mtl_buffers[i];
        unsafe {
            ptr::copy_nonoverlapping(
                mtl_buffer.contents() as *const u8,
                backing_data,
                backing_len,
            );
        }
    }
    for (i, image) in images.iter().enumerate() {
        let (_, bpp) = storage_image_format(image.format);
        let texture = &mtl_images[i];
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.get_bytes(
            image.data as *mut _,
            (image.width as u64) * (bpp as u64),
            region,
            0,
        );
    }
    clear_err(err);
    Status::OK
}

// The same input set as `compute_encode_on_encoder`, one encoder shorter.
#[allow(clippy::too_many_arguments)]
pub fn compute_core(
    mtlb: &[u8],
    buffers: &mut [ReimsVgpuBuffer],
    images: &mut [ReimsVgpuStorageImage],
    sampled: &[ReimsVgpuComputeSampledImage],
    samplers: &[ReimsVgpuSampler],
    threadgroup_memory: &[ReimsVgpuThreadgroupMemory],
    stage_in_region: Option<&ReimsVgpuComputeStageInRegion>,
    stage_in_region_indirect: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
    imageblock_dimensions: Option<&ReimsVgpuComputeImageblockDimensions>,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    // A `bool`, and forwarded as one — see `compute_encode_on_encoder`, which
    // consumes it. It sits beside `dispatch_type` here, which is why.
    dispatch_threads: bool,
    dispatch_type: u32,
    grid: Extent3,
    threadgroup: Extent3,
    err: ErrOut<'_>,
) -> Status {
    let Some(metal_dispatch_type) = mtl_dispatch_type(dispatch_type) else {
        set_err(
            err,
            format!("invalid compute dispatch type {dispatch_type}"),
        );
        return Status::args("metal_compute_dispatch_type_invalid")
            .field("dispatch_type", dispatch_type);
    };

    let Some(device) = system_device() else {
        set_err(err, "MTLCreateSystemDefaultDevice returned nil");
        return Status::execute("metal_compute_device_unavailable");
    };

    let queue = thread_queue(device);
    let Some(command_buffer) = crate::backend::metal::raw_metal::new_command_buffer(&queue) else {
        return Status::execute("metal_compute_command_buffer_unavailable");
    };
    let command_buffer = command_buffer.to_owned();
    let Some(encoder) =
        crate::backend::metal::raw_metal::new_compute_command_encoder_with_dispatch_type(
            &command_buffer,
            metal_dispatch_type,
        )
    else {
        return Status::execute("metal_compute_encoder_unavailable");
    };

    let retain = match compute_encode_on_encoder(
        device,
        encoder,
        mtlb,
        buffers,
        images,
        sampled,
        samplers,
        threadgroup_memory,
        stage_in_region,
        stage_in_region_indirect,
        imageblock_dimensions,
        stage_input,
        dispatch_threads,
        grid,
        threadgroup,
        err,
    ) {
        Ok(r) => r,
        Err(st) => {
            encoder.end_encoding();
            return st;
        }
    };

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        let detail = command_buffer_error_description(&command_buffer);
        set_err(err, format!("Metal command buffer failed: {detail}"));
        return Status::execute("metal_compute_command_buffer_failed");
    }

    let rc = compute_writeback_from_mtl(buffers, &retain.buffers, images, &retain.images, err);
    let _ = retain.sampled;
    let _ = retain.indirect;
    rc
}

/// The texture bindings a compute kernel's own reflection declares, as a list
/// this function owns.
///
/// # Why the caller does not size this
///
/// This used to take `(*mut usage, usage_cap, *out_count)` and refuse with
/// `..._capacity_exceeded` once the reflection named more bindings than the
/// caller's buffer held. Both callers sized that buffer with a bare `32`, so a
/// kernel declaring a 33rd texture had its whole dispatch refused — while
/// [`REIMS_VGPU_METAL_MAX_TEXTURES`] is 128 and
/// [`crate::runtime::draw::MAX_TEXTURE_BIND_SLOTS`] tells the rest of the device
/// that 128 is bindable. That is a cap the guest cannot see, below the one it is
/// told about, and losing a dispatch to it is not a limit any GPU has.
///
/// Nothing here crosses the C boundary — no shim names this function or
/// `ReimsVgpuComputeTextureUsage` — so the out-pointer shape bought nothing, and
/// the entry is already built as a `Vec` for [`reflect_insert`] regardless.
/// Returning it leaves exactly one bound on this path: Metal's own 128-entry
/// texture table, refused per *index* below.
pub fn reflect_compute_textures_mtlb(
    mtlb: &[u8],
    err: ErrOut<'_>,
) -> Result<Vec<ReimsVgpuComputeTextureUsage>, Status> {
    if mtlb.is_empty() {
        set_err(err, "compute MTLB is empty");
        return Err(Status::args("metal_compute_reflection_mtlb_empty"));
    }

    let key = BlobKey::new(mtlb);
    if let Some(cached) = reflect_lookup(&key) {
        clear_err(err);
        return Ok(cached);
    }

    let Some(device) = system_device() else {
        set_err(err, "MTLCreateSystemDefaultDevice returned nil");
        return Err(Status::execute(
            "metal_compute_reflection_device_unavailable",
        ));
    };
    let function = load_only_function(device, mtlb, "compute", err)?;

    // MTLPipelineOptionArgumentInfo == BindingInfo == 1
    let (pso, reflection) = match new_compute_pso_with_function_reflection(device, &function, 1) {
        Ok(v) => v,
        Err(e) => {
            crate::observe::Emit::decline("metal_compute_reflection_pso", &e)
                .field("mtlb_hash", format!("{:#x}", key.hash))
                .fail_once(key.hash);
            set_err(err, format!("compute reflection PSO failed: {e}"));
            return Err(
                Status::execute("metal_compute_reflection_pso_create_failed")
                    .field("mtlb_hash", key.hash),
            );
        }
    };
    if reflection.is_null() {
        set_err(err, "compute pipeline reflection unavailable");
        return Err(
            Status::execute("metal_compute_reflection_unavailable").field("mtlb_hash", key.hash)
        );
    }

    let bindings = reflection_bindings(reflection);
    // Drop reflection retain.
    unsafe {
        let _: () = msg_send_release(reflection);
    }

    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    let mut local: Vec<ReimsVgpuComputeTextureUsage> = Vec::new();
    for b in bindings {
        if !b.used || b.type_ != BINDING_TYPE_TEXTURE {
            continue;
        }
        let access = match b.access {
            BINDING_ACCESS_READ_ONLY => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
            BINDING_ACCESS_READ_WRITE => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE,
            BINDING_ACCESS_WRITE_ONLY => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
            other => {
                set_err(err, format!("unsupported compute texture access {other}"));
                return Err(
                    Status::args("metal_compute_reflection_texture_access_unsupported")
                        .field("access", other),
                );
            }
        };
        for elem in 0..b.array_length {
            let texture_index = b.index + elem;
            if texture_index < b.index || texture_index as usize >= REIMS_VGPU_METAL_MAX_TEXTURES {
                set_err(
                    err,
                    format!("compute texture index {texture_index} exceeds backend cap"),
                );
                return Err(
                    Status::args("metal_compute_reflection_texture_index_exceeded")
                        .field("index", texture_index)
                        .field("base", b.index)
                        .field("limit", REIMS_VGPU_METAL_MAX_TEXTURES),
                );
            }
            if seen[texture_index as usize] {
                set_err(
                    err,
                    format!("duplicate compute texture binding {texture_index}"),
                );
                return Err(
                    Status::args("metal_compute_reflection_texture_binding_duplicate")
                        .field("index", texture_index),
                );
            }
            // No length check on `local`: the index band above admits only
            // `texture_index < REIMS_VGPU_METAL_MAX_TEXTURES` and `seen` refuses
            // a repeat, so one push per distinct in-band index bounds this list
            // at the table width by construction. The check that used to sit
            // here compared the same width a second time and could not fire.
            seen[texture_index as usize] = true;
            local.push(ReimsVgpuComputeTextureUsage {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + texture_index as u32,
                access,
            });
        }
    }

    reflect_insert(&key, local.clone());
    clear_err(err);
    let _ = pso;
    Ok(local)
}

unsafe fn msg_send_release(obj: *mut objc::runtime::Object) {
    use objc::{msg_send, sel, sel_impl};
    let _: () = msg_send![obj, release];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    fn buffer(backing: &mut [u8]) -> ReimsVgpuBuffer {
        ReimsVgpuBuffer {
            binding: 7,
            data: std::ptr::null_mut(),
            len: 0,
            attribute_stride: 0,
            has_attribute_stride: 0,
            reserved0: 0,
            backing_data: backing.as_mut_ptr(),
            backing_len: backing.len(),
            backing_offset: 0,
        }
    }

    fn backing_refusal(buffer: &ReimsVgpuBuffer) -> String {
        let status = match compute_buffer_backing(buffer) {
            Ok(_) => panic!("invalid compute backing unexpectedly succeeded"),
            Err(status) => status,
        };
        Emit::refusal("metal_compute_test", &status)
            .expect("invalid compute backing must carry a refusal")
            .render()
    }

    #[test]
    fn compute_backing_rejections_preserve_the_failed_bound() {
        let mut empty = Vec::<u8>::with_capacity(1);
        let zero_len = buffer(empty.as_mut_slice());
        assert_eq!(
            backing_refusal(&zero_len),
            "metal_compute_test reason=metal_compute_backing_length_zero class=args binding=7"
        );

        let mut storage = vec![0u8; 8];
        let mut offset = buffer(&mut storage);
        offset.backing_offset = 9;
        assert_eq!(
            backing_refusal(&offset),
            "metal_compute_test reason=metal_compute_backing_offset_out_of_range class=args binding=7 offset=9 backing_len=8"
        );

        let mut span = buffer(&mut storage);
        span.backing_offset = 4;
        span.len = 5;
        assert_eq!(
            backing_refusal(&span),
            "metal_compute_test reason=metal_compute_backing_span_out_of_range class=args binding=7 len=5 offset=4 backing_len=8"
        );
    }

    /// A reflection wider than any caller-side buffer is served whole.
    ///
    /// `reflect_compute_textures_mtlb` took an out-pointer and a capacity, and
    /// both call sites passed a bare `32`; a kernel declaring a 33rd texture lost
    /// its entire dispatch to `..._capacity_exceeded`, against the 128 the device
    /// tells every other rail it binds. Drive the cached arm — which returns
    /// before the function needs an `MTLDevice`, so this runs anywhere — with a
    /// list well past the retired 32 and assert every entry comes back.
    #[test]
    fn a_reflection_past_the_retired_caller_capacity_is_served_whole() {
        // Distinct from any real kernel blob, so this cannot collide with an
        // entry another test in this process inserted.
        let mtlb: Vec<u8> = (0..64u8).map(|b| b ^ 0xa5).collect();
        let wide: Vec<ReimsVgpuComputeTextureUsage> = (0..REIMS_VGPU_METAL_MAX_TEXTURES as u32)
            .map(|i| ReimsVgpuComputeTextureUsage {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + i,
                access: REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
            })
            .collect();
        reflect_insert(&BlobKey::new(&mtlb), wide.clone());

        let mut err_buf = [0i8; 256];
        let served = reflect_compute_textures_mtlb(&mtlb, (err_buf.as_mut_ptr(), err_buf.len()))
            .expect("a cached reflection is served without a device");
        assert_eq!(
            served.len(),
            REIMS_VGPU_METAL_MAX_TEXTURES,
            "every reflected binding is returned, not the first 32"
        );
        assert_eq!(
            served.last().map(|u| u.binding),
            wide.last().map(|u| u.binding),
            "and the entries past the retired capacity are the ones reflected"
        );
    }
}
