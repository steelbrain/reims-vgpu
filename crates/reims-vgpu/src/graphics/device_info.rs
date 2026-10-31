//! The stage-3 device-info caps blob: the host fills [`RgfxDeviceInfo`] from the
//! platform's [`DeviceCaps`] and [`IdentityCaps`], so the guest bundle's `MTLDevice`
//! answers `supportsFamily:` of every family, `supportsFeatureSet:` of every
//! feature set, its identity (the host GPU's real name among it), the limits and
//! the feature queries as the host device does (Metal on macOS, the Vulkan
//! platform on Linux). A query the platform cannot answer (a runtime gap, an `Err`
//! or `None` in the caps) is reported as unsupported here, never guessed yes.
//! Where the lane's own device can do less than the host GPU, its answer is the
//! lane's ([`lane`]'s table).
//!
//! The decoder does not touch this: [`super::GraphicsDevice`] writes it once at
//! the device's init into the region at
//! [`RGFX_DEVICE_INFO_OFFSET`](reims_graphics::wire::RGFX_DEVICE_INFO_OFFSET),
//! and the guest bundle reads it there, not a guest command.

use reims_graphics::wire::{
    RGFX_DEVICE_ARCHITECTURE_NAME_LEN, RGFX_DEVICE_INFO_COUNTER_POINTS, RGFX_DEVICE_INFO_FAMILIES,
    RGFX_DEVICE_INFO_FEATURE_SETS, RGFX_DEVICE_INFO_FORMATS, RGFX_DEVICE_INFO_IDENTITY_FLAGS,
    RGFX_DEVICE_INFO_SIZES, RGFX_DEVICE_INFO_SPARSE_TILES, RGFX_DEVICE_INFO_VERSION,
    RGFX_DEVICE_NAME_LEN, RgfxDeviceInfo, RgfxFormatAnswers,
};
use reims_vgpu_core::platform::caps::{
    COUNTER_SAMPLING_POINTS, DeviceCaps, FAMILIES, FEATURE_SETS, FormatAnswers, IdentityCaps,
    SDK_FAMILIES, SDK_PIXEL_FORMATS, SPARSE_PAGE_SIZES, SPARSE_SAMPLE_COUNTS,
};
use reims_vgpu_core::raise::RaiseResult;

/// `MTLGPUFamilyMetal4`.
const MTL_GPU_FAMILY_METAL4: i64 = 5002;

/// 1 when the query answered a present `true`; 0 for a present `false` or a host
/// gap (an `Err`). A gap is never reported as support.
fn bit(answer: &RaiseResult<bool>) -> u8 {
    u8::from(matches!(answer, Ok(true)))
}

/// 1 for a present `true`; 0 for a `false` or a query the platform has no answer of.
fn known(answer: Option<bool>) -> u8 {
    u8::from(answer == Some(true))
}

/// `text` in `out`, cut at `out`'s length on a character's boundary; its length.
fn put_text(out: &mut [u8], text: &str) -> u32 {
    let mut len = text.len().min(out.len());
    while !text.is_char_boundary(len) {
        len -= 1;
    }
    out[..len].copy_from_slice(&text.as_bytes()[..len]);
    // At most 256 bytes: lossless.
    len as u32
}

/// `answers` as the blob holds them; a value the blob's field cannot hold (no such answer is
/// known) is 0, no answer.
fn format_answers(answers: &FormatAnswers) -> RgfxFormatAnswers {
    let mut out = RgfxFormatAnswers {
        format: u32::try_from(answers.format).unwrap_or(0),
        texture_buffer_alignment: answers
            .texture_buffer_alignment
            .and_then(|a| u32::try_from(a).ok())
            .unwrap_or(0),
        ..RgfxFormatAnswers::default()
    };
    if let Some(tiles) = answers.sparse_tiles {
        for (out, tile) in out.sparse_tiles.iter_mut().zip(tiles) {
            if let (Ok(w), Ok(h), Ok(d)) = (
                u16::try_from(tile[0]),
                u16::try_from(tile[1]),
                u16::try_from(tile[2]),
            ) {
                *out = [w, h, d];
            }
        }
    }
    out
}

/// The guest's device's answers where they are not the platform's: what the lane does not
/// carry yet. The guest's `MTLDevice` is the lane's own, which can do only what the wire, the
/// bundle and the decoder carry, so it answers no to a feature they lack whatever the host GPU
/// answers, and an app takes its fallback instead of sending a selector the bundle does not
/// implement. Each row names what it waits on, and goes when that lands:
///
/// | answer | the lane's | waits on |
/// |---|---|---|
/// | `supportsFamily:` of Metal4 | no | the `MTL4` API: its command queues and buffers, argument tables, compiler and allocators |
/// | `supportsRaytracing`, `supportsRaytracingFromRender`, `supportsPrimitiveMotionBlur` | no | acceleration structures, their builds, and intersection function tables |
/// | `supportsFunctionPointers`, `supportsFunctionPointersFromRender` | no | visible function tables and function handles |
/// | `supportsDynamicLibraries`, `supportsRenderDynamicLibraries` | no | `newDynamicLibrary:` and linked functions |
/// | `supportsPlacementSparse` | no | placement sparse buffers and textures |
/// | `supportsCounterSampling:` | no, at every point | counter sample buffers |
/// | `supportsRasterizationRateMapWithLayerCount:` | no, of every count | rasterization rate maps |
/// | `maximumConcurrentCompilationTaskCount` | the device's compile threads, which run its pipeline compiles | nothing: the device's own answer |
///
/// `hasUnifiedMemory` is the platform's: the decoder keeps `Managed` resources in one copy or two
/// by the same answer (docs/graphics-kext/design.md, Decided 4).
fn lane(mut identity: IdentityCaps, compile_threads: u64) -> IdentityCaps {
    let metal4 = SDK_FAMILIES
        .iter()
        .position(|&f| f == MTL_GPU_FAMILY_METAL4);
    if let Some(i) = metal4 {
        identity.families[i] = Some(false);
    }
    let f = &mut identity.flags;
    for no in [
        &mut f.supports_raytracing,
        &mut f.supports_raytracing_from_render,
        &mut f.supports_primitive_motion_blur,
        &mut f.supports_function_pointers,
        &mut f.supports_function_pointers_from_render,
        &mut f.supports_dynamic_libraries,
        &mut f.supports_render_dynamic_libraries,
        &mut f.supports_placement_sparse,
    ] {
        *no = Some(false);
    }
    identity.counter_sampling = [Some(false); 5];
    identity.sizes.largest_rasterization_rate_map_layer_count = Some(0);
    identity.sizes.maximum_concurrent_compilation_task_count = Some(compile_threads);
    identity
}

/// The device-info caps blob for `caps` and `identity`, of a device that runs `compile_threads`
/// pipeline compiles at once: every value the guest bundle reads once at init, the host device's
/// answers but where the lane's own differ ([`lane`]).
pub fn device_info(
    caps: &DeviceCaps,
    identity: IdentityCaps,
    compile_threads: u64,
) -> RgfxDeviceInfo {
    let identity = lane(identity, compile_threads);
    let mut info = RgfxDeviceInfo {
        version: RGFX_DEVICE_INFO_VERSION,
        registry_id: caps.registry_id,
        gpu_core_count: caps.gpu_core_count.unwrap_or(0),
        max_vertex_amplification_count: caps.max_vertex_amplification_count,
        heap_buffer_page_private: match &caps.heap_buffer_page_private {
            Ok(sa) => [sa.size, sa.align],
            Err(_) => [0, 0],
        },
        min_linear_alignment_bgra8: *caps.min_linear_alignment_bgra8.as_ref().unwrap_or(&0),
        ..RgfxDeviceInfo::default()
    };
    info.name_len = put_text(&mut info.name, caps.name.as_deref().unwrap_or(""));
    info.architecture_name_len = put_text(
        &mut info.architecture_name,
        identity.architecture_name.as_deref().unwrap_or(""),
    );
    let t = caps.max_threads_per_threadgroup;
    info.max_threads_per_threadgroup = [t.width, t.height, t.depth];

    let l = &caps.limits;
    // In the host `Limits` field order; the bundle reads them in this order.
    info.limits = [
        l.device_linear_texture_alignment_bytes,
        l.iosurface_read_only_texture_alignment_bytes,
        l.iosurface_texture_alignment_bytes,
        l.linear_texture_alignment_bytes,
        l.max_compute_buffers,
        l.max_compute_local_memory_sizes,
        l.max_compute_samplers,
        l.max_compute_textures,
        l.max_compute_threadgroup_memory,
        l.max_compute_threadgroup_memory_alignment_bytes,
        l.max_fragment_buffers,
        l.max_fragment_samplers,
        l.max_fragment_textures,
        l.max_predicated_nesting_depth,
        l.max_rasterization_rate_layer_count,
        l.max_texture_height_2d,
        l.max_texture_layers,
        l.max_texture_width_2d,
        l.max_threadgroup_memory_length,
        l.max_tile_buffers,
        l.max_tile_samplers,
        l.max_tile_textures,
        l.max_total_compute_threads_per_threadgroup,
        l.max_vertex_buffers,
        l.max_vertex_samplers,
        l.max_vertex_textures,
        l.max_viewport_count,
    ];
    let z = &identity.sizes;
    // In the host `IdentitySizes` field order; the bundle reads them in this order.
    let sizes: [Option<u64>; RGFX_DEVICE_INFO_SIZES] = [
        z.recommended_max_working_set_size,
        z.location,
        z.location_number,
        z.max_transfer_rate,
        z.peer_group_id,
        z.peer_index,
        z.peer_count,
        z.max_buffer_length,
        z.max_argument_buffer_sampler_count,
        z.argument_buffers_support,
        z.read_write_texture_support,
        z.sparse_tile_size_in_bytes,
        z.maximum_concurrent_compilation_task_count,
        z.largest_vertex_amplification_count,
        z.largest_rasterization_rate_map_layer_count,
    ];
    info.sizes = sizes.map(|size| size.unwrap_or(0));

    // Every family of the SDK, then -1009, which core's caps answer last.
    let families = SDK_FAMILIES
        .iter()
        .zip(identity.families)
        .map(|(&family, answer)| (family, known(answer)))
        .chain(std::iter::once((FAMILIES[6], bit(&caps.families[6]))));
    let mut family_count = 0;
    for (k, (family, answer)) in families.enumerate() {
        info.family_values[k] = family;
        info.family_answers[k] = answer;
        family_count = k + 1;
    }
    // At most RGFX_DEVICE_INFO_FAMILIES: lossless.
    info.family_count = family_count as u32;
    for (k, (&set, answer)) in FEATURE_SETS.iter().zip(identity.feature_sets).enumerate() {
        info.feature_set_values[k] = set;
        info.feature_set_answers[k] = known(answer);
    }
    info.feature_set_count = FEATURE_SETS.len() as u32;

    info.sample_counts = caps.sample_counts.each_ref().map(bit);
    info.primitive_types = caps.primitive_types.each_ref().map(bit);
    info.write_rounding_modes = caps.write_rounding_modes.each_ref().map(bit);
    let f = &caps.flags;
    // In the host `Flags` field order; the bundle reads them in this order.
    info.flags = [
        f.are_raster_order_groups_supported,
        f.is_depth24_stencil8_pixel_format_supported,
        f.is_framebuffer_read_supported,
        f.is_large_mrt_supported,
        f.is_rgb10a2_gamma_supported,
        f.supports_buffer_with_address_ranges,
        f.supports_buffer_with_iosurface,
        f.supports_command_buffer_jump,
        f.supports_compute_compressed_texture_write,
        f.supports_conditional_load_store,
        f.supports_dynamic_attribute_stride,
        f.supports_extended_xr10_formats,
        f.supports_float16_bcubic_filtering,
        f.supports_heap_with_address_ranges,
        f.supports_image_blocks,
        f.supports_memory_order_atomics,
        f.supports_native_hardware_fp16,
        f.supports_non_square_tile_shaders,
        f.supports_programmable_sample_positions,
        f.supports_resource_detach_backing,
        f.supports_shared_texture_handles,
        f.supports_simd_reduction,
        f.supports_simd_shuffle_and_fill,
        f.supports_texture_2d_multisample_array,
        f.supports_tile_shaders,
    ]
    .map(u8::from);
    let g = &identity.flags;
    // In the host `IdentityFlags` field order; the bundle reads them in this order.
    let identity_flags: [Option<bool>; RGFX_DEVICE_INFO_IDENTITY_FLAGS] = [
        g.is_low_power,
        g.is_headless,
        g.is_removable,
        g.has_unified_memory,
        g.supports_32bit_float_filtering,
        g.supports_32bit_msaa,
        g.supports_query_texture_lod,
        g.supports_bc_texture_compression,
        g.supports_pull_model_interpolation,
        g.are_barycentric_coords_supported,
        g.supports_shader_barycentric_coordinates,
        g.supports_dynamic_libraries,
        g.supports_render_dynamic_libraries,
        g.supports_raytracing,
        g.supports_function_pointers,
        g.supports_function_pointers_from_render,
        g.supports_raytracing_from_render,
        g.supports_primitive_motion_blur,
        g.supports_placement_sparse,
    ];
    info.identity_flags = identity_flags.map(known);
    info.counter_sampling = identity.counter_sampling.map(known);
    let mut format_count = 0;
    for (out, answers) in info.formats.iter_mut().zip(&identity.formats) {
        *out = format_answers(answers);
        format_count += 1;
    }
    info.format_count = format_count;
    info
}

// The blob holds every answer the host asks.
const _: () = assert!(SDK_FAMILIES.len() < RGFX_DEVICE_INFO_FAMILIES);
const _: () = assert!(FEATURE_SETS.len() <= RGFX_DEVICE_INFO_FEATURE_SETS);
const _: () = assert!(COUNTER_SAMPLING_POINTS.len() == RGFX_DEVICE_INFO_COUNTER_POINTS);
const _: () = assert!(RGFX_DEVICE_NAME_LEN <= 256 && RGFX_DEVICE_ARCHITECTURE_NAME_LEN <= 256);
const _: () = assert!(FAMILIES[6] == -1009);
const _: () = assert!(SDK_PIXEL_FORMATS.len() <= RGFX_DEVICE_INFO_FORMATS);
const _: () =
    assert!(SPARSE_SAMPLE_COUNTS.len() * SPARSE_PAGE_SIZES.len() == RGFX_DEVICE_INFO_SPARSE_TILES);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::raise::Raise;
    use reims_vgpu_core::testkit::FakePlatform;

    use super::*;

    /// The answer the blob gives of `family`, as the bundle looks it up: `None` where it lists
    /// no such family.
    fn family(info: &RgfxDeviceInfo, family: i64) -> Option<u8> {
        let n = info.family_count as usize;
        info.family_values[..n]
            .iter()
            .position(|&f| f == family)
            .map(|k| info.family_answers[k])
    }

    /// The answer the blob gives of feature set `set`.
    fn feature_set(info: &RgfxDeviceInfo, set: u64) -> Option<u8> {
        let n = info.feature_set_count as usize;
        info.feature_set_values[..n]
            .iter()
            .position(|&f| f == set)
            .map(|k| info.feature_set_answers[k])
    }

    /// The blob mirrors the device's caps: the name, a limit and a flag by
    /// position, and the version.
    #[test]
    fn mirrors_the_device_caps() {
        let p = Arc::new(FakePlatform::new());
        let caps = p.device().caps();
        let info = device_info(&caps, p.device().identity_caps(), 4);

        assert_eq!(info.version, RGFX_DEVICE_INFO_VERSION);
        assert_eq!(info.registry_id, caps.registry_id);
        assert_eq!(
            &info.name[..info.name_len as usize],
            caps.name.as_deref().unwrap().as_bytes()
        );
        assert_eq!(info.gpu_core_count, caps.gpu_core_count.unwrap());
        // The last limit by position is max_viewport_count.
        assert_eq!(info.limits[26], caps.limits.max_viewport_count);
        // The first flag by position is are_raster_order_groups_supported.
        assert_eq!(
            info.flags[0],
            u8::from(caps.flags.are_raster_order_groups_supported)
        );
    }

    /// Every family of the SDK and -1009 is answered, each the platform's (the fake answers as
    /// an M3 Max: Apple1 to Apple9, Mac1 and Mac2, the Common families and Metal3), but Metal4,
    /// which the lane cannot carry; a value that names no family is listed nowhere
    #[test]
    fn every_family_is_the_platforms_or_the_lanes() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        caps.families[6] = Ok(false);
        let info = device_info(&caps, p.device().identity_caps(), 4);
        assert_eq!(info.family_count, 21);
        for yes in [
            1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 2001, 2002, 3001, 3002, 3003,
            5001,
        ] {
            assert_eq!(family(&info, yes), Some(1), "family {yes}");
        }
        for no in [1010, 1011, 4001, 4002, 5002, -1009] {
            assert_eq!(family(&info, no), Some(0), "family {no}");
        }
        assert_eq!(family(&info, 7), None);
    }

    /// Every feature set is answered as the platform answers it: the fake's (an M3 Max's) every
    /// iOS one and macOS' but 10002, no tvOS one
    #[test]
    fn every_feature_set_is_the_platforms() {
        let p = Arc::new(FakePlatform::new());
        let info = device_info(&p.device().caps(), p.device().identity_caps(), 4);
        assert_eq!(info.feature_set_count as usize, FEATURE_SETS.len());
        for set in (0..=16).chain([10000, 10001, 10003, 10004, 10005]) {
            assert_eq!(feature_set(&info, set), Some(1), "feature set {set}");
        }
        for set in [10002, 30000, 30005] {
            assert_eq!(feature_set(&info, set), Some(0), "feature set {set}");
        }
    }

    /// The identity is the platform's, a name longer than v1's 32 bytes whole and the memory as
    /// the platform shares it, but where the lane's table says otherwise: what the wire does
    /// not carry answers no
    #[test]
    fn the_identity_is_the_platforms_but_the_lanes_table() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        caps.name = Some(String::from("NVIDIA GeForce RTX 5080 Laptop GPU"));
        let mut identity = p.device().identity_caps();
        identity.flags.has_unified_memory = Some(false);
        identity.flags.is_low_power = Some(true);
        let info = device_info(&caps, identity.clone(), 4);
        assert_eq!(
            &info.name[..info.name_len as usize],
            b"NVIDIA GeForce RTX 5080 Laptop GPU"
        );
        assert_eq!(
            &info.architecture_name[..info.architecture_name_len as usize],
            b"fake_gpu"
        );
        // is_low_power, then is_headless, is_removable, has_unified_memory.
        assert_eq!(info.identity_flags[..4], [1, 0, 0, 0]);
        // supports_bc_texture_compression is the platform's yes; raytracing (13) the lane's no.
        assert_eq!(info.identity_flags[7], 1);
        assert_eq!(info.identity_flags[13], 0);
        assert_eq!(info.identity_flags[18], 0);
        assert_eq!(info.counter_sampling, [0; 5]);
        // recommended_max_working_set_size first, the rate maps' layers last (the lane's 0).
        assert_eq!(
            info.sizes[0],
            identity.sizes.recommended_max_working_set_size.unwrap()
        );
        assert_eq!(info.sizes[7], identity.sizes.max_buffer_length.unwrap());
        // maximum_concurrent_compilation_task_count is the device's compile threads.
        assert_eq!(info.sizes[12], 4);
        assert_eq!(info.sizes[13], 8);
        assert_eq!(info.sizes[14], 0);
        // Every SDK format, each the platform's: BGRA8Unorm (80) 16-byte texture buffers and
        // 64x64 tiles; BC1_RGBA (130) no texture buffers.
        assert_eq!(info.format_count as usize, SDK_PIXEL_FORMATS.len());
        let at = |f: u32| info.formats.iter().find(|a| a.format == f).unwrap();
        assert_eq!(at(80).texture_buffer_alignment, 16);
        assert_eq!(at(80).sparse_tiles[0], [64, 64, 1]);
        assert_eq!(at(80).sparse_tiles[8], [128, 128, 1]);
        assert_eq!(at(130).texture_buffer_alignment, 0);
    }

    /// A present no reads 0, a present yes reads 1, and a platform gap (an `Err` or `None`)
    /// reads 0 — never a spurious yes; a name longer than the blob's is cut on a character
    #[test]
    fn a_platform_gap_is_reported_as_unsupported() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        let mut identity = p.device().identity_caps();
        // The fake answers sample count 8 (index 3) with a present no.
        assert_eq!(device_info(&caps, identity.clone(), 4).sample_counts[3], 0);
        // A family the platform could not answer at all (a gap) also reads 0.
        caps.families[6] = Err(Raise::foreign("device_info_test_gap"));
        identity.families[8] = None;
        identity.feature_sets[0] = None;
        identity.flags.is_low_power = None;
        identity.sizes.location = None;
        let info = device_info(&caps, identity.clone(), 4);
        assert_eq!(family(&info, -1009), Some(0));
        assert_eq!(family(&info, 1009), Some(0));
        assert_eq!(feature_set(&info, 0), Some(0));
        assert_eq!(info.identity_flags[0], 0);
        assert_eq!(info.sizes[1], 0);
        // A present yes still reads 1.
        caps.families[6] = Ok(true);
        assert_eq!(
            family(&device_info(&caps, identity.clone(), 4), -1009),
            Some(1)
        );
        // 255 ASCII bytes and a two-byte character: the character does not fit, and is left out.
        caps.name = Some(format!("{}\u{e9}", "a".repeat(255)));
        let info = device_info(&caps, identity, 4);
        assert_eq!(info.name_len, 255);
    }
}
