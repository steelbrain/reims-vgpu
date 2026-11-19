//! The rt device's answers: its capabilities ([`caps`]) and its identity ([`identity()`]), from
//! what Vulkan answers for its physical device ([`Facts`]) and rt's own device answers. Its
//! memory's are the Vulkan platform's (`crate::gpu::identity`'s), read at the call.
//!
//! **Metal's answers, as the Vulkan platform gives them.** rt answers the device-level questions
//! only where they depend on the device (`hasUnifiedMemory`, `supportsTextureSampleCount:`,
//! `minimumLinearTextureAlignmentForPixelFormat:`); the rest of a Metal device's answers are the
//! consumer's. So each is the Vulkan platform's answer where that answer is the platform's own
//! (its stand-ins, `crate::gpu::caps::stand_in`, and its fixed answers), and otherwise Metal's
//! answer on an Apple M3 Max, recorded in
//! `gpu/caps/tests/metal_answers/metal_tests.every_cap_is_metals.txt` ([`metal`]), held to the
//! device's Vulkan limits where one bounds it ([`caps`]). The guest's device info is made from
//! them as from the Vulkan platform's (reims-vgpu's `graphics::device_info`), which caps the
//! families and the features the graphics lane does not offer.
//!
//! **Where they differ from the Vulkan platform's**, which asks metal2vulkan-runtime for its
//! device's private properties and its family audit (`crate::gpu::caps`, `crate::gpu::identity`),
//! the guest sees another answer. The rt platform's GPU test lists every such difference on its
//! GPU (`device`'s tests); the expected ones:
//! - the counts the runtime clamped to its device's descriptor and stage limits are Metal's here,
//!   and the device's Vulkan limits bound only sizes, layers, threadgroup memory, threads and
//!   viewports;
//! - `gpuCoreCount` is none: Vulkan reports no core count;
//! - the families beyond `caps`' are their closure under containment
//!   (`crate::gpu::identity::closed`), not the runtime's `supportsFamily:`;
//! - the shader features of the runtime's family audit are Metal's answers where rt's translator
//!   lowers them, Vulkan's feature where one decides them (BC compression, barycentrics, 32-bit
//!   float filtering and MSAA), and no for what rt has none of (dynamic libraries, function
//!   pointers, ray tracing);
//! - no sparse tile answers: rt has no sparse textures;
//! - `maxBufferLength` is rt's [`MAX_BUFFER_LENGTH`](metal2vulkan_rt::MAX_BUFFER_LENGTH), and
//!   `maxArgumentBufferSamplerCount` the indices of rt's sampler class;
//! - the linear texture alignments (the device's, an IOSurface's, a linear texture's) are the
//!   largest of rt's `minimumLinearTextureAlignmentForPixelFormat:` answers, as the runtime's were
//!   the largest of its, and BGRA8's is rt's.
//!
//! A feature answered here that rt does not build yet is refused where the guest uses it (a
//! pipeline of it is rt's `Gap`), counted (`super::unbuilt`).

use std::sync::Arc;

use ash::vk;

use crate::core::platform::caps::{
    DeviceCaps, FAMILIES, FEATURE_SETS, Flags, FormatAnswers, IdentityCaps, IdentityFlags,
    IdentitySizes, Limits, SDK_FAMILIES, SDK_PIXEL_FORMATS, TEXTURE_BUFFER_FORMATS,
};
use crate::core::platform::types::{MtlSize, SizeAndAlign};
use crate::core::raise::Raise;
use crate::gpu::caps::{METAL_LINEAR_ALIGNMENT, stand_in};
use crate::gpu::identity;

/// Metal's answers on an Apple M3 Max (macOS 27.0.1, recorded 2026-11-07 in
/// `gpu/caps/tests/metal_answers/metal_tests.every_cap_is_metals.txt`), which the rt device
/// answers but where the module's docs say otherwise.
pub mod metal {
    use crate::core::platform::caps::{Flags, Limits};

    /// `maxThreadsPerThreadgroup`, each dimension.
    pub const MAX_THREADS_PER_THREADGROUP: u64 = 1024;
    /// `maxVertexAmplificationCount`.
    pub const MAX_VERTEX_AMPLIFICATION_COUNT: u64 = 2;
    /// `supportsPrimitiveType:` of the raw values 0 to 4, the public ones.
    pub const PUBLIC_PRIMITIVE_TYPES: u64 = 5;

    /// The device's counts.
    pub const LIMITS: Limits = Limits {
        device_linear_texture_alignment_bytes: 16,
        iosurface_read_only_texture_alignment_bytes: 16,
        iosurface_texture_alignment_bytes: 16,
        linear_texture_alignment_bytes: 16,
        max_compute_buffers: 31,
        max_compute_local_memory_sizes: 31,
        max_compute_samplers: 16,
        max_compute_textures: 128,
        max_compute_threadgroup_memory: 65536,
        max_compute_threadgroup_memory_alignment_bytes: 16,
        max_fragment_buffers: 31,
        max_fragment_samplers: 16,
        max_fragment_textures: 128,
        max_predicated_nesting_depth: 4_294_967_295,
        max_rasterization_rate_layer_count: 2,
        max_texture_height_2d: 16384,
        max_texture_layers: 2048,
        max_texture_width_2d: 16384,
        max_threadgroup_memory_length: 32768,
        max_tile_buffers: 31,
        max_tile_samplers: 16,
        max_tile_textures: 128,
        max_total_compute_threads_per_threadgroup: 1024,
        max_vertex_buffers: 31,
        max_vertex_samplers: 16,
        max_vertex_textures: 128,
        max_viewport_count: 16,
    };

    /// The device's flags.
    pub const FLAGS: Flags = Flags {
        are_raster_order_groups_supported: true,
        is_depth24_stencil8_pixel_format_supported: false,
        is_framebuffer_read_supported: true,
        is_large_mrt_supported: true,
        is_rgb10a2_gamma_supported: true,
        supports_buffer_with_address_ranges: true,
        supports_buffer_with_iosurface: true,
        supports_command_buffer_jump: true,
        supports_compute_compressed_texture_write: false,
        supports_conditional_load_store: true,
        supports_dynamic_attribute_stride: true,
        supports_extended_xr10_formats: true,
        supports_float16_bcubic_filtering: true,
        supports_heap_with_address_ranges: true,
        supports_image_blocks: true,
        supports_memory_order_atomics: true,
        supports_native_hardware_fp16: true,
        supports_non_square_tile_shaders: true,
        supports_programmable_sample_positions: true,
        supports_resource_detach_backing: true,
        supports_shared_texture_handles: true,
        supports_simd_reduction: true,
        supports_simd_shuffle_and_fill: true,
        supports_texture_2d_multisample_array: true,
        supports_tile_shaders: true,
    };

    /// `supportsTextureWriteRoundingMode:` of each of core's `WRITE_ROUNDING_MODES`
    /// (`crate::core::platform::caps::WRITE_ROUNDING_MODES`).
    pub const WRITE_ROUNDING_MODES: [bool; 2] = [true, true];
    /// `argumentBuffersSupport`: `MTLArgumentBuffersTier2`.
    pub const ARGUMENT_BUFFERS_TIER: u64 = 1;
    /// `readWriteTextureSupport`: `MTLReadWriteTextureTier2`.
    pub const READ_WRITE_TEXTURE_TIER: u64 = 2;
}

/// What the rt platform reads of its device's physical device from Vulkan and rt, once, as the
/// device is made: what its answers depend on beyond Metal's. Not `Debug`: Vulkan's limits are
/// not.
#[derive(Clone)]
pub struct Facts {
    /// `deviceName`.
    pub name: String,
    /// The device's registry ID, the platform's ([`registry_id`]).
    pub registry_id: u64,
    /// `vendorID`.
    pub vendor_id: u32,
    /// `deviceID`.
    pub device_id: u32,
    /// `pipelineCacheUUID`.
    pub pipeline_cache_uuid: [u8; vk::UUID_SIZE],
    /// The device's limits.
    pub limits: vk::PhysicalDeviceLimits,
    /// `shaderFloat16`, which rt's floor reads.
    pub float16: bool,
    /// rt's `hasUnifiedMemory`.
    pub unified_memory: bool,
    /// rt's `supportsTextureSampleCount:` of each of core's sample counts
    /// ([`SAMPLE_COUNTS`](crate::core::platform::caps::SAMPLE_COUNTS)).
    pub sample_counts: [bool; 4],
    /// `textureCompressionBC`.
    pub bc: bool,
    /// Whether rt enabled `VK_KHR_fragment_shader_barycentric`.
    pub barycentrics: bool,
    /// Whether the device filters `R32G32B32A32_SFLOAT` linearly in an optimal image.
    pub float32_filtering: bool,
    /// Whether the device renders `R32G32B32A32_SFLOAT` at 4 samples.
    pub float32_msaa: bool,
    /// The bytes of its device-local memory heaps, summed.
    pub device_local: u64,
    /// The host's page size.
    pub page_size: u64,
    /// rt's `heapBufferSizeAndAlignWithLength:options:` of a page of Private storage.
    pub heap_buffer_page_private: SizeAndAlign,
    /// The largest of rt's `minimumLinearTextureAlignmentForPixelFormat:` answers of the SDK's
    /// pixel formats, and Metal's 16 where every one is smaller.
    pub linear_alignment_largest: u64,
    /// rt's `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8; none where rt refuses it.
    pub linear_alignment_bgra8: Option<u64>,
}

/// The registry ID the rt platform gives the device of `uuid`, its `deviceUUID`: its two halves'
/// little-endian words, exclusive-or'ed. Stable across processes and boots for one device and
/// driver, as Metal's `registryID` is for one GPU.
pub fn registry_id(uuid: &[u8; vk::UUID_SIZE]) -> u64 {
    let (low, high) = uuid.split_at(8);
    let word = |half: &[u8]| u64::from_le_bytes(<[u8; 8]>::try_from(half).unwrap_or_default());
    word(low) ^ word(high)
}

/// `maxArgumentBufferSamplerCount`: the indices of rt's sampler class, each of which a sampler
/// state an argument buffer names takes.
fn sampler_count() -> u64 {
    let range = metal2vulkan_rt::IndexClass::Sampler.range();
    u64::from(range.end.saturating_sub(range.start))
}

/// `x` at most `bound`, a Vulkan limit.
fn held(x: u64, bound: u32) -> u64 {
    x.min(u64::from(bound))
}

/// The selector of a linear alignment's query.
const LINEAR_ALIGNMENT: &str = "minimumLinearTextureAlignmentForPixelFormat:";

/// The raise of `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8 where rt refuses it.
fn bgra8_refused() -> Raise {
    Raise::platform(LINEAR_ALIGNMENT).with_detail("metal2vulkan-rt refused BGRA8Unorm")
}

/// The device's capabilities: Metal's answers ([`metal`]), held to `facts`' Vulkan limits where
/// one bounds them, with the Vulkan platform's own answers where they are its own (the module
/// docs).
///
/// - The Vulkan limits: 2D sizes to `maxImageDimension2D`, layers to `maxImageArrayLayers`,
///   threadgroup memory to `maxComputeSharedMemorySize`, a threadgroup's threads to
///   `maxComputeWorkGroupSize` and `maxComputeWorkGroupInvocations`, viewports to `maxViewports`.
/// - The Vulkan platform's own: no buffer or heap over address ranges and no detach of a
///   resource's backing, which it does not build; the stand-ins; `supportsNativeHardwareFP16` as
///   the device's `shaderFloat16`.
/// - rt's linear alignments: the device's, an IOSurface's and a linear texture's alignment are the
///   largest of every format's ([`Facts::linear_alignment_largest`]), and
///   `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8 rt's
///   ([`Facts::linear_alignment_bgra8`]), raised where rt refuses it. So a guest that lays a
///   linear texture out by the device's alignment lays it as the device rounds rows.
/// - rt's: the heap buffer query of a page of Private storage
///   ([`Facts::heap_buffer_page_private`]), Metal's answer on every device: the length unrounded,
///   at a multiple of 256 bytes.
pub fn caps(facts: &Facts) -> Arc<DeviceCaps> {
    let v = &facts.limits;
    let total = held(
        metal::LIMITS.max_total_compute_threads_per_threadgroup,
        v.max_compute_work_group_invocations,
    );
    let [width, height, depth] = v
        .max_compute_work_group_size
        .map(|bound| held(metal::MAX_THREADS_PER_THREADGROUP, bound).min(total));
    let limits = Limits {
        max_texture_height_2d: held(
            metal::LIMITS.max_texture_height_2d,
            v.max_image_dimension2_d,
        ),
        max_texture_width_2d: held(metal::LIMITS.max_texture_width_2d, v.max_image_dimension2_d),
        max_texture_layers: held(metal::LIMITS.max_texture_layers, v.max_image_array_layers),
        max_threadgroup_memory_length: held(
            metal::LIMITS.max_threadgroup_memory_length,
            v.max_compute_shared_memory_size,
        ),
        max_total_compute_threads_per_threadgroup: total,
        max_viewport_count: held(metal::LIMITS.max_viewport_count, v.max_viewports),
        device_linear_texture_alignment_bytes: facts.linear_alignment_largest,
        iosurface_read_only_texture_alignment_bytes: facts.linear_alignment_largest,
        iosurface_texture_alignment_bytes: facts.linear_alignment_largest,
        linear_texture_alignment_bytes: facts.linear_alignment_largest,
        ..metal::LIMITS
    };
    let flags = Flags {
        is_rgb10a2_gamma_supported: stand_in::RGB10A2_GAMMA,
        supports_buffer_with_address_ranges: false,
        supports_extended_xr10_formats: stand_in::EXTENDED_XR10_FORMATS,
        supports_float16_bcubic_filtering: stand_in::FLOAT16_BCUBIC_FILTERING,
        supports_heap_with_address_ranges: false,
        supports_image_blocks: stand_in::IMAGE_BLOCKS,
        supports_native_hardware_fp16: facts.float16,
        supports_non_square_tile_shaders: stand_in::TILE_SHADERS,
        supports_resource_detach_backing: false,
        supports_texture_2d_multisample_array: stand_in::TEXTURE_2D_MULTISAMPLE_ARRAY,
        supports_tile_shaders: stand_in::TILE_SHADERS,
        ..metal::FLAGS
    };
    let primitive_type = |t: u64| match t.checked_sub(metal::PUBLIC_PRIMITIVE_TYPES) {
        None => true,
        Some(private) => usize::try_from(private)
            .ok()
            .and_then(|at| stand_in::PRIVATE_PRIMITIVE_TYPES.get(at).copied())
            .unwrap_or(false),
    };
    Arc::new(DeviceCaps {
        name: Some(facts.name.clone()),
        registry_id: facts.registry_id,
        gpu_core_count: None,
        max_threads_per_threadgroup: MtlSize::new(width, height, depth),
        max_vertex_amplification_count: metal::MAX_VERTEX_AMPLIFICATION_COUNT,
        limits,
        flags,
        sample_counts: facts.sample_counts.map(Ok),
        primitive_types: std::array::from_fn(|t| {
            Ok(primitive_type(u64::try_from(t).unwrap_or(u64::MAX)))
        }),
        families: stand_in::FAMILIES.map(Ok),
        write_rounding_modes: metal::WRITE_ROUNDING_MODES.map(Ok),
        heap_buffer_page_private: Ok(facts.heap_buffer_page_private),
        min_linear_alignment_bgra8: facts.linear_alignment_bgra8.ok_or_else(bgra8_refused),
    })
}

/// The device's identity and the answers beyond [`caps`] (`GpuDevice::identity_caps`): the
/// families `caps` claims and every family they contain, the feature sets of those, the
/// architecture's name as the Vulkan platform makes it, each SDK format's alignments
/// (`linear` answers rt's `minimumLinearTextureAlignmentForPixelFormat:` of a raw format, `None`
/// where rt makes no texture of it), the sizes and the flags (the module docs).
pub fn identity(
    facts: &Facts,
    caps: &DeviceCaps,
    linear: impl Fn(u64) -> Option<u64>,
) -> IdentityCaps {
    let mut claimed = [false; 20];
    for (&family, answer) in FAMILIES.iter().zip(&caps.families) {
        if let Some(at) = SDK_FAMILIES.iter().position(|&f| f == family) {
            claimed[at] = matches!(answer, Ok(true));
        }
    }
    let families = identity::closed(claimed);
    let [
        feature_profile,
        half_fp_config,
        single_fp_config,
        double_fp_config,
    ] = identity::driver_sizes(facts.registry_id);
    IdentityCaps {
        families: families.map(Some),
        feature_sets: FEATURE_SETS.map(|set| Some(identity::feature_set(set, &families))),
        counter_sampling: [None; 5],
        architecture_name: Some(identity::architecture_name(
            facts.vendor_id,
            facts.device_id,
            facts.pipeline_cache_uuid,
        )),
        formats: SDK_PIXEL_FORMATS
            .iter()
            .map(|&format| {
                let linear_alignment = linear(format);
                FormatAnswers {
                    format,
                    texture_buffer_alignment: (TEXTURE_BUFFER_FORMATS.contains(&format)
                        && linear_alignment.is_some())
                    .then_some(METAL_LINEAR_ALIGNMENT),
                    linear_alignment,
                    sparse_tiles: None,
                }
            })
            .collect(),
        sizes: IdentitySizes {
            recommended_max_working_set_size: Some(facts.device_local),
            max_buffer_length: Some(metal2vulkan_rt::MAX_BUFFER_LENGTH),
            max_argument_buffer_sampler_count: Some(sampler_count()),
            argument_buffers_support: Some(metal::ARGUMENT_BUFFERS_TIER),
            read_write_texture_support: Some(metal::READ_WRITE_TEXTURE_TIER),
            largest_vertex_amplification_count: Some(caps.max_vertex_amplification_count),
            largest_rasterization_rate_map_layer_count: Some(
                caps.limits.max_rasterization_rate_layer_count,
            ),
            feature_profile,
            half_fp_config,
            single_fp_config,
            double_fp_config,
            ..IdentitySizes::default()
        },
        flags: IdentityFlags {
            has_unified_memory: Some(facts.unified_memory),
            supports_32bit_float_filtering: Some(facts.float32_filtering),
            supports_32bit_msaa: Some(facts.float32_msaa),
            supports_query_texture_lod: Some(true),
            supports_bc_texture_compression: Some(facts.bc),
            supports_pull_model_interpolation: Some(true),
            are_barycentric_coords_supported: Some(facts.barycentrics),
            supports_shader_barycentric_coordinates: Some(facts.barycentrics),
            supports_dynamic_libraries: Some(false),
            supports_render_dynamic_libraries: Some(false),
            supports_raytracing: Some(false),
            supports_function_pointers: Some(false),
            supports_function_pointers_from_render: Some(false),
            supports_raytracing_from_render: Some(false),
            ..IdentityFlags::default()
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Facts of a device with Metal's limits or more, and every feature.
    pub(crate) fn roomy() -> Facts {
        Facts {
            name: "a GPU".to_owned(),
            registry_id: 7,
            vendor_id: 0x1002,
            device_id: 0x15bf,
            pipeline_cache_uuid: [0xab; vk::UUID_SIZE],
            limits: vk::PhysicalDeviceLimits {
                max_image_dimension2_d: 32768,
                max_image_array_layers: 8192,
                max_compute_shared_memory_size: 65536,
                max_compute_work_group_invocations: 1024,
                max_compute_work_group_size: [1024, 1024, 1024],
                max_viewports: 16,
                ..vk::PhysicalDeviceLimits::default()
            },
            float16: true,
            unified_memory: true,
            sample_counts: [true, true, true, false],
            bc: true,
            barycentrics: true,
            float32_filtering: true,
            float32_msaa: true,
            device_local: 8 << 30,
            page_size: 4096,
            heap_buffer_page_private: SizeAndAlign {
                size: 4096,
                align: 256,
            },
            linear_alignment_largest: 16,
            linear_alignment_bgra8: Some(16),
        }
    }

    /// On a device of Metal's limits or more, the counts are Metal's, the flags Metal's but the
    /// Vulkan platform's own, and the heap buffer query answers rt's of the page.
    #[test]
    fn a_roomy_device_answers_metal_s_counts() {
        let caps = caps(&roomy());
        assert_eq!(caps.limits, metal::LIMITS);
        assert_eq!(
            caps.max_threads_per_threadgroup,
            MtlSize::new(1024, 1024, 1024)
        );
        assert_eq!(caps.name.as_deref(), Some("a GPU"));
        assert_eq!(caps.gpu_core_count, None);
        let f = caps.flags;
        assert!(!f.supports_buffer_with_address_ranges);
        assert!(!f.supports_heap_with_address_ranges);
        assert!(!f.supports_resource_detach_backing);
        assert!(!f.is_depth24_stencil8_pixel_format_supported);
        assert!(f.is_framebuffer_read_supported && f.supports_native_hardware_fp16);
        assert_eq!(
            caps.sample_counts.clone().map(Result::unwrap),
            [true, true, true, false]
        );
        let primitives: Vec<bool> = (caps.primitive_types.iter())
            .map(|t| *t.as_ref().unwrap())
            .collect();
        assert_eq!(primitives, [[true; 9].as_slice(), &[false]].concat());
        assert_eq!(
            caps.heap_buffer_page_private(crate::core::raise::OnFault::Die),
            Ok(SizeAndAlign {
                size: 4096,
                align: 256
            })
        );
        assert_eq!(caps.min_linear_alignment_bgra8(), Ok(16));
    }

    /// A device's Vulkan limits bound the sizes, layers, threadgroup memory, threads and viewports
    /// it answers; no fp16 arithmetic answers no native fp16.
    #[test]
    fn vulkan_limits_bound_the_answers() {
        let mut facts = roomy();
        facts.limits.max_image_dimension2_d = 8192;
        facts.limits.max_image_array_layers = 256;
        facts.limits.max_compute_shared_memory_size = 16384;
        facts.limits.max_compute_work_group_invocations = 512;
        facts.limits.max_compute_work_group_size = [1024, 1024, 64];
        facts.limits.max_viewports = 1;
        facts.float16 = false;
        let caps = caps(&facts);
        let l = caps.limits;
        assert_eq!(
            (l.max_texture_width_2d, l.max_texture_height_2d),
            (8192, 8192)
        );
        assert_eq!(l.max_texture_layers, 256);
        assert_eq!(l.max_threadgroup_memory_length, 16384);
        assert_eq!(l.max_total_compute_threads_per_threadgroup, 512);
        assert_eq!(caps.max_threads_per_threadgroup, MtlSize::new(512, 512, 64));
        assert_eq!(l.max_viewport_count, 1);
        assert_eq!(l.max_fragment_textures, metal::LIMITS.max_fragment_textures);
        assert!(!caps.flags.supports_native_hardware_fp16);
    }

    /// The linear texture alignments are the largest of rt's per-format answers, and BGRA8's is
    /// rt's: a refused one raises.
    #[test]
    fn the_linear_alignments_are_rt_s() {
        let mut facts = roomy();
        facts.linear_alignment_largest = 256;
        facts.linear_alignment_bgra8 = Some(128);
        let answered = caps(&facts);
        let l = answered.limits;
        let all = [
            l.device_linear_texture_alignment_bytes,
            l.iosurface_read_only_texture_alignment_bytes,
            l.iosurface_texture_alignment_bytes,
            l.linear_texture_alignment_bytes,
        ];
        assert_eq!(all, [256; 4]);
        assert_eq!(answered.min_linear_alignment_bgra8, Ok(128));
        facts.linear_alignment_bgra8 = None;
        assert_eq!(caps(&facts).min_linear_alignment_bgra8, Err(bgra8_refused()));
    }

    /// The identity claims the families caps claims and those they contain, their feature sets,
    /// none of what rt lacks, and each format's alignments as rt answers them.
    #[test]
    fn the_identity_claims_the_families_caps_contain() {
        let facts = roomy();
        let caps = caps(&facts);
        let id = identity(&facts, &caps, |format| (format == 80).then_some(64));
        let claimed: Vec<i64> = (SDK_FAMILIES.iter().zip(id.families))
            .filter_map(|(&f, yes)| (yes == Some(true)).then_some(f))
            .collect();
        assert_eq!(
            claimed,
            [
                1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 2001, 2002, 3001, 3002, 3003,
                5001
            ]
        );
        assert!(id.feature_sets.iter().all(Option::is_some));
        let bgra8 = id.formats.iter().find(|f| f.format == 80).unwrap();
        assert_eq!(
            (
                bgra8.texture_buffer_alignment,
                bgra8.linear_alignment,
                bgra8.sparse_tiles
            ),
            (Some(16), Some(64), None)
        );
        let rgba8 = id.formats.iter().find(|f| f.format == 70).unwrap();
        assert_eq!(
            (rgba8.texture_buffer_alignment, rgba8.linear_alignment),
            (None, None)
        );
        assert_eq!(id.formats.len(), SDK_PIXEL_FORMATS.len());
        assert_eq!(id.flags.supports_raytracing, Some(false));
        assert_eq!(id.flags.supports_dynamic_libraries, Some(false));
        assert_eq!(id.flags.has_unified_memory, Some(true));
        assert_eq!(id.sizes.max_buffer_length, Some(1 << 31));
        assert_eq!(id.sizes.max_argument_buffer_sampler_count, Some(3999));
        assert_eq!(
            id.architecture_name.as_deref(),
            Some("vulkan_1002_15bf_abababababababababababababababab")
        );
    }

    /// The registry ID folds the UUID's halves, so two devices of UUIDs that differ in either half
    /// differ.
    #[test]
    fn the_registry_id_folds_the_uuid() {
        let mut uuid = [0u8; vk::UUID_SIZE];
        uuid[0] = 1;
        uuid[8] = 2;
        assert_eq!(registry_id(&uuid), 3);
        uuid[15] = 0x80;
        assert_eq!(registry_id(&uuid), 3 | 0x80 << 56);
    }
}
