//! `DeviceCaps`: everything core asks a device that does not change while the device lives,
//! read once per device by the platform ([`GpuDevice::caps`](super::device::GpuDevice::caps)).
//!
//! [`Limits`] and [`Flags`] hold one field per `NSUInteger` and `BOOL` property the device is
//! asked, named after its selector (`registryID` and `maxVertexAmplificationCount` are
//! [`DeviceCaps`]'s own fields). A query that can raise is kept as the `RaiseResult` it answered:
//! its accessor raises it again at the site that uses it, with the same identity and origin.

use super::types::{MtlSize, SizeAndAlign};
use crate::raise::{OnFault, RaiseResult};

/// The sample counts core asks `supportsTextureSampleCount:` about, in order.
pub const SAMPLE_COUNTS: [u64; 4] = [1, 2, 4, 8];
/// The primitive types core asks `supportsPrimitiveType:` about: `0..10`, raw.
pub const PRIMITIVE_TYPE_COUNT: u64 = 10;
/// The families core asks `supportsFamily:` about, in order: `MTLGPUFamilyApple5` to
/// `MTLGPUFamilyApple10`, then -1009, which no SDK family is.
pub const FAMILIES: [i64; 7] = [1005, 1006, 1007, 1008, 1009, 1010, -1009];
/// The modes core asks `supportsTextureWriteRoundingMode:` about, in order.
pub const WRITE_ROUNDING_MODES: [i64; 2] = [1, 2];
/// The storage mode of the heap-buffer size query: `MTLResourceStorageModePrivate`.
pub const HEAP_QUERY_OPTIONS: u64 = 0x20;
/// The pixel format of the linear-alignment query: `MTLPixelFormatBGRA8Unorm`.
pub const LINEAR_ALIGNMENT_FORMAT: u64 = 80;

/// Declares a struct with one `pub` field of `$ty` per property, named after its selector.
macro_rules! caps_struct {
    ($(#[$m:meta])* $name:ident: $ty:ty; $( $field:ident, )+) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        #[allow(missing_docs)] // each field is its selector's answer, named after it
        pub struct $name {
            $( pub $field: $ty, )+
        }
    };
}

caps_struct! {
    /// The device's `NSUInteger` properties, one field per selector.
    Limits: u64;
    device_linear_texture_alignment_bytes,
    iosurface_read_only_texture_alignment_bytes,
    iosurface_texture_alignment_bytes,
    linear_texture_alignment_bytes,
    max_compute_buffers,
    max_compute_local_memory_sizes,
    max_compute_samplers,
    max_compute_textures,
    max_compute_threadgroup_memory,
    max_compute_threadgroup_memory_alignment_bytes,
    max_fragment_buffers,
    max_fragment_samplers,
    max_fragment_textures,
    max_predicated_nesting_depth,
    max_rasterization_rate_layer_count,
    max_texture_height_2d,
    max_texture_layers,
    max_texture_width_2d,
    max_threadgroup_memory_length,
    max_tile_buffers,
    max_tile_samplers,
    max_tile_textures,
    max_total_compute_threads_per_threadgroup,
    max_vertex_buffers,
    max_vertex_samplers,
    max_vertex_textures,
    max_viewport_count,
}

caps_struct! {
    /// The device's `BOOL` properties, one field per selector.
    Flags: bool;
    are_raster_order_groups_supported,
    is_depth24_stencil8_pixel_format_supported,
    is_framebuffer_read_supported,
    is_large_mrt_supported,
    is_rgb10a2_gamma_supported,
    supports_buffer_with_address_ranges,
    supports_buffer_with_iosurface,
    supports_command_buffer_jump,
    supports_compute_compressed_texture_write,
    supports_conditional_load_store,
    supports_dynamic_attribute_stride,
    supports_extended_xr10_formats,
    supports_float16_bcubic_filtering,
    supports_heap_with_address_ranges,
    supports_image_blocks,
    supports_memory_order_atomics,
    supports_native_hardware_fp16,
    supports_non_square_tile_shaders,
    supports_programmable_sample_positions,
    supports_resource_detach_backing,
    supports_shared_texture_handles,
    supports_simd_reduction,
    supports_simd_shuffle_and_fill,
    supports_texture_2d_multisample_array,
    supports_tile_shaders,
}

/// Everything core asks a device that does not change while it lives.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCaps {
    /// `name`; `None` for nil.
    pub name: Option<String>,
    /// `registryID`.
    pub registry_id: u64,
    /// `gpuCoreCount`; `None` when the device does not respond to it.
    pub gpu_core_count: Option<u64>,
    /// `maxThreadsPerThreadgroup`.
    pub max_threads_per_threadgroup: MtlSize,
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u64,
    /// The other `NSUInteger` properties.
    pub limits: Limits,
    /// The `BOOL` properties.
    pub flags: Flags,
    /// `supportsTextureSampleCount:` of each of [`SAMPLE_COUNTS`].
    pub sample_counts: [RaiseResult<bool>; 4],
    /// `supportsPrimitiveType:` of `0..10`.
    pub primitive_types: [RaiseResult<bool>; 10],
    /// `supportsFamily:` of each of [`FAMILIES`].
    pub families: [RaiseResult<bool>; 7],
    /// `supportsTextureWriteRoundingMode:` of each of [`WRITE_ROUNDING_MODES`].
    pub write_rounding_modes: [RaiseResult<bool>; 2],
    /// `heapBufferSizeAndAlignWithLength:options:` of (`vm_page_size`, [`HEAP_QUERY_OPTIONS`]).
    pub heap_buffer_page_private: RaiseResult<SizeAndAlign>,
    /// `minimumLinearTextureAlignmentForPixelFormat:` of [`LINEAR_ALIGNMENT_FORMAT`].
    pub min_linear_alignment_bgra8: RaiseResult<u64>,
}

/// Every `MTLGPUFamily` of the macOS 27 SDK, in its order: Apple1 to Apple11, Mac1 and Mac2,
/// Common1 to Common3, MacCatalyst1 and MacCatalyst2, Metal3 and Metal4. [`IdentityCaps`]
/// answers `supportsFamily:` of each.
pub const SDK_FAMILIES: [i64; 20] = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 2001, 2002, 3001, 3002, 3003,
    4001, 4002, 5001, 5002,
];

/// Every `MTLFeatureSet` of the SDK, in its order: iOS 0 to 16, macOS 10000 to 10005, tvOS 30000
/// to 30005. [`IdentityCaps`] answers `supportsFeatureSet:` of each.
pub const FEATURE_SETS: [u64; 29] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 10000, 10001, 10002, 10003, 10004,
    10005, 30000, 30001, 30002, 30003, 30004, 30005,
];

/// The family of [`SDK_FAMILIES`] an `MTLFeatureSet` of [`FEATURE_SETS`] goes with: a device answers
/// the set where it answers the family, as Metal answers on the M3 Max (macOS 27, measured). The iOS
/// sets go with the Apple family of their number, macOS' 1_v1 to v4 with Mac1 and 2_v1 with Mac2;
/// `None` for the sets no family brings (macOS_ReadWriteTextureTier2 and the tvOS sets), which the
/// M3 Max answers no.
pub const fn feature_set_family(set: u64) -> Option<i64> {
    match set {
        // iOS_GPUFamily1_v1 to v5.
        0 | 2 | 5 | 8 | 12 => Some(1001),
        // iOS_GPUFamily2_v1 to v5.
        1 | 3 | 6 | 9 | 13 => Some(1002),
        // iOS_GPUFamily3_v1 to v4.
        4 | 7 | 10 | 14 => Some(1003),
        // iOS_GPUFamily4_v1 and v2.
        11 | 15 => Some(1004),
        // iOS_GPUFamily5_v1.
        16 => Some(1005),
        // macOS_GPUFamily1_v1 to v4.
        10000 | 10001 | 10003 | 10004 => Some(2001),
        // macOS_GPUFamily2_v1.
        10005 => Some(2002),
        _ => None,
    }
}

/// The `MTLCounterSamplingPoint`s [`IdentityCaps`] answers `supportsCounterSampling:` of, in order:
/// at stage, draw, dispatch, tile and blit boundaries.
pub const COUNTER_SAMPLING_POINTS: [u64; 5] = [0, 1, 2, 3, 4];

/// The largest count `supportsVertexAmplificationCount:` and
/// `supportsRasterizationRateMapWithLayerCount:` are asked about: each answer of
/// [`IdentitySizes`] is the largest count up to it the query takes.
pub const LARGEST_COUNT_ASKED: u64 = 16;

/// Every `MTLPixelFormat` of the macOS 27 SDK but Invalid, in its order. [`IdentityCaps`] answers
/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:` of each.
pub const SDK_PIXEL_FORMATS: [u64; 151] = [
    1, 10, 11, 12, 13, 14, 20, 22, 23, 24, 25, 30, 31, 32, 33, 34, 40, 41, 42, 43, 45, 46, 47, 48,
    53, 54, 55, 60, 62, 63, 64, 65, 70, 71, 72, 73, 74, 80, 81, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    99, 103, 104, 105, 110, 112, 113, 114, 115, 120, 121, 122, 123, 124, 125, 130, 131, 132, 133,
    134, 135, 140, 141, 142, 143, 150, 151, 152, 153, 160, 161, 162, 163, 164, 165, 166, 167, 170,
    172, 174, 176, 178, 179, 180, 181, 182, 183, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196,
    197, 198, 199, 200, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 222,
    223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 240, 241, 250, 252, 253, 255,
    260, 261, 262, 263, 552, 553, 554, 555,
];

/// The formats of [`SDK_PIXEL_FORMATS`] Metal answers `minimumTextureBufferAlignmentForPixelFormat:`
/// of, as it does `minimumLinearTextureAlignmentForPixelFormat:`: the uncompressed colour
/// formats that are neither packed nor YUV (measured on an Apple M3 Max, macOS 27.2: 16 for each,
/// and a failed assertion, which ends the process, for every other format of the SDK, so no
/// other is asked).
pub const TEXTURE_BUFFER_FORMATS: [u64; 56] = [
    1, 10, 11, 12, 13, 14, 20, 22, 23, 24, 25, 30, 31, 32, 33, 34, 40, 41, 42, 43, 53, 54, 55, 60,
    62, 63, 64, 65, 70, 71, 72, 73, 74, 80, 81, 90, 91, 92, 93, 94, 103, 104, 105, 110, 112, 113,
    114, 115, 123, 124, 125, 263, 552, 553, 554, 555,
];

/// The sample counts [`FormatAnswers::sparse_tiles`] answers, in order.
pub const SPARSE_SAMPLE_COUNTS: [u64; 3] = [1, 2, 4];

/// The `MTLSparsePageSize`s [`FormatAnswers::sparse_tiles`] answers, in order: 16, 64 and 256
/// KiB. The query without a page size answers the first's.
pub const SPARSE_PAGE_SIZES: [i64; 3] = [101, 102, 103];

/// One pixel format's answers in [`IdentityCaps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FormatAnswers {
    /// The raw `MTLPixelFormat`.
    pub format: u64,
    /// `minimumTextureBufferAlignmentForPixelFormat:`; `None` for a format Metal does not answer
    /// (it asserts) or where the platform has no answer.
    pub texture_buffer_alignment: Option<u64>,
    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:` of a 2D texture,
    /// width, height and depth, at each of [`SPARSE_SAMPLE_COUNTS`] (major) and each of
    /// [`SPARSE_PAGE_SIZES`]: the same for every texture type (measured on the M3 Max); `None`
    /// where the platform has no answer.
    pub sparse_tiles: Option<[[u64; 3]; 9]>,
}

caps_struct! {
    /// The device's identity and size answers [`IdentityCaps`] holds, one field per selector:
    /// `None` where the platform has no answer of its own. The last four are `MTLDeviceSPI`'s
    /// `featureProfile`, `halfFPConfig`, `singleFPConfig` and `doubleFPConfig`, which SkyLight
    /// asks (measured on the M3 Max, macOS 27: 10, 0xe, 0x54 and 0).
    IdentitySizes: Option<u64>;
    recommended_max_working_set_size,
    location,
    location_number,
    max_transfer_rate,
    peer_group_id,
    peer_index,
    peer_count,
    max_buffer_length,
    max_argument_buffer_sampler_count,
    argument_buffers_support,
    read_write_texture_support,
    sparse_tile_size_in_bytes,
    maximum_concurrent_compilation_task_count,
    largest_vertex_amplification_count,
    largest_rasterization_rate_map_layer_count,
    feature_profile,
    half_fp_config,
    single_fp_config,
    double_fp_config,
}

caps_struct! {
    /// The device's `BOOL` identity and capability answers [`IdentityCaps`] holds, one field per
    /// selector: `None` where the platform has no answer of its own.
    IdentityFlags: Option<bool>;
    is_low_power,
    is_headless,
    is_removable,
    has_unified_memory,
    supports_32bit_float_filtering,
    supports_32bit_msaa,
    supports_query_texture_lod,
    supports_bc_texture_compression,
    supports_pull_model_interpolation,
    are_barycentric_coords_supported,
    supports_shader_barycentric_coordinates,
    supports_dynamic_libraries,
    supports_render_dynamic_libraries,
    supports_raytracing,
    supports_function_pointers,
    supports_function_pointers_from_render,
    supports_raytracing_from_render,
    supports_primitive_motion_blur,
    supports_placement_sparse,
}

/// The device's answers a guest's own `MTLDevice` gives beyond [`DeviceCaps`]: `supportsFamily:`
/// of every family of the SDK, `supportsFeatureSet:` of every feature set, and the identity and
/// capability getters. Only the graphics lane asks it ([`GpuDevice::identity_caps`]), once per
/// device; core's guest ABI never reads it.
///
/// Each answer is the platform's own: Metal's getter on the Metal platform, the runtime's answer
/// on the Vulkan platform. `None` is a query the platform has no answer of, which the asker must
/// not read as support.
///
/// [`GpuDevice::identity_caps`]: super::device::GpuDevice::identity_caps
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IdentityCaps {
    /// `supportsFamily:` of each of [`SDK_FAMILIES`].
    pub families: [Option<bool>; 20],
    /// `supportsFeatureSet:` of each of [`FEATURE_SETS`].
    pub feature_sets: [Option<bool>; 29],
    /// `supportsCounterSampling:` of each of [`COUNTER_SAMPLING_POINTS`].
    pub counter_sampling: [Option<bool>; 5],
    /// `architecture.name`.
    pub architecture_name: Option<String>,
    /// The answers of each of [`SDK_PIXEL_FORMATS`], in its order.
    pub formats: Vec<FormatAnswers>,
    /// The integer answers.
    pub sizes: IdentitySizes,
    /// The `BOOL` answers.
    pub flags: IdentityFlags,
}

/// `r` again: its value, or its raise thrown here, as `on` says.
fn again<T: Clone>(on: OnFault, r: &RaiseResult<T>) -> RaiseResult<T> {
    match r {
        Ok(v) => Ok(v.clone()),
        Err(e) => on.throw(e.clone()),
    }
}

impl DeviceCaps {
    /// `heapBufferSizeAndAlignWithLength:options:` as read, its raise thrown again here as `on`
    /// says: the device info writer asks it both under a handler and with none.
    pub fn heap_buffer_page_private(&self, on: OnFault) -> RaiseResult<SizeAndAlign> {
        again(on, &self.heap_buffer_page_private)
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:` as read, its raise thrown again here. Only
    /// the EFI display's registers, with no handler, ask it, so the raise dies here.
    pub fn min_linear_alignment_bgra8(&self) -> RaiseResult<u64> {
        again(OnFault::Die, &self.min_linear_alignment_bgra8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::Raise;
    use crate::testkit::impls::device::fake_caps;

    /// A raise kept in the caps is raised again where it is asked: the caller's under a handler,
    /// and a death with none (the alignment is asked only with none); a kept answer is the
    /// answer
    #[test]
    fn a_raise_is_kept_and_raised_again() {
        let raised = Raise::foreign("heapBufferSizeAndAlignWithLength:options:").with_detail("no");
        let caps = DeviceCaps {
            heap_buffer_page_private: Err(raised.clone()),
            min_linear_alignment_bgra8: Err(Raise::foreign(
                "minimumLinearTextureAlignmentForPixelFormat:",
            )
            .with_detail("refused")),
            ..fake_caps()
        };
        assert_eq!(caps.heap_buffer_page_private(OnFault::Return), Err(raised));
        assert_eq!(
            fake_caps().heap_buffer_page_private(OnFault::Die),
            Ok(SizeAndAlign {
                size: 0x4000,
                align: 0x4000
            })
        );
        let death = crate::testkit::capture::catch_death(|| {
            let _ = caps.min_linear_alignment_bgra8();
        });
        assert_eq!(
            death.map_err(|d| d.file),
            Err("minimumLinearTextureAlignmentForPixelFormat:")
        );
        let death = crate::testkit::capture::catch_death(|| {
            let _ = caps.heap_buffer_page_private(OnFault::Die);
        });
        assert!(death.is_err());
    }
}
