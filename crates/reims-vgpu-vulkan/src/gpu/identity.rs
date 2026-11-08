//! `IdentityCaps` on the Vulkan platform: the families, feature sets, identity and capability
//! answers a guest's own `MTLDevice` gives beyond `DeviceCaps`, read for the graphics lane.
//!
//! Each answer comes from one of these sources, and is `None` where none gives it:
//! - **the runtime's answer**: `name`'s companions in `DeviceProperties` (`hasUnifiedMemory`,
//!   `maxBufferLength`, the argument-buffer and read-write texture tiers,
//!   `supports32BitFloatFiltering`), `sparseTileSizeInBytes`, each format's sparse tiles,
//!   `supportsFamily:`, `supportsVertexAmplificationCount:` and
//!   `supportsRasterizationRateMapWithLayerCount:`;
//! - **a stand-in**, [`crate::gpu::caps`]' (its table), for
//!   `minimumTextureBufferAlignmentForPixelFormat:`: `minimumLinearTextureAlignmentForPixelFormat:`
//!   as this platform answers it ([`crate::gpu::caps::linear_alignment`]), which Metal answers
//!   alike (measured);
//! - **the runtime's audit of a feature row** of Apple's tables, in the Apple9 column, which lists
//!   every row read here, as [`crate::gpu::caps`]' SIMD flags are read ([`ROWS`]): a getter is yes
//!   where the row its feature is is exact on the device;
//! - **the device's Vulkan memory heaps**, for `recommendedMaxWorkingSetSize`, which the runtime
//!   does not answer: the sum of the device-local heaps, the memory the GPU reaches at its own
//!   speed, as Metal answers on a Mac of a discrete GPU from its VRAM;
//! - **the runtime's device properties**, for `maxArgumentBufferSamplerCount`: the most sampler
//!   states an argument buffer can name at once, as the runtime places them in its set-1 sampler
//!   arrays (`DeviceProperties::max_argument_buffer_sampler_count`);
//! - **Metal's documented relations**:
//!   - a family the platform claims claims the families it contains (`implied`): each Apple
//!     family every Apple family before it, Apple1 Common1, Apple3 Common2, Apple5 Common3, and
//!     Apple7 Mac1, Mac2 and Metal3 (the SDK's replacements of the deprecated families, and
//!     Metal3's own Apple7 floor). Apple5 to Apple10 are the platform's `supportsFamily:`
//!     answers in `DeviceCaps` (the stand-ins of [`crate::gpu::caps`]), so a guest of either lane
//!     on one host is told the same families;
//!   - each feature set is its family's, as Metal on an Apple M3 Max answers them (measured,
//!     macOS 27.2, `guest/bundle/tools/metal_enum.m`): every iOS feature set of Apple1 to Apple5
//!     with its family, `macOS_GPUFamily1` with Mac1, `macOS_GPUFamily2` with Mac2, and none of
//!     `macOS_ReadWriteTextureTier2` (10002) and the tvOS ones (`feature_set`).
//!
//! **The runtime gives no answer** of `isLowPower`, `isHeadless`, `isRemovable`, `location`,
//! `locationNumber`, `maxTransferRate`, the peer group, `maxArgumentBufferSamplerCount`,
//! `maximumConcurrentCompilationTaskCount`, `architecture.name`, `supportsPrimitiveMotionBlur`,
//! `supportsPlacementSparse`, `supportsCounterSampling:` and `recommendedMaxWorkingSetSize`: each
//! is `None` here (the working set the heaps' sum, the sampler count the one its sampler arrays
//! hold), and a query of the runtime's to come.
//! `supports32BitMSAA` and `supportsPullModelInterpolation` are the runtime's answers; a gap in the
//! latter is `None`.

use ash::vk;
use metal2vulkan::metal::TextureType;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::caps::{
    DeviceCaps, FEATURE_SETS, FormatAnswers, IdentityCaps, IdentityFlags, IdentitySizes,
    LARGEST_COUNT_ASKED, SDK_FAMILIES, SDK_PIXEL_FORMATS, SPARSE_PAGE_SIZES, SPARSE_SAMPLE_COUNTS,
    TEXTURE_BUFFER_FORMATS, feature_set_family,
};

use crate::gpu::runtime_pixel_format;

/// The feature rows of Apple's tables the getters are read from, each with its getter's field
/// in [`IdentityFlags`], in the order `flags` reads them.
const ROWS: [rt::FeatureRow; 9] = [
    rt::FeatureRow::QueryTextureLod,
    rt::FeatureRow::BcPixelFormats,
    rt::FeatureRow::BarycentricCoordinates,
    rt::FeatureRow::ComputeDynamicLibraries,
    rt::FeatureRow::RenderDynamicLibraries,
    rt::FeatureRow::RayTracingCompute,
    rt::FeatureRow::RayTracingRender,
    rt::FeatureRow::FunctionPointersCompute,
    rt::FeatureRow::FunctionPointersRender,
];

/// The first of [`SDK_FAMILIES`] the platform's `DeviceCaps` answers: Apple5, then Apple6 to
/// Apple10, as core's `FAMILIES` asks them.
const DEVICE_CAPS_FIRST: usize = 4;

/// Whether each of [`ROWS`] is exact on `device`, in the runtime's audit of the Apple9 column;
/// `None` for a row the column does not list.
fn rows_exact(device: &rt::Device) -> [Option<bool>; 9] {
    let audit = device.family_rows(rt::GpuFamily::Apple9);
    ROWS.map(|row| {
        audit
            .iter()
            .find(|a| a.row == rt::FamilyRow::Feature(row))
            .map(|a| a.state == rt::RowState::Exact)
    })
}

/// The index in [`SDK_FAMILIES`] of `family`.
fn at(family: i64) -> usize {
    SDK_FAMILIES
        .iter()
        .position(|&f| f == family)
        .unwrap_or_else(|| unreachable!("{family} is an SDK family"))
}

/// The families each claimed family contains, by Metal's documented relations (the module docs).
fn implied(family: i64) -> &'static [i64] {
    match family {
        1001 => &[3001],
        1002 => &[1001],
        1003 => &[1002, 3002],
        1004 => &[1003],
        1005 => &[1004, 3003],
        1006 => &[1005],
        1007 => &[1006, 2001, 2002, 5001],
        1008 => &[1007],
        1009 => &[1008],
        1010 => &[1009],
        1011 => &[1010],
        _ => &[],
    }
}

/// `families` with every family a claimed one contains claimed too.
fn closed(mut families: [bool; 20]) -> [bool; 20] {
    loop {
        let before = families;
        for (i, &family) in SDK_FAMILIES.iter().enumerate() {
            if families[i] {
                for &contained in implied(family) {
                    families[at(contained)] = true;
                }
            }
        }
        if families == before {
            return families;
        }
    }
}

/// `supportsFamily:` of each of [`SDK_FAMILIES`]: the platform's `DeviceCaps` answer for Apple5 to
/// Apple10, the runtime's for the others, closed under [`implied`].
fn families(device: &rt::Device, caps: &DeviceCaps) -> [bool; 20] {
    let answers = std::array::from_fn(|i| {
        // The first six of core's families are Apple5 to Apple10; the seventh is -1009.
        match i.checked_sub(DEVICE_CAPS_FIRST).filter(|&k| k < 6) {
            Some(k) => matches!(caps.families[k], Ok(true)),
            None => device.supports_family(SDK_FAMILIES[i]),
        }
    });
    closed(answers)
}

/// `supportsFeatureSet:` of `set` on a device of `families`, as Metal answers on the M3 Max (the
/// module docs): the iOS feature sets of each Apple family, macOS' two families, and none of the
/// others ([`feature_set_family`]).
fn feature_set(set: u64, families: &[bool; 20]) -> bool {
    feature_set_family(set).is_some_and(|family| families[at(family)])
}

/// The largest count in `1..=LARGEST_COUNT_ASKED` `accepts` takes, counting up from 1 until it
/// refuses one or answers a gap; 0 where it refuses 1.
fn largest(accepts: impl Fn(u64) -> Result<bool, rt::RuntimeGap>) -> u64 {
    (1..=LARGEST_COUNT_ASKED)
        .take_while(|&count| matches!(accepts(count), Ok(true)))
        .last()
        .unwrap_or(0)
}

/// The runtime's page of Metal's raw `MTLSparsePageSize` `page`.
fn sparse_page(page: i64) -> Option<rt::SparsePageSize> {
    [
        rt::SparsePageSize::Size16,
        rt::SparsePageSize::Size64,
        rt::SparsePageSize::Size256,
    ]
    .into_iter()
    .zip(SPARSE_PAGE_SIZES)
    .find_map(|(p, raw)| (raw == page).then_some(p))
}

/// The answers of `format` [`FormatAnswers`] holds: the runtime's sparse tiles, and, of a format
/// Metal answers it of ([`TEXTURE_BUFFER_FORMATS`]), the alignment of its texture buffers:
/// `minimumLinearTextureAlignmentForPixelFormat:` as this platform answers it, a stand-in
/// ([`crate::gpu::caps::linear_alignment`]), which Metal answers alike for every format of the
/// SDK (measured, an M3 Max under macOS 27.2). `None` for a format the runtime has no enumerator
/// of.
fn format_answers(device: &rt::Device, format: u64) -> FormatAnswers {
    let Some(pf) = runtime_pixel_format(format) else {
        return FormatAnswers {
            format,
            ..FormatAnswers::default()
        };
    };
    let texture_buffer_alignment = TEXTURE_BUFFER_FORMATS
        .contains(&format)
        .then(|| crate::gpu::caps::linear_alignment(device, pf).ok())
        .flatten();
    let mut tiles = [[0; 3]; 9];
    let pairs = SPARSE_SAMPLE_COUNTS
        .iter()
        .flat_map(|&samples| SPARSE_PAGE_SIZES.iter().map(move |&page| (samples, page)));
    for (tile, (samples, page)) in tiles.iter_mut().zip(pairs) {
        if let Some(page) = sparse_page(page) {
            let size = device.sparse_tile_size_for_page(TextureType::Type2D, pf, samples, page);
            *tile = [size.width, size.height, size.depth];
        }
    }
    FormatAnswers {
        format,
        texture_buffer_alignment,
        sparse_tiles: Some(tiles),
    }
}

/// The sum of `device`'s device-local memory heaps.
fn device_local_bytes(device: &rt::Device) -> u64 {
    // SAFETY: the instance and physical device are the runtime's, read only, while `device`
    // lives.
    let memory = unsafe {
        let raw = device.raw();
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    memory
        .memory_heaps_as_slice()
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .fold(0u64, |sum, heap| sum.saturating_add(heap.size))
}

/// The bytes of `device`'s device-local memory heaps the process may hold now
/// ([`GpuDevice::own_memory_budget`]): the heaps' budgets (`VK_EXT_memory_budget`) where the
/// physical device lists the extension, which a physical-device query may chain without the device
/// enabling it, else the heaps' sizes ([`device_local_bytes`]).
///
/// [`GpuDevice::own_memory_budget`]: reims_vgpu_core::platform::device::GpuDevice::own_memory_budget
pub(crate) fn device_local_budget(device: &rt::Device) -> u64 {
    // SAFETY: the instance and physical device are the runtime's, read only, while `device`
    // lives.
    let lists = unsafe {
        let raw = device.raw();
        raw.instance
            .enumerate_device_extension_properties(raw.physical_device)
            .unwrap_or_default()
            .iter()
            .any(|p| p.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME))
    };
    if !lists {
        return device_local_bytes(device);
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut memory = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: as above; the one structure chained is of an extension the device lists.
    unsafe {
        let raw = device.raw();
        raw.instance
            .get_physical_device_memory_properties2(raw.physical_device, &mut memory);
    }
    let properties = memory.memory_properties;
    (properties.memory_heaps_as_slice().iter())
        .zip(budget.heap_budget)
        .filter(|(heap, _)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .fold(0u64, |sum, (_, budget)| sum.saturating_add(budget))
}

/// `maxArgumentBufferSamplerCount` of `device`: how many sampler states an argument buffer can
/// name at once, as the runtime answers it (`DeviceProperties::max_argument_buffer_sampler_count`:
/// its set-1 sampler arrays' elements and `maxSamplerAllocationCount`, less the neutral sampler's
/// one), where Metal answers 500000 on an M3 Max and 2048 on an AMD Radeon Pro 5300M and an Intel
/// UHD Graphics 630 (macOS 26.7.1; docs/graphics-kext/dgpu-oracle.md, Q9).
fn argument_buffer_sampler_count(device: &rt::Device) -> u64 {
    device.properties().max_argument_buffer_sampler_count
}

/// `architecture.name` of a device of PCI vendor `vendor` and device `device` whose driver
/// keys its compiled pipelines by `pipeline_cache`: `vulkan_VVVV_DDDD_` and the key in hex.
///
/// The name keys the guest's binary archives, so it stays the same for one GPU and driver from
/// boot to boot, and differs where compiled pipelines differ: between GPUs, and between drivers
/// or driver builds, which Vulkan's `pipelineCacheUUID` tells apart (a pipeline cache of another
/// UUID is one the driver does not take). The name claims no Apple GPU (Metal's are
/// `applegpu_gNN`); the device's families say what it supports.
fn architecture_name(vendor: u32, device: u32, pipeline_cache: [u8; vk::UUID_SIZE]) -> String {
    let key: String = pipeline_cache.iter().map(|b| format!("{b:02x}")).collect();
    format!("vulkan_{vendor:04x}_{device:04x}_{key}")
}

/// `VkPhysicalDeviceProperties::pipelineCacheUUID` of `device`.
fn pipeline_cache_uuid(device: &rt::Device) -> [u8; vk::UUID_SIZE] {
    // SAFETY: the instance and physical device are the runtime's, read only, while `device`
    // lives.
    unsafe {
        let raw = device.raw();
        raw.instance
            .get_physical_device_properties(raw.physical_device)
            .pipeline_cache_uuid
    }
}

/// `featureProfile`, `halfFPConfig`, `singleFPConfig` and `doubleFPConfig` measured on the M3 Max
/// (macOS 27), the one device they are measured on.
const M3_MAX_DRIVER_SIZES: [u64; 4] = [10, 0xe, 0x54, 0];

/// `featureProfile`, `halfFPConfig`, `singleFPConfig` and `doubleFPConfig` (private; SkyLight asks
/// them) of the device this platform answers registry ID `registry_id` for:
/// - on macOS, for the device this platform answers Metal's registry ID for, Metal's own answers
///   for that GPU ([`crate::os::default_driver_sizes`]);
/// - else the M3 Max's ([`M3_MAX_DRIVER_SIZES`]). No Vulkan query reports them, and what Metal
///   answers for a family other than the M3 Max's is not measured, so every other device answers
///   the one measured device's values (a gap, in the graphics lane's selector sweep).
fn driver_sizes(registry_id: u64) -> [Option<u64>; 4] {
    let metals = if crate::os::default_registry_id() == Some(registry_id) {
        crate::os::default_driver_sizes()
    } else {
        [None; 4]
    };
    let mut sizes = [None; 4];
    for ((size, metal), m3) in sizes.iter_mut().zip(metals).zip(M3_MAX_DRIVER_SIZES) {
        *size = Some(metal.unwrap_or(m3));
    }
    sizes
}

/// Every identity answer of `device`, whose `DeviceCaps` are `caps`, read once.
pub fn read(device: &rt::Device, caps: &DeviceCaps) -> IdentityCaps {
    let props = device.properties();
    let [
        feature_profile,
        half_fp_config,
        single_fp_config,
        double_fp_config,
    ] = driver_sizes(caps.registry_id);
    let families = families(device, caps);
    let [
        lod,
        bc,
        barycentrics,
        dynamic,
        render_dynamic,
        rt_compute,
        rt_render,
        fp,
        fp_render,
    ] = rows_exact(device);
    IdentityCaps {
        families: families.map(Some),
        feature_sets: FEATURE_SETS.map(|set| Some(feature_set(set, &families))),
        counter_sampling: [None; 5],
        architecture_name: Some(architecture_name(
            props.vendor_id,
            props.device_id,
            pipeline_cache_uuid(device),
        )),
        formats: SDK_PIXEL_FORMATS
            .iter()
            .map(|&format| format_answers(device, format))
            .collect(),
        sizes: IdentitySizes {
            recommended_max_working_set_size: Some(device_local_bytes(device)),
            max_buffer_length: Some(props.max_buffer_length),
            max_argument_buffer_sampler_count: Some(argument_buffer_sampler_count(device)),
            argument_buffers_support: Some(u64::from(props.argument_buffers_support.as_raw())),
            read_write_texture_support: Some(u64::from(props.read_write_texture_support.as_raw())),
            sparse_tile_size_in_bytes: Some(device.sparse_tile_size_in_bytes()),
            largest_vertex_amplification_count: Some(largest(|count| {
                props.supports_vertex_amplification_count(count)
            })),
            largest_rasterization_rate_map_layer_count: Some(largest(|count| {
                props.supports_rasterization_rate_map_with_layer_count(count)
            })),
            feature_profile,
            half_fp_config,
            single_fp_config,
            double_fp_config,
            ..IdentitySizes::default()
        },
        flags: IdentityFlags {
            has_unified_memory: Some(props.has_unified_memory),
            supports_32bit_float_filtering: Some(props.supports_32bit_float_filtering),
            supports_query_texture_lod: lod,
            supports_bc_texture_compression: bc,
            are_barycentric_coords_supported: barycentrics,
            supports_shader_barycentric_coordinates: barycentrics,
            supports_dynamic_libraries: dynamic,
            supports_render_dynamic_libraries: render_dynamic,
            supports_raytracing: rt_compute,
            supports_raytracing_from_render: rt_render,
            supports_function_pointers: fp,
            supports_function_pointers_from_render: fp_render,
            supports_32bit_msaa: Some(device.supports_32bit_msaa()),
            supports_pull_model_interpolation: props.supports_pull_model_interpolation().ok(),
            ..IdentityFlags::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::device::GpuDevice;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A device no Metal device stands for answers the M3 Max's feature profile and FP configs;
    /// the one Metal's registry ID names, Metal's own (on macOS)
    #[test]
    fn driver_sizes_are_metals_or_the_m3_maxs() {
        let elsewhere = crate::os::default_registry_id().map_or(1, |id| id ^ 1);
        assert_eq!(driver_sizes(elsewhere), M3_MAX_DRIVER_SIZES.map(Some));
        if let Some(id) = crate::os::default_registry_id() {
            let metals = crate::os::default_driver_sizes();
            let answered = driver_sizes(id);
            for ((answer, metal), m3) in answered.iter().zip(metals).zip(M3_MAX_DRIVER_SIZES) {
                assert_eq!(*answer, Some(metal.unwrap_or(m3)));
            }
        }
    }

    /// A claimed family claims every family Metal documents it contains, and nothing more: Apple9
    /// claims Apple1 to Apple8, the Common families, Mac1, Mac2 and Metal3, but not Apple10,
    /// Apple11, the Mac Catalyst families or Metal4
    #[test]
    fn a_family_claims_the_families_it_contains() {
        let mut families = [false; 20];
        families[at(1009)] = true;
        let claimed: Vec<i64> = SDK_FAMILIES
            .iter()
            .zip(closed(families))
            .filter_map(|(&f, yes)| yes.then_some(f))
            .collect();
        assert_eq!(
            claimed,
            [
                1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 2001, 2002, 3001, 3002, 3003,
                5001
            ]
        );
        assert_eq!(closed([false; 20]), [false; 20]);
    }

    /// The architecture's name is the GPU's vendor and device and its driver's pipeline-cache
    /// key, in hex, under a prefix no Apple GPU's name has
    #[test]
    fn the_architecture_names_the_gpu_and_its_compiled_pipelines() {
        let mut key = [0u8; vk::UUID_SIZE];
        key[0] = 0xab;
        key[15] = 0x01;
        let name = architecture_name(0x10de, 0x2c58, key);
        assert_eq!(name, "vulkan_10de_2c58_ab000000000000000000000000000001");
        assert!(name.len() < 64);
        key[15] = 0x02;
        assert_ne!(architecture_name(0x10de, 0x2c58, key), name);
        assert_ne!(
            architecture_name(0x1002, 0x2c58, [0; vk::UUID_SIZE]),
            architecture_name(0x10de, 0x2c58, [0; vk::UUID_SIZE])
        );
    }

    /// The feature sets of the M3 Max's families are the M3 Max's (measured, macOS 27.2): every
    /// iOS one and the macOS ones but 10002, no tvOS one
    #[test]
    fn feature_sets_are_the_m3_maxs_of_its_families() {
        let mut families = [false; 20];
        for f in [
            1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 2001, 2002,
        ] {
            families[at(f)] = true;
        }
        let answers: Vec<u64> = FEATURE_SETS
            .iter()
            .copied()
            .filter(|&set| feature_set(set, &families))
            .collect();
        let mut expected: Vec<u64> = (0..=16).collect();
        expected.extend([10000, 10001, 10003, 10004, 10005]);
        assert_eq!(answers, expected);
    }

    /// The device's answers: its families are the platform's Apple5 to Apple10 and what they
    /// contain, so every family `DeviceCaps` claims is claimed here too, and the runtime's
    /// properties are the ones it answers
    #[test]
    fn identity_is_the_runtimes_and_the_platforms() {
        let d = dev();
        let caps = d.caps();
        let identity = d.identity_caps();
        for (k, answer) in caps.families.iter().take(6).enumerate() {
            assert_eq!(
                identity.families[DEVICE_CAPS_FIRST + k],
                Some(matches!(answer, Ok(true)))
            );
        }
        let props = d.raw.properties();
        assert_eq!(
            identity.flags.has_unified_memory,
            Some(props.has_unified_memory)
        );
        assert_eq!(
            identity.sizes.max_buffer_length,
            Some(props.max_buffer_length)
        );
        assert!(
            identity
                .sizes
                .recommended_max_working_set_size
                .is_some_and(|bytes| bytes > 0)
        );
        assert_eq!(identity.flags.is_low_power, None);
        // The sampler states an argument buffer names: the runtime's answer, never none on a
        // device whose functions read argument buffers' samplers, and within the device's sampler
        // allocations and the handle search's reach.
        // SAFETY: the instance and physical device are the runtime's, read only, while `d` lives.
        let allocations = unsafe {
            let raw = d.raw.raw();
            raw.instance
                .get_physical_device_properties(raw.physical_device)
                .limits
                .max_sampler_allocation_count
        };
        let samplers = identity.sizes.max_argument_buffer_sampler_count.unwrap();
        assert_eq!(
            samplers,
            d.raw.properties().max_argument_buffer_sampler_count
        );
        assert!(samplers > 0);
        assert!(samplers < u64::from(allocations));
        assert!(samplers < u64::from(metal2vulkan::bind::HANDLE_TABLE_CAPACITY));
        // BGRA8Unorm (80): 16-byte texture buffers, and a 4-byte texel's 64x64 tile in 16 KiB,
        // as Metal answers on the M3 Max; a format Metal asserts on (BC1_RGBA, 130) has none.
        let bgra8 = identity.formats.iter().find(|f| f.format == 80).unwrap();
        assert_eq!(bgra8.texture_buffer_alignment, Some(16));
        assert_eq!(bgra8.sparse_tiles.unwrap()[0], [64, 64, 1]);
        let bc1 = identity.formats.iter().find(|f| f.format == 130).unwrap();
        assert_eq!(bc1.texture_buffer_alignment, None);
    }

    /// The budget of the device's own memory is some of it, no more than its device-local heaps
    /// hold, as `VK_EXT_memory_budget` bounds each heap's budget by the heap's size
    #[test]
    fn the_own_memory_budget_is_within_the_device_local_heaps() {
        let d = dev();
        let budget = d.own_memory_budget();
        assert!(budget > 0);
        assert!(budget <= device_local_bytes(&d.raw));
    }
}
