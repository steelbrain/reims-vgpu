//! `IdentityCaps` on the Vulkan platform: the families, feature sets, identity and capability
//! answers a guest's own `MTLDevice` gives beyond `DeviceCaps`, read for the graphics lane.
//!
//! Each answer comes from one of these sources, and is `None` where none gives it:
//! - **the runtime's answer**: `name`'s companions in `DeviceProperties` (`hasUnifiedMemory`,
//!   `maxBufferLength`, the argument-buffer and read-write texture tiers,
//!   `supports32BitFloatFiltering`), `sparseTileSizeInBytes`, `supportsFamily:`,
//!   `supportsVertexAmplificationCount:` and `supportsRasterizationRateMapWithLayerCount:`;
//! - **the runtime's audit of a feature row** of Apple's tables, in the Apple9 column, which lists
//!   every row read here, as [`crate::gpu::caps`]' SIMD flags are read ([`ROWS`]): a getter is yes
//!   where the row its feature is is exact on the device;
//! - **the device's Vulkan memory heaps**, for `recommendedMaxWorkingSetSize`, which the runtime
//!   does not answer: the sum of the device-local heaps, the memory the GPU reaches at its own
//!   speed, as Metal answers on a Mac of a discrete GPU from its VRAM;
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
//! `maximumConcurrentCompilationTaskCount`, `architecture.name`, `supports32BitMSAA`,
//! `supportsPullModelInterpolation`, `supportsPrimitiveMotionBlur`, `supportsPlacementSparse`,
//! `supportsCounterSampling:` and `recommendedMaxWorkingSetSize`: each is `None` here (the working
//! set the heaps' sum), and a query of the runtime's to come.

use ash::vk;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::caps::{
    DeviceCaps, FEATURE_SETS, IdentityCaps, IdentityFlags, IdentitySizes, LARGEST_COUNT_ASKED,
    SDK_FAMILIES,
};

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
/// others.
fn feature_set(set: u64, families: &[bool; 20]) -> bool {
    let has = |family: i64| families[at(family)];
    match set {
        // iOS_GPUFamily1_v1 to v5.
        0 | 2 | 5 | 8 | 12 => has(1001),
        // iOS_GPUFamily2_v1 to v5.
        1 | 3 | 6 | 9 | 13 => has(1002),
        // iOS_GPUFamily3_v1 to v4.
        4 | 7 | 10 | 14 => has(1003),
        // iOS_GPUFamily4_v1 and v2.
        11 | 15 => has(1004),
        // iOS_GPUFamily5_v1.
        16 => has(1005),
        // macOS_GPUFamily1_v1 to v4.
        10000 | 10001 | 10003 | 10004 => has(2001),
        // macOS_GPUFamily2_v1.
        10005 => has(2002),
        // macOS_ReadWriteTextureTier2 and the tvOS feature sets.
        _ => false,
    }
}

/// The largest count in `1..=LARGEST_COUNT_ASKED` `accepts` takes, counting up from 1 until it
/// refuses one or answers a gap; 0 where it refuses 1.
fn largest(accepts: impl Fn(u64) -> Result<bool, rt::RuntimeGap>) -> u64 {
    (1..=LARGEST_COUNT_ASKED)
        .take_while(|&count| matches!(accepts(count), Ok(true)))
        .last()
        .unwrap_or(0)
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

/// Every identity answer of `device`, whose `DeviceCaps` are `caps`, read once.
pub fn read(device: &rt::Device, caps: &DeviceCaps) -> IdentityCaps {
    let props = device.properties();
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
        architecture_name: None,
        sizes: IdentitySizes {
            recommended_max_working_set_size: Some(device_local_bytes(device)),
            max_buffer_length: Some(props.max_buffer_length),
            argument_buffers_support: Some(u64::from(props.argument_buffers_support.as_raw())),
            read_write_texture_support: Some(u64::from(props.read_write_texture_support.as_raw())),
            sparse_tile_size_in_bytes: Some(device.sparse_tile_size_in_bytes()),
            largest_vertex_amplification_count: Some(largest(|count| {
                props.supports_vertex_amplification_count(count)
            })),
            largest_rasterization_rate_map_layer_count: Some(largest(|count| {
                props.supports_rasterization_rate_map_with_layer_count(count)
            })),
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
            ..IdentityFlags::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::device::GpuDevice;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A claimed family claims every family Metal documents it contains, and nothing more: Apple9
    /// claims Apple1 to Apple8, the Common families, Mac1, Mac2 and Metal3, but not Apple10,
    /// Apple11, the Mac Catalyst families or Metal4
    #[test]
    fn a_family_claims_the_families_it_contains() {
        let mut families = [false; 20];
        families[at(1009)] = true;
        let closed = closed(families);
        let claimed: Vec<i64> = SDK_FAMILIES
            .iter()
            .zip(closed)
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
    }
}
