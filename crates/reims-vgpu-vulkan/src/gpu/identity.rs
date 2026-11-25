//! The parts of `IdentityCaps` the rt platform's answers take from Metal's documented relations
//! and from the device's Vulkan memory heaps (`crate::rt_platform::answers`):
//! - a family the platform claims claims the families it contains (`implied`): each Apple
//!   family every Apple family before it, Apple1 Common1, Apple3 Common2, Apple5 Common3, and
//!   Apple7 Mac1, Mac2 and Metal3 (the SDK's replacements of the deprecated families, and
//!   Metal3's own Apple7 floor) ([`closed`]);
//! - each feature set is its family's, as Metal on an Apple M3 Max answers them (measured,
//!   macOS 27.2, `guest/bundle/tools/metal_enum.m`): every iOS feature set of Apple1 to Apple5
//!   with its family, `macOS_GPUFamily1` with Mac1, `macOS_GPUFamily2` with Mac2, and none of
//!   `macOS_ReadWriteTextureTier2` (10002) and the tvOS ones ([`feature_set`]);
//! - `recommendedMaxWorkingSetSize` is the sum of the device-local heaps, the memory the GPU
//!   reaches at its own speed, as Metal answers on a Mac of a discrete GPU from its VRAM
//!   ([`device_local_bytes_of`]), and the budget of the device's own memory its heaps' budgets
//!   ([`device_local_budget_of`]);
//! - `architecture.name` names the GPU and its driver's compiled pipelines
//!   ([`architecture_name`]), and the driver's private sizes are Metal's or the M3 Max's
//!   ([`driver_sizes`]).

use crate::core::platform::caps::{SDK_FAMILIES, feature_set_family};
use ash::vk;

/// The index in [`SDK_FAMILIES`] of `family`.
pub(crate) fn at(family: i64) -> usize {
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
pub(crate) fn closed(mut families: [bool; 20]) -> [bool; 20] {
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

/// `supportsFeatureSet:` of `set` on a device of `families`, as Metal answers on the M3 Max (the
/// module docs): the iOS feature sets of each Apple family, macOS' two families, and none of the
/// others ([`feature_set_family`]).
pub(crate) fn feature_set(set: u64, families: &[bool; 20]) -> bool {
    feature_set_family(set).is_some_and(|family| families[at(family)])
}

/// The bytes of the device-local memory heaps of the physical device `physical` of `instance`,
/// summed.
///
/// # Safety
///
/// `physical` is a physical device `instance` lists, and `instance` is live.
pub(crate) unsafe fn device_local_bytes_of(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> u64 {
    // SAFETY: the caller's.
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    memory
        .memory_heaps_as_slice()
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .fold(0u64, |sum, heap| sum.saturating_add(heap.size))
}

/// The bytes of the device-local memory heaps of the physical device `physical` of `instance` the
/// process may hold now (`GpuDevice::own_memory_budget`): the heaps' budgets
/// (`VK_EXT_memory_budget`) where the physical device lists the extension, which a physical-device
/// query may chain without the device enabling it, each no more than its heap's size, else the
/// heaps' sizes ([`device_local_bytes_of`]). A driver may answer a budget past its heap's size,
/// which the extension bounds it by: Intel's 101.2145 driver for the UHD Graphics 630, on Windows
/// 11, answers 47.10 GiB for its one heap of 31.92 GiB with nothing used (measured with
/// `vulkaninfo`, 2026-11-15).
///
/// # Safety
///
/// `physical` is a physical device `instance` lists, and `instance` is live.
pub(crate) unsafe fn device_local_budget_of(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> u64 {
    // SAFETY: the caller's.
    let lists = unsafe {
        instance
            .enumerate_device_extension_properties(physical)
            .unwrap_or_default()
            .iter()
            .any(|p| p.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME))
    };
    if !lists {
        // SAFETY: the caller's.
        return unsafe { device_local_bytes_of(instance, physical) };
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut memory = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: the caller's; the one structure chained is of an extension the device lists.
    unsafe { instance.get_physical_device_memory_properties2(physical, &mut memory) };
    let properties = memory.memory_properties;
    (properties.memory_heaps_as_slice().iter())
        .zip(budget.heap_budget)
        .filter(|(heap, _)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .fold(0u64, |sum, (heap, budget)| {
            sum.saturating_add(budget.min(heap.size))
        })
}

/// The bytes of `memory`'s heaps, every one of them, summed (`GpuDevice::memory_heaps_total`),
/// which bound the guest's Shared pool's range. Every heap, not only the heap of
/// metal2vulkan-runtime's Shared memory type (host-visible and host-cached, a host-coherent one
/// first), whose blocks fill the pool: a heap's size is how the driver accounts for its memory,
/// not what it will allocate (on a 780M, RADV gives that type a 5.52 GiB heap beside an 11.05 GiB
/// device-local one, measured with `vulkaninfo` on 2026-11-18, both from the same system memory),
/// and the runtime takes Shared blocks until the driver refuses one. The sizes, not
/// `VK_EXT_memory_budget`'s budgets: the pool's range is guest physical addresses held for the
/// device's life, which RAM fills only as the pool takes blocks, while a budget is what the process
/// may allocate at the moment it is read, which other processes move, and which a driver may answer
/// past its heap's size ([`device_local_budget_of`]).
pub(crate) fn memory_heaps_total_of(memory: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    memory
        .memory_heaps_as_slice()
        .iter()
        .fold(0u64, |sum, heap| sum.saturating_add(heap.size))
}

/// `architecture.name` of a device of PCI vendor `vendor` and device `device` whose driver
/// keys its compiled pipelines by `pipeline_cache`: `vulkan_VVVV_DDDD_` and the key in hex.
///
/// The name keys the guest's binary archives, so it stays the same for one GPU and driver from
/// boot to boot, and differs where compiled pipelines differ: between GPUs, and between drivers
/// or driver builds, which Vulkan's `pipelineCacheUUID` tells apart (a pipeline cache of another
/// UUID is one the driver does not take). The name claims no Apple GPU (Metal's are
/// `applegpu_gNN`); the device's families say what it supports.
pub(crate) fn architecture_name(
    vendor: u32,
    device: u32,
    pipeline_cache: [u8; vk::UUID_SIZE],
) -> String {
    let key: String = pipeline_cache.iter().map(|b| format!("{b:02x}")).collect();
    format!("vulkan_{vendor:04x}_{device:04x}_{key}")
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
pub(crate) fn driver_sizes(registry_id: u64) -> [Option<u64>; 4] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::caps::FEATURE_SETS;

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

    /// Memory properties of heaps of `sizes`, with no memory types.
    fn memory(sizes: &[u64]) -> vk::PhysicalDeviceMemoryProperties {
        let mut m = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: u32::try_from(sizes.len()).unwrap(),
            ..Default::default()
        };
        for (slot, &size) in m.memory_heaps.iter_mut().zip(sizes) {
            slot.size = size;
        }
        m
    }

    /// The memory heaps' total is every heap's size summed, whatever lies over it: a discrete
    /// GPU's own memory, the host's and the window the host sees of the GPU's, and an APU's two
    /// heaps over the same memory; 0 for no heaps, and no wrap past the most a `u64` holds
    #[test]
    fn the_memory_heaps_total_is_every_heap_summed() {
        // An RTX 3090's layout: 24 GiB of its own, the host's 31 GiB, and its 246 MiB window.
        let discrete = memory(&[24 << 30, 31 << 30, 246 << 20]);
        assert_eq!(memory_heaps_total_of(&discrete), (55 << 30) + (246 << 20));
        // A 780M's under RADV: the host-cached types' heap and the device-local one.
        let apu = memory(&[5_932_310_528, 11 << 30]);
        assert_eq!(memory_heaps_total_of(&apu), 5_932_310_528 + (11 << 30));
        assert_eq!(memory_heaps_total_of(&memory(&[])), 0);
        assert_eq!(memory_heaps_total_of(&memory(&[u64::MAX, 1])), u64::MAX);
    }
}
