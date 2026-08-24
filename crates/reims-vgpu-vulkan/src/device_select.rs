//! Physical-device selection: rank, then enforce the API floor.
//!
//! Both Vulkan devices this crate creates — the engine device and the host
//! window's present device — select through here. They used to rank
//! independently on different scales, so on a hybrid host they could disagree
//! about which GPU is best, and the window's scale did not demote a CPU
//! software rasterizer below an unclassified device. Neither applied an API
//! floor at all.

use ash::vk;

use super::api_floor::meets_floor;

/// Rank a physical device for engine selection: prefer a real GPU, and among
/// GPUs prefer discrete over integrated (the RTX 5080 dev host) while still
/// accepting an iGPU-only host (portability directive). A CPU/software device
/// (llvmpipe) ranks lowest so it is chosen ONLY when nothing else exists —
/// never as a silent fallback ahead of a usable GPU that merely enumerated
/// second. Higher is better.
///
/// Ranking is orthogonal to the API floor: a device below
/// [`crate::api_floor::MIN_SUPPORTED_API`] is filtered out before ranking, so a
/// top-ranked device the engine cannot drive never wins over a usable one.
pub fn rank_physical_device(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 0,
        // OTHER / future types: a real device we do not classify — above CPU,
        // below any named GPU.
        _ => 1,
    }
}

/// Pick the best device that meets the API floor, keeping the first-enumerated
/// on a rank tie. `candidates` is `(api_version, device_type, payload)`; the
/// payload is whatever the caller needs back (the `VkPhysicalDevice` in
/// production, an index in tests). Returns the payload plus the chosen device's
/// own `apiVersion`, so the caller logs the version of the device it actually
/// bound rather than re-deriving it from another device's properties.
///
/// Returns `Err` with the rejected devices' versions when nothing clears the
/// floor, so the decline names what was found rather than reporting a bare "no
/// Vulkan devices".
pub fn select_physical_device<T: Copy>(
    candidates: &[(u32, vk::PhysicalDeviceType, T)],
) -> Result<(T, u32), Vec<u32>> {
    let mut below_floor = Vec::new();
    let mut best: Option<(u8, T, u32)> = None;
    for (api, device_type, payload) in candidates {
        if !meets_floor(*api) {
            below_floor.push(*api);
            continue;
        }
        let rank = rank_physical_device(*device_type);
        if best.is_none_or(|(best_rank, _, _)| rank > best_rank) {
            best = Some((rank, *payload, *api));
        }
    }
    best.map(|(_, payload, api)| (payload, api))
        .ok_or(below_floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_device_rank_prefers_real_gpu_over_software() {
        use vk::PhysicalDeviceType as T;
        // Strict ordering: discrete > integrated > virtual > other > cpu.
        assert!(rank_physical_device(T::DISCRETE_GPU) > rank_physical_device(T::INTEGRATED_GPU));
        assert!(rank_physical_device(T::INTEGRATED_GPU) > rank_physical_device(T::VIRTUAL_GPU));
        assert!(rank_physical_device(T::VIRTUAL_GPU) > rank_physical_device(T::OTHER));
        assert!(rank_physical_device(T::OTHER) > rank_physical_device(T::CPU));
        // The load-bearing property: a real GPU of any kind outranks a CPU
        // software rasterizer, so llvmpipe is never chosen ahead of an iGPU.
        for gpu in [T::DISCRETE_GPU, T::INTEGRATED_GPU, T::VIRTUAL_GPU] {
            assert!(rank_physical_device(gpu) > rank_physical_device(T::CPU));
        }
    }

    #[test]
    fn physical_device_selection_keeps_first_on_tie() {
        use vk::PhysicalDeviceType as T;
        // Highest rank wins, first enumerated wins on a tie (two integrated
        // GPUs → index 0).
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_3, T::INTEGRATED_GPU, 0usize),
            (vk::API_VERSION_1_3, T::INTEGRATED_GPU, 1),
        ]);
        assert_eq!(chosen, Ok((0, vk::API_VERSION_1_3)));

        // A CPU device enumerated FIRST must lose to a discrete GPU second.
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_3, T::CPU, 0usize),
            (vk::API_VERSION_1_3, T::DISCRETE_GPU, 1),
        ]);
        assert_eq!(chosen, Ok((1, vk::API_VERSION_1_3)));
    }

    /// A device below the 1.2 floor is filtered out BEFORE ranking, so a
    /// top-ranked discrete GPU on a Vulkan 1.1 driver never beats a usable
    /// integrated one. Getting this backwards would bind a device the engine
    /// cannot drive and fail much later.
    #[test]
    fn below_floor_devices_lose_to_a_usable_lower_ranked_device() {
        use vk::PhysicalDeviceType as T;
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_1, T::DISCRETE_GPU, 'a'),
            (vk::API_VERSION_1_2, T::INTEGRATED_GPU, 'b'),
        ]);
        assert_eq!(chosen, Ok(('b', vk::API_VERSION_1_2)));
    }

    /// When nothing clears the floor, the decline carries the versions that
    /// were found so the log names the actual situation.
    #[test]
    fn no_device_above_floor_reports_what_was_found() {
        use vk::PhysicalDeviceType as T;
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_0, T::CPU, 0usize),
            (vk::API_VERSION_1_1, T::DISCRETE_GPU, 1),
        ]);
        assert_eq!(chosen, Err(vec![vk::API_VERSION_1_0, vk::API_VERSION_1_1]));
    }

    /// No devices at all is distinguishable from "devices, all too old".
    #[test]
    fn empty_enumeration_declines_with_an_empty_list() {
        let chosen = select_physical_device::<usize>(&[]);
        assert_eq!(chosen, Err(Vec::new()));
    }

    /// The chosen device's own `apiVersion` comes back with it, so the caller
    /// never logs a different device's version.
    #[test]
    fn selection_returns_the_chosen_devices_own_api_version() {
        use vk::PhysicalDeviceType as T;
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_2, T::INTEGRATED_GPU, 'i'),
            (vk::API_VERSION_1_3, T::DISCRETE_GPU, 'd'),
        ]);
        assert_eq!(chosen, Ok(('d', vk::API_VERSION_1_3)));

        // …and the 1.2 device wins when it is the only real GPU. A higher API
        // version is NOT a tiebreaker: rank decides, and 1.2 is the baseline
        // every pathway runs on, so a 1.2 GPU beats a 1.3 software rasterizer.
        let chosen = select_physical_device(&[
            (vk::API_VERSION_1_2, T::DISCRETE_GPU, 'd'),
            (vk::API_VERSION_1_3, T::CPU, 'c'),
        ]);
        assert_eq!(chosen, Ok(('d', vk::API_VERSION_1_2)));
    }
}
