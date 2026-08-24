//! The Vulkan API floor. **One** baseline, not a tier ladder.
//!
//! Every host this backend supports runs **Vulkan 1.2**, and the engine
//! requires nothing above 1.2 core on any pathway. A device below
//! [`MIN_SUPPORTED_API`] is **declined by name**, not silently degraded:
//! decoding the guest Metal stream needs 8-bit storage and
//! `shaderOutputViewportIndex`, both of which are Vulkan 1.2 core and both of
//! which `caps::device_features` queries and enables. Vulkan 1.1 cannot express
//! the command stream, so pretending otherwise would fail later and further
//! from the cause.
//!
//! This paragraph used to name four requirements. Two of them — descriptor
//! indexing and timeline semaphores — appear nowhere in the crate: not queried,
//! not enabled, not used. A floor justified by capabilities nothing reaches for
//! is a floor nobody can re-derive, and the two that are real are enough on
//! their own. Timeline semaphores are worth knowing about for the opposite
//! reason: they are 1.2 core and therefore already inside this floor, so
//! adopting them needs no new gate.
//!
//! # There is deliberately no `ApiFloor` tier enum
//!
//! An earlier design classified devices into `Vk12` / `Vk13` tiers and made that
//! an axis of the support matrix. It described the hosts this project happens to
//! own rather than anything the code does: nothing in the engine has ever used a
//! 1.3 core feature, so the "1.3 row" only ever meant "a host that also happens
//! to have feature X". Device capability is the property that actually varies
//! — whether the device can share memory with another device at all — and it is
//! orthogonal to the API version: a 1.2 driver can advertise an extension and
//! a 1.3 driver can lack it. Classifying on the version invited exactly the
//! coupling that gating on capability exists to prevent, so the version is now a
//! floor check and nothing more.
//!
//! A capability promoted into 1.3 core must therefore be reached through its
//! `KHR`/`EXT` form, gated on runtime presence, with the 1.2 path still
//! implemented and tested. A source scan used to fail when a 1.3-core feature
//! struct or promoted entry point appeared in the crate; it is gone, so this is
//! a review obligation now. One trap it could never see anyway, and neither can
//! a reader skimming for `vk::` names: ash gives the
//! `synchronization2` family one token for both spellings, so only the entry
//! points separate correct extension use from 1.3-core use there.

use ash::vk;

/// Lowest device `apiVersion` the engine will bind. Below this the guest Metal
/// command stream cannot be expressed (Vulkan 1.2 core: 8-bit storage,
/// `shaderOutputViewportIndex`).
pub const MIN_SUPPORTED_API: u32 = vk::API_VERSION_1_2;

/// Ceiling for the version the **instance** requests. This is a loader
/// negotiation bound, NOT a requirement and NOT a tier: asking for more than the
/// loader supports is `VK_ERROR_INCOMPATIBLE_DRIVER` on a Vulkan 1.0 loader, and
/// asking for an unbounded version claims capability nobody checks.
///
/// A consequence worth knowing when reading logs: because the instance caps
/// here, the loader clamps every physical device's reported `apiVersion` to this
/// value too — an Apple M3 Max's native 1.4.334 reads back as 1.3. That is the
/// clamp, not a driver limitation, and it changes nothing about what the engine
/// requires, which is [`MIN_SUPPORTED_API`].
pub const MAX_USEFUL_API: u32 = vk::API_VERSION_1_3;

/// Inverting the pair would decline every device on earth with a confusing
/// reason. Both are literals, so the check belongs at compile time.
const _: () = assert!(MAX_USEFUL_API >= MIN_SUPPORTED_API);

/// Whether a device's `apiVersion` clears the baseline every pathway needs.
///
/// The caller must decline a `false` **by name** rather than degrading — see
/// `vk_device_select_fail reason=vk_init_below_api_floor`.
pub fn meets_floor(device_api_version: u32) -> bool {
    device_api_version >= MIN_SUPPORTED_API
}

/// Pick the `VkApplicationInfo::apiVersion` to request for the instance.
///
/// `loader_version` is `vkEnumerateInstanceVersion`'s answer (or
/// `vk::API_VERSION_1_0` when the entry point is absent, which means a Vulkan
/// 1.0 loader). Clamped by both the loader and [`MAX_USEFUL_API`].
pub fn instance_api_version(loader_version: u32) -> u32 {
    loader_version.min(MAX_USEFUL_API)
}

/// `major.minor` for a log line, so declines name the versions actually found
/// instead of printing a packed `u32`.
pub fn version_str(api_version: u32) -> String {
    format!(
        "{}.{}",
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The floor is 1.2: a 1.0/1.1 device is declined, not degraded.
    #[test]
    fn below_floor_is_declined() {
        assert!(!meets_floor(vk::API_VERSION_1_0));
        assert!(!meets_floor(vk::API_VERSION_1_1));
        // One patch below 1.2 is still below the floor.
        assert!(!meets_floor(vk::API_VERSION_1_2 - 1));
    }

    /// Everything at or above the baseline is accepted on equal terms — there
    /// is no tier, so a 1.4 device gets no different treatment than a 1.2 one.
    #[test]
    fn every_version_at_or_above_the_floor_is_accepted_equally() {
        for version in [
            vk::API_VERSION_1_2,
            vk::make_api_version(0, 1, 2, 300),
            vk::API_VERSION_1_3,
            // The M3 Max reports 1.4.334 natively.
            vk::make_api_version(0, 1, 4, 334),
        ] {
            assert!(meets_floor(version), "{}", version_str(version));
        }
    }

    /// The instance version is clamped by BOTH the loader and our ceiling.
    #[test]
    fn instance_version_clamps_both_ways() {
        // A 1.0 loader must not be asked for 1.3 (INCOMPATIBLE_DRIVER).
        assert_eq!(
            instance_api_version(vk::API_VERSION_1_0),
            vk::API_VERSION_1_0
        );
        assert_eq!(
            instance_api_version(vk::API_VERSION_1_2),
            vk::API_VERSION_1_2
        );
        // A 1.4 loader (Homebrew vulkan-loader 1.4.350) is clamped to the
        // negotiation ceiling.
        assert_eq!(
            instance_api_version(vk::make_api_version(0, 1, 4, 350)),
            MAX_USEFUL_API
        );
    }

    /// The ceiling itself clears the floor, so clamping to it never produces an
    /// instance version that would then be declined.
    #[test]
    fn ceiling_is_at_or_above_the_floor() {
        assert!(meets_floor(instance_api_version(MAX_USEFUL_API)));
    }

    #[test]
    fn version_str_is_major_minor() {
        assert_eq!(version_str(vk::API_VERSION_1_2), "1.2");
        assert_eq!(version_str(vk::make_api_version(0, 1, 4, 334)), "1.4");
    }
}
