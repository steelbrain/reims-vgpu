//! Whether a resident render target can use imported linear host memory.
//!
//! Resource synchronization is an operation on a retained backing. On shared
//! memory, an image that uses the guest allocation as that backing needs only
//! ordering at synchronize time; a distinct device image instead owes a
//! full-frame copy into the guest allocation. This module asks whether Vulkan
//! can express the shared-backing shape for the formats resident colour targets
//! use.
//!
//! The answer is necessary, not sufficient. A particular target must also have
//! a stable contiguous host alias, compatible image memory requirements, and a
//! driver-selected linear row pitch equal to the guest's declared pitch. Those
//! are per-target facts and stay out of this device-level report.

use ash::vk;

/// Formats used by the ordinary eight-bit resident colour-target rails.
const PROBED: &[(&str, vk::Format)] = &[
    ("bgra8_unorm", vk::Format::B8G8R8A8_UNORM),
    ("bgra8_srgb", vk::Format::B8G8R8A8_SRGB),
    ("rgba8_unorm", vk::Format::R8G8B8A8_UNORM),
    ("rgba8_srgb", vk::Format::R8G8B8A8_SRGB),
];

/// The usage common to resident colour targets before optional feedback-loop
/// usage is added.
pub(crate) const BASE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::INPUT_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw(),
);

/// Device-level answer for one linear target format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearTargetVerdict {
    /// Linear tiling carries every format feature the target usage needs.
    usable: bool,
    /// `HOST_ALLOCATION_EXT` is importable with [`BASE_USAGE`].
    importable: bool,
    /// That import is reported `DEDICATED_ONLY`: the memory must be bound to an
    /// image created solely for it.
    ///
    /// A shared-backing target imports the guest allocation that the guest's
    /// own buffer already covers, so a handle type that will only be imported
    /// into a dedicated allocation cannot serve it. `importable` reads `true`
    /// on such a device, so without this bit the report claims a shape the
    /// device will not build.
    dedicated_only: bool,
}

impl LinearTargetVerdict {
    fn from_tiling(features: vk::FormatFeatureFlags) -> Self {
        let required = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        Self {
            usable: features.contains(required),
            importable: false,
            dedicated_only: false,
        }
    }

    fn device_conditions_hold(self) -> bool {
        self.usable && self.importable && !self.dedicated_only
    }

    fn slug(self) -> &'static str {
        match (self.usable, self.importable, self.dedicated_only) {
            (true, true, false) => "alias_possible",
            (true, true, true) => "dedicated_only",
            (true, false, _) => "not_importable",
            (false, _, _) => "usage_unsupported",
        }
    }
}

/// Report whether each resident colour format can be a linear imported target.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub(crate) unsafe fn report(instance: &ash::Instance, pd: vk::PhysicalDevice) {
    let mut fields = String::new();
    let mut possible = 0usize;
    for (name, format) in PROBED {
        let props = unsafe { instance.get_physical_device_format_properties(pd, *format) };
        let mut verdict = LinearTargetVerdict::from_tiling(props.linear_tiling_features);

        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(*format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(BASE_USAGE)
            .push_next(&mut external);
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut out = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        if unsafe { instance.get_physical_device_image_format_properties2(pd, &info, &mut out) }
            .is_ok()
        {
            let features = external_props
                .external_memory_properties
                .external_memory_features;
            verdict.importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
            verdict.dedicated_only =
                features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        }

        if verdict.device_conditions_hold() {
            possible += 1;
        }
        fields.push_str(&format!(" {name}={}", verdict.slug()));
    }
    crate::observe::off(format!(
        "vk_linear_target alias_possible={possible}/{}{fields} (device conditions only; each target also needs a stable alias, compatible memory type, and matching row pitch)",
        PROBED.len()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_target_format_feature_is_required() {
        let all = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        assert!(LinearTargetVerdict::from_tiling(all).usable);
        for bit in [
            vk::FormatFeatureFlags::SAMPLED_IMAGE,
            vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
            vk::FormatFeatureFlags::TRANSFER_SRC,
            vk::FormatFeatureFlags::TRANSFER_DST,
        ] {
            assert!(
                !LinearTargetVerdict::from_tiling(all & !bit).usable,
                "missing {bit:?} must refuse the target alias"
            );
        }
    }

    #[test]
    fn tiling_support_does_not_claim_external_importability() {
        let all = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        let verdict = LinearTargetVerdict::from_tiling(all);
        assert!(verdict.usable);
        assert!(!verdict.importable);
        assert!(!verdict.device_conditions_hold());
        assert_eq!(verdict.slug(), "not_importable");
    }

    /// A dedicated-only import cannot alias an allocation the guest's own
    /// buffer already covers, so it refuses the rail even though every other
    /// device condition holds and `importable` reads true.
    #[test]
    fn a_dedicated_only_import_is_not_an_alias() {
        let all = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        let mut verdict = LinearTargetVerdict::from_tiling(all);
        verdict.importable = true;
        assert!(verdict.device_conditions_hold());
        assert_eq!(verdict.slug(), "alias_possible");

        verdict.dedicated_only = true;
        assert!(
            !verdict.device_conditions_hold(),
            "a dedicated allocation cannot also be the guest allocation's own buffer"
        );
        assert_eq!(
            verdict.slug(),
            "dedicated_only",
            "the refusing condition must be named, not folded into not_importable"
        );
    }

    #[test]
    fn probed_formats_are_named_and_distinct() {
        let mut names: Vec<_> = PROBED.iter().map(|(name, _)| *name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);

        let mut formats: Vec<_> = PROBED.iter().map(|(_, format)| *format).collect();
        formats.sort_unstable_by_key(|format| format.as_raw());
        formats.dedup();
        assert_eq!(formats.len(), count);
    }
}
