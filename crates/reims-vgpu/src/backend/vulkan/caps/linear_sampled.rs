//! Whether this device could sample guest pages **as an image**, with no detile
//! copy — the step Apple's own host framework does not have.
//!
//! # Why the question exists
//!
//! [`super::host_pointer`] brings guest RAM to the GPU as a `VkBuffer`, and
//! buffers-only is deliberate there: an optimally-tiled image backed by linear
//! guest bytes is not something this device may assume works on an unknown
//! driver. So guest texels reach a sampled image through
//! `vkCmdCopyBufferToImage`, and the sampled cache in `pools` exists to memoize
//! the result of that copy.
//!
//! Apple's `ParavirtualizedGraphics` host framework has no such cache, and the
//! reason is visible in the Metal selectors it references:
//! `newBufferWithBytesNoCopy:length:options:deallocator:` — the same import
//! primitive, the one MoltenVK implements `VK_EXT_external_memory_host` over —
//! together with `newTextureWithDescriptor:offset:bytesPerRow:`, which makes an
//! `MTLTexture` that *is* the buffer, and
//! `minimumLinearTextureAlignmentForPixelFormat:`, which is the alignment that
//! makes it legal. No copy, so nothing to cache.
//!
//! # Three conditions, and the third is not a capability
//!
//! Vulkan can express the same thing, conditionally. Measured on one discrete
//! NVIDIA (RTX 5080, 580.x) — one driver, and every number below is that
//! driver's:
//!
//! 1. **`linearTilingFeatures`** must carry `SAMPLED_IMAGE` *and*
//!    `SAMPLED_IMAGE_FILTER_LINEAR` for the format. Read by [`report`]. All four
//!    probed formats passed, which is not what "buffers only, deliberately"
//!    predicted.
//! 2. **The device must import a host pointer as an image**, not just as a
//!    buffer — `vkGetPhysicalDeviceImageFormatProperties2` with a
//!    `VkPhysicalDeviceExternalImageFormatInfo` chain naming
//!    `HOST_ALLOCATION_EXT`. This is the exact image-side analogue of the buffer
//!    query in [`super::host_pointer::query`], and it also passed, up to
//!    32768x32768.
//! 3. **The driver's chosen row pitch must equal the guest's own**, and this one
//!    is *not* answerable at capability time. It is the asymmetry between the
//!    two APIs: Apple's call **takes** `bytesPerRow`, so a texture can always be
//!    made to match the guest's layout, while a Vulkan linear image's `rowPitch`
//!    is chosen by the driver and only reported back, through
//!    `vkGetImageSubresourceLayout` on a created image. Where the two disagree
//!    the bytes cannot be aliased at all and the copy is compulsory.
//!
//! Condition 3 held for every standard display width and failed otherwise,
//! because this driver rounds the pitch up to 32 bytes:
//!
//! ```text
//!   1920x1080   guest 7680   driver  7680   alias
//!   2560x1440   guest 10240  driver 10240   alias
//!   3840x2160   guest 15360  driver 15360   alias
//!   1366x768    guest 5464   driver  5472   copy
//!   17x5        guest 68     driver    96   copy
//! ```
//!
//! So a rail built on this is a per-window question — resolvable, but only
//! against a created image and the guest's *declared* pitch, which is
//! `row_length_texels` and not always `width * bpp`. It belongs at the bind
//! site with a named decline, not here.
//!
//! # This module reports and does not gate
//!
//! Nothing branches on any of it yet, and per the rules in [`super`] a
//! capability nothing reads does not go on `HostGpuCaps`; it is measured and
//! reported where it is measured. Two further things are unmeasured and would
//! decide whether the rail is worth building at all:
//!
//! * **Whether sampling host memory in place is actually faster on a discrete
//!   host.** Every texel fetch would cross PCIe instead of reading VRAM. The
//!   support matrix says the copy into VRAM is the point on such a host, so
//!   "possible" here is not "wanted" — the unified-memory cells are where this
//!   would pay. Do not build the rail on condition 1-3 passing alone.
//! * **Whether the imported memory type intersects the linear image's
//!   `vkGetImageMemoryRequirements`.** A per-image question like condition 3.

use ash::vk;

/// Formats the sampled rail builds images in, and therefore the ones worth
/// asking about.
///
/// Taken from `translate::pixel`'s mapping of the Metal pixel formats this
/// device decodes — the 8-bit RGBA/BGRA pairs are what a macOS guest's
/// compositing actually uses. Representative rather than exhaustive: a format
/// absent here is unmeasured, not unsupported, and the line names what it asked
/// about so a reader cannot mistake one for the other.
const PROBED: &[(&str, vk::Format)] = &[
    ("bgra8_unorm", vk::Format::B8G8R8A8_UNORM),
    ("bgra8_srgb", vk::Format::B8G8R8A8_SRGB),
    ("rgba8_unorm", vk::Format::R8G8B8A8_UNORM),
    ("rgba8_srgb", vk::Format::R8G8B8A8_SRGB),
];

/// What one format can do under `VK_IMAGE_TILING_LINEAR`.
///
/// Three bits rather than one because they fail differently. A format that is
/// `SAMPLED_IMAGE` but not `SAMPLED_IMAGE_FILTER_LINEAR` can be sampled only
/// with nearest filtering, which is a *different picture* rather than a slower
/// one — so a rail built on the first without the second would land wrong pixels
/// wherever the guest asked for a linear filter. And a device that samples the
/// format linearly but will not import a host pointer as an image cannot reach
/// the guest's bytes at all, however good the first two look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearFormatVerdict {
    /// `linearTilingFeatures` contains `SAMPLED_IMAGE`.
    pub sampled: bool,
    /// `linearTilingFeatures` contains `SAMPLED_IMAGE_FILTER_LINEAR`.
    pub filter_linear: bool,
    /// The device reports `HOST_ALLOCATION_EXT` importable for a `LINEAR`,
    /// `SAMPLED` image of this format.
    pub importable: bool,
}

impl LinearFormatVerdict {
    /// Read the tiling features. `importable` is answered separately, by a query
    /// that can fail as a whole, so it is set by the caller.
    fn from_tiling(features: vk::FormatFeatureFlags) -> Self {
        Self {
            sampled: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
            filter_linear: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
            importable: false,
        }
    }

    /// Whether the *device-level* conditions all hold. Never sufficient on its
    /// own: the row-pitch agreement in this module's doc is a per-window
    /// question and is not represented here.
    pub fn device_conditions_hold(self) -> bool {
        self.sampled && self.filter_linear && self.importable
    }

    /// Stable slug for the report line, naming which condition refused.
    fn slug(self) -> &'static str {
        match (self.sampled, self.filter_linear, self.importable) {
            (true, true, true) => "alias_possible",
            (true, true, false) => "not_importable",
            (true, false, _) => "sampled_nearest_only",
            (false, _, _) => "not_sampled",
        }
    }
}

/// Ask the device about every format in [`PROBED`] and emit the answer.
///
/// Emitted on the `OFF` channel at device create, beside `vk_caps`, because it
/// is a fact about the host and not a loss. One line per boot.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn report(instance: &ash::Instance, pd: vk::PhysicalDevice) {
    let mut fields = String::new();
    let mut possible = 0usize;
    for (name, format) in PROBED {
        let props = unsafe { instance.get_physical_device_format_properties(pd, *format) };
        let mut verdict = LinearFormatVerdict::from_tiling(props.linear_tiling_features);

        let mut ext_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(*format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .push_next(&mut ext_info);
        let mut ext_props = vk::ExternalImageFormatProperties::default();
        let mut out = vk::ImageFormatProperties2::default().push_next(&mut ext_props);
        // A device that cannot make this image at all answers `Err`, which is a
        // "no" for this format rather than an error for the boot.
        if unsafe { instance.get_physical_device_image_format_properties2(pd, &info, &mut out) }
            .is_ok()
        {
            verdict.importable = ext_props
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
        }

        if verdict.device_conditions_hold() {
            possible += 1;
        }
        fields.push_str(&format!(" {name}={}", verdict.slug()));
    }
    crate::observe::off(format!(
        "vk_linear_sampled alias_possible={possible}/{}{fields} (device conditions only — a window \
         also needs vkGetImageSubresourceLayout's rowPitch to equal the guest's declared pitch, \
         which is per-window and not asked here)",
        PROBED.len()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every condition is load-bearing, and each one missing is its own slug. A
    /// format that samples but cannot filter linearly would land nearest-filtered
    /// pixels wherever the guest asked for a linear filter — a wrong picture, not
    /// a slow one — so no single bit may satisfy the verdict alone.
    #[test]
    fn every_device_condition_is_required() {
        let all = LinearFormatVerdict {
            sampled: true,
            filter_linear: true,
            importable: true,
        };
        assert!(all.device_conditions_hold());
        assert_eq!(all.slug(), "alias_possible");

        for (drop_field, want_slug) in [
            ("filter", "sampled_nearest_only"),
            ("import", "not_importable"),
            ("sampled", "not_sampled"),
        ] {
            let mut v = all;
            match drop_field {
                "filter" => v.filter_linear = false,
                "import" => v.importable = false,
                _ => v.sampled = false,
            }
            assert!(!v.device_conditions_hold(), "{drop_field} must be required");
            assert_eq!(v.slug(), want_slug, "{drop_field}");
        }
    }

    /// The tiling flags are read from `linearTilingFeatures`, not invented, and a
    /// neighbouring bit must not be mistaken for either of ours.
    #[test]
    fn the_verdict_reads_the_two_tiling_flags_it_names() {
        let none = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::empty());
        assert!(!none.sampled && !none.filter_linear);

        let sampled = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::SAMPLED_IMAGE);
        assert!(sampled.sampled && !sampled.filter_linear);

        let both = LinearFormatVerdict::from_tiling(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        );
        assert!(both.sampled && both.filter_linear);

        let storage = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::STORAGE_IMAGE);
        assert!(!storage.sampled && !storage.filter_linear);
    }

    /// `from_tiling` never claims importability. That answer comes from a
    /// separate query which can fail as a whole, and a verdict that defaulted it
    /// to `true` would report an alias as possible on a device that was never
    /// asked.
    #[test]
    fn tiling_alone_never_claims_importable() {
        let both = LinearFormatVerdict::from_tiling(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        );
        assert!(!both.importable);
        assert!(!both.device_conditions_hold());
    }

    /// Every probed format is named and distinct. A duplicate would report one
    /// device answer twice and inflate the `alias_possible=` numerator.
    #[test]
    fn every_probed_format_is_named_once() {
        let mut names: Vec<_> = PROBED.iter().map(|(n, _)| *n).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two probed formats share a name");

        let mut formats: Vec<_> = PROBED.iter().map(|(_, f)| *f).collect();
        formats.sort_unstable_by_key(|f| f.as_raw());
        formats.dedup();
        assert_eq!(formats.len(), count, "one format is probed twice");
    }
}
