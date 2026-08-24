//! Capability and limit for command-buffer-local descriptor state.
//!
//! Metal resource setters mutate encoder state directly.  Vulkan's closest
//! optional spelling is `VK_KHR_push_descriptor`: descriptor writes become
//! commands in the command buffer instead of updates to a separately allocated
//! set.  The extension has a hard per-layout descriptor limit, so this is a
//! positive capability rung rather than a new requirement.  Layouts wider than
//! the reported limit keep the ordinary allocated-set path.

use ash::vk;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PushDescriptorCaps {
    /// `VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors`.
    pub max_descriptors: u32,
}

impl PushDescriptorCaps {
    pub fn is_available(self) -> bool {
        self.max_descriptors != 0
    }

    /// Whether one set layout fits the device's push-descriptor limit.
    ///
    /// The sum is checked because descriptor array counts come from translated
    /// shader reflection.  Overflow is a refusal of this optional rail, never a
    /// reason to reject a layout the allocated-set path can still represent.
    pub fn supports_counts(self, counts: impl IntoIterator<Item = u32>) -> bool {
        self.is_available()
            && counts
                .into_iter()
                .try_fold(0u32, u32::checked_add)
                .is_some_and(|total| total <= self.max_descriptors)
    }

    pub fn required_extensions(self) -> Vec<*const i8> {
        self.is_available()
            .then_some(ash::khr::push_descriptor::NAME.as_ptr())
            .into_iter()
            .collect()
    }
}

/// Resolve the extension and its mandatory limit as one answer.
///
/// # Safety
///
/// `pd` must belong to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
    enabled: bool,
) -> PushDescriptorCaps {
    if !enabled {
        return PushDescriptorCaps::default();
    }
    if !has_extension(ash::khr::push_descriptor::NAME) {
        return PushDescriptorCaps::default();
    }
    let mut push = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut push);
    unsafe { instance.get_physical_device_properties2(pd, &mut properties) };
    PushDescriptorCaps {
        max_descriptors: push.max_push_descriptors,
    }
}

/// Query push descriptors with the operator's capability-narrowing switch.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query_configured(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> PushDescriptorCaps {
    let enabled = reims_vgpu_config::switch(reims_vgpu_config::PUSH_DESCRIPTORS)
        != reims_vgpu_config::Switch::Off;
    unsafe { query(instance, pd, has_extension, enabled) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_must_fit_the_reported_total() {
        let caps = PushDescriptorCaps { max_descriptors: 8 };
        assert!(caps.supports_counts([2, 1, 5]));
        assert!(!caps.supports_counts([2, 1, 6]));
        assert!(!caps.supports_counts([u32::MAX, 2]));
        assert!(!PushDescriptorCaps::default().supports_counts([1]));
    }
}
