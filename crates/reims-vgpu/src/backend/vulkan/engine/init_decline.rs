//! Typed Vulkan-engine initialization failures.
//!
//! Engine bring-up used to collapse ten distinct checks into
//! `DrawError::Init(String)` and the single `vk_engine_init_untyped` slug. The
//! process-global negative cache then flattened that error through `Display`
//! and wrapped the resulting prose in a new `DrawError::Init`, losing the
//! original check permanently. This type names each bring-up check and keeps
//! the driver/loader value as a structured, log-safe field.

use ash::vk;

use crate::observe::Decline;

/// A specific check that prevented the Vulkan engine from initializing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitDecline {
    /// Loading the platform Vulkan loader failed before an ash entry existed.
    LoadVulkanLoader { detail: String },
    /// `vkEnumerateInstanceVersion` failed; initialization continues at 1.0.
    EnumerateInstanceVersion { result: vk::Result },
    /// `vkEnumerateInstanceExtensionProperties` failed.
    EnumerateInstanceExtensions { result: vk::Result },
    /// `vkCreateInstance` failed.
    CreateInstance { result: vk::Result },
    /// `vkEnumeratePhysicalDevices` failed.
    EnumeratePhysicalDevices { result: vk::Result },
    /// The loader exposed no physical Vulkan device.
    NoPhysicalDevice,
    /// Devices existed, but every one was below the Vulkan API floor.
    BelowApiFloor { minimum: u32, found: Vec<u32> },
    /// The chosen device exposed no graphics-capable queue family.
    NoGraphicsQueueFamily,
    /// `vkEnumerateDeviceExtensionProperties` failed.
    EnumerateDeviceExtensions { result: vk::Result },
    /// `vkCreateDevice` failed.
    CreateDevice { result: vk::Result },
    /// Both warm-cache and cold `vkCreatePipelineCache` attempts failed.
    CreatePipelineCache { result: vk::Result },
}

impl InitDecline {
    fn squash(value: impl std::fmt::Display) -> String {
        value.to_string().replace(char::is_whitespace, "_")
    }

    /// The driver's `vk::Result`, for the checks that are a Vulkan call
    /// refusing; `None` for the checks this device decided itself.
    ///
    /// The four that answer `None` are judgements rather than calls — no
    /// loader, no device, no graphics queue, every device below the API floor —
    /// and none of them carries a result to report. Every arm is spelled out so
    /// a new variant has to choose a side rather than defaulting into one.
    ///
    /// Read by [`super::types::DrawError::out_of_memory`], which is why the
    /// grouping matches [`Decline::fields`]'s: both answer "did a Vulkan call
    /// refuse, and with what".
    pub(crate) fn vk_result(&self) -> Option<vk::Result> {
        match self {
            Self::EnumerateInstanceVersion { result }
            | Self::EnumerateInstanceExtensions { result }
            | Self::CreateInstance { result }
            | Self::EnumeratePhysicalDevices { result }
            | Self::EnumerateDeviceExtensions { result }
            | Self::CreateDevice { result }
            | Self::CreatePipelineCache { result } => Some(*result),
            Self::LoadVulkanLoader { .. }
            | Self::NoPhysicalDevice
            | Self::BelowApiFloor { .. }
            | Self::NoGraphicsQueueFamily => None,
        }
    }
}

impl Decline for InitDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::LoadVulkanLoader { .. } => "vk_init_load_loader",
            Self::EnumerateInstanceVersion { .. } => "vk_init_enumerate_instance_version",
            Self::EnumerateInstanceExtensions { .. } => "vk_init_enumerate_instance_extensions",
            Self::CreateInstance { .. } => "vk_init_create_instance",
            Self::EnumeratePhysicalDevices { .. } => "vk_init_enumerate_physical_devices",
            Self::NoPhysicalDevice => "vk_init_no_physical_device",
            Self::BelowApiFloor { .. } => "vk_init_below_api_floor",
            Self::NoGraphicsQueueFamily => "vk_init_no_graphics_queue_family",
            Self::EnumerateDeviceExtensions { .. } => "vk_init_enumerate_device_extensions",
            Self::CreateDevice { .. } => "vk_init_create_device",
            Self::CreatePipelineCache { .. } => "vk_init_create_pipeline_cache",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::LoadVulkanLoader { detail } => {
                vec![("detail", Self::squash(detail))]
            }
            Self::EnumerateInstanceVersion { result }
            | Self::EnumerateInstanceExtensions { result }
            | Self::CreateInstance { result }
            | Self::EnumeratePhysicalDevices { result }
            | Self::EnumerateDeviceExtensions { result }
            | Self::CreateDevice { result }
            | Self::CreatePipelineCache { result } => {
                vec![("vk_result", Self::squash(result))]
            }
            Self::BelowApiFloor { minimum, found } => vec![
                (
                    "minimum",
                    crate::backend::vulkan::caps::api_floor::version_str(*minimum),
                ),
                (
                    "found",
                    found
                        .iter()
                        .map(|version| {
                            crate::backend::vulkan::caps::api_floor::version_str(*version)
                        })
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
            Self::NoPhysicalDevice | Self::NoGraphicsQueueFamily => Vec::new(),
        }
    }
}

crate::observe::decline_display!(InitDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<InitDecline> {
        vec![
            InitDecline::LoadVulkanLoader {
                detail: "loader not found".to_string(),
            },
            InitDecline::EnumerateInstanceVersion {
                result: vk::Result::ERROR_INITIALIZATION_FAILED,
            },
            InitDecline::EnumerateInstanceExtensions {
                result: vk::Result::ERROR_INITIALIZATION_FAILED,
            },
            InitDecline::CreateInstance {
                result: vk::Result::ERROR_INCOMPATIBLE_DRIVER,
            },
            InitDecline::EnumeratePhysicalDevices {
                result: vk::Result::ERROR_INITIALIZATION_FAILED,
            },
            InitDecline::NoPhysicalDevice,
            InitDecline::BelowApiFloor {
                minimum: vk::API_VERSION_1_2,
                found: vec![vk::API_VERSION_1_0, vk::API_VERSION_1_1],
            },
            InitDecline::NoGraphicsQueueFamily,
            InitDecline::EnumerateDeviceExtensions {
                result: vk::Result::ERROR_INITIALIZATION_FAILED,
            },
            InitDecline::CreateDevice {
                result: vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            },
            InitDecline::CreatePipelineCache {
                result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            },
        ]
    }

    #[test]
    fn every_initialization_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_init_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate initialization slug");
    }

    #[test]
    fn driver_loader_and_api_versions_reach_log_safe_fields() {
        for decline in all() {
            let line = crate::observe::Emit::decline("vk_init_test", &decline).render();
            assert!(line.starts_with(&format!("vk_init_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }
}
