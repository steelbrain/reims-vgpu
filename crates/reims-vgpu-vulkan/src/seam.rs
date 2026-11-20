//! Our side of metal2vulkan-rt's seam: what the Vulkan platform gives rt, built on rt at the commit
//! the workspace pins. Its tests run it on the GPU each test host holds, but on the Mac, whose Vulkan, MoltenVK, rt's floor refuses (it lacks `nullDescriptor`).
//!
//! - **Sampler IDs** ([`sampler_ids`]): rt's creation of a pipeline state takes one sampler ID
//!   for each sampler the state numbers, each constexpr sampler's an index of a live grant.
//!   [`SamplerIds`] grants the device's sampler range once and numbers each state's samplers.
//! - **Blit options** ([`blit`]): rt's copies between a buffer and a texture take no
//!   `MTLBlitOption`. Options 0 is rt's copy, and any other value is [`BlitError::Options`],
//!   with nothing recorded.
//!
//! The pool's block source over rt, which keeps rt's block handles by their numbers, is
//! reims-vgpu's (`pool::rt_blocks`), beside the pool's `BlockSource`.

pub mod blit;
pub mod sampler_ids;

pub use blit::BlitError;
pub use sampler_ids::{FIRST_SAMPLER_ID, Numbered, PipelineError, SAMPLER_ID_COUNT, SamplerIds};

#[cfg(any(test, feature = "testkit"))]
pub use testkit::test_device;

#[cfg(any(test, feature = "testkit"))]
mod testkit {
    use metal2vulkan_rt::{Device, DeviceOptions, Instance, PhysicalDevice};

    /// The variables that name the host's pick of GPU, in the order they are read, as the
    /// platform's device choice reads them (`crate::device_choice`): `vendor:device` in hex, with
    /// or without `0x` and Mesa's `!`.
    const SELECTION: [&str; 2] = ["MESA_VK_DEVICE_SELECT", "VK_LOADER_DEVICE_SELECT"];

    /// Tests: an rt device, made with rt's default options, on the physical device the first
    /// selection variable set names, else on the first one that meets rt's floor; with its name
    /// and `vendor:device` IDs, which a test prints.
    ///
    /// Err: what rt answered, or that no physical device meets the floor, as text.
    pub fn test_device() -> Result<(Device, String), String> {
        let instance = Instance::new(&[]).map_err(|error| format!("rt's instance: {error}"))?;
        let physical = (instance.physical_devices())
            .map_err(|error| format!("rt's physical devices: {error}"))?;
        let (_, raw) = instance.raw();
        let listed: Vec<(PhysicalDevice, u32, u32, String)> = (physical.into_iter())
            .map(|device| {
                // SAFETY: the instance listed the device and is alive.
                let properties = unsafe { raw.get_physical_device_properties(device.raw()) };
                let name = (properties.device_name_as_c_str())
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (device, properties.vendor_id, properties.device_id, name)
            })
            .collect();
        let named = SELECTION
            .iter()
            .find_map(|variable| std::env::var(variable).ok().and_then(|v| ids(&v)));
        let (chosen, vendor, id, name) = (listed.iter())
            .find(|(_, vendor, id, _)| named == Some((*vendor, *id)))
            .or_else(|| listed.iter().find(|(device, ..)| device.check().is_ok()))
            .ok_or_else(|| below_floor(&listed))?;
        let device = (Device::new(chosen, DeviceOptions::default()))
            .map_err(|error| format!("rt's device on {name}: {error}"))?;
        Ok((device, format!("{name} {vendor:04x}:{id:04x}")))
    }

    /// The error where no listed physical device meets rt's floor: each one's name and IDs, with
    /// what rt says it lacks.
    fn below_floor(listed: &[(PhysicalDevice, u32, u32, String)]) -> String {
        let each: Vec<String> = (listed.iter())
            .filter_map(|(device, vendor, id, name)| {
                let below = device.check().err()?;
                Some(format!("{name} {vendor:04x}:{id:04x}: {below}"))
            })
            .collect();
        let count = listed.len();
        format!(
            "none of {count} physical devices meets rt's floor: {}",
            each.join("; ")
        )
    }

    /// The `vendor:device` IDs a selection variable's value names.
    fn ids(value: &str) -> Option<(u32, u32)> {
        let (vendor, device) = value.trim_end_matches('!').split_once(':')?;
        let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
        Some((hex(vendor)?, hex(device)?))
    }

    #[cfg(test)]
    mod tests {
        use super::ids;

        #[test]
        fn a_selection_names_its_ids_with_or_without_prefix_and_bang() {
            assert_eq!(ids("1002:15bf!"), Some((0x1002, 0x15bf)));
            assert_eq!(ids("0x10de:0x2204"), Some((0x10de, 0x2204)));
            assert_eq!(ids("pci-0000_03_00_0"), None);
        }
    }
}
