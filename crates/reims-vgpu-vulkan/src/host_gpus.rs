//! The render node of the GPU a runtime device runs on, which a host video API opens the GPU by:
//! the reims-graphics device's VA-API arm (reims-vgpu's `graphics::vaapi`) opens the runtime
//! device's. It is the node the device's `VK_EXT_physical_device_drm` names, else the one sysfs
//! lists for the PCI address its `VK_EXT_pci_bus_info` names; none where neither does. The rt
//! platform's device answers the same two, which `render_node_at` takes.
//! On Windows the GPU is opened by its DXGI adapter's LUID instead (`adapter_luid_of`, built for
//! Windows alone): the one Vulkan answers for the physical device of the runtime device's UUID.

#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

use metal2vulkan_runtime as rt;

/// The render node of `device`'s GPU: the one its DRM node names, else the one sysfs lists for
/// its PCI address; `None` where neither does (module doc).
#[cfg(target_os = "linux")]
pub fn render_node_of(device: &rt::Device) -> Option<PathBuf> {
    let properties = device.properties();
    let pci_address = properties.pci_bus.map(pci_address_of);
    (properties.drm_render)
        .and_then(|(major, minor)| node_of(major, minor))
        .or_else(|| pci_address.as_deref().and_then(node_at_pci))
}

/// The render node of the GPU whose render node's numbers (`VK_EXT_physical_device_drm`) and PCI
/// address (`VK_EXT_pci_bus_info`) are `render` and `pci`, found as [`render_node_of`] finds a
/// runtime device's: the rt platform's device answers them (`rt_platform::HostGpu`).
#[cfg(target_os = "linux")]
pub fn render_node_at(
    render: Option<(u32, u32)>,
    pci: Option<(u32, u32, u32, u32)>,
) -> Option<PathBuf> {
    render
        .and_then(|(major, minor)| node_of(major.into(), minor.into()))
        .or_else(|| pci.map(pci_address_of).as_deref().and_then(node_at_pci))
}

/// The LUID of the DXGI adapter `device`'s GPU is, as Vulkan answers it for the physical device
/// whose UUID is the runtime device's (`VkPhysicalDeviceIDProperties::deviceLUID`, read through an
/// instance of this call's own at Vulkan 1.1); `None` where no loader, no such device, or no valid
/// LUID is had. Windows's: libva-win32 opens a display on the adapter of this LUID.
#[cfg(windows)]
pub fn adapter_luid_of(device: &rt::Device) -> Option<[u8; 8]> {
    use ash::vk;
    let uuid = device.properties().device_uuid;
    // SAFETY: loads the system's Vulkan loader, whose initialisers register nothing this process
    // reads.
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: a valid create info with no layers and no extension.
    let instance = unsafe { entry.create_instance(&info, None) }.ok()?;
    // SAFETY: `instance` is live, each physical device is one it lists, and the ID properties are
    // core at the instance's Vulkan 1.1; the instance is destroyed once, after the last use.
    unsafe {
        let luid = (instance.enumerate_physical_devices().unwrap_or_default())
            .into_iter()
            .filter(|&physical| {
                instance
                    .get_physical_device_properties(physical)
                    .api_version
                    >= vk::API_VERSION_1_1
            })
            .find_map(|physical| {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                instance.get_physical_device_properties2(physical, &mut asked);
                (id.device_uuid == uuid && id.device_luid_valid == vk::TRUE)
                    .then_some(id.device_luid)
            });
        instance.destroy_instance(None);
        luid
    }
}

/// The address sysfs names a PCI device by, `0000:03:00.0`, of its domain, bus, device and
/// function.
#[cfg(target_os = "linux")]
fn pci_address_of((domain, bus, device, function): (u32, u32, u32, u32)) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// The DRM node of character device `major:minor`, as sysfs names it
/// (`/sys/dev/char/226:128` links to its `renderD128`), in `/dev/dri`.
#[cfg(target_os = "linux")]
fn node_of(major: i64, minor: i64) -> Option<PathBuf> {
    let link = std::fs::read_link(format!("/sys/dev/char/{major}:{minor}")).ok()?;
    Some(Path::new("/dev/dri").join(link.file_name()?))
}

/// The DRM render node sysfs lists for the PCI device at `address`
/// (`/sys/bus/pci/devices/0000:03:00.0/drm/renderD128`), in `/dev/dri`.
#[cfg(target_os = "linux")]
fn node_at_pci(address: &str) -> Option<PathBuf> {
    let listed = std::fs::read_dir(format!("/sys/bus/pci/devices/{address}/drm")).ok()?;
    let mut nodes: Vec<_> = listed
        .flatten()
        .map(|entry| entry.file_name())
        .filter(|name| name.to_string_lossy().starts_with("renderD"))
        .collect();
    nodes.sort();
    Some(Path::new("/dev/dri").join(nodes.first()?))
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;

    /// A PCI address is written as sysfs names the device: four hex digits of domain, two of bus
    /// and of device, one of function
    #[test]
    fn a_pci_address_reads_as_sysfs_names_it() {
        assert_eq!(pci_address_of((0, 3, 0, 0)), "0000:03:00.0");
        assert_eq!(pci_address_of((1, 0xc5, 0x1f, 7)), "0001:c5:1f.7");
    }
}
