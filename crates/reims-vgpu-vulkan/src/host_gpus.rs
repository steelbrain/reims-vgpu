//! The render node of the GPU a runtime device runs on, which a host video API opens the GPU by:
//! the reims-graphics device's VA-API arm (reims-vgpu's `graphics::vaapi`) opens the runtime
//! device's. It is the node the device's `VK_EXT_physical_device_drm` names, else the one sysfs
//! lists for the PCI address its `VK_EXT_pci_bus_info` names; none where neither does.

use std::path::{Path, PathBuf};

use metal2vulkan_runtime as rt;

/// The render node of `device`'s GPU: the one its DRM node names, else the one sysfs lists for
/// its PCI address; `None` where neither does (module doc).
pub fn render_node_of(device: &rt::Device) -> Option<PathBuf> {
    let properties = device.properties();
    let pci_address = properties.pci_bus.map(pci_address_of);
    (properties.drm_render)
        .and_then(|(major, minor)| node_of(major, minor))
        .or_else(|| pci_address.as_deref().and_then(node_at_pci))
}

/// The address sysfs names a PCI device by, `0000:03:00.0`, of its domain, bus, device and
/// function.
fn pci_address_of((domain, bus, device, function): (u32, u32, u32, u32)) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// The DRM node of character device `major:minor`, as sysfs names it
/// (`/sys/dev/char/226:128` links to its `renderD128`), in `/dev/dri`.
fn node_of(major: i64, minor: i64) -> Option<PathBuf> {
    let link = std::fs::read_link(format!("/sys/dev/char/{major}:{minor}")).ok()?;
    Some(Path::new("/dev/dri").join(link.file_name()?))
}

/// The DRM render node sysfs lists for the PCI device at `address`
/// (`/sys/bus/pci/devices/0000:03:00.0/drm/renderD128`), in `/dev/dri`.
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
