//! The window system's Vulkan extensions, which a display's window presents through
//! (`crate::rt_platform::swapchain`), named in the rt platform's instance and device
//! (`crate::rt_platform::device`), and what the library's choice of device reads of each physical
//! device ([`seen`]).
//!
//! The instance takes each of the OS's surface extensions the loader lists (`crate::os`'s
//! `WINDOW_SURFACE_EXTENSIONS`: on Linux Wayland's, XCB's and Xlib's, so a window opens over
//! whichever the window library picks; on Windows `VK_KHR_win32_surface`; on macOS
//! `VK_EXT_metal_surface`, over a window's `CAMetalLayer`) ([`rt_instance_extensions`]);
//! `VK_KHR_surface` and the device's `VK_KHR_swapchain` are rt's floor's.
//!
//! **Present fences.** The device also takes `VK_KHR_swapchain_maintenance1` (or, where a device
//! lists only that, `VK_EXT_swapchain_maintenance1`), and the instance its surface counterpart
//! and `VK_KHR_get_surface_capabilities2`, where the device and the loader list them
//! ([`rt_present_fences`]): a swapchain then knows when each present is done. Where any of them
//! is missing, the swapchain waits as unextended Vulkan lets it.

use std::ffi::{CStr, CString};

use ash::vk;

use crate::device_choice::{DrmNodes, Seen};

use super::host_os::WINDOW_SURFACE_EXTENSIONS;

/// The device extensions that give present fences, preferred first, each with the instance
/// extension it requires beside `VK_KHR_get_surface_capabilities2`.
const PRESENT_FENCES: [(&CStr, &CStr); 2] = [
    (
        c"VK_KHR_swapchain_maintenance1",
        c"VK_KHR_surface_maintenance1",
    ),
    (
        ash::ext::swapchain_maintenance1::NAME,
        ash::ext::surface_maintenance1::NAME,
    ),
];

/// The instance extensions the rt platform's instance names (`crate::rt_platform::device`), which
/// rt's `Instance::new` enables where the loader lists them: the OS's surface extensions
/// (`WINDOW_SURFACE_EXTENSIONS`), and the instance extensions of present fences
/// (`VK_KHR_get_surface_capabilities2` and each of [`PRESENT_FENCES`]' surface counterparts). rt's
/// floor takes `VK_KHR_surface` itself. A test build adds `VK_EXT_headless_surface`, which the
/// tests' swapchains present to.
pub(crate) fn rt_instance_extensions() -> Vec<&'static CStr> {
    let mut names: Vec<&'static CStr> = WINDOW_SURFACE_EXTENSIONS.to_vec();
    names.push(ash::khr::get_surface_capabilities2::NAME);
    names.extend(PRESENT_FENCES.iter().map(|&(_, surface)| surface));
    #[cfg(any(test, feature = "testkit"))]
    names.push(ash::ext::headless_surface::NAME);
    names
}

/// The device extension that gives the rt platform's device present fences: the first of
/// [`PRESENT_FENCES`] the device lists (`lists`) whose instance extensions the instance enabled
/// (`enabled`, rt's `Instance::extensions`); `None` where there is none. rt enables it with its
/// feature where the device has the feature, and passes over it otherwise.
pub(crate) fn rt_present_fences(
    enabled: &[CString],
    lists: impl Fn(&CStr) -> bool,
) -> Option<&'static CStr> {
    let on = |name: &CStr| enabled.iter().any(|n| n.as_c_str() == name);
    if !on(ash::khr::get_surface_capabilities2::NAME) {
        return None;
    }
    (PRESENT_FENCES.iter())
        .find(|&&(device, surface)| on(surface) && lists(device))
        .map(|&(device, _)| device)
}

/// What the library's choice of device reads of `device` ([`Seen`]): its properties, its
/// largest device-local heap, whether a queue family does graphics and compute work, and its UUID,
/// read where the instance (`properties2`) and the device are at Vulkan 1.1 or later, its DRM
/// nodes where it also lists `VK_EXT_physical_device_drm` (`drm`), and its PCI address where it
/// also lists `VK_EXT_pci_bus_info` (`pci`); `window_lacks`, the window's device extensions it
/// does not list (`Seen::window_lacks`).
///
/// # Safety
///
/// `device` is a physical device `instance` lists, and `instance` is live.
pub(crate) unsafe fn seen(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    properties2: bool,
    drm: bool,
    pci: bool,
    window_lacks: Vec<&'static CStr>,
) -> Seen {
    // SAFETY: the caller's.
    let (properties, memory, families) = unsafe {
        (
            instance.get_physical_device_properties(device),
            instance.get_physical_device_memory_properties(device),
            instance.get_physical_device_queue_family_properties(device),
        )
    };
    let uuid = (properties2 && properties.api_version >= vk::API_VERSION_1_1).then(|| {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: the caller's; the instance and the device are at Vulkan 1.1 or later, where
        // the structure is core.
        unsafe { instance.get_physical_device_properties2(device, &mut asked) };
        id.device_uuid
    });
    let drm = (drm && properties2 && properties.api_version >= vk::API_VERSION_1_1).then(|| {
        let mut nodes = vk::PhysicalDeviceDrmPropertiesEXT::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut nodes);
        // SAFETY: the caller's; the device lists the extension whose structure this is.
        unsafe { instance.get_physical_device_properties2(device, &mut asked) };
        let node = |has: vk::Bool32, major: i64, minor: i64| {
            (has == vk::TRUE).then(|| (u32::try_from(major).ok(), u32::try_from(minor).ok()))
        };
        let pair = |n: Option<(Option<u32>, Option<u32>)>| match n {
            Some((Some(a), Some(b))) => Some((a, b)),
            _ => None,
        };
        DrmNodes {
            primary: pair(node(
                nodes.has_primary,
                nodes.primary_major,
                nodes.primary_minor,
            )),
            render: pair(node(
                nodes.has_render,
                nodes.render_major,
                nodes.render_minor,
            )),
        }
    });
    let pci = (pci && properties2 && properties.api_version >= vk::API_VERSION_1_1).then(|| {
        let mut bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut bus);
        // SAFETY: the caller's; the device lists the extension whose structure this is.
        unsafe { instance.get_physical_device_properties2(device, &mut asked) };
        (
            bus.pci_domain,
            bus.pci_bus,
            bus.pci_device,
            bus.pci_function,
        )
    });
    let work = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    Seen {
        name: properties.device_name_as_c_str().map_or_else(
            |_| String::new(),
            |name| name.to_string_lossy().into_owned(),
        ),
        kind: properties.device_type,
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        graphics_compute: families.iter().any(|f| f.queue_flags.contains(work)),
        device_local: memory
            .memory_heaps_as_slice()
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0),
        uuid,
        drm,
        pci,
        window_lacks,
    }
}
