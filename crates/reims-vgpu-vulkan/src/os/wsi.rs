//! The window system's Vulkan extensions, which a display's window presents through
//! (`crate::swapchain`): the runtime's device is made before any window, so its options name
//! them up front ([`add_window_extensions`]).
//!
//! The instance takes `VK_KHR_surface` and each of the OS's surface extensions the loader lists
//! (`crate::os`'s `WINDOW_SURFACE_EXTENSIONS`: on Linux Wayland's, XCB's and Xlib's, so a window
//! opens over whichever the window library picks; on macOS `VK_EXT_metal_surface`, over a
//! window's `CAMetalLayer`). The device takes `VK_KHR_swapchain`, and the OS's own device
//! extensions (`WINDOW_DEVICE_EXTENSIONS`: on macOS `VK_EXT_metal_objects`, whose export of an
//! image's Metal texture fetches its drawable, `crate::swapchain`), where every physical device
//! lists them: the runtime refuses a device whose extension it asked for is missing, and which
//! physical device it picks is its own choice, so a host where one of them lacks one gets no
//! swapchain at all, and its windows say so and stay closed, rather than losing the device. The
//! physical devices' lists are read through an instance of the probe's own, made with no
//! extensions but `VK_KHR_portability_enumeration` where the loader lists it, as the runtime
//! makes its own (so the probe lists MoltenVK, a portability driver, as the runtime does), and
//! destroyed at once. The loader is the one the runtime opens (`crate::os`'s `open_loader`).
//!
//! **Present fences.** The device also takes `VK_KHR_swapchain_maintenance1` (or, where a device
//! lists only that, `VK_EXT_swapchain_maintenance1`), and the instance its surface counterpart
//! and `VK_KHR_get_surface_capabilities2`, where every physical device lists the extension and
//! has its feature, `swapchainMaintenance1`, and the loader lists the instance's: a swapchain
//! then knows when each present is done ([`crate::swapchain`]). The runtime enables the feature
//! of a device extension it knows where the device lists it, and refuses a device that lacks it
//! (`Error::ExtensionFeature`), which dies at the creator ([`crate::platform`]): the probe asks
//! each device for the feature as the runtime does, so no device it lists is refused so. Where
//! any of them is missing, the swapchain waits as unextended Vulkan lets it, and the reason is
//! logged.
//!
//! **Requirements.** Each device extension named here comes with the instance extensions it
//! requires (`VK_KHR_swapchain` with `VK_KHR_surface`; `swapchain_maintenance1` with its surface
//! counterpart and `VK_KHR_get_surface_capabilities2`), which the runtime checks before it makes
//! the instance (`Error::InstanceExtensionRequired`).
//!
//! **The loader's list in one call.** The loader's instance extensions are read with one
//! `vkEnumerateInstanceExtensionProperties` call into room for [`LISTED_FIRST`] of them, asked
//! again with twice the room while the loader answers `VK_INCOMPLETE`
//! ([`instance_extensions`]), where ash's read makes two calls, for the count and then the list.
//! The loader loads and asks every driver its manifests name at each such call, and at each
//! `vkCreateInstance`, and unloads one that makes no instance, and a driver's load can be slow.
//! Measured on a host with a Radeon RX 550, a Radeon 780M and NVIDIA's driver package but no
//! NVIDIA GPU (Vulkan loader 1.4.363, NVIDIA 615.78.08, Mesa 26.2.6, 2026-11-07): at each call,
//! NVIDIA's driver library read the PCI configuration of each of the host's 45 PCI devices, which
//! resumed each one runtime power management had suspended, 13 ms for most and 140 ms for a USB
//! controller, 0.25 to 0.7 s a call. A test process that made the platform's device made 13 such
//! calls and ran 7 to 10 s, against 0.05 s with that driver's manifest left out of the loader's
//! search; 3 of the calls were this module's, 2 of them its list's read through ash.

use std::ffi::{CStr, CString};

use ash::vk;
use metal2vulkan_runtime as rt;

use super::host_os::{WINDOW_DEVICE_EXTENSIONS, WINDOW_SURFACE_EXTENSIONS, open_loader};

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

/// Adds to `options` the window system's extensions this host's loader and devices list (the
/// module doc). Where the loader does not open, or lists no `VK_KHR_surface`, it adds nothing,
/// and says why once.
pub fn add_window_extensions(options: &mut rt::DeviceOptions) {
    let entry = match open_loader() {
        Ok(entry) => entry,
        Err(e) => {
            log::warn!(
                target: "reims_vgpu_vulkan::wsi",
                "the Vulkan loader does not open ({e}): a display's window cannot present"
            );
            return;
        }
    };
    let listed = instance_extensions(&entry);
    let listed: Vec<&CStr> = listed
        .iter()
        .filter_map(|p| p.extension_name_as_c_str().ok())
        .collect();
    if !listed.contains(&ash::khr::surface::NAME) {
        log::warn!(
            target: "reims_vgpu_vulkan::wsi",
            "the Vulkan loader lists no VK_KHR_surface: a display's window cannot present"
        );
        return;
    }
    add(&mut options.instance_extensions, ash::khr::surface::NAME);
    for name in WINDOW_SURFACE_EXTENSIONS {
        if listed.contains(&name) {
            add(&mut options.instance_extensions, name);
        }
    }
    let devices = probe(&entry, &listed);
    let wanted = std::iter::once(ash::khr::swapchain::NAME).chain(WINDOW_DEVICE_EXTENSIONS);
    for name in wanted.clone() {
        if !devices.every_lists(name) {
            log::warn!(
                target: "reims_vgpu_vulkan::wsi",
                "a physical device lists no {}: a display's window cannot present",
                name.to_string_lossy()
            );
            return;
        }
    }
    for name in wanted {
        add(&mut options.device_extensions, name);
    }
    match fence_extensions(&listed, &devices) {
        Ok((device, surface)) => {
            add(
                &mut options.instance_extensions,
                ash::khr::get_surface_capabilities2::NAME,
            );
            add(&mut options.instance_extensions, surface);
            add(&mut options.device_extensions, device);
        }
        Err(why) => log::info!(
            target: "reims_vgpu_vulkan::wsi",
            "no present fences ({why}): a window's swapchain waits for its queue to idle where a \
             present is not known done"
        ),
    }
}

/// Whether `options` give a device present fences: they list one of the device extensions
/// [`add_window_extensions`] adds for them, whose feature the runtime then enables on every
/// device it makes of them (the module doc).
pub fn present_fences(options: &rt::DeviceOptions) -> bool {
    options.device_extensions.iter().any(|name| {
        PRESENT_FENCES
            .iter()
            .any(|(device, _)| name.as_c_str() == *device)
    })
}

/// The room the first call of [`instance_extensions`] has: more than any loader measured lists
/// (26 on Linux, with the loader 1.4.363 and Mesa's and NVIDIA's drivers).
const LISTED_FIRST: u32 = 64;

/// The most room [`instance_extensions`] asks with, past which a loader that still answers
/// `VK_INCOMPLETE` is taken to list nothing, as a failed call is.
const LISTED_MOST: u32 = 1 << 16;

/// The instance extensions `entry`'s loader lists, in its order, read in one call where they fit
/// in room for [`LISTED_FIRST`] (the module doc); none where the loader fails the call.
pub(crate) fn instance_extensions(entry: &ash::Entry) -> Vec<vk::ExtensionProperties> {
    instance_extensions_from(entry, LISTED_FIRST)
}

/// [`instance_extensions`], its first call with room for `first` (at least one), and each next
/// with twice the last's, while the loader answers `VK_INCOMPLETE`: it then writes as many as
/// there is room for, and the count it leaves is that room, not how many it lists.
fn instance_extensions_from(entry: &ash::Entry, first: u32) -> Vec<vk::ExtensionProperties> {
    let mut room = first.max(1);
    while room <= LISTED_MOST {
        let mut listed: Vec<vk::ExtensionProperties> = Vec::with_capacity(room as usize);
        let mut count = room;
        // SAFETY: a global command of a loaded entry, with no layer named, and room for `count`
        // properties at the pointer.
        let result = unsafe {
            (entry.fp_v1_0().enumerate_instance_extension_properties)(
                std::ptr::null(),
                &mut count,
                listed.as_mut_ptr(),
            )
        };
        match result {
            vk::Result::SUCCESS => {
                // SAFETY: the loader wrote `count` properties, at most the room asked for.
                unsafe { listed.set_len(count.min(room) as usize) };
                return listed;
            }
            vk::Result::INCOMPLETE => room *= 2,
            _ => break,
        }
    }
    Vec::new()
}

/// Adds `name` to `names` where it is not there yet.
fn add(names: &mut Vec<CString>, name: &CStr) {
    if !names.iter().any(|n| n.as_c_str() == name) {
        names.push(name.to_owned());
    }
}

/// The device extension that gives present fences on every physical device of `devices`, and
/// the instance extension it requires, of those the loader lists (`listed`): the first of
/// [`PRESENT_FENCES`] every device lists, where every device has the feature. Otherwise why
/// not.
fn fence_extensions(
    listed: &[&CStr],
    devices: &Probed,
) -> Result<(&'static CStr, &'static CStr), String> {
    if !listed.contains(&ash::khr::get_surface_capabilities2::NAME) {
        return Err("the loader lists no VK_KHR_get_surface_capabilities2".to_owned());
    }
    let Some(&(device, surface)) = PRESENT_FENCES
        .iter()
        .find(|(device, surface)| listed.contains(surface) && devices.every_lists(device))
    else {
        return Err(
            "no swapchain_maintenance1 that every physical device and the loader list".to_owned(),
        );
    };
    if !devices.every_has_maintenance1 {
        return Err("a physical device lacks the feature swapchainMaintenance1".to_owned());
    }
    Ok((device, surface))
}

/// What the probe read of every physical device ([`probe`]).
#[derive(Debug, Default)]
struct Probed {
    /// Each device's extensions; empty where there is no device.
    extensions: Vec<Vec<CString>>,
    /// Whether every device has the feature `swapchainMaintenance1`, asked of those that list
    /// one of [`PRESENT_FENCES`]' device extensions; false where there is no device, or the
    /// loader cannot ask.
    every_has_maintenance1: bool,
}

impl Probed {
    /// Whether there is a device and every one lists `name`.
    fn every_lists(&self, name: &CStr) -> bool {
        !self.extensions.is_empty()
            && self
                .extensions
                .iter()
                .all(|names| names.iter().any(|n| n.as_c_str() == name))
    }
}

/// Every physical device `entry`'s loader lists, read through an instance of this call's own, at
/// Vulkan 1.1 where the loader takes it, for the features query, with portability drivers
/// enumerated where the loader lists the instance extensions it names (`listed`), as the runtime
/// enumerates them: nothing where there is none.
fn probe(entry: &ash::Entry, listed: &[&CStr]) -> Probed {
    // SAFETY: a global command of a loaded entry.
    let version = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(vk::API_VERSION_1_0);
    let features2 = version >= vk::API_VERSION_1_1;
    let app = vk::ApplicationInfo::default().api_version(if features2 {
        vk::API_VERSION_1_1
    } else {
        vk::API_VERSION_1_0
    });
    let portability = listed.contains(&ash::khr::portability_enumeration::NAME);
    let extensions = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let mut info = vk::InstanceCreateInfo::default().application_info(&app);
    if portability {
        info = info
            .enabled_extension_names(&extensions)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    // SAFETY: a valid create info with no layers, and no extension but one the loader lists.
    let Ok(instance) = (unsafe { entry.create_instance(&info, None) }) else {
        return Probed::default();
    };
    // SAFETY: `instance` is live, and each physical device is one it lists.
    let probed = unsafe {
        let devices = instance.enumerate_physical_devices().unwrap_or_default();
        let extensions: Vec<Vec<CString>> = devices
            .iter()
            .map(|&device| {
                instance
                    .enumerate_device_extension_properties(device)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
                    .collect()
            })
            .collect();
        let every_has_maintenance1 = features2
            && !devices.is_empty()
            && devices.iter().zip(&extensions).all(|(&device, names)| {
                let lists = names
                    .iter()
                    .any(|n| PRESENT_FENCES.iter().any(|(d, _)| n.as_c_str() == *d));
                // A device that lists neither extension is not asked: none is chosen then.
                !lists || {
                    let properties = instance.get_physical_device_properties(device);
                    properties.api_version >= vk::API_VERSION_1_1 && {
                        let mut maintenance1 =
                            vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
                        let mut features =
                            vk::PhysicalDeviceFeatures2::default().push_next(&mut maintenance1);
                        instance.get_physical_device_features2(device, &mut features);
                        maintenance1.swapchain_maintenance1 == vk::TRUE
                    }
                }
            });
        Probed {
            extensions,
            every_has_maintenance1,
        }
    };
    // SAFETY: nothing made of `instance` outlives it.
    unsafe { instance.destroy_instance(None) };
    probed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host with a Vulkan driver gets `VK_KHR_surface` and a swapchain, and nothing twice
    #[test]
    fn the_window_systems_extensions_are_named_once() {
        let mut options = rt::DeviceOptions::default();
        options
            .instance_extensions
            .push(ash::khr::surface::NAME.to_owned());
        add_window_extensions(&mut options);
        let surface = ash::khr::surface::NAME;
        let count = options
            .instance_extensions
            .iter()
            .filter(|n| n.as_c_str() == surface)
            .count();
        assert_eq!(count, 1);
        assert!(
            options
                .device_extensions
                .iter()
                .any(|n| n.as_c_str() == ash::khr::swapchain::NAME)
        );
    }

    /// Present fences take the KHR extension where every device lists it, else the EXT one,
    /// each with its own surface extension, and none where a device lacks the feature, a device
    /// lists neither, or the loader lists the instance's extensions not
    #[test]
    fn present_fences_take_what_every_device_lists_and_has() {
        let khr = c"VK_KHR_swapchain_maintenance1";
        let ext = ash::ext::swapchain_maintenance1::NAME;
        let probed = |lists: &[&[&CStr]], has: bool| Probed {
            extensions: lists
                .iter()
                .map(|names| names.iter().map(|n| (*n).to_owned()).collect())
                .collect(),
            every_has_maintenance1: has,
        };
        let loader = [
            ash::khr::get_surface_capabilities2::NAME,
            c"VK_KHR_surface_maintenance1",
            ash::ext::surface_maintenance1::NAME,
        ];
        let both: &[&CStr] = &[khr, ext];
        let ext_only: &[&CStr] = &[ext];
        assert_eq!(
            fence_extensions(&loader, &probed(&[both, both], true)),
            Ok((khr, c"VK_KHR_surface_maintenance1"))
        );
        assert_eq!(
            fence_extensions(&loader, &probed(&[both, ext_only], true)),
            Ok((ext, ash::ext::surface_maintenance1::NAME))
        );
        assert!(fence_extensions(&loader, &probed(&[both, both], false)).is_err());
        assert!(fence_extensions(&loader, &probed(&[both, &[]], true)).is_err());
        assert!(fence_extensions(&loader, &probed(&[], true)).is_err());
        assert!(fence_extensions(&loader[..1], &probed(&[both], true)).is_err());
        assert!(fence_extensions(&loader[1..], &probed(&[both], true)).is_err());
    }

    /// The loader's instance extensions read in one call are those ash's read of two calls lists,
    /// in its order, whatever room the first call has
    #[test]
    fn the_instance_extensions_are_the_loaders_whatever_the_first_room() {
        let Ok(entry) = open_loader() else {
            return;
        };
        let named = |listed: &[vk::ExtensionProperties]| -> Vec<(CString, u32)> {
            listed
                .iter()
                .filter_map(|p| {
                    Some((p.extension_name_as_c_str().ok()?.to_owned(), p.spec_version))
                })
                .collect()
        };
        // SAFETY: a global command of a loaded entry.
        let asked = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
        assert!(!asked.is_empty());
        assert_eq!(named(&instance_extensions(&entry)), named(&asked));
        let len = u32::try_from(asked.len()).unwrap();
        for first in [0, 1, 2, len - 1, len, len + 1] {
            assert_eq!(
                named(&instance_extensions_from(&entry, first)),
                named(&asked),
                "first room {first}"
            );
        }
    }

    /// Options that list a present-fence extension give present fences; others do not
    #[test]
    fn present_fences_are_read_of_the_options() {
        let mut options = rt::DeviceOptions::default();
        assert!(!present_fences(&options));
        add(&mut options.device_extensions, ash::khr::swapchain::NAME);
        assert!(!present_fences(&options));
        add(
            &mut options.device_extensions,
            ash::ext::swapchain_maintenance1::NAME,
        );
        assert!(present_fences(&options));
    }
}
