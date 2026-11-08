//! The host's GPUs as the video device picks its own (reims-vgpu's
//! `video::backend_choice`): the one the library's GPU device runs on, and every hardware GPU
//! the host's Vulkan drivers show, each with its PCI address and DRM render node, which the video
//! backends open a GPU by.
//!
//! - **The GPU device's GPU** ([`gpu_device_gpu`]) is the physical device the runtime makes its
//!   system default device of with this platform's options ([`crate::platform::device_options`]),
//!   as the GPU device makes its own: the runtime's identity of it
//!   (`Device::system_default_identity_with`), which it reads by the same selection over the same
//!   physical devices, without making a device. So the answer is the one the GPU device's would
//!   be, whether a GPU device was made first or not, or at all. Its PCI address and DRM render
//!   node are the identity's, as the device made of it answers them (`DeviceProperties::pci_bus`
//!   and `drm_render`, from `VK_EXT_pci_bus_info` and `VK_EXT_physical_device_drm`): no address
//!   where the device lists no such extension, and the render node sysfs lists for the address
//!   where the device names none.
//! - **Every hardware GPU** ([`hardware_gpus`]), for a GPU device on a software rasterizer or a
//!   GPU whose backend has no part: the loader's environment may name one driver alone
//!   (`VK_DRIVER_FILES`, as reims-ctl pins a VM to lavapipe), which the runtime's instance then
//!   loads alone, so these are read on an instance of this module's own that loads every driver
//!   the loader's own search finds itself (`VK_LUNARG_direct_driver_loading`, exclusive), from
//!   the driver manifests of its Linux search paths. Measured with the Vulkan loader 1.4.363 on a
//!   host with an RX 7900 XTX, a Radeon 780M and lavapipe (2026-10-25): such an instance lists all
//!   three under `VK_DRIVER_FILES` naming lavapipe's manifest alone, and still only the one
//!   `VK_LOADER_DEVICE_ID_FILTER` names. A loader without the extension gives the devices of its
//!   environment. CPU implementations are left out, and the rest ranked as the runtime ranks its
//!   devices: discrete, then integrated, virtual and other, each kind in the loader's order. A GPU
//!   two drivers show is listed once, by the first. No file of sysfs states a GPU's kind, which
//!   that ranking needs, so the list is the drivers'. The runtime lists the physical devices of
//!   its own loader environment alone, so this instance is the module's own
//!   (docs/video/bugs.md, U18); `video::backend_choice` asks for the list, and so makes the
//!   instance, only where the GPU device's GPU is a software rasterizer, has no backend with any
//!   part, or is none, and never where that GPU serves.
//!
//! Each is read once a process. The drivers this module loads stay loaded for the process's life,
//! as the runtime keeps its own (`Device::system_default_with`): a driver unloaded and loaded
//! again is one some drivers cannot take indefinitely.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use ash::vk;
use metal2vulkan_runtime as rt;

/// The kind of a GPU, from `VkPhysicalDeviceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuKind {
    /// A discrete GPU.
    Discrete,
    /// A GPU integrated with the host.
    Integrated,
    /// A GPU of a virtual machine.
    Virtual,
    /// A CPU implementation: lavapipe, llvmpipe.
    Cpu,
    /// Any other device.
    Other,
}

impl GpuKind {
    fn of(ty: vk::PhysicalDeviceType) -> GpuKind {
        match ty {
            vk::PhysicalDeviceType::DISCRETE_GPU => GpuKind::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => GpuKind::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => GpuKind::Virtual,
            vk::PhysicalDeviceType::CPU => GpuKind::Cpu,
            _ => GpuKind::Other,
        }
    }

    fn of_runtime(kind: rt::DeviceKind) -> GpuKind {
        match kind {
            rt::DeviceKind::Discrete => GpuKind::Discrete,
            rt::DeviceKind::Integrated => GpuKind::Integrated,
            rt::DeviceKind::Virtual => GpuKind::Virtual,
            rt::DeviceKind::Cpu => GpuKind::Cpu,
            _ => GpuKind::Other,
        }
    }

    /// The order [`hardware_gpus`] ranks kinds in, the runtime's: discrete, integrated, virtual,
    /// other.
    fn rank(self) -> u8 {
        match self {
            GpuKind::Discrete => 0,
            GpuKind::Integrated => 1,
            GpuKind::Virtual => 2,
            GpuKind::Other => 3,
            GpuKind::Cpu => 4,
        }
    }
}

/// One of the host's GPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostGpu {
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub name: String,
    /// Its kind.
    pub kind: GpuKind,
    /// `VkPhysicalDeviceProperties::vendorID`: its PCI vendor ID (0x10de, NVIDIA's).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`.
    pub device_id: u32,
    /// Its PCI address as sysfs names it, `0000:03:00.0`, from `VK_EXT_pci_bus_info`; `None`
    /// where the device lists no such extension, as a CPU implementation does.
    pub pci_address: Option<String>,
    /// Its DRM render node, `/dev/dri/renderD128`: the one `VK_EXT_physical_device_drm` names,
    /// else the one sysfs lists for its PCI address; `None` where neither does.
    pub render_node: Option<PathBuf>,
}

/// The GPU the library's GPU device runs on: the runtime's identity of the physical device it
/// makes its system default device of (module doc); `None` where it makes none.
pub fn gpu_device_gpu() -> Option<HostGpu> {
    static GPU: LazyLock<Option<HostGpu>> = LazyLock::new(|| {
        let identity =
            rt::Device::system_default_identity_with(crate::platform::device_options()).ok()?;
        Some(of_identity(&identity))
    });
    GPU.clone()
}

/// The render node of `device`'s GPU, as [`HostGpu::render_node`] finds it: the one its DRM node
/// names, else the one sysfs lists for its PCI address; `None` where neither does. A host API's
/// video on the GPU a runtime device runs on opens there (the reims-graphics device's VA-API).
pub fn render_node_of(device: &rt::Device) -> Option<PathBuf> {
    let properties = device.properties();
    let pci_address = properties.pci_bus.map(pci_address_of);
    (properties.drm_render)
        .and_then(|(major, minor)| node_of(major, minor))
        .or_else(|| pci_address.as_deref().and_then(node_at_pci))
}

/// The GPU of `identity`, the runtime's identity of a physical device: its address of its PCI
/// bus, and its render node of its DRM node, else of its address (module doc).
fn of_identity(identity: &rt::PhysicalDeviceIdentity) -> HostGpu {
    let pci_address = identity.pci_bus.map(pci_address_of);
    let render_node = (identity.drm_render)
        .and_then(|(major, minor)| node_of(major, minor))
        .or_else(|| pci_address.as_deref().and_then(node_at_pci));
    HostGpu {
        name: identity.name.clone(),
        kind: GpuKind::of_runtime(identity.kind),
        vendor_id: identity.vendor_id,
        device_id: identity.device_id,
        pci_address,
        render_node,
    }
}

/// The address sysfs names a PCI device by, `0000:03:00.0`, of its domain, bus, device and
/// function.
fn pci_address_of((domain, bus, device, function): (u32, u32, u32, u32)) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// The host's hardware GPUs, whichever driver the loader's environment selects, ranked (module
/// doc): for the fallback from the GPU device's GPU alone, as the first call makes a Vulkan
/// instance of this module's own.
pub fn hardware_gpus() -> Vec<HostGpu> {
    static GPUS: LazyLock<Vec<HostGpu>> = LazyLock::new(|| {
        let mut gpus: Vec<HostGpu> = Vec::new();
        for gpu in every_driver_gpu() {
            let seen =
                gpu.pci_address.is_some() && gpus.iter().any(|g| g.pci_address == gpu.pci_address);
            if gpu.kind != GpuKind::Cpu && !seen {
                gpus.push(gpu);
            }
        }
        gpus.sort_by_key(|gpu| gpu.kind.rank());
        gpus
    });
    GPUS.clone()
}

/// Every physical device an instance of this call's own lists, over every driver the loader's
/// search finds (module doc), or, on a loader without `VK_LUNARG_direct_driver_loading`, those of
/// its environment. Nothing where the loader does not open or makes no instance; a device below
/// Vulkan 1.1 is left out.
fn every_driver_gpu() -> Vec<HostGpu> {
    let Ok(entry) = crate::os::open_loader() else {
        return Vec::new();
    };
    // SAFETY: a global command of a loaded entry.
    let version = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(vk::API_VERSION_1_0);
    if version < vk::API_VERSION_1_1 {
        return Vec::new();
    }
    let listed = crate::os::instance_extensions(&entry);
    let direct = listed.iter().any(|p| {
        p.extension_name_as_c_str()
            .is_ok_and(|n| n == ash::lunarg::direct_driver_loading::NAME)
    });
    if !direct {
        log::info!(
            "the Vulkan loader has no VK_LUNARG_direct_driver_loading: the hardware GPUs are those \
             its environment shows"
        );
    }
    let drivers: Vec<vk::DirectDriverLoadingInfoLUNARG<'_>> = if direct {
        driver_entries()
            .into_iter()
            .map(|get| {
                vk::DirectDriverLoadingInfoLUNARG::default().pfn_get_instance_proc_addr(Some(get))
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut list = vk::DirectDriverLoadingListLUNARG::default()
        .mode(vk::DirectDriverLoadingModeLUNARG::EXCLUSIVE)
        .drivers(&drivers);
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let extensions = [ash::lunarg::direct_driver_loading::NAME.as_ptr()];
    let mut info = vk::InstanceCreateInfo::default().application_info(&app);
    if direct && !drivers.is_empty() {
        info = info
            .enabled_extension_names(&extensions)
            .push_next(&mut list);
    }
    // SAFETY: a valid create info with no layers, and no extension but one the loader lists, its
    // drivers' entry points those of libraries kept loaded for the process's life.
    let Ok(instance) = (unsafe { entry.create_instance(&info, None) }) else {
        return Vec::new();
    };
    // SAFETY: `instance` is live, at Vulkan 1.1, and each physical device is one it lists.
    let found = unsafe {
        let devices = instance.enumerate_physical_devices().unwrap_or_default();
        devices
            .into_iter()
            .filter_map(|device| read_gpu(&instance, device))
            .collect()
    };
    // SAFETY: nothing made of `instance` outlives it.
    unsafe { instance.destroy_instance(None) };
    found
}

/// The identity of `device`, a physical device of `instance`, at Vulkan 1.1 or later; `None`
/// below it.
///
/// # Safety
/// `instance` is live, at Vulkan 1.1, and `device` is one it lists.
unsafe fn read_gpu(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<HostGpu> {
    // SAFETY: the caller's contract.
    let core = unsafe { instance.get_physical_device_properties(device) };
    if core.api_version < vk::API_VERSION_1_1 {
        return None;
    }
    // SAFETY: the caller's contract.
    let names: Vec<CString> = unsafe { instance.enumerate_device_extension_properties(device) }
        .unwrap_or_default()
        .iter()
        .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
        .collect();
    let lists = |name: &CStr| names.iter().any(|n| n.as_c_str() == name);
    let (has_pci, has_drm) = (
        lists(ash::ext::pci_bus_info::NAME),
        lists(ash::ext::physical_device_drm::NAME),
    );
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if has_pci {
        properties = properties.push_next(&mut pci);
    }
    if has_drm {
        properties = properties.push_next(&mut drm);
    }
    // SAFETY: the caller's contract; each structure chained is of an extension the device lists.
    unsafe { instance.get_physical_device_properties2(device, &mut properties) };
    let name = core
        .device_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let pci_address = has_pci.then(|| {
        pci_address_of((
            pci.pci_domain,
            pci.pci_bus,
            pci.pci_device,
            pci.pci_function,
        ))
    });
    let render_node = (has_drm && drm.has_render == vk::TRUE)
        .then(|| node_of(drm.render_major, drm.render_minor))
        .flatten()
        .or_else(|| pci_address.as_deref().and_then(node_at_pci));
    Some(HostGpu {
        name,
        kind: GpuKind::of(core.device_type),
        vendor_id: core.vendor_id,
        device_id: core.device_id,
        pci_address,
        render_node,
    })
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

/// A driver's `vk_icdGetInstanceProcAddr` (the loader's driver interface).
type DriverEntry = unsafe extern "system" fn(vk::Instance, *const c_char) -> vk::PFN_vkVoidFunction;

/// `vk_icdGetInstanceProcAddr` of every driver the loader's Linux search finds ([`manifests`]),
/// each library loaded once and kept loaded (module doc); a manifest whose library does not load
/// here, as a 32-bit driver's does not, is left out.
fn driver_entries() -> Vec<DriverEntry> {
    let mut entries: Vec<DriverEntry> = Vec::new();
    for manifest in manifests() {
        let Some(library) = library_of(&manifest) else {
            continue;
        };
        let Ok(path) = CString::new(library.into_os_string().into_encoded_bytes()) else {
            continue;
        };
        // SAFETY: loading a Vulkan driver runs its initializers, as the loader's load of it
        // does; the handle is never closed (module doc).
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }
        // SAFETY: `handle` is a live library handle, and the name is NUL-terminated.
        let symbol: *mut c_void =
            unsafe { libc::dlsym(handle, c"vk_icdGetInstanceProcAddr".as_ptr()) };
        if symbol.is_null() {
            continue;
        }
        // SAFETY: a Vulkan driver's `vk_icdGetInstanceProcAddr` has this signature (the loader's
        // driver interface), and its library stays loaded.
        let entry = unsafe { std::mem::transmute::<*mut c_void, DriverEntry>(symbol) };
        if !entries.iter().any(|e| std::ptr::fn_addr_eq(*e, entry)) {
            entries.push(entry);
        }
    }
    entries
}

/// The driver manifests (`*.json`) of the loader's Linux search paths, in its order:
/// `$XDG_CONFIG_HOME`, `$XDG_CONFIG_DIRS`, `/etc`, `$XDG_DATA_HOME` and `$XDG_DATA_DIRS`, each
/// with `vulkan/icd.d`, with their defaults (`~/.config`, `/etc/xdg`, `~/.local/share`,
/// `/usr/local/share:/usr/share`), each directory's in name order; the loader's environment
/// (`VK_DRIVER_FILES`, `VK_ADD_DRIVER_FILES`) is not read.
fn manifests() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    let list = |name: &str, default: &str| -> Vec<PathBuf> {
        let value = var(name).unwrap_or_else(|| default.into());
        std::env::split_paths(&value)
            .filter(|p| !p.as_os_str().is_empty())
            .collect()
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    dirs.extend(
        var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join(".config"))),
    );
    dirs.extend(list("XDG_CONFIG_DIRS", "/etc/xdg"));
    dirs.push("/etc".into());
    dirs.extend(
        var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join(".local/share"))),
    );
    dirs.extend(list("XDG_DATA_DIRS", "/usr/local/share:/usr/share"));
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        let Ok(listed) = std::fs::read_dir(dir.join("vulkan/icd.d")) else {
            continue;
        };
        let mut here: Vec<PathBuf> = listed
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "json"))
            .collect();
        here.sort();
        for path in here {
            if !files.contains(&path) {
                files.push(path);
            }
        }
    }
    files
}

/// The driver library a manifest names (`ICD.library_path`): a bare file name as it stands, for
/// the dynamic linker's search, and a relative path from the manifest's directory, as the loader
/// takes them; `None` for a manifest that does not read.
fn library_of(manifest: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(manifest).ok()?;
    library_in(&text, manifest.parent()?)
}

/// [`library_of`] over a manifest's text, `dir` its directory.
fn library_in(text: &str, dir: &Path) -> Option<PathBuf> {
    let json: serde_json::Value = serde_json::from_str(text).ok()?;
    let path = Path::new(json.get("ICD")?.get("library_path")?.as_str()?);
    Some(if path.is_absolute() || path.components().count() == 1 {
        path.to_path_buf()
    } else {
        dir.join(path)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest's library is a bare name for the linker's search, an absolute path as it is,
    /// and a relative one from the manifest's directory; a manifest without one names none
    #[test]
    fn a_manifest_names_its_library_as_the_loader_reads_it() {
        let dir = Path::new("/usr/share/vulkan/icd.d");
        let manifest = |path: &str| {
            format!(r#"{{"file_format_version": "1.0.1", "ICD": {{"library_path": "{path}"}}}}"#)
        };
        assert_eq!(
            library_in(&manifest("libvulkan_lvp.so"), dir),
            Some(PathBuf::from("libvulkan_lvp.so"))
        );
        assert_eq!(
            library_in(&manifest("/usr/lib/libvulkan_radeon.so"), dir),
            Some(PathBuf::from("/usr/lib/libvulkan_radeon.so"))
        );
        assert_eq!(
            library_in(&manifest("../../../lib/libvulkan_intel.so"), dir),
            Some(dir.join("../../../lib/libvulkan_intel.so"))
        );
        assert_eq!(library_in(r#"{"ICD": {}}"#, dir), None);
        assert_eq!(library_in("not json", dir), None);
    }

    /// The hardware GPUs are ranked as the runtime ranks devices, with no CPU implementation
    #[test]
    fn kinds_rank_as_the_runtime_ranks_them() {
        let mut kinds = [
            GpuKind::Other,
            GpuKind::Integrated,
            GpuKind::Virtual,
            GpuKind::Discrete,
        ];
        kinds.sort_by_key(|k| k.rank());
        assert_eq!(
            kinds,
            [
                GpuKind::Discrete,
                GpuKind::Integrated,
                GpuKind::Virtual,
                GpuKind::Other
            ]
        );
    }

    /// On a host with a Vulkan driver, the GPU device's GPU is the physical device the runtime
    /// makes its default device of: its name, kind and IDs, and its PCI address and DRM render
    /// node as that device answers them; a hardware one has its PCI address, and one with a
    /// render node is listed among the hardware GPUs with the same address and node
    #[test]
    fn the_gpu_devices_gpu_is_the_runtimes_default_device() {
        let Ok(device) = rt::Device::system_default_with(crate::platform::device_options()) else {
            return;
        };
        let p = device.properties();
        let gpu = gpu_device_gpu().expect("the default device's GPU");
        assert_eq!(gpu.name, p.name);
        assert_eq!(gpu.kind, GpuKind::of_runtime(p.kind));
        assert_eq!((gpu.vendor_id, gpu.device_id), (p.vendor_id, p.device_id));
        assert_eq!(gpu.pci_address, p.pci_bus.map(pci_address_of));
        if let Some((major, minor)) = p.drm_render {
            assert_eq!(gpu.render_node, node_of(major, minor), "{gpu:?}");
        }
        let hardware = hardware_gpus();
        assert!(hardware.iter().all(|g| g.kind != GpuKind::Cpu));
        if gpu.kind != GpuKind::Cpu {
            assert!(gpu.pci_address.is_some(), "{gpu:?}");
            let listed = hardware.iter().find(|g| g.pci_address == gpu.pci_address);
            let listed = listed.expect("the GPU device's GPU among the hardware GPUs");
            assert_eq!(listed.render_node, gpu.render_node, "{listed:?}");
        }
    }

    /// A PCI address is written as sysfs names the device: four hex digits of domain, two of bus
    /// and of device, one of function
    #[test]
    fn a_pci_address_reads_as_sysfs_names_it() {
        assert_eq!(pci_address_of((0, 3, 0, 0)), "0000:03:00.0");
        assert_eq!(pci_address_of((1, 0xc5, 0x1f, 7)), "0001:c5:1f.7");
    }
}
