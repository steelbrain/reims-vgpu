//! The OS services on Windows: the clocks ([`WindowsClock`]), threads ([`WindowsThreads`]), Mach
//! VM's calls over views of pagefile-backed sections ([`WindowsVm`]), whose views of the guest's
//! pages map the sections the backing table names (`crate::os`'s `files`), the runtime's imports of
//! host memory (`import`), with the guest's RAM handed to the devices whole (`crate::os`'s
//! `host_ram`), what the window system's Vulkan extensions take here (`crate::os`'s `wsi`: the
//! Win32 surface and the loader), and the user's cache directory (`app_data`). No Objective-C runs
//! here: core pushes no autorelease pool where none is installed.

mod app_data;
mod clock;
mod hints;
mod import;
mod lease;
mod plane_copy;
mod threads;
mod vm;

use std::ffi::CStr;

use ash::vk;
use metal2vulkan_runtime as rt;

pub use clock::WindowsClock;
pub use lease::{GuestMemoryLease, lease_guest_memory};
pub use plane_copy::{CopyError, PlaneCopier, PlaneCopy, SharedPlanes, TextureFormat};
pub use threads::WindowsThreads;
pub use vm::{RegionMemory, WindowsVm};

pub(crate) use app_data::cache_directory;
pub(crate) use import::{
    COPIES_READ_ONLY, ImportMade, dma_buf_refusal, host_memory, host_memory_vouched,
    import_heap_view, import_view,
};
pub(crate) use vm::allocate_aliasable;
#[cfg(test)]
pub(crate) use vm::section;
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The instance's surface extensions a display's window may present over, besides
/// `VK_KHR_surface`, each asked for where the loader lists it (`crate::os`'s `wsi`): Win32's.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];

/// The device extensions a window's swapchain needs besides `VK_KHR_swapchain`: none.
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];

/// Adds to `options` the device extension the video's copies take here
/// (`plane_copy`): `VK_KHR_external_memory_win32`, which a device enables where it lists it, so
/// that it imports the video's shared textures by their NT handles.
pub(crate) fn add_video_extensions(options: &mut rt::DeviceOptions) {
    let name = ash::khr::external_memory_win32::NAME;
    if !options
        .optional_device_extensions
        .iter()
        .any(|n| n.as_c_str() == name)
    {
        options.optional_device_extensions.push(name.to_owned());
    }
}

/// The Vulkan loader, opened by its usual name (`vulkan-1.dll`), as the runtime opens it.
pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: opening the loader runs its initialization, and it is kept open while the entry
    // lives.
    unsafe { ash::Entry::load() }
}

/// What fetches a swapchain image's drawable ahead of its frame: nothing here, where an image's
/// memory is the driver's own once it is acquired (`crate::swapchain`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drawables {}

impl Drawables {
    /// None on Windows.
    pub(crate) fn load(_raw: &rt::Device) -> Option<Drawables> {
        None
    }

    /// Never called: there is no value.
    ///
    /// # Safety
    ///
    /// None needed.
    pub(crate) unsafe fn fetch(&self, _image: vk::Image) -> bool {
        match *self {}
    }
}

/// The guest's RAM as the devices are handed it (`crate::os`'s `host_ram`): `ram` as it is, as
/// Windows has no dma-bufs to give it sources of.
pub(crate) fn with_ram_sources(ram: rt::HostRam) -> rt::HostRam {
    ram
}

/// Whether the RAM has sources of dma-bufs ([`with_ram_sources`]): never on Windows.
#[cfg(test)]
pub(crate) fn ram_has_sources() -> bool {
    false
}

/// The clocks.
pub type OsClock = WindowsClock;
/// The threads.
pub type OsThreads = WindowsThreads;
/// Mach VM's calls.
pub type OsVm = WindowsVm;

/// Installs what the process's platform needs before it runs, once: nothing on Windows. A view
/// holds its section's handle where Linux's holds a file open, and Windows has no soft limit on a
/// process's handles to raise, as Linux has on its open files.
pub fn install_process_services() {}

/// The registry ID the platform answers for the runtime's default device in place of its own:
/// none, as there is no Metal here to ask, so the runtime's own stands.
pub fn default_registry_id() -> Option<u64> {
    None
}

/// The GPU core count Metal answers for the device [`default_registry_id`] names: none, as there
/// is no Metal here to ask.
pub fn default_gpu_core_count() -> Option<u64> {
    None
}

/// The feature profile and FP configs Metal answers for the device [`default_registry_id`]
/// names: none, as there is no Metal here to ask.
pub fn default_driver_sizes() -> [Option<u64>; 4] {
    [None; 4]
}
