//! The OS services on Windows: the clocks ([`WindowsClock`]), threads ([`WindowsThreads`]), Mach
//! VM's calls over views of pagefile-backed sections ([`WindowsVm`]), whose views of the guest's
//! pages map the sections the backing table names (`crate::os`'s `files`), what the window
//! system's Vulkan extensions take here (`crate::os`'s `wsi`: the Win32 surface), and the user's
//! cache directory (`app_data`). No Objective-C runs here: core pushes no autorelease pool where
//! none is installed.

mod app_data;
mod clock;
mod hints;
mod lease;
mod threads;
mod vm;

use std::ffi::CStr;

pub use clock::WindowsClock;
pub use lease::{GuestMemoryLease, lease_guest_memory};
pub use threads::WindowsThreads;
pub use vm::WindowsVm;

pub(crate) use app_data::cache_directory;
#[cfg(test)]
pub(crate) use vm::allocate_aliasable;
#[cfg(test)]
pub(crate) use vm::section;
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The instance's surface extensions a display's window may present over, besides
/// `VK_KHR_surface`, each asked for where the loader lists it (`crate::os`'s `wsi`): Win32's.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];

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

/// A compile-pool make ended after `took` of wall time: nothing on Windows. Handing back the heap
/// a heavy make left free is glibc's on Linux (`os/linux/heap_trim.rs`); Windows' allocator
/// differs, and is out of its scope.
pub fn make_ended(_took: std::time::Duration) {}

/// The registry ID the platform answers for the runtime's default device in place of its own:
/// none, as there is no Metal here to ask, so the runtime's own stands.
pub fn default_registry_id() -> Option<u64> {
    None
}

/// The feature profile and FP configs Metal answers for the device [`default_registry_id`]
/// names: none, as there is no Metal here to ask.
pub fn default_driver_sizes() -> [Option<u64>; 4] {
    [None; 4]
}
