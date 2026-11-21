//! The OS services on Linux: the clocks ([`LinuxClock`]), threads ([`LinuxThreads`]), Mach VM's
//! calls over `mmap` ([`LinuxVm`]), whose views of the guest's pages map the files the backing
//! table names (`files`), dma-bufs of host memory lent to the device (`device_lend`), what
//! the window system's Vulkan extensions take here (`crate::os`'s `wsi`: the surface
//! extensions), and, with glibc, the heap a heavy pipeline make left free handed back to Linux
//! (`heap_trim`). No Objective-C runs here: core pushes no autorelease pool where none is
//! installed.

mod budget;
mod clock;
mod device_lend;
mod dma_buf;
#[cfg(target_env = "gnu")]
mod heap_trim;
mod hints;
mod import;
mod lease;
mod threads;
mod vm;

use std::ffi::CStr;

pub use clock::LinuxClock;
pub use device_lend::{LendRefusal, LentDmaBuf, dma_buf_of_host, map_for_pointer};
pub(crate) use device_lend::{MappingsHeld, dma_bufs_of_host, hold_mappings, through_dma_bufs};
#[cfg(target_env = "gnu")]
pub use heap_trim::make_ended;
#[cfg(test)]
pub(crate) use import::{imports_need_files, several_dma_bufs_refused};
pub(crate) use import::{learn_dma_buf_granule, views_need_files};
pub use lease::{GuestMemoryLease, lease_guest_memory};
pub use threads::LinuxThreads;
pub use vm::LinuxVm;

#[cfg(test)]
pub(crate) use vm::allocate_aliasable;
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The instance's surface extensions a display's window may present over, besides
/// `VK_KHR_surface`, each asked for where the loader lists it (`crate::os`'s `wsi`): Wayland's,
/// XCB's and Xlib's, so a window opens over whichever the window library picks.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 3] = [
    ash::khr::wayland_surface::NAME,
    ash::khr::xcb_surface::NAME,
    ash::khr::xlib_surface::NAME,
];

/// The clocks.
pub type OsClock = LinuxClock;
/// The threads.
pub type OsThreads = LinuxThreads;
/// Mach VM's calls.
pub type OsVm = LinuxVm;

/// Installs what the process's platform needs before it runs, once: the soft limit of open files
/// raised to the hard limit, as each range a view can be made of holds a file open
/// (`memfd_create`). A limit left lower is logged.
pub fn install_process_services() {
    static RAISED: std::sync::Once = std::sync::Once::new();
    RAISED.call_once(|| {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `limit` is a valid rlimit to fill and then to set.
        let raised = unsafe {
            libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0
                && (limit.rlim_cur >= limit.rlim_max || {
                    limit.rlim_cur = limit.rlim_max;
                    libc::setrlimit(libc::RLIMIT_NOFILE, &limit) == 0
                })
        };
        if !raised {
            log::warn!(
                target: "reims_vgpu_vulkan::os",
                "the open-file limit stays where it is ({}): {}",
                limit.rlim_cur,
                std::io::Error::last_os_error()
            );
        }
    });
}

/// A compile-pool make ended after `took` of wall time: nothing where the allocator is not
/// glibc's, whose heap `heap_trim` trims after a heavy make. Other allocators keep free memory
/// otherwise, and are out of its scope.
#[cfg(not(target_env = "gnu"))]
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
