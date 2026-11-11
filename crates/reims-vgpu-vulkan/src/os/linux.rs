//! The OS services on Linux: the clocks ([`LinuxClock`]), threads ([`LinuxThreads`]), Mach VM's
//! calls over `mmap` ([`LinuxVm`]), whose views of the guest's pages map the files the backing
//! table names (`files`), IOSurface's stand-in over that memory ([`LinuxSurface`]), what the
//! window system's Vulkan extensions take here (`crate::os`'s `wsi`: the surface extensions and
//! the loader), and the GPU's copies of another driver's planes in dma-bufs into a runtime buffer
//! ([`PlaneCopier`]). No Objective-C runs here: core pushes no autorelease pool where none is
//! installed.

mod budget;
mod clock;
mod device_lend;
mod dma_buf;
mod files;
mod hints;
mod host_ram;
mod import;
mod plane_copy;
mod surface;
mod threads;
mod vm;

use std::ffi::CStr;

use ash::vk;
use metal2vulkan_runtime as rt;

pub use clock::LinuxClock;
pub use device_lend::{LendRefusal, LentDmaBuf, dma_buf_of_host};
pub use files::{GuestMemoryLease, lease_guest_memory};
pub use plane_copy::{CopyError, DmaBufPlane, PlaneCopier, PlaneCopy, PlaneFormat};
pub use surface::{LinuxSurface, LinuxSurfaces};
pub use threads::LinuxThreads;
pub use vm::{LinuxVm, RegionMemory};

pub(crate) use dma_buf::{
    fits_one as one_dma_buf_fits, fits_one_of as one_dma_buf_of_fits,
    fits_pieces as one_dma_buf_of_pieces_fits,
};
pub(crate) use host_ram::{
    copies as copies_guest_ram, copy_from_host_ram, copy_to_host_ram, ram_over,
    take_devices as take_ram_devices,
};
#[cfg(test)]
pub(crate) use import::views_through_dma_bufs;
pub(crate) use import::{
    COPIES_READ_ONLY, ImportView, TWINS_READ_ONLY, allocate_zeros, dma_buf_refusal, host_memory,
    host_memory_vouched, import_view, learn_view_imports, shown_at,
};
pub(crate) use surface::{
    own_alias, plane_at, plane_shape, recorded_mut, surface_memory_hold, ycbcr_matrix,
};
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

/// The device extensions a window's swapchain needs besides `VK_KHR_swapchain`: none.
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];

/// The Vulkan loader, opened by its usual name, as the runtime opens it.
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
    /// None on Linux.
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

/// The clocks.
pub type OsClock = LinuxClock;
/// The threads.
pub type OsThreads = LinuxThreads;
/// Mach VM's calls.
pub type OsVm = LinuxVm;
/// An IOSurface: the stand-in's, one reference.
pub type OsSurface = LinuxSurface;
/// IOSurface creation: the stand-in's.
pub type OsSurfaces = LinuxSurfaces;

/// Installs what the process's platform needs before it runs, once: the soft limit of open files
/// raised to the hard limit, as each IOSurface stand-in of its own memory holds a file open
/// (`memfd_create`), as does each range a view can be made of. A limit left lower is logged.
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
