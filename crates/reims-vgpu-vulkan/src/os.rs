//! The OS services: core's threads, clocks and host VM, each OS's own behind one facade
//! (`OsClock`, `OsThreads`, `OsVm`).
//!
//! - **macOS** (`os/macos.rs`): the system's threads, clocks, Mach VM and autorelease pool, and
//!   the Objective-C message layer the window's AppKit side sends through (`msg`).
//! - **Linux** (`os/linux.rs`): the clocks (`CLOCK_BOOTTIME`, `CLOCK_REALTIME`), threads, and a
//!   host VM over `mmap` that makes views of the guest's pages by mapping their files (the backing
//!   table).
//!
//! Either way the platform's VM reports the pages it changes to the runtime
//! ([`crate::remap::VulkanVm`]), and a display's window presents through a swapchain over the
//! window system's Vulkan extensions (`wsi.rs`), each OS naming its own surface extensions and
//! its loader, and fetching an image's drawable ahead where the driver needs it
//! (`Drawables`: MoltenVK's, on macOS). The user's cache directory, where the library keeps what
//! later processes reuse, is each OS's own too ([`user_cache_directory`]).

#[cfg(any(target_os = "linux", windows))]
mod files;
#[cfg(unix)]
mod home;
#[cfg(any(target_os = "linux", windows))]
mod host_ram;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;
mod wsi;

#[cfg(target_os = "linux")]
use linux as host_os;
#[cfg(target_os = "macos")]
use macos as host_os;
#[cfg(windows)]
use windows as host_os;

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(unix)]
use home::cache_directory as host_cache_directory;
#[cfg(windows)]
use host_os::cache_directory as host_cache_directory;

use crate::core::platform::os::vm::{HostAddr, HostVm as _};

/// Learns whether a device imports views of guest pages mapped through dma-bufs, on Linux
/// (`os/linux/import.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::learn_view_imports;
/// The calling thread's user-space instructions and cycles, which cost tests count in, on Linux
/// (`os/linux/thread_counts.rs`).
#[cfg(all(test, target_os = "linux"))]
pub(crate) use host_os::thread_counts;
pub(crate) use host_os::{
    COPIES_READ_ONLY, Drawables, ImportMade, allocate_aliasable, dma_buf_refusal, host_memory,
    host_memory_vouched, import_view,
};
/// Copies by the GPU of planes another driver wrote into dma-bufs into a runtime buffer, on Linux
/// (`os/linux/plane_copy.rs`).
#[cfg(target_os = "linux")]
pub use host_os::{CopyError, DmaBufPlane, PlaneCopier, PlaneCopy, PlaneFormat};
pub use host_os::{
    GuestMemoryLease, OsClock, OsThreads, OsVm, RegionMemory, default_driver_sizes,
    default_gpu_core_count, default_registry_id, install_process_services, lease_guest_memory,
};
/// Dma-bufs of host memory lent to a device outside the runtime, on Linux
/// (`os/linux/device_lend.rs`).
#[cfg(target_os = "linux")]
pub use host_os::{LendRefusal, LentDmaBuf, dma_buf_of_host};
#[cfg(test)]
pub(crate) use host_os::{PAGE, mapped, writable};
/// Whether one dma-buf takes the pages of a range, as the runtime asks for a texture's own, of a
/// list of ranges, before a view of them is made, or of pieces of pages not made yet, on Linux
/// (`os/linux/dma_buf.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::{one_dma_buf_fits, one_dma_buf_of_fits, one_dma_buf_of_pieces_fits};
/// Hands a device the guest's RAM as it is adopted, on Windows (`os/host_ram.rs`); on Linux the
/// learning of its imports of views hands it (`learn_view_imports`).
#[cfg(windows)]
pub(crate) use host_ram::hand_ram_to;
/// Takes the devices handed the guest's RAM, on Linux and Windows (`os/host_ram.rs`), for the
/// platform's let-go of its devices ([`crate::let_go`]).
#[cfg(any(target_os = "linux", windows))]
pub(crate) use host_ram::take_devices as take_ram_devices;
/// The guest's RAM a device copies by runs of it, on Linux and Windows (`os/host_ram.rs`).
#[cfg(any(target_os = "linux", windows))]
pub(crate) use host_ram::{
    copies as copies_guest_ram, copy_from_host_ram, copy_runs_from_host_ram, copy_to_host_ram,
    ram_over,
};
/// The Objective-C message layer the window's AppKit side sends through, on macOS
/// (`os/macos/msg.rs`).
#[cfg(target_os = "macos")]
pub use macos::msg;
/// Runs a test's calls with their imports refused, so the runtime copies the pages, on macOS
/// (`os/macos.rs`).
#[cfg(all(test, target_os = "macos"))]
pub(crate) use macos::refusing_imports;
/// Every physical device the host's loader lists, as the window's probe read it, which the
/// library's choice of device reads (`crate::device_choice`).
pub(crate) use wsi::seen_devices;
pub use wsi::{add_window_extensions, present_fences};

/// The user's cache directory, where the library keeps for the host's later processes what they
/// would otherwise work out again: the shader cache and its device checks
/// (`crate::platform::cache_directory`), the pipeline records (`crate::records`) and the
/// reims-graphics device's kept answers, each in a directory of its own in it. `.cache` under
/// `HOME` on Linux and macOS (`os/home.rs`), and none where `HOME` is not set; on Windows
/// `reims-vgpu` in the user's local application data folder (`os/windows/app_data.rs`), which a
/// process has whatever its environment, as a QEMU started outside MSYS2's shell has no `HOME`.
/// In a test build, the directory [`TESTS_CACHE_DIRECTORY`] names where it is set.
pub fn user_cache_directory() -> Option<PathBuf> {
    #[cfg(any(test, feature = "testkit"))]
    if let Some(dir) = std::env::var_os(TESTS_CACHE_DIRECTORY) {
        return Some(PathBuf::from(dir));
    }
    host_cache_directory()
}

/// The variable that names a test build's user cache directory ([`user_cache_directory`]), for a
/// test's child process that starts with nothing kept, as a user's first boot does: on Windows the
/// directory is not under `HOME`, so a new `HOME` does not move it.
#[cfg(any(test, feature = "testkit"))]
pub const TESTS_CACHE_DIRECTORY: &str = "RVG_TESTS_CACHE_DIRECTORY";

/// A file of the host's that memory is a shared mapping of, as the library holds it: an open
/// descriptor on Linux and macOS.
#[cfg(unix)]
pub type HostFile = std::os::fd::OwnedFd;
/// A file of the host's that memory is a shared mapping of, as the library holds it: a handle of
/// a section on Windows (`CreateFileMappingW`), whose views the memory is.
#[cfg(windows)]
pub type HostFile = std::os::windows::io::OwnedHandle;

/// One range of the guest's RAM as the host maps it, and the file behind it: `length` bytes at
/// `host`, a shared mapping (`MAP_SHARED`; a view on Windows) of `file` from byte `offset`. A device's lease of the
/// guest's RAM ([`lease_guest_memory`]) is made of these.
#[derive(Clone, Debug)]
pub struct GuestRamFile {
    /// The host address of the range's first byte.
    pub host: u64,
    /// The guest physical address of the range's first byte.
    pub guest: u64,
    /// The range's length in bytes.
    pub length: u64,
    /// The file.
    pub file: Arc<HostFile>,
    /// The file offset of the range's first byte.
    pub offset: u64,
}

/// Pages of the library's own, private and zero-filled: the OS VM's allocation anywhere (Mach's on
/// macOS, an anonymous mapping on Linux), which commits a page only when something first touches
/// it, and which Metal and MoltenVK import in place, as RADV and ANV import it by its pointer.
/// Its drop deallocates the pages through the platform's VM ([`crate::remap::VulkanVm`]), so the
/// unmap waits until the runtime has freed every import of them ([`crate::remap::UnmapHold`]):
/// the runtime runs a no-copy buffer's deallocator at Metal's moment and frees the buffer's import
/// after it, and the deallocator may drop the memory's last holder, as a graphics device's region
/// goes with the last reference its sink's deallocators hold.
#[derive(Debug)]
pub struct PrivateMemory {
    /// The first byte.
    first: HostAddr,
    /// The length in bytes, whole pages.
    length: u64,
}

impl PrivateMemory {
    /// `len` bytes of zeros, rounded up to whole pages; `None` for none, or where the host gives
    /// no address space.
    pub fn new(len: u64) -> Option<PrivateMemory> {
        let vm = OsVm::new();
        let length = len.checked_next_multiple_of(vm.page_size())?;
        if length == 0 {
            return None;
        }
        let first = vm.allocate_anywhere(length).ok()?;
        Some(PrivateMemory { first, length })
    }

    /// Its first byte.
    pub fn base(&self) -> *mut u8 {
        self.first.as_ptr()
    }
}

impl Drop for PrivateMemory {
    /// Deallocates the pages through the platform's VM, which leaves those an import holds mapped
    /// until its hold goes.
    fn drop(&mut self) {
        let _ = crate::remap::VulkanVm::new().deallocate(self.first, self.length);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The tests' page is the host's
    #[test]
    fn the_tests_page_is_the_hosts() {
        assert_eq!(OsVm::new().page_size(), PAGE);
    }

    /// Private memory whose pages a device's import holds stays mapped past its drop, until the
    /// import's hold goes, as a graphics device's region does whose last reference goes in a
    /// no-copy buffer's deallocator, before the runtime frees the buffer's import
    #[test]
    fn private_memory_an_import_holds_is_unmapped_once_its_hold_goes() {
        let memory = PrivateMemory::new(2 * PAGE).unwrap();
        let at = HostAddr::from_ptr(memory.base());
        let held = at.wrapping_add(PAGE);
        let hold = crate::remap::UnmapHold::new(
            std::ptr::NonNull::new(held.as_ptr::<u8>()).unwrap(),
            PAGE,
        );
        // SAFETY: a byte of the memory's second page, mapped read and write.
        unsafe { held.as_ptr::<u8>().write_volatile(0x5a) };
        drop(memory);
        assert!(crate::remap::unmap_waits(at, 2 * PAGE), "the unmap waits");
        // SAFETY: the held page stays mapped while the hold lives.
        let byte = unsafe { held.as_ptr::<u8>().read_volatile() };
        assert_eq!(byte, 0x5a, "the held page stays mapped");
        drop(hold);
        assert!(!crate::remap::unmap_waits(at, 2 * PAGE), "unmapped");
    }

    /// Private memory is whole pages of zeros, and none for no bytes
    #[test]
    fn private_memory_is_whole_pages_of_zeros() {
        use crate::core::num::{ToU64 as _, ToUsize as _};

        assert!(PrivateMemory::new(0).is_none());
        let memory = PrivateMemory::new(PAGE + 1).unwrap();
        assert_eq!(memory.length, 2 * PAGE);
        let at = HostAddr::from_ptr(memory.base());
        assert_eq!(at.addr().to_u64() % PAGE, 0, "page-aligned");
        // SAFETY: the memory's two pages, mapped read and write.
        let bytes = unsafe { std::slice::from_raw_parts(memory.base(), (2 * PAGE).to_usize()) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    /// Each call takes whole pages as Mach's does, which this checks on macOS (measured on the M3
    /// Max, macOS 27.0.1): a fixed overwrite and a remap take their length rounded up to whole
    /// pages from the page of their address, a remap's source from the page of its first byte,
    /// though the bytes run into the next page, and the remap answers the target's page; a
    /// protect and a deallocate take every page the bytes touch; a call of no bytes off a page's
    /// start changes nothing, a remap's refused as an invalid argument
    #[test]
    fn the_vm_spans_its_calls_as_mach_does() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::{HostAddr, KernReturn, VmProt};

        let vm = OsVm::new();
        let page = vm.page_size();
        let at = |base: HostAddr, i: u64| base.wrapping_add(i * page);
        // Four pages that a view can be made of, page `i` filled with `first + i`.
        let pages = |first: u8| {
            let base = allocate_aliasable(4 * page).unwrap();
            for (i, byte) in (0..4).zip(first..) {
                // SAFETY: the page just allocated, readable and writable.
                unsafe {
                    at(base, i)
                        .as_ptr::<u8>()
                        .write_bytes(byte, page.to_usize())
                };
            }
            base
        };
        // The first byte of each page.
        let bytes = |base: HostAddr| -> Vec<u8> {
            // SAFETY: the test's pages, mapped readable.
            (0..4)
                .map(|i| unsafe { at(base, i).as_ptr::<u8>().read_volatile() })
                .collect()
        };
        let near_the_end = page - 0x80;

        let zeroed = pages(0xa0);
        vm.allocate_fixed_overwrite(zeroed.wrapping_add(near_the_end), 0x100)
            .unwrap();
        assert_eq!(bytes(zeroed), [0, 0xa1, 0xa2, 0xa3], "a fixed overwrite");

        let (source, target) = (pages(0x10), pages(0xb0));
        let into_page_1 = at(target, 1).wrapping_add(near_the_end);
        let remapped =
            vm.remap_shared_overwrite(into_page_1, source.wrapping_add(near_the_end), page);
        assert_eq!(remapped, Ok(at(target, 1)));
        assert_eq!(
            bytes(target),
            [0xb0, 0x10, 0xb2, 0xb3],
            "a remap of a page's length"
        );
        let remapped =
            vm.remap_shared_overwrite(target, source.wrapping_add(near_the_end), page + 1);
        assert_eq!(remapped, Ok(target));
        assert_eq!(
            bytes(target),
            [0x10, 0x11, 0xb2, 0xb3],
            "and of a byte more"
        );

        vm.protect_max(target.wrapping_add(near_the_end), 0x100, VmProt::READ)
            .unwrap();
        let writable_pages: Vec<bool> = (0..4).map(|i| writable(at(target, i), page)).collect();
        assert_eq!(writable_pages, [false, false, true, true], "a protect");
        assert!(
            vm.deallocate(zeroed.wrapping_add(near_the_end), 0x100)
                .is_success()
        );
        let mapped_pages: Vec<bool> = (0..4).map(|i| mapped(at(zeroed, i))).collect();
        assert_eq!(mapped_pages, [false, false, true, true], "a deallocate");

        let off = at(target, 2).wrapping_add(0x80);
        assert_eq!(
            vm.remap_shared_overwrite(off, source, 0),
            Err(KernReturn(4))
        );
        vm.allocate_fixed_overwrite(off, 0).unwrap();
        vm.protect_max(off, 0, VmProt::NONE).unwrap();
        assert!(vm.deallocate(off, 0).is_success());
        assert_eq!(bytes(target)[2], 0xb2, "no bytes: the page as it was");
        assert!(mapped(at(target, 2)) && writable(at(target, 2), page));

        for base in [zeroed, source, target] {
            assert!(vm.deallocate(base, 4 * page).is_success());
        }
    }
}
