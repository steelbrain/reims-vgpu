//! The OS services: core's threads, clocks and host VM, each OS's own behind one facade
//! (`OsClock`, `OsThreads`, `OsVm`).
//!
//! - **macOS** (`os/macos.rs`): the system's threads, clocks, Mach VM and autorelease pool, and
//!   the Objective-C message layer the window's AppKit side sends through (`msg`).
//! - **Linux** (`os/linux.rs`): the clocks (`CLOCK_BOOTTIME`, `CLOCK_REALTIME`), threads, and a
//!   host VM over `mmap` that makes views of the guest's pages by mapping their files (the backing
//!   table).
//!
//! - **Windows** (`os/windows.rs`): the clocks, threads, and a host VM over views of
//!   pagefile-backed sections, whose views of the guest's pages map the sections the backing
//!   table names.
//!
//! A display's window presents through a swapchain over the window system's Vulkan extensions
//! (`wsi.rs`), each OS naming its own surface extensions. The user's cache directory, where the
//! library keeps what later processes reuse, is each OS's own too ([`user_cache_directory`]).

#[cfg(any(target_os = "linux", windows))]
mod files;
#[cfg(unix)]
mod home;
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

#[cfg(test)]
pub(crate) use host_os::allocate_aliasable;
pub use host_os::{
    GuestMemoryLease, OsClock, OsThreads, OsVm, default_driver_sizes, default_registry_id,
    install_process_services, lease_guest_memory,
};
/// Dma-bufs of host memory lent to a device outside the runtime, and that memory mapped from its
/// files for one that pins a pointer's pages, on Linux (`os/linux/device_lend.rs`).
#[cfg(target_os = "linux")]
pub use host_os::{LendRefusal, LentDmaBuf, dma_buf_of_host, map_for_pointer};
/// Whether some of a range's pages lie in a view mapped through a dma-buf, which no kernel pin
/// takes, as the rt platform asks before it hands rt their pointer, the hold that keeps a range's
/// pages as they are mapped while rt may take them by it, and dma-bufs of a range's pages cut for
/// rt to bind behind one buffer where one dma-buf does not hold them, on Linux
/// (`os/linux/device_lend.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::{MappingsHeld, dma_bufs_of_host, hold_mappings, through_dma_bufs};
#[cfg(test)]
pub(crate) use host_os::{PAGE, mapped, writable};
/// Whether views are mapped from their files for the imports of their pages, and whether those
/// more than one dma-buf holds are, on Linux (`os/linux/import.rs`).
#[cfg(all(test, target_os = "linux"))]
pub(crate) use host_os::{imports_need_files, several_dma_bufs_refused};
/// The settings, from the rt platform's device as it is made and at rt's first refusal of a
/// view's one dma-buf, of views mapped from their files for the imports of their pages, on Linux
/// (`os/linux/import.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::{learn_dma_buf_granule, views_need_files};
/// The Objective-C message layer the window's AppKit side sends through, on macOS
/// (`os/macos/msg.rs`).
#[cfg(target_os = "macos")]
pub use macos::msg;
/// What the library's choice of device reads of one physical device, from an instance of the
/// caller's (the rt platform's, `crate::rt_platform`).
pub(crate) use wsi::seen;
/// The window system's extensions of the rt platform's instance and device
/// (`crate::rt_platform::device`).
pub(crate) use wsi::{rt_instance_extensions, rt_present_fences};

/// The rt platform's instance with `extensions`: on macOS with the Vulkan driver the platform
/// runs there, Mesa's KosmicKrisp, which the process loads (`os/macos/rt_driver.rs`); elsewhere
/// rt's `Instance::new`, over the drivers the host's loader finds.
#[cfg(target_os = "macos")]
pub(crate) use macos::rt_instance;
/// The rt platform's instance with `extensions`: on macOS with the Vulkan driver the platform
/// runs there, Mesa's KosmicKrisp, which the process loads (`os/macos/rt_driver.rs`); elsewhere
/// rt's `Instance::new`, over the drivers the host's loader finds.
#[cfg(not(target_os = "macos"))]
pub(crate) fn rt_instance(
    extensions: &[&std::ffi::CStr],
) -> Result<metal2vulkan_rt::Instance, metal2vulkan_rt::Error> {
    metal2vulkan_rt::Instance::new(extensions)
}

/// `CLOCK_MONOTONIC_RAW` in nanoseconds, on Linux and macOS, for a diagnostic line whose reader
/// joins it to its own reading of that clock (reims-vgpu's `graphics::summary`); `None` on
/// Windows, which has no such clock, or where it cannot be read.
pub fn raw_mono_nanos() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: a valid clock id and a live out-parameter.
        let read = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
        let secs = u64::try_from(ts.tv_sec).ok()?;
        let nanos = u64::try_from(ts.tv_nsec).ok()?;
        (read == 0).then(|| secs.saturating_mul(1_000_000_000).saturating_add(nanos))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// The user's cache directory, where the library keeps for the host's later processes what they
/// would otherwise work out again: the shader cache and its device checks ([`cache_directory`])
/// and the reims-graphics device's kept answers, each in a directory of its own in it. `.cache` under
/// `HOME` on Linux and macOS (`os/home.rs`), and none where `HOME` is not set; on Windows
/// `reims-vgpu` in the user's local application data folder (`os/windows/app_data.rs`), which a
/// process has whatever its environment, as a QEMU started outside MSYS2's shell has no `HOME`.
/// In a test build, the directory `TESTS_CACHE_DIRECTORY` names where it is set.
pub fn user_cache_directory() -> Option<PathBuf> {
    #[cfg(any(test, feature = "testkit"))]
    if let Some(dir) = std::env::var_os(TESTS_CACHE_DIRECTORY) {
        return Some(PathBuf::from(dir));
    }
    host_cache_directory()
}

/// The library's cache directory, [`CACHE`] in the user's cache directory
/// ([`user_cache_directory`]): the device checks' `device-checks` (the host video's checks of its
/// driver). `None` where the OS names no user cache directory (on Linux and macOS, where `HOME` is
/// not set).
pub fn cache_directory() -> Option<PathBuf> {
    Some(user_cache_directory()?.join(CACHE))
}

/// The library's cache directory's name in the user's cache directory ([`cache_directory`]).
#[cfg(not(any(test, feature = "testkit")))]
const CACHE: &str = "reims-vgpu-vulkan";

/// A test build's cache directory, the tests' own: this crate's unit tests', and every crate's
/// tests', which turn on its `testkit` feature (only dev-dependencies do, `cargo xtask testkit`).
/// The tests write nothing into the one the library's hosts share, nor read the device checks kept
/// there.
#[cfg(any(test, feature = "testkit"))]
const CACHE: &str = "reims-vgpu-vulkan-tests";

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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::platform::os::vm::HostVm as _;

    /// The tests' page is the host's
    #[test]
    fn the_tests_page_is_the_hosts() {
        assert_eq!(OsVm::new().page_size(), PAGE);
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
