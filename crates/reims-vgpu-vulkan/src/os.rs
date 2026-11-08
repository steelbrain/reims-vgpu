//! The OS services: core's threads, clocks, host VM and IOSurface, each OS's own behind one
//! facade (`OsClock`, `OsThreads`, `OsVm`, `OsSurface`, `OsSurfaces`), and what this platform
//! reads of a surface beyond core's trait, written against it (`surface.rs`).
//!
//! - **macOS** (`os/macos.rs`): the metal crate's services, as they are, and IOSurface itself.
//! - **Linux** (`os/linux.rs`): the clocks (`CLOCK_BOOTTIME`, `CLOCK_REALTIME`), threads, a host
//!   VM over `mmap` that makes views of the guest's pages by mapping their files (the backing
//!   table), and IOSurface's stand-in over that memory.
//!
//! Either way the platform's VM reports the pages it changes to the runtime
//! ([`crate::remap::VulkanVm`]), and a display's window presents through a swapchain over the
//! window system's Vulkan extensions (`wsi.rs`), each OS naming its own surface extensions and
//! its loader, and fetching an image's drawable ahead where the driver needs it
//! (`Drawables`: MoltenVK's, on macOS).

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
mod surface;
mod wsi;

#[cfg(target_os = "linux")]
use linux as host_os;
#[cfg(target_os = "macos")]
use macos as host_os;

use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Learns whether a device imports views of guest pages mapped through dma-bufs, on Linux
/// (`os/linux/import.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::learn_view_imports;
/// Whether a view of several runs of pages made now is mapped through dma-bufs, on Linux
/// (`os/linux/import.rs`).
#[cfg(all(test, target_os = "linux"))]
pub(crate) use host_os::views_through_dma_bufs;
pub(crate) use host_os::{
    COPIES_READ_ONLY, Drawables, ImportView, TWINS_READ_ONLY, allocate_aliasable, allocate_zeros,
    dma_buf_refusal, host_memory, host_memory_vouched, import_view, plane_shape, shown_at,
    surface_memory_hold,
};
pub use host_os::{
    GuestMemoryLease, OsClock, OsSurface, OsSurfaces, OsThreads, OsVm, RegionMemory,
    default_driver_sizes, default_gpu_core_count, default_registry_id, install_process_services,
    lease_guest_memory,
};
/// Dma-bufs of host memory lent to a device outside the runtime, on Linux
/// (`os/linux/device_lend.rs`).
#[cfg(target_os = "linux")]
pub use host_os::{LendRefusal, LentDmaBuf, dma_buf_of_host};
#[cfg(test)]
pub(crate) use host_os::{PAGE, mapped, writable};
/// The guest's RAM a device copies by runs of it, on Linux (`os/linux/host_ram.rs`).
#[cfg(target_os = "linux")]
pub(crate) use host_os::{copies_guest_ram, copy_from_host_ram, copy_to_host_ram, ram_over};
#[cfg(all(test, target_os = "macos"))]
pub(crate) use macos::MATRIX_KEY;
/// Runs a test's calls with their imports refused, so the runtime copies the pages, on macOS
/// (`os/macos.rs`).
#[cfg(all(test, target_os = "macos"))]
pub(crate) use macos::refusing_imports;
pub use wsi::{add_window_extensions, present_fences};

#[cfg(test)]
pub(crate) use surface::in_runs;
pub(crate) use surface::{
    MadeOver, READ_ONLY_DIRECTION, base_in_pages, dropped_write, gpu_read_only, memory_kind,
    page_runs, plane, plane_facts, surface_alias, surface_bytes_memory, surface_memory,
    surface_offset, write_surface_bytes, ycbcr_facts,
};

/// One range of the guest's RAM as the host maps it, and the file behind it: `length` bytes at
/// `host`, a shared mapping (`MAP_SHARED`) of `fd` from byte `offset`. A device's lease of the
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
    pub fd: Arc<OwnedFd>,
    /// The file offset of the range's first byte.
    pub offset: u64,
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::platform::os::vm::HostVm as _;

    // The surface tests' helpers, which the crate's tests name here.
    pub(crate) use super::surface::tests::*;
    use super::*;

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
        use reims_vgpu_core::num::ToUsize as _;
        use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn, VmProt};

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
