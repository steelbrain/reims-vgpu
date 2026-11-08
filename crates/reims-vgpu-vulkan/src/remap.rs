//! What the platform tells the runtime of host pages that change under an address.
//!
//! The runtime lets a no-copy buffer made over exactly the bytes a live one was made over lie in
//! that one's import, which holds the pages that were there when that one was made. Metal wires
//! the pages at the address when it makes each buffer. So a host address that keeps its value
//! while its pages are replaced (an unmap and a map at the same address) would give a buffer
//! made after the change the old pages, unless the runtime is told:
//! `rt::Device::host_range_remapped`, which [`crate::gpu::device::host_range_remapped`] calls on
//! every device of the process.
//!
//! A host address takes new pages through core's Mach VM calls, and each reports what it changed,
//! with the call's own arguments and answer passed through unchanged. A call that maps reports
//! after it, and one that unmaps before it, at the unmap: a mapping that reports nothing (an
//! IOSurface's, `malloc`'s) could take the freed address between the unmap and a later report,
//! while a live buffer still lies in an import of the old pages there:
//! - **Core's Mach VM calls** ([`VulkanVm`], the platform's `HostVm`): a memory map's mappings
//!   (`remap_shared_overwrite`), fixed overwrites (`allocate_fixed_overwrite`), every
//!   mapping's teardown, core's detach temporaries among them (`deallocate`, before the call:
//!   one Mach refuses reports pages that did not change, which only has a later buffer over
//!   them import them anew), and fresh ranges, which may lie where others were freed
//!   (`allocate_anywhere`). Each reports the pages the call changes, as Mach takes them
//!   ([`length_pages`], [`pages`]): a fixed overwrite's and a remap's length rounded up to whole
//!   pages from the page of the address, a deallocate's every page its bytes touch; a call of no
//!   bytes changes none. `protect_max` changes no pages.
//!
//! The platform's own Mach VM calls in [`crate::gpu::backing`] report nothing: an alias's
//! mapping (`gather`), the block of zeros, and an alias's release; nor do those of Linux's
//! surface stand-in (`os::linux::surface`), whose view of the guest's pages is the alias of a
//! surface over several ranges of them. Each range is a fresh one of the platform's own, which
//! backs a heap, a rebind, or the no-copy buffers over a surface at an offset into its allocation
//! or whose GPU writes never reach its memory (`buffer_over_surface_at` in
//! [`crate::gpu::device`]). The runtime lets no buffer lie in the import of a heap's bytes or of
//! the memory a rebound buffer takes, and a surface's alias is released by the keep-alive of the
//! last buffer over it, which the runtime drops after it freed that buffer's import, or, where it
//! is the surface's view, by the surface's last release where that comes later, so no buffer made
//! later at the address could lie in one.
//!
//! **Not reported**: the pages of the host's own RAM a memory map names, which core reads at the
//! host's addresses with no mapping of its own. The host changes those without the library: no call
//! of the C API tells it, and Metal needs none. The only no-copy buffers at those addresses are
//! those under a surface whose base address lies in the guest's pages it was made over
//! ([`crate::os::base_in_pages`]: one over a single page; one over several pages, a range each, has
//! a mapping of its own there, measured), at no offset into its allocation and with GPU writes that
//! reach its memory (the others lie over an alias): one made after the host replaced the pages lies
//! in a live one's import, over the old pages, and what Metal's texture on such a surface reads
//! then is not measured. Detached buffers and heaps lie in aliases of their own, fresh at each
//! attach ([`crate::gpu::backing`]), never there.
//!
//! **Unmaps an import waits for** ([`UnmapHold`]). The runtime runs a no-copy buffer's
//! deallocator, given with a keep-alive, as the buffer is released, Metal's moment, and then
//! frees the device's import of its pages (`vkFreeMemory`), dropping the keep-alive after that.
//! A command buffer that uses the buffer holds that release: Metal runs the deallocator of a
//! no-copy buffer released while a blit of a command buffer reads it, behind a running kernel,
//! only once the command buffer's completed handler has run, and before `waitUntilCompleted`
//! returns but in 3 of 600 under load, those within 1 µs after (measured on an Apple M3 Max,
//! macOS 27.0.1, 2026-10-26, with Metal itself: 300 trials without other load, 600 beside 16 CPU
//! spinners and another process's kernels; never while the kernel ran).
//! Vulkan requires imported pages to stay mapped while the import lives
//! (`VK_EXT_external_memory_host`). A deallocator of core's unmaps core's mapping of the guest's
//! pages through [`VulkanVm`] (a memory map's `Mapping`, core's detach temporaries), so each
//! no-copy buffer this platform makes over memory it was given gives the runtime an
//! [`UnmapHold`] on the pages in its keep-alive ([`crate::gpu::backing::keep_alive`]). While the
//! hold lives, [`VulkanVm`]'s `deallocate` of pages it meets reports them, as every unmap does,
//! answers success and leaves them mapped; the hold's drop, after the free, unmaps them, whole,
//! once no other hold meets them. So the pages go after the import, as they did when the runtime
//! ran the deallocator after the free, and the deallocator, which is the guest's and the host's
//! (core's mapping's release, and with it what the host sees), stays at Metal's moment.
//!
//! - **Nothing waits.** `deallocate` never waits for a hold: it records the unmap under a leaf
//!   lock and returns. The hold's drop takes the same lock, takes the unmaps no hold meets any
//!   more, lets the lock go, and unmaps them (`munmap` or `mach_vm_deallocate`), calling nothing
//!   else. So neither side waits for another thread, and no thread can wait on a hold whose drop
//!   needs it, or a thread it blocks: not the graphics device's threads, which release buffers and
//!   so may run core's deallocators, nor a vCPU thread of QEMU's that holds its BQL inside a call
//!   of the C API. The calls of `deallocate` are core's: a mapping's last release on the thread
//!   that lets it go (a device thread, the host's thread inside a call, or the runtime's thread
//!   that runs a buffer's deallocator, the releasing thread or one of its handler threads as a
//!   command buffer completes), a detach temporary's and a failed map's. The holds go on the runtime's thread that frees
//!   the import: the releasing thread, or a handler thread once the last command buffer that held
//!   the import completed, or the creating thread where the runtime refused the call.
//! - **What it keeps.** The pages stay mapped, at their address, until the hold goes: the kernel
//!   gives no range over them meanwhile. Core deallocates only whole ranges it made and never
//!   reaches again, so nothing maps over a waiting unmap but a fixed overwrite, which takes those
//!   pages out of it (`Holds::replaced`). A released hold's unmaps run from the decision to the
//!   unmap in one order with the fixed overwrites ([`OVERWRITES`]), so an overwrite never lands
//!   between them, to be unmapped with the pages that waited there.
//! - **What is not ours to unmap**: memory a no-copy buffer lies in whose unmap is the system's,
//!   which no hold defers: on macOS an IOSurface's memory that a buffer over it at no offset
//!   imports as it is. There the deallocator (the surface's release) may unmap the pages before the free, in the same release, or, where
//!   a command buffer submitted before a texture over them was released still holds the import,
//!   until it completes. Ground: MoltenVK's no-copy buffer keeps the pages it was made over,
//!   which Metal wires as it makes the buffer (`tests/no_copy_remap.rs` measures a remap under
//!   one), and so do NVIDIA's host-pointer import and an import of dma-bufs of a file's pages,
//!   which the dma-buf pinned (RADV's, for any page with a file behind it):
//!   `tests/no_copy_remap_linux.rs` measures a remap under each, on an RTX 5080 Laptop GPU and
//!   on RADV's Radeon 780M and RX 7900 XTX. On those devices the GPU still reads the pages it
//!   imported, as Metal's reads those it wired. Other drivers' host-pointer imports follow their
//!   address (the runtime's `Device::pointer_imports_follow_their_address`, which it checks once
//!   per driver, release and kernel: ANV, RADV and lavapipe answer yes), and one that validates
//!   the pages again at each submission fails a submission that lists the import in that window.
//!   RADV's pointer imports take only pages with no file behind them (amdgpu's userptr), and it
//!   lists every import in every submission: once a file's pages are mapped under one, every
//!   submission of the device fails, the runtime answering
//!   `DeviceLost`, and other pages with no file behind them are what the GPU reads from then
//!   (measured on both, Mesa 26.2.4). On Linux the surface stand-in's own memory and views
//!   (`os::linux::surface`) are held until the free, by the keep-alive of a buffer over a surface
//!   at no offset (`surface_memory_hold`). A surface over a single range of the guest's pages
//!   maps nothing: core makes one only over a page of the host's own RAM ("Not reported",
//!   above), which no hold defers, and the [`UnmapHold`] its hold carries defers an unmap only of
//!   pages [`VulkanVm`] maps.
//!   Measured on an Iris Xe (Mesa 26.2.4's ANV, i915) before that hold: ANV lists every live
//!   import in each execbuffer, and one listing an import whose pages a surface's release had
//!   unmapped failed with `EFAULT`, which ANV answers as the device lost.
//!
//! What a report costs, measured on the M3 Max (release build, 100,000 calls, three runs): with
//! no no-copy buffer live, nothing a 270 ns `mach_vm_deallocate` shows above its noise; with
//! 1,000 live no-copy buffers the runtime imported, 40 to 62 ns per `deallocate`.

#[cfg(test)]
use std::collections::BTreeMap;
use std::ops::Range;
use std::ptr::NonNull;

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::sync::Mutex;

use crate::os::OsVm;

mod holds;

use holds::Holds;

/// The pages `len` bytes at `addr` touch, as Mach's `deallocate` takes them: from the page of
/// `addr` to the page past its last byte; none for no bytes.
fn pages(addr: HostAddr, len: u64, page: u64) -> (HostPtr, u64) {
    let mask = page.wrapping_sub(1);
    let start = addr.addr().to_u64();
    let first = start & !mask;
    let end = match len {
        0 => first,
        _ => start.saturating_add(len).saturating_add(mask) & !mask,
    };
    (
        HostPtr::from_addr(first.to_usize()),
        end.saturating_sub(first),
    )
}

/// The pages Mach's fixed overwrite and remap of `len` bytes at `addr` change: `len` rounded up
/// to whole pages from the page of `addr`, though the bytes run into the page after (measured on
/// the M3 Max, macOS 27.0.1: `crate::os`'s `the_vm_spans_its_calls_as_mach_does`); none for no
/// bytes.
fn length_pages(addr: HostAddr, len: u64, page: u64) -> (HostPtr, u64) {
    let mask = page.wrapping_sub(1);
    let first = addr.addr().to_u64() & !mask;
    let length = len.saturating_add(mask) & !mask;
    (HostPtr::from_addr(first.to_usize()), length)
}

/// Core's mappings of the guest's pages, for tests ([`guest_mappings_of`]): every remap through
/// [`VulkanVm`] is one of core's memory map's guest pages, which it maps afresh for a no-copy
/// buffer over them. Each run of a mapping's host bytes, keyed by its first byte, with
/// its end and the first byte of the pages it shows, as the memory map names them, from the remap
/// to the unmap. The runs share no byte. A leaf lock: nothing is called under it.
#[cfg(test)]
#[derive(Debug, Default)]
struct GuestMappings(BTreeMap<u64, (u64, u64)>);

/// The process's [`GuestMappings`], which [`VulkanVm`] keeps.
#[cfg(test)]
static GUEST_MAPPINGS: Mutex<GuestMappings> = Mutex::new(GuestMappings::new());

#[cfg(test)]
impl GuestMappings {
    /// None.
    const fn new() -> GuestMappings {
        GuestMappings(BTreeMap::new())
    }

    /// The run that holds the byte `at`: its first byte, end and source.
    fn holding(&self, at: u64) -> Option<(u64, u64, u64)> {
        (self.0.range(..=at).next_back())
            .map(|(&first, &(end, source))| (first, end, source))
            .filter(|&(_, end, _)| at < end)
    }

    /// `first..end` shows the pages from `source` from now on, which a mapping of a mapping
    /// names as that one does where it lies in one run of it.
    fn mapped(&mut self, first: u64, end: u64, source: u64) {
        let length = end - first;
        let source = match self.holding(source) {
            Some((at, run_end, from)) if source.saturating_add(length) <= run_end => {
                from + (source - at)
            }
            _ => source,
        };
        self.unmapped(first, end);
        if first < end {
            self.0.insert(first, (end, source));
        }
    }

    /// `first..end` shows none of the guest's pages from now on: what of a run lies in it goes.
    fn unmapped(&mut self, first: u64, end: u64) {
        let meeting: Vec<(u64, u64, u64)> = (self.0.range(..end).rev())
            .map(|(&at, &(run_end, source))| (at, run_end, source))
            .take_while(|&(_, run_end, _)| run_end > first)
            .collect();
        for (at, run_end, source) in meeting {
            self.0.remove(&at);
            if at < first {
                self.0.insert(at, (first, source));
            }
            if end < run_end {
                self.0.insert(end, (run_end, source + (end - at)));
            }
        }
    }
}

/// The process's holds on pages runtime imports cover, and the unmaps that wait for them
/// ([`UnmapHold`]). A leaf lock: nothing is called under it.
static HOLDS: Mutex<Holds> = Mutex::new(Holds::new());

/// The order of [`VulkanVm`]'s fixed overwrites, each with the waiting unmaps it takes its pages
/// out of, and a released hold's unmaps, each from the decision to the unmap (module docs, "What
/// it keeps"): an overwrite between those would be unmapped with the pages that waited. Taken
/// before [`HOLDS`], around the OS's calls alone; a hold's release with no unmap waiting skips it.
static OVERWRITES: Mutex<()> = Mutex::new(());

/// `call`, a fixed overwrite of the `length` pages at `at`, then what waited to be unmapped there
/// taken out of the waiting unmaps, in [`OVERWRITES`]' order.
fn overwrite<T>(
    (at, length): (HostPtr, u64),
    call: impl FnOnce() -> Result<T, KernReturn>,
) -> Result<T, KernReturn> {
    let _order = OVERWRITES.lock();
    let made = call()?;
    let first = at.addr().to_u64();
    HOLDS
        .lock()
        .replaced(&(first..first.saturating_add(length)));
    Ok(made)
}

/// A hold on the host pages a runtime import covers, which the runtime is given in a no-copy
/// buffer's keep-alive: while it lives, [`VulkanVm`]'s `deallocate` of pages it meets leaves them
/// mapped, and its drop, which the runtime makes once it freed the import, unmaps them (module
/// docs, "Unmaps an import waits for").
#[derive(Debug)]
pub(crate) struct UnmapHold {
    /// The first byte of its pages.
    first: u64,
    /// Its number among the holds.
    id: u64,
}

impl UnmapHold {
    /// A hold on the pages the `length` bytes at `bytes` lie in (one page for a length of 0).
    pub(crate) fn new(bytes: NonNull<u8>, length: u64) -> UnmapHold {
        let at = HostPtr::from_ptr(bytes.as_ptr());
        let (at, length) = pages(at, length.max(1), OsVm::new().page_size());
        let first = at.addr().to_u64();
        let id = HOLDS.lock().hold(first..first.saturating_add(length));
        UnmapHold { first, id }
    }
}

impl Drop for UnmapHold {
    /// Unmaps what waited for this hold and no other, in [`OVERWRITES`]' order where an unmap
    /// waits, calling nothing else.
    fn drop(&mut self) {
        {
            let mut holds = HOLDS.lock();
            if !holds.unmaps_wait() {
                holds.release(self.first, self.id);
                return;
            }
        }
        let _order = OVERWRITES.lock();
        let free = HOLDS.lock().release(self.first, self.id);
        let vm = OsVm::new();
        for pages in free {
            let at = HostPtr::from_addr(pages.start.to_usize());
            let _ = vm.deallocate(at, pages.end - pages.start);
        }
    }
}

/// Whether an unmap of a page of the `length` bytes at `at` waits for a hold, for tests: read in
/// [`OVERWRITES`]' order, so a hold whose release has begun has unmapped what waited for it
/// before this answers `false` (its drop takes the unmaps out of [`HOLDS`], then makes them).
#[cfg(test)]
pub(crate) fn unmap_waits(at: HostPtr, length: u64) -> bool {
    let _order = OVERWRITES.lock();
    let (at, length) = pages(at, length, OsVm::new().page_size());
    let first = at.addr().to_u64();
    HOLDS.lock().waits(&(first..first.saturating_add(length)))
}

/// The host bytes of each run of core's mappings of the guest's pages that shows a page of
/// `source` (named by the memory map's host addresses), for tests: a run leaves at the unmap of
/// its pages through [`VulkanVm`], whether or not that unmap waits for a hold.
#[cfg(test)]
pub(crate) fn guest_mappings_of(source: Range<u64>) -> Vec<Range<u64>> {
    (GUEST_MAPPINGS.lock().0.iter())
        .filter(|&(&first, &(end, from))| from < source.end && source.start < from + (end - first))
        .map(|(&first, &(end, _))| first..end)
        .collect()
}

/// The platform's `HostVm`: the OS's ([`OsVm`]: Mach VM on macOS, its calls over `mmap` on
/// Linux), each call that changes pages then reported (module docs), and an unmap of pages a
/// runtime import holds made once its hold goes ([`UnmapHold`]).
#[derive(Debug)]
pub struct VulkanVm {
    /// The calls themselves.
    vm: OsVm,
    /// Where the changed pages go: every device's runtime, but in this module's tests.
    report: fn(HostPtr, u64),
}

impl VulkanVm {
    /// The process's `HostVm`, reporting to every device made in the process.
    pub fn new() -> VulkanVm {
        VulkanVm {
            vm: OsVm::new(),
            report: crate::gpu::device::host_range_remapped,
        }
    }

    /// Reports `length` pages at `at`, which a call changed: their range. None reports nothing.
    fn changed(&self, (at, length): (HostPtr, u64)) -> Range<u64> {
        let first = at.addr().to_u64();
        let changed = first..first.saturating_add(length);
        if changed.is_empty() {
            return changed;
        }
        #[cfg(test)]
        GUEST_MAPPINGS.lock().unmapped(changed.start, changed.end);
        (self.report)(at, length);
        changed
    }
}

impl Default for VulkanVm {
    fn default() -> VulkanVm {
        VulkanVm::new()
    }
}

impl HostVm for VulkanVm {
    fn page_size(&self) -> u64 {
        self.vm.page_size()
    }

    fn page_shift(&self) -> u32 {
        self.vm.page_shift()
    }

    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let at = self.vm.allocate_anywhere(len)?;
        self.changed(pages(at, len, self.page_size()));
        Ok(at)
    }

    /// A range a view can be made of ([`crate::os::allocate_aliasable`]): Mach's allocation on
    /// macOS, a file's on Linux, whose anonymous memory no view can be made of.
    fn allocate_aliasable(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let at = crate::os::allocate_aliasable(len)?;
        self.changed(pages(at, len, self.page_size()));
        Ok(at)
    }

    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let pages = length_pages(addr, len, self.page_size());
        overwrite(pages, || self.vm.allocate_fixed_overwrite(addr, len))?;
        self.changed(pages);
        Ok(())
    }

    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let pages = length_pages(dst, len, self.page_size());
        let at = overwrite(pages, || self.vm.remap_shared_overwrite(dst, src, len))?;
        self.changed(pages);
        #[cfg(test)]
        {
            // The kernel maps as many of the source's pages, from the page of its first byte.
            let (source, _) = length_pages(src, len, self.page_size());
            let first = pages.0.addr().to_u64();
            (GUEST_MAPPINGS.lock()).mapped(first, first + pages.1, source.addr().to_u64());
        }
        Ok(at)
    }

    /// The OS's view of the runs at once, reported as their remaps one at a time are: the pages
    /// of the runs it mapped, the runs before a refusal's, taken out of the waiting unmaps in
    /// [`OVERWRITES`]' order and then reported changed.
    fn remap_runs_shared_overwrite(
        &self,
        dst: HostAddr,
        runs: &[(HostAddr, u64)],
    ) -> Result<(), (usize, KernReturn)> {
        let page = self.page_size();
        let order = OVERWRITES.lock();
        let made = self.vm.remap_runs_shared_overwrite(dst, runs);
        let done = match made {
            Ok(()) => runs.len(),
            Err((failed, _)) => failed,
        };
        let mapped = (runs[..done].iter()).fold(0u64, |sum, &(_, len)| {
            sum.saturating_add(len.saturating_add(page - 1) & !(page - 1))
        });
        let pages = length_pages(dst, mapped, page);
        let first = pages.0.addr().to_u64();
        HOLDS
            .lock()
            .replaced(&(first..first.saturating_add(pages.1)));
        drop(order);
        self.changed(pages);
        #[cfg(test)]
        {
            let mut at = first;
            for &(src, len) in &runs[..done] {
                let (source, length) = length_pages(src, len, page);
                (GUEST_MAPPINGS.lock()).mapped(at, at + length, source.addr().to_u64());
                at += length;
            }
        }
        made
    }

    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        self.vm.protect_max(addr, len, prot)
    }

    fn idle_view_mappings(&self) -> Option<u64> {
        self.vm.idle_view_mappings()
    }

    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        // At the unmap (module docs).
        let pages = self.changed(pages(addr, len, self.page_size()));
        // Pages a runtime import holds are unmapped once its hold goes (module docs).
        if HOLDS.lock().unmap(pages) {
            return KernReturn::SUCCESS;
        }
        self.vm.deallocate(addr, len)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    thread_local! {
        /// What [`seen`] was told on this thread.
        static SEEN: RefCell<Vec<(HostPtr, u64, bool)>> = const { RefCell::new(Vec::new()) };
    }

    /// A report kept on this thread, with whether its first page was mapped as it came.
    fn seen(at: HostPtr, length: u64) {
        SEEN.with(|s| s.borrow_mut().push((at, length, crate::os::mapped(at))));
    }

    /// What this thread was told since the last take.
    fn take() -> Vec<(HostPtr, u64, bool)> {
        SEEN.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    /// Each Mach VM call that changes pages reports them as Mach takes them: a fixed overwrite's
    /// and a remap's length in whole pages from the page of the address, though the bytes run
    /// into the next, the remap's mapping of as many of the source's pages from the page of its
    /// first byte; a fresh range's and a deallocate's every page the bytes touch. One that maps
    /// reports after it, `deallocate` at the unmap, its pages still mapped; `protect_max`, a
    /// refused remap and a call of no bytes off a page's start report nothing
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn the_vm_reports_the_pages_its_calls_change() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let at = vm.allocate_anywhere(page + 0x100).unwrap();
        assert_eq!(take(), [(at, 2 * page, true)]);
        vm.allocate_fixed_overwrite(at.wrapping_add(page + 0x80), 0x100)
            .unwrap();
        assert_eq!(take(), [(at.wrapping_add(page), page, true)]);
        vm.allocate_fixed_overwrite(at.wrapping_add(page - 0x80), 0x100)
            .unwrap();
        assert_eq!(take(), [(at, page, true)]);
        let src = crate::os::allocate_aliasable(2 * page).unwrap();
        vm.remap_shared_overwrite(at, src.wrapping_add(0x200), page)
            .unwrap();
        assert_eq!(take(), [(at, page, true)]);
        let (first, mapped) = (src.addr().to_u64(), at.addr().to_u64());
        assert_eq!(
            guest_mappings_of(first..first + 2 * page),
            [mapped..mapped + page]
        );
        vm.protect_max(at, page, VmProt::READ | VmProt::WRITE)
            .unwrap();
        assert_eq!(take(), []);
        vm.allocate_fixed_overwrite(at.wrapping_add(0x80), 0)
            .unwrap();
        assert!(vm.deallocate(at.wrapping_add(0x80), 0).is_success());
        assert_eq!(take(), []);
        assert!(crate::os::mapped(at));
        assert!(vm.deallocate(at, 2 * page).is_success());
        assert_eq!(take(), [(at, 2 * page, true)]);
        assert!(!crate::os::mapped(at));
        // The source is no longer mapped: the remap fails, and changes nothing.
        assert!(vm.remap_shared_overwrite(src, at, page).is_err());
        assert_eq!(take(), []);
        assert!(OsVm::new().deallocate(src, 2 * page).is_success());
    }

    /// Core's aliasable range (the detach heap's temporary) reports its pages as a fresh range
    /// does, and a view can be made of it, as Mach's remap of any allocation: on Linux a range
    /// of `allocate_anywhere`'s has no file behind it, and a view of it is refused
    #[test]
    fn an_aliasable_range_reports_its_pages_and_takes_a_view() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let at = vm.allocate_aliasable(page + 0x100).unwrap();
        assert_eq!(take(), [(at, 2 * page, true)]);
        let view = vm.allocate_anywhere(2 * page).unwrap();
        let _ = take();
        vm.remap_shared_overwrite(view, at, 2 * page).unwrap();
        // SAFETY: both ranges are mapped, read and write, and show the same pages.
        unsafe {
            at.wrapping_add(page + 5)
                .as_ptr::<u8>()
                .write_volatile(0x3c);
            assert_eq!(
                view.wrapping_add(page + 5).as_ptr::<u8>().read_volatile(),
                0x3c
            );
        }
        let anonymous = vm.allocate_anywhere(page).unwrap();
        let refused = vm.remap_shared_overwrite(view, anonymous, page);
        assert_eq!(refused.is_err(), cfg!(target_os = "linux"));
        for (range, length) in [(view, 2 * page), (at, 2 * page), (anonymous, page)] {
            assert!(vm.deallocate(range, length).is_success());
        }
    }

    /// A `deallocate` of pages a hold meets reports them at the call and answers success,
    /// leaving them mapped until the hold goes, which unmaps them; one of pages no hold meets
    /// unmaps at once. Unmapped pages may be mapped again at once by another test's thread, so
    /// an unmap shows as pages unmapped or no longer holding the bytes written there
    #[test]
    fn an_unmap_of_held_pages_waits_for_the_hold() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let at = vm.allocate_anywhere(2 * page).unwrap();
        let other = vm.allocate_anywhere(page).unwrap();
        for (p, length) in [(at, 2 * page), (other, page)] {
            // SAFETY: pages this test just mapped, readable and writable.
            unsafe { std::ptr::write_bytes(p.as_ptr::<u8>(), 0xa5, length.to_usize()) };
        }
        let _ = take();
        let gone = |p: HostPtr| {
            // SAFETY: a read of a page that is mapped, the first test's.
            !crate::os::mapped(p) || unsafe { p.as_ptr::<u8>().read_volatile() } != 0xa5
        };
        let first = NonNull::new(at.wrapping_add(0x10).as_ptr::<u8>()).unwrap();
        let hold = UnmapHold::new(first, page);
        assert!(vm.deallocate(other, page).is_success());
        assert!(!unmap_waits(other, page), "no hold meets it");
        assert!(gone(other), "unmapped at once");
        assert!(vm.deallocate(at, 2 * page).is_success());
        assert_eq!(
            take(),
            [(other, page, true), (at, 2 * page, true)],
            "each reported at the call"
        );
        assert!(unmap_waits(at, 2 * page));
        assert!(
            !gone(at) && !gone(at.wrapping_add(page)),
            "still mapped, whole"
        );
        drop(hold);
        assert!(!unmap_waits(at, 2 * page));
        assert!(
            gone(at) && gone(at.wrapping_add(page)),
            "unmapped as the hold went"
        );
    }
}
