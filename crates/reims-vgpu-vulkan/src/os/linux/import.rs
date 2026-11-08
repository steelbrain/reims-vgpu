//! What the runtime imports for host memory on Linux: the memory itself, and for a surface's
//! read-only alias, or a range core maps read-only and makes a buffer over, a writable twin of it.
//!
//! Linux drivers pin imported pages for writing, so the runtime imports no read-only memory
//! (typed, `HostMemory`), where Metal reads it: the guest's ranges core maps read-only (the EFI
//! framebuffer among them), aliases of a read-only replace or heap, and surfaces the GPU only
//! reads. For aliases of a read-only replace or heap, and a no-copy buffer a descriptor names or a
//! heap over bytes, over pages mapped read-only, the runtime is given the pages themselves: the
//! views' gate refuses
//! them ([`import_gate`]), and the runtime makes the buffer or heap over a copy of them whose GPU
//! writes it drops, as Metal's GPU drops them (its `copied-host-pages`, a kind of
//! `crate::answer::DIES`), so the GPU reads what Metal's reads and its writes reach nothing past
//! the copy ([`ImportView::read_only`]); so does an object made with `MTLResourceOptions` bit 24,
//! whatever its pages. What Metal answers otherwise than a dropped write there (a pass's store
//! into such memory, a texture's writes) is `crate::gpu::writes`'s.
//!
//! A surface's read-only alias, and a buffer core makes over a range it maps read-only, are given
//! a **twin** instead (`crate::gpu::backing::ImportSite::twins`): a second view of the same file
//! pages, read and write, at an address of its own ([`import_view`]), imported in place, so a
//! surface's buffers lie in one import of the pages, as a writer's GPU writes through another
//! import reach its readers, where a copy would read them as they were at its commit, and core's
//! ranges cost no compare of their bytes at each commit. The GPU then reads what Metal's reads,
//! but its writes reach those pages, where Metal drops them: every command that may write
//! through an object over such an import dies where it is encoded, naming it
//! (`crate::gpu::writes::kept_write`).
//!
//! The block of zeros a read-only alias maps past its ranges' pages is a file of zeros all such
//! aliases share ([`allocate_zeros`]). Its twin is private memory of zeros instead, so the GPU's
//! writes there stay in that twin and never reach the block, though they are kept for the
//! alias's life, where Metal drops them, which the same deaths stop. A detached object lies in
//! no memory (the runtime's detach, `crate::gpu::backing`), so nothing of it is imported.
//!
//! A buffer made over a twin answers the address it was made over as its contents
//! ([`shown_at`]): the runtime's contents are the twin's. The twin goes once the runtime frees the
//! import (its keep-alive, [`ImportView::release`]).
//!
//! Every import also carries a dma-buf source ([`host_memory`]): a device that refuses the pages
//! as host memory, as RADV refuses any page with a file behind it, imports a dma-buf of the same
//! pages instead (`super::dma_buf`). The runtime asks for it only where it would refuse the
//! import otherwise, so every device that imports host memory imports as it did.
//!
//! And every import carries the views' gate ([`import_gate`]), which the runtime asks before it
//! imports a no-copy buffer's or heap's pages, or a buffer's new pages, by their pointer: where
//! the pages were left in views through dma-bufs for want of mappings, which a device that pins a
//! pointer's pages refuses, or are read-only with no twin, it refuses, and the runtime makes the
//! buffer or heap over a copy of them in memory of the device's own instead, which takes no
//! mapping (its `copied-host-pages`, a kind of `crate::answer::DIES`), and whose GPU writes it
//! drops where the process cannot write the pages.
//!
//! **What the imported pages cost the GPU** (**measured** 2026-10-26, raw Vulkan with no runtime
//! between, kernels of our own reading 64 MiB and writing 64 MiB, the median GPU time of five in
//! ms, device-local memory / the imported pages): the RTX 5080 Laptop GPU on NVIDIA's 615.71.09,
//! its link at PCIe 5.0 x8 under load, importing by pointer; the RX 7900 XTX, behind a PCIe 4.0 x4
//! port, and the Radeon 780M, both on RADV of Mesa 26.2.4, importing dma-bufs. Each GPU's IOMMU
//! group passes its addresses through (`identity`).
//!
//! | pattern | RTX 5080 Laptop | RX 7900 XTX | Radeon 780M |
//! |---|---|---|---|
//! | 16-byte loads, stores, in order | 0.08 / 3.4 to 4.2 | 0.41 / 11.3 to 11.6 | 1.76 / 1.96 to 2.00 |
//! | 4-byte stores in order | 0.05 / 2.54 | 0.07 / 8.6 to 9.0 | 0.83 / 0.90 to 1.19 |
//! | 4-byte stores 4,099 words apart | 3.03 / 180 to 277 | 0.77 / 91 to 101 | 18.2 / 19 to 27 |
//! | 4-byte loads 4,099 words apart | 0.75 / 47 to 61 | 0.73 / 95 to 104 | 8.7 / 9.0 to 19.7 |
//! | 25 dependent 4-byte loads a thread | 1.57 / 70.8 | 0.99 / 452 to 454 | 39.2 / 39.4 to 56.0 |
//! | 64 rows a workgroup, 4 MiB window | 0.10 / 9.3 | 0.07 / 1.0 to 1.7 | 3.5 / 3.1 to 4.2 |
//! | atomic adds to one word | 0.03 / 12 to 29 | 0.06 / 0.05 | 0.08 / 0.06 to 0.07 |
//! | atomic adds 4,099 words apart | 1.37 / 167 to 246 | 0.60 / 121 to 192 | 27.6 / 27 to 34 |
//! | a linear RGBA8 image's texels | 0.01 / 2.7 to 4.7 | 0.01 / 1.4 to 1.5 | 0.20 / 0.20 to 0.23 |
//!
//! - **On the RTX 5080 Laptop GPU nothing of the pages changes the time**, by pattern: 4 KiB or
//!   2 MiB pages (transparent huge pages, by `MADV_HUGEPAGE`), a memfd or anonymous memory, a
//!   memfd's views laid out of order at 4 KiB to 2 MiB granules, either memory type the driver
//!   offers for the import (host-visible and coherent, with or without cached), and the driver's
//!   own host memory took as long. Its L2 does not hold host memory for reads: the window's loads
//!   took about 10 ms from 1 MiB to 64 MiB, and only 16 to 64 KiB windows ran from its L1.
//!   Contended atomics there cost 400 to 1,000 times device-local ones (a compare-and-swap loop
//!   on one word 26 to 28 ms against 0.03; a 2-byte store made of an atomic AND and an atomic OR
//!   16 ms against 0.65 for a plain 2-byte store). The link trains to Gen 1 at idle and to its slot's
//!   Gen 5 x8 under load, and stayed there through every run.
//! - **The same GPU stores into host memory hundreds of times slower where a warp's stores leave
//!   gaps in the bytes they span**: 8 MiB as four consecutive words an invocation, a store each,
//!   took 393 to 620 ms, against 0.32 ms as one word an invocation in order, as four words an
//!   invocation a workgroup apart, or as one 16-byte store an invocation; one word in each 16
//!   bytes took 140 ms and one in each 64 bytes 1.15 ms. Loads of the same shapes took 0.32 ms.
//!   The two Radeon GPUs show no such cost. The guest's transfer kernel's seconds come from it,
//!   through the runtime's merge of a shadowed texture's writes into a surface's pages
//!   (`docs/video/bugs.md`, B26).
//! - **The RX 7900 XTX's L2 holds host memory**: the window's loads took 1.0 to 1.5 ms at 4 MiB and
//!   26 to 27 ms at 16 MiB, and atomic adds to one word ran as fast as in its own memory. Its
//!   scattered loads in a 4 MiB window ran faster over 2 MiB pages (`super::dma_buf`).
//! - **On the Radeon 780M host memory is the GPU's memory**: within 1.4 times device-local in
//!   order, and over 4 KiB pages 1.4 to 2.3 times for scattered loads and stores, which 2 MiB
//!   pages bring back to device-local's (`super::dma_buf`).
//! - The runtime imports the pages as the best type of its `Shared` class the import allows:
//!   host-visible and coherent, cached where it can, never AMD's device-coherent and uncached
//!   types (**read**, metal2vulkan 67c64f99d8, `memory.rs`'s `candidates` and `allocatable`). So
//!   on the RTX 5080 Laptop GPU it takes the cached one of the two types the pointer allows, and
//!   on RADV the cached one of a dma-buf's two, whose other, device-coherent and uncached, took
//!   as long or longer (the scattered stores 120 ms against 91 on the RX 7900 XTX).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use reims_vgpu_core::sync::Mutex;

use super::budget;
use super::files;
use super::vm;

/// Whether the runtime imports a surface's read-only alias through a writable twin of its files:
/// on Linux, so read-only memory the library makes has a file behind every byte (module docs).
pub(crate) const TWINS_READ_ONLY: bool = true;

/// Whether the runtime makes read-only memory other than a surface's, and memory made with
/// `MTLResourceOptions` bit 24, over a copy of it whose GPU writes it drops, where Metal's GPU
/// keeps them from the memory itself: on Linux, where the runtime's device answers no to
/// `drops_gpu_writes_to_read_only_memory` (module docs).
pub(crate) const COPIES_READ_ONLY: bool = true;

/// Every live twin, by its first byte: its length and the first byte of the memory it twins.
static TWINS: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// The files of zeros ([`allocate_zeros`]), held for the process.
static ZERO_FILES: Mutex<Vec<Arc<OwnedFd>>> = Mutex::new(Vec::new());

/// A fresh range of `len` bytes of zeros that a view can be made of, as
/// [`vm::allocate_aliasable`] makes, whose file is kept for the process: an import's twin of it
/// is private zeros, never the file (module docs).
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    let at = vm::allocate_aliasable(len)?;
    if let Some(piece) = files::pieces(at.addr().to_u64(), 1).and_then(|p| p.into_iter().next()) {
        ZERO_FILES.lock().push(piece.backing.fd);
    }
    Ok(at)
}

/// Whether `fd` is a file of zeros.
#[cfg(test)]
fn zeros(fd: &Arc<OwnedFd>) -> bool {
    ZERO_FILES.lock().iter().any(|z| Arc::ptr_eq(z, fd))
}

/// Whether each of `pieces` is of a file of zeros, the files looked at once.
pub(super) fn of_zeros(pieces: &[files::Piece]) -> Vec<bool> {
    let zero_files = ZERO_FILES.lock();
    let zeros = |fd: &Arc<OwnedFd>| zero_files.iter().any(|z| Arc::ptr_eq(z, fd));
    pieces.iter().map(|p| zeros(&p.backing.fd)).collect()
}

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
    /// The record of its pages a device may pin, where it imports them as they are mapped.
    pinned: Option<Pinned>,
    /// The table shows every page of the import mapped writable ([`ImportView::vouched`]).
    vouched: bool,
    /// The table shows a page of the memory mapped read-only ([`ImportView::read_only`]).
    read_only: bool,
}

/// The pages a device of the process may pin now: each live import's, of pages it takes as they
/// are mapped from their files (a pointer's, NVIDIA's and i915's), by first byte and number, to
/// their end, from before the import maps them from their files until the import is freed
/// ([`Pinned`]). A fold of the views maps none of them through dma-bufs
/// (`super::vm::fold_idle`): a device took them by their mappings.
static PINNED: Mutex<BTreeMap<(u64, u64), u64>> = Mutex::new(BTreeMap::new());

/// One import's pages in [`PINNED`], until it drops.
#[derive(Debug)]
pub(crate) struct Pinned((u64, u64));

impl Pinned {
    /// The pages of `first..end`, where a device of the process takes imported pages as they are
    /// mapped from their files ([`imports_need_files`]); `None` where none does.
    fn new(first: u64, end: u64) -> Option<Pinned> {
        imports_need_files().then(|| Pinned::of(first, end))
    }

    /// The pages of `first..end`, whatever the devices.
    pub(super) fn of(first: u64, end: u64) -> Pinned {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let key = (first, NEXT.fetch_add(1, Ordering::Relaxed));
        PINNED.lock().insert(key, end);
        Pinned(key)
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        PINNED.lock().remove(&self.0);
        UNPINNED.fetch_add(1, Ordering::Relaxed);
    }
}

/// How many records of [`PINNED`] have dropped, each freeing pages a fold may map through dma-bufs
/// (`super::vm`'s `fold_short`).
static UNPINNED: AtomicU64 = AtomicU64::new(0);

/// How many records of [`PINNED`] have dropped ([`UNPINNED`]).
pub(super) fn unpinned() -> u64 {
    UNPINNED.load(Ordering::Relaxed)
}

/// The pages devices may pin now ([`PINNED`]), in order, those that meet joined.
pub(super) fn pinned() -> Vec<std::ops::Range<u64>> {
    let mut ranges: Vec<std::ops::Range<u64>> = (PINNED.lock().iter())
        .map(|(&(first, _), &end)| first..end)
        .collect();
    ranges.sort_by_key(|r| r.start);
    let mut joined: Vec<std::ops::Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match joined.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => joined.push(range),
        }
    }
    joined
}

/// Whether a device of the process imports by pointer only pages mapped from their files: learned
/// of each device as it is made ([`learn_view_imports`]), and set where any refuses a view through
/// dma-bufs. From then on views are mapped from their files (`super::vm`'s module docs, "Imports
/// of views"), and an import of a view made through dma-bufs before maps its pages from their
/// files ([`pages_from_files`]).
static IMPORTS_NEED_FILES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a device of the process imports by pointer only pages mapped from their files
/// ([`learn_view_imports`]).
pub(crate) fn imports_need_files() -> bool {
    IMPORTS_NEED_FILES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a view of several runs of pages made now is mapped through dma-bufs: udmabuf is the
/// process's, and no device of it needs imported pages mapped from their files (`super::vm`'s
/// module docs, "Views through dma-bufs").
#[cfg(test)]
pub(crate) fn views_through_dma_bufs() -> bool {
    super::dma_buf::opens() && !imports_need_files()
}

/// Imports refused at the views' budget, of one kind: how many and their bytes, each written to
/// the trace (its first byte, its length and the mappings it would have taken) and the count
/// logged at the first and at each power of two.
struct Refused {
    /// The trace record's name.
    record: &'static str,
    /// The refusals so far.
    count: AtomicU64,
    /// Their bytes.
    bytes: AtomicU64,
}

impl Refused {
    const fn new(record: &'static str) -> Refused {
        Refused {
            record,
            count: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Counts the refusal of `first..end`, which would have taken `adding` mappings: the count
    /// and the bytes so far, where the count is one to log.
    fn count(&self, first: u64, end: u64, adding: u64) -> Option<(u64, u64)> {
        reims_vgpu_core::trace::event(self.record, &[first, end - first, adding]);
        let bytes = self.bytes.fetch_add(end - first, Ordering::Relaxed) + (end - first);
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        count.is_power_of_two().then_some((count, bytes))
    }
}

/// The imports the gate refused for their pages left in views through dma-bufs, which the runtime
/// copied ([`import_gate`]).
static COPIED: Refused = Refused::new("nrcopy");

/// The imports of read-only pages no twin was made for, short of the headroom of the views'
/// budget ([`twin`]), which the gate refuses, so the runtime copies them.
static TWINLESS: Refused = Refused::new("notwin");

/// The views' gate over the runtime's import of a no-copy buffer's or heap's pages, or a buffer's
/// new pages, by their pointer (`rt::HostMemory::with_import_gate`), which every [`host_memory`]
/// carries, asked on the thread that makes the buffer, heap or rebind, with no lock of the
/// runtime's held, after [`import_view`] made what it imports. It refuses read-only pages, which
/// reach it where no twin was asked for them (all but a surface's read-only alias), or none was
/// made short of the headroom of the views' budget ([`twin`], counted there), so the runtime
/// copies them, dropping the GPU's writes; and, where a device of the process needs imported
/// pages mapped from their files ([`imports_need_files`]), pages [`import_view`] left in views
/// through dma-bufs, as
/// mapping them from their files would have come within that headroom (`super::budget`'s
/// `claim`), each such refusal counted with its bytes ([`Refused`], trace record `nrcopy`). The
/// runtime then makes the buffer or heap over a copy of the pages, which takes no mapping. A copy
/// costs every commit and completion a compare of its bytes, so the warning names the setting
/// that gives room.
fn import_gate(request: rt::ImportRequest) -> bool {
    let first = addr_of(request.start);
    gate(first, first.saturating_add(request.length))
}

/// [`import_gate`]'s answer for the pages of `first..end`.
fn gate(first: u64, end: u64) -> bool {
    if files::writable(first, end - first) == Some(false) {
        return false;
    }
    if !imports_need_files() {
        return true;
    }
    let pieces = files::through_pieces(first, end);
    if pieces.is_empty() {
        return true;
    }
    let adding = vm::mappings_added(&pieces);
    if let Some((count, bytes)) = COPIED.count(first, end, adding) {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{count} buffers and heaps of {} KiB in all copied rather than imported, which costs \
             each command buffer a compare of their bytes: mapping their pages from their files \
             would have taken the views of memory within {} mappings of their budget \
             (vm.max_map_count), the latest's {adding} mappings for {:#x} bytes at {first:#x}",
            bytes >> 10,
            budget::HEADROOM,
            end - first
        );
    }
    false
}

/// The address of `at`.
fn addr_of(at: NonNull<u8>) -> u64 {
    at.as_ptr().addr().to_u64()
}

/// Maps what of the whole pages of a device lend's `length` bytes at `at` lies in a view through
/// dma-bufs from their files, which a device given their address pins as they are mapped (a
/// mapping of a dma-buf it cannot pin), for as long as the view lives (`super::vm::unfold`), short
/// of the headroom of the views' budget (`super::budget`'s `claim`, `Reach::ShortOfHeadroom`),
/// which stays for what has no other way. Past it the pages stay as they are: the
/// device refuses to register them, declines the frame, and the session copies it into the guest's
/// buffer on the CPU instead (`super::device_lend`), so the guest's buffer holds the same bytes.
/// The record of the pages the device may pin ([`Pinned`]), which the caller holds while the
/// device may.
pub(crate) fn pages_from_files(at: NonNull<u8>, length: u64) -> Pinned {
    let page = vm::page_size();
    let start = addr_of(at);
    let end = start.saturating_add(length.max(1)).next_multiple_of(page);
    let first = start - start % page;
    let pinned = Pinned::of(first, end);
    // Past the headroom the lend's way out is the CPU's copy, so nothing is counted or refused.
    let _ = vm::unfold(first, end);
    pinned
}

/// Learns whether `device` imports a view mapped through dma-bufs as it is mapped, as the
/// platform's imports hand the runtime a view ([`host_memory`]): a no-copy buffer over a view of
/// pages backwards, one more than udmabuf takes in one dma-buf, so two, made as an import is,
/// placed outside the guest's numbering and released at once. Where the device does not take it
/// through the dma-bufs its source makes, views are mapped from their files from then on
/// ([`IMPORTS_NEED_FILES`]): where it refuses it, or makes a copy of the view in its place as
/// its import failed (the runtime's `copied-host-pages`, which the probe's call hears unheard,
/// `crate::answer::unheard`), and where it takes the view's pointer, as no
/// driver pins a `VM_PFNMAP` mapping's pages and one that pins them at the GPU's first use of
/// them, not at the import (as i915's userptr does without `I915_USERPTR_PROBE`, not measured
/// here), would fail there. A device
/// imports such a view by its dma-bufs where it imports none by pointer (RADV), and refuses it
/// where it pins the pointer's pages and imports no foreign dma-buf (NVIDIA's 615 driver, measured
/// 2026-10-27, `super::vm`'s module docs, "Imports of views"). Nothing is learned where the view is
/// not mapped through dma-bufs (udmabuf not the process's): no import then needs it.
pub(crate) fn learn_view_imports(device: &rt::Device) {
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    if IMPORTS_NEED_FILES.load(Ordering::Relaxed) {
        return;
    }
    let vm = super::LinuxVm::new();
    let page = vm::page_size();
    let count = super::dma_buf::limits()
        .pieces
        .saturating_add(1)
        .min(1 << 16);
    let length = count * page;
    let (Ok(src), Ok(view)) = (vm::allocate_aliasable(length), vm.allocate_anywhere(length)) else {
        return;
    };
    let runs: Vec<_> = (0..count)
        .map(|i| (src.wrapping_add((count - 1 - i) * page), page))
        .collect();
    let first = view.addr().to_u64();
    let through = vm.remap_runs_shared_overwrite(view, &runs).is_ok()
        && !files::through_pieces(first, first + length).is_empty();
    if through && let Some(at) = address(first) {
        let asked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // SAFETY: the view is mapped, read and write, and stays so until after the buffer, which
        // has no deallocator or keep-alive, is released: its import is freed in the release.
        let memory = unsafe { host_memory_noting(at, Some(Arc::clone(&asked))) };
        let contents = rt::BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: None,
        };
        let descriptor = rt::BufferDescriptor::new(length, rt::ResourceOptions::shared(), contents)
            .with_placement(rt::AddressPlacement::Outside);
        // A copy of the view in the import's place (`copied-host-pages`), as its import failed,
        // is the device's refusal too: the probe's own, which no one hears of.
        let copies = || device.work_counts().copied_host_buffers_made;
        let before = copies();
        let made = crate::answer::unheard(|| device.new_buffer_with_descriptor(descriptor));
        let refused = made.is_err() || copies() > before;
        drop(made);
        let through_dma_bufs = !refused && asked.load(Ordering::Relaxed);
        let why = dma_buf_refusal();
        log::info!(
            target: "reims_vgpu_vulkan::os",
            "{}: an import of a view of guest pages mapped through dma-bufs {}",
            device.properties().name,
            match (refused, through_dma_bufs) {
                (true, _) => {
                    "is refused, or copied as its import failed, so views are mapped from their \
                     files"
                }
                (false, false) => {
                    "is taken by its pointer, which a driver pinning the pages at the GPU's first \
                     use of them would fail then, so views are mapped from their files"
                }
                (false, true) => "is taken through its dma-bufs",
            }
        );
        if let Some(why) = why {
            log::debug!(target: "reims_vgpu_vulkan::os", "the view's dma-buf source: {why}");
        }
        if !through_dma_bufs {
            IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
            budget::warn_if_imports_short();
            super::host_ram::device_needs_ram(device);
        }
    }
    let _ = vm.deallocate(view, length);
    let _ = vm.deallocate(src, length);
}

// SAFETY: `at` is an address only, which nothing here reads through: it is handed to the runtime;
// the twin is unmapped once, by whichever thread drops the view.
unsafe impl Send for ImportView {}

// SAFETY: as for `Send`: a shared view only answers where it is.
unsafe impl Sync for ImportView {}

/// A writable twin's address range, unmapped when it drops.
#[derive(Debug)]
struct Twin {
    first: u64,
    length: u64,
    /// The kernel's mappings it takes, counted in the budget once it is whole.
    mappings: u64,
}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&self.first);
        // SAFETY: the twin's own mapping, which the runtime let go of (it freed the import and
        // dropped its keep-alive, or the call that would import it was refused).
        unsafe { libc::munmap(vm::ptr(self.first), self.length.to_usize()) };
        budget::twins_hold(self.mappings, false);
    }
}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the import is a writable twin of memory mapped read-only, the guest's pages or
    /// zeros, whose GPU writes reach them where Metal drops them (module docs).
    pub(crate) fn twinned(&self) -> bool {
        self.twin.is_some()
    }

    /// Whether the table shows a page of the memory mapped read-only: the runtime is given its
    /// twin, or the pages themselves, which the gate refuses ([`import_gate`]), so the runtime
    /// makes the object over a copy of them whose GPU writes it drops (module docs).
    pub(crate) fn read_only(&self) -> bool {
        self.read_only
    }

    /// Whether the runtime may be told the import's pages are writable ([`host_memory_vouched`]):
    /// the memory itself, not a twin, every page of which the table shows mapped with write
    /// permission now, as the library's own views and the guest's RAM are. Where it does not
    /// know a page (memory of the library's own, an IOSurface stand-in's), or a page is
    /// read-only, the runtime asks the kernel as before.
    pub(crate) fn vouched(&self) -> bool {
        self.vouched
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves. With a twin: `inner`, given `original`, the address the bytes were
    /// given at, and the twin, which the runtime drops once it freed the import, so the twin is
    /// unmapped after it. Without one, `inner` as it is, and the record of the pages a device may
    /// pin ([`Pinned`]), where there is one.
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            let pinned = self.pinned.map(|p| -> rt::KeepAlive { Box::new(p) });
            return (inner, pinned);
        };
        let original = SendPtr(original);
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| inner(original.get(), length))
        });
        (inner, Some(Box::new(twin)))
    }
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]), with a
/// source of dma-bufs of its pages for a device that refuses them as host memory, and the views'
/// gate ([`import_gate`]) (module docs).
/// The runtime keeps the source while the import lives, and calls it with no lock of its own
/// held, with a window of the import's pages and the granule it binds chunks at: the whole
/// import's when it makes it, and a texture's own pages, at a granule of their length, when it
/// makes a texture over it. The source answers dma-bufs that are the window's pages in order, each
/// within udmabuf's limits (`super::dma_buf`), which the runtime imports. A twin's window is the
/// twinned memory's pages ([`shown_at`]), which a dma-buf takes whatever their protection. A call
/// that makes none keeps its refusal on its thread for the death there ([`dma_buf_refusal`]), as
/// the runtime then refuses the import as it would have, or makes the texture as it does where
/// it cannot alias the memory.
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: NonNull<u8>) -> rt::HostMemory {
    // SAFETY: the caller's.
    unsafe { host_memory_noting(at, None) }
}

/// [`host_memory`], its pages' write permission vouched for where `vouched`
/// ([`ImportView::vouched`]): the runtime then asks the kernel nothing of the pages before it
/// imports them, where it would ask about every mapping over them, one `PROCMAP_QUERY` each
/// (measured on an RTX 5080 Laptop GPU, NVIDIA 615.71.09, Linux 7.2.8, a view of 9,000
/// mappings: its import took 13.3 ms checked and 7.7 ms vouched, and one of 1,024 mappings 1.46
/// against 0.77 ms).
///
/// # Safety
///
/// [`host_memory`]'s; and where `vouched`, every page of the import is mapped writable by this
/// process now (the table's record, [`ImportView::vouched`]), and stays mapped so while it is
/// imported, as `rt::HostMemory::assume_writable` asks: the library lowers no protection of a
/// view or of the guest's RAM once made, and an unmap of its own waits for the import's free
/// (`crate::remap`'s `UnmapHold`).
pub(crate) unsafe fn host_memory_vouched(at: NonNull<u8>, vouched: bool) -> rt::HostMemory {
    // SAFETY: the caller's.
    let memory = unsafe { host_memory_noting(at, None) };
    match vouched {
        // SAFETY: the caller's.
        true => unsafe { memory.assume_writable() },
        false => memory,
    }
}

/// [`host_memory`], setting `asked` where the runtime asks its source for dma-bufs.
///
/// # Safety
///
/// [`host_memory`]'s.
unsafe fn host_memory_noting(
    at: NonNull<u8>,
    asked: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> rt::HostMemory {
    // An earlier import's refusal is not this one's.
    REFUSAL.with_borrow_mut(|kept| *kept = None);
    // The import's files of zeros, which every request of its source over the same pages takes.
    let zeros = super::dma_buf::Zeros::default();
    let source: rt::DmaBufSource = Arc::new(move |request: rt::DmaBufRequest| {
        if let Some(asked) = &asked {
            asked.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let first = shown_at(request.start).as_ptr().addr().to_u64();
        let (length, granule) = (request.length, request.granule);
        match super::dma_buf::of_pages(first, length, granule, &zeros) {
            Ok(chunks) => {
                log::debug!(
                    "{length:#x} bytes at {first:#x} as {} dma-bufs at a granule of {granule:#x}",
                    chunks.len()
                );
                Ok(chunks)
            }
            Err(refusal) => {
                let line = format!("no dma-buf of its {length:#x} bytes at {first:#x}: {refusal}");
                log::debug!("{line}");
                REFUSAL.with_borrow_mut(|kept| *kept = Some(line));
                Err(std::io::Error::other(refusal.to_string()))
            }
        }
    });
    static GATE: std::sync::LazyLock<rt::ImportGate> =
        std::sync::LazyLock::new(|| Arc::new(import_gate));
    static RUNS: std::sync::LazyLock<rt::RunResolver> =
        std::sync::LazyLock::new(|| Arc::new(super::host_ram::runs));
    // SAFETY: the caller's.
    let memory =
        unsafe { rt::HostMemory::with_dma_buf(at, source) }.with_import_gate(Arc::clone(&GATE));
    // Where the devices take the guest's RAM whole (`super::host_ram`), the runs of it the pages
    // are, which the runtime asks of the import's window.
    match imports_need_files() {
        true => memory.with_runs(Arc::clone(&RUNS)),
        false => memory,
    }
}

thread_local! {
    /// The last refusal of a dma-buf source on this thread ([`host_memory`]).
    static REFUSAL: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Why this thread's last dma-buf source made none, taken, for the death of an import the runtime
/// refused after it ([`host_memory`]).
pub(crate) fn dma_buf_refusal() -> Option<String> {
    REFUSAL.with_borrow_mut(Option::take)
}

/// An address moved into a deallocator, which may run on any thread.
struct SendPtr(NonNull<u8>);

// SAFETY: an address only, handed back to the deallocator it was given for.
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// The address.
    fn get(self) -> NonNull<u8> {
        self.0
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes where every page of
/// them is mapped writable or one has no file behind it (which the runtime then refuses as it
/// refuses any import there), or where one is mapped read-only and `twin` is false (which the
/// gate refuses, so the runtime copies them, dropping the GPU's writes); and otherwise, a
/// surface's read-only alias, a writable twin of their pages, at the same offset in its first
/// page (module docs). Where a device needs imported pages mapped from their files, what of the
/// bytes lies in views through dma-bufs is mapped from them, and a twin made, short of the
/// headroom of the views' budget, past which the gate refuses the pages and the runtime copies
/// them ([`import_gate`]): the runtime asks the gate about every import, a no-copy buffer's,
/// heap's or rebind's. Where none does, a twin may take the whole budget.
pub(crate) fn import_view(at: NonNull<u8>, length: u64, twin: bool) -> ImportView {
    let mut plain = ImportView {
        at,
        twin: None,
        pinned: None,
        vouched: false,
        read_only: false,
    };
    let start = at.as_ptr().addr().to_u64();
    let page = vm::page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|e| e.checked_next_multiple_of(page))
    else {
        return plain;
    };
    // Only where the gate refuses what does not fit short of the headroom.
    let reach = match imports_need_files() {
        true => budget::Reach::ShortOfHeadroom,
        false => budget::Reach::Whole,
    };
    // Looked up without collecting the pieces, which only a twin needs.
    let writable = files::writable(first, end - first);
    if writable != Some(false) {
        // Recorded before its pages are mapped from their files, so no fold maps them back.
        let pinned = Pinned::new(first, end);
        // Pages left in views through dma-bufs for want of room are refused at the gate.
        if imports_need_files() {
            let _ = vm::unfold(first, end);
        }
        return ImportView {
            pinned,
            vouched: writable == Some(true),
            ..plain
        };
    }
    plain.read_only = true;
    if !twin {
        return plain;
    }
    let Some(pieces) = files::pieces(first, end - first) else {
        return plain;
    };
    let Some(twin) = vm::reclaiming(|| self::twin(first, end - first, &pieces, reach)) else {
        return plain;
    };
    log::debug!(
        "an import of {length:#x} bytes at {start:#x}: a writable twin at {:#x} of {} mappings",
        twin.first,
        twin.mappings
    );
    let Some(twin_at) = address(twin.first + (start - first)) else {
        return plain;
    };
    ImportView {
        at: twin_at,
        twin: Some(twin),
        ..plain
    }
}

/// A writable twin of the `length` bytes from `first` that `pieces` show: each piece's file
/// pages, shared, read and write, but zeros', which stay the fresh reservation's own private
/// zeros; `None` where the host maps none, or the views' budget has no room for it as far as
/// `reach` lets it: short of the headroom, where a device needs imported pages mapped from their
/// files ([`twinless`]), or the whole budget ([`budget::admit`]).
fn twin(first: u64, length: u64, pieces: &[files::Piece], reach: budget::Reach) -> Option<Twin> {
    let zeros = of_zeros(pieces);
    let mappings = twin_mappings(pieces, &zeros);
    // Held until the twin's mappings are counted.
    let _claim = match reach {
        budget::Reach::ShortOfHeadroom => match vm::claim_short(mappings) {
            Some(claim) => Some(claim),
            None => {
                twinless(first, first + length, mappings);
                return None;
            }
        },
        // Past the budget no twin, counted: the runtime refuses the read-only pages as it
        // refuses any import of them.
        budget::Reach::Whole => match budget::admit(mappings, "an import's twin") {
            true => None,
            false => return None,
        },
    };
    let base = vm::anonymous(None, length, "an import's twin").ok()?;
    let mut twin = Twin {
        first: base,
        length,
        mappings: 0,
    };
    let rw = libc::PROT_READ | libc::PROT_WRITE;
    for (piece, zeros) in pieces.iter().zip(zeros) {
        if !zeros {
            vm::file(Some(base + piece.at), piece.length, &piece.backing, rw).ok()?;
        }
    }
    budget::twins_hold(mappings, true);
    twin.mappings = mappings;
    TWINS.lock().insert(base, (length, first));
    Some(twin)
}

/// Counts an import of the read-only pages `first..end` that no twin was made for, as its
/// `adding` mappings would have come within the headroom of the views' budget: the gate refuses
/// the pages, and the runtime copies them ([`TWINLESS`], trace record `notwin`).
fn twinless(first: u64, end: u64, adding: u64) {
    if let Some((count, bytes)) = TWINLESS.count(first, end, adding) {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{count} buffers and heaps over read-only pages, of {} KiB in all, copied rather \
             than imported, which costs each command buffer a compare of their bytes: a writable \
             twin of their pages would have taken the views of memory within {} mappings of \
             their budget (vm.max_map_count), the latest's {adding} mappings for {:#x} bytes at \
             {first:#x}",
            bytes >> 10,
            budget::HEADROOM,
            end - first
        );
    }
}

/// The kernel's mappings a twin of `pieces` takes, `zeros` saying which are of zeros: one for each
/// run of them that maps as one, pieces of zeros next to each other (private zeros, as the twin's
/// reservation is) or each piece following the one before in the same file (all read and write
/// in the twin).
fn twin_mappings(pieces: &[files::Piece], zeros: &[bool]) -> u64 {
    let apart = (1..pieces.len())
        .filter(|&i| {
            let (a, b) = (&pieces[i - 1], &pieces[i]);
            let together = if zeros[i - 1] || zeros[i] {
                zeros[i - 1] && zeros[i]
            } else {
                Arc::ptr_eq(&a.backing.fd, &b.backing.fd)
                    && a.backing.offset + a.length == b.backing.offset
            };
            !together
        })
        .count();
    apart.to_u64() + 1
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, but for an address in a
/// writable twin, the twinned memory's at the same offset (module docs).
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let addr = at.as_ptr().addr().to_u64();
    let twins = TWINS.lock();
    let Some((&twin, &(length, first))) = twins.range(..=addr).next_back() else {
        return at;
    };
    if addr >= twin + length {
        return at;
    }
    address(first + (addr - twin)).unwrap_or(at)
}

/// A pointer to `at` in the process's address space; `None` for 0.
fn address(at: u64) -> Option<NonNull<u8>> {
    NonNull::new(vm::ptr(at).cast::<u8>())
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::os::vm::{HostVm as _, VmProt};

    use super::super::LinuxVm;
    use super::*;

    const PAGE: u64 = 0x1000;

    fn read(at: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { address(at).unwrap().as_ptr().read_volatile() }
    }

    fn write(at: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { address(at).unwrap().as_ptr().write_volatile(v) }
    }

    fn addr(p: NonNull<u8>) -> u64 {
        p.as_ptr().addr().to_u64()
    }

    /// Writable memory is imported as it is; read-only memory through a writable twin of its
    /// pages, at the same offset, whose writes reach them, so the import keeps writes, and whose
    /// addresses answer the memory's own until the runtime drops its keep-alive; the deallocator
    /// is given the memory's own address, and the twin outlives it
    #[test]
    fn read_only_memory_is_imported_through_a_writable_twin() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first + 0x40).unwrap();
        let view = import_view(p, PAGE, true);
        assert_eq!(view.at(), p, "writable: the memory itself");
        assert!(!view.twinned());
        let (run, kept) = view.release(p, None);
        assert!(run.is_none() && kept.is_none());
        assert!(vm.protect_max(at, 2 * PAGE, VmProt::READ).is_ok());
        let view = import_view(p, PAGE, true);
        assert!(view.twinned());
        let twin = addr(view.at());
        assert_ne!(twin, addr(p));
        assert_eq!(twin & (PAGE - 1), 0x40, "the offset in its page");
        write(twin + PAGE, 0x5a);
        assert_eq!(
            read(first + 0x40 + PAGE),
            0x5a,
            "the twin's writes reach the pages"
        );
        assert_eq!(shown_at(view.at()), p);
        let inside = address(twin + 0x10).unwrap();
        assert_eq!(addr(shown_at(inside)), addr(p) + 0x10);
        let given = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&given);
        let inner: rt::Deallocator = Box::new(move |at, length| {
            *seen.lock() = Some((at.as_ptr().addr().to_u64(), length));
        });
        let (free, kept) = view.release(p, Some(inner));
        free.unwrap()(address(twin).unwrap(), PAGE);
        assert_eq!(*given.lock(), Some((addr(p), PAGE)));
        assert_eq!(
            addr(shown_at(inside)),
            addr(p) + 0x10,
            "the twin outlives it"
        );
        drop(kept);
        assert_eq!(addr(shown_at(inside)), twin + 0x10, "the twin is gone");
        assert!(vm.deallocate(at, 2 * PAGE).is_success());
    }

    /// Read-only memory imported without a twin (all but a surface's read-only alias) is given as
    /// it is, read-only, and the gate refuses its pages, so the runtime copies them and drops the
    /// GPU's writes; writable memory is not read-only, with a twin asked for or not
    #[test]
    fn read_only_memory_without_a_twin_is_refused_at_the_gate() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first).unwrap();
        for twin in [false, true] {
            let view = import_view(p, PAGE, twin);
            assert_eq!((view.at(), view.read_only()), (p, false), "writable");
            assert!(gate(first, first + PAGE));
        }
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let view = import_view(p, PAGE, false);
        assert_eq!(view.at(), p, "the pages themselves");
        assert!(view.read_only() && !view.twinned());
        assert!(!view.vouched());
        assert!(!gate(first, first + PAGE), "refused, so copied");
        let (run, kept) = view.release(p, None);
        assert!(run.is_none() && kept.is_none());
        let twinned = import_view(p, PAGE, true);
        assert!(twinned.read_only() && twinned.twinned());
        drop(twinned);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// The runtime may be told an import's pages are writable only where the table shows every
    /// one mapped writable as it is made: a range a view can be made of is vouched for; once one
    /// of its pages is protected read-only, the next import over it is a twin and is not; and
    /// memory the table does not know (no file behind it) is not either
    #[test]
    fn only_pages_the_table_shows_writable_are_vouched_for() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first).unwrap();
        let view = import_view(p, 2 * PAGE, true);
        assert_eq!(view.at(), p);
        assert!(view.vouched(), "every page writable");
        drop(view);
        let second = address(first + PAGE).unwrap();
        let second = reims_vgpu_core::host::HostPtr::from_ptr(second.as_ptr());
        assert!(vm.protect_max(second, PAGE, VmProt::READ).is_ok());
        let view = import_view(p, 2 * PAGE, true);
        assert!(view.twinned(), "a twin of the read-only page");
        assert!(!view.vouched(), "a page read-only since the first import");
        drop(view);
        let view = import_view(p, PAGE, true);
        assert!(view.vouched(), "the page still writable");
        drop(view);
        assert!(vm.deallocate(at, 2 * PAGE).is_success());
        // SAFETY: a fresh anonymous mapping, unmapped below.
        let anon = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGE.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(anon, libc::MAP_FAILED);
        let view = import_view(NonNull::new(anon.cast::<u8>()).unwrap(), PAGE, true);
        assert!(!view.vouched(), "memory the table does not know");
        drop(view);
        // SAFETY: the mapping made above.
        assert_eq!(unsafe { libc::munmap(anon, PAGE.to_usize()) }, 0);
    }

    /// A twin takes a mapping for each run of its pieces that maps as one, following pieces of
    /// one file or neighbouring pieces of zeros, and the kernel maps it so
    #[test]
    fn a_twin_counts_its_mappings_as_the_kernel_maps_them() {
        let f = vm::memfd(4 * PAGE).unwrap();
        let zero_at = allocate_zeros(PAGE).unwrap();
        let z = files::pieces(zero_at.addr().to_u64(), 1).unwrap()[0]
            .backing
            .fd
            .clone();
        let piece = |at: u64, fd: &Arc<OwnedFd>, offset: u64| files::Piece {
            at: at * PAGE,
            length: PAGE,
            backing: files::Backing {
                fd: Arc::clone(fd),
                offset: offset * PAGE,
            },
            prot: libc::PROT_READ,
        };
        let mappings = |pieces: &[files::Piece]| twin_mappings(pieces, &of_zeros(pieces));
        assert_eq!(mappings(&[piece(0, &f, 0), piece(1, &f, 1)]), 1);
        assert_eq!(mappings(&[piece(0, &f, 1), piece(1, &f, 0)]), 2);
        assert_eq!(mappings(&[piece(0, &z, 0), piece(1, &z, 5)]), 1);
        let mixed = [piece(0, &f, 0), piece(1, &z, 0), piece(2, &f, 2)];
        assert_eq!(mappings(&mixed), 3);
        // A read-only view of the file's pages 2, 3, 0 and 1: its twin maps as two.
        let vm = LinuxVm::new();
        let source = vm::allocate_aliasable(4 * PAGE).unwrap();
        let view = vm.allocate_anywhere(4 * PAGE).unwrap();
        for (to, from) in [(0, 2), (1, 3), (2, 0), (3, 1)] {
            vm.remap_shared_overwrite(
                view.wrapping_add(to * PAGE),
                source.wrapping_add(from * PAGE),
                PAGE,
            )
            .unwrap();
        }
        assert!(vm.protect_max(view, 4 * PAGE, VmProt::READ).is_ok());
        let p = address(view.addr().to_u64()).unwrap();
        let import = import_view(p, 4 * PAGE, true);
        let twin = addr(import.at());
        assert_ne!(twin, addr(p));
        assert_eq!(vm::tests::kernel_mappings(twin, twin + 4 * PAGE), 2);
        drop(import);
        assert!(vm.deallocate(view, 4 * PAGE).is_success());
        assert!(vm.deallocate(source, 4 * PAGE).is_success());
        assert!(vm.deallocate(zero_at, PAGE).is_success());
    }

    /// A twin of zeros is private zeros: its writes never reach the shared file
    #[test]
    fn a_twin_of_zeros_keeps_its_writes_to_itself() {
        let vm = LinuxVm::new();
        let at = allocate_zeros(PAGE).unwrap();
        let first = at.addr().to_u64();
        assert!(zeros(&files::pieces(first, 1).unwrap()[0].backing.fd));
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let p = address(first).unwrap();
        let view = import_view(p, PAGE, true);
        let twin = addr(view.at());
        assert_ne!(twin, first);
        assert_eq!(read(twin), 0);
        write(twin, 7);
        assert_eq!((read(twin), read(first)), (7, 0));
        drop(view);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// A read-only alias of guest pages shorter than its size, a heap's or a replace's, is
    /// imported through a writable twin: the pages' bytes, then zeros of the twin's own past
    /// them, as its zeros map the block of zeros
    #[test]
    fn a_read_only_alias_with_zeros_past_its_pages_is_twinned() {
        use reims_vgpu_core::platform::types::ResourceAddressRange;

        let vm = LinuxVm::new();
        let guest = vm::allocate_aliasable(PAGE).unwrap();
        let first = guest.addr().to_u64();
        write(first + 5, 0x6b);
        let ranges = [ResourceAddressRange {
            address: first,
            length: PAGE,
        }];
        let alias = crate::gpu::backing::gather("test", &ranges, 3 * PAGE, true).unwrap();
        let at = alias.first();
        let view = import_view(at, 3 * PAGE, true);
        let twin = addr(view.at());
        assert_ne!(twin, addr(at), "a twin");
        assert_eq!(read(twin + 5), 0x6b, "the guest's page");
        assert_eq!((read(twin + PAGE), read(twin + 3 * PAGE - 1)), (0, 0));
        write(twin + 2 * PAGE, 9);
        assert_eq!(
            read(addr(at) + 2 * PAGE),
            0,
            "the alias's zeros keep none of it"
        );
        drop(view);
        drop(alias);
        assert!(vm.deallocate(guest, PAGE).is_success());
    }

    /// Readable pages with no file behind them, which no view is made of, end the process at the
    /// call that gathers them into an alias, where Metal takes them; a call whose refusal is nil
    /// answers nil there by default, counted, as when memory runs out, and dies in strict mode
    #[test]
    fn a_gather_of_pages_without_a_file_is_the_hosts_refusal() {
        use reims_vgpu_core::platform::types::ResourceAddressRange;
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;

        let vm = LinuxVm::new();
        let at = vm.allocate_anywhere(PAGE).unwrap();
        let ranges = [ResourceAddressRange {
            address: at.addr().to_u64(),
            length: PAGE,
        }];
        let gathered = || crate::gpu::backing::gather("test", &ranges, PAGE, false).is_some();
        let site = catch_death(gathered).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "test"));
        let or_nil = || crate::gpu::backing::gather_or_nil("test", &ranges, PAGE, false).is_some();
        let site = catch_death(or_nil).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "test"), "strict");
        let before = crate::answer::out_of_host_memory_count();
        assert!(!crate::answer::relaxed(or_nil), "nil");
        assert_eq!(crate::answer::out_of_host_memory_count(), before + 1);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// Memory with no file behind it is given as it is, for the runtime to refuse
    #[test]
    fn memory_without_a_file_is_imported_as_it_is() {
        let vm = LinuxVm::new();
        let at = vm.allocate_anywhere(PAGE).unwrap();
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let p = NonNull::new(at.as_ptr::<u8>()).unwrap();
        assert_eq!(import_view(p, PAGE, true).at(), p);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// Where a device of the process needs imported pages mapped from their files, a view that
    /// from its files would come within the headroom of the views' budget is mapped through a
    /// dma-buf, though the budget has room for it. An import of it (a buffer's, a heap's or a
    /// rebind's) leaves it so, and the gate refuses its pages, counted with their bytes, in strict
    /// mode too, as the runtime copies them; with room short of the headroom, an import maps the
    /// pages from their files, which the gate lets through. Where udmabuf does not open, the view
    /// is mapped from its files, and ends the process past the budget. The flag is the process's,
    /// which nextest runs this test alone in.
    #[test]
    fn past_the_headroom_imports_are_refused_at_the_gate() {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        fn dies<R>(f: impl FnOnce() -> R) -> bool {
            catch_death(f).is_err_and(|site| (site.kind, site.file) == (DeathKind::Abort, "mmap"))
        }
        let counted = |r: &Refused| {
            (
                r.count.load(Ordering::Relaxed),
                r.bytes.load(Ordering::Relaxed),
            )
        };
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        let vm = LinuxVm::new();
        let n = 32u64;
        let src = vm::allocate_aliasable(n * PAGE).unwrap();
        let view = vm.allocate_anywhere(n * PAGE).unwrap();
        let runs: Vec<_> = (0..n)
            .map(|i| (src.wrapping_add((n - 1 - i) * PAGE), PAGE))
            .collect();
        let through = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .is_ok();
        // Room in the budget for the view from its files, none short of it by the headroom.
        let limit = |room| budget::held() + budget::RESERVE + room;
        if through {
            budget::set_test_limit(Some(limit(budget::HEADROOM + 8)));
            vm.remap_runs_shared_overwrite(view, &runs).unwrap();
            let p = NonNull::new(view.as_ptr::<u8>()).unwrap();
            let (first, end) = (addr(p), addr(p) + n * PAGE);
            let mappings = || vm::tests::kernel_mappings(first, end);
            assert_eq!(mappings(), 1, "the view, through a dma-buf");
            let import = import_view(p, n * PAGE, true);
            assert_eq!((import.at(), mappings()), (p, 1), "an import's, left");
            let (copied, copied_bytes) = counted(&COPIED);
            assert!(!gate(first, end), "refused at the gate");
            assert_eq!(counted(&COPIED), (copied + 1, copied_bytes + n * PAGE));
            drop(import);
            // Room short of the headroom.
            budget::set_test_limit(Some(limit(budget::HEADROOM + 2 * n)));
            let import = import_view(p, n * PAGE, true);
            assert_eq!((import.at(), mappings()), (p, n), "from their files");
            assert!(gate(first, end), "let through");
        } else {
            budget::set_test_limit(Some(limit(n)));
            assert!(dies(|| vm.remap_runs_shared_overwrite(view, &runs)));
        }
        budget::set_test_limit(None);
        assert!(vm.deallocate(view, n * PAGE).is_success());
        assert!(vm.deallocate(src, n * PAGE).is_success());
    }

    /// Where a device of the process needs imported pages mapped from their files, a gated
    /// import of read-only pages whose twin would come within the headroom of the views' budget
    /// gets none, counted with its bytes: the gate refuses the pages, read-only, so the runtime
    /// copies them. With room, the twin is made, which the gate lets through. The flag is the
    /// process's, which nextest runs this test alone in.
    #[test]
    fn a_gated_import_with_no_room_for_a_twin_is_refused_at_the_gate() {
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        assert!(vm.protect_max(at, 2 * PAGE, VmProt::READ).is_ok());
        let p = NonNull::new(at.as_ptr::<u8>()).unwrap();
        let (first, end) = (addr(p), addr(p) + PAGE);
        let before = TWINLESS.count.load(Ordering::Relaxed);
        budget::set_test_limit(Some(budget::held() + budget::RESERVE + budget::HEADROOM));
        let view = import_view(p, PAGE, true);
        assert_eq!(view.at(), p, "no twin");
        assert!(!gate(first, end), "refused, read-only");
        assert_eq!(TWINLESS.count.load(Ordering::Relaxed), before + 1);
        budget::set_test_limit(None);
        let view = import_view(p, PAGE, true);
        assert_ne!(view.at(), p, "a twin");
        let twin = addr(view.at());
        assert!(gate(twin, twin + PAGE), "the twin let through");
        drop(view);
        assert!(vm.deallocate(at, 2 * PAGE).is_success());
    }

    /// Where a device of the process imports by pointer only pages mapped from their files, an
    /// import of a view mapped through a dma-buf maps its pages from their files, the bytes as
    /// they were, and they stay so once it goes; a second import of them maps nothing more, and a
    /// view made from then on is mapped from its files. Where udmabuf does not open, every view
    /// is mapped from its files. The flag is the process's, which nextest runs this test alone in.
    #[test]
    fn an_import_maps_its_pages_from_their_files() {
        let vm = LinuxVm::new();
        let n = 32u64;
        let src = vm::allocate_aliasable(n * PAGE).unwrap();
        let backwards = |view: HostAddr| {
            let runs: Vec<_> = (0..n)
                .map(|i| (src.wrapping_add((n - 1 - i) * PAGE), PAGE))
                .collect();
            vm.remap_runs_shared_overwrite(view, &runs).unwrap();
            let first = addr(NonNull::new(view.as_ptr::<u8>()).unwrap());
            (first, first + n * PAGE)
        };
        let view = vm.allocate_anywhere(n * PAGE).unwrap();
        let (first, end) = backwards(view);
        let through = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .is_ok();
        let mappings = |first, end| vm::tests::kernel_mappings(first, end);
        assert_eq!(
            mappings(first, end),
            if through { 1 } else { n },
            "the view"
        );
        write(first + 7, 0x77);
        IMPORTS_NEED_FILES.store(true, std::sync::atomic::Ordering::Relaxed);
        let p = NonNull::new(view.as_ptr::<u8>()).unwrap();
        let whole = import_view(p, n * PAGE, true);
        assert_eq!((whole.at(), mappings(first, end)), (p, n), "imported");
        assert_eq!(read(first + 7), 0x77);
        drop(whole);
        let again = import_view(address(first + 4 * PAGE).unwrap(), 2 * PAGE, true);
        drop(again);
        assert_eq!(mappings(first, end), n, "no import");
        assert_eq!(read(first + 7), 0x77);
        let other = vm.allocate_anywhere(n * PAGE).unwrap();
        let (o_first, o_end) = backwards(other);
        assert_eq!(mappings(o_first, o_end), n, "a view made since");
        assert!(vm.deallocate(other, n * PAGE).is_success());
        assert!(vm.deallocate(view, n * PAGE).is_success());
        assert!(vm.deallocate(src, n * PAGE).is_success());
    }
}
