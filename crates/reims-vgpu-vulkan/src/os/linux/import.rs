//! What the runtime imports for host memory on Linux: the memory itself, and for a surface's
//! read-only alias, a writable twin of it.
//!
//! Linux drivers pin imported pages for writing, so the runtime imports no read-only memory
//! (typed, `HostMemory`), where Metal reads it: the guest's ranges core maps read-only, aliases
//! of a read-only replace or heap, and surfaces the GPU only reads. For aliases of a read-only
//! replace or heap, and a no-copy buffer or heap over pages mapped read-only, the runtime is
//! given the pages themselves: the views' gate refuses them ([`import_gate`]), and the runtime
//! makes the buffer or heap over a copy of them whose GPU writes it drops, as Metal's GPU drops
//! them (its `copied-host-pages`, a kind of `crate::answer::DIES`), so the GPU reads what Metal's
//! reads and its writes reach nothing past the copy ([`ImportView::read_only`]); so does an
//! object made with `MTLResourceOptions` bit 24, whatever its pages. What Metal answers otherwise
//! than a dropped write there (a pass's store into such memory, a texture's writes) is
//! `crate::gpu::writes`'s.
//!
//! A surface's read-only alias is given a **twin** instead
//! (`crate::gpu::backing::ImportSite::twins`): a second view of the same file pages, read and
//! write, at an address of its own ([`import_view`]), imported in place, so a surface's buffers
//! lie in one import of the pages, as a writer's GPU writes through another import reach its
//! readers, where a copy would read them as they were at its commit. The GPU then reads what
//! Metal's reads, but its writes reach those pages, where Metal drops them: every command that
//! may write through an object over such an import dies where it is encoded, naming it
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

use crate::core::num::{ToU64, ToUsize};
use crate::core::platform::os::vm::{HostAddr, KernReturn};
use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

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
/// ([`Pinned`]), with how many of its imports are still being made. A fold of the views maps
/// none of them through dma-bufs (`super::vm::fold_idle`): a device took them by their mappings.
/// Where the devices keep what they pinned as the import was made ([`folds_after_import`]), a
/// record none of whose imports is still being made is left out ([`pinned`]).
static PINNED: Mutex<BTreeMap<(u64, u64), Record>> = Mutex::new(BTreeMap::new());

/// One import's pages in [`PINNED`]: their end, and how many of its imports the runtime is still
/// making ([`ImportMade`]).
#[derive(Debug, Clone, Copy)]
struct Record {
    end: u64,
    making: u32,
}

/// One import's pages in [`PINNED`], until it drops.
#[derive(Debug)]
pub(crate) struct Pinned((u64, u64));

impl Pinned {
    /// The pages of `first..end`, where a device of the process takes imported pages as they are
    /// mapped from their files ([`imports_need_files`]); `None` where none does. Its import is
    /// being made until its [`ImportMade`] says otherwise.
    fn new(first: u64, end: u64) -> Option<Pinned> {
        imports_need_files().then(|| Pinned::of(first, end))
    }

    /// The pages of `first..end`, whatever the devices, being made.
    pub(super) fn of(first: u64, end: u64) -> Pinned {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let key = (first, NEXT.fetch_add(1, Ordering::Relaxed));
        PINNED.lock().insert(key, Record { end, making: 1 });
        Pinned(key)
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        PINNED.lock().remove(&self.0);
        UNPINNED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The word that an import of the pages a [`Pinned`] records was made, which a device that keeps
/// what it pinned at the import lets a fold map through dma-bufs ([`folds_after_import`]): told
/// by the caller once the runtime's call that imports them returned ([`ImportMade::made`]).
/// Dropped untold, the record stays as one being made, which no fold maps.
#[derive(Debug)]
#[must_use = "an import never told made stays out of every fold"]
pub(crate) struct ImportMade(Option<(u64, u64)>);

impl ImportMade {
    /// The runtime's call that imports the pages on `device` returned: where the device keeps an
    /// import of a pointer it made until it frees it, rather than releasing it while idle and
    /// importing the pointer again at the next use (`rt::Device::releases_idle_pointer_imports`,
    /// which answers `false` for an import made before its check answered), the import is made,
    /// and once none of the record's imports is being made, its pages may be folded.
    pub(crate) fn made(self, device: &rt::Device) {
        let Some(key) = self.0 else {
            return;
        };
        if !folds_after_import(device) || device.releases_idle_pointer_imports() {
            return;
        }
        made_now(key);
    }

    /// [`ImportMade::made`], as on a device that keeps what it pinned at the import.
    #[cfg(test)]
    fn made_here(self) {
        if let Some(key) = self.0 {
            made_now(key);
        }
    }
}

/// One import of the record `key` of [`PINNED`] made.
fn made_now(key: (u64, u64)) {
    if let Some(record) = PINNED.lock().get_mut(&key) {
        record.making = record.making.saturating_sub(1);
    }
    // A fold may free more now ([`unpinned`]).
    UNPINNED.fetch_add(1, Ordering::Relaxed);
}

/// How many records of [`PINNED`] have dropped, or had an import made, each freeing pages a fold
/// may map through dma-bufs (`super::vm`'s `fold_short`).
static UNPINNED: AtomicU64 = AtomicU64::new(0);

/// How many records of [`PINNED`] have dropped or had an import made ([`UNPINNED`]).
pub(super) fn unpinned() -> u64 {
    UNPINNED.load(Ordering::Relaxed)
}

/// The pages devices may pin now ([`PINNED`]), in order, those that meet joined: those of each
/// record with an import still being made, or made on a device that does not keep what it pinned
/// at the import ([`ImportMade::made`]).
pub(super) fn pinned() -> Vec<std::ops::Range<u64>> {
    let mut ranges: Vec<std::ops::Range<u64>> = (PINNED.lock().iter())
        .filter(|(_, record)| record.making > 0)
        .map(|(&(first, _), record)| first..record.end)
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

/// The GPUs, by registry ID, whose imports of a pointer's pages a fold may map through dma-bufs
/// once made ([`ImportMade::made`]), and those it may not: learned of the first device of each
/// that is made once a device of the process needs imported pages mapped from their files
/// ([`probe_fold`]). NVIDIA's driver pins the pages as it imports them and keeps those pages,
/// whatever is mapped over their range later (measured, `tests/no_copy_remap_linux.rs`), so the
/// view's own mappings, a mapping a run of scattered pages, may go once the import is made; a
/// driver that gets the pages again from their mapping at a later use (i915's userptr, through
/// its notifier) fails the probe, and its imports keep their views' mappings.
static FOLDS: Mutex<Vec<(u64, bool)>> = Mutex::new(Vec::new());

/// Whether a GPU of the process passed [`probe_fold`] ([`FOLDS`]).
pub(super) fn folds_any() -> bool {
    FOLDS.lock().iter().any(|&(_, folds)| folds)
}

/// Whether `device`'s GPU passed [`probe_fold`] ([`FOLDS`]).
fn folds_after_import(device: &rt::Device) -> bool {
    let id = device.registry_id();
    FOLDS.lock().iter().any(|&(at, folds)| at == id && folds)
}

/// Whether a device of the process imports by pointer only pages mapped from their files: learned
/// of each device as it is made ([`learn_view_imports`]), and set where any refuses a view through
/// dma-bufs. From then on views are mapped from their files (`super::vm`'s module docs, "Imports
/// of views").
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
/// logged at the first and at each power of two, at debug after the first (`core::logging`'s once
/// per kind).
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
        crate::core::trace::event(self.record, &[first, end - first, adding]);
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

/// Learns whether `device` imports a view mapped through dma-bufs as it is mapped, as the
/// platform's imports hand the runtime a view ([`host_memory`]): a probe of a view of one page
/// more than udmabuf takes in one dma-buf, so two ([`probe_view`]). Where the device does not take
/// it through the dma-bufs its source makes, views are mapped from their files from then on
/// ([`IMPORTS_NEED_FILES`]): where it refuses it, or makes a copy of the view in its place as its
/// import failed, and where it takes the view's pointer, as no driver pins a `VM_PFNMAP`
/// mapping's pages and one that pins them at the GPU's first use of them, not at the import (as
/// i915's userptr does without `I915_USERPTR_PROBE`, not measured here), would fail there. A
/// device imports such a view by its dma-bufs where it imports none by pointer (RADV), and
/// refuses it where it pins the pointer's pages and imports no foreign dma-buf under a sparse
/// buffer (NVIDIA's 615 driver, measured 2026-10-27, `super::vm`'s module docs, "Imports of
/// views"). Either way the device is handed the guest's RAM then
/// (`super::host_ram`). Nothing is learned where the view is not mapped through dma-bufs (udmabuf
/// not the process's): no import then needs it, and the device takes no RAM.
pub(crate) fn learn_view_imports(device: &rt::Device) {
    if IMPORTS_NEED_FILES.load(Ordering::Relaxed) {
        learn_folds(device);
        return;
    }
    let limits = super::dma_buf::limits();
    let Some((refused, through_dma_bufs)) =
        probe_view(device, limits.pieces.saturating_add(1).min(1 << 16))
    else {
        return;
    };
    let why = dma_buf_refusal();
    let name = &device.properties().name;
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "{name}: an import of a view of guest pages mapped through dma-bufs {}",
        match (refused, through_dma_bufs) {
            (true, _) => {
                "is refused, or copied as its import failed, so views are mapped from their files"
            }
            (false, false) => {
                "is taken by its pointer, which a driver pinning the pages at the GPU's first use \
                 of them would fail then, so views are mapped from their files"
            }
            (false, true) => "is taken through its dma-bufs",
        }
    );
    if let Some(why) = why {
        log::debug!(target: "reims_vgpu_vulkan::os", "the view's dma-buf source: {why}");
    }
    if !through_dma_bufs {
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        learn_folds(device);
        budget::warn_if_imports_short();
    }
    // Either way the device copies by runs of the guest's RAM: one whose views are mapped from
    // their files takes objects over the RAM too (`super::host_ram`).
    super::host_ram::hand_ram_to(device);
}

/// How long the last of [`learn_folds`]' answers took, a kept verdict's read or the probe, in
/// nanoseconds: what it added to its device's making.
static PROBE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Learns of `device`'s GPU, once, whether a fold may map the pages of its imports of a pointer
/// through dma-bufs once they are made ([`FOLDS`]), where a device of the process needs imported
/// pages mapped from their files: from the verdict kept for the GPU's driver and the kernel
/// ([`kept_fold`]), else by the probe ([`probe_fold`]), about 20 ms of the device's making
/// (measured on the Iris Xe of RPL-P, 2026-11-11, 17 to 26 ms over ten makings, against 2 to 3
/// ms for the probe made again once the device is), whose answer is kept where it is the GPU's
/// own ([`lasting`]).
fn learn_folds(device: &rt::Device) {
    if !imports_need_files() {
        return;
    }
    let id = device.registry_id();
    if FOLDS.lock().iter().any(|&(at, _)| at == id) {
        return;
    }
    use crate::core::platform::os::Clock as _;
    let clock = crate::os::OsClock::new();
    let start = clock.now_mono();
    let kept = fold_key(device).map(|(file, key)| {
        let kept = kept_fold(&file, &key);
        (file, key, kept)
    });
    let (answer, how) = match kept {
        Some((_, _, Some(answer))) => (answer, "the verdict kept for its driver"),
        kept => {
            let answer = probe_fold(device).map_err(str::to_owned);
            if let Some((file, key, _)) = kept
                && answer.as_ref().err().is_none_or(|why| lasting(why))
                && let Err(error) = keep_fold(&file, &key, &answer)
            {
                log::debug!(
                    target: "reims_vgpu_vulkan::os",
                    "the fold probe's verdict was not kept in {}: {error}",
                    file.display()
                );
            }
            (answer, "the probe")
        }
    };
    let nanos = clock.now_mono().as_nanos().saturating_sub(start.as_nanos());
    PROBE_NANOS.store(nanos, Ordering::Relaxed);
    let took = nanos as f64 / 1e6;
    let name = &device.properties().name;
    match &answer {
        Ok(()) => log::info!(
            target: "reims_vgpu_vulkan::os",
            "{name}: an import of a view of guest pages mapped from their files keeps its pages \
             once the view is mapped through a dma-buf, so a made import's view is folded at the \
             views' budget ({how}, {took:.1} ms)"
        ),
        Err(why) => log::info!(
            target: "reims_vgpu_vulkan::os",
            "{name}: a made import's view stays mapped from its files: {why} ({how}, {took:.1} \
             ms)"
        ),
    }
    let mut folds = FOLDS.lock();
    if !folds.iter().any(|&(at, _)| at == id) {
        folds.push((id, answer.is_ok()));
    }
}

/// The first line of a kept fold verdict ([`keep_fold`]): its format, which a change to what
/// [`probe_fold`] asks moves on.
const FOLD_HEADER: &str = "reims-vgpu fold after import 1";

/// Where `device`'s GPU's fold verdict is kept, `device-checks/fold-after-import-UUID` under the
/// library's cache directory, and the key it is kept under: the GPU's UUID, its driver's version
/// as the loader's probe read it (`crate::os`'s `seen_devices`), and the kernel's release, whose
/// udmabuf and memory map the probe goes through. `None` where the cache directory, the GPU's
/// driver version or the release is not known: the probe runs, and nothing is kept.
fn fold_key(device: &rt::Device) -> Option<(std::path::PathBuf, String)> {
    let uuid = device.properties().device_uuid;
    let seen = (crate::os::seen_devices().iter()).find(|seen| seen.uuid == Some(uuid))?;
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").ok()?;
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    let file = crate::platform::cache_directory()?
        .join("device-checks")
        .join(format!("fold-after-import-{hex}"));
    let key = format!(
        "{hex} | driver {} | kernel {}",
        seen.driver_version,
        kernel.trim()
    );
    Some((file, key))
}

/// The fold verdict `file` keeps under `key` ([`keep_fold`]): `None` where it is missing,
/// unreadable, of another format or another key.
fn kept_fold(file: &std::path::Path, key: &str) -> Option<Result<(), String>> {
    let text = std::fs::read_to_string(file).ok()?;
    let mut lines = text.lines();
    if lines.next()? != FOLD_HEADER || lines.next()?.strip_prefix("key ")? != key {
        return None;
    }
    match lines.next()? {
        "passed" => Some(Ok(())),
        line => Some(Err(line.strip_prefix("failed ")?.to_owned())),
    }
}

/// Keeps `answer` in `file` under `key`, through a file beside it renamed over it: three lines,
/// [`FOLD_HEADER`], the key, and `passed` or `failed` and why.
fn keep_fold(
    file: &std::path::Path,
    key: &str,
    answer: &Result<(), String>,
) -> std::io::Result<()> {
    let verdict = match answer {
        Ok(()) => "passed".to_owned(),
        Err(why) => format!("failed {}", why.replace(['\n', '\r'], " ")),
    };
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut beside = file.as_os_str().to_owned();
    beside.push(format!(".{}", std::process::id()));
    std::fs::write(&beside, format!("{FOLD_HEADER}\nkey {key}\n{verdict}\n"))?;
    std::fs::rename(&beside, file)
}

/// Whether [`probe_fold`]'s refusal `why` is the GPU's own answer, which is kept for its driver
/// ([`keep_fold`]): what the GPU read or wrote, or that its command buffer failed or did not
/// complete, or that the runtime did not import the pages by their pointer; not the probe's want
/// of memory, of a queue or of udmabuf, which the next device's making asks again.
fn lasting(why: &str) -> bool {
    why.starts_with("its GPU")
        || why.starts_with("its command buffer")
        || why == "its pages were not imported by their pointer"
}

/// [`learn_folds`]' probe of `device`: a no-copy buffer over a view of [`FOLD_PROBE_PAGES`] pages
/// backwards, mapped from their files and made as an import is ([`import_view`]), placed outside
/// the guest's numbering. First other pages of a file of their own are mapped over the view and
/// the GPU copies the buffer out: a driver that pinned the pages at the import reads the old ones
/// (NVIDIA's, `tests/no_copy_remap_linux.rs`), and one that gets them again from their mapping
/// at a later use reads the new ones, which fails the probe there, before any fold, as its next
/// use of a view folded into a dma-buf would fail where nothing pins a `VM_PFNMAP` mapping's
/// pages (measured: on the Iris Xe of RPL-P, ANV over i915, Linux 7.3.1,
/// 2026-11-11, lost the device on the probe's command buffer when the probe folded first). Then
/// the view is mapped from its files again and folded into a dma-buf of its pages
/// (`super::vm::fold_held`), as a fold maps it; bytes are written through the pages' own
/// mapping, as QEMU writes the guest's RAM; and one command buffer copies the buffer out and
/// fills its first half ([`FoldCheck`]). `Ok` where the runtime imported the pages by their
/// pointer, the GPU read the old pages through the remap and the bytes written after the fold,
/// and its fill shows through the pages' own mapping and through the folded view; else what
/// failed. The buffer is released at once, its pages unmapped once the runtime freed its import
/// ([`ProbePages`]).
fn probe_fold(device: &rt::Device) -> Result<(), &'static str> {
    use crate::core::platform::os::vm::HostVm as _;
    let vm = super::LinuxVm::new();
    let page = vm::page_size();
    let length = FOLD_PROBE_PAGES * page;
    let (Ok(src), Ok(view)) = (vm::allocate_aliasable(length), vm.allocate_anywhere(length)) else {
        return Err("no memory for its pages");
    };
    let check = FoldCheck {
        own: src.addr().to_u64(),
        view: view.addr().to_u64(),
        pages: FOLD_PROBE_PAGES,
        round: 0,
    };
    let end = check.view + length;
    // Unmapped as they drop: once the runtime freed the probe's import, where it makes one.
    let pages = ProbePages::new(check.view, check.own, length);
    if vm.remap_runs_shared_overwrite(view, &check.runs()).is_err()
        || !files::through_pieces(check.view, end).is_empty()
    {
        return Err("its view was not mapped from its files");
    }
    check.write();
    let at = address(check.view).ok_or("its view has no address")?;
    let import = import_view(at, length, false);
    let imported_at = import.at();
    let (_, kept) = import.release(at, None);
    let imports = || device.work_counts().host_pointer_imports_made;
    let before = imports();
    // SAFETY: the view is mapped, read and write, until the runtime drops the keep-alive, which
    // owns it and its pages' file, once it freed the import; the remap and the fold map other
    // mappings over it in one call each, never leaving it unmapped.
    let memory = unsafe { host_memory(imported_at) };
    let contents = rt::BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: Some(crate::gpu::backing::keep_alive((pages, kept))),
    };
    let descriptor = rt::BufferDescriptor::new(length, rt::ResourceOptions::shared(), contents)
        .with_placement(rt::AddressPlacement::Outside);
    let buffer = crate::answer::unheard(|| device.new_buffer_with_descriptor(descriptor))
        .map_err(|_| "its buffer was refused")?;
    if imports() == before {
        return Err("its pages were not imported by their pointer");
    }
    // First the one question no driver can fault on: other pages over the view, which have a
    // file behind them and can be pinned as the old ones were. A driver that kept the pages it
    // pinned at the import reads the old ones; one that gets them again from their mapping reads
    // these, and is refused here, before the fold below, whose `VM_PFNMAP` mapping such a driver
    // cannot get pages from (i915's userptr lost the device there when the fold came first).
    let other = vm::allocate_aliasable(length).map_err(|_| "no memory for other pages")?;
    let other = Unmapped {
        at: other.addr().to_u64(),
        length,
    };
    // SAFETY: the other pages' own mapping, read and write.
    unsafe { std::ptr::write_bytes(vm::ptr(other.at).cast::<u8>(), 0xe5, length.to_usize()) };
    let other_at = HostAddr::from_addr(other.at.to_usize());
    if vm.remap_shared_overwrite(view, other_at, length).is_err() {
        return Err("other pages were not mapped over its view");
    }
    let (out, copied) = check.run(device, &buffer, false)?;
    if (0..FOLD_PROBE_PAGES).any(|j| FoldCheck::ends(copied, j) == [0xe5; 2]) {
        return Err("its GPU read the pages mapped over the view since the import");
    }
    check.copied_in_view_order(copied)?;
    drop((out, other));
    if vm.remap_runs_shared_overwrite(view, &check.runs()).is_err() {
        return Err("its view was not mapped from its files again");
    }
    if !vm::fold_held(check.view..end) {
        return Err("its view was not mapped through a dma-buf");
    }
    let check = FoldCheck { round: 1, ..check };
    check.write();
    check.gpu(device, &buffer)
}

/// A mapping of the library's own, `length` bytes from `at`, unmapped as it drops.
#[derive(Debug)]
struct Unmapped {
    at: u64,
    length: u64,
}

impl Drop for Unmapped {
    fn drop(&mut self) {
        use crate::core::platform::os::vm::HostVm as _;
        let at = HostAddr::from_addr(self.at.to_usize());
        let _ = super::LinuxVm::new().deallocate(at, self.length);
    }
}

/// The pages of [`probe_fold`]'s view.
const FOLD_PROBE_PAGES: u64 = 64;

/// What [`probe_fold`] checks of an import of a view whose pages were mapped through a dma-buf
/// once it was made: a view at `view` of `pages` pages of the mapping at `own`, backwards (its
/// `j`th page is the mapping's `pages - 1 - j`th), the probe's `round`.
#[derive(Debug, Clone, Copy)]
struct FoldCheck {
    own: u64,
    view: u64,
    pages: u64,
    round: u64,
}

impl FoldCheck {
    /// The byte the GPU fills the first half of the view's pages with.
    const FILL: u8 = 0xc3;

    /// The byte written through the pages' own mapping into its `i`th page.
    fn written(&self, i: u64) -> u8 {
        ((i ^ self.round.wrapping_mul(31)) as u8) ^ 0x5a
    }

    /// The view's runs: the mapping's pages, backwards.
    fn runs(&self) -> Vec<(HostAddr, u64)> {
        let page = vm::page_size();
        (0..self.pages)
            .map(|i| {
                let at = self.own + (self.pages - 1 - i) * page;
                (HostAddr::from_addr(at.to_usize()), page)
            })
            .collect()
    }

    /// Writes [`FoldCheck::written`] through the pages' own mapping.
    fn write(&self) {
        let page = vm::page_size();
        for i in 0..self.pages {
            // SAFETY: the pages' own mapping, read and write while the check's buffer lives.
            unsafe {
                std::ptr::write_bytes(
                    vm::ptr(self.own + i * page).cast::<u8>(),
                    self.written(i),
                    page.to_usize(),
                );
            }
        }
    }

    /// The first and last bytes of the `j`th page at `first`.
    fn ends(first: u64, j: u64) -> [u8; 2] {
        let page = vm::page_size();
        let at = first + j * page;
        // SAFETY: the caller's pages, mapped while the check's buffer lives.
        unsafe { [0, page - 1].map(|o| vm::ptr(at + o).cast::<u8>().read_volatile()) }
    }

    /// One command buffer on `device` that copies `buffer`, over the view, into a buffer of the
    /// device's own, then fills the first half of `buffer` with [`FoldCheck::FILL`]; then whether
    /// the copy holds what was written through the pages' own mapping, in the view's order, and
    /// the fill shows through the pages' own mapping and through the view.
    fn gpu(&self, device: &rt::Device, buffer: &rt::Buffer) -> Result<(), &'static str> {
        self.gpu_seen(device, buffer, Some(self.view))
    }

    /// [`FoldCheck::gpu`], the view's bytes checked at `view_shows` where `Some`: other pages may
    /// be mapped over the view since.
    fn gpu_seen(
        &self,
        device: &rt::Device,
        buffer: &rt::Buffer,
        view_shows: Option<u64>,
    ) -> Result<(), &'static str> {
        let (_out, copied) = self.run(device, buffer, true)?;
        self.copied_in_view_order(copied)?;
        for j in 0..self.pages {
            let source = self.pages - 1 - j;
            let now = match j < self.pages / 2 {
                true => Self::FILL,
                false => self.written(source),
            };
            if Self::ends(self.own, source) != [now; 2]
                || view_shows.is_some_and(|view| Self::ends(view, j) != [now; 2])
            {
                return Err("its GPU's fill did not show through the pages' mappings");
            }
        }
        Ok(())
    }

    /// Whether the copy at `copied` holds what was written through the pages' own mapping, in
    /// the view's order.
    fn copied_in_view_order(&self, copied: u64) -> Result<(), &'static str> {
        for j in 0..self.pages {
            if Self::ends(copied, j) != [self.written(self.pages - 1 - j); 2] {
                return Err(
                    "its GPU did not read the bytes written through the pages' own mapping",
                );
            }
        }
        Ok(())
    }

    /// One command buffer on `device` that copies `buffer` into a buffer of the device's own, and
    /// where `fill`, then fills the first half of `buffer` with [`FoldCheck::FILL`], completed:
    /// the device's buffer, and the address of its contents.
    fn run(
        &self,
        device: &rt::Device,
        buffer: &rt::Buffer,
        fill: bool,
    ) -> Result<(rt::Buffer, u64), &'static str> {
        let length = self.pages * vm::page_size();
        let descriptor = rt::BufferDescriptor::new(
            length,
            rt::ResourceOptions::shared(),
            rt::BufferContents::Zero,
        )
        .with_placement(rt::AddressPlacement::Outside);
        let out = crate::answer::unheard(|| device.new_buffer_with_descriptor(descriptor))
            .map_err(|_| "no buffer of the device's own for its copy")?;
        let queue = device.new_command_queue().map_err(|_| "no command queue")?;
        let cb = queue.command_buffer().map_err(|_| "no command buffer")?;
        let mut blit = cb.blit_command_encoder().map_err(|_| "no blit encoder")?;
        blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)
            .and_then(|()| match fill {
                true => blit.fill_buffer(buffer, 0..length / 2, Self::FILL),
                false => Ok(()),
            })
            .and_then(|()| blit.end_encoding())
            .and_then(|()| cb.commit())
            .map_err(|_| "its commands were refused")?;
        if (cb.wait_until_completed_for(std::time::Duration::from_secs(10))).is_err() {
            return Err("its command buffer did not complete within 10 s");
        }
        if cb.status() != rt::CommandBufferStatus::Completed {
            return Err("its command buffer failed");
        }
        let copied = out.contents().ok_or("its copy has no contents")?;
        let copied = copied.as_ptr().addr().to_u64();
        Ok((out, copied))
    }
}

/// [`learn_view_imports`]' probe of `device`: a no-copy buffer over a view of `count` pages
/// backwards, mapped through dma-bufs, made as an import is, placed outside the guest's numbering
/// and released at once, its pages unmapped once the runtime freed its import ([`ProbePages`]).
/// Whether the device refused it, or made a copy of the view in its place as its import failed
/// (the runtime's `copied-host-pages`, which the probe's call hears unheard,
/// `crate::answer::unheard`), and whether it took it through the dma-bufs its source makes;
/// `None` where the view is not mapped through dma-bufs.
fn probe_view(device: &rt::Device, count: u64) -> Option<(bool, bool)> {
    use crate::core::platform::os::vm::HostVm as _;
    let vm = super::LinuxVm::new();
    let page = vm::page_size();
    let length = count * page;
    let (Ok(src), Ok(view)) = (vm::allocate_aliasable(length), vm.allocate_anywhere(length)) else {
        return None;
    };
    // Unmapped as they drop: once the runtime freed the probe's import, where it makes one.
    let probe = ProbePages::new(view.addr().to_u64(), src.addr().to_u64(), length);
    let runs: Vec<_> = (0..count)
        .map(|i| (src.wrapping_add((count - 1 - i) * page), page))
        .collect();
    let first = view.addr().to_u64();
    let through = vm.remap_runs_shared_overwrite(view, &runs).is_ok()
        && !files::through_pieces(first, first + length).is_empty();
    let at = address(first).filter(|_| through)?;
    let asked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // SAFETY: the view is mapped, read and write, and stays so until the runtime drops the
    // keep-alive, which owns it and its pages' file, once it freed the import.
    let memory = unsafe { host_memory_noting(at, Some(Arc::clone(&asked))) };
    let contents = rt::BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: Some(crate::gpu::backing::keep_alive(probe)),
    };
    let descriptor = rt::BufferDescriptor::new(length, rt::ResourceOptions::shared(), contents)
        .with_placement(rt::AddressPlacement::Outside);
    // A copy of the view in the import's place (`copied-host-pages`), as its import failed, is
    // the device's refusal too: the probe's own, which no one hears of.
    let copies = || device.work_counts().copied_host_buffers_made;
    let before = copies();
    let made = crate::answer::unheard(|| device.new_buffer_with_descriptor(descriptor));
    let refused = made.is_err() || copies() > before;
    drop(made);
    Some((refused, !refused && asked.load(Ordering::Relaxed)))
}

/// The pages of [`learn_view_imports`]' probe, its view and the file pages it shows, each
/// `length` bytes from its first byte: unmapped as it drops, which, where the probe's buffer was
/// made, is the runtime's drop of the buffer's keep-alive, once it freed the import (Vulkan
/// requires imported pages to stay mapped while the import lives, `crate::remap`'s module docs);
/// a free on another of the runtime's threads (its residency thread's) may come after the
/// buffer's release returned.
#[derive(Debug)]
struct ProbePages {
    view: u64,
    src: u64,
    length: u64,
}

impl ProbePages {
    fn new(view: u64, src: u64, length: u64) -> Self {
        #[cfg(test)]
        tests::PROBE_PAGES_LIVE.fetch_add(1, Ordering::Relaxed);
        Self { view, src, length }
    }
}

impl Drop for ProbePages {
    fn drop(&mut self) {
        use crate::core::platform::os::vm::HostVm as _;
        let vm = super::LinuxVm::new();
        for first in [self.view, self.src] {
            let _ = vm.deallocate(HostAddr::from_addr(first.to_usize()), self.length);
        }
        #[cfg(test)]
        tests::PROBE_PAGES_LIVE.fetch_sub(1, Ordering::Relaxed);
    }
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
    #[cfg(test)]
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

    /// The word that the runtime's import of the memory was made ([`ImportMade`]), which the
    /// caller gives once the runtime's call that imports it returned.
    pub(crate) fn made(&self) -> ImportMade {
        ImportMade(self.pinned.as_ref().map(|p| p.0))
    }

    /// Another import of the same memory while the view lives (`crate::gpu::backing`'s aliases,
    /// whose imports share one view): its pages counted as being made again, so no fold maps
    /// them through dma-bufs meanwhile, and what of them a fold mapped so mapped from their files
    /// again, as for the first ([`import_view`]); the word that it was made.
    pub(crate) fn again(&self) -> ImportMade {
        let Some(pinned) = &self.pinned else {
            return ImportMade(None);
        };
        let end = match PINNED.lock().get_mut(&pinned.0) {
            Some(record) => {
                record.making = record.making.saturating_add(1);
                record.end
            }
            None => return ImportMade(None),
        };
        // Pages left in views through dma-bufs for want of room are refused at the gate.
        let _ = vm::unfold(pinned.0.0, end);
        ImportMade(Some(pinned.0))
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
    use crate::core::platform::os::vm::{HostVm as _, VmProt};

    use super::super::LinuxVm;
    use super::*;

    const PAGE: u64 = 0x1000;

    /// The device-making probes' pages not yet unmapped ([`ProbePages`]): the runtime frees a
    /// probe's import, and so drops its pages, on a thread of its own, after the device is made.
    pub(super) static PROBE_PAGES_LIVE: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    /// Waits up to 10 s for the probes' pages to go, so a test that counts the views' mappings
    /// counts its own alone: on the RTX 2080 two of the probes' mappings were still there as the
    /// test began, and gone by its end. Says how many are left where any are.
    fn probes_gone() {
        for _ in 0..10_000 {
            if PROBE_PAGES_LIVE.load(Ordering::Relaxed) == 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        eprintln!(
            "{} of the probes' pages still mapped after 10 s",
            PROBE_PAGES_LIVE.load(Ordering::Relaxed)
        );
    }

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
        let second = crate::core::host::HostPtr::from_ptr(second.as_ptr());
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
        use crate::core::platform::types::ResourceAddressRange;

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
        use crate::core::platform::types::ResourceAddressRange;
        use crate::core::raise::death::DeathKind;
        use crate::core::testkit::capture::catch_death;

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
        use crate::core::raise::death::DeathKind;
        use crate::core::testkit::capture::catch_death;
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

    /// A view of `n` pages of a fresh range a view can be made of, backwards, mapped from their
    /// files where a device of the process needs imported pages so: the range and the view.
    fn backwards(vm: &LinuxVm, n: u64) -> (HostAddr, HostAddr) {
        let src = vm::allocate_aliasable(n * PAGE).unwrap();
        let view = vm.allocate_anywhere(n * PAGE).unwrap();
        let runs: Vec<_> = (0..n)
            .map(|i| (src.wrapping_add((n - 1 - i) * PAGE), PAGE))
            .collect();
        vm.remap_runs_shared_overwrite(view, &runs).unwrap();
        (src, view)
    }

    /// An import's view is folded into a dma-buf as a view no import holds is once the import is
    /// made on a device that keeps what it pinned at the import, and not while it is being made;
    /// another import of the same view (an alias's, `crate::gpu::backing`) maps it from its files
    /// again and keeps it so until that import is made too. The flag is the process's, which
    /// nextest runs this test alone in.
    #[test]
    fn a_made_import_s_view_folds_and_another_maps_it_from_its_files_again() {
        if !super::super::dma_buf::opens() {
            return;
        }
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        let vm = LinuxVm::new();
        let n = 32;
        let (src, view) = backwards(&vm, n);
        let p = NonNull::new(view.as_ptr::<u8>()).unwrap();
        let (first, end) = (addr(p), addr(p) + n * PAGE);
        let maps = || vm::tests::kernel_mappings(first, end);
        // Folded: its pages lie in a dma-buf, a mapping or two (as `super::vm`'s fold test
        // counts them); else a mapping a page from their files.
        let folded = || !files::through_pieces(first, end).is_empty() && maps() <= 2;
        let from_files = || files::through_pieces(first, end).is_empty() && maps() == n;
        assert!(from_files(), "mapped from its files");
        let import = import_view(p, n * PAGE, false);
        vm::fold_idle();
        assert!(from_files(), "being made: not folded");
        import.made().made_here();
        vm::fold_idle();
        assert!(folded(), "made: folded ({} mappings)", maps());
        let again = import.again();
        assert!(from_files(), "another import: mapped from its files again");
        vm::fold_idle();
        assert!(from_files(), "the other being made: not folded");
        again.made_here();
        vm::fold_idle();
        assert!(folded(), "both made: folded ({} mappings)", maps());
        drop(import);
        IMPORTS_NEED_FILES.store(false, Ordering::Relaxed);
        assert!(vm.deallocate(view, n * PAGE).is_success());
        assert!(vm.deallocate(src, n * PAGE).is_success());
    }

    /// The test device, its checks settled, where its GPU's imports of a pointer are folded once
    /// made: it passed the probe ([`learn_folds`]) and does not release idle pointer imports.
    /// Where they are not, said on stderr, the fold is checked off: a made import's view mapped
    /// from its files stays so through a fold ([`fold_stays_off`]), and `None`.
    fn folding_device(test: &str) -> Option<crate::gpu::device::VulkanDevice> {
        let d = crate::gpu::device::tests::dev();
        crate::gpu::device::tests::settle_device_checks(&d);
        probes_gone();
        let (passed, releases) = (
            folds_after_import(&d.raw),
            d.raw.releases_idle_pointer_imports(),
        );
        if imports_need_files() && passed && !releases {
            return Some(d);
        }
        eprintln!(
            "{test}: {}: its imports' views are not folded once made (needs files {}, probe \
             passed {passed}, releases idle pointer imports {releases}): the fold checked off",
            d.raw.properties().name,
            imports_need_files(),
        );
        fold_stays_off(&d);
        None
    }

    /// On a device whose imports' views are not folded once made, a no-copy buffer's view mapped
    /// from its files, made, stays so through a fold, where the device needs imported pages so;
    /// elsewhere views are mapped through dma-bufs, and nothing is folded or unfolded for an
    /// import.
    fn fold_stays_off(d: &crate::gpu::device::VulkanDevice) {
        if !imports_need_files() {
            return;
        }
        let vm = LinuxVm::new();
        let n = 64;
        let (src, view) = backwards(&vm, n);
        let (first, end) = (view.addr().to_u64(), view.addr().to_u64() + n * PAGE);
        let (buffer, gone) = no_copy(d, view, n);
        vm::fold_idle();
        assert_eq!(
            vm::tests::kernel_mappings(first, end),
            n,
            "a made import's view stays mapped from its files"
        );
        released(buffer, &gone, first, end);
        for range in [view, src] {
            assert!(vm.deallocate(range, n * PAGE).is_success());
        }
    }

    /// A kept fold verdict answers under its own key alone, whatever it says, and nothing where
    /// the file is missing or of another format; a refusal's reason comes back on its line.
    #[test]
    fn a_kept_fold_verdict_answers_under_its_key_alone() {
        let dir = std::env::temp_dir().join(format!("fold-verdict-{}", std::process::id()));
        let file = dir.join("device-checks").join("fold-after-import-00");
        assert_eq!(kept_fold(&file, "k"), None, "missing");
        for answer in [Ok(()), Err("its GPU read the old\npages".to_owned())] {
            keep_fold(&file, "k", &answer).unwrap();
            let back = answer.map_err(|why| why.replace('\n', " "));
            assert_eq!(kept_fold(&file, "k"), Some(back));
            assert_eq!(kept_fold(&file, "other"), None, "another key");
        }
        std::fs::write(&file, "reims-vgpu fold after import 0\nkey k\npassed\n").unwrap();
        assert_eq!(kept_fold(&file, "k"), None, "another format");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What of the probe's refusals is the GPU's own answer, kept for its driver.
    #[test]
    fn only_the_gpus_own_answers_are_kept() {
        assert!(lasting(
            "its GPU read the pages mapped over the view since the import"
        ));
        assert!(lasting("its command buffer failed"));
        assert!(lasting("its pages were not imported by their pointer"));
        assert!(!lasting("no memory for its pages"));
        assert!(!lasting("its view was not mapped through a dma-buf"));
        assert!(!lasting("no command queue"));
    }

    /// The fold probe answers alike ten times over on the test device, as the GPU's verdict
    /// ([`learn_folds`]) the adopted device holds, and says on stderr what it answered and what
    /// it cost; not run where views are mapped through dma-bufs, where nothing calls it.
    #[test]
    fn the_fold_probe_answers_alike_ten_times() {
        use crate::core::platform::os::Clock as _;
        let d = crate::gpu::device::tests::dev();
        let name = &d.raw.properties().name;
        if !imports_need_files() {
            eprintln!("the fold probe on {name}: not run, views are mapped through dma-bufs");
            return;
        }
        let clock = crate::os::OsClock::new();
        let mut took = Vec::new();
        let mut answers = Vec::new();
        for _ in 0..10 {
            let start = clock.now_mono();
            answers.push(probe_fold(&d.raw));
            took.push(clock.now_mono().as_nanos() - start.as_nanos());
        }
        took.sort_unstable();
        eprintln!(
            "the fold probe on {name}: {:?}, at the device's making {:.2} ms, then median {:.2} \
             ms, longest {:.2} ms",
            answers[0],
            PROBE_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
            took[5] as f64 / 1e6,
            took[9] as f64 / 1e6
        );
        assert!(answers.iter().all(|a| *a == answers[0]), "{answers:?}");
        assert_eq!(
            answers[0].is_ok(),
            folds_after_import(&d.raw),
            "the adopted device's verdict"
        );
    }

    /// A no-copy buffer of the platform's over the `n` pages of the view at `view`, and the
    /// receiver its deallocator answers on.
    fn no_copy(
        d: &crate::gpu::device::VulkanDevice,
        view: HostAddr,
        n: u64,
    ) -> (
        crate::gpu::resource::VulkanBuffer,
        std::sync::mpsc::Receiver<()>,
    ) {
        use crate::core::platform::device::GpuDevice as _;
        let (tx, rx) = std::sync::mpsc::channel();
        let deallocator: crate::core::platform::device::NoCopyDeallocator = Box::new(move || {
            let _ = tx.send(());
        });
        // SAFETY: the view stays mapped until the buffer's deallocator ran and its import was
        // freed (its record gone, [`released`]); a fold or remap maps other mappings over it in
        // one call, never leaving it unmapped.
        let range = unsafe { crate::core::platform::HostRange::new(view, n * PAGE) };
        let buffer = d
            .new_buffer_over_host_memory(
                crate::core::raise::OnFault::Return,
                range,
                0,
                Some(deallocator),
            )
            .unwrap()
            .unwrap();
        (buffer, rx)
    }

    /// `buffer` released: its deallocator ran, and its import was freed, which drops its record
    /// of [`PINNED`], within 10 s.
    fn released(
        buffer: crate::gpu::resource::VulkanBuffer,
        gone: &std::sync::mpsc::Receiver<()>,
        first: u64,
        end: u64,
    ) {
        drop(buffer);
        let wait = std::time::Duration::from_secs(10);
        assert_eq!(gone.recv_timeout(wait), Ok(()), "the deallocator ran");
        let held =
            || (PINNED.lock().iter()).any(|(&(at, _), record)| at < end && first < record.end);
        for _ in 0..wait.as_millis() {
            if !held() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!held(), "the import was freed");
    }

    /// The process's mappings, as the kernel lists them.
    fn process_mappings() -> usize {
        std::fs::read_to_string("/proc/self/maps")
            .unwrap()
            .lines()
            .count()
    }

    /// On a device whose imports' views are folded once made ([`folding_device`], NVIDIA's; on
    /// others the fold is checked off), a no-copy buffer's view made 100 times over pages of a
    /// file of their own, folded into a dma-buf once made: the GPU
    /// reads the bytes written through the pages' own mapping after the fold and its fill shows
    /// through both mappings, and the buffer's release gives everything back: the views' mappings,
    /// their records, the process's mappings, and the pages, which the kernel then lets the file
    /// be sealed against writes, as it refuses while a page of it is pinned
    /// (`memfd_wait_for_pins`).
    #[test]
    fn a_folded_import_gives_back_its_pages_and_mappings() {
        let Some(d) = folding_device("a_folded_import_gives_back_its_pages_and_mappings") else {
            return;
        };
        let vm = LinuxVm::new();
        let n = 256;
        let before = (files::mappings(), PINNED.lock().len(), process_mappings());
        for round in 0..100 {
            let (src, view) = backwards(&vm, n);
            let check = FoldCheck {
                own: src.addr().to_u64(),
                view: view.addr().to_u64(),
                pages: n,
                round,
            };
            let (first, end) = (check.view, check.view + n * PAGE);
            let file = Arc::clone(&files::pieces(check.own, n * PAGE).unwrap()[0].backing.fd);
            assert!(
                files::through_pieces(first, end).is_empty(),
                "mapped from its files"
            );
            let (buffer, gone) = no_copy(&d, view, n);
            vm::fold_idle();
            assert!(
                !files::through_pieces(first, end).is_empty()
                    && vm::tests::kernel_mappings(first, end) <= 2,
                "round {round}: folded"
            );
            check.write();
            assert_eq!(check.gpu(&d.raw, &buffer.raw), Ok(()), "round {round}");
            released(buffer, &gone, first, end);
            assert!(vm.deallocate(view, n * PAGE).is_success());
            assert!(vm.deallocate(src, n * PAGE).is_success());
            use std::os::fd::AsRawFd as _;
            // SAFETY: a seal of a file of the test's own.
            let sealed =
                unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_WRITE) };
            assert_eq!(
                sealed,
                0,
                "round {round}: no page of the file pinned ({})",
                std::io::Error::last_os_error()
            );
        }
        assert_eq!(files::mappings(), before.0, "the views' mappings");
        assert_eq!(PINNED.lock().len(), before.1, "the records");
        let now = process_mappings();
        assert!(
            now <= before.2 + 16,
            "the process's mappings: {} then {now}",
            before.2
        );
    }

    /// On a device whose imports' views are folded once made ([`folding_device`], NVIDIA's; on
    /// others the fold is checked off), a no-copy buffer whose view was folded into a dma-buf once
    /// made keeps the pages it was made over when other pages are
    /// mapped over its range, untold, as an import of a view mapped from its files keeps them
    /// (`tests/no_copy_remap_linux.rs`): the GPU reads the old pages and fills them, and the range
    /// shows the new pages, untouched.
    #[test]
    fn a_folded_import_keeps_its_pages_through_a_remap() {
        let Some(d) = folding_device("a_folded_import_keeps_its_pages_through_a_remap") else {
            return;
        };
        let vm = LinuxVm::new();
        let n = 64;
        let (src, view) = backwards(&vm, n);
        let check = FoldCheck {
            own: src.addr().to_u64(),
            view: view.addr().to_u64(),
            pages: n,
            round: 7,
        };
        let (first, end) = (check.view, check.view + n * PAGE);
        let (buffer, gone) = no_copy(&d, view, n);
        vm::fold_idle();
        assert!(
            !files::through_pieces(first, end).is_empty()
                && vm::tests::kernel_mappings(first, end) <= 2,
            "folded"
        );
        check.write();
        let new = vm::allocate_aliasable(n * PAGE).unwrap();
        // SAFETY: the test's own fresh mapping, read and write.
        unsafe { std::ptr::write_bytes(new.as_ptr::<u8>(), 0xe5, (n * PAGE).to_usize()) };
        vm.remap_shared_overwrite(view, new, n * PAGE).unwrap();
        assert_eq!(read(first), 0xe5, "the CPU sees the new pages");
        assert_eq!(
            check.gpu_seen(&d.raw, &buffer.raw, None),
            Ok(()),
            "the old pages"
        );
        for at in [
            first,
            end - 1,
            addr(NonNull::new(new.as_ptr::<u8>()).unwrap()),
        ] {
            assert_eq!(read(at), 0xe5, "the new pages untouched");
        }
        released(buffer, &gone, first, end);
        for range in [view, src, new] {
            assert!(vm.deallocate(range, n * PAGE).is_success());
        }
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
