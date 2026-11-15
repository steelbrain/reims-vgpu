//! What the runtime imports for host memory on Linux: the memory itself.
//!
//! Linux drivers pin imported pages for writing, so the runtime imports no read-only memory
//! (typed, `HostMemory`), where Metal reads it: the guest's ranges core maps read-only and
//! aliases of a read-only heap. For those, and a no-copy buffer or heap over pages mapped
//! read-only, the runtime is given the pages themselves: the views' gate refuses them
//! ([`import_gate`]), and the runtime makes the buffer or heap over a copy of them whose GPU
//! writes it drops, as Metal's GPU drops them (its `copied-host-pages`, a kind of
//! `crate::answer::DIES`), so the GPU reads what Metal's reads and its writes reach nothing past
//! the copy ([`ImportView::read_only`]); so does an object made with `MTLResourceOptions` bit 24,
//! whatever its pages. What Metal answers otherwise than a dropped write there (a pass's store
//! into such memory, a texture's writes) is `crate::gpu::writes`'s.
//!
//! Every import also carries a dma-buf source ([`host_memory`]): a device that refuses the pages
//! as host memory, as RADV refuses any page with a file behind it, imports a dma-buf of the same
//! pages instead (`super::dma_buf`). The runtime asks for it only where it would refuse the
//! import otherwise, so every device that imports host memory imports as it did. A buffer's or a
//! heap's request over a view in a group dma-buf is answered with the view's piece of the group's
//! file (`super::dma_buf`'s module docs, "Groups"), where that piece's offset is a multiple of the
//! alignment the request asks; one over the guest's RAM with pieces of the RAM's region dma-bufs,
//! where they fit the request ("Regions"); a texture's and any other request with a dma-buf of
//! its own, as before.
//!
//! And every import carries the views' gate ([`import_gate`]), which the runtime asks before it
//! imports a no-copy buffer's or heap's pages, or a buffer's new pages, by their pointer: where
//! the pages were left in views through dma-bufs for want of mappings, which a device that pins a
//! pointer's pages refuses, or are read-only, it refuses, and the runtime makes the buffer or heap
//! over a copy of them in memory of the device's own instead, which takes no mapping (its
//! `copied-host-pages`, a kind of `crate::answer::DIES`), and whose GPU writes it drops where the
//! process cannot write the pages.
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
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::num::{ToU64, ToUsize};
use crate::core::platform::os::vm::HostAddr;
use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

use super::budget;
use super::vm;
use crate::os::files;

/// Whether the runtime makes read-only memory, and memory made with `MTLResourceOptions` bit 24, over a copy of it whose GPU writes it drops, where Metal's GPU
/// keeps them from the memory itself: on Linux, where the runtime's device answers no to
/// `drops_gpu_writes_to_read_only_memory` (module docs).
pub(crate) const COPIES_READ_ONLY: bool = true;

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
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
/// ([`Pinned`]), with whether its import is still being made. A fold of the views maps none of
/// them through dma-bufs (`super::vm::fold_idle`): a device took them by their mappings. Where
/// the devices keep what they pinned as the import was made ([`folds_after_import`]), a record
/// whose import is made is left out ([`pinned`]).
static PINNED: Mutex<BTreeMap<(u64, u64), Record>> = Mutex::new(BTreeMap::new());

/// One import's pages in [`PINNED`]: their end, and whether the runtime is still making the
/// import ([`ImportMade`]).
#[derive(Debug, Clone, Copy)]
struct Record {
    end: u64,
    making: bool,
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
        PINNED.lock().insert(key, Record { end, making: true });
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
    /// and its pages may be folded.
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

/// The import of the record `key` of [`PINNED`] made.
fn made_now(key: (u64, u64)) {
    if let Some(record) = PINNED.lock().get_mut(&key) {
        record.making = false;
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
        .filter(|(_, record)| record.making)
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

/// Whether the device that needs imported pages mapped from their files ([`IMPORTS_NEED_FILES`])
/// takes a view of pages one dma-buf holds through that dma-buf, which its pointer's refusal
/// leaves the runtime to import ([`learn_view_imports`]): so the gate lets such pages through
/// though they lie in views through dma-bufs ([`import_gate`]), and they take no mapping of their
/// own. NVIDIA's driver imports a dma-buf of scattered guest pages as an allocation dedicated to
/// its buffer, one dma-buf a buffer (the runtime's `DmaBufFor::Buffer`), and refuses several
/// under one buffer (no type its dma-bufs import as takes a sparse buffer): so it imports one
/// udmabuf's pages, at most `list_limit` runs (1,024 by default, 4 MiB of scattered 4 KiB pages)
/// and `size_limit_mb` MiB, and no more.
static ONE_DMA_BUF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`ONE_DMA_BUF`].
fn one_dma_buf_imports() -> bool {
    ONE_DMA_BUF.load(Ordering::Relaxed)
}

/// Whether a device of the process took a view of guest pages through the dma-bufs its source
/// made ([`learn_view_imports`]), as RADV does, which imports no page with a file behind it by
/// pointer ([`groups_views`]).
static VIEWS_BY_SOURCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the views no-copy buffers lie over are put in group dma-bufs
/// (`crate::core::platform::os::vm::HostVm::group_views`, `super::dma_buf`'s module docs,
/// "Groups"): a device of the process takes views through its source's dma-bufs, so its import of
/// a grouped view's pages is a piece of the group's file ([`host_memory`]); none needs imported
/// pages mapped from their files, whose imports are made by pointer, or of one dma-buf dedicated
/// to its buffer, which no shared file serves; and udmabuf is the process's.
pub(super) fn groups_views() -> bool {
    VIEWS_BY_SOURCE.load(Ordering::Relaxed) && !imports_need_files() && super::dma_buf::opens()
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

/// The views' gate over the runtime's import of a no-copy buffer's or heap's pages, by their
/// pointer (`rt::HostMemory::with_import_gate`), which every [`host_memory`] carries, asked on
/// the thread that makes the buffer or heap, with no lock of the
/// runtime's held, after [`import_view`] made what it imports. It refuses read-only pages, so the
/// runtime copies them, dropping the GPU's writes; and, where a device of the process needs imported
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
    // Pages one dma-buf holds, on devices that take such a dma-buf in place of the pointer: the
    // runtime imports them through it, which takes no mapping ([`ONE_DMA_BUF`]).
    if one_dma_buf_imports() && super::dma_buf::fits_one(first, end - first).is_ok() {
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
/// views"). There a second probe, of a view of as many pages as one dma-buf takes, learns whether
/// the device takes the pages one dma-buf holds through it ([`ONE_DMA_BUF`]), as NVIDIA's driver
/// does, dedicated to the buffer. Either way the device is handed the guest's RAM then
/// (`crate::os`'s `host_ram`). Nothing is learned where the view is not mapped through dma-bufs (udmabuf
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
    if through_dma_bufs {
        VIEWS_BY_SOURCE.store(true, Ordering::Relaxed);
    }
    if !through_dma_bufs {
        // The pages one dma-buf holds, as many runs as it takes, within its bytes.
        let one = (limits.pieces)
            .min(limits.bytes / vm::page_size())
            .min(1 << 16);
        let taken = matches!(probe_view(device, one), Some((false, true)));
        let why = dma_buf_refusal();
        log::info!(
            target: "reims_vgpu_vulkan::os",
            "{name}: an import of a view of guest pages one dma-buf holds {}",
            match taken {
                true => "is taken through it, so such imports take no mapping",
                false => "is refused too, so their pages are mapped from their files as well",
            }
        );
        if let Some(why) = why {
            log::debug!(target: "reims_vgpu_vulkan::os", "the view's dma-buf source: {why}");
        }
        ONE_DMA_BUF.store(taken, Ordering::Relaxed);
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        learn_folds(device);
        budget::warn_if_imports_short();
    }
    // Either way the device copies by runs of the guest's RAM, and binds objects over it in the
    // RAM's chunks it imported (`crate::os`'s `host_ram`).
    crate::os::host_ram::hand_ram_to(device);
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
    let import = import_view(at, length);
    let imported_at = import.at();
    let (_, kept) = import.release(None);
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

// SAFETY: `at` is an address only, which nothing here reads through: it is handed to the runtime.
unsafe impl Send for ImportView {}

// SAFETY: as for `Send`: a shared view only answers where it is.
unsafe impl Sync for ImportView {}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the table shows a page of the memory mapped read-only: the runtime is given the
    /// pages themselves, which the gate refuses ([`import_gate`]), so the runtime makes the object
    /// over a copy of them whose GPU writes it drops (module docs).
    pub(crate) fn read_only(&self) -> bool {
        self.read_only
    }

    /// Whether the runtime may be told the import's pages are writable ([`host_memory_vouched`]):
    /// every page of the memory the table shows mapped with write permission now, as the
    /// library's own views and the guest's RAM are. Where it does not know a page (memory of the
    /// library's own), or a page is read-only, the runtime asks the kernel as before.
    pub(crate) fn vouched(&self) -> bool {
        self.vouched
    }

    /// The word that the runtime's import of the memory was made ([`ImportMade`]), which the
    /// caller gives once the runtime's call that imports it returned.
    pub(crate) fn made(&self) -> ImportMade {
        ImportMade(self.pinned.as_ref().map(|p| p.0))
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves: `inner` as it is, and the record of the pages a device may pin
    /// ([`Pinned`]), where there is one.
    pub(crate) fn release(
        self,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let pinned = self.pinned.map(|p| -> rt::KeepAlive { Box::new(p) });
        (inner, pinned)
    }
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]), with a
/// source of dma-bufs of its pages for a device that refuses them as host memory, and the views'
/// gate ([`import_gate`]) (module docs).
/// The runtime keeps the source while the import lives, and calls it with no lock of its own
/// held, with a window of the import's pages and the granule it binds chunks at: the whole
/// import's when it makes it, and a texture's own pages, at a granule of their length, when it
/// makes a texture over it. The source answers dma-bufs that are the window's pages in order, each
/// within udmabuf's limits (`super::dma_buf`), which the runtime imports. A call that makes none keeps its refusal on its thread for the death there ([`dma_buf_refusal`]), as
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
    // The region dma-bufs the source answered with, held while the runtime keeps the source (with
    // the imports it made of them): the runtime takes a chunk at offset 0 whose `Arc` it alone holds
    // once the source returns as a dma-buf of its own, and a region's file, which the table of
    // regions holds weakly, would be that at its first piece from its first page.
    let answered: Mutex<Vec<Arc<std::os::fd::OwnedFd>>> = Mutex::new(Vec::new());
    let source: rt::DmaBufSource = Arc::new(move |request: rt::DmaBufRequest| {
        if let Some(asked) = &asked {
            asked.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let first = addr_of(request.start);
        let (length, granule) = (request.length, request.granule);
        // A buffer's or a heap's import of a grouped view's pages: a piece of the group's file
        // (`super::dma_buf`'s module docs, "Groups"); else, of the guest's RAM, pieces of its
        // regions' files ("Regions").
        if request.shared_files {
            let alignment = request.alignment;
            super::dma_buf::asked_alignment(alignment);
            if let Some((file, offset)) = super::dma_buf::piece_of(first, length, alignment) {
                log::debug!(
                    "{length:#x} bytes at {first:#x} as the piece at {offset:#x} of a group dma-buf"
                );
                return Ok(vec![rt::DmaBufChunk {
                    file,
                    offset,
                    length,
                }]);
            }
            if let Some(pieces) = super::dma_buf::ram_pieces(first, length, granule, alignment) {
                log::debug!(
                    "{length:#x} bytes at {first:#x} as {} pieces of the guest RAM's region \
                     dma-bufs",
                    pieces.len()
                );
                let mut held = answered.lock();
                for piece in &pieces {
                    if !held.iter().any(|f| Arc::ptr_eq(f, &piece.file)) {
                        held.push(Arc::clone(&piece.file));
                    }
                }
                drop(held);
                return Ok(pieces);
            }
        }
        match super::dma_buf::of_pages(first, length, granule) {
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
        std::sync::LazyLock::new(|| Arc::new(crate::os::host_ram::runs));
    // SAFETY: the caller's.
    let memory =
        unsafe { rt::HostMemory::with_dma_buf(at, source) }.with_import_gate(Arc::clone(&GATE));
    // The runs of the guest's RAM the pages are (`crate::os`'s `host_ram`), which the runtime asks of the
    // import's window, on every device: where the device imports the RAM's chunks for its objects,
    // the object lies in its chunk; where it lists its imports in every submission, in a chunk a
    // copy imported, and else in an import of its own, as with no runs.
    memory.with_runs(Arc::clone(&RUNS))
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

/// The memory the runtime imports for `length` bytes at `at`: those bytes, where every page of
/// them is mapped writable or one has no file behind it (which the runtime then refuses as it
/// refuses any import there), and where one is mapped read-only, which the gate refuses, so the
/// runtime copies them, dropping the GPU's writes (module docs). Where a device needs imported
/// pages mapped from their files, what of writable bytes lies in views through dma-bufs is mapped
/// from them, short of the headroom of the views' budget, past which the gate refuses the pages
/// and the runtime copies them ([`import_gate`]): the runtime asks the gate about every import, a
/// no-copy buffer's or heap's.
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView {
        at,
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
    ImportView {
        read_only: true,
        ..plain
    }
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

    /// Read-only memory is given as it is, read-only, and the gate refuses its pages, so the
    /// runtime copies them and drops the GPU's writes; writable memory is not read-only
    #[test]
    fn read_only_memory_is_refused_at_the_gate() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first).unwrap();
        let view = import_view(p, PAGE);
        assert_eq!((view.at(), view.read_only()), (p, false), "writable");
        assert!(gate(first, first + PAGE));
        drop(view);
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let view = import_view(p, PAGE);
        assert_eq!(view.at(), p, "the pages themselves");
        assert!(view.read_only());
        assert!(!view.vouched());
        assert!(!gate(first, first + PAGE), "refused, so copied");
        let (run, kept) = view.release(None);
        assert!(run.is_none() && kept.is_none());
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// The runtime may be told an import's pages are writable only where the table shows every
    /// one mapped writable as it is made: a range a view can be made of is vouched for; once one
    /// of its pages is protected read-only, the next import over it is read-only and is not; and
    /// memory the table does not know (no file behind it) is not either
    #[test]
    fn only_pages_the_table_shows_writable_are_vouched_for() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first).unwrap();
        let view = import_view(p, 2 * PAGE);
        assert_eq!(view.at(), p);
        assert!(view.vouched(), "every page writable");
        drop(view);
        let second = address(first + PAGE).unwrap();
        let second = crate::core::host::HostPtr::from_ptr(second.as_ptr());
        assert!(vm.protect_max(second, PAGE, VmProt::READ).is_ok());
        let view = import_view(p, 2 * PAGE);
        assert!(view.read_only(), "a read-only page");
        assert!(!view.vouched(), "a page read-only since the first import");
        drop(view);
        let view = import_view(p, PAGE);
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
        let view = import_view(NonNull::new(anon.cast::<u8>()).unwrap(), PAGE);
        assert!(!view.vouched(), "memory the table does not know");
        drop(view);
        // SAFETY: the mapping made above.
        assert_eq!(unsafe { libc::munmap(anon, PAGE.to_usize()) }, 0);
    }

    /// Memory with no file behind it is given as it is, for the runtime to refuse
    #[test]
    fn memory_without_a_file_is_imported_as_it_is() {
        let vm = LinuxVm::new();
        let at = vm.allocate_anywhere(PAGE).unwrap();
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let p = NonNull::new(at.as_ptr::<u8>()).unwrap();
        assert_eq!(import_view(p, PAGE).at(), p);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// Where a device of the process needs imported pages mapped from their files, a view that
    /// from its files would come within the headroom of the views' budget is mapped through a
    /// dma-buf, though the budget has room for it. An import of it (a buffer's or a heap's)
    /// leaves it so, and the gate refuses its pages, counted with their bytes, in strict
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
            let import = import_view(p, n * PAGE);
            assert_eq!((import.at(), mappings()), (p, 1), "an import's, left");
            let (copied, copied_bytes) = counted(&COPIED);
            assert!(!gate(first, end), "refused at the gate");
            assert_eq!(counted(&COPIED), (copied + 1, copied_bytes + n * PAGE));
            drop(import);
            // Room short of the headroom.
            budget::set_test_limit(Some(limit(budget::HEADROOM + 2 * n)));
            let import = import_view(p, n * PAGE);
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

    /// Where the device that needs imported pages mapped from their files takes the pages one
    /// dma-buf holds through it, an import of a view left in a dma-buf past the headroom of the
    /// views' budget is let through the gate, uncounted, and maps nothing; one of more runs than
    /// one dma-buf takes is still refused, counted. The flags are the process's, which nextest
    /// runs this test alone in.
    #[test]
    fn past_the_headroom_one_dma_buf_s_pages_pass_the_gate() {
        let through = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .is_ok();
        if !through {
            return;
        }
        IMPORTS_NEED_FILES.store(true, Ordering::Relaxed);
        ONE_DMA_BUF.store(true, Ordering::Relaxed);
        let vm = LinuxVm::new();
        // A view of `n` runs backwards, past the headroom: through dma-bufs.
        let view_of = |n: u64| {
            let src = vm::allocate_aliasable(n * PAGE).unwrap();
            let view = vm.allocate_anywhere(n * PAGE).unwrap();
            let runs: Vec<_> = (0..n)
                .map(|i| (src.wrapping_add((n - 1 - i) * PAGE), PAGE))
                .collect();
            vm.remap_runs_shared_overwrite(view, &runs).unwrap();
            (src, view)
        };
        budget::set_test_limit(Some(
            budget::held() + budget::RESERVE + budget::HEADROOM + 8,
        ));
        let copied = || COPIED.count.load(Ordering::Relaxed);
        let before = copied();
        let one = super::super::dma_buf::limits().pieces.min(32);
        let (src, view) = view_of(one);
        let p = NonNull::new(view.as_ptr::<u8>()).unwrap();
        let (first, end) = (addr(p), addr(p) + one * PAGE);
        let import = import_view(p, one * PAGE);
        assert_eq!(import.at(), p, "the view itself");
        assert_eq!(
            vm::tests::kernel_mappings(first, end),
            1,
            "left in its dma-buf"
        );
        assert!(gate(first, end), "let through");
        assert_eq!(copied(), before, "uncounted");
        drop(import);
        assert!(vm.deallocate(view, one * PAGE).is_success());
        assert!(vm.deallocate(src, one * PAGE).is_success());
        let more = super::super::dma_buf::limits().pieces + 1;
        let (src, view) = view_of(more);
        let p = NonNull::new(view.as_ptr::<u8>()).unwrap();
        let (first, end) = (addr(p), addr(p) + more * PAGE);
        let import = import_view(p, more * PAGE);
        assert!(!gate(first, end), "more than one dma-buf takes: refused");
        assert_eq!(copied(), before + 1, "counted");
        drop(import);
        budget::set_test_limit(None);
        ONE_DMA_BUF.store(false, Ordering::Relaxed);
        assert!(vm.deallocate(view, more * PAGE).is_success());
        assert!(vm.deallocate(src, more * PAGE).is_success());
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
    /// made on a device that keeps what it pinned at the import, and not while it is being made.
    /// The flag is the process's, which nextest runs this test alone in.
    #[test]
    fn a_made_import_s_view_folds_once_made() {
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
        let import = import_view(p, n * PAGE);
        vm::fold_idle();
        assert!(from_files(), "being made: not folded");
        import.made().made_here();
        vm::fold_idle();
        assert!(folded(), "made: folded ({} mappings)", maps());
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
            let file = Arc::clone(&files::pieces(check.own, n * PAGE).unwrap()[0].backing.file);
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
        let whole = import_view(p, n * PAGE);
        assert_eq!((whole.at(), mappings(first, end)), (p, n), "imported");
        assert_eq!(read(first + 7), 0x77);
        drop(whole);
        let again = import_view(address(first + 4 * PAGE).unwrap(), 2 * PAGE);
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

    /// Where views are imported through their dma-bufs and udmabuf opens ([`groups_views`],
    /// RADV's), the platform's VM (`crate::remap::VulkanVm`, which the library's devices are
    /// given) answers a shared file's limits, and no-copy buffers over views it puts into one
    /// group dma-buf (`super::dma_buf`'s module docs, "Groups") are imported as pieces of its
    /// file: the device imports the file at the first piece, and the others join that import; the
    /// GPU reads each view's own pages, in the view's order, and its fill shows through them; once
    /// the buffers go, so does the file's import. Elsewhere, said on stderr, nothing is checked.
    #[test]
    fn grouped_views_are_imported_as_pieces_of_one_file() {
        let d = crate::gpu::device::tests::dev();
        crate::gpu::device::tests::settle_device_checks(&d);
        probes_gone();
        if !groups_views() {
            eprintln!(
                "skipped: {} takes no view through its dma-bufs here, or udmabuf does not open",
                d.raw.properties().name
            );
            return;
        }
        // The members at offsets every request's alignment divides.
        super::super::dma_buf::asked_alignment(1 << 16);
        let vm = LinuxVm::new();
        // Pages of each view; 32 pages are more than the runtime's 64 KiB granule.
        let made: Vec<(HostAddr, HostAddr, u64)> = [4, 32, 4]
            .into_iter()
            .map(|n| {
                let (src, view) = backwards(&vm, n);
                (src, view, n)
            })
            .collect();
        let platform_vm = crate::remap::VulkanVm::new();
        let limits = platform_vm.view_groups().expect("a shared file's limits");
        assert!(limits.bytes <= super::super::dma_buf::GROUP_BYTES && limits.runs > 0);
        let views: Vec<(HostAddr, u64)> = (made.iter())
            .map(|&(_, view, n)| (view, n * PAGE))
            .collect();
        let groups = platform_vm.group_views(&views);
        let (files, members) = (groups.making, groups.members);
        let shared = (files.as_ref())
            .and_then(|f| f.downcast_ref::<Vec<Arc<std::os::fd::OwnedFd>>>())
            .map(Vec::len);
        assert_eq!(shared, Some(1), "one group");
        assert!(members.iter().all(Option::is_some), "every view in it");
        let before = d.raw.work_counts();
        let mut buffers = Vec::new();
        for (round, &(src, view, n)) in made.iter().enumerate() {
            let check = FoldCheck {
                own: src.addr().to_u64(),
                view: view.addr().to_u64(),
                pages: n,
                round: round.to_u64(),
            };
            check.write();
            let (buffer, gone) = no_copy(&d, view, n);
            assert_eq!(check.gpu(&d.raw, &buffer.raw), Ok(()), "view {round}");
            buffers.push((buffer, gone));
        }
        let after = d.raw.work_counts();
        assert_eq!(
            (
                after.dma_buf_imports - before.dma_buf_imports,
                after.dma_buf_pieces_joined - before.dma_buf_pieces_joined,
                after.dma_buf_shared_imports_live,
            ),
            (1, 2, before.dma_buf_shared_imports_live + 1),
            "the file imported once, the other pieces joining it"
        );
        drop(files);
        let wait = std::time::Duration::from_secs(10);
        for (buffer, gone) in buffers {
            drop(buffer);
            assert_eq!(gone.recv_timeout(wait), Ok(()), "the deallocator ran");
        }
        let live = || d.raw.work_counts().dma_buf_shared_imports_live;
        for _ in 0..wait.as_millis() {
            if live() == before.dma_buf_shared_imports_live {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            live(),
            before.dma_buf_shared_imports_live,
            "the file's import freed"
        );
        drop(members);
        for (src, view, n) in made {
            assert!(vm.deallocate(view, n * PAGE).is_success());
            assert!(vm.deallocate(src, n * PAGE).is_success());
        }
    }

    /// Where views are imported through their dma-bufs and udmabuf opens (RADV's), no-copy
    /// buffers over the guest's RAM in one region of its file are imported as pieces of the
    /// region's dma-buf (`super::dma_buf`'s module docs, "Regions"): the device imports the file
    /// at the first piece and the other joins that import; the GPU reads each buffer's own pages
    /// and its fill reaches them; once the buffers go, so does the file's import. Elsewhere, said
    /// on stderr, nothing is checked.
    #[test]
    fn buffers_over_the_guests_ram_are_pieces_of_its_region() {
        let d = crate::gpu::device::tests::dev();
        crate::gpu::device::tests::settle_device_checks(&d);
        probes_gone();
        if !groups_views() {
            eprintln!(
                "skipped: {} takes no view through its dma-bufs here, or udmabuf does not open",
                d.raw.properties().name
            );
            return;
        }
        let length: u64 = 8 << 20;
        let file = vm::memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, which the test unmaps after the lease and
        // every buffer over it.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&*file),
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        let at = start.addr().to_u64();
        let lease = super::super::lease_guest_memory([crate::os::GuestRamFile {
            host: at,
            guest: 0,
            length,
            file,
            offset: 0,
        }]);
        let pages = 16;
        let before = d.raw.work_counts();
        let mut buffers = Vec::new();
        for (round, first) in [at, at + (1 << 20)].into_iter().enumerate() {
            let check = FoldCheck {
                own: first,
                view: first,
                pages,
                round: round.to_u64(),
            };
            check.write();
            let (buffer, gone) = no_copy(&d, HostAddr::from_addr(first.to_usize()), pages);
            let (_out, copied) = check.run(&d.raw, &buffer.raw, true).unwrap();
            for i in 0..pages {
                assert_eq!(
                    FoldCheck::ends(copied, i),
                    [check.written(i); 2],
                    "buffer {round} read its own pages"
                );
                let now = match i < pages / 2 {
                    true => FoldCheck::FILL,
                    false => check.written(i),
                };
                assert_eq!(
                    FoldCheck::ends(first, i),
                    [now; 2],
                    "buffer {round}'s fill reached its pages"
                );
            }
            buffers.push((buffer, gone));
        }
        let after = d.raw.work_counts();
        assert_eq!(
            (
                after.dma_buf_imports - before.dma_buf_imports,
                after.dma_buf_pieces_joined - before.dma_buf_pieces_joined,
                after.dma_buf_shared_imports_live,
            ),
            (1, 1, before.dma_buf_shared_imports_live + 1),
            "the region's file imported once, the other piece joining it"
        );
        let wait = std::time::Duration::from_secs(10);
        for (buffer, gone) in buffers {
            drop(buffer);
            assert_eq!(gone.recv_timeout(wait), Ok(()), "the deallocator ran");
        }
        let live = || d.raw.work_counts().dma_buf_shared_imports_live;
        for _ in 0..wait.as_millis() {
            if live() == before.dma_buf_shared_imports_live {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            live(),
            before.dma_buf_shared_imports_live,
            "the file's import freed"
        );
        drop(lease);
        let unmapped =
            crate::remap::wait_unimported(NonNull::new(start.cast::<u8>()).unwrap(), length, wait);
        assert!(unmapped, "the RAM's imports went");
        // SAFETY: the mapping made above, which nothing imports any more.
        assert_eq!(unsafe { libc::munmap(start, length.to_usize()) }, 0);
    }
}
