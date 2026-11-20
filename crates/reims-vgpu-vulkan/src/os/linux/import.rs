//! What Linux devices learn of imported host memory: the pages a device of the process may pin
//! now ([`Pinned`], which a fold of the views maps none of through dma-bufs, `super::vm`'s
//! `fold_idle`), and the rt platform's settings of views mapped from their files for the imports
//! of their pages ([`views_need_files`], [`learn_dma_buf_granule`]).
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
//!   The two Radeon GPUs show no such cost.
//! - **The RX 7900 XTX's L2 holds host memory**: the window's loads took 1.0 to 1.5 ms at 4 MiB and
//!   26 to 27 ms at 16 MiB, and atomic adds to one word ran as fast as in its own memory. Its
//!   scattered loads in a 4 MiB window ran faster over 2 MiB pages (`super::dma_buf`).
//! - **On the Radeon 780M host memory is the GPU's memory**: within 1.4 times device-local in
//!   order, and over 4 KiB pages 1.4 to 2.3 times for scattered loads and stores, which 2 MiB
//!   pages bring back to device-local's (`super::dma_buf`).
//! - Of a dma-buf's two memory types on RADV, the device-coherent and uncached one took as long
//!   as the cached one or longer (the scattered stores 120 ms against 91 on the RX 7900 XTX).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::sync::Mutex;

use super::budget;

/// The pages a device of the process may pin now: each live import's, of pages it takes as they
/// are mapped from their files (a pointer's, NVIDIA's and i915's), by first byte and number, to
/// their end, from before the import maps them from their files until the import is freed
/// ([`Pinned`]). A fold of the views maps none of them through dma-bufs (`super::vm::fold_idle`):
/// a device took them by their mappings.
static PINNED: Mutex<BTreeMap<(u64, u64), u64>> = Mutex::new(BTreeMap::new());

/// One import's pages in [`PINNED`], until it drops.
#[derive(Debug)]
pub(crate) struct Pinned((u64, u64));

impl Pinned {
    /// The pages of `first..end`.
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

/// How many records of [`PINNED`] have dropped, each freeing pages a fold may map through
/// dma-bufs (`super::vm`'s `fold_short`).
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

/// Whether a device of the process imports by pointer only pages mapped from their files: set at
/// rt's first refusal of a view's one dma-buf ([`views_need_files`]). From then on views are mapped from their files (`super::vm`'s module
/// docs, "Imports of views").
static IMPORTS_NEED_FILES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a device of the process imports by pointer only pages mapped from their files
/// ([`IMPORTS_NEED_FILES`]).
pub(crate) fn imports_need_files() -> bool {
    IMPORTS_NEED_FILES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Views are mapped from their files from now on ([`IMPORTS_NEED_FILES`]): rt refused a dma-buf
/// of the pages of a view mapped through dma-bufs, one that holds them all, for `why`, so they go
/// to the device by their pointer, mapped from their files first (`super::vm`'s `unfold`). The rt
/// platform's answer at that refusal (`crate::rt_platform`'s `buffer` docs, "Pages no pin
/// takes"), which rt gives only at the call, where no probe runs as the device is made: a later
/// view is mapped from its files at once, so its import maps none of its pages again. Logged the
/// first time, with what the budget then leaves imports (`super::budget`'s
/// `warn_if_imports_short`).
pub(crate) fn views_need_files(why: &str) {
    if IMPORTS_NEED_FILES.swap(true, Ordering::Relaxed) {
        return;
    }
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "rt refused a dma-buf of the pages of a view mapped through dma-bufs ({why}): views are \
         mapped from their files from now on"
    );
    budget::warn_if_imports_short();
}

/// Whether a device of the process imports no several dma-bufs behind one buffer, so the pages of
/// a view more than one dma-buf holds go to it by their pointer, mapped from their files: the rt
/// platform's device's answer as it is made, rt's `dma_buf_granule` (`None`), read by
/// `learn_dma_buf_granule`. From then on such a view is mapped from its files at once
/// (`super::vm`'s module docs, "Imports of views").
static SEVERAL_DMA_BUFS_REFUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a device of the process imports no several dma-bufs behind one buffer
/// ([`SEVERAL_DMA_BUFS_REFUSED`]).
pub(crate) fn several_dma_bufs_refused() -> bool {
    SEVERAL_DMA_BUFS_REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Learns of the rt platform's device, as it is made, whether it imports several dma-bufs behind
/// one buffer: rt's `dma_buf_granule`, `granule`, where `None` it imports none
/// ([`SEVERAL_DMA_BUFS_REFUSED`]). Logged where it imports none.
pub(crate) fn learn_dma_buf_granule(granule: Option<u64>) {
    if granule.is_some() || SEVERAL_DMA_BUFS_REFUSED.swap(true, Ordering::Relaxed) {
        return;
    }
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "the device imports no several dma-bufs behind one buffer: views of guest pages more than \
         one dma-buf holds are mapped from their files"
    );
}

/// Whether a view of several runs of pages made now is mapped through dma-bufs: udmabuf is the
/// process's, and no device of it needs imported pages mapped from their files (`super::vm`'s
/// module docs, "Views through dma-bufs").
#[cfg(test)]
pub(crate) fn views_through_dma_bufs() -> bool {
    super::dma_buf::opens() && !imports_need_files()
}
