//! [`Region`]: the device's BAR 2, host memory the guest maps write-back, with the control rings
//! at fixed offsets and the guest's allocations after them (`wire`).
//!
//! It is memory of the host's process: the platform's own where it gives the device some
//! ([`Region::over`]; on Linux a shared mapping of a memfd of the library's, which every GPU
//! imports, RADV through dma-bufs of its pages), else private anonymous memory (`mmap` with
//! `MAP_PRIVATE | MAP_ANONYMOUS`, [`Region::new`]). Either is handed out zeroed and commits a page
//! only when something first touches it, so a page neither the guest nor the host has used costs
//! no host memory; a GPU import that pins the region commits what it pins. The host maps it into
//! the guest once, at the device's start, and it stays mapped, at the same address, until the
//! device's end.
//!
//! **Guest writes.** The guest writes the region while the host reads it. The host reads the
//! rings' counters and flags with atomic loads and stores at their offsets, and copies out what a
//! message or an image holds before it reads it ([`Region::read`]), so a value the guest changes
//! later never changes what the host has checked.
//!
//! **Blocks.** The guest's accelerator kext hands the processes blocks of the region and tells
//! the host of each (`BLOCK_GRANT`) before the process learns of it. The region keeps which
//! context holds each block until the block's `BLOCK_RELEASE` or the context's `CONTEXT_CLOSE`,
//! so a span a message names is checked against the sender's own blocks ([`Region::held_by`]),
//! on the queues' rings, which bypass the kext, as on ring 0. A shared block (`BLOCK_SHARE`, an
//! IOSurface plane's) is held by every context the kext shared it with, each until its
//! `BLOCK_UNSHARE` or its close, and by none after the block's `BLOCK_RELEASE`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::wire::{RgfxSurfaceWriter, RgfxSurfaceWriters};

/// The region's memory.
pub struct Region {
    /// Its first byte, page-aligned.
    base: *mut u8,
    /// Its length in bytes.
    len: usize,
    /// The blocks the processes hold, by offset.
    blocks: Mutex<BTreeMap<u64, Held>>,
    /// What holds the memory where the platform gave it ([`Region::over`]), which frees it as it
    /// drops; `None` for the region's own mapping ([`Region::new`]), which its drop unmaps.
    memory: Option<Box<dyn Send + Sync>>,
}

/// A block's size and who holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Held {
    size: u64,
    /// Its holders' contexts: one for a granted block, any number for a shared one.
    holders: Vec<u32>,
    /// Whether it is a shared block (`BLOCK_SHARE`), which more contexts may join.
    shared: bool,
}

// SAFETY: the region is plain memory the struct owns, reached only through the methods below,
// which use atomics or copies; the guest's writes to it are the device's protocol.
unsafe impl Send for Region {}
// SAFETY: as for `Send`: no method hands out a reference a write through another could change,
// other than the atomics.
unsafe impl Sync for Region {}

impl std::fmt::Debug for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Region")
            .field("base", &self.base)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Region {
    /// `len` bytes of private anonymous memory, which read as zero and are committed as they are
    /// first touched. `None` where the host refuses the mapping, or `len` is 0.
    pub fn new(len: usize) -> Option<Region> {
        if len == 0 {
            return None;
        }
        // SAFETY: a fresh anonymous mapping, at an address the kernel picks.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        Some(Region {
            base: base.cast::<u8>(),
            len,
            blocks: Mutex::new(BTreeMap::new()),
            memory: None,
        })
    }

    /// The `len` bytes at `base`, which `memory` holds and frees as it drops, after the region's
    /// last use. `None` where `base` is null or `len` is 0.
    ///
    /// # Safety
    ///
    /// `base` is page-aligned, and its `len` bytes are zeros, mapped readable and writable by this
    /// process and reached by nothing else of it, from now until `memory` drops.
    pub unsafe fn over(base: *mut u8, len: usize, memory: Box<dyn Send + Sync>) -> Option<Region> {
        if base.is_null() || len == 0 {
            return None;
        }
        Some(Region {
            base,
            len,
            blocks: Mutex::new(BTreeMap::new()),
            memory: Some(memory),
        })
    }

    /// Context `context` holds the `size` bytes at `offset` (`BLOCK_GRANT`), in place of what
    /// was held there.
    pub fn grant(&self, context: u32, offset: u64, size: u64) {
        let mut blocks = self.blocks.lock();
        Self::clear(&mut blocks, offset, size);
        blocks.insert(
            offset,
            Held {
                size,
                holders: vec![context],
                shared: false,
            },
        );
    }

    /// Context `context` holds the shared block of `size` bytes at `offset` too (`BLOCK_SHARE`),
    /// beside its holders; a span that meets any other block takes it from its holders, as a
    /// grant does.
    pub fn share(&self, context: u32, offset: u64, size: u64) {
        let mut blocks = self.blocks.lock();
        if let Some(held) = blocks.get_mut(&offset)
            && held.shared
            && held.size == size
        {
            if !held.holders.contains(&context) {
                held.holders.push(context);
            }
            return;
        }
        Self::clear(&mut blocks, offset, size);
        blocks.insert(
            offset,
            Held {
                size,
                holders: vec![context],
                shared: true,
            },
        );
    }

    /// Context `context` no longer holds the shared block of `size` bytes at `offset`
    /// (`BLOCK_UNSHARE`); its other holders keep it. Whether it held it: not where no such block
    /// is shared, nor where the hold ended already, at the block's release or the context's
    /// close, which the kext's unshare may come behind.
    pub fn unshare(&self, context: u32, offset: u64, size: u64) -> bool {
        let mut blocks = self.blocks.lock();
        let Some(held) = blocks.get_mut(&offset) else {
            return false;
        };
        if !held.shared || held.size != size || !held.holders.contains(&context) {
            return false;
        }
        held.holders.retain(|&c| c != context);
        if held.holders.is_empty() {
            blocks.remove(&offset);
        }
        true
    }

    /// Drops every block that meets the `size` bytes at `offset`.
    fn clear(blocks: &mut BTreeMap<u64, Held>, offset: u64, size: u64) {
        let end = offset.saturating_add(size);
        blocks.retain(|&at, held| at.saturating_add(held.size) <= offset || end <= at);
    }

    /// The block at `offset` is no longer anyone's to name (`BLOCK_RELEASE`): its holder's, or
    /// every holder's of a shared one. The contexts that held it, none where no block lies there.
    pub fn revoke(&self, offset: u64) -> Vec<u32> {
        self.blocks
            .lock()
            .remove(&offset)
            .map(|held| held.holders)
            .unwrap_or_default()
    }

    /// No block is context `context`'s any more (`CONTEXT_CLOSE`): a block only it holds goes,
    /// and a shared block's other holders keep it. The shared blocks it held, each as its offset
    /// and size, those it alone held among them: another context's share of one may not have
    /// been carried out yet, as it runs on that context's lane.
    pub fn revoke_context(&self, context: u32) -> Vec<(u64, u64)> {
        let mut shared = Vec::new();
        self.blocks.lock().retain(|&at, held| {
            let before = held.holders.len();
            held.holders.retain(|&c| c != context);
            if held.shared && held.holders.len() != before {
                shared.push((at, held.size));
            }
            !held.holders.is_empty()
        });
        shared
    }

    /// No block is anyone's any more (the device's reset): the kext grants them again.
    pub fn revoke_all(&self) {
        self.blocks.lock().clear();
    }

    /// Whether the `len` bytes at `offset` lie in one block context `context` holds.
    pub fn held_by(&self, context: u32, offset: u64, len: u64) -> bool {
        Self::in_held(&self.blocks.lock(), context, offset, len)
    }

    /// The block the `len` bytes at `offset` lie in, whoever holds it: its offset and size;
    /// `None` where they lie in no one block.
    pub fn block_of(&self, offset: u64, len: u64) -> Option<(u64, u64)> {
        let blocks = self.blocks.lock();
        let (&at, held) = blocks.range(..=offset).next_back()?;
        let end = offset.checked_add(len)?;
        (end <= at.saturating_add(held.size)).then_some((at, held.size))
    }

    /// A copy of the `len` bytes at `offset`, where they lie in one block context `context`
    /// holds; the block cannot be revoked, and so given to another process, while it is copied.
    pub fn read_held(&self, context: u32, offset: u64, len: usize) -> Option<Vec<u8>> {
        let blocks = self.blocks.lock();
        if !Self::in_held(&blocks, context, offset, len as u64) {
            return None;
        }
        self.read(offset, len)
    }

    fn in_held(blocks: &BTreeMap<u64, Held>, context: u32, offset: u64, len: u64) -> bool {
        blocks
            .range(..=offset)
            .next_back()
            .is_some_and(|(&at, held)| {
                held.holders.contains(&context)
                    && offset
                        .checked_add(len)
                        .is_some_and(|end| end <= at.saturating_add(held.size))
            })
    }

    /// The region's first byte: what the host maps into the guest.
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Its size in bytes.
    pub fn size(&self) -> u64 {
        // usize is 64 bits on every target the library builds for.
        self.len as u64
    }

    /// Whether `len` bytes at `offset` lie inside the region.
    pub fn contains(&self, offset: u64, len: u64) -> bool {
        offset
            .checked_add(len)
            .is_some_and(|end| end <= self.size())
    }

    /// A copy of the `len` bytes at `offset`, or `None` where they do not lie inside the region.
    pub fn read(&self, offset: u64, len: usize) -> Option<Vec<u8>> {
        let mut out = vec![0u8; len];
        self.read_into(offset, &mut out).then_some(out)
    }

    /// Copies the bytes at `offset` into `out`, allocating nothing; false, with `out` left as it
    /// was, where they do not lie inside the region.
    pub fn read_into(&self, offset: u64, out: &mut [u8]) -> bool {
        if !self.contains(offset, out.len() as u64) {
            return false;
        }
        // SAFETY: the bytes lie inside the mapping, which lives as long as `self`, and `out` is
        // the process's own; the guest's concurrent writes change only what is copied, never
        // memory Rust owns.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.base.add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        true
    }

    /// The `N` bytes at `offset`, allocating nothing, or `None` where they do not lie inside the
    /// region.
    pub(crate) fn read_array<const N: usize>(&self, offset: u64) -> Option<[u8; N]> {
        let mut out = [0u8; N];
        self.read_into(offset, &mut out).then_some(out)
    }

    /// A copy of the `count` records of `stride` bytes from `offset`, each with its first `taken`
    /// `u64` words swapped for 0 as it is copied, the swaps before the rest of the record is
    /// read: the bits the guest set there since the last take, with the bytes it wrote before
    /// setting them ([`crate::icb`], "Slots"). `None`, taking nothing, where the records do not
    /// lie inside the region, `offset` and `stride` are not 8-aligned, or `stride` holds fewer
    /// than `taken` words.
    pub fn take_records(
        &self,
        offset: u64,
        stride: u64,
        count: u64,
        taken: usize,
    ) -> Option<Vec<u8>> {
        let len = stride.checked_mul(count)?;
        let words = taken as u64 * 8;
        if !offset.is_multiple_of(8)
            || !stride.is_multiple_of(8)
            || stride < words
            || !self.contains(offset, len)
        {
            return None;
        }
        let mut out = vec![0u8; usize::try_from(len).ok()?];
        for (i, record) in out.chunks_exact_mut(stride as usize).enumerate() {
            let at = offset + i as u64 * stride;
            for w in 0..taken {
                let word = self.atomic_u64(at + w as u64 * 8).swap(0, Ordering::AcqRel);
                record[w * 8..w * 8 + 8].copy_from_slice(&word.to_le_bytes());
            }
            // SAFETY: the record's bytes lie inside the mapping (checked above), which lives as
            // long as `self`; the guest's concurrent writes change only what is copied.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.base.add((at + words) as usize),
                    record[taken * 8..].as_mut_ptr(),
                    record.len() - taken * 8,
                );
            }
        }
        Some(out)
    }

    /// [`take_records`](Self::take_records), where the records lie in one block context
    /// `context` holds, which cannot be revoked while they are taken.
    pub fn take_records_held(
        &self,
        context: u32,
        offset: u64,
        stride: u64,
        count: u64,
        taken: usize,
    ) -> Option<Vec<u8>> {
        let blocks = self.blocks.lock();
        if !Self::in_held(&blocks, context, offset, stride.checked_mul(count)?) {
            return None;
        }
        self.take_records(offset, stride, count, taken)
    }

    /// Writes `bytes` at `offset`. False, writing nothing, where they do not lie inside the
    /// region.
    pub fn write(&self, offset: u64, bytes: &[u8]) -> bool {
        if !self.contains(offset, bytes.len() as u64) {
            return false;
        }
        // SAFETY: as in `read`, the destination lies inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.base.add(offset as usize),
                bytes.len(),
            );
        }
        true
    }

    /// The `u64` the guest keeps at `offset`, read atomically (`Acquire`), as the guest writes it;
    /// `None` where it is not 8-aligned and inside the region.
    pub fn load_u64(&self, offset: u64) -> Option<u64> {
        (offset.is_multiple_of(8) && self.contains(offset, 8))
            .then(|| self.atomic_u64(offset).load(Ordering::Acquire))
    }

    /// The `u64` at `offset`, which is 8-aligned and inside the region, as an atomic.
    pub(crate) fn atomic_u64(&self, offset: u64) -> &AtomicU64 {
        assert!(
            offset.is_multiple_of(8) && self.contains(offset, 8),
            "a u64 field of the region"
        );
        // SAFETY: the 8 bytes lie inside the mapping, aligned, and live as long as `self`; every
        // access of the host's to them is atomic.
        unsafe { AtomicU64::from_ptr(self.base.add(offset as usize).cast()) }
    }

    /// The `u32` at `offset`, which is 4-aligned and inside the region, as an atomic.
    pub(crate) fn atomic_u32(&self, offset: u64) -> &AtomicU32 {
        assert!(
            offset.is_multiple_of(4) && self.contains(offset, 4),
            "a u32 field of the region"
        );
        // SAFETY: as in `atomic_u64`.
        unsafe { AtomicU32::from_ptr(self.base.add(offset as usize).cast()) }
    }

    /// The [`RgfxSurfaceWriters`] the guest keeps at `offset`, each slot read as the guest's
    /// writes allow (the record's doc): its queue, its seqno, then its queue again, a slot whose
    /// queue changed meanwhile reading as free. `None` where the record is not 8-aligned and
    /// wholly inside the region.
    pub fn surface_writers(&self, offset: u64) -> Option<RgfxSurfaceWriters> {
        let slot = size_of::<RgfxSurfaceWriter>() as u64;
        if !offset.is_multiple_of(8)
            || !self.contains(offset, size_of::<RgfxSurfaceWriters>() as u64)
        {
            return None;
        }
        let mut record = RgfxSurfaceWriters::default();
        for (i, writer) in record.writers.iter_mut().enumerate() {
            let at = offset + i as u64 * slot;
            let queue = self.atomic_u64(at).load(Ordering::Acquire);
            let seqno = self.atomic_u64(at + 8).load(Ordering::Acquire);
            if queue != 0 && self.atomic_u64(at).load(Ordering::Acquire) == queue {
                *writer = RgfxSurfaceWriter {
                    queue_obj_id: queue,
                    seqno,
                };
            }
        }
        Some(record)
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        match self.memory.take() {
            // The platform's memory, which its holder frees.
            Some(memory) => drop(memory),
            // SAFETY: the mapping `new` made, which nothing reaches after this.
            None => unsafe {
                libc::munmap(self.base.cast(), self.len);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new region reads as zeros, and takes writes inside it only
    #[test]
    fn a_region_starts_zeroed_and_bounds_its_accesses() {
        let region = Region::new(64 * 1024).unwrap();
        assert_eq!(region.size(), 64 * 1024);
        assert!(region.read(0, 64 * 1024).unwrap().iter().all(|&b| b == 0));
        assert!(region.write(100, &[1, 2, 3]));
        assert_eq!(region.read(100, 3).unwrap(), [1, 2, 3]);
        assert!(!region.write(64 * 1024 - 2, &[1, 2, 3]));
        assert!(region.read(64 * 1024 - 2, 3).is_none());
        assert!(region.read(u64::MAX, 1).is_none());
        assert!(Region::new(0).is_none());
    }

    /// A span's block is the granted or shared block it lies in, whoever holds it; a span past
    /// its block's end, or in none, has none
    #[test]
    fn a_span_s_block_is_the_one_it_lies_in() {
        let region = Region::new(1 << 20).unwrap();
        region.grant(3, 0x1_0000, 0x4000);
        region.share(4, 0x2_0000, 0x2000);
        assert_eq!(region.block_of(0x1_1000, 0x1000), Some((0x1_0000, 0x4000)));
        assert_eq!(region.block_of(0x1_0000, 0x4000), Some((0x1_0000, 0x4000)));
        assert_eq!(region.block_of(0x2_1000, 0x1000), Some((0x2_0000, 0x2000)));
        assert_eq!(region.block_of(0x1_3000, 0x2000), None);
        assert_eq!(region.block_of(0x8000, 0x10), None);
        assert_eq!(region.block_of(u64::MAX, 2), None);
    }

    /// A region over memory it is given reads and writes that memory, and frees it, through its
    /// holder, only as it drops; none is made over a null base or 0 bytes
    #[test]
    fn a_region_over_given_memory_frees_it_as_it_drops() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        struct Holder {
            at: usize,
            len: usize,
            freed: Arc<AtomicBool>,
        }
        impl Drop for Holder {
            fn drop(&mut self) {
                // SAFETY: the test's own mapping, which nothing reaches after the region drops.
                unsafe { libc::munmap(std::ptr::with_exposed_provenance_mut(self.at), self.len) };
                self.freed.store(true, Ordering::Relaxed);
            }
        }
        let len = 64 * 1024;
        // SAFETY: a fresh anonymous mapping, at an address the kernel picks.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        let freed = Arc::new(AtomicBool::new(false));
        let holder = Holder {
            at: at.expose_provenance(),
            len,
            freed: Arc::clone(&freed),
        };
        // SAFETY: the mapping's zeros, which only the region reaches until the holder drops.
        let region = unsafe { Region::over(at.cast(), len, Box::new(holder)) }.unwrap();
        assert_eq!(region.size(), 64 * 1024);
        assert!(region.read(0, 64 * 1024).unwrap().iter().all(|&b| b == 0));
        assert!(region.write(100, &[1, 2, 3]));
        assert_eq!(region.read(100, 3).unwrap(), [1, 2, 3]);
        assert!(!freed.load(Ordering::Relaxed));
        drop(region);
        assert!(freed.load(Ordering::Relaxed));
        // SAFETY: neither reaches any memory.
        assert!(unsafe { Region::over(std::ptr::null_mut(), len, Box::new(())) }.is_none());
        let mut byte = 0u8;
        // SAFETY: as above.
        assert!(unsafe { Region::over(&raw mut byte, 0, Box::new(())) }.is_none());
    }

    /// A surface plane's writers read slot by slot, a free slot (queue 0) as free, and nothing
    /// for a record misaligned or past the region's end
    #[test]
    fn a_surface_s_writers_are_read_slot_by_slot() {
        let region = Region::new(64 * 1024).unwrap();
        let at = 0x1000;
        let mut record = RgfxSurfaceWriters::default();
        record.writers[0] = RgfxSurfaceWriter {
            queue_obj_id: 0x3_0000_0001,
            seqno: 7,
        };
        record.writers[2] = RgfxSurfaceWriter {
            queue_obj_id: 0x4_0000_0002,
            seqno: 9,
        };
        // A free slot's stale seqno reads as free.
        record.writers[3].seqno = 5;
        assert!(region.write(at, zerocopy::IntoBytes::as_bytes(&record)));
        let read = region.surface_writers(at).unwrap();
        let expected = RgfxSurfaceWriters {
            writers: [
                record.writers[0],
                RgfxSurfaceWriter::default(),
                record.writers[2],
                RgfxSurfaceWriter::default(),
            ],
        };
        assert_eq!(read, expected);
        assert!(region.surface_writers(at + 4).is_none());
        assert!(region.surface_writers(64 * 1024 - 32).is_none());
        assert!(region.surface_writers(u64::MAX - 7).is_none());
    }

    /// A shared block is held by each context it is shared with until that context lets it go
    /// or closes, and by none after its release; a grant over it takes it from them all.
    #[test]
    fn a_shared_block_is_each_holder_s_until_it_lets_it_go() {
        let region = Region::new(1 << 20).unwrap();
        region.share(1, 0x1_0000, 0x2000);
        region.share(2, 0x1_0000, 0x2000);
        region.share(2, 0x1_0000, 0x2000);
        assert!(region.held_by(1, 0x1_0000, 0x2000));
        assert!(region.held_by(2, 0x1_1000, 0x1000));
        assert!(!region.held_by(3, 0x1_0000, 0x10));
        assert!(!region.held_by(2, 0x1_1000, 0x1001));
        // One holder lets it go; the other keeps it.
        assert!(region.unshare(1, 0x1_0000, 0x2000));
        assert!(!region.held_by(1, 0x1_0000, 0x10));
        assert!(region.held_by(2, 0x1_0000, 0x10));
        // An unshare of another size, or of a granted block, or by a context no longer holding
        // it, changes nothing.
        region.grant(4, 0x4_0000, 0x1000);
        assert!(!region.unshare(4, 0x4_0000, 0x1000));
        assert!(!region.unshare(2, 0x1_0000, 0x1000));
        assert!(!region.unshare(1, 0x1_0000, 0x2000));
        assert!(region.held_by(4, 0x4_0000, 0x1000));
        assert!(region.held_by(2, 0x1_0000, 0x10));
        // A context's close ends its holds; the block goes with its last holder.
        region.share(3, 0x1_0000, 0x2000);
        assert_eq!(region.revoke_context(2), [(0x1_0000, 0x2000)]);
        assert!(!region.held_by(2, 0x1_0000, 0x10));
        assert!(region.held_by(3, 0x1_0000, 0x10));
        // Its release ends every hold.
        region.share(5, 0x1_0000, 0x2000);
        assert_eq!(region.revoke(0x1_0000), [3, 5]);
        assert!(!region.held_by(3, 0x1_0000, 0x10));
        assert!(!region.held_by(5, 0x1_0000, 0x10));
        assert_eq!(region.revoke(0x1_0000), [] as [u32; 0]);
        // A share over a granted block, or a grant over a shared one, takes the span.
        region.share(6, 0x4_0000, 0x1000);
        assert!(!region.held_by(4, 0x4_0000, 0x10));
        assert!(region.held_by(6, 0x4_0000, 0x10));
        region.share(7, 0x4_0000, 0x1000);
        region.grant(8, 0x4_0000, 0x2000);
        assert!(!region.held_by(6, 0x4_0000, 0x10));
        assert!(!region.held_by(7, 0x4_0000, 0x10));
        assert!(region.held_by(8, 0x4_1000, 0x1000));
        // A share of another size at a shared block's offset starts the block anew.
        region.share(1, 0x8_0000, 0x1000);
        region.share(2, 0x8_0000, 0x2000);
        assert!(!region.held_by(1, 0x8_0000, 0x10));
        assert!(region.held_by(2, 0x8_1000, 0x1000));
    }

    /// A context's close names every shared block it held, those it alone held too, whose other
    /// holders' shares may still be on their lanes; not a granted block, which only its own
    /// context names, nor one it never held.
    #[test]
    fn a_close_names_every_shared_block_its_context_held() {
        let region = Region::new(1 << 20).unwrap();
        region.share(1, 0x1_0000, 0x2000);
        region.share(2, 0x1_0000, 0x2000);
        region.share(1, 0x2_0000, 0x1000);
        region.share(3, 0x3_0000, 0x1000);
        region.share(2, 0x3_0000, 0x1000);
        region.grant(1, 0x4_0000, 0x1000);
        assert_eq!(
            region.revoke_context(1),
            [(0x1_0000, 0x2000), (0x2_0000, 0x1000)]
        );
        assert!(!region.held_by(1, 0x2_0000, 0x10));
        assert!(!region.held_by(1, 0x4_0000, 0x10));
        assert!(region.held_by(2, 0x1_0000, 0x10));
        assert_eq!(region.revoke_context(1), []);
        assert_eq!(
            region.revoke_context(2),
            [(0x1_0000, 0x2000), (0x3_0000, 0x1000)]
        );
        assert_eq!(region.revoke(0x1_0000), [] as [u32; 0]);
        assert_eq!(region.revoke(0x3_0000), [3]);
    }

    /// The process's peak resident size in bytes: `ru_maxrss` is in bytes on macOS and in KiB
    /// on Linux.
    fn peak_resident() -> u64 {
        // SAFETY: getrusage writes the struct it is given.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
        let maxrss = u64::try_from(usage.ru_maxrss).unwrap();
        if cfg!(target_os = "macos") {
            maxrss
        } else {
            maxrss * 1024
        }
    }

    /// A region costs host memory only for the pages that are touched: making a 1 GiB one does
    /// not raise the process's peak resident size by anything near its size.
    #[test]
    fn a_region_commits_only_the_pages_that_are_touched() {
        const LEN: usize = 1 << 30;
        let before = peak_resident();
        let region = Region::new(LEN).unwrap();
        assert!(region.write(0, &[1]));
        assert_eq!(region.read(LEN as u64 - 4096, 4096).unwrap(), vec![0; 4096]);
        let grown = peak_resident().saturating_sub(before);
        assert!(
            grown < (LEN as u64) / 4,
            "a 1 GiB region raised the peak by {grown} bytes"
        );
    }
}
