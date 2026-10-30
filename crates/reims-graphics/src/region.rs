//! [`Region`]: the device's BAR 2, host memory the guest maps write-back, with the control rings
//! at fixed offsets and the guest's allocations after them (`wire`).
//!
//! It is private anonymous memory of the host's process (`mmap` with `MAP_PRIVATE |
//! MAP_ANONYMOUS`), zeroed with stores, so that every page is the process's own before the guest
//! or a GPU import touches it: RADV refuses to import a memfd's or any shared mapping (measured on
//! the 780M and the RX 7900 XTX; `docs/graphics-kext/design.md`, Decided 5). The host maps it into
//! the guest once, at the device's start, and it stays mapped, at the same address, until the
//! device's end.
//!
//! **Guest writes.** The guest writes the region while the host reads it. The host reads the
//! rings' counters and flags with atomic loads and stores at their offsets, and copies out what a
//! message or an image holds before it reads it ([`Region::read`]), so a value the guest changes
//! later never changes what the host has checked.

use std::sync::atomic::{AtomicU32, AtomicU64};

/// The region's memory.
pub struct Region {
    /// Its first byte, page-aligned.
    base: *mut u8,
    /// Its length in bytes.
    len: usize,
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
            .finish()
    }
}

impl Region {
    /// `len` bytes of private anonymous memory, every byte written with zero. `None` where the
    /// host refuses the mapping, or `len` is 0.
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
        let base = base.cast::<u8>();
        // Zeroed with stores, so each page is faulted in as the process's own (the module doc).
        // SAFETY: the mapping is `len` bytes, readable and writable, and nothing else holds it.
        unsafe { std::ptr::write_bytes(base, 0, len) };
        Some(Region { base, len })
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
        if !self.contains(offset, len as u64) {
            return None;
        }
        let mut out = vec![0u8; len];
        // SAFETY: the bytes lie inside the mapping, which lives as long as `self`, and `out` is
        // `len` bytes of the process's own; the guest's concurrent writes change only what is
        // copied, never memory Rust owns.
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(offset as usize), out.as_mut_ptr(), len);
        }
        Some(out)
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
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made, which nothing reaches after this.
        unsafe { libc::munmap(self.base.cast(), self.len) };
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
}
