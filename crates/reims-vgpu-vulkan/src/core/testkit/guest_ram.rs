//! Fake: `GuestRam`: fake guest memory, and `RawRegion`, the real host memory behind every fake
//! address (`unsafe` allowed, architecture 17.3).
//!
//! Core reaches guest memory through raw host addresses (`guest`, D18), so the fake's memory must
//! be real: every fake address (guest RAM, `FakeVm` allocations, buffer contents, IOSurface base
//! addresses, rate-map sample storage) points into a [`RawRegion`], a page-aligned zeroed
//! allocation that lives as long as its last `Arc`. This module is the only place in `fake` that
//! dereferences one.

#![allow(unsafe_code)] // architecture 17.3: fake/guest_ram.rs

use std::alloc::{Layout, alloc_zeroed};
use std::sync::Arc;

use crate::core::sync::Mutex;

use crate::core::host::{HostPtr, PhysicalMemoryRange};
use crate::core::num::{ToU64, ToUsize};

// A vCPU's one 16-byte store and load, in one file per architecture.
#[cfg_attr(target_arch = "aarch64", path = "guest_ram/raw/aarch64.rs")]
#[cfg_attr(target_arch = "x86_64", path = "guest_ram/raw/x86_64.rs")]
mod raw;

/// The page size of the fake's memory: the host's `vm_page_size`, 16 KiB on the suite's arm64 host
/// and 4 KiB on an x86-64 one.
pub const PAGE_SIZE: u64 = if cfg!(target_arch = "x86_64") {
    0x1000
} else {
    0x4000
};

/// One page-aligned, zero-filled block of real host memory.
#[derive(Debug)]
pub struct RawRegion {
    ptr: *mut u8,
    len: usize,
    layout: Option<Layout>,
}

// SAFETY: the region is plain bytes; concurrent access through raw addresses is what core does to
// guest memory (D18), and this type only hands out copies.
unsafe impl Send for RawRegion {}
// SAFETY: as above; no reference into the bytes is ever handed out.
unsafe impl Sync for RawRegion {}

impl RawRegion {
    /// `len` zero bytes, page aligned. `None` when the allocator refuses (or `len` does not fit
    /// a layout). A zero length gives a region with a dangling, page-aligned address.
    pub fn new(len: u64) -> Option<Arc<RawRegion>> {
        let len = usize::try_from(len).ok()?;
        if len == 0 {
            return Some(Arc::new(RawRegion {
                ptr: std::ptr::without_provenance_mut(PAGE_SIZE.to_usize()),
                len: 0,
                layout: None,
            }));
        }
        let layout = Layout::from_size_align(len, PAGE_SIZE.to_usize()).ok()?;
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(Arc::new(RawRegion {
            ptr,
            len,
            layout: Some(layout),
        }))
    }

    /// The first byte's host address.
    pub fn addr(&self) -> HostPtr {
        HostPtr::from_ptr(self.ptr)
    }

    /// The length in bytes.
    pub fn len(&self) -> u64 {
        self.len.to_u64()
    }

    /// Whether the region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `addr..addr+len` lies inside the region.
    pub fn contains(&self, addr: HostPtr, len: u64) -> bool {
        self.offset_of(addr, len).is_some()
    }

    /// The offset of `addr` in the region, if `addr..addr+len` lies inside it.
    pub fn offset_of(&self, addr: HostPtr, len: u64) -> Option<u64> {
        let start = self.ptr.addr().to_u64();
        let a = addr.addr().to_u64();
        let off = a.checked_sub(start)?;
        let end = off.checked_add(len)?;
        (end <= self.len()).then_some(off)
    }

    fn range(&self, off: u64, len: usize) -> Option<*mut u8> {
        let end = off.checked_add(len.to_u64())?;
        if end > self.len() {
            return None;
        }
        // SAFETY: `off + len <= self.len`, so the pointer stays inside the allocation.
        Some(unsafe { self.ptr.add(off.to_usize()) })
    }

    /// Copies `dst.len()` bytes from offset `off`; `false` (nothing copied) if out of range.
    pub fn read(&self, off: u64, dst: &mut [u8]) -> bool {
        let Some(p) = self.range(off, dst.len()) else {
            return false;
        };
        // SAFETY: `p..p+dst.len()` is inside the allocation (checked above); `dst` is a distinct
        // Rust buffer.
        unsafe { std::ptr::copy_nonoverlapping(p, dst.as_mut_ptr(), dst.len()) };
        true
    }

    /// Copies `src` to offset `off`; `false` (nothing copied) if out of range.
    pub fn write(&self, off: u64, src: &[u8]) -> bool {
        let Some(p) = self.range(off, src.len()) else {
            return false;
        };
        // SAFETY: `p..p+src.len()` is inside the allocation (checked above); `src` is a distinct
        // Rust buffer.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len()) };
        true
    }

    /// `len` bytes from offset `off`, or `None` if out of range.
    pub fn read_vec(&self, off: u64, len: u64) -> Option<Vec<u8>> {
        let mut v = vec![0u8; usize::try_from(len).ok()?];
        self.read(off, &mut v).then_some(v)
    }

    /// Zero-fills `len` bytes at offset `off`.
    pub fn zero(&self, off: u64, len: u64) -> bool {
        let Ok(n) = usize::try_from(len) else {
            return false;
        };
        let Some(p) = self.range(off, n) else {
            return false;
        };
        // SAFETY: `p..p+n` is inside the allocation (checked above).
        unsafe { std::ptr::write_bytes(p, 0, n) };
        true
    }

    /// A little-endian `u32` at `off`.
    pub fn read_u32(&self, off: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.read(off, &mut b).then(|| u32::from_le_bytes(b))
    }

    /// A little-endian `u64` at `off`.
    pub fn read_u64(&self, off: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(off, &mut b).then(|| u64::from_le_bytes(b))
    }

    /// Writes a little-endian `u32` at `off`.
    pub fn write_u32(&self, off: u64, v: u32) -> bool {
        self.write(off, &v.to_le_bytes())
    }

    /// Writes a little-endian `u64` at `off`.
    pub fn write_u64(&self, off: u64, v: u64) -> bool {
        self.write(off, &v.to_le_bytes())
    }

    /// A vCPU's one 16-byte store at a 16-byte-aligned `off` (`stp lo, hi` on arm64, `movdqa` on
    /// x86-64; single-copy atomic there): what a concurrent guest writes whole. False when `off` is
    /// not aligned or the 16 bytes are not all inside the region.
    pub fn store_16_atomic(&self, off: u64, lo: u64, hi: u64) -> bool {
        let Some(a) = self.aligned_16(off) else {
            return false;
        };
        // SAFETY: `a..a+16` is inside the allocation and 16-aligned (`aligned_16`).
        unsafe { raw::store_16(a, lo, hi) };
        true
    }

    /// A vCPU's one 16-byte load at a 16-byte-aligned `off` (`ldp` on arm64, `movdqa` on
    /// x86-64; single-copy atomic there): `(bytes 0..8, bytes 8..16)`, little-endian.
    pub fn load_16_atomic(&self, off: u64) -> Option<(u64, u64)> {
        let a = self.aligned_16(off)?;
        // SAFETY: as `store_16_atomic`.
        Some(unsafe { raw::load_16(a) })
    }

    /// The address of `off` when it is 16-aligned and 16 bytes from it are inside the region.
    fn aligned_16(&self, off: u64) -> Option<*mut u8> {
        let a = self.addr().wrapping_add(off);
        (a.addr() & 0xf == 0 && self.contains(a, 16)).then(|| a.as_ptr::<u8>())
    }
}

impl Drop for RawRegion {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            // SAFETY: `ptr` came from `alloc_zeroed(layout)` and is freed once, here.
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

/// Copies `len` bytes between two regions (`FakeVm`'s remap and alias sync). `false` if either
/// range is out of bounds.
pub fn copy_between(
    src: &RawRegion,
    src_off: u64,
    dst: &RawRegion,
    dst_off: u64,
    len: u64,
) -> bool {
    let Ok(n) = usize::try_from(len) else {
        return false;
    };
    let (Some(s), Some(d)) = (src.range(src_off, n), dst.range(dst_off, n)) else {
        return false;
    };
    // SAFETY: both ranges are inside their allocations (checked above); `copy` allows overlap.
    unsafe { std::ptr::copy(s, d, n) };
    true
}

/// Writes `src` to the raw host address `dst`: a host block (`readMemory`) filling core's buffer.
///
/// # Safety
/// `dst..dst+src.len()` must be writable memory the caller owns for the duration of the call
/// (the contract of the host block the fake stands in for).
pub unsafe fn write_raw(dst: HostPtr, src: &[u8]) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr::<u8>(), src.len()) };
}

/// One guest-physical range backed by a region.
#[derive(Debug, Clone)]
pub struct GuestRange {
    /// The guest-physical address of the first byte.
    pub physical_address: u64,
    /// The backing host memory (its length is the range's).
    pub region: Arc<RawRegion>,
}

impl GuestRange {
    /// The host address of the first byte.
    pub fn host(&self) -> HostPtr {
        self.region.addr()
    }

    /// The range as the SDK's `PGPhysicalMemoryRange_t`.
    pub fn physical(&self) -> PhysicalMemoryRange {
        PhysicalMemoryRange {
            physical_address: self.physical_address,
            physical_length: self.region.len(),
        }
    }

    fn offset_of(&self, pa: u64, len: u64) -> Option<u64> {
        let off = pa.checked_sub(self.physical_address)?;
        (off.checked_add(len)? <= self.region.len()).then_some(off)
    }
}

/// Fake guest RAM: guest-physical ranges backed by real host memory, as QEMU's RAM blocks are.
/// `Clone` shares the RAM.
#[derive(Debug, Clone, Default)]
pub struct GuestRam {
    ranges: Arc<Mutex<Vec<GuestRange>>>,
}

impl GuestRam {
    /// No RAM.
    pub fn new() -> GuestRam {
        GuestRam::default()
    }

    /// One range of `len` bytes at guest-physical `pa`.
    pub fn with_range(pa: u64, len: u64) -> Option<GuestRam> {
        let ram = GuestRam::new();
        ram.add_range(pa, len)?;
        Some(ram)
    }

    /// Adds `len` zero bytes at guest-physical `pa`; returns the host address. `None` if the
    /// allocation fails.
    pub fn add_range(&self, pa: u64, len: u64) -> Option<HostPtr> {
        let region = RawRegion::new(len)?;
        let host = region.addr();
        self.ranges.lock().push(GuestRange {
            physical_address: pa,
            region,
        });
        Some(host)
    }

    /// Every range, in the order added (what a test puts in its memory map descriptor).
    pub fn ranges(&self) -> Vec<GuestRange> {
        self.ranges.lock().clone()
    }

    /// The range holding `pa..pa+len`, and the offset of `pa` in it.
    pub fn locate(&self, pa: u64, len: u64) -> Option<(GuestRange, u64)> {
        self.ranges
            .lock()
            .iter()
            .find_map(|r| r.offset_of(pa, len).map(|off| (r.clone(), off)))
    }

    /// The host address of guest-physical `pa`.
    pub fn host_addr(&self, pa: u64) -> Option<HostPtr> {
        self.locate(pa, 0)
            .map(|(r, off)| r.host().wrapping_add(off))
    }

    /// Reads guest-physical `pa..pa+dst.len()` (inside one range); `false` if unbacked.
    pub fn read(&self, pa: u64, dst: &mut [u8]) -> bool {
        match self.locate(pa, dst.len().to_u64()) {
            Some((r, off)) => r.region.read(off, dst),
            None => false,
        }
    }

    /// `len` bytes at guest-physical `pa`.
    pub fn read_vec(&self, pa: u64, len: u64) -> Option<Vec<u8>> {
        let (r, off) = self.locate(pa, len)?;
        r.region.read_vec(off, len)
    }

    /// Writes `src` at guest-physical `pa` (inside one range); `false` if unbacked.
    pub fn write(&self, pa: u64, src: &[u8]) -> bool {
        match self.locate(pa, src.len().to_u64()) {
            Some((r, off)) => r.region.write(off, src),
            None => false,
        }
    }

    /// A little-endian `u32` at guest-physical `pa`.
    pub fn read_u32(&self, pa: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.read(pa, &mut b).then(|| u32::from_le_bytes(b))
    }

    /// A little-endian `u64` at guest-physical `pa`.
    pub fn read_u64(&self, pa: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(pa, &mut b).then(|| u64::from_le_bytes(b))
    }

    /// Writes a little-endian `u32` at guest-physical `pa`.
    pub fn write_u32(&self, pa: u64, v: u32) -> bool {
        self.write(pa, &v.to_le_bytes())
    }

    /// Writes a little-endian `u64` at guest-physical `pa`.
    pub fn write_u64(&self, pa: u64, v: u64) -> bool {
        self.write(pa, &v.to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regions are zeroed, page aligned and bounds checked
    #[test]
    fn raw_region_bounds() {
        let r = RawRegion::new(0x8000).unwrap();
        assert_eq!(r.len(), 0x8000);
        assert!(!r.is_empty());
        assert_eq!(r.addr().addr() % usize::try_from(PAGE_SIZE).unwrap(), 0);
        assert_eq!(r.read_u64(0x7ff8), Some(0));
        assert!(r.write_u32(4, 0xdead_beef));
        assert_eq!(r.read_u32(4), Some(0xdead_beef));
        assert!(r.write_u64(8, 7));
        assert_eq!(r.read_u64(8), Some(7));
        assert!(!r.write(0x7fff, &[1, 2]));
        assert_eq!(r.read_u32(0x7ffd), None);
        assert!(r.contains(r.addr().wrapping_add(0x10), 0x7ff0));
        assert!(!r.contains(r.addr().wrapping_add(0x10), 0x7ff1));
        assert_eq!(r.offset_of(r.addr().wrapping_add(3), 1), Some(3));
        assert!(r.zero(4, 4));
        assert_eq!(r.read_u32(4), Some(0));
        assert!(!r.zero(0x8000, 1));
        assert_eq!(r.read_vec(8, 1), Some(vec![7]));
        let e = RawRegion::new(0).unwrap();
        assert!(e.is_empty());
        assert!(!e.write(0, &[1]));
    }

    /// Copies between regions, and the raw write
    #[test]
    fn copies() {
        let a = RawRegion::new(0x4000).unwrap();
        let b = RawRegion::new(0x4000).unwrap();
        a.write(0, b"hello");
        assert!(copy_between(&a, 0, &b, 10, 5));
        assert_eq!(b.read_vec(10, 5).unwrap(), b"hello");
        assert!(!copy_between(&a, 0x3fff, &b, 0, 2));
        // SAFETY: the address is inside `b`, which outlives the call.
        unsafe { write_raw(b.addr().wrapping_add(20), b"xy") };
        assert_eq!(b.read_vec(20, 2).unwrap(), b"xy");
    }

    /// Guest-physical addressing over several ranges
    #[test]
    fn guest_ram_ranges() {
        let ram = GuestRam::with_range(0x1_0000_0000, 0x10000).unwrap();
        let host2 = ram.add_range(0x2_0000_0000, 0x4000).unwrap();
        assert_eq!(ram.ranges().len(), 2);
        assert_eq!(ram.host_addr(0x2_0000_0010), Some(host2.wrapping_add(0x10)));
        assert!(ram.write_u32(0x1_0000_0100, 5));
        assert_eq!(ram.read_u32(0x1_0000_0100), Some(5));
        assert!(ram.write_u64(0x2_0000_0000, 9));
        assert_eq!(ram.read_u64(0x2_0000_0000), Some(9));
        assert!(
            !ram.write(0x1_0000_fffe, &[0; 4]),
            "no write across the end"
        );
        assert!(ram.host_addr(0x3_0000_0000).is_none());
        assert_eq!(ram.read_vec(0x1_0000_0100, 4).unwrap(), vec![5, 0, 0, 0]);
        let p = ram.ranges()[0].physical();
        assert_eq!(
            (p.physical_address, p.physical_length),
            (0x1_0000_0000, 0x10000)
        );
        // A clone shares the bytes.
        let r2 = ram.clone();
        assert_eq!(r2.read_u32(0x1_0000_0100), Some(5));
    }
}
