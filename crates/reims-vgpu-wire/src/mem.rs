//! The guest-memory seam.
//!
//! Every layout in [`crate::ops`] is reached from a buffer someone already
//! handed us. The structures in [`crate::page_table`] are not: a page table is
//! a tree the guest scattered across its own physical pages, and reading it
//! means fetching one page, reading four bytes, and fetching the page those
//! bytes name. That traversal cannot live in a crate that only takes `&[u8]`.
//!
//! [`GuestMemory`] is the one thing this crate asks the device for so the
//! traversal *can* live here. It is deliberately the smallest possible seam —
//! two methods, no lifetimes to manage, no notion of tasks, resources, mappings
//! or caches — because everything above it is device policy and pulling any of
//! it in would make the crate depend on the device model it exists to feed.
//!
//! # Why two methods
//!
//! `slice_at` is the one you want and the one you often cannot have. Borrowing
//! guest bytes requires the span to be contiguous *in host address space*, and
//! on the x86 pathway it usually is not: a 12-bit page shift fragments nearly
//! every span, which is the same reason `reims_vgpu::runtime::gva_view`'s reuse
//! counter reads zero there. So `read_at` — which copies into a caller buffer —
//! is the required method, and `slice_at` is optional and defaults to `None`.
//!
//! An implementation that can borrow should override `slice_at`; one that
//! cannot should leave it alone. A caller must treat `None` as "not contiguous",
//! never as "not present", and fall back to `read_at`.
//!
//! # What this is not
//!
//! It is not a resource or object-ref lookup. Resolving a ref to a live texture
//! is device state with a lifetime, and a trait for it would drag the device
//! model in behind it. This trait answers exactly one question — *what bytes are
//! at this address* — and the address space it answers in is the caller's to
//! keep straight.
//!
//! That last point is not decoration. `reims-vgpu` has already shipped one bug
//! from mixing a guest-virtual number into a guest-physical predicate, and the
//! rule it broke is the one restated here: an address has no meaning apart from
//! the space it is resolved in. A single `GuestMemory` implementation must serve
//! exactly one space. Do not write one that inspects the address to decide.

/// Read-only access to one guest address space.
///
/// Implementors serve a single space — guest-physical *or* one task's
/// guest-virtual, never both. See the module docs for why that is a hard rule
/// rather than a preference.
pub trait GuestMemory {
    /// Copy `out.len()` bytes starting at `addr` into `out`.
    ///
    /// Returns `false` if any byte of the span is unreadable, in which case
    /// `out` holds unspecified bytes and must not be used. Partial success is
    /// not representable on purpose: a half-read page-table entry is not a
    /// smaller problem than an unreadable one.
    fn read_at(&self, addr: u64, out: &mut [u8]) -> bool;

    /// Borrow `len` bytes at `addr` without copying.
    ///
    /// Returns `None` when the span cannot be borrowed — most often because it
    /// is not host-contiguous, which on a 12-bit page shift is the common case.
    /// `None` says nothing about whether the bytes exist; callers fall back to
    /// [`GuestMemory::read_at`].
    ///
    /// The default refuses every borrow, so an implementation that has no
    /// contiguous view is complete without writing this.
    fn slice_at(&self, addr: u64, len: usize) -> Option<&[u8]> {
        let _ = (addr, len);
        None
    }

    /// Read one little-endian `u32`, the width of a page-table entry.
    ///
    /// Provided rather than left to callers because the walk does nothing else,
    /// and because `PTE_SIZE` being 4 is a fact about the guest's format
    /// (see [`crate::page_table`]) rather than about any particular caller.
    fn u32_at(&self, addr: u64) -> Option<u32> {
        let mut bytes = [0u8; 4];
        if !self.read_at(addr, &mut bytes) {
            return None;
        }
        Some(u32::from_le_bytes(bytes))
    }
}

/// A reference to a `GuestMemory` is one, including `&dyn GuestMemory`.
///
/// This is what lets a caller that holds the trait object — the layer above
/// this crate erases the concrete host type — hand it to the generic walk
/// without a forwarding wrapper. Every method forwards, including `slice_at`:
/// dropping to the refusing default here would silently take the copy path for
/// an implementation that can borrow.
impl<T: GuestMemory + ?Sized> GuestMemory for &T {
    fn read_at(&self, addr: u64, out: &mut [u8]) -> bool {
        (**self).read_at(addr, out)
    }

    fn slice_at(&self, addr: u64, len: usize) -> Option<&[u8]> {
        (**self).slice_at(addr, len)
    }

    fn u32_at(&self, addr: u64) -> Option<u32> {
        (**self).u32_at(addr)
    }
}

/// A `GuestMemory` over a plain byte slice, addressed from zero.
///
/// This is the test double the walk is exercised against, and it is in the
/// library rather than behind `#[cfg(test)]` so `reims-vgpu`'s own tests and
/// any future fuzz target can build page tables without restating it.
pub struct SliceMemory<'a> {
    bytes: &'a [u8],
}

impl<'a> SliceMemory<'a> {
    #[inline]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
}

impl GuestMemory for SliceMemory<'_> {
    fn read_at(&self, addr: u64, out: &mut [u8]) -> bool {
        let Ok(start) = usize::try_from(addr) else {
            return false;
        };
        let Some(end) = start.checked_add(out.len()) else {
            return false;
        };
        match self.bytes.get(start..end) {
            Some(src) => {
                out.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    fn slice_at(&self, addr: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(addr).ok()?;
        let end = start.checked_add(len)?;
        self.bytes.get(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_that_runs_past_the_end_fails_rather_than_truncating() {
        let backing = [1u8, 2, 3, 4];
        let mem = SliceMemory::new(&backing);
        let mut out = [0u8; 8];
        assert!(!mem.read_at(0, &mut out));
        assert!(!mem.read_at(3, &mut out[..2]));
        assert!(mem.read_at(2, &mut out[..2]));
        assert_eq!(&out[..2], &[3, 4]);
    }

    #[test]
    fn an_address_too_large_for_this_host_is_refused_rather_than_wrapped() {
        // The guest supplies these; on a 32-bit host `as usize` would truncate
        // u64::MAX to 0 and read the first bytes of the buffer instead.
        let backing = [0u8; 16];
        let mem = SliceMemory::new(&backing);
        let mut out = [0u8; 4];
        assert!(!mem.read_at(u64::MAX, &mut out));
        assert!(mem.slice_at(u64::MAX, 4).is_none());
        assert!(mem.slice_at(12, usize::MAX).is_none());
    }

    #[test]
    fn the_default_slice_at_refuses_every_borrow_so_callers_must_have_a_copy_path() {
        struct CopyOnly;
        impl GuestMemory for CopyOnly {
            fn read_at(&self, _addr: u64, out: &mut [u8]) -> bool {
                out.fill(0x5a);
                true
            }
        }
        assert!(CopyOnly.slice_at(0, 1).is_none());
        assert_eq!(CopyOnly.u32_at(0), Some(0x5a5a_5a5a));
    }

    /// The reference impl forwards `slice_at` rather than inheriting the
    /// refusing default — a borrowable implementation must stay borrowable
    /// through a `&dyn`.
    #[test]
    fn a_reference_forwards_all_three_methods() {
        let backing = [1u8, 2, 3, 4];
        let mem = SliceMemory::new(&backing);
        let dyn_mem: &dyn GuestMemory = &mem;
        let mut out = [0u8; 2];
        assert!((&dyn_mem).read_at(1, &mut out));
        assert_eq!(out, [2, 3]);
        assert_eq!((&dyn_mem).slice_at(0, 4), Some(&backing[..]));
        assert_eq!((&dyn_mem).u32_at(0), Some(u32::from_le_bytes(backing)));
    }

    #[test]
    fn u32_at_reads_little_endian_and_propagates_a_failed_read() {
        let backing = [0x2c, 0x00, 0x00, 0x00, 0xff];
        let mem = SliceMemory::new(&backing);
        assert_eq!(mem.u32_at(0), Some(44));
        assert_eq!(mem.u32_at(2), None);
    }
}
