//! The host's values: addresses and physical ranges.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A host virtual address (`void *`): a range of the memory map, a no-copy buffer. Carries no
/// provenance of its own: a user turns it back into a pointer (`with_exposed_provenance`). NULL
/// is representable: the host passes it and core tests it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct HostPtr(usize);

impl HostPtr {
    /// `NULL`.
    pub const NULL: HostPtr = HostPtr(0);

    /// From an address.
    pub const fn from_addr(addr: usize) -> HostPtr {
        HostPtr(addr)
    }

    /// From a pointer, exposing its provenance (the metal and C crates' side).
    pub fn from_ptr<T>(p: *mut T) -> HostPtr {
        HostPtr(p.expose_provenance())
    }

    /// Back to a pointer with the exposed provenance. Creating it is safe; dereferencing it
    /// is its user's to make safe.
    pub fn as_ptr<T>(self) -> *mut T {
        std::ptr::with_exposed_provenance_mut(self.0)
    }

    /// The address.
    pub const fn addr(self) -> usize {
        self.0
    }

    /// Whether it is `NULL`.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// `self + off`, wrapping.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "lossless on the 64-bit target, and `ToUsize` cannot be called in a const fn"
    )]
    pub const fn wrapping_add(self, off: u64) -> HostPtr {
        HostPtr(self.0.wrapping_add(off as usize))
    }
}

/// One range of guest physical memory, `{physicalAddress, physicalLength}`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct PhysicalMemoryRange {
    /// `physicalAddress`.
    pub physical_address: u64,
    /// `physicalLength`.
    pub physical_length: u64,
}

/// The short name of [`PhysicalMemoryRange`].
pub type PhysRange = PhysicalMemoryRange;

const _: () = assert!(size_of::<PhysicalMemoryRange>() == 16);

/// Where the zero region ends: every arm64 macOS executable, QEMU included, maps
/// `__PAGEZERO` over `[0, 4 GiB)` (ld64's default `-pagezero_size`, `otool -l`), so an access below
/// it faults. Name is ours: the "NULL or near-NULL" access.
pub const ZERO_REGION_END: usize = 0x1_0000_0000;

#[cfg(test)]
mod tests {
    use super::*;

    /// HostPtr round-trips and wraps
    #[test]
    fn host_ptr() {
        let mut x = 5u32;
        let raw = std::ptr::from_mut(&mut x);
        let p = HostPtr::from_ptr(raw);
        assert!(!p.is_null());
        assert_eq!(p.as_ptr::<u32>(), raw);
        assert!(HostPtr::NULL.is_null());
        assert_eq!(HostPtr::from_addr(usize::MAX).wrapping_add(2).addr(), 1);
        assert_eq!(HostPtr::default(), HostPtr::NULL);
    }

    /// Its layout: the address, then the length
    #[test]
    fn phys_range_layout() {
        let bytes = [1u8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        let r = PhysicalMemoryRange::read_from_bytes(&bytes).ok();
        assert_eq!(
            r,
            Some(PhysRange {
                physical_address: 1,
                physical_length: 2
            })
        );
    }
}
