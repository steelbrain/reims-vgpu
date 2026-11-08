//! The host's description of guest memory: its ranges, as the host added them.

use crate::host::HostPtr;

/// One range of guest physical memory and where the host mapped it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GuestPhysicalRange {
    /// `physicalAddress`.
    pub physical_address: u64,
    /// `physicalLength`.
    pub physical_length: u64,
    /// `hostAddress`: where the host mapped it (`mach_vm_allocate`d by the host).
    pub host_address: HostPtr,
}

/// The host's ranges, in the order the host added them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryMapDescriptor {
    ranges: Vec<GuestPhysicalRange>,
}

impl MemoryMapDescriptor {
    /// No ranges.
    pub fn new() -> MemoryMapDescriptor {
        MemoryMapDescriptor { ranges: Vec::new() }
    }

    /// Appends a copy: no validation, no dedup; call order is kept.
    pub fn add_range(&mut self, range: GuestPhysicalRange) {
        self.ranges.push(range);
    }

    /// The ranges, in the order added.
    pub fn ranges(&self) -> &[GuestPhysicalRange] {
        &self.ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(pa: u64, len: u64, host: usize) -> GuestPhysicalRange {
        GuestPhysicalRange {
            physical_address: pa,
            physical_length: len,
            host_address: HostPtr::from_addr(host),
        }
    }

    #[test]
    fn new_is_empty() {
        assert!(MemoryMapDescriptor::new().ranges().is_empty());
        assert_eq!(MemoryMapDescriptor::new(), MemoryMapDescriptor::default());
    }

    #[test]
    fn add_range_keeps_order_and_duplicates() {
        let mut d = MemoryMapDescriptor::new();
        d.add_range(r(0x8000, 0x4000, 0x10000));
        d.add_range(r(0x4000, 0, 0));
        d.add_range(r(0x8000, 0x4000, 0x10000));
        d.add_range(r(u64::MAX, u64::MAX, 1));
        assert_eq!(
            d.ranges(),
            &[
                r(0x8000, 0x4000, 0x10000),
                r(0x4000, 0, 0),
                r(0x8000, 0x4000, 0x10000),
                r(u64::MAX, u64::MAX, 1),
            ]
        );
    }
}
