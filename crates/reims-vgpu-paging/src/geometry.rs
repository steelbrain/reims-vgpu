//! Page-table geometry names shared by the device and paging fixtures.
//!
//! The byte-format owner remains `reims-vgpu-wire`; these constants are direct
//! projections or explicit-width adaptations and therefore cannot drift from
//! the walk implementation.

use reims_vgpu_wire::page_table as wire;

pub const DIRECTORY_ROOT_PFN: u32 = wire::DIRECTORY_ROOT_PFN as u32;
pub const DIRECTORY_DEPTH: u32 = wire::DIRECTORY_DEPTH as u32;

pub const PAGE_SHIFT_ARM64E: u32 = wire::ARM64E.page_shift;
pub const PAGE_SIZE_ARM64E: u32 = wire::ARM64E.page_size() as u32;
pub const PAGE_SHIFT_X86: u32 = wire::X86_64.page_shift;
pub const PAGE_SIZE_X86: u32 = wire::X86_64.page_size() as u32;

/// Mapper page-list entry validity and PFN placement.
pub const MAPPER_PAGE_ENTRY_VALID: u32 = 0x1;
pub const MAPPER_PAGE_ENTRY_PFN_SHIFT: u32 = 2;

#[inline]
pub fn pfn_to_gpa(pfn: u32, page_shift: u32) -> u64 {
    (pfn as u64) << page_shift
}

#[inline]
pub fn page_size(page_shift: u32) -> u64 {
    1u64 << page_shift
}

pub fn span_page_count(min_size: u64, page_shift: u32) -> u64 {
    if min_size == 0 {
        1
    } else {
        ((min_size - 1) >> page_shift) + 1
    }
}

pub fn mapper_entry_gpa(entry: u32, page_shift: u32) -> Option<u64> {
    if (entry & MAPPER_PAGE_ENTRY_VALID) == 0 {
        return None;
    }
    Some(u64::from(entry >> MAPPER_PAGE_ENTRY_PFN_SHIFT) << page_shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pathways_keep_distinct_lossless_geometry() {
        assert_ne!(PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86);
        assert_ne!(PAGE_SIZE_ARM64E, PAGE_SIZE_X86);
        for geometry in [wire::X86_64, wire::ARM64E] {
            assert_eq!(geometry.page_size() as u32 as u64, geometry.page_size());
        }
        assert_eq!(DIRECTORY_ROOT_PFN as u64, wire::DIRECTORY_ROOT_PFN);
        assert_eq!(DIRECTORY_DEPTH as u64, wire::DIRECTORY_DEPTH);
    }

    #[test]
    fn pfn_conversion_requires_explicit_geometry() {
        for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
            let gpa = pfn_to_gpa(0x1234, shift);
            assert_eq!((gpa + 0x321) >> shift, 0x1234);
        }
    }

    #[test]
    fn mapper_entries_and_span_counts_take_explicit_page_geometry() {
        assert_eq!(span_page_count(0, PAGE_SHIFT_X86), 1);
        assert_eq!(span_page_count(1, PAGE_SHIFT_X86), 1);
        assert_eq!(
            span_page_count(u64::from(PAGE_SIZE_X86) + 1, PAGE_SHIFT_X86),
            2
        );
        assert_eq!(mapper_entry_gpa(0, PAGE_SHIFT_ARM64E), None);
        let entry = (5 << MAPPER_PAGE_ENTRY_PFN_SHIFT) | MAPPER_PAGE_ENTRY_VALID;
        assert_eq!(
            mapper_entry_gpa(entry, PAGE_SHIFT_ARM64E),
            Some(5u64 << PAGE_SHIFT_ARM64E)
        );
    }
}
