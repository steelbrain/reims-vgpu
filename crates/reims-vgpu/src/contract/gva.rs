//! GVA page-table geometry, as this device names it.
//!
//! Nothing here is declared. The format itself lives in
//! `reims_vgpu_wire::page_table`, which owns it — so every value below is either
//! re-exported from there or computed from it, and the two cannot drift because
//! there is only one of each. What this module adds is *this device's names for them*, which the
//! rest of the crate uses and which encode a rule the wire crate has no reason
//! to care about (see the arch-prefix note below).
//!
//! An earlier version of this file declared the same constants a second time,
//! with a test asserting the two agreed. That is the right remedy where the two
//! sides cannot see each other — the QEMU ABI header, which Rust does not
//! include — but both of these are Rust, so a re-export is strictly better: it
//! makes the drift impossible rather than detectable.

use reims_vgpu_wire::page_table as wire;

/// Offsets within a task's directory page. Narrowed from the wire crate's `u64`
/// because every consumer here indexes a `u32` field set.
pub const DIRECTORY_ROOT_PFN: u32 = wire::DIRECTORY_ROOT_PFN as u32;
pub const DIRECTORY_DEPTH: u32 = wire::DIRECTORY_DEPTH as u32;

// No `MAX_SPAN_PAGES`. There was one, `1 << 20`, whose doc said the guest's page
// table could describe a longer span and this device declined instead. No such
// decline existed: the value was carried as a `Geometry` field, set from the
// constant in both of the two geometries there are, compared against the same
// constant by `validate_geometry`, and dropped by `wire_geometry` before the
// walk ever saw it. A span of any length resolved, which is the faithful
// behaviour — so the constant is gone rather than the behaviour, and this note
// stands where a reader would otherwise "restore" a refusal that never ran.

pub const PAGE_SHIFT_ARM64E: u32 = wire::ARM64E.page_shift;
pub const PAGE_SIZE_ARM64E: u32 = wire::ARM64E.page_size() as u32;

pub const PAGE_SHIFT_X86: u32 = wire::X86_64.page_shift;
pub const PAGE_SIZE_X86: u32 = wire::X86_64.page_size() as u32;

// No `*_INDEX_BITS`, `*_INDEX_MASK`, `*_ENTRIES_PER_TABLE`, `*_PAGE_OFFSET_MASK`
// or `*_MAX_DEPTH`. Ten names, none of them read anywhere outside this file:
// every consumer of a level index or a fan-out is inside the walk, which takes a
// `wire::Geometry` and asks it. Re-exporting a derived value the device does not
// name only gives a future caller a second way to spell what the geometry
// already answers, at the arch it happens to be prefixed with rather than at the
// one the device booted on — which is the cross-arch bug the prefixes exist to
// prevent. Call the accessor on the geometry.

// No bare `PAGE_SHIFT`, `PAGE_SIZE`, `INDEX_BITS`, `INDEX_MASK` or
// `ENTRIES_PER_TABLE`. Every one of those silently meant arm64e and caused
// cross-arch bugs. Use the arch-prefixed name or the device `page_shift`.

/// PFN → GPA at an explicit guest page shift (12 or 14). No default.
///
/// `model::regs` re-exports this rather than restating it. It had its own copy
/// with the same body and the same doc, and the ring drains reached that one
/// while this one was reached by nothing but the round-trip test below — two
/// definitions of one shift, either of which could have been changed alone.
#[inline]
pub fn pfn_to_gpa(pfn: u32, page_shift: u32) -> u64 {
    (pfn as u64) << page_shift
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two pathways do not share a page size.
    ///
    /// What is left of a test that also asserted each pathway's page size was
    /// `1 << its shift`, its offset mask was `size - 1`, and its fan-out was
    /// `1 << its index bits`. Every one of those compared two values computed
    /// from the same `wire::Geometry` accessor, where the right-hand side *is*
    /// the left-hand side's definition — they could not fail. This line can:
    /// it is the one claim here about two different geometries, and a device
    /// that let the pathways collapse onto one page size would resolve every
    /// arm64e address at an x86 stride.
    #[test]
    fn the_two_pathways_do_not_share_a_page_size() {
        assert_ne!(PAGE_SIZE_ARM64E, PAGE_SIZE_X86);
        assert_ne!(PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86);
    }

    /// The wire crate states page geometry in `u64`; this module narrows it.
    ///
    /// The narrowing is silent, so it gets a test. There is no drift test here
    /// on purpose — these names are re-exports and computations, not a second
    /// declaration, so there is nothing that *can* disagree. What a widened
    /// page shift would do instead is truncate, and this is what catches that.
    #[test]
    fn narrowing_the_wire_geometry_to_this_devices_width_loses_nothing() {
        for g in [wire::X86_64, wire::ARM64E] {
            assert_eq!(g.page_size() as u32 as u64, g.page_size());
            assert_eq!(g.page_offset_mask() as u32 as u64, g.page_offset_mask());
            assert_eq!(g.index_mask() as u32 as u64, g.index_mask());
            assert_eq!(g.entries_per_table() as u32 as u64, g.entries_per_table());
        }
        assert_eq!(DIRECTORY_ROOT_PFN as u64, wire::DIRECTORY_ROOT_PFN);
        assert_eq!(DIRECTORY_DEPTH as u64, wire::DIRECTORY_DEPTH);
    }

    /// A PFN shifted to a GPA still names its own page at any offset inside it.
    ///
    /// Stated at both shifts because that is the whole reason `pfn_to_gpa` takes
    /// one: a helper that assumed 14 is what put x86 stamp writes on the wrong
    /// page. The inverse is written out rather than called, because no product
    /// path wanted a named `page_index` helper and an unused one is a second
    /// place this shift could be changed.
    #[test]
    fn a_pfn_shifted_to_a_gpa_names_its_own_page_at_either_shift() {
        for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
            let gpa = pfn_to_gpa(0x1234, shift);
            assert_eq!((gpa + 0x321) >> shift, 0x1234);
        }
    }
}
