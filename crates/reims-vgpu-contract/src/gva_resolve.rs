//! The walk's typed statuses on this device's failure channel.
//!
//! **This module is the impl below and nothing else.** It used to re-export the
//! whole of `reims_vgpu_paging::resolve` under device-local names, so a reader
//! at a call site could not tell whether `translate_root` was ours or the
//! crate's, and two of those names were renamed on the way through
//! (`ARM64E` arrived as `ARM64E_GEOMETRY`), which is a second vocabulary for one
//! set of items. Callers now name `reims_vgpu_paging` directly and the boundary
//! is visible in every `use` line that crosses it.
//!
//! What genuinely cannot move is here: the walk's statuses come from
//! `reims_vgpu_paging` and the fail channel's vocabulary comes from
//! `reims_vgpu_observe`, so neither crate can carry the mapping between them.
//! It is a function rather than a `Refusal` impl for the same reason — both
//! types are foreign to this crate, and Rust says so.
//!
//! The guest-memory seam is the wire crate's
//! [`GuestMemory`](reims_vgpu_wire::mem::GuestMemory); the device implements
//! it over `crate::runtime::host::HostMemory` at each caller.

use reims_vgpu_paging::resolve::ResolveStatus;

/// Every distinct check in the guest page-table walk, each with its own
/// slug.
///
/// They were already distinct *variants* — the walk has been honest about
/// which check refused since it was written. What was missing is that every
/// caller collapsed them all into one `MemError::Unmapped`, and
/// `MemError` reaches the always-on log at no site in the crate. So "the
/// guest asked for a GVA and we could not produce it" was
/// indistinguishable from "the directory PFN is zero", from "the PTE is
/// malformed", from "the span overflowed" — and none of them was visible at
/// all.
///
/// `gva_` prefix: these names (`args`, `zero_pfn`, `span_overflow`) are
/// generic enough to collide with half the crate.
pub fn walk_refusal(status: &ResolveStatus) -> Option<&'static str> {
    Some(match status {
        ResolveStatus::Ok => return None,
        ResolveStatus::ErrArgs => "gva_args",
        ResolveStatus::ErrInactiveTask => "gva_inactive_task",
        ResolveStatus::ErrNoDirectory => "gva_no_directory",
        ResolveStatus::ErrDirectoryRead => "gva_directory_read",
        ResolveStatus::ErrZeroRootPfn => "gva_zero_root_pfn",
        ResolveStatus::ErrZeroDepth => "gva_zero_depth",
        ResolveStatus::ErrDepthTooDeep => "gva_depth_too_deep",
        ResolveStatus::ErrPageTableRead => "gva_page_table_read",
        ResolveStatus::ErrZeroPfn => "gva_zero_pfn",
        ResolveStatus::ErrMalformedPte => "gva_malformed_pte",
        ResolveStatus::ErrUnsupportedGeometry => "gva_unsupported_geometry",
    })
}
