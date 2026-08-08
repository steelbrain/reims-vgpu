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
//! What genuinely cannot move is here: [`crate::observe::Refusal`] is this
//! crate's trait, so the impl that puts a walk status on the fail channel has to
//! be written on this side of the boundary.
//!
//! The guest-memory seam is the wire crate's
//! [`GuestMemory`](reims_vgpu_wire::mem::GuestMemory); the device implements
//! it over [`crate::runtime::host::HostMemory`] at each caller.

use reims_vgpu_paging::resolve::ResolveStatus;

impl crate::observe::Refusal for ResolveStatus {
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
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::Ok => return None,
            Self::ErrArgs => "gva_args",
            Self::ErrInactiveTask => "gva_inactive_task",
            Self::ErrNoDirectory => "gva_no_directory",
            Self::ErrDirectoryRead => "gva_directory_read",
            Self::ErrZeroRootPfn => "gva_zero_root_pfn",
            Self::ErrZeroDepth => "gva_zero_depth",
            Self::ErrDepthTooDeep => "gva_depth_too_deep",
            Self::ErrPageTableRead => "gva_page_table_read",
            Self::ErrZeroPfn => "gva_zero_pfn",
            Self::ErrMalformedPte => "gva_malformed_pte",
            Self::ErrUnsupportedGeometry => "gva_unsupported_geometry",
        })
    }
}
