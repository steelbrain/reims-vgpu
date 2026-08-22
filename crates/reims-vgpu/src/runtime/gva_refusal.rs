//! Device failure vocabulary for page-table resolution statuses.

use reims_vgpu_paging::resolve::ResolveStatus;

pub fn slug(status: ResolveStatus) -> Option<&'static str> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_has_no_failure_name_and_every_failure_does() {
        assert_eq!(slug(ResolveStatus::Ok), None);
        for status in [
            ResolveStatus::ErrArgs,
            ResolveStatus::ErrInactiveTask,
            ResolveStatus::ErrNoDirectory,
            ResolveStatus::ErrDirectoryRead,
            ResolveStatus::ErrZeroRootPfn,
            ResolveStatus::ErrZeroDepth,
            ResolveStatus::ErrDepthTooDeep,
            ResolveStatus::ErrPageTableRead,
            ResolveStatus::ErrZeroPfn,
            ResolveStatus::ErrMalformedPte,
            ResolveStatus::ErrUnsupportedGeometry,
        ] {
            assert!(slug(status).is_some(), "missing slug for {status:?}");
        }
    }
}
