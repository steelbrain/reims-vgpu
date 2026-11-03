//! The stage-3 command-buffer completion: when a command buffer finishes, the
//! host sends the guest an [`RgfxCbComplete`] carrying the GPU times and the
//! outcome. [`cb_complete`] builds it, mapping the host's
//! `MTLCommandBufferError` to our `RGFX_CB_*` status one for one (the IOGPU
//! error boundary the metal2vulkan lane measured): page fault, timeout (a GPU
//! hang), internal (an innocent recovery victim), access revoked (a queue's
//! submissions ignored after its second fault) and invalid resource. Every
//! other `MTLCommandBufferError` is reported as a generic internal error, with
//! the raw code kept for diagnostics.
//!
//! The host emits [`cb_complete`] from the decoder's completion sink once a
//! submitted command buffer's work lands, and [`cb_failed`] when the host could
//! not carry a command buffer out at all, so a guest blocked on it unblocks.

use reims_graphics::wire::{
    RGFX_CB_ACCESS_REVOKED, RGFX_CB_INTERNAL, RGFX_CB_INVALID_RESOURCE, RGFX_CB_OK,
    RGFX_CB_PAGE_FAULT, RGFX_CB_TIMEOUT, RGFX_EVT_CB_COMPLETE, RgfxCbComplete, RgfxMsgHeader,
};
use reims_vgpu_core::platform::types::NsErrorText;

/// `MTLCommandBufferErrorInternal`.
const MTL_CB_ERROR_INTERNAL: i64 = 1;
/// `MTLCommandBufferErrorTimeout`.
const MTL_CB_ERROR_TIMEOUT: i64 = 2;
/// `MTLCommandBufferErrorPageFault`.
const MTL_CB_ERROR_PAGE_FAULT: i64 = 3;
/// `MTLCommandBufferErrorAccessRevoked`.
const MTL_CB_ERROR_ACCESS_REVOKED: i64 = 4;
/// `MTLCommandBufferErrorInvalidResource`.
const MTL_CB_ERROR_INVALID_RESOURCE: i64 = 9;

/// The `RGFX_CB_*` status and the raw error code for a finished command buffer:
/// `None` is success, `Some` the host's command-buffer error. An error whose
/// code is not one of the mapped boundary cases is a generic internal error; the
/// raw `MTLCommandBufferError` code always travels in the second value for
/// diagnostics.
fn cb_status(error: Option<&NsErrorText>) -> (u32, u32) {
    let Some(e) = error else {
        return (RGFX_CB_OK, 0);
    };
    let status = match e.code {
        MTL_CB_ERROR_PAGE_FAULT => RGFX_CB_PAGE_FAULT,
        MTL_CB_ERROR_TIMEOUT => RGFX_CB_TIMEOUT,
        MTL_CB_ERROR_INTERNAL => RGFX_CB_INTERNAL,
        MTL_CB_ERROR_ACCESS_REVOKED => RGFX_CB_ACCESS_REVOKED,
        MTL_CB_ERROR_INVALID_RESOURCE => RGFX_CB_INVALID_RESOURCE,
        _ => RGFX_CB_INTERNAL,
    };
    (status, e.code as u32)
}

/// The completion message for the command buffer `cb_obj_id` finished at
/// submission `seqno`, with the GPU start and end times (the caller passes a 0
/// start for a recovery victim, which has none) and the outcome `error` (`None`
/// is success).
pub fn cb_complete(
    cb_obj_id: u64,
    seqno: u64,
    gpu_start_ns: u64,
    gpu_end_ns: u64,
    error: Option<&NsErrorText>,
) -> RgfxCbComplete {
    let (status, error_code) = cb_status(error);
    RgfxCbComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_COMPLETE,
            size: size_of::<RgfxCbComplete>() as u32,
            seqno,
        },
        cb_obj_id,
        gpu_start_ns,
        gpu_end_ns,
        status,
        error_code,
    }
}

/// The completion record for a command buffer the host could not carry out: a malformed op
/// stream, a reference the host does not hold, or an exec fault before the buffer was committed.
/// No GPU ran it and no completion handler will fire for it, so the host posts this itself — an
/// internal error with no GPU times — and the guest blocked on the buffer unblocks. Its code is
/// `MTLCommandBufferErrorInternal`'s, as [`cb_complete`]'s is for that status: the guest's buffer
/// carries the code as its error's, and a failed buffer whose error is
/// `MTLCommandBufferErrorNone` (0) is one Metal never gives.
pub fn cb_failed(cb_obj_id: u64, seqno: u64) -> RgfxCbComplete {
    RgfxCbComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_COMPLETE,
            size: size_of::<RgfxCbComplete>() as u32,
            seqno,
        },
        cb_obj_id,
        gpu_start_ns: 0,
        gpu_end_ns: 0,
        status: RGFX_CB_INTERNAL,
        error_code: MTL_CB_ERROR_INTERNAL as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb_error(code: i64) -> NsErrorText {
        NsErrorText {
            domain: String::from("MTLCommandBufferErrorDomain"),
            code,
            description: String::new(),
        }
    }

    /// A clean finish is OK with no error code; the header carries the type, the
    /// size and the submission seqno, and the GPU times pass through.
    #[test]
    fn a_clean_finish_is_ok() {
        let m = cb_complete(7, 42, 1_000, 2_000, None);
        assert_eq!(m.header.r#type, RGFX_EVT_CB_COMPLETE);
        assert_eq!(m.header.size as usize, size_of::<RgfxCbComplete>());
        assert_eq!(m.header.seqno, 42);
        assert_eq!(m.cb_obj_id, 7);
        assert_eq!((m.gpu_start_ns, m.gpu_end_ns), (1_000, 2_000));
        assert_eq!(m.status, RGFX_CB_OK);
        assert_eq!(m.error_code, 0);
    }

    /// Each measured boundary error maps to its own status, keeping the raw code.
    #[test]
    fn the_boundary_errors_map_one_for_one() {
        for (code, status) in [
            (MTL_CB_ERROR_PAGE_FAULT, RGFX_CB_PAGE_FAULT),
            (MTL_CB_ERROR_TIMEOUT, RGFX_CB_TIMEOUT),
            (MTL_CB_ERROR_INTERNAL, RGFX_CB_INTERNAL),
            (MTL_CB_ERROR_ACCESS_REVOKED, RGFX_CB_ACCESS_REVOKED),
            (MTL_CB_ERROR_INVALID_RESOURCE, RGFX_CB_INVALID_RESOURCE),
        ] {
            let m = cb_complete(1, 1, 0, 0, Some(&cb_error(code)));
            assert_eq!(m.status, status);
            assert_eq!(m.error_code, code as u32);
        }
    }

    /// A command buffer the host could not carry out fails as an internal error, with
    /// `MTLCommandBufferErrorInternal`'s code and no GPU times, never with code 0.
    #[test]
    fn a_buffer_the_host_could_not_carry_out_is_an_internal_error() {
        let m = cb_failed(7, 42);
        assert_eq!(m.header.r#type, RGFX_EVT_CB_COMPLETE);
        assert_eq!(m.header.size as usize, size_of::<RgfxCbComplete>());
        assert_eq!(m.header.seqno, 42);
        assert_eq!(m.cb_obj_id, 7);
        assert_eq!((m.gpu_start_ns, m.gpu_end_ns), (0, 0));
        assert_eq!(m.status, RGFX_CB_INTERNAL);
        assert_eq!(m.error_code, MTL_CB_ERROR_INTERNAL as u32);
        let internal = cb_complete(7, 42, 0, 0, Some(&cb_error(MTL_CB_ERROR_INTERNAL)));
        assert_eq!(
            (m.status, m.error_code),
            (internal.status, internal.error_code)
        );
    }

    /// An error outside the boundary cases (here out of memory, 8) is a generic
    /// internal error, and keeps its raw code for diagnostics.
    #[test]
    fn an_unmapped_error_is_internal() {
        let m = cb_complete(1, 1, 0, 0, Some(&cb_error(8)));
        assert_eq!(m.status, RGFX_CB_INTERNAL);
        assert_eq!(m.error_code, 8);
    }
}
