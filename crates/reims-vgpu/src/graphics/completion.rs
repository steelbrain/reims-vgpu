//! The stage-3 command-buffer completion: when a command buffer finishes, the
//! host sends the guest an [`RgfxCbComplete`] carrying the GPU times and the
//! outcome. [`cb_complete`] builds it, mapping the host's
//! `MTLCommandBufferError` to our `RGFX_CB_*` status one for one (the IOGPU
//! error boundary the metal2vulkan lane measured): page fault, timeout (a GPU
//! hang), internal (an innocent recovery victim), access revoked (a queue's
//! submissions ignored after its second fault) and invalid resource. Every
//! other code of Metal's `MTLCommandBufferError` (not permitted, out of memory,
//! memoryless, device removed, stack overflow) is reported as a generic
//! internal status with its own code, which the guest's error carries. An error
//! Metal never gives a failed buffer (another domain's, `None`'s code 0, or a
//! code `MTLCommandBufferError` does not name) is Metal's internal error, "an
//! internal error that doesn't fit into the other categories" (Metal's
//! `MTLCommandBuffer.h`), with `MTLCommandBufferErrorInternal`'s code.
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
/// `MTLCommandBufferErrorNotPermitted`.
const MTL_CB_ERROR_NOT_PERMITTED: i64 = 7;
/// `MTLCommandBufferErrorOutOfMemory`.
const MTL_CB_ERROR_OUT_OF_MEMORY: i64 = 8;
/// `MTLCommandBufferErrorInvalidResource`.
const MTL_CB_ERROR_INVALID_RESOURCE: i64 = 9;
/// `MTLCommandBufferErrorMemoryless`.
const MTL_CB_ERROR_MEMORYLESS: i64 = 10;
/// `MTLCommandBufferErrorDeviceRemoved`.
const MTL_CB_ERROR_DEVICE_REMOVED: i64 = 11;
/// `MTLCommandBufferErrorStackOverflow`.
const MTL_CB_ERROR_STACK_OVERFLOW: i64 = 12;
/// `MTLCommandBufferErrorDomain`, the domain of every error Metal gives a failed command buffer.
const MTL_CB_ERROR_DOMAIN: &str = "MTLCommandBufferErrorDomain";

/// The `RGFX_CB_*` status and the `MTLCommandBufferError` code for a finished command buffer:
/// `None` is success, `Some` the host's command-buffer error. An error of Metal's domain and
/// one of its codes keeps the code, its status that of its boundary case or a generic internal
/// one; any other error is `MTLCommandBufferErrorInternal`, as Metal reports a failure that
/// fits none of its codes.
fn cb_status(error: Option<&NsErrorText>) -> (u32, u32) {
    let Some(e) = error else {
        return (RGFX_CB_OK, 0);
    };
    let status = match e.code {
        _ if e.domain != MTL_CB_ERROR_DOMAIN => None,
        MTL_CB_ERROR_PAGE_FAULT => Some(RGFX_CB_PAGE_FAULT),
        MTL_CB_ERROR_TIMEOUT => Some(RGFX_CB_TIMEOUT),
        MTL_CB_ERROR_INTERNAL => Some(RGFX_CB_INTERNAL),
        MTL_CB_ERROR_ACCESS_REVOKED => Some(RGFX_CB_ACCESS_REVOKED),
        MTL_CB_ERROR_INVALID_RESOURCE => Some(RGFX_CB_INVALID_RESOURCE),
        MTL_CB_ERROR_NOT_PERMITTED
        | MTL_CB_ERROR_OUT_OF_MEMORY
        | MTL_CB_ERROR_MEMORYLESS
        | MTL_CB_ERROR_DEVICE_REMOVED
        | MTL_CB_ERROR_STACK_OVERFLOW => Some(RGFX_CB_INTERNAL),
        _ => None,
    };
    match status {
        // Each of Metal's codes fits in a u32.
        Some(status) => (status, e.code as u32),
        None => {
            log::debug!(
                target: "reims_vgpu::graphics",
                "a command buffer failed with {} code {}, which Metal never gives one: it fails \
                 as Metal's internal error",
                e.domain,
                e.code
            );
            (RGFX_CB_INTERNAL, MTL_CB_ERROR_INTERNAL as u32)
        }
    }
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

    /// Metal's codes outside the boundary cases (not permitted, out of memory, memoryless, device
    /// removed, stack overflow) are a generic internal status, and keep their own code, which the
    /// guest's error carries.
    #[test]
    fn metal_s_other_codes_are_internal_with_their_own_code() {
        for code in [
            MTL_CB_ERROR_NOT_PERMITTED,
            MTL_CB_ERROR_OUT_OF_MEMORY,
            MTL_CB_ERROR_MEMORYLESS,
            MTL_CB_ERROR_DEVICE_REMOVED,
            MTL_CB_ERROR_STACK_OVERFLOW,
        ] {
            let m = cb_complete(1, 1, 0, 0, Some(&cb_error(code)));
            assert_eq!((m.status, m.error_code), (RGFX_CB_INTERNAL, code as u32));
        }
    }

    /// An error Metal never gives a failed command buffer is Metal's internal error, with
    /// `MTLCommandBufferErrorInternal`'s code: another domain's (whatever its code, one of
    /// Metal's among them), `MTLCommandBufferErrorNone`'s 0, and codes `MTLCommandBufferError`
    /// does not name, negative or past `u32` among them.
    #[test]
    fn an_error_metal_never_gives_is_metal_s_internal_error() {
        let other = |domain: &str, code| NsErrorText {
            domain: domain.to_owned(),
            code,
            description: String::new(),
        };
        let internal = (RGFX_CB_INTERNAL, MTL_CB_ERROR_INTERNAL as u32);
        for e in [
            other("NSPOSIXErrorDomain", 3),
            other("IOGPUCommandQueueErrorDomain", 9),
            other("", 2),
            cb_error(0),
            cb_error(5),
            cb_error(6),
            cb_error(13),
            cb_error(-1),
            cb_error(1 << 40),
        ] {
            let m = cb_complete(1, 1, 0, 0, Some(&e));
            assert_eq!((m.status, m.error_code), internal, "{e:?}");
        }
    }
}
