//! [`RvgStatus`]: what a function that can fail answers, and its message.
//!
//! The codes are plain `u32` constants, so a value a host makes up is never a Rust enum. A host
//! treats a code it does not know as a failure.

use std::ffi::c_char;

use reims_vgpu::Error;

use crate::ffi;

/// A function's result: RVG_OK or one of RVG_ERR_*.
pub type RvgStatus = u32;

// The constants below are RvgStatus values, typed `u32` so that cbindgen writes them with the
// header's other constants, ahead of its types.

/// Success.
pub const RVG_OK: u32 = 0;

/// A NULL handle, config or out-pointer where one is needed; a `struct_size` below the
/// struct's; a value outside its documented range.
pub const RVG_ERR_INVALID_ARGUMENT: u32 = 1;

/// A `struct_size` above the library's with a non-zero byte past the library's size.
pub const RVG_ERR_UNSUPPORTED: u32 = 2;

/// The platform refused an object the call needs: the display's texture or its draw
/// (rvg_graphics_device_read_display).
pub const RVG_ERR_METAL: u32 = 3;

/// The call's GPU work ended with an error.
pub const RVG_ERR_GPU: u32 = 4;

/// The host refused the memory, a pipe or a thread the call needs: the graphics device's region
/// or its threads (rvg_graphics_device_create).
pub const RVG_ERR_HOST_RESOURCES: u32 = 5;

/// The status of a Rust API error.
pub(crate) fn status(e: Error) -> RvgStatus {
    match e {
        Error::Metal => RVG_ERR_METAL,
        Error::Gpu => RVG_ERR_GPU,
        Error::HostResources => RVG_ERR_HOST_RESOURCES,
        // InvalidArgument, and any error a later Rust API adds before this layer names it.
        _ => RVG_ERR_INVALID_ARGUMENT,
    }
}

/// The messages, by code.
const MESSAGES: [&std::ffi::CStr; 6] = [
    c"success",
    c"invalid argument",
    c"unsupported struct size",
    c"the platform refused an object",
    c"GPU error",
    c"the host refused memory, a pipe or a thread",
];

/// A static, NUL-terminated description of `status`; "unknown status" for a value this
/// library does not define. Any thread, inside any callback.
#[unsafe(no_mangle)]
pub extern "C" fn rvg_status_message(status: RvgStatus) -> *const c_char {
    ffi::export(|| {
        usize::try_from(status)
            .ok()
            .and_then(|i| MESSAGES.get(i))
            .map_or(c"unknown status", |m| m)
            .as_ptr()
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    fn message(s: RvgStatus) -> &'static str {
        // SAFETY: every message is a static C string.
        unsafe { CStr::from_ptr(rvg_status_message(s)) }
            .to_str()
            .unwrap()
    }

    /// One message per code, and one for every other value
    #[test]
    fn messages() {
        assert_eq!(message(RVG_OK), "success");
        assert_eq!(message(RVG_ERR_GPU), MESSAGES[4].to_str().unwrap());
        assert_eq!(
            message(RVG_ERR_HOST_RESOURCES),
            "the host refused memory, a pipe or a thread"
        );
        assert_eq!(message(6), "unknown status");
        assert_eq!(message(u32::MAX), "unknown status");
    }

    /// Each error has its code
    #[test]
    fn codes() {
        for (e, s) in [
            (Error::InvalidArgument, RVG_ERR_INVALID_ARGUMENT),
            (Error::Metal, 3),
            (Error::Gpu, 4),
            (Error::HostResources, 5),
        ] {
            assert_eq!(status(e), s, "{e:?}");
        }
    }
}
