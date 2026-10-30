//! [`RvgStatus`]: what a function that can fail answers, and its message.
//!
//! The codes are plain `u32` constants, so a value a host makes up is never a Rust enum. They
//! are additive: none is ever renumbered or removed, and a host treats an unknown one as a
//! failure.

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
/// struct's first version; a value outside its documented range; `add_trace_range` without
/// `remove_trace_range`.
pub const RVG_ERR_INVALID_ARGUMENT: u32 = 1;

/// A `struct_size` above the library's with a non-zero byte past the library's size.
pub const RVG_ERR_UNSUPPORTED: u32 = 2;

/// `display_port_count` is 0 or above RVG_MAX_DISPLAY_PORTS.
pub const RVG_ERR_DISPLAY_PORT_COUNT: u32 = 3;

/// No Metal device has `metal_registry_id`, or the system has none.
pub const RVG_ERR_NO_METAL_DEVICE: u32 = 4;

/// A memory map together with any of `create_task`, `destroy_task`, `map_memory`,
/// `unmap_memory` or `read_memory`.
pub const RVG_ERR_MEMORY_CONFLICT: u32 = 5;

/// The memory map was refused: ranges that overlap or overflow.
pub const RVG_ERR_MEMORY_MAP: u32 = 6;

/// `process_isolation` without a memory map.
pub const RVG_ERR_ISOLATION_WITHOUT_MAP: u32 = 7;

/// Metal refused an object the call needs: the device's blit pipelines (rvg_device_create), the
/// display's pipeline cache (rvg_display_create), a frame target's texture
/// (rvg_frame_target_create), or a frame's command buffer or the process's frame queue it comes
/// from (rvg_display_encode_frame).
pub const RVG_ERR_METAL: u32 = 8;

/// The display's port is not below the device's `display_port_count`.
pub const RVG_ERR_PORT_OUT_OF_RANGE: u32 = 9;

/// A display already holds the port.
pub const RVG_ERR_PORT_IN_USE: u32 = 10;

/// A display in this process already holds the serial number.
pub const RVG_ERR_SERIAL_IN_USE: u32 = 11;

/// An IOSurface device config without a memory map.
pub const RVG_ERR_NO_MEMORY_MAP: u32 = 12;

/// The device has no suspend state to give. The device serializes its state at every
/// rvg_device_finish_suspend, so no call returns it.
pub const RVG_ERR_SUSPEND: u32 = 13;

/// The device refused the resume state; the RvgResumeError says why when the device stored a
/// reason.
pub const RVG_ERR_RESUME: u32 = 14;

/// The frame's GPU work ended with an error.
pub const RVG_ERR_GPU: u32 = 15;

/// A frame committed twice, or waited on before its commit.
pub const RVG_ERR_FRAME_STATE: u32 = 16;

/// The host refused the memory, a pipe or a thread the call needs: the graphics device's region
/// or its threads (rvg_graphics_device_create).
pub const RVG_ERR_HOST_RESOURCES: u32 = 17;

/// The status of a Rust API error.
pub(crate) fn status(e: Error) -> RvgStatus {
    match e {
        Error::DisplayPortCount => RVG_ERR_DISPLAY_PORT_COUNT,
        Error::NoMetalDevice => RVG_ERR_NO_METAL_DEVICE,
        Error::MemoryConflict => RVG_ERR_MEMORY_CONFLICT,
        Error::MemoryMap => RVG_ERR_MEMORY_MAP,
        Error::IsolationWithoutMap => RVG_ERR_ISOLATION_WITHOUT_MAP,
        Error::Metal => RVG_ERR_METAL,
        Error::PortOutOfRange => RVG_ERR_PORT_OUT_OF_RANGE,
        Error::PortInUse => RVG_ERR_PORT_IN_USE,
        Error::SerialInUse => RVG_ERR_SERIAL_IN_USE,
        Error::NoMemoryMap => RVG_ERR_NO_MEMORY_MAP,
        Error::Suspend => RVG_ERR_SUSPEND,
        Error::Resume(_) => RVG_ERR_RESUME,
        Error::Gpu => RVG_ERR_GPU,
        Error::FrameState => RVG_ERR_FRAME_STATE,
        Error::HostResources => RVG_ERR_HOST_RESOURCES,
        // InvalidArgument, and any error a later Rust API adds before this layer names it.
        _ => RVG_ERR_INVALID_ARGUMENT,
    }
}

/// The messages, by code.
const MESSAGES: [&std::ffi::CStr; 18] = [
    c"success",
    c"invalid argument",
    c"unsupported struct size",
    c"display port count out of range",
    c"no such Metal device",
    c"memory map with task or memory callbacks",
    c"memory map refused",
    c"process isolation without a memory map",
    c"Metal refused an object",
    c"display port out of range",
    c"display port in use",
    c"display serial number in use",
    c"no memory map",
    c"no suspend state",
    c"resume state refused",
    c"GPU error",
    c"frame committed twice or waited on before its commit",
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

    use reims_vgpu::ResumeError;

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
        assert_eq!(message(RVG_ERR_FRAME_STATE), MESSAGES[16].to_str().unwrap());
        assert_eq!(
            message(RVG_ERR_HOST_RESOURCES),
            "the host refused memory, a pipe or a thread"
        );
        assert_eq!(message(18), "unknown status");
        assert_eq!(message(u32::MAX), "unknown status");
    }

    /// Each error has its code
    #[test]
    fn codes() {
        for (e, s) in [
            (Error::InvalidArgument, RVG_ERR_INVALID_ARGUMENT),
            (Error::DisplayPortCount, 3),
            (Error::NoMetalDevice, 4),
            (Error::MemoryConflict, 5),
            (Error::MemoryMap, 6),
            (Error::IsolationWithoutMap, 7),
            (Error::Metal, 8),
            (Error::PortOutOfRange, 9),
            (Error::PortInUse, 10),
            (Error::SerialInUse, 11),
            (Error::NoMemoryMap, 12),
            (Error::Suspend, 13),
            (Error::Resume(None), 14),
            (Error::Resume(Some(ResumeError::InvalidContent)), 14),
            (Error::Gpu, 15),
            (Error::FrameState, 16),
            (Error::HostResources, 17),
        ] {
            assert_eq!(status(e), s, "{e:?}");
        }
    }
}
