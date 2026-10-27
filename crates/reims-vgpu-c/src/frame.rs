//! Frames copied into host memory: [`RvgFrameTarget`], the encode into it, and [`RvgFrame`].

use std::ffi::c_void;

use parking_lot::RwLock;
use reims_vgpu::{Frame, FrameTarget, PixelFormat, Region};

use crate::Plat;
use crate::device::RvgDevice;
use crate::display::RvgDisplay;
use crate::ffi;
use crate::handle;
use crate::sized::{self, HostStruct};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_OK, RvgStatus, status};

/// A frame target's pixel format: one of RVG_PIXEL_FORMAT_*.
pub type RvgPixelFormat = u32;

// The constants below are RvgPixelFormat values, typed `u32` so that cbindgen writes them with the
// header's other constants, ahead of its types.

/// 8-bit RGBA, normalized (Metal's MTLPixelFormatRGBA8Unorm, 70).
pub const RVG_PIXEL_FORMAT_RGBA8_UNORM: u32 = 70;

/// 8-bit RGBA, sRGB (71).
pub const RVG_PIXEL_FORMAT_RGBA8_UNORM_SRGB: u32 = 71;

/// 8-bit BGRA, normalized (80): the byte order of a pixman x8r8g8b8 surface.
pub const RVG_PIXEL_FORMAT_BGRA8_UNORM: u32 = 80;

/// 8-bit BGRA, sRGB (81).
pub const RVG_PIXEL_FORMAT_BGRA8_UNORM_SRGB: u32 = 81;

/// Where a frame is drawn in its target, in pixels. The whole target is cleared first.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgRegion {
    pub x: u64,
    pub y: u64,
    pub width: u64,
    pub height: u64,
}

/// A frame target's config.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RvgFrameTargetConfig {
    /// sizeof(RvgFrameTargetConfig) as the host was compiled.
    pub struct_size: u32,
    /// 1 to 16384.
    pub width: u32,
    /// 1 to 16384.
    pub height: u32,
    /// One of RVG_PIXEL_FORMAT_*.
    pub format: RvgPixelFormat,
}

// SAFETY: repr(C), a u32 first; integers only.
unsafe impl HostStruct for RvgFrameTargetConfig {}

/// A 2D texture the library draws frames into and reads pixels back from.
pub struct RvgFrameTarget {
    target: FrameTarget<Plat>,
}

/// One encode of a display's current frame into a frame target: uncommitted, committed, or
/// complete.
pub struct RvgFrame {
    /// Committed under the write lock; waited on under the read lock, so waits may overlap.
    frame: RwLock<Frame<Plat>>,
}

/// A frame target on `device`'s Metal device: a 2D texture of `config`'s size and format that
/// the library can draw into and the CPU can read. RVG_ERR_INVALID_ARGUMENT for a size or format
/// out of range, RVG_ERR_METAL where Metal refuses the texture. The target holds no reference to
/// the device. Calls back: never.
///
/// # Safety
/// `device` is NULL or a live device handle; `config` is NULL or a live config; `target` is
/// NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_target_create(
    device: *mut RvgDevice,
    config: *const RvgFrameTargetConfig,
    target: *mut *mut RvgFrameTarget,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let c = match unsafe { sized::read(config) } {
            Ok(Some(c)) => c,
            Ok(None) => return RVG_ERR_INVALID_ARGUMENT,
            Err(s) => return s,
        };
        // SAFETY: the caller's contract.
        let Some(device) = (unsafe { handle::get(device) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        if target.is_null() {
            return RVG_ERR_INVALID_ARGUMENT;
        }
        let Some(format) = PixelFormat::from_metal(c.format) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        match device.device.new_frame_target(c.width, c.height, format) {
            Ok(t) => {
                let h = handle::new(RvgFrameTarget { target: t });
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { target.write(h) };
                RVG_OK
            }
            Err(e) => status(e),
        }
    })
}

/// One more reference to `target`; returns it.
///
/// # Safety
/// `target` is NULL or a live frame target.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_target_retain(
    target: *mut RvgFrameTarget,
) -> *mut RvgFrameTarget {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::retain(target) }
    })
}

/// Drops one reference to `target`; frames still drawing into it keep their own.
///
/// # Safety
/// `target` is NULL or a live frame target whose reference the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_target_release(target: *mut RvgFrameTarget) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(target) }
    })
}

/// Copies the whole target into `dst`: `height` rows of `width * 4` bytes, each row
/// `bytes_per_row` bytes after the last, in the target's format. It reads what the GPU has
/// written so far: wait for the frames that draw into it first. RVG_ERR_INVALID_ARGUMENT when
/// `bytes_per_row` is below `width * 4` or `dst_length` is below the copy's extent. Calls
/// back: never.
///
/// # Safety
/// `target` is NULL or a live frame target; `dst` is NULL or holds `dst_length` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_target_read_pixels(
    target: *const RvgFrameTarget,
    dst: *mut c_void,
    bytes_per_row: usize,
    dst_length: usize,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(t) = (unsafe { handle::get(target) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        if dst.is_null() {
            return RVG_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the caller's contract.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), dst_length) };
        match t.target.read_pixels(dst, bytes_per_row) {
            Ok(()) => RVG_OK,
            Err(e) => status(e),
        }
    })
}

/// Brings `dst` up to the target's pixels, laid out as rvg_frame_target_read_pixels lays them
/// out: each row that differs from `dst`'s is written and `changed[row]` set to 1; each equal row
/// is left as it is and `changed[row]` set to 0. A host that keeps `dst` as the last call left it
/// learns which rows changed since, and only those are written. Where the library reads the
/// display's frame in place (a frame on an IOSurface), each row is read once and nothing is
/// copied whole. It reads what the GPU has written so far: wait for the frames that draw into it
/// first. RVG_ERR_INVALID_ARGUMENT as for rvg_frame_target_read_pixels, and when
/// `changed_length` is below the target's height. Calls back: never.
///
/// # Safety
/// `target` is NULL or a live frame target; `dst` is NULL or holds `dst_length` writable bytes;
/// `changed` is NULL or holds `changed_length` writable bytes, apart from `dst`'s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_target_update_pixels(
    target: *const RvgFrameTarget,
    dst: *mut c_void,
    bytes_per_row: usize,
    dst_length: usize,
    changed: *mut u8,
    changed_length: usize,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(t) = (unsafe { handle::get(target) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        if dst.is_null() || changed.is_null() {
            return RVG_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the caller's contract.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), dst_length) };
        // SAFETY: the caller's contract.
        let changed = unsafe { std::slice::from_raw_parts_mut(changed, changed_length) };
        match t.target.update_pixels(dst, bytes_per_row, changed) {
            Ok(()) => RVG_OK,
            Err(e) => status(e),
        }
    })
}

/// Encodes the display's current frame into `target` over `region` (NULL: the whole target),
/// on the process's frame queue, and writes the uncommitted frame through `frame`. The display
/// counts the frame in flight until it completes; its presents wait for frames in flight. The
/// first encode after each new guest frame counts one host present. RVG_ERR_INVALID_ARGUMENT
/// for a target on another Metal device, RVG_ERR_METAL where Metal gives no command buffer.
/// Calls back: never.
///
/// # Safety
/// `display` is NULL or a live display handle; `target` is NULL or a live frame target;
/// `region` is NULL or a live region; `frame` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_encode_frame(
    display: *mut RvgDisplay,
    target: *mut RvgFrameTarget,
    region: *const RvgRegion,
    frame: *mut *mut RvgFrame,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let (Some(d), Some(t)) = (unsafe { handle::get(display) }, unsafe {
            handle::get(target)
        }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        if frame.is_null() {
            return RVG_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the caller's contract.
        let region = unsafe { region.as_ref() }.map(|r| Region {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        });
        match d.display.encode_frame(&t.target, region) {
            Ok(f) => {
                let h = handle::new(RvgFrame {
                    frame: RwLock::new(f),
                });
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { frame.write(h) };
                RVG_OK
            }
            Err(e) => status(e),
        }
    })
}

/// Submits the frame to the GPU. RVG_ERR_FRAME_STATE if it was committed already. Calls
/// back: never.
///
/// # Safety
/// `frame` is NULL or a live frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_commit(frame: *mut RvgFrame) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(f) = (unsafe { handle::get(frame) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        match f.frame.write().commit() {
            Ok(()) => RVG_OK,
            Err(e) => status(e),
        }
    })
}

/// Waits until the committed frame completes: RVG_OK, or RVG_ERR_GPU when the GPU work ended
/// with an error. RVG_ERR_FRAME_STATE before rvg_frame_commit. Calls back: never.
///
/// # Safety
/// `frame` is NULL or a live frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_wait(frame: *mut RvgFrame) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(f) = (unsafe { handle::get(frame) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        match f.frame.read().wait() {
            Ok(()) => RVG_OK,
            Err(e) => status(e),
        }
    })
}

/// Drops the frame. A committed frame completes on its own. An uncommitted one is never drawn,
/// and stays counted in flight for the display's life. Calls back: never.
///
/// # Safety
/// `frame` is NULL or a live frame, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_frame_release(frame: *mut RvgFrame) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(frame) }
    })
}

#[cfg(test)]
mod tests;
