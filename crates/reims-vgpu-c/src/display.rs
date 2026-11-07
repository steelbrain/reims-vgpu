//! Displays: [`RvgDisplayConfig`], [`RvgDisplayCallbacks`], and the `rvg_display_*` functions
//! other than the frame encode ([`crate::frame`]).
//!
//! A display's callbacks run on the process's display thread, one at a time across every
//! display, in the order core queued them (docs/api.md section 5). Each calls the host's
//! function under `ffi::host_call`. A display with a window has a second table, the window's
//! ([`crate::window`]), which its handle holds beside the display's.

use std::ffi::{c_char, c_void};
use std::sync::Arc;

use reims_vgpu::{
    CursorGlyph, CursorGlyphFn, CursorMoveFn, CursorShowFn, Display, DisplayCallbacks,
    DisplayConfig, DisplayMode, ModeChangeFn, NewFrameFn,
};

use crate::device::RvgDevice;
use crate::ffi;
use crate::handle;
use crate::sized::{self, HostStruct};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_OK, RvgStatus, status};
use crate::table::{RvgDestroyFn, Table, WithTable};
use crate::window::{RvgWindowConfig, window_callbacks};
use crate::{Plat, null_handle};

/// The mode supports EDR: the guest reads the flag in the mode's entry.
pub const RVG_DISPLAY_MODE_EDR: u32 = 1;

/// The mode supports 10 bits per channel: the guest reads the flag in the mode's entry.
pub const RVG_DISPLAY_MODE_10BPC: u32 = 2;

/// A display mode offered to the guest.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgDisplayMode {
    pub width: u16,
    pub height: u16,
    /// RVG_DISPLAY_MODE_* bits; 0 for a plain mode.
    pub flags: u32,
    pub refresh_rate_hz: f64,
}

impl RvgDisplayMode {
    /// The Rust API's mode; bits the library does not define are ignored.
    fn to_mode(self) -> DisplayMode {
        DisplayMode {
            width: self.width,
            height: self.height,
            refresh_rate_hz: self.refresh_rate_hz,
            supports_edr: self.flags & RVG_DISPLAY_MODE_EDR != 0,
            supports_10bpc: self.flags & RVG_DISPLAY_MODE_10BPC != 0,
        }
    }
}

/// A position or size in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgDisplayCoord {
    pub x: u16,
    pub y: u16,
}

/// `mode_change(ctx, width, height, pixel_format)`: the guest set a mode. `pixel_format` is the
/// guest's IOSurface pixel format code.
///
/// Thread: the library's display thread, a serial worker that runs every display callback of
/// the process one at a time, in the order the library queued them, and never inside a call of
/// the host's. After 5 s idle it exits, and the next callback starts a new one.
/// May call: every function of this header. A display callback may block (the host's big lock,
/// for example): the library never waits for this thread, and the next display callback waits
/// until it returns.
/// Without it, the guest's mode changes are not reported.
pub type RvgModeChangeFn = Option<
    unsafe extern "C-unwind" fn(ctx: *mut c_void, width: u16, height: u16, pixel_format: u32),
>;

/// `new_frame(ctx)`: the guest presented a frame since the last call (calls coalesce).
///
/// Thread, and what it may call: as RvgModeChangeFn. Without it, the EFI display's first present
/// to this display ends the process, and a guest's present to it completes with no call.
pub type RvgNewFrameFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// The cursor glyph, borrowed for the call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgCursorGlyph {
    /// sizeof(RvgCursorGlyph) as the library was compiled.
    pub struct_size: u32,
    pub width: u32,
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: RvgDisplayCoord,
    /// `width * 4` bytes a row, `height` rows: the guest's BGRA pixels with red and blue swapped.
    pub rgba: *const u8,
    pub rgba_length: usize,
}

/// `cursor_glyph(ctx, glyph)`: the guest set a cursor glyph.
///
/// Thread, and what it may call: as RvgModeChangeFn. Without it, the guest reads a cursor glyph
/// size limit of 0, and 256 with it.
pub type RvgCursorGlyphFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, glyph: *const RvgCursorGlyph)>;

/// `cursor_show(ctx, show)`: the guest showed or hid the cursor. `show` is the guest's byte as
/// written: 0 hides, any other value shows.
///
/// Thread, and what it may call: as RvgModeChangeFn. The guest reads whether the display has it.
pub type RvgCursorShowFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, show: u8)>;

/// `cursor_move(ctx)`: the guest moved the cursor; rvg_display_cursor_position reads where.
///
/// Thread, and what it may call: as RvgModeChangeFn. The guest reads whether the display has it.
pub type RvgCursorMoveFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// A display's callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgDisplayCallbacks {
    /// sizeof(RvgDisplayCallbacks) as the host was compiled.
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Runs once after the display's end, when no callback of the table runs again.
    pub destroy: RvgDestroyFn,
    pub mode_change: RvgModeChangeFn,
    pub new_frame: RvgNewFrameFn,
    pub cursor_glyph: RvgCursorGlyphFn,
    pub cursor_show: RvgCursorShowFn,
    pub cursor_move: RvgCursorMoveFn,
}

// SAFETY: repr(C), a u32 first; a pointer and optional function pointers.
unsafe impl HostStruct for RvgDisplayCallbacks {}

/// A display's config.
#[repr(C)]
#[derive(Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgDisplayConfig {
    /// sizeof(RvgDisplayConfig) as the host was compiled.
    pub struct_size: u32,
    /// The guest's serial number for the display, unique in the process.
    pub serial_number: u32,
    /// The port, below the device's `display_port_count`.
    pub port: u64,
    /// UTF-8, `name_length` bytes, not NUL-terminated; invalid sequences become U+FFFD. NULL is
    /// "Apple Virtual".
    pub name: *const c_char,
    pub name_length: usize,
    pub width_millimeters: f64,
    pub height_millimeters: f64,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgDisplayCallbacks,
    /// NULL is no window. Otherwise the library opens a window for the display (RvgWindowConfig
    /// says what the display and the host then see).
    pub window: *const RvgWindowConfig,
}

// SAFETY: repr(C), a u32 first; integers, floats and pointers.
unsafe impl HostStruct for RvgDisplayConfig {
    /// The first version ends at `callbacks`.
    const V1_SIZE: usize = 56;
}

const _: () = assert!(std::mem::offset_of!(RvgDisplayConfig, window) == RvgDisplayConfig::V1_SIZE);

/// A display on one of the device's ports.
pub struct RvgDisplay {
    /// The window's table, for a display with a window: held here so that a table with no
    /// function still has its destroy at the release. It goes before the display, whose window
    /// holds the table's other references until it closes.
    window_table: Option<Arc<Table>>,
    pub(crate) display: WithTable<Display<Plat>>,
}

impl Drop for RvgDisplay {
    /// The window's table first, then the display: with functions, the window's closures, which
    /// the display's last reference lets go of when it closes the window, hold its last
    /// references.
    fn drop(&mut self) {
        drop(self.window_table.take());
    }
}

/// The Rust API's closures over a display table: each holds the table, and calls its function
/// with the table's `ctx`.
fn display_callbacks(t: &Arc<Table>, c: &RvgDisplayCallbacks) -> DisplayCallbacks {
    DisplayCallbacks {
        mode_change: c.mode_change.map(|f| {
            let t = t.clone();
            let c: ModeChangeFn = Arc::new(move |width, height, format| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("mode_change", || unsafe {
                    f(t.ctx(), width, height, format)
                });
            });
            c
        }),
        new_frame: c.new_frame.map(|f| {
            let t = t.clone();
            let c: NewFrameFn = Arc::new(move || {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("new_frame", || unsafe { f(t.ctx()) });
            });
            c
        }),
        cursor_glyph: cursor_glyph(t, c.cursor_glyph),
        cursor_show: cursor_show(t, c.cursor_show),
        cursor_move: cursor_move(t, c.cursor_move),
    }
}

/// The Rust API's closure over a table's `cursor_glyph`, a display's or a graphics device's: it
/// holds the table, and lends the host the glyph for the call.
pub(crate) fn cursor_glyph(t: &Arc<Table>, f: RvgCursorGlyphFn) -> Option<CursorGlyphFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorGlyphFn = Arc::new(move |g: &CursorGlyph<'_>| {
            let glyph = RvgCursorGlyph {
                struct_size: sized::size_u32::<RvgCursorGlyph>(),
                width: g.width,
                height: g.height,
                hot_spot: RvgDisplayCoord {
                    x: g.hot_spot.0,
                    y: g.hot_spot.1,
                },
                rgba: g.rgba.as_ptr(),
                rgba_length: g.rgba.len(),
            };
            let glyph = &raw const glyph;
            // SAFETY: the host's function over its ctx; the glyph and its pixels are live for the
            // call.
            ffi::host_call("cursor_glyph", || unsafe { f(t.ctx(), glyph) });
        });
        c
    })
}

/// The Rust API's closure over a table's `cursor_show`, as [`cursor_glyph`]'s.
pub(crate) fn cursor_show(t: &Arc<Table>, f: RvgCursorShowFn) -> Option<CursorShowFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorShowFn = Arc::new(move |show| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("cursor_show", || unsafe { f(t.ctx(), show) });
        });
        c
    })
}

/// The Rust API's closure over a table's `cursor_move`, as [`cursor_glyph`]'s.
pub(crate) fn cursor_move(t: &Arc<Table>, f: RvgCursorMoveFn) -> Option<CursorMoveFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorMoveFn = Arc::new(move || {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("cursor_move", || unsafe { f(t.ctx()) });
        });
        c
    })
}

/// The display behind a handle, or the process's end for NULL.
///
/// # Safety
/// `display` is NULL or a live display handle.
unsafe fn live<'a>(display: *const RvgDisplay, call: &'static str) -> &'a RvgDisplay {
    // SAFETY: the caller's contract.
    unsafe { handle::get(display) }.unwrap_or_else(|| null_handle(call))
}

/// A new display on `device`. Refusals, checked in this order: RVG_ERR_INVALID_ARGUMENT,
/// RVG_ERR_UNSUPPORTED, RVG_ERR_PORT_OUT_OF_RANGE, RVG_ERR_METAL, RVG_ERR_SERIAL_IN_USE,
/// RVG_ERR_PORT_IN_USE. The display keeps the device alive. May call back.
///
/// The config's `window`, and its callbacks, are read with the config, before either table is
/// the library's: a `struct_size` refusal of any of the four structs runs no destroy, and every
/// later refusal runs both tables' destroys before it returns, the window's first. With a
/// window, the display has all five display callbacks as the guest reads them (a cursor glyph
/// size limit of 256, and each cursor callback), whichever of `callbacks` the host gives, and
/// calls those it gives. With RVG_WINDOW_MATCH_HOST_SCREEN, the display's size in millimetres,
/// its scale factor and its first modes are those of the window's host screen, where there is
/// one, or, with the window config's mode, that mode's at a 1× density or the config's
/// `mode_pixels_per_inch`, and the screen's refresh rate; a later rvg_display_set_modes
/// replaces the modes. With no screen, the size, scale factor and first modes are those of the
/// config's mode, or of 1920 × 1080 at 1× without one, at 60 Hz.
///
/// # Safety
/// `device` is NULL or a live device handle; `config` is NULL or a live config; `display` is
/// NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_create(
    device: *mut RvgDevice,
    config: *const RvgDisplayConfig,
    display: *mut *mut RvgDisplay,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let c = match unsafe { sized::read(config) } {
            Ok(Some(c)) => c,
            Ok(None) => return RVG_ERR_INVALID_ARGUMENT,
            Err(s) => return s,
        };
        // SAFETY: the config's contract: NULL or a live table.
        let callbacks = match unsafe { sized::read_or_zero(c.callbacks) } {
            Ok(t) => t,
            Err(s) => return s,
        };
        // SAFETY: the config's contract: NULL or a live window config.
        let window = match unsafe { sized::read(c.window) } {
            Ok(w) => w,
            Err(s) => return s,
        };
        let window = match window {
            None => None,
            // SAFETY: the window config's contract: NULL or a live table.
            Some(w) => match unsafe { sized::read_or_zero(w.callbacks) } {
                Ok(t) => Some((w, t)),
                Err(s) => return s,
            },
        };
        // From here the tables are the library's: a refusal ends them before it returns, the
        // window's first, as a release does.
        let table = Table::new(callbacks.ctx, callbacks.destroy);
        let window_table = window.as_ref().map(|(_, t)| Table::new(t.ctx, t.destroy));
        let refuse = |s: RvgStatus| {
            if let Some(t) = &window_table {
                t.end();
            }
            table.end();
            s
        };
        // SAFETY: the caller's contract.
        let Some(device) = (unsafe { handle::get(device) }) else {
            return refuse(RVG_ERR_INVALID_ARGUMENT);
        };
        if display.is_null() {
            return refuse(RVG_ERR_INVALID_ARGUMENT);
        }
        let name = (!c.name.is_null()).then(|| {
            // SAFETY: the config's contract: `name_length` bytes at `name`.
            let bytes = unsafe { std::slice::from_raw_parts(c.name.cast::<u8>(), c.name_length) };
            String::from_utf8_lossy(bytes).into_owned()
        });
        let mut config = DisplayConfig::new(c.port, c.serial_number);
        config.name = name;
        config.size_in_millimeters = (c.width_millimeters, c.height_millimeters);
        config.callbacks = display_callbacks(&table, &callbacks);
        if let (Some((w, wc)), Some(t)) = (&window, &window_table) {
            config.window = Some(w.to_config(window_callbacks(t, wc)));
        }
        match device.device.new_display(config) {
            Ok(d) => {
                let h = handle::new(RvgDisplay {
                    window_table,
                    display: WithTable::new(d, table),
                });
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { display.write(h) };
                RVG_OK
            }
            Err(e) => refuse(status(e)),
        }
    })
}

/// Drops the host's reference; the last one unplugs the display from the guest, and may be the
/// device's last reference. May block, and may call back. A display with a window closes it
/// first, waiting neither for the main thread nor for the window's frames: the window table's
/// destroy runs here, or once a window callback running on the main thread returns, and a
/// frame the window has in flight keeps the display until it completes, as an RvgFrame does.
///
/// # Safety
/// `display` is NULL or a live display handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_release(display: *mut RvgDisplay) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(display) }
    })
}

/// Offers the guest `count` modes, in order (copied). `count` 0 changes nothing, and `modes`
/// may then be NULL; a NULL `modes` with a `count` above 0 ends the process. The guest's display
/// page for the port takes the first 128: the first list it takes connects the display to the
/// guest, and a later one that differs updates it. Until the guest gives the port a page, the
/// list waits for it. The EFI mode registers read port 0's list. May call back.
///
/// # Safety
/// `display` is a live display handle; `modes` holds `count` modes (it may be NULL when `count`
/// is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_set_modes(
    display: *mut RvgDisplay,
    modes: *const RvgDisplayMode,
    count: usize,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let d = unsafe { live(display, "rvg_display_set_modes") };
        // Core is given the empty list too: it changes nothing there.
        let modes: &[RvgDisplayMode] = if count == 0 {
            &[]
        } else if modes.is_null() {
            null_handle("rvg_display_set_modes")
        } else {
            // SAFETY: the caller's contract.
            unsafe { std::slice::from_raw_parts(modes, count) }
        };
        let modes: Vec<DisplayMode> = modes.iter().map(|m| m.to_mode()).collect();
        d.display.set_modes(&modes);
    })
}

/// The cursor's position; {65535, 65535} before the guest has linked the display. While the
/// device is paused (rvg_device_pause, or a finished suspend), a display the guest has linked
/// ends the process here. Calls back: never.
///
/// # Safety
/// `display` is a live display handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_cursor_position(
    display: *const RvgDisplay,
) -> RvgDisplayCoord {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let (x, y) = unsafe { live(display, "rvg_display_cursor_position") }
            .display
            .cursor_position();
        RvgDisplayCoord { x, y }
    })
}

/// The guest's presents, and the host's encodes of a new frame, counted so far. Either pointer
/// may be NULL. Calls back: never.
///
/// # Safety
/// `display` is a live display handle; `guest` and `host` are each NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_display_present_counts(
    display: *const RvgDisplay,
    guest: *mut u64,
    host: *mut u64,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let d = unsafe { live(display, "rvg_display_present_counts") };
        // SAFETY: the caller's contract.
        unsafe {
            handle::write_out(guest, d.display.guest_present_count());
            handle::write_out(host, d.display.host_present_count());
        }
    })
}

#[cfg(test)]
mod tests;
