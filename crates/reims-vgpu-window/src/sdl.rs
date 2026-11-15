//! A display's window through SDL3. SDL is the window library: it opens and sizes the window,
//! follows the screen's scale (fractional ones included), reports the window's changes, sets the
//! guest's cursor, and gives the surface the window is drawn into, so this crate speaks no window
//! system's protocol of its own but where an OS's extension does ([`os`]).
//!
//! - **Linux** (`native/sdl_thread/sdl.rs`, the only window there): Wayland's or X11's window,
//!   presented through Vulkan on the runtime's own device. SDL runs on a thread of the library's
//!   own, `reims-vgpu.ui` (`native/sdl_thread/ui.rs`), started with the first window or the first look
//!   at the host's screens: QEMU's main thread is never taken nor waited on. Work for it is posted
//!   and runs between its events; the window's input reaches the host's callbacks there, one at a
//!   time, in the order the events came (`native/sdl_thread/input.rs`, `native/sdl_thread/keymap.rs`). Each
//!   window has a render thread too, `reims-vgpu.window` (`native/sdl_thread/render.rs`), which draws
//!   the display's current frame into the window's swapchain through the display's sink.
//! - **macOS** (`native/macos/sdl.rs`, with `REIMS_VGPU_WINDOW_IMPL=sdl`): Cocoa's window, drawn
//!   into a `CAMetalLayer` in SDL's Metal view by the same render thread as AppKit's window
//!   (`native/macos/render.rs`). SDL runs on the main thread, which AppKit needs, inside the
//!   application the library runs there (`native/macos/app.rs`): SDL never runs a loop of its
//!   own, and its events are taken as they come. The window's input is AppKit's events, as the
//!   application dispatches them, to the same input as AppKit's window
//!   (`native/macos/input.rs`), and the extensions over SDL's `NSWindow` are AppKit's: where a
//!   window opens and how it follows the guest's mode, borderless windows, the camera housing's
//!   room, the menu and the end.
//!
//! Modules shared by both: [`shell`] (the windows open, each one's opening, size, title, layout,
//! events and end), [`cursor`] (the guest's cursor as SDL's) and [`units`] (SDL's sizes, points
//! or pixels, and a window's first size in them).

pub(crate) mod cursor;
pub(crate) mod shell;
pub(crate) mod units;

/// What the window does its own way on each OS: the window's making and placing, its surface,
/// its input and its end.
pub(crate) use crate::native::sdl_os as os;

use std::ffi::{CStr, CString};

/// SDL's last error, for a log line.
pub(crate) fn sdl_error() -> String {
    // SAFETY: SDL's error is a string it keeps for the thread, valid until its next call.
    unsafe { CStr::from_ptr(sdl3_sys::everything::SDL_GetError()) }
        .to_string_lossy()
        .into_owned()
}

/// A string SDL takes, from `text`, with any NUL dropped.
pub(crate) fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', "")).unwrap_or_default()
}
