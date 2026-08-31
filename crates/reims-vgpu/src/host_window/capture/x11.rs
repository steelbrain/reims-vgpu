//! X11 shortcut capture via `XGrabKeyboard`.
//!
//! # Contract
//!
//! `XGrabKeyboard(display, window, owner_events, pointer_mode, keyboard_mode,
//! time)` actively grabs the keyboard: while the grab is held, all key events
//! are reported to `window`, and — the part that matters here — the passive
//! grabs a window manager installed for its own chords do not trigger. It
//! returns `GrabSuccess` (0) or one of `AlreadyGrabbed` / `GrabInvalidTime` /
//! `GrabNotViewable` / `GrabFrozen`, which are refusals rather than errors and
//! must be reported as such. `XUngrabKeyboard` releases it.
//!
//! `owner_events = True` keeps events that would normally go to this window
//! being reported normally, so the window's own event handling is unchanged; it
//! is only the redirection of everything *else* that is added.
//!
//! Both modes are `GrabModeAsync` (1): a synchronous grab would require this
//! code to call `XAllowEvents` for every event to keep the keyboard unfrozen,
//! and freezing the X server's keyboard on a window that is also presenting is
//! how a desktop becomes unrecoverable.

use std::os::raw::{c_int, c_ulong};

use super::{Capture, CaptureError};

/// `GrabModeAsync` from `X.h`. Named rather than spelled inline because the
/// difference from `GrabModeSync` is a wedged desktop.
const GRAB_MODE_ASYNC: c_int = 1;
/// `CurrentTime` from `X.h`.
const CURRENT_TIME: c_ulong = 0;
/// `GrabSuccess` from `X.h`.
const GRAB_SUCCESS: c_int = 0;

/// Shortcut capture over the window's existing Xlib display.
pub struct X11Capture {
    xlib: x11_dl::xlib::Xlib,
    display: *mut x11_dl::xlib::Display,
    window: c_ulong,
    grabbed: bool,
}

// The display pointer is used only from the window's own event-loop thread,
// which is the thread that owns this value; it is never shared.
unsafe impl Send for X11Capture {}

impl X11Capture {
    pub fn new(
        display: raw_window_handle::XlibDisplayHandle,
        window: raw_window_handle::XlibWindowHandle,
    ) -> Result<Self, CaptureError> {
        let xlib = x11_dl::xlib::Xlib::open()
            .map_err(|e| CaptureError::Protocol(e.detail().to_string()))?;
        let display = display
            .display
            .ok_or(CaptureError::UnsupportedWindowSystem("xlib_no_display"))?;
        Ok(Self {
            xlib,
            display: display.as_ptr().cast(),
            window: window.window,
            grabbed: false,
        })
    }
}

impl Capture for X11Capture {
    fn set(&mut self, active: bool) -> Result<(), CaptureError> {
        if active == self.grabbed {
            return Ok(());
        }
        if active {
            // SAFETY: `display` and `window` come from the live window this
            // capture belongs to, and both calls are made only from the thread
            // that owns the event loop.
            let status = unsafe {
                (self.xlib.XGrabKeyboard)(
                    self.display,
                    self.window,
                    x11_dl::xlib::True,
                    GRAB_MODE_ASYNC,
                    GRAB_MODE_ASYNC,
                    CURRENT_TIME,
                )
            };
            if status != GRAB_SUCCESS {
                // A refused grab leaves the window manager owning the chords.
                // Reported, not retried: the usual cause is another client
                // holding the keyboard, and spinning on it would flood the log.
                return Err(CaptureError::GrabRefused(status as u8));
            }
            self.grabbed = true;
        } else {
            // SAFETY: as above.
            unsafe {
                (self.xlib.XUngrabKeyboard)(self.display, CURRENT_TIME);
                (self.xlib.XFlush)(self.display);
            }
            self.grabbed = false;
        }
        Ok(())
    }

    fn describe(&self) -> &'static str {
        "x11_grab_keyboard"
    }
}

impl Drop for X11Capture {
    fn drop(&mut self) {
        if self.grabbed {
            // Never leave an X keyboard grab behind: it would survive this
            // window and leave the whole session unable to type.
            // SAFETY: as in `set`.
            unsafe {
                (self.xlib.XUngrabKeyboard)(self.display, CURRENT_TIME);
                (self.xlib.XFlush)(self.display);
            }
        }
    }
}
