//! macOS shortcut capture via `NSApplication.presentationOptions`.
//!
//! # Contract, and its limit
//!
//! `NSApplicationPresentationDisableHideApplication` keeps Cmd+H from hiding
//! the presenting application. AppKit permits `DisableProcessSwitching` only
//! while the Dock is hidden or auto-hidden; this window must not change the
//! operator's Dock, so Cmd+Tab remains host-owned. That is the whole of what an
//! unprivileged, non-Dock-changing application may claim.
//!
//! It is **not** full capture, and this file does not pretend otherwise. The
//! window server keeps a small reserved set regardless — Cmd+Space for the
//! system search field, the screenshot chords, Ctrl+F-key accessibility
//! bindings, and anything the user has bound in Keyboard Shortcuts. Taking those
//! requires a `CGEventTap` at the HID tap point, which requires Accessibility
//! permission: a user-granted, per-application entitlement that cannot be
//! obtained programmatically and would silently produce no capture at all if
//! absent.
//!
//! So this reports [`CaptureError::PartialOnly`] on the first activation. The
//! guest gets Cmd+H; it does not get Cmd+Tab or the reserved set, and the
//! operator learns that from the fail log rather than from a guest that ignores
//! a keystroke. Nothing here infers the reserved set or works around it.

use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use super::{Capture, CaptureError};

/// `NSApplicationPresentationDisableHideApplication` — suppresses Cmd+H.
const DISABLE_HIDE_APPLICATION: usize = 1 << 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresentationChange {
    next: usize,
    added: usize,
}

/// Add the non-Dock-changing capture option.
///
/// `DisableProcessSwitching` is deliberately absent: AppKit would require this
/// application to hide or auto-hide the Dock alongside it. The delta remembers
/// only the bit introduced here so release does not clobber fullscreen/window
/// state.
fn engage_options(current: usize) -> PresentationChange {
    let added = DISABLE_HIDE_APPLICATION & !current;
    PresentationChange {
        next: current | added,
        added,
    }
}

/// Remove only the capture-owned bit.
fn release_options(current: usize, added: usize) -> usize {
    current & !added
}

/// Shortcut capture over the process's `NSApplication`.
///
/// There is one `NSApplication` per process and the host window is the only
/// thing in this process that presents, so this needs no window handle: the
/// presentation options are an application-wide property.
pub struct MacCapture {
    active: bool,
    /// Bits this capture added on activation. Release removes this delta rather
    /// than restoring a stale snapshot of the whole application property.
    added_options: usize,
    /// Whether the partial-capture refusal has already been reported. It is a
    /// standing limitation, not a per-transition event, so it is logged once.
    reported_partial: bool,
}

impl MacCapture {
    pub fn new() -> Self {
        Self {
            active: false,
            added_options: 0,
            reported_partial: false,
        }
    }

    /// Set `NSApp.presentationOptions`, preserving every bit this file does not
    /// own — a fullscreen presentation sets its own options through the same
    /// property, and clobbering them would take the window out of fullscreen.
    fn apply(&mut self, add: bool) {
        // SAFETY: `NSApp` is the process's shared application object; both the
        // getter and the setter are main-thread-only, and on this platform the
        // window's event loop *is* the process main thread (see
        // `present::run_main_thread`).
        unsafe {
            let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
            if app.is_null() {
                return;
            }
            let current: usize = msg_send![app, presentationOptions];
            let next = if add {
                let change = engage_options(current);
                self.added_options = change.added;
                change.next
            } else {
                release_options(current, self.added_options)
            };
            let _: () = msg_send![app, setPresentationOptions: next];
            if !add {
                self.added_options = 0;
            }
        }
    }
}

impl Default for MacCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl Capture for MacCapture {
    fn set(&mut self, active: bool) -> Result<(), CaptureError> {
        if active == self.active {
            return Ok(());
        }
        self.apply(active);
        self.active = active;
        if active && !self.reported_partial {
            self.reported_partial = true;
            // Say once, on the always-on channel, exactly which keystrokes the
            // guest will still not receive on this host.
            return Err(CaptureError::PartialOnly(
                "process_switching_and_window_server_reserved_chords",
            ));
        }
        Ok(())
    }

    fn describe(&self) -> &'static str {
        "macos_presentation_options"
    }
}

impl Drop for MacCapture {
    fn drop(&mut self) {
        if self.active {
            self.apply(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sized_window_capture_does_not_change_the_dock() {
        let change = engage_options(0);
        let dock_options = (1 << 0) | (1 << 1);
        assert_eq!(change.next, DISABLE_HIDE_APPLICATION);
        assert_eq!(change.next & dock_options, 0);
        assert_eq!(release_options(change.next, change.added), 0);
    }

    #[test]
    fn existing_presentation_options_are_preserved_and_not_owned() {
        let unrelated = 1 << 10; // NSApplicationPresentationFullScreen
        let current = (1 << 1) | unrelated; // NSApplicationPresentationHideDock
        let change = engage_options(current);
        assert_eq!(release_options(change.next, change.added), current);
    }

    #[test]
    fn release_preserves_options_added_by_another_owner() {
        let change = engage_options(0);
        let later = 1 << 10; // NSApplicationPresentationFullScreen
        assert_eq!(release_options(change.next | later, change.added), later);
    }
}
