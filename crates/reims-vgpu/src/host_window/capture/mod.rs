//! Asking the host desktop to stop consuming keyboard shortcuts before the
//! host-owned window sees them ([[host-window]]).
//!
//! # The contract this implements
//!
//! A desktop compositor or window manager claims chords for itself and acts on
//! them *instead of* delivering them to the focused client. On the session this
//! was recovered from, 63 `Meta`/`Alt`/`Ctrl` chords were registered as global
//! shortcuts, among them bare `Meta`, `Meta+A`, `Meta+V`, `Meta+Q`, `Meta+W`,
//! `Meta+T`, `Meta+0`..`Meta+9` and `Alt+Tab`. A macOS guest reads host `Meta`
//! as `Cmd`, so that list is close to the guest's entire shortcut surface. None
//! of it reaches [`super::input_map`]; the keys are gone before any code in this
//! crate runs.
//!
//! Each window system has a defined way to ask for them back:
//!
//! - **Wayland** — `keyboard-shortcuts-inhibit-unstable-v1`. Creating a
//!   `zwp_keyboard_shortcuts_inhibitor_v1` for a (surface, seat) obliges the
//!   compositor to forward all key events from that seat to that surface rather
//!   than acting on its own shortcuts; destroying it restores them.
//! - **X11** — `XGrabKeyboard`, which redirects key events to the grabbing
//!   window and overrides the window manager's passive grabs for its duration.
//! - **macOS** — `NSApplicationPresentationDisableHideApplication`, which keeps
//!   Cmd+H from hiding the presenting application without changing Dock
//!   visibility. AppKit makes Dock hiding a prerequisite for suppressing
//!   Cmd+Tab, so process switching remains host-owned; see [`macos`] for the
//!   typed partial-capture result.
//!
//! # Refusal, not partial capture
//!
//! Every platform here can fail to deliver: a compositor need not implement the
//! inhibit protocol, an X server can hand back `AlreadyGrabbed`, macOS keeps
//! some chords regardless. None of those are allowed to be silent. A capture
//! that did not take emits a typed reason on the always-on fail channel and the
//! window keeps running with whatever the desktop leaves it — which is exactly
//! today's behaviour, now named. [`Capture::describe`] carries what actually
//! happened so a boot log says which of the three it got.

use crate::observe::Decline;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(all(unix, not(target_os = "macos")))]
pub mod wayland;
#[cfg(all(unix, not(target_os = "macos")))]
pub mod x11;

/// Why a capture request could not be honoured.
///
/// Every variant is a real refusal the operator should be able to read out of
/// `/tmp/reims-vgpu-fail.log`: it means guest keystrokes are being eaten by the
/// host desktop.
#[derive(Debug)]
pub enum CaptureError {
    /// The window handle was not one this build knows how to capture on. Carries
    /// the handle kind so the log names the window system that was found.
    UnsupportedWindowSystem(&'static str),
    /// The compositor does not advertise `zwp_keyboard_shortcuts_inhibit_manager_v1`.
    /// There is no other way to ask on Wayland; the guest keeps losing chords.
    NoInhibitManager,
    /// The compositor advertises the manager but exposes no seat to inhibit on.
    NoSeat,
    /// A Wayland protocol call failed. Carries the library's own detail.
    Protocol(String),
    /// `XGrabKeyboard` returned a non-success status. Carries the X status code.
    GrabRefused(u8),
    /// The platform can only take part of the shortcut surface. Carries what is
    /// still owned by the host, so the log says what the guest will not receive.
    PartialOnly(&'static str),
}

impl Decline for CaptureError {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnsupportedWindowSystem(_) => "window_capture_unsupported",
            Self::NoInhibitManager => "window_capture_no_inhibit_manager",
            Self::NoSeat => "window_capture_no_seat",
            Self::Protocol(_) => "window_capture_protocol",
            Self::GrabRefused(_) => "window_capture_grab_refused",
            Self::PartialOnly(_) => "window_capture_partial",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnsupportedWindowSystem(k) => vec![("window_system", (*k).to_string())],
            Self::Protocol(d) => vec![("detail", super::present::detail_field(d))],
            Self::GrabRefused(s) => vec![("status", s.to_string())],
            Self::PartialOnly(w) => vec![("still_host_owned", (*w).to_string())],
            Self::NoInhibitManager | Self::NoSeat => Vec::new(),
        }
    }
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.slug())
    }
}

impl std::error::Error for CaptureError {}

/// The capture mechanism a boot actually got, for the always-on census line.
///
/// An observation, not a failure — it is sent through `off()`. It exists because
/// "the guest is missing keystrokes" has three very different causes, and the
/// first thing a reader needs is which of the three mechanisms this boot had.
pub struct CaptureMode(pub &'static str);

impl Decline for CaptureMode {
    fn slug(&self) -> &'static str {
        "window_capture_mode"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("mechanism", self.0.to_string())]
    }
}

/// The first time capture actually engaged, for the always-on census line.
///
/// [`CaptureMode`] says which mechanism the boot *has*; this says the compositor
/// or server actually handed the shortcuts over at least once. The two are
/// genuinely different outcomes and the first live boot showed why: a window
/// that builds its capture and is then never focused logs exactly what a working
/// capture logs, and the operator's real question — "did the guest ever get my
/// Cmd key" — is answered by neither the mode line nor the silence after it.
///
/// The line carries the release chord because this is the moment the operator's
/// desktop shortcuts stop working. A grab whose escape hatch is documented only
/// in this crate's source is a grab the operator cannot get out of, so the
/// chord is emitted where they are already looking rather than left to be found.
pub struct CaptureEngaged(pub &'static str);

impl Decline for CaptureEngaged {
    fn slug(&self) -> &'static str {
        "window_capture_engaged"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("mechanism", self.0.to_string()),
            // Read from the constant the recogniser itself uses, so the message
            // and the chord that actually works cannot drift apart.
            ("release", super::keyboard::UNGRAB_CHORD.to_string()),
        ]
    }
}

/// A platform's shortcut capture, if it has one.
///
/// `set` is idempotent and total: the window calls it with the grab state the
/// pure [`super::keyboard::GrabState`] decided, and the implementation makes the
/// platform agree. Errors are reported, never propagated into the event loop —
/// a window that cannot capture must still present.
pub trait Capture: Send {
    /// Make the platform's capture state match `active`.
    fn set(&mut self, active: bool) -> Result<(), CaptureError>;

    /// What this capture actually covers, for the boot log.
    fn describe(&self) -> &'static str;
}

/// A capture that does nothing, for a window system this build cannot ask.
///
/// It exists so the window's own code path is the same everywhere: there is
/// always a `Capture`, and the difference between platforms is what it reports,
/// not whether the call site has to branch.
pub struct NoCapture {
    reason: &'static str,
}

impl NoCapture {
    pub fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl Capture for NoCapture {
    fn set(&mut self, _active: bool) -> Result<(), CaptureError> {
        Ok(())
    }

    fn describe(&self) -> &'static str {
        self.reason
    }
}

/// Build the capture for this window, or a [`NoCapture`] plus a typed reason.
///
/// Called once, after the native window exists — every platform's request is
/// bound to a live surface or window id, so there is nothing to build before it.
///
/// The per-platform choice is a `cfg`-selected [`platform`] function rather than
/// `cfg` blocks inside this body. Blocks would leave which one is the function's
/// tail expression depending on the target, so the same source is idiomatic on
/// one host and a lint failure on another — and the host that fails is the one
/// no required clippy arm compiles.
pub fn for_window(
    display: raw_window_handle::RawDisplayHandle,
    window: raw_window_handle::RawWindowHandle,
) -> Box<dyn Capture> {
    platform::build(display, window)
}

/// Report `error` and fall back to an uncaptured window named by `reason`.
///
/// Shared by every platform arm: a capture that cannot be built is a real
/// refusal — the host desktop keeps eating guest keystrokes — so it is always
/// named on the fail channel, and the window always gets a working `Capture` to
/// call so no call site has to branch on whether capture exists.
///
/// Not compiled on macOS: that arm's capture is an application-wide property
/// with nothing to bind and no way to fail while building, so it declines at
/// `set` time (see [`macos::MacCapture`]) rather than here.
#[cfg(not(target_os = "macos"))]
fn declined(error: CaptureError, reason: &'static str) -> Box<dyn Capture> {
    crate::observe::Emit::decline("host_window_capture", &error).fail();
    Box::new(NoCapture::new(reason))
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use super::{declined, wayland, x11, Capture, CaptureError};
    use raw_window_handle::{RawDisplayHandle as D, RawWindowHandle as W};

    pub fn build(display: D, window: W) -> Box<dyn Capture> {
        match (display, window) {
            (D::Wayland(d), W::Wayland(w)) => match wayland::WaylandCapture::new(d, w) {
                Ok(c) => Box::new(c),
                Err(e) => declined(e, "wayland_unavailable"),
            },
            (D::Xlib(d), W::Xlib(w)) => match x11::X11Capture::new(d, w) {
                Ok(c) => Box::new(c),
                Err(e) => declined(e, "x11_unavailable"),
            },
            // The window was built on the XCB backend, whose window id this
            // build has no Xlib `Display*` to grab against. Named rather than
            // silently uncaptured.
            (D::Xcb(_), _) | (_, W::Xcb(_)) => declined(
                CaptureError::UnsupportedWindowSystem("xcb"),
                "xcb_unsupported",
            ),
            _ => declined(
                CaptureError::UnsupportedWindowSystem("unknown"),
                "unknown_unsupported",
            ),
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{macos, Capture};
    use raw_window_handle::{RawDisplayHandle as D, RawWindowHandle as W};

    /// macOS capture is an application-wide presentation-options change, so it
    /// needs neither handle; they are taken to keep one signature across arms.
    pub fn build(_display: D, _window: W) -> Box<dyn Capture> {
        Box::new(macos::MacCapture::new())
    }
}

#[cfg(not(unix))]
mod platform {
    use super::{declined, Capture, CaptureError};
    use raw_window_handle::{RawDisplayHandle as D, RawWindowHandle as W};

    pub fn build(_display: D, _window: W) -> Box<dyn Capture> {
        declined(
            CaptureError::UnsupportedWindowSystem("unknown"),
            "unknown_unsupported",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `NoCapture` accepts both states without error — the window's call site
    /// must not have to know whether capture is real.
    #[test]
    fn no_capture_is_total() {
        let mut c = NoCapture::new("test");
        assert!(c.set(true).is_ok());
        assert!(c.set(false).is_ok());
        assert_eq!(c.describe(), "test");
    }

    /// Every refusal carries a distinct slug; a log that cannot tell "no
    /// compositor support" from "the grab was refused" cannot be acted on.
    #[test]
    fn capture_error_slugs_are_distinct() {
        let all = [
            CaptureError::UnsupportedWindowSystem("xcb"),
            CaptureError::NoInhibitManager,
            CaptureError::NoSeat,
            CaptureError::Protocol("x".into()),
            CaptureError::GrabRefused(1),
            CaptureError::PartialOnly("y"),
        ];
        let mut seen = std::collections::HashSet::new();
        for e in &all {
            assert!(seen.insert(e.slug()), "duplicate slug {}", e.slug());
            assert!(e.slug().starts_with("window_capture_"));
        }
    }
}
