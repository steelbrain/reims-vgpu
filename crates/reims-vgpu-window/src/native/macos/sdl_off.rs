//! SDL3's window on macOS, not built in: a build without the `macos-sdl` feature has no SDL, and
//! its windows are AppKit's whatever `REIMS_VGPU_WINDOW_IMPL` names (`super::launched` logs it
//! once). Nothing here is asked (`super::sdl_windows`); each answers as no SDL window would.

use objc2::MainThreadMarker;
use objc2_app_kit::NSEvent;

/// SDL3's window is not built in.
pub(crate) const BUILT_IN: bool = false;

/// SDL's video never starts.
pub(crate) fn start(_mtm: MainThreadMarker) {}

/// No SDL window's input takes an event.
pub(crate) fn takes(_event: &NSEvent, _mtm: MainThreadMarker) -> bool {
    false
}

/// No SDL window has input to let go.
pub(crate) fn release_key_input(_mtm: MainThreadMarker) {}

/// No SDL window has the keyboard.
pub(crate) fn key_input_captured(_mtm: MainThreadMarker) -> bool {
    false
}

/// No SDL window is open to ask about.
pub(crate) fn quit_requested(_mtm: MainThreadMarker) {}

/// No SDL window opens.
pub(crate) mod windows {
    use std::sync::Arc;

    use crate::handle::Window;

    /// No SDL window opens.
    pub(crate) fn open(_window: &Arc<Window>) {}

    /// No SDL window follows the guest's mode.
    pub(crate) fn guest_mode_changed(_id: u64) {}

    /// No SDL window shows the guest's cursor.
    pub(crate) fn cursor_changed(_id: u64) {}

    /// No SDL window closes.
    pub(crate) fn close(_id: u64) {}

    /// No SDL window lays out again.
    pub(crate) fn screens_changed() {}
}
