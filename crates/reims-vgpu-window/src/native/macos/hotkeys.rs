//! The system's own hot keys (⌘Tab, ⌘Space, Mission Control), turned off while a window
//! captures the keyboard so they reach the guest.
//!
//! WindowServer's `CGSSetGlobalHotKeyOperatingMode(CGSMainConnectionID(), mode)` sets them for
//! the process's connection: mode 1 turns them off, 0 back on. Both are SkyLight's, a private
//! framework the dist tree's pkg-config link line refuses, so they are looked up when first
//! needed with `dlsym`: among the images already loaded (`RTLD_DEFAULT`), then in SkyLight itself
//! (`dlopen` with `RTLD_NOLOAD`, then loading it). Where neither finds both, the hot keys stay as
//! they are, and one line says so.
//!
//! Measured on the M3 Max, macOS 26.7 (25G229):
//! - a process that links AppKit or CoreGraphics, as this crate does, has SkyLight loaded, and
//!   `RTLD_DEFAULT` finds both functions there;
//! - a process that links neither finds neither that way, and has no SkyLight to open with
//!   `RTLD_NOLOAD`; `dlopen` then loads it, and finds both;
//! - the call succeeds from an unbundled, ad hoc signed process, with no TCC prompt: the mode
//!   reads back as set.
//!
//! Every window that captures the keyboard holds the hot keys off, and they come back when the
//! last holder lets go, so two windows never undo each other.

use std::ffi::{CStr, c_int, c_void};
use std::sync::OnceLock;

use reims_vgpu_vulkan::core::sync::Mutex;

/// `CGSMainConnectionID()`: the process's connection to WindowServer.
type MainConnectionId = unsafe extern "C" fn() -> c_int;
/// `CGSSetGlobalHotKeyOperatingMode(connection, mode)`: a `CGError`, 0 on success.
type SetGlobalHotKeyOperatingMode = unsafe extern "C" fn(c_int, c_int) -> c_int;

/// `kCGSGlobalHotKeyEnable`: the system's hot keys work.
const MODE_ENABLE: c_int = 0;
/// `kCGSGlobalHotKeyDisable`: the system's hot keys are off, and their keys reach the app.
const MODE_DISABLE: c_int = 1;

/// SkyLight, where both functions are.
const SKYLIGHT: &CStr = c"/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight";
/// The functions' names.
const MAIN_CONNECTION_ID: &CStr = c"CGSMainConnectionID";
/// See [`MAIN_CONNECTION_ID`].
const SET_MODE: &CStr = c"CGSSetGlobalHotKeyOperatingMode";

/// The two functions, found.
#[derive(Clone, Copy)]
struct HotKeyCalls {
    /// `CGSMainConnectionID`.
    main_connection_id: MainConnectionId,
    /// `CGSSetGlobalHotKeyOperatingMode`.
    set_mode: SetGlobalHotKeyOperatingMode,
}

/// The functions, looked up once; `None` where the system does not have them.
static CALLS: OnceLock<Option<HotKeyCalls>> = OnceLock::new();

/// How many windows hold the hot keys off.
static HOLDERS: Mutex<usize> = Mutex::new(0);

/// Both functions in `handle` (a `dlopen` handle, or `RTLD_DEFAULT`), or `None` unless both are
/// there.
fn lookup(handle: *mut c_void) -> Option<HotKeyCalls> {
    // SAFETY: the names are NUL-terminated, and `handle` is `RTLD_DEFAULT` or a live handle.
    let (main, set) = unsafe {
        (
            libc::dlsym(handle, MAIN_CONNECTION_ID.as_ptr()),
            libc::dlsym(handle, SET_MODE.as_ptr()),
        )
    };
    if main.is_null() || set.is_null() {
        return None;
    }
    // SAFETY: both are SkyLight's exported functions of these C signatures (`CGSConnectionID`
    // and the operating mode are C `int`s, `CGError` an `int32_t`), and a function's address
    // converts to a function pointer.
    Some(unsafe {
        HotKeyCalls {
            main_connection_id: std::mem::transmute::<*mut c_void, MainConnectionId>(main),
            set_mode: std::mem::transmute::<*mut c_void, SetGlobalHotKeyOperatingMode>(set),
        }
    })
}

/// Looks the functions up (module doc): loaded images first, then SkyLight, opened only if it
/// is loaded, then loaded. Says once when they are not found.
fn resolve() -> Option<HotKeyCalls> {
    if let Some(calls) = lookup(libc::RTLD_DEFAULT) {
        return Some(calls);
    }
    // SAFETY: the path is NUL-terminated. The handle is never closed: the functions it gives
    // are kept for the process's life.
    let handle = unsafe {
        let loaded = libc::dlopen(SKYLIGHT.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD);
        if loaded.is_null() {
            libc::dlopen(SKYLIGHT.as_ptr(), libc::RTLD_LAZY)
        } else {
            loaded
        }
    };
    let calls = if handle.is_null() {
        None
    } else {
        lookup(handle)
    };
    if calls.is_none() {
        log::warn!(
            "the system's hot keys stay on while input is captured: \
             CGSSetGlobalHotKeyOperatingMode was not found"
        );
    }
    calls
}

/// The count of holders after one takes (`disabled`) or lets go of the hot keys, and the state
/// to set them to when that changes: `Some(true)` off, `Some(false)` back on. A letting go with
/// no holder changes nothing.
fn next_holders(holders: usize, disabled: bool) -> (usize, Option<bool>) {
    let next = if disabled {
        holders.saturating_add(1)
    } else {
        holders.saturating_sub(1)
    };
    let (held, holds) = (holders > 0, next > 0);
    (next, (held != holds).then_some(holds))
}

/// One window takes (`true`) or lets go of the system's hot keys: they are off while any window
/// holds them, and back on once none does. A window takes them once while it captures the
/// keyboard, and lets go once when capture ends. Does nothing where the system does not offer
/// it.
///
/// Main thread only, where capture changes: WindowServer applies the mode to the process's
/// connection, and the windows' holds are counted in the order they are made.
pub(crate) fn set_system_hotkeys_disabled(disabled: bool) {
    let mut holders = HOLDERS.lock();
    let (next, change) = next_holders(*holders, disabled);
    *holders = next;
    let Some(off) = change else {
        return;
    };
    let Some(calls) = *CALLS.get_or_init(resolve) else {
        return;
    };
    let mode = if off { MODE_DISABLE } else { MODE_ENABLE };
    // SAFETY: both take and return plain integers; the connection is the process's own.
    let error = unsafe { (calls.set_mode)((calls.main_connection_id)(), mode) };
    if error != 0 {
        log::warn!("CGSSetGlobalHotKeyOperatingMode({mode}) failed: CGError {error}");
    }
}

#[cfg(test)]
mod tests {
    use objc2::ClassType;

    use super::*;

    /// Two windows holding the hot keys off: they come back only when both let go
    #[test]
    fn the_hot_keys_come_back_when_the_last_holder_lets_go() {
        assert_eq!(next_holders(0, true), (1, Some(true)));
        assert_eq!(next_holders(1, true), (2, None));
        assert_eq!(next_holders(2, false), (1, None));
        assert_eq!(next_holders(1, false), (0, Some(false)));
        // A letting go with no holder changes nothing.
        assert_eq!(next_holders(0, false), (0, None));
    }

    /// The functions are found in a process that links AppKit, among the loaded images
    #[test]
    fn the_hot_key_functions_are_found() {
        let _link = objc2_app_kit::NSEvent::class();
        assert!(lookup(libc::RTLD_DEFAULT).is_some());
        assert!(resolve().is_some());
    }
}
