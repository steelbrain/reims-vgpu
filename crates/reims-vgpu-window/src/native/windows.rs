//! The window on Windows: SDL3's Win32 window, on the library's UI thread
//! (`super::sdl_thread`), and what Windows adds under it.
//!
//! - **The UI thread owns the window.** A Win32 window's messages go to the thread that made it,
//!   and SDL makes the window and pumps its messages on the thread its video started on, the UI
//!   thread, which waits in SDL between its events as on Linux. **Measured** 2026-11-15 with a
//!   probe of this crate's calls on Windows 11 26H2, an Intel UHD 630 (driver 101.2145): SDL's
//!   video starts on a spawned thread (85 ms), which SDL then takes for its main thread; the
//!   window's HWND belongs to that thread; a second thread makes the window's Vulkan surface
//!   through SDL (`VK_KHR_win32_surface`) and presents 297 frames, FIFO at the screen's 59.94 Hz,
//!   while the first resizes the window, grabs and frees the keyboard, makes it full screen and
//!   windowed again, and hides and shows it, with no error and no stall of either; an event
//!   another thread pushes wakes the UI thread's wait in 0.11 ms (median of 50, at most 10.6 ms).
//! - **DPI.** SDL's video start declares the process per-monitor DPI aware (v2), unless the
//!   process has declared its awareness already (read from SDL 3.4.18's code,
//!   `WIN_InitDPIAwareness`): SDL's sizes are then pixels, its modes have a pixel density of 1,
//!   and a screen's content scale is its DPI over 96, as on X11 (`crate::sdl::units`). A window's
//!   pixels over its size are 1 at any scale, so the scale its guest is told is SDL's display
//!   scale of the window, its monitor's content scale (`super::sdl_thread::sdl::screen_scale`):
//!   1.5 at 150%.
//! - **The keyboard grab** while captured (`SDL_SetWindowKeyboardGrab`) is a low-level keyboard
//!   hook SDL installs on the UI thread while the window has the keyboard, which takes the
//!   system's shortcuts (the Windows key, Alt+Tab, Alt+Esc) before Windows does (read from SDL
//!   3.4.18's code, `WIN_GrabKeyboard`). Windows calls the hook on that thread, so it is answered
//!   between the thread's events and jobs.
//! - **The compositor's GPU**: Windows names none to a window's client ([`compositor_device`]),
//!   so the window presents on the runtime's device, as a window does where Linux names none.
//! - **The release chord's keys** ([`HOST_OS`]): the Windows keys are SDL's GUI keys, the keys
//!   the chord holds with Ctrl and Alt, so the title names the chord Ctrl+Alt+Win. SDL gives the
//!   left and right Windows keys (`VK_LWIN` and `VK_RWIN` in the keyboard grab's hook, scan codes
//!   E0 5B and E0 5C otherwise) as `SDL_SCANCODE_LGUI` and `SDL_SCANCODE_RGUI`, the guest's left
//!   and right ⌘ (read from SDL 3.4.18's code, `WIN_KeyboardHookProc` and
//!   `scancodes_windows.h`).
//! - **Full screen to the shell.** A borderless window is SDL's desktop full screen: a popup
//!   window over its monitor's whole rect (read from SDL 3.4.18's code,
//!   `WIN_SetWindowFullscreen`). SDL never tells the shell a window is full screen (read from its
//!   code: it makes the taskbar list only for a window's progress), and a taskbar set to hide
//!   itself then comes up over the window when the pointer meets the screen's bottom edge
//!   (reported on a Windows 11 host). So the window marks it full screen
//!   ([`FullScreenMark`], `ITaskbarList2::MarkFullscreenWindow`) from the moment SDL has made it
//!   so until it hides at the display's end: Microsoft's documentation of the call says the
//!   taskbar then goes to the bottom of the z-order while the window is active, and that taking
//!   the mark away leaves the window to the shell's own detection.

use std::ffi::c_void;
use std::ptr::{NonNull, null, null_mut};

use sdl3_sys::everything::{
    SDL_GetPointerProperty, SDL_GetWindowFlags, SDL_GetWindowProperties,
    SDL_PROP_WINDOW_WIN32_HWND_POINTER, SDL_WINDOW_FULLSCREEN, SDL_Window,
};
use windows_sys::Win32::Foundation::{FALSE, HWND, RPC_E_CHANGED_MODE, S_FALSE, S_OK, TRUE};
use windows_sys::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
    CoUninitialize,
};
use windows_sys::Win32::UI::Shell::TaskbarList;
use windows_sys::core::{BOOL, GUID, HRESULT};

use crate::capture::HostOs;

/// The OS whose names the title gives the release chord's keys: Ctrl+Alt+Win (the module doc).
pub(super) const HOST_OS: HostOs = HostOs::Windows;

/// The GPU the compositor composites SDL's window on: none known on Windows (the module doc).
///
/// # Safety
///
/// None of its own: `unsafe` as Linux's, which reads SDL's window.
pub(super) unsafe fn compositor_device(_window: *mut SDL_Window) -> Option<(u32, u32)> {
    None
}

/// `ITaskbarList2`'s interface ID, as the Windows SDK's `shobjidl.h` defines it (windows-sys has
/// the taskbar list's class ID, `CLSID_TaskbarList`, but no COM interfaces).
const IID_ITASKBARLIST2: GUID = GUID::from_u128(0x602d4995_b13a_429b_a66e_1935e44f4317);

/// An `ITaskbarList2` object: a pointer to its methods.
#[repr(C)]
struct TaskbarList2 {
    /// Its methods.
    methods: *const TaskbarList2Methods,
}

/// `ITaskbarList2`'s methods, in `shobjidl.h`'s order: `IUnknown`'s, `ITaskbarList`'s, then its
/// own.
#[repr(C)]
struct TaskbarList2Methods {
    /// `QueryInterface` and `AddRef`, which the mark does not call.
    _unknown: [*const c_void; 2],
    /// `Release`.
    release: unsafe extern "system" fn(*mut TaskbarList2) -> u32,
    /// `HrInit`, which comes before any other of the list's methods.
    hr_init: unsafe extern "system" fn(*mut TaskbarList2) -> HRESULT,
    /// `AddTab`, `DeleteTab`, `ActivateTab` and `SetActiveAlt`, which the mark does not call.
    _tabs: [*const c_void; 4],
    /// `MarkFullscreenWindow`.
    mark_fullscreen_window: unsafe extern "system" fn(*mut TaskbarList2, HWND, BOOL) -> HRESULT,
}

/// The shell's mark of a full-screen window as full screen (the module doc): made marking it,
/// dropped taking the mark away, on the UI thread, which made the window.
///
/// It follows the call's documented contract: the taskbar list (`CLSID_TaskbarList`) is made with
/// `CoCreateInstance` on a thread where COM is initialised, and `HrInit` is called before any of
/// its other methods. COM is SDL's on this thread already, single-threaded (its video start
/// initialises it, `WIN_VideoInit`, read from SDL 3.4.18's code); the mark initialises it again
/// for itself, which COM counts, and lets that go after the list. The list is held while the mark
/// is.
pub(super) struct FullScreenMark {
    /// The taskbar list.
    list: NonNull<TaskbarList2>,
    /// The window marked.
    hwnd: HWND,
    /// Whether the mark's `CoInitializeEx` is matched by a `CoUninitialize`: not where COM was
    /// the thread's already, multi-threaded (`RPC_E_CHANGED_MODE`).
    uninitialize: bool,
}

impl FullScreenMark {
    /// Marks SDL's window `window` full screen to the shell, where SDL has it full screen; `None`
    /// where it is not, or where Windows refuses a step, which is logged for the window titled
    /// `title`.
    ///
    /// # Safety
    ///
    /// On the UI thread, which made `window`, a window SDL has open.
    pub(super) unsafe fn of(window: *mut SDL_Window, title: &str) -> Option<Self> {
        // SAFETY: SDL's window, on its thread (the caller's promise); the handle lives while
        // SDL's window does.
        let (flags, hwnd) = unsafe {
            (
                SDL_GetWindowFlags(window),
                SDL_GetPointerProperty(
                    SDL_GetWindowProperties(window),
                    SDL_PROP_WINDOW_WIN32_HWND_POINTER,
                    null_mut(),
                ),
            )
        };
        if (flags & SDL_WINDOW_FULLSCREEN).0 == 0 {
            return None;
        }
        if hwnd.is_null() {
            unmarked(title, "SDL names no window handle", None);
            return None;
        }
        // SAFETY: SDL's window's handle, on the thread that made the window.
        match unsafe { Self::mark(hwnd) } {
            Ok(mark) => {
                log::info!(
                    target: "reims_vgpu_window",
                    "window {title:?}: marked full screen to the shell, so a taskbar set to hide \
                     itself stays down over it"
                );
                Some(mark)
            }
            Err((step, refused)) => {
                unmarked(title, step, Some(refused));
                None
            }
        }
    }

    /// Marks the window `hwnd` full screen to the shell; else the step Windows refused, and the
    /// `HRESULT` it refused it with.
    ///
    /// # Safety
    ///
    /// On the thread that made `hwnd`, a live window.
    unsafe fn mark(hwnd: HWND) -> Result<Self, (&'static str, HRESULT)> {
        // SAFETY: COM's start on this thread, matched by `let_go` where it is this call's.
        let started = unsafe { CoInitializeEx(null(), COINIT_APARTMENTTHREADED as u32) };
        let uninitialize = match started {
            S_OK | S_FALSE => true,
            RPC_E_CHANGED_MODE => false,
            refused => return Err(("CoInitializeEx", refused)),
        };
        let mut object = null_mut();
        // SAFETY: COM is initialised on this thread; the IDs are the class's and the interface's.
        let made = unsafe {
            CoCreateInstance(
                &TaskbarList,
                null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_ITASKBARLIST2,
                &mut object,
            )
        };
        let Some(list) = NonNull::new(object.cast::<TaskbarList2>()).filter(|_| made >= 0) else {
            // SAFETY: no list was made; COM's start is let go where it was this call's.
            unsafe { let_go(None, uninitialize) };
            return Err(("CoCreateInstance", made));
        };
        // SAFETY: the list just made, on this thread: `HrInit` first, as its contract asks, then
        // the mark, of the caller's window.
        let refused = unsafe {
            let methods = &*list.as_ref().methods;
            let init = (methods.hr_init)(list.as_ptr());
            if init < 0 {
                Some(("HrInit", init))
            } else {
                let marked = (methods.mark_fullscreen_window)(list.as_ptr(), hwnd, TRUE);
                (marked < 0).then_some(("MarkFullscreenWindow", marked))
            }
        };
        if let Some(refused) = refused {
            // SAFETY: the list is this call's, and marks nothing; COM's start as above.
            unsafe { let_go(Some(list), uninitialize) };
            return Err(refused);
        }
        Ok(Self {
            list,
            hwnd,
            uninitialize,
        })
    }
}

impl Drop for FullScreenMark {
    /// Takes the mark away, then lets the list and COM go, in that order.
    fn drop(&mut self) {
        // SAFETY: the mark's list, initialised, on the UI thread, which made it and the window;
        // the window has not gone yet (`crate::sdl::shell` destroys SDL's window after its OS
        // side lets go).
        let unmarked = unsafe {
            let methods = &*self.list.as_ref().methods;
            (methods.mark_fullscreen_window)(self.list.as_ptr(), self.hwnd, FALSE)
        };
        if unmarked < 0 {
            log::debug!(
                target: "reims_vgpu_window",
                "the shell's full-screen mark was not taken away: {}",
                hresult(unmarked)
            );
        }
        // SAFETY: the mark's list, released once, here; COM's start as in `mark`.
        unsafe { let_go(Some(self.list), self.uninitialize) };
    }
}

/// Releases `list`, then lets COM go on this thread where `uninitialize`.
///
/// # Safety
///
/// `list` is a live list of this thread's, released no more after this; `uninitialize` only for
/// a `CoInitializeEx` of this thread's that nothing else matches.
unsafe fn let_go(list: Option<NonNull<TaskbarList2>>, uninitialize: bool) {
    if let Some(list) = list {
        // SAFETY: the caller's promise.
        unsafe { ((*list.as_ref().methods).release)(list.as_ptr()) };
    }
    if uninitialize {
        // SAFETY: the caller's promise.
        unsafe { CoUninitialize() };
    }
}

/// Logs that the window titled `title` is not marked full screen: at `step`, which Windows
/// refused with `refused` where it returned an `HRESULT`.
fn unmarked(title: &str, step: &str, refused: Option<HRESULT>) {
    let refused = refused.map_or_else(String::new, |r| format!(": {}", hresult(r)));
    log::warn!(
        target: "reims_vgpu_window", kind:% = "no full-screen mark";
        "window {title:?}: not marked full screen to the shell ({step}{refused}), so a taskbar \
         set to hide itself can come up over it"
    );
}

/// An `HRESULT` as Windows prints one: `0x80004005`.
fn hresult(result: HRESULT) -> String {
    format!("{:#010x}", result.cast_unsigned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Windows build's title names the release chord in Windows' keys, Ctrl+Alt+Win: the
    /// constant the SDL thread passes is Windows' (the module doc), not Linux's Super.
    #[test]
    fn a_windows_build_names_the_release_chord_in_windows_keys() {
        assert_eq!(HOST_OS, HostOs::Windows);
        let title = crate::capture::captured_suffix(HOST_OS);
        assert!(
            title.contains("Ctrl+Alt+Win") && !title.contains("Super"),
            "{title}"
        );
    }
}
