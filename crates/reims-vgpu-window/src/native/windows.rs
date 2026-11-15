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
//!   and a screen's content scale is its DPI over 96, as on X11 (`crate::sdl::units`).
//! - **The keyboard grab** while captured (`SDL_SetWindowKeyboardGrab`) is a low-level keyboard
//!   hook SDL installs on the UI thread while the window has the keyboard, which takes the
//!   system's shortcuts (the Windows key, Alt+Tab, Alt+Esc) before Windows does (read from SDL
//!   3.4.18's code, `WIN_GrabKeyboard`). Windows calls the hook on that thread, so it is answered
//!   between the thread's events and jobs.
//! - **The compositor's GPU**: Windows names none to a window's client ([`compositor_device`]),
//!   so the window presents on the runtime's device, as a window does where Linux names none.

use sdl3_sys::everything::SDL_Window;

/// The GPU the compositor composites SDL's window on: none known on Windows (the module doc).
///
/// # Safety
///
/// None of its own: `unsafe` as Linux's, which reads SDL's window.
pub(super) unsafe fn compositor_device(_window: *mut SDL_Window) -> Option<(u32, u32)> {
    None
}
