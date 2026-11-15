//! The window's OS side, one file per OS: the render thread and what it draws on (`Surface`), the
//! window's opening, changes and end on the OS's UI thread, the
//! host's screens, and the keyboard's kind.
//!
//! - **macOS** (`native/macos.rs`): AppKit, on the main thread but for the render thread; or,
//!   built with the `macos-sdl` feature and run with `REIMS_VGPU_WINDOW_IMPL=sdl`, SDL3 in
//!   AppKit's application there, with AppKit's window's extensions over SDL's `NSWindow`
//!   (`native/macos/sdl.rs`).
//! - **Linux and Windows** (`native/sdl_thread.rs`): SDL3, on Wayland or X11, or Win32, on the
//!   library's UI thread (`reims-vgpu.ui`) but for the render thread, which draws into the
//!   window's Vulkan swapchain; the window's shell and cursor are SDL's shared ones (`sdl`), with
//!   the SDL thread's side in `native/sdl_thread/sdl.rs`, and what each OS adds in its own file
//!   (`native/linux.rs`: the GPU a Wayland compositor composites on; `native/windows.rs`).

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod sdl_thread;
#[cfg(target_os = "windows")]
mod windows;

// What the OS adds under the SDL thread's windows: the GPU its compositor composites on.
#[cfg(target_os = "linux")]
use linux::compositor_device;
#[cfg(target_os = "windows")]
use windows::compositor_device;

#[cfg(all(target_os = "macos", feature = "macos-sdl"))]
pub(crate) use macos::sdl as sdl_os;
#[cfg(target_os = "macos")]
pub(crate) use macos::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, input_takes, is_iso_keyboard,
    key_input_captured, launched, monitored_events, now, open, quit_requested, release_key_input,
    screens_changed,
};
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) use sdl_thread::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, now, open, sdl as sdl_os,
};
