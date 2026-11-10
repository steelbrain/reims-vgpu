//! The window's OS side, one file per OS: what draws into the window and where it opens
//! ([`FrameSink`], `Surface`), the window's opening, changes and end on the OS's UI thread, the
//! host's screens, and the keyboard's kind.
//!
//! - **macOS** (`native/macos.rs`): AppKit, on the main thread but for the render thread; or,
//!   built with the `macos-sdl` feature and run with `REIMS_VGPU_WINDOW_IMPL=sdl`, SDL3 in
//!   AppKit's application there, with AppKit's window's extensions over SDL's `NSWindow`
//!   (`native/macos/sdl.rs`).
//! - **Linux** (`native/linux.rs`): SDL3, on Wayland or X11, on the library's UI thread
//!   (`reims-vgpu.ui`) but for the render thread, which draws into the window's Vulkan swapchain;
//!   the window's shell and cursor are SDL's shared ones (`sdl`), with Linux's side in
//!   `native/linux/sdl.rs`.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod sdl_vulkan;
#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::FrameSink;
#[cfg(target_os = "windows")]
pub(crate) use windows::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, now, open, sdl as sdl_os,
};
#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "linux")]
pub use linux::FrameSink;
#[cfg(target_os = "linux")]
pub(crate) use linux::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, now, open, sdl as sdl_os,
};
#[cfg(target_os = "macos")]
pub use macos::FrameSink;
#[cfg(all(target_os = "macos", feature = "macos-sdl"))]
pub(crate) use macos::sdl as sdl_os;
#[cfg(target_os = "macos")]
pub(crate) use macos::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, input_takes, is_iso_keyboard,
    key_input_captured, launched, monitored_events, now, open, quit_requested, release_key_input,
    screens_changed,
};
