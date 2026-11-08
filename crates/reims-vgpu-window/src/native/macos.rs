//! The window on macOS: AppKit and Core Animation. The window's AppKit side (the application,
//! its `NSWindow`, view and cursor) lives on the main thread, reached through work posted there
//! (`main_thread`); its render thread draws into the view's `CAMetalLayer` (`render`), through
//! the sink's presenter over a Vulkan surface of the layer (`metal_surface`, `crate::present`).
//! This file is what the shared window code asks of it: what draws into the layer
//! ([`FrameSink`](crate::FrameSink)), the window's opening, changes and end, the host's screens
//! (`screens`, from CoreGraphics) and the keyboard's kind (`iso`, from Carbon).
//!
//! With `REIMS_VGPU_WINDOW_IMPL=sdl`, in a build with the `macos-sdl` feature, the window is
//! SDL3's (`sdl`), in the same application, drawn by the same render thread into its Metal view's
//! layer: the window's opening, changes and end go to the SDL window's shell (`sdl::windows`, the
//! shared `crate::sdl::shell`) in place of AppKit's (`shell`), and what the application asks of
//! the windows (its events, the menu, Quit and the screens) goes to whichever opens. A build
//! without the feature has no SDL (`sdl_off`): its windows are AppKit's whatever the variable
//! names, which the application's launch logs once.

// AppKit's side: the application, its windows, their views, render threads, input and cursor.
mod app;
mod cursor;
mod hotkeys;
mod input;
mod iso;
mod main_thread;
mod metal_surface;
mod render;
mod screens;
mod shell;
mod view;
// SDL3's window, or, without the `macos-sdl` feature, none.
#[cfg(feature = "macos-sdl")]
pub(crate) mod sdl;
#[cfg(not(feature = "macos-sdl"))]
mod sdl_off;
#[cfg(not(feature = "macos-sdl"))]
use sdl_off as sdl;

use std::sync::Arc;

use objc2::MainThreadMarker;
use objc2_app_kit::{NSEvent, NSEventMask};
use reims_vgpu_vulkan::core::platform::os::{Clock as _, MonoTime};
use reims_vgpu_vulkan::os::OsClock;

pub(crate) use self::iso::is_iso_keyboard;
pub(crate) use self::screens::host_screens;
pub(crate) use render::Surface;

use crate::env::WindowImpl;
use crate::handle::Window;
use main_thread::post_main;

/// Whether `REIMS_VGPU_WINDOW_IMPL` names SDL3's window.
fn sdl_named() -> bool {
    crate::env::settings().window_impl == WindowImpl::Sdl
}

/// Whether the windows are SDL3's: the variable names SDL's, and the build has it.
fn sdl_windows() -> bool {
    sdl::BUILT_IN && sdl_named()
}

/// Starts `window`'s render thread and, once per process, the application, and posts the
/// window's opening to the main thread.
pub(crate) fn open(window: &Arc<Window>) {
    render::start(Arc::clone(window));
    app::start();
    let window = Arc::downgrade(window);
    post_main(move |mtm| {
        if let Some(window) = window.upgrade() {
            window.opening();
            if sdl_windows() {
                sdl::windows::open(&window);
            } else {
                shell::open(&window, mtm);
            }
        }
    });
}

/// Posts to the main thread that the guest of window `id` set a mode.
pub(crate) fn guest_mode_changed(id: u64) {
    post_main(move |mtm| {
        if sdl_windows() {
            sdl::windows::guest_mode_changed(id);
        } else {
            shell::guest_mode_changed(id, mtm);
        }
    });
}

/// Posts a cursor refresh of window `id` to the main thread.
pub(crate) fn cursor_changed(id: u64) {
    post_main(move |mtm| {
        if sdl_windows() {
            sdl::windows::cursor_changed(id);
        } else {
            shell::cursor_changed(id, mtm);
        }
    });
}

/// Posts the close of window `id` to the main thread.
pub(crate) fn close(id: u64) {
    post_main(move |mtm| {
        if sdl_windows() {
            sdl::windows::close(id);
        } else {
            shell::close(id, mtm);
        }
    });
}

/// Now, on the monotonic clock the library's platform reads (`OsClock`).
pub(crate) fn now() -> MonoTime {
    OsClock::new().now_mono()
}

/// The application has launched, before the windows posted so far open: SDL's video starts,
/// where the windows are SDL3's; where the variable names SDL's in a build without it, that is
/// logged once, and the windows are AppKit's.
pub(crate) fn launched(mtm: MainThreadMarker) {
    if sdl_windows() {
        sdl::start(mtm);
    } else if sdl_named() {
        log::warn!(
            target: "reims_vgpu_window",
            "{}=sdl, but this build has no SDL (the `macos-sdl` feature): the windows are \
             AppKit's",
            crate::env::IMPL_VAR
        );
    }
}

/// Whether the key window's input takes `event` before AppKit dispatches it; for SDL3's windows
/// their input sees every mouse and scroll event too (`sdl`'s `input`).
pub(crate) fn input_takes(event: &NSEvent, mtm: MainThreadMarker) -> bool {
    if sdl_windows() {
        sdl::takes(event, mtm)
    } else {
        view::key_view_takes(event, mtm)
    }
}

/// The events a host's application gives the windows' input first (`app`'s monitor): the keys,
/// and for SDL3's windows the mouse and the wheel too.
pub(crate) fn monitored_events() -> NSEventMask {
    let keys = NSEventMask::KeyDown
        | NSEventMask::KeyUp
        | NSEventMask::FlagsChanged
        | NSEventMask::SystemDefined;
    if !sdl_windows() {
        return keys;
    }
    keys | NSEventMask::LeftMouseDown
        | NSEventMask::LeftMouseUp
        | NSEventMask::RightMouseDown
        | NSEventMask::RightMouseUp
        | NSEventMask::OtherMouseDown
        | NSEventMask::OtherMouseUp
        | NSEventMask::LeftMouseDragged
        | NSEventMask::RightMouseDragged
        | NSEventMask::OtherMouseDragged
        | NSEventMask::MouseMoved
        | NSEventMask::ScrollWheel
}

/// Window ▸ Release Input: the key window's input lets the keyboard and the pointer go.
pub(crate) fn release_key_input(mtm: MainThreadMarker) {
    if sdl_windows() {
        sdl::release_key_input(mtm);
    } else if let Some(view) = view::key_view(mtm) {
        view.release_capture();
    }
}

/// Whether the key window's input has the keyboard captured.
pub(crate) fn key_input_captured(mtm: MainThreadMarker) -> bool {
    if sdl_windows() {
        sdl::key_input_captured(mtm)
    } else {
        view::key_view(mtm).is_some_and(|view| view.is_captured())
    }
}

/// Quit: every window lets its input go and tells the host, which ends the VM.
pub(crate) fn quit_requested(mtm: MainThreadMarker) {
    if sdl_windows() {
        sdl::quit_requested();
    } else {
        for view in shell::views(mtm) {
            view.close_requested();
        }
    }
}

/// The screens were added, removed, moved or resized.
pub(crate) fn screens_changed(mtm: MainThreadMarker) {
    if sdl_windows() {
        sdl::windows::screens_changed();
    } else {
        shell::screens_changed(mtm);
    }
}
