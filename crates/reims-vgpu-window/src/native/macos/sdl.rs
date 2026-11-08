//! SDL3's window on macOS (`REIMS_VGPU_WINDOW_IMPL=sdl`): the shared SDL window's macOS side
//! (`crate::sdl::os`), and the extensions over SDL's `NSWindow` that make it AppKit's window's
//! equal.
//!
//! SDL makes the window, its delegate and its content view, and reports the window's changes
//! (its size, scale, screen, key status, occlusion, a close by the person), on the main thread
//! inside the application (`ui`). The rest is AppKit's window's, over SDL's `NSWindow`:
//! - **Where it opens**, and how it follows the guest's mode: the frames AppKit's window takes
//!   (`super::shell`'s), on the `NSScreen` that is the CoreGraphics screen picked for the
//!   display's port: centred in the screen's visible frame, clear of the other windowed windows'
//!   top left corners, and, following the guest's mode, its top left corner kept inside the
//!   visible frame. The host's screens stay CoreGraphics', as AppKit's window lists them.
//! - **Borderless**: a titled window made to look borderless, filling its screen's whole frame,
//!   the Dock and the menu bar hidden while the application is active, its image inside the
//!   screen's safe area (below a camera housing); when the screens change, it fills its screen
//!   again, or the main screen while its own is gone.
//! - **The surface**: SDL's Metal view, whose `CAMetalLayer` is set up as AppKit's view's
//!   (`super::render::configure_layer`) and drawn into by the same render thread
//!   (`super::render`). The layout is the content view's points at the window's backing scale.
//! - **Input**: AppKit's events, given to AppKit's window's input (`super::input`) by the
//!   application before it dispatches them (`input`), with the system's hot keys held off while
//!   captured; SDL's own input events are not taken.
//! - **The size while captured**: as AppKit's window, a windowed window is not resizable while
//!   captured, and is again once released ([`hold_size`]).
//! - **The end**: at the display's end the window lets its input go and hides, and SDL's window
//!   and Metal view go at once: the render thread holds the layer itself, and lets go of it on
//!   the main thread.

mod input;
mod ui;

use std::sync::Arc;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSWindow, NSWindowOcclusionState, NSWindowStyleMask};
use objc2_foundation::NSRect;
use objc2_quartz_core::CAMetalLayer;
use reims_vgpu_vulkan::os::msg;
use sdl3_sys::everything::*;

pub(crate) use self::input::{Input, key_input_captured, quit_requested, release_key_input, takes};
pub(crate) use self::ui::start;
use super::render::{LayerRef, configure_layer};
use super::shell as appkit;
use crate::geometry::Insets;
use crate::handle::Window;
use crate::screens;
use crate::sdl::c_string;
/// The SDL windows' shell, which the application's requests go to (`super`).
pub(crate) use crate::sdl::shell as windows;
use crate::sdl::shell::{self, DEFAULT_MODE, MIN_CONTENT, Opened, Shell};

/// SDL3's window is built in (`super::sdl_windows`).
pub(crate) const BUILT_IN: bool = true;

/// Where an open windowed window is, for placing others: its frame, in screen coordinates.
pub(crate) type Frame = NSRect;

/// What macOS keeps of a window.
pub(crate) struct State {
    /// SDL's `NSWindow`.
    ns_window: Retained<NSWindow>,
    /// SDL's Metal view, whose layer the render thread draws into.
    metal_view: SDL_MetalView,
    /// Whether it is borderless: it fills its screen.
    borderless: bool,
    /// The CoreGraphics display of the screen it opened on, which a borderless window fills
    /// again when the screens change.
    screen: Option<u32>,
}

impl State {
    /// SDL's `NSWindow`.
    pub(crate) fn ns_window(&self) -> &NSWindow {
        &self.ns_window
    }
}

/// Where `shell`'s window is, for placing others: its frame, where it is an open windowed one.
pub(crate) fn frame_of(shell: &Shell) -> Option<Frame> {
    (!shell.os.borderless && !shell.closed)
        .then(|| msg::get("frame", || shell.os.ns_window.frame()))
}

/// SDL's `NSWindow` of `sdl`.
fn cocoa_window(sdl: *mut SDL_Window) -> Option<Retained<NSWindow>> {
    // SAFETY: `sdl` is a live window of SDL's, on the main thread; the property is SDL's pointer
    // to its NSWindow, which lives while SDL's window does, and is retained here.
    unsafe {
        let properties = SDL_GetWindowProperties(sdl);
        let window = SDL_GetPointerProperty(
            properties,
            SDL_PROP_WINDOW_COCOA_WINDOW_POINTER,
            std::ptr::null_mut(),
        );
        Retained::retain(window.cast::<NSWindow>())
    }
}

/// Sets `window`'s style mask to `style`, keeping its content view's next responder, which SDL
/// set (its window's listener) and AppKit may reset with the style.
fn restyle(window: &NSWindow, style: NSWindowStyleMask) {
    let content = msg::get("contentView", || window.contentView());
    // SAFETY: a getter.
    let next = content
        .as_deref()
        .and_then(|c| msg::get("nextResponder", || unsafe { c.nextResponder() }));
    msg::get("setStyleMask:", || window.setStyleMask(style));
    if let (Some(content), Some(next)) = (content, next) {
        // SAFETY: the next responder is SDL's listener of the window, which SDL keeps while the
        // window lives.
        msg::get("setNextResponder:", || unsafe {
            content.setNextResponder(Some(&next));
        });
    }
}

/// Makes SDL's Metal view in `sdl`, sets its layer up, and hands `window`'s render thread the
/// layer. `None` where SDL makes no view.
fn attach_layer(window: &Window, sdl: *mut SDL_Window) -> Option<SDL_MetalView> {
    // SAFETY: SDL's window, made with `SDL_WINDOW_METAL`, on the main thread.
    let view = unsafe { SDL_Metal_CreateView(sdl) };
    if view.is_null() {
        shell::refused("no Metal view", window.title());
        return None;
    }
    // SAFETY: the view's layer is a CAMetalLayer, which the view keeps while it lives; it is
    // retained here for the render thread.
    let layer = unsafe { SDL_Metal_GetLayer(view).cast::<CAMetalLayer>().as_ref() };
    let Some(layer) = layer else {
        shell::refused("no Metal layer", window.title());
        // SAFETY: the view just made.
        unsafe { SDL_Metal_DestroyView(view) };
        return None;
    };
    configure_layer(layer);
    window.surface().set_layer(LayerRef::new(layer));
    Some(view)
}

/// Tells the render thread whether any of `window` is on screen.
fn occlusion_changed(window: &Window, ns_window: &NSWindow) {
    let visible = msg::get("occlusionState", || ns_window.occlusionState())
        .contains(NSWindowOcclusionState::Visible);
    window.surface().set_visible(visible);
}

/// SDL's size of `sdl` in points.
fn sdl_size(sdl: *mut SDL_Window) -> (i32, i32) {
    let (mut w, mut h) = (0, 0);
    // SAFETY: SDL's window, on the main thread.
    unsafe { SDL_GetWindowSize(sdl, &mut w, &mut h) };
    (w, h)
}

/// Makes and shows `window`'s SDL window where AppKit's would open, clear of the windowed frames
/// `taken` (the module doc), and hands the render thread its layer.
pub(crate) fn open(window: &Arc<Window>, borderless: bool, taken: &[Frame]) -> Option<Opened> {
    let mtm = MainThreadMarker::new()?;
    let target = screens::screen_for_port(window.port()).map(|s| s.id);
    let screen = target
        .and_then(|id| appkit::screen_with_id(id, mtm))
        .or_else(|| appkit::main_screen(mtm));
    let frame = if borderless {
        appkit::screen_frame(screen.as_deref())
    } else {
        let mode = window.mode().or(window.offered()).unwrap_or(DEFAULT_MODE);
        appkit::windowed_frame(
            screen.as_deref(),
            mode,
            window.guest_scale(mode),
            taken,
            mtm,
        )
    };
    let title = c_string(window.title());
    let flags =
        SDL_WINDOW_METAL | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN;
    // SAFETY: SDL's video runs on the main thread; the title is copied.
    let sdl = unsafe { SDL_CreateWindow(title.as_ptr(), MIN_CONTENT.0, MIN_CONTENT.1, flags) };
    if sdl.is_null() {
        shell::refused("no window", window.title());
        return None;
    }
    let destroy = || {
        // SAFETY: the window just made, which nothing else holds.
        unsafe { SDL_DestroyWindow(sdl) };
    };
    let Some(ns_window) = cocoa_window(sdl) else {
        shell::refused("no NSWindow", window.title());
        destroy();
        return None;
    };
    // SAFETY: SDL's window, on the main thread.
    unsafe { SDL_SetWindowMinimumSize(sdl, MIN_CONTENT.0, MIN_CONTENT.1) };
    if borderless {
        restyle(&ns_window, appkit::borderless_style());
        appkit::look_borderless(&ns_window);
    }
    msg::get("setFrame:display:", || {
        ns_window.setFrame_display(frame, false);
    });
    if !borderless {
        appkit::step_resizes(&ns_window);
    }
    let Some(metal_view) = attach_layer(window, sdl) else {
        destroy();
        return None;
    };
    if borderless {
        appkit::hide_dock_and_menu_bar(mtm);
    }
    // SAFETY: SDL's window, on the main thread.
    unsafe { SDL_ShowWindow(sdl) };
    if borderless {
        // Ordering it front may have moved it below the menu bar.
        msg::get("setFrame:display:", || {
            ns_window.setFrame_display(frame, true);
        });
    }
    occlusion_changed(window, &ns_window);
    Some(Opened {
        sdl,
        asked: sdl_size(sdl),
        input: Input::new(window),
        state: State {
            screen: screen.as_deref().and_then(appkit::display_id),
            ns_window,
            metal_view,
            borderless,
        },
    })
}

/// Resizes a window following the guest's mode for `mode` as AppKit's would
/// (`super::shell::followed_frame`), clear of the windowed frames `taken`.
pub(crate) fn follow(shell: &mut Shell, mode: (u16, u16), taken: &[Frame]) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let ns_window = &shell.os.ns_window;
    let frame = msg::get("frame", || ns_window.frame());
    let resized =
        appkit::followed_frame(ns_window, mode, shell.window.guest_scale(mode), taken, mtm);
    if resized != frame {
        msg::get("setFrame:display:", || {
            ns_window.setFrame_display(resized, true);
        });
    }
    shell.asked = sdl_size(shell.sdl);
}

/// The window's size in points and its pixels per point: its content view's bounds, at its
/// backing scale, as AppKit's view lays out. `None` before it has a size.
pub(crate) fn metrics(shell: &Shell) -> Option<((f64, f64), f64)> {
    let ns_window = &shell.os.ns_window;
    let content = msg::get("contentView", || ns_window.contentView())?;
    let size = msg::get("bounds", || content.bounds()).size;
    let scale = msg::get("backingScaleFactor", || ns_window.backingScaleFactor());
    (size.width > 0.0 && size.height > 0.0 && scale > 0.0)
        .then_some(((size.width, size.height), scale))
}

/// The room the window keeps clear around the image: a borderless window's screen's safe area.
pub(crate) fn insets(shell: &Shell) -> Insets {
    if !shell.os.borderless {
        return Insets::default();
    }
    msg::get("screen", || shell.os.ns_window.screen())
        .map(|screen| appkit::safe_area(&screen))
        .unwrap_or_default()
}

/// The screens changed: a borderless window fills its screen again, or the main screen while its
/// own is gone.
pub(crate) fn screens_changed(shell: &mut Shell) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    if !shell.os.borderless {
        return;
    }
    let screen = shell
        .os
        .screen
        .and_then(|id| appkit::screen_with_id(id, mtm))
        .or_else(|| appkit::main_screen(mtm));
    if let Some(screen) = screen {
        appkit::fill(&shell.os.ns_window, &screen);
    }
}

/// SDL's input events are not taken: AppKit's events are, before the application dispatches
/// them (`input`).
pub(crate) fn input_event(_shell: &mut Shell, _kind: u32, _event: &SDL_Event) -> bool {
    false
}

/// Capture changed: the input holds the system's hot keys off while captured (`super::hotkeys`),
/// and nothing more is grabbed.
pub(crate) fn grab_keyboard(_shell: &Shell, _captured: bool) {}

/// Makes a windowed window not resizable while its input is captured, and resizable again when
/// not, as AppKit's window does (`super::shell::resizable_change`): SDL's resizable flag, which
/// SDL's Cocoa code makes the window's style. A borderless window keeps its style, which SDL would
/// replace with its own windowed one. So does one in native full screen, where SDL's Cocoa code
/// would still set the window's collection behaviour at once, to no full screen while not
/// resizable (read from SDL 3.4.18's code); the shell asks again as the window leaves it.
pub(crate) fn hold_size(shell: &Shell) {
    // SAFETY: SDL's window, on the main thread.
    let flags = unsafe { SDL_GetWindowFlags(shell.sdl) };
    let style = msg::get("styleMask", || shell.os.ns_window.styleMask());
    let Some(resizable) = appkit::resizable_change(
        style,
        shell.input.window_resizable(),
        shell.os.borderless,
        (flags & SDL_WINDOW_FULLSCREEN).0 != 0,
    ) else {
        return;
    };
    // SAFETY: as above.
    if !unsafe { SDL_SetWindowResizable(shell.sdl, resizable) } {
        let what = if resizable {
            "the window stays not resizable"
        } else {
            "the window stays resizable while captured"
        };
        shell::refused(what, shell.window.title());
    }
}

/// The display ended: the window hides, and the surface's reference to the layer goes, on the
/// main thread. SDL's window may go at once.
pub(crate) fn close(shell: &mut Shell) -> bool {
    // SAFETY: SDL's window, on the main thread.
    unsafe { SDL_HideWindow(shell.sdl) };
    drop(shell.window.surface().take_layer());
    true
}

/// The window goes: its Metal view goes, and the Dock and the menu bar come back once no
/// borderless window is open.
pub(crate) fn destroy(shell: Shell) {
    // SAFETY: SDL's view, made with the window, which nothing else destroys.
    unsafe { SDL_Metal_DestroyView(shell.os.metal_view) };
    if shell.os.borderless
        && shell::each_open(|s| s.os.borderless.then_some(())).is_empty()
        && let Some(mtm) = MainThreadMarker::new()
    {
        appkit::show_dock_and_menu_bar(mtm);
    }
}
