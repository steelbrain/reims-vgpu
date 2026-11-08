//! A guest display's own window: on macOS an AppKit window per display, whose `CAMetalLayer`
//! the display's current frame is drawn into in one GPU pass, and whose keyboard, pointer and
//! scroll input goes to the host through [`WindowCallbacks`] (or, built with the `macos-sdl`
//! feature and run with `REIMS_VGPU_WINDOW_IMPL=sdl`, SDL3's window over Cocoa, drawn and taking
//! input the same way);
//! on Linux an SDL3 window per display, on Wayland or X11, whose Vulkan swapchain the frame is
//! drawn into the same way, on the runtime's own device. On macOS's Vulkan platform the frame goes
//! into a Vulkan swapchain over the window's `CAMetalLayer` likewise, through the same presenter
//! (`present`).
//!
//! What the window does for the person at the host is listed in `features.md`, next to this
//! crate's manifest: each entry there is deliberate, and names the test that holds it.
//!
//! The crate knows nothing of devices or displays. `reims-vgpu` makes a [`Window`] for a display
//! created with a [`WindowConfig`], tells it what the guest does (a new frame, a mode, the cursor)
//! and gives it a [`FrameSink`] that presents the display's current frame in it. On macOS
//! the window opens once the process's main thread runs its run loop, as QEMU's does under
//! `-display none`: nothing here waits for the main thread. On Linux it opens on the library's
//! UI thread, and the process's main thread is never used.
//!
//! Modules:
//! - what a window decides, with no AppKit in it, each unit-tested: [`geometry`] (the image's
//!   rectangle and the pointer's position in it), [`keymap`] (macOS key codes to USB usages),
//!   [`keys`] (modifiers and the keys held down), [`capture`] (who gets the keyboard, and where
//!   the pointer may go), [`scroll`] (wheel and trackpad deltas to notches or smooth scrolling),
//!   [`policy`] (the modes and density the guest is offered for a host screen) and
//!   [`env`](mod@env) (the `REIMS_VGPU_WINDOW_*` variables);
//! - the host's screens ([`screens`]), as the OS lists them;
//! - the window itself: [`handle`] (the [`Window`] the display holds), and its OS side
//!   (`native`, one directory per OS behind one file). On macOS (`native/macos/`) that is the
//!   AppKit side, on the main thread but for the render thread: `main_thread` (work posted to the
//!   main thread, and the state only it touches), `app` (the application, its menu and its end),
//!   `shell` (a display's `NSWindow` and its delegate), `view` (its content view, layer and
//!   layout), `render` (the thread that draws each frame into the layer), `input` (the view's and
//!   the application's events to the host's callbacks), `cursor` (the guest's cursor as the
//!   host's) and `hotkeys` (the system's hot keys while captured). On Linux (`native/linux/`) it
//!   is SDL's: a UI thread of the library's own that SDL runs on (`ui`), a window's input there
//!   (`input`, `keymap`), and its render thread (`render`), which draws through a presenter the
//!   display's sink makes over the window's swapchain (`present`, which macOS's render thread
//!   draws through too on the Vulkan platform), with the window's shell and cursor SDL's shared
//!   ones (`sdl`). SDL's window on macOS shares `sdl`'s shell and cursor, and AppKit's
//!   application, render thread, input and hot keys, with its own side in `native/macos/sdl.rs`.

pub mod capture;
pub mod config;
pub mod env;
pub mod geometry;
pub mod handle;
pub mod keymap;
pub mod keys;
mod native;
pub mod policy;
mod present;
pub mod screens;
pub mod scroll;

// The SDL3 window: Linux's (`native/linux.rs`), and macOS's with the `macos-sdl` feature, behind
// `REIMS_VGPU_WINDOW_IMPL`.
#[cfg(any(target_os = "linux", feature = "macos-sdl"))]
mod sdl;

pub use config::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, CloseFn, KeyFn,
    MomentumPhase, PointerFn, ScrollEvent, ScrollFn, ScrollPhase, WheelFn, WindowCallbacks,
    WindowConfig,
};
pub use handle::{DrawnCursor, FrameSink, Glyph, PixelRect, Window};
pub use policy::{Density, OfferedMode};
pub use present::{Drawn, Presenter, VulkanTarget};
pub use screens::HostScreen;

#[cfg(test)]
mod features;
