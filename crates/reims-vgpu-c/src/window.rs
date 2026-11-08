//! Windows: [`RvgWindowConfig`] and [`RvgWindowCallbacks`], the graphics device's display's own
//! window (`RvgGraphicsDeviceConfig.window`).
//!
//! A window's callbacks run one at a time across every window of the process: on macOS on the
//! process's main thread, where AppKit runs, and on Linux on the library's thread
//! `reims-vgpu.ui`. Each calls the host's function under `ffi::host_call`.

use std::ffi::c_void;
use std::sync::Arc;

use reims_vgpu::{
    CloseFn, KeyFn, MomentumPhase, PointerFn, ScrollEvent, ScrollFn, ScrollPhase, WheelFn,
    WindowCallbacks, WindowConfig,
};

use crate::ffi;
use crate::sized::{self, HostStruct};
use crate::table::{RvgDestroyFn, Table};

/// The left button, in an RvgWindowPointerFn's `buttons`.
pub const RVG_WINDOW_BUTTON_LEFT: u32 = 1;

/// The right button.
pub const RVG_WINDOW_BUTTON_RIGHT: u32 = 2;

/// The middle button (the host's third).
pub const RVG_WINDOW_BUTTON_MIDDLE: u32 = 4;

/// The first side button (the host's fourth, "back").
pub const RVG_WINDOW_BUTTON_SIDE: u32 = 8;

/// The second side button (the host's fifth, "forward").
pub const RVG_WINDOW_BUTTON_EXTRA: u32 = 16;

/// `pointer(ctx, x, y, width, height, buttons)`: the pointer is at (`x`, `y`) in the guest's
/// pixels, in a mode `width` by `height`, with the RVG_WINDOW_BUTTON_* bits of the buttons held.
/// `x` is below `width` and `y` below `height`: over the window but off the image, and outside
/// the window during a drag, the pointer holds at the image's nearest edge.
///
/// Thread: on macOS the process's main thread, on Linux the library's thread reims-vgpu.ui;
/// either runs every window callback of the process one at a time, in the order the events
/// arrived, never inside a call of the host's.
/// May call: every function of this header. It must not block: until it returns, no window of
/// the process takes input or redraws its frame.
/// Without it, the pointer goes nowhere.
pub type RvgWindowPointerFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        buttons: u32,
    ),
>;

/// `wheel(ctx, dy, dx)`: wheel notches. `dy` above 0 is the wheel turned away from the person
/// (up), `dx` above 0 is to the left: the device's own direction, whatever the host's scrolling
/// setting, so the guest's applies.
///
/// Thread, and what it may call: as RvgWindowPointerFn. Not called where the table has
/// `scroll`. Without either, scrolling goes nowhere.
pub type RvgWindowWheelFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, dy: i32, dx: i32)>;

/// `key(ctx, usage, down)`: the key with USB HID keyboard usage `usage` (page 0x07) went down
/// or up. A held key is one down: the guest repeats it. Every key still down gets its up when
/// the window loses focus, releases capture or closes.
///
/// Thread, and what it may call: as RvgWindowPointerFn. Without it, keys go nowhere.
pub type RvgWindowKeyFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, usage: u16, down: bool)>;

/// `close(ctx)`: the person closed the window, or chose Quit. The window stays open: the host
/// ends the VM, and its release of the device closes the window.
///
/// Thread, and what it may call: as RvgWindowPointerFn. Without it, closing the window does
/// nothing, and Quit is ignored.
pub type RvgWindowCloseFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// In an RvgWindowScroll's `phase`: the event is in no gesture (a notched wheel, a device that
/// reports no phases, or momentum scrolling).
pub const RVG_WINDOW_SCROLL_PHASE_NONE: u32 = 0;

/// Fingers touched the trackpad, and may scroll.
pub const RVG_WINDOW_SCROLL_PHASE_MAY_BEGIN: u32 = 1;

/// The gesture's first scroll.
pub const RVG_WINDOW_SCROLL_PHASE_BEGAN: u32 = 2;

/// The gesture scrolled on, or its fingers held still.
pub const RVG_WINDOW_SCROLL_PHASE_CHANGED: u32 = 3;

/// The fingers lifted: momentum scrolling may follow.
pub const RVG_WINDOW_SCROLL_PHASE_ENDED: u32 = 4;

/// The gesture ended without scrolling on, as when fingers that may have begun lift.
pub const RVG_WINDOW_SCROLL_PHASE_CANCELLED: u32 = 5;

/// In an RvgWindowScroll's `momentum`: the event is not momentum scrolling.
pub const RVG_WINDOW_SCROLL_MOMENTUM_NONE: u32 = 0;

/// The first event of the momentum scrolling after a gesture.
pub const RVG_WINDOW_SCROLL_MOMENTUM_BEGAN: u32 = 1;

/// Momentum scrolling went on.
pub const RVG_WINDOW_SCROLL_MOMENTUM_CHANGED: u32 = 2;

/// Momentum scrolling stopped.
pub const RVG_WINDOW_SCROLL_MOMENTUM_ENDED: u32 = 3;

/// A scroll event, smoothly and as wheel notches, for an RvgWindowScrollFn.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvgWindowScroll {
    /// sizeof(RvgWindowScroll) as the library was compiled.
    pub struct_size: u32,
    /// The distance up (away from the person), in 1/256 of the host's point: the device's own
    /// direction, whatever the host's scrolling setting, so the guest's applies. A notched
    /// wheel's event is its lines at 10 points each.
    pub dy: i32,
    /// The distance left, as `dy`.
    pub dx: i32,
    /// Where the event is in its gesture: RVG_WINDOW_SCROLL_PHASE_*.
    pub phase: u32,
    /// Where the event is in the momentum scrolling after its gesture:
    /// RVG_WINDOW_SCROLL_MOMENTUM_*.
    pub momentum: u32,
    /// The event as the wheel notches RvgWindowWheelFn would have had, for a guest that takes
    /// no smooth scrolling: up.
    pub notches_dy: i32,
    /// As `notches_dy`, to the left.
    pub notches_dx: i32,
}

/// `scroll(ctx, scroll)`: the wheel or a trackpad scrolled. `scroll` holds the event smoothly,
/// as a trackpad gives it, with the phases of its gesture, and as wheel notches; it lives for
/// the call. A gesture that begins over the image comes here to its end, momentum included,
/// wherever the pointer goes; one that begins elsewhere does not, and an event of no phase comes
/// where the pointer is over the image.
///
/// Thread, and what it may call: as RvgWindowPointerFn. With it, scrolling comes here and not to
/// `wheel`.
pub type RvgWindowScrollFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, scroll: *const RvgWindowScroll)>;

/// A window's callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgWindowCallbacks {
    /// sizeof(RvgWindowCallbacks) as the host was compiled.
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Runs once after the device's release has closed the window, when no callback of the
    /// table runs again.
    pub destroy: RvgDestroyFn,
    pub pointer: RvgWindowPointerFn,
    pub wheel: RvgWindowWheelFn,
    pub key: RvgWindowKeyFn,
    pub close: RvgWindowCloseFn,
    pub scroll: RvgWindowScrollFn,
}

// SAFETY: repr(C), a u32 first; a pointer and optional function pointers.
unsafe impl HostStruct for RvgWindowCallbacks {}

/// `phase`'s RVG_WINDOW_SCROLL_PHASE_*.
fn phase_value(phase: ScrollPhase) -> u32 {
    match phase {
        ScrollPhase::None => RVG_WINDOW_SCROLL_PHASE_NONE,
        ScrollPhase::MayBegin => RVG_WINDOW_SCROLL_PHASE_MAY_BEGIN,
        ScrollPhase::Began => RVG_WINDOW_SCROLL_PHASE_BEGAN,
        ScrollPhase::Changed => RVG_WINDOW_SCROLL_PHASE_CHANGED,
        ScrollPhase::Ended => RVG_WINDOW_SCROLL_PHASE_ENDED,
        ScrollPhase::Cancelled => RVG_WINDOW_SCROLL_PHASE_CANCELLED,
    }
}

/// `momentum`'s RVG_WINDOW_SCROLL_MOMENTUM_*.
fn momentum_value(momentum: MomentumPhase) -> u32 {
    match momentum {
        MomentumPhase::None => RVG_WINDOW_SCROLL_MOMENTUM_NONE,
        MomentumPhase::Began => RVG_WINDOW_SCROLL_MOMENTUM_BEGAN,
        MomentumPhase::Changed => RVG_WINDOW_SCROLL_MOMENTUM_CHANGED,
        MomentumPhase::Ended => RVG_WINDOW_SCROLL_MOMENTUM_ENDED,
    }
}

/// The graphics device's window (RvgGraphicsDeviceConfig's `window`).
///
/// The library opens a window for the display on one of the host's screens, and draws the
/// guest's scanout into it on the GPU. Its keyboard, pointer and wheel go to `callbacks`. On
/// macOS the window opens once the process's main thread runs its run loop (CFRunLoopRun, or
/// NSApplication's), and on Linux on the library's thread reims-vgpu.ui, which needs nothing of
/// the main thread; the library never waits for the main thread. The environment picks the
/// screen and the style (REIMS_VGPU_WINDOW_SCREENS, REIMS_VGPU_WINDOW_BORDERLESS).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgWindowConfig {
    /// sizeof(RvgWindowConfig) as the host was compiled.
    pub struct_size: u32,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgWindowCallbacks,
    /// The size in pixels the window opens at, and the mode the guest is offered until the
    /// window's own size is known; 0 in either is 1920 × 1080.
    pub mode_width: u16,
    /// As `mode_width`.
    pub mode_height: u16,
}

// SAFETY: repr(C), a u32 first; integers and a pointer.
unsafe impl HostStruct for RvgWindowConfig {}

impl RvgWindowConfig {
    /// The Rust API's window config over `callbacks`, the table's closures: the mode is `mode`,
    /// `None` where either side is 0.
    pub(crate) fn to_config(self, callbacks: WindowCallbacks) -> WindowConfig {
        let mode = (self.mode_width != 0 && self.mode_height != 0)
            .then_some((self.mode_width, self.mode_height));
        WindowConfig {
            mode,
            keeps_its_size: false,
            callbacks,
        }
    }
}

/// The Rust API's closures over a window table: each holds the table, and calls its function
/// with the table's `ctx`.
pub(crate) fn window_callbacks(t: &Arc<Table>, c: &RvgWindowCallbacks) -> WindowCallbacks {
    WindowCallbacks {
        pointer: c.pointer.map(|f| {
            let t = t.clone();
            let c: PointerFn = Arc::new(move |x, y, width, height, buttons| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("pointer", || unsafe {
                    f(t.ctx(), x, y, width, height, buttons)
                });
            });
            c
        }),
        wheel: c.wheel.map(|f| {
            let t = t.clone();
            let c: WheelFn = Arc::new(move |dy, dx| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("wheel", || unsafe { f(t.ctx(), dy, dx) });
            });
            c
        }),
        key: c.key.map(|f| {
            let t = t.clone();
            let c: KeyFn = Arc::new(move |usage, down| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("key", || unsafe { f(t.ctx(), usage, down) });
            });
            c
        }),
        close: c.close.map(|f| {
            let t = t.clone();
            let c: CloseFn = Arc::new(move || {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("close", || unsafe { f(t.ctx()) });
            });
            c
        }),
        scroll: c.scroll.map(|f| {
            let t = t.clone();
            let c: ScrollFn = Arc::new(move |e: ScrollEvent| {
                let scroll = RvgWindowScroll {
                    struct_size: sized::size_u32::<RvgWindowScroll>(),
                    dy: e.dy,
                    dx: e.dx,
                    phase: phase_value(e.phase),
                    momentum: momentum_value(e.momentum),
                    notches_dy: e.notches.0,
                    notches_dx: e.notches.1,
                };
                // SAFETY: the host's function over its ctx, and the event, live for the call.
                ffi::host_call("scroll", || unsafe { f(t.ctx(), &raw const scroll) });
            });
            c
        }),
    }
}

#[cfg(test)]
mod tests;
