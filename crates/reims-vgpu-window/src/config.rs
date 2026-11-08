//! What a host gives a display's window: its flags and the callbacks that carry the window's
//! input to the host.
//!
//! Every callback runs on the window's UI thread, one at a time across every window of the
//! process, in the order the events arrived. On macOS that is the process's main thread, the one
//! AppKit runs on; none may block there: the main thread is the host's too (QEMU's parks there in
//! its run loop), and a callback that waits for something the host's main thread would do never
//! returns. On Linux it is the library's own `reims-vgpu.ui`, where SDL runs: a callback that
//! blocks holds every window's events and their changes, until it returns.

use std::sync::Arc;

/// The left button, in a [`PointerFn`]'s `buttons`.
pub const BUTTON_LEFT: u32 = 1 << 0;
/// The right button.
pub const BUTTON_RIGHT: u32 = 1 << 1;
/// The middle button (the host's third).
pub const BUTTON_MIDDLE: u32 = 1 << 2;
/// The first side button (the host's fourth, "back").
pub const BUTTON_SIDE: u32 = 1 << 3;
/// The second side button (the host's fifth, "forward").
pub const BUTTON_EXTRA: u32 = 1 << 4;

/// `(x, y, width, height, buttons)`: the pointer is at `(x, y)` in the guest's pixels, in a
/// mode `width` by `height`, with the `BUTTON_*` bits of the buttons held. `x` is below `width`
/// and `y` below `height`: a pointer outside the image holds at its nearest edge.
pub type PointerFn = Arc<dyn Fn(u16, u16, u16, u16, u32) + Send + Sync>;
/// `(dy, dx)`: wheel notches. `dy` above 0 is the wheel turned away from the person (up), `dx`
/// above 0 is to the left: the device's own direction, so the guest's scrolling setting applies.
pub type WheelFn = Arc<dyn Fn(i32, i32) + Send + Sync>;
/// Where a scroll event is in the gesture it belongs to, in a [`ScrollFn`]: `NSEvent.phase`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScrollPhase {
    /// In no gesture: a notched wheel, a device that reports no phases, or momentum scrolling.
    #[default]
    None,
    /// Fingers touched the trackpad, and may scroll.
    MayBegin,
    /// The gesture's first scroll.
    Began,
    /// The gesture scrolled on, or its fingers held still (`NSEventPhaseStationary`).
    Changed,
    /// The fingers lifted: momentum scrolling may follow.
    Ended,
    /// The gesture ended without scrolling on, as when fingers that may have begun lift.
    Cancelled,
}

/// Where a scroll event is in the momentum scrolling that follows a gesture, in a [`ScrollFn`]:
/// `NSEvent.momentumPhase`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MomentumPhase {
    /// Not momentum scrolling.
    #[default]
    None,
    /// The first event of momentum scrolling.
    Began,
    /// Momentum scrolling went on.
    Changed,
    /// Momentum scrolling stopped.
    Ended,
}

/// A scroll event, smoothly and as wheel notches, for a [`ScrollFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrollEvent {
    /// The distance up (away from the person), in 1/256 of the host's point: the device's own
    /// direction, so the guest's scrolling setting applies. A notched wheel's event is its lines
    /// at 10 points each.
    pub dy: i32,
    /// The distance left, as `dy`.
    pub dx: i32,
    /// Where the event is in its gesture.
    pub phase: ScrollPhase,
    /// Where the event is in the momentum scrolling after its gesture.
    pub momentum: MomentumPhase,
    /// `(dy, dx)`: the event as the wheel notches a [`WheelFn`] would have had, for a guest that
    /// takes no smooth scrolling.
    pub notches: (i32, i32),
}

/// Smooth scrolling, as a trackpad gives it, with the phases of its gesture.
pub type ScrollFn = Arc<dyn Fn(ScrollEvent) + Send + Sync>;
/// `(usage, down)`: a key of the USB HID keyboard page (0x07) went down or up. The guest's own
/// key repeat applies: a held key is one down.
pub type KeyFn = Arc<dyn Fn(u16, bool) + Send + Sync>;
/// The person closed the window, or chose Quit: the host ends the VM.
pub type CloseFn = Arc<dyn Fn() + Send + Sync>;

/// A window's callbacks, each on the UI thread (the module doc). `None` is a callback the host does not give:
/// that input goes nowhere.
#[derive(Clone, Default)]
pub struct WindowCallbacks {
    /// The pointer moved, or a button went down or up.
    pub pointer: Option<PointerFn>,
    /// The wheel turned.
    pub wheel: Option<WheelFn>,
    /// A key went down or up.
    pub key: Option<KeyFn>,
    /// The window was closed.
    pub close: Option<CloseFn>,
    /// The wheel or a trackpad scrolled, smoothly and as notches. With it, scrolling comes here
    /// and not to `wheel`.
    pub scroll: Option<ScrollFn>,
}

impl std::fmt::Debug for WindowCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowCallbacks")
            .field("pointer", &self.pointer.is_some())
            .field("wheel", &self.wheel.is_some())
            .field("key", &self.key.is_some())
            .field("close", &self.close.is_some())
            .field("scroll", &self.scroll.is_some())
            .finish()
    }
}

/// A display's window.
#[derive(Clone, Debug, Default)]
pub struct WindowConfig {
    /// The mode the display offers the guest first, `None` leaving it to the display. The
    /// window does not read it: the display tells it the mode it offers
    /// ([`Window::offered_mode`](crate::Window::offered_mode)), which the window opens sized for
    /// until the guest sets one.
    pub mode: Option<(u16, u16)>,
    /// The guest follows the window: the window keeps the size it opened at, or the person gave
    /// it, and never resizes for the guest's mode, whose image it fits between black bars as it
    /// does once the person has resized it. The reims-graphics display's window, whose guest
    /// takes the window's size as its mode.
    pub keeps_its_size: bool,
    /// The callbacks.
    pub callbacks: WindowCallbacks,
}
