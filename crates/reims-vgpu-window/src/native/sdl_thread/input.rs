//! A window's input through SDL: its pointer, wheel and key events to the host's callbacks
//! ([`crate::config::WindowCallbacks`]), on the UI thread.
//!
//! [`Input`] belongs to one window. The UI thread reads each SDL event's values and hands them
//! to it with what the window knows ([`At`]: the event's point in the window, the image's
//! rectangle, the guest's mode). It decides with the crate's pure logic ([`crate::keys`],
//! [`crate::capture`], [`crate::geometry`], [`crate::scroll`]) and queues what to send, which
//! [`send`] then hands the host's callbacks, taken from the [`crate::handle::Window`] at each
//! event: after the window's end there are none, and nothing is sent.
//!
//! What the guest gets:
//! - **The pointer** once the guest has a mode, at the guest pixel under the host's pointer, held
//!   at the image's nearest edge outside it; a move is sent where that pixel changes. A click in
//!   the image captures the keyboard and the pointer; the click that focuses the window (one that
//!   comes within [`FOCUS_CLICK_NS`] of the window gaining the keyboard) is not sent. A click
//!   outside the image is not sent and starts nothing. A button's release is sent wherever it
//!   happens, so a drag that leaves the image ends in the guest. The left, middle and right
//!   buttons and the two side buttons each have their bit.
//! - **The wheel** over the image, once the guest has a mode: SDL's whole notches as wheel
//!   notches, or, where the host takes smooth scrolling, each event's distance at
//!   [`crate::scroll::POINTS_PER_LINE`] points a line, with its notches, in no gesture (SDL gives
//!   no phases). The guest gets the device's own direction: SDL's "flipped" (natural scrolling)
//!   deltas are flipped back, so the guest's own setting applies; left is above 0, as the host's
//!   callbacks take it.
//! - **Keys** by position ([`super::keymap`]), one down and one up each: the host's repeats are
//!   dropped, since the guest repeats a held key itself. With no menu of the host's to drive,
//!   every key goes to the guest, captured or not, but what the compositor keeps for itself while
//!   not captured. Pressing Ctrl, Alt and Super together while captured lifts every key held and
//!   releases capture: the chord the title names.
//! - **No stuck keys**: losing the keyboard, releasing capture and closing send an up for every
//!   key held; losing the keyboard and closing lift the buttons held too.
//!
//! While captured, the window grabs the keyboard (`SDL_SetWindowKeyboardGrab`): on Wayland the
//! compositor's shortcuts inhibitor, so its own shortcuts reach the guest; and SDL holds the
//! pointer over the image ([`Input::pointer_confinement`], `crate::sdl::shell`).

use crate::capture::{Capture, UNCAPTURED_SUFFIX};
use crate::config::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, ScrollEvent,
    WindowCallbacks,
};
use crate::geometry::{self, Rect};
use crate::keys::{KeyEvent, KeyState};
use crate::scroll::POINTS_PER_LINE;

/// How soon after the window gains the keyboard a click is the one that focused it, in
/// nanoseconds of SDL's event clock: the compositor gives the keyboard and the click together.
pub(crate) const FOCUS_CLICK_NS: u64 = 100_000_000;

/// What the title ends with while captured: what capture is on and how to release it.
const CAPTURED_SUFFIX: &str = " \u{2014} input captured (press Ctrl+Alt+Super to release)";

/// What the window knows when an event reaches it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct At {
    /// The event's point in the window, in points (y down), for a pointer or wheel event.
    pub(crate) point: Option<(f64, f64)>,
    /// The image's rectangle in the window, in points.
    pub(crate) image: Rect,
    /// The guest's mode, once it has set one.
    pub(crate) guest: Option<(u16, u16)>,
}

/// What the window does after an event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reaction {
    /// The capture state changed: the keyboard grab, the pointer's hold, the title and the cursor
    /// follow it.
    pub(crate) capture_changed: bool,
}

/// What the input sends the host, in the order it is sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToHost {
    /// `pointer(x, y, width, height, buttons)`.
    Pointer {
        /// The guest pixel.
        at: (u16, u16),
        /// The guest's mode.
        mode: (u16, u16),
        /// The `BUTTON_*` bits held.
        buttons: u32,
    },
    /// `wheel(dy, dx)` or, where the host takes it, `scroll(event)`.
    Wheel {
        /// Notches up, then left.
        notches: (i32, i32),
        /// The smooth distance up, then left, in 1/256 point.
        smooth: (i32, i32),
    },
    /// `key(usage, down)`.
    Key(KeyEvent),
}

/// One SDL wheel event's values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Wheel {
    /// SDL's whole notches, right then away from the person (`integer_x`, `integer_y`).
    pub(crate) notches: (i32, i32),
    /// SDL's distance in lines, right then away (`x`, `y`).
    pub(crate) lines: (f32, f32),
    /// SDL reports the deltas flipped (natural scrolling on).
    pub(crate) flipped: bool,
}

/// The `BUTTON_*` bit of SDL's button `button` (`SDL_BUTTON_LEFT` 1 to `SDL_BUTTON_X2` 5), 0 for
/// any other.
pub(crate) fn button_bit(button: u8) -> u32 {
    match button {
        1 => BUTTON_LEFT,
        2 => BUTTON_MIDDLE,
        3 => BUTTON_RIGHT,
        4 => BUTTON_SIDE,
        5 => BUTTON_EXTRA,
        _ => 0,
    }
}

/// The guest's mode in `at`, where it has one with pixels: the pointer is sent only then.
fn guest_mode(at: &At) -> Option<(u16, u16)> {
    at.guest.filter(|&(width, height)| width > 0 && height > 0)
}

/// `points` in 1/256 point, rounded; 0 for what is not a number.
fn fixed(points: f64) -> i32 {
    if points.is_finite() {
        (points * 256.0)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    } else {
        0
    }
}

/// One window's input decisions (the module doc).
#[derive(Debug, Default)]
pub(crate) struct Input {
    /// The keys the guest was told are down.
    keys: KeyState,
    /// Who gets the keyboard.
    capture: Capture,
    /// The `BUTTON_*` bits of the buttons the guest was told are down.
    buttons: u32,
    /// The pointer event last sent.
    last: Option<ToHost>,
    /// Whether the window has the keyboard.
    focused: bool,
    /// When the window last gained the keyboard, on SDL's event clock.
    focused_at: Option<u64>,
    /// What to send, in order, since [`send`] last took it.
    out: Vec<ToHost>,
}

impl Input {
    /// Nothing held, nothing captured, no keyboard.
    pub(crate) fn new() -> Input {
        Input::default()
    }

    /// Whether input, the keyboard and the pointer, is captured.
    pub(crate) fn is_captured(&self) -> bool {
        self.capture.is_captured()
    }

    /// Where the pointer is held over `image`, the image's rectangle in the window's points: the
    /// image while captured ([`Capture::pointer_confinement`]).
    pub(crate) fn pointer_confinement(&self, image: Rect) -> Option<Rect> {
        self.capture.pointer_confinement(image)
    }

    /// What the title ends with: the release chord while captured, that a click captures while not.
    pub(crate) fn title_suffix(&self) -> &'static str {
        if self.is_captured() {
            CAPTURED_SUFFIX
        } else {
            UNCAPTURED_SUFFIX
        }
    }

    /// What to send, taken.
    pub(crate) fn take(&mut self) -> Vec<ToHost> {
        std::mem::take(&mut self.out)
    }

    /// Queues the pointer at `point` over `image` in `mode`, with the buttons held, where it
    /// differs from the one last sent (or `always`).
    fn pointer(&mut self, point: (f64, f64), image: Rect, mode: (u16, u16), always: bool) {
        let event = ToHost::Pointer {
            at: geometry::pointer_to_guest(point, image, mode),
            mode,
            buttons: self.buttons,
        };
        if always || self.last != Some(event) {
            self.last = Some(event);
            self.out.push(event);
        }
    }

    /// The pointer moved.
    pub(crate) fn moved(&mut self, at: &At) {
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            self.pointer(point, at.image, mode, false);
        }
    }

    /// Button `bit` went down at `timestamp` (SDL's event clock).
    pub(crate) fn button_down(&mut self, bit: u32, timestamp: u64, at: &At) -> Reaction {
        let (Some(point), Some(mode)) = (at.point, guest_mode(at)) else {
            return Reaction::default();
        };
        if bit == 0 || !geometry::contains(at.image, point) {
            return Reaction::default();
        }
        let focusing = !self.focused
            || self
                .focused_at
                .is_some_and(|t| timestamp.saturating_sub(t) < FOCUS_CLICK_NS);
        let was_captured = self.is_captured();
        let forward = if was_captured {
            true
        } else {
            self.capture.click_in_image(!focusing).forward
        };
        if forward {
            self.buttons |= bit;
            self.pointer(point, at.image, mode, true);
        }
        Reaction {
            capture_changed: self.is_captured() != was_captured,
        }
    }

    /// Button `bit` went up: sent where the guest was told it is down, wherever the pointer is.
    pub(crate) fn button_up(&mut self, bit: u32, at: &At) {
        if self.buttons & bit == 0 {
            return;
        }
        self.buttons &= !bit;
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            self.pointer(point, at.image, mode, true);
        }
    }

    /// The wheel turned (the module doc).
    pub(crate) fn wheel(&mut self, wheel: Wheel, at: &At) {
        let (Some(point), Some(_)) = (at.point, guest_mode(at)) else {
            return;
        };
        if !geometry::contains(at.image, point) {
            return;
        }
        self.moved(at);
        let sign = if wheel.flipped { -1 } else { 1 };
        let notches = (
            wheel.notches.1.saturating_mul(sign),
            wheel.notches.0.saturating_mul(-sign),
        );
        let lines = (f64::from(wheel.lines.1), f64::from(wheel.lines.0));
        let sign = f64::from(sign);
        let smooth = (
            fixed(sign * POINTS_PER_LINE * lines.0),
            fixed(-sign * POINTS_PER_LINE * lines.1),
        );
        if notches != (0, 0) || smooth != (0, 0) {
            self.out.push(ToHost::Wheel { notches, smooth });
        }
    }

    /// A key, `usage`, went down (`down`) or up; `repeat` for the host's repeat of a held key.
    pub(crate) fn key(&mut self, usage: u16, down: bool, repeat: bool) -> Reaction {
        if !self.focused {
            return Reaction::default();
        }
        if !down {
            if let Some(event) = self.keys.key_up(usage) {
                self.out.push(ToHost::Key(event));
            }
            return Reaction::default();
        }
        if let Some(event) = self.keys.key_down(usage, repeat) {
            self.out.push(ToHost::Key(event));
        }
        if self.capture.release_chord(self.keys.held_modifiers()) {
            return self.release_capture();
        }
        Reaction::default()
    }

    /// The window gained the keyboard at `timestamp`.
    pub(crate) fn focus_gained(&mut self, timestamp: u64) {
        self.focused = true;
        self.focused_at = Some(timestamp);
    }

    /// The window lost the keyboard: every key and button held is lifted, and capture released.
    pub(crate) fn focus_lost(&mut self) -> Reaction {
        self.focused = false;
        self.lift_keys();
        self.lift_buttons();
        Reaction {
            capture_changed: self.capture.resign_key(),
        }
    }

    /// Releases capture, lifting every key held.
    pub(crate) fn release_capture(&mut self) -> Reaction {
        self.lift_keys();
        Reaction {
            capture_changed: self.capture.release(),
        }
    }

    /// The window is closing: every key and button held is lifted.
    pub(crate) fn closing(&mut self) {
        self.lift_keys();
        self.lift_buttons();
        self.capture.release();
    }

    /// Ups for every key held.
    fn lift_keys(&mut self) {
        let ups = self.keys.release_all();
        self.out.extend(ups.into_iter().map(ToHost::Key));
    }

    /// A pointer event with no button, where one is held, at the last pointer's pixel.
    fn lift_buttons(&mut self) {
        if self.buttons == 0 {
            return;
        }
        self.buttons = 0;
        if let Some(ToHost::Pointer { at, mode, .. }) = self.last {
            let event = ToHost::Pointer {
                at,
                mode,
                buttons: 0,
            };
            self.last = Some(event);
            self.out.push(event);
        }
    }
}

/// Hands `event` to the host's `callbacks`. Smooth scrolling goes to `scroll` where the host
/// gives it, else notches to `wheel`.
pub(crate) fn send(callbacks: &WindowCallbacks, event: ToHost) {
    match event {
        ToHost::Pointer { at, mode, buttons } => {
            if let Some(f) = &callbacks.pointer {
                f(at.0, at.1, mode.0, mode.1, buttons);
            }
        }
        ToHost::Wheel { notches, smooth } => {
            if let Some(f) = &callbacks.scroll {
                f(ScrollEvent {
                    dy: smooth.0,
                    dx: smooth.1,
                    notches,
                    ..ScrollEvent::default()
                });
            } else if let Some(f) = &callbacks.wheel
                && notches != (0, 0)
            {
                f(notches.0, notches.1);
            }
        }
        ToHost::Key(KeyEvent { usage, down }) => {
            if let Some(f) = &callbacks.key {
                f(usage, down);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 100×50 image at the window's origin, a 200×100 guest mode, the pointer at `point`.
    fn at(point: (f64, f64)) -> At {
        At {
            point: Some(point),
            image: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0,
            },
            guest: Some((200, 100)),
        }
    }

    /// An input whose window has had the keyboard for a second.
    fn focused() -> Input {
        let mut input = Input::new();
        input.focus_gained(0);
        input
    }

    /// The pointer event at guest pixel `(x, y)` with `buttons`.
    fn pointer(x: u16, y: u16, buttons: u32) -> ToHost {
        ToHost::Pointer {
            at: (x, y),
            mode: (200, 100),
            buttons,
        }
    }

    /// The click that focuses the window is not sent, but captures; the next one is sent
    #[test]
    fn the_click_that_focuses_the_window_is_not_sent() {
        let mut input = Input::new();
        input.focus_gained(1_000_000_000);
        let r = input.button_down(BUTTON_LEFT, 1_010_000_000, &at((10.0, 10.0)));
        assert!(r.capture_changed);
        input.button_up(BUTTON_LEFT, &at((10.0, 10.0)));
        assert_eq!(input.take(), vec![]);
        input.button_down(BUTTON_LEFT, 2_000_000_000, &at((10.0, 10.0)));
        assert_eq!(input.take(), vec![pointer(20, 20, BUTTON_LEFT)]);
    }

    /// A click in the image captures and is sent; one outside starts nothing
    #[test]
    fn a_click_in_the_image_captures_and_is_sent() {
        let mut input = focused();
        assert!(
            !input
                .button_down(BUTTON_RIGHT, 1_000_000_000, &at((120.0, 10.0)))
                .capture_changed
        );
        assert_eq!(input.take(), vec![]);
        let r = input.button_down(BUTTON_RIGHT, 1_000_000_000, &at((50.0, 25.0)));
        assert!(r.capture_changed && input.is_captured());
        assert_eq!(input.take(), vec![pointer(100, 50, BUTTON_RIGHT)]);
    }

    /// A drag ends in the guest wherever the button comes up, held at the nearest edge
    #[test]
    fn a_drag_ends_in_the_guest_outside_the_window() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.moved(&at((150.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((150.0, 10.0)));
        assert_eq!(
            input.take(),
            vec![
                pointer(20, 20, BUTTON_LEFT),
                pointer(199, 20, BUTTON_LEFT),
                pointer(199, 20, 0),
            ]
        );
    }

    /// Before the guest has a mode, the pointer sends nothing
    #[test]
    fn before_the_guest_has_a_mode_the_pointer_sends_nothing() {
        let mut input = focused();
        let no_mode = At {
            guest: None,
            ..at((10.0, 10.0))
        };
        input.moved(&no_mode);
        input.button_down(BUTTON_LEFT, 1_000_000_000, &no_mode);
        assert_eq!(input.take(), vec![]);
    }

    /// Moves are sent where the guest pixel changes
    #[test]
    fn moves_are_sent_where_the_guest_pixel_changes() {
        let mut input = focused();
        input.moved(&at((10.0, 10.0)));
        input.moved(&at((10.2, 10.2)));
        input.moved(&at((11.0, 10.0)));
        assert_eq!(input.take(), vec![pointer(20, 20, 0), pointer(22, 20, 0)]);
    }

    /// Host repeats are dropped; each key is one down and one up
    #[test]
    fn host_repeats_are_dropped() {
        let mut input = focused();
        input.key(0x04, true, false);
        input.key(0x04, true, true);
        input.key(0x04, false, false);
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0x04)),
                ToHost::Key(KeyEvent::up(0x04))
            ]
        );
    }

    /// Ctrl+Alt+Super while captured lifts every key and releases capture
    #[test]
    fn ctrl_alt_super_releases_capture() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((10.0, 10.0)));
        assert!(input.is_captured());
        input.take();
        input.key(0xE0, true, false);
        input.key(0xE2, true, false);
        let r = input.key(0xE3, true, false);
        assert!(r.capture_changed && !input.is_captured());
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0xE0)),
                ToHost::Key(KeyEvent::down(0xE2)),
                ToHost::Key(KeyEvent::down(0xE3)),
                ToHost::Key(KeyEvent::up(0xE3)),
                ToHost::Key(KeyEvent::up(0xE2)),
                ToHost::Key(KeyEvent::up(0xE0)),
            ]
        );
        assert_eq!(input.title_suffix(), UNCAPTURED_SUFFIX);
    }

    /// Losing the keyboard lifts every key and button and releases capture
    #[test]
    fn losing_the_keyboard_leaves_nothing_stuck() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.key(0x04, true, false);
        input.take();
        let r = input.focus_lost();
        assert!(r.capture_changed);
        assert_eq!(
            input.take(),
            vec![ToHost::Key(KeyEvent::up(0x04)), pointer(20, 20, 0)]
        );
        input.key(0x05, true, false);
        assert_eq!(input.take(), vec![], "no keys without the keyboard");
    }

    /// Closing lifts every key and button and releases capture
    #[test]
    fn closing_leaves_nothing_stuck() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.key(0x04, true, false);
        input.take();
        assert!(input.is_captured());
        input.closing();
        assert!(!input.is_captured());
        assert_eq!(
            input.take(),
            vec![ToHost::Key(KeyEvent::up(0x04)), pointer(20, 20, 0)]
        );
    }

    /// The wheel over the image sends notches up and left in the device's own direction, and
    /// lines as smooth scrolling at 10 points each
    #[test]
    fn the_wheel_sends_the_devices_own_direction() {
        let mut input = focused();
        let up = Wheel {
            notches: (0, 1),
            lines: (0.0, 1.0),
            flipped: false,
        };
        input.wheel(up, &at((10.0, 10.0)));
        let flipped_right = Wheel {
            notches: (-2, 0),
            lines: (-2.0, 0.0),
            flipped: true,
        };
        input.wheel(flipped_right, &at((10.0, 10.0)));
        input.wheel(up, &at((150.0, 10.0)));
        assert_eq!(
            input.take(),
            vec![
                pointer(20, 20, 0),
                ToHost::Wheel {
                    notches: (1, 0),
                    smooth: (2560, 0)
                },
                ToHost::Wheel {
                    notches: (0, -2),
                    smooth: (0, -5120)
                },
            ]
        );
    }

    /// SDL's buttons to the host's bits
    #[test]
    fn sdl_buttons_are_the_hosts_bits() {
        assert_eq!(button_bit(1), BUTTON_LEFT);
        assert_eq!(button_bit(2), BUTTON_MIDDLE);
        assert_eq!(button_bit(3), BUTTON_RIGHT);
        assert_eq!(button_bit(4), BUTTON_SIDE);
        assert_eq!(button_bit(5), BUTTON_EXTRA);
        assert_eq!(button_bit(6), 0);
    }
}
