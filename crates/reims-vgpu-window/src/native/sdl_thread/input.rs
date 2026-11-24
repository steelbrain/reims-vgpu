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
//!   not captured. Pressing Ctrl, Alt and SDL's GUI key together while captured lifts every key
//!   held and releases capture: the chord the title names, as the host's OS names its keys
//!   (`super::super::HOST_OS`): Ctrl+Alt+Super on Linux, Ctrl+Alt+Win on Windows.
//! - **No stuck keys**: losing the keyboard, releasing capture and closing send an up for every
//!   key held; losing the keyboard and closing lift the buttons held too.
//! - **With capture off** ([`crate::capture`], the VM's setting): a click in the image is sent,
//!   but nothing is captured, grabbed, held or locked; the pointer is sent over the image alone,
//!   and outside it only in a drag; keys go to the guest while the window has the keyboard, but
//!   what the compositor keeps. A game gets the absolute pointer alone, which stops at the
//!   image's edge.
//!
//! - **For games**, where the guest has its own mouse ([`crate::PointerMotion`]): captured, while
//!   the guest hides its cursor on every display, the window locks the host's pointer (SDL's
//!   relative mode, its motion the device's own counts) and hands its motion, its buttons
//!   pressed meanwhile and its wheel to the guest's mouse ([`crate::game`]), until the guest shows
//!   its cursor again, where the host's pointer is let go over it; a button comes up by the route
//!   it went down by. Releasing capture, losing the keyboard and closing let the pointer go where
//!   it was held.
//! - **Past the image's edge**, where the host gives the window the guest's other displays
//!   ([`crate::PointerBeyond`]): the captured pointer moving toward an edge another display meets,
//!   within reach of it, goes on onto it ([`crate::beyond`]). The window locks the host's pointer
//!   (SDL's relative mode, [`Input::take_hold`]) and hands its motion to the guest's desktop,
//!   which sends the guest the pointer, and the buttons and their changes, on whichever display
//!   it lands on, through that display's window; the wheel goes on where the pointer is. Back
//!   over the window's display, out of reach of the edge, the pointer is unlocked there and is
//!   the window's again; releasing capture, losing the keyboard and closing unlock it where it
//!   was held.
//!
//! While captured, the window grabs the keyboard (`SDL_SetWindowKeyboardGrab`): on Wayland the
//! compositor's shortcuts inhibitor, so its own shortcuts reach the guest; and SDL holds the
//! pointer over the image ([`Input::pointer_confinement`], `crate::sdl::shell`).

use std::sync::Arc;

use super::super::HOST_OS;
use crate::beyond::{Back, Beyond, Hold, point_of};
use crate::capture::Capture;
use crate::config::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, PointerBeyond,
    PointerMotion, ScrollEvent, WindowCallbacks,
};
use crate::game::Game;
use crate::geometry::{self, Rect};
use crate::keys::{KeyEvent, KeyState};
use crate::scroll::POINTS_PER_LINE;

/// How soon after the window gains the keyboard a click is the one that focused it, in
/// nanoseconds of SDL's event clock: the compositor gives the keyboard and the click together.
pub(crate) const FOCUS_CLICK_NS: u64 = 100_000_000;

/// What the window knows when an event reaches it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct At {
    /// The event's point in the window, in points (y down), for a pointer or wheel event.
    pub(crate) point: Option<(f64, f64)>,
    /// The image's rectangle in the window, in points.
    pub(crate) image: Rect,
    /// The guest's mode, once it has set one.
    pub(crate) guest: Option<(u16, u16)>,
    /// The guest's cursor, in its mode's pixels, where it shows over the window's display.
    pub(crate) cursor: Option<(u16, u16)>,
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

/// The window's point over the guest's cursor in `at`, where it shows over the window's display.
fn cursor_point(at: &At) -> Option<(f64, f64)> {
    let mode = guest_mode(at)?;
    let cursor = at.cursor?;
    let inside = (cursor.0.min(mode.0 - 1), cursor.1.min(mode.1 - 1));
    Some(point_of(inside, at.image, mode))
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
#[derive(Default)]
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
    /// Where the captured pointer goes past the image's edge, where the host gives it.
    beyond: Option<Arc<dyn PointerBeyond>>,
    /// Whether it has gone.
    away: Beyond,
    /// What the window does with the host's pointer next, since [`Self::take_hold`] last took it.
    hold: Option<Hold>,
    /// The guest's own mouse, where the host gives it.
    motion: Option<Arc<dyn PointerMotion>>,
    /// Whether the captured pointer drives it.
    game: Game,
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("capture", &self.capture)
            .field("buttons", &self.buttons)
            .field("away", &self.away)
            .field("game", &self.game)
            .finish_non_exhaustive()
    }
}

impl Input {
    /// Nothing held, nothing captured, no keyboard.
    #[cfg(test)]
    pub(crate) fn new() -> Input {
        Input::default()
    }

    /// `Self::new`, its captured pointer going past the image's edge through `beyond`, and
    /// driving the guest's own mouse `motion` while the guest hides its cursor, where the host
    /// gives each.
    pub(crate) fn with_links(
        beyond: Option<Arc<dyn PointerBeyond>>,
        motion: Option<Arc<dyn PointerMotion>>,
    ) -> Input {
        Input {
            beyond,
            motion,
            ..Input::default()
        }
    }

    /// This input with `capture` in place of capture on: off where the host's setting turns it
    /// off ([`Capture::with_setting`]).
    pub(crate) fn with_capture(self, capture: Capture) -> Input {
        Input { capture, ..self }
    }

    /// Whether the captured pointer has gone past the image's edge.
    #[cfg(test)]
    pub(crate) fn is_away(&self) -> bool {
        self.away.is_away()
    }

    /// Whether the host's pointer is locked: gone past the image's edge, or driving the guest's
    /// mouse. SDL reports its moves as motion alone ([`Self::moved_by`]).
    pub(crate) fn is_locked(&self) -> bool {
        self.away.is_away() || self.game.is_on()
    }

    /// Whether the captured pointer drives the guest's own mouse.
    #[cfg(test)]
    pub(crate) fn drives_the_guest_s_mouse(&self) -> bool {
        self.game.is_on()
    }

    /// What the window does with the host's pointer next, taken: lock it as it goes past the
    /// edge, or let it go where it comes back.
    pub(crate) fn take_hold(&mut self) -> Option<Hold> {
        self.hold.take()
    }

    /// The window could not lock the host's pointer: the pointer is the window's again, where it
    /// was held.
    pub(crate) fn lock_refused(&mut self) {
        if let Some(hold) = self.game.refused() {
            self.hold = Some(hold);
            return;
        }
        let beyond = self.beyond.clone();
        if let Some(back) = self.away.let_go(beyond.as_deref()) {
            self.came_back(back);
        }
    }

    /// The pointer came back to the window at `back`: the host's pointer is let go there, and the
    /// guest pixel is the last sent.
    fn came_back(&mut self, back: Back) {
        self.hold = Some(Hold::Unlock(back.point));
        if let Some(ToHost::Pointer { mode, .. }) = self.last {
            self.last = Some(ToHost::Pointer {
                at: back.at,
                mode,
                buttons: self.buttons,
            });
        }
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

    /// What the title ends with: the release chord while captured, in the host's OS's names for
    /// its keys, and that a click captures while not ([`Capture::title_suffix`]).
    pub(crate) fn title_suffix(&self) -> &'static str {
        self.capture.title_suffix(HOST_OS)
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

    /// The pointer moved. Captured, with the guest's cursor hidden, it drives the guest's mouse
    /// ([`crate::game`]); moving toward an edge another display meets and within reach of it, it
    /// goes past it ([`crate::beyond`]). Outside the image with capture off, it is sent only in a
    /// drag ([`Capture::pointer_outside_image`]). A place SDL reports while the pointer is locked
    /// is SDL's own (the lock's warp): the pointer's moves are [`Self::moved_by`]'s.
    pub(crate) fn moved(&mut self, at: &At) {
        if self.is_locked() {
            return;
        }
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            if self.capture.is_captured()
                && let Some(motion) = self.motion.clone()
                && let Some(hold) = self.game.begin(motion.as_ref(), point)
            {
                self.hold = Some(hold);
                return;
            }
            let from = match self.last {
                Some(ToHost::Pointer { at, .. }) => Some(at),
                _ => None,
            };
            if self.capture.is_captured()
                && let Some(beyond) = self.beyond.clone()
                && let Some(hold) =
                    self.away
                        .reached(beyond.as_ref(), point, at.image, mode, self.buttons, from)
            {
                self.hold = Some(hold);
                return;
            }
            if !geometry::contains(at.image, point)
                && !self.capture.pointer_outside_image(self.buttons != 0)
            {
                return;
            }
            self.pointer(point, at.image, mode, false);
        }
    }

    /// The locked pointer moved by `by` (SDL's relative motion): driving the guest's mouse, by
    /// the device's counts, until the guest shows its cursor; gone past the edge, by the window's
    /// points, on, coming back where it lands on the window's display, away from the edge, or
    /// driving the guest's mouse from where it is once the guest hides its cursor.
    pub(crate) fn moved_by(&mut self, by: (f64, f64), at: &At) {
        if let Some(motion) = self.motion.clone() {
            if self.game.is_on() {
                let cursor = cursor_point(at);
                if let Some(hold) = self.game.moved(motion.as_ref(), by, cursor) {
                    self.hold = Some(hold);
                }
                return;
            }
            if self.away.is_away()
                && let Some(back) = self.away.held()
                && let Some(hold) = self.game.begin(motion.as_ref(), back)
            {
                // Away, the guest hid its cursor: the lock stays, its motion now raw.
                let beyond = self.beyond.clone();
                self.away.let_go(beyond.as_deref());
                self.hold = Some(hold);
                return;
            }
        }
        let Some(beyond) = self.beyond.clone() else {
            return;
        };
        if let Some(back) = self.away.moved(beyond.as_ref(), by, at.image, self.buttons) {
            self.came_back(back);
        }
    }

    /// The buttons changed while away: the guest's desktop sends them where the pointer is.
    fn away_buttons(&mut self) {
        if let Some(beyond) = self.beyond.clone() {
            self.away.buttons(beyond.as_ref(), self.buttons);
        }
    }

    /// Lets go of the pointer gone past the edge, where it is: it is the window's again, where it
    /// was held.
    fn let_go_away(&mut self) {
        let beyond = self.beyond.clone();
        if let Some(back) = self.away.let_go(beyond.as_deref()) {
            self.came_back(back);
        }
    }

    /// Button `bit` went down at `timestamp` (SDL's event clock).
    pub(crate) fn button_down(&mut self, bit: u32, timestamp: u64, at: &At) -> Reaction {
        if self.game.press(self.motion.as_deref(), bit) {
            return Reaction::default();
        }
        if self.away.is_away() {
            // Captured: the button goes where the pointer is.
            if bit != 0 {
                self.buttons |= bit;
                self.away_buttons();
            }
            return Reaction::default();
        }
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

    /// Button `bit` went up: sent where the guest was told it is down, wherever the pointer is,
    /// by the route it went down by (the guest's mouse, or the tablet).
    pub(crate) fn button_up(&mut self, bit: u32, at: &At) {
        if self.game.release(self.motion.as_deref(), bit) {
            return;
        }
        if self.buttons & bit == 0 {
            return;
        }
        self.buttons &= !bit;
        if self.away.is_away() {
            self.away_buttons();
            return;
        }
        if self.game.is_on() {
            // The tablet's button, pressed before the guest's mouse was driven: up where the
            // tablet is.
            self.resend_buttons();
            return;
        }
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            self.pointer(point, at.image, mode, true);
        }
    }

    /// The wheel turned (the module doc).
    pub(crate) fn wheel(&mut self, wheel: Wheel, at: &At) {
        let (Some(point), Some(_)) = (at.point, guest_mode(at)) else {
            return;
        };
        if self.game.is_on() {
            let sign = if wheel.flipped { -1 } else { 1 };
            let up = wheel.notches.1.saturating_mul(sign);
            if up != 0 {
                self.game.wheel(self.motion.as_deref(), up);
                return;
            }
        } else if !self.away.is_away() {
            if !geometry::contains(at.image, point) {
                return;
            }
            self.moved(at);
        }
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
        self.let_go_game();
        self.let_go_away();
        Reaction {
            capture_changed: self.capture.resign_key(),
        }
    }

    /// Releases capture, lifting every key held.
    pub(crate) fn release_capture(&mut self) -> Reaction {
        self.lift_keys();
        self.end_game();
        self.let_go_away();
        Reaction {
            capture_changed: self.capture.release(),
        }
    }

    /// The window is closing: every key and button held is lifted.
    pub(crate) fn closing(&mut self) {
        self.lift_keys();
        self.lift_buttons();
        self.let_go_game();
        self.let_go_away();
        self.capture.release();
    }

    /// Ups for every key held.
    fn lift_keys(&mut self) {
        let ups = self.keys.release_all();
        self.out.extend(ups.into_iter().map(ToHost::Key));
    }

    /// The guest showed or hid its cursor: driving the guest's mouse, the pointer is the
    /// window's again once the guest shows it, the host's pointer let go over it.
    pub(crate) fn cursor_changed(&mut self, at: &At) {
        if let Some(motion) = self.motion.clone()
            && let Some(hold) = self.game.shown(motion.as_ref(), cursor_point(at))
        {
            self.hold = Some(hold);
        }
    }

    /// Lifts the buttons held through the guest's mouse, and lets go of the pointer driving it
    /// where it was held.
    fn let_go_game(&mut self) {
        self.game.lift(self.motion.as_deref());
        self.end_game();
    }

    /// The pointer driving the guest's mouse is the window's again, where it was held; a refused
    /// lock may be tried again.
    fn end_game(&mut self) {
        if let Some(hold) = self.game.end() {
            self.hold = Some(hold);
        }
    }

    /// The tablet's buttons changed while the pointer drives the guest's mouse: the last
    /// pointer's pixel with them.
    fn resend_buttons(&mut self) {
        if let Some(ToHost::Pointer { at, mode, .. }) = self.last {
            let event = ToHost::Pointer {
                at,
                mode,
                buttons: self.buttons,
            };
            self.last = Some(event);
            self.out.push(event);
        }
    }

    /// A pointer event with no button, where one is held, at the last pointer's pixel.
    fn lift_buttons(&mut self) {
        if self.buttons == 0 {
            return;
        }
        self.buttons = 0;
        if self.away.is_away() {
            self.away_buttons();
            return;
        }
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
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::*;
    use crate::capture::{OFF_SUFFIX, UNCAPTURED_SUFFIX, captured_suffix};
    use crate::game::SETTLE_NS;
    use crate::game::tests::Mouse;

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
            cursor: None,
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

    /// Ctrl+Alt+Super (Ctrl+Alt+Win on Windows: SDL's GUI key) while captured lifts every key and
    /// releases capture, and the title names it in the host's OS's names while captured
    #[test]
    fn ctrl_alt_super_releases_capture() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((10.0, 10.0)));
        assert!(input.is_captured());
        assert_eq!(input.title_suffix(), captured_suffix(HOST_OS));
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

    /// With capture off, a click in the image is sent but captures nothing, so nothing holds the
    /// pointer; the pointer outside the image is sent only in a drag; Ctrl, Alt and the GUI key
    /// are the guest's keys and release nothing; the title says capture is off
    #[test]
    fn capture_off_captures_nothing_and_sends_the_pointer_over_the_image_alone() {
        let mut input = focused().with_capture(Capture::with_setting(false));
        assert_eq!(input.title_suffix(), OFF_SUFFIX);
        input.moved(&at((10.0, 10.0)));
        input.moved(&at((120.0, 10.0)));
        assert_eq!(
            input.take(),
            vec![pointer(20, 20, 0)],
            "none outside the image"
        );
        let r = input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        assert!(!r.capture_changed && !input.is_captured());
        assert_eq!(input.pointer_confinement(at((0.0, 0.0)).image), None);
        // A drag goes on outside the image, held at its edge, and ends there.
        input.moved(&at((120.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((120.0, 10.0)));
        input.moved(&at((130.0, 10.0)));
        assert_eq!(
            input.take(),
            vec![
                pointer(20, 20, BUTTON_LEFT),
                pointer(199, 20, BUTTON_LEFT),
                pointer(199, 20, 0),
            ]
        );
        input.key(0xE0, true, false);
        input.key(0xE2, true, false);
        let r = input.key(0xE3, true, false);
        assert!(!r.capture_changed && !input.is_captured());
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0xE0)),
                ToHost::Key(KeyEvent::down(0xE2)),
                ToHost::Key(KeyEvent::down(0xE3)),
            ]
        );
        assert_eq!(input.title_suffix(), OFF_SUFFIX);
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

    /// A close the person cancels at the question leaves nothing held and capture released, and
    /// the input goes on: a key reaches the guest, and a click in the image captures again
    #[test]
    fn a_cancelled_close_leaves_nothing_held_and_the_input_goes_on() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.key(0x04, true, false);
        input.take();
        input.closing();
        assert!(!input.is_captured());
        assert_eq!(
            input.take(),
            vec![ToHost::Key(KeyEvent::up(0x04)), pointer(20, 20, 0)]
        );
        input.key(0x05, true, false);
        input.key(0x05, false, false);
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0x05)),
                ToHost::Key(KeyEvent::up(0x05))
            ]
        );
        let r = input.button_down(BUTTON_LEFT, 2_000_000_000, &at((50.0, 25.0)));
        assert!(r.capture_changed && input.is_captured());
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

    /// A guest desktop of display 0 (the window's, 200 x 100) and display 1 right of it, in the
    /// same mode, recording what it is told.
    #[derive(Default)]
    struct RightOfIt {
        /// The pointer's column across both, and its row, while away.
        at: Mutex<Option<(f64, f64)>>,
        /// What it was told.
        told: Mutex<Vec<String>>,
    }

    impl RightOfIt {
        fn told(&self) -> Vec<String> {
            std::mem::take(&mut *self.told.lock())
        }
    }

    impl PointerBeyond for RightOfIt {
        fn adjoins(&self, _at: (u16, u16), _mode: (u16, u16), outward: (i8, i8)) -> bool {
            outward == (1, 0)
        }

        fn leave(&self, at: (u16, u16), _mode: (u16, u16)) -> bool {
            *self.at.lock() = Some((f64::from(at.0) + 0.5, f64::from(at.1) + 0.5));
            true
        }

        fn moved(&self, by: (f64, f64), buttons: u32) -> crate::config::Landed {
            let mut at = self.at.lock();
            let Some(point) = at.as_mut() else {
                return crate::config::Landed::Lost;
            };
            *point = (point.0 + by.0, point.1 + by.1);
            let (x, y) = (point.0.floor() as u16, point.1.floor() as u16);
            if x < 200 {
                self.told.lock().push(format!("0 {x},{y} {buttons}"));
                crate::config::Landed::Here {
                    at: (x, y),
                    mode: (200, 100),
                }
            } else {
                self.told
                    .lock()
                    .push(format!("1 {},{y} {buttons}", x - 200));
                crate::config::Landed::Elsewhere
            }
        }

        fn come_back(&self) {
            self.told.lock().push("back".into());
        }
    }

    /// A captured input with `desktop` beyond its image's edges.
    fn captured_beside(desktop: &Arc<RightOfIt>) -> Input {
        let beyond: Arc<dyn PointerBeyond> = desktop.clone();
        let mut input = Input::with_links(Some(beyond), None);
        input.focus_gained(0);
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((10.0, 10.0)));
        input.take();
        input
    }

    /// Captured, the pointer moving toward the image's edge where the guest's desktop puts another
    /// display, within reach of it, goes past it, the host's pointer locked: its motion, a drag's
    /// button and its release go to the desktop, keys and the wheel to the host as ever, and back
    /// over the window's display, out of reach of the edge, it is the window's again, let go
    /// where it is
    #[test]
    fn the_captured_pointer_at_a_shared_edge_goes_on_past_it() {
        let desktop = Arc::new(RightOfIt::default());
        let mut input = captured_beside(&desktop);
        input.moved(&at((60.0, 25.0)));
        assert_eq!(input.take_hold(), None, "out of reach of the edge");
        input.button_down(BUTTON_LEFT, 2_000_000_000, &at((60.0, 25.0)));
        input.moved(&at((75.0, 25.0)));
        assert_eq!(input.take_hold(), Some(Hold::Lock));
        assert!(input.is_away());
        assert_eq!(
            input.take(),
            vec![pointer(120, 50, 0), pointer(120, 50, BUTTON_LEFT)]
        );
        assert_eq!(desktop.told(), ["0 150,50 1"]);
        // SDL's motion, in points: two pixels each. A place SDL reports itself is ignored.
        input.moved_by((30.0, 1.0), &at((75.0, 25.0)));
        input.moved(&at((50.0, 25.0)));
        input.key(0x04, true, false);
        input.wheel(
            Wheel {
                notches: (0, 1),
                lines: (0.0, 1.0),
                flipped: false,
            },
            &at((99.9, 25.0)),
        );
        input.button_up(BUTTON_LEFT, &at((99.9, 25.0)));
        assert_eq!(desktop.told(), ["1 10,52 1", "1 10,52 0"]);
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0x04)),
                ToHost::Wheel {
                    notches: (1, 0),
                    smooth: (2560, 0)
                },
            ]
        );
        assert_eq!(input.take_hold(), None);
        // Back across the edge, then in: the host's pointer goes where the guest's cursor is.
        input.moved_by((-10.0, 0.0), &at((75.0, 25.0)));
        assert_eq!(input.take_hold(), None, "within reach of the edge");
        input.moved_by((-30.0, 0.0), &at((75.0, 25.0)));
        assert_eq!(desktop.told(), ["0 190,52 0", "0 130,52 0", "back"]);
        assert_eq!(input.take_hold(), Some(Hold::Unlock((65.25, 26.25))));
        assert!(!input.is_away());
        input.moved(&at((65.25, 26.25)));
        assert_eq!(input.take(), vec![], "the pixel came back sent");
        input.moved(&at((60.0, 26.25)));
        assert_eq!(input.take(), vec![pointer(120, 52, 0)]);
    }

    /// Releasing capture, losing the keyboard or closing while away gives the window the host's
    /// pointer where it was held, a button held lifted where the pointer is; without a desktop,
    /// or uncaptured, the pointer holds at the image's edge as ever
    #[test]
    fn letting_go_while_away_gives_the_pointer_back_where_it_was_held() {
        let desktop = Arc::new(RightOfIt::default());
        let mut input = captured_beside(&desktop);
        input.moved(&at((99.9, 10.0)));
        assert_eq!(input.take_hold(), Some(Hold::Lock));
        input.moved_by((10.0, 0.0), &at((99.9, 10.0)));
        input.key(0xE0, true, false);
        input.key(0xE2, true, false);
        input.key(0xE3, true, false);
        assert!(!input.is_captured() && !input.is_away());
        assert_eq!(input.take_hold(), Some(Hold::Unlock((99.9, 10.0))));
        desktop.told();

        let mut input = captured_beside(&desktop);
        input.button_down(BUTTON_LEFT, 2_000_000_000, &at((90.0, 10.0)));
        input.moved(&at((99.9, 10.0)));
        assert_eq!(input.take_hold(), Some(Hold::Lock));
        input.moved_by((10.0, 0.0), &at((99.9, 10.0)));
        desktop.told();
        input.focus_lost();
        assert_eq!(desktop.told(), ["1 19,20 0", "back"]);
        assert_eq!(input.take_hold(), Some(Hold::Unlock((99.9, 10.0))));

        let mut input = captured_beside(&desktop);
        input.moved(&at((99.9, 10.0)));
        input.closing();
        assert_eq!(input.take_hold(), Some(Hold::Unlock((99.9, 10.0))));

        let mut alone = focused();
        alone.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        alone.moved(&at((99.9, 10.0)));
        assert_eq!(alone.take_hold(), None);
        let mut uncaptured =
            Input::with_links(Some(desktop.clone() as Arc<dyn PointerBeyond>), None);
        uncaptured.moved(&at((99.9, 10.0)));
        assert_eq!(uncaptured.take_hold(), None);
        assert_eq!(uncaptured.take(), vec![pointer(199, 20, 0)]);
    }

    /// An input captured by a click, driving the guest's mouse `motion` where it is given.
    fn captured_by(motion: Option<Arc<Mouse>>) -> Input {
        let mut input = Input::with_links(None, motion.map(|m| m as Arc<dyn PointerMotion>));
        input.focus_gained(0);
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.button_up(BUTTON_LEFT, &at((10.0, 10.0)));
        input.take();
        input
    }

    /// Moves, a drag past the image's edge and the wheel, as the tablet hears them.
    fn desk_work(input: &mut Input) -> Vec<ToHost> {
        input.moved(&at((20.0, 10.0)));
        input.moved(&at((40.0, 20.0)));
        input.button_down(BUTTON_RIGHT, 2_000_000_000, &at((40.0, 20.0)));
        input.moved(&at((150.0, 10.0)));
        input.button_up(BUTTON_RIGHT, &at((150.0, 10.0)));
        input.moved(&at((60.0, 30.0)));
        input.wheel(
            Wheel {
                notches: (0, 1),
                lines: (0.0, 1.0),
                flipped: false,
            },
            &at((60.0, 30.0)),
        );
        input.take()
    }

    /// Outside a game the tablet hears exactly what it hears with no guest's mouse, and the
    /// guest's mouse nothing: captured while the guest shows its cursor, or has hidden it for
    /// less than the settle, and uncaptured however long it hides it
    #[test]
    fn the_tablet_s_events_are_unchanged_outside_a_game() {
        let plain = desk_work(&mut captured_by(None));
        assert!(!plain.is_empty());
        for hidden in [None, Some(SETTLE_NS - 1)] {
            let mouse = Arc::new(Mouse::default());
            *mouse.hidden.lock() = hidden;
            let mut input = captured_by(Some(Arc::clone(&mouse)));
            assert_eq!(desk_work(&mut input), plain, "hidden for {hidden:?}");
            assert_eq!(input.take_hold(), None);
            assert_eq!(mouse.take(), vec![]);
        }
        // Uncaptured: moves alone, since a click in the image captures.
        let roam = |input: &mut Input| {
            input.moved(&at((20.0, 10.0)));
            input.moved(&at((150.0, 10.0)));
            input.moved(&at((60.0, 30.0)));
            input.take()
        };
        let uncaptured = roam(&mut focused());
        let mouse = Arc::new(Mouse::default());
        mouse.hide_for(SETTLE_NS);
        let mut input = Input::with_links(None, Some(Arc::clone(&mouse) as Arc<dyn PointerMotion>));
        input.focus_gained(0);
        assert_eq!(roam(&mut input), uncaptured);
        assert_eq!(input.take_hold(), None);
        assert_eq!(mouse.take(), vec![]);
    }

    /// Captured, once the guest has hidden its cursor for the settle, the host's pointer is
    /// locked where it is, and its motion, by whole counts, and the wheel's notches drive the
    /// guest's mouse, however far: the tablet hears nothing. Once the guest shows its cursor the
    /// pointer is the window's again, let go over the guest's cursor on the window's display, else
    /// where it was held
    #[test]
    fn a_hidden_cursor_s_motion_drives_the_guest_s_mouse_until_it_shows() {
        let mouse = Arc::new(Mouse::default());
        let mut input = captured_by(Some(Arc::clone(&mouse)));
        mouse.hide_for(SETTLE_NS);
        input.moved(&at((99.0, 10.0)));
        assert_eq!(input.take_hold(), Some(Hold::Raw));
        assert!(input.is_locked() && input.drives_the_guest_s_mouse());
        input.moved_by((500.0, 0.0), &at((99.0, 10.0)));
        input.moved_by((300.5, -2.0), &at((99.0, 10.0)));
        input.moved_by((0.5, 0.0), &at((99.0, 10.0)));
        input.moved(&at((10.0, 10.0)));
        input.wheel(
            Wheel {
                notches: (0, 2),
                lines: (0.0, 2.0),
                flipped: true,
            },
            &at((99.0, 10.0)),
        );
        assert_eq!(
            mouse.take(),
            vec![
                ((500, 0), 0, 0),
                ((300, -2), 0, 0),
                ((1, 0), 0, 0),
                ((0, 0), -2, 0)
            ]
        );
        assert_eq!(input.take(), vec![], "the tablet hears nothing");
        // The guest shows its cursor at pixel (50, 40) of the window's display.
        mouse.show();
        input.cursor_changed(&At {
            cursor: Some((50, 40)),
            ..at((99.0, 10.0))
        });
        assert_eq!(input.take_hold(), Some(Hold::Unlock((25.25, 20.25))));
        assert!(!input.is_locked());
        input.moved(&at((25.25, 20.25)));
        input.moved(&at((30.0, 20.25)));
        assert_eq!(input.take(), vec![pointer(50, 40, 0), pointer(60, 40, 0)]);
        // Hidden again, then shown on another display, seen at the next motion.
        mouse.hide_for(SETTLE_NS);
        input.moved(&at((30.0, 20.0)));
        assert_eq!(input.take_hold(), Some(Hold::Raw));
        mouse.show();
        input.moved_by((3.0, 0.0), &at((30.0, 20.0)));
        assert_eq!(mouse.take(), vec![((3, 0), 0, 0)]);
        assert_eq!(input.take_hold(), Some(Hold::Unlock((30.0, 20.0))));
        assert!(!input.drives_the_guest_s_mouse());
    }

    /// A button comes up by the route it went down by, held across either switch: one pressed on
    /// the tablet before the guest's mouse is driven comes up on the tablet where it last was; one
    /// pressed through the guest's mouse comes up through it after the guest shows its cursor
    #[test]
    fn a_button_held_across_the_switch_comes_up_by_its_own_route() {
        let mouse = Arc::new(Mouse::default());
        let mut input = captured_by(Some(Arc::clone(&mouse)));
        input.button_down(BUTTON_RIGHT, 2_000_000_000, &at((20.0, 10.0)));
        assert_eq!(input.take(), vec![pointer(40, 20, BUTTON_RIGHT)]);
        mouse.hide_for(SETTLE_NS);
        input.moved(&at((21.0, 10.0)));
        assert_eq!(input.take_hold(), Some(Hold::Raw));
        input.moved_by((4.0, 0.0), &at((21.0, 10.0)));
        input.button_down(BUTTON_LEFT, 3_000_000_000, &at((21.0, 10.0)));
        input.moved_by((1.0, 1.0), &at((21.0, 10.0)));
        input.button_up(BUTTON_RIGHT, &at((21.0, 10.0)));
        assert_eq!(
            mouse.take(),
            vec![
                ((4, 0), 0, 0),
                ((0, 0), 0, BUTTON_LEFT),
                ((1, 1), 0, BUTTON_LEFT)
            ]
        );
        assert_eq!(
            input.take(),
            vec![pointer(40, 20, 0)],
            "the tablet's comes up there"
        );
        mouse.show();
        input.cursor_changed(&at((21.0, 10.0)));
        assert_eq!(input.take_hold(), Some(Hold::Unlock((21.0, 10.0))));
        input.button_up(BUTTON_LEFT, &at((21.0, 10.0)));
        assert_eq!(
            mouse.take(),
            vec![((0, 0), 0, 0)],
            "the mouse's comes up there"
        );
        assert_eq!(input.take(), vec![]);
        input.button_down(BUTTON_LEFT, 4_000_000_000, &at((21.0, 10.0)));
        assert_eq!(input.take(), vec![pointer(42, 20, BUTTON_LEFT)]);
        assert_eq!(mouse.take(), vec![]);
    }

    /// Releasing capture (the chord), losing the keyboard and closing give the window back the
    /// pointer driving the guest's mouse where it was held, a button held through the mouse lifted
    /// on the keyboard's loss and the close; a lock the OS refuses gives it back too, and is not
    /// tried again until capture changes
    #[test]
    fn letting_go_of_a_game_gives_the_pointer_back_where_it_was_held() {
        let mouse = Arc::new(Mouse::default());
        mouse.hide_for(SETTLE_NS);
        let in_game = || {
            let mut input = captured_by(Some(Arc::clone(&mouse)));
            input.moved(&at((30.0, 10.0)));
            assert_eq!(input.take_hold(), Some(Hold::Raw));
            input
        };
        let mut input = in_game();
        for usage in [0xE0, 0xE2, 0xE3] {
            input.key(usage, true, false);
        }
        assert!(!input.is_captured() && !input.is_locked());
        assert_eq!(input.take_hold(), Some(Hold::Unlock((30.0, 10.0))));

        let mut input = in_game();
        input.button_down(BUTTON_LEFT, 2_000_000_000, &at((30.0, 10.0)));
        mouse.take();
        input.focus_lost();
        assert_eq!(mouse.take(), vec![((0, 0), 0, 0)]);
        assert!(!input.is_locked());
        assert_eq!(input.take_hold(), Some(Hold::Unlock((30.0, 10.0))));

        let mut input = in_game();
        input.button_down(BUTTON_MIDDLE, 2_000_000_000, &at((30.0, 10.0)));
        mouse.take();
        input.closing();
        assert_eq!(mouse.take(), vec![((0, 0), 0, 0)]);
        assert_eq!(input.take_hold(), Some(Hold::Unlock((30.0, 10.0))));

        let mut input = in_game();
        input.lock_refused();
        assert_eq!(input.take_hold(), Some(Hold::Unlock((30.0, 10.0))));
        assert!(!input.is_locked());
        input.moved(&at((31.0, 10.0)));
        assert_eq!(input.take_hold(), None, "not tried again while captured");
        assert_eq!(input.take(), vec![pointer(62, 20, 0)]);
        input.release_capture();
        input.button_down(BUTTON_LEFT, 5_000_000_000, &at((31.0, 10.0)));
        input.moved(&at((32.0, 10.0)));
        assert_eq!(
            input.take_hold(),
            Some(Hold::Raw),
            "tried again at the next capture"
        );
        assert_eq!(mouse.take(), vec![]);
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
