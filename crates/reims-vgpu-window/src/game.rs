//! A pointer for games: what a window decides of its captured pointer while the guest hides its
//! cursor, with no OS in it (the SDL window's input, `native/sdl_thread/input.rs`, drives it on
//! Linux and Windows, and the macOS window's input on macOS).
//!
//! The guest's pointer is the host's absolute one (QEMU's USB tablet), held at the image's edge
//! outside it, and captured, the host's pointer is held over the image (`capture`). A game that
//! hides the cursor and turns its camera by the mouse's motion (a disassociated cursor's deltas,
//! GCMouse) then gets no motion past the edge: the tablet stays where it is. Where the guest has
//! its own mouse ([`PointerMotion`], the reims-graphics device's `POINTER_MOTION`), the window,
//! captured, with the guest's cursor hidden on every display for [`SETTLE_NS`], holds the host's
//! pointer still where it is ([`Hold::Raw`]: the OS's pointer lock, which hides it and reports
//! its moves as the device's own counts) and hands each move to the guest's mouse
//! ([`Game::moved`]), which the guest accelerates once, as its own settings say: the camera turns
//! as far as the mouse moves. Once the guest shows its cursor again ([`Game::shown`]), the pointer
//! is the window's again, the host's pointer let go over the guest's cursor where it shows on the
//! window's display, else where it was held; letting go of capture (the release chord, the
//! keyboard lost, the window's end) lets it go where it was held ([`Game::end`]). On the desktop,
//! where the guest shows its cursor, the pointer stays the absolute one, exact.
//!
//! A button goes up by the route it went down by: one pressed while the guest's mouse is driven
//! goes down and up through it, one pressed before through the tablet, whenever each comes up
//! ([`Game::release`]), so a button held across either switch is never left down. The wheel's
//! notches go to the guest's mouse while it is driven ([`Game::wheel`]).

use crate::beyond::Hold;
use crate::config::PointerMotion;

/// How long the guest's cursor stays hidden on every display it sees before the captured
/// pointer drives the guest's mouse: past a hide and a show in a row. WindowServer hides and
/// shows the cursor a dozen times as it starts (lane gamemouse, potluck job m3-max-35167), and the
/// cursor going from one display to the next is hidden on the one before it shows on the other.
pub const SETTLE_NS: u64 = 50_000_000;

/// The captured pointer driving the guest's mouse, or not (the module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Game {
    /// Where the host's pointer was held as it began, in the window's points, while it drives the
    /// guest's mouse.
    held: Option<(f64, f64)>,
    /// The `BUTTON_*` buttons that went down through the guest's mouse and have not come up.
    buttons: u32,
    /// The motion's parts below a count, not sent yet.
    rest: (f64, f64),
    /// The OS refused to lock the host's pointer: no new start until capture changes.
    refused: bool,
}

impl Game {
    /// Not driving the guest's mouse.
    pub fn new() -> Game {
        Game::default()
    }

    /// Whether the captured pointer drives the guest's mouse: the window reports its moves as
    /// motion alone ([`Self::moved`]).
    pub fn is_on(&self) -> bool {
        self.held.is_some()
    }

    /// The buttons held through the guest's mouse.
    pub fn buttons(&self) -> u32 {
        self.buttons
    }

    /// Whether the captured pointer at `point`, in the window's points, begins to drive the
    /// guest's mouse `motion`: the guest has hidden its cursor on every display for
    /// [`SETTLE_NS`], and the OS did not refuse the lock since capture last changed. Where it
    /// does, the host's pointer is held there ([`Hold::Raw`]).
    pub fn begin(&mut self, motion: &dyn PointerMotion, point: (f64, f64)) -> Option<Hold> {
        if self.held.is_some() || self.refused {
            return None;
        }
        if !motion.hidden_for().is_some_and(|ns| ns >= SETTLE_NS) {
            return None;
        }
        self.held = Some(point);
        self.rest = (0.0, 0.0);
        Some(Hold::Raw)
    }

    /// The host's pointer moved by `by`, its device's counts: the guest's mouse moves by the whole
    /// counts, the parts below a count kept for the next. Where the guest shows its cursor again,
    /// the pointer is the window's again ([`Self::shown`]). `None` while it drives the mouse, or
    /// did not.
    pub fn moved(
        &mut self,
        motion: &dyn PointerMotion,
        by: (f64, f64),
        cursor: Option<(f64, f64)>,
    ) -> Option<Hold> {
        self.held?;
        let whole = |rest: &mut f64, by: f64| -> i32 {
            let total = *rest + if by.is_finite() { by } else { 0.0 };
            let counts = total.trunc();
            *rest = total - counts;
            counts.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
        };
        let counts = (whole(&mut self.rest.0, by.0), whole(&mut self.rest.1, by.1));
        if counts != (0, 0) {
            motion.moved(counts, 0, self.buttons);
        }
        self.shown(motion, cursor)
    }

    /// Whether the guest shows its cursor again: the pointer is then the window's again, the
    /// host's pointer let go at `cursor`, the window's point over the guest's cursor where it
    /// shows on the window's display, else where it was held. `None` while the guest hides it,
    /// or where the pointer did not drive the mouse.
    pub fn shown(
        &mut self,
        motion: &dyn PointerMotion,
        cursor: Option<(f64, f64)>,
    ) -> Option<Hold> {
        let held = self.held?;
        if motion.hidden_for().is_some() {
            return None;
        }
        self.held = None;
        Some(Hold::Unlock(cursor.unwrap_or(held)))
    }

    /// Button `bit` went down: where the pointer drives the guest's mouse, it goes down there.
    /// Whether it did; else it is the tablet's.
    pub fn press(&mut self, motion: Option<&dyn PointerMotion>, bit: u32) -> bool {
        let Some(motion) = motion.filter(|_| self.held.is_some()) else {
            return false;
        };
        if bit != 0 && self.buttons & bit == 0 {
            self.buttons |= bit;
            motion.moved((0, 0), 0, self.buttons);
        }
        true
    }

    /// Button `bit` went up: where it went down through the guest's mouse, it comes up there,
    /// whether or not the pointer still drives it. Whether it did; else it is the tablet's.
    pub fn release(&mut self, motion: Option<&dyn PointerMotion>, bit: u32) -> bool {
        if bit == 0 || self.buttons & bit == 0 {
            return false;
        }
        self.buttons &= !bit;
        if let Some(motion) = motion {
            motion.moved((0, 0), 0, self.buttons);
        }
        true
    }

    /// The wheel turned `notches` up: where the pointer drives the guest's mouse, they go there.
    /// Whether they did.
    pub fn wheel(&mut self, motion: Option<&dyn PointerMotion>, notches: i32) -> bool {
        let Some(motion) = motion.filter(|_| self.held.is_some()) else {
            return false;
        };
        if notches != 0 {
            motion.moved((0, 0), notches, self.buttons);
        }
        true
    }

    /// Every button held through the guest's mouse comes up (the keyboard lost, the window's
    /// end).
    pub fn lift(&mut self, motion: Option<&dyn PointerMotion>) {
        if self.buttons == 0 {
            return;
        }
        self.buttons = 0;
        if let Some(motion) = motion {
            motion.moved((0, 0), 0, 0);
        }
    }

    /// Capture changed (released, the keyboard lost, the window's end): the pointer is the
    /// window's again, the host's pointer let go where it was held, and a refused lock may be
    /// tried again at the next capture. `None` where it did not drive the mouse.
    pub fn end(&mut self) -> Option<Hold> {
        self.refused = false;
        let held = self.held.take()?;
        Some(Hold::Unlock(held))
    }

    /// The OS refused to lock the host's pointer: the pointer is the window's again, where it
    /// was held, and it does not try again until capture changes.
    pub fn refused(&mut self) -> Option<Hold> {
        let held = self.held.take()?;
        self.refused = true;
        Some(Hold::Unlock(held))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::*;

    /// What a guest's mouse was sent, in order: each motion, wheel and buttons.
    pub(crate) type Sent = Vec<((i32, i32), i32, u32)>;

    /// A guest's mouse: how long its cursor is hidden, and what it was sent.
    #[derive(Debug, Default)]
    pub(crate) struct Mouse {
        pub(crate) hidden: Mutex<Option<u64>>,
        pub(crate) sent: Mutex<Sent>,
    }

    impl PointerMotion for Mouse {
        fn hidden_for(&self) -> Option<u64> {
            *self.hidden.lock()
        }
        fn moved(&self, by: (i32, i32), wheel: i32, buttons: u32) -> bool {
            self.sent.lock().push((by, wheel, buttons));
            true
        }
    }

    impl Mouse {
        pub(crate) fn hide_for(&self, ns: u64) {
            *self.hidden.lock() = Some(ns);
        }
        pub(crate) fn show(&self) {
            *self.hidden.lock() = None;
        }
        pub(crate) fn take(&self) -> Sent {
            std::mem::take(&mut *self.sent.lock())
        }
    }

    /// The captured pointer drives the guest's mouse only once the guest has hidden its cursor
    /// for the settle: not while it shows it, not over a hide and a show in a row
    #[test]
    fn the_guest_s_mouse_is_driven_once_its_cursor_stays_hidden() {
        let mouse = Mouse::default();
        let mut game = Game::new();
        assert_eq!(game.begin(&mouse, (5.0, 5.0)), None, "shown");
        mouse.hide_for(SETTLE_NS - 1);
        assert_eq!(
            game.begin(&mouse, (5.0, 5.0)),
            None,
            "hidden for less than the settle"
        );
        mouse.show();
        assert_eq!(
            game.begin(&mouse, (5.0, 5.0)),
            None,
            "shown again: the count starts over"
        );
        mouse.hide_for(SETTLE_NS);
        assert_eq!(game.begin(&mouse, (5.0, 5.0)), Some(Hold::Raw));
        assert!(game.is_on());
        assert_eq!(game.begin(&mouse, (6.0, 6.0)), None, "begun once");
    }

    /// Each move goes to the guest's mouse in whole counts, the parts below one carried on, and
    /// the guest's show of its cursor gives the pointer back over its cursor, or where it was held
    #[test]
    fn moves_go_to_the_guest_s_mouse_until_it_shows_its_cursor() {
        let mouse = Mouse::default();
        let mut game = Game::new();
        mouse.hide_for(SETTLE_NS);
        game.begin(&mouse, (5.0, 6.0));
        assert_eq!(game.moved(&mouse, (2.5, -1.25), Some((9.0, 9.0))), None);
        assert_eq!(game.moved(&mouse, (0.75, -0.75), None), None);
        assert_eq!(game.moved(&mouse, (0.25, 0.0), None), None);
        assert_eq!(mouse.take(), vec![((2, -1), 0, 0), ((1, -1), 0, 0)]);
        mouse.show();
        assert_eq!(
            game.moved(&mouse, (1.0, 0.0), Some((40.0, 30.0))),
            Some(Hold::Unlock((40.0, 30.0))),
            "let go over the guest's cursor"
        );
        assert!(!game.is_on());
        assert_eq!(
            mouse.take(),
            vec![((1, 0), 0, 0)],
            "the move before the show goes"
        );
        assert_eq!(game.moved(&mouse, (1.0, 0.0), None), None, "not driven now");
        assert_eq!(mouse.take(), vec![]);
        // Shown elsewhere: let go where it was held.
        mouse.hide_for(SETTLE_NS);
        game.begin(&mouse, (7.0, 8.0));
        mouse.show();
        assert_eq!(game.shown(&mouse, None), Some(Hold::Unlock((7.0, 8.0))));
    }

    /// A button goes up by the route it went down by, whichever way the pointer switched while it
    /// was held: one pressed through the tablet comes up there while the guest's mouse is driven,
    /// and one pressed through the guest's mouse comes up there once the pointer is the window's
    /// again
    #[test]
    fn a_button_comes_up_by_the_route_it_went_down_by() {
        let mouse = Mouse::default();
        let mut game = Game::new();
        // Pressed through the tablet, before: not the mouse's, then or after the switch.
        assert!(!game.press(Some(&mouse), 2));
        mouse.hide_for(SETTLE_NS);
        game.begin(&mouse, (0.0, 0.0));
        assert!(!game.release(Some(&mouse), 2), "the tablet's to lift");
        // Pressed through the mouse, released after the switch back: the mouse's.
        assert!(game.press(Some(&mouse), 1));
        mouse.show();
        assert!(game.shown(&mouse, None).is_some());
        assert!(!game.press(Some(&mouse), 4), "pressed now: the tablet's");
        assert!(game.release(Some(&mouse), 1), "the mouse's to lift");
        assert!(!game.release(Some(&mouse), 1), "once");
        assert_eq!(mouse.take(), vec![((0, 0), 0, 1), ((0, 0), 0, 0)]);
    }

    /// The wheel goes to the guest's mouse while it is driven, with the buttons held there
    #[test]
    fn the_wheel_goes_to_the_driven_mouse() {
        let mouse = Mouse::default();
        let mut game = Game::new();
        assert!(!game.wheel(Some(&mouse), 1), "not driven");
        mouse.hide_for(SETTLE_NS);
        game.begin(&mouse, (0.0, 0.0));
        game.press(Some(&mouse), 1);
        assert!(game.wheel(Some(&mouse), -2));
        assert_eq!(mouse.take(), vec![((0, 0), 0, 1), ((0, 0), -2, 1)]);
    }

    /// Capture's end lets the host's pointer go where it was held, and a refused lock is tried
    /// again only once capture changed; the keyboard lost lifts the mouse's buttons
    #[test]
    fn capture_s_end_and_a_refused_lock_give_the_pointer_back_where_held() {
        let mouse = Mouse::default();
        let mut game = Game::new();
        mouse.hide_for(SETTLE_NS);
        game.begin(&mouse, (3.0, 4.0));
        game.press(Some(&mouse), 1);
        game.lift(Some(&mouse));
        assert_eq!(game.end(), Some(Hold::Unlock((3.0, 4.0))));
        assert_eq!(game.end(), None);
        assert_eq!(mouse.take(), vec![((0, 0), 0, 1), ((0, 0), 0, 0)]);
        game.begin(&mouse, (1.0, 1.0));
        assert_eq!(game.refused(), Some(Hold::Unlock((1.0, 1.0))));
        assert_eq!(
            game.begin(&mouse, (1.0, 1.0)),
            None,
            "refused until capture changes"
        );
        assert_eq!(game.end(), None);
        assert_eq!(game.begin(&mouse, (1.0, 1.0)), Some(Hold::Raw));
    }
}
