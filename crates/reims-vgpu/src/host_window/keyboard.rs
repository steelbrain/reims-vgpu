//! Keyboard ownership for the host-owned window ([[host-window]]): which keys
//! the guest currently believes are held, and whether the host window is
//! capturing the compositor's own shortcuts.
//!
//! Two invariants live here, both of which used to be nobody's job.
//!
//! # Every key that goes down comes back up
//!
//! The guest keyboard is a level-triggered device: a key-down with no matching
//! key-up leaves the guest holding that key for the rest of the boot. The window
//! system does not guarantee the up. Wayland's `wl_keyboard.leave` says only
//! "the client must treat all keys as released" — it delivers a focus-loss
//! notification, not a release per key — and the windowing library's Wayland
//! backend passes that through unchanged, where its X11 backend synthesises the
//! individual releases. So on Wayland a key held at the moment focus is lost
//! never comes up.
//!
//! That is not a rare corner. It is the *normal* outcome of every chord the
//! compositor claims: the operator presses Alt, the compositor takes Alt+Tab and
//! the focus with it, and the guest is left holding Alt forever. Every later
//! keystroke is then read by the guest as Alt-modified, which presents as
//! "the window does not forward modifiers" even for the keys that do arrive.
//!
//! [`HeldKeys`] is the only thing in this crate that can emit a key-down, so
//! "every down has a matching up" is a property of the type rather than an
//! obligation re-assembled at each call site. Release is total: releasing a key
//! that is not held emits nothing, because there is no down for it to close.
//!
//! # The compositor must be asked to stop eating shortcuts
//!
//! A desktop compositor claims chords before any client sees them. On the host
//! this was recovered from, the session registered 63 `Meta`/`Alt`/`Ctrl` chords
//! as global shortcuts, including bare `Meta`, `Meta+A`, `Meta+V`, `Meta+Q`,
//! `Meta+W` and `Alt+Tab`. A macOS guest reads host `Meta` as `Cmd`, so that set
//! is very close to "every shortcut the guest has". None of them reach
//! [`super::input_map`] to be mapped; the mapping table was never the defect.
//!
//! Asking the compositor to stop is a per-platform request ([`super::capture`]).
//! *When* to ask is not: it is the focus-and-latch machine in [`GrabState`],
//! which is pure and lives here.
//!
//! # The escape hatch is not optional
//!
//! Capture that cannot be released strands the operator: with the compositor's
//! shortcuts inhibited there is no Alt+Tab to leave with. [`UNGRAB_CHORD`] is
//! the way out. It is consumed rather than forwarded, and it releases every held
//! key on its way out so the guest is not left holding the Ctrl and Alt that
//! produced it.
//!
//! The chord requires a Ctrl *and* an Alt, which is what keeps it clear of the
//! one guest shortcut it could plausibly have shadowed. A macOS guest's Force
//! Quit is Cmd+Option+Esc; the host presses that as Meta+Alt+Esc, which carries
//! no Ctrl and is therefore forwarded whole. `force_quit_chord_reaches_the_guest`
//! holds that apart — an escape hatch that ate the guest's own escape hatch
//! would be a poor trade.

use crate::runtime::host::HostAction;

/// evdev `KEY_*` codes this module reasons about by name.
///
/// These are Linux `input-event-codes.h` ABI values, the same wire form
/// [`super::input_map`] produces. They are duplicated here rather than derived
/// because [`super::input_map::keycode_to_evdev`] is a `match` on a windowing
/// type and cannot be evaluated in a `const`; `chord_codes_match_input_map`
/// pins each one against that table so the two cannot drift.
mod key {
    pub const ESC: u32 = 1;
    pub const LEFTCTRL: u32 = 29;
    pub const LEFTALT: u32 = 56;
    pub const RIGHTCTRL: u32 = 97;
    pub const RIGHTALT: u32 = 100;
}

/// The chord that releases the grab, as the operator presses it: Ctrl+Alt+Esc.
///
/// Either Ctrl and either Alt count, so the hand that reaches for it does not
/// have to be the left one. Named as a constant because the doc comment on
/// [`GrabState::key_down`] and the operator-facing message must not be able to
/// disagree about which keys they are.
pub const UNGRAB_CHORD: &str = "Ctrl+Alt+Esc";

/// Whether the host window is currently asking the compositor for its shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grab {
    /// The compositor keeps its own shortcuts; the guest sees only what is left.
    Released,
    /// The window has asked the compositor to forward everything.
    Held,
}

/// What a keyboard or focus event asks the window to do.
///
/// Actions and grab state travel together because they are decided together: the
/// ungrab chord both changes the grab and suppresses its own key, and focus loss
/// both drops the grab and releases every held key. Returning them as one value
/// is what keeps a caller from applying half of a transition.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct KeyEffect {
    /// Guest input to emit, in order. May be empty.
    pub actions: Vec<HostAction>,
    /// The grab state that should now be in force, if it changed.
    pub grab: Option<Grab>,
}

impl KeyEffect {
    fn nothing() -> Self {
        Self::default()
    }
}

/// The set of evdev codes the guest currently believes are held.
///
/// A `Vec` rather than a set: a human hand holds a handful of keys at once, the
/// membership test is over single-digit lengths, and release order is then the
/// order they were pressed in — which is the order a real keyboard would have
/// reported had the window stayed focused.
#[derive(Debug, Default)]
pub struct HeldKeys {
    down: Vec<u32>,
}

impl HeldKeys {
    /// Record `evdev` as held and produce its key-down.
    ///
    /// A repeat of a key already held still emits: the guest's own autorepeat
    /// contract is that it sees the repeats, and filtering them here would be a
    /// guest-visible change no decoded term asks for. Only the membership set is
    /// idempotent.
    pub fn press(&mut self, evdev: u32) -> HostAction {
        if !self.down.contains(&evdev) {
            self.down.push(evdev);
        }
        HostAction::input_key(evdev, true)
    }

    /// Produce `evdev`'s key-up, or nothing when the guest does not hold it.
    ///
    /// The `Option` is the invariant: an up is only ever emitted to close a down
    /// this type recorded, so a stray release — a synthetic one from the X11
    /// backend after this type has already released the key, or the operator
    /// lifting a key that was consumed by the ungrab chord — cannot reach the
    /// guest as an unpaired event.
    pub fn release(&mut self, evdev: u32) -> Option<HostAction> {
        let at = self.down.iter().position(|&k| k == evdev)?;
        self.down.remove(at);
        Some(HostAction::input_key(evdev, false))
    }

    /// Close every outstanding key-down, in press order, and forget them.
    ///
    /// The whole reason this module exists. Called on focus loss, on ungrab, and
    /// at teardown — every path after which this window will stop being told
    /// what the operator's hands are doing.
    pub fn release_all(&mut self) -> Vec<HostAction> {
        self.down
            .drain(..)
            .map(|evdev| HostAction::input_key(evdev, false))
            .collect()
    }

    /// Whether `evdev` is currently held. For chord recognition.
    fn holds(&self, evdev: u32) -> bool {
        self.down.contains(&evdev)
    }

    /// How many keys are outstanding. Test and census only.
    pub fn len(&self) -> usize {
        self.down.len()
    }

    /// Whether the guest holds nothing.
    pub fn is_empty(&self) -> bool {
        self.down.is_empty()
    }
}

/// Focus and ungrab-latch machine: decides whether capture *should* be active.
///
/// Two inputs, one derived answer. The window has keyboard focus or it does not,
/// and the operator has asked for release or has not; capture is wanted exactly
/// when the window is focused and no release is latched.
///
/// The latch clears on focus loss rather than on focus gain, which is what makes
/// the escape hatch a one-shot: Ctrl+Alt+Esc releases the grab, the operator
/// leaves with the Alt+Tab that is now theirs again, and coming back to the
/// window re-arms capture. An operator who wants to work beside the window
/// simply does not click into it.
#[derive(Debug, Default)]
pub struct GrabState {
    focused: bool,
    /// The operator pressed [`UNGRAB_CHORD`] and has not left the window since.
    released_by_operator: bool,
    /// What was last reported to the caller, so a transition is reported once.
    applied: Option<Grab>,
}

impl GrabState {
    /// The grab this state wants, ignoring what is currently applied.
    fn wanted(&self) -> Grab {
        if self.focused && !self.released_by_operator {
            Grab::Held
        } else {
            Grab::Released
        }
    }

    /// The wanted grab, but only when it differs from the last one reported.
    fn transition(&mut self) -> Option<Grab> {
        let wanted = self.wanted();
        if self.applied == Some(wanted) {
            return None;
        }
        self.applied = Some(wanted);
        Some(wanted)
    }

    /// Whether capture is currently meant to be in force.
    pub fn is_held(&self) -> bool {
        self.wanted() == Grab::Held
    }
}

/// The window's whole keyboard state: what the guest holds, and whether the
/// compositor's shortcuts are being captured.
///
/// One type because the two are not separable — the ungrab chord is read out of
/// the held set, and every grab transition has to settle the held set with it.
#[derive(Debug, Default)]
pub struct Keyboard {
    held: HeldKeys,
    grab: GrabState,
}

impl Keyboard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle a physical key transition, in evdev codes.
    ///
    /// Returns the guest input to emit and any grab change. The [`UNGRAB_CHORD`]
    /// is recognised on the *press* of Esc while both a Ctrl and an Alt are
    /// held; it is consumed, and it takes the rest of the held set with it so
    /// the guest does not keep the modifiers that formed it.
    pub fn key(&mut self, evdev: u32, down: bool) -> KeyEffect {
        if down && evdev == key::ESC && self.chord_modifiers_held() {
            return self.release_grab();
        }
        let actions = if down {
            vec![self.held.press(evdev)]
        } else {
            self.held.release(evdev).into_iter().collect()
        };
        KeyEffect {
            actions,
            grab: None,
        }
    }

    /// Handle a focus change.
    ///
    /// Losing focus releases every held key — the window is about to stop being
    /// told what the operator's hands are doing, and a key left down here stays
    /// down in the guest for the life of the boot. It also clears the operator's
    /// ungrab latch, so returning to the window re-arms capture.
    pub fn focus(&mut self, focused: bool) -> KeyEffect {
        self.grab.focused = focused;
        let actions = if focused {
            Vec::new()
        } else {
            self.grab.released_by_operator = false;
            self.held.release_all()
        };
        KeyEffect {
            actions,
            grab: self.grab.transition(),
        }
    }

    /// Release everything at teardown: every held key, and the grab.
    ///
    /// Distinct from `focus(false)` in intent though not currently in effect;
    /// the window calls this on its way out, where there is no focus event to
    /// rely on.
    pub fn shutdown(&mut self) -> KeyEffect {
        self.grab.focused = false;
        self.grab.released_by_operator = false;
        KeyEffect {
            actions: self.held.release_all(),
            grab: self.grab.transition(),
        }
    }

    /// Whether both halves of the ungrab chord's modifier set are held.
    fn chord_modifiers_held(&self) -> bool {
        let ctrl = self.held.holds(key::LEFTCTRL) || self.held.holds(key::RIGHTCTRL);
        let alt = self.held.holds(key::LEFTALT) || self.held.holds(key::RIGHTALT);
        ctrl && alt
    }

    /// Latch the operator's release and settle the guest's held set.
    fn release_grab(&mut self) -> KeyEffect {
        // Already released: the chord is still consumed (it is ours, not the
        // guest's) but there is nothing left to change.
        if self.grab.released_by_operator {
            return KeyEffect::nothing();
        }
        self.grab.released_by_operator = true;
        KeyEffect {
            actions: self.held.release_all(),
            grab: self.grab.transition(),
        }
    }

    /// Whether capture is currently meant to be in force. Census and tests.
    pub fn is_grabbed(&self) -> bool {
        self.grab.is_held()
    }

    /// How many keys the guest is believed to hold. Census and tests.
    pub fn held_count(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::HostActionKind;
    use winit::keyboard::KeyCode;

    /// Every code this module names must be the same code the window's own
    /// mapping table produces for that physical key. Nothing derives one from
    /// the other — this is the assertion that keeps them equal.
    #[test]
    fn chord_codes_match_input_map() {
        use super::super::input_map::keycode_to_evdev as ev;
        assert_eq!(ev(KeyCode::Escape), Some(key::ESC));
        assert_eq!(ev(KeyCode::ControlLeft), Some(key::LEFTCTRL));
        assert_eq!(ev(KeyCode::ControlRight), Some(key::RIGHTCTRL));
        assert_eq!(ev(KeyCode::AltLeft), Some(key::LEFTALT));
        assert_eq!(ev(KeyCode::AltRight), Some(key::RIGHTALT));
    }

    fn is_key(a: &HostAction, evdev: u32, down: bool) -> bool {
        a.kind == HostActionKind::InputKey && a.a0 as u32 == evdev && (a.a1 != 0) == down
    }

    /// The defect this module was written for: a key held when focus is lost
    /// must come up. Without the release-all, the guest holds it forever.
    #[test]
    fn focus_loss_releases_every_held_key() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        for code in [key::LEFTALT, key::LEFTCTRL, 30] {
            kb.key(code, true);
        }
        assert_eq!(kb.held_count(), 3);

        let effect = kb.focus(false);
        assert_eq!(effect.actions.len(), 3, "one up per held key, no more");
        assert!(is_key(&effect.actions[0], key::LEFTALT, false));
        assert!(is_key(&effect.actions[1], key::LEFTCTRL, false));
        assert!(is_key(&effect.actions[2], 30, false));
        assert_eq!(kb.held_count(), 0);
        assert_eq!(effect.grab, Some(Grab::Released));
    }

    /// Focus loss with nothing held emits nothing. A window that pumps spurious
    /// key-ups every time it is switched away from would be its own defect.
    #[test]
    fn focus_loss_with_nothing_held_emits_nothing() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(30, true);
        kb.key(30, false);
        let effect = kb.focus(false);
        assert!(effect.actions.is_empty());
    }

    /// An up for a key the guest does not hold is not forwarded. The X11 backend
    /// synthesises releases on focus loss; after this type has already released
    /// them, those must not reach the guest as unpaired ups.
    #[test]
    fn release_of_unheld_key_emits_nothing() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        assert!(kb.key(30, false).actions.is_empty());
        kb.key(30, true);
        kb.key(30, false);
        assert!(kb.key(30, false).actions.is_empty());
    }

    /// Repeats still reach the guest — the guest owns autorepeat — but they do
    /// not accumulate held state, so one release still closes the key.
    #[test]
    fn repeats_forward_but_do_not_accumulate() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        for _ in 0..4 {
            let e = kb.key(30, true);
            assert_eq!(e.actions.len(), 1);
            assert!(is_key(&e.actions[0], 30, true));
        }
        assert_eq!(kb.held_count(), 1);
        assert_eq!(kb.key(30, false).actions.len(), 1);
        assert_eq!(kb.held_count(), 0);
    }

    /// Focus alone drives the grab, and the transition is reported once.
    #[test]
    fn grab_follows_focus_and_reports_once() {
        let mut kb = Keyboard::new();
        assert_eq!(kb.focus(true).grab, Some(Grab::Held));
        assert!(kb.is_grabbed());
        assert_eq!(kb.focus(true).grab, None, "no change, no transition");
        assert_eq!(kb.focus(false).grab, Some(Grab::Released));
        assert!(!kb.is_grabbed());
    }

    /// The escape hatch: Ctrl+Alt+Esc releases the grab, is never forwarded to
    /// the guest, and leaves the guest holding neither Ctrl nor Alt.
    #[test]
    fn ungrab_chord_releases_grab_and_is_not_forwarded() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(key::LEFTCTRL, true);
        kb.key(key::LEFTALT, true);

        let effect = kb.key(key::ESC, true);
        assert_eq!(effect.grab, Some(Grab::Released));
        assert!(!kb.is_grabbed());
        // Esc itself never reaches the guest.
        assert!(
            !effect.actions.iter().any(|a| is_key(a, key::ESC, true)),
            "the ungrab chord's Esc must be consumed, not forwarded"
        );
        // And the modifiers that formed it are released.
        assert_eq!(effect.actions.len(), 2);
        assert!(is_key(&effect.actions[0], key::LEFTCTRL, false));
        assert!(is_key(&effect.actions[1], key::LEFTALT, false));
        assert_eq!(kb.held_count(), 0);
    }

    /// Either Ctrl and either Alt form the chord.
    #[test]
    fn ungrab_chord_accepts_either_hand() {
        for (ctrl, alt) in [
            (key::LEFTCTRL, key::LEFTALT),
            (key::LEFTCTRL, key::RIGHTALT),
            (key::RIGHTCTRL, key::LEFTALT),
            (key::RIGHTCTRL, key::RIGHTALT),
        ] {
            let mut kb = Keyboard::new();
            kb.focus(true);
            kb.key(ctrl, true);
            kb.key(alt, true);
            assert_eq!(
                kb.key(key::ESC, true).grab,
                Some(Grab::Released),
                "ctrl={ctrl} alt={alt} must form the chord"
            );
        }
    }

    /// Esc without the full modifier set is an ordinary guest key. A guest that
    /// could not send Esc, or could not send Ctrl+Esc, would be a worse defect
    /// than the one being fixed.
    #[test]
    fn esc_without_both_modifiers_reaches_the_guest() {
        for held in [vec![], vec![key::LEFTCTRL], vec![key::LEFTALT]] {
            let mut kb = Keyboard::new();
            kb.focus(true);
            for k in &held {
                kb.key(*k, true);
            }
            let effect = kb.key(key::ESC, true);
            assert_eq!(effect.grab, None, "held={held:?} must not touch the grab");
            assert!(
                effect.actions.iter().any(|a| is_key(a, key::ESC, true)),
                "held={held:?} must forward Esc to the guest"
            );
            assert!(kb.is_grabbed());
        }
    }

    /// The advertised chord must be the one the recogniser actually accepts.
    ///
    /// `UNGRAB_CHORD` is what the operator is told — on stderr and in the
    /// `window_capture_engaged` census line — at the moment their desktop
    /// shortcuts stop working. If the string and the recogniser ever disagreed,
    /// the operator would be holding a keyboard grab with printed instructions
    /// that do not release it. This drives the recogniser with exactly the keys
    /// the string names, so the two cannot drift.
    #[test]
    fn the_advertised_chord_is_the_one_that_releases() {
        let named: Vec<&str> = UNGRAB_CHORD.split('+').collect();
        assert_eq!(named, vec!["Ctrl", "Alt", "Esc"], "chord spelling changed");

        let code = |part: &str| match part {
            "Ctrl" => key::LEFTCTRL,
            "Alt" => key::LEFTALT,
            "Esc" => key::ESC,
            other => panic!("no evdev code wired for advertised chord part {other:?}"),
        };
        let mut kb = Keyboard::new();
        kb.focus(true);
        assert!(kb.is_grabbed());
        // Press exactly what the advertised string names, in the order it names
        // them; the last one must be what releases the grab.
        let mut effect = KeyEffect::nothing();
        for part in &named {
            effect = kb.key(code(part), true);
        }
        assert_eq!(
            effect.grab,
            Some(Grab::Released),
            "pressing the advertised chord {UNGRAB_CHORD} must release the grab"
        );
        assert!(!kb.is_grabbed());
    }

    /// The guest's own Force Quit (Cmd+Option+Esc, pressed on the host as
    /// Meta+Alt+Esc) carries no Ctrl, so it is not the ungrab chord and must
    /// reach the guest whole — including while the grab is held, which is
    /// exactly when the guest is likely to need it.
    #[test]
    fn force_quit_chord_reaches_the_guest() {
        const LEFTMETA: u32 = 125;
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(LEFTMETA, true);
        kb.key(key::LEFTALT, true);
        let effect = kb.key(key::ESC, true);
        assert_eq!(effect.grab, None, "Meta+Alt+Esc must not touch the grab");
        assert!(
            effect.actions.iter().any(|a| is_key(a, key::ESC, true)),
            "the guest's Force Quit must reach it"
        );
        assert!(kb.is_grabbed(), "and the grab is still held");
    }

    /// The Meta code this file names must be the one the window's mapping table
    /// produces, like the chord codes above.
    #[test]
    fn force_quit_test_uses_the_real_meta_code() {
        use super::super::input_map::keycode_to_evdev as ev;
        assert_eq!(ev(winit::keyboard::KeyCode::SuperLeft), Some(125));
    }

    /// The latch is a one-shot: leaving the window and coming back re-arms
    /// capture, so the operator does not have to know the grab is a mode.
    #[test]
    fn returning_focus_rearms_the_grab() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(key::LEFTCTRL, true);
        kb.key(key::LEFTALT, true);
        kb.key(key::ESC, true);
        assert!(!kb.is_grabbed());

        assert_eq!(kb.focus(false).grab, None, "already released");
        assert_eq!(kb.focus(true).grab, Some(Grab::Held), "return re-arms");
        assert!(kb.is_grabbed());
    }

    /// Pressing the chord twice without leaving is idempotent, and the second
    /// Esc is still consumed rather than leaking to the guest.
    #[test]
    fn repeated_ungrab_chord_is_idempotent_and_still_consumed() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(key::LEFTCTRL, true);
        kb.key(key::LEFTALT, true);
        kb.key(key::ESC, true);

        kb.key(key::LEFTCTRL, true);
        kb.key(key::LEFTALT, true);
        let again = kb.key(key::ESC, true);
        assert_eq!(again.grab, None, "already released; no second transition");
        assert!(!again.actions.iter().any(|a| is_key(a, key::ESC, true)));
    }

    /// Teardown closes the held set even though no focus event arrives.
    #[test]
    fn shutdown_releases_held_keys() {
        let mut kb = Keyboard::new();
        kb.focus(true);
        kb.key(key::LEFTALT, true);
        kb.key(30, true);
        let effect = kb.shutdown();
        assert_eq!(effect.actions.len(), 2);
        assert!(effect.actions.iter().all(|a| a.a1 == 0), "all are key-ups");
        assert_eq!(kb.held_count(), 0);
    }
}
