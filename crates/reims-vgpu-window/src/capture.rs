//! Input capture: whether the guest or the host gets a key, and where the host's pointer may go.
//!
//! A click into the image captures the keyboard and the pointer together: from then on every key
//! goes to the guest, ⌘ shortcuts included (and the host's system keys, which the AppKit side
//! turns off while captured), and the host's pointer is held over the image
//! ([`Capture::pointer_confinement`]): it cannot leave for the title bar or the host's other
//! windows, and every click is the guest's. Nor can the window be resized while captured
//! ([`Capture::window_resizable`]): where the image reaches the window's edges, the pointer held
//! over it is over the window's resize edges too, where a click meant for the guest would drag
//! the window smaller instead, and never larger, the hold keeping the pointer from going
//! outward. The keyboard is never captured without the pointer,
//! nor the pointer without the keyboard. Pressing ⌃⌥⌘ together releases both (the same keys,
//! named Ctrl+Alt+Super on Linux and Ctrl+Alt+Win on Windows: [`captured_suffix`]), and so does
//! the window losing the keyboard. Uncaptured, a key with ⌘ held drives the host's menu and every
//! other key goes to the guest, and the pointer goes where the person moves it.
//!
//! The click that makes the window key only brings it forward: the view does not take a first
//! click (`acceptsFirstMouse:`), so that click neither captures nor reaches the guest. Clicks
//! after it capture, and are sent to the guest.
//!
//! With capture off, the host's setting for the VM ([`Capture::with_setting`],
//! `crate::env`'s `REIMS_VGPU_WINDOW_CAPTURE`), nothing is ever captured: a click into the image
//! is sent as before but captures nothing, so the host keeps its system keys and the pointer goes
//! where the person moves it, and the window stays resizable. Every key goes to the guest while
//! the window has the keyboard, ⌘ shortcuts included, as while captured
//! ([`Capture::keys_are_the_guest_s`]): with no capture there is no other way to give the guest
//! one, and the host's menu is in the pointer's reach. The pointer reaches the guest over the
//! image alone, and outside it only in a drag, so the drag ends in the guest
//! ([`Capture::pointer_outside_image`]). The title says so (`OFF_SUFFIX`).

use crate::geometry::Rect;
use crate::keys::Modifiers;

/// What to do with a click into the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClickAction {
    /// Whether the click goes to the guest.
    pub forward: bool,
}

/// Who a key goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The guest's keyboard.
    Guest,
    /// The host: the app's menu and AppKit's own handling.
    Host,
}

/// Whether input, the keyboard and the pointer, is captured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capture {
    /// Captured.
    captured: bool,
    /// Capture is off: nothing is ever captured (the module doc).
    off: bool,
}

/// The host's OS, whose names for the release chord's keys the title gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostOs {
    /// macOS: ⌃⌥⌘.
    MacOs,
    /// Linux: Ctrl+Alt+Super.
    Linux,
    /// Windows: Ctrl+Alt+Win. The Windows key is SDL's GUI key there, as Super is on Linux, the
    /// key the chord holds (`native/windows.rs`).
    Windows,
}

/// What the title says while captured on `host`: that input is captured, and the release chord in
/// the names `host` gives its keys.
pub fn captured_suffix(host: HostOs) -> &'static str {
    match host {
        HostOs::MacOs => " \u{2014} input captured (press \u{2303}\u{2325}\u{2318} to release)",
        HostOs::Linux => " \u{2014} input captured (press Ctrl+Alt+Super to release)",
        HostOs::Windows => " \u{2014} input captured (press Ctrl+Alt+Win to release)",
    }
}

/// What the title says while not captured, on every OS: that a click captures.
pub(crate) const UNCAPTURED_SUFFIX: &str = " \u{2014} click to capture input";

/// What the title says with capture off, on every OS: that nothing captures.
pub(crate) const OFF_SUFFIX: &str = " \u{2014} input capture off";

impl Capture {
    /// Not captured, capture on.
    pub fn new() -> Capture {
        Capture::default()
    }

    /// Not captured, capture on where `on`, else off: nothing is ever captured (the module doc).
    pub fn with_setting(on: bool) -> Capture {
        Capture {
            captured: false,
            off: !on,
        }
    }

    /// Whether input, the keyboard and the pointer, is captured.
    pub fn is_captured(&self) -> bool {
        self.captured
    }

    /// A click landed in the image: the keyboard and the pointer are captured. `window_was_key`
    /// is whether the window already had the keyboard before the click; a click that made it
    /// key, where one reaches the input, is not forwarded.
    /// With capture off, the click captures nothing.
    pub fn click_in_image(&mut self, window_was_key: bool) -> ClickAction {
        self.captured = !self.off;
        ClickAction {
            forward: window_was_key,
        }
    }

    /// Releases the keyboard and the pointer. Whether they were captured.
    pub fn release(&mut self) -> bool {
        std::mem::replace(&mut self.captured, false)
    }

    /// The window lost the keyboard: capture is released. Whether it was captured, so that the
    /// host's system keys are given back.
    pub fn resign_key(&mut self) -> bool {
        self.release()
    }

    /// Where the host's pointer is held, `image` the image's rectangle in the window's points:
    /// over the image while captured, where it has an area. `None` while not captured, and over
    /// an image of no area, which would hold the pointer still: the pointer then goes where the
    /// person moves it.
    pub fn pointer_confinement(&self, image: Rect) -> Option<Rect> {
        (self.captured && image.width > 0.0 && image.height > 0.0).then_some(image)
    }

    /// Whether the person may resize the window: not while captured (module doc), and as before
    /// once released.
    pub fn window_resizable(&self) -> bool {
        !self.captured
    }

    /// Who a key goes to, `command_held` whether ⌘ is down: the guest while every key is its
    /// ([`Capture::keys_are_the_guest_s`]); else the host for a ⌘ key, and the guest for others.
    pub fn route(&self, command_held: bool) -> Route {
        if !self.keys_are_the_guest_s() && command_held {
            Route::Host
        } else {
            Route::Guest
        }
    }

    /// Whether every key goes to the guest, ⌘ shortcuts included: while captured, and always
    /// with capture off. Only capture turns the host's system keys off.
    pub fn keys_are_the_guest_s(&self) -> bool {
        self.captured || self.off
    }

    /// Whether the pointer outside the image goes to the guest, held at the image's nearest
    /// edge, `dragging` whether a button the guest was told is down is still down: always with
    /// capture on, and with it off only in a drag, so the drag ends in the guest.
    pub fn pointer_outside_image(&self, dragging: bool) -> bool {
        !self.off || dragging
    }

    /// Whether `held` is the release chord, ⌃⌥⌘ all down (Shift either way), while captured.
    /// The caller then sends the ups of the keys down ([`crate::keys::KeyState::release_all`])
    /// and releases capture ([`Capture::release`]).
    pub fn release_chord(&self, held: Modifiers) -> bool {
        self.captured && held.control && held.option && held.command
    }

    /// The text the window's title ends with on `host`: what capture is on and how to release it
    /// while captured ([`captured_suffix`]), that a click captures while not, and that capture
    /// is off where it is (`OFF_SUFFIX`).
    pub fn title_suffix(&self, host: HostOs) -> &'static str {
        if self.off {
            OFF_SUFFIX
        } else if self.captured {
            captured_suffix(host)
        } else {
            UNCAPTURED_SUFFIX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHORD: Modifiers = Modifiers {
        control: true,
        option: true,
        command: true,
        shift: false,
    };

    /// A click into the image captures; the click that makes the window key is not forwarded
    #[test]
    fn a_click_in_the_image_captures() {
        let mut capture = Capture::new();
        assert!(!capture.is_captured());
        assert_eq!(
            capture.click_in_image(false),
            ClickAction { forward: false }
        );
        assert!(capture.is_captured());
        assert_eq!(capture.click_in_image(true), ClickAction { forward: true });
        assert!(capture.is_captured());
        // Released, the window still key: the next click captures and is forwarded.
        assert!(capture.release());
        assert_eq!(capture.click_in_image(true), ClickAction { forward: true });
        assert!(capture.is_captured());
    }

    /// Captured, every key goes to the guest, ⌘ shortcuts included
    #[test]
    fn captured_keys_go_to_the_guest() {
        let mut capture = Capture::new();
        capture.click_in_image(true);
        assert_eq!(capture.route(true), Route::Guest);
        assert_eq!(capture.route(false), Route::Guest);
    }

    /// Uncaptured, ⌘ keys go to the host's menu and other keys to the guest
    #[test]
    fn uncaptured_command_goes_to_the_host() {
        let capture = Capture::new();
        assert_eq!(capture.route(true), Route::Host);
        assert_eq!(capture.route(false), Route::Guest);
    }

    /// ⌃⌥⌘ together releases capture, with or without Shift; fewer do not
    #[test]
    fn control_option_command_releases() {
        let mut capture = Capture::new();
        assert!(!capture.release_chord(CHORD), "not captured");
        capture.click_in_image(true);
        assert!(capture.release_chord(CHORD));
        assert!(capture.release_chord(Modifiers {
            shift: true,
            ..CHORD
        }));
        for held in [
            Modifiers {
                control: false,
                ..CHORD
            },
            Modifiers {
                option: false,
                ..CHORD
            },
            Modifiers {
                command: false,
                ..CHORD
            },
            Modifiers::default(),
        ] {
            assert!(!capture.release_chord(held), "{held:?}");
        }
        assert!(capture.release());
        assert!(!capture.is_captured());
        assert!(!capture.release());
    }

    /// Losing the keyboard releases capture, and says whether it was captured
    #[test]
    fn losing_focus_releases() {
        let mut capture = Capture::new();
        capture.click_in_image(false);
        assert!(capture.resign_key());
        assert!(!capture.is_captured());
        assert_eq!(capture.route(true), Route::Host);
        assert!(!capture.resign_key());
    }

    /// Captured, the pointer is held over the image; released, by the chord or by losing the
    /// keyboard, it is free; an image of no area holds nothing
    #[test]
    fn captured_input_holds_the_pointer_over_the_image() {
        let image = Rect {
            x: 100.0,
            y: 50.5,
            width: 960.0,
            height: 600.0,
        };
        let mut capture = Capture::new();
        assert_eq!(capture.pointer_confinement(image), None);
        capture.click_in_image(false);
        assert_eq!(capture.pointer_confinement(image), Some(image));
        for empty in [
            Rect::default(),
            Rect {
                width: 0.0,
                ..image
            },
            Rect {
                height: 0.0,
                ..image
            },
        ] {
            assert_eq!(capture.pointer_confinement(empty), None, "{empty:?}");
        }
        capture.release();
        assert_eq!(capture.pointer_confinement(image), None);
        capture.click_in_image(true);
        capture.resign_key();
        assert_eq!(capture.pointer_confinement(image), None);
    }

    /// Captured, the window cannot be resized; released, by the chord or by losing the keyboard,
    /// it can again
    #[test]
    fn a_captured_window_cannot_be_resized() {
        let mut capture = Capture::new();
        assert!(capture.window_resizable());
        capture.click_in_image(false);
        assert!(!capture.window_resizable());
        capture.release();
        assert!(capture.window_resizable());
        capture.click_in_image(true);
        assert!(!capture.window_resizable());
        capture.resign_key();
        assert!(capture.window_resizable());
    }

    /// The title says a click captures while not captured, and how to release while captured
    #[test]
    fn the_title_says_whether_input_is_captured() {
        let mut capture = Capture::new();
        assert_eq!(
            capture.title_suffix(HostOs::MacOs),
            " \u{2014} click to capture input"
        );
        capture.click_in_image(true);
        assert!(capture.title_suffix(HostOs::MacOs).contains("captured"));
        assert!(
            capture
                .title_suffix(HostOs::MacOs)
                .contains("\u{2303}\u{2325}\u{2318}")
        );
        capture.release();
        assert_eq!(capture.title_suffix(HostOs::MacOs), UNCAPTURED_SUFFIX);
        capture.click_in_image(false);
        capture.resign_key();
        assert_eq!(capture.title_suffix(HostOs::MacOs), UNCAPTURED_SUFFIX);
    }

    /// Captured, the title names the release chord's keys as the host's OS does: ⌃⌥⌘ on macOS,
    /// Ctrl+Alt+Super on Linux and Ctrl+Alt+Win on Windows; not captured, it is the same on each
    #[test]
    fn the_title_names_the_release_chord_in_the_host_s_keys() {
        let mut capture = Capture::new();
        for host in [HostOs::MacOs, HostOs::Linux, HostOs::Windows] {
            assert_eq!(capture.title_suffix(host), UNCAPTURED_SUFFIX, "{host:?}");
        }
        capture.click_in_image(true);
        assert_eq!(
            capture.title_suffix(HostOs::MacOs),
            " \u{2014} input captured (press \u{2303}\u{2325}\u{2318} to release)"
        );
        assert_eq!(
            capture.title_suffix(HostOs::Linux),
            " \u{2014} input captured (press Ctrl+Alt+Super to release)"
        );
        assert_eq!(
            capture.title_suffix(HostOs::Windows),
            " \u{2014} input captured (press Ctrl+Alt+Win to release)"
        );
    }

    /// With capture off, a click into the image is sent but captures nothing: the pointer is
    /// never held, the window stays resizable, no chord releases anything, and every key goes to
    /// the guest, ⌘ shortcuts included; the title says capture is off on every OS
    #[test]
    fn capture_off_never_captures() {
        let image = Rect {
            x: 0.0,
            y: 0.0,
            width: 960.0,
            height: 600.0,
        };
        let mut capture = Capture::with_setting(false);
        assert_eq!(
            capture.click_in_image(false),
            ClickAction { forward: false }
        );
        assert_eq!(capture.click_in_image(true), ClickAction { forward: true });
        assert!(!capture.is_captured());
        assert_eq!(capture.pointer_confinement(image), None);
        assert!(capture.window_resizable());
        assert!(!capture.release_chord(CHORD));
        assert!(!capture.release());
        assert!(!capture.resign_key());
        assert!(capture.keys_are_the_guest_s());
        assert_eq!(capture.route(true), Route::Guest);
        assert_eq!(capture.route(false), Route::Guest);
        for host in [HostOs::MacOs, HostOs::Linux, HostOs::Windows] {
            assert_eq!(
                capture.title_suffix(host),
                " \u{2014} input capture off",
                "{host:?}"
            );
        }
        // On, as before.
        let mut on = Capture::with_setting(true);
        assert_eq!(on, Capture::new());
        assert!(!on.keys_are_the_guest_s());
        on.click_in_image(true);
        assert!(on.is_captured() && on.keys_are_the_guest_s());
    }

    /// The pointer outside the image reaches the guest with capture on, and with it off only in
    /// a drag
    #[test]
    fn capture_off_sends_the_pointer_outside_the_image_only_in_a_drag() {
        let mut on = Capture::new();
        assert!(on.pointer_outside_image(false) && on.pointer_outside_image(true));
        on.click_in_image(true);
        assert!(on.pointer_outside_image(false));
        let off = Capture::with_setting(false);
        assert!(!off.pointer_outside_image(false));
        assert!(off.pointer_outside_image(true));
    }
}
