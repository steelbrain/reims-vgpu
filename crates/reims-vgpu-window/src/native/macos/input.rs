//! A window's input: AppKit's events, as the window's view and the application see them, to the
//! host's callbacks ([`crate::config::WindowCallbacks`]).
//!
//! An [`InputState`] belongs to one window and lives on the main thread. The view calls it for
//! each event it receives, and the application calls [`InputState::app_send_event`] for the key
//! window's events before AppKit dispatches them. It reads each event's values (every send
//! through the message layer, `reims_vgpu_vulkan::os::msg`), decides with the crate's pure logic
//! ([`crate::keys`], [`crate::capture`], [`crate::scroll`], [`crate::geometry`],
//! [`crate::keymap`]) in an [`InputLogic`] that takes plain values, and calls the host's
//! callbacks, taken from the [`Window`] at each event: after the window's end there are none,
//! and nothing is sent.
//!
//! What the guest gets:
//! - **The pointer** once the guest has a mode, at the guest pixel under the host's pointer,
//!   held at the image's nearest edge outside it. A click in the image captures the keyboard and
//!   the pointer; the click that makes the window key is not sent. A click outside the image,
//!   and one of a button past the guest tablet's five, is not sent and starts nothing. A
//!   button's release is sent wherever it happens, so a drag that leaves the image ends in the
//!   guest; AppKit keeps sending the drag's moves to the view meanwhile.
//! - **The wheel** over the image, once the guest has a mode, after the pointer's position where
//!   it moved: a window that is not key gets no moves, and the guest scrolls where the host's
//!   pointer is. Where the host takes smooth scrolling, each event goes with its phases, and a
//!   gesture that began over the image goes to its end, momentum included ([`crate::scroll`]);
//!   else the guest gets wheel notches.
//! - **Keys** by position ([`crate::keymap`]), one down and one up each: the host's repeats are
//!   dropped, since the guest repeats a held key itself. A key, or a click sent to the guest,
//!   that AppKit reports with a modifier the window did not see change (one pressed before the
//!   window was key) brings the guest's modifiers into line first. Uncaptured, a key with
//!   ⌘ held is the host's (its menu); captured, every key is the guest's, and the application
//!   takes a ⌘ key from AppKit before its menu sees it. A key-up under ⌘, which AppKit never
//!   hands a view, is taken there too. Pressing ⌃⌥⌘ while captured lifts every key held and
//!   releases capture.
//! - **No stuck keys**: losing the keyboard, releasing capture and closing send an up for every
//!   key held; losing the keyboard and closing lift the buttons held too.
//!
//! While captured, the window holds the system's hot keys off ([`super::hotkeys`]), the host's
//! pointer over the image ([`InputState::pointer_confinement`], which the view applies), and its
//! own size ([`InputState::window_resizable`], which the view applies too).

use std::sync::Arc;

use objc2_app_kit::{NSEvent, NSEventPhase, NSEventType};
use objc2_core_graphics::{CGEvent, CGEventField};
use reims_vgpu_vulkan::os::msg;

use super::hotkeys;
use crate::capture::{Capture, Route};
use crate::config::{MomentumPhase, ScrollEvent, ScrollPhase, WindowCallbacks};
use crate::geometry::{self, Rect};
use crate::handle::Window;
use crate::keymap;
use crate::keys::{FLAG_COMMAND, KeyEvent, KeyState};
use crate::scroll::{self, Gesture, Scroll, ScrollInput};

/// What the view knows when an event reaches it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ViewContext {
    /// The event's location in the view, in flipped points (y down), for a mouse or scroll
    /// event.
    pub(crate) point: Option<(f64, f64)>,
    /// The image's rectangle in the view, in flipped points.
    pub(crate) image: Rect,
    /// The guest's mode, once it has set one.
    pub(crate) guest: Option<(u16, u16)>,
    /// Whether the window was key when the event arrived.
    pub(crate) window_is_key: bool,
}

/// What the view does after an event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reaction {
    /// The capture state changed: the window's title, the cursor and the pointer's hold follow.
    pub(crate) capture_changed: bool,
    /// The event is AppKit's too: the view passes it to `super` (a ⌘ shortcut to the menu).
    pub(crate) pass_on: bool,
}

impl Reaction {
    /// Capture changed when `changed`; the event is the input's.
    fn capture(changed: bool) -> Reaction {
        Reaction {
            capture_changed: changed,
            pass_on: false,
        }
    }
}

/// What the input sends the host, in the order it is sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToHost {
    /// `pointer(x, y, width, height, buttons)`.
    Pointer(PointerAt),
    /// `wheel(dy, dx)`.
    Wheel {
        /// Notches up.
        dy: i32,
        /// Notches left.
        dx: i32,
    },
    /// `scroll(event)`.
    Scroll(ScrollEvent),
    /// `key(usage, down)`.
    Key(KeyEvent),
}

/// A pointer event: where, in which mode, with which buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PointerAt {
    /// The guest pixel, `x` then `y`.
    at: (u16, u16),
    /// The guest's mode, width then height, neither 0.
    mode: (u16, u16),
    /// The `BUTTON_*` bits held.
    buttons: u32,
}

/// A key event's values, read from its `NSEvent`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KeyInput {
    /// `keyCode`.
    keycode: u16,
    /// `modifierFlags`, device-dependent bits included.
    flags: u64,
    /// `isARepeat`; false for `flagsChanged:`, which has none.
    repeat: bool,
}

/// The guest's mode in `cx`, where it has one with pixels: the pointer is sent only then, since
/// the host scales it by the mode.
fn guest_mode(cx: &ViewContext) -> Option<(u16, u16)> {
    cx.guest.filter(|&(width, height)| width > 0 && height > 0)
}

/// Whether the application takes a key event from AppKit (`down` a key-down, else a key-up,
/// with `flags` its `modifierFlags`) for the guest, before AppKit dispatches it: a key-down with
/// ⌘ while `captured`, which would otherwise go to the menu; a key-up with ⌘ held, captured or
/// not, which AppKit never hands the view.
fn app_takes(captured: bool, down: bool, flags: u64) -> bool {
    flags & FLAG_COMMAND != 0 && (captured || !down)
}

/// A scroll event's `phase`. Stationary (fingers held still) is Changed.
fn scroll_phase(phase: NSEventPhase) -> ScrollPhase {
    [
        (NSEventPhase::MayBegin, ScrollPhase::MayBegin),
        (NSEventPhase::Began, ScrollPhase::Began),
        (NSEventPhase::Ended, ScrollPhase::Ended),
        (NSEventPhase::Cancelled, ScrollPhase::Cancelled),
        (NSEventPhase::Changed, ScrollPhase::Changed),
        (NSEventPhase::Stationary, ScrollPhase::Changed),
    ]
    .into_iter()
    .find_map(|(bit, ours)| phase.contains(bit).then_some(ours))
    .unwrap_or(ScrollPhase::None)
}

/// A scroll event's `momentumPhase`. Cancelled is Ended, and Stationary is Changed.
fn momentum_phase(phase: NSEventPhase) -> MomentumPhase {
    [
        (NSEventPhase::Began, MomentumPhase::Began),
        (NSEventPhase::Ended, MomentumPhase::Ended),
        (NSEventPhase::Cancelled, MomentumPhase::Ended),
        (NSEventPhase::Changed, MomentumPhase::Changed),
        (NSEventPhase::Stationary, MomentumPhase::Changed),
    ]
    .into_iter()
    .find_map(|(bit, ours)| phase.contains(bit).then_some(ours))
    .unwrap_or(MomentumPhase::None)
}

/// One window's input decisions, from plain values: what each event sends the host
/// ([`InputLogic::out`]) and what the view does after it.
#[derive(Debug, Default)]
struct InputLogic {
    /// Whether the keyboard in use is ISO ([`crate::native::is_iso_keyboard`]), asked again at
    /// each key event.
    iso: bool,
    /// The keys the guest was told are down.
    keys: KeyState,
    /// The usage each key the guest was told is down went down as, by its `keyCode`: its up
    /// names the same usage though `iso` changed between them (another keyboard typed on), so
    /// the guest is never left holding the down.
    downs: Vec<(u16, u16)>,
    /// Who gets the keyboard.
    capture: Capture,
    /// Precise scrolling's points not yet a notch.
    scroll: Scroll,
    /// Which smooth scrolling events go to the guest.
    gesture: Gesture,
    /// The `BUTTON_*` bits of the buttons the guest was told are down.
    buttons: u32,
    /// The pointer event last sent.
    last: Option<PointerAt>,
    /// What to send, in order, since the host's callbacks last took it.
    out: Vec<ToHost>,
}

impl InputLogic {
    /// Nothing held and nothing captured, on an ISO keyboard (`iso`) or not.
    fn new(iso: bool) -> InputLogic {
        InputLogic {
            iso,
            ..InputLogic::default()
        }
    }

    /// Sends the pointer at the guest pixel under `point`, with the buttons held.
    fn send_pointer(&mut self, point: (f64, f64), image: Rect, mode: (u16, u16)) {
        let at = geometry::pointer_to_guest(point, image, mode);
        self.send_pointer_at(at, mode);
    }

    /// Sends the pointer at `at` in `mode`, with the buttons held.
    fn send_pointer_at(&mut self, at: (u16, u16), mode: (u16, u16)) {
        let event = PointerAt {
            at,
            mode,
            buttons: self.buttons,
        };
        self.last = Some(event);
        self.out.push(ToHost::Pointer(event));
    }

    /// Sends the pointer at the guest pixel under `point` unless it is there already, with the
    /// same buttons.
    fn move_pointer(&mut self, point: (f64, f64), image: Rect, mode: (u16, u16)) {
        let at = geometry::pointer_to_guest(point, image, mode);
        let event = PointerAt {
            at,
            mode,
            buttons: self.buttons,
        };
        if self.last != Some(event) {
            self.send_pointer_at(at, mode);
        }
    }

    /// Sends `events`, keys down and up.
    fn send_keys(&mut self, events: Vec<KeyEvent>) {
        self.out.extend(events.into_iter().map(ToHost::Key));
    }

    /// Brings the modifiers the guest holds into line with an event's `flags`.
    fn sync_modifiers(&mut self, flags: u64) {
        let events = self.keys.sync_modifiers(flags);
        self.send_keys(events);
    }

    /// Ups for every key held.
    fn lift_keys(&mut self) {
        let events = self.keys.release_all();
        self.downs.clear();
        self.send_keys(events);
    }

    /// Lifts every button held, where the pointer was last sent.
    fn lift_buttons(&mut self) {
        if self.buttons == 0 {
            return;
        }
        self.buttons = 0;
        if let Some(last) = self.last {
            self.send_pointer_at(last.at, last.mode);
        }
    }

    /// A button went down: `bit` its `BUTTON_*` bit, 0 for one past the guest tablet's five,
    /// `flags` the event's `modifierFlags`.
    fn button_down(&mut self, bit: u32, flags: u64, cx: &ViewContext) -> Reaction {
        let (Some(mode), Some(point)) = (guest_mode(cx), cx.point) else {
            return Reaction::default();
        };
        if bit == 0 || !geometry::contains(cx.image, point) {
            return Reaction::default();
        }
        let was = self.capture.is_captured();
        let click = self.capture.click_in_image(cx.window_is_key);
        let reaction = Reaction::capture(self.capture.is_captured() != was);
        if !click.forward {
            return reaction;
        }
        self.sync_modifiers(flags);
        self.buttons |= bit;
        self.send_pointer(point, cx.image, mode);
        reaction
    }

    /// A button went up: `bit` its `BUTTON_*` bit. Sent only for a button the guest was told
    /// is down, wherever the pointer is.
    fn button_up(&mut self, bit: u32, cx: &ViewContext) -> Reaction {
        if bit == 0 || self.buttons & bit == 0 {
            return Reaction::default();
        }
        self.buttons &= !bit;
        match (guest_mode(cx), cx.point) {
            (Some(mode), Some(point)) => self.send_pointer(point, cx.image, mode),
            _ => {
                if let Some(last) = self.last {
                    self.send_pointer_at(last.at, last.mode);
                }
            }
        }
        Reaction::default()
    }

    /// The pointer moved, or dragged.
    fn moved(&mut self, cx: &ViewContext) -> Reaction {
        if let (Some(mode), Some(point)) = (guest_mode(cx), cx.point) {
            self.move_pointer(point, cx.image, mode);
        }
        Reaction::default()
    }

    /// The wheel or a trackpad scrolled: as smooth scrolling when `smooth` (the host takes
    /// it), else as notches.
    fn scrolled(&mut self, input: ScrollInput, smooth: bool, cx: &ViewContext) -> Reaction {
        let over = match (guest_mode(cx), cx.point) {
            (Some(mode), Some(point)) if geometry::contains(cx.image, point) => Some((point, mode)),
            _ => None,
        };
        if smooth {
            self.scrolled_smoothly(input, cx.point, over, cx.image);
            return Reaction::default();
        }
        let Some((point, mode)) = over else {
            return Reaction::default();
        };
        let (dy, dx) = self.scroll.feed(input);
        if dy != 0 || dx != 0 {
            self.move_pointer(point, cx.image, mode);
            self.out.push(ToHost::Wheel { dy, dx });
        }
        Reaction::default()
    }

    /// Smooth scrolling: the event goes where [`Gesture::route`] sends it, the pointer at `at`,
    /// after the pointer's position where the pointer is over the image (`over`: where, in which
    /// mode), with its notches. An event of no phase, no distance and no notch is not sent.
    fn scrolled_smoothly(
        &mut self,
        input: ScrollInput,
        at: Option<(f64, f64)>,
        over: Option<((f64, f64), (u16, u16))>,
        image: Rect,
    ) {
        if !self
            .gesture
            .route(input.phase, input.momentum, at, over.is_some())
        {
            return;
        }
        let (dy, dx) = scroll::smooth(input);
        let event = ScrollEvent {
            dy,
            dx,
            phase: input.phase,
            momentum: input.momentum,
            notches: self.scroll.feed(input),
        };
        if event == ScrollEvent::default() {
            return;
        }
        if let Some((point, mode)) = over {
            self.move_pointer(point, image, mode);
        }
        self.out.push(ToHost::Scroll(event));
    }

    /// A key went down (`keyDown:`).
    fn key_down(&mut self, key: KeyInput) -> Reaction {
        if key.repeat {
            return Reaction::default();
        }
        self.sync_modifiers(key.flags);
        if self.capture.route(key.flags & FLAG_COMMAND != 0) == Route::Host {
            return Reaction {
                capture_changed: false,
                pass_on: true,
            };
        }
        if let Some(usage) = keymap::usb_usage(key.keycode, self.iso) {
            let event = self.keys.key_down(usage, false);
            if event.is_some() {
                self.downs.retain(|&(keycode, _)| keycode != key.keycode);
                self.downs.push((key.keycode, usage));
            }
            self.out.extend(event.map(ToHost::Key));
        }
        Reaction::default()
    }

    /// A key went up (`keyUp:`): sent for a key the guest was told is down, as the usage it went
    /// down as ([`Self::downs`]).
    fn key_up(&mut self, key: KeyInput) -> Reaction {
        self.sync_modifiers(key.flags);
        let went_down = (self.downs.iter())
            .position(|&(keycode, _)| keycode == key.keycode)
            .map(|at| self.downs.remove(at).1);
        if let Some(usage) = went_down.or_else(|| keymap::usb_usage(key.keycode, self.iso)) {
            let event = self.keys.key_up(usage);
            self.out.extend(event.map(ToHost::Key));
        }
        Reaction::default()
    }

    /// A modifier changed (`flagsChanged:`); ⌃⌥⌘ all held while captured releases capture.
    fn flags_changed(&mut self, key: KeyInput) -> Reaction {
        let events = self.keys.flags_changed(key.keycode, key.flags);
        self.send_keys(events);
        if !self.capture.release_chord(self.keys.held_modifiers()) {
            return Reaction::default();
        }
        self.lift_keys();
        Reaction::capture(self.capture.release())
    }

    /// The application saw a key event before AppKit dispatches it: true when the input takes
    /// it ([`app_takes`]).
    fn app_key(&mut self, down: bool, key: KeyInput) -> bool {
        if !app_takes(self.capture.is_captured(), down, key.flags) {
            return false;
        }
        if down {
            self.key_down(key);
        } else {
            self.key_up(key);
        }
        true
    }

    /// The window stopped being key.
    fn focus_lost(&mut self) -> Reaction {
        let was = self.capture.resign_key();
        self.lift_keys();
        self.lift_buttons();
        Reaction::capture(was)
    }

    /// Release Input.
    fn release_capture(&mut self) -> Reaction {
        let was = self.capture.release();
        self.lift_keys();
        Reaction::capture(was)
    }

    /// The window is closing.
    fn window_closing(&mut self) {
        self.lift_keys();
        self.lift_buttons();
        self.capture.release();
    }
}

/// Calls the host's callback for `event`, if the host gave one.
fn call_host(callbacks: &WindowCallbacks, event: ToHost) {
    match event {
        ToHost::Pointer(p) => {
            if let Some(pointer) = &callbacks.pointer {
                pointer(p.at.0, p.at.1, p.mode.0, p.mode.1, p.buttons);
            }
        }
        ToHost::Wheel { dy, dx } => {
            if let Some(wheel) = &callbacks.wheel {
                wheel(dy, dx);
            }
        }
        ToHost::Scroll(event) => {
            if let Some(scroll) = &callbacks.scroll {
                scroll(event);
            }
        }
        ToHost::Key(key) => {
            if let Some(callback) = &callbacks.key {
                callback(key.usage, key.down);
            }
        }
    }
}

/// `event.type`.
fn event_type(event: &NSEvent) -> NSEventType {
    msg::get("type", || event.r#type())
}

/// `event.modifierFlags`, device-dependent bits included.
fn modifier_flags(event: &NSEvent) -> u64 {
    msg::get("modifierFlags", || event.modifierFlags()).0 as u64
}

/// `event.buttonNumber`'s `BUTTON_*` bit.
fn button_bit(event: &NSEvent) -> u32 {
    geometry::button_bit(msg::get("buttonNumber", || event.buttonNumber()) as i64)
}

/// A key event's values; `repeat` whether to read `isARepeat`, which only `keyDown:` and
/// `keyUp:` events answer.
fn key_input(event: &NSEvent, repeat: bool) -> KeyInput {
    KeyInput {
        keycode: msg::get("keyCode", || event.keyCode()),
        flags: modifier_flags(event),
        repeat: repeat && msg::get("isARepeat", || event.isARepeat()),
    }
}

/// A scroll event's values ([`ScrollInput`]).
fn scroll_input(event: &NSEvent) -> ScrollInput {
    let cg_event = msg::get("CGEvent", || event.CGEvent());
    let line = |axis: CGEventField| {
        cg_event
            .as_deref()
            .map_or(0, |e| CGEvent::integer_value_field(Some(e), axis))
    };
    ScrollInput {
        precise: msg::get("hasPreciseScrollingDeltas", || {
            event.hasPreciseScrollingDeltas()
        }),
        delta: (
            msg::get("scrollingDeltaX", || event.scrollingDeltaX()),
            msg::get("scrollingDeltaY", || event.scrollingDeltaY()),
        ),
        lines: (
            line(CGEventField::ScrollWheelEventDeltaAxis2),
            line(CGEventField::ScrollWheelEventDeltaAxis1),
        ),
        inverted: msg::get("isDirectionInvertedFromDevice", || {
            event.isDirectionInvertedFromDevice()
        }),
        phase: scroll_phase(msg::get("phase", || event.phase())),
        momentum: momentum_phase(msg::get("momentumPhase", || event.momentumPhase())),
    }
}

/// One window's input state.
#[derive(Debug)]
pub(crate) struct InputState {
    /// The window whose callbacks the input goes to.
    window: Arc<Window>,
    /// The decisions.
    logic: InputLogic,
    /// Whether this window holds the system's hot keys off: while captured.
    holds_hotkeys: bool,
}

impl InputState {
    /// The input state of `window`, made on the main thread.
    pub(crate) fn new(window: Arc<Window>) -> InputState {
        InputState {
            window,
            logic: InputLogic::new(crate::native::is_iso_keyboard()),
            holds_hotkeys: false,
        }
    }

    /// Asks HIToolbox whether the keyboard in use is ISO, before a key event is mapped: it
    /// follows the keyboard last typed on, so one plugged in after the window opened maps its
    /// own way.
    fn follow_keyboard(&mut self) {
        self.logic.iso = crate::native::is_iso_keyboard();
    }

    /// Sends what the event decided through the host's callbacks while the window lives
    /// ([`Window::send_each`]), each pointer event telling the window first
    /// ([`Window::pointer_sent`]), and has the system's hot keys follow capture.
    fn finish(&mut self, reaction: Reaction) -> Reaction {
        let window = &self.window;
        window.send_each(std::mem::take(&mut self.logic.out), |callbacks, event| {
            if matches!(event, ToHost::Pointer(_)) {
                window.pointer_sent();
            }
            call_host(callbacks, event);
        });
        let captured = self.logic.capture.is_captured();
        if captured != self.holds_hotkeys {
            self.holds_hotkeys = captured;
            hotkeys::set_system_hotkeys_disabled(captured);
        }
        reaction
    }

    /// A button went down (`mouseDown:`, `rightMouseDown:`, `otherMouseDown:`).
    pub(crate) fn mouse_down(&mut self, event: &NSEvent, cx: &ViewContext) -> Reaction {
        let reaction = self
            .logic
            .button_down(button_bit(event), modifier_flags(event), cx);
        self.finish(reaction)
    }

    /// A button went up (`mouseUp:`, `rightMouseUp:`, `otherMouseUp:`).
    pub(crate) fn mouse_up(&mut self, event: &NSEvent, cx: &ViewContext) -> Reaction {
        let reaction = self.logic.button_up(button_bit(event), cx);
        self.finish(reaction)
    }

    /// The pointer moved, with or without a button held (`mouseMoved:`, `*MouseDragged:`).
    pub(crate) fn mouse_moved(&mut self, _event: &NSEvent, cx: &ViewContext) -> Reaction {
        let reaction = self.logic.moved(cx);
        self.finish(reaction)
    }

    /// The wheel or a trackpad scrolled (`scrollWheel:`): smoothly where the host's callbacks
    /// take it.
    pub(crate) fn scroll_wheel(&mut self, event: &NSEvent, cx: &ViewContext) -> Reaction {
        let smooth = self
            .window
            .callbacks()
            .is_some_and(|callbacks| callbacks.scroll.is_some());
        let reaction = self.logic.scrolled(scroll_input(event), smooth, cx);
        self.finish(reaction)
    }

    /// `keyDown:`.
    pub(crate) fn key_down(&mut self, event: &NSEvent) -> Reaction {
        self.follow_keyboard();
        let reaction = self.logic.key_down(key_input(event, true));
        self.finish(reaction)
    }

    /// `keyUp:`.
    pub(crate) fn key_up(&mut self, event: &NSEvent) -> Reaction {
        self.follow_keyboard();
        let reaction = self.logic.key_up(key_input(event, true));
        self.finish(reaction)
    }

    /// `flagsChanged:`.
    pub(crate) fn flags_changed(&mut self, event: &NSEvent) -> Reaction {
        self.follow_keyboard();
        let reaction = self.logic.flags_changed(key_input(event, false));
        self.finish(reaction)
    }

    /// The application's `sendEvent:` saw `event` while this window is key, before AppKit
    /// dispatches it: true when the input took it, and AppKit must not see it (a ⌘ key-down
    /// while captured, a key-up under ⌘). Every other event is AppKit's to dispatch.
    pub(crate) fn app_send_event(&mut self, event: &NSEvent, _cx: &ViewContext) -> bool {
        let kind = event_type(event);
        let down = match kind {
            NSEventType::KeyDown => true,
            NSEventType::KeyUp => false,
            _ => return false,
        };
        if !app_takes(
            self.logic.capture.is_captured(),
            down,
            modifier_flags(event),
        ) {
            return false;
        }
        self.follow_keyboard();
        let taken = self.logic.app_key(down, key_input(event, true));
        self.finish(Reaction::default());
        taken
    }

    /// The window stopped being key: capture ends and every key and button held goes up.
    pub(crate) fn focus_lost(&mut self) -> Reaction {
        let reaction = self.logic.focus_lost();
        self.finish(reaction)
    }

    /// The person chose Release Input: capture ends and every key held goes up.
    pub(crate) fn release_capture(&mut self) -> Reaction {
        let reaction = self.logic.release_capture();
        self.finish(reaction)
    }

    /// The window is closing: every key and button held goes up, and capture ends.
    pub(crate) fn window_closing(&mut self) {
        self.logic.window_closing();
        self.finish(Reaction::default());
    }

    /// Whether input, the keyboard and the pointer, is captured.
    pub(crate) fn is_captured(&self) -> bool {
        self.logic.capture.is_captured()
    }

    /// Where the host's pointer is held over `image`, the image's rectangle in the view's points:
    /// the image while captured ([`Capture::pointer_confinement`]).
    pub(crate) fn pointer_confinement(&self, image: Rect) -> Option<Rect> {
        self.logic.capture.pointer_confinement(image)
    }

    /// Whether the person may resize the window: not while captured
    /// ([`Capture::window_resizable`]).
    pub(crate) fn window_resizable(&self) -> bool {
        self.logic.capture.window_resizable()
    }

    /// What the window's title ends with in the current capture state.
    pub(crate) fn title_suffix(&self) -> &'static str {
        self.logic.capture.title_suffix()
    }
}

impl Drop for InputState {
    /// A window gone while captured lets go of the system's hot keys, which would otherwise stay
    /// off for the process.
    fn drop(&mut self) {
        if self.holds_hotkeys {
            hotkeys::set_system_hotkeys_disabled(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::*;
    use crate::config::{BUTTON_LEFT, BUTTON_RIGHT};
    use crate::keys::{FLAG_CONTROL, FLAG_OPTION, FLAG_SHIFT};

    /// Left Shift, Control, Option and Command as `modifierFlags` carries them.
    const L_SHIFT: u64 = FLAG_SHIFT | 0x02;
    const L_CONTROL: u64 = FLAG_CONTROL | 0x01;
    const L_OPTION: u64 = FLAG_OPTION | 0x20;
    const L_COMMAND: u64 = FLAG_COMMAND | 0x08;

    /// Key codes: A, Q, the ISO section key, Fn and the left modifiers.
    const KVK_A: u16 = 0x00;
    const KVK_Q: u16 = 0x0C;
    const KVK_ISO_SECTION: u16 = 0x0A;
    const KVK_FN: u16 = 0x3F;
    const KVK_COMMAND: u16 = 0x37;
    const KVK_CONTROL: u16 = 0x3B;
    const KVK_OPTION: u16 = 0x3A;

    /// Usages: A, Q and the left modifiers.
    const A: u16 = 0x04;
    const Q: u16 = 0x14;
    const CONTROL: u16 = 0xE0;
    const SHIFT: u16 = 0xE1;
    const OPTION: u16 = 0xE2;
    const COMMAND: u16 = 0xE3;

    const MODE: (u16, u16) = (1920, 1200);
    /// The guest shown at half a point per pixel, 100 points in and 50 down.
    const IMAGE: Rect = Rect {
        x: 100.0,
        y: 50.0,
        width: 960.0,
        height: 600.0,
    };

    /// A key window's view with the pointer at `point`.
    fn at(point: (f64, f64)) -> ViewContext {
        ViewContext {
            point: Some(point),
            image: IMAGE,
            guest: Some(MODE),
            window_is_key: true,
        }
    }

    fn key(keycode: u16, flags: u64) -> KeyInput {
        KeyInput {
            keycode,
            flags,
            repeat: false,
        }
    }

    fn pointer(x: u16, y: u16, buttons: u32) -> ToHost {
        ToHost::Pointer(PointerAt {
            at: (x, y),
            mode: MODE,
            buttons,
        })
    }

    fn down(usage: u16) -> ToHost {
        ToHost::Key(KeyEvent::down(usage))
    }

    fn up(usage: u16) -> ToHost {
        ToHost::Key(KeyEvent::up(usage))
    }

    /// What the events so far sent, taken.
    fn sent(logic: &mut InputLogic) -> Vec<ToHost> {
        std::mem::take(&mut logic.out)
    }

    /// A logic whose keyboard is captured, with nothing sent yet.
    fn captured() -> InputLogic {
        let mut logic = InputLogic::new(false);
        logic.button_down(BUTTON_LEFT, 0, &at((500.0, 300.0)));
        logic.button_up(BUTTON_LEFT, &at((500.0, 300.0)));
        sent(&mut logic);
        assert!(logic.capture.is_captured());
        logic
    }

    /// A key goes up as the usage it went down as, though the keyboard's ISO answer changed
    /// between its down and its up (another keyboard typed on meanwhile): the guest is left
    /// holding no key
    #[test]
    fn a_key_goes_up_as_it_went_down_across_a_keyboard_change() {
        let on_iso = keymap::usb_usage(KVK_ISO_SECTION, true).unwrap();
        let on_ansi = keymap::usb_usage(KVK_ISO_SECTION, false).unwrap();
        assert_ne!(on_iso, on_ansi, "the ISO answer moves the key");
        for (at_down, at_up, usage) in [(true, false, on_iso), (false, true, on_ansi)] {
            let mut logic = captured();
            logic.iso = at_down;
            logic.key_down(key(KVK_ISO_SECTION, 0));
            logic.iso = at_up;
            logic.key_up(key(KVK_ISO_SECTION, 0));
            assert_eq!(sent(&mut logic), vec![down(usage), up(usage)]);
            logic.lift_keys();
            assert_eq!(sent(&mut logic), vec![], "a key left held");
            // A key down and up on one keyboard maps as that keyboard does.
            logic.key_down(key(KVK_ISO_SECTION, 0));
            logic.key_up(key(KVK_ISO_SECTION, 0));
            let now = keymap::usb_usage(KVK_ISO_SECTION, at_up).unwrap();
            assert_eq!(sent(&mut logic), vec![down(now), up(now)]);
        }
    }

    /// The click that makes the window key captures the keyboard, and is not sent, nor is its
    /// release
    #[test]
    fn the_click_that_makes_the_window_key_is_not_sent() {
        let mut logic = InputLogic::new(false);
        let cx = ViewContext {
            window_is_key: false,
            ..at((500.0, 300.0))
        };
        assert_eq!(
            logic.button_down(BUTTON_LEFT, 0, &cx),
            Reaction::capture(true)
        );
        assert!(logic.capture.is_captured());
        assert_eq!(
            logic.button_up(BUTTON_LEFT, &at((500.0, 300.0))),
            Reaction::default()
        );
        assert_eq!(sent(&mut logic), vec![]);
        // The next click is the guest's.
        assert_eq!(
            logic.button_down(BUTTON_LEFT, 0, &at((500.0, 300.0))),
            Reaction::capture(false)
        );
        assert_eq!(sent(&mut logic), vec![pointer(800, 500, BUTTON_LEFT)]);
    }

    /// A click in the image is sent with its button, and its release without; a click with
    /// another button held carries both
    #[test]
    fn a_click_in_the_image_is_sent_with_its_button() {
        let mut logic = InputLogic::new(false);
        logic.capture.click_in_image(true);
        logic.button_down(BUTTON_LEFT, 0, &at((100.0, 50.0)));
        logic.button_down(BUTTON_RIGHT, 0, &at((100.5, 50.5)));
        logic.button_up(BUTTON_LEFT, &at((101.0, 51.0)));
        logic.button_up(BUTTON_RIGHT, &at((101.0, 51.0)));
        assert_eq!(
            sent(&mut logic),
            vec![
                pointer(0, 0, BUTTON_LEFT),
                pointer(1, 1, BUTTON_LEFT | BUTTON_RIGHT),
                pointer(2, 2, BUTTON_RIGHT),
                pointer(2, 2, 0),
            ]
        );
    }

    /// A click of a button past the guest tablet's five (the host's sixth on) is not sent and
    /// does not capture, as on Linux; nor is its release
    #[test]
    fn a_button_past_the_fifth_starts_nothing() {
        let mut logic = InputLogic::new(false);
        let sixth = geometry::button_bit(5);
        assert_eq!(sixth, 0);
        assert_eq!(
            logic.button_down(sixth, L_COMMAND, &at((500.0, 300.0))),
            Reaction::default()
        );
        assert_eq!(
            logic.button_up(sixth, &at((500.0, 300.0))),
            Reaction::default()
        );
        assert!(!logic.capture.is_captured());
        assert_eq!(sent(&mut logic), vec![]);
        // Captured, it sends nothing either.
        let mut logic = captured();
        logic.button_down(sixth, 0, &at((500.0, 300.0)));
        logic.button_up(sixth, &at((500.0, 300.0)));
        assert_eq!(sent(&mut logic), vec![]);
    }

    /// A click outside the image (on the bars) is not sent and does not capture
    #[test]
    fn a_click_outside_the_image_starts_nothing() {
        let mut logic = InputLogic::new(false);
        assert_eq!(
            logic.button_down(BUTTON_LEFT, 0, &at((50.0, 300.0))),
            Reaction::default()
        );
        assert_eq!(
            logic.button_up(BUTTON_LEFT, &at((50.0, 300.0))),
            Reaction::default()
        );
        assert!(!logic.capture.is_captured());
        assert_eq!(sent(&mut logic), vec![]);
    }

    /// Before the guest has a mode, or with a mode of no pixels, the pointer sends nothing
    #[test]
    fn before_the_guest_has_a_mode_the_pointer_sends_nothing() {
        for guest in [None, Some((0, 768)), Some((1024, 0))] {
            let mut logic = InputLogic::new(false);
            let cx = ViewContext {
                guest,
                ..at((500.0, 300.0))
            };
            assert_eq!(logic.button_down(BUTTON_LEFT, 0, &cx), Reaction::default());
            logic.moved(&cx);
            for smooth in [false, true] {
                logic.scrolled(
                    ScrollInput {
                        delta: (0.0, 1.0),
                        lines: (0, 1),
                        ..ScrollInput::default()
                    },
                    smooth,
                    &cx,
                );
            }
            logic.button_up(BUTTON_LEFT, &cx);
            assert!(!logic.capture.is_captured());
            assert_eq!(sent(&mut logic), vec![], "{guest:?}");
        }
    }

    /// A drag keeps going outside the window, held at the image's edge, and its release there
    /// ends it in the guest
    #[test]
    fn a_drag_keeps_going_outside_the_window() {
        let mut logic = captured();
        logic.button_down(BUTTON_LEFT, 0, &at((580.0, 350.0)));
        logic.moved(&at((5000.0, 350.0)));
        logic.moved(&at((5000.0, -40.0)));
        logic.button_up(BUTTON_LEFT, &at((6000.0, -80.0)));
        assert_eq!(
            sent(&mut logic),
            vec![
                pointer(960, 600, BUTTON_LEFT),
                pointer(1919, 600, BUTTON_LEFT),
                pointer(1919, 0, BUTTON_LEFT),
                pointer(1919, 0, 0),
            ]
        );
    }

    /// Moves are sent where the guest pixel changes, over the bars too, held at the edge
    #[test]
    fn moves_are_sent_where_the_guest_pixel_changes() {
        let mut logic = InputLogic::new(false);
        logic.moved(&at((580.0, 350.0)));
        logic.moved(&at((580.2, 350.2)));
        logic.moved(&at((580.5, 350.0)));
        logic.moved(&at((20.0, 350.0)));
        assert_eq!(
            sent(&mut logic),
            vec![
                pointer(960, 600, 0),
                pointer(961, 600, 0),
                pointer(0, 600, 0)
            ]
        );
    }

    /// Scrolling over the image sends its notches after the pointer's position; outside it,
    /// nothing
    #[test]
    fn scrolling_over_the_image_sends_notches() {
        let wheel = ScrollInput {
            lines: (0, 2),
            ..ScrollInput::default()
        };
        let mut logic = InputLogic::new(false);
        logic.scrolled(wheel, false, &at((580.0, 350.0)));
        logic.scrolled(wheel, false, &at((580.0, 350.0)));
        logic.scrolled(wheel, false, &at((20.0, 350.0)));
        logic.scrolled(ScrollInput::default(), false, &at((600.0, 350.0)));
        assert_eq!(
            sent(&mut logic),
            vec![
                pointer(960, 600, 0),
                ToHost::Wheel { dy: 2, dx: 0 },
                ToHost::Wheel { dy: 2, dx: 0 },
            ]
        );
    }

    /// A gesture begins at its phase's Began or its momentum's
    #[test]
    fn a_gesture_begins_at_either_phase() {
        let input = |phase, momentum| ScrollInput {
            phase: scroll_phase(phase),
            momentum: momentum_phase(momentum),
            ..ScrollInput::default()
        };
        assert!(input(NSEventPhase::Began, NSEventPhase::None).began());
        assert!(input(NSEventPhase::None, NSEventPhase::Began).began());
        assert!(!input(NSEventPhase::Changed, NSEventPhase::None).began());
        assert!(!input(NSEventPhase::None, NSEventPhase::Changed).began());
    }

    /// AppKit's phases are the guest's: Stationary is Changed, and a cancelled momentum Ended
    #[test]
    fn appkit_phases_are_the_guests() {
        use MomentumPhase as M;
        use NSEventPhase as N;
        use ScrollPhase as P;
        let phases = [
            (N::None, P::None, M::None),
            (N::MayBegin, P::MayBegin, M::None),
            (N::Began, P::Began, M::Began),
            (N::Changed, P::Changed, M::Changed),
            (N::Stationary, P::Changed, M::Changed),
            (N::Ended, P::Ended, M::Ended),
            (N::Cancelled, P::Cancelled, M::Ended),
        ];
        for (appkit, phase, momentum) in phases {
            assert_eq!(scroll_phase(appkit), phase, "{appkit:?}");
            assert_eq!(momentum_phase(appkit), momentum, "{appkit:?}");
        }
    }

    /// A precise scroll event, `dy` points down as AppKit reports it (natural scrolling off),
    /// with its phases
    fn swipe(dy: f64, phase: ScrollPhase, momentum: MomentumPhase) -> ScrollInput {
        ScrollInput {
            precise: true,
            delta: (0.0, dy),
            phase,
            momentum,
            ..ScrollInput::default()
        }
    }

    /// Smooth scrolling sends each event of a gesture begun over the image with its phases,
    /// after the pointer's position, to the gesture's end wherever the pointer goes
    #[test]
    fn smooth_scrolling_sends_the_gesture_to_its_end() {
        use MomentumPhase as M;
        use ScrollPhase as P;
        let mut logic = InputLogic::new(false);
        let over = at((580.0, 350.0));
        let off = at((20.0, 350.0));
        logic.scrolled(swipe(0.0, P::MayBegin, M::None), true, &over);
        logic.scrolled(swipe(6.0, P::Began, M::None), true, &over);
        logic.scrolled(swipe(2.5, P::Changed, M::None), true, &off);
        logic.scrolled(swipe(0.0, P::Ended, M::None), true, &off);
        logic.scrolled(swipe(4.0, P::None, M::Began), true, &off);
        logic.scrolled(swipe(0.0, P::None, M::Ended), true, &over);
        let scroll = |dy, phase, momentum, notches| {
            ToHost::Scroll(ScrollEvent {
                dy,
                dx: 0,
                phase,
                momentum,
                notches: (notches, 0),
            })
        };
        assert_eq!(
            sent(&mut logic),
            vec![
                pointer(960, 600, 0),
                scroll(0, P::MayBegin, M::None, 0),
                scroll(1536, P::Began, M::None, 0),
                scroll(640, P::Changed, M::None, 0),
                scroll(0, P::Ended, M::None, 0),
                scroll(1024, P::None, M::Began, 0),
                scroll(0, P::None, M::Ended, 0),
            ]
        );
        // Its notches count as the wheel's do: one per 40 points, afresh at each Began.
        for (dy, phase) in [(30.0, P::Began), (30.0, P::Changed), (90.0, P::Changed)] {
            logic.scrolled(swipe(dy, phase, M::None), true, &over);
        }
        logic.scrolled(swipe(0.0, P::Ended, M::None), true, &over);
        assert_eq!(
            sent(&mut logic),
            vec![
                scroll(7680, P::Began, M::None, 0),
                scroll(7680, P::Changed, M::None, 1),
                scroll(23040, P::Changed, M::None, 2),
                scroll(0, P::Ended, M::None, 0),
            ]
        );
        // A gesture begun off the image is not sent, wherever it goes; a wheel's lines over the
        // image are, at 10 points each, and one of no distance is not.
        logic.scrolled(swipe(0.0, P::MayBegin, M::None), true, &off);
        logic.scrolled(swipe(6.0, P::Began, M::None), true, &off);
        logic.scrolled(swipe(2.0, P::Changed, M::None), true, &over);
        logic.scrolled(swipe(0.0, P::Ended, M::None), true, &over);
        let wheel = ScrollInput {
            delta: (0.0, 1.5),
            lines: (0, 1),
            ..ScrollInput::default()
        };
        logic.scrolled(wheel, true, &over);
        logic.scrolled(ScrollInput::default(), true, &over);
        logic.scrolled(wheel, true, &off);
        assert_eq!(sent(&mut logic), vec![scroll(3840, P::None, M::None, 1)]);
    }

    /// The host's repeats are dropped at the view: a held key is one down
    #[test]
    fn host_repeats_are_dropped_at_the_view() {
        let mut logic = InputLogic::new(false);
        logic.key_down(key(KVK_A, 0));
        for _ in 0..3 {
            logic.key_down(KeyInput {
                repeat: true,
                ..key(KVK_A, 0)
            });
        }
        logic.key_up(key(KVK_A, 0));
        assert_eq!(sent(&mut logic), vec![down(A), up(A)]);
    }

    /// Keys with no usage (Fn) are dropped
    #[test]
    fn unmapped_keys_are_dropped() {
        let mut logic = InputLogic::new(false);
        assert_eq!(logic.key_down(key(KVK_FN, 0)), Reaction::default());
        logic.key_up(key(KVK_FN, 0));
        assert_eq!(sent(&mut logic), vec![]);
    }

    /// A modifier the window did not see go down reaches the guest before the key it came with
    #[test]
    fn a_missed_modifier_is_made_good_before_the_key() {
        let mut logic = InputLogic::new(false);
        logic.key_down(key(KVK_A, L_SHIFT));
        logic.key_up(key(KVK_A, 0));
        assert_eq!(
            sent(&mut logic),
            vec![down(SHIFT), down(A), up(SHIFT), up(A)]
        );
    }

    /// A modifier held down before the window was key reaches the guest before the click it
    /// came with; a click not sent brings nothing
    #[test]
    fn a_missed_modifier_is_made_good_before_the_click() {
        let mut logic = InputLogic::new(false);
        let cx = ViewContext {
            window_is_key: false,
            ..at((100.0, 50.0))
        };
        logic.button_down(BUTTON_LEFT, L_COMMAND, &cx);
        assert_eq!(sent(&mut logic), vec![]);
        logic.button_down(BUTTON_LEFT, L_COMMAND, &at((100.0, 50.0)));
        assert_eq!(
            sent(&mut logic),
            vec![down(COMMAND), pointer(0, 0, BUTTON_LEFT)]
        );
    }

    /// Uncaptured, a key with ⌘ is the menu's: passed on, not sent (⌘ itself is); other keys
    /// are the guest's, and AppKit keeps the ⌘ key-down
    #[test]
    fn uncaptured_command_keys_go_to_the_menu() {
        let mut logic = InputLogic::new(false);
        logic.flags_changed(key(KVK_COMMAND, L_COMMAND));
        assert!(!logic.app_key(true, key(KVK_Q, L_COMMAND)));
        assert_eq!(
            logic.key_down(key(KVK_Q, L_COMMAND)),
            Reaction {
                capture_changed: false,
                pass_on: true
            }
        );
        logic.flags_changed(key(KVK_COMMAND, 0));
        assert!(!logic.app_key(true, key(KVK_A, 0)));
        assert_eq!(logic.key_down(key(KVK_A, 0)), Reaction::default());
        assert_eq!(sent(&mut logic), vec![down(COMMAND), up(COMMAND), down(A)]);
    }

    /// Captured, a ⌘ key-down is taken from AppKit before the menu sees it, and reaches the
    /// guest
    #[test]
    fn captured_command_keys_reach_the_guest_not_the_menu() {
        let mut logic = captured();
        logic.flags_changed(key(KVK_COMMAND, L_COMMAND));
        assert!(logic.app_key(true, key(KVK_Q, L_COMMAND)));
        // Its repeats are taken too, and dropped.
        assert!(logic.app_key(
            true,
            KeyInput {
                repeat: true,
                ..key(KVK_Q, L_COMMAND)
            }
        ));
        assert!(logic.app_key(false, key(KVK_Q, L_COMMAND)));
        // A key without ⌘ is AppKit's to hand the view.
        assert!(!logic.app_key(true, key(KVK_A, 0)));
        assert!(!logic.app_key(false, key(KVK_A, 0)));
        assert_eq!(sent(&mut logic), vec![down(COMMAND), down(Q), up(Q)]);
    }

    /// A key-up under ⌘, which AppKit never hands the view, is taken and reaches the guest,
    /// captured or not
    #[test]
    fn key_ups_under_command_reach_the_guest() {
        assert!(app_takes(false, false, L_COMMAND));
        assert!(app_takes(true, false, L_COMMAND));
        assert!(app_takes(true, true, L_COMMAND));
        assert!(!app_takes(false, true, L_COMMAND));
        assert!(!app_takes(true, false, L_SHIFT));
        assert!(!app_takes(true, true, 0));
        let mut logic = InputLogic::new(false);
        logic.key_down(key(KVK_A, 0));
        logic.flags_changed(key(KVK_COMMAND, L_COMMAND));
        assert!(logic.app_key(false, key(KVK_A, L_COMMAND)));
        // The up of a key the menu had is taken, and sends nothing.
        assert!(logic.app_key(false, key(KVK_Q, L_COMMAND)));
        assert_eq!(sent(&mut logic), vec![down(A), down(COMMAND), up(A)]);
    }

    /// ⌃⌥⌘ while captured lifts every key held and releases capture
    #[test]
    fn the_release_chord_lifts_every_key_and_releases() {
        let mut logic = captured();
        logic.key_down(key(KVK_A, 0));
        logic.flags_changed(key(KVK_CONTROL, L_CONTROL));
        logic.flags_changed(key(KVK_OPTION, L_CONTROL | L_OPTION));
        assert_eq!(sent(&mut logic), vec![down(A), down(CONTROL), down(OPTION)]);
        assert_eq!(
            logic.flags_changed(key(KVK_COMMAND, L_CONTROL | L_OPTION | L_COMMAND)),
            Reaction::capture(true)
        );
        assert!(!logic.capture.is_captured());
        assert_eq!(
            sent(&mut logic),
            vec![down(COMMAND), up(COMMAND), up(OPTION), up(CONTROL), up(A)]
        );
        // The modifiers still held are the guest's again at their next change, and uncaptured
        // the chord is only modifiers.
        assert_eq!(
            logic.flags_changed(key(KVK_COMMAND, L_CONTROL | L_OPTION)),
            Reaction::default()
        );
        assert_eq!(
            logic.flags_changed(key(KVK_COMMAND, L_CONTROL | L_OPTION | L_COMMAND)),
            Reaction::default()
        );
        assert_eq!(
            sent(&mut logic),
            vec![down(CONTROL), down(OPTION), down(COMMAND)]
        );
    }

    /// Losing the keyboard releases capture, and lifts every key and button held
    #[test]
    fn losing_focus_lifts_every_key_and_releases() {
        let mut logic = captured();
        logic.button_down(BUTTON_LEFT, 0, &at((100.0, 50.0)));
        logic.flags_changed(key(KVK_CONTROL, L_CONTROL));
        logic.key_down(key(KVK_A, L_CONTROL));
        sent(&mut logic);
        assert_eq!(logic.focus_lost(), Reaction::capture(true));
        assert!(!logic.capture.is_captured());
        assert_eq!(sent(&mut logic), vec![up(A), up(CONTROL), pointer(0, 0, 0)]);
        // The button's release that follows sends nothing, and a second loss changes nothing.
        logic.button_up(BUTTON_LEFT, &at((100.0, 50.0)));
        assert_eq!(logic.focus_lost(), Reaction::capture(false));
        assert_eq!(sent(&mut logic), vec![]);
    }

    /// Release Input releases capture and lifts every key held
    #[test]
    fn releasing_capture_lifts_every_key() {
        let mut logic = captured();
        logic.key_down(key(KVK_A, 0));
        logic.key_down(key(KVK_Q, 0));
        sent(&mut logic);
        assert_eq!(logic.release_capture(), Reaction::capture(true));
        assert_eq!(sent(&mut logic), vec![up(Q), up(A)]);
        assert_eq!(logic.release_capture(), Reaction::capture(false));
    }

    /// Closing lifts every key and button held, and releases capture
    #[test]
    fn closing_lifts_every_key() {
        let mut logic = captured();
        logic.button_down(BUTTON_RIGHT, 0, &at((100.0, 50.0)));
        logic.key_down(key(KVK_A, 0));
        sent(&mut logic);
        logic.window_closing();
        assert!(!logic.capture.is_captured());
        assert_eq!(sent(&mut logic), vec![up(A), pointer(0, 0, 0)]);
    }

    /// The window's end, while the host's callback for the first key-up of a lift runs, drops
    /// the other key-ups: no callback of the host's starts after the end has returned
    #[test]
    fn no_callback_starts_after_the_window_s_end() {
        let ups = Arc::new(Mutex::new(Vec::new()));
        let this: Arc<Mutex<Option<std::sync::Weak<Window>>>> = Arc::default();
        let (seen, ends) = (Arc::clone(&ups), Arc::clone(&this));
        let config = crate::config::WindowConfig {
            callbacks: WindowCallbacks {
                key: Some(Arc::new(move |usage, down| {
                    seen.lock().push((usage, down));
                    // The display's release, on another thread, returns meanwhile.
                    if let Some(window) = ends.lock().as_ref().and_then(std::sync::Weak::upgrade) {
                        window.close();
                    }
                })),
                ..WindowCallbacks::default()
            },
            ..crate::config::WindowConfig::default()
        };
        let window = Window::new(config, 0, "input".to_owned());
        *this.lock() = Some(Arc::downgrade(&window));
        let mut input = InputState::new(Arc::clone(&window));
        input.logic.key_down(key(KVK_A, 0));
        input.logic.key_down(key(KVK_Q, 0));
        sent(&mut input.logic);
        input.focus_lost();
        assert!(window.is_closed());
        assert_eq!(*ups.lock(), [(Q, false)]);
        // After the end, nothing is sent.
        input.logic.key_down(key(KVK_A, 0));
        input.focus_lost();
        assert_eq!(*ups.lock(), [(Q, false)]);
    }

    /// Each event goes to its callback, and a callback the host did not give takes nothing
    #[test]
    fn each_event_goes_to_its_callback() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (p, w, k) = (Arc::clone(&log), Arc::clone(&log), Arc::clone(&log));
        let s = Arc::clone(&log);
        let callbacks = WindowCallbacks {
            pointer: Some(Arc::new(move |x, y, width, height, buttons| {
                p.lock()
                    .push(format!("pointer {x} {y} {width} {height} {buttons}"));
            })),
            wheel: Some(Arc::new(move |dy, dx| {
                w.lock().push(format!("wheel {dy} {dx}"))
            })),
            key: Some(Arc::new(move |usage, down| {
                k.lock().push(format!("key {usage:#x} {down}"));
            })),
            close: None,
            scroll: Some(Arc::new(move |e: ScrollEvent| {
                s.lock().push(format!(
                    "scroll {} {} {:?} {:?} {:?}",
                    e.dy, e.dx, e.phase, e.momentum, e.notches
                ));
            })),
        };
        call_host(&callbacks, pointer(3, 4, BUTTON_RIGHT));
        call_host(&callbacks, ToHost::Wheel { dy: -1, dx: 2 });
        call_host(
            &callbacks,
            ToHost::Scroll(ScrollEvent {
                dy: 256,
                dx: -3,
                phase: ScrollPhase::Changed,
                momentum: MomentumPhase::None,
                notches: (1, 0),
            }),
        );
        call_host(&callbacks, down(A));
        call_host(&WindowCallbacks::default(), up(A));
        assert_eq!(
            *log.lock(),
            vec![
                "pointer 3 4 1920 1200 2",
                "wheel -1 2",
                "scroll 256 -3 Changed None (1, 0)",
                "key 0x4 true"
            ]
        );
    }
}
