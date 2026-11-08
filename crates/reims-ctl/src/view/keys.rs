//! The terminal's keys, pasted text and mouse as QMP's `input-send-event` gives them to the
//! guest: a US keyboard's keys by QEMU's qcodes, and the USB tablet's absolute position and
//! buttons.
//!
//! A terminal reports a key press as the text it types, not as a key going down and up, so each
//! press is a tap: its modifiers down, the key down and up, its modifiers up. A character no key
//! of a US keyboard types (é, ü) has no tap.
//!
//! Most terminals cannot send Command, so the terminal's Alt (and Super, where the terminal
//! reports it) is the guest's Command; the viewer's toolbar holds Option, Control, Command or
//! Shift for the next key or click.

use super::encode::Place;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton};
use serde_json::{Value, json};

/// Modifiers held around a key or a click.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods(u8);

impl Mods {
    pub const NONE: Mods = Mods(0);
    pub const CMD: Mods = Mods(1);
    pub const OPT: Mods = Mods(2);
    pub const CTRL: Mods = Mods(4);
    pub const SHIFT: Mods = Mods(8);
    /// Each modifier, with its name on the toolbar and its qcode, in the order they go down.
    pub const EACH: [(Mods, &'static str, &'static str); 4] = [
        (Mods::CMD, "Cmd", "meta_l"),
        (Mods::OPT, "Opt", "alt"),
        (Mods::CTRL, "Ctrl", "ctrl"),
        (Mods::SHIFT, "Shift", "shift"),
    ];

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, m: Mods) -> bool {
        self.0 & m.0 == m.0
    }

    pub fn toggle(&mut self, m: Mods) {
        self.0 ^= m.0;
    }

    /// The modifiers a terminal reports with a key or a click: Alt, Super and Meta are Command.
    pub fn from_terminal(m: KeyModifiers) -> Mods {
        let mut out = Mods::NONE;
        if m.contains(KeyModifiers::CONTROL) {
            out = out | Mods::CTRL;
        }
        if m.intersects(KeyModifiers::ALT | KeyModifiers::SUPER | KeyModifiers::META) {
            out = out | Mods::CMD;
        }
        if m.contains(KeyModifiers::SHIFT) {
            out = out | Mods::SHIFT;
        }
        out
    }

    fn qcodes(self) -> impl DoubleEndedIterator<Item = &'static str> {
        Mods::EACH
            .into_iter()
            .filter(move |(m, _, _)| self.contains(*m))
            .map(|(_, _, q)| q)
    }
}

impl std::ops::BitOr for Mods {
    type Output = Mods;
    fn bitor(self, o: Mods) -> Mods {
        Mods(self.0 | o.0)
    }
}

/// One input event for `input-send-event`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ev {
    /// A key, by its qcode, going down (true) or up.
    Key(&'static str, bool),
    /// A mouse button, by QMP's name for it, going down or up.
    Btn(&'static str, bool),
    /// The tablet's position, each axis 0 to 0x7fff across the display.
    Abs(u16, u16),
}

impl Ev {
    /// The slots of QEMU's USB keyboard queue the event takes: a key's scancodes, two for a key
    /// behind the 0xe0 prefix (hw/input/hid.c, `hid_keyboard_event`), which the guest takes one
    /// a poll.
    pub fn slots(self) -> u32 {
        match self {
            Ev::Key(
                "meta_l" | "left" | "right" | "up" | "down" | "home" | "end" | "pgup" | "pgdn"
                | "insert" | "delete",
                _,
            ) => 2,
            Ev::Key(..) => 1,
            _ => 0,
        }
    }

    /// A button going down or up, but for the wheel's, which only move.
    pub fn is_click(self) -> bool {
        matches!(self, Ev::Btn(b, _) if !b.starts_with("wheel"))
    }

    /// The event as `input-send-event` takes it; a position is two, one for each axis.
    pub fn json(self, out: &mut Vec<Value>) {
        match self {
            Ev::Key(k, down) => out.push(json!({
                "type": "key",
                "data": { "down": down, "key": { "type": "qcode", "data": k } }
            })),
            Ev::Btn(b, down) => out.push(json!({
                "type": "btn",
                "data": { "down": down, "button": b }
            })),
            Ev::Abs(x, y) => {
                out.push(json!({ "type": "abs", "data": { "axis": "x", "value": x } }));
                out.push(json!({ "type": "abs", "data": { "axis": "y", "value": y } }));
            }
        }
    }
}

/// The events of a tap of `key` with `mods` held.
pub fn tap(mods: Mods, key: &'static str) -> Vec<Ev> {
    let mut v: Vec<Ev> = mods.qcodes().map(|q| Ev::Key(q, true)).collect();
    v.push(Ev::Key(key, true));
    v.push(Ev::Key(key, false));
    v.extend(mods.qcodes().rev().map(|q| Ev::Key(q, false)));
    v
}

/// `mods` going down before a click, or up after it.
pub fn hold(mods: Mods, down: bool) -> Vec<Ev> {
    let v = mods.qcodes().map(|q| Ev::Key(q, down));
    if down { v.collect() } else { v.rev().collect() }
}

const LETTERS: [&str; 26] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];
const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const F_KEYS: [&str; 12] = [
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
];

/// The key of a US keyboard that types `c`, and whether it takes Shift.
pub fn char_key(c: char) -> Option<(&'static str, bool)> {
    Some(match c {
        'a'..='z' => (LETTERS[c as usize - 'a' as usize], false),
        'A'..='Z' => (LETTERS[c as usize - 'A' as usize], true),
        '0'..='9' => (DIGITS[c as usize - '0' as usize], false),
        ' ' => ("spc", false),
        '\n' => ("ret", false),
        '\t' => ("tab", false),
        '!' => ("1", true),
        '@' => ("2", true),
        '#' => ("3", true),
        '$' => ("4", true),
        '%' => ("5", true),
        '^' => ("6", true),
        '&' => ("7", true),
        '*' => ("8", true),
        '(' => ("9", true),
        ')' => ("0", true),
        '-' => ("minus", false),
        '_' => ("minus", true),
        '=' => ("equal", false),
        '+' => ("equal", true),
        '[' => ("bracket_left", false),
        '{' => ("bracket_left", true),
        ']' => ("bracket_right", false),
        '}' => ("bracket_right", true),
        '\\' => ("backslash", false),
        '|' => ("backslash", true),
        ';' => ("semicolon", false),
        ':' => ("semicolon", true),
        '\'' => ("apostrophe", false),
        '"' => ("apostrophe", true),
        '`' => ("grave_accent", false),
        '~' => ("grave_accent", true),
        ',' => ("comma", false),
        '<' => ("comma", true),
        '.' => ("dot", false),
        '>' => ("dot", true),
        '/' => ("slash", false),
        '?' => ("slash", true),
        _ => return None,
    })
}

/// What a terminal key press sends the guest.
#[derive(Debug, PartialEq, Eq)]
pub enum Key {
    Tap(Vec<Ev>),
    /// A character no US key types.
    Untypable(char),
    /// A key the guest is not sent (Caps Lock, media keys, a modifier alone).
    Ignored,
}

/// The tap a terminal key press is, with `extra` held besides its own modifiers.
pub fn key(k: &KeyEvent, extra: Mods) -> Key {
    let mut mods = Mods::from_terminal(k.modifiers) | extra;
    let q = match k.code {
        KeyCode::Char(c) => match char_key(c) {
            Some((q, shift)) => {
                if shift {
                    mods = mods | Mods::SHIFT;
                }
                q
            }
            None => return Key::Untypable(c),
        },
        KeyCode::Enter => "ret",
        KeyCode::Tab => "tab",
        KeyCode::BackTab => {
            mods = mods | Mods::SHIFT;
            "tab"
        }
        KeyCode::Backspace => "backspace",
        KeyCode::Esc => "esc",
        KeyCode::Left => "left",
        KeyCode::Right => "right",
        KeyCode::Up => "up",
        KeyCode::Down => "down",
        KeyCode::Home => "home",
        KeyCode::End => "end",
        KeyCode::PageUp => "pgup",
        KeyCode::PageDown => "pgdn",
        KeyCode::Insert => "insert",
        KeyCode::Delete => "delete",
        KeyCode::F(n @ 1..=12) => F_KEYS[n as usize - 1],
        _ => return Key::Ignored,
    };
    Key::Tap(tap(mods, q))
}

/// Whether a key press is the viewer's own disconnect, Control-]. Without the kitty keyboard
/// protocol a terminal sends it as the byte 0x1d, which crossterm reports as Control-5.
pub fn is_disconnect(k: &KeyEvent, enhanced: bool) -> bool {
    k.modifiers.contains(KeyModifiers::CONTROL)
        && (k.code == KeyCode::Char(']') || (!enhanced && k.code == KeyCode::Char('5')))
}

/// Pasted text as taps, a character each, and the characters no US key types.
pub fn paste(s: &str) -> (Vec<Vec<Ev>>, Vec<char>) {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut taps = Vec::new();
    let mut untypable = Vec::new();
    for c in s.chars() {
        match char_key(c) {
            Some((q, shift)) => taps.push(tap(if shift { Mods::SHIFT } else { Mods::NONE }, q)),
            None => untypable.push(c),
        }
    }
    (taps, untypable)
}

pub fn button(b: MouseButton) -> &'static str {
    match b {
        MouseButton::Left => "left",
        MouseButton::Right => "right",
        MouseButton::Middle => "middle",
    }
}

/// Where a mouse report at `x`, `y` falls on the frame placed at `p`: the fraction of the way
/// across and down, and whether it is on the frame. The report is in cells, or in pixels when
/// `pixel` gives the cells' size (SGR-Pixels, mode 1016); a cell's report is its middle.
pub fn point(x: u16, y: u16, pixel: Option<(u16, u16)>, p: Place) -> (f64, f64, bool) {
    let (fx, fy) = match pixel {
        Some((cw, ch)) => (
            (f64::from(x) - f64::from(p.col) * f64::from(cw)) / (f64::from(p.cols) * f64::from(cw)),
            (f64::from(y) - f64::from(p.row) * f64::from(ch)) / (f64::from(p.rows) * f64::from(ch)),
        ),
        None => (
            (f64::from(x) - f64::from(p.col) + 0.5) / f64::from(p.cols),
            (f64::from(y) - f64::from(p.row) + 0.5) / f64::from(p.rows),
        ),
    };
    let on = (0.0..1.0).contains(&fx) && (0.0..1.0).contains(&fy);
    (fx, fy, on)
}

/// The tablet's position at a fraction of the way across and down the display: the pixel there
/// times 0x7fff over the display's size, as QEMU's own windows send it.
pub fn abs(fx: f64, fy: f64) -> Ev {
    let axis = |f: f64| (f.clamp(0.0, 1.0) * 32767.0).round() as u16;
    Ev::Abs(axis(fx), axis(fy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode, m: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, m)
    }

    #[test]
    fn a_letter_taps_its_key_and_a_capital_holds_shift() {
        assert_eq!(
            key(&press(KeyCode::Char('a'), KeyModifiers::NONE), Mods::NONE),
            Key::Tap(vec![Ev::Key("a", true), Ev::Key("a", false)])
        );
        assert_eq!(
            key(&press(KeyCode::Char('A'), KeyModifiers::SHIFT), Mods::NONE),
            Key::Tap(vec![
                Ev::Key("shift", true),
                Ev::Key("a", true),
                Ev::Key("a", false),
                Ev::Key("shift", false)
            ])
        );
        // A shifted symbol needs Shift whether or not the terminal reports it.
        assert_eq!(
            key(&press(KeyCode::Char('?'), KeyModifiers::NONE), Mods::NONE),
            Key::Tap(tap(Mods::SHIFT, "slash"))
        );
    }

    #[test]
    fn alt_is_command_and_the_toolbar_adds_its_modifiers() {
        assert_eq!(
            key(&press(KeyCode::Char(' '), KeyModifiers::ALT), Mods::NONE),
            Key::Tap(vec![
                Ev::Key("meta_l", true),
                Ev::Key("spc", true),
                Ev::Key("spc", false),
                Ev::Key("meta_l", false)
            ])
        );
        assert_eq!(
            key(
                &press(KeyCode::Esc, KeyModifiers::NONE),
                Mods::CMD | Mods::OPT
            ),
            Key::Tap(vec![
                Ev::Key("meta_l", true),
                Ev::Key("alt", true),
                Ev::Key("esc", true),
                Ev::Key("esc", false),
                Ev::Key("alt", false),
                Ev::Key("meta_l", false)
            ])
        );
        assert_eq!(
            key(
                &press(KeyCode::Char('c'), KeyModifiers::CONTROL),
                Mods::NONE
            ),
            Key::Tap(tap(Mods::CTRL, "c"))
        );
    }

    #[test]
    fn keys_without_a_us_key_are_reported_or_ignored() {
        assert_eq!(
            key(&press(KeyCode::Char('é'), KeyModifiers::NONE), Mods::NONE),
            Key::Untypable('é')
        );
        assert_eq!(
            key(&press(KeyCode::CapsLock, KeyModifiers::NONE), Mods::NONE),
            Key::Ignored
        );
        assert_eq!(
            key(&press(KeyCode::BackTab, KeyModifiers::SHIFT), Mods::NONE),
            Key::Tap(tap(Mods::SHIFT, "tab"))
        );
        assert_eq!(
            key(&press(KeyCode::F(12), KeyModifiers::NONE), Mods::NONE),
            Key::Tap(tap(Mods::NONE, "f12"))
        );
    }

    #[test]
    fn control_bracket_disconnects() {
        let ctrl = KeyModifiers::CONTROL;
        assert!(is_disconnect(&press(KeyCode::Char(']'), ctrl), true));
        assert!(is_disconnect(&press(KeyCode::Char('5'), ctrl), false));
        // With the kitty keyboard protocol Control-5 is Control-5.
        assert!(!is_disconnect(&press(KeyCode::Char('5'), ctrl), true));
        assert!(!is_disconnect(
            &press(KeyCode::Char(']'), KeyModifiers::NONE),
            false
        ));
    }

    #[test]
    fn paste_types_each_character_and_lists_the_rest() {
        let (taps, bad) = paste("sudo ls\r\nÅ!");
        assert_eq!(taps.len(), 9);
        assert_eq!(taps[4], tap(Mods::NONE, "spc"));
        assert_eq!(taps[7], tap(Mods::NONE, "ret"));
        assert_eq!(taps[8], tap(Mods::SHIFT, "1"));
        assert_eq!(bad, vec!['Å']);
    }

    #[test]
    fn every_printable_ascii_character_has_a_key() {
        for c in ' '..='~' {
            assert!(char_key(c).is_some(), "{c:?}");
        }
    }

    #[test]
    fn a_point_maps_cells_and_pixels_onto_the_frame() {
        let p = Place {
            col: 10,
            row: 2,
            cols: 100,
            rows: 50,
        };
        let (fx, fy, on) = point(10, 2, None, p);
        assert!(on);
        assert!((fx - 0.005).abs() < 1e-9 && (fy - 0.01).abs() < 1e-9);
        assert!(!point(9, 20, None, p).2);
        assert!(!point(110, 20, None, p).2);
        // In pixels, with 8x16 cells: the frame starts at 80, 32 and is 800x800.
        let (fx, fy, on) = point(480, 432, Some((8, 16)), p);
        assert!(on);
        assert!((fx - 0.5).abs() < 1e-9 && (fy - 0.5).abs() < 1e-9);
        assert_eq!(abs(0.5, 1.5), Ev::Abs(16384, 32767));
        assert_eq!(abs(-1.0, 0.0), Ev::Abs(0, 0));
    }

    #[test]
    fn keys_behind_the_e0_prefix_take_two_slots() {
        let slots = |evs: Vec<Ev>| evs.iter().map(|e| e.slots()).sum::<u32>();
        assert_eq!(slots(tap(Mods::NONE, "a")), 2);
        assert_eq!(slots(tap(Mods::CMD, "z")), 6);
        assert_eq!(slots(tap(Mods::NONE, "up")), 4);
        assert_eq!(
            slots(tap(
                Mods::CMD | Mods::OPT | Mods::CTRL | Mods::SHIFT,
                "left"
            )),
            14
        );
        assert_eq!(slots(vec![Ev::Abs(1, 1), Ev::Btn("left", true)]), 0);
        assert!(Ev::Btn("left", false).is_click() && !Ev::Btn("wheel-up", true).is_click());
    }

    #[test]
    fn events_are_input_send_events() {
        let mut v = Vec::new();
        Ev::Key("a", true).json(&mut v);
        Ev::Btn("left", false).json(&mut v);
        Ev::Abs(1, 2).json(&mut v);
        assert_eq!(
            Value::Array(v),
            json!([
                { "type": "key", "data": { "down": true, "key": { "type": "qcode", "data": "a" } } },
                { "type": "btn", "data": { "down": false, "button": "left" } },
                { "type": "abs", "data": { "axis": "x", "value": 1 } },
                { "type": "abs", "data": { "axis": "y", "value": 2 } },
            ])
        );
    }
}
