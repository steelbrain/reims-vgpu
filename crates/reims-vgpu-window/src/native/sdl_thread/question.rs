//! The question a close asks on Linux, drawn by the library in the VM's own window
//! (`crate::close`): it needs nothing installed on the host and shows the same on every desktop,
//! Wayland's and X11's. [`Question`] lays it out for the window's drawable and pixel density
//! ([`shape`]), draws it into an image ([`draw`]) that the window's render thread draws over the
//! frame ([`DrawnOverlay`], `super::render`), and takes the window's keys and clicks while it is
//! open (`super::sdl`).
//!
//! **What it shows**: macOS's words and buttons ([`close::QUESTION`], [`close::DETAIL`],
//! [`close::TURN_OFF`], [`close::CANCEL`]), in ASCII: the title over the detail, wrapped at a
//! word to at most [`MAX_COLUMNS`] characters a line, then the buttons at the right, Cancel left
//! of Turn Off as GNOME's and KDE's dialogs place them, on a dark panel centred in the drawable.
//! Cancel, the default, is filled with the accent colour, and the button with the focus has a
//! ring, Cancel's at first.
//!
//! **The font** ([`FONT`]) is the 8 by 8 one SDL draws its debug text with, its glyphs `!` to `~`
//! copied from SDL 3.4.18's `src/render/SDL_render_debug_font.h`: Marcel Sondaar's `font8_8`, of
//! IBM's VGA font, which that file marks public domain. Each of its pixels is a square of a whole
//! number of the drawable's pixels ([`scale_for`]): about 1.5 a point of the window's, so a line is
//! about 12 points high at every scale, fractional ones included; less where the question would
//! not fit the drawable otherwise.
//!
//! **Its keys** ([`Question::key`]), by their USB usages: Return, the keypad's Enter and Escape
//! answer Cancel, the default; Space answers the button with the focus; Tab and the left and right
//! arrows move the focus. A click answers the button it went down and up on
//! ([`Question::click`]).

use std::sync::Arc;

use crate::close::{self, Answer};
use crate::handle::{DrawnOverlay, Glyph};

/// The font's glyphs for `!` (0x21) to `~` (0x7e), 8 rows each from the top, a row's lowest bit
/// its leftmost pixel: copied from SDL 3.4.18's `src/render/SDL_render_debug_font.h`, whose font,
/// Marcel Sondaar's `font8_8` of IBM's VGA font, is in the public domain (the module doc).
#[rustfmt::skip]
const FONT: [[u8; 8]; 94] = [
    [0x18, 0x3c, 0x3c, 0x18, 0x18, 0x00, 0x18, 0x00], // '!'
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '"'
    [0x36, 0x36, 0x7f, 0x36, 0x7f, 0x36, 0x36, 0x00], // '#'
    [0x0c, 0x3e, 0x03, 0x1e, 0x30, 0x1f, 0x0c, 0x00], // '$'
    [0x00, 0x63, 0x33, 0x18, 0x0c, 0x66, 0x63, 0x00], // '%'
    [0x1c, 0x36, 0x1c, 0x6e, 0x3b, 0x33, 0x6e, 0x00], // '&'
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // '\''
    [0x18, 0x0c, 0x06, 0x06, 0x06, 0x0c, 0x18, 0x00], // '('
    [0x06, 0x0c, 0x18, 0x18, 0x18, 0x0c, 0x06, 0x00], // ')'
    [0x00, 0x66, 0x3c, 0xff, 0x3c, 0x66, 0x00, 0x00], // '*'
    [0x00, 0x0c, 0x0c, 0x3f, 0x0c, 0x0c, 0x00, 0x00], // '+'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c, 0x06], // ','
    [0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00], // '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c, 0x00], // '.'
    [0x60, 0x30, 0x18, 0x0c, 0x06, 0x03, 0x01, 0x00], // '/'
    [0x3e, 0x63, 0x73, 0x7b, 0x6f, 0x67, 0x3e, 0x00], // '0'
    [0x0c, 0x0e, 0x0c, 0x0c, 0x0c, 0x0c, 0x3f, 0x00], // '1'
    [0x1e, 0x33, 0x30, 0x1c, 0x06, 0x33, 0x3f, 0x00], // '2'
    [0x1e, 0x33, 0x30, 0x1c, 0x30, 0x33, 0x1e, 0x00], // '3'
    [0x38, 0x3c, 0x36, 0x33, 0x7f, 0x30, 0x78, 0x00], // '4'
    [0x3f, 0x03, 0x1f, 0x30, 0x30, 0x33, 0x1e, 0x00], // '5'
    [0x1c, 0x06, 0x03, 0x1f, 0x33, 0x33, 0x1e, 0x00], // '6'
    [0x3f, 0x33, 0x30, 0x18, 0x0c, 0x0c, 0x0c, 0x00], // '7'
    [0x1e, 0x33, 0x33, 0x1e, 0x33, 0x33, 0x1e, 0x00], // '8'
    [0x1e, 0x33, 0x33, 0x3e, 0x30, 0x18, 0x0e, 0x00], // '9'
    [0x00, 0x0c, 0x0c, 0x00, 0x00, 0x0c, 0x0c, 0x00], // ':'
    [0x00, 0x0c, 0x0c, 0x00, 0x00, 0x0c, 0x0c, 0x06], // ';'
    [0x18, 0x0c, 0x06, 0x03, 0x06, 0x0c, 0x18, 0x00], // '<'
    [0x00, 0x00, 0x3f, 0x00, 0x00, 0x3f, 0x00, 0x00], // '='
    [0x06, 0x0c, 0x18, 0x30, 0x18, 0x0c, 0x06, 0x00], // '>'
    [0x1e, 0x33, 0x30, 0x18, 0x0c, 0x00, 0x0c, 0x00], // '?'
    [0x3e, 0x63, 0x7b, 0x7b, 0x7b, 0x03, 0x1e, 0x00], // '@'
    [0x0c, 0x1e, 0x33, 0x33, 0x3f, 0x33, 0x33, 0x00], // 'A'
    [0x3f, 0x66, 0x66, 0x3e, 0x66, 0x66, 0x3f, 0x00], // 'B'
    [0x3c, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3c, 0x00], // 'C'
    [0x1f, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1f, 0x00], // 'D'
    [0x7f, 0x46, 0x16, 0x1e, 0x16, 0x46, 0x7f, 0x00], // 'E'
    [0x7f, 0x46, 0x16, 0x1e, 0x16, 0x06, 0x0f, 0x00], // 'F'
    [0x3c, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7c, 0x00], // 'G'
    [0x33, 0x33, 0x33, 0x3f, 0x33, 0x33, 0x33, 0x00], // 'H'
    [0x1e, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x1e, 0x00], // 'I'
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1e, 0x00], // 'J'
    [0x67, 0x66, 0x36, 0x1e, 0x36, 0x66, 0x67, 0x00], // 'K'
    [0x0f, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7f, 0x00], // 'L'
    [0x63, 0x77, 0x7f, 0x7f, 0x6b, 0x63, 0x63, 0x00], // 'M'
    [0x63, 0x67, 0x6f, 0x7b, 0x73, 0x63, 0x63, 0x00], // 'N'
    [0x1c, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1c, 0x00], // 'O'
    [0x3f, 0x66, 0x66, 0x3e, 0x06, 0x06, 0x0f, 0x00], // 'P'
    [0x1e, 0x33, 0x33, 0x33, 0x3b, 0x1e, 0x38, 0x00], // 'Q'
    [0x3f, 0x66, 0x66, 0x3e, 0x36, 0x66, 0x67, 0x00], // 'R'
    [0x1e, 0x33, 0x07, 0x0e, 0x38, 0x33, 0x1e, 0x00], // 'S'
    [0x3f, 0x2d, 0x0c, 0x0c, 0x0c, 0x0c, 0x1e, 0x00], // 'T'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3f, 0x00], // 'U'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1e, 0x0c, 0x00], // 'V'
    [0x63, 0x63, 0x63, 0x6b, 0x7f, 0x77, 0x63, 0x00], // 'W'
    [0x63, 0x63, 0x36, 0x1c, 0x1c, 0x36, 0x63, 0x00], // 'X'
    [0x33, 0x33, 0x33, 0x1e, 0x0c, 0x0c, 0x1e, 0x00], // 'Y'
    [0x7f, 0x63, 0x31, 0x18, 0x4c, 0x66, 0x7f, 0x00], // 'Z'
    [0x1e, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1e, 0x00], // '['
    [0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0x40, 0x00], // '\\'
    [0x1e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1e, 0x00], // ']'
    [0x08, 0x1c, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff], // '_'
    [0x0c, 0x0c, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // '`'
    [0x00, 0x00, 0x1e, 0x30, 0x3e, 0x33, 0x6e, 0x00], // 'a'
    [0x07, 0x06, 0x06, 0x3e, 0x66, 0x66, 0x3b, 0x00], // 'b'
    [0x00, 0x00, 0x1e, 0x33, 0x03, 0x33, 0x1e, 0x00], // 'c'
    [0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6e, 0x00], // 'd'
    [0x00, 0x00, 0x1e, 0x33, 0x3f, 0x03, 0x1e, 0x00], // 'e'
    [0x1c, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0f, 0x00], // 'f'
    [0x00, 0x00, 0x6e, 0x33, 0x33, 0x3e, 0x30, 0x1f], // 'g'
    [0x07, 0x06, 0x36, 0x6e, 0x66, 0x66, 0x67, 0x00], // 'h'
    [0x0c, 0x00, 0x0e, 0x0c, 0x0c, 0x0c, 0x1e, 0x00], // 'i'
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1e], // 'j'
    [0x07, 0x06, 0x66, 0x36, 0x1e, 0x36, 0x67, 0x00], // 'k'
    [0x0e, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x1e, 0x00], // 'l'
    [0x00, 0x00, 0x33, 0x7f, 0x7f, 0x6b, 0x63, 0x00], // 'm'
    [0x00, 0x00, 0x1f, 0x33, 0x33, 0x33, 0x33, 0x00], // 'n'
    [0x00, 0x00, 0x1e, 0x33, 0x33, 0x33, 0x1e, 0x00], // 'o'
    [0x00, 0x00, 0x3b, 0x66, 0x66, 0x3e, 0x06, 0x0f], // 'p'
    [0x00, 0x00, 0x6e, 0x33, 0x33, 0x3e, 0x30, 0x78], // 'q'
    [0x00, 0x00, 0x3b, 0x6e, 0x66, 0x06, 0x0f, 0x00], // 'r'
    [0x00, 0x00, 0x3e, 0x03, 0x1e, 0x30, 0x1f, 0x00], // 's'
    [0x08, 0x0c, 0x3e, 0x0c, 0x0c, 0x2c, 0x18, 0x00], // 't'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6e, 0x00], // 'u'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1e, 0x0c, 0x00], // 'v'
    [0x00, 0x00, 0x63, 0x6b, 0x7f, 0x7f, 0x36, 0x00], // 'w'
    [0x00, 0x00, 0x63, 0x36, 0x1c, 0x36, 0x63, 0x00], // 'x'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3e, 0x30, 0x1f], // 'y'
    [0x00, 0x00, 0x3f, 0x19, 0x0c, 0x26, 0x3f, 0x00], // 'z'
    [0x38, 0x0c, 0x0c, 0x07, 0x0c, 0x0c, 0x38, 0x00], // '{'
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // '|'
    [0x07, 0x0c, 0x0c, 0x38, 0x0c, 0x0c, 0x07, 0x00], // '}'
    [0x6e, 0x3b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '~'
];

/// The longest a line of the detail is, in characters, before it wraps at a word.
const MAX_COLUMNS: u32 = 40;
/// A glyph's side, in the font's pixels.
const CELL: u32 = 8;
/// The panel's margin around what it shows, in the font's pixels.
const PAD: u32 = 16;
/// The room between the title and the detail, in the font's pixels.
const TITLE_GAP: u32 = 12;
/// The room between the detail's lines, in the font's pixels.
const LEADING: u32 = 4;
/// The room between the detail and the buttons, in the font's pixels.
const BUTTONS_GAP: u32 = 16;
/// A button's height, then its width, in the font's pixels: room around its label, the longer
/// one (Turn Off) setting both buttons' width.
const BUTTON: (u32, u32) = (24, 96);
/// The room between the buttons, in the font's pixels.
const BUTTON_GAP: u32 = 8;
/// The fewest characters a line of the detail holds: the buttons' width.
const MIN_COLUMNS: u32 = (2 * BUTTON.1 + BUTTON_GAP) / CELL;
/// The most drawable pixels a font pixel takes.
const MAX_SCALE: u32 = 16;

/// The panel's colour, RGBA.
const PANEL: [u8; 4] = [0x2b, 0x2b, 0x2b, 0xff];
/// The panel's edge.
const EDGE: [u8; 4] = [0x6b, 0x6b, 0x6b, 0xff];
/// The title's and the buttons' labels.
const BRIGHT: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
/// The detail's.
const TEXT: [u8; 4] = [0xd4, 0xd4, 0xd4, 0xff];
/// The default button, Cancel.
const ACCENT: [u8; 4] = [0x35, 0x74, 0xf0, 0xff];
/// The other button.
const PLAIN: [u8; 4] = [0x4a, 0x4a, 0x4a, 0xff];
/// The ring around the button with the focus.
const RING: [u8; 4] = [0xa8, 0xc8, 0xff, 0xff];

/// Return, the keypad's Enter, Escape, Tab, Space, and the right and left arrows: USB usages.
const RETURN: u16 = 0x28;
/// The keypad's Enter.
const KEYPAD_ENTER: u16 = 0x58;
/// Escape.
const ESCAPE: u16 = 0x29;
/// Tab.
const TAB: u16 = 0x2B;
/// Space.
const SPACE: u16 = 0x2C;
/// The right arrow.
const RIGHT: u16 = 0x4F;
/// The left arrow.
const LEFT: u16 = 0x50;

/// One of the question's buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Button {
    /// Cancel, the default.
    Cancel,
    /// Turn Off.
    TurnOff,
}

/// The buttons, left to right.
const BUTTONS: [Button; 2] = [Button::Cancel, Button::TurnOff];

impl Button {
    /// What choosing it answers.
    fn answer(self) -> Answer {
        match self {
            Button::Cancel => Answer::Cancel,
            Button::TurnOff => Answer::TurnOff,
        }
    }

    /// Its label.
    fn label(self) -> &'static str {
        match self {
            Button::Cancel => close::CANCEL,
            Button::TurnOff => close::TURN_OFF,
        }
    }
}

/// The question laid out for a drawable ([`shape`]): its sizes in the font's pixels, each `scale`
/// drawable pixels on a side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    /// The drawable's pixels on a side of a font pixel.
    pub(super) scale: u32,
    /// The detail's lines.
    lines: Vec<String>,
    /// The panel's width and height, in font pixels.
    size: (u32, u32),
    /// The panel's top left in the drawable, in its pixels.
    at: (u64, u64),
}

impl Shape {
    /// The panel's width and height in the drawable's pixels.
    fn pixels(&self) -> (u32, u32) {
        (self.size.0 * self.scale, self.size.1 * self.scale)
    }

    /// Where `button` is in the panel, in font pixels: its left, top, width and height.
    fn button(&self, button: Button) -> (u32, u32, u32, u32) {
        let top = self.size.1 - PAD - BUTTON.0;
        let right = self.size.0 - PAD;
        let left = match button {
            Button::TurnOff => right - BUTTON.1,
            Button::Cancel => right - 2 * BUTTON.1 - BUTTON_GAP,
        };
        (left, top, BUTTON.1, BUTTON.0)
    }

    /// The top of the detail's first line, in font pixels.
    fn detail_top() -> u32 {
        PAD + CELL + TITLE_GAP
    }

    /// The button under `point`, in the drawable's pixels.
    fn button_at(&self, point: (f64, f64)) -> Option<Button> {
        let scale = f64::from(self.scale);
        let (x, y) = (
            (point.0 - self.at.0 as f64) / scale,
            (point.1 - self.at.1 as f64) / scale,
        );
        BUTTONS.into_iter().find(|&button| {
            let (left, top, width, height) = self.button(button);
            let (left, top) = (f64::from(left), f64::from(top));
            (left..left + f64::from(width)).contains(&x)
                && (top..top + f64::from(height)).contains(&y)
        })
    }
}

/// The drawable's pixels a font pixel takes at `density` drawable pixels a point (the module
/// doc): 1.5 a point, rounded, at least 1.
pub(super) fn scale_for(density: f64) -> u32 {
    let scale = (1.5 * density).round();
    if scale.is_finite() && scale >= 1.0 {
        scale.min(f64::from(MAX_SCALE)) as u32
    } else {
        1
    }
}

/// `text`'s paragraphs (its lines), each wrapped at a word to at most `columns` characters a line;
/// a word longer than that has a line of its own.
fn wrap(text: &str, columns: u32) -> Vec<String> {
    let columns = usize::try_from(columns).unwrap_or(usize::MAX);
    let mut lines = Vec::new();
    for paragraph in text.lines() {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > columns {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}

/// `text`'s width in font pixels.
fn width_of(text: &str) -> u32 {
    u32::try_from(text.len())
        .unwrap_or(u32::MAX)
        .saturating_mul(CELL)
}

/// The question laid out for a drawable of `drawable` pixels at `density` pixels a point: at
/// [`scale_for`]'s scale, or the largest below it at which the panel fits the drawable, its
/// detail as wide as the drawable takes, from [`MIN_COLUMNS`] to [`MAX_COLUMNS`] characters a
/// line, centred in the drawable.
pub(super) fn shape(drawable: (u64, u64), density: f64) -> Shape {
    let mut scale = scale_for(density);
    loop {
        let across = u32::try_from(drawable.0 / u64::from(scale)).unwrap_or(u32::MAX);
        let columns = (across.saturating_sub(2 * PAD) / CELL).clamp(MIN_COLUMNS, MAX_COLUMNS);
        let lines = wrap(close::DETAIL, columns);
        let text = (lines.iter().map(|l| width_of(l)))
            .chain([width_of(close::QUESTION), 2 * BUTTON.1 + BUTTON_GAP])
            .max()
            .unwrap_or(0);
        let rows = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        let size = (
            text + 2 * PAD,
            Shape::detail_top() + rows * (CELL + LEADING) - LEADING + BUTTONS_GAP + BUTTON.0 + PAD,
        );
        let (width, height) = (u64::from(size.0 * scale), u64::from(size.1 * scale));
        let fits = width <= drawable.0 && height <= drawable.1;
        if fits || scale == 1 {
            let at = (
                drawable.0.saturating_sub(width) / 2,
                drawable.1.saturating_sub(height) / 2,
            );
            return Shape {
                scale,
                lines,
                size,
                at,
            };
        }
        scale -= 1;
    }
}

/// What the question draws on: its pixels, RGBA, in rows of `width`, painted in font pixels of
/// `scale` drawable pixels a side.
struct Canvas {
    /// The pixels.
    rgba: Vec<u8>,
    /// Its width, in drawable pixels.
    width: u32,
    /// The drawable pixels on a side of a font pixel.
    scale: u32,
}

impl Canvas {
    /// Fills the rectangle `left`, `top`, `width`, `height` (in font pixels) with `colour`.
    fn fill(&mut self, (left, top, width, height): (u32, u32, u32, u32), colour: [u8; 4]) {
        let s = self.scale;
        let row = usize::try_from(self.width).unwrap_or(0) * 4;
        let x0 = usize::try_from(left * s).unwrap_or(0) * 4;
        let x1 = usize::try_from((left + width) * s).unwrap_or(0) * 4;
        for y in top * s..(top + height) * s {
            let start = usize::try_from(y).unwrap_or(0) * row;
            if let Some(span) = self.rgba.get_mut(start + x0..start + x1.min(row)) {
                for pixel in span.as_chunks_mut::<4>().0 {
                    *pixel = colour;
                }
            }
        }
    }

    /// Draws `text` from `left`, `top` (in font pixels) in `colour`, a glyph a cell: what the
    /// font has no glyph for, a space among them, leaves its cell as it is.
    fn text(&mut self, left: u32, top: u32, text: &str, colour: [u8; 4]) {
        for (column, c) in (0u32..).zip(text.chars()) {
            let Some(glyph) = glyph_of(c) else {
                continue;
            };
            for (y, bits) in (0u32..).zip(glyph) {
                for x in 0..CELL {
                    if bits & (1 << x) != 0 {
                        self.fill((left + column * CELL + x, top + y, 1, 1), colour);
                    }
                }
            }
        }
    }
}

/// The font's glyph of `c`, or `None` where it has none: a space, or what is not `!` to `~`.
fn glyph_of(c: char) -> Option<[u8; 8]> {
    let index = u32::from(c).checked_sub(0x21)?;
    FONT.get(usize::try_from(index).ok()?).copied()
}

/// The question as `shape` lays it out, with the focus on `focus`: the panel, its edge, the title,
/// the detail and the buttons (the module doc). Opaque.
pub(super) fn draw(shape: &Shape, focus: Button) -> Glyph {
    let (width, height) = shape.pixels();
    let length = usize::try_from(u64::from(width) * u64::from(height) * 4).unwrap_or(0);
    let mut canvas = Canvas {
        rgba: vec![0; length],
        width,
        scale: shape.scale,
    };
    let (w, h) = shape.size;
    canvas.fill((0, 0, w, h), EDGE);
    canvas.fill((1, 1, w - 2, h - 2), PANEL);
    canvas.text(PAD, PAD, close::QUESTION, BRIGHT);
    for (row, line) in (0u32..).zip(&shape.lines) {
        canvas.text(
            PAD,
            Shape::detail_top() + row * (CELL + LEADING),
            line,
            TEXT,
        );
    }
    for button in BUTTONS {
        let (left, top, bw, bh) = shape.button(button);
        if button == focus {
            canvas.fill((left - 2, top - 2, bw + 4, bh + 4), RING);
            canvas.fill((left - 1, top - 1, bw + 2, bh + 2), PANEL);
        }
        let fill = if button == Button::Cancel {
            ACCENT
        } else {
            PLAIN
        };
        canvas.fill((left, top, bw, bh), fill);
        let label = button.label();
        let x = left + (bw - width_of(label)) / 2;
        canvas.text(x, top + (bh - CELL) / 2, label, BRIGHT);
    }
    Glyph {
        width,
        height,
        hot_spot: (0, 0),
        rgba: canvas.rgba,
    }
}

/// What a key or a click did to the question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    /// Nothing.
    Nothing,
    /// The focus moved: the question is drawn again ([`Question::overlay`]).
    Redraw,
    /// The person answered.
    Answer(Answer),
}

/// The question open in a window: for which windows, its focus, and its image.
pub(super) struct Question {
    /// The windows the answer is for (`crate::sdl::shell::answered`).
    pub(super) ids: Vec<u64>,
    /// The button with the focus.
    focus: Button,
    /// The button the left button went down on, until it goes up.
    pressed: Option<Button>,
    /// Its layout.
    shape: Shape,
    /// Its image, drawn for the shape and the focus.
    image: Arc<Glyph>,
}

impl Question {
    /// The question for windows `ids`, laid out for a drawable of `drawable` pixels at `density`
    /// pixels a point, the focus on Cancel.
    pub(super) fn new(ids: Vec<u64>, drawable: (u64, u64), density: f64) -> Question {
        let shape = shape(drawable, density);
        let image = Arc::new(draw(&shape, Button::Cancel));
        Question {
            ids,
            focus: Button::Cancel,
            pressed: None,
            shape,
            image,
        }
    }

    /// What the window's render thread draws over the frame.
    pub(super) fn overlay(&self) -> DrawnOverlay {
        DrawnOverlay {
            image: Arc::clone(&self.image),
            at: self.shape.at,
        }
    }

    /// The window was laid out again, its drawable now `drawable` pixels at `density` pixels a
    /// point: whether the question changed, and is drawn again.
    pub(super) fn relayout(&mut self, drawable: (u64, u64), density: f64) -> bool {
        let shape = shape(drawable, density);
        if shape == self.shape {
            return false;
        }
        self.shape = shape;
        self.image = Arc::new(draw(&self.shape, self.focus));
        true
    }

    /// Moves the focus to `button`.
    fn focus(&mut self, button: Button) -> Step {
        if button == self.focus {
            return Step::Nothing;
        }
        self.focus = button;
        self.image = Arc::new(draw(&self.shape, self.focus));
        Step::Redraw
    }

    /// Key `usage` went down (`down`) or up; `repeat` for the host's repeat of a held key: only
    /// a first down does anything (the module doc).
    pub(super) fn key(&mut self, usage: u16, down: bool, repeat: bool) -> Step {
        if !down || repeat {
            return Step::Nothing;
        }
        match usage {
            RETURN | KEYPAD_ENTER | ESCAPE => Step::Answer(Answer::Cancel),
            SPACE => Step::Answer(self.focus.answer()),
            TAB => match self.focus {
                Button::Cancel => self.focus(Button::TurnOff),
                Button::TurnOff => self.focus(Button::Cancel),
            },
            LEFT => self.focus(Button::Cancel),
            RIGHT => self.focus(Button::TurnOff),
            _ => Step::Nothing,
        }
    }

    /// The left button went down (`down`) or up at `point`, in the drawable's pixels: going up
    /// over the button it went down on answers that button.
    pub(super) fn click(&mut self, down: bool, point: (f64, f64)) -> Step {
        let under = self.shape.button_at(point);
        if down {
            self.pressed = under;
            return Step::Nothing;
        }
        match (self.pressed.take(), under) {
            (Some(pressed), Some(under)) if pressed == under => Step::Answer(under.answer()),
            _ => Step::Nothing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A question for window 1 in a 1920 x 1080 drawable at 1 pixel a point.
    fn question() -> Question {
        Question::new(vec![1], (1920, 1080), 1.0)
    }

    /// The centre of `button` in the drawable's pixels.
    fn centre(q: &Question, button: Button) -> (f64, f64) {
        let (left, top, width, height) = q.shape.button(button);
        let s = f64::from(q.shape.scale);
        (
            q.shape.at.0 as f64 + (f64::from(left) + f64::from(width) / 2.0) * s,
            q.shape.at.1 as f64 + (f64::from(top) + f64::from(height) / 2.0) * s,
        )
    }

    /// The question says macOS's words, in ASCII, every character of them one the font draws
    #[test]
    fn the_question_s_words_are_macos_s_in_the_font() {
        for text in [
            close::QUESTION,
            close::DETAIL,
            close::TURN_OFF,
            close::CANCEL,
        ] {
            for c in text.chars().filter(|&c| c != ' ' && c != '\n') {
                assert!(c.is_ascii() && glyph_of(c).is_some(), "{c:?} in {text:?}");
            }
        }
        assert_eq!(glyph_of(' '), None);
        assert_eq!(
            glyph_of('A'),
            Some([0x0c, 0x1e, 0x33, 0x33, 0x3f, 0x33, 0x33, 0x00])
        );
        assert_eq!(glyph_of('~').map(|g| g[0]), Some(0x6e));
        let shape = shape((1920, 1080), 1.0);
        assert_eq!(
            shape.lines,
            [
                "The VM turns off at once, as if its",
                "power were cut.",
                "Unsaved work in it is lost."
            ]
        );
    }

    /// A font pixel is a whole number of drawable pixels, about 1.5 a point at every scale, so the
    /// question reads at 4K and at fractional scales; the question shrinks only where it would
    /// not fit, and is centred
    #[test]
    fn the_font_scales_with_the_density() {
        for (density, scale) in [
            (1.0, 2),
            (1.25, 2),
            (1.5, 2),
            (1.75, 3),
            (2.0, 3),
            (2.1, 3),
            (3.0, 5),
            (0.0, 1),
            (f64::NAN, 1),
            (100.0, MAX_SCALE),
        ] {
            assert_eq!(scale_for(density), scale, "{density}");
        }
        let wide = shape((3840, 2160), 2.0);
        assert_eq!(wide.scale, 3);
        let (width, height) = wide.pixels();
        assert_eq!(
            wide.at,
            (
                (3840 - u64::from(width)) / 2,
                (2160 - u64::from(height)) / 2
            )
        );
        // A window of 320 x 200 points at 1 pixel a point: one pixel a font pixel, the detail
        // wrapped narrower, and the whole question inside the drawable.
        let small = shape((320, 200), 1.0);
        assert_eq!(small.scale, 1);
        let (width, height) = small.pixels();
        assert!(width <= 320 && height <= 200, "{width} x {height}");
        assert_eq!(
            small.at,
            ((320 - u64::from(width)) / 2, (200 - u64::from(height)) / 2)
        );
    }

    /// The image is the panel at the shape's size, each font pixel a square of the scale's pixels:
    /// the edge at the corner, the title's first glyph (T) in its colour, the default button
    /// filled with the accent and the focus's ring around it
    #[test]
    fn the_question_is_drawn_in_squares_of_the_scale() {
        let q = Question::new(vec![1], (3840, 2160), 2.0);
        let image = &q.image;
        let s = q.shape.scale;
        assert_eq!((image.width, image.height), q.shape.pixels());
        assert_eq!(image.hot_spot, (0, 0));
        let pixel = |x: u32, y: u32| {
            let at = usize::try_from((y * image.width + x) * 4).unwrap();
            <[u8; 4]>::try_from(&image.rgba[at..at + 4]).unwrap()
        };
        assert_eq!(pixel(0, 0), EDGE);
        assert_eq!(pixel(s, s), PANEL);
        // T's first row is its left six pixels.
        for x in 0..6 * s {
            assert_eq!(pixel(PAD * s + x, PAD * s), BRIGHT, "{x}");
        }
        assert_eq!(pixel((PAD + 6) * s, PAD * s), PANEL);
        let (left, top, ..) = q.shape.button(Button::Cancel);
        assert_eq!(pixel(left * s, top * s), ACCENT);
        assert_eq!(pixel((left - 2) * s, (top - 2) * s), RING);
        assert_eq!(pixel((left - 1) * s, (top - 1) * s), PANEL);
        let (left, top, ..) = q.shape.button(Button::TurnOff);
        assert_eq!(pixel(left * s, top * s), PLAIN);
        assert_eq!(pixel((left - 2) * s, (top - 2) * s), PANEL, "no ring");
    }

    /// Return, the keypad's Enter and Escape answer Cancel, the default, wherever the focus is;
    /// Space answers the button with the focus, which Tab and the arrows move; ups and repeats do
    /// nothing
    #[test]
    fn return_and_escape_cancel_and_space_answers_the_focus() {
        for key in [RETURN, KEYPAD_ENTER, ESCAPE] {
            let mut q = question();
            assert_eq!(q.key(TAB, true, false), Step::Redraw);
            assert_eq!(q.key(key, true, true), Step::Nothing, "a repeat");
            assert_eq!(q.key(key, false, false), Step::Nothing, "an up");
            assert_eq!(q.key(key, true, false), Step::Answer(Answer::Cancel));
        }
        let mut q = question();
        let before = Arc::clone(&q.image);
        assert_eq!(q.key(SPACE, true, false), Step::Answer(Answer::Cancel));
        assert_eq!(q.key(TAB, true, false), Step::Redraw);
        assert!(!Arc::ptr_eq(&before, &q.image), "drawn again");
        assert_eq!(q.key(SPACE, true, false), Step::Answer(Answer::TurnOff));
        assert_eq!(q.key(RIGHT, true, false), Step::Nothing, "the focus stays");
        assert_eq!(q.key(LEFT, true, false), Step::Redraw);
        assert_eq!(q.key(SPACE, true, false), Step::Answer(Answer::Cancel));
        assert_eq!(q.key(RIGHT, true, false), Step::Redraw);
        assert_eq!(q.key(TAB, true, false), Step::Redraw);
        assert_eq!(q.key(SPACE, true, false), Step::Answer(Answer::Cancel));
        assert_eq!(q.key(0x04, true, false), Step::Nothing, "a");
    }

    /// A click answers the button it went down and up on; one that goes up off it, or on the
    /// other, or anywhere else, answers nothing
    #[test]
    fn a_click_answers_the_button_it_went_down_and_up_on() {
        let mut q = question();
        let (cancel, turn_off) = (centre(&q, Button::Cancel), centre(&q, Button::TurnOff));
        assert_eq!(q.click(true, turn_off), Step::Nothing);
        assert_eq!(q.click(false, turn_off), Step::Answer(Answer::TurnOff));
        assert_eq!(q.click(true, cancel), Step::Nothing);
        assert_eq!(q.click(false, cancel), Step::Answer(Answer::Cancel));
        assert_eq!(q.click(true, cancel), Step::Nothing);
        assert_eq!(q.click(false, turn_off), Step::Nothing, "up on the other");
        assert_eq!(
            q.click(false, turn_off),
            Step::Nothing,
            "an up with no down"
        );
        assert_eq!(q.click(true, (0.0, 0.0)), Step::Nothing);
        assert_eq!(q.click(false, (0.0, 0.0)), Step::Nothing);
        let (left, top, ..) = q.shape.button(Button::TurnOff);
        let s = f64::from(q.shape.scale);
        let edge = (
            q.shape.at.0 as f64 + f64::from(left) * s - 0.5,
            q.shape.at.1 as f64 + f64::from(top) * s,
        );
        assert_eq!(q.click(true, edge), Step::Nothing);
        assert_eq!(q.click(false, edge), Step::Nothing, "just left of Turn Off");
    }

    /// The question laid out again for the drawable it has changes nothing; for a new drawable or
    /// density it is placed and drawn again
    #[test]
    fn a_new_layout_places_and_draws_the_question_again() {
        let mut q = question();
        let image = Arc::clone(&q.image);
        assert!(!q.relayout((1920, 1080), 1.0));
        assert!(Arc::ptr_eq(&image, &q.image));
        assert!(q.relayout((1280, 720), 1.0));
        assert_eq!(q.overlay().at, q.shape.at);
        assert!(q.relayout((2560, 1440), 2.0));
        assert_eq!(q.shape.scale, 3);
        assert!(!Arc::ptr_eq(&image, &q.overlay().image));
    }
}
