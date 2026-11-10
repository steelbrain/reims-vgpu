//! The SDL window's Linux side ([`crate::sdl::os`]): Wayland's or X11's window, drawn through
//! Vulkan, on the library's UI thread.
//!
//! **The screen.** The screen a window opens on is SDL's display of that ID (the host's screens
//! are SDL's on Linux, `super::ui`), else SDL's primary display. Wayland lets no client place a
//! windowed window: the compositor puts it where it likes, and only a borderless window is sure to
//! be on its screen. SDL's answer to which screen a window is on is not used: on Wayland it names
//! the first screen for a window the compositor shows on another. A window resized for a guest's
//! mode fits the screen it opened on.
//!
//! **Windowed**: decorated and resizable, centred on its screen where the window system lets it
//! be placed, its content shrunk to fit the screen's usable area less [`TITLE_ROOM`] for the title
//! bar and a panel, which Wayland does not tell a client. Its size and its smallest are asked for
//! in SDL's sizes (`crate::sdl::units`): points on Wayland, and on X11 pixels, the desktop's
//! content scale of them to the point.
//!
//! **Borderless**: through the compositor's own full screen on its screen, over its panels.
//!
//! **The surface.** The window is made for Vulkan: its render thread makes the window's Vulkan
//! surface of SDL's window (`super::render`), so SDL's window goes only once that thread has let
//! its swapchain and surface go ([`close`], `crate::sdl::shell::released`).
//!
//! **Input** is SDL's events ([`input_event`], `super::input`). While captured, the window grabs
//! the keyboard (`SDL_SetWindowKeyboardGrab`), so the compositor's own shortcuts reach the guest:
//! on Wayland through the compositor's shortcuts inhibitor (keyboard-shortcuts-inhibit), under
//! XWayland through an X keyboard grab, which XWayland hands the compositor
//! (xwayland-keyboard-grab). SDL holds the grab only while the window has the keyboard, and takes
//! it when the window gains it. SDL 3.4.18 opens windows through XWayland on a Wayland compositor
//! without fifo-v1 (read from its code), and cosmic-comp has none. Measured 2026-10-13 on
//! cosmic-comp 1.9.0, nested: through either route, while captured, its shortcuts (Super alone,
//! Super+Q, Super+A, Alt+Tab) stay unused and their keys reach the guest, a click that focuses the
//! window included; not captured, Super+Q closes the window. Each capture and release is logged
//! with whether SDL holds the grab ([`grab_keyboard`]).

use std::sync::Arc;

use sdl3_sys::everything::*;

use super::input::{self, At, Wheel};
use super::keymap;
use super::render::Target;
use crate::geometry::{Insets, Rect};
use crate::handle::Window;
use crate::screens;
use crate::sdl::c_string;
use crate::sdl::shell::{self, DEFAULT_MODE, Opened, Shell};
use crate::sdl::units::{self, Units};

/// Points of a screen's height kept for a window's title bar and a panel: Wayland gives a client
/// no work area, nor the size of the compositor's decorations.
pub(crate) const TITLE_ROOM: f64 = 48.0;

/// Where an open window is, for placing others: nothing on Linux, where the window system places
/// them.
pub(crate) type Frame = std::convert::Infallible;

/// Where `shell`'s window is: nothing on Linux.
pub(crate) fn frame_of(_shell: &Shell) -> Option<Frame> {
    None
}

/// What Linux keeps of a window.
pub(crate) struct State {
    /// The screen it opened on: SDL's answer to which screen a window is on is not used (the
    /// module doc).
    display: SDL_DisplayID,
}

/// A window's input: SDL's events, queued until the shell sends them (`super::input`).
pub(crate) struct Input(input::Input);

impl Input {
    /// The window is closing: every key and button held is lifted, and capture ends.
    pub(crate) fn closing(&mut self) {
        self.0.closing();
    }

    /// The person closed the window: every key and button held goes up, and capture ends, before
    /// the host is told, as on macOS.
    pub(crate) fn close_requested(&mut self) {
        self.0.closing();
    }

    /// The window gained the keyboard at `timestamp` (SDL's event clock).
    pub(crate) fn focus_gained(&mut self, timestamp: u64) {
        self.0.focus_gained(timestamp);
    }

    /// The window lost the keyboard: whether capture changed.
    pub(crate) fn focus_lost(&mut self) -> bool {
        self.0.focus_lost().capture_changed
    }

    /// Whether input, the keyboard and the pointer, is captured.
    pub(crate) fn is_captured(&self) -> bool {
        self.0.is_captured()
    }

    /// Where the pointer is held over `image`, in the window's points: the image while captured.
    pub(crate) fn pointer_confinement(&self, image: Rect) -> Option<Rect> {
        self.0.pointer_confinement(image)
    }

    /// What the title ends with in the current capture state.
    pub(crate) fn title_suffix(&self) -> &'static str {
        self.0.title_suffix()
    }

    /// Sends what the input queued to `window`'s callbacks, while the window lives
    /// ([`Window::send_each`]), each pointer event telling the window first
    /// ([`Window::pointer_sent`]).
    pub(crate) fn flush(&mut self, window: &Window) {
        window.send_each(self.0.take(), |callbacks, event| {
            if matches!(event, input::ToHost::Pointer { .. }) {
                window.pointer_sent();
            }
            input::send(callbacks, event);
        });
    }
}

/// SDL's sizes on screen `display` (`crate::sdl::units`).
fn units_of(display: SDL_DisplayID) -> Units {
    // SAFETY: SDL's video runs on this thread; the mode is SDL's, read at once.
    let density =
        unsafe { SDL_GetCurrentDisplayMode(display).as_ref() }.map_or(1.0, |m| m.pixel_density);
    // SAFETY: as above.
    let content = unsafe { SDL_GetDisplayContentScale(display) };
    Units::new(density, content)
}

/// The content size, in SDL's sizes, of a windowed window for a guest mode `mode` at
/// `guest_scale`, on screen `display` (the module doc): at least `shell::MIN_CONTENT` points.
fn windowed_size(display: SDL_DisplayID, mode: (u16, u16), guest_scale: u8) -> (i32, i32) {
    let mut usable = SDL_Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };
    // SAFETY: SDL's video runs on this thread.
    let known = unsafe { SDL_GetDisplayUsableBounds(display, &mut usable) };
    let usable = known.then_some((usable.w, usable.h));
    units::windowed_size(mode, guest_scale, usable, TITLE_ROOM, units_of(display))
}

/// Makes and shows `window`'s SDL window (the module doc), and hands its render thread SDL's
/// window.
pub(crate) fn open(window: &Arc<Window>, borderless: bool, _taken: &[Frame]) -> Option<Opened> {
    // SAFETY: SDL's video runs on this thread.
    let display = screens::screen_for_port(window.port()).map_or_else(
        || unsafe { SDL_GetPrimaryDisplay() },
        |s| SDL_DisplayID(s.id),
    );
    let mode = window.mode().or(window.offered()).unwrap_or(DEFAULT_MODE);
    let size = windowed_size(display, mode, window.guest_scale(mode));
    let min = units_of(display).min_content();
    let title = c_string(window.title());
    let flags = SDL_WINDOW_VULKAN
        | SDL_WINDOW_HIGH_PIXEL_DENSITY
        | SDL_WINDOW_RESIZABLE
        | SDL_WINDOW_HIDDEN;
    // SAFETY: SDL's video runs on this thread; the title is copied.
    let sdl = unsafe { SDL_CreateWindow(title.as_ptr(), size.0, size.1, flags) };
    if sdl.is_null() {
        shell::refused("no window", window.title());
        return None;
    }
    let centered = SDL_WINDOWPOS_CENTERED_DISPLAY(display);
    // SAFETY: `sdl` is the window just made, on this thread.
    unsafe {
        SDL_SetWindowMinimumSize(sdl, min.0, min.1);
        SDL_SetWindowPosition(sdl, centered, centered);
        if borderless {
            SDL_SetWindowFullscreen(sdl, true);
        }
        SDL_ShowWindow(sdl);
    }
    window.surface().set_target(Target(sdl));
    window.surface().set_visible(true);
    log::debug!(
        target: "reims_vgpu_window",
        "window {:?} on screen {}",
        window.title(),
        display.0
    );
    Some(Opened {
        sdl,
        asked: size,
        input: Input(input::Input::new()),
        state: State { display },
    })
}

/// Resizes a window following the guest's mode for `mode`, as it would open on its screen.
pub(crate) fn follow(shell: &mut Shell, mode: (u16, u16), _taken: &[Frame]) {
    let size = windowed_size(shell.os.display, mode, shell.window.guest_scale(mode));
    if size != shell.asked {
        shell.asked = size;
        // SAFETY: the window is SDL's, on this thread.
        unsafe { SDL_SetWindowSize(shell.sdl, size.0, size.1) };
    }
}

/// The window's size in points and its pixels per point: SDL's sizes of it in points and in
/// pixels, whose ratio is the scale, fractional ones included. `None` before it has a size.
pub(crate) fn metrics(shell: &Shell) -> Option<((f64, f64), f64)> {
    let (mut w, mut h, mut pw, mut ph) = (0, 0, 0, 0);
    // SAFETY: the window is SDL's, on this thread.
    unsafe {
        SDL_GetWindowSize(shell.sdl, &mut w, &mut h);
        SDL_GetWindowSizeInPixels(shell.sdl, &mut pw, &mut ph);
    }
    if w <= 0 || h <= 0 || pw <= 0 || ph <= 0 {
        return None;
    }
    let density = f64::from(pw) / f64::from(w);
    Some(((f64::from(pw) / density, f64::from(ph) / density), density))
}

/// The room the window keeps clear around the image: none.
pub(crate) fn insets(_shell: &Shell) -> Insets {
    Insets::default()
}

/// The screens changed: nothing more than the layout.
pub(crate) fn screens_changed(_shell: &mut Shell) {}

/// What the window knows for an event at `point` (SDL's, in points).
fn at(shell: &Shell, point: (f32, f32)) -> At {
    At {
        point: Some((f64::from(point.0), f64::from(point.1))),
        image: shell.layout.image,
        guest: shell.window.mode(),
    }
}

/// An input event of type `kind` for the window: the pointer, a button, the wheel or a key, to
/// the window's input. Whether capture changed.
pub(crate) fn input_event(shell: &mut Shell, kind: u32, event: &SDL_Event) -> bool {
    match kind {
        k if k == SDL_EVENT_MOUSE_MOTION.0 => {
            // SAFETY: a motion event.
            let point = unsafe { (event.motion.x, event.motion.y) };
            let at = at(shell, point);
            shell.input.0.moved(&at);
            shell.pointer_over((f64::from(point.0), f64::from(point.1)));
            false
        }
        k if k == SDL_EVENT_MOUSE_BUTTON_DOWN.0 || k == SDL_EVENT_MOUSE_BUTTON_UP.0 => {
            // SAFETY: a button event.
            let b = unsafe { event.button };
            let at = at(shell, (b.x, b.y));
            let bit = input::button_bit(b.button);
            if b.down {
                shell
                    .input
                    .0
                    .button_down(bit, b.timestamp, &at)
                    .capture_changed
            } else {
                shell.input.0.button_up(bit, &at);
                false
            }
        }
        k if k == SDL_EVENT_MOUSE_WHEEL.0 => {
            // SAFETY: a wheel event.
            let w = unsafe { event.wheel };
            let at = at(shell, (w.mouse_x, w.mouse_y));
            let wheel = Wheel {
                notches: (w.integer_x, w.integer_y),
                lines: (w.x, w.y),
                flipped: w.direction == SDL_MOUSEWHEEL_FLIPPED,
            };
            shell.input.0.wheel(wheel, &at);
            false
        }
        k if k == SDL_EVENT_KEY_DOWN.0 || k == SDL_EVENT_KEY_UP.0 => {
            // SAFETY: a key event.
            let key = unsafe { event.key };
            keymap::usb_usage(key.scancode.0).is_some_and(|usage| {
                shell
                    .input
                    .0
                    .key(usage, key.down, key.repeat)
                    .capture_changed
            })
        }
        _ => false,
    }
}

/// Grabs the keyboard while captured, and lets it go (the module doc), logging what that left.
pub(crate) fn grab_keyboard(shell: &Shell, captured: bool) {
    // SAFETY: the window is SDL's, on this thread.
    let asked = unsafe { SDL_SetWindowKeyboardGrab(shell.sdl, captured) };
    let refused = (!asked).then(crate::sdl::sdl_error);
    // SAFETY: as above.
    let held = unsafe { SDL_GetWindowKeyboardGrab(shell.sdl) };
    // SAFETY: as above.
    let flags = unsafe { SDL_GetWindowFlags(shell.sdl) };
    log::info!(
        target: "reims_vgpu_window",
        "window {:?}: {}",
        shell.window.title(),
        grab_outcome(
            captured,
            (flags & SDL_WINDOW_INPUT_FOCUS).0 != 0,
            held,
            refused.as_deref()
        )
    );
}

/// Nothing: the window stays resizable while captured. The hold keeps the pointer on the window's
/// own surface, and the window's resize edges lie outside it, the compositor's (xdg-decoration's
/// decorations, libdecor's frame, an X11 window manager's frame window), where the pointer held
/// over the image cannot go. A window SDL made not resizable would also be held on X11 at its
/// size from before it was maximized or tiled (`X11_SetWindowMinMax`, read from SDL 3.4.18's
/// code).
pub(crate) fn hold_size(_shell: &Shell) {}

/// What a change of capture left: whether `captured`, whether the window has the keyboard
/// (`focused`), whether SDL then holds the keyboard grab (`held`), and the error where SDL refused
/// it.
fn grab_outcome(captured: bool, focused: bool, held: bool, refused: Option<&str>) -> String {
    match (captured, held, refused) {
        (true, true, _) => "keyboard captured; SDL holds the keyboard grab, so the compositor's \
                            shortcuts go to the guest"
            .into(),
        (true, false, Some(error)) => format!(
            "keyboard captured, but SDL cannot grab the keyboard ({error}): the compositor keeps \
             its own shortcuts"
        ),
        (true, false, None) => {
            "keyboard captured; SDL grabs the keyboard once the window has it".into()
        }
        (false, _, _) if !focused => "keyboard released: the window lost the keyboard, and the \
                                      compositor's shortcuts are its own again"
            .into(),
        (false, _, _) => "keyboard released: the compositor's shortcuts are its own again".into(),
    }
}

/// The display ended: the window hides. SDL's window goes once the render thread has let go
/// (`crate::sdl::shell::released`), never before.
pub(crate) fn close(shell: &mut Shell) -> bool {
    // SAFETY: the window is SDL's, on this thread.
    unsafe { SDL_HideWindow(shell.sdl) };
    false
}

/// The window goes: nothing of Linux's stays.
pub(crate) fn destroy(_shell: Shell) {}

#[cfg(test)]
mod tests {
    use super::input::ToHost;
    use super::*;
    use crate::geometry::Rect;
    use crate::keys::KeyEvent;

    /// The person's close lets the input go before the shell tells the host
    /// (`shell::Shell::close_requested`): a held key and button go up, queued for the host, and
    /// capture ends
    #[test]
    fn a_close_the_person_asks_for_lifts_every_key_and_ends_capture() {
        let mut input = Input(input::Input::new());
        input.focus_gained(0);
        let at = At {
            point: Some((10.0, 10.0)),
            image: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0,
            },
            guest: Some((200, 100)),
        };
        input
            .0
            .button_down(crate::config::BUTTON_LEFT, 1_000_000_000, &at);
        input.0.key(0x04, true, false);
        input.0.take();
        assert!(input.is_captured());
        input.close_requested();
        assert!(!input.is_captured());
        let pointer = ToHost::Pointer {
            at: (20, 20),
            mode: (200, 100),
            buttons: 0,
        };
        assert_eq!(
            input.0.take(),
            vec![ToHost::Key(KeyEvent::up(0x04)), pointer]
        );
    }

    /// Each capture and release is logged with whether SDL holds the keyboard grab, and a release
    /// with whether the window lost the keyboard
    #[test]
    fn a_change_of_capture_says_whether_sdl_holds_the_grab() {
        assert_eq!(
            grab_outcome(true, true, true, None),
            "keyboard captured; SDL holds the keyboard grab, so the compositor's shortcuts go to \
             the guest"
        );
        assert_eq!(
            grab_outcome(true, false, false, None),
            "keyboard captured; SDL grabs the keyboard once the window has it"
        );
        assert_eq!(
            grab_outcome(true, true, false, Some("no inhibitor")),
            "keyboard captured, but SDL cannot grab the keyboard (no inhibitor): the compositor \
             keeps its own shortcuts"
        );
        assert_eq!(
            grab_outcome(false, false, false, None),
            "keyboard released: the window lost the keyboard, and the compositor's shortcuts are \
             its own again"
        );
        assert_eq!(
            grab_outcome(false, true, false, None),
            "keyboard released: the compositor's shortcuts are its own again"
        );
    }
}
