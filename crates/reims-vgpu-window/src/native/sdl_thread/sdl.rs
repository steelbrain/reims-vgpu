//! The SDL window's side on Linux and Windows ([`crate::sdl::os`]): Wayland's or X11's window, or
//! Win32's, drawn through Vulkan, on the library's UI thread.
//!
//! **The screen.** The screen a window opens on is SDL's display of that ID (the host's screens
//! are SDL's on Linux and Windows, `super::ui`), else SDL's primary display. Wayland lets no
//! client place a windowed window: the compositor puts it where it likes, and only a borderless
//! window is sure to be on its screen. SDL's answer to which screen a window is on is not used: on Wayland it names
//! the first screen for a window the compositor shows on another. A window resized for a guest's
//! mode fits the screen it opened on.
//!
//! **Windowed**: decorated and resizable, centred on its screen where the window system lets it
//! be placed (X11, Windows), its content shrunk to fit the screen's usable area less
//! [`TITLE_ROOM`] for the title bar and a panel, which Wayland does not tell a client. Its size
//! and its smallest are asked for in SDL's sizes (`crate::sdl::units`): points on Wayland, and on
//! X11 and Windows pixels, the desktop's content scale of them to the point.
//!
//! **The scale the guest is told** is the screen's ([`screen_scale`]): on Wayland the window's
//! pixels a point, and on X11 and Windows, where its drawable is a pixel a unit of SDL's sizes at
//! any scale, the content scale of the screen SDL has it on. The drawable, the image and the
//! pointer stay in SDL's sizes; the question a close asks in the window is laid out at the told
//! scale (`super::question`).
//!
//! **Borderless**: through the compositor's own full screen on its screen, over its panels. On
//! Windows the shell is also told the window is full screen, so a taskbar set to hide itself stays
//! down over it (`super::super::windows`, "Full screen to the shell"), until the window hides at
//! the display's end ([`close()`]).
//!
//! **The surface.** The window is made for Vulkan: its render thread makes the window's Vulkan
//! surface of SDL's window (`super::render`), so SDL's window goes only once that thread has let
//! its swapchain and surface go ([`close()`], `crate::sdl::shell::released`).
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
//! with whether SDL holds the grab ([`grab_keyboard`]). On Windows the grab is a low-level keyboard
//! hook on this thread, which takes the system's shortcuts while the window has the keyboard
//! (`super::super::windows`).

use std::ffi::{CStr, c_int};
use std::sync::Arc;

use sdl3_sys::everything::*;

use super::input::{self, At, Wheel};
use super::keymap;
use super::render::Target;
use crate::beyond::Hold;
use crate::capture::Capture;
use crate::close::{self, Answer};
use crate::geometry::{Insets, Rect};
use crate::handle::Window;
use crate::screens;
use crate::sdl::shell::{self, DEFAULT_MODE, Opened, Shell};
use crate::sdl::units::{self, Units};
use crate::sdl::{c_string, sdl_error};

/// Points of a screen's height kept for a window's title bar and a panel: Wayland gives a client
/// no work area, nor the size of the compositor's decorations, and Windows's work area leaves out
/// the task bar but not a window's title bar.
pub(crate) const TITLE_ROOM: f64 = 48.0;

/// Where an open window is, for placing others: nothing here, where each window is centred on its
/// screen or placed by the window system.
pub(crate) type Frame = std::convert::Infallible;

/// Where `shell`'s window is: nothing here.
pub(crate) fn frame_of(_shell: &Shell) -> Option<Frame> {
    None
}

/// What this side keeps of a window.
pub(crate) struct State {
    /// The screen it opened on: SDL's answer to which screen a window is on is not used (the
    /// module doc).
    display: SDL_DisplayID,
    /// The question a close asks in the window, while it is open ([`ask`]).
    #[cfg(target_os = "linux")]
    question: Option<super::question::Question>,
    /// The shell's mark of the window as full screen, while SDL has it so.
    #[cfg(target_os = "windows")]
    full_screen: Option<super::super::windows::FullScreenMark>,
}

/// A window's input: SDL's events, queued until the shell sends them (`super::input`).
pub(crate) struct Input(input::Input);

impl Input {
    /// The window is closing: every key and button held is lifted, and capture ends.
    pub(crate) fn closing(&mut self) {
        self.0.closing();
    }

    /// The person closed the window: every key and button held goes up, and capture ends, before
    /// the person is asked whether to turn the VM off, as on macOS.
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

/// The content size, in SDL's sizes, of a windowed window for a guest mode `mode`, on screen
/// `display` (the module doc): at least `shell::MIN_CONTENT` points.
fn windowed_size(display: SDL_DisplayID, mode: (u16, u16)) -> (i32, i32) {
    let mut usable = SDL_Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };
    // SAFETY: SDL's video runs on this thread.
    let known = unsafe { SDL_GetDisplayUsableBounds(display, &mut usable) };
    let usable = known.then_some((usable.w, usable.h));
    units::windowed_size(mode, usable, TITLE_ROOM, units_of(display))
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
    let size = windowed_size(display, mode);
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
    // SAFETY: `sdl` is the window just made, on this thread; a borderless one is full screen once
    // shown (SDL applies the full screen asked of a hidden window as it shows it).
    #[cfg(target_os = "windows")]
    let full_screen = unsafe { super::super::windows::FullScreenMark::of(sdl, window.title()) };
    window.surface().set_target(Target(sdl));
    window.surface().set_visible(true);
    log::debug!(
        target: "reims_vgpu_window",
        "window {:?} on screen {}",
        window.title(),
        display.0
    );
    let callbacks = window.callbacks();
    Some(Opened {
        sdl,
        asked: size,
        input: Input(
            input::Input::with_links(
                callbacks
                    .as_ref()
                    .and_then(|callbacks| callbacks.beyond.clone()),
                callbacks.and_then(|callbacks| callbacks.motion),
            )
            .with_capture(Capture::with_setting(crate::env::settings().capture)),
        ),
        state: State {
            display,
            #[cfg(target_os = "linux")]
            question: None,
            #[cfg(target_os = "windows")]
            full_screen,
        },
    })
}

/// Resizes a window following the guest's mode for `mode`, as it would open on its screen.
pub(crate) fn follow(shell: &mut Shell, mode: (u16, u16), _taken: &[Frame]) {
    let size = windowed_size(shell.os.display, mode);
    if size != shell.asked {
        shell.asked = size;
        // SAFETY: the window is SDL's, on this thread.
        unsafe { SDL_SetWindowSize(shell.sdl, size.0, size.1) };
    }
}

/// After the person's resize to `size`, makes the window the next smaller size whose pixels the
/// guest is offered whole (`units::offered_size`): the image then fills the window, where a size
/// between two leaves up to 15 columns and a row of black beside it. A maximized, full-screen or
/// borderless window keeps the size the system gave it. On Windows alone: the person's drag runs
/// in a loop of Windows' own, and its size events reach the UI thread once it is over, so the
/// size set here follows the drag and never fights it.
#[cfg(windows)]
pub(crate) fn fit_offered(shell: &Shell, size: (i32, i32)) {
    // SAFETY: the window is SDL's, on this thread.
    let (flags, display) = unsafe {
        (
            SDL_GetWindowFlags(shell.sdl),
            SDL_GetDisplayForWindow(shell.sdl),
        )
    };
    if (flags & SDL_WINDOW_MAXIMIZED).0 != 0
        || (flags & SDL_WINDOW_FULLSCREEN).0 != 0
        || (flags & SDL_WINDOW_BORDERLESS).0 != 0
    {
        return;
    }
    let fitted = units::offered_size(size, units_of(display));
    if fitted != size {
        // SAFETY: as above.
        unsafe { SDL_SetWindowSize(shell.sdl, fitted.0, fitted.1) };
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

/// The pixels a host point of the screen the window is on, which its guest is told
/// (`crate::geometry::Layout::screen_scale`), for a window of `density` pixels a unit of SDL's
/// sizes ([`metrics`]): the density on Wayland, whose sizes are points; on X11 and Windows, whose
/// sizes are pixels and whose windows have a density of 1 at any scale, SDL's display scale of
/// the window, its screen's content scale (`crate::sdl::units`): 1.5 on a Windows screen at 150%.
/// SDL tells a change of it (`SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED`), which lays the window out
/// again (`crate::sdl::shell`).
pub(crate) fn screen_scale(shell: &Shell, density: f64) -> f64 {
    // SAFETY: the window is SDL's, on this thread.
    let display_scale = unsafe { SDL_GetWindowDisplayScale(shell.sdl) };
    units::pixels_per_point(density, display_scale)
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
        ..at_no_point(shell)
    }
}

/// What the window knows, with no pointer event.
fn at_no_point(shell: &Shell) -> At {
    At {
        point: None,
        image: shell.layout.image,
        guest: shell.window.mode(),
        cursor: shell.window.shown_cursor_at(),
    }
}

/// The guest showed or hid its cursor: the input may let go of the pointer driving the guest's
/// mouse ([`input::Input::cursor_changed`]).
pub(crate) fn cursor_changed(shell: &mut Shell) {
    let at = at_no_point(shell);
    shell.input.0.cursor_changed(&at);
    apply_hold(shell);
}

/// An input event of type `kind` for the window: the pointer, a button, the wheel or a key, to
/// the window's input, or to the question open in the window ([`ask`]). Whether capture changed.
pub(crate) fn input_event(shell: &mut Shell, kind: u32, event: &SDL_Event) -> bool {
    #[cfg(target_os = "linux")]
    if shell.os.question.is_some() {
        question_event(shell, kind, event);
        return false;
    }
    match kind {
        k if k == SDL_EVENT_MOUSE_MOTION.0 => {
            // SAFETY: a motion event.
            let motion = unsafe { event.motion };
            let point = (motion.x, motion.y);
            let at = at(shell, point);
            if shell.input.0.is_locked() {
                // The pointer is locked: SDL reports its motion alone.
                let by = (f64::from(motion.xrel), f64::from(motion.yrel));
                shell.input.0.moved_by(by, &at);
            } else {
                shell.input.0.moved(&at);
                shell.pointer_over((f64::from(point.0), f64::from(point.1)));
            }
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

/// Locks the host's pointer as the captured pointer goes past the image's edge, or lets it go
/// where the pointer came back, as the input asks ([`input::Input::take_hold`]): SDL's relative
/// mode, which on Wayland locks the pointer (pointer-constraints' lock) and reports its motion
/// (relative-pointer), hiding it; on X11 it holds it in the window, hidden, and reports its raw
/// motion. Letting go warps the pointer to the point first, so SDL puts it there as the lock
/// ends (on Wayland the lock's cursor position hint, or pointer-warp). A lock SDL refuses gives
/// the pointer back to the window, where it was held.
///
/// Gone past the edge, the motion is the host's pointer's, in the window's points, scaled as
/// the host moves its pointer (`SDL_MOUSE_RELATIVE_SYSTEM_SCALE` 1); driving the guest's mouse,
/// it is the device's own counts (0: on Wayland relative-pointer's unaccelerated motion), which
/// the guest scales as it scales a mouse's.
pub(crate) fn apply_hold(shell: &mut Shell) {
    let Some(hold) = shell.input.0.take_hold() else {
        return;
    };
    match hold {
        Hold::Lock | Hold::Raw => {
            let raw = hold == Hold::Raw;
            let scale = if raw { c"0" } else { c"1" };
            // SAFETY: the window is SDL's, on this thread; both strings are NUL-terminated, and
            // SDL copies them.
            let locked = unsafe {
                SDL_SetHint(c"SDL_MOUSE_RELATIVE_SYSTEM_SCALE".as_ptr(), scale.as_ptr());
                SDL_SetWindowRelativeMouseMode(shell.sdl, true)
            };
            log::debug!(
                target: "reims_vgpu_window",
                "window {:?}: the captured pointer {} (locked: {locked})",
                shell.window.title(),
                if raw {
                    "drives the guest's mouse"
                } else {
                    "goes past the image's edge"
                }
            );
            if !locked {
                shell::refused(
                    if raw {
                        "no pointer lock, so a game's mouse stops at the image's edge"
                    } else {
                        "no pointer lock, so the pointer stays at the image's edge"
                    },
                    shell.window.title(),
                );
                shell.input.0.lock_refused();
                apply_hold(shell);
            }
        }
        Hold::Unlock(point) => {
            // SAFETY: as above; in relative mode the warp sets where the pointer goes as it ends.
            unsafe {
                SDL_WarpMouseInWindow(shell.sdl, point.0 as f32, point.1 as f32);
                SDL_SetWindowRelativeMouseMode(shell.sdl, false);
            }
            log::debug!(
                target: "reims_vgpu_window",
                "window {:?}: the pointer is back at {point:?}",
                shell.window.title()
            );
            shell.refresh_cursor();
        }
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
/// decorations, libdecor's frame, an X11 window manager's frame window, Windows's non-client
/// frame), where the pointer held over the image cannot go. A window SDL made not resizable would also be held on X11 at its
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

/// The button of the question's Turn Off (`crate::close`).
const TURN_OFF_ID: c_int = 1;

/// The button of the question's Cancel, the default, and the answer where no button is chosen.
const CANCEL_ID: c_int = 0;

/// Asks the person whether to turn the VM off, for windows `ids` (`crate::close`), over `shell`'s
/// window: on Linux in the window itself, which the library draws (`ask_in_the_window`), and in
/// SDL's message box where the window cannot draw it; on Windows in SDL's message box, a task
/// dialog the window owns ([`ask_in_a_dialog`]).
///
/// From the close to the answer, nothing of the window's input reaches the guest
/// ([`input::Input::question_opened`]): the events at hand before the question shows, a key that
/// went down after the close's lift among them, the keys and clicks the question takes, and the
/// ups of the keys a dialog takes, which come once it is answered.
pub(crate) fn ask(shell: &mut Shell, ids: Vec<u64>) {
    shell.input.0.question_opened();
    #[cfg(target_os = "linux")]
    let Err(ids) = ask_in_the_window(shell, ids) else {
        return;
    };
    ask_in_a_dialog(shell.window.id(), ids);
}

/// Asks the question in `shell`'s window (`super::question`): drawn over the frame by its render
/// thread, the window's keys and clicks going to it ([`question_event`]), the host's arrow over it
/// ([`asks`]); the window restored where it is minimized, and raised. `ids` back where the window
/// has no drawable to draw it in yet. A render thread that can draw no more hands the question to
/// SDL's message box ([`overlay_unshown`]).
#[cfg(target_os = "linux")]
fn ask_in_the_window(shell: &mut Shell, ids: Vec<u64>) -> Result<(), Vec<u64>> {
    let layout = shell.layout;
    if layout.drawable.0 == 0 || layout.drawable.1 == 0 {
        return Err(ids);
    }
    let question = super::question::Question::new(ids, &layout);
    shell.window.surface().set_overlay(Some(question.overlay()));
    shell.os.question = Some(question);
    // SAFETY: the window is SDL's, on this thread.
    unsafe {
        if (SDL_GetWindowFlags(shell.sdl) & SDL_WINDOW_MINIMIZED).0 != 0 {
            SDL_RestoreWindow(shell.sdl);
        }
        SDL_RaiseWindow(shell.sdl);
    }
    shell.refresh_cursor();
    log::info!(
        target: "reims_vgpu_window",
        "window {:?}: the question before turning the VM off shows in the window",
        shell.window.title()
    );
    Ok(())
}

/// An input event of type `kind` for the question open in `shell`'s window: its keys, and the
/// left button's clicks at their pixels of the drawable (`super::question`). A focus moved draws
/// it again; an answer ends it ([`answered_in_the_window`]).
#[cfg(target_os = "linux")]
fn question_event(shell: &mut Shell, kind: u32, event: &SDL_Event) {
    use super::question::Step;
    let scale = shell.layout.scale;
    let Some(question) = shell.os.question.as_mut() else {
        return;
    };
    let step = match kind {
        k if k == SDL_EVENT_KEY_DOWN.0 || k == SDL_EVENT_KEY_UP.0 => {
            // SAFETY: a key event.
            let key = unsafe { event.key };
            keymap::usb_usage(key.scancode.0).map_or(Step::Nothing, |usage| {
                question.key(usage, key.down, key.repeat)
            })
        }
        k if k == SDL_EVENT_MOUSE_BUTTON_DOWN.0 || k == SDL_EVENT_MOUSE_BUTTON_UP.0 => {
            // SAFETY: a button event.
            let b = unsafe { event.button };
            if input::button_bit(b.button) == crate::config::BUTTON_LEFT {
                let point = (f64::from(b.x) * scale, f64::from(b.y) * scale);
                question.click(b.down, point)
            } else {
                Step::Nothing
            }
        }
        _ => Step::Nothing,
    };
    match step {
        Step::Nothing => {}
        Step::Redraw => shell.window.surface().set_overlay(Some(question.overlay())),
        Step::Answer(answer) => answered_in_the_window(shell, answer),
    }
}

/// The person answered the question in `shell`'s window: it goes, the window's input goes on, and
/// the answer goes to [`shell::answered`], posted, to run with no window borrowed.
#[cfg(target_os = "linux")]
fn answered_in_the_window(shell: &mut Shell, answer: Answer) {
    let Some(question) = shell.os.question.take() else {
        return;
    };
    shell.window.surface().set_overlay(None);
    shell.input.0.question_answered();
    shell.refresh_cursor();
    let ids = question.ids;
    super::ui::post(move || shell::answered(&ids, answer));
}

/// Window `id`'s render thread can draw no more (its GPU device lost, or no swapchain), so the
/// question open in it does not show (`super::render`): it is asked in SDL's message box instead
/// ([`ask_in_a_dialog`]), the input held as it was.
#[cfg(target_os = "linux")]
pub(super) fn overlay_unshown(id: u64) {
    let ids = shell::with_shell(id, |s| {
        let question = s.os.question.take()?;
        s.window.surface().set_overlay(None);
        s.refresh_cursor();
        Some(question.ids)
    })
    .flatten();
    if let Some(ids) = ids {
        log::warn!(
            target: "reims_vgpu_window",
            "the window draws no more: the question before turning the VM off shows in SDL's \
             message box"
        );
        ask_in_a_dialog(id, ids);
    }
}

/// Whether a question is open in `shell`'s window, over which the host's arrow shows.
#[cfg(target_os = "linux")]
pub(crate) fn asks(shell: &Shell) -> bool {
    shell.os.question.is_some()
}

/// Whether a question is open in the window: none on Windows, whose question is a dialog.
#[cfg(not(target_os = "linux"))]
pub(crate) fn asks(_shell: &Shell) -> bool {
    false
}

/// The window was laid out again: the question open in it is laid out for the new drawable and
/// drawn again where that changed it.
#[cfg(target_os = "linux")]
pub(crate) fn laid_out(shell: &mut Shell) {
    if let Some(question) = shell.os.question.as_mut()
        && question.relayout(&shell.layout)
    {
        shell.window.surface().set_overlay(Some(question.overlay()));
    }
}

/// The window was laid out again: nothing more on Windows, whose question is a dialog.
#[cfg(not(target_os = "linux"))]
pub(crate) fn laid_out(_shell: &mut Shell) {}

/// Asks the question for windows `ids` in SDL's message box over window `id`: on Linux zenity's
/// question where zenity is installed, else SDL's own X11 dialog, which the window owns on X11 and
/// under XWayland; on Windows a task dialog the window owns. Read from SDL 3.4.18's code: zenity's
/// question is a window of its own, which SDL waits for, and on Wayland SDL tries nothing else.
/// SDL's message box holds the thread it runs on until it is answered, the UI thread here, which
/// made the window, as SDL asks: the other windows' events and the work posted wait meanwhile,
/// while the render threads draw on. So it is posted, to run after the events at hand with no
/// window borrowed; the answer goes to [`shell::answered`], after the window's input goes on. A
/// window whose display ended before the question showed asks nothing.
fn ask_in_a_dialog(id: u64, ids: Vec<u64>) {
    super::ui::post(move || {
        let parent = shell::with_shell(id, |s| (!s.closed).then_some(s.sdl)).flatten();
        let answer = match parent {
            Some(parent) => question(parent),
            None => Answer::Cancel,
        };
        shell::with_shell(id, |s| s.input.0.question_answered());
        shell::answered(&ids, answer);
    });
}

/// Shows the question over `parent`, and waits for its answer ([`answer_of`]). Where SDL shows
/// none (on Linux no zenity on Wayland, or no X server), that is logged, and the VM runs on.
fn question(parent: *mut SDL_Window) -> Answer {
    let title = c_string(close::QUESTION);
    let message = c_string(close::DETAIL);
    let turn_off = c_string(close::TURN_OFF);
    let cancel = c_string(close::CANCEL);
    let buttons = question_buttons(&turn_off, &cancel);
    let data = SDL_MessageBoxData {
        flags: SDL_MESSAGEBOX_WARNING | SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT,
        window: parent,
        title: title.as_ptr(),
        message: message.as_ptr(),
        numbuttons: 2,
        buttons: buttons.as_ptr(),
        colorScheme: std::ptr::null(),
    };
    let mut chosen = CANCEL_ID;
    // SAFETY: SDL's video runs on this thread, which made `parent`, a window open on it; the
    // strings and the buttons outlive the call.
    let shown = unsafe { SDL_ShowMessageBox(&data, &mut chosen) };
    if !shown {
        log::warn!(
            target: "reims_vgpu_window",
            "the question before turning the VM off does not show ({}): the VM runs on",
            sdl_error()
        );
    }
    answer_of(shown, chosen)
}

/// The question's buttons, left to right as the OS's own dialogs place them: on Linux Cancel,
/// then Turn Off; on Windows Turn Off, then Cancel. Cancel is the default, for Return and for
/// Escape (zenity takes neither flag).
fn question_buttons(turn_off: &CStr, cancel: &CStr) -> [SDL_MessageBoxButtonData; 2] {
    let turn_off = SDL_MessageBoxButtonData {
        flags: SDL_MessageBoxButtonFlags(0),
        buttonID: TURN_OFF_ID,
        text: turn_off.as_ptr(),
    };
    let cancel = SDL_MessageBoxButtonData {
        flags: SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT,
        buttonID: CANCEL_ID,
        text: cancel.as_ptr(),
    };
    if cfg!(target_os = "windows") {
        [turn_off, cancel]
    } else {
        [cancel, turn_off]
    }
}

/// The answer of SDL's message box, `shown` or not, whose chosen button is `chosen`: Turn Off
/// only by its button; any other, the question dismissed included (`chosen` left as it was, or
/// -1 from Windows' task dialog), is Cancel.
fn answer_of(shown: bool, chosen: c_int) -> Answer {
    match (shown, chosen) {
        (false, _) => Answer::Unshown,
        (true, TURN_OFF_ID) => Answer::TurnOff,
        (true, _) => Answer::Cancel,
    }
}

/// The display ended: the window hides, and a question open in it ends as cancelled; on Windows
/// the shell's full-screen mark goes first. SDL's window goes once the render thread has let go
/// (`crate::sdl::shell::released`), never before.
pub(crate) fn close(shell: &mut Shell) -> bool {
    #[cfg(target_os = "linux")]
    if let Some(question) = shell.os.question.take() {
        shell.window.surface().set_overlay(None);
        let ids = question.ids;
        super::ui::post(move || shell::answered(&ids, Answer::Cancel));
    }
    #[cfg(target_os = "windows")]
    drop(shell.os.full_screen.take());
    // SAFETY: the window is SDL's, on this thread.
    unsafe { SDL_HideWindow(shell.sdl) };
    false
}

/// The window goes: nothing of this side's stays.
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
            cursor: None,
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

    /// The question's default, for Return and for Escape, is Cancel, and Turn Off is no default
    #[test]
    fn the_question_s_default_is_cancel() {
        let buttons = question_buttons(c"Turn Off", c"Cancel");
        let defaults =
            SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT;
        let cancel = buttons.iter().find(|b| b.buttonID == CANCEL_ID).unwrap();
        let turn_off = buttons.iter().find(|b| b.buttonID == TURN_OFF_ID).unwrap();
        assert_eq!(cancel.flags.0, defaults.0);
        assert_eq!(turn_off.flags.0, 0);
        // SAFETY: the texts are the C strings above, which live to the end of the test.
        let (cancel, turn_off) =
            unsafe { (CStr::from_ptr(cancel.text), CStr::from_ptr(turn_off.text)) };
        assert_eq!((cancel, turn_off), (c"Cancel", c"Turn Off"));
    }

    /// Only Turn Off's button is Turn Off; the question dismissed, or any other button, is
    /// Cancel, and a question that did not show is said so
    #[test]
    fn an_answer_is_turn_off_only_by_its_button() {
        assert_eq!(answer_of(true, TURN_OFF_ID), Answer::TurnOff);
        assert_eq!(answer_of(true, CANCEL_ID), Answer::Cancel);
        assert_eq!(answer_of(true, -1), Answer::Cancel);
        assert_eq!(answer_of(false, TURN_OFF_ID), Answer::Unshown);
        assert_eq!(answer_of(false, CANCEL_ID), Answer::Unshown);
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
