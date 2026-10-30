//! A display's SDL window, on SDL's thread: its opening, size, title and layout, its events, and
//! its end. The windows open are kept here, by the [`Window`]'s id, for SDL's thread alone. What
//! each OS does its own way (where the window opens and how it follows the guest's mode, its
//! surface, its input, its end) is its [`os`]'s.
//!
//! **Opening.** A window opens on the screen [`crate::screens::screen_for_port`] picks for the
//! display's port, else on the main screen, so the screen whose density the guest was told of is
//! the one it opens on. Windowed (the default), its content is the guest's mode, else the first
//! mode the display offers, else [`DEFAULT_MODE`], at one guest point per host point, shrunk to fit
//! the screen ([`crate::geometry::initial_content_size`]), on whole pixels at the screen's scale,
//! and never below [`MIN_CONTENT`]. Borderless (`REIMS_VGPU_WINDOW_BORDERLESS=1`), it fills its
//! screen.
//!
//! **Following the guest's mode.** Until the person resizes it (an edge, tiling, maximizing or
//! zooming, full screen), a windowed window follows the guest's mode: each mode change resizes its
//! content as it would open. A resize the library did not ask for, more than a point off the size
//! it asked for ([`is_asked`]), is the person's, which ends the following; from then on a mode
//! change re-fits the image and the window keeps its size.
//!
//! **The layout.** Whenever the window's size, its scale or the guest's mode changes, the image is
//! laid out again ([`geometry::layout`]) over the window's points at its pixels per point
//! ([`os::metrics`]), inside the room the OS keeps clear ([`os::insets`]), and the render thread
//! draws a changed one: the drawable is the window's size in pixels, one pixel per screen pixel at
//! any scale.
//!
//! **Events.** Closing the window (its close button) tells the host, which ends the VM, and leaves
//! the window open: it closes when the display is released ([`close`]). Losing the keyboard
//! releases the input. The window shown, exposed, hidden, minimized, restored or covered tells the
//! render thread whether to draw ([`shows`]): nothing is presented into a window the person cannot
//! see. On Wayland a covered window is one the compositor suspends, whose FIFO present would wait
//! for a frame callback that never comes, on the queue the guest's frames use too.
//!
//! **Capture.** While the input is captured, SDL holds the pointer over the image, SDL's mouse
//! rect ([`mouse_rect`]), set again whenever the layout changes, and frees it when capture ends.
//! SDL keeps the rect on Wayland with the compositor's pointer constraints, on X11 with XFixes
//! pointer barriers, and on macOS with the window's `mouseConfinementRect`, each only while the
//! window has the keyboard (read from SDL 3.4.18's code). Where SDL has no mouse rect (an X11
//! build without XFixes), its mouse grab holds the pointer in the whole window. The window's OS
//! decides whether it may be resized while captured ([`os::hold_size`]), asked at each change of
//! capture and as the window leaves full screen.
//!
//! **The end.** At the display's end the window lets its input go, and the keyboard's grab, the
//! pointer's hold and the size's hold with it ([`ended`]), and hides; SDL's window is
//! destroyed once the OS lets it go ([`os::close`]): on Linux once its render thread has let its
//! swapchain and Vulkan surface go too ([`released`]), whichever of the two comes last.

use std::cell::RefCell;
use std::sync::Arc;

use sdl3_sys::everything::*;

use super::cursor::CursorCache;
use super::os::{self, Input};
use super::{c_string, sdl_error};
use crate::geometry::{self, Layout, Rect};
use crate::handle::Window;

/// The guest's mode a window is sized for before the guest sets one or the display offers one.
pub(crate) const DEFAULT_MODE: (u16, u16) = (1920, 1080);

/// The smallest a windowed window's content may be made, in points.
pub(crate) const MIN_CONTENT: (i32, i32) = (320, 200);

/// A window open on SDL's thread.
pub(crate) struct Shell {
    /// The window.
    pub(crate) window: Arc<Window>,
    /// SDL's window, which only SDL's thread uses.
    pub(crate) sdl: *mut SDL_Window,
    /// SDL's ID of it, which its events name.
    sdl_id: u32,
    /// Whether it follows the guest's mode (the module doc).
    pub(crate) follows_guest: bool,
    /// The size in points the library last asked for: a resize to another is the person's.
    pub(crate) asked: (i32, i32),
    /// The layout, once laid out.
    pub(crate) layout: Layout,
    /// The window's pixels per point.
    density: f64,
    /// Its input.
    pub(crate) input: Input,
    /// What its OS keeps of it.
    pub(crate) os: os::State,
    /// The guest's cursor, built for SDL.
    cursor: CursorCache,
    /// Whether the pointer is over the image, which shows the guest's cursor.
    over_image: bool,
    /// The display has ended: the window is hidden, and goes once its OS lets it go.
    pub(crate) closed: bool,
    /// Whether the events of the batch SDL is handing over show the window or hide it, the last
    /// of them, applied once the batch is over ([`batch_over`]); `None` where none said.
    shown: Option<bool>,
    /// Its render thread has let its surface go.
    released: bool,
}

/// What an OS made of a window that opened.
pub(crate) struct Opened {
    /// SDL's window, shown.
    pub(crate) sdl: *mut SDL_Window,
    /// The size in points the library asked for.
    pub(crate) asked: (i32, i32),
    /// The window's input.
    pub(crate) input: Input,
    /// What the OS keeps of it.
    pub(crate) state: os::State,
}

thread_local! {
    /// The windows open, on SDL's thread.
    static OPEN: RefCell<Vec<Shell>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on window `id`'s shell, if it is open.
pub(crate) fn with_shell<R>(id: u64, f: impl FnOnce(&mut Shell) -> R) -> Option<R> {
    OPEN.with(|open| {
        let mut open = open.borrow_mut();
        open.iter_mut().find(|s| s.window.id() == id).map(f)
    })
}

/// What `f` answers of each open window, in the order they opened.
pub(crate) fn each_open<R>(mut f: impl FnMut(&Shell) -> Option<R>) -> Vec<R> {
    OPEN.with(|open| open.borrow().iter().filter_map(&mut f).collect())
}

/// The id of the window SDL calls `sdl_id`.
fn window_of(sdl_id: u32) -> Option<u64> {
    each_open(|s| (s.sdl_id == sdl_id).then(|| s.window.id()))
        .first()
        .copied()
}

/// Whether a window resized to `size` points is at the size the library asked for, `asked`:
/// within a point each way, which a compositor's rounding at a fractional scale takes (KWin gives
/// 1920×1199 points for 1920×1200 at 190%).
fn is_asked(size: (i32, i32), asked: (i32, i32)) -> bool {
    (size.0 - asked.0).abs() <= 1 && (size.1 - asked.1).abs() <= 1
}

/// Opens `window`'s SDL window (the module doc), unless the window has ended.
pub(crate) fn open(window: &Arc<Window>) {
    if window.is_closed() || with_shell(window.id(), |_| ()).is_some() {
        return;
    }
    let borderless = crate::env::settings().borderless;
    let taken = each_open(os::frame_of);
    let Some(opened) = os::open(window, borderless, &taken) else {
        return;
    };
    // SAFETY: SDL's window, just made on this thread.
    let sdl_id = unsafe { SDL_GetWindowID(opened.sdl).0 };
    OPEN.with(|open| {
        open.borrow_mut().push(Shell {
            window: Arc::clone(window),
            sdl: opened.sdl,
            sdl_id,
            follows_guest: !borderless && !window.keeps_its_size(),
            asked: opened.asked,
            layout: Layout::default(),
            density: 1.0,
            input: opened.input,
            os: opened.state,
            cursor: CursorCache::default(),
            over_image: false,
            closed: false,
            shown: None,
            released: false,
        });
    });
    with_shell(window.id(), |shell| {
        shell.relayout();
        shell.refresh_title();
    });
    log::info!(
        target: "reims_vgpu_window",
        "window {:?} open: {}, {}x{} points",
        window.title(),
        if borderless { "borderless" } else { "windowed" },
        opened.asked.0,
        opened.asked.1
    );
}

/// Window `id`'s display ended: its input lets go ([`ended`]) and it hides; SDL's window goes
/// once its OS lets it go ([`os::close`]).
pub(crate) fn close(id: u64) {
    let gone = with_shell(id, |shell| {
        ended(shell);
        shell.closed = true;
        os::close(shell) || shell.released
    });
    if gone == Some(true) {
        destroy(id);
    }
}

/// Window `id`'s render thread let its surface go: SDL's window goes where the display has ended.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only Linux's render thread holds SDL's window")
)]
pub(crate) fn released(id: u64) {
    let gone = with_shell(id, |shell| {
        shell.released = true;
        shell.closed
    });
    if gone == Some(true) {
        destroy(id);
    }
}

/// Destroys window `id`'s SDL window and forgets it.
fn destroy(id: u64) {
    let shell = OPEN.with(|open| {
        let mut open = open.borrow_mut();
        let at = open.iter().position(|s| s.window.id() == id)?;
        Some(open.remove(at))
    });
    if let Some(shell) = shell {
        let sdl = shell.sdl;
        let title = shell.window.title().to_owned();
        os::destroy(shell);
        // SAFETY: the window is SDL's, on this thread; its OS and render thread have let go.
        unsafe { SDL_DestroyWindow(sdl) };
        log::info!(target: "reims_vgpu_window", "window {title:?} closed");
    }
}

/// The guest of window `id` set a mode: a window following the guest's mode is resized for it
/// (the module doc), and the image is laid out again.
pub(crate) fn guest_mode_changed(id: u64) {
    let taken = each_open(|s| (s.window.id() != id).then(|| os::frame_of(s)).flatten());
    with_shell(id, |shell| {
        if shell.follows_guest
            && !shell.closed
            && let Some(mode) = shell.window.mode()
        {
            os::follow(shell, mode, &taken);
        }
        shell.relayout();
    });
}

/// The guest of window `id` changed its cursor or showed or hid it.
pub(crate) fn cursor_changed(id: u64) {
    with_shell(id, |shell| {
        shell.window.take_cursor_refresh();
        shell.refresh_cursor();
    });
}

/// The screens changed: every window takes its OS's say ([`os::screens_changed`]) and lays out
/// again.
pub(crate) fn screens_changed() {
    let ids = each_open(|s| Some(s.window.id()));
    for id in ids {
        with_shell(id, |shell| {
            if !shell.closed {
                os::screens_changed(shell);
            }
            shell.relayout();
        });
    }
}

/// SDL has handed over every event it had queued: each window that an event of the batch showed
/// or hid is drawn or not as the last of them said. SDL sends a suspended Wayland window whose
/// size changes `EXPOSED` and then `OCCLUDED` back to back, so a window woken at the first would
/// present into a surface the compositor does not show, which under FIFO blocks the present
/// until it shows again.
pub(crate) fn batch_over() {
    OPEN.with(|open| {
        for shell in open.borrow_mut().iter_mut() {
            if let Some(shown) = shell.shown.take()
                && !shell.closed
            {
                shell.window.surface().set_visible(shown);
            }
        }
    });
}

/// Whether the window shows after an event of type `kind`, where `before` is what the batch's
/// earlier events said ([`shows`]): the event's say, else the earlier one.
fn settled(before: Option<bool>, kind: u32) -> Option<bool> {
    shows(kind).or(before)
}

/// An SDL event for a window: its window's, else nothing.
pub(crate) fn event(event: &SDL_Event) {
    // SAFETY: every event starts with its type, and each arm reads the variant its type names.
    let kind = unsafe { event.r#type };
    let sdl_id = match kind {
        k if (SDL_EVENT_WINDOW_FIRST.0..=SDL_EVENT_WINDOW_LAST.0).contains(&k) => unsafe {
            event.window.windowID.0
        },
        k if k == SDL_EVENT_KEY_DOWN.0 || k == SDL_EVENT_KEY_UP.0 => unsafe {
            event.key.windowID.0
        },
        k if k == SDL_EVENT_MOUSE_MOTION.0 => unsafe { event.motion.windowID.0 },
        k if k == SDL_EVENT_MOUSE_BUTTON_DOWN.0 || k == SDL_EVENT_MOUSE_BUTTON_UP.0 => unsafe {
            event.button.windowID.0
        },
        k if k == SDL_EVENT_MOUSE_WHEEL.0 => unsafe { event.wheel.windowID.0 },
        _ => return,
    };
    let Some(id) = window_of(sdl_id) else {
        return;
    };
    with_shell(id, |shell| {
        if !shell.closed {
            shell.event(kind, event);
        }
    });
}

impl Shell {
    /// Handles `event`, of type `kind`, for this window.
    fn event(&mut self, kind: u32, event: &SDL_Event) {
        let capture_changed = match kind {
            k if k == SDL_EVENT_WINDOW_RESIZED.0 => {
                // SAFETY: a window event.
                let size = unsafe { (event.window.data1, event.window.data2) };
                if self.follows_guest && !is_asked(size, self.asked) {
                    self.follows_guest = false;
                    log::debug!(
                        target: "reims_vgpu_window",
                        "window {:?} sized by the person: it keeps its size",
                        self.window.title()
                    );
                }
                self.relayout();
                false
            }
            k if k == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED.0
                || k == SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED.0
                || k == SDL_EVENT_WINDOW_DISPLAY_CHANGED.0
                || k == SDL_EVENT_WINDOW_SAFE_AREA_CHANGED.0 =>
            {
                self.relayout();
                false
            }
            k if k == SDL_EVENT_WINDOW_MAXIMIZED.0 || k == SDL_EVENT_WINDOW_ENTER_FULLSCREEN.0 => {
                self.follows_guest = false;
                false
            }
            k if k == SDL_EVENT_WINDOW_LEAVE_FULLSCREEN.0 => {
                os::hold_size(self);
                false
            }
            k if shows(k).is_some() => {
                self.shown = settled(self.shown, k);
                false
            }
            k if k == SDL_EVENT_WINDOW_CLOSE_REQUESTED.0 => {
                self.close_requested();
                false
            }
            k if k == SDL_EVENT_WINDOW_FOCUS_GAINED.0 => {
                // SAFETY: a window event.
                self.input.focus_gained(unsafe { event.window.timestamp });
                false
            }
            k if k == SDL_EVENT_WINDOW_FOCUS_LOST.0 => self.input.focus_lost(),
            k if k == SDL_EVENT_WINDOW_MOUSE_ENTER.0 || k == SDL_EVENT_WINDOW_MOUSE_LEAVE.0 => {
                // SDL's cursor is the process's: another window may have hidden it.
                self.over_image = false;
                self.refresh_cursor();
                false
            }
            _ => os::input_event(self, kind, event),
        };
        self.flush();
        if capture_changed {
            self.capture_changed();
        }
    }

    /// The person closed the window: the input lets go ([`os::Input::close_requested`]), and the
    /// host is told, which ends the VM ([`close_asked`]). The window stays until the display's
    /// release closes it.
    pub(crate) fn close_requested(&mut self) {
        log::info!(
            target: "reims_vgpu_window",
            "window {:?}: closed by the person",
            self.window.title()
        );
        close_asked(self);
    }

    /// Sends what the input queued to the host's callbacks, while the window has them.
    pub(crate) fn flush(&mut self) {
        self.input.flush(&self.window);
    }

    /// The input was captured or released: the holds follow ([`Shell::holds_follow_capture`]),
    /// and the title and the cursor too.
    pub(crate) fn capture_changed(&mut self) {
        self.holds_follow_capture();
        self.refresh_title();
        self.refresh_cursor();
    }

    /// The holds follow the capture: the OS grabs the keyboard or lets it go, SDL holds the
    /// pointer over the image or frees it, and the OS holds the window's size or lets it go.
    fn holds_follow_capture(&mut self) {
        os::grab_keyboard(self, self.input.is_captured());
        self.confine_pointer();
        os::hold_size(self);
    }

    /// Holds the pointer over the image while captured, and frees it when not (the module doc):
    /// SDL's mouse rect, else its mouse grab of the whole window where SDL refuses the rect.
    fn confine_pointer(&self) {
        let rect = self
            .input
            .pointer_confinement(self.layout.image)
            .map(mouse_rect);
        let rect_ptr = rect.as_ref().map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: the window is SDL's, on this thread; SDL copies the rect.
        let ruled = unsafe { SDL_SetWindowMouseRect(self.sdl, rect_ptr) };
        let grab = rect.is_some() && !ruled;
        if grab {
            refused(
                "no mouse rect, so the pointer is held in the whole window",
                self.window.title(),
            );
        }
        // SAFETY: as above.
        unsafe { SDL_SetWindowMouseGrab(self.sdl, grab) };
    }

    /// Sets the title: the window's, then what capture adds.
    fn refresh_title(&self) {
        let title = c_string(&format!(
            "{}{}",
            self.window.title(),
            self.input.title_suffix()
        ));
        // SAFETY: the window is SDL's, on this thread; the title is copied.
        unsafe { SDL_SetWindowTitle(self.sdl, title.as_ptr()) };
    }

    /// Lays the image out again over the window's size and scale (the module doc), and tells the
    /// render thread.
    pub(crate) fn relayout(&mut self) {
        let Some((bounds, density)) = os::metrics(self) else {
            return;
        };
        let insets = os::insets(self);
        let layout = geometry::layout(bounds, density, insets, self.window.mode());
        if layout != self.layout || density.to_bits() != self.density.to_bits() {
            log::debug!(
                target: "reims_vgpu_window",
                "layout: {bounds:?} points at {density} pixels a point, insets {insets:?}: \
                 {layout:?}"
            );
            self.layout = layout;
            self.density = density;
            self.refresh_cursor();
            if self.input.is_captured() {
                self.confine_pointer();
            }
        }
        self.window.surface().set_layout(layout);
    }

    /// The pointer moved to `point` (in points): the cursor changes as it enters or leaves the
    /// image.
    pub(crate) fn pointer_over(&mut self, point: (f64, f64)) {
        let over = geometry::contains(self.layout.image, point);
        if over != self.over_image {
            self.over_image = over;
            self.refresh_cursor();
        }
    }

    /// Sets the cursor: over the image the guest's, or none while the guest hides its own or the
    /// window draws it itself (`handle`'s module doc); the host's arrow elsewhere and before the
    /// guest has set one.
    fn refresh_cursor(&mut self) {
        let guest = self.window.guest();
        let ours = if self.over_image {
            let per_pixel =
                geometry::cursor_points_per_pixel(self.layout.image, guest.mode.unwrap_or((0, 0)));
            if !guest.host_cursor_visible() && guest.cursor.is_some() {
                // SAFETY: SDL's video runs on this thread.
                unsafe { SDL_HideCursor() };
                return;
            }
            self.cursor
                .cursor(guest.cursor.as_deref(), per_pixel, self.density)
                .map(|c| c.raw())
        } else {
            None
        };
        // SAFETY: SDL's video runs on this thread; the cursor is one it made, or its default.
        unsafe {
            SDL_SetCursor(ours.unwrap_or_else(|| SDL_GetDefaultCursor()));
            SDL_ShowCursor();
        }
    }
}

/// `image`, a rectangle in the window's points, as SDL's mouse rect: in whole points, rounded out,
/// so that the image's last row and column stay in reach of the pointer held inside it, which
/// holds at the image's edge over the sliver of the bars it may reach.
fn mouse_rect(image: Rect) -> SDL_Rect {
    let (left, top) = (image.x.floor(), image.y.floor());
    let right = (image.x + image.width).ceil();
    let bottom = (image.y + image.height).ceil();
    SDL_Rect {
        x: left as i32,
        y: top as i32,
        w: (right - left) as i32,
        h: (bottom - top) as i32,
    }
}

/// Logs that SDL refused `what` for the window titled `title`.
pub(crate) fn refused(what: &str, title: &str) {
    log::warn!(
        target: "reims_vgpu_window",
        "window {title:?}: {what}: {}",
        sdl_error()
    );
}

/// Whether a window event of type `kind` shows the window to the person, or hides it: shown,
/// exposed or restored; hidden, minimized or covered (`SDL_EVENT_WINDOW_OCCLUDED`, which SDL
/// sends on Wayland for a window the compositor suspends, and on macOS for one fully covered).
/// `None` for any other event.
fn shows(kind: u32) -> Option<bool> {
    match kind {
        k if k == SDL_EVENT_WINDOW_SHOWN.0
            || k == SDL_EVENT_WINDOW_EXPOSED.0
            || k == SDL_EVENT_WINDOW_RESTORED.0 =>
        {
            Some(true)
        }
        k if k == SDL_EVENT_WINDOW_HIDDEN.0
            || k == SDL_EVENT_WINDOW_MINIMIZED.0
            || k == SDL_EVENT_WINDOW_OCCLUDED.0 =>
        {
            Some(false)
        }
        _ => None,
    }
}

/// The steps of a close the person asks for ([`close_asked`]) and of the display's end
/// ([`ended`]).
trait Closing {
    /// Whether the keyboard is captured.
    fn is_captured(&self) -> bool;
    /// The person closed the window: the input lets go, every key and button held goes up, and
    /// capture ends.
    fn let_go(&mut self);
    /// The display ended: the input lets go as for [`Closing::let_go`].
    fn let_go_at_the_end(&mut self);
    /// What the input queued goes to the host.
    fn flush(&mut self);
    /// The grab, the pointer's hold, the size's hold, the title and the cursor follow capture.
    fn capture_changed(&mut self);
    /// The grab, the pointer's hold and the size's hold follow capture.
    fn holds_follow_capture(&mut self);
    /// The host is told of the close.
    fn tell_host(&mut self);
}

impl Closing for Shell {
    fn is_captured(&self) -> bool {
        self.input.is_captured()
    }

    fn let_go(&mut self) {
        self.input.close_requested();
    }

    fn let_go_at_the_end(&mut self) {
        self.input.closing();
    }

    fn flush(&mut self) {
        Shell::flush(self);
    }

    fn capture_changed(&mut self) {
        Shell::capture_changed(self);
    }

    fn holds_follow_capture(&mut self) {
        Shell::holds_follow_capture(self);
    }

    fn tell_host(&mut self) {
        if let Some(close) = self.window.callbacks().and_then(|c| c.close) {
            close();
        }
    }
}

/// A close the person asks for, in order: the input lets go, what it queued (the key-ups and the
/// pointer's buttons up) goes to the host, the grab, the title and the cursor follow where
/// capture ended, and only then is the host told, which takes its callbacks back, so the guest
/// gets the key-ups first.
fn close_asked(c: &mut impl Closing) {
    let captured = c.is_captured();
    c.let_go();
    c.flush();
    if c.is_captured() != captured {
        c.capture_changed();
    }
    c.tell_host();
}

/// The display's end, in order, as AppKit's window closes (`DisplayView::window_closing`): the
/// input lets go, what it queued goes to the host, and where that ended capture, the keyboard's
/// grab, the pointer's hold and the size's hold go with it, so none stays on the window as it
/// hides. The title and the cursor are left: the window no longer shows, and SDL's cursor is the
/// process's, which another window sets as the pointer enters it.
fn ended(c: &mut impl Closing) {
    let captured = c.is_captured();
    c.let_go_at_the_end();
    c.flush();
    if c.is_captured() != captured {
        c.holds_follow_capture();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pointer is held over the image in whole points, rounded out; an image on whole points
    /// is the rect itself
    #[test]
    fn the_pointer_is_held_over_the_image_in_whole_points() {
        let held = |x, y, width, height| {
            let r = mouse_rect(Rect {
                x,
                y,
                width,
                height,
            });
            (r.x, r.y, r.w, r.h)
        };
        assert_eq!(held(80.5, 0.0, 959.0, 600.25), (80, 0, 960, 601));
        assert_eq!(held(0.0, 0.0, 1920.0, 1200.0), (0, 0, 1920, 1200));
        assert_eq!(held(10.4, 20.6, 0.2, 0.2), (10, 20, 1, 1));
    }

    /// A close's steps, recorded, over a capture that letting go ends or not.
    struct Recorded {
        /// Whether the keyboard is captured.
        captured: bool,
        /// Whether letting go ends capture.
        releases: bool,
        /// The steps, in order.
        steps: Vec<&'static str>,
    }

    impl Closing for Recorded {
        fn is_captured(&self) -> bool {
            self.captured
        }

        fn let_go(&mut self) {
            self.steps.push("let go");
            if self.releases {
                self.captured = false;
            }
        }

        fn let_go_at_the_end(&mut self) {
            self.steps.push("let go at the end");
            if self.releases {
                self.captured = false;
            }
        }

        fn flush(&mut self) {
            self.steps.push("flush");
        }

        fn capture_changed(&mut self) {
            self.steps.push("capture changed");
        }

        fn holds_follow_capture(&mut self) {
            self.steps.push("holds follow");
        }

        fn tell_host(&mut self) {
            self.steps.push("tell host");
        }
    }

    /// The person's close lets the input go and sends what it queued, then has the grab, title
    /// and cursor follow the capture it ended, and tells the host last; with no capture to end,
    /// nothing follows it
    #[test]
    fn a_close_sends_the_key_ups_and_ends_capture_before_the_host_hears_of_it() {
        let mut captured = Recorded {
            captured: true,
            releases: true,
            steps: Vec::new(),
        };
        close_asked(&mut captured);
        assert_eq!(
            captured.steps,
            ["let go", "flush", "capture changed", "tell host"]
        );
        assert!(!captured.captured);
        let mut free = Recorded {
            captured: false,
            releases: true,
            steps: Vec::new(),
        };
        close_asked(&mut free);
        assert_eq!(free.steps, ["let go", "flush", "tell host"]);
    }

    /// The display's end lets the input go and sends what it queued, then lets go of the
    /// keyboard's grab, the pointer's hold and the size's hold where capture ended; with no
    /// capture to end, nothing follows it
    #[test]
    fn the_end_lets_go_of_the_grab_and_the_holds() {
        let mut captured = Recorded {
            captured: true,
            releases: true,
            steps: Vec::new(),
        };
        ended(&mut captured);
        assert_eq!(
            captured.steps,
            ["let go at the end", "flush", "holds follow"]
        );
        assert!(!captured.captured);
        let mut free = Recorded {
            captured: false,
            releases: true,
            steps: Vec::new(),
        };
        ended(&mut free);
        assert_eq!(free.steps, ["let go at the end", "flush"]);
    }

    /// A window hidden, minimized or covered (a Wayland compositor's suspended one) stops
    /// drawing until it is shown, restored or exposed again; no other event says either
    #[test]
    fn a_covered_window_stops_drawing_until_exposed() {
        for hides in [
            SDL_EVENT_WINDOW_HIDDEN,
            SDL_EVENT_WINDOW_MINIMIZED,
            SDL_EVENT_WINDOW_OCCLUDED,
        ] {
            assert_eq!(shows(hides.0), Some(false), "{}", hides.0);
        }
        for shows_it in [
            SDL_EVENT_WINDOW_SHOWN,
            SDL_EVENT_WINDOW_EXPOSED,
            SDL_EVENT_WINDOW_RESTORED,
        ] {
            assert_eq!(shows(shows_it.0), Some(true), "{}", shows_it.0);
        }
        for other in [
            SDL_EVENT_WINDOW_RESIZED,
            SDL_EVENT_WINDOW_FOCUS_LOST,
            SDL_EVENT_WINDOW_MOUSE_LEAVE,
            SDL_EVENT_KEY_DOWN,
        ] {
            assert_eq!(shows(other.0), None, "{}", other.0);
        }
    }

    /// Within one batch of events the last that shows or hides the window decides: a suspended
    /// Wayland window's `EXPOSED` then `OCCLUDED` leaves it hidden, and other events say nothing
    #[test]
    fn the_last_show_or_hide_of_a_batch_decides() {
        let batch = |kinds: &[SDL_EventType]| kinds.iter().fold(None, |s, k| settled(s, k.0));
        let resized = SDL_EVENT_WINDOW_RESIZED;
        assert_eq!(
            batch(&[SDL_EVENT_WINDOW_EXPOSED, SDL_EVENT_WINDOW_OCCLUDED, resized]),
            Some(false)
        );
        assert_eq!(
            batch(&[SDL_EVENT_WINDOW_OCCLUDED, resized, SDL_EVENT_WINDOW_EXPOSED]),
            Some(true)
        );
        assert_eq!(batch(&[resized, SDL_EVENT_WINDOW_FOCUS_LOST]), None);
    }

    /// A resize to the size asked for, or within a point of it, keeps the window following the
    /// guest's mode; any other is the person's
    #[test]
    fn a_resize_within_a_point_of_the_asked_size_is_the_library_s() {
        assert!(is_asked((1920, 1200), (1920, 1200)));
        assert!(is_asked((1920, 1199), (1920, 1200)));
        assert!(is_asked((1921, 1201), (1920, 1200)));
        assert!(!is_asked((1918, 1200), (1920, 1200)));
        assert!(!is_asked((1920, 1100), (1920, 1200)));
    }
}
