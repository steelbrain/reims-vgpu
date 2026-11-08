//! SDL's window's input on macOS: AppKit's events, given to AppKit's window's input
//! ([`InputState`]) by the application before it dispatches them ([`takes`]), as AppKit's view
//! would receive them, so the guest gets what it gets from AppKit's window: keys by position on
//! ISO keyboards too, ⌘ shortcuts to the menu uncaptured and to the guest captured, key-ups under
//! ⌘, scrolling with its phases and momentum, the system's hot keys held off while captured. The
//! pointer is held over the image while captured by SDL's mouse rect (`crate::sdl::shell`).
//!
//! Which events, and when:
//! - **Keys** of the key window, where it is one of SDL's windows: those the input takes before
//!   AppKit (a ⌘ key-down while captured, a key-up under ⌘) never reach AppKit; the others go to
//!   the input as the view's `keyDown:`, `keyUp:` and `flagsChanged:`, and on to AppKit only where
//!   the input passes them on (an uncaptured ⌘ shortcut, for the menu).
//! - **Clicks** in a window that is key, as the view's: the click that makes a window key is
//!   AppKit's alone, as the view does not take the first mouse. Releases, drags and the wheel go
//!   to the window they are in; moves to the key window, while the pointer is over its content, as
//!   the view's tracking area gives them.
//! - Every mouse and scroll event goes on to AppKit afterwards, which SDL's view takes and the
//!   window ignores, so AppKit's own handling (key status, cursor rectangles, the title bar) is as
//!   for any window.
//!
//! The event's point is in the content view's points, y down, as AppKit's flipped view sees it.

use std::sync::Arc;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSEvent, NSEventType, NSWindow};
use reims_vgpu_vulkan::os::msg;

use crate::geometry::Rect;
use crate::handle::Window;
use crate::native::macos::input::{InputState, Reaction, ViewContext};
use crate::native::macos::main_thread::shared_app;
use crate::sdl::shell::{self, Shell};

/// A window's input: AppKit's window's ([`InputState`]), which calls the host's callbacks as each
/// event comes.
pub(crate) struct Input(InputState);

impl Input {
    /// The input of `window`.
    pub(crate) fn new(window: &Arc<Window>) -> Input {
        Input(InputState::new(Arc::clone(window)))
    }

    /// The window is closing: every key and button held goes up, and capture ends.
    pub(crate) fn closing(&mut self) {
        self.0.window_closing();
    }

    /// The person closed the window or chose Quit: as for AppKit's window, every key and button
    /// held goes up, and capture ends, before the host is told.
    pub(crate) fn close_requested(&mut self) {
        self.0.window_closing();
    }

    /// The window gained the keyboard: AppKit gives the click that did it to no one.
    pub(crate) fn focus_gained(&mut self, _timestamp: u64) {}

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

    /// Whether the person may resize the window: not while captured.
    pub(crate) fn window_resizable(&self) -> bool {
        self.0.window_resizable()
    }

    /// What the title ends with in the current capture state.
    pub(crate) fn title_suffix(&self) -> &'static str {
        self.0.title_suffix()
    }

    /// Nothing waits: the input calls the host's callbacks as each event comes.
    pub(crate) fn flush(&mut self, _window: &Window) {}
}

/// The open window whose `NSWindow` is `window`.
fn shell_of(window: &NSWindow) -> Option<u64> {
    shell::each_open(|s| std::ptr::eq(s.os.ns_window(), window).then(|| s.window.id()))
        .first()
        .copied()
}

/// The key window's open window, where it is one of SDL's.
fn key_shell(mtm: MainThreadMarker) -> Option<u64> {
    let key = msg::get("keyWindow", || shared_app(mtm).keyWindow())?;
    shell_of(&key)
}

/// What the view would know as `event` reaches it; the event's point when `located`.
fn context(shell: &Shell, event: &NSEvent, located: bool) -> ViewContext {
    let window = shell.os.ns_window();
    let point = located
        .then(|| {
            let content = msg::get("contentView", || window.contentView())?;
            let location = msg::get("locationInWindow", || event.locationInWindow());
            let point = msg::get("convertPoint:fromView:", || {
                content.convertPoint_fromView(location, None)
            });
            let height = msg::get("bounds", || content.bounds()).size.height;
            Some((point.x, height - point.y))
        })
        .flatten();
    ViewContext {
        point,
        image: shell.layout.image,
        guest: shell.window.mode(),
        window_is_key: msg::get("isKeyWindow", || window.isKeyWindow()),
    }
}

/// Whether `point` is over `shell`'s content, in its points.
fn over_content(shell: &Shell, point: (f64, f64)) -> bool {
    let Some(content) = msg::get("contentView", || shell.os.ns_window().contentView()) else {
        return false;
    };
    let size = msg::get("bounds", || content.bounds()).size;
    (0.0..size.width).contains(&point.0) && (0.0..size.height).contains(&point.1)
}

/// Acts on the input's reaction: a changed capture refreshes the title, the cursor and the
/// pointer's hold.
fn react(shell: &mut Shell, reaction: Reaction) {
    if reaction.capture_changed {
        shell.capture_changed();
    }
}

/// The application saw `event`, before AppKit dispatches it: true where one of SDL's windows'
/// input took it, and AppKit must not see it (the module doc).
pub(crate) fn takes(event: &NSEvent, mtm: MainThreadMarker) -> bool {
    let kind = msg::get("type", || event.r#type());
    match kind {
        NSEventType::KeyDown | NSEventType::KeyUp | NSEventType::FlagsChanged => key_shell(mtm)
            .is_some_and(|id| {
                shell::with_shell(id, |shell| key(shell, event, kind)).unwrap_or(false)
            }),
        NSEventType::LeftMouseDown
        | NSEventType::RightMouseDown
        | NSEventType::OtherMouseDown
        | NSEventType::LeftMouseUp
        | NSEventType::RightMouseUp
        | NSEventType::OtherMouseUp
        | NSEventType::LeftMouseDragged
        | NSEventType::RightMouseDragged
        | NSEventType::OtherMouseDragged
        | NSEventType::MouseMoved
        | NSEventType::ScrollWheel => {
            let window: Option<Retained<NSWindow>> = msg::get("window", || event.window(mtm));
            if let Some(id) = window.as_deref().and_then(shell_of) {
                shell::with_shell(id, |shell| mouse(shell, event, kind));
            }
            false
        }
        _ => false,
    }
}

/// A key event for `shell`'s window, the key window: true where AppKit must not see it.
fn key(shell: &mut Shell, event: &NSEvent, kind: NSEventType) -> bool {
    if shell.closed {
        return false;
    }
    let cx = context(shell, event, false);
    let input = &mut shell.input.0;
    let before = input.is_captured();
    if input.app_send_event(event, &cx) {
        let changed = input.is_captured() != before;
        react(
            shell,
            Reaction {
                capture_changed: changed,
                pass_on: false,
            },
        );
        return true;
    }
    let reaction = match kind {
        NSEventType::KeyDown => input.key_down(event),
        NSEventType::KeyUp => input.key_up(event),
        _ => input.flags_changed(event),
    };
    react(shell, reaction);
    !reaction.pass_on
}

/// A mouse or scroll event in `shell`'s window (the module doc).
fn mouse(shell: &mut Shell, event: &NSEvent, kind: NSEventType) {
    if shell.closed {
        return;
    }
    let cx = context(shell, event, true);
    let over = cx.point.filter(|&point| over_content(shell, point));
    let input = &mut shell.input.0;
    let reaction = match kind {
        NSEventType::LeftMouseDown | NSEventType::RightMouseDown | NSEventType::OtherMouseDown
            if cx.window_is_key && over.is_some() =>
        {
            input.mouse_down(event, &cx)
        }
        NSEventType::LeftMouseUp | NSEventType::RightMouseUp | NSEventType::OtherMouseUp => {
            input.mouse_up(event, &cx)
        }
        NSEventType::LeftMouseDragged
        | NSEventType::RightMouseDragged
        | NSEventType::OtherMouseDragged
            if cx.window_is_key =>
        {
            input.mouse_moved(event, &cx)
        }
        NSEventType::MouseMoved if cx.window_is_key && over.is_some() => {
            input.mouse_moved(event, &cx)
        }
        NSEventType::ScrollWheel => input.scroll_wheel(event, &cx),
        _ => return,
    };
    react(shell, reaction);
    if let Some(point) = over
        && cx.window_is_key
    {
        shell.pointer_over(point);
    }
}

/// Window ▸ Release Input: the key window's input lets the keyboard and the pointer go.
pub(crate) fn release_key_input(mtm: MainThreadMarker) {
    if let Some(id) = key_shell(mtm) {
        shell::with_shell(id, |shell| {
            let reaction = shell.input.0.release_capture();
            react(shell, reaction);
        });
    }
}

/// Whether the key window's input has the keyboard captured, which turns Release Input on.
pub(crate) fn key_input_captured(mtm: MainThreadMarker) -> bool {
    key_shell(mtm)
        .and_then(|id| shell::with_shell(id, |shell| shell.input.is_captured()))
        .unwrap_or(false)
}

/// Quit: every window lets its input go and tells the host, which ends the VM.
pub(crate) fn quit_requested() {
    for id in shell::each_open(|s| (!s.closed).then(|| s.window.id())) {
        shell::with_shell(id, Shell::close_requested);
    }
}
