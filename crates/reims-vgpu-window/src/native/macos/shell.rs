//! A display's `NSWindow`, on the main thread: where it opens, how it looks, its delegate, and
//! its end. The windows open are kept here, by the [`Window`]'s id, for the main thread alone.
//!
//! **The screen.** A window opens on the `NSScreen` whose `deviceDescription["NSScreenNumber"]`
//! is the CoreGraphics display [`screens::screen_for_port`] picks for the display's port, else on
//! `NSScreen.mainScreen`, so the screen whose density the guest was told of is the one it opens
//! on.
//!
//! **Windowed** (the default): titled, closable, miniaturizable and resizable, and able to go
//! full screen; while its input is captured it is not resizable, and it is again once released
//! ([`resizable_change`], which the view applies). Its content is the guest's mode, else the first mode the display offers, else
//! 1920×1080, at one guest point per host point (a HiDPI guest's pixel is a screen pixel, a 1×
//! guest's a point, which the draw scales on the GPU), shrunk to fit the screen's visible area
//! less the title bar ([`geometry::initial_content_size`]), centred there. The guest's scale
//! for a mode follows the size in millimetres the display told it ([`Window::told_size`]). A display offers its
//! modes when it is made, before its window opens, and the guest sets a mode later, so a window
//! opens at the mode offered first. A window whose top left would be another windowed window's
//! moves 24 points right and down, made smaller, its ratio kept, where it would leave the
//! visible area, but not below [`MIN_CONTENT`] ([`cascade`]).
//!
//! Until the person resizes it (an edge, tiling, zoom, full screen), a windowed window follows
//! the guest's mode: each mode change resizes its content as the window would open on its
//! current screen, keeping the frame's top left corner, then moves the frame inside the
//! screen's visible area, its top edge on it ([`geometry::resized_frame`]), and off another
//! windowed window's top left as it would open. The guest's first modes (at boot, then at login
//! under a saved setting) can differ from the one offered first, and a window that kept its size
//! would show the guest scaled. Once the person has sized it, a mode change re-fits the image
//! and the window keeps its size.
//!
//! **Borderless** (`REIMS_VGPU_WINDOW_BORDERLESS=1`): each window fills its screen's whole frame,
//! and the view fits the image inside the screen's safe area (below a camera housing). The
//! window is a titled one made to look borderless (a full-size content view, a transparent title
//! bar with no title and no buttons, not movable): on macOS 26.3 and later a window without a
//! title bar may let clicks through to what is behind it, and a titled one takes keys and clicks.
//! Its title bar's views are hidden, so a click at the screen's top edge reaches the view. The
//! window keeps its frame where AppKit would move it below the menu bar
//! (`constrainFrameRect:toScreen:`). While the application is active the Dock and the menu bar
//! hide (`presentationOptions`, which apply only while it is frontmost). When the screens change,
//! each borderless window fills its screen again, or the main screen while its own is gone.
//!
//! **The delegate.** Closing the window (its close button) tells the host, which ends the VM, and
//! leaves the window open: it closes when the display is released ([`close`]). Losing key status
//! releases the input. Any resize the library did not make is the person's sizing, which ends
//! the following of the guest's mode: tiling, Stage Manager and window managers set the frame
//! with no other sign. A live resize's start, a zoom and entering full screen mark it before
//! their first resize. Entering full screen and leaving it, or failing to enter it, tell the view,
//! which leaves the window's style alone in between. The window's occlusion tells the render thread whether to draw, and its
//! backing scale and screen lay the view out again.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplicationPresentationOptions, NSBackingStoreType, NSResponder, NSScreen, NSWindow,
    NSWindowButton, NSWindowCollectionBehavior, NSWindowDelegate, NSWindowStyleMask,
    NSWindowTitleVisibility,
};
use objc2_foundation::{
    NSNotification, NSNumber, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, ns_string,
};
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_metal::msg;

use super::main_thread::{MainOnly, responds, shared_app};
use super::view::DisplayView;
use crate::geometry::{self, Insets, Rect};
use crate::handle::Window;
use crate::screens;

/// The guest's mode a window is sized for before the guest sets one or the display offers one.
const DEFAULT_MODE: (u16, u16) = (1920, 1080);

/// The smallest a windowed window's content may be made, in points.
const MIN_CONTENT: NSSize = NSSize::new(320.0, 200.0);

/// A window's `NSWindow` state.
#[derive(Debug)]
pub(crate) struct WindowIvars {
    /// Whether the window is borderless (module doc).
    borderless: bool,
}

define_class!(
    /// A display's `NSWindow` (module doc): it can always become key and main, and a borderless
    /// one keeps the frame it is given.
    // SAFETY:
    // - NSWindow has no subclassing requirements for these overrides.
    // - The window does not implement `Drop`.
    #[unsafe(super(NSWindow, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = WindowIvars]
    pub(crate) struct DisplayWindow;

    impl DisplayWindow {
        #[unsafe(method(canBecomeKeyWindow))]
        fn can_become_key_window(&self) -> bool {
            let _env = BlockEnv::foreign();
            true
        }

        #[unsafe(method(canBecomeMainWindow))]
        fn can_become_main_window(&self) -> bool {
            let _env = BlockEnv::foreign();
            true
        }

        #[unsafe(method(constrainFrameRect:toScreen:))]
        fn constrain_frame_rect(&self, frame: NSRect, screen: Option<&NSScreen>) -> NSRect {
            let _env = BlockEnv::foreign();
            if self.ivars().borderless {
                return frame;
            }
            // SAFETY: `constrainFrameRect:toScreen:` takes an NSRect and an optional screen and
            // returns an NSRect.
            msg::get("constrainFrameRect:toScreen:", || unsafe {
                msg_send![super(self), constrainFrameRect: frame, toScreen: screen]
            })
        }
    }

    unsafe impl NSObjectProtocol for DisplayWindow {}
);

impl DisplayWindow {
    /// A window with content at `content` (screen coordinates) of `style`, released only by its
    /// references (never by its own close).
    fn new(
        mtm: MainThreadMarker,
        content: NSRect,
        style: NSWindowStyleMask,
        borderless: bool,
    ) -> Retained<DisplayWindow> {
        // SAFETY: `initWithContentRect:styleMask:backing:defer:` is one of NSWindow's
        // designated initializers, with these argument types.
        let window: Retained<DisplayWindow> =
            msg::get("initWithContentRect:styleMask:backing:defer:", || unsafe {
                let this = DisplayWindow::alloc(mtm).set_ivars(WindowIvars { borderless });
                msg_send![
                    super(this),
                    initWithContentRect: content,
                    styleMask: style,
                    backing: NSBackingStoreType::Buffered,
                    defer: false
                ]
            });
        // SAFETY: the window is released only through its `Retained`s, never by its own close,
        // so none of them is left dangling.
        msg::get("setReleasedWhenClosed:", || unsafe {
            window.setReleasedWhenClosed(false);
        });
        window
    }
}

/// A window delegate's state.
pub(crate) struct DelegateIvars {
    /// The window's view.
    view: Retained<DisplayView>,
    /// Whether the window is borderless.
    borderless: bool,
    /// Whether the person has resized the window: from then on it keeps its size when the
    /// guest's mode changes (module doc).
    sized_by_person: Cell<bool>,
    /// The library is resizing the window ([`follow`]): the resize is not the person's.
    resizing: Cell<bool>,
}

define_class!(
    /// A display window's delegate (module doc).
    // SAFETY:
    // - NSObject has no subclassing requirements.
    // - The delegate does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = DelegateIvars]
    pub(crate) struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        /// The window stays: the host ends the VM, and the display's release closes it.
        #[unsafe(method(windowShouldClose:))]
        fn window_should_close(&self, _sender: &NSWindow) -> bool {
            let _env = BlockEnv::foreign();
            self.ivars().view.close_requested();
            false
        }

        /// A borderless window keeps its screen's frame. A windowed one zooms, which is the
        /// person's sizing: the green button with ⌥, a double click on the title bar, or
        /// Window ▸ Zoom.
        #[unsafe(method(windowShouldZoom:toFrame:))]
        fn window_should_zoom(&self, _window: &NSWindow, _frame: NSRect) -> bool {
            let _env = BlockEnv::foreign();
            self.person_sized("zoom");
            !self.ivars().borderless
        }

        /// The person started dragging an edge or a corner.
        #[unsafe(method(windowWillStartLiveResize:))]
        fn window_will_start_live_resize(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.person_sized("live resize");
        }

        #[unsafe(method(windowWillEnterFullScreen:))]
        fn window_will_enter_full_screen(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.person_sized("full screen");
            self.ivars().view.full_screen_changed(true);
        }

        #[unsafe(method(windowDidExitFullScreen:))]
        fn window_did_exit_full_screen(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.ivars().view.full_screen_changed(false);
        }

        #[unsafe(method(windowDidFailToEnterFullScreen:))]
        fn window_did_fail_to_enter_full_screen(&self, _window: &NSWindow) {
            let _env = BlockEnv::foreign();
            self.ivars().view.full_screen_changed(false);
        }

        #[unsafe(method(windowDidResignKey:))]
        fn window_did_resign_key(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.ivars().view.focus_lost();
        }

        #[unsafe(method(windowDidChangeOcclusionState:))]
        fn window_did_change_occlusion_state(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.ivars().view.occlusion_changed();
        }

        #[unsafe(method(windowDidChangeBackingProperties:))]
        fn window_did_change_backing_properties(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.ivars().view.relayout();
        }

        #[unsafe(method(windowDidChangeScreen:))]
        fn window_did_change_screen(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            self.ivars().view.relayout();
        }

        #[unsafe(method(windowDidResize:))]
        fn window_did_resize(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            if !self.ivars().resizing.get() {
                self.person_sized("resize");
            }
            self.ivars().view.relayout();
        }
    }
);

impl WindowDelegate {
    /// The delegate of the window showing `view`.
    fn new(
        mtm: MainThreadMarker,
        view: Retained<DisplayView>,
        borderless: bool,
    ) -> Retained<WindowDelegate> {
        // SAFETY: `init` is NSObject's designated initializer.
        msg::get("init", || unsafe {
            let this = WindowDelegate::alloc(mtm).set_ivars(DelegateIvars {
                view,
                borderless,
                sized_by_person: Cell::new(false),
                resizing: Cell::new(false),
            });
            msg_send![super(this), init]
        })
    }

    /// The person sized the window, through `hook`: a windowed window keeps its size from now on.
    fn person_sized(&self, hook: &str) {
        let ivars = self.ivars();
        if !ivars.borderless && !ivars.sized_by_person.replace(true) {
            log::debug!(
                "window {:?} sized by the person ({hook}): it keeps its size",
                ivars.view.title()
            );
        }
    }

    /// Whether the window follows the guest's mode: windowed, not sized by the person, and not
    /// a window that keeps its size (`Window::keeps_its_size`).
    fn follows_guest_mode(&self) -> bool {
        let ivars = self.ivars();
        !ivars.borderless && !ivars.sized_by_person.get() && !ivars.view.keeps_its_size()
    }
}

/// An open window, as the main thread keeps it.
struct Shell {
    /// Its [`Window`]'s id.
    id: u64,
    window: Retained<DisplayWindow>,
    view: Retained<DisplayView>,
    /// Held here: a window's delegate is a weak reference.
    delegate: Retained<WindowDelegate>,
    /// Whether it is borderless.
    borderless: bool,
    /// The CoreGraphics display of the screen it opened on, which a borderless window fills
    /// again when the screens change.
    screen: Option<u32>,
}

/// The windows open, in the order opened.
static OPEN: MainOnly<RefCell<Vec<Shell>>> = MainOnly::new(RefCell::new(Vec::new()));

/// The view of the open window `id`.
fn view_of(id: u64, mtm: MainThreadMarker) -> Option<Retained<DisplayView>> {
    let open = OPEN.get(mtm).borrow();
    open.iter().find(|s| s.id == id).map(|s| s.view.clone())
}

/// The views of every open window.
pub(crate) fn views(mtm: MainThreadMarker) -> Vec<Retained<DisplayView>> {
    OPEN.get(mtm)
        .borrow()
        .iter()
        .map(|s| s.view.clone())
        .collect()
}

/// The frames of the open windowed windows, but window `except`'s.
fn windowed_frames(except: Option<u64>, mtm: MainThreadMarker) -> Vec<NSRect> {
    OPEN.get(mtm)
        .borrow()
        .iter()
        .filter(|s| !s.borderless && Some(s.id) != except)
        .map(|s| msg::get("frame", || s.window.frame()))
        .collect()
}

/// The CoreGraphics display `screen` is.
pub(crate) fn display_id(screen: &NSScreen) -> Option<u32> {
    let description = msg::get("deviceDescription", || screen.deviceDescription());
    let number = msg::get("objectForKey:", || {
        description.objectForKey(ns_string!("NSScreenNumber"))
    })?;
    let number = msg::get("isKindOfClass:", || number.downcast::<NSNumber>()).ok()?;
    Some(msg::get("unsignedIntValue", || number.unsignedIntValue()))
}

/// The screen that is CoreGraphics display `id`.
pub(crate) fn screen_with_id(id: u32, mtm: MainThreadMarker) -> Option<Retained<NSScreen>> {
    let screens = msg::get("screens", || NSScreen::screens(mtm).to_vec());
    screens
        .into_iter()
        .find(|screen| display_id(screen) == Some(id))
}

/// `NSScreen.mainScreen`.
pub(crate) fn main_screen(mtm: MainThreadMarker) -> Option<Retained<NSScreen>> {
    msg::get("mainScreen", || NSScreen::mainScreen(mtm))
}

/// The room `screen` keeps clear at its edges (a camera housing), in points; none before
/// macOS 12, which has no `safeAreaInsets`.
pub(crate) fn safe_area(screen: &NSScreen) -> Insets {
    if !responds(screen, sel!(safeAreaInsets)) {
        return Insets::default();
    }
    let insets = msg::get("safeAreaInsets", || screen.safeAreaInsets());
    Insets {
        top: insets.top,
        left: insets.left,
        bottom: insets.bottom,
        right: insets.right,
    }
}

/// A windowed window's style.
fn windowed_style() -> NSWindowStyleMask {
    NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable
}

/// A borderless window's style: titled, so that it takes clicks (module doc).
pub(crate) fn borderless_style() -> NSWindowStyleMask {
    NSWindowStyleMask::Titled | NSWindowStyleMask::FullSizeContentView
}

/// Whether a window of `style` is to be made resizable, or not, for `resizable`
/// ([`crate::capture::Capture::window_resizable`]): `Some(resizable)` for a windowed window
/// whose style says otherwise. A borderless window's style has no `Resizable`, and stays as it
/// is. So does that of a window in native full screen or on its way in or out (`full_screen`, or
/// `FullScreen` in its style): SDL's Cocoa code leaves the style alone there too, as a window
/// made not resizable in its full screen space can be stuck in it (read from SDL 3.4.18's code).
/// The window asks again once it has left full screen.
pub(crate) fn resizable_change(
    style: NSWindowStyleMask,
    resizable: bool,
    borderless: bool,
    full_screen: bool,
) -> Option<bool> {
    let windowed = !borderless && !full_screen && !style.contains(NSWindowStyleMask::FullScreen);
    (windowed && style.contains(NSWindowStyleMask::Resizable) != resizable).then_some(resizable)
}

/// How far a window that would open exactly over another moves, right and down, in points.
const CASCADE: f64 = 24.0;

/// How a windowed window is sized for a guest mode on a screen.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sizing {
    /// The guest's mode, whose ratio the content keeps.
    mode: (u16, u16),
    /// The scale the guest runs the mode at: its pixels per point.
    guest_scale: f64,
    /// The screen's backing scale.
    scale: f64,
    /// The title bar and borders: the frame less the content.
    chrome: NSSize,
}

impl Sizing {
    /// The content of a frame at most `room`: one guest point per host point, shrunk with its
    /// ratio kept where it does not fit ([`geometry::initial_content_size`]), on an even number
    /// of whole pixels, so the drawable is the content exactly and the mode the guest is offered
    /// for it ([`geometry::offered_mode`]) is the whole drawable.
    fn content(&self, room: NSSize) -> NSSize {
        let (width, height) = geometry::initial_content_size(
            self.mode,
            self.guest_scale,
            (
                room.width - self.chrome.width,
                room.height - self.chrome.height,
            ),
        );
        let whole = |points: f64| {
            let pixels = (points * self.scale).floor();
            let even = if pixels >= 2.0 {
                pixels - pixels % 2.0
            } else {
                pixels
            };
            (even / self.scale).max(1.0)
        };
        NSSize::new(whole(width), whole(height))
    }

    /// The frame around `content`.
    fn frame(&self, content: NSSize) -> NSSize {
        NSSize::new(
            content.width + self.chrome.width,
            content.height + self.chrome.height,
        )
    }
}

/// `frame`, a windowed window's in `visible` as `sizing` sizes it, moved right and down by
/// [`CASCADE`] while a frame of `taken` has its top left where it has: two windows on one screen
/// do not show one exactly over the other. Where the moved frame would leave `visible`, it is
/// made smaller from its new top left, its content keeping the guest's ratio on whole pixels;
/// where that would take the content below [`MIN_CONTENT`], the frame stays where it last was.
/// Screen coordinates, y up.
fn cascade(frame: NSRect, sizing: &Sizing, visible: NSRect, taken: &[NSRect]) -> NSRect {
    let top_left = |f: &NSRect| (f.origin.x, f.origin.y + f.size.height);
    let shares_top_left = |at: &NSRect| {
        let (x, y) = top_left(at);
        taken.iter().any(|f| {
            let (fx, fy) = top_left(f);
            (fx - x).abs() < 1.0 && (fy - y).abs() < 1.0
        })
    };
    let right = visible.origin.x + visible.size.width;
    let bottom = visible.origin.y;
    let mut at = frame;
    while shares_top_left(&at) {
        let (x, y) = top_left(&at);
        let (x, y) = (x + CASCADE, y - CASCADE);
        let room = NSSize::new(right - x, y - bottom);
        let fits = |size: NSSize| size.width <= room.width && size.height <= room.height;
        let size = if fits(frame.size) {
            frame.size
        } else {
            let content = sizing.content(room);
            let size = sizing.frame(content);
            if !fits(size)
                || content.width < MIN_CONTENT.width
                || content.height < MIN_CONTENT.height
            {
                break;
            }
            size
        };
        at = NSRect::new(NSPoint::new(x, y - size.height), size);
    }
    at
}

/// A windowed window's size on a screen for a guest mode, and the room there is for it.
struct Fit {
    /// The screen's visible frame.
    visible: NSRect,
    /// How the window is sized there.
    sizing: Sizing,
    /// The content, on whole pixels.
    content: NSSize,
    /// The frame: the content, the title bar and the borders.
    frame: NSSize,
}

/// The size of a windowed window showing a guest mode `mode`, which the guest runs at
/// `guest_scale`, on `screen` (module doc): one guest point per host point, shrunk to fit the
/// screen's visible area less the title bar, on whole pixels.
fn windowed_fit(
    screen: Option<&NSScreen>,
    mode: (u16, u16),
    guest_scale: u8,
    mtm: MainThreadMarker,
) -> Fit {
    let (visible, scale) = screen.map_or(
        (
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1920.0, 1080.0)),
            1.0,
        ),
        |s| {
            (
                msg::get("visibleFrame", || s.visibleFrame()),
                msg::get("backingScaleFactor", || s.backingScaleFactor()),
            )
        },
    );
    // The title bar and borders, from a probe content rectangle.
    let probe = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(100.0, 100.0));
    let framed = msg::get("frameRectForContentRect:styleMask:", || {
        NSWindow::frameRectForContentRect_styleMask(probe, windowed_style(), mtm)
    });
    let sizing = Sizing {
        mode,
        guest_scale: f64::from(guest_scale.max(1)),
        scale,
        chrome: NSSize::new(
            framed.size.width - probe.size.width,
            framed.size.height - probe.size.height,
        ),
    };
    let content = sizing.content(visible.size);
    Fit {
        visible,
        sizing,
        content,
        frame: sizing.frame(content),
    }
}

/// The frame of a windowed window for a guest mode `mode`, which the guest runs at
/// `guest_scale`, on `screen`, clear of the frames `taken` (module doc): centred in the screen's
/// visible frame, then cascaded.
pub(crate) fn windowed_frame(
    screen: Option<&NSScreen>,
    mode: (u16, u16),
    guest_scale: u8,
    taken: &[NSRect],
    mtm: MainThreadMarker,
) -> NSRect {
    let Fit {
        visible,
        sizing,
        frame: frame_size,
        ..
    } = windowed_fit(screen, mode, guest_scale, mtm);
    let origin = NSPoint::new(
        visible.origin.x + ((visible.size.width - frame_size.width) / 2.0).floor(),
        visible.origin.y + ((visible.size.height - frame_size.height) / 2.0).floor(),
    );
    let frame = cascade(NSRect::new(origin, frame_size), &sizing, visible, taken);
    log::debug!(
        "windowed: visible {visible:?} at scale {}, mode {mode:?} at guest scale \
         {guest_scale}: frame {frame:?}",
        sizing.scale
    );
    frame
}

/// A windowed window for a guest mode `mode`, which the guest runs at `guest_scale`, on
/// `screen`, clear of the frames `taken` (module doc).
fn windowed(
    screen: Option<&NSScreen>,
    mode: (u16, u16),
    guest_scale: u8,
    taken: &[NSRect],
    mtm: MainThreadMarker,
) -> Retained<DisplayWindow> {
    let style = windowed_style();
    let frame = windowed_frame(screen, mode, guest_scale, taken, mtm);
    let content_rect = msg::get("contentRectForFrameRect:styleMask:", || {
        NSWindow::contentRectForFrameRect_styleMask(frame, style, mtm)
    });
    let window = DisplayWindow::new(mtm, content_rect, style, false);
    msg::get("setContentMinSize:", || {
        window.setContentMinSize(MIN_CONTENT)
    });
    msg::get("setCollectionBehavior:", || {
        window.setCollectionBehavior(NSWindowCollectionBehavior::FullScreenPrimary);
    });
    msg::get("setFrame:display:", || {
        window.setFrame_display(frame, false)
    });
    window
}

/// `rect`, in AppKit's screen coordinates (y up), in [`geometry`]'s (y down).
fn y_down(rect: NSRect) -> Rect {
    Rect {
        x: rect.origin.x,
        y: -(rect.origin.y + rect.size.height),
        width: rect.size.width,
        height: rect.size.height,
    }
}

/// `rect`, in [`geometry`]'s coordinates (y down), in AppKit's screen coordinates (y up).
fn y_up(rect: Rect) -> NSRect {
    NSRect::new(
        NSPoint::new(rect.x, -(rect.y + rect.height)),
        NSSize::new(rect.width, rect.height),
    )
}

/// Resizes windowed `window`, whose delegate is `delegate`, for the guest mode `mode`, which the
/// guest runs at `guest_scale`, as it would open on its screen, keeping its top left corner, moved inside the screen's visible area
/// ([`geometry::resized_frame`]), then clear of the frames `taken` ([`cascade`]). The delegate
/// knows the resize for the library's.
fn follow(
    window: &DisplayWindow,
    delegate: &WindowDelegate,
    mode: (u16, u16),
    guest_scale: u8,
    taken: &[NSRect],
    mtm: MainThreadMarker,
) {
    let frame = msg::get("frame", || window.frame());
    let resized = followed_frame(window, mode, guest_scale, taken, mtm);
    if resized != frame {
        delegate.ivars().resizing.set(true);
        msg::get("setFrame:display:", || {
            window.setFrame_display(resized, true)
        });
        delegate.ivars().resizing.set(false);
    }
}

/// The frame windowed `window` takes for the guest mode `mode`, which the guest runs at
/// `guest_scale`, as it would open on its screen, keeping its top left corner, moved inside the
/// screen's visible area ([`geometry::resized_frame`]), then clear of the frames `taken`
/// ([`cascade`]).
pub(crate) fn followed_frame(
    window: &NSWindow,
    mode: (u16, u16),
    guest_scale: u8,
    taken: &[NSRect],
    mtm: MainThreadMarker,
) -> NSRect {
    let screen = msg::get("screen", || window.screen()).or_else(|| main_screen(mtm));
    let fit = windowed_fit(screen.as_deref(), mode, guest_scale, mtm);
    let frame = msg::get("frame", || window.frame());
    let resized = y_up(geometry::resized_frame(
        y_down(frame),
        (fit.frame.width, fit.frame.height),
        y_down(fit.visible),
    ));
    let resized = cascade(resized, &fit.sizing, fit.visible, taken);
    log::debug!(
        "following guest mode {mode:?} at guest scale {guest_scale}, scale {}: content {:?}, \
         frame {frame:?} to {resized:?}",
        fit.sizing.scale,
        fit.content
    );
    resized
}

/// The frame a borderless window fills on `screen`: its whole frame.
pub(crate) fn screen_frame(screen: Option<&NSScreen>) -> NSRect {
    screen.map_or(
        NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1920.0, 1080.0)),
        |s| msg::get("frame", || s.frame()),
    )
}

/// A borderless window filling `screen` (module doc).
fn borderless(screen: Option<&NSScreen>, mtm: MainThreadMarker) -> Retained<DisplayWindow> {
    let style = borderless_style();
    let frame = screen_frame(screen);
    let window = DisplayWindow::new(mtm, frame, style, true);
    look_borderless(&window);
    msg::get("setFrame:display:", || {
        window.setFrame_display(frame, false)
    });
    window
}

/// Makes a titled `window` look borderless (module doc): a transparent title bar with no title
/// and no buttons, not movable, with no shadow, and no full screen of its own.
pub(crate) fn look_borderless(window: &NSWindow) {
    msg::get("setTitlebarAppearsTransparent:", || {
        window.setTitlebarAppearsTransparent(true);
    });
    msg::get("setTitleVisibility:", || {
        window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
    });
    msg::get("setMovable:", || window.setMovable(false));
    msg::get("setHasShadow:", || window.setHasShadow(false));
    msg::get("setCollectionBehavior:", || {
        window.setCollectionBehavior(NSWindowCollectionBehavior::FullScreenNone);
    });
    hide_title_bar(window);
}

/// Hides a borderless window's standard buttons, and the view that holds its (transparent)
/// title bar, which would otherwise take the clicks along the screen's top edge.
fn hide_title_bar(window: &NSWindow) {
    let buttons = [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ];
    let mut title_bar = None;
    for kind in buttons {
        let Some(button) = msg::get("standardWindowButton:", || {
            window.standardWindowButton(kind)
        }) else {
            continue;
        };
        msg::get("setHidden:", || button.setHidden(true));
        if title_bar.is_none() {
            // SAFETY: `superview` is a getter; the button's superview is the title bar, and
            // its superview the title bar's container.
            title_bar = msg::get("superview", || unsafe { button.superview() })
                .and_then(|bar| msg::get("superview", || unsafe { bar.superview() }));
        }
    }
    let content = msg::get("contentView", || window.contentView());
    // SAFETY: `superview` is a getter.
    let frame_view = content
        .as_deref()
        .and_then(|c| msg::get("superview", || unsafe { c.superview() }));
    if let Some(bar) = title_bar {
        // Never the window's frame view, nor its content.
        let is_frame = frame_view
            .as_deref()
            .is_some_and(|f| std::ptr::eq(f, &*bar));
        let is_content = content.as_deref().is_some_and(|c| std::ptr::eq(c, &*bar));
        if !is_frame && !is_content {
            msg::get("setHidden:", || bar.setHidden(true));
        }
    }
}

/// Fills `screen` with borderless `window`.
pub(crate) fn fill(window: &NSWindow, screen: &NSScreen) {
    let frame = msg::get("frame", || screen.frame());
    msg::get("setFrame:display:", || window.setFrame_display(frame, true));
}

/// Hides the Dock and the menu bar while the application is active (module doc).
pub(crate) fn hide_dock_and_menu_bar(mtm: MainThreadMarker) {
    let app = shared_app(mtm);
    let options =
        NSApplicationPresentationOptions::HideDock | NSApplicationPresentationOptions::HideMenuBar;
    if let Err(caught) = msg::catch("setPresentationOptions:", || {
        app.setPresentationOptions(options);
    }) {
        log::warn!(
            "the Dock and menu bar stay: setPresentationOptions: raised {}: {}",
            caught.name,
            caught.reason
        );
    }
}

/// Gives the Dock and the menu bar back once no borderless window is open.
fn restore_dock_and_menu_bar(mtm: MainThreadMarker) {
    if OPEN.get(mtm).borrow().iter().any(|s| s.borderless) {
        return;
    }
    show_dock_and_menu_bar(mtm);
}

/// Gives the Dock and the menu bar back: the application's default presentation.
pub(crate) fn show_dock_and_menu_bar(mtm: MainThreadMarker) {
    let app = shared_app(mtm);
    let _ = msg::catch("setPresentationOptions:", || {
        app.setPresentationOptions(NSApplicationPresentationOptions::Default);
    });
}

/// Opens `window`'s `NSWindow` (module doc), unless the window has ended.
pub(crate) fn open(window: &Arc<Window>, mtm: MainThreadMarker) {
    if window.is_closed() || view_of(window.id(), mtm).is_some() {
        return;
    }
    let borderless = crate::env::settings().borderless;
    let target = screens::screen_for_port(window.port()).map(|s| s.id);
    let screen = target
        .and_then(|id| screen_with_id(id, mtm))
        .or_else(|| main_screen(mtm));
    let ns_window = if borderless {
        self::borderless(screen.as_deref(), mtm)
    } else {
        let taken = windowed_frames(None, mtm);
        let mode = window.mode().or(window.offered()).unwrap_or(DEFAULT_MODE);
        windowed(
            screen.as_deref(),
            mode,
            window.guest_scale(mode),
            &taken,
            mtm,
        )
    };
    let sink = window.sink();
    let device = sink.as_ref().and_then(|sink| sink.device());
    let framebuffer_only = sink.as_ref().is_none_or(|sink| sink.render_target_only());
    // Let go of the sink as soon as it has answered: when a release races the opening, this may
    // be the sink's last reference, released on the main thread outside an entry.
    drop(sink);
    let frame = msg::get("frame", || ns_window.frame());
    let bounds = msg::get("contentRectForFrameRect:", || {
        ns_window.contentRectForFrameRect(frame)
    })
    .size;
    let view = DisplayView::new(
        mtm,
        Arc::clone(window),
        borderless,
        bounds,
        device.as_deref(),
        framebuffer_only,
    );
    msg::get("setContentView:", || ns_window.setContentView(Some(&view)));
    let delegate = WindowDelegate::new(mtm, view.clone(), borderless);
    msg::get("setDelegate:", || {
        ns_window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    });
    OPEN.get(mtm).borrow_mut().push(Shell {
        id: window.id(),
        window: ns_window.clone(),
        view: view.clone(),
        delegate,
        borderless,
        screen: screen.as_deref().and_then(display_id),
    });
    window.surface().set_layer(view.layer_ref());
    view.refresh_title();
    if borderless {
        hide_dock_and_menu_bar(mtm);
    }
    msg::get("makeKeyAndOrderFront:", || {
        ns_window.makeKeyAndOrderFront(None)
    });
    msg::get("makeFirstResponder:", || {
        ns_window.makeFirstResponder(Some(&view))
    });
    view.relayout();
    view.occlusion_changed();
    log::info!(
        "window {:?} open: {}",
        window.title(),
        if borderless { "borderless" } else { "windowed" }
    );
}

/// Closes window `id`, at its display's end: its input lets go, and the `NSWindow`, its view,
/// delegate and layer are released here.
pub(crate) fn close(id: u64, mtm: MainThreadMarker) {
    let shell = {
        let mut open = OPEN.get(mtm).borrow_mut();
        let Some(at) = open.iter().position(|s| s.id == id) else {
            return;
        };
        open.remove(at)
    };
    let Shell {
        window,
        view,
        delegate,
        borderless,
        ..
    } = shell;
    view.window_closing();
    msg::get("setDelegate:", || window.setDelegate(None));
    msg::get("close", || window.close());
    if borderless {
        restore_dock_and_menu_bar(mtm);
    }
    // The layer's last references, the surface's and the view's, go here, on the main thread.
    let layer = view.take_surface_layer();
    drop((delegate, window, view, layer));
}

/// The guest of window `id` set a mode: a window following the guest's mode is resized for it
/// (module doc), and the image is laid out again.
pub(crate) fn guest_mode_changed(id: u64, mtm: MainThreadMarker) {
    let found = OPEN
        .get(mtm)
        .borrow()
        .iter()
        .find(|s| s.id == id)
        .map(|s| (s.window.clone(), s.view.clone(), s.delegate.clone()));
    let Some((window, view, delegate)) = found else {
        return;
    };
    if delegate.follows_guest_mode()
        && let Some(mode) = view.guest_mode()
    {
        let taken = windowed_frames(Some(id), mtm);
        follow(
            &window,
            &delegate,
            mode,
            view.guest_scale(mode),
            &taken,
            mtm,
        );
    }
    view.relayout();
}

/// The guest of window `id` changed its cursor or showed or hid it.
pub(crate) fn cursor_changed(id: u64, mtm: MainThreadMarker) {
    if let Some(view) = view_of(id, mtm) {
        view.cursor_changed();
    }
}

/// The screens changed: each borderless window fills its screen again, or the main screen while
/// its own is gone, and every view lays out again.
pub(crate) fn screens_changed(mtm: MainThreadMarker) {
    let shells: Vec<_> = OPEN
        .get(mtm)
        .borrow()
        .iter()
        .map(|s| (s.window.clone(), s.view.clone(), s.borderless, s.screen))
        .collect();
    for (window, view, borderless, screen) in shells {
        if borderless {
            let screen = screen
                .and_then(|id| screen_with_id(id, mtm))
                .or_else(|| main_screen(mtm));
            if let Some(screen) = screen {
                fill(&window, &screen);
            }
        }
        view.relayout();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame `size` with its bottom left at `origin`.
    fn rect(origin: (f64, f64), size: (f64, f64)) -> NSRect {
        NSRect::new(
            NSPoint::new(origin.0, origin.1),
            NSSize::new(size.0, size.1),
        )
    }

    /// While captured, a windowed window is made not resizable, and resizable again once
    /// released; a borderless window, and one in native full screen or on its way in or out, keep
    /// their style
    #[test]
    fn a_windowed_window_is_not_resizable_while_captured() {
        let windowed = windowed_style();
        let held = windowed.difference(NSWindowStyleMask::Resizable);
        assert_eq!(resizable_change(windowed, false, false, false), Some(false));
        assert_eq!(resizable_change(held, true, false, false), Some(true));
        // Already as asked: nothing changes.
        assert_eq!(resizable_change(windowed, true, false, false), None);
        assert_eq!(resizable_change(held, false, false, false), None);
        // In full screen, or on its way in or out.
        let full_screen = windowed | NSWindowStyleMask::FullScreen;
        assert_eq!(resizable_change(full_screen, false, false, false), None);
        assert_eq!(resizable_change(windowed, false, false, true), None);
        assert_eq!(resizable_change(held, true, false, true), None);
        // Borderless, captured or not.
        for resizable in [false, true] {
            assert_eq!(
                resizable_change(borderless_style(), resizable, true, false),
                None
            );
        }
    }

    /// A second window on one screen opens below and right of the first, not over it
    #[test]
    fn windows_on_one_screen_cascade() {
        let visible = rect((0.0, 100.0), (1728.0, 962.0));
        let sizing = Sizing {
            mode: (1920, 1080),
            guest_scale: 2.0,
            scale: 2.0,
            chrome: NSSize::new(0.0, 32.0),
        };
        let first = rect((384.0, 295.0), (960.0, 572.0));
        assert_eq!(cascade(first, &sizing, visible, &[]), first);
        let second = rect((408.0, 271.0), (960.0, 572.0));
        assert_eq!(cascade(first, &sizing, visible, &[first]), second);
        assert_eq!(
            cascade(first, &sizing, visible, &[first, second]),
            rect((432.0, 247.0), (960.0, 572.0))
        );
        // Elsewhere on the screen: no move.
        let other = rect((0.0, 100.0), (960.0, 572.0));
        assert_eq!(cascade(first, &sizing, visible, &[other]), first);
    }

    /// A cascaded window stays inside the visible area; where even its smallest content would
    /// not fit, it stays where it last was
    #[test]
    fn a_cascade_stays_on_the_screen() {
        let visible = rect((0.0, 0.0), (400.0, 260.0));
        let sizing = Sizing {
            mode: (640, 400),
            guest_scale: 1.0,
            scale: 1.0,
            chrome: NSSize::new(0.0, 30.0),
        };
        // The fit: 368x229 of content, centred.
        let first = rect((16.0, 0.0), (368.0, 259.0));
        let second = cascade(first, &sizing, visible, &[first]);
        assert_eq!(second, rect((40.0, 1.0), (328.0, 234.0)));
        // A third would have 289x181 of content, below the smallest.
        assert_eq!(cascade(first, &sizing, visible, &[first, second]), second);
        for frame in [first, second] {
            assert!(frame.origin.x >= visible.origin.x && frame.origin.y >= visible.origin.y);
            assert!(frame.origin.x + frame.size.width <= visible.size.width);
            assert!(frame.origin.y + frame.size.height <= visible.size.height);
        }
    }

    /// A window that fills the visible area does not open over another such window: it opens 24
    /// points right and down, made smaller from there, its content keeping the guest's ratio on
    /// an even number of whole pixels
    #[test]
    fn a_cascaded_window_shrinks_rather_than_overlap() {
        let visible = rect((0.0, 100.0), (1728.0, 1110.0));
        let sizing = Sizing {
            mode: (3456, 2160),
            guest_scale: 2.0,
            scale: 2.0,
            chrome: NSSize::new(0.0, 30.0),
        };
        let content = sizing.content(visible.size);
        assert_eq!(content, NSSize::new(1728.0, 1080.0));
        let first = NSRect::new(visible.origin, sizing.frame(content));
        assert_eq!(first, visible);
        // 1689.6x1056 of content, on an even number of whole pixels: 3378x2112.
        let second = cascade(first, &sizing, visible, &[first]);
        assert_eq!(second, rect((24.0, 100.0), (1689.0, 1086.0)));
        let third = cascade(first, &sizing, visible, &[first, second]);
        assert_eq!(third, rect((48.0, 100.0), (1651.0, 1062.0)));
    }
}
