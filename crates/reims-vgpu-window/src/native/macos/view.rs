//! A window's view: the content view that hosts the `CAMetalLayer` the render thread draws into,
//! lays the guest's image out in it, gives its events to the window's input, and shows the
//! guest's cursor over the image. Main thread only.
//!
//! **The layer.** The view is layer-hosting: its `CAMetalLayer` is set before the view wants a
//! layer, so AppKit never draws into it. The layer is BGRA8, tagged sRGB (with no colour space,
//! Core Animation shows the pixels in the screen's own gamut, oversaturated on a P3 screen),
//! opaque on black, at the window's backing scale, its drawables not framebuffer-only, as the
//! swapchain's images take transfers; the swapchain sets its device.
//!
//! **The layout.** The view is flipped, so its points run y down as [`geometry`]'s do. Whenever
//! its size, its window's backing scale, its screen or the guest's mode changes, it lays the
//! image out ([`geometry::layout`]): inside the screen's safe area when the window is borderless,
//! so the image sits below a camera housing. It keeps the layout for its events and cursor, and
//! gives it to the render thread, which draws a changed one.
//!
//! **Events.** Every mouse, scroll and key event the view receives goes to the window's
//! [`InputState`] with a [`ViewContext`]: the event's point in the view (a mouse or scroll event),
//! the image's rectangle, the guest's mode and whether the window is key. A reaction that changed
//! the capture refreshes the title, the cursor and the pointer's hold; one that passes the event on
//! sends it to `super`, down AppKit's responder chain. The click that makes the window key is
//! AppKit's alone (`acceptsFirstMouse:` is NO). A tracking area gives the view the pointer's moves
//! while its window is key.
//!
//! **The cursor.** A cursor rectangle over the image shows the guest's cursor
//! ([`CursorCache`]), sized as the image shows it; the bars and the title bar keep the host's
//! arrow. The rectangle is set again when the glyph, its visibility, the layout or the capture
//! changes, and that puts the new cursor on screen at once, the pointer still. Measured on the
//! M3 Max, macOS 27.0.1 (26A434), with a scratch AppKit window opened under the pointer, key in
//! the active application, the pointer never moving: after `invalidateCursorRectsForView:`,
//! AppKit called `resetCursorRects` at once, and within 50 ms the rectangle's new cursor was the
//! application's (`currentCursor`) and the system's (`currentSystemCursor`, by its image's
//! size), each of three changes. A window that is not key has no cursor rectangles in effect, as
//! for any AppKit window.
//!
//! **The pointer's hold.** While the input is captured, the window holds the host's pointer over
//! the image ([`InputState::pointer_confinement`]): NSWindow's `mouseConfinementRect`, the image's
//! rectangle in the window's coordinates, set again whenever the capture or the layout changes,
//! and freed (`NSZeroRect`) when capture ends and as the window closes. AppKit has had the
//! property since macOS 10.13.2 and leaves it out of its headers; SDL3 holds its windows' pointers
//! with it (read from SDL 3.4.18's code), so SDL's window on macOS gets the same hold through
//! SDL's mouse rect (`crate::sdl::shell`). Where a window does not answer it, the pointer stays
//! free, and one line says so. Measured on the M3 Max, macOS 27.0.1 (26A434): NSWindow answers
//! it; a pointer warped (`CGWarpMouseCursorPosition`) or moved by events posted at the HID event
//! tap lands outside the rectangle, so only a person's own mouse shows the hold (the manual walk,
//! `scripts/internal/walk_window_macos.sh`).
//!
//! **The window's size.** While the input is captured, a windowed window cannot be resized
//! ([`InputState::window_resizable`]): where the image reaches the window's edges, the pointer
//! held over it is over the window's resize edges too, where a click meant for the guest would
//! drag the window smaller. The view drops `Resizable` from the window's style as capture starts,
//! and puts it back as capture ends and as the window closes
//! ([`super::shell::resizable_change`]). It leaves the style alone in native full screen and on
//! the way in or out, which the delegate tells it, and sets it again once the window has left; a
//! borderless window's style has no `Resizable`. Measured on the M3 Max, macOS 27.0.1 (26A434),
//! on a titled window not on screen: dropping `Resizable` and putting it back keeps the frame,
//! the content's rectangle and the view's bounds, the first responder, the frame view, the
//! minimum size, the collection behaviour and the `mouseConfinementRect`, and sends no
//! `windowDidResize:` (which would end the following of the guest's mode) and no
//! `viewDidMoveToWindow`.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel,
};
use objc2_app_kit::{
    NSEvent, NSEventType, NSResponder, NSTrackingArea, NSTrackingAreaOptions, NSView, NSWindow,
    NSWindowOcclusionState, NSWindowStyleMask,
};
use objc2_core_foundation::CGFloat;
use objc2_foundation::{NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize};
use objc2_quartz_core::CAMetalLayer;
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_vulkan::os::msg;

use super::cursor::CursorCache;
use super::input::{InputState, Reaction, ViewContext};
use super::main_thread::{ns_text, responds, shared_app};
use super::render::{LayerRef, configure_layer};
use crate::geometry::{self, Insets, Layout, Rect};
use crate::handle::Window;

/// A view's state.
pub(crate) struct ViewIvars {
    /// The window the view shows.
    window: Arc<Window>,
    /// Whether the window is borderless: the image then fits inside its screen's safe area.
    borderless: bool,
    /// The layer the render thread draws into.
    layer: Retained<CAMetalLayer>,
    /// The window's input.
    input: RefCell<InputState>,
    /// The cursors built for the guest's glyph.
    cursors: RefCell<CursorCache>,
    /// The last layout.
    layout: Cell<Layout>,
    /// Whether the window is in native full screen, or on its way in or out: its style is then
    /// left alone (module doc).
    full_screen: Cell<bool>,
}

define_class!(
    /// A window's content view (module doc).
    // SAFETY:
    // - NSView has no subclassing requirements beyond calling `super` where the overridden
    //   method needs it, which each override here does.
    // - The view does not implement `Drop`.
    #[unsafe(super(NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ViewIvars]
    pub(crate) struct DisplayView;

    impl DisplayView {
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            let _env = BlockEnv::foreign();
            true
        }

        #[unsafe(method(isOpaque))]
        fn is_opaque(&self) -> bool {
            let _env = BlockEnv::foreign();
            true
        }

        #[unsafe(method(acceptsFirstResponder))]
        fn accepts_first_responder(&self) -> bool {
            let _env = BlockEnv::foreign();
            true
        }

        /// The click that brings the window forward is not the guest's.
        #[unsafe(method(acceptsFirstMouse:))]
        fn accepts_first_mouse(&self, _event: Option<&NSEvent>) -> bool {
            let _env = BlockEnv::foreign();
            false
        }

        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: NSSize) {
            let _env = BlockEnv::foreign();
            // SAFETY: `setFrameSize:` takes an NSSize and returns nothing.
            msg::get("setFrameSize:", || unsafe {
                let _: () = msg_send![super(self), setFrameSize: size];
            });
            self.relayout();
        }

        #[unsafe(method(viewDidChangeBackingProperties))]
        fn did_change_backing_properties(&self) {
            let _env = BlockEnv::foreign();
            // SAFETY: `viewDidChangeBackingProperties` takes nothing and returns nothing.
            msg::get("viewDidChangeBackingProperties", || unsafe {
                let _: () = msg_send![super(self), viewDidChangeBackingProperties];
            });
            self.relayout();
        }

        #[unsafe(method(viewDidMoveToWindow))]
        fn did_move_to_window(&self) {
            let _env = BlockEnv::foreign();
            // SAFETY: `viewDidMoveToWindow` takes nothing and returns nothing.
            msg::get("viewDidMoveToWindow", || unsafe {
                let _: () = msg_send![super(self), viewDidMoveToWindow];
            });
            self.relayout();
        }

        #[unsafe(method(resetCursorRects))]
        fn reset_cursor_rects(&self) {
            let _env = BlockEnv::foreign();
            self.add_cursor_rect();
        }

        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_down) {
                // SAFETY: an NSResponder event method takes the event and returns nothing.
                pass("mouseDown:", || unsafe { msg_send![super(self), mouseDown: event] });
            }
        }

        #[unsafe(method(rightMouseDown:))]
        fn right_mouse_down(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_down) {
                // SAFETY: as for `mouseDown:`.
                pass("rightMouseDown:", || unsafe {
                    msg_send![super(self), rightMouseDown: event]
                });
            }
        }

        #[unsafe(method(otherMouseDown:))]
        fn other_mouse_down(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_down) {
                // SAFETY: as for `mouseDown:`.
                pass("otherMouseDown:", || unsafe {
                    msg_send![super(self), otherMouseDown: event]
                });
            }
        }

        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_up) {
                // SAFETY: as for `mouseDown:`.
                pass("mouseUp:", || unsafe { msg_send![super(self), mouseUp: event] });
            }
        }

        #[unsafe(method(rightMouseUp:))]
        fn right_mouse_up(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_up) {
                // SAFETY: as for `mouseDown:`.
                pass("rightMouseUp:", || unsafe {
                    msg_send![super(self), rightMouseUp: event]
                });
            }
        }

        #[unsafe(method(otherMouseUp:))]
        fn other_mouse_up(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_up) {
                // SAFETY: as for `mouseDown:`.
                pass("otherMouseUp:", || unsafe {
                    msg_send![super(self), otherMouseUp: event]
                });
            }
        }

        #[unsafe(method(mouseMoved:))]
        fn mouse_moved(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_moved) {
                // SAFETY: as for `mouseDown:`.
                pass("mouseMoved:", || unsafe { msg_send![super(self), mouseMoved: event] });
            }
        }

        #[unsafe(method(mouseDragged:))]
        fn mouse_dragged(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_moved) {
                // SAFETY: as for `mouseDown:`.
                pass("mouseDragged:", || unsafe {
                    msg_send![super(self), mouseDragged: event]
                });
            }
        }

        #[unsafe(method(rightMouseDragged:))]
        fn right_mouse_dragged(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_moved) {
                // SAFETY: as for `mouseDown:`.
                pass("rightMouseDragged:", || unsafe {
                    msg_send![super(self), rightMouseDragged: event]
                });
            }
        }

        #[unsafe(method(otherMouseDragged:))]
        fn other_mouse_dragged(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::mouse_moved) {
                // SAFETY: as for `mouseDown:`.
                pass("otherMouseDragged:", || unsafe {
                    msg_send![super(self), otherMouseDragged: event]
                });
            }
        }

        #[unsafe(method(scrollWheel:))]
        fn scroll_wheel(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.mouse(event, InputState::scroll_wheel) {
                // SAFETY: as for `mouseDown:`.
                pass("scrollWheel:", || unsafe { msg_send![super(self), scrollWheel: event] });
            }
        }

        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.key(event, InputState::key_down) {
                // SAFETY: as for `mouseDown:`.
                pass("keyDown:", || unsafe { msg_send![super(self), keyDown: event] });
            }
        }

        #[unsafe(method(keyUp:))]
        fn key_up(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.key(event, InputState::key_up) {
                // SAFETY: as for `mouseDown:`.
                pass("keyUp:", || unsafe { msg_send![super(self), keyUp: event] });
            }
        }

        #[unsafe(method(flagsChanged:))]
        fn flags_changed(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if self.key(event, InputState::flags_changed) {
                // SAFETY: as for `mouseDown:`.
                pass("flagsChanged:", || unsafe {
                    msg_send![super(self), flagsChanged: event]
                });
            }
        }
    }

    unsafe impl NSObjectProtocol for DisplayView {}
);

/// Sends an event on to `super` (`send`, a send of `sel`). An exception inside AppKit's handling
/// of it stops at the send and is logged, as `-[NSApplication run]` would log it.
pub(crate) fn pass(sel: &'static str, send: impl FnOnce()) {
    if let Err(caught) = msg::catch(sel, send) {
        log::error!(
            "{sel} raised {}: {}; the event is dropped",
            caught.name,
            caught.reason
        );
    }
}

impl DisplayView {
    /// The view of `window`, `size` points, with a `CAMetalLayer` (module doc).
    pub(crate) fn new(
        mtm: MainThreadMarker,
        window: Arc<Window>,
        borderless: bool,
        size: NSSize,
    ) -> Retained<DisplayView> {
        let layer = new_layer();
        let input = InputState::new(Arc::clone(&window));
        let ivars = ViewIvars {
            window,
            borderless,
            layer,
            input: RefCell::new(input),
            cursors: RefCell::new(CursorCache::new()),
            layout: Cell::new(Layout::default()),
            full_screen: Cell::new(false),
        };
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), size);
        // SAFETY: `initWithFrame:` is NSView's designated initializer, taking an NSRect.
        let this: Retained<DisplayView> = msg::get("initWithFrame:", || unsafe {
            msg_send![super(DisplayView::alloc(mtm).set_ivars(ivars)), initWithFrame: frame]
        });
        let layer: &CAMetalLayer = &this.ivars().layer;
        // A layer-hosting view: the layer first, then the wish for one.
        msg::get("setLayer:", || this.setLayer(Some(layer)));
        msg::get("setWantsLayer:", || this.setWantsLayer(true));
        let options = NSTrackingAreaOptions::MouseMoved
            | NSTrackingAreaOptions::MouseEnteredAndExited
            | NSTrackingAreaOptions::ActiveInKeyWindow
            | NSTrackingAreaOptions::InVisibleRect;
        let owner: &DisplayView = &this;
        // SAFETY: the owner is the view, which receives the area's events; there is no user info.
        let area = msg::get("initWithRect:options:owner:userInfo:", || unsafe {
            NSTrackingArea::initWithRect_options_owner_userInfo(
                NSTrackingArea::alloc(),
                NSRect::ZERO,
                options,
                Some(owner),
                None,
            )
        });
        msg::get("addTrackingArea:", || this.addTrackingArea(&area));
        this
    }

    /// A reference to the view's layer, for the render thread.
    pub(crate) fn layer_ref(&self) -> LayerRef {
        LayerRef::new(&self.ivars().layer)
    }

    /// The render thread's surface's reference to the layer, taken back at the window's end to
    /// be released on the main thread.
    pub(crate) fn take_surface_layer(&self) -> Option<LayerRef> {
        self.ivars().window.surface().take_layer()
    }

    /// Lays the image out again, for the view's size, its window's backing scale and screen, and
    /// the guest's mode; a changed layout goes to the render thread, and moves the cursor's
    /// rectangle.
    pub(crate) fn relayout(&self) {
        let ivars = self.ivars();
        let window = msg::get("window", || self.window());
        let scale = window.as_deref().map_or(1.0, |w| {
            msg::get("backingScaleFactor", || w.backingScaleFactor())
        });
        let insets = match (&window, ivars.borderless) {
            (Some(w), true) => msg::get("screen", || w.screen())
                .map(|screen| super::shell::safe_area(&screen))
                .unwrap_or_default(),
            _ => Insets::default(),
        };
        let bounds = msg::get("bounds", || self.bounds()).size;
        let layout = geometry::layout(
            (bounds.width, bounds.height),
            scale,
            insets,
            ivars.window.mode(),
        );
        let layer = &ivars.layer;
        if msg::get("contentsScale", || layer.contentsScale()) != scale {
            msg::get("setContentsScale:", || layer.setContentsScale(scale));
        }
        if ivars.layout.replace(layout) != layout {
            log::debug!(
                "layout: {bounds:?} points at scale {scale}, insets {insets:?}: {layout:?}"
            );
            self.invalidate_cursor();
        }
        // The image's rectangle in the window's coordinates moves with the window's height, which
        // a layout in the view's flipped points may not show.
        if self.is_captured() {
            self.confine_pointer();
        }
        ivars.window.surface().set_layout(layout);
    }

    /// The guest's mode, once it has set one.
    pub(crate) fn guest_mode(&self) -> Option<(u16, u16)> {
        self.ivars().window.mode()
    }

    /// Whether the window never resizes for the guest's mode (`Window::keeps_its_size`).
    pub(crate) fn keeps_its_size(&self) -> bool {
        self.ivars().window.keeps_its_size()
    }

    /// The window's title, without what the capture state adds.
    pub(crate) fn title(&self) -> &str {
        self.ivars().window.title()
    }

    /// Tells the render thread whether any of the window is on screen.
    pub(crate) fn occlusion_changed(&self) {
        let visible = msg::get("window", || self.window()).is_some_and(|w| {
            msg::get("occlusionState", || w.occlusionState())
                .contains(NSWindowOcclusionState::Visible)
        });
        log::debug!(
            "window {:?} visible: {visible}",
            self.ivars().window.title()
        );
        self.ivars().window.surface().set_visible(visible);
    }

    /// The window's title: its `Window`'s, then what the capture state adds.
    pub(crate) fn refresh_title(&self) {
        let suffix = self
            .ivars()
            .input
            .try_borrow()
            .map_or("", |input| input.title_suffix());
        let title = ns_text(&format!("{}{}", self.ivars().window.title(), suffix));
        if let Some(window) = msg::get("window", || self.window()) {
            msg::get("setTitle:", || window.setTitle(&title));
        }
    }

    /// Has AppKit set the cursor's rectangle again (`resetCursorRects`).
    pub(crate) fn invalidate_cursor(&self) {
        if let Some(window) = msg::get("window", || self.window()) {
            msg::get("invalidateCursorRectsForView:", || {
                window.invalidateCursorRectsForView(self);
            });
        }
    }

    /// The guest's cursor or its visibility changed.
    pub(crate) fn cursor_changed(&self) {
        self.ivars().window.take_cursor_refresh();
        self.invalidate_cursor();
    }

    /// The cursor rectangle over the image, with the guest's cursor, invisible while the window
    /// draws it itself (`handle`'s module doc); none (the host's arrow) before the guest has set
    /// one.
    fn add_cursor_rect(&self) {
        let ivars = self.ivars();
        let guest = ivars.window.guest();
        let image = ivars.layout.get().image;
        let per_pixel = geometry::cursor_points_per_pixel(image, guest.mode.unwrap_or((0, 0)));
        let cursor = ivars.cursors.try_borrow_mut().ok().and_then(|mut cursors| {
            cursors.cursor(
                guest.cursor.as_deref(),
                guest.host_cursor_visible(),
                per_pixel,
                self.mtm(),
            )
        });
        if let Some(cursor) = cursor {
            msg::get("addCursorRect:cursor:", || {
                self.addCursorRect_cursor(ns_rect(image), &cursor);
            });
        }
    }

    /// What the view knows as `event` reaches it; the event's point when `located`.
    fn context(&self, event: &NSEvent, located: bool) -> ViewContext {
        let point = located.then(|| {
            let location = msg::get("locationInWindow", || event.locationInWindow());
            let point = msg::get("convertPoint:fromView:", || {
                self.convertPoint_fromView(location, None)
            });
            (point.x, point.y)
        });
        let window_is_key = msg::get("window", || self.window())
            .is_some_and(|w| msg::get("isKeyWindow", || w.isKeyWindow()));
        ViewContext {
            point,
            image: self.ivars().layout.get().image,
            guest: self.ivars().window.mode(),
            window_is_key,
        }
    }

    /// Gives a mouse or scroll event to the input with `f`: true when it passes the event on.
    fn mouse(
        &self,
        event: &NSEvent,
        f: impl FnOnce(&mut InputState, &NSEvent, &ViewContext) -> Reaction,
    ) -> bool {
        let cx = self.context(event, true);
        let reaction = match self.ivars().input.try_borrow_mut() {
            Ok(mut input) => f(&mut input, event, &cx),
            Err(_) => Reaction::default(),
        };
        self.react(reaction)
    }

    /// Gives a key event to the input with `f`: true when it passes the event on.
    fn key(&self, event: &NSEvent, f: impl FnOnce(&mut InputState, &NSEvent) -> Reaction) -> bool {
        let reaction = match self.ivars().input.try_borrow_mut() {
            Ok(mut input) => f(&mut input, event),
            Err(_) => Reaction::default(),
        };
        self.react(reaction)
    }

    /// Acts on the input's reaction: a changed capture refreshes the title, the cursor, the
    /// pointer's hold and whether the window may be resized. True when the event goes on to
    /// `super`.
    fn react(&self, reaction: Reaction) -> bool {
        if reaction.capture_changed {
            self.refresh_title();
            self.invalidate_cursor();
            self.confine_pointer();
            self.hold_size();
        }
        reaction.pass_on
    }

    /// Makes a windowed window not resizable while the input is captured, and resizable again
    /// when not, but in native full screen and on the way in or out (module doc).
    fn hold_size(&self) {
        let Ok(input) = self.ivars().input.try_borrow() else {
            return;
        };
        let resizable = input.window_resizable();
        drop(input);
        let Some(window) = msg::get("window", || self.window()) else {
            return;
        };
        let mut style = msg::get("styleMask", || window.styleMask());
        let ivars = self.ivars();
        let Some(resizable) = super::shell::resizable_change(
            style,
            resizable,
            ivars.borderless,
            ivars.full_screen.get(),
        ) else {
            return;
        };
        style.set(NSWindowStyleMask::Resizable, resizable);
        msg::get("setStyleMask:", || window.setStyleMask(style));
        log::debug!(
            "window {:?}: {}",
            ivars.window.title(),
            if resizable {
                "resizable again"
            } else {
                "not resizable while captured"
            }
        );
    }

    /// The window is entering native full screen (`entering`), or has left it or failed to
    /// enter it: its style is left alone until then, and follows the capture again after.
    pub(crate) fn full_screen_changed(&self, entering: bool) {
        self.ivars().full_screen.set(entering);
        if !entering {
            self.hold_size();
        }
    }

    /// Holds the host's pointer over the image while the input is captured, and frees it when
    /// not (module doc).
    fn confine_pointer(&self) {
        let Ok(input) = self.ivars().input.try_borrow() else {
            return;
        };
        let held = input.pointer_confinement(self.ivars().layout.get().image);
        drop(input);
        let Some(window) = msg::get("window", || self.window()) else {
            return;
        };
        let rect = held.map_or(NSRect::ZERO, |image| {
            msg::get("convertRect:toView:", || {
                self.convertRect_toView(ns_rect(image), None)
            })
        });
        set_mouse_confinement(&window, rect);
    }

    /// Runs `f` on the input and refreshes the title, the cursor and the pointer's hold when the
    /// capture changed under it.
    fn with_input<R>(&self, f: impl FnOnce(&mut InputState) -> R) -> Option<R> {
        let (result, changed) = {
            let mut input = self.ivars().input.try_borrow_mut().ok()?;
            let before = input.is_captured();
            let result = f(&mut input);
            (result, input.is_captured() != before)
        };
        if changed {
            self.react(Reaction {
                capture_changed: true,
                pass_on: false,
            });
        }
        Some(result)
    }

    /// The application saw `event` while this view's window is key, before AppKit dispatches
    /// it: true when the input took it.
    pub(crate) fn app_send_event(&self, event: &NSEvent) -> bool {
        let located = self.is_located_here(event);
        let cx = self.context(event, located);
        self.with_input(|input| input.app_send_event(event, &cx))
            .unwrap_or(false)
    }

    /// Whether `event` is a mouse or scroll event in this view's window.
    fn is_located_here(&self, event: &NSEvent) -> bool {
        let located = matches!(
            msg::get("type", || event.r#type()),
            NSEventType::LeftMouseDown
                | NSEventType::LeftMouseUp
                | NSEventType::RightMouseDown
                | NSEventType::RightMouseUp
                | NSEventType::OtherMouseDown
                | NSEventType::OtherMouseUp
                | NSEventType::MouseMoved
                | NSEventType::LeftMouseDragged
                | NSEventType::RightMouseDragged
                | NSEventType::OtherMouseDragged
                | NSEventType::ScrollWheel
        );
        located
            && msg::get("window", || self.window()).is_some_and(|w| {
                msg::get("windowNumber", || w.windowNumber())
                    == msg::get("windowNumber", || event.windowNumber())
            })
    }

    /// The window stopped being key.
    pub(crate) fn focus_lost(&self) {
        let reaction = self
            .ivars()
            .input
            .try_borrow_mut()
            .map_or_else(|_| Reaction::default(), |mut input| input.focus_lost());
        self.react(reaction);
    }

    /// The person chose Release Input.
    pub(crate) fn release_capture(&self) {
        let reaction = self
            .ivars()
            .input
            .try_borrow_mut()
            .map_or_else(|_| Reaction::default(), |mut input| input.release_capture());
        self.react(reaction);
    }

    /// Whether the window's input, the keyboard and the pointer, is captured.
    pub(crate) fn is_captured(&self) -> bool {
        self.ivars()
            .input
            .try_borrow()
            .is_ok_and(|input| input.is_captured())
    }

    /// The window is closing, or about to: every key held goes up, capture ends, the pointer
    /// goes free, and the window is resizable again.
    pub(crate) fn window_closing(&self) {
        if let Ok(mut input) = self.ivars().input.try_borrow_mut() {
            input.window_closing();
        }
        self.confine_pointer();
        self.hold_size();
    }

    /// The person closed the window or chose Quit: the input lets go, and the host is told,
    /// which ends the VM. The window stays until the display's release closes it.
    pub(crate) fn close_requested(&self) {
        log::info!(
            "window {:?}: closed by the person",
            self.ivars().window.title()
        );
        self.window_closing();
        let close = self.ivars().window.callbacks().and_then(|c| c.close);
        if let Some(close) = close {
            close();
        }
    }
}

/// The key window's view, when it is a display's.
pub(crate) fn key_view(mtm: MainThreadMarker) -> Option<Retained<DisplayView>> {
    let app = shared_app(mtm);
    let window = msg::get("keyWindow", || app.keyWindow())?;
    let content = msg::get("contentView", || window.contentView())?;
    msg::get("isKindOfClass:", || content.downcast::<DisplayView>()).ok()
}

/// Whether the key window's view, a display's, takes `event` before AppKit dispatches it.
pub(crate) fn key_view_takes(event: &NSEvent, mtm: MainThreadMarker) -> bool {
    key_view(mtm).is_some_and(|view| view.app_send_event(event))
}

/// A new `CAMetalLayer`, set up for the window (module doc).
fn new_layer() -> Retained<CAMetalLayer> {
    let layer = msg::get("new", CAMetalLayer::new);
    configure_layer(&layer);
    layer
}

/// Whether a window not answering `mouseConfinementRect` was logged: it is said once.
static NO_HOLD_SAID: AtomicBool = AtomicBool::new(false);

/// Holds the pointer in `rect`, in `window`'s coordinates, or frees it for `NSRect::ZERO`: the
/// window's `mouseConfinementRect` (module doc). Where the window does not answer it, the pointer
/// stays free, and one line says so.
fn set_mouse_confinement(window: &NSWindow, rect: NSRect) {
    if !responds(window, sel!(setMouseConfinementRect:)) {
        if rect != NSRect::ZERO && !NO_HOLD_SAID.swap(true, Ordering::Relaxed) {
            log::warn!(
                "the pointer is not held over the image while input is captured: NSWindow has \
                 no mouseConfinementRect"
            );
        }
        return;
    }
    // SAFETY: `setMouseConfinementRect:` takes an NSRect and returns nothing.
    msg::get("setMouseConfinementRect:", || unsafe {
        let _: () = msg_send![window, setMouseConfinementRect: rect];
    });
}

/// `rect` as an NSRect in the flipped view's points.
fn ns_rect(rect: Rect) -> NSRect {
    NSRect::new(
        NSPoint::new(rect.x as CGFloat, rect.y as CGFloat),
        NSSize::new(rect.width as CGFloat, rect.height as CGFloat),
    )
}

#[cfg(test)]
mod tests {
    use objc2_metal::MTLPixelFormat;

    use super::*;

    /// A layer is BGRA8 and opaque, and its drawables are not framebuffer-only: the swapchain's
    /// images take transfers
    #[test]
    fn a_layer_is_bgra8_opaque_and_takes_transfers() {
        let layer = new_layer();
        assert!(!layer.framebufferOnly());
        assert_eq!(layer.pixelFormat(), MTLPixelFormat::BGRA8Unorm);
        assert!(layer.isOpaque());
    }
}
