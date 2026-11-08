//! A window's render thread, which draws the display's current frame into the window's
//! `CAMetalLayer` when there is something new to show, and the [`Surface`] it shares with the
//! window and its view.
//!
//! One thread per window (`reims-vgpu.window`), started when the window attaches. It waits until
//! there is something to draw: a new guest frame, a new layout (the view resized or moved to a
//! screen of another scale, or the guest set a mode), the window showing again, or a change of
//! the guest's cursor where the window draws it ([`Window::drawn_cursor`], which each draw hands
//! the sink). It draws only while the window is at least partly visible, once its layer and
//! layout exist.
//!
//! Each draw goes through the sink's presenter over a Vulkan surface of the layer
//! ([`super::metal_surface::LayerTarget`]), as on Linux (`crate::present`): made at the first
//! draw, tried again after [`RETRY`] where no image came in time, made again where one ends.
//! Its swapchain sizes the layer's drawables (from the layer's bounds and scale, which the view
//! keeps), and takes them. The presenter goes before the thread lets its reference to the layer
//! go.
//!
//! Requests that come while it draws fold into one: it draws the latest frame once more. It
//! copies the sink out of the window's lock before a draw, and holds no lock of the window's
//! while it draws.
//!
//! It ends at the window's end ([`Surface::close`]). Nothing joins it: a release that races a
//! draw ends the display on this thread (the draw's hold on the display is then the last one),
//! and the window's end waits for nothing. The thread holds the window, so the window's last
//! reference may go here.
//!
//! **The layer off the main thread.** objc2 does not make a `CAMetalLayer` `Send`. This thread
//! only has a presenter make a Vulkan surface of it, whose driver retains it and sets it up as
//! any thread may (`metal_surface`). Its reference to the layer is released on the main thread,
//! as the view's is ([`LayerRef`]).

use std::sync::Arc;

use objc2::Message;
use objc2::rc::Retained;
use objc2_core_graphics::{CGColor, CGColorSpace, kCGColorBlack, kCGColorSpaceSRGB};
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;
use reims_vgpu_vulkan::core::rt::spawn_named;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::exit_gate;
use reims_vgpu_vulkan::os::OsThreads;
use reims_vgpu_vulkan::os::msg;

use super::main_thread::post_main;
use super::metal_surface::LayerTarget;
use crate::geometry::Layout;
use crate::handle::Window;
use crate::present::{Presenting, RETRY};

/// The render thread's stack.
const STACK: usize = 512 * 1024;

/// A window's `CAMetalLayer`, as the render thread holds it.
///
/// Made on the main thread with the view, and released on the main thread: the surface's
/// reference is taken back there at the window's end ([`Surface::take_layer`]), and the render
/// thread posts its own there as it ends.
pub(crate) struct LayerRef(Retained<CAMetalLayer>);

// SAFETY: the render thread only retains the layer and makes a Vulkan surface of it, whose driver
// retains it and sets it up as any thread may, and lets its retain go before the layer's last
// (`metal_surface`); every release of a `LayerRef` is on the main thread (above).
unsafe impl Send for LayerRef {}

impl LayerRef {
    /// A reference to `layer`, made on the main thread.
    pub(crate) fn new(layer: &CAMetalLayer) -> LayerRef {
        LayerRef(layer.retain())
    }

    /// The layer's pointer, which a Vulkan surface is made of (`metal_surface`).
    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_void {
        Retained::as_ptr(&self.0).cast()
    }

    /// Another reference to the same layer.
    fn retain(&self) -> LayerRef {
        LayerRef(self.0.clone())
    }
}

/// Sets up `layer` in BGRA8, tagged sRGB, opaque on black, its drawables not framebuffer-only,
/// as the swapchain's images take transfers: with no colour space, Core Animation shows the
/// pixels in the screen's own gamut, oversaturated on a P3 screen. The swapchain sets the layer's
/// device. On the main thread, before the render thread has the layer.
pub(crate) fn configure_layer(layer: &CAMetalLayer) {
    msg::get("setPixelFormat:", || {
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    });
    // SAFETY: `kCGColorSpaceSRGB` is a constant CoreGraphics defines.
    let srgb = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }));
    msg::get("setColorspace:", || layer.setColorspace(srgb.as_deref()));
    // SAFETY: `kCGColorBlack` is a constant CoreGraphics defines.
    let black = CGColor::constant_color(Some(unsafe { kCGColorBlack }));
    msg::get("setBackgroundColor:", || {
        layer.setBackgroundColor(black.as_deref());
    });
    msg::get("setOpaque:", || layer.setOpaque(true));
    msg::get("setFramebufferOnly:", || layer.setFramebufferOnly(false));
}

/// What the render thread draws with, shared with the window and its view.
#[derive(Debug, Default)]
pub(crate) struct Surface {
    state: Mutex<SurfaceState>,
    wake: Condvar,
}

/// The surface's state.
#[derive(Default)]
struct SurfaceState {
    /// The view's layer, once the window has opened on the main thread.
    layer: Option<LayerRef>,
    /// Where the drawable and the image are, once the view has laid out.
    layout: Option<Layout>,
    /// Whether any part of the window is on screen (its occlusion state).
    visible: bool,
    /// Something new to show: a frame, a layout, the window showing again.
    dirty: bool,
    /// The window has ended: the render thread ends.
    closed: bool,
}

impl std::fmt::Debug for SurfaceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceState")
            .field("layer", &self.layer.is_some())
            .field("layout", &self.layout)
            .field("visible", &self.visible)
            .field("dirty", &self.dirty)
            .field("closed", &self.closed)
            .finish()
    }
}

/// A draw the render thread takes: the layer, the first time, and the layout.
struct Work {
    layer: Option<LayerRef>,
    layout: Layout,
}

impl SurfaceState {
    /// The draw to make now, if any: something new, the window visible, its layer and layout
    /// there. `want_layer` asks for a reference to the layer.
    fn take(&mut self, want_layer: bool) -> Option<Work> {
        if !(self.dirty && self.visible) {
            return None;
        }
        let (Some(layer), Some(layout)) = (&self.layer, self.layout) else {
            return None;
        };
        self.dirty = false;
        Some(Work {
            layer: want_layer.then(|| layer.retain()),
            layout,
        })
    }

    /// The draw to make again after a presenter's that had no image in time (`layout`): the
    /// newest, else the same, while the window is visible. None while it is not: the draw waits
    /// for it to show, which asks for one ([`Surface::set_visible`]).
    fn retake(&mut self, want_layer: bool, layout: Layout) -> Option<Work> {
        if !self.visible {
            return None;
        }
        Some(self.take(want_layer).unwrap_or(Work {
            layer: None,
            layout,
        }))
    }
}

impl Surface {
    /// An empty surface: no layer, no layout, not visible.
    pub(crate) fn new() -> Surface {
        Surface::default()
    }

    /// Changes the state with `f` and wakes the render thread.
    fn update(&self, f: impl FnOnce(&mut SurfaceState)) {
        f(&mut self.state.lock());
        self.wake.notify_all();
    }

    /// The view's layer, on the main thread as the window opens.
    pub(crate) fn set_layer(&self, layer: LayerRef) {
        self.update(|s| {
            s.layer = Some(layer);
            s.dirty = true;
        });
    }

    /// The surface's reference to the layer, taken back on the main thread at the window's end,
    /// to be released there.
    pub(crate) fn take_layer(&self) -> Option<LayerRef> {
        self.state.lock().layer.take()
    }

    /// The view's layout: a changed one is drawn.
    pub(crate) fn set_layout(&self, layout: Layout) {
        self.update(|s| {
            if s.layout != Some(layout) {
                s.layout = Some(layout);
                s.dirty = true;
            }
        });
    }

    /// Whether the window is at least partly on screen: nothing is drawn while it is not, and it
    /// is drawn again when it shows.
    pub(crate) fn set_visible(&self, visible: bool) {
        self.update(|s| {
            if visible && !s.visible {
                s.dirty = true;
            }
            s.visible = visible;
        });
    }

    /// The guest presented a frame: it is drawn.
    pub(crate) fn request_draw(&self) {
        self.update(|s| s.dirty = true);
    }

    /// The window's end: the render thread ends.
    pub(crate) fn close(&self) {
        self.update(|s| s.closed = true);
    }

    /// Waits for the next draw; `None` at the window's end. `retry` asks for a draw of that
    /// layout again after [`RETRY`], where nothing newer comes first.
    fn next(&self, want_layer: bool, retry: Option<Layout>) -> Option<Work> {
        let mut state = self.state.lock();
        if let Some(layout) = retry
            && !state.closed
        {
            self.wake.wait_for(&mut state, RETRY);
            if state.closed {
                return None;
            }
            if let Some(work) = state.retake(want_layer, layout) {
                return Some(work);
            }
        }
        loop {
            if state.closed {
                return None;
            }
            if let Some(work) = state.take(want_layer) {
                return Some(work);
            }
            self.wake.wait(&mut state);
        }
    }
}

/// Starts `window`'s render thread.
pub(crate) fn start(window: Arc<Window>) {
    spawn_named(
        &OsThreads::new(),
        "reims-vgpu.window",
        STACK,
        Box::new(()),
        move || run(&window),
    );
}

/// The render thread's body: draws until the window's end, then lets its presenter go, and hands
/// its reference to the layer to the main thread.
fn run(window: &Window) {
    let surface = window.surface();
    let mut layer: Option<LayerRef> = None;
    let mut presenting = Presenting::default();
    let mut retry = None;
    while let Some(work) = surface.next(layer.is_none(), retry.take()) {
        // A frame is a step of the exit gate's: the process's exit waits for the frame in
        // progress, and the next one parks the thread for good.
        let Some(_step) = exit_gate::step() else {
            exit_gate::stay()
        };
        if work.layer.is_some() {
            layer = work.layer;
        }
        let Some(layer) = &layer else {
            continue;
        };
        // The sink is copied out of the window's lock: none is held while it draws.
        let Some(sink) = window.sink() else {
            continue;
        };
        let cursor = window.drawn_cursor();
        if presenting.needs_presenter() {
            presenting.make(|| sink.presenter(&LayerTarget(layer)));
        }
        if presenting.draw(work.layout.drawable, work.layout.pixels, cursor.as_ref()) {
            retry = Some(work.layout);
        }
    }
    // The swapchain and the surface go before the layer under them.
    drop(presenting);
    if let Some(layer) = layer {
        post_main(move |_| drop(layer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{self, Insets};

    /// A layout for a 100×50 point view at scale 2.
    fn a_layout() -> Layout {
        geometry::layout((100.0, 50.0), 2.0, Insets::default(), Some((1920, 1080)))
    }

    /// A layer, made off the main thread as Core Animation allows.
    fn a_layer() -> LayerRef {
        LayerRef(CAMetalLayer::new())
    }

    /// Nothing is drawn while the window is hidden, and the window is drawn again as it shows
    #[test]
    fn a_hidden_window_is_drawn_when_it_shows() {
        let mut s = SurfaceState {
            layer: Some(a_layer()),
            layout: Some(a_layout()),
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take(false).is_none());
        s.visible = true;
        assert!(s.take(false).is_some());
        assert!(s.take(false).is_none(), "one draw per request");
    }

    /// Nothing is drawn before the view has a layer and a layout
    #[test]
    fn nothing_is_drawn_before_the_layer_and_layout() {
        let mut s = SurfaceState {
            visible: true,
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take(false).is_none());
        s.layout = Some(a_layout());
        assert!(s.take(false).is_none());
        s.layer = Some(a_layer());
        let work = s.take(true).unwrap();
        assert!(work.layer.is_some());
        assert_eq!(work.layout, a_layout());
    }

    /// Showing again, a new layout and a new frame each ask for a draw; an unchanged layout
    /// does not
    #[test]
    fn what_asks_for_a_draw() {
        let surface = Surface::new();
        surface.set_layer(a_layer());
        surface.set_layout(a_layout());
        surface.set_visible(true);
        assert!(surface.state.lock().take(false).is_some());
        surface.set_layout(a_layout());
        assert!(surface.state.lock().take(false).is_none());
        surface.request_draw();
        assert!(surface.state.lock().take(false).is_some());
        surface.set_visible(false);
        surface.set_visible(true);
        assert!(surface.state.lock().take(false).is_some());
        surface.close();
        assert!(surface.next(false, None).is_none());
        drop(surface.take_layer());
    }

    /// A presenter's draw with no image in time comes again, the newest layout first, while the
    /// window is visible; hidden, it waits for the window to show
    #[test]
    fn a_draw_with_no_image_comes_again_while_visible() {
        let surface = Surface::new();
        surface.set_layer(a_layer());
        surface.set_layout(a_layout());
        surface.set_visible(true);
        let work = surface.next(true, None).unwrap();
        assert!(work.layer.is_some());
        let again = surface.next(false, Some(work.layout)).unwrap();
        assert_eq!(again.layout, a_layout(), "the draw comes again");
        assert!(again.layer.is_none());
        let mut s = SurfaceState {
            layer: Some(a_layer()),
            layout: Some(a_layout()),
            ..SurfaceState::default()
        };
        assert!(s.retake(false, a_layout()).is_none(), "hidden, it does not");
        s.visible = true;
        assert!(s.retake(false, a_layout()).is_some(), "visible, it does");
        drop(s.layer.take());
        drop(surface.take_layer());
    }
}
