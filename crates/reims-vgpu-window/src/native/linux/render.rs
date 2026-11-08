//! A window's render thread, which draws the display's current frame into the window's swapchain
//! when there is something new to show, the [`Surface`] it shares with the window and the UI
//! thread, and what the display's sink gives it ([`FrameSink`](crate::FrameSink)).
//!
//! One thread per window (`reims-vgpu.window`), started when the window attaches. It waits until
//! there is something to draw: a new guest frame, a new layout (the window resized, or moved to a
//! screen of another scale, or the guest set a mode), the window showing again, or a change of
//! the guest's cursor where the window draws it ([`Window::drawn_cursor`], which each draw hands
//! the presenter). It draws only
//! while the window is shown, once SDL's window and the layout exist, through a presenter the sink
//! makes over the window ([`FrameSink::presenter`](crate::FrameSink::presenter)), which makes the
//! window's Vulkan surface ([`VulkanTarget`]), as `crate::present` says: made at the first draw,
//! tried again after [`RETRY`] where no image came in time, made again where one ends.
//!
//! Requests that come while it draws fold into one: it draws the latest frame once more. It
//! copies the sink out of the window's lock before asking it, and holds no lock of the window's
//! while it draws.
//!
//! It ends at the window's end ([`Surface::close`]): it drops its presenter, which destroys the
//! swapchain and the Vulkan surface, and only then posts to the UI thread that SDL's window may go
//! ([`crate::sdl::shell::released`]), so the surface always goes before the window under it.
//! Nothing joins it, and the window's end waits for nothing.

use std::sync::Arc;

use ash::vk;
use reims_vgpu_vulkan::core::rt::spawn_named;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::exit_gate;
use reims_vgpu_vulkan::os::OsThreads;
use sdl3_sys::everything::{SDL_Vulkan_CreateSurface, SDL_Window};

use crate::geometry::Layout;
use crate::handle::Window;
use crate::present::{Presenting, RETRY, VulkanTarget};

/// The render thread's stack.
const STACK: usize = 512 * 1024;

/// SDL's window, as its render thread holds it: only to make the window's Vulkan surface of it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target(pub(super) *mut SDL_Window);

// SAFETY: SDL's thread makes and destroys the window, and destroys it only once the render thread
// has let its surface go ([`crate::sdl::shell::released`]); the render thread, the only other one
// the target reaches, only has SDL make the window's Vulkan surface of it ([`Target`]'s
// `create_surface`), while the window lives.
unsafe impl Send for Target {}

impl VulkanTarget for Target {
    /// SDL's surface of the window: SDL looks the instance's commands up through the loader it
    /// opened, the same library as the runtime's, so the instance's own look-up is not needed.
    fn create_surface(
        &self,
        _get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR> {
        let mut surface = vk::SurfaceKHR::null();
        // SAFETY: the window is SDL's, made with `SDL_WINDOW_VULKAN`, and lives until the render
        // thread has posted its end ([`crate::sdl::shell::released`]); SDL only reads it here.
        let made =
            unsafe { SDL_Vulkan_CreateSurface(self.0, instance, std::ptr::null(), &mut surface) };
        if made {
            Some(surface)
        } else {
            log::warn!(
                target: "reims_vgpu_window",
                "SDL made no Vulkan surface: {}",
                crate::sdl::sdl_error()
            );
            None
        }
    }
}

/// What the render thread draws with, shared with the window and the UI thread.
#[derive(Debug, Default)]
pub(crate) struct Surface {
    state: Mutex<SurfaceState>,
    wake: Condvar,
}

/// The surface's state.
#[derive(Debug, Default)]
struct SurfaceState {
    /// SDL's window, once it has opened on the UI thread.
    target: Option<Target>,
    /// Where the drawable and the image are, once the window has laid out.
    layout: Option<Layout>,
    /// Whether the window is shown.
    visible: bool,
    /// Something new to show: a frame, a layout, the window showing again.
    dirty: bool,
    /// The window has ended: the render thread ends.
    closed: bool,
}

/// A draw the render thread takes: SDL's window and the layout.
struct Work {
    target: Target,
    layout: Layout,
}

impl SurfaceState {
    /// The draw to make now, if any: something new, the window shown, its window and layout
    /// there.
    fn take(&mut self) -> Option<Work> {
        if !(self.dirty && self.visible) {
            return None;
        }
        let (Some(target), Some(layout)) = (self.target, self.layout) else {
            return None;
        };
        self.dirty = false;
        Some(Work { target, layout })
    }

    /// The draw to make again after one that had no image in time: the newest, else the same,
    /// while the window is shown. None while it is hidden: the draw waits for it to show, which
    /// asks for one ([`Surface::set_visible`]).
    fn retake(&mut self, work: Work) -> Option<Work> {
        if !self.visible {
            return None;
        }
        Some(self.take().unwrap_or(work))
    }
}

impl Surface {
    /// An empty surface: no window, no layout, not shown.
    pub(crate) fn new() -> Surface {
        Surface::default()
    }

    /// Changes the state with `f` and wakes the render thread.
    fn update(&self, f: impl FnOnce(&mut SurfaceState)) {
        f(&mut self.state.lock());
        self.wake.notify_all();
    }

    /// SDL's window, from the UI thread as the window opens.
    pub(crate) fn set_target(&self, target: Target) {
        self.update(|s| {
            s.target = Some(target);
            s.dirty = true;
        });
    }

    /// The window's layout: a changed one is drawn.
    pub(crate) fn set_layout(&self, layout: Layout) {
        self.update(|s| {
            if s.layout != Some(layout) {
                s.layout = Some(layout);
                s.dirty = true;
            }
        });
    }

    /// Whether the window is shown: nothing is drawn while it is not, and it is drawn again when
    /// it shows.
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

    /// Waits for the next draw; `None` at the window's end. `retry` asks for the last draw
    /// again after [`RETRY`], where nothing newer comes first.
    fn next(&self, retry: Option<Work>) -> Option<Work> {
        let mut state = self.state.lock();
        if let Some(work) = retry
            && !state.closed
        {
            self.wake.wait_for(&mut state, RETRY);
            if state.closed {
                return None;
            }
            if let Some(work) = state.retake(work) {
                return Some(work);
            }
        }
        loop {
            if state.closed {
                return None;
            }
            if let Some(work) = state.take() {
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

/// The render thread's body: draws until the window's end, then lets its presenter go and posts
/// the window's release to the UI thread.
fn run(window: &Window) {
    let surface = window.surface();
    let mut presenting = Presenting::default();
    let mut retry = None;
    while let Some(work) = surface.next(retry.take()) {
        // A frame is a step of the exit gate's: the process's exit waits for the frame in
        // progress, and the next one parks the thread for good.
        let Some(_step) = exit_gate::step() else {
            exit_gate::stay()
        };
        if presenting.needs_presenter() {
            // The sink is copied out of the window's lock: none is held while it answers.
            let Some(sink) = window.sink() else {
                continue;
            };
            presenting.make(|| sink.presenter(&work.target));
        }
        let cursor = window.drawn_cursor();
        if presenting.draw(work.layout.drawable, work.layout.pixels, cursor.as_ref()) {
            retry = Some(work);
        }
    }
    drop(presenting);
    let id = window.id();
    super::ui::post(move || crate::sdl::shell::released(id));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{self, Insets};

    /// A layout for a 100×50 point window at scale 2.
    fn a_layout() -> Layout {
        geometry::layout((100.0, 50.0), 2.0, Insets::default(), Some((1920, 1080)))
    }

    /// A target that names no window, for the state's logic.
    fn a_target() -> Target {
        Target(std::ptr::null_mut())
    }

    /// Nothing is drawn while the window is hidden, and the window is drawn again as it shows
    #[test]
    fn a_hidden_window_is_drawn_when_it_shows() {
        let mut s = SurfaceState {
            target: Some(a_target()),
            layout: Some(a_layout()),
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take().is_none());
        s.visible = true;
        assert!(s.take().is_some());
        assert!(s.take().is_none(), "one draw per request");
    }

    /// Nothing is drawn before SDL's window and the layout
    #[test]
    fn nothing_is_drawn_before_the_window_and_layout() {
        let mut s = SurfaceState {
            visible: true,
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take().is_none());
        s.layout = Some(a_layout());
        assert!(s.take().is_none());
        s.target = Some(a_target());
        assert_eq!(s.take().map(|w| w.layout), Some(a_layout()));
    }

    /// Showing again, a new layout and a new frame each ask for a draw; an unchanged layout does
    /// not; a draw with no image comes again
    #[test]
    fn what_asks_for_a_draw() {
        let surface = Surface::new();
        surface.set_target(a_target());
        surface.set_layout(a_layout());
        surface.set_visible(true);
        assert!(surface.state.lock().take().is_some());
        surface.set_layout(a_layout());
        assert!(surface.state.lock().take().is_none());
        surface.request_draw();
        let work = surface.next(None).unwrap();
        let again = surface.next(Some(work)).unwrap();
        assert_eq!(
            again.layout,
            a_layout(),
            "the draw with no image comes again"
        );
        surface.set_visible(false);
        surface.set_visible(true);
        assert!(surface.state.lock().take().is_some());
        surface.close();
        assert!(surface.next(None).is_none());
    }

    /// A draw that had no image in time is not made again while the window is hidden: it is
    /// made once the window shows
    #[test]
    fn a_hidden_window_takes_no_retry() {
        let mut s = SurfaceState {
            target: Some(a_target()),
            layout: Some(a_layout()),
            visible: true,
            dirty: true,
            ..SurfaceState::default()
        };
        let work = s.take().unwrap();
        assert!(s.retake(work).is_some(), "shown, the draw comes again");
        let work = Work {
            target: a_target(),
            layout: a_layout(),
        };
        s.visible = false;
        assert!(s.retake(work).is_none(), "hidden, it does not");
        let surface = Surface::new();
        surface.set_target(a_target());
        surface.set_layout(a_layout());
        surface.set_visible(false);
        surface.set_visible(true);
        assert!(
            surface.state.lock().take().is_some(),
            "shown again, it is drawn"
        );
    }
}
