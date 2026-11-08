//! What a window's render thread presents through: the display's sink presents the frames itself,
//! over a Vulkan swapchain of the window's surface, on either OS. The sink makes a [`Presenter`]
//! over the window ([`VulkanTarget`]: SDL's window on Linux, the window's `CAMetalLayer` on macOS),
//! and the render thread has it draw each frame ([`Presenting`]).
//!
//! **The presenter's life.** At its first draw the render thread asks the sink for a presenter;
//! without one (no swapchain on this host) it never draws, and the window stays empty. Each draw
//! has the presenter draw the current frame into its next image and present it
//! ([`Presenter::draw`]): taking the image is the pacing, at the screen's own rate. Where no image
//! came in time the render thread tries again after [`RETRY`]. A presenter that can draw no more
//! ([`Drawn::Ended`]: its surface is lost) is let go, and the next draw asks the sink for a new
//! one, up to [`RENEWALS`] times in a row without a frame presented; the window then stays as it
//! is.

use std::time::Duration;

use ash::vk;

use crate::handle::{DrawnCursor, PixelRect};

/// How long the render thread waits before it draws again where no image came in time: about a
/// refresh at 120 Hz. Before it answers [`Drawn::Retry`] for want of an image, the presenter has
/// waited for one, up to 33 ms, and drawn as soon as the presentation engine gave one back
/// (`reims_vgpu_vulkan::swapchain`): where the engine has every image queued or shown, the draws
/// follow the screen's pace, not this one.
pub(crate) const RETRY: Duration = Duration::from_millis(8);

/// How many presenters in a row the render thread makes again after one ends, with no frame
/// presented between: past that, the window stays as it is.
const RENEWALS: u32 = 3;

/// What a draw did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    /// The frame was drawn and presented.
    Presented,
    /// Nothing was drawn: the display has ended, or refused the encode. The image taken is kept
    /// for the next draw.
    Nothing,
    /// No image came in time: the draw is tried again.
    Retry,
    /// The presenter can draw no more: the render thread asks the sink for a new one at its next
    /// draw (the module doc).
    Ended,
}

/// What a window's render thread presents through: the sink's, over the window's swapchain.
pub trait Presenter: Send {
    /// Draws the display's current frame into the next image of `drawable` pixels (width,
    /// height), over `region` of it, clearing the rest, with `cursor` over it where the window
    /// draws the guest's cursor, and presents it. Blocks only for the next image, up to a few
    /// frames.
    fn draw(
        &mut self,
        drawable: (u64, u64),
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
    ) -> Drawn;
}

/// What draws a display's current frame into a window: `reims-vgpu`'s, over the display. It holds
/// the display weakly: a window never keeps its display alive, except for a frame in flight, as
/// any frame does. It draws through a presenter of its own over the window's Vulkan surface
/// ([`FrameSink::presenter`]): of SDL's window on Linux, of the window's `CAMetalLayer` on macOS,
/// whose swapchain takes the layer's drawables and sets the layer up for them.
pub trait FrameSink: Send + Sync {
    /// A presenter over `target`, the window: one that makes the window's Vulkan surface on the
    /// sink's instance and a swapchain over it. `None` where there is none (the display has ended,
    /// or this host has no swapchain): the window then stays empty.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>>;

    /// The guest's cursor position, in its mode's pixels, as the display reads it now: `None`
    /// once the display has ended, without a guest display, while it is paused, and for a sink
    /// that cannot tell. Asked on the thread of the display's callbacks
    /// ([`Window::cursor_moved`](crate::Window::cursor_moved)).
    fn guest_cursor_position(&self) -> Option<(u16, u16)> {
        None
    }
}

/// What makes a window's Vulkan surface, on its render thread: on Linux SDL's window, on macOS
/// the window's `CAMetalLayer`.
pub trait VulkanTarget {
    /// The window's surface, made on `instance` (whose commands `get_instance_proc_addr`, the
    /// loader's the instance was made with, looks up), which the caller owns from then on (and
    /// destroys before its instance), or `None` where none is made.
    fn create_surface(
        &self,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR>;
}

/// A render thread's presenter, made of the sink at the first draw and made again after one ends,
/// up to [`RENEWALS`] times in a row (the module doc).
#[derive(Default)]
pub(crate) struct Presenting {
    /// The presenter, once made.
    presenter: Option<Box<dyn Presenter>>,
    /// The window stays as it is: the sink made no presenter, or the presenters ended too many
    /// times in a row.
    ended: bool,
    /// The presenters made again since the last frame presented.
    renewals: Renewals,
}

impl Presenting {
    /// Whether the next draw needs a presenter made first ([`Presenting::make`]).
    pub(crate) fn needs_presenter(&self) -> bool {
        !self.ended && self.presenter.is_none()
    }

    /// Makes the presenter with `make`, the sink's: where it makes none, the window stays as it
    /// is.
    pub(crate) fn make(&mut self, make: impl FnOnce() -> Option<Box<dyn Presenter>>) {
        self.presenter = make();
        self.ended |= self.presenter.is_none();
    }

    /// Has the presenter draw the current frame into an image of `drawable` pixels over
    /// `region`, with `cursor` over it: whether to draw again after [`RETRY`] (no image came in
    /// time). Nothing without a presenter.
    pub(crate) fn draw(
        &mut self,
        drawable: (u64, u64),
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
    ) -> bool {
        if self.ended {
            return false;
        }
        let Some(presenter) = self.presenter.as_mut() else {
            return false;
        };
        let drawn = presenter.draw(drawable, region, cursor);
        if drawn == Drawn::Ended {
            self.presenter = None;
        }
        if !self.renewals.after(drawn) {
            log::warn!(
                target: "reims_vgpu_window",
                "the window's presenter ended {RENEWALS} times in a row: the window stays as it is"
            );
            self.ended = true;
        }
        drawn == Drawn::Retry
    }
}

/// The presenters a render thread has made again since its last frame presented.
#[derive(Default)]
struct Renewals(u32);

impl Renewals {
    /// Counts what a draw did: whether the render thread may make a new presenter, where it
    /// needs one.
    fn after(&mut self, drawn: Drawn) -> bool {
        match drawn {
            Drawn::Presented => self.0 = 0,
            Drawn::Ended => self.0 = self.0.saturating_add(1),
            Drawn::Nothing | Drawn::Retry => {}
        }
        self.0 <= RENEWALS
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    /// A presenter that answers what it is told, in order, then `Presented`.
    struct Scripted(VecDeque<Drawn>);

    impl Presenter for Scripted {
        fn draw(
            &mut self,
            _drawable: (u64, u64),
            _region: PixelRect,
            _cursor: Option<&DrawnCursor>,
        ) -> Drawn {
            self.0.pop_front().unwrap_or(Drawn::Presented)
        }
    }

    /// A presenter answering `drawn` in order.
    fn scripted(drawn: &[Drawn]) -> Option<Box<dyn Presenter>> {
        Some(Box::new(Scripted(drawn.iter().copied().collect())))
    }

    /// A region.
    const REGION: PixelRect = PixelRect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };

    /// A presenter that ends is made again, up to [`RENEWALS`] times in a row, and a frame
    /// presented starts the count again
    #[test]
    fn an_ended_presenter_is_made_again_a_few_times() {
        let mut r = Renewals::default();
        for _ in 0..RENEWALS {
            assert!(r.after(Drawn::Ended));
            assert!(r.after(Drawn::Retry) && r.after(Drawn::Nothing));
        }
        assert!(r.after(Drawn::Presented));
        for _ in 0..RENEWALS {
            assert!(r.after(Drawn::Ended));
        }
        assert!(!r.after(Drawn::Ended), "the window stays as it is");
    }

    /// The presenter is made at the first draw and kept; a draw with no image asks for a retry;
    /// one that ends is made again at the next draw; a sink that makes none leaves the window as
    /// it is, asked no more
    #[test]
    fn presenting_makes_keeps_and_renews_the_presenter() {
        let mut p = Presenting::default();
        assert!(p.needs_presenter());
        p.make(|| scripted(&[Drawn::Retry, Drawn::Presented, Drawn::Ended]));
        assert!(!p.needs_presenter());
        assert!(p.draw((4, 4), REGION, None), "no image: a retry");
        assert!(!p.draw((4, 4), REGION, None));
        assert!(!p.draw((4, 4), REGION, None));
        assert!(p.needs_presenter(), "an ended presenter is made again");
        p.make(|| None);
        assert!(
            !p.needs_presenter(),
            "no presenter: the window stays as it is"
        );
        assert!(!p.draw((4, 4), REGION, None));
    }

    /// Presenters that end too many times in a row leave the window as it is
    #[test]
    fn presenters_ending_in_a_row_leave_the_window_as_it_is() {
        let mut p = Presenting::default();
        for _ in 0..=RENEWALS {
            assert!(p.needs_presenter());
            p.make(|| scripted(&[Drawn::Ended]));
            p.draw((4, 4), REGION, None);
        }
        assert!(!p.needs_presenter());
    }
}
