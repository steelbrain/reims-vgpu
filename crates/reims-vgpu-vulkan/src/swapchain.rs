//! What a display window's swapchain takes of the window system's rules, on Linux, Windows and
//! macOS: the surface's format, the present mode and the images asked for, the size, the take of
//! an image, the remake, the retired swapchains and the one lock over the window system's calls.
//! The rt platform's swapchain (`crate::rt_platform::swapchain`) presents through them. The
//! surface is the window's: on Linux SDL's (Wayland's or X11's), on Windows SDL's
//! (`VK_KHR_win32_surface`), on macOS one over the window's `CAMetalLayer`
//! (`VK_EXT_metal_surface`).
//!
//! **Making it.** The presenting queue's family must present to the surface, which must offer
//! `B8G8R8A8_UNORM` in `SRGB_NONLINEAR` ([`FORMAT`], [`COLOR_SPACE`]): the format the display
//! draws into, which the compositor reads as sRGB. Otherwise there is no swapchain, and the window
//! stays empty; the reason is logged.
//!
//! **Present mode.** Never one that tears, and never one that makes the guest wait: MAILBOX
//! where the surface offers it, with at least four images, so that taking an image rarely waits
//! for the compositor to give one back; else FIFO, with at least three ([`present_mode`]). Either
//! way a frame faster than the screen replaces or queues behind the one before: the window shows
//! up to the screen's rate. FIFO_LATEST_READY (`VK_EXT_present_mode_fifo_latest_ready`) is not
//! asked for: it would serve only a surface without MAILBOX, and every surface measured offers
//! MAILBOX (an RTX 5080 Laptop GPU's NVIDIA driver under KWin).
//!
//! **A compositor on another GPU (measured).** The swapchain is made on the runtime's device.
//! Where the compositor composites on another GPU (a GPU in an eGPU enclosure or an x4 slot beside
//! the integrated GPU that drives the screen, or a laptop's discrete GPU beside the panel's), each
//! present crosses the GPU's link once. An RTX 3090 on PCIe 3.0 x4 under KWin on a Radeon 780M
//! (NVIDIA 620.70.06, 2026-11-13) handed KWin four LINEAR XRGB8888 buffers of the window's size
//! (3648 x 2039, 29.75 MB), and each present moved one of them; on a desktop walk the window
//! presented 0.93 times for each of the guest's presents. The 3090's link carries 1.83 GB/s from
//! the GPU to the host, whichever queue family copies, so a present of that size takes 16.4 ms of
//! it: 61 presents a second at most, with nothing else crossing. Only its graphics family presents
//! to the surface. Where the compositor runs on the runtime's GPU (an RTX 5080 Laptop GPU driving
//! its panel), the buffers are the GPU's own tiled images, and nothing crosses.
//! docs/graphics-kext/design.md (Decided, 14) has the walks.
//!
//! **Pacing, on the render thread.** Each draw takes the next image with a fence, and waits for the
//! fence, up to [`WAIT_NS`]: that is the render thread's wait for the compositor, as `nextDrawable`
//! is on a Metal window. Where the presentation engine has no image to give (every one
//! queued or shown, as a FIFO-only surface's are when frames come faster than the screen shows
//! them), the take waits for one in the window system, in calls of [`TAKE_NS`] each, for up to
//! [`WAIT_NS`] in all, read on the monotonic clock ([`take_image`]): the draw goes on as the engine
//! gives one back, at the screen's own pace, rather than after the window's fixed retry (8 ms,
//! longer than a refresh at 144 Hz and above, by when the frames queued before the draw may all
//! have been shown, and the screen shows one twice). Where none comes in that time, the draw is
//! tried again. The frame's command buffer waits on no semaphore of the compositor's, so a frame
//! the display counts as in flight completes when the GPU does, never when the compositor gives an
//! image back: a window the compositor stops showing (a closed lid, a minimized window) slows its
//! own render thread, never the guest's presents. A fence wait that runs out gives
//! [`Acquired::Retry`]: the image stays taken, and the next draw waits for it again.
//!
//! **Size.** The swapchain is made at the size the window asks for, in pixels: Wayland gives a
//! surface no size of its own and reports no out-of-date swapchain when the window's pixels
//! change, so a draw at another size makes a new one. Where the surface has a size of its own
//! (X11), the swapchain takes that one, and the window's size otherwise within the surface's
//! limits ([`extent_for`]): during a resize the window's size can be a step behind the
//! surface's. One the compositor reports out of date or suboptimal is made again at the next
//! draw. Making a new one waits for the old one's frames to complete, then passes the old one as
//! `oldSwapchain` and retires it (below), whether or not the create makes the new one: Vulkan
//! retires it either way. A make refused before the create (the surface has no pixels for now,
//! as a minimized window's may, or refuses its capabilities) hands Vulkan nothing, so the old
//! one stays the current one, stale, and the next draw hands it over. With none to hand over
//! (the first make, or the one after a refused create), the retired swapchains are destroyed
//! first, waited for as at the swapchain's end (below): Vulkan refuses a create with no
//! `oldSwapchain` while the window has a swapchain, and a retired one is the window's until it
//! is destroyed (`VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`, read from the specification of
//! `vkCreateSwapchainKHR` and `VkSwapchainCreateInfoKHR`). A swapchain the surface refuses for
//! now is tried again at the next draw ([`remake`]); only a lost surface, or a lost device, ends
//! the swapchain's draws ([`Acquired::Failed`]).
//!
//! **Retiring one, with no wait on the queue.** A retired swapchain, its images' textures and
//! their semaphores are destroyed once nothing uses them: its frames' command buffers have
//! completed (waited for as it retires), and so has its last present, its wait on a semaphore
//! included. Where the device has present fences (`swapchainMaintenance1`,
//! [`crate::os::rt_present_fences`]), each present signals a fence of its own once the presentation
//! engine is done with its semaphore and image, which tells it exactly ([`Fences`]). Two rules of
//! Vulkan's tell it too, with no fence ([`Presents`]): an image taken again, its fence waited for,
//! has its previous present complete, the wait on its semaphore included (the Vulkan Guide's
//! "Swapchain Semaphore Reuse"); and the presents made on one queue are performed in the order
//! they were made (`vkQueuePresentKHR`), whatever the swapchain, and a present the compositor
//! rejects as out of date or lost still waits on its semaphores. So once a present made after a
//! swapchain's last is known complete, so is that one. Until then the retired swapchain is kept,
//! up to [`RETIRED`] of them. Past that (a resize dragged out over many frames retires one a
//! frame, faster than any present is known complete), and at the swapchain's end, where a present
//! is not known complete, the swapchain waits for the presents' fences, up to [`END_WAIT_NS`];
//! where it has none, or they do not signal in that time, it waits for the presenting queue to
//! idle, the only other way unextended Vulkan has, and one its specification does not cover
//! either.
//!
//! **One window-system call at a time.** Every call any swapchain of the process makes into the
//! window system (the surface's queries and its end, making and destroying swapchains, taking
//! images and presenting) runs under one lock, [`WSI`]. Measured on an RTX 5080 Laptop GPU (NVIDIA
//! 615.71.09, KWin on Wayland, two windows): `vkQueuePresentKHR` on one window's render thread
//! crashed inside the driver while the other's made its swapchain, though Vulkan allows both at
//! once. The lock is held for those calls alone, never while a draw waits for a fence or a frame
//! for the GPU. A take asks for at most [`TAKE_NS`] in one call, and between its calls the lock is
//! handed to a window waiting for it, so no window waits on another for more than a call (NVIDIA's
//! driver returns from such a call later than asked: [`TAKE_NS`]). A present takes it before the
//! presenting queue's own lock.

use std::collections::VecDeque;

use crate::core::sync::Mutex;
use ash::vk;

/// How long a draw waits for the compositor to give an image, in nanoseconds: about two frames
/// at 60 Hz.
pub const WAIT_NS: u64 = 33_000_000;

/// How long one take of an image asks to wait in the window system, in nanoseconds
/// ([`take_image`]): the take runs under the lock every call into the window system holds
/// ([`WSI`]), so another window's call waits for it about this long. NVIDIA's driver returns from
/// such a call with no image later than asked: after 1.9 ms on average on a nested KWin's virtual
/// output, and about 3.9 ms on KWin whose only output was off (measured on an RTX 5080 Laptop
/// GPU, NVIDIA 615.71.09, Wayland, FIFO, 2026-10-22). So the take counts its wait on the
/// monotonic clock as well as in calls.
const TAKE_NS: u64 = 1_000_000;

/// How long a swapchain's end waits for an image the compositor has yet to give, in
/// nanoseconds, before it leaves the fence behind rather than destroy it in use; and how long it
/// waits for presents' fences before it waits for the queue to idle (the module doc).
pub(crate) const END_WAIT_NS: u64 = 1_000_000_000;

/// How many retired swapchains are kept for their last presents to be known complete before the
/// queue is waited idle (the module doc): each keeps its images until it is destroyed.
pub(crate) const RETIRED: usize = 4;

/// The images a swapchain asks for at least, by present mode.
pub(crate) const MAILBOX_IMAGES: u32 = 4;
/// With FIFO.
pub(crate) const FIFO_IMAGES: u32 = 3;

/// The format the display draws into, and its colour space.
pub(crate) const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
/// The colour space the compositor reads it in.
pub(crate) const COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;
/// The usages a swapchain asks its images for, of those the surface supports.
pub(crate) const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw(),
);

/// The lock every call into the window system runs under (the module doc).
pub(crate) static WSI: Mutex<()> = Mutex::new(());

/// What a draw takes.
#[derive(Debug)]
pub enum Acquired {
    /// An image to draw into, ready.
    Image(Frame),
    /// No image yet: the compositor has not given one back in time, or the swapchain was made
    /// again. The draw is tried again.
    Retry,
    /// No image, nor will there be: the surface is lost, or the device.
    Failed,
}

/// An image a draw took, until it is presented or kept for the next draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Its index in the swapchain.
    index: u32,
}

impl Frame {
    /// The frame of image `index`, for the rt platform's swapchain
    /// (`crate::rt_platform::swapchain`).
    pub(crate) fn at(index: u32) -> Frame {
        Frame { index }
    }

    /// Its image's index in the swapchain.
    pub(crate) fn index(self) -> u32 {
        self.index
    }
}

/// The fences of a swapchain's presents, where the device has present fences (the module doc).
#[derive(Debug, Default)]
pub(crate) struct Fences {
    /// Fences unsignalled, with no present pending on them.
    pub(crate) free: Vec<vk::Fence>,
    /// The fences of the presents not yet seen signalled, each with its present's number, in the
    /// order the presents were made.
    pub(crate) pending: VecDeque<(u64, vk::Fence)>,
}

impl Fences {
    /// A fence for a present: a free one, else a new one; none where the device makes none, and
    /// the present goes without.
    pub(crate) fn take(&mut self, vk: &ash::Device) -> Option<vk::Fence> {
        if let Some(fence) = self.free.pop() {
            return Some(fence);
        }
        // SAFETY: a valid create info, on the runtime's live device.
        match unsafe { vk.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(fence) => Some(fence),
            Err(e) => {
                log::warn!(
                    target: "reims_vgpu_vulkan::swapchain", kind:? = e;
                    "vkCreateFence: {e}"
                );
                None
            }
        }
    }

    /// `fence` was chained into present number `number`, which answered `result`: pending where
    /// the present's operations were enqueued (it was taken, or rejected as out of date or for a
    /// lost surface, which still waits on its semaphores), else free again, as a present that
    /// enqueues nothing leaves its fence as it was.
    pub(crate) fn presented(
        &mut self,
        number: u64,
        fence: vk::Fence,
        result: Result<bool, vk::Result>,
    ) {
        match result {
            Ok(_)
            | Err(
                vk::Result::ERROR_OUT_OF_DATE_KHR
                | vk::Result::ERROR_SURFACE_LOST_KHR
                | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
            ) => self.pending.push_back((number, fence)),
            Err(_) => self.free.push(fence),
        }
    }

    /// Every fence, free or pending.
    pub(crate) fn all(&self) -> impl Iterator<Item = vk::Fence> + '_ {
        (self.free.iter().copied()).chain(self.pending.iter().map(|&(_, fence)| fence))
    }
}

/// What a swapchain knows of its presents (the module doc): they are numbered from 1 in the order
/// they are made, all on one queue, and the newest known complete, its semaphore wait included,
/// tells that every one before it is too.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Presents {
    /// The presents made.
    pub(crate) made: u64,
    /// The newest present known complete; 0 for none.
    pub(crate) complete: u64,
}

impl Presents {
    /// A present made: its number.
    pub(crate) fn make(&mut self) -> u64 {
        self.made += 1;
        self.made
    }

    /// An image taken again, its fence waited for, whose last present was number `last` (0 for
    /// none): that present is complete, and every one made before it.
    pub(crate) fn taken_again(&mut self, last: u64) {
        self.complete = self.complete.max(last);
    }

    /// Whether present number `last`, and so every one before it, is known complete.
    pub(crate) fn done(&self, last: u64) -> bool {
        last <= self.complete
    }

    /// Present number `number`'s fence signalled: it is complete, and every one made before it.
    pub(crate) fn signalled(&mut self, number: u64) {
        self.complete = self.complete.max(number);
    }

    /// The queue was waited idle: every present made is taken as complete (the module doc).
    pub(crate) fn idle(&mut self) {
        self.complete = self.made;
    }
}

/// Why a swapchain is not made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a swapchain's make made none, by whether Vulkan was asked for one with the old
/// swapchain as its `oldSwapchain`, which retires the old one whatever the answer.
#[derive(Debug)]
pub(crate) enum Unmade {
    /// Refused before the create: the old swapchain is still the window's.
    Unasked(Refused),
    /// The create, or the wrap of its images, refused: the old swapchain is retired.
    Asked(Refused),
}

/// The steps of a remake ([`remake`]), apart from the swapchain, so that their order is tested
/// without a window system.
pub(crate) trait Remaking {
    /// A swapchain.
    type Chain;
    /// Takes the current swapchain out, once its frames have completed.
    fn take_current(&mut self) -> Option<Self::Chain>;
    /// Whether a retired swapchain lives.
    fn has_retired(&self) -> bool;
    /// Destroys every retired swapchain, once its presents are complete.
    fn end_retired(&mut self);
    /// A new swapchain, with `old` as its `oldSwapchain`.
    fn make(&mut self, old: Option<&Self::Chain>) -> Result<Self::Chain, Unmade>;
    /// Retires `chain`: it is destroyed once its presents are complete.
    fn retire(&mut self, chain: Self::Chain);
    /// Makes `chain` the current swapchain, stale or not.
    fn set_current(&mut self, chain: Option<Self::Chain>, stale: bool);
}

/// Makes the swapchain again (the module doc, "Size"). With no current swapchain to hand over,
/// the retired ones go first: a create that passes no `oldSwapchain` is refused while one lives
/// (`VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`). A make refused before the create leaves the old
/// swapchain current, stale, since Vulkan was never handed it: retired there, it would live on
/// as the window's, and every later create, with none to hand over, would be refused. One the
/// create refused, or the wrap of its images, retired the old one all the same.
pub(crate) fn remake<R: Remaking>(r: &mut R) -> Result<(), Refused> {
    let old = r.take_current();
    if old.is_none() && r.has_retired() {
        r.end_retired();
    }
    match r.make(old.as_ref()) {
        Ok(chain) => {
            if let Some(old) = old {
                r.retire(old);
            }
            r.set_current(Some(chain), false);
            Ok(())
        }
        Err(Unmade::Unasked(e)) => {
            r.set_current(old, true);
            Err(e)
        }
        Err(Unmade::Asked(e)) => {
            if let Some(old) = old {
                r.retire(old);
            }
            Err(e)
        }
    }
}

/// A refusal naming `what` failed with `result`.
pub(crate) fn refused(what: &str, result: vk::Result) -> Refused {
    Refused(format!("{what}: {result}"))
}

/// An image taken with `take`, a call of `vkAcquireNextImageKHR` that waits in the window system
/// up to the nanoseconds it is given: each call asks for [`TAKE_NS`], and one that answers no
/// image is made again, up to [`WAIT_NS`] in all (the module doc). So the draw goes on as soon as
/// the presentation engine gives an image back, rather than a retry after a sleep that knows
/// nothing of when it does. The wait ends at whichever comes first of `WAIT_NS / TAKE_NS` calls,
/// so a window system that answers at once whatever the time it is given is asked no more often,
/// and [`WAIT_NS`] passed since the take began, as `elapsed` reads it in nanoseconds, so one that
/// returns later than asked (NVIDIA's, [`TAKE_NS`]) is asked no longer. What the last call
/// answered: a timeout, or `NOT_READY`, where none came in that time.
pub(crate) fn take_image(
    mut take: impl FnMut(u64) -> Result<(u32, bool), vk::Result>,
    mut elapsed: impl FnMut() -> u64,
) -> Result<(u32, bool), vk::Result> {
    let mut calls = 0;
    loop {
        let taken = take(TAKE_NS);
        calls += 1;
        match taken {
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY)
                if calls < WAIT_NS / TAKE_NS && elapsed() < WAIT_NS => {}
            taken => return taken,
        }
    }
}

/// The present mode of a swapchain over a surface offering `modes` (the module doc): MAILBOX
/// where offered, else FIFO, which every surface offers.
pub(crate) fn present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// The size in pixels of the images of a swapchain for a window of `size` pixels on a surface
/// of `caps`: the surface's own size where it has one, else `size` within the surface's limits.
/// `None` where that has no pixel (a minimized window's surface may allow none).
pub(crate) fn extent_for(
    size: (u64, u64),
    caps: &vk::SurfaceCapabilitiesKHR,
) -> Option<vk::Extent2D> {
    let extent = if caps.current_extent.width == u32::MAX {
        let within = |wanted: u64, min: u32, max: u32| {
            u32::try_from(wanted)
                .unwrap_or(u32::MAX)
                .clamp(min, max.max(min))
        };
        vk::Extent2D {
            width: within(
                size.0,
                caps.min_image_extent.width,
                caps.max_image_extent.width,
            ),
            height: within(
                size.1,
                caps.min_image_extent.height,
                caps.max_image_extent.height,
            ),
        }
    } else {
        caps.current_extent
    };
    (extent.width > 0 && extent.height > 0).then_some(extent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A surface's capabilities: its own size (`u32::MAX` wide where it has none) and its limits.
    fn caps(current: (u32, u32), min: (u32, u32), max: (u32, u32)) -> vk::SurfaceCapabilitiesKHR {
        let extent = |(width, height)| vk::Extent2D { width, height };
        vk::SurfaceCapabilitiesKHR {
            current_extent: extent(current),
            min_image_extent: extent(min),
            max_image_extent: extent(max),
            ..vk::SurfaceCapabilitiesKHR::default()
        }
    }

    /// The pixels of an extent, for comparing.
    fn pixels(extent: Option<vk::Extent2D>) -> Option<(u32, u32)> {
        extent.map(|e| (e.width, e.height))
    }

    /// MAILBOX where the surface offers it, else FIFO; never a mode that tears
    #[test]
    fn mailbox_is_taken_where_offered_else_fifo() {
        use vk::PresentModeKHR as M;
        assert!(present_mode(&[M::IMMEDIATE, M::FIFO, M::MAILBOX]) == M::MAILBOX);
        assert!(present_mode(&[M::FIFO, M::FIFO_RELAXED, M::IMMEDIATE]) == M::FIFO);
        assert!(present_mode(&[M::FIFO]) == M::FIFO);
    }

    /// A present is known complete once an image presented at it or after it is taken again, and
    /// with it every present before it; a swapchain whose last present is known complete goes
    #[test]
    fn a_present_is_known_complete_once_a_later_image_is_taken_again() {
        let mut p = Presents::default();
        assert!(p.done(0), "a swapchain that presented nothing goes at once");
        let old_last = [p.make(), p.make(), p.make()][2];
        let new_first = p.make();
        assert!(!p.done(old_last));
        p.taken_again(0);
        assert!(!p.done(old_last), "an image never presented tells nothing");
        p.taken_again(new_first);
        assert!(p.done(old_last) && p.done(new_first));
        p.taken_again(1);
        assert!(
            p.done(new_first),
            "an older present taken again does not move it back"
        );
        let later = p.make();
        assert!(!p.done(later));
        p.idle();
        assert!(p.done(later), "the queue waited idle");
    }

    /// A surface with a size of its own (X11's) gives the swapchain that size, whatever the
    /// window's, which can be a step behind it during a resize
    #[test]
    fn a_surface_of_its_own_size_sizes_the_swapchain() {
        let x11 = caps((800, 600), (800, 600), (800, 600));
        assert_eq!(pixels(extent_for((1024, 768), &x11)), Some((800, 600)));
        assert_eq!(pixels(extent_for((800, 600), &x11)), Some((800, 600)));
    }

    /// A surface with no size of its own (Wayland's) takes the window's, within its limits
    #[test]
    fn a_surface_without_a_size_takes_the_windows_within_its_limits() {
        let wayland = caps((u32::MAX, u32::MAX), (1, 1), (16384, 16384));
        assert_eq!(pixels(extent_for((1024, 768), &wayland)), Some((1024, 768)));
        assert_eq!(
            pixels(extent_for((20000, 768), &wayland)),
            Some((16384, 768))
        );
        assert_eq!(
            pixels(extent_for((u64::MAX, 1), &wayland)),
            Some((16384, 1))
        );
        let small = caps((u32::MAX, u32::MAX), (64, 64), (4096, 4096));
        assert_eq!(pixels(extent_for((10, 5000), &small)), Some((64, 4096)));
    }

    /// Limits the wrong way round give the smallest the surface allows, not a panic
    #[test]
    fn crossed_limits_give_the_smallest() {
        let crossed = caps((u32::MAX, u32::MAX), (100, 100), (50, 50));
        assert_eq!(pixels(extent_for((10, 1000), &crossed)), Some((100, 100)));
    }

    /// A surface with no pixels for now (a minimized window's may say 0×0) makes no swapchain:
    /// the draw is tried again
    #[test]
    fn a_surface_without_pixels_makes_no_swapchain() {
        let minimized = caps((0, 0), (0, 0), (0, 0));
        assert_eq!(pixels(extent_for((1024, 768), &minimized)), None);
        let flat = caps((u32::MAX, u32::MAX), (0, 0), (0, 4096));
        assert_eq!(pixels(extent_for((1024, 768), &flat)), None);
    }

    /// The window system's answers to the takes of one draw, in order: the last answers every
    /// take after it. What it gives back: each take's timeout.
    fn takes(
        answers: &[Result<(u32, bool), vk::Result>],
    ) -> (Result<(u32, bool), vk::Result>, Vec<u64>) {
        takes_of(answers, 0)
    }

    /// [`takes`], each call taking `call_ns` of the clock the take reads.
    fn takes_of(
        answers: &[Result<(u32, bool), vk::Result>],
        call_ns: u64,
    ) -> (Result<(u32, bool), vk::Result>, Vec<u64>) {
        let asked = std::cell::RefCell::new(Vec::new());
        let taken = take_image(
            |timeout| {
                let mut asked = asked.borrow_mut();
                let answer = answers[asked.len().min(answers.len() - 1)];
                asked.push(timeout);
                answer
            },
            || asked.borrow().len() as u64 * call_ns,
        );
        (taken, asked.into_inner())
    }

    /// Where the presentation engine has no image free, the take waits for one in the window
    /// system, a call at a time, and the draw goes on with the first one given back
    #[test]
    fn a_take_waits_in_the_window_system_for_the_next_image() {
        let none = Err(vk::Result::NOT_READY);
        let late = Err(vk::Result::TIMEOUT);
        let (taken, asked) = takes(&[none, late, late, Ok((2, false))]);
        assert_eq!(taken, Ok((2, false)));
        assert_eq!(asked, [TAKE_NS; 4]);
    }

    /// Where no image comes within the wait, the draw is tried again: the take stops after
    /// `WAIT_NS` of calls, answering the last call's timeout
    #[test]
    fn a_take_with_no_image_in_the_wait_gives_a_retry() {
        for none in [vk::Result::TIMEOUT, vk::Result::NOT_READY] {
            let (taken, asked) = takes(&[Err(none)]);
            assert_eq!(taken, Err(none));
            assert_eq!(asked.len() as u64, WAIT_NS.div_ceil(TAKE_NS));
            assert!(asked.iter().all(|&timeout| timeout == TAKE_NS));
        }
    }

    /// A window system that returns from each call later than asked, as NVIDIA's does
    /// ([`TAKE_NS`]), is asked until `WAIT_NS` of the clock has passed, not `WAIT_NS / TAKE_NS`
    /// times
    #[test]
    fn a_take_ends_once_its_wait_has_passed_on_the_clock() {
        let (taken, asked) = takes_of(&[Err(vk::Result::TIMEOUT)], 4 * TAKE_NS);
        assert_eq!(taken, Err(vk::Result::TIMEOUT));
        assert_eq!(asked.len() as u64, WAIT_NS.div_ceil(4 * TAKE_NS));
        let (taken, asked) = takes_of(&[Err(vk::Result::TIMEOUT), Ok((1, false))], 4 * TAKE_NS);
        assert_eq!((taken, asked.len()), (Ok((1, false)), 2));
    }

    /// An image, a suboptimal one included, and every other answer end the take at once
    #[test]
    fn a_take_answers_anything_but_a_wait_at_once() {
        for answer in [
            Ok((1, false)),
            Ok((0, true)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR),
            Err(vk::Result::ERROR_DEVICE_LOST),
        ] {
            let (taken, asked) = takes(&[answer]);
            assert_eq!(taken, answer);
            assert_eq!(asked, [TAKE_NS]);
        }
    }

    /// A window system as a remake meets it ([`Remaking`]), where nothing tells that a retired
    /// swapchain's presents are complete (a device without present fences, the window drawing
    /// nothing new): its swapchains are numbers, and its create keeps Vulkan's rule that a window
    /// with a swapchain, a retired one included, refuses a create with no `oldSwapchain`
    #[derive(Default)]
    struct Window {
        /// Whether the surface has pixels for now.
        pixels: bool,
        /// Whether the create refuses (as for want of memory).
        refusing: bool,
        /// The swapchains the window has: made and not destroyed.
        live: Vec<u32>,
        /// The current swapchain.
        current: Option<u32>,
        /// Whether it is stale.
        stale: bool,
        /// The retired swapchains.
        retired: Vec<u32>,
        /// Each create's `oldSwapchain`.
        creates: Vec<Option<u32>>,
        /// The last swapchain made.
        made: u32,
    }

    impl Remaking for Window {
        type Chain = u32;

        fn take_current(&mut self) -> Option<u32> {
            self.current.take()
        }

        fn has_retired(&self) -> bool {
            !self.retired.is_empty()
        }

        fn end_retired(&mut self) {
            for chain in std::mem::take(&mut self.retired) {
                self.live.retain(|&c| c != chain);
            }
        }

        fn make(&mut self, old: Option<&u32>) -> Result<u32, Unmade> {
            if !self.pixels {
                return Err(Unmade::Unasked(Refused("no pixels for now".to_owned())));
            }
            self.creates.push(old.copied());
            let refusal = if old.is_none() && !self.live.is_empty() {
                Some(vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR)
            } else {
                self.refusing
                    .then_some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
            };
            if let Some(result) = refusal {
                return Err(Unmade::Asked(refused("vkCreateSwapchainKHR", result)));
            }
            self.made += 1;
            self.live.push(self.made);
            Ok(self.made)
        }

        fn retire(&mut self, chain: u32) {
            self.retired.push(chain);
        }

        fn set_current(&mut self, chain: Option<u32>, stale: bool) {
            self.current = chain;
            self.stale = stale;
        }
    }

    /// A window whose first swapchain is made, its surface with pixels.
    fn window() -> Window {
        let mut w = Window {
            pixels: true,
            ..Window::default()
        };
        assert_eq!(remake(&mut w), Ok(()));
        assert_eq!((w.current, &w.creates[..]), (Some(1), &[None][..]));
        w
    }

    /// A make refused before the create (a surface with no pixels for now) keeps the old
    /// swapchain current and stale, so the next make hands it over as `oldSwapchain`: retired
    /// there, it stayed the window's, and every later create, with none to hand over, was refused
    /// (`VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`), the window keeping its last frame for good
    #[test]
    fn a_make_refused_before_the_create_keeps_the_old_swapchain_to_hand_over() {
        let mut w = window();
        w.pixels = false;
        for _ in 0..3 {
            assert!(remake(&mut w).is_err());
            assert_eq!((w.current, w.stale), (Some(1), true));
            assert!(w.retired.is_empty());
        }
        assert_eq!(w.creates, [None], "no create was asked");
        w.pixels = true;
        assert_eq!(remake(&mut w), Ok(()));
        assert_eq!(w.creates, [None, Some(1)]);
        assert_eq!(
            (w.current, w.stale, &w.retired[..]),
            (Some(2), false, &[1][..])
        );
    }

    /// A refused create retires the old swapchain all the same, and a retired swapchain is the
    /// window's until it is destroyed: the next make, with none to hand over, destroys the
    /// retired ones first, so its create is not refused for them
    #[test]
    fn a_make_after_a_refused_create_ends_the_retired_swapchains_first() {
        let mut w = window();
        w.refusing = true;
        let out_of_memory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        assert_eq!(
            remake(&mut w),
            Err(refused("vkCreateSwapchainKHR", out_of_memory))
        );
        assert_eq!(
            (w.current, &w.retired[..], &w.live[..]),
            (None, &[1][..], &[1][..])
        );
        w.refusing = false;
        assert_eq!(remake(&mut w), Ok(()));
        assert_eq!(w.creates, [None, Some(1), None]);
        assert_eq!((w.current, &w.live[..]), (Some(2), &[2][..]));
        assert!(w.retired.is_empty());
    }
}
