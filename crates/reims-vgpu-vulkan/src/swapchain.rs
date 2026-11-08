//! A display window's swapchain on the runtime's own device: what the window's render thread
//! presents through, on either OS. No copy: each of the swapchain's images is a runtime texture
//! (`new_texture_with_vk_image`), which the display's present pass renders into in a command
//! buffer of the frame queue, and the image is presented from the swapchain's own presenting
//! queue. The surface is the window's: on Linux SDL's (Wayland's or X11's), on macOS one over
//! the window's `CAMetalLayer` (`VK_EXT_metal_surface`), whose drawables MoltenVK's images are.
//!
//! **The presenting queue.** Each swapchain presents on a queue made for presenting
//! (`Device::new_command_queue_for_presenting`), whose `VkQueue` the runtime reserves apart from
//! every queue of `newCommandQueue`'s where the device has one to spare (NVIDIA's graphics family
//! has 16): a present that blocks under its lock (FIFO on a window the compositor does not show,
//! the idle wait below) then holds none of the guest's submissions, as Metal's present holds no
//! queue of the guest's. A device with none to spare (RADV's and lavapipe's one graphics queue)
//! gives it a `VkQueue` its other queues share, and the runtime announces it once
//! (`shared-present-queue`, a performance kind, relayed).
//!
//! **Making it.** Over a `VkSurfaceKHR` of the runtime's instance, which the swapchain owns from
//! then on and destroys after its last image. The presenting queue's family must present to the
//! surface, which must offer `B8G8R8A8_UNORM` in `SRGB_NONLINEAR`: the format the display draws
//! into, which the compositor reads as sRGB. Otherwise there is no swapchain, and the window
//! stays empty; the reason is logged.
//!
//! **Present mode.** Never one that tears, and never one that makes the guest wait: MAILBOX
//! where the surface offers it, with at least four images, so that taking an image rarely waits
//! for the compositor to give one back; else FIFO, with at least three (MoltenVK offers FIFO
//! alone beside IMMEDIATE, and makes the layer's drawables as many as the images). Either way a
//! frame faster than the screen replaces or queues behind the one before: the window shows up to
//! the screen's rate. FIFO_LATEST_READY (`VK_EXT_present_mode_fifo_latest_ready`) is not asked
//! for: it would serve only a surface without MAILBOX, and every surface measured offers MAILBOX
//! (an RTX 5080 Laptop GPU's NVIDIA driver under KWin), while its feature, like any the runtime
//! enables for a listed extension, would be one more a device could be refused for.
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
//! tried again. MoltenVK's take answers an image at once whatever the time it is given, and its
//! fence does the waiting. The frame's command buffer waits on no semaphore of the compositor's, so
//! a frame the display counts as in flight completes when the GPU does, never when the compositor
//! gives an image back: a window the compositor stops showing (a closed lid, a minimized window)
//! slows its own render thread, never the guest's presents. A fence wait that runs out gives
//! [`Acquired::Retry`]: the image stays taken, and the next draw waits for it again. Where the OS
//! has an image's drawable to fetch (MoltenVK's `CAMetalDrawable`, [`crate::os::Drawables`]), the
//! render thread fetches it once the image is taken, so any wait for one is its own, never the
//! frame's submission's; where none comes, the draw is tried again likewise.
//!
//! **Each frame.** The command buffer signals the image's own semaphore after the present pass
//! ([`Swapchain::encode_ready`]); once the runtime has scheduled it, the image is presented on
//! the presenting queue's `VkQueue`, waiting on that semaphore ([`Swapchain::present`]). An image
//! taken for a
//! frame the display then drew nothing into is kept for the next draw. An image's semaphore is
//! signalled again only by a frame drawn into it after it was taken again, its fence waited for:
//! by then its last present has waited on the semaphore (below).
//!
//! **Size.** The swapchain is made at the size the window asks for, in pixels: Wayland gives a
//! surface no size of its own and reports no out-of-date swapchain when the window's pixels
//! change, so a draw at another size makes a new one. Where the surface has a size of its own
//! (X11), the swapchain takes that one, and the window's size otherwise within the surface's
//! limits ([`extent_for`]): during a resize the window's size can be a step behind the
//! surface's. One the compositor reports out of date or suboptimal is made again at the next
//! draw. Making a new one waits for the old one's frames to complete, then passes the old one as
//! `oldSwapchain` and retires it (below). A swapchain the surface refuses for now (it has no
//! pixels, as a minimized window's may) is tried again at the next draw; only a lost surface
//! ends the swapchain's draws ([`Acquired::Failed`]).
//!
//! **Retiring one, with no wait on the queue.** A retired swapchain, its images' textures and
//! their semaphores are destroyed once nothing uses them: its frames' command buffers have
//! completed (waited for as it retires), and so has its last present, its wait on a semaphore
//! included. Where the device has present fences (`swapchainMaintenance1`,
//! [`crate::os::present_fences`]), each present signals a fence of its own once the presentation
//! engine is done with its semaphore and image, which tells it exactly. Two rules of Vulkan's
//! tell it too, with no fence ([`Presents`]): an image taken again, its fence waited for, has its
//! previous present complete, the wait on its semaphore included (the Vulkan Guide's "Swapchain
//! Semaphore Reuse"); and the presents made on one queue are performed in the order they were
//! made (`vkQueuePresentKHR`), whatever the swapchain, and a present the compositor rejects as
//! out of date or lost still waits on its semaphores. So once a present made after a swapchain's
//! last is known complete, so is that one. Until then the retired swapchain is kept, up to
//! [`RETIRED`] of them. Past that (a resize dragged out over many frames retires one a frame,
//! faster than any present is known complete), and at the swapchain's end, where a present is not
//! known complete, the swapchain waits for the presents' fences, up to [`END_WAIT_NS`]; where it
//! has none, or they do not signal in that time, it waits for the presenting queue to idle, the
//! only other way unextended Vulkan has, and one its specification does not cover either. That
//! queue holds the window's presents alone where the device reserved it (above).
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
//! presenting queue's own lock, which the runtime's submissions take without it.
//!
//! **The commands.** `VK_KHR_swapchain`'s, and the OS's drawable fetch, are looked up as each
//! swapchain is made ([`SwapchainFns`]), through the runtime's look-up
//! (`rt::Device::device_proc_addr`), which first puts back the device's first word, the loader's
//! dispatch table, that MoltenVK writes over whenever it names the device in a debug message.
//! The commands themselves are the driver's, which read no word, but for `vkCreateSwapchainKHR`,
//! the loader's terminator, which finds the device by that word: it is called with the word put
//! back too (`rt::Device::with_loader_word`).
//!
//! **What the runtime announces.** Each batch that uses an image moves it from
//! `PRESENT_SRC_KHR`, its layout between batches, into `GENERAL` and back: two whole-image
//! barriers per frame, which the runtime announces once as a degradation of its own.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

use ash::{khr, vk};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::os::Clock as _;
use reims_vgpu_core::sync::{Mutex, MutexGuard};

use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::device::VulkanDevice;
use crate::gpu::metal_space::Hold;
use crate::gpu::resource::VulkanTexture;

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
const END_WAIT_NS: u64 = 1_000_000_000;

/// How many retired swapchains are kept for their last presents to be known complete before the
/// queue is waited idle (the module doc): each keeps its images until it is destroyed.
const RETIRED: usize = 4;

/// The images a swapchain asks for at least, by present mode.
const MAILBOX_IMAGES: u32 = 4;
/// With FIFO.
const FIFO_IMAGES: u32 = 3;

/// The format the display draws into, and its colour space.
const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
/// The colour space the compositor reads it in.
const COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;
/// The usages a swapchain asks its images for, of those the surface supports.
const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw(),
);

/// The lock every call into the window system runs under (the module doc).
static WSI: Mutex<()> = Mutex::new(());

/// `VK_KHR_swapchain`'s commands on a device, looked up as a swapchain is made, through the
/// runtime's guarded look-up (`rt::Device::device_proc_addr`, the module doc).
struct SwapchainFns {
    /// The device, whose loader word `vkCreateSwapchainKHR` reads.
    device: rt::Device,
    /// Its handle.
    handle: vk::Device,
    /// The commands.
    fp: khr::swapchain::DeviceFn,
}

impl SwapchainFns {
    /// The commands on `device`: `None` where it has none, its options asking for no
    /// `VK_KHR_swapchain` (`crate::os::add_window_extensions`).
    fn load(device: &rt::Device) -> Option<SwapchainFns> {
        device.device_proc_addr(c"vkCreateSwapchainKHR")?;
        let fp = khr::swapchain::DeviceFn::load(|name| {
            device
                .device_proc_addr(name)
                .map_or(std::ptr::null(), |f| f as *const c_void)
        });
        // SAFETY: the handle is the device's, read while it lives; the swapchain that calls the
        // commands holds the device.
        let handle = unsafe { device.raw() }.device.handle();
        Some(SwapchainFns {
            device: device.clone(),
            handle,
            fp,
        })
    }

    /// `vkCreateSwapchainKHR`: the loader's terminator, which finds the device by its first word,
    /// so the call is made with the runtime's word put back (`rt::Device::with_loader_word`).
    ///
    /// # Safety
    ///
    /// `info` is valid, over a live surface of the device's instance.
    unsafe fn create_swapchain(
        &self,
        info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<vk::SwapchainKHR, vk::Result> {
        let mut swapchain = vk::SwapchainKHR::null();
        let made = self.device.with_loader_word(|| {
            // SAFETY: the caller's; the device is live while `self` is.
            unsafe {
                (self.fp.create_swapchain_khr)(self.handle, info, std::ptr::null(), &mut swapchain)
            }
        });
        made.result_with_success(swapchain)
    }

    /// `vkDestroySwapchainKHR`.
    ///
    /// # Safety
    ///
    /// `swapchain` is the device's, and nothing uses it any more.
    unsafe fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
        // SAFETY: the caller's.
        unsafe { (self.fp.destroy_swapchain_khr)(self.handle, swapchain, std::ptr::null()) };
    }

    /// `vkGetSwapchainImagesKHR`: every image of `swapchain`.
    ///
    /// # Safety
    ///
    /// `swapchain` is a live swapchain of the device.
    unsafe fn get_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        let get = self.fp.get_swapchain_images_khr;
        loop {
            let mut count = 0_u32;
            // SAFETY: the caller's; a count query writes no image.
            unsafe { get(self.handle, swapchain, &mut count, std::ptr::null_mut()) }.result()?;
            let mut images = vec![vk::Image::null(); usize::try_from(count).unwrap_or(0)];
            // SAFETY: as above; `images` holds `count` images.
            let read = unsafe { get(self.handle, swapchain, &mut count, images.as_mut_ptr()) };
            if read != vk::Result::INCOMPLETE {
                read.result()?;
                images.truncate(usize::try_from(count).unwrap_or(0));
                return Ok(images);
            }
        }
    }

    /// `vkAcquireNextImageKHR`: the image's index, and whether the swapchain is suboptimal.
    ///
    /// # Safety
    ///
    /// As `vkAcquireNextImageKHR`'s valid usage asks, of a live swapchain of the device.
    unsafe fn acquire_next_image(
        &self,
        swapchain: vk::SwapchainKHR,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(u32, bool), vk::Result> {
        let mut index = 0_u32;
        // SAFETY: the caller's.
        let taken = unsafe {
            (self.fp.acquire_next_image_khr)(
                self.handle,
                swapchain,
                timeout,
                semaphore,
                fence,
                &mut index,
            )
        };
        match taken {
            vk::Result::SUCCESS => Ok((index, false)),
            vk::Result::SUBOPTIMAL_KHR => Ok((index, true)),
            other => Err(other),
        }
    }

    /// `vkQueuePresentKHR` on `queue`: whether the swapchains are suboptimal.
    ///
    /// # Safety
    ///
    /// As `vkQueuePresentKHR`'s valid usage asks, on a queue of the device.
    unsafe fn queue_present(
        &self,
        queue: vk::Queue,
        info: &vk::PresentInfoKHR<'_>,
    ) -> Result<bool, vk::Result> {
        // SAFETY: the caller's.
        match unsafe { (self.fp.queue_present_khr)(queue, info) } {
            vk::Result::SUCCESS => Ok(false),
            vk::Result::SUBOPTIMAL_KHR => Ok(true),
            other => Err(other),
        }
    }
}

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

/// One image of the swapchain.
struct Image {
    /// The image.
    raw: vk::Image,
    /// The runtime's texture over it.
    texture: VulkanTexture,
    /// The semaphore its frame's command buffer signals, and its present waits on.
    ready: vk::Semaphore,
    /// The command buffer of its last frame, until the swapchain ends.
    last: Option<VulkanCommandBuffer>,
    /// Its last present's number ([`Presents`]), 0 before its first.
    presented: u64,
}

/// A swapchain and its images.
struct Chain {
    /// The swapchain.
    raw: vk::SwapchainKHR,
    /// The size in pixels it was made for: the window's.
    size: (u64, u64),
    /// Its images' size in pixels, within the surface's limits ([`extent_for`]).
    extent: (u64, u64),
    /// Its images, by index.
    images: Vec<Image>,
}

impl Chain {
    /// The number of its last present, 0 where none was made.
    fn last_present(&self) -> u64 {
        self.images.iter().map(|i| i.presented).max().unwrap_or(0)
    }
}

/// The fences of a swapchain's presents, where the device has present fences (the module doc).
#[derive(Debug, Default)]
struct Fences {
    /// Fences unsignalled, with no present pending on them.
    free: Vec<vk::Fence>,
    /// The fences of the presents not yet seen signalled, each with its present's number, in the
    /// order the presents were made.
    pending: VecDeque<(u64, vk::Fence)>,
}

impl Fences {
    /// A fence for a present: a free one, else a new one; none where the device makes none, and
    /// the present goes without.
    fn take(&mut self, vk: &ash::Device) -> Option<vk::Fence> {
        if let Some(fence) = self.free.pop() {
            return Some(fence);
        }
        // SAFETY: a valid create info, on the runtime's live device.
        match unsafe { vk.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(fence) => Some(fence),
            Err(e) => {
                log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkCreateFence: {e}");
                None
            }
        }
    }

    /// `fence` was chained into present number `number`, which answered `result`: pending where
    /// the present's operations were enqueued (it was taken, or rejected as out of date or for a
    /// lost surface, which still waits on its semaphores), else free again, as a present that
    /// enqueues nothing leaves its fence as it was.
    fn presented(&mut self, number: u64, fence: vk::Fence, result: Result<bool, vk::Result>) {
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
    fn all(&self) -> impl Iterator<Item = vk::Fence> + '_ {
        (self.free.iter().copied()).chain(self.pending.iter().map(|&(_, fence)| fence))
    }
}

/// What a swapchain knows of its presents (the module doc): they are numbered from 1 in the order
/// they are made, all on one queue, and the newest known complete, its semaphore wait included,
/// tells that every one before it is too.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Presents {
    /// The presents made.
    made: u64,
    /// The newest present known complete; 0 for none.
    complete: u64,
}

impl Presents {
    /// A present made: its number.
    fn make(&mut self) -> u64 {
        self.made += 1;
        self.made
    }

    /// An image taken again, its fence waited for, whose last present was number `last` (0 for
    /// none): that present is complete, and every one made before it.
    fn taken_again(&mut self, last: u64) {
        self.complete = self.complete.max(last);
    }

    /// Whether present number `last`, and so every one before it, is known complete.
    fn done(&self, last: u64) -> bool {
        last <= self.complete
    }

    /// Present number `number`'s fence signalled: it is complete, and every one made before it.
    fn signalled(&mut self, number: u64) {
        self.complete = self.complete.max(number);
    }

    /// The queue was waited idle: every present made is taken as complete (the module doc).
    fn idle(&mut self) {
        self.complete = self.made;
    }
}

/// A window's swapchain (the module doc), on the render thread.
pub struct Swapchain {
    /// The presenting queue, whose `VkQueue` presents (the module doc).
    present: rt::CommandQueue,
    /// The queue's device, whose textures the images are.
    device: VulkanDevice,
    /// The runtime's logical device.
    vk: ash::Device,
    /// The physical device.
    physical: vk::PhysicalDevice,
    /// The queue families the runtime's images are shared among.
    families: Vec<u32>,
    /// `VK_KHR_surface`'s commands.
    surface_api: khr::surface::Instance,
    /// `VK_KHR_swapchain`'s commands.
    swapchain_api: SwapchainFns,
    /// The drawable fetch, where the OS and the driver have one ([`crate::os::Drawables`]).
    drawables: Option<crate::os::Drawables>,
    /// The surface, owned.
    surface: vk::SurfaceKHR,
    /// The present mode.
    mode: vk::PresentModeKHR,
    /// The fence each image is taken with.
    fence: vk::Fence,
    /// The swapchain, once made.
    chain: Option<Chain>,
    /// Swapchains retired, kept until nothing uses them (the module doc).
    retired: Vec<Chain>,
    /// The presents made, and the newest known complete.
    presents: Presents,
    /// The presents' fences, where the device has present fences.
    fences: Option<Fences>,
    /// The swapchain is out of date, or suboptimal: the next draw makes it again.
    stale: bool,
    /// An image taken whose fence has not signalled yet.
    pending: Option<u32>,
    /// An image taken and ready, kept for the next draw.
    held: Option<u32>,
    /// The surface is lost: no swapchain will be made of it.
    lost: bool,
    /// The last making of the swapchain was refused, which was logged: the next refusal is not.
    refusing: bool,
}

impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("mode", &self.mode.as_raw())
            .field("size", &self.chain.as_ref().map(|c| c.size))
            .field("stale", &self.stale)
            .finish_non_exhaustive()
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

/// Has `report` called with `command_buffer`'s GPU time once it completes, on the runtime's
/// completion thread: its GPU end less its GPU start, as the runtime answers them (on a device
/// that times its command buffers from the host, from the end of its submission to the host's
/// sight of its completion). Nothing is reported where the runtime gives no time, nor where it
/// refuses the handler (a command buffer already committed): false then.
pub fn report_gpu_time(
    command_buffer: &VulkanCommandBuffer,
    report: impl FnOnce(Duration) + Send + 'static,
) -> bool {
    command_buffer
        .raw
        .add_completed_handler(Box::new(move |cb: &rt::CommandBuffer| {
            if let (Ok(start), Ok(end)) = (cb.gpu_start_time(), cb.gpu_end_time())
                && let Ok(time) = Duration::try_from_secs_f64(end - start)
            {
                report(time);
            }
        }))
        .is_ok()
}

/// A refusal naming `what` failed with `result`.
fn refused(what: &str, result: vk::Result) -> Refused {
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
fn take_image(
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
fn present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// The size in pixels of the images of a swapchain for a window of `size` pixels on a surface
/// of `caps`: the surface's own size where it has one, else `size` within the surface's limits.
/// `None` where that has no pixel (a minimized window's surface may allow none).
fn extent_for(size: (u64, u64), caps: &vk::SurfaceCapabilitiesKHR) -> Option<vk::Extent2D> {
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

impl Swapchain {
    /// A swapchain over `surface`, presented on a presenting queue of `queue`'s device, made at
    /// the first draw. Refused, with the surface destroyed, where the device has no swapchain
    /// commands ([`SwapchainFns`]), the runtime makes no presenting queue (a lost device), the
    /// presenting queue's family does not present to the surface, or it offers no
    /// `B8G8R8A8_UNORM` in `SRGB_NONLINEAR` (the module doc).
    ///
    /// # Safety
    ///
    /// `surface` is a live surface of the instance of `queue`'s device, owned by no one else.
    pub unsafe fn new(
        queue: &VulkanCommandQueue,
        surface: vk::SurfaceKHR,
    ) -> Result<Swapchain, Refused> {
        let device = queue.device.clone_ref();
        // SAFETY: the handles are the runtime's, used only while `device` lives, which the
        // swapchain holds; nothing here destroys them.
        let raw = unsafe { device.raw.raw() };
        let surface_api = khr::surface::Instance::new(raw.entry, raw.instance);
        let (vk, physical) = (raw.device.clone(), raw.physical_device);
        let families = raw.queue_families.to_vec();
        let Some(swapchain_api) = SwapchainFns::load(&device.raw) else {
            let _wsi = WSI.lock();
            // SAFETY: the surface is this call's, and nothing was made of it.
            unsafe { surface_api.destroy_surface(surface, None) };
            return Err(Refused(
                "the device has no swapchain commands (no VK_KHR_swapchain)".to_owned(),
            ));
        };
        let drawables = crate::os::Drawables::load(&device.raw);
        let present = match device.raw.new_command_queue_for_presenting() {
            Ok(present) => present,
            Err(e) => {
                let _wsi = WSI.lock();
                // SAFETY: the surface is this call's, and nothing was made of it.
                unsafe { surface_api.destroy_surface(surface, None) };
                return Err(Refused(format!("no presenting queue: {e}")));
            }
        };
        let wsi = WSI.lock();
        let destroy = |reason: Refused| {
            // SAFETY: the surface is this call's, and nothing was made of it.
            unsafe { surface_api.destroy_surface(surface, None) };
            reason
        };
        // SAFETY: the closure only reads the family; it submits nothing.
        let family = unsafe { present.with_vk_queue(|_, family| family) };
        // SAFETY: `surface` and `physical` are live, of one instance.
        let presents =
            unsafe { surface_api.get_physical_device_surface_support(physical, family, surface) }
                .map_err(|e| destroy(refused("vkGetPhysicalDeviceSurfaceSupportKHR", e)))?;
        if !presents {
            return Err(destroy(Refused(format!(
                "queue family {family} does not present to the window's surface"
            ))));
        }
        // SAFETY: as above.
        let formats = unsafe { surface_api.get_physical_device_surface_formats(physical, surface) }
            .map_err(|e| destroy(refused("vkGetPhysicalDeviceSurfaceFormatsKHR", e)))?;
        if !formats
            .iter()
            .any(|f| f.format == FORMAT && f.color_space == COLOR_SPACE)
        {
            return Err(destroy(Refused(
                "the window's surface offers no B8G8R8A8_UNORM in SRGB_NONLINEAR".to_owned(),
            )));
        }
        // SAFETY: as above.
        let modes =
            unsafe { surface_api.get_physical_device_surface_present_modes(physical, surface) }
                .map_err(|e| destroy(refused("vkGetPhysicalDeviceSurfacePresentModesKHR", e)))?;
        let mode = present_mode(&modes);
        // SAFETY: a valid create info, on the runtime's live device.
        let fence = unsafe { vk.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| destroy(refused("vkCreateFence", e)))?;
        drop(wsi);
        let fences =
            crate::os::present_fences(crate::platform::device_options()).then(Fences::default);
        Ok(Swapchain {
            present,
            device,
            vk,
            physical,
            families,
            surface_api,
            swapchain_api,
            drawables,
            surface,
            mode,
            fence,
            chain: None,
            retired: Vec::new(),
            presents: Presents::default(),
            fences,
            stale: false,
            pending: None,
            held: None,
            lost: false,
            refusing: false,
        })
    }

    /// A swapchain over the surface `make` makes on the instance of `queue`'s device, given the
    /// instance's `vkGetInstanceProcAddr` (its loader's, which the runtime opened) and the
    /// instance: [`Swapchain::new`] over it, or refused where `make` makes none.
    ///
    /// # Safety
    ///
    /// A surface `make` returns is one it made on the instance it was given, owned by no one
    /// else.
    pub unsafe fn over(
        queue: &VulkanCommandQueue,
        make: impl FnOnce(vk::PFN_vkGetInstanceProcAddr, vk::Instance) -> Option<vk::SurfaceKHR>,
    ) -> Result<Swapchain, Refused> {
        // SAFETY: the instance's handle and the loader's entry, read while the queue's device
        // lives.
        let raw = unsafe { queue.device.raw.raw() };
        let (get_instance_proc_addr, instance) = (
            raw.entry.static_fn().get_instance_proc_addr,
            raw.instance.handle(),
        );
        let surface = {
            let _wsi = WSI.lock();
            make(get_instance_proc_addr, instance)
        }
        .ok_or_else(|| Refused("no surface".to_owned()))?;
        // SAFETY: the caller's.
        unsafe { Swapchain::new(queue, surface) }
    }

    /// Whether the swapchain presents with MAILBOX (else FIFO).
    pub fn is_mailbox(&self) -> bool {
        self.mode == vk::PresentModeKHR::MAILBOX
    }

    /// Whether each present signals a fence of its own (the module doc).
    pub fn has_present_fences(&self) -> bool {
        self.fences.is_some()
    }

    /// The size in pixels of the images, once the swapchain is made: the window's, or the
    /// surface's own ([`extent_for`]).
    pub fn extent(&self) -> Option<(u64, u64)> {
        self.chain.as_ref().map(|c| c.extent)
    }

    /// The next image to draw a frame of `size` pixels into: the one kept from the last draw,
    /// else the next the compositor gives, waited for up to [`WAIT_NS`] ([`take_image`]), its
    /// fence waited for up to [`WAIT_NS`], else a retry (the module doc). Makes the swapchain
    /// first where there is none, where it is stale, or where its size is another.
    pub fn acquire(&mut self, size: (u64, u64)) -> Acquired {
        if size.0 == 0 || size.1 == 0 {
            return Acquired::Retry;
        }
        let current = self.chain.as_ref().map(|c| c.size);
        if self.stale || current != Some(size) {
            match self.remake(size) {
                Ok(()) => self.refusing = false,
                Err(e) if self.lost => {
                    log::warn!(
                        target: "reims_vgpu_vulkan::swapchain",
                        "no swapchain ({e}): the surface is lost"
                    );
                    return Acquired::Failed;
                }
                Err(e) => {
                    if !self.refusing {
                        log::warn!(
                            target: "reims_vgpu_vulkan::swapchain",
                            "no swapchain yet ({e}): tried again at the next draw"
                        );
                    }
                    self.refusing = true;
                    return Acquired::Retry;
                }
            }
        }
        if let Some(index) = self.held {
            return self.fetched(index);
        }
        if let Some(index) = self.pending {
            return self.wait_for(index);
        }
        let Some(chain) = &self.chain else {
            return Acquired::Failed;
        };
        let clock = crate::os::OsClock::new();
        let began = clock.now_mono();
        let elapsed = || {
            let waited = clock.now_mono().saturating_duration_since(began).as_nanos();
            u64::try_from(waited).unwrap_or(u64::MAX)
        };
        let taken = take_image(
            |timeout| {
                let wsi = WSI.lock();
                // SAFETY: the swapchain and fence are live, and the fence unsignalled with nothing
                // pending on it: a take that answers no image leaves it so.
                let taken = unsafe {
                    self.swapchain_api.acquire_next_image(
                        chain.raw,
                        timeout,
                        vk::Semaphore::null(),
                        self.fence,
                    )
                };
                // Handed to a window waiting for it, if any, before this one takes it again.
                MutexGuard::unlock_fair(wsi);
                taken
            },
            elapsed,
        );
        match taken {
            Ok((index, suboptimal)) => {
                self.stale |= suboptimal;
                self.wait_for(index)
            }
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Acquired::Retry,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stale = true;
                Acquired::Retry
            }
            Err(e) => {
                log::warn!(
                    target: "reims_vgpu_vulkan::swapchain",
                    "vkAcquireNextImageKHR: {e}: the window stops presenting"
                );
                Acquired::Failed
            }
        }
    }

    /// Waits up to [`WAIT_NS`] for image `index`'s fence: the image, or a retry with it pending.
    fn wait_for(&mut self, index: u32) -> Acquired {
        // SAFETY: the fence is live, and image `index`'s acquire signals it.
        match unsafe { self.vk.wait_for_fences(&[self.fence], true, WAIT_NS) } {
            Ok(()) => {
                // SAFETY: the fence has signalled, and nothing else waits on it.
                if let Err(e) = unsafe { self.vk.reset_fences(&[self.fence]) } {
                    log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkResetFences: {e}");
                    return Acquired::Failed;
                }
                self.pending = None;
                self.held = Some(index);
                self.taken_again(index);
                self.collect();
                self.fetched(index)
            }
            Err(vk::Result::TIMEOUT) => {
                self.pending = Some(index);
                Acquired::Retry
            }
            Err(e) => {
                log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkWaitForFences: {e}");
                Acquired::Failed
            }
        }
    }

    /// Image `index`, taken and held, once its drawable is fetched where the OS has one to fetch
    /// ([`crate::os::Drawables`], the module doc); a retry, the image still held, where none came
    /// in time.
    fn fetched(&self, index: u32) -> Acquired {
        let image = self.chain.as_ref().and_then(|c| {
            usize::try_from(index)
                .ok()
                .and_then(|i| c.images.get(i))
                .map(|i| i.raw)
        });
        let Some((drawables, image)) = self.drawables.as_ref().zip(image) else {
            return Acquired::Image(Frame { index });
        };
        // SAFETY: the image is the swapchain's, taken, its fence waited for; only this thread
        // uses it until its frame is committed.
        if unsafe { drawables.fetch(image) } {
            Acquired::Image(Frame { index })
        } else {
            Acquired::Retry
        }
    }

    /// The texture over `frame`'s image, which the display's present pass draws into.
    pub fn texture(&self, frame: Frame) -> Option<&VulkanTexture> {
        let chain = self.chain.as_ref()?;
        chain
            .images
            .get(usize::try_from(frame.index).ok()?)
            .map(|i| &i.texture)
    }

    /// Encodes in `command_buffer` the signal of `frame`'s semaphore, after what it encoded:
    /// false where the runtime refuses it.
    pub fn encode_ready(&self, frame: Frame, command_buffer: &VulkanCommandBuffer) -> bool {
        let Some(image) = self.image(frame) else {
            return false;
        };
        // SAFETY: the semaphore is a binary semaphore of the command buffer's device, which
        // lives until the swapchain's end, after every frame's completion; its present consumes
        // the signal.
        unsafe { command_buffer.raw.encode_signal_vk_semaphore(image.ready) }.is_ok()
    }

    /// Presents `frame`, whose command buffer `command_buffer` has been committed with its
    /// [`Swapchain::encode_ready`]: once the runtime has scheduled it, on the presenting queue's
    /// `VkQueue`, with a fence of its own where the device has present fences. A swapchain the
    /// compositor reports out of date or suboptimal is made again at the next draw. Waits for
    /// the scheduling alone.
    pub fn present(&mut self, frame: Frame, command_buffer: VulkanCommandBuffer) {
        self.held = None;
        // The image's frame is waited for before its swapchain goes, presented or not.
        if let Some(image) = self.chain.as_mut().and_then(|c| {
            usize::try_from(frame.index)
                .ok()
                .and_then(|i| c.images.get_mut(i))
        }) {
            image.last = Some(command_buffer.clone_ref());
        }
        if let Err(e) = command_buffer.raw.wait_until_scheduled() {
            // The image stays taken, its semaphore perhaps never signalled: the swapchain is
            // made again, which gives every image back.
            log::warn!(
                target: "reims_vgpu_vulkan::swapchain",
                "a window frame was not scheduled ({e:?}): the swapchain is made again"
            );
            self.stale = true;
            return;
        }
        let Some(chain) = self.chain.as_mut() else {
            return;
        };
        let Some(image) = usize::try_from(frame.index)
            .ok()
            .and_then(|i| chain.images.get_mut(i))
        else {
            return;
        };
        let waits = [image.ready];
        let chains = [chain.raw];
        let indices = [frame.index];
        let fence = self.fences.as_mut().and_then(|f| f.take(&self.vk));
        let fences = [fence.unwrap_or_default()];
        let mut fenced = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
        let mut info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&chains)
            .image_indices(&indices);
        if fence.is_some() {
            info = info.push_next(&mut fenced);
        }
        let api = &self.swapchain_api;
        let wsi = WSI.lock();
        // SAFETY: the present is the only submission inside, on the presenting queue's own
        // `VkQueue`; the image was taken from this swapchain and its semaphore's signal is
        // submitted. A fence is chained only where the device enabled `swapchainMaintenance1`
        // ([`crate::os::present_fences`]), and is unsignalled with nothing pending on it.
        let presented = unsafe {
            self.present
                .with_vk_queue(|q, _| api.queue_present(q, &info))
        };
        drop(wsi);
        image.last = Some(command_buffer);
        // Numbered whatever the compositor answers: a present it rejects still waits on its
        // semaphore (the module doc).
        image.presented = self.presents.make();
        if let (Some(fences), Some(fence)) = (self.fences.as_mut(), fence) {
            fences.presented(image.presented, fence, presented);
        }
        match presented {
            Ok(suboptimal) => self.stale |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.stale = true,
            Err(e) => {
                log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkQueuePresentKHR: {e}");
                self.stale = true;
            }
        }
    }

    /// `frame`'s image.
    fn image(&self, frame: Frame) -> Option<&Image> {
        let chain = self.chain.as_ref()?;
        chain.images.get(usize::try_from(frame.index).ok()?)
    }

    /// Makes the swapchain again at `size` pixels, after the old one's frames (the module doc).
    fn remake(&mut self, size: (u64, u64)) -> Result<(), Refused> {
        let old = self.retire();
        let made = self.make(
            size,
            old.as_ref().map_or(vk::SwapchainKHR::null(), |c| c.raw),
        );
        // Retired whether or not the new one was made.
        self.retired.extend(old);
        self.collect();
        self.chain = Some(made?);
        self.stale = false;
        Ok(())
    }

    /// Takes the swapchain out once its frames have completed, waiting on nothing else; its
    /// images' textures and semaphores stay until [`Swapchain::destroy`].
    fn retire(&mut self) -> Option<Chain> {
        let mut chain = self.chain.take()?;
        for image in &mut chain.images {
            if let Some(last) = image.last.take() {
                // A frame that failed has completed too; what failed it is the display's.
                let _ = last.raw.wait_until_completed();
            }
        }
        if let Some(index) = self.settle_fence()
            && let Some(image) = usize::try_from(index)
                .ok()
                .and_then(|i| chain.images.get(i))
        {
            self.presents.taken_again(image.presented);
        }
        self.held = None;
        Some(chain)
    }

    /// Image `index` of the swapchain was taken again, its fence waited for: its last present is
    /// complete ([`Presents::taken_again`]).
    fn taken_again(&mut self, index: u32) {
        let last = self.chain.as_ref().and_then(|c| {
            usize::try_from(index)
                .ok()
                .and_then(|i| c.images.get(i))
                .map(|i| i.presented)
        });
        if let Some(last) = last {
            self.presents.taken_again(last);
        }
    }

    /// Destroys the retired swapchains whose last present is known complete; past [`RETIRED`]
    /// kept, waits first for the oldest one's presents' fences, or where they do not tell it, for
    /// the presenting queue to idle (the module doc).
    fn collect(&mut self) {
        self.poll_fences();
        if self.retired.len() > RETIRED {
            let oldest = self.retired.first().map_or(0, Chain::last_present);
            if !self.wait_fences(oldest) {
                self.wait_idle("more retired swapchains than are kept");
            }
        }
        let (done, kept): (Vec<Chain>, Vec<Chain>) = std::mem::take(&mut self.retired)
            .into_iter()
            .partition(|c| self.presents.done(c.last_present()));
        self.retired = kept;
        if !done.is_empty() {
            log::trace!(
                target: "reims_vgpu_vulkan::swapchain",
                "{} retired swapchain(s) destroyed, {} kept: presents complete up to {} of {}",
                done.len(),
                self.retired.len(),
                self.presents.complete,
                self.presents.made
            );
        }
        for chain in done {
            self.destroy(chain);
        }
    }

    /// Waits for the presenting queue's `VkQueue` to idle (where the device shares it, other
    /// queues' work on it included), after which every present made is taken as complete (the
    /// module doc); `why` is logged.
    fn wait_idle(&mut self, why: &str) {
        log::debug!(
            target: "reims_vgpu_vulkan::swapchain",
            "waiting for the presenting queue to idle: {why}"
        );
        // SAFETY: the wait submits nothing; the queue is the swapchain's presenting queue.
        let idle = unsafe {
            self.present
                .with_vk_queue(|q, _| self.vk.queue_wait_idle(q))
        };
        if let Err(e) = idle {
            log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkQueueWaitIdle: {e}");
        }
        self.presents.idle();
        // Every enqueued present's fence has signalled.
        self.poll_fences();
    }

    /// Takes as complete the presents whose fences have signalled, the oldest first, as far as
    /// the first that has not ([`Presents::signalled`]); each such fence is reset and free again.
    fn poll_fences(&mut self) {
        let Some(fences) = self.fences.as_mut() else {
            return;
        };
        while let Some(&(number, fence)) = fences.pending.front() {
            // SAFETY: the fence is live.
            match unsafe { self.vk.get_fence_status(fence) } {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    log::warn!(target: "reims_vgpu_vulkan::swapchain", "vkGetFenceStatus: {e}");
                    break;
                }
            }
            fences.pending.pop_front();
            // SAFETY: the fence has signalled, and nothing waits on it.
            match unsafe { self.vk.reset_fences(&[fence]) } {
                Ok(()) => fences.free.push(fence),
                // SAFETY: as above.
                Err(_) => unsafe { self.vk.destroy_fence(fence, None) },
            }
            self.presents.signalled(number);
        }
    }

    /// Waits up to [`END_WAIT_NS`] for the fences of the presents up to number `last`, then
    /// takes those that signalled as complete ([`Swapchain::poll_fences`]): whether present
    /// `last`, and so every one before it, is known complete then. False where the device has no
    /// present fences and it is not known.
    fn wait_fences(&mut self, last: u64) -> bool {
        let Some(fences) = &self.fences else {
            return self.presents.done(last);
        };
        let waited: Vec<vk::Fence> = (fences.pending.iter())
            .filter(|&&(number, _)| number <= last)
            .map(|&(_, fence)| fence)
            .collect();
        if !waited.is_empty() {
            // SAFETY: the fences are live, each chained into a present that was enqueued.
            let _ = unsafe { self.vk.wait_for_fences(&waited, true, END_WAIT_NS) };
        }
        self.poll_fences();
        self.presents.done(last)
    }

    /// Waits for an image still pending on the fence, up to [`END_WAIT_NS`], so the fence is
    /// unsignalled with nothing pending on it; where the wait runs out, it is left behind and
    /// another made. The image's index where its fence signalled: it was taken.
    fn settle_fence(&mut self) -> Option<u32> {
        let index = self.pending.take()?;
        // SAFETY: the fence is live.
        let waited = unsafe { self.vk.wait_for_fences(&[self.fence], true, END_WAIT_NS) };
        // SAFETY: the fence has signalled, or is left behind unused.
        let settled = waited.is_ok() && unsafe { self.vk.reset_fences(&[self.fence]) }.is_ok();
        if !settled {
            // SAFETY: a valid create info, on the runtime's live device.
            if let Ok(fence) =
                unsafe { self.vk.create_fence(&vk::FenceCreateInfo::default(), None) }
            {
                self.fence = fence;
            }
        }
        waited.is_ok().then_some(index)
    }

    /// Destroys `chain`: its textures first, then its semaphores and the swapchain.
    fn destroy(&self, chain: Chain) {
        let _wsi = WSI.lock();
        let Chain { raw, images, .. } = chain;
        let semaphores: Vec<vk::Semaphore> = images.iter().map(|i| i.ready).collect();
        drop(images);
        // SAFETY: every frame of the swapchain has completed ([`Swapchain::retire`]), and so has
        // its last present, its semaphore's wait included ([`Swapchain::collect`]): nothing uses
        // its semaphores or images any more.
        unsafe {
            for semaphore in semaphores {
                self.vk.destroy_semaphore(semaphore, None);
            }
            self.swapchain_api.destroy_swapchain(raw);
        }
    }

    /// A swapchain for a window of `size` pixels, passing `old` as its `oldSwapchain`, its images
    /// wrapped as runtime textures (the module doc). A refusal for a lost surface marks it lost.
    fn make(&mut self, size: (u64, u64), old: vk::SwapchainKHR) -> Result<Chain, Refused> {
        let _wsi = WSI.lock();
        // SAFETY: the surface and physical device are live, of one instance.
        let caps = unsafe {
            self.surface_api
                .get_physical_device_surface_capabilities(self.physical, self.surface)
        }
        .map_err(|e| self.refused_by("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", e))?;
        let extent = extent_for(size, &caps)
            .ok_or_else(|| Refused("the window's surface has no pixels for now".to_owned()))?;
        let wanted = if self.is_mailbox() {
            MAILBOX_IMAGES
        } else {
            FIFO_IMAGES
        };
        let mut count = caps.min_image_count.saturating_add(1).max(wanted);
        if caps.max_image_count != 0 {
            count = count.min(caps.max_image_count);
        }
        let usage = caps.supported_usage_flags & USAGE;
        let alpha = if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };
        let sharing = sharing_among(&self.families);
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(FORMAT)
            .image_color_space(COLOR_SPACE)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(sharing)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(self.mode)
            .clipped(true)
            .old_swapchain(old);
        if sharing == vk::SharingMode::CONCURRENT {
            info = info.queue_family_indices(&self.families);
        }
        // SAFETY: a valid create info over the live surface; `old` is this swapchain's last.
        let raw = unsafe { self.swapchain_api.create_swapchain(&info) }
            .map_err(|e| self.refused_by("vkCreateSwapchainKHR", e))?;
        let images = self.wrap(raw, extent, usage, sharing).inspect_err(|_| {
            // SAFETY: nothing was presented from the new swapchain.
            unsafe { self.swapchain_api.destroy_swapchain(raw) };
        })?;
        Ok(Chain {
            raw,
            size,
            extent: (u64::from(extent.width), u64::from(extent.height)),
            images,
        })
    }

    /// `what`'s refusal with `result`, which marks the surface lost where it says so.
    fn refused_by(&mut self, what: &str, result: vk::Result) -> Refused {
        self.lost |= result == vk::Result::ERROR_SURFACE_LOST_KHR;
        refused(what, result)
    }

    /// `swapchain`'s images, each wrapped as a runtime texture with a semaphore of its own.
    fn wrap(
        &self,
        swapchain: vk::SwapchainKHR,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        sharing: vk::SharingMode,
    ) -> Result<Vec<Image>, Refused> {
        // SAFETY: `swapchain` is live.
        let raw_images = unsafe { self.swapchain_api.get_swapchain_images(swapchain) }
            .map_err(|e| refused("vkGetSwapchainImagesKHR", e))?;
        let info = image_info(extent, usage, sharing, &self.families);
        let descriptor = image_descriptor(extent);
        let layouts = rt::VkImageLayouts::new(
            metal2vulkan::vk::ImageLayout::UNDEFINED,
            metal2vulkan::vk::ImageLayout::PRESENT_SRC_KHR,
        );
        let mut images: Vec<Image> = Vec::with_capacity(raw_images.len());
        let undo = |images: Vec<Image>, e: Refused| {
            let semaphores: Vec<vk::Semaphore> = images.iter().map(|i| i.ready).collect();
            drop(images);
            for semaphore in semaphores {
                // SAFETY: no frame used the semaphore.
                unsafe { self.vk.destroy_semaphore(semaphore, None) };
            }
            e
        };
        for raw in raw_images {
            // SAFETY: the image is the swapchain's, made with `info` (the swapchain's format,
            // extent, usage and sharing, with one level, one layer and optimal tiling), bound,
            // in UNDEFINED until its first frame, and it outlives the texture, which is dropped
            // before the swapchain is destroyed ([`Swapchain::destroy`]).
            let texture = match unsafe {
                self.device
                    .raw
                    .new_texture_with_vk_image(&descriptor, raw, &info, layouts)
            } {
                Ok(texture) => texture,
                Err(e) => {
                    return Err(undo(
                        images,
                        Refused(format!("new_texture_with_vk_image: {e:?}")),
                    ));
                }
            };
            // SAFETY: a valid create info, on the runtime's live device.
            let ready = match unsafe {
                self.vk
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(ready) => ready,
                Err(e) => return Err(undo(images, refused("vkCreateSemaphore", e))),
            };
            images.push(Image {
                raw,
                texture: self
                    .device
                    .texture_of(texture, None, 0, Arc::new(Hold::NONE)),
                ready,
                last: None,
                presented: 0,
            });
        }
        Ok(images)
    }
}

/// How a swapchain's images are shared among the runtime's queue `families`: concurrently where
/// it has several.
fn sharing_among(families: &[u32]) -> vk::SharingMode {
    match families.len() {
        0 | 1 => vk::SharingMode::EXCLUSIVE,
        _ => vk::SharingMode::CONCURRENT,
    }
}

/// The create info of a swapchain's images of `extent`, `usage` and `sharing`: one level and one
/// layer of [`FORMAT`], optimal tiling, shared among `families` where `sharing` is concurrent.
fn image_info(
    extent: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    sharing: vk::SharingMode,
    families: &[u32],
) -> vk::ImageCreateInfo<'_> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(FORMAT)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(sharing)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    match sharing {
        vk::SharingMode::CONCURRENT => info.queue_family_indices(families),
        _ => info,
    }
}

/// The runtime's descriptor of a swapchain's image of `extent`: a `Private` `BGRA8Unorm` render
/// target, placed outside the program's resource IDs (`IdPlacement::Outside`, a runtime
/// extension), as the device's own textures are. It takes no texture resource ID, so the
/// `gpuResourceID` and `uniqueIdentifier` of the guest's textures are the same whether or not a
/// window is open.
fn image_descriptor(extent: vk::Extent2D) -> rt::TextureDescriptor {
    let mut descriptor = rt::TextureDescriptor::texture_2d(
        PixelFormat::BGRA8Unorm,
        u64::from(extent.width),
        u64::from(extent.height),
        false,
    );
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = rt::ResourceOptions::private();
    descriptor.placement = rt::IdPlacement::Outside;
    descriptor
}

impl Drop for Swapchain {
    /// The swapchain's end, on the render thread: its frames complete, and where a present is not
    /// known complete, the swapchain waits for its presents' fences, or for the presenting queue
    /// to idle (the module doc); then every swapchain's images' textures, its semaphores and the
    /// swapchain go, then the fences and the surface.
    fn drop(&mut self) {
        let last = self.retire();
        self.retired.extend(last);
        // An image pending with no swapchain left: its fence is settled before it goes.
        let _ = self.settle_fence();
        let newest = self.retired.iter().map(Chain::last_present).max();
        if let Some(newest) = newest
            && !self.wait_fences(newest)
        {
            self.wait_idle("the window's end");
        }
        for chain in std::mem::take(&mut self.retired) {
            self.destroy(chain);
        }
        let _wsi = WSI.lock();
        // SAFETY: every present is complete, so nothing waits on a fence any more, nor uses the
        // surface.
        unsafe {
            for fence in self.fences.iter().flat_map(Fences::all) {
                self.vk.destroy_fence(fence, None);
            }
            self.vk.destroy_fence(self.fence, None);
            self.surface_api.destroy_surface(self.surface, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A swapchain's image's texture takes no resource ID: over an image made as the swapchain's
    /// are ([`image_info`], [`image_descriptor`]), it answers 0, and a guest texture made while it
    /// lives takes the ID it takes without it
    #[test]
    fn a_swapchain_images_texture_takes_no_resource_id() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Texture as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;

        let p = crate::platform::tests::platform();
        let d = crate::gpu::device::tests::dev();
        let desc = p.texture_descriptor(&TextureDesc {
            width: 32,
            height: 32,
            ..TextureDesc::default()
        });
        let guest =
            || (d.new_texture(OnFault::Return, &desc).unwrap().unwrap()).unique_identifier();
        let without = guest();
        // SAFETY: the handles are the runtime's, used only while `d` lives.
        let raw = unsafe { d.raw.raw() };
        let extent = vk::Extent2D {
            width: 32,
            height: 32,
        };
        let families = raw.queue_families.to_vec();
        let info = image_info(extent, USAGE, sharing_among(&families), &families);
        // SAFETY: a valid create info, on the runtime's live device; the image and its memory
        // are this test's, bound once and destroyed after the texture over them is gone.
        let (image, memory) = unsafe {
            let image = raw.device.create_image(&info, None).unwrap();
            let needs = raw.device.get_image_memory_requirements(image);
            let types = (raw.instance).get_physical_device_memory_properties(raw.physical_device);
            let index = (0..types.memory_type_count)
                .find(|&i| needs.memory_type_bits & (1 << i) != 0)
                .unwrap();
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(index);
            let memory = raw.device.allocate_memory(&allocate, None).unwrap();
            raw.device.bind_image_memory(image, memory, 0).unwrap();
            (image, memory)
        };
        let layouts = rt::VkImageLayouts::new(
            metal2vulkan::vk::ImageLayout::UNDEFINED,
            metal2vulkan::vk::ImageLayout::PRESENT_SRC_KHR,
        );
        // SAFETY: the image is made with `info`, bound, in UNDEFINED, and outlives the texture.
        let texture = unsafe {
            (d.raw).new_texture_with_vk_image(&image_descriptor(extent), image, &info, layouts)
        }
        .unwrap();
        let texture = d.texture_of(texture, None, 0, Arc::new(Hold::NONE));
        assert_eq!(texture.unique_identifier(), 0);
        assert_eq!(guest(), without);
        drop(texture);
        // SAFETY: nothing uses the image any more: its texture is gone and ran no command.
        unsafe {
            raw.device.destroy_image(image, None);
            raw.device.free_memory(memory, None);
        }
    }

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
}
