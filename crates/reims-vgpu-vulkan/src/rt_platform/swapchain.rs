//! A display window's swapchain on the rt platform's device ([`RtSwapchain`]): what reims-vgpu's
//! window sink on the rt platform presents through. It follows rt's presenting contract (rt's
//! guide, "Presenting"): the swapchain, the surface, the acquires, the presents and their pacing
//! are ours, and rt renders into the swapchain's images, each a drawable texture of rt's
//! (`Device::new_drawable_texture`) on the process's one device. Nothing is exported, imported or
//! copied.
//!
//! **What it shares with the Vulkan platform's swapchain** (`crate::swapchain`, whose module doc
//! has the reasons and the measurements): the surface's checks (`B8G8R8A8_UNORM` in
//! `SRGB_NONLINEAR`), the present mode (MAILBOX where offered, else FIFO) and the images asked
//! for, the size ([`extent_for`]), the take of an image in calls of 1 ms for up to 33 ms
//! ([`take_image`]), the wait for its fence, the remake that hands the old swapchain over as
//! `oldSwapchain` ([`remake`]), the retired swapchains destroyed once their presents are known
//! complete, from present fences or Vulkan's rules ([`Presents`], [`Fences`]), and the one lock
//! every call into the window system runs under ([`WSI`]), which the two platforms' windows share.
//!
//! **The present mode is logged** at info at the first swapchain made, with the images asked
//! for and the images made; a later make logs it again only where its counts differ from the
//! last line's. Which mode a window presents with depends on what its surface offers, and no
//! other line says it.
//!
//! **Each frame, in rt's order.** The take of an image with a fence, the fence waited for; then
//! `drawable_acquired`, rt's part of `nextDrawable`, which moves the image to its home layout in
//! the upload of rt's next submission; the display's encoders into the image's texture;
//! `present_drawable` with the image's binary semaphore ([`RtSwapchain::encode_ready`]), before
//! the commit; the commit; `wait_until_scheduled`; and, where the command buffer did not end
//! without running, `vkQueuePresentKHR` on rt's presenting queue (`Device::with_present_queue`),
//! waiting on the semaphore ([`RtSwapchain::present`]). A command buffer that ends without running
//! signals nothing: nothing is presented, and the swapchain is made again (rt's guide, "Known gaps
//! of presenting"). An image taken for a frame that then encodes nothing stays taken, its
//! `drawable_acquired` made, for the next draw.
//!
//! **The presenting queue** is rt's: a queue of its own where the device's queue family has a
//! second (NVIDIA's has 16), else the runtime's queue, under the lock every submission of rt's
//! takes. A present takes [`WSI`] before it.
//!
//! **Its calls are timed** for the host's summary, where it counts them
//! ([`super::window_times`]): each `vkAcquireNextImageKHR` call and each `vkQueuePresentKHR` call,
//! each present's wait before its call for rt's lock of the presenting queue, and each
//! `vkCreateSwapchainKHR` call.
//!
//! **Destroying a swapchain** (rt's guide, "Destroying a swapchain"). Every command buffer that
//! drew into or presented one of its images has completed, waited for as it retires, and so has
//! each of its presents (above). An image taken and not presented since, its `drawable_acquired`
//! made (a frame that encoded nothing), has had a command buffer with commands committed after
//! that call complete, which carried its move home: the retire commits a clear of the image on a
//! queue of the swapchain's own and waits for it ([`RtSwapchain::settle_acquired`]). Then its
//! images' textures go, then their semaphores and the swapchain.
//!
//! **Without present fences** (no `swapchainMaintenance1` on the device), a retired swapchain's
//! last present is known complete from Vulkan's rules alone, and where it is not in time, the
//! swapchain waits for the presenting queue to idle inside `with_present_queue`. Where the device
//! has one queue, that queue is the runtime's, so the wait holds the lock of rt's submissions and
//! waits for rt's submitted work too, which rt's contract of `with_present_queue` asks its callers
//! not to do. The wait ends all the same: rt holds a command buffer that waits for an event's
//! value until a signal of it is submitted, and submits no semaphore wait for it (rt's
//! `encode_wait_for_event`), so submitted work waits on nothing a later submission signals.
//!
//! **A lost device.** rt makes its device once and recovers none. Once the device is lost
//! (`RtDevice::is_lost`, rt's `Device::loss`: set at rt's own Vulkan call that met the loss, before
//! any of its command buffers ends with the device lost, or at a Vulkan call of the swapchain's
//! that answered `VK_ERROR_DEVICE_LOST`, which the swapchain reports to rt as the device's loss,
//! rt's `Device::report_loss`, so that rt ends the device's work as at a loss it meets itself), the
//! swapchain makes no Vulkan call on it but its end's destroys: no create, take, wait, present or
//! remake, and no clear of rt's at a retire. Every draw answers [`Acquired::Lost`] at once, and a
//! present presents nothing; the window keeps the frame it shows. Vulkan allows any call on a lost
//! device ("It is still legal to call any API command on the device and child objects", the
//! specification's "Lost Device"), and the calls achieve nothing there; Intel's Windows driver for
//! a UHD 630 died of an access violation inside one of them (`vkCreateSwapchainKHR`, retried at
//! every draw for about 6 minutes after the loss, **measured** 2026-11-20). The destroys are the
//! calls the specification still asks for: a lost device's objects stay valid until destroyed, and
//! are destroyed before their device, which rt destroys at the host's let-go of it. None waits
//! first: nothing runs on a lost device any more.

use ash::{khr, vk};
use metal2vulkan_runtime as rt;

use crate::core::long_waits;
use crate::core::platform::Handle as _;
use crate::core::platform::os::Clock as _;
use crate::core::sync::MutexGuard;
use crate::core::trace;
use crate::swapchain::{
    Acquired, COLOR_SPACE, END_WAIT_NS, FIFO_IMAGES, FORMAT, Fences, Frame, MAILBOX_IMAGES,
    Presents, RETIRED, Refused, Remaking, USAGE, Unmade, WAIT_NS, WSI, extent_for, present_mode,
    refused, remake, take_image,
};

use super::RtObj;
use super::command::RtCommandBuffer;
use super::device::RtDevice;
use super::texture::{self, RtTexture};
use super::unbuilt::rt_refused;
use super::window_times;

/// The log target.
const TARGET: &str = "reims_vgpu_vulkan::rt_swapchain";

/// The call rt's refusals of a drawable's texture and of its acquire are counted under.
const NEXT_DRAWABLE: &str = "nextDrawable";

/// The call rt's refusals of a present are counted under.
const PRESENT_DRAWABLE: &str = "presentDrawable:";

/// One image of the swapchain.
struct Image {
    /// rt's drawable texture over it.
    texture: RtObj<RtTexture>,
    /// The semaphore its frame's command buffer signals (`present_drawable`), and its present
    /// waits on.
    ready: vk::Semaphore,
    /// The command buffer of its last frame, until the swapchain ends.
    last: Option<rt::CommandBuffer>,
    /// Its last present's number ([`Presents`]), 0 before its first.
    presented: u64,
    /// Its `drawable_acquired` was made and no committed command buffer presented it since: its
    /// move home may be owed to rt's next submission (the module doc, "Destroying a swapchain").
    acquired: bool,
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

/// A window's swapchain on the rt platform's device (the module doc), on the window's render
/// thread.
pub struct RtSwapchain {
    /// The device, whose instance the surface is of and whose presenting queue presents.
    device: RtObj<RtDevice>,
    /// rt's logical device's commands.
    vk: ash::Device,
    /// The physical device.
    physical: vk::PhysicalDevice,
    /// `VK_KHR_surface`'s commands.
    surface_api: khr::surface::Instance,
    /// `VK_KHR_swapchain`'s commands.
    swapchain_api: khr::swapchain::Device,
    /// The surface, owned.
    surface: vk::SurfaceKHR,
    /// The present mode.
    mode: vk::PresentModeKHR,
    /// The fence each image is taken with.
    fence: vk::Fence,
    /// The swapchain, once made.
    chain: Option<Chain>,
    /// Swapchains retired, kept until nothing uses them.
    retired: Vec<Chain>,
    /// The presents made, and the newest known complete.
    presents: Presents,
    /// The presents' fences, where the device has present fences.
    fences: Option<Fences>,
    /// The swapchain is out of date, or suboptimal: the next draw makes it again.
    stale: bool,
    /// An image taken whose fence has not signalled yet.
    pending: Option<u32>,
    /// An image taken and ready, its `drawable_acquired` made, kept for the next draw.
    held: Option<u32>,
    /// The surface is lost: no swapchain will be made of it.
    lost: bool,
    /// The last making of the swapchain was refused, which was logged: the next refusal is not.
    refusing: bool,
    /// The image counts the last line of its present mode gave: the images asked for and those
    /// made ([`RtSwapchain::make`]).
    told: Option<(u32, usize)>,
    /// The queue a retire commits its clears on ([`RtSwapchain::settle_acquired`]), made at the
    /// first.
    settling: Option<rt::CommandQueue>,
}

impl std::fmt::Debug for RtSwapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtSwapchain")
            .field("mode", &self.mode.as_raw())
            .field("size", &self.chain.as_ref().map(|c| c.size))
            .field("stale", &self.stale)
            .finish_non_exhaustive()
    }
}

/// A remake of `swapchain` at `size` pixels ([`remake`]).
struct Remake<'a> {
    /// The swapchain.
    swapchain: &'a mut RtSwapchain,
    /// The window's size in pixels.
    size: (u64, u64),
}

impl Remaking for Remake<'_> {
    type Chain = Chain;

    fn take_current(&mut self) -> Option<Chain> {
        self.swapchain.retire()
    }

    fn has_retired(&self) -> bool {
        !self.swapchain.retired.is_empty()
    }

    fn end_retired(&mut self) {
        self.swapchain
            .end_retired("a swapchain made with none to hand over");
    }

    fn make(&mut self, old: Option<&Chain>) -> Result<Chain, Unmade> {
        let old = old.map_or(vk::SwapchainKHR::null(), |c| c.raw);
        self.swapchain.make(self.size, old)
    }

    fn retire(&mut self, chain: Chain) {
        self.swapchain.retired.push(chain);
        self.swapchain.collect();
    }

    fn set_current(&mut self, chain: Option<Chain>, stale: bool) {
        self.swapchain.chain = chain;
        self.swapchain.stale = stale;
    }
}

impl RtSwapchain {
    /// A swapchain over `surface`, presented on `device`'s presenting queue, made at the first
    /// draw. Refused, with the surface destroyed, where the device's queue family does not
    /// present to the surface, or it offers no `B8G8R8A8_UNORM` in `SRGB_NONLINEAR`.
    ///
    /// # Safety
    ///
    /// `surface` is a live surface of `device`'s instance, owned by no one else.
    pub unsafe fn new(
        device: &RtObj<RtDevice>,
        surface: vk::SurfaceKHR,
    ) -> Result<RtSwapchain, Refused> {
        let (entry, instance) = device.instance().raw();
        let surface_api = khr::surface::Instance::new(entry, instance);
        let raw = device.rt().raw();
        let (vk, physical, family) = (
            raw.device.clone(),
            raw.physical_device,
            raw.queue_family_index,
        );
        let swapchain_api = khr::swapchain::Device::new(raw.instance, raw.device);
        let wsi = WSI.lock();
        let destroy = |reason: Refused| {
            // SAFETY: the surface is this call's, and nothing was made of it.
            unsafe { surface_api.destroy_surface(surface, None) };
            reason
        };
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
        // SAFETY: a valid create info, on rt's live device.
        let fence = unsafe { vk.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| destroy(refused("vkCreateFence", e)))?;
        drop(wsi);
        let fences = device.has_present_fences().then(Fences::default);
        Ok(RtSwapchain {
            device: device.clone_ref(),
            vk,
            physical,
            surface_api,
            swapchain_api,
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
            told: None,
            settling: None,
        })
    }

    /// A swapchain over the surface `make` makes on `device`'s instance, given the instance's
    /// `vkGetInstanceProcAddr` (its loader's) and the instance: [`RtSwapchain::new`] over it, or
    /// refused where `make` makes none.
    ///
    /// # Safety
    ///
    /// A surface `make` returns is one it made on the instance it was given, owned by no one
    /// else.
    pub unsafe fn over(
        device: &RtObj<RtDevice>,
        make: impl FnOnce(vk::PFN_vkGetInstanceProcAddr, vk::Instance) -> Option<vk::SurfaceKHR>,
    ) -> Result<RtSwapchain, Refused> {
        let (entry, instance) = device.instance().raw();
        let surface = {
            let _wsi = WSI.lock();
            make(entry.static_fn().get_instance_proc_addr, instance.handle())
        }
        .ok_or_else(|| Refused("no surface".to_owned()))?;
        // SAFETY: the caller's.
        unsafe { RtSwapchain::new(device, surface) }
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
    /// fence waited for up to [`WAIT_NS`] and its `drawable_acquired` made, else a retry (the
    /// module doc). Makes the swapchain first where there is none, where it is stale, or where
    /// its size is another. [`Acquired::Lost`], with no call made, once rt's device is known lost
    /// (the module doc, "A lost device").
    pub fn acquire(&mut self, size: (u64, u64)) -> Acquired {
        if self.device_lost() {
            return self.lost_device();
        }
        if size.0 == 0 || size.1 == 0 {
            return Acquired::Retry;
        }
        let current = self.chain.as_ref().map(|c| c.size);
        if self.stale || current != Some(size) {
            match self.remake(size) {
                Ok(()) => self.refusing = false,
                Err(_) if self.device_lost() => return self.lost_device(),
                Err(e) if self.lost => {
                    log::warn!(
                        target: TARGET, kind:? = e;
                        "no swapchain ({e}): the surface is lost"
                    );
                    return Acquired::Failed;
                }
                Err(e) => {
                    if !self.refusing {
                        log::warn!(
                            target: TARGET, kind:? = e;
                            "no swapchain yet ({e}): tried again at the next draw"
                        );
                    }
                    self.refusing = true;
                    return Acquired::Retry;
                }
            }
        }
        if let Some(index) = self.held {
            return Acquired::Image(Frame::at(index));
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
        let long = long_waits::begin();
        let counted = window_times::on();
        let taken = take_image(
            |timeout| {
                let asked = trace::on().then(trace::now);
                let wsi = WSI.lock();
                let timed = asked.is_some() || counted;
                let locked = timed.then(trace::now);
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
                let returned = timed.then(trace::now);
                // Handed to a window waiting for it, if any, before this one takes it again.
                MutexGuard::unlock_fair(wsi);
                if counted && let (Some(locked), Some(returned)) = (locked, returned) {
                    window_times::acquired(returned.saturating_sub(locked));
                }
                if let (Some(asked), Some(locked), Some(returned)) = (asked, locked, returned) {
                    // winacquire (timed at the call's return): when the take of the window
                    // system's lock began, when the `vkAcquireNextImageKHR` call began, holding
                    // it, and 1 where the call answered an image, else 0
                    let image = u64::from(taken.is_ok());
                    trace::event_at(returned, "winacquire", &[asked, locked, image]);
                }
                taken
            },
            elapsed,
        );
        long.end("window-take");
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
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.note("vkAcquireNextImageKHR", vk::Result::ERROR_DEVICE_LOST);
                self.lost_device()
            }
            Err(e) => {
                log::warn!(
                    target: TARGET, kind:? = e;
                    "vkAcquireNextImageKHR: {e}: the window stops presenting"
                );
                Acquired::Failed
            }
        }
    }

    /// Whether rt's device is lost (the module doc, "A lost device"): the swapchain makes no
    /// Vulkan call on it from then on but its end's destroys.
    fn device_lost(&self) -> bool {
        self.device.is_lost()
    }

    /// The answer of a draw once rt's device is known lost: [`Acquired::Lost`], logged.
    fn lost_device(&self) -> Acquired {
        log::warn!(
            target: TARGET,
            "rt's device is lost, which nothing makes again: the window stops presenting and keeps \
             the frame it shows, and its swapchain makes no Vulkan call on the device but its \
             end's destroys"
        );
        Acquired::Lost
    }

    /// Notes `result`, the answer of the swapchain's Vulkan call `call` on rt's device
    /// ([`answer_seen`]).
    fn note(&self, call: &'static str, result: vk::Result) {
        answer_seen(&self.device, call, result);
    }

    /// Waits up to [`WAIT_NS`] for image `index`'s fence: the image, its `drawable_acquired`
    /// made, or a retry with it pending.
    fn wait_for(&mut self, index: u32) -> Acquired {
        let began = trace::on().then(trace::now);
        let long = long_waits::begin();
        // SAFETY: the fence is live, and image `index`'s acquire signals it.
        let waited = unsafe { self.vk.wait_for_fences(&[self.fence], true, WAIT_NS) };
        long.end("window-fence");
        if let Some(began) = began {
            // winfence (timed at the wait's return): when the wait for a taken image's fence
            // began, and 1 where the fence signalled, else 0
            trace::event("winfence", &[began, u64::from(waited.is_ok())]);
        }
        match waited {
            Ok(()) => {
                // SAFETY: the fence has signalled, and nothing else waits on it.
                if let Err(e) = unsafe { self.vk.reset_fences(&[self.fence]) } {
                    self.note("vkResetFences", e);
                    if self.device_lost() {
                        return self.lost_device();
                    }
                    log::warn!(target: TARGET, kind:? = e; "vkResetFences: {e}");
                    return Acquired::Failed;
                }
                self.pending = None;
                self.taken_again(index);
                self.collect();
                self.acquired(index)
            }
            Err(vk::Result::TIMEOUT) => {
                self.pending = Some(index);
                Acquired::Retry
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.note("vkWaitForFences", vk::Result::ERROR_DEVICE_LOST);
                self.lost_device()
            }
            Err(e) => {
                log::warn!(target: TARGET, kind:? = e; "vkWaitForFences: {e}");
                Acquired::Failed
            }
        }
    }

    /// Image `index`, taken, its fence waited for: held once rt's `drawable_acquired` of its
    /// texture is made; the swapchain's draws end where rt refuses it, counted.
    fn acquired(&mut self, index: u32) -> Acquired {
        let Some(image) = self.image_mut(index) else {
            return Acquired::Failed;
        };
        // SAFETY: the image was taken from the swapchain since its texture was made or last
        // presented, its fence waited for, and the swapchain lives.
        match unsafe { image.texture.rt().drawable_acquired() } {
            Ok(()) => {
                image.acquired = true;
                self.held = Some(index);
                Acquired::Image(Frame::at(index))
            }
            Err(_) if self.device_lost() => self.lost_device(),
            Err(e) => {
                rt_refused(NEXT_DRAWABLE, &e);
                log::warn!(
                    target: TARGET, kind:? = e;
                    "rt refused a window image's acquire ({e}): the window stops presenting"
                );
                Acquired::Failed
            }
        }
    }

    /// The texture over `frame`'s image, which the display's draw draws into. A recorded stream
    /// makes it before its first use, as the Vulkan platform's swapchain's
    /// (`crate::recorder::adopt_rt_texture`).
    pub fn texture(&self, frame: Frame) -> Option<&RtObj<RtTexture>> {
        let texture = self.image(frame.index()).map(|i| &i.texture);
        if let Some(t) = texture {
            crate::recorder::adopt_rt_texture(t);
        }
        texture
    }

    /// Has `command_buffer` present `frame`'s image once its commands have run, signalling the
    /// image's semaphore (rt's `present_drawable`), after what it encoded and before its commit:
    /// false where rt refuses it, counted.
    pub fn encode_ready(&self, frame: Frame, command_buffer: &RtCommandBuffer) -> bool {
        let Some(image) = self.image(frame.index()) else {
            return false;
        };
        // SAFETY: the semaphore is a binary semaphore of rt's device. It is unsignalled with
        // nothing pending on it: the last frame that signalled it was presented, its present
        // waiting on it, and that present is complete, as the image was taken again since and its
        // fence waited for (the Vulkan Guide's "Swapchain Semaphore Reuse"); a frame that ended
        // without running signalled nothing. It lives until the swapchain's end, after the command
        // buffer completes ([`RtSwapchain::retire`]), and only this swapchain's present waits on
        // it.
        let made =
            unsafe { (command_buffer.rt()).present_drawable(image.texture.rt(), image.ready) };
        match made {
            Ok(()) => true,
            Err(e) => {
                rt_refused(PRESENT_DRAWABLE, &e);
                false
            }
        }
    }

    /// Presents `frame`, whose command buffer `command_buffer` was committed with its
    /// [`RtSwapchain::encode_ready`]: once rt has scheduled it, on rt's presenting queue, with a
    /// fence of its own where the device has present fences. A swapchain the compositor reports
    /// out of date or suboptimal, or whose frame ended without running, is made again at the next
    /// draw. Waits for the scheduling alone. False where nothing was presented: the commit was
    /// refused (counted, `RtCommandBuffer`'s commit), and the image is kept for the next draw, the
    /// frame ended without running, or rt's device is lost, as it is before any frame ends with
    /// rt's error naming a lost device (the module doc, "A lost device").
    pub fn present(&mut self, frame: Frame, command_buffer: &RtCommandBuffer) -> bool {
        let raw = command_buffer.rt().clone();
        if matches!(
            raw.status(),
            rt::CommandBufferStatus::NotEnqueued | rt::CommandBufferStatus::Enqueued
        ) {
            // Not committed: its present goes with it, and the image stays taken for the next
            // draw (rt's guide, "Presenting").
            return false;
        }
        self.held = None;
        let index = frame.index();
        let Some(chain) = self.chain.as_mut() else {
            return false;
        };
        let Some(image) = usize::try_from(index)
            .ok()
            .and_then(|i| chain.images.get_mut(i))
        else {
            return false;
        };
        // The image's frame is waited for before its swapchain goes, presented or not.
        image.last = Some(raw.clone());
        image.acquired = false;
        let scheduling = trace::on().then(trace::now);
        let began = long_waits::begin();
        let scheduled = raw.wait_until_scheduled();
        began.end("window-scheduled");
        if scheduled.is_err() || raw.status() == rt::CommandBufferStatus::Error {
            if self.device.is_lost() {
                // rt's loss, set before the frame ended with it: nothing is presented or made
                // again (the module doc, "A lost device").
                return false;
            }
            // It signals nothing, and the image stays taken, its move home perhaps not run: the
            // swapchain is made again, its retire settling the image (the module doc).
            log::warn!(
                target: TARGET, kind:? = scheduled;
                "a window frame ended without running ({scheduled:?}): the swapchain is made again"
            );
            image.acquired = true;
            self.stale = true;
            return false;
        }
        if self.device.is_lost() {
            // Lost since the take: nothing is presented (the module doc, "A lost device").
            return false;
        }
        let waits = [image.ready];
        let chains = [chain.raw];
        let indices = [index];
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
        let counted = window_times::on();
        let timed = scheduling.is_some() || counted;
        let mut called = 0;
        let mut call_returned = 0;
        let present = |queue: vk::Queue| {
            if timed {
                called = trace::now();
            }
            // SAFETY: the present is the only call on rt's presenting queue, inside its lock; the
            // image was taken from this swapchain and its semaphore's signal is submitted, as its
            // command buffer is scheduled. A fence is chained only where the device enabled
            // `swapchainMaintenance1`, and is unsignalled with nothing pending on it.
            let presented = unsafe { api.queue_present(queue, &info) };
            if counted {
                call_returned = trace::now();
            }
            presented
        };
        let asked = scheduling.map(|_| trace::now());
        let began = long_waits::begin();
        let wsi = WSI.lock();
        let locking = counted.then(trace::now);
        let presented = self.device.rt().with_present_queue(present);
        let returned = scheduling.map(|_| trace::now());
        drop(wsi);
        began.end("window-present");
        if let Some(locking) = locking {
            // From the window system's lock taken to the call's start, rt's lock of its
            // presenting queue; then the call.
            window_times::presented(
                called.saturating_sub(locking),
                call_returned.saturating_sub(called),
            );
        }
        if let (Some(scheduling), Some(asked), Some(returned)) = (scheduling, asked, returned) {
            // winpresent (timed at the call's return): when the wait for the frame's command
            // buffer to be scheduled began, when the take of the window system's lock began, when
            // the `vkQueuePresentKHR` call began, holding it and rt's lock of its presenting
            // queue, and 1 where the call answered success, else 0
            let ok = u64::from(presented.is_ok());
            trace::event_at(returned, "winpresent", &[scheduling, asked, called, ok]);
        }
        // Numbered whatever the compositor answers: a present it rejects still waits on its
        // semaphore.
        image.presented = self.presents.make();
        if let (Some(fences), Some(fence)) = (self.fences.as_mut(), fence) {
            fences.presented(image.presented, fence, presented);
        }
        match presented {
            Ok(suboptimal) => self.stale |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.stale = true,
            Err(e) => {
                self.note("vkQueuePresentKHR", e);
                log::warn!(target: TARGET, kind:? = e; "vkQueuePresentKHR: {e}");
                self.stale = true;
            }
        }
        true
    }

    /// Image `index` of the current swapchain.
    fn image(&self, index: u32) -> Option<&Image> {
        let chain = self.chain.as_ref()?;
        chain.images.get(usize::try_from(index).ok()?)
    }

    /// Image `index` of the current swapchain, to change.
    fn image_mut(&mut self, index: u32) -> Option<&mut Image> {
        let chain = self.chain.as_mut()?;
        chain.images.get_mut(usize::try_from(index).ok()?)
    }

    /// Makes the swapchain again at `size` pixels, after the old one's frames ([`remake`]).
    fn remake(&mut self, size: (u64, u64)) -> Result<(), Refused> {
        remake(&mut Remake {
            swapchain: self,
            size,
        })
    }

    /// Destroys every retired swapchain once its presents are known complete, waiting for their
    /// fences, or where they do not tell it, for the presenting queue to idle (the module doc);
    /// `why` is logged where it waits for the queue. With rt's device known lost, it waits for
    /// nothing (the module doc, "A lost device").
    fn end_retired(&mut self, why: &str) {
        let newest = self.retired.iter().map(Chain::last_present).max();
        if let Some(newest) = newest
            && !self.device_lost()
            && !self.wait_fences(newest)
        {
            self.wait_idle(why);
        }
        for chain in std::mem::take(&mut self.retired) {
            self.destroy(chain);
        }
    }

    /// Takes the swapchain out once its frames have completed and every image taken has its move
    /// home run ([`RtSwapchain::settle_acquired`]), waiting on nothing else; its images' textures
    /// and semaphores stay until [`RtSwapchain::destroy`].
    fn retire(&mut self) -> Option<Chain> {
        let mut chain = self.chain.take()?;
        for image in &mut chain.images {
            if let Some(last) = image.last.take() {
                // A frame that failed has completed too; what failed it is the display's.
                let _ = last.wait_until_completed();
            }
        }
        self.settle_acquired(&mut chain);
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

    /// Commits a clear of each image of `chain` whose `drawable_acquired` was made and that no
    /// committed command buffer presented since, on a queue of the swapchain's own, and waits for
    /// it: a command buffer with commands committed after the call has then completed, which
    /// carried the image's move home (the module doc, "Destroying a swapchain"). What the image
    /// held is never shown: the swapchain goes. A clear rt refuses is logged, and the swapchain
    /// goes all the same. With rt's device known lost, none is committed: nothing runs there any
    /// more, a move home included (the module doc, "A lost device").
    fn settle_acquired(&mut self, chain: &mut Chain) {
        if self.device_lost() {
            return;
        }
        for image in chain.images.iter_mut().filter(|i| i.acquired) {
            image.acquired = false;
            if let Err(e) = self.clear(&image.texture) {
                log::warn!(
                    target: TARGET, kind:? = e;
                    "rt refused the clear of a window image taken and not presented ({e}), which \
                     settles its move home before its swapchain goes"
                );
            }
        }
    }

    /// A clear of `texture` in a command buffer of the swapchain's own queue, committed and waited
    /// for.
    fn clear(&mut self, texture: &RtObj<RtTexture>) -> Result<(), rt::Error> {
        let queue = match &self.settling {
            Some(queue) => queue.clone(),
            None => {
                let queue = self.device.rt().new_command_queue(1)?;
                self.settling = Some(queue.clone());
                queue
            }
        };
        let command_buffer = queue.command_buffer()?;
        let mut pass = rt::RenderPassDescriptor::default();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.rt().clone());
        attachment.load_action = rt::LoadAction::Clear;
        attachment.store_action = rt::StoreAction::Store;
        command_buffer.render_command_encoder(&pass)?.end_encoding();
        command_buffer.commit()?;
        command_buffer.wait_until_completed()
    }

    /// Image `index` of the swapchain was taken again, its fence waited for: its last present is
    /// complete ([`Presents::taken_again`]).
    fn taken_again(&mut self, index: u32) {
        if let Some(last) = self.image(index).map(|i| i.presented) {
            self.presents.taken_again(last);
        }
    }

    /// Destroys the retired swapchains whose last present is known complete; past [`RETIRED`]
    /// kept, waits first for the oldest one's presents' fences, or where they do not tell it, for
    /// the presenting queue to idle (the module doc). With rt's device known lost, it does
    /// nothing: the end destroys them (the module doc, "A lost device").
    fn collect(&mut self) {
        if self.device_lost() {
            return;
        }
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
                target: TARGET,
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

    /// Waits for rt's presenting queue to idle (where the device has one queue, rt's own work on
    /// it included, the module doc), after which every present made is taken as complete; `why`
    /// is logged.
    fn wait_idle(&mut self, why: &str) {
        log::debug!(target: TARGET, "waiting for the presenting queue to idle: {why}");
        let commands = &self.vk;
        // SAFETY: the wait submits nothing, on rt's presenting queue inside its lock.
        let wait = |queue: vk::Queue| unsafe { commands.queue_wait_idle(queue) };
        let idle = self.device.rt().with_present_queue(wait);
        if let Err(e) = idle {
            self.note("vkQueueWaitIdle", e);
            log::warn!(target: TARGET, kind:? = e; "vkQueueWaitIdle: {e}");
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
                    answer_seen(&self.device, "vkGetFenceStatus", e);
                    log::warn!(target: TARGET, kind:? = e; "vkGetFenceStatus: {e}");
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
    /// takes those that signalled as complete ([`RtSwapchain::poll_fences`]): whether present
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
            let answer = unsafe { self.vk.wait_for_fences(&waited, true, END_WAIT_NS) };
            if let Err(e) = answer {
                self.note("vkWaitForFences", e);
            }
        }
        self.poll_fences();
        self.presents.done(last)
    }

    /// Waits for an image still pending on the fence, up to [`END_WAIT_NS`], so the fence is
    /// unsignalled with nothing pending on it; where the wait runs out, it is left behind and
    /// another made. The image's index where its fence signalled: it was taken. With rt's device
    /// known lost, it waits for nothing and makes nothing, and the fence goes with the swapchain
    /// (the module doc, "A lost device").
    fn settle_fence(&mut self) -> Option<u32> {
        let index = self.pending.take()?;
        if self.device_lost() {
            return None;
        }
        // SAFETY: the fence is live.
        let waited = unsafe { self.vk.wait_for_fences(&[self.fence], true, END_WAIT_NS) };
        if let Err(e) = waited {
            self.note("vkWaitForFences", e);
        }
        // SAFETY: the fence has signalled, or is left behind unused.
        let settled = waited.is_ok() && unsafe { self.vk.reset_fences(&[self.fence]) }.is_ok();
        if !settled {
            // SAFETY: a valid create info, on rt's live device.
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
        // SAFETY: every frame of the swapchain has completed and every image taken has its move
        // home run ([`RtSwapchain::retire`]), its last present is complete, its semaphore's wait
        // included ([`RtSwapchain::collect`]), or the device is lost, where nothing runs; and its
        // textures are gone: nothing uses its semaphores or images any more.
        unsafe {
            for semaphore in semaphores {
                self.vk.destroy_semaphore(semaphore, None);
            }
            self.swapchain_api.destroy_swapchain(raw, None);
        }
    }

    /// A swapchain for a window of `size` pixels, passing `old` as its `oldSwapchain`, its images
    /// rt's drawable textures (the module doc). A refusal for a lost surface marks it lost.
    fn make(&mut self, size: (u64, u64), old: vk::SwapchainKHR) -> Result<Chain, Unmade> {
        let _wsi = WSI.lock();
        // SAFETY: the surface and physical device are live, of one instance.
        let caps = unsafe {
            self.surface_api
                .get_physical_device_surface_capabilities(self.physical, self.surface)
        }
        .map_err(|e| {
            Unmade::Unasked(self.refused_by("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", e))
        })?;
        let extent = extent_for(size, &caps).ok_or_else(|| {
            Unmade::Unasked(Refused(
                "the window's surface has no pixels for now".to_owned(),
            ))
        })?;
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
        let alpha = if (caps.supported_composite_alpha).contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(FORMAT)
            .image_color_space(COLOR_SPACE)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(self.mode)
            .clipped(true)
            .old_swapchain(old);
        let began = window_times::on().then(trace::now);
        // SAFETY: a valid create info over the live surface; `old` is this swapchain's last.
        let made = unsafe { self.swapchain_api.create_swapchain(&info, None) };
        if let Some(began) = began {
            window_times::created(trace::now().saturating_sub(began));
        }
        let raw = made.map_err(|e| Unmade::Asked(self.refused_by("vkCreateSwapchainKHR", e)))?;
        let images = self
            .wrap(raw, &info)
            .inspect_err(|_| {
                // SAFETY: nothing was presented from the new swapchain.
                unsafe { self.swapchain_api.destroy_swapchain(raw, None) };
            })
            .map_err(Unmade::Asked)?;
        let told = (count, images.len());
        if self.told != Some(told) {
            self.told = Some(told);
            // One of the two modes `present_mode` picks, by name: ash builds its enums without `Debug`.
            let mode = if self.is_mailbox() { "MAILBOX" } else { "FIFO" };
            log::info!(
                target: TARGET,
                "a window's swapchain presents with {mode}, {} images ({count} asked for)",
                images.len()
            );
        }
        Ok(Chain {
            raw,
            size,
            extent: (u64::from(extent.width), u64::from(extent.height)),
            images,
        })
    }

    /// `what`'s refusal with `result`, which marks the surface lost where it says so, and reports
    /// rt's device lost where it says that ([`answer_seen`]; the module doc, "A lost device").
    fn refused_by(&mut self, what: &'static str, result: vk::Result) -> Refused {
        self.lost |= result == vk::Result::ERROR_SURFACE_LOST_KHR;
        self.note(what, result);
        refused(what, result)
    }

    /// `swapchain`'s images, made with `info`, each wrapped as rt's drawable texture with a
    /// semaphore of its own; rt's refusal of a texture is counted.
    fn wrap(
        &self,
        swapchain: vk::SwapchainKHR,
        info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<Vec<Image>, Refused> {
        // SAFETY: `swapchain` is live.
        let raw_images =
            unsafe { self.swapchain_api.get_swapchain_images(swapchain) }.map_err(|e| {
                self.note("vkGetSwapchainImagesKHR", e);
                refused("vkGetSwapchainImagesKHR", e)
            })?;
        let extent = info.image_extent;
        let descriptor = texture::drawable_descriptor(extent.width, extent.height);
        let reached = texture::reaches(descriptor.usage, descriptor.pixel_format);
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
            // A colour target, it takes an index as any texture shaders reach (`texture::reaches`).
            let Some((index, own)) = self.device.index_for(NEXT_DRAWABLE, reached, 0) else {
                return Err(undo(
                    images,
                    Refused("no texture index for a drawable".into()),
                ));
            };
            // SAFETY: the image is one `vkGetSwapchainImagesKHR` gives of `swapchain`, made on
            // rt's device with `info`, exclusive to rt's one queue family; no other drawable
            // texture is over it, and it is not taken. The swapchain is destroyed only once every
            // command buffer that used the texture has completed, and the move home of its last
            // `drawable_acquired` has run, and its texture is gone ([`RtSwapchain::retire`],
            // [`RtSwapchain::destroy`]).
            let made =
                unsafe { (self.device.rt()).new_drawable_texture(&descriptor, index, raw, info) };
            let texture = match made {
                Ok(made) => texture::drawable(&self.device, made, own),
                Err(e) => {
                    if let Some(own) = own {
                        self.device.indices().give_back(own);
                    }
                    rt_refused(NEXT_DRAWABLE, &e);
                    return Err(undo(images, Refused(format!("new_drawable_texture: {e}"))));
                }
            };
            // SAFETY: a valid create info, on rt's live device.
            let ready = match unsafe {
                self.vk
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(ready) => ready,
                Err(e) => {
                    self.note("vkCreateSemaphore", e);
                    return Err(undo(images, refused("vkCreateSemaphore", e)));
                }
            };
            images.push(Image {
                texture,
                ready,
                last: None,
                presented: 0,
                acquired: false,
            });
        }
        Ok(images)
    }
}

/// Where the swapchain's Vulkan call `call` on `device`, rt's, answered `result`: the one place a
/// call of the window's sees rt's device lost. `VK_ERROR_DEVICE_LOST` is reported to rt as the
/// device's loss, of `call` and `result` with no fault report (rt's `Device::report_loss`): rt
/// ends the device's work as at a loss it meets itself, every command buffer the loss reaches
/// failing with the device lost, and names the call as its cause where it is the device's first
/// loss. The device is lost from then on (`RtDevice::is_lost`), so the swapchain makes no Vulkan
/// call on it but its end's destroys, and the window's sink makes no swapchain (the module doc, "A
/// lost device").
fn answer_seen(device: &RtDevice, call: &'static str, result: vk::Result) {
    if result == vk::Result::ERROR_DEVICE_LOST {
        if !device.is_lost() {
            log::info!(target: TARGET, "{call} answered VK_ERROR_DEVICE_LOST: rt's device is lost");
        }
        device.rt().report_loss(rt::DeviceLoss {
            call,
            result,
            fault: None,
        });
    }
}

impl Drop for RtSwapchain {
    /// The swapchain's end, on the render thread: its frames complete and its taken images'
    /// moves home run, and where a present is not known complete, the swapchain waits for its
    /// presents' fences, or for the presenting queue to idle (the module doc); then every
    /// swapchain's images' textures, its semaphores and the swapchain go, then the fences and the
    /// surface. With rt's device known lost, it waits for none of it, and its only Vulkan calls
    /// are the destroys (the module doc, "A lost device").
    fn drop(&mut self) {
        let last = self.retire();
        self.retired.extend(last);
        // An image pending with no swapchain left: its fence is settled before it goes.
        let _ = self.settle_fence();
        self.end_retired("the window's end");
        let _wsi = WSI.lock();
        // SAFETY: every present is complete, or the device is lost, where nothing runs: nothing
        // waits on a fence any more, nor uses the surface.
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
    use crate::core::host::HostPtr;
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::CommandBufferStatus;
    use crate::core::raise::OnFault;
    use crate::rt_platform::RtPlatform;
    use crate::rt_platform::unbuilt::RtRefusal;

    /// Every refusal of rt's counted so far, by kind.
    fn refusals() -> Vec<u64> {
        let counts = crate::answer::degradation_counts();
        let counted = |code: &str| {
            (counts.iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| *n)
        };
        RtRefusal::ALL.iter().map(|k| counted(k.code())).collect()
    }

    /// The next image of `swapchain` for a window of `size` pixels, taken again while the
    /// compositor has none to give.
    fn take(swapchain: &mut RtSwapchain, size: (u64, u64)) -> Frame {
        for _ in 0..100 {
            match swapchain.acquire(size) {
                Acquired::Image(frame) => return frame,
                Acquired::Retry => {}
                ended => panic!("no image of {size:?}: {ended:?}"),
            }
        }
        panic!("no image of {size:?} in 100 takes");
    }

    /// A headless surface (`VK_EXT_headless_surface`) on `device`'s instance, where the loader
    /// offers one and the device's driver presents to it; `None`, saying why, where not.
    fn headless_surface(device: &RtDevice) -> Option<vk::SurfaceKHR> {
        let headless = ash::ext::headless_surface::NAME;
        if !(device.instance().extensions().iter()).any(|e| e.as_c_str() == headless) {
            eprintln!("the loader offers no VK_EXT_headless_surface: no window swapchain tested");
            return None;
        }
        let (entry, instance) = device.instance().raw();
        let surfaces = khr::surface::Instance::new(entry, instance);
        let made = ash::ext::headless_surface::Instance::new(entry, instance);
        let info = vk::HeadlessSurfaceCreateInfoEXT::default();
        // SAFETY: a valid create info, on rt's live instance, which enabled the extension.
        let surface = unsafe { made.create_headless_surface(&info, None) }.expect("a surface");
        let raw = device.rt().raw();
        let (physical, family) = (raw.physical_device, raw.queue_family_index);
        // SAFETY: the surface and the physical device are live, of one instance.
        let presents =
            unsafe { surfaces.get_physical_device_surface_support(physical, family, surface) };
        if presents != Ok(true) {
            eprintln!("the device presents to no headless surface ({presents:?}): none tested");
            // SAFETY: the surface is this test's, and nothing was made of it.
            unsafe { surfaces.destroy_surface(surface, None) };
            return None;
        }
        Some(surface)
    }

    /// A window's swapchain on rt's device over a headless surface (`VK_EXT_headless_surface`,
    /// where the device's driver presents to one): each frame takes an image, clears its
    /// drawable and copies it out in a command buffer that presents it, and the copy holds the
    /// clear; the swapchain is made again at a new size; an image taken for a frame that encodes
    /// nothing is the next draw's, and the swapchain ends with it taken, its move home settled.
    /// rt refuses nothing on the way. The present mode is logged once, at the first make: the
    /// remake at the new size makes as many images.
    #[test]
    fn a_window_presents_rt_drawables_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let Some(surface) = headless_surface(&device) else {
            return;
        };
        let before = refusals();
        let logs = crate::core::testkit::logs::capture();
        // SAFETY: the surface is live, of the device's instance, and the swapchain's alone.
        let made = unsafe { RtSwapchain::new(&device, surface) };
        let mut swapchain = made.expect("a swapchain over the headless surface");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let sizes = [(64, 48), (64, 48), (64, 48), (80, 40), (80, 40)];
        for (n, (width, height)) in sizes.into_iter().enumerate() {
            let size = (u64::from(width), u64::from(height));
            let frame = take(&mut swapchain, size);
            assert_eq!(swapchain.extent(), Some(size), "frame {n}");
            let texture = swapchain.texture(frame).expect("the frame's drawable");
            let length = size.0 * size.1 * 4;
            let buffer = device.new_buffer_of_length(on, length, 0);
            let buffer = buffer.expect("never a raise").expect("a Shared buffer");
            let made = queue.command_buffer(on).expect("never a raise");
            let made = made.expect("a command buffer");
            let mut pass = rt::RenderPassDescriptor::default();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(texture.rt().clone());
            attachment.load_action = rt::LoadAction::Clear;
            attachment.clear_color = rt::ClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            };
            let clear = made.rt().render_command_encoder(&pass);
            clear.expect("a clear of the drawable").end_encoding();
            let mut blit = made.rt().blit_command_encoder().expect("a blit encoder");
            let layout = rt::BufferLayout {
                offset: 0,
                bytes_per_row: size.0 * 4,
                bytes_per_image: length,
            };
            let from = rt::TexturePlace::default();
            let copied = blit.copy_from_texture_to_buffer(
                texture.rt(),
                from,
                [width, height, 1],
                buffer.rt(),
                layout,
            );
            copied.expect("a copy out of the drawable");
            blit.end_encoding();
            assert!(swapchain.encode_ready(frame, &made), "frame {n}");
            assert_eq!(made.commit(on), Ok(()));
            assert!(swapchain.present(frame, &made), "frame {n}");
            assert_eq!(made.wait_until_completed(on), Ok(()));
            assert_eq!(made.status(), CommandBufferStatus::COMPLETED, "frame {n}");
            let at: HostPtr = buffer.contents();
            let length = usize::try_from(length).expect("a small frame");
            // SAFETY: a live Shared buffer's contents, `length` bytes long.
            let bytes = unsafe { std::slice::from_raw_parts(at.as_ptr::<u8>(), length) };
            // `BGRA8Unorm`: opaque red.
            assert!(bytes.chunks(4).all(|p| p == [0, 0, 255, 255]), "frame {n}");
        }
        // A frame that took an image and encoded nothing: the next draw takes it again.
        let kept = take(&mut swapchain, (80, 40));
        assert_eq!(take(&mut swapchain, (80, 40)), kept);
        let told: Vec<String> = (logs.take().into_iter())
            .filter(|(_, target, line)| target == TARGET && line.contains("presents with"))
            .map(|(_, _, line)| line)
            .collect();
        let images = (swapchain.chain.as_ref()).map_or(0, |c| c.images.len());
        let mode = if swapchain.is_mailbox() {
            "MAILBOX"
        } else {
            "FIFO"
        };
        let start = format!("a window's swapchain presents with {mode}, {images} images (");
        assert!(
            told.len() == 1 && told[0].starts_with(&start) && told[0].ends_with(" asked for)"),
            "{told:?}"
        );
        drop(swapchain);
        assert_eq!(refusals(), before);
    }

    /// A traced frame of a window's swapchain over a headless surface, on the GPU the test holds:
    /// each take of its image writes `winacquire`, the wait for the image's fence `winfence`, and
    /// its present `winpresent`, all on the thread that draws, on the trace's clock (`cbdone`'s):
    /// each time a value holds is at or after the one before it, and none is after its line's own
    /// time; the last take answered an image, its fence signalled, and the present succeeded.
    #[test]
    fn a_traced_window_frame_writes_its_takes_and_present_on_vulkan() {
        // nextest runs each test in a process of its own, whose trace file is this one
        let path = std::env::temp_dir().join(format!("rvg-rt-window-{}.txt", std::process::id()));
        assert!(trace::to_file(&path));
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let Some(surface) = headless_surface(&device) else {
            std::fs::remove_file(&path).unwrap();
            return;
        };
        // SAFETY: the surface is live, of the device's instance, and the swapchain's alone.
        let made = unsafe { RtSwapchain::new(&device, surface) };
        let mut swapchain = made.expect("a swapchain over the headless surface");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let frame = take(&mut swapchain, (64, 48));
        let texture = swapchain.texture(frame).expect("the frame's drawable");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let mut pass = rt::RenderPassDescriptor::default();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.rt().clone());
        attachment.load_action = rt::LoadAction::Clear;
        let clear = made.rt().render_command_encoder(&pass);
        clear.expect("a clear of the drawable").end_encoding();
        assert!(swapchain.encode_ready(frame, &made));
        assert_eq!(made.commit(on), Ok(()));
        assert!(swapchain.present(frame, &made));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        drop(swapchain);
        trace::flush_before_death();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        // Each line of `kind`: its thread, then its time and its values.
        let lines = |kind: &str| -> Vec<(String, Vec<u64>)> {
            (text.lines())
                .map(|l| l.split(' ').collect::<Vec<_>>())
                .filter(|f| f.get(2) == Some(&kind))
                .map(|f| {
                    let at = f[0].parse::<u64>().unwrap();
                    let values = f[3..].iter().map(|v| v.parse::<u64>().unwrap());
                    (f[1].to_owned(), std::iter::once(at).chain(values).collect())
                })
                .collect()
        };
        let (takes, fences, presents) =
            (lines("winacquire"), lines("winfence"), lines("winpresent"));
        assert!(
            !takes.is_empty() && !fences.is_empty() && presents.len() == 1,
            "{text}"
        );
        let thread = &presents[0].0;
        for (th, f) in takes.iter().chain(&fences).chain(&presents) {
            assert_eq!(th, thread, "{text}");
            // The values are times but the last, which is the call's answer.
            let (&answer, times) = f[1..].split_last().unwrap();
            assert!(times[0] > 0 && answer <= 1, "{text}");
            assert!(times.windows(2).all(|w| w[0] <= w[1]), "{text}");
            assert!(times.last() <= Some(&f[0]), "{text}");
        }
        let answered =
            |lines: &[(String, Vec<u64>)]| lines.last().and_then(|(_, f)| f.last().copied());
        assert_eq!(answered(&takes), Some(1), "{text}");
        assert_eq!(answered(&fences), Some(1), "{text}");
        assert_eq!(answered(&presents), Some(1), "{text}");
    }

    /// A frame of a window's swapchain over a headless surface, on the GPU the test holds, with
    /// the window's times counted (`super::window_times`): its takes' `vkAcquireNextImageKHR`
    /// calls, one at least, its one `vkQueuePresentKHR` call and its wait for the presenting
    /// queue's lock, its swapchain's one `vkCreateSwapchainKHR` call, and, through a completed
    /// handler as the window's draw adds one, its command buffer's GPU time; each call took some
    /// time, and none longer than the period's in all.
    #[test]
    fn a_counted_window_frame_times_its_calls_and_gpu_time_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let Some(surface) = headless_surface(&device) else {
            return;
        };
        // SAFETY: the surface is live, of the device's instance, and the swapchain's alone.
        let made = unsafe { RtSwapchain::new(&device, surface) };
        let mut swapchain = made.expect("a swapchain over the headless surface");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        window_times::count();
        let _ = window_times::take();
        let frame = take(&mut swapchain, (64, 48));
        let texture = swapchain.texture(frame).expect("the frame's drawable");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let mut pass = rt::RenderPassDescriptor::default();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.rt().clone());
        attachment.load_action = rt::LoadAction::Clear;
        let clear = made.rt().render_command_encoder(&pass);
        clear.expect("a clear of the drawable").end_encoding();
        assert!(swapchain.encode_ready(frame, &made));
        let (ran, done) = std::sync::mpsc::channel();
        let handler = Box::new(move |info: crate::core::platform::command::CompletedInfo| {
            window_times::frame_ran(info.gpu_times_ns);
            let _ = ran.send(());
        });
        assert_eq!(made.add_completed_handler(on, handler), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert!(swapchain.present(frame, &made));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        let waited = done.recv_timeout(std::time::Duration::from_secs(10));
        waited.expect("the frame's completed handler ran");
        drop(swapchain);
        let counted = window_times::take();
        let (acquires, presents) = (counted.acquires, counted.presents);
        let (waits, frames) = (counted.present_lock_waits, counted.frames_on_gpu);
        let creates = counted.creates;
        assert!(acquires.count >= 1, "{counted:?}");
        assert_eq!(
            (presents.count, waits.count, frames.count, creates.count),
            (1, 1, 1, 1),
            "{counted:?}"
        );
        for spans in [acquires, presents, frames, creates] {
            assert!(spans.total > std::time::Duration::ZERO, "{counted:?}");
        }
        for spans in [acquires, presents, waits, frames, creates] {
            assert!(spans.longest <= spans.total, "{counted:?}");
        }
    }

    /// A clear of `frame`'s drawable on `queue`, `swapchain`'s present of it encoded and the
    /// command buffer committed and waited for: the command buffer.
    fn clear_and_commit(
        device: &RtObj<RtDevice>,
        swapchain: &RtSwapchain,
        frame: Frame,
    ) -> RtObj<RtCommandBuffer> {
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let texture = swapchain.texture(frame).expect("the frame's drawable");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let mut pass = rt::RenderPassDescriptor::default();
        pass.color_attachments[0].texture = Some(texture.rt().clone());
        pass.color_attachments[0].load_action = rt::LoadAction::Clear;
        let clear = made.rt().render_command_encoder(&pass);
        clear.expect("a clear of the drawable").end_encoding();
        assert!(swapchain.encode_ready(frame, &made));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        made
    }

    /// Once rt's device is lost, by `lose`, which answers the loss it made, a window's swapchain
    /// over a headless surface on the GPU the test holds makes no Vulkan call on it but its end's
    /// destroys (the module doc, "A lost device"): rt names that loss; a frame committed before it
    /// presents nothing; every draw, at its size or another, answers [`Acquired::Lost`] with no
    /// take, present or create counted and the swapchain made at the first draw kept, retiring
    /// none; a command buffer committed after it fails with the device lost, not in flight; and
    /// the swapchain's end commits no clear for the image taken and not presented, and waits on
    /// nothing. rt refuses nothing on the way, and the loss is logged.
    fn lost_swapchain(lose: impl FnOnce(&RtDevice, &mut RtSwapchain) -> rt::DeviceLoss) {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let Some(surface) = headless_surface(&device) else {
            return;
        };
        let before = refusals();
        let logs = crate::core::testkit::logs::capture();
        // SAFETY: the surface is live, of the device's instance, and the swapchain's alone.
        let made = unsafe { RtSwapchain::new(&device, surface) };
        let mut swapchain = made.expect("a swapchain over the headless surface");
        let size = (64, 48);
        let frame = take(&mut swapchain, size);
        let shown = clear_and_commit(&device, &swapchain, frame);
        assert!(
            swapchain.present(frame, &shown),
            "the frame before the loss"
        );
        let kept = take(&mut swapchain, size);
        let unshown = clear_and_commit(&device, &swapchain, kept);
        let raw = (swapchain.chain.as_ref()).map(|c| c.raw);
        assert!(!device.is_lost());
        let loss = lose(&device, &mut swapchain);
        assert!(device.is_lost(), "the device is lost");
        assert_eq!(device.rt().loss(), Some(loss), "rt's loss");
        window_times::count();
        let _ = window_times::take();
        assert!(!swapchain.present(kept, &unshown), "nothing is presented");
        for size in [size, (80, 40), size] {
            assert!(
                matches!(swapchain.acquire(size), Acquired::Lost),
                "{size:?}"
            );
        }
        assert_eq!(
            swapchain.chain.as_ref().map(|c| c.raw),
            raw,
            "none made again"
        );
        assert_eq!(swapchain.extent(), Some(size));
        assert!(swapchain.retired.is_empty());
        let counted = window_times::take();
        assert_eq!(
            (
                counted.acquires.count,
                counted.presents.count,
                counted.creates.count
            ),
            (0, 0, 0),
            "{counted:?}"
        );
        // rt ends the device's work.
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let late = queue.command_buffer(on).expect("never a raise");
        let late = late.expect("a command buffer");
        assert_eq!(late.commit(on), Ok(()));
        assert_eq!(late.wait_until_completed(on), Ok(()));
        assert_eq!(
            late.rt().error(),
            Some(rt::CommandBufferError::DeviceLost { in_flight: false }),
            "a commit after the loss"
        );
        // The end's retire: the image taken and not presented gets no clear, and no queue is made
        // for one.
        let chain = swapchain.retire();
        assert!(chain.is_some() && swapchain.settling.is_none());
        swapchain.retired.extend(chain);
        drop(swapchain);
        let lost: Vec<String> = (logs.take().into_iter())
            .filter(|(_, target, line)| target == TARGET && line.contains("rt's device is lost"))
            .map(|(_, _, line)| line)
            .collect();
        assert!(!lost.is_empty(), "the loss is logged");
        assert_eq!(refusals(), before);
    }

    /// A Vulkan call of a window's swapchain that answers `VK_ERROR_DEVICE_LOST` (here its
    /// create's refusal, as on Intel's Windows driver after rt's device was lost) is rt's device's
    /// loss, reported to rt (`Device::report_loss`), which names that call and result; the
    /// swapchain makes no Vulkan call on the device from then on but its end's destroys
    /// ([`lost_swapchain`]).
    #[test]
    fn a_lost_device_s_swapchain_makes_no_vulkan_call_but_its_end_s_destroys_on_vulkan() {
        lost_swapchain(|_, swapchain| {
            let _ = swapchain.refused_by("vkCreateSwapchainKHR", vk::Result::ERROR_DEVICE_LOST);
            rt::DeviceLoss {
                call: "vkCreateSwapchainKHR",
                result: vk::Result::ERROR_DEVICE_LOST,
                fault: None,
            }
        });
    }

    /// A loss of rt's own, as at a submission of rt's that meets it (here reported as rt's
    /// `vkQueueSubmit2`), stops a window's swapchain from rt's `Device::loss` alone, before any
    /// command buffer has failed with the device lost: it makes no Vulkan call on the device from
    /// then on but its end's destroys ([`lost_swapchain`]).
    #[test]
    fn a_swapchain_makes_no_vulkan_call_from_rt_s_own_loss_on_vulkan() {
        lost_swapchain(|device, _| {
            let loss = rt::DeviceLoss {
                call: "vkQueueSubmit2",
                result: vk::Result::ERROR_DEVICE_LOST,
                fault: None,
            };
            device.rt().report_loss(loss.clone());
            loss
        });
    }
}
