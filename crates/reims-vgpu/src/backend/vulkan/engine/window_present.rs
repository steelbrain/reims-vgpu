//! Engine-device WSI presentation for the macOS Vulkan host window.
//!
//! The final compositor resident stays on the engine `VkDevice`. A short
//! queue-ordered blit writes it into the acquired MoltenVK swapchain image; no
//! host readback, staging upload, or second Vulkan device exists
//! on this pathway.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::time::Instant;

use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::facade_decline::EngineFacadeDecline;
use super::pools::ResourcePools;
use super::types::{DrawError, PresentRect, TargetIdentity, WindowPresentSource};
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::translate;

/// Consecutive suboptimal-flagged presents (each of which arms a swapchain
/// recreation) before the always-on alarm names the class. Recreation normally
/// clears the flag on the next frame, and a live user resize clears the streak
/// whenever the extent actually changes. A streak this long at an unchanged
/// extent means recreation is not converging and the window may be presenting
/// invisibly (the CAMetalLayer drawableSize-clobber class).
const SUBOPTIMAL_ALARM_STREAK: u32 = 60;

/// The pre-content / letterbox-bar clear color (linear BGRA channels).
const SLATE_CLEAR: [f32; 4] = [0.05, 0.06, 0.08, 1.0];

const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// One frame at the refresh rate exposed to the guest bounds an acquire retry.
///
/// A zero-timeout probe stays first so the common path never sleeps with the
/// shared engine lock held. If all images are busy, the bounded retry gives a
/// WSI implementation a chance to process presentation-engine image releases.
/// This is an engine-lock budget derived from the guest display contract; it
/// makes no claim about the host presentation interval.
const fn guest_frame_budget_ns(refresh_hz: u32) -> Option<u64> {
    if refresh_hz == 0 {
        None
    } else {
        Some(NANOS_PER_SECOND / refresh_hz as u64)
    }
}

const ACQUIRE_RETRY_BUDGET_NS: u64 = match guest_frame_budget_ns(crate::model::DISPLAY_REFRESH_HZ) {
    Some(timeout) => timeout,
    None => 0,
};
const _: () = assert!(ACQUIRE_RETRY_BUDGET_NS > 0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AcquireRetry {
    attempted: bool,
    rescued: bool,
}

/// Probe without waiting, then enter WSI's bounded wait/dispatch path only when
/// every image is currently busy. Keeping the policy here makes the accepted
/// retry results and the one-retry bound independently testable.
fn acquire_with_bounded_retry<T>(
    mut acquire: impl FnMut(u64) -> Result<T, vk::Result>,
) -> (Result<T, vk::Result>, AcquireRetry) {
    let first = acquire(0);
    match first {
        Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
            let retried = acquire(ACQUIRE_RETRY_BUDGET_NS);
            let retry = AcquireRetry {
                attempted: true,
                rescued: retried.is_ok(),
            };
            (retried, retry)
        }
        result => (result, AcquireRetry::default()),
    }
}

/// A host-window present degradation that does not abort the whole present.
///
/// This is not a [`SlateReason`]: a persistent suboptimal flag still queues
/// presents while warning that swapchain recreation is not converging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowPresentDecline {
    SuboptimalPersistent {
        streak: u32,
        width: u32,
        height: u32,
    },
}

impl crate::observe::Decline for WindowPresentDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::SuboptimalPersistent { .. } => "window_present_suboptimal_persistent",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::SuboptimalPersistent {
                streak,
                width,
                height,
            } => vec![
                ("streak", streak.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
        }
    }
}

/// Why a present cleared to slate instead of blitting a guest resident.
///
/// A slate present is the window showing *nothing* — on the arm64 MoltenVK
/// pathway it is the whole "blank window" failure class, and it used to happen
/// with no log line at all: the caller only reported the FIRST direct present,
/// so a later regression into slate was invisible except as a drop in
/// `direct_frac`. Every slate run now names its cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlateReason {
    /// No frame source was published for this present. Expected before the
    /// present boundary and while the guest is idle.
    NoSource,
    /// The source named candidate identities but none is in the resident
    /// registry — the resident was evicted, or never created.
    NoResident,
    /// A resident exists but its content has not landed yet.
    ContentNotReady,
    /// A resident exists and is ready, but is not BGRA. The present blit does
    /// no format conversion, so it cannot be shown.
    NotBgra,
    /// A resident exists and is ready, but at different dimensions than the
    /// source claims — presenting it would show a torn or scaled frame.
    GeomMismatch,
}

impl crate::observe::Decline for SlateReason {
    /// Slugs carry a `slate_` prefix.
    ///
    /// They were bare (`no_source`, `geom_mismatch`, …) while this type was an
    /// island with its own `slug()`. Crate-wide they read as claims about the
    /// whole present path rather than about the window's blit choice, and
    /// `geom_mismatch` is also a `THRASH` proxy name while `no_resident` sits
    /// one word away from the capture rail's `no_resident_content`. A grep for
    /// a bare one would mix three different subsystems.
    fn slug(&self) -> &'static str {
        match self {
            Self::NoSource => "slate_no_source",
            Self::NoResident => "slate_no_resident",
            Self::ContentNotReady => "slate_content_not_ready",
            Self::NotBgra => "slate_not_bgra",
            Self::GeomMismatch => "slate_geom_mismatch",
        }
    }
}

/// Why the CPU fallback source could not be staged.
///
/// Its own type rather than a `DrawError` because the present does not abort on
/// it: the swapchain image is already acquired, so the frame degrades to slate
/// and the window stays alive. Each variant names the exact call that refused,
/// so the fix is not a guess about which of five allocations failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagingError {
    Call(VkCall),
    /// No memory type satisfies `MemoryClass::Upload` for the staging image.
    /// Vulkan guarantees a `HOST_VISIBLE|HOST_COHERENT` type exists, so this
    /// means the image's own `memoryTypeBits` excluded every one of them.
    NoUploadMemoryType {
        type_bits: u32,
    },
}

impl crate::observe::Decline for StagingError {
    fn slug(&self) -> &'static str {
        match self {
            Self::Call(call) => call.slug(),
            Self::NoUploadMemoryType { .. } => "window_staging_no_upload_memory_type",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Call(call) => call.fields(),
            Self::NoUploadMemoryType { type_bits } => {
                vec![("type_bits", format!("{type_bits:#x}"))]
            }
        }
    }
}

/// What the registry knows about the identity a present named, flattened so the
/// classification below is pure and testable without a GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CandidateState {
    /// The identity resolved to a registry slot.
    pub resident: bool,
    pub content_ready: bool,
    pub bgra: bool,
    pub width: u32,
    pub height: u32,
}

/// Name why the resident the present named could not carry it.
///
/// Each arm is a distinct blocker with a distinct remedy, checked in the order
/// a slot progresses through them (created → content landed → correct format →
/// correct size). Collapsing them into one "no_resident" is the exact "N
/// distinct checks share one status" trap the failure-logging rules call out.
pub(crate) fn classify_slate(
    source_present: bool,
    want: (u32, u32),
    state: CandidateState,
) -> SlateReason {
    if !source_present {
        return SlateReason::NoSource;
    }
    if !state.resident {
        return SlateReason::NoResident;
    }
    if !state.content_ready {
        return SlateReason::ContentNotReady;
    }
    if !state.bgra {
        return SlateReason::NotBgra;
    }
    if (state.width, state.height) != want {
        return SlateReason::GeomMismatch;
    }
    // Resident, ready, BGRA and the right size — `slot_presentable` agreed with
    // none of the blockers above, so the caller took the resident and never got
    // here. Reaching this arm means the two disagree; report the residual class
    // rather than inventing a sixth.
    SlateReason::ContentNotReady
}

/// A CPU-BGRA frame offered as the present source when no resident carries the
/// display.
///
/// The resident is always preferred: taking it is the whole point of presenting
/// on the engine device, and it costs no host memory traffic. This exists for
/// the presents that have no resident at all — the firmware/boot framebuffer, a
/// mapping the compositor has cleared but never rendered into, and the frames
/// after a device reset. Without it those presents would show slate, which on
/// Linux would be a blank window for the whole of early boot.
///
/// Measured on x86/Vulkan, and the numbers say exactly that and no more. Once
/// the guest is compositing, `host_window_cadence` reports `direct_frac=1.00`
/// across every sampling window of a driven Safari session — every present
/// comes from a resident and this path carries none of them. Before that, one
/// boot logged a single `slate_no_source` run of 358 frames with `covered=1`,
/// which is this path holding the window through firmware boot and then handing
/// over.
///
/// So it is boot-scope, not dead: a reader who deletes it because steady-state
/// traffic is zero blanks the window for the first several hundred frames.
#[derive(Clone, Copy, Debug)]
pub struct WindowCpuFrame<'a> {
    pub bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// Publish sequence of the frame these bytes came from. The staging image
    /// keeps the last one it uploaded, so a forced redraw (resize, suboptimal
    /// self-heal) re-blits without re-copying 8 MB that have not changed.
    pub seq: u64,
}

/// Whether a published CPU frame holds every byte of the geometry it claims.
///
/// A short buffer is not a degraded frame, it is a torn one: the blit would
/// read whatever the staging image held below the copied rows, which is the
/// previous frame at whatever geometry it had. Kept out of the staging code so
/// the rejection is a value test rather than a length check buried in an unsafe
/// copy loop.
fn cpu_frame_complete(frame: &WindowCpuFrame<'_>) -> bool {
    if frame.width == 0 || frame.height == 0 {
        return false;
    }
    let need = (frame.width as usize)
        .saturating_mul(frame.height as usize)
        .saturating_mul(4);
    need != 0 && frame.bgra.len() >= need
}

/// The staging image's persistent host mapping.
///
/// A raw pointer is not `Send`, and [`WindowPresenter`] lives inside the global
/// engine mutex, which must be. The mapping is created with the image, lives
/// exactly as long as it, and is only ever dereferenced by the thread holding
/// that mutex — so moving the address across threads is sound. Saying so in a
/// wrapper keeps it a pointer in the type system; laundering it through a
/// `usize` would hide the same claim behind an integer.
struct MappedStaging(*mut u8);

// SAFETY: see the type's documentation — ownership is exclusive under the engine
// mutex and the mapping outlives every dereference.
unsafe impl Send for MappedStaging {}

/// Host-visible LINEAR image the CPU fallback frame is copied into, then
/// scale-blitted into the acquired swapchain image.
///
/// LINEAR because the copy is a host write through a persistent map, and a host
/// write to an OPTIMAL image has no defined layout. Row pitch comes from the
/// driver rather than from the width: it is free to pad, and copying tightly
/// into a padded image shears the picture.
struct StagingImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    mapped: MappedStaging,
    width: u32,
    height: u32,
    row_pitch: u64,
    offset: u64,
    /// Whether the image has ever been transitioned out of `PREINITIALIZED`.
    /// The first blit must declare that layout as the old one so the host
    /// writes are not discarded; every later blit declares `GENERAL`.
    transitioned: bool,
    /// [`WindowCpuFrame::seq`] of the bytes currently held, or `None` for a
    /// freshly created image that holds no frame.
    staged_seq: Option<u64>,
}

impl StagingImage {
    unsafe fn destroy(self, device: &ash::Device) {
        device.unmap_memory(self.memory);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// The image this present blits from, and what it takes to make it readable.
///
/// A resident lives in whatever layout its last draw left it and moves to
/// `TRANSFER_SRC_OPTIMAL`. The staging image is host-written through a
/// persistent map, so it must stay in a layout that permits host access —
/// `GENERAL` — and needs a `HOST_WRITE → TRANSFER_READ` barrier instead of a
/// layout transition. Reading a host-written image from `TRANSFER_SRC_OPTIMAL`
/// is the defect this distinction exists to prevent.
#[derive(Clone, Copy)]
enum BlitSource {
    Resident {
        image: vk::Image,
        access: super::pools::ResidentAccess,
        width: u32,
        height: u32,
    },
    Staged {
        image: vk::Image,
        /// The image has never left `PREINITIALIZED`, so that is the layout the
        /// barrier must declare — only `PREINITIALIZED` and `GENERAL` preserve
        /// contents, and declaring the wrong one discards the frame just
        /// uploaded.
        first_use: bool,
        width: u32,
        height: u32,
    },
}

impl BlitSource {
    fn image(&self) -> vk::Image {
        match self {
            Self::Resident { image, .. } | Self::Staged { image, .. } => *image,
        }
    }

    fn extent(&self) -> (u32, u32) {
        match self {
            Self::Resident { width, height, .. } | Self::Staged { width, height, .. } => {
                (*width, *height)
            }
        }
    }

    /// Record the barrier that makes this source readable by the blit, and
    /// return the layout the blit must name.
    unsafe fn record_read_barrier(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
    ) -> vk::ImageLayout {
        match self {
            Self::Resident { image, access, .. } => {
                super::exec::barrier_resident_for_transfer_read(device, cmd, *image, *access);
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            }
            Self::Staged {
                image, first_use, ..
            } => {
                let old = if *first_use {
                    vk::ImageLayout::PREINITIALIZED
                } else {
                    vk::ImageLayout::GENERAL
                };
                image_barrier(
                    device,
                    cmd,
                    *image,
                    old,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::HOST_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::TRANSFER,
                );
                vk::ImageLayout::GENERAL
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPresentOutcome {
    Busy,
    Presented {
        direct: bool,
        width: u32,
        height: u32,
        swapchain_images: usize,
        /// The surface reported suboptimal at acquire or present, so a
        /// recreation is armed. The window must schedule another redraw
        /// promptly instead of waiting for the next guest frame — boot-era
        /// presents can be seconds apart, which would leave a mismatched
        /// drawable on screen for that long.
        suboptimal: bool,
    },
}

/// MAILBOX where the surface offers it, FIFO where it does not.
///
/// FIFO is the only mode Vulkan guarantees, so it is the fallback — including
/// when the mode query itself fails, which reaches here as an empty slice.
pub(crate) fn choose_present_mode(supported: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if supported.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// The two swapchain decisions that have to be made together, returned together.
///
/// They were computed separately and only one of them reached
/// `vkCreateSwapchainKHR`: the mode was chosen, handed to
/// [`swapchain_image_count`], and then dropped in favour of a literal `FIFO` in
/// the create info. The census printed the *chosen* mode, so a log read
/// `present_mode=mailbox` beside a swapchain that was FIFO, and the change that
/// introduced the choice measured "no effect" because it never reached the
/// driver. One value carried in one struct is what stops that shape: the count
/// is derived from the very mode the create info is given.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SwapchainPlan {
    pub present_mode: vk::PresentModeKHR,
    pub image_count: u32,
}

/// The mode the surface offers and the image count that mode needs.
pub(crate) fn swapchain_plan(
    caps_min: u32,
    caps_max: u32,
    supported: &[vk::PresentModeKHR],
) -> SwapchainPlan {
    let present_mode = choose_present_mode(supported);
    SwapchainPlan {
        present_mode,
        image_count: swapchain_image_count(caps_min, caps_max, present_mode),
    }
}

/// The MAILBOX floor: one image queued, one being drawn, one to replace with.
///
/// Named rather than written inline because [`PRESENT_IN_FLIGHT`] is bounded by
/// it, and a bound that restates a literal from another function is the kind
/// that stops being true silently.
const MAILBOX_MIN_IMAGES: u32 = 3;

/// Swapchain image count for a mode, inside the surface's own bounds.
///
/// `min + 1` is the usual one-spare rule. MAILBOX needs a third image to have
/// something to replace while one is queued and one is being drawn, so the floor
/// is raised on that arm only — and `max_image_count` still wins, since a
/// surface that caps at two cannot be argued with (0 means no maximum).
pub(crate) fn swapchain_image_count(caps_min: u32, caps_max: u32, mode: vk::PresentModeKHR) -> u32 {
    let mut count = caps_min.saturating_add(1);
    if mode == vk::PresentModeKHR::MAILBOX {
        count = count.max(MAILBOX_MIN_IMAGES);
    }
    if caps_max != 0 {
        count = count.min(caps_max);
    }
    count
}

/// Presentation-wait resources indexed by the image returned from acquire.
///
/// A submit fence only retires the rendering submission. It does not prove that
/// the presentation engine has finished waiting on that submission's semaphore,
/// so one semaphore reused for every frame can be signalled again while a prior
/// present still owns it. Reacquiring image N is the portable proof that the
/// prior present of image N has completed, which makes N the safe reuse key.
struct PerSwapchainImage<T> {
    slots: Vec<T>,
}

impl<T> PerSwapchainImage<T> {
    fn new(slots: Vec<T>) -> Self {
        Self { slots }
    }

    fn for_acquired(&self, image_index: u32) -> &T {
        &self.slots[image_index as usize]
    }

    fn drain(&mut self) -> std::vec::Drain<'_, T> {
        self.slots.drain(..)
    }
}

pub(crate) struct WindowPresenter {
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    desired_extent: vk::Extent2D,
    recreate_pending: bool,
    /// Why the next recreation was armed — carried into the always-on
    /// `host_window_swapchain` line so a live log separates guest/user resizes
    /// from suboptimal-surface self-heals.
    recreate_reason: &'static str,
    /// Consecutive presents whose acquire or present reported a suboptimal
    /// surface. Each one arms a recreation; see [`SUBOPTIMAL_ALARM_STREAK`].
    suboptimal_streak: u32,
    /// Reason the resident could not carry the run currently in progress,
    /// `None` while presenting a resident directly. A line is emitted when a
    /// run STARTS or its reason CHANGES, and a summary when it ends — so a
    /// window blank for a minute at 120 Hz costs two lines, not 7200.
    slate_reason: Option<SlateReason>,
    /// Consecutive presents in the current non-resident run.
    slate_run: u64,
    /// Whether the run in progress is being covered by CPU bytes. A covered run
    /// shows the guest's frame and only costs the host copy the resident rail
    /// exists to remove; an uncovered one is a blank window. They share a
    /// `SlateReason` and have completely different severities, so the run
    /// tracker carries which it is rather than reporting both as blank.
    slate_covered: bool,
    /// Host-visible staging for the CPU fallback source. Allocated on the first
    /// present that needs it and kept until the geometry changes, so a boot that
    /// never falls back never allocates it.
    staging: Option<StagingImage>,
    cmd_pool: vk::CommandPool,
    /// One render-complete semaphore per swapchain image. Reacquiring an image
    /// proves presentation has finished waiting on that image's semaphore;
    /// retiring a submit fence alone does not.
    render_finished: PerSwapchainImage<vk::Semaphore>,
    /// One entry per present that may be in flight at once, used round-robin.
    ///
    /// Every field in [`PresentFrame`] is per-present and none may be shared: a
    /// second present recording into the first's command buffer is a
    /// use-after-submit the validation layers this device does not run would be
    /// the only thing to catch. The render-finished semaphores are the deliberate
    /// exception above: presentation lifetime makes those per swapchain image.
    frames: Vec<PresentFrame>,
    /// Which entry the next present will use. Advances only on a successful
    /// submit, so a `Busy` return does not burn a slot.
    frame_ix: usize,
    cadence_started: Instant,
    cadence_presents: u64,
    cadence_direct: u64,
    cadence_busy: u64,
    /// Distinct frame sequences offered in the window, and the last one seen.
    ///
    /// `presents` alone cannot separate "the device published 20 frames this
    /// second" from "the device published 100 and the presenter could only show
    /// 20": a `Busy` return leaves the window's seq gate unchanged, so the same
    /// frame is re-offered every poll and `busy` counts retries, not frames.
    /// Offered-vs-presented is the ratio that says which side is the limit.
    cadence_offered: u64,
    cadence_last_offered: Option<u64>,
    /// `Busy` returns split by which of the two gates refused: the previous
    /// present's blit fence still running (`fence` — the engine queue is behind,
    /// since the blit is submitted to the same queue as every guest draw), or
    /// the swapchain having no free image (`acquire` — the display's own pacing).
    /// They have opposite fixes, and one `busy` count cannot tell them apart.
    cadence_busy_fence: u64,
    cadence_busy_acquire: u64,
    /// Zero-timeout acquires that invoked the bounded retry, and the subset for
    /// which that retry returned an image. An attempt may complete immediately;
    /// this is not a measurement of time spent asleep.
    cadence_acquire_retried: u64,
    cadence_acquire_rescued: u64,
}

/// Everything one in-flight present owns for as long as its blit is running.
struct PresentFrame {
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    /// Whether this entry's blit has been submitted and not yet retired.
    submitted: bool,
    /// Resident targets pinned for this present, released when its fence
    /// retires. Per entry because two in-flight presents may pin different
    /// surfaces and the earlier one's pins must not be dropped by the later
    /// one's retire.
    pinned: Vec<TargetIdentity>,
}

/// How many presents may be in flight at once.
///
/// # Why this is not 1
///
/// It was 1, and that made the presenter a hard ceiling rather than a pacer.
/// Twelve driven macos-13 boots across three builds put `presents` at 1599-1696
/// — a 5 % spread — while the device *published* 1760-2015 frames to it. Around
/// 15 % of every boot's frames were built and thrown away, `busy_acquire` 0
/// throughout, so the swapchain always had an image free and every refusal was
/// the previous blit's fence still running.
///
/// The blit shares a queue with every guest draw, so that fence retires behind
/// whatever guest work is queued rather than behind the copy itself — ~24 ms at
/// the observed rate, against a blit of one surface. That is latency, and depth
/// is what hides latency.
///
/// # Why the swapchain's floor and not more
///
/// A present past the image count cannot acquire an image, so it would refuse on
/// `acquire` rather than on the fence — trading a wait we can see for one that
/// reports as the display's pacing. Depth past what the swapchain serves buys
/// nothing and moves the evidence.
///
/// A surface that caps `max_image_count` below this leaves the last entry
/// unable to acquire. That is safe and self-limiting — it refuses as
/// `busy_acquire`, which is exactly the counter that says so — and it is why
/// this is a ceiling on ambition rather than a promise about the surface.
///
/// # It is transparent on every x86 rail, at every rate they offer
///
/// The depth was measured on macos-13 and shipped for all of them, so the other
/// rails were owed a boot each. One driven boot per rail, same binary:
///
/// ```text
/// rail       present_hz  offered_hz  busy_fence  busy_acquire  panic
/// macos-11        45.20       45.20           0             0  no
/// macos-12        47.20       47.20           0             0  no
/// macos-14        45.60       45.60           0             0  no
/// macos-15        14.45       14.45           0             0  no
/// macos-26        40.00       40.00           0             0  no
/// macos-26        21.05       21.05           0             0  no
/// macos-26        36.20       36.20           0             0  no
/// ```
///
/// `presents == offered` exactly on all seven boots, with both refusal counters
/// at zero. Two readings carry it. macos-15 offers **14 Hz**, a third of what
/// macos-13 does, and is equally transparent; and macos-26 was booted three
/// times, landing at 40, 21 and 36 Hz, and tracked its own offer each time. So
/// this is not a clamp that happens to sit above what these guests ask for — a
/// clamp shows as the two columns diverging at the top of the range, and nothing
/// here diverges at any rate between 14 and 47 Hz.
///
/// The macos-26 boots did not panic, which is worth stating precisely: that rail
/// panics on roughly a third of driven boots for reasons of its own, three clean
/// boots is an unremarkable draw from that rate, and this says nothing about
/// whether the rate moved. It is the presenter that was being measured.
const PRESENT_IN_FLIGHT: usize = MAILBOX_MIN_IMAGES as usize;

/// At least as deep as the single-flight presenter this replaced, so indexing
/// `frames` is always valid, and no deeper than the swapchain's own floor.
/// Both ends are relations against values derived elsewhere rather than
/// restatements of this line.
const _: () = assert!(PRESENT_IN_FLIGHT >= 1 && PRESENT_IN_FLIGHT <= MAILBOX_MIN_IMAGES as usize);

impl WindowPresenter {
    /// How deep to run, after the environment has had its say.
    ///
    /// `REIMS_VGPU_PRESENT_DEPTH=off` returns 1, which is exactly the
    /// single-flight presenter this replaced. It narrows — one present in flight
    /// is strictly less concurrency, never more — so it obeys the rule that a
    /// switch may only turn a rail off.
    fn present_depth() -> usize {
        static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *DEPTH.get_or_init(|| {
            let (state, value) = crate::env::read(crate::env::PRESENT_DEPTH);
            match state {
                crate::env::Switch::Off => {
                    crate::observe::off("present_depth reason=present_depth_disabled_by_env");
                    1
                }
                crate::env::Switch::Unrecognized => {
                    crate::observe::fail(format!(
                        "present_depth reason=present_depth_env_unrecognized value={}",
                        value.unwrap_or_default()
                    ));
                    PRESENT_IN_FLIGHT
                }
                crate::env::Switch::Unset | crate::env::Switch::On => PRESENT_IN_FLIGHT,
            }
        })
    }

    pub(crate) unsafe fn create(
        ctx: &DeviceContext,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, DrawError> {
        if !ctx.swapchain {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainUnavailable,
            ));
        }
        let surface = ash_window::create_surface(&ctx._entry, &ctx.instance, display, window, None)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowCreateSurface, error)))?;
        let surface_loader = ash::khr::surface::Instance::new(&ctx._entry, &ctx.instance);
        let present_capable = surface_loader
            .get_physical_device_surface_support(ctx.pd, ctx.gq, surface)
            .map_err(|error| {
                surface_loader.destroy_surface(surface, None);
                DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceSupport, error))
            })?;
        if !present_capable {
            surface_loader.destroy_surface(surface, None);
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::QueueCannotPresent {
                    queue_family: ctx.gq,
                },
            ));
        }

        let cmd_pool = match ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.gq)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateCommandPool,
                    error,
                )));
            }
        };
        let depth = Self::present_depth();
        let cmds = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(depth as u32),
        ) {
            Ok(buffers) => buffers,
            Err(error) => {
                ctx.device.destroy_command_pool(cmd_pool, None);
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowAllocCommandBuffer,
                    error,
                )));
            }
        };
        // One set of per-present objects per entry. Built in a loop that unwinds
        // everything already made on any failure, because a half-built presenter
        // is returned as an error and never dropped — nothing else would free
        // the entries that did succeed.
        let mut frames: Vec<PresentFrame> = Vec::with_capacity(cmds.len());
        let mut build = || -> Result<(), (VkOp, vk::Result)> {
            for &cmd in &cmds {
                let image_available = ctx
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| (VkOp::WindowCreateAcquireSemaphore, e))?;
                // Created signaled: the first present through each entry finds
                // it retired rather than waiting on a fence nothing submitted.
                let in_flight = match ctx.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                ) {
                    Ok(fence) => fence,
                    Err(error) => {
                        ctx.device.destroy_semaphore(image_available, None);
                        return Err((VkOp::WindowCreateFence, error));
                    }
                };
                frames.push(PresentFrame {
                    cmd,
                    image_available,
                    in_flight,
                    submitted: false,
                    pinned: Vec::new(),
                });
            }
            Ok(())
        };
        if let Err((op, error)) = build() {
            for frame in frames.drain(..) {
                ctx.device.destroy_fence(frame.in_flight, None);
                ctx.device.destroy_semaphore(frame.image_available, None);
            }
            ctx.device.destroy_command_pool(cmd_pool, None);
            surface_loader.destroy_surface(surface, None);
            return Err(DrawError::VkCall(VkCall::new(op, error)));
        }

        let mut presenter = Self {
            surface_loader,
            surface,
            swapchain_loader: ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device),
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            extent: vk::Extent2D::default(),
            desired_extent: vk::Extent2D {
                width: width.max(1),
                height: height.max(1),
            },
            recreate_pending: true,
            recreate_reason: "init",
            suboptimal_streak: 0,
            slate_reason: None,
            slate_run: 0,
            slate_covered: false,
            staging: None,
            cmd_pool,
            render_finished: PerSwapchainImage::new(Vec::new()),
            frames,
            frame_ix: 0,
            cadence_started: Instant::now(),
            cadence_presents: 0,
            cadence_direct: 0,
            cadence_busy: 0,
            cadence_offered: 0,
            cadence_last_offered: None,
            cadence_busy_fence: 0,
            cadence_busy_acquire: 0,
            cadence_acquire_retried: 0,
            cadence_acquire_rescued: 0,
        };
        if let Err(error) = presenter.recreate_swapchain(ctx) {
            presenter.destroy(ctx, None);
            return Err(error);
        }
        Ok(presenter)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let requested = vk::Extent2D {
            width: width.max(1),
            height: height.max(1),
        };
        if requested != self.desired_extent {
            self.recreate_pending = true;
            self.recreate_reason = "resize";
        }
        self.desired_extent = requested;
    }

    /// Release every entry whose blit has finished, and say whether the entry
    /// the next present would use is free.
    ///
    /// Sweeping all of them rather than only the next one matters for the pins:
    /// an entry that completed is holding resident targets off the reclaim path,
    /// and with several in flight the round-robin might not revisit it for
    /// another two presents. The return value is still about one entry, because
    /// that is the only one the caller is about to record into.
    unsafe fn retire(
        &mut self,
        ctx: &DeviceContext,
        pools: &mut ResourcePools,
    ) -> Result<bool, DrawError> {
        for ix in 0..self.frames.len() {
            if !self.frames[ix].submitted {
                continue;
            }
            let signaled = ctx
                .device
                .get_fence_status(self.frames[ix].in_flight)
                .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowFenceStatus, error)))?;
            if !signaled {
                continue;
            }
            let pinned = std::mem::take(&mut self.frames[ix].pinned);
            for identity in pinned {
                let _ = pools.pin_resident_target(&identity, false);
            }
            self.frames[ix].submitted = false;
        }
        Ok(!self.frames[self.frame_ix].submitted)
    }

    /// Block until every submitted entry's blit has finished.
    ///
    /// Only the CPU-fallback staging path needs this, and only because that one
    /// image is shared by every entry. The `submitted` latches are left alone:
    /// clearing them is [`Self::retire`]'s job because that is where the pins
    /// are released, and doing it in two places would let a pin outlive the
    /// entry that took it.
    ///
    /// A wait failure is reported and swallowed rather than propagated. The
    /// caller is already on a degraded path, and the honest options at that
    /// point are "present a stale frame" or "abort the whole draw chain over a
    /// fence" — the first is what a lost device is going to produce anyway.
    unsafe fn wait_for_in_flight(&mut self, ctx: &DeviceContext) {
        let fences: Vec<vk::Fence> = self
            .frames
            .iter()
            .filter(|frame| frame.submitted)
            .map(|frame| frame.in_flight)
            .collect();
        if fences.is_empty() {
            return;
        }
        if let Err(error) = ctx.device.wait_for_fences(&fences, true, u64::MAX) {
            let decline = VkCall::new(VkOp::WindowFenceStatus, error);
            crate::observe::Emit::decline("host_window_staging_wait", &decline).fail_once(0);
        }
    }

    unsafe fn recreate_swapchain(&mut self, ctx: &DeviceContext) -> Result<(), DrawError> {
        ctx.device
            .queue_wait_idle(ctx.queue())
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowQueueWaitIdle, error)))?;
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceCaps, error)))?;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainLacksTransferDst,
            ));
        }
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceFormats, error)))?;
        let format = formats
            .iter()
            .find(|format| {
                format.format == translate::pixel::SCANOUT_FORMAT
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainNoSurfaceFormat,
            ))?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: self
                    .desired_extent
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: self
                    .desired_extent
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        // MAILBOX where the surface offers it, FIFO where it does not.
        //
        // Acquisition probes with a zero timeout first, because the window
        // thread holds the shared engine lock here. An exhausted swapchain then
        // gets one guest frame budget to let WSI process image releases; it
        // never waits indefinitely behind the presentation engine or vblank.
        //
        // MAILBOX is the mode whose contract matches this consumer: the
        // presentation engine keeps one pending image and *replaces* it, so the
        // newest submitted frame is the one eventually displayed. The WSI may
        // still need the bounded acquire above to process image releases;
        // MAILBOX changes which queued frame survives, not whether the client
        // must dispatch those releases.
        //
        // Capability-gated with no vendor or driver test, and FIFO remains the
        // fallback because it is the only mode Vulkan guarantees. MAILBOX also
        // wants a third image to have something to replace, so the count floor
        // is raised only on that arm and only within `max_image_count`.
        let plan = swapchain_plan(
            caps.min_image_count,
            caps.max_image_count,
            self.surface_loader
                .get_physical_device_surface_present_modes(ctx.pd, self.surface)
                .as_deref()
                .unwrap_or(&[]),
        );
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|flag| caps.supported_composite_alpha.contains(*flag))
        .ok_or(DrawError::Unsupported(
            super::reason::DrawReason::SwapchainNoCompositeAlpha,
        ))?;
        // Destroy the old swapchain BEFORE creating its replacement, and create
        // the replacement without `old_swapchain`. MoltenVK (verified against
        // v1.4.1 MVKSwapchain.mm) works around a Metal present-callback
        // regression by setting the CAMetalLayer drawableSize to {1,1} when a
        // swapchain that still has 1-2 unpresented images is retired; with
        // `old_swapchain`, that clobber runs AFTER the new swapchain has
        // already configured the layer, and nothing restores the size — every
        // later present then succeeds (flagged suboptimal only) while the
        // window displays a single stretched pixel. Destroy-first makes the new
        // swapchain's layer configuration the final write, the ordering that
        // workaround assumes. The queue idle above retires rendering submissions;
        // unextended `vkQueuePresentKHR` has no completion fence, so final
        // presentation-resource teardown here retains Vulkan's conventional
        // wait-idle limitation. `VK_EXT_swapchain_maintenance1` would be needed
        // for a strict host-visible completion proof.
        let from = self.extent;
        if self.swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
            self.images.clear();
        }
        let swapchain = self
            .swapchain_loader
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(plan.image_count)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(composite_alpha)
                    .present_mode(plan.present_mode)
                    .clipped(true),
                None,
            )
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowCreateSwapchain, error)))?;
        let images = self
            .swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|error| {
                self.swapchain_loader.destroy_swapchain(swapchain, None);
                DrawError::VkCall(VkCall::new(VkOp::WindowGetSwapchainImages, error))
            })?;
        // Fresh per-recreation semaphores: an acquire whose submit later failed
        // leaves `image_available` with a signal nobody consumed, which is
        // invalid to reuse on the new swapchain's first acquire. Created before
        // the old set is destroyed so a failure leaves the presenter
        // consistent.
        // Every entry gets a fresh acquire semaphore, not just the one about to be used. The
        // queue idled above, so no entry has work outstanding — but an entry
        // whose acquire succeeded and whose submit then failed still holds an
        // unconsumed signal on its `image_available`, and that is invalid to
        // reuse against the new swapchain whichever entry it belongs to.
        let mut fresh_acquire = Vec::with_capacity(self.frames.len());
        let mut make = || -> Result<(), (VkOp, vk::Result)> {
            for _ in 0..self.frames.len() {
                let image_available = ctx
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| (VkOp::WindowCreateAcquireSemaphore, e))?;
                fresh_acquire.push(image_available);
            }
            Ok(())
        };
        if let Err((op, error)) = make() {
            for image_available in fresh_acquire {
                ctx.device.destroy_semaphore(image_available, None);
            }
            self.swapchain_loader.destroy_swapchain(swapchain, None);
            return Err(DrawError::VkCall(VkCall::new(op, error)));
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            match ctx
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            {
                Ok(semaphore) => render_finished.push(semaphore),
                Err(error) => {
                    for semaphore in render_finished.drain(..) {
                        ctx.device.destroy_semaphore(semaphore, None);
                    }
                    for image_available in fresh_acquire {
                        ctx.device.destroy_semaphore(image_available, None);
                    }
                    self.swapchain_loader.destroy_swapchain(swapchain, None);
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::WindowCreateRenderSemaphore,
                        error,
                    )));
                }
            }
        }
        for semaphore in self.render_finished.drain() {
            ctx.device.destroy_semaphore(semaphore, None);
        }
        self.render_finished = PerSwapchainImage::new(render_finished);
        for (frame, image_available) in self.frames.iter_mut().zip(fresh_acquire) {
            ctx.device.destroy_semaphore(frame.image_available, None);
            frame.image_available = image_available;
            // The queue idled, so nothing is outstanding regardless of what the
            // latch said before.
            frame.submitted = false;
        }
        self.swapchain = swapchain;
        self.images = images;
        self.extent = extent;
        self.desired_extent = extent;
        self.recreate_pending = false;
        if extent != from {
            // A geometry change is progress; only a same-extent suboptimal
            // loop should keep accumulating toward the alarm.
            self.suboptimal_streak = 0;
        }
        crate::observe::off(swapchain_recreated_line(
            from,
            extent,
            self.recreate_reason,
            plan.present_mode,
            self.images.len(),
        ));
        Ok(())
    }

    pub(crate) unsafe fn present(
        &mut self,
        ctx: &DeviceContext,
        pools: &mut ResourcePools,
        counters: &EngineCounters,
        source: Option<&WindowPresentSource>,
        cpu: Option<WindowCpuFrame<'_>>,
    ) -> Result<WindowPresentOutcome, DrawError> {
        if let Some(seq) = cpu.map(|frame| frame.seq) {
            if self.cadence_last_offered != Some(seq) {
                self.cadence_last_offered = Some(seq);
                self.cadence_offered = self.cadence_offered.saturating_add(1);
            }
        }
        if !self.retire(ctx, pools)? {
            self.cadence_busy_fence = self.cadence_busy_fence.saturating_add(1);
            self.note_cadence(false, false);
            return Ok(WindowPresentOutcome::Busy);
        }
        if self.swapchain == vk::SwapchainKHR::null() || self.recreate_pending {
            self.recreate_swapchain(ctx)?;
        }
        // Bound after any swapchain recreation above, which resets every latch.
        let frame_ix = self.frame_ix;
        let frame_cmd = self.frames[frame_ix].cmd;
        let frame_image_available = self.frames[frame_ix].image_available;
        let frame_in_flight = self.frames[frame_ix].in_flight;
        let (acquire, retry) = acquire_with_bounded_retry(|timeout| {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                timeout,
                frame_image_available,
                vk::Fence::null(),
            )
        });
        self.cadence_acquire_retried = self
            .cadence_acquire_retried
            .saturating_add(u64::from(retry.attempted));
        self.cadence_acquire_rescued = self
            .cadence_acquire_rescued
            .saturating_add(u64::from(retry.rescued));
        let (image_index, acquire_suboptimal) = match acquire {
            Ok((index, suboptimal)) => (index, suboptimal),
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                self.cadence_busy_acquire = self.cadence_busy_acquire.saturating_add(1);
                self.note_cadence(false, false);
                return Ok(WindowPresentOutcome::Busy);
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "acquire_out_of_date";
                self.cadence_busy_acquire = self.cadence_busy_acquire.saturating_add(1);
                self.note_cadence(false, false);
                return Ok(WindowPresentOutcome::Busy);
            }
            Err(error) => {
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowAcquireImage,
                    error,
                )));
            }
        };

        pools.batch_flush(ctx, counters)?;
        let selected = source.and_then(|source| {
            let slot = pools.registry_get(&source.identity)?;
            super::pools::slot_presentable(slot, source.width, source.height).then(|| {
                (
                    source.identity.clone(),
                    slot.image,
                    slot.access,
                    slot.width,
                    slot.height,
                )
            })
        });
        // Only reached when no resident carries this present: upload the CPU
        // bytes instead. `None` here means the window shows slate.
        let staged = if selected.is_some() {
            self.note_slate_end();
            None
        } else {
            // Failure path only: re-read the slot to name WHY the resident could
            // not carry. Cheap because it never runs on a good frame.
            let state = source
                .and_then(|source| pools.registry_get(&source.identity))
                .map_or(CandidateState::default(), |slot| CandidateState {
                    resident: true,
                    content_ready: slot.content_ready,
                    bgra: slot.scanout_order(),
                    width: slot.width,
                    height: slot.height,
                });
            let want = source.map_or((0, 0), |s| (s.width, s.height));
            let reason = classify_slate(source.is_some(), want, state);
            let staged = cpu
                .filter(cpu_frame_complete)
                .and_then(|frame| self.stage_cpu_frame(ctx, frame));
            self.note_slate(reason, want, state, staged.is_some());
            staged
        };
        let mut pinned = Vec::with_capacity(1);
        if let Some((identity, _, _, _, _)) = selected.as_ref() {
            if !pools.pin_resident_target(identity, true) {
                return Err(DrawError::Facade(
                    EngineFacadeDecline::WindowSourceDisappearedBeforePin {
                        identity: identity.clone(),
                    },
                ));
            }
            pinned.push(identity.clone());
        }

        // One blit body for both sources: they differ only in which image is
        // read and how it is made readable. Keeping them separate is how the
        // aspect-fit and letterbox-clear rules drift apart between the two
        // rails.
        let blit = selected
            .as_ref()
            .map(
                |(_, image, access, base_width, base_height)| BlitSource::Resident {
                    image: *image,
                    access: *access,
                    width: *base_width,
                    height: *base_height,
                },
            )
            .or(staged);

        let submit_result = (|| {
            ctx.device
                .reset_fences(&[frame_in_flight])
                .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowResetFence, error)))?;
            ctx.device
                .reset_command_buffer(frame_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|error| {
                    DrawError::VkCall(VkCall::new(VkOp::WindowResetCommandBuffer, error))
                })?;
            ctx.device
                .begin_command_buffer(
                    frame_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| {
                    DrawError::VkCall(VkCall::new(VkOp::WindowBeginCommandBuffer, error))
                })?;

            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let dst = self.images[image_index as usize];
            image_barrier(
                &ctx.device,
                frame_cmd,
                dst,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            );
            if let Some(blit) = blit {
                let (base_width, base_height) = blit.extent();
                // Aspect-fit placement: the guest frame keeps its aspect ratio
                // inside whatever drawable exists right now (a guest-driven
                // native resize normally makes this the full window within
                // milliseconds). The window input path maps pointer positions
                // through this same transform.
                let vp = crate::host_window::viewport::aspect_fit(
                    (base_width, base_height),
                    (self.extent.width, self.extent.height),
                );
                if !vp.covers((self.extent.width, self.extent.height)) {
                    // Letterbox bars: clear the whole image first so stale
                    // swapchain pixels never frame the guest content.
                    ctx.device.cmd_clear_color_image(
                        frame_cmd,
                        dst,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue {
                            float32: SLATE_CLEAR,
                        },
                        &[color_range],
                    );
                }
                let src_layout = blit.record_read_barrier(&ctx.device, frame_cmd);
                blit_rect(
                    &ctx.device,
                    frame_cmd,
                    blit.image(),
                    dst,
                    src_layout,
                    (0, 0, base_width, base_height),
                    (vp.x, vp.y, vp.x + vp.width, vp.y + vp.height),
                );
                if let Some((identity, _, _, _, _)) = selected.as_ref() {
                    pools
                        .registry_note_access(identity, super::pools::ResidentAccess::TransferRead);
                }
            } else {
                ctx.device.cmd_clear_color_image(
                    frame_cmd,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: SLATE_CLEAR,
                    },
                    &[color_range],
                );
            }
            image_barrier(
                &ctx.device,
                frame_cmd,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            );
            ctx.device.end_command_buffer(frame_cmd).map_err(|error| {
                DrawError::VkCall(VkCall::new(VkOp::WindowEndCommandBuffer, error))
            })?;
            let waits = [frame_image_available];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let render_finished = *self.render_finished.for_acquired(image_index);
            let signals = [render_finished];
            let commands = [frame_cmd];
            ctx.device
                .queue_submit(
                    ctx.queue(),
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&waits)
                        .wait_dst_stage_mask(&wait_stages)
                        .command_buffers(&commands)
                        .signal_semaphores(&signals)],
                    frame_in_flight,
                )
                .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSubmitPresent, error)))
        })();
        if let Err(error) = submit_result {
            for identity in pinned.drain(..) {
                let _ = pools.pin_resident_target(&identity, false);
            }
            return Err(error);
        }
        self.frames[frame_ix].pinned = pinned;
        self.frames[frame_ix].submitted = true;
        // Only a successful submit advances the ring; a `Busy` return above
        // leaves the slot for the next attempt.
        self.frame_ix = (frame_ix + 1) % self.frames.len();
        if matches!(blit, Some(BlitSource::Staged { .. })) {
            // The barrier that leaves the staging image in GENERAL is now
            // queued. Recorded only after the submit succeeds: a failed submit
            // never executes it, and declaring GENERAL as the old layout of an
            // image still in PREINITIALIZED discards the frame it holds.
            if let Some(staging) = self.staging.as_mut() {
                staging.transitioned = true;
            }
        }

        let swapchains = [self.swapchain];
        let indices = [image_index];
        let waits = [*self.render_finished.for_acquired(image_index)];
        match self.swapchain_loader.queue_present(
            ctx.queue(),
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&swapchains)
                .image_indices(&indices),
        ) {
            Ok(present_suboptimal) => {
                // ash reports VK_SUBOPTIMAL_KHR as `Ok(true)` (a success code),
                // never through the `Err` arm. MoltenVK returns it from both
                // acquire and present for as long as the CAMetalLayer's
                // drawable or natural size diverges from the swapchain extent —
                // including after a retired swapchain clobbered the layer's
                // drawableSize — so ignoring the flag leaves an invisible
                // window that still counts successful presents.
                let suboptimal = acquire_suboptimal || present_suboptimal;
                if suboptimal {
                    self.recreate_pending = true;
                    self.recreate_reason = "suboptimal";
                    self.suboptimal_streak = self.suboptimal_streak.saturating_add(1);
                    if self.suboptimal_streak == SUBOPTIMAL_ALARM_STREAK {
                        let decline = WindowPresentDecline::SuboptimalPersistent {
                            streak: self.suboptimal_streak,
                            width: self.extent.width,
                            height: self.extent.height,
                        };
                        crate::observe::Emit::decline("host_window_present", &decline).fail();
                    }
                } else {
                    self.suboptimal_streak = 0;
                }
                let direct = selected.is_some();
                self.note_cadence(true, direct);
                Ok(WindowPresentOutcome::Presented {
                    direct,
                    width: self.extent.width,
                    height: self.extent.height,
                    swapchain_images: self.images.len(),
                    suboptimal,
                })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "present_out_of_date";
                self.note_cadence(false, false);
                Ok(WindowPresentOutcome::Busy)
            }
            Err(error) => Err(DrawError::VkCall(VkCall::new(
                VkOp::WindowQueuePresent,
                error,
            ))),
        }
    }

    /// Copy a published CPU frame into the staging image and describe it as a
    /// blit source. `None` means the staging image could not be provided, and
    /// the present falls through to slate with the reason already named.
    ///
    /// Seq-gated: a forced redraw (resize, suboptimal self-heal) re-blits the
    /// bytes already staged rather than copying a full frame again.
    unsafe fn stage_cpu_frame(
        &mut self,
        ctx: &DeviceContext,
        frame: WindowCpuFrame<'_>,
    ) -> Option<BlitSource> {
        // The staging image is **one allocation shared by every entry**, written
        // here by the CPU and read by the blit the caller is about to record. So
        // before touching it, drain any other present still in flight: with a
        // depth of one that was free, and `ensure_staging`'s doc relies on it
        // (a geometry change destroys the previous image, which is only safe
        // while nothing queued still reads it).
        //
        // Waiting rather than refusing, because refusing here would mean
        // returning `Busy` after the swapchain image is already acquired, and
        // that leaves an unconsumed signal on `image_available` — the exact
        // state `recreate_swapchain` documents as invalid to reuse. Waiting is
        // affordable precisely here: this is the failure path taken when no
        // resident can carry the present, and it never runs on a good frame.
        self.wait_for_in_flight(ctx);
        if let Err(error) = self.ensure_staging(ctx, frame.width, frame.height) {
            // A host that cannot allocate staging cannot allocate it next frame
            // either, so this latches to one line per boot rather than one per
            // present.
            crate::observe::Emit::decline("host_window_staging", &error).fail_once(0);
            return None;
        }
        let staging = self.staging.as_mut()?;
        if staging.staged_seq != Some(frame.seq) {
            // Row by row: the driver is free to pad a LINEAR image's rows, and
            // copying tightly into a padded image shears the picture.
            let src_row = frame.width as usize * 4;
            for y in 0..frame.height as usize {
                let dst = staging
                    .mapped
                    .0
                    .add(staging.offset as usize + y * staging.row_pitch as usize);
                std::ptr::copy_nonoverlapping(frame.bgra.as_ptr().add(y * src_row), dst, src_row);
            }
            staging.staged_seq = Some(frame.seq);
        }
        Some(BlitSource::Staged {
            image: staging.image,
            first_use: !staging.transitioned,
            width: staging.width,
            height: staging.height,
        })
    }

    /// Provide a host-visible LINEAR staging image at exactly `width`x`height`.
    ///
    /// A geometry change destroys the previous one, which is safe here because
    /// [`Self::present`] retires the in-flight fence before reaching this point,
    /// so no queued blit still reads it.
    unsafe fn ensure_staging(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
    ) -> Result<(), StagingError> {
        if self
            .staging
            .as_ref()
            .is_some_and(|s| s.width == width && s.height == height)
        {
            return Ok(());
        }
        if let Some(old) = self.staging.take() {
            old.destroy(&ctx.device);
        }
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(translate::pixel::SCANOUT_FORMAT)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                    .initial_layout(vk::ImageLayout::PREINITIALIZED),
                None,
            )
            .map_err(|result| {
                StagingError::Call(VkCall::new(VkOp::WindowCreateStagingImage, result))
            })?;
        let req = ctx.device.get_image_memory_requirements(image);
        let Some(mem_type) = ctx.memory_type_for(
            req.memory_type_bits,
            req.size,
            crate::backend::vulkan::caps::MemoryClass::Upload,
        ) else {
            ctx.device.destroy_image(image, None);
            return Err(StagingError::NoUploadMemoryType {
                type_bits: req.memory_type_bits,
            });
        };
        let memory = match ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        ) {
            Ok(memory) => memory,
            Err(result) => {
                ctx.device.destroy_image(image, None);
                return Err(StagingError::Call(VkCall::new(
                    VkOp::WindowAllocateStagingMemory,
                    result,
                )));
            }
        };
        if let Err(result) = ctx.device.bind_image_memory(image, memory, 0) {
            ctx.device.destroy_image(image, None);
            ctx.device.free_memory(memory, None);
            return Err(StagingError::Call(VkCall::new(
                VkOp::WindowBindStagingMemory,
                result,
            )));
        }
        let layout = ctx.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        );
        let mapped = match ctx
            .device
            .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
        {
            Ok(pointer) => MappedStaging(pointer as *mut u8),
            Err(result) => {
                ctx.device.destroy_image(image, None);
                ctx.device.free_memory(memory, None);
                return Err(StagingError::Call(VkCall::new(
                    VkOp::WindowMapStagingMemory,
                    result,
                )));
            }
        };
        self.staging = Some(StagingImage {
            image,
            memory,
            mapped,
            width,
            height,
            row_pitch: layout.row_pitch,
            offset: layout.offset,
            transitioned: false,
            staged_seq: None,
        });
        Ok(())
    }

    /// Record a present that no resident carried. Emits a line when a run
    /// starts or its reason changes; silent for every repeat within a run.
    ///
    /// `covered` splits two very different outcomes that share a reason: the
    /// window showing the guest's frame from CPU bytes (correct, and only as
    /// expensive as the host copy this rail exists to remove — a census line),
    /// and the window showing nothing at all (a visible loss — a failure line).
    fn note_slate(
        &mut self,
        reason: SlateReason,
        want: (u32, u32),
        state: CandidateState,
        covered: bool,
    ) {
        if self.slate_reason == Some(reason) && self.slate_covered == covered {
            self.slate_run = self.slate_run.saturating_add(1);
            return;
        }
        if self.slate_reason.is_some() {
            self.note_slate_end();
        }
        self.slate_reason = Some(reason);
        self.slate_covered = covered;
        self.slate_run = 1;
        let seen = if state.resident {
            format!(
                "{}x{}/{}{}",
                state.width, state.height, state.content_ready as u8, state.bgra as u8
            )
        } else {
            "absent".to_string()
        };
        let emit = crate::observe::Emit::decline(
            if covered {
                "host_window_cpu_fallback"
            } else {
                "host_window_slate"
            },
            &reason,
        )
        .field("want", format!("{}x{}", want.0, want.1))
        .field("seen", seen);
        if covered {
            // The guest's frame IS on screen; what was lost is the direct
            // handoff, which costs host copies rather than pixels. Expected for
            // the whole of firmware boot, so a failure line here would cry wolf
            // on every run.
            emit.off();
        } else {
            emit.fail();
        }
    }

    /// Close an in-progress non-resident run, reporting how long it lasted.
    fn note_slate_end(&mut self) {
        let Some(reason) = self.slate_reason.take() else {
            return;
        };
        // `off()`, not `fail()`: the run *ending* is the window recovering, so
        // it is a census line rather than a drop, per the curated-fail rule.
        crate::observe::Emit::decline("host_window_slate_end", &reason)
            .field("frames", self.slate_run)
            .field("covered", u8::from(self.slate_covered))
            .off();
        self.slate_run = 0;
        self.slate_covered = false;
    }

    fn note_cadence(&mut self, presented: bool, direct: bool) {
        if presented {
            self.cadence_presents = self.cadence_presents.saturating_add(1);
            self.cadence_direct = self.cadence_direct.saturating_add(u64::from(direct));
        } else {
            self.cadence_busy = self.cadence_busy.saturating_add(1);
        }
        let elapsed = self.cadence_started.elapsed();
        if elapsed.as_millis() < 1_000 {
            return;
        }
        crate::observe::off(window_cadence_line(
            elapsed.as_millis() as u64,
            self.cadence_presents,
            self.cadence_direct,
            CadenceBusy {
                total: self.cadence_busy,
                fence: self.cadence_busy_fence,
                acquire: self.cadence_busy_acquire,
                acquire_retried: self.cadence_acquire_retried,
                acquire_rescued: self.cadence_acquire_rescued,
            },
            self.cadence_offered,
        ));
        self.cadence_started = Instant::now();
        self.cadence_presents = 0;
        self.cadence_direct = 0;
        self.cadence_busy = 0;
        self.cadence_offered = 0;
        self.cadence_busy_fence = 0;
        self.cadence_busy_acquire = 0;
        self.cadence_acquire_retried = 0;
        self.cadence_acquire_rescued = 0;
    }

    pub(crate) fn release_pins_after_idle(&mut self, pools: &mut ResourcePools) {
        for frame in &mut self.frames {
            for identity in frame.pinned.drain(..) {
                let _ = pools.pin_resident_target(&identity, false);
            }
            frame.submitted = false;
        }
    }

    pub(crate) unsafe fn destroy(
        &mut self,
        ctx: &DeviceContext,
        pools: Option<&mut ResourcePools>,
    ) {
        if let Err(error) = ctx.device.queue_wait_idle(ctx.queue()) {
            let decline = VkCall::new(VkOp::WindowDestroyQueueWaitIdle, error);
            crate::observe::Emit::decline("host_window_destroy", &decline).fail_once(0);
        }
        if let Some(pools) = pools {
            for frame in &mut self.frames {
                for identity in frame.pinned.drain(..) {
                    let _ = pools.pin_resident_target(&identity, false);
                }
            }
        } else {
            for frame in &mut self.frames {
                frame.pinned.clear();
            }
        }
        if let Some(staging) = self.staging.take() {
            staging.destroy(&ctx.device);
        }
        for semaphore in self.render_finished.drain() {
            ctx.device.destroy_semaphore(semaphore, None);
        }
        // Drained rather than iterated, so a second `destroy` — `create` calls
        // it on a failed `recreate_swapchain`, and the caller may call it again
        // — cannot double-free a handle.
        for frame in self.frames.drain(..) {
            ctx.device.destroy_fence(frame.in_flight, None);
            ctx.device.destroy_semaphore(frame.image_available, None);
        }
        ctx.device.destroy_command_pool(self.cmd_pool, None);
        if self.swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
        }
        self.surface_loader.destroy_surface(self.surface, None);
    }
}

fn swapchain_recreated_line(
    from: vk::Extent2D,
    to: vk::Extent2D,
    reason: &str,
    mode: vk::PresentModeKHR,
    images: usize,
) -> String {
    // Without these a `busy_acquire` rate is uninterpretable: the same number
    // means "the display is pacing us" under FIFO and "we are out of images"
    // under MAILBOX, and those have different fixes.
    //
    // `images` is what `vkGetSwapchainImagesKHR` returned. `mode` is the one the
    // create info was given — which `vkCreateSwapchainKHR` either honours or
    // fails on, so there is no third answer to report. It comes from the same
    // [`SwapchainPlan`] the create info reads, because when the two were spelled
    // separately this line printed `present_mode=mailbox` for a swapchain
    // created FIFO, and a whole session's measurement was read against it.
    let mode = match mode {
        vk::PresentModeKHR::MAILBOX => "mailbox",
        vk::PresentModeKHR::FIFO => "fifo",
        vk::PresentModeKHR::FIFO_RELAXED => "fifo_relaxed",
        vk::PresentModeKHR::IMMEDIATE => "immediate",
        _ => "other",
    };
    format!(
        "host_window_swapchain status=recreated from={}x{} to={}x{} trigger={reason} \
         present_mode={mode} images={images}",
        from.width, from.height, to.width, to.height
    )
}

/// The two gates that can refuse a present, kept apart because they have
/// opposite fixes: `fence` is the engine queue still running the previous blit
/// behind however much guest work was submitted ahead of it, `acquire` is the
/// swapchain having no free image, which is the display's own pacing.
struct CadenceBusy {
    total: u64,
    fence: u64,
    acquire: u64,
    acquire_retried: u64,
    acquire_rescued: u64,
}

fn window_cadence_line(
    window_ms: u64,
    presents: u64,
    direct: u64,
    busy: CadenceBusy,
    offered: u64,
) -> String {
    let hz = presents as f64 * 1_000.0 / window_ms.max(1) as f64;
    let direct_fraction = direct as f64 / presents.max(1) as f64;
    let offered_hz = offered as f64 * 1_000.0 / window_ms.max(1) as f64;
    format!(
        "host_window_cadence window_ms={window_ms} presents={presents} direct={direct} \
         busy={} busy_fence={} busy_acquire={} acquire_retried={} acquire_rescued={} \
         offered={offered} present_hz={hz:.1} \
         offered_hz={offered_hz:.1} direct_frac={direct_fraction:.2}",
        busy.total, busy.fence, busy.acquire, busy.acquire_retried, busy.acquire_rescued
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the helper mirrors the complete Vulkan image barrier state"
)]
unsafe fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    device.cmd_pipeline_barrier(
        cmd,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)],
    );
}

unsafe fn blit_rect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_layout: vk::ImageLayout,
    src_rect: PresentRect,
    dst_rect: PresentRect,
) {
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    device.cmd_blit_image(
        cmd,
        src,
        src_layout,
        dst,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D {
                    x: src_rect.0 as i32,
                    y: src_rect.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: src_rect.2 as i32,
                    y: src_rect.3 as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D {
                    x: dst_rect.0 as i32,
                    y: dst_rect.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: dst_rect.2 as i32,
                    y: dst_rect.3 as i32,
                    z: 1,
                },
            ])],
        crate::backend::vulkan::translate::sampler::PRESENT_BLIT_FILTER,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapchain_recreation_line_names_geometry_and_reason() {
        let from = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let to = vk::Extent2D {
            width: 1440,
            height: 1080,
        };
        assert_eq!(
            swapchain_recreated_line(from, to, "resize", vk::PresentModeKHR::MAILBOX, 3),
            "host_window_swapchain status=recreated from=1920x1080 to=1440x1080 \
             trigger=resize present_mode=mailbox images=3"
        );
        // The granted mode, not the requested one — a surface that refuses
        // MAILBOX must be visible as FIFO in the log, or a `busy_acquire` rate
        // gets read against the wrong contract.
        assert!(
            swapchain_recreated_line(from, to, "init", vk::PresentModeKHR::FIFO, 2)
                .contains("present_mode=fifo images=2")
        );
    }

    /// The staging upload copies `height` rows of `width * 4` bytes out of the
    /// published buffer. A buffer short of that would read whatever the staging
    /// image held below the copied rows — the previous frame, at whatever
    /// geometry it had — and blit the result as though it were current.
    ///
    /// The short case is not hypothetical: every present the device elides the
    /// readback for publishes an EMPTY buffer, because the resident is carrying
    /// that frame. Those arrive here whenever the resident then turns out not to
    /// be presentable, which is exactly when the fallback runs.
    #[test]
    fn a_cpu_frame_shorter_than_its_own_geometry_is_refused() {
        let full = vec![0u8; 8 * 4 * 4];
        assert!(cpu_frame_complete(&WindowCpuFrame {
            bgra: &full,
            width: 8,
            height: 4,
            seq: 1,
        }));
        // Slop is fine — the copy reads exactly what the geometry names.
        assert!(cpu_frame_complete(&WindowCpuFrame {
            bgra: &full,
            width: 8,
            height: 3,
            seq: 1,
        }));
        assert!(
            !cpu_frame_complete(&WindowCpuFrame {
                bgra: &full[..full.len() - 1],
                width: 8,
                height: 4,
                seq: 1,
            }),
            "one byte short is still a torn last row"
        );
        assert!(
            !cpu_frame_complete(&WindowCpuFrame {
                bgra: &[],
                width: 8,
                height: 4,
                seq: 1,
            }),
            "the elided-readback publish carries no bytes at all"
        );
        assert!(
            !cpu_frame_complete(&WindowCpuFrame {
                bgra: &full,
                width: 0,
                height: 4,
                seq: 1,
            }),
            "a zero dimension names no pixels and blits nothing"
        );
    }

    #[test]
    fn cadence_proxy_reports_actual_queue_presents_and_direct_fraction() {
        let line = window_cadence_line(
            1_000,
            120,
            119,
            CadenceBusy {
                total: 131,
                fence: 100,
                acquire: 31,
                acquire_retried: 37,
                acquire_rescued: 6,
            },
            240,
        );
        assert!(line.contains("presents=120"), "{line}");
        assert!(line.contains("direct=119"), "{line}");
        assert!(line.contains("busy=131"), "{line}");
        assert!(line.contains("busy_fence=100"), "{line}");
        assert!(line.contains("busy_acquire=31"), "{line}");
        assert!(line.contains("acquire_retried=37"), "{line}");
        assert!(line.contains("acquire_rescued=6"), "{line}");
        assert!(line.contains("present_hz=120.0"), "{line}");
        assert!(line.contains("direct_frac=0.99"), "{line}");
    }

    /// `offered` is the denominator `presents` needs. A window that presents 20
    /// frames is healthy if 20 were published and a 6x drop if 120 were, and
    /// `busy` cannot tell them apart — a `Busy` return leaves the window's seq
    /// gate unchanged, so one frame is re-offered every poll and `busy` counts
    /// retries.
    #[test]
    fn the_cadence_line_carries_the_rate_frames_were_offered_at() {
        let line = window_cadence_line(
            1_000,
            20,
            20,
            CadenceBusy {
                total: 420,
                fence: 400,
                acquire: 20,
                acquire_retried: 29,
                acquire_rescued: 9,
            },
            109,
        );
        assert!(line.contains("offered=109"), "{line}");
        assert!(line.contains("offered_hz=109.0"), "{line}");
        assert!(line.contains("present_hz=20.0"), "{line}");
    }

    fn ready(width: u32, height: u32) -> CandidateState {
        CandidateState {
            resident: true,
            content_ready: true,
            bgra: true,
            width,
            height,
        }
    }

    /// No published source is the expected pre-boundary / idle case and must be
    /// distinguishable from a source whose residents are missing.
    #[test]
    fn slate_without_a_source_is_named_separately() {
        assert_eq!(
            classify_slate(false, (0, 0), CandidateState::default()),
            SlateReason::NoSource
        );
        assert_eq!(
            classify_slate(true, (1440, 1080), CandidateState::default()),
            SlateReason::NoResident
        );
    }

    /// A resident that exists but has not landed content yet is the boot-era
    /// case; it must not be reported as a missing resident.
    #[test]
    fn unready_resident_reports_content_not_ready() {
        let pending = CandidateState {
            resident: true,
            content_ready: false,
            bgra: true,
            width: 1440,
            height: 1080,
        };
        assert_eq!(
            classify_slate(true, (1440, 1080), pending),
            SlateReason::ContentNotReady
        );
    }

    /// A resident that is ready and BGRA but the wrong size is the geometry
    /// class — the actionable fact is the size, and it must not be folded into
    /// `ContentNotReady`, whose remedy (wait a frame) would never converge.
    #[test]
    fn a_ready_resident_at_the_wrong_size_is_the_geometry_class() {
        assert_eq!(
            classify_slate(true, (1440, 1080), ready(1920, 1080)),
            SlateReason::GeomMismatch
        );
    }

    /// A ready non-BGRA resident is its own class — the present blit does no
    /// format conversion, so collapsing it into content_not_ready would send a
    /// reader hunting the wrong bug.
    #[test]
    fn non_bgra_resident_is_its_own_reason() {
        let state = CandidateState {
            resident: true,
            content_ready: true,
            bgra: false,
            width: 1440,
            height: 1080,
        };
        assert_eq!(
            classify_slate(true, (1440, 1080), state),
            SlateReason::NotBgra
        );
    }

    /// Every reason has a distinct, `slate_`-prefixed slug.
    ///
    /// What the prefix buys beyond distinctness is keeping a grep for this
    /// window's blit choice from also matching the capture rail's `no_resident_content`
    /// and the `THRASH geom_mismatch` proxy.
    #[test]
    fn slate_reason_slugs_are_distinct_and_namespaced() {
        use crate::observe::Decline;
        let mut slugs = [
            SlateReason::NoSource,
            SlateReason::NoResident,
            SlateReason::ContentNotReady,
            SlateReason::NotBgra,
            SlateReason::GeomMismatch,
        ]
        .map(|r| r.slug());
        for s in slugs {
            assert!(s.starts_with("slate_"), "{s} is not namespaced");
        }
        slugs.sort_unstable();
        let unique = slugs.len();
        let mut dedup = slugs.to_vec();
        dedup.dedup();
        assert_eq!(dedup.len(), unique);
    }

    #[test]
    fn non_aborting_present_degradations_keep_exact_geometry() {
        use crate::observe::Decline as _;
        let suboptimal = WindowPresentDecline::SuboptimalPersistent {
            streak: 60,
            width: 1440,
            height: 1080,
        };
        assert_eq!(suboptimal.slug(), "window_present_suboptimal_persistent");
        assert_eq!(
            suboptimal.fields(),
            vec![
                ("streak", "60".into()),
                ("width", "1440".into()),
                ("height", "1080".into()),
            ]
        );
        assert_eq!(
            crate::observe::Emit::decline("host_window_present", &suboptimal).render(),
            "host_window_present reason=window_present_suboptimal_persistent \
             streak=60 width=1440 height=1080"
        );
    }

    /// A resident that clears every blocker still has to come back with *some*
    /// reason, because the classifier only runs after `slot_presentable` already
    /// refused. The two disagreeing is a defect in one of them, and the residual
    /// class is what makes it visible instead of a panic or a sixth variant that
    /// nothing else ever reads.
    #[test]
    fn a_resident_that_clears_every_blocker_falls_to_the_residual_class() {
        assert_eq!(
            classify_slate(true, (1440, 1080), ready(1440, 1080)),
            SlateReason::ContentNotReady
        );
    }

    /// A submit fence says the blit finished, but it does not retire the wait
    /// performed by `vkQueuePresentKHR`. Only reacquiring the same swapchain
    /// image proves that present is done with its wait semaphore, so the
    /// resource follows the acquired image rather than a frame counter.
    #[test]
    fn a_present_wait_semaphore_is_reused_only_with_its_swapchain_image() {
        let per_image = PerSwapchainImage::new(vec!["image-0", "image-1", "image-2"]);
        let acquired = [0, 1, 2, 1, 0];
        let selected: Vec<_> = acquired
            .into_iter()
            .map(|index| *per_image.for_acquired(index))
            .collect();

        assert_eq!(
            selected,
            ["image-0", "image-1", "image-2", "image-1", "image-0"]
        );
    }

    /// An exhausted swapchain must enter the WSI wait/progress path, but the
    /// engine lock may not be held indefinitely. The policy retries only
    /// NOT_READY/TIMEOUT, retries exactly once, and reports whether it recovered.
    #[test]
    fn an_exhausted_acquire_gets_one_bounded_retry_to_dispatch_releases() {
        assert_eq!(guest_frame_budget_ns(0), None);
        assert_eq!(guest_frame_budget_ns(1), Some(NANOS_PER_SECOND));
        assert_eq!(
            guest_frame_budget_ns(crate::model::DISPLAY_REFRESH_HZ),
            Some(ACQUIRE_RETRY_BUDGET_NS)
        );

        let mut calls = Vec::new();
        let mut answers = [Err(vk::Result::NOT_READY), Ok(7)].into_iter();
        let (result, retry) = acquire_with_bounded_retry(|timeout| {
            calls.push(timeout);
            answers.next().expect("policy made more than two attempts")
        });
        assert_eq!(calls, [0, ACQUIRE_RETRY_BUDGET_NS]);
        assert_eq!(result, Ok(7));
        assert_eq!(
            retry,
            AcquireRetry {
                attempted: true,
                rescued: true
            }
        );

        let mut calls = Vec::new();
        let mut answers = [
            Err::<(), _>(vk::Result::TIMEOUT),
            Err::<(), _>(vk::Result::NOT_READY),
        ]
        .into_iter();
        let (result, retry) = acquire_with_bounded_retry(|timeout| {
            calls.push(timeout);
            answers.next().expect("policy made more than two attempts")
        });
        assert_eq!(calls, [0, ACQUIRE_RETRY_BUDGET_NS]);
        assert_eq!(result, Err(vk::Result::NOT_READY));
        assert_eq!(
            retry,
            AcquireRetry {
                attempted: true,
                rescued: false
            }
        );

        let mut calls = Vec::new();
        let (result, retry) = acquire_with_bounded_retry(|timeout| {
            calls.push(timeout);
            Err::<(), _>(vk::Result::ERROR_OUT_OF_DATE_KHR)
        });
        assert_eq!(calls, [0]);
        assert_eq!(result, Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert_eq!(retry, AcquireRetry::default());
    }

    /// A producer faster than the display must supersede frames, not lose them.
    ///
    /// The window probes acquire without waiting, then grants an exhausted WSI
    /// one bounded retry budget. FIFO routinely needs that wait because an image
    /// becomes available at a refresh boundary. MAILBOX can need it too when
    /// buffer releases are waiting in the client event queue, but replacing the
    /// pending image still ensures the newest submitted frame survives.
    #[test]
    fn the_swapchain_prefers_mailbox_and_falls_back_to_fifo() {
        use super::{choose_present_mode, swapchain_image_count};
        let fifo = vk::PresentModeKHR::FIFO;
        let mailbox = vk::PresentModeKHR::MAILBOX;

        assert_eq!(choose_present_mode(&[fifo, mailbox]), mailbox);
        assert_eq!(
            choose_present_mode(&[fifo, vk::PresentModeKHR::IMMEDIATE]),
            fifo,
            "IMMEDIATE tears and is not a substitute"
        );
        // A failed mode query arrives as an empty slice, and FIFO is the only
        // mode Vulkan guarantees, so it is what an unknown surface gets.
        assert_eq!(choose_present_mode(&[]), fifo);

        // MAILBOX needs a third image; FIFO keeps the one-spare rule.
        assert_eq!(swapchain_image_count(2, 0, mailbox), 3);
        assert_eq!(swapchain_image_count(1, 0, mailbox), 3);
        assert_eq!(swapchain_image_count(1, 0, fifo), 2);
        assert_eq!(swapchain_image_count(3, 0, mailbox), 4);
        // The surface's own maximum still wins over the MAILBOX floor: a
        // surface that caps at two cannot be argued into three.
        assert_eq!(swapchain_image_count(1, 2, mailbox), 2);
        assert_eq!(swapchain_image_count(2, 3, mailbox), 3);
    }

    /// The mode that sizes the image count must be the mode the swapchain gets.
    ///
    /// It was not. `choose_present_mode` picked MAILBOX, the count was raised to
    /// three on that basis, and the create info was then handed a literal
    /// `FIFO` — while the census printed the *chosen* mode, so a live log read
    /// `present_mode=mailbox images=3` for a swapchain that was FIFO. The
    /// session that introduced the choice measured "no effect on presents" and
    /// recorded that MAILBOX does not help, because the driver never saw it.
    ///
    /// The test the old shape could pass is the one above: both halves were
    /// correct in isolation and only their pairing was not. So this asserts the
    /// pairing — one plan, whose count is derived from the very mode the create
    /// info reads.
    #[test]
    fn the_swapchain_plan_sizes_its_images_for_the_mode_it_will_actually_ask_for() {
        use super::swapchain_plan;
        let fifo = vk::PresentModeKHR::FIFO;
        let mailbox = vk::PresentModeKHR::MAILBOX;

        let offered = swapchain_plan(2, 0, &[fifo, mailbox]);
        assert_eq!(offered.present_mode, mailbox);
        assert_eq!(
            offered.image_count, 3,
            "a three-image count is only justified by the mode that needs it"
        );

        let bare = swapchain_plan(2, 0, &[fifo]);
        assert_eq!(bare.present_mode, fifo);
        assert_eq!(
            bare.image_count, 3,
            "min+1 with caps_min=2 is three under either mode"
        );
        assert_eq!(swapchain_plan(1, 0, &[fifo]).image_count, 2);

        // A surface that caps at two forces FIFO's count onto a MAILBOX plan;
        // the mode is still MAILBOX, because the cap is about images.
        let capped = swapchain_plan(1, 2, &[fifo, mailbox]);
        assert_eq!(capped.present_mode, mailbox);
        assert_eq!(capped.image_count, 2);
    }

    /// The present depth must be servable by the swapchain the plan asks for.
    ///
    /// The two numbers are derived from one constant now, so this cannot drift
    /// by a rename — but it can drift by someone raising [`PRESENT_IN_FLIGHT`]
    /// for its own sake. An entry past the image count cannot acquire, so it
    /// would refuse as `busy_acquire` forever: the presenter would look like it
    /// had depth while permanently wasting its last slot, and the counter that
    /// says so is the one nobody reads when `busy_fence` is the suspect.
    ///
    /// The capped case is asserted too, and asserted as *tolerated* rather than
    /// as correct: a surface that will only give two images leaves the third
    /// entry unusable, and that is safe and self-reporting rather than a bug.
    #[test]
    fn the_present_depth_is_servable_by_the_swapchain_it_asks_for() {
        use super::{swapchain_plan, MAILBOX_MIN_IMAGES, PRESENT_IN_FLIGHT};
        let fifo = vk::PresentModeKHR::FIFO;
        let mailbox = vk::PresentModeKHR::MAILBOX;

        assert_eq!(
            PRESENT_IN_FLIGHT, MAILBOX_MIN_IMAGES as usize,
            "the depth is the swapchain's own floor, not a number of its own"
        );
        let plan = swapchain_plan(1, 0, &[fifo, mailbox]);
        assert_eq!(plan.present_mode, mailbox);
        assert!(
            plan.image_count as usize >= PRESENT_IN_FLIGHT,
            "a MAILBOX swapchain must be able to serve every in-flight present: \
             {} images against depth {PRESENT_IN_FLIGHT}",
            plan.image_count
        );

        let capped = swapchain_plan(1, 2, &[fifo, mailbox]);
        assert!(
            (capped.image_count as usize) < PRESENT_IN_FLIGHT,
            "a surface capped at two is the case the depth cannot be served in"
        );
    }
}
