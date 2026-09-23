//! Engine-device WSI presentation for the macOS Vulkan host window.
//!
//! The final compositor resident stays on the engine `VkDevice`. A short
//! queue-ordered blit writes it into the acquired MoltenVK swapchain image; no
//! host readback, staging upload, or second Vulkan device exists
//! on this pathway.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use super::context::DeviceContext;
use super::pools::ResourcePools;
use super::types::{DrawError, PresentRect, WindowPresentSource};
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::translate;

/// Host-window present transactions submitted to the queue and not yet retired.
///
/// Read by [`super::pools::ResourcePools::open_slot_mask`], which is what
/// decides whether `dispose` may destroy a host object on the spot or must park
/// it in the graveyard. That mask covered the engine's own command-buffer slots
/// and its open batch, and nothing else — so a resident retired out from under a
/// window blit that was submitted and still executing was destroyed
/// immediately, and the blit read freed memory. `open_slot_mask`'s own
/// documentation names this case and its remedy: a caller whose recording
/// outlives its bookkeeping "must either do all of its bookkeeping after the
/// submit, or make its recording slot visible here the way `open_batch` is".
/// This is that slot.
///
/// A count and not a flag: the presenter runs a ring of entries
/// ([`WindowPresenter::present_depth`]) and several can be in flight at once, so
/// the last one to retire is what clears the slot.
///
/// A `static` rather than a field on either side, because it is a fact about the
/// process: `backend::select` latches one rail, the engine owns one
/// `VkInstance`/`VkDevice`, and `window_presenter` hangs off that same owner as
/// one host window. It is deliberately *not* on `ResourcePools`: the registry is
/// guest-derived state on its way to the device the guest declared it against,
/// and a presenter that had to reach into it to say "I am still running" would
/// be the coupling that move exists to remove.
static WINDOW_PRESENTS_IN_FLIGHT: AtomicU32 = AtomicU32::new(0);

/// Diagnostic A/B switch for the resident-vs-CPU window source.
///
/// The normal path prefers the engine resident. The screenshot evidence from
/// T009 shows a stable, corrupted frame being reused, so this switch lets a
/// controlled runtime bypass that choice without changing the production
/// default or the guest protocol. A corrected image here implicates resident
/// content or its handoff; an unchanged image implicates the guest CPU frame
/// or its writeback instead.
static FORCE_CPU_WINDOW_PRESENT: OnceLock<bool> = OnceLock::new();
static FORCE_COPY_WINDOW_PRESENT: OnceLock<bool> = OnceLock::new();
static FRAME_TIMING: OnceLock<bool> = OnceLock::new();

fn frame_timing_enabled() -> bool {
    *FRAME_TIMING.get_or_init(|| {
        matches!(
            crate::config::read(crate::config::FRAME_TIMING).0,
            crate::config::Switch::On
        )
    })
}

fn force_cpu_window_present() -> bool {
    *FORCE_CPU_WINDOW_PRESENT.get_or_init(|| {
        std::env::var("REIMS_VGPU_WINDOW_FORCE_CPU")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "on" | "true"
                )
            })
            .unwrap_or(false)
    })
}

fn force_copy_window_present() -> bool {
    *FORCE_COPY_WINDOW_PRESENT.get_or_init(|| {
        std::env::var("REIMS_VGPU_WINDOW_FORCE_COPY")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "on" | "true"
                )
            })
            .unwrap_or(false)
    })
}

/// Whether any host-window present is submitted and unretired.
pub(crate) fn window_presents_in_flight() -> bool {
    WINDOW_PRESENTS_IN_FLIGHT.load(Ordering::Acquire) != 0
}

/// Claim the window's graveyard slot for one submitted present.
///
/// Called only where `PresentFrame::submitted` is set, and paired with
/// [`WindowPresenter::end_present_in_flight`] at every place it is cleared.
fn begin_present_in_flight() {
    WINDOW_PRESENTS_IN_FLIGHT.fetch_add(1, Ordering::Release);
}

/// Hold the window's graveyard slot for the body of a test, and give it back on
/// drop.
///
/// A guard rather than a bare pair of calls, because the counter is a `static`
/// and the suite is serial: a test that asserted its way to a panic between a
/// claim and its release would leave every later test believing a present is
/// outstanding, and the failure would land on an innocent test.
#[cfg(test)]
pub(crate) struct PresentInFlightForTest;

#[cfg(test)]
impl PresentInFlightForTest {
    pub(crate) fn claim() -> Self {
        begin_present_in_flight();
        Self
    }
}

#[cfg(test)]
impl Drop for PresentInFlightForTest {
    fn drop(&mut self) {
        WINDOW_PRESENTS_IN_FLIGHT.fetch_sub(1, Ordering::Release);
    }
}

/// Clear `frame`'s `submitted` latch and give back the graveyard slot it
/// claimed.
///
/// The two are one action and are never written apart. The latch says this
/// entry's blit is outstanding; the slot is what `dispose` consults before
/// destroying anything that blit may still be reading. Clearing the latch alone
/// would open the graveyard while the blit runs — the defect the slot exists to
/// close — and clearing the slot alone would hold it shut forever.
///
/// A free function over one frame rather than a method, because two of the four
/// sites clear latches while iterating `&mut self.frames` and could not call a
/// `&mut self` method; splitting the rule across a method and two hand-written
/// copies is how the pair would come apart.
///
/// `mem::replace` and not an unconditional decrement: `recreate_swapchain` and
/// `destroy` both clear latches that may already be clear, so the decrement has
/// to be per claim rather than per call or it underflows.
fn end_present_in_flight(frame: &mut PresentFrame) {
    if std::mem::replace(&mut frame.submitted, false) {
        WINDOW_PRESENTS_IN_FLIGHT.fetch_sub(1, Ordering::Release);
    }
}

/// Consecutive suboptimal-flagged presents (each of which arms a swapchain
/// recreation) before the always-on alarm names the class. Recreation normally
/// clears the flag on the next frame, and a live user resize clears the streak
/// whenever the extent actually changes. A streak this long at an unchanged
/// extent means recreation is not converging and the window may be presenting
/// invisibly (the CAMetalLayer drawableSize-clobber class).
const SUBOPTIMAL_ALARM_STREAK: u32 = 60;

/// The pre-content / letterbox-bar clear color (linear BGRA channels).
const SLATE_CLEAR: [f32; 4] = [0.05, 0.06, 0.08, 1.0];

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
///
/// # Two, where there were six
///
/// The other four — `no_resident`, `content_not_ready`, `not_bgra`,
/// `geom_mismatch` — were the presenter's own judgement of a resident it had
/// re-resolved out of the registry. It resolves nothing now: the publish decides
/// presentability a frame earlier, under the engine lock, where the registry is,
/// and each of those four is reported there as `winpub_no_resident`,
/// `winpub_content_not_ready`, `winpub_scanout_order` or `winpub_geometry` on
/// the drain's route channel. A refused publish reaches this thread as no source
/// at all, which is what [`Self::NoSource`] means.
///
/// The pair that remains is the pair only this thread can answer: nothing was
/// published, or what was published has since been withdrawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlateReason {
    /// No frame source was published for this present. Expected before the
    /// present boundary and while the guest is idle — and also what a publish
    /// that *declined* looks like from here, the reason for the decline being
    /// named at the publish.
    NoSource,
    /// The resident this source was resolved against has since been withdrawn.
    /// Not a statement about any resident now under that identity: the stamp
    /// says the device replaced, dropped, re-laid-out or re-declared the one
    /// that was promised, so the resolution the source carries may name an
    /// image that is already destroyed.
    SourceStale,
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
            Self::SourceStale => "slate_source_stale",
            Self::NoSource => "slate_no_source",
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

    /// Delegated arm for arm with `slug`; see
    /// [`crate::observe::slugs`].
    fn owner(&self) -> &'static str {
        match self {
            Self::Call(call) => call.owner(),
            Self::NoUploadMemoryType { .. } => std::any::type_name::<Self>(),
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
pub(crate) use crate::backend::window::WindowCpuFrame;

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
        next_access: super::pools::ResidentAccess,
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
            Self::Resident {
                image,
                access,
                next_access,
                ..
            } => {
                super::exec::barrier_resident_for_transfer_read(
                    device,
                    cmd,
                    *image,
                    *access,
                    *next_access,
                );
                next_access.layout()
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

pub(crate) use crate::backend::window::WindowPresentOutcome;

/// Result of the lock-held half of a display transaction.
pub(crate) enum WindowPresentDispatch {
    Complete(WindowPresentOutcome),
    Pending(PendingWindowPresent),
}

/// Queue-owner completion plus the immutable facts needed to classify it.
///
/// No engine or pool reference crosses this boundary.  The resident was pinned
/// and its next access recorded before the transaction entered the ordered
/// queue; only the host driver's result remains outstanding.
pub(crate) struct PendingWindowPresent {
    wait: super::queue_owner::PendingPresent,
    acquire_suboptimal: bool,
    direct: bool,
    width: u32,
    height: u32,
    swapchain_images: usize,
}

pub(crate) struct FinishedWindowPresent {
    result: Result<bool, vk::Result>,
    acquire_suboptimal: bool,
    direct: bool,
    width: u32,
    height: u32,
    swapchain_images: usize,
}

impl PendingWindowPresent {
    /// Wait only for the display transaction's own completion.  This method
    /// carries no engine reference and is therefore safe to call after the
    /// global engine guard has been dropped.
    pub(crate) fn wait(self) -> FinishedWindowPresent {
        FinishedWindowPresent {
            result: self.wait.wait(),
            acquire_suboptimal: self.acquire_suboptimal,
            direct: self.direct,
            width: self.width,
            height: self.height,
            swapchain_images: self.swapchain_images,
        }
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
    /// One entry per present that may be in flight at once, used round-robin.
    ///
    /// Every field in [`PresentFrame`] is per-present and none may be shared: a
    /// second present recording into the first's command buffer, or waiting on
    /// its `image_available` while the first's acquire is still outstanding, is
    /// a use-after-submit.
    frames: Vec<PresentFrame>,
    /// The semaphore `queue_submit` signals and `queue_present` waits on, **one
    /// per swapchain image**, indexed by the acquired image index.
    ///
    /// # Why this is not per entry, which is where it used to live
    ///
    /// A binary semaphore signalled by a submit and waited on by a present is
    /// free to be signalled again only once that present has completed — and a
    /// present's completion is not observable. `vkAcquireNextImageKHR` returning
    /// an image index says *that image* is reusable; it says nothing about the
    /// presents of the other images. So the only index under which reuse is
    /// safe is the image's.
    ///
    /// Keyed by the round-robin entry instead, the entry index and the image
    /// index advance independently — the acquire hands back whatever image the
    /// presentation engine has free — so entry N could be signalled while a
    /// present that entry N started for a *different* image was still pending.
    /// The Khronos validation layer reported exactly that on a driven macos-11
    /// boot: `VUID-vkQueueSubmit-pSignalSemaphores-00067`, "is being signaled by
    /// VkQueue, but it may still be in use by VkSwapchainKHR ... Swapchain image
    /// 0 was presented but was not re-acquired".
    ///
    /// Rebuilt with the swapchain, because its length is the image count.
    render_finished: Vec<vk::Semaphore>,
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
    /// `Busy` returns from the third gate: the surface has no area, so there is
    /// no swapchain to acquire from. A minimized window, and its own counter
    /// because `busy = fence + acquire` was an identity worth keeping true.
    cadence_busy_no_area: u64,
    timing_started: Instant,
    timing_frames: u64,
    timing_retire_us: u64,
    timing_acquire_us: u64,
    timing_record_us: u64,
    timing_submit_us: u64,
    timing_total_us: u64,
    timing_max_total_us: u64,
    /// Whether the run in progress is a window with no area, so the reason is
    /// stated when the run starts and not on every frame of it.
    surface_had_no_area: bool,
    /// `vkGetPhysicalDeviceSurfaceSupportKHR` for this surface and the graphics
    /// queue, asked once at attach.
    ///
    /// Carried rather than re-queried per recreate: it is a property of the
    /// pairing, which does not change while the presenter owns the surface, and
    /// a second query is a second answer that could disagree with the one
    /// attach already refused on.
    present_family_supported: bool,
}

/// Everything one in-flight present owns for as long as its blit is running.
struct PresentFrame {
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    /// Whether this entry's blit has been submitted and not yet retired.
    ///
    /// The only per-entry state a present still carries about its source. It
    /// used to carry the resident identities it had pinned as well; the pin is
    /// gone — `ResourcePools::window_holds` keeps a displayed resident off the
    /// reclaim paths from the publish that named it, which is both wider in time
    /// and not a write this thread has to make.
    submitted: bool,
}

/// The stage the acquire semaphore is waited at, and therefore the stage the
/// first layout transition must name as its source.
///
/// One constant because the two must agree. A submit that waits at `TRANSFER`
/// while the barrier ahead of it declares `TOP_OF_PIPE` puts the transition
/// before the wait it exists to be ordered after, and the pair being written
/// eleven lines apart is what let them disagree.
const ACQUIRE_WAIT_STAGE: vk::PipelineStageFlags = vk::PipelineStageFlags::TRANSFER;

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
const PRESENT_IN_FLIGHT: usize = reims_vgpu_vulkan::swapchain::MAILBOX_MIN_IMAGES as usize;

/// At least as deep as the single-flight presenter this replaced, so indexing
/// `frames` is always valid, and no deeper than the swapchain's own floor.
/// Both ends are relations against values derived elsewhere rather than
/// restatements of this line.
const _: () = assert!(
    PRESENT_IN_FLIGHT >= 1
        && PRESENT_IN_FLIGHT <= reims_vgpu_vulkan::swapchain::MAILBOX_MIN_IMAGES as usize
);

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
            let (state, value) = crate::config::read(crate::config::PRESENT_DEPTH);
            match state {
                crate::config::Switch::Off => {
                    crate::observe::off("present_depth reason=present_depth_disabled_by_env");
                    1
                }
                crate::config::Switch::Unrecognized => {
                    crate::observe::fail(format!(
                        "present_depth reason=present_depth_env_unrecognized value={}",
                        value.unwrap_or_default()
                    ));
                    PRESENT_IN_FLIGHT
                }
                crate::config::Switch::Unset | crate::config::Switch::On => PRESENT_IN_FLIGHT,
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
            frames,
            // Sized by the swapchain's image count, which the recreate below is
            // the first thing to know.
            render_finished: Vec::new(),
            frame_ix: 0,
            cadence_started: Instant::now(),
            cadence_presents: 0,
            cadence_direct: 0,
            cadence_busy: 0,
            cadence_offered: 0,
            cadence_last_offered: None,
            cadence_busy_fence: 0,
            cadence_busy_acquire: 0,
            cadence_busy_no_area: 0,
            timing_started: Instant::now(),
            timing_frames: 0,
            timing_retire_us: 0,
            timing_acquire_us: 0,
            timing_record_us: 0,
            timing_submit_us: 0,
            timing_total_us: 0,
            timing_max_total_us: 0,
            surface_had_no_area: false,
            // Attach refused above unless this was true, so the presenter that
            // exists is one whose queue can address its surface.
            present_family_supported: present_capable,
        };
        // A window created while minimized has no swapchain yet, which is not a
        // failure to attach: `begin_present` retries every frame until the
        // window comes back.
        if let Err(error) = presenter.recreate_swapchain(ctx) {
            presenter.destroy(ctx);
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
    /// Sweeping all of them rather than only the next one matters for the
    /// graveyard: an entry that completed is still holding `WINDOW_PRESENT_SLOT`
    /// shut, and with several in flight the round-robin might not revisit it for
    /// another two presents. The return value is still about one entry, because
    /// that is the only one the caller is about to record into.
    unsafe fn retire(&mut self, ctx: &DeviceContext) -> Result<bool, DrawError> {
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
            end_present_in_flight(&mut self.frames[ix]);
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

    /// Choose and build a swapchain for the surface as it is right now.
    ///
    /// `Ok(false)` is a surface with no area — a minimized window — which is
    /// not a failure and not a capability this host lacks: it resolves when the
    /// window comes back. The presenter keeps `recreate_pending` and the caller
    /// reports the frame busy rather than acquiring against a null swapchain.
    unsafe fn recreate_swapchain(&mut self, ctx: &DeviceContext) -> Result<bool, DrawError> {
        ctx.queue_wait_idle()
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowQueueWaitIdle, error)))?;
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceCaps, error)))?;
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceFormats, error)))?;
        // A failed mode query arrives as an empty slice: FIFO is the only mode
        // the specification requires of every surface, so a surface that could
        // not be asked still gets the rung it is guaranteed to have.
        let modes = self
            .surface_loader
            .get_physical_device_surface_present_modes(ctx.pd, self.surface)
            .unwrap_or_default();
        // Every question the swapchain is chosen from, answered, and the choice
        // made in one place with no device — see `reims_vgpu_vulkan::swapchain`
        // for why the format is refused rather than substituted, why the
        // composite-alpha ladder stops before the blending modes, and why
        // MAILBOX asks for a third image.
        let plan = match reims_vgpu_vulkan::swapchain::plan(
            &reims_vgpu_vulkan::swapchain::Surface {
                supported_by_present_family: self.present_family_supported,
                capabilities: caps,
                formats: &formats,
                present_modes: &modes,
            },
            reims_vgpu_vulkan::swapchain::Wanted {
                format: translate::pixel::SCANOUT_FORMAT,
                extent: self.desired_extent,
                // The composition blits the guest's frame in rather than
                // rendering to the image.
                transfer_destination: true,
            },
            reims_vgpu_vulkan::swapchain::Narrowing::from_env(),
        ) {
            Ok(reims_vgpu_vulkan::swapchain::Outcome::Ready(plan)) => plan,
            Ok(reims_vgpu_vulkan::swapchain::Outcome::NotReady(why)) => {
                // Expected control flow, so it is quiet on the failure channel
                // and reported once per run instead: a minimized window at
                // 120 Hz would otherwise cost 120 lines a second, and the
                // cadence line's `busy_no_area` carries the rate.
                if !self.surface_had_no_area {
                    self.surface_had_no_area = true;
                    crate::observe::off(format!("host_window_swapchain status=not_ready {why}"));
                }
                self.recreate_pending = true;
                return Ok(false);
            }
            Err(refusal) => {
                let reason = super::reason::DrawReason::SwapchainSurface(refusal);
                crate::observe::Emit::decline("host_window_swapchain", &reason).fail();
                return Err(DrawError::Unsupported(reason));
            }
        };
        self.surface_had_no_area = false;
        let extent = plan.extent;
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
        // workaround assumes. The queue idled above, so no submitted work
        // references the old swapchain.
        let from = self.extent;
        if self.swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
            self.images.clear();
        }
        // `SwapchainKHR::null()` rather than the swapchain destroyed above: the
        // whole point of the paragraph above is that this one carries no old
        // swapchain.
        let swapchain = self
            .swapchain_loader
            .create_swapchain(
                &plan.create_info(self.surface, vk::SwapchainKHR::null()),
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
        // the old pair is destroyed so a failure leaves the presenter
        // consistent.
        // Every entry gets a fresh pair, not just the one about to be used. The
        // queue idled above, so no entry has work outstanding — but an entry
        // whose acquire succeeded and whose submit then failed still holds an
        // unconsumed signal on its `image_available`, and that is invalid to
        // reuse against the new swapchain whichever entry it belongs to.
        // One acquire semaphore per entry and one render semaphore per swapchain
        // **image** — the two counts are independent and the image count is only
        // known here, which is the other reason the render semaphores live with
        // the swapchain rather than with the entries.
        let mut fresh_acquire: Vec<vk::Semaphore> = Vec::with_capacity(self.frames.len());
        let mut fresh_render: Vec<vk::Semaphore> = Vec::with_capacity(images.len());
        let mut make = || -> Result<(), (VkOp, vk::Result)> {
            for _ in 0..self.frames.len() {
                fresh_acquire.push(
                    ctx.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| (VkOp::WindowCreateAcquireSemaphore, e))?,
                );
            }
            for _ in 0..images.len() {
                fresh_render.push(
                    ctx.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| (VkOp::WindowCreateRenderSemaphore, e))?,
                );
            }
            Ok(())
        };
        if let Err((op, error)) = make() {
            for semaphore in fresh_acquire.into_iter().chain(fresh_render) {
                ctx.device.destroy_semaphore(semaphore, None);
            }
            self.swapchain_loader.destroy_swapchain(swapchain, None);
            return Err(DrawError::VkCall(VkCall::new(op, error)));
        }
        for (frame, image_available) in self.frames.iter_mut().zip(fresh_acquire) {
            ctx.device.destroy_semaphore(frame.image_available, None);
            frame.image_available = image_available;
            // The queue idled, so nothing is outstanding regardless of what the
            // latch said before.
            end_present_in_flight(frame);
        }
        for semaphore in self.render_finished.drain(..) {
            ctx.device.destroy_semaphore(semaphore, None);
        }
        self.render_finished = fresh_render;
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
        Ok(true)
    }

    pub(crate) unsafe fn begin_present(
        &mut self,
        ctx: &DeviceContext,
        pools: &mut ResourcePools,
        source: Option<&WindowPresentSource>,
        cpu: Option<WindowCpuFrame<'_>>,
    ) -> Result<WindowPresentDispatch, DrawError> {
        let timing = frame_timing_enabled();
        let total_started = timing.then(Instant::now);
        if let Some(seq) = cpu.map(|frame| frame.seq) {
            if self.cadence_last_offered != Some(seq) {
                self.cadence_last_offered = Some(seq);
                self.cadence_offered = self.cadence_offered.saturating_add(1);
            }
        }
        let retire_started = timing.then(Instant::now);
        let retired = self.retire(ctx)?;
        if let Some(started) = retire_started {
            self.timing_retire_us = self
                .timing_retire_us
                .saturating_add(started.elapsed().as_micros() as u64);
        }
        if !retired {
            self.cadence_busy_fence = self.cadence_busy_fence.saturating_add(1);
            self.note_cadence(false, false);
            return Ok(WindowPresentDispatch::Complete(WindowPresentOutcome::Busy));
        }
        if self.swapchain == vk::SwapchainKHR::null() || self.recreate_pending {
            // A minimized window has no swapchain to acquire from and nothing
            // is wrong: the recreation stays armed and this frame is busy. The
            // gate is counted apart from the other two because it has a
            // different fix from both — the fence gate says the engine queue is
            // behind and the acquire gate says the display is pacing us, while
            // this one says there is no window to pace against.
            if !self.recreate_swapchain(ctx)? {
                self.cadence_busy_no_area = self.cadence_busy_no_area.saturating_add(1);
                self.note_cadence(false, false);
                return Ok(WindowPresentDispatch::Complete(WindowPresentOutcome::Busy));
            }
        }
        // Bound after any swapchain recreation above, which resets every latch.
        let frame_ix = self.frame_ix;
        let frame_cmd = self.frames[frame_ix].cmd;
        let frame_image_available = self.frames[frame_ix].image_available;
        let frame_in_flight = self.frames[frame_ix].in_flight;
        let acquire_timeout_ns: u64 = std::env::var("REIMS_VGPU_ACQUIRE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(100)
            * 1_000_000;
        let acquire_started = timing.then(Instant::now);
        let (image_index, acquire_suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            acquire_timeout_ns,
            frame_image_available,
            vk::Fence::null(),
        ) {
            Ok((index, suboptimal)) => (index, suboptimal),
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                self.cadence_busy_acquire = self.cadence_busy_acquire.saturating_add(1);
                self.note_cadence(false, false);
                return Ok(WindowPresentDispatch::Complete(WindowPresentOutcome::Busy));
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "acquire_out_of_date";
                self.cadence_busy_acquire = self.cadence_busy_acquire.saturating_add(1);
                self.note_cadence(false, false);
                return Ok(WindowPresentDispatch::Complete(WindowPresentOutcome::Busy));
            }
            Err(error) => {
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowAcquireImage,
                    error,
                )));
            }
        };
        let acquire_us = acquire_started.map(|started| started.elapsed().as_micros() as u64);

        // Keyed by the acquired image and not by the entry: the acquire is what
        // says this image's previous present has completed, and nothing says
        // that about the entry's previous present of some other image.
        debug_assert_eq!(
            self.render_finished.len(),
            self.images.len(),
            "one render semaphore per swapchain image; both are set from the \
             same `images` in `recreate_swapchain` and nothing else writes either"
        );
        let frame_render_finished = self.render_finished[image_index as usize];

        // The whole of this present's source selection, and it reads no
        // registry.
        //
        // The window thread has no device to reach one through — the lock a
        // `&DeviceState` comes behind is the one the drain holds for a whole
        // render tranche, measured at 935-979 ms per exec packet — so the
        // registry, which is guest-derived state on its way to the device the
        // guest declared it against, cannot come with it. What replaced the read
        // is one atomic load.
        //
        // `source.resolved` is what the publish resolved this identity to under
        // the engine lock, and `source.epoch` is `WINDOW_SOURCE_EPOCH` as it
        // stood at that moment. The epoch moves at every point the resolution
        // can stop being true: `unregister_resident` (the registry's sole
        // `remove`), `destroy_all` (its sole `drain`), `Drop for ResourcePools`,
        // `set_registry_access` and `set_registry_format` — the last two being
        // the only writers of the two fields production mutates on a live slot.
        // Every other field of a slot is written once, where the slot is built.
        //
        // So a stamp that still compares equal is the publish's whole decision,
        // still standing: the image is registered, its pixels landed, its byte
        // order is the scanout's, its extent is the one being presented, and the
        // barrier below names the access the image is actually in.
        let stale =
            source.is_some_and(|source| source.epoch != super::pools::window_source_epoch());
        let selected = source
            .filter(|_| !stale && !force_cpu_window_present())
            .map(|source| (source.identity.clone(), source.resolved));
        // Only reached when no resident carries this present: upload the CPU
        // bytes instead. `None` here means the window shows slate.
        let staged = if selected.is_some() {
            self.note_slate_end();
            None
        } else {
            // Two reasons reach here and no more. The presenter no longer judges
            // a resident — it cannot, having no registry — so "resident but not
            // ready", "resident but not BGRA" and "resident at the wrong size"
            // are decided by `resident_present_decision` a frame earlier and
            // reported there as `winpub_content_not_ready`, `winpub_scanout_order`
            // and `winpub_geometry`. What is left is: nothing was published, or
            // what was published has since been withdrawn.
            let want = source.map_or((0, 0), |s| (s.width, s.height));
            let reason = if stale {
                SlateReason::SourceStale
            } else {
                SlateReason::NoSource
            };
            let staged = cpu
                .filter(WindowCpuFrame::complete)
                .and_then(|frame| self.stage_cpu_frame(ctx, frame));
            self.note_slate(reason, want, source.map(|s| s.resolved), staged.is_some());
            staged
        };
        // One blit body for both sources: they differ only in which image is
        // read and how it is made readable. Keeping them separate is how the
        // aspect-fit and letterbox-clear rules drift apart between the two
        // rails.
        let blit = selected
            .as_ref()
            .map(|(_, now)| BlitSource::Resident {
                image: now.image,
                access: now.access,
                next_access: super::pools::ResidentAccess::transfer_read(now.guest_imported),
                width: now.width,
                height: now.height,
            })
            .or(staged);

        let record_started = timing.then(Instant::now);
        let mut submit_us = None;
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
            // `srcStageMask` is `TRANSFER` and not `TOP_OF_PIPE`, and it has to
            // match the `pWaitDstStageMask` of the submit below.
            //
            // A layout transition is a write, and the acquire semaphore's wait
            // is what says the presentation engine has finished reading this
            // image. `TOP_OF_PIPE` puts the transition ahead of that wait, so
            // the transition is ordered against nothing — reported by the
            // Khronos validation layer on a driven macos-11 boot as
            // `SYNC-HAZARD-WRITE-AFTER-READ`, "vkCmdPipelineBarrier writes to
            // VkImage, which was previously accessed by vkAcquireNextImageKHR
            // ... layout transition does not synchronize with these stages".
            image_barrier(
                &ctx.device,
                frame_cmd,
                dst,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                ACQUIRE_WAIT_STAGE,
                vk::PipelineStageFlags::TRANSFER,
            );
            if let Some(blit) = blit {
                let (base_width, base_height) = blit.extent();
                // Aspect-fit placement: the guest frame keeps its aspect ratio
                // inside whatever drawable exists right now (a guest-driven
                // native resize normally makes this the full window within
                // milliseconds). The window input path maps pointer positions
                // through this same transform.
                let vp = crate::backend::window::viewport::aspect_fit(
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
                    // The blit below overwrites the middle of what that clear
                    // just wrote. Both are transfer writes and Vulkan orders
                    // neither against the other — the clear stage and the blit
                    // stage are distinct, so "same command buffer, recorded
                    // earlier" says nothing. Without this the two race for the
                    // letterboxed pixels, which is a frame of slate over guest
                    // content: `SYNC-HAZARD-WRITE-AFTER-WRITE`, "vkCmdBlitImage
                    // writes to VkImage, which was previously written by
                    // vkCmdClearColorImage".
                    //
                    // Only on the letterbox arm, because that is the only arm
                    // that writes the image twice.
                    ctx.device.cmd_pipeline_barrier(
                        frame_cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                        &[],
                        &[],
                    );
                }
                let src_layout = blit.record_read_barrier(&ctx.device, frame_cmd);
                let src_rect = (0, 0, base_width, base_height);
                let dst_rect = (vp.x, vp.y, vp.x + vp.width, vp.y + vp.height);
                if force_copy_window_present()
                    && src_rect == dst_rect
                    && base_width == self.extent.width
                    && base_height == self.extent.height
                {
                    copy_rect(
                        &ctx.device,
                        frame_cmd,
                        blit.image(),
                        dst,
                        src_layout,
                        base_width,
                        base_height,
                    );
                } else {
                    blit_rect(
                        &ctx.device,
                        frame_cmd,
                        blit.image(),
                        dst,
                        src_layout,
                        src_rect,
                        dst_rect,
                    );
                }
                // The window's last contact with the resident registry, and the
                // one that stays. Two ways out of it were looked for and both
                // are unsound; recording that here so the third reader does not
                // spend the afternoon finding out again.
                //
                // **Not the publish, recording it eagerly.** The publish would
                // be promising a transition that a `Busy` present, a failed
                // acquire or a detached window never makes, and the registry
                // would then describe an image the device has not moved. The lie
                // is not the layout — a primary target rests in
                // `TRANSFER_SRC_OPTIMAL` and `transfer_read(false)` is the same
                // layout — it is the *access*: the next barrier over the image
                // would name `TRANSFER_READ` as its source scope and the colour
                // write that really happened last would never be made available.
                //
                // **Not a restore, transitioning back after the blit so the
                // registry stays true.** The layout would come back and the
                // dependency would not. A guest colour write barriering from the
                // restored `ColorWrite` names `COLOR_ATTACHMENT_OUTPUT` as its
                // source stage, which does not wait for this blit's `TRANSFER`
                // read — a write-after-read on the pixels the screen is reading.
                // The write-back exists to make that next barrier say `TRANSFER`,
                // which is the whole of its job.
                //
                // What both attempts have in common is that the barrier and the
                // record of it have to be one action, taken by whoever holds the
                // registry, adjacent to the blit. So this write leaves when the
                // present becomes a transaction against the device that owns the
                // registry — not before, and not by moving the write alone.
                if let Some((identity, now)) = selected.as_ref() {
                    pools.registry_note_access(
                        identity,
                        super::pools::ResidentAccess::transfer_read(now.guest_imported),
                    );
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
            let wait_stages = [ACQUIRE_WAIT_STAGE];
            let signals = [frame_render_finished];
            let commands = [frame_cmd];
            let submit_started = timing.then(Instant::now);
            let result = ctx.submit_present_transaction(super::context::PresentTransaction {
                command_buffers: &commands,
                wait_semaphores: &waits,
                wait_stages: &wait_stages,
                signal_semaphores: &signals,
                fence: frame_in_flight,
                loader: self.swapchain_loader.clone(),
                present_wait: frame_render_finished,
                swapchain: self.swapchain,
                image_index,
            });
            if let Some(started) = submit_started {
                submit_us = Some(started.elapsed().as_micros() as u64);
            }
            result.map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSubmitPresent, error)))
        })();
        // A plain `?` now the failure arm has nothing to undo: it used to have to
        // drop the pins this present had taken before returning.
        let submission = submit_result?;
        if timing {
            let record_us = record_started
                .map(|started| started.elapsed().as_micros() as u64)
                .unwrap_or_default();
            let submit_us = submit_us.unwrap_or_default();
            let total_us = total_started
                .map(|started| started.elapsed().as_micros() as u64)
                .unwrap_or_default();
            self.timing_frames = self.timing_frames.saturating_add(1);
            self.timing_acquire_us = self
                .timing_acquire_us
                .saturating_add(acquire_us.unwrap_or_default());
            self.timing_record_us = self.timing_record_us.saturating_add(record_us);
            self.timing_submit_us = self.timing_submit_us.saturating_add(submit_us);
            self.timing_total_us = self.timing_total_us.saturating_add(total_us);
            self.timing_max_total_us = self.timing_max_total_us.max(total_us);
            self.note_timing();
        }
        // Claimed before the latch, so the slot is never observed clear while
        // the latch says an entry is outstanding.
        begin_present_in_flight();
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

        let direct = selected.is_some();
        match submission {
            super::context::PresentSubmission::Complete(result) => {
                self.finish_present(FinishedWindowPresent {
                    result,
                    acquire_suboptimal,
                    direct,
                    width: self.extent.width,
                    height: self.extent.height,
                    swapchain_images: self.images.len(),
                })
            }
            super::context::PresentSubmission::Pending(wait) => {
                Ok(WindowPresentDispatch::Pending(PendingWindowPresent {
                    wait,
                    acquire_suboptimal,
                    direct,
                    width: self.extent.width,
                    height: self.extent.height,
                    swapchain_images: self.images.len(),
                }))
            }
        }
    }

    fn note_timing(&mut self) {
        let elapsed = self.timing_started.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let frames = self.timing_frames.max(1);
        crate::observe::off(format!(
            "host_window_timing window_ms={} frames={} retire_avg_us={} acquire_avg_us={} \
             record_avg_us={} submit_avg_us={} total_avg_us={} total_max_us={}",
            elapsed.as_millis(),
            self.timing_frames,
            self.timing_retire_us / frames,
            self.timing_acquire_us / frames,
            self.timing_record_us / frames,
            self.timing_submit_us / frames,
            self.timing_total_us / frames,
            self.timing_max_total_us,
        ));
        self.timing_started = Instant::now();
        self.timing_frames = 0;
        self.timing_retire_us = 0;
        self.timing_acquire_us = 0;
        self.timing_record_us = 0;
        self.timing_submit_us = 0;
        self.timing_total_us = 0;
        self.timing_max_total_us = 0;
    }

    pub(crate) fn finish_present(
        &mut self,
        finished: FinishedWindowPresent,
    ) -> Result<WindowPresentDispatch, DrawError> {
        match finished.result {
            Ok(present_suboptimal) => {
                // ash reports VK_SUBOPTIMAL_KHR as `Ok(true)` (a success code),
                // never through the `Err` arm. MoltenVK returns it from both
                // acquire and present for as long as the CAMetalLayer's
                // drawable or natural size diverges from the swapchain extent —
                // including after a retired swapchain clobbered the layer's
                // drawableSize — so ignoring the flag leaves an invisible
                // window that still counts successful presents.
                let suboptimal = finished.acquire_suboptimal || present_suboptimal;
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
                self.note_cadence(true, finished.direct);
                Ok(WindowPresentDispatch::Complete(
                    WindowPresentOutcome::Presented {
                        direct: finished.direct,
                        width: finished.width,
                        height: finished.height,
                        buffers: finished.swapchain_images,
                        suboptimal,
                    },
                ))
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "present_out_of_date";
                self.note_cadence(false, false);
                Ok(WindowPresentDispatch::Complete(WindowPresentOutcome::Busy))
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
    /// `promised` is the resolution the *publish* handed this thread, not a
    /// re-read of the registry — there is no re-read any more. On
    /// [`SlateReason::SourceStale`] it is what was withdrawn, which is the value
    /// a reader wants; on [`SlateReason::NoSource`] there is nothing to name.
    fn note_slate(
        &mut self,
        reason: SlateReason,
        want: (u32, u32),
        promised: Option<super::pools::ResolvedResident>,
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
        let seen = promised.map_or_else(
            || "absent".to_string(),
            |promised| {
                format!(
                    "{}x{}/{:?}",
                    promised.width, promised.height, promised.access
                )
            },
        );
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
                no_area: self.cadence_busy_no_area,
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
        self.cadence_busy_no_area = 0;
    }

    /// Give every entry's graveyard claim back after the caller has waited the
    /// queue idle.
    ///
    /// It released pins as well until the presenter stopped taking any. What is
    /// left is the claim on `WINDOW_PRESENT_SLOT`, which is the whole reason a
    /// guest reset must call this: the bit is a `static` the graveyard consults,
    /// so an entry that never retires would hold it shut for the rest of the
    /// process.
    pub(crate) fn release_claims_after_idle(&mut self) {
        for frame in &mut self.frames {
            end_present_in_flight(frame);
        }
    }

    pub(crate) unsafe fn destroy(&mut self, ctx: &DeviceContext) {
        if let Err(error) = ctx.queue_wait_idle() {
            let decline = VkCall::new(VkOp::WindowDestroyQueueWaitIdle, error);
            crate::observe::Emit::decline("host_window_destroy", &decline).fail_once(0);
        }
        if let Some(staging) = self.staging.take() {
            staging.destroy(&ctx.device);
        }
        // Drained rather than iterated, so a second `destroy` — `create` calls
        // it on a failed `recreate_swapchain`, and the caller may call it again
        // — cannot double-free a handle.
        for mut frame in self.frames.drain(..) {
            // `queue_wait_idle` above is what makes this sound: every entry's
            // blit has completed, so giving the slot back here cannot open the
            // graveyard under live work. Per entry that actually held a claim,
            // for the reason `end_present_in_flight` uses `mem::replace` —
            // `destroy` may run twice and a second pass has no frames left.
            end_present_in_flight(&mut frame);
            ctx.device.destroy_fence(frame.in_flight, None);
            ctx.device.destroy_semaphore(frame.image_available, None);
        }
        // Drained for the same reason as the entries above: `destroy` may run
        // twice and these handles are per swapchain image, so a second pass
        // would double-free every one of them.
        for semaphore in self.render_finished.drain(..) {
            ctx.device.destroy_semaphore(semaphore, None);
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

/// The three gates that can refuse a present, kept apart because they have
/// different fixes: `fence` is the engine queue still running the previous blit
/// behind however much guest work was submitted ahead of it, `acquire` is the
/// swapchain having no free image, which is the display's own pacing, and
/// `no_area` is a surface with no swapchain at all — a minimized window, which
/// is not a rate problem and resolves by itself.
struct CadenceBusy {
    total: u64,
    fence: u64,
    acquire: u64,
    no_area: u64,
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
         busy={} busy_fence={} busy_acquire={} busy_no_area={} offered={offered} \
         present_hz={hz:.1} offered_hz={offered_hz:.1} direct_frac={direct_fraction:.2}",
        busy.total, busy.fence, busy.acquire, busy.no_area
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

unsafe fn copy_rect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_layout: vk::ImageLayout,
    width: u32,
    height: u32,
) {
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    device.cmd_copy_image(
        cmd,
        src,
        src_layout,
        dst,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })],
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

    #[test]
    fn cadence_proxy_reports_actual_queue_presents_and_direct_fraction() {
        let line = window_cadence_line(
            1_000,
            120,
            119,
            CadenceBusy {
                total: 131,
                fence: 100,
                acquire: 30,
                no_area: 1,
            },
            240,
        );
        assert!(line.contains("presents=120"), "{line}");
        assert!(line.contains("direct=119"), "{line}");
        assert!(line.contains("busy=131"), "{line}");
        assert!(line.contains("busy_fence=100"), "{line}");
        assert!(line.contains("busy_acquire=30"), "{line}");
        // The third gate is its own number, so `busy` stays the sum of the
        // three rather than of the two that used to be named.
        assert!(line.contains("busy_no_area=1"), "{line}");
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
                no_area: 0,
            },
            109,
        );
        assert!(line.contains("offered=109"), "{line}");
        assert!(line.contains("offered_hz=109.0"), "{line}");
        assert!(line.contains("present_hz=20.0"), "{line}");
    }

    /// Every reason has a distinct, `slate_`-prefixed slug.
    ///
    /// What the prefix buys beyond distinctness is keeping a grep for this
    /// window's blit choice from also matching the capture rail's `no_resident_content`
    /// and the `THRASH geom_mismatch` proxy.
    #[test]
    fn slate_reason_slugs_are_distinct_and_namespaced() {
        use crate::observe::Decline;
        let mut slugs = [SlateReason::NoSource, SlateReason::SourceStale].map(|r| r.slug());
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
        use reims_vgpu_vulkan::swapchain::{
            plan, Narrowing, Outcome, Surface, Wanted, MAILBOX_MIN_IMAGES,
        };
        let fifo = vk::PresentModeKHR::FIFO;
        let mailbox = vk::PresentModeKHR::MAILBOX;
        let format = vk::SurfaceFormatKHR {
            format: translate::pixel::SCANOUT_FORMAT,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let images = |max_image_count| {
            let capabilities = vk::SurfaceCapabilitiesKHR {
                min_image_count: 1,
                max_image_count,
                current_extent: vk::Extent2D {
                    width: 64,
                    height: 64,
                },
                supported_usage_flags: vk::ImageUsageFlags::TRANSFER_DST,
                supported_composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                ..Default::default()
            };
            let Ok(Outcome::Ready(plan)) = plan(
                &Surface {
                    supported_by_present_family: true,
                    capabilities,
                    formats: &[format],
                    present_modes: &[fifo, mailbox],
                },
                Wanted {
                    format: translate::pixel::SCANOUT_FORMAT,
                    extent: vk::Extent2D {
                        width: 64,
                        height: 64,
                    },
                    transfer_destination: true,
                },
                Narrowing::default(),
            ) else {
                panic!("this surface carries a swapchain");
            };
            assert_eq!(plan.present_mode, mailbox);
            plan.requested_images as usize
        };

        assert_eq!(
            PRESENT_IN_FLIGHT, MAILBOX_MIN_IMAGES as usize,
            "the depth is the swapchain's own floor, not a number of its own"
        );
        assert!(
            images(0) >= PRESENT_IN_FLIGHT,
            "a MAILBOX swapchain must be able to serve every in-flight present: \
             {} images against depth {PRESENT_IN_FLIGHT}",
            images(0)
        );
        assert!(
            images(2) < PRESENT_IN_FLIGHT,
            "a surface capped at two is the case the depth cannot be served in"
        );
    }
}
