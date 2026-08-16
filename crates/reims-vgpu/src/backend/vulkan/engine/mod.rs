//! Persistent Vulkan draw + compute engine for the Linux metal2vulkan product path.
//!
//! Facade: [`execute_draw_request`] / [`execute_compute_request`] /
//! [`read_target`]. Caches L2–L7 + Lc + memory pools so a warm
//! identical static key performs zero `vkCreate*` and zero `vkAllocateMemory` on
//! the product path.

#![allow(unsafe_op_in_unsafe_fn)]

mod buffer_slab;
mod caches;
mod compute_execution;
mod compute_validation;
mod context;
mod counters;
mod desc_arena;
mod device_lost;
mod digest;
mod draw_execution;
mod draw_phase;
mod draw_preparation;
pub(crate) mod draw_validation;
mod driver_breadcrumb;
mod exec;
mod exec_compute;
mod facade_decline;
mod guest_scatter;
mod host_ram;
pub mod init_decline;
mod linear_target_import;
mod pools;
mod queue_owner;
mod scatter_shader;
pub(crate) mod stamp_completion;
/// The requested draw-time buffer-gather working set. Re-exported for the same
/// reason: `pools` is private, and the number this reports is what a content
/// cache on that rail would have to be sized from.
pub use pools::buffer_gather_working_set::census as buffer_gather_working_set_census;
/// The requested sampled working set, re-exported for the same reason and to
/// the same place: `pools` is private, and this line is only interpretable
/// beside the eviction routes the census already emits.
pub use pools::sampled_working_set::census as sampled_working_set_census;
/// Reference interval used only to keep reuse-distance census bands stable.
/// Residency policy does not read it.
pub(crate) use pools::IDLE_MAINTENANCE_START_MS;
pub mod gather_phase;
pub mod gpu_span;
pub mod reason;
mod slab;
pub mod stage_phase;
pub mod types;
pub mod vk_call;
#[cfg(feature = "host-window")]
mod window_present;

pub use context::MAX_DEVICE_RECREATES;
pub(crate) use counters::{CounterSnapshot, EngineCounters};
pub(crate) use draw_phase::take_window as draw_phase_window;
pub(crate) use draw_preparation::DrawPreparationDecline;
pub(crate) use facade_decline::EngineFacadeDecline;
pub(crate) use host_ram::GuestWriteDecline;
pub use types::viewport_slot_count;
pub use types::{
    BlendFactor, BlendOp, BlendStateResource, BufferContent, ColorWriteMask, ComputeBufferResource,
    ComputeOutput, ComputeRequest, ComputeResidentSampleBind, ComputeSampledImageResource,
    ComputeStorageImageResource, ComputeStorageResidency, CullMode, DepthClipMode, DepthState,
    DrawError, DrawOutput, DrawRequest, FillMode, GuestRun, GuestRunSource, GuestSampledBacking,
    GuestTargetBacking, GuestTargetMemory, GuestTargetSeed, IndexType, IndexedDrawResource,
    PipelineObjectIdentity, PrimitiveTopology, SampledByteOrigin, SampledContentIdentity,
    SampledImageResource, SampledSource, SamplerAddressMode, SamplerBorderColor,
    SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource, ScissorResource,
    SecondaryColorTarget, SeedOrder,
    StencilFaceOps, StencilOp, StencilState, StorageBufferResource, StorageImageFormat,
    TargetIdentity, TargetKeyDivergence, VertexAttributeFormat, VertexAttributeResource, VertexStepFunction,
    ViewportResource, VisibilityResultMode, WindowPresentSource, COLOR_INPUT_BINDING,
};
pub(crate) use vk_call::{VkCall, VkOp};
#[cfg(feature = "host-window")]
pub(crate) use window_present::{WindowCpuFrame, WindowPresentOutcome};

use caches::ObjectCaches;
use context::ContextOwner;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pools::ResourcePools;
use std::sync::atomic::{AtomicU64, Ordering};
use types::ComputeError;

/// The colour aspect of a single-mip, single-layer image — the shape of every
/// image this engine creates.
///
/// Twenty-two barriers, copies and views across the engine spelled it out
/// longhand, and the longhand is five fields of which four are zero or one. A
/// `level_count: 0` typed once is a barrier that covers nothing, and the shape
/// carries nothing a reader could check it against, so the only defence is not
/// writing it out again. Callers with a real array range or a depth aspect
/// still spell theirs out; those are saying something.
pub(crate) fn color_subresource_range() -> ash::vk::ImageSubresourceRange {
    ash::vk::ImageSubresourceRange {
        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Whether a registry resident of this format is a depth(-stencil) attachment
/// rather than a colour one.
///
/// The two depth formats this device ever creates are
/// [`crate::backend::vulkan::translate::pixel::TRANSIENT_DEPTH_FORMAT`] and
/// whichever combined format `DeviceContext::depth_stencil_format` selected, and
/// that second one is device-queried — so this asks the format's own aspect
/// rather than comparing against a list a third format would not be on.
pub(crate) fn format_is_depth(format: ash::vk::Format) -> bool {
    matches!(
        format,
        ash::vk::Format::D16_UNORM
            | ash::vk::Format::X8_D24_UNORM_PACK32
            | ash::vk::Format::D32_SFLOAT
            | ash::vk::Format::D16_UNORM_S8_UINT
            | ash::vk::Format::D24_UNORM_S8_UINT
            | ash::vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Whether a format carries a stencil aspect as well as depth.
pub(crate) fn format_has_stencil(format: ash::vk::Format) -> bool {
    matches!(
        format,
        ash::vk::Format::D16_UNORM_S8_UINT
            | ash::vk::Format::D24_UNORM_S8_UINT
            | ash::vk::Format::D32_SFLOAT_S8_UINT
            | ash::vk::Format::S8_UINT
    )
}

/// The subresource range a registry resident's view is created with, derived
/// from its format. See [`registry_target_usage`] for why these are functions of
/// the format rather than parameters beside it.
pub(crate) fn registry_subresource_range(
    format: ash::vk::Format,
) -> ash::vk::ImageSubresourceRange {
    if !format_is_depth(format) {
        return color_subresource_range();
    }
    let mut aspect = ash::vk::ImageAspectFlags::DEPTH;
    if format_has_stencil(format) {
        aspect |= ash::vk::ImageAspectFlags::STENCIL;
    }
    ash::vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// The usage set a registry resident's image is created with, derived from its
/// format.
///
/// **A function of the format, deliberately, and not a parameter.** The recycle
/// free-list buckets displaced images by `(geometry, format)` and hands one back
/// to the next resident of that bucket, so a bucket whose members disagree about
/// usage would eventually bind an image to an attachment it was not created for
/// — invalid, and invalid in the quiet way, because the image is real and the
/// geometry matches. Deriving usage here makes "same bucket implies same usage"
/// true by construction, so a fourth creation site cannot get it wrong and
/// nothing has to scan for one that did.
pub(crate) fn registry_target_usage(format: ash::vk::Format) -> ash::vk::ImageUsageFlags {
    if format_is_depth(format) {
        // A depth resident is only ever attachment N of an ad-hoc framebuffer.
        // No SAMPLED and no TRANSFER: nothing in this device reads a depth
        // buffer back or copies one, and asking for usage the host need not
        // support for a depth format would refuse the image on hosts that
        // support the attachment alone.
        ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    } else {
        ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
            | ash::vk::ImageUsageFlags::INPUT_ATTACHMENT
            | ash::vk::ImageUsageFlags::TRANSFER_SRC
            | ash::vk::ImageUsageFlags::TRANSFER_DST
            | ash::vk::ImageUsageFlags::SAMPLED
    }
}

/// [`color_subresource_range`] as a copy's subresource selector: colour aspect,
/// base mip, single layer.
pub(crate) fn color_subresource_layers() -> ash::vk::ImageSubresourceLayers {
    ash::vk::ImageSubresourceLayers {
        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    }
}

struct EngineState {
    owner: ContextOwner,
    caches: ObjectCaches,
    pools: ResourcePools,
    counters: EngineCounters,
    #[cfg(feature = "host-window")]
    window_presenter: Option<window_present::WindowPresenter>,
}

impl EngineState {
    fn new() -> Self {
        Self {
            owner: ContextOwner::new(),
            caches: ObjectCaches::new(),
            pools: ResourcePools::new(),
            counters: EngineCounters::default(),
            #[cfg(feature = "host-window")]
            window_presenter: None,
        }
    }

    /// Everything a `VK_ERROR_DEVICE_LOST` obliges this device to do, in one
    /// place: count it, poison the context, drop everything derived from the old
    /// `VkDevice`, and bring a new one up.
    ///
    /// # Why this is a method and not eight lines at the call site
    ///
    /// It *was* eight lines at the call site, twice — the draw path and the
    /// compute path — and every other way of observing a lost device had none of
    /// them. `read_target`, `quiesce_guest_reads`, `flush_batched_draws`,
    /// `retire_all`, `window_present_frame` and the stamp-completion thread all
    /// propagated `DeviceLost` upward without poisoning anything, so the engine
    /// went on submitting to a dead device for the rest of the boot.
    ///
    /// A driven macos-11 boot is what named it. The Maps leg's loss surfaced in
    /// the stamp-completion thread's `vkWaitSemaphores` as
    /// `stamp_wait_failed err=ERROR_DEVICE_LOST`; nothing poisoned, nothing
    /// recreated — `vk_device_recreate_proven` appears zero times in that boot —
    /// and every leg after it reported **`draws=0`** with a clean fail channel
    /// and a healthy 1 s census cadence. A device that is dead and does not know
    /// it looks exactly like a guest that stopped asking.
    ///
    /// Returns whether a fresh context came up, so a caller that can retry knows
    /// whether retrying is worth anything.
    fn on_device_lost(&mut self) -> bool {
        self.counters.device_lost.fetch_add(1, Ordering::Relaxed);
        self.owner.mark_device_lost();
        self.flush_device_derived();
        let EngineState {
            ref mut owner,
            ref counters,
            ..
        } = self;
        match owner.ensure(counters) {
            Ok(_) => true,
            Err(error) => {
                crate::observe::Emit::decline("vk_device_recreate", &error).fail_once(1);
                false
            }
        }
    }

    /// Drop everything derived from the current `VkDevice`, so the caller can
    /// bring a new one up under it.
    ///
    /// Every *guest* resource comes back by itself after this: the registry is
    /// keyed by a `TargetIdentity` the guest re-presents and `registry_ensure`
    /// recreates whatever is missing, and pipelines, modules, passes and
    /// samplers re-derive through `ObjectCaches` on next use. The presenter is
    /// the exception — its only constructor is on the window thread — so this
    /// takes it and publishes that it is gone, and
    /// `host_window::present::App::reattach_engine` is what builds it again.
    fn flush_device_derived(&mut self) {
        clear_device_capabilities();
        RESIDENT_RESOURCE_EPOCH.fetch_add(1, Ordering::Release);
        // Taken and published unconditionally, before anything fallible. The
        // presenter's swapchain, surface and semaphores were made against the
        // device that is going away; whether or not there is still a `ctx` to
        // destroy them through, they are not usable after this and the flag that
        // says a window rail exists must not outlive them. It did once — the
        // take lived inside the `ctx` arm and the flag was published from the
        // window thread, which had no idea this path existed — and the cost was
        // the whole rest of the boot: `window_publish` kept routing frames at a
        // presenter that was `None`.
        #[cfg(feature = "host-window")]
        let presenter = self.window_presenter.take();
        #[cfg(feature = "host-window")]
        note_window_present_attached(false);
        if let Some(ctx) = self.owner.ctx.as_ref() {
            // A device-loss observer can run while an ended batch is waiting in
            // the submission owner's FIFO. The device is already unusable, so
            // no GPU wait is owed, but the queue thread must release every
            // request carrying pool handles before those pools are destroyed.
            ctx.queue_barrier();
            unsafe {
                #[cfg(feature = "host-window")]
                if let Some(mut presenter) = presenter {
                    presenter.destroy(ctx, Some(&mut self.pools));
                }
                self.caches.destroy_all(&ctx.device);
                self.pools.destroy_all(&ctx.device);
            }
        } else {
            self.caches.clear_logical();
        }
        self.pools = ResourcePools::new();
        self.caches = ObjectCaches::new();
    }
}

static ENGINE: Lazy<Mutex<EngineState>> = Lazy::new(|| Mutex::new(EngineState::new()));

/// Whether the current device owns a recorded batch that has not been
/// submitted yet.
///
/// The batch itself stays exclusively under [`ENGINE`]. This publication says
/// only whether the drain's idle-tail boundary has work that justifies entering
/// that transaction. The recording thread publishes after installing the batch
/// and clears while removing it, so `false` never permits an open batch to idle.
static BATCH_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Time of the last successful drain-tail batch submission, consumed by the
/// next batch opener. This measures whether the host tranche boundary found a
/// real idle gap or split a run that resumed immediately.
static LAST_TAIL_BATCH_FLUSH_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TailReopenBand {
    Le100Us,
    Le1Ms,
    Le4Ms,
    Le16Ms,
    Gt16Ms,
}

impl TailReopenBand {
    fn of(gap_us: u64) -> Self {
        match gap_us {
            0..=100 => Self::Le100Us,
            101..=1_000 => Self::Le1Ms,
            1_001..=4_000 => Self::Le4Ms,
            4_001..=16_000 => Self::Le16Ms,
            _ => Self::Gt16Ms,
        }
    }

    fn count(self, counters: &EngineCounters) {
        let counter = match self {
            Self::Le100Us => &counters.batch_tail_reopen_le100us,
            Self::Le1Ms => &counters.batch_tail_reopen_le1ms,
            Self::Le4Ms => &counters.batch_tail_reopen_le4ms,
            Self::Le16Ms => &counters.batch_tail_reopen_le16ms,
            Self::Gt16Ms => &counters.batch_tail_reopen_gt16ms,
        };
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(super) fn publish_batch_open(open: bool) {
    BATCH_OPEN.store(open, std::sync::atomic::Ordering::Release);
}

fn note_batch_open_after_tail(counters: &EngineCounters) {
    let flushed_at = LAST_TAIL_BATCH_FLUSH_US.swap(0, std::sync::atomic::Ordering::AcqRel);
    if flushed_at == 0 {
        return;
    }
    let gap = crate::observe::elapsed_us().saturating_sub(flushed_at);
    TailReopenBand::of(gap).count(counters);
}

#[cfg(test)]
mod tail_reopen_band_tests {
    use super::TailReopenBand;

    #[test]
    fn every_boundary_enters_exactly_one_elapsed_time_band() {
        use TailReopenBand::*;
        assert_eq!(TailReopenBand::of(0), Le100Us);
        assert_eq!(TailReopenBand::of(100), Le100Us);
        assert_eq!(TailReopenBand::of(101), Le1Ms);
        assert_eq!(TailReopenBand::of(1_000), Le1Ms);
        assert_eq!(TailReopenBand::of(1_001), Le4Ms);
        assert_eq!(TailReopenBand::of(4_000), Le4Ms);
        assert_eq!(TailReopenBand::of(4_001), Le16Ms);
        assert_eq!(TailReopenBand::of(16_000), Le16Ms);
        assert_eq!(TailReopenBand::of(16_001), Gt16Ms);
    }
}

fn end_of_tranche_requires_engine(batch_open: bool, device_lost: bool) -> bool {
    batch_open || device_lost
}

/// Lock-free publication of the immutable answers attached to the current
/// Vulkan device.
///
/// These values are read while a draw request is being assembled, before the
/// request enters [`execute_draw_request`]'s engine transaction. Taking the
/// global engine lock for each answer serialized ordinary CPU preparation with
/// presentation even though none of the answers can change while that device
/// exists. One packed word makes publication atomic across device replacement:
/// readers see either the old device, the conservative no-device snapshot, or
/// the newly created device, never a mixture of their fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceCapabilitySnapshot(u64);

const CAP_MAX_DIMENSION_BITS: u32 = u32::BITS;
const CAP_LAYOUT_SHIFT: u32 = CAP_MAX_DIMENSION_BITS;
const CAP_GPU_ONLY_BIT: u32 =
    CAP_LAYOUT_SHIFT + crate::contract::pixel_format::TexelLayout::ALL.len() as u32;
const CAP_SAMPLED_FILTER_SHIFT: u32 = CAP_GPU_ONLY_BIT + 1;
const CAP_PUBLISHED_BIT: u32 =
    CAP_SAMPLED_FILTER_SHIFT + crate::contract::pixel_format::TexelLayout::ALL.len() as u32;
const _: () = assert!(CAP_PUBLISHED_BIT < u64::BITS);

impl DeviceCapabilitySnapshot {
    const CONSERVATIVE: Self =
        Self(crate::backend::vulkan::caps::device_features::VULKAN_MIN_IMAGE_DIMENSION_2D as u64);

    fn from_device(ctx: &context::DeviceContext) -> Self {
        Self::from_parts(&ctx.features, ctx.caps.quirks)
    }

    fn from_parts(
        features: &crate::backend::vulkan::caps::device_features::DeviceFeatures,
        quirks: crate::backend::vulkan::caps::DriverQuirk,
    ) -> Self {
        let mut word = u64::from(features.max_image_dimension_2d);
        for (index, supported) in features.color_attachment_blend.iter().enumerate() {
            if *supported {
                word |= 1_u64 << (CAP_LAYOUT_SHIFT + index as u32);
            }
        }
        if !quirks.guest_pages_stay_authoritative {
            word |= 1_u64 << CAP_GPU_ONLY_BIT;
        }
        for (index, supported) in features.sampled_linear_filter.iter().enumerate() {
            if *supported {
                word |= 1_u64 << (CAP_SAMPLED_FILTER_SHIFT + index as u32);
            }
        }
        word |= 1_u64 << CAP_PUBLISHED_BIT;
        Self(word)
    }

    fn max_render_target_dimension(self) -> u32 {
        self.0 as u32
    }

    fn render_target_layout_supported(
        self,
        layout: crate::contract::pixel_format::TexelLayout,
    ) -> bool {
        self.0 & (1_u64 << (CAP_LAYOUT_SHIFT + layout.index() as u32)) != 0
    }

    fn deferred_gpu_only_content_allowed(self) -> bool {
        self.0 & (1_u64 << CAP_GPU_ONLY_BIT) != 0
    }

    fn sampled_layout_linear_filter_if_published(
        self,
        layout: crate::contract::pixel_format::TexelLayout,
    ) -> Option<bool> {
        (self.0 & (1_u64 << CAP_PUBLISHED_BIT) != 0)
            .then_some(self.0 & (1_u64 << (CAP_SAMPLED_FILTER_SHIFT + layout.index() as u32)) != 0)
    }
}

static DEVICE_CAPABILITIES: AtomicU64 = AtomicU64::new(DeviceCapabilitySnapshot::CONSERVATIVE.0);

fn device_capabilities() -> DeviceCapabilitySnapshot {
    DeviceCapabilitySnapshot(DEVICE_CAPABILITIES.load(Ordering::Acquire))
}

pub(super) fn publish_device_capabilities(ctx: &context::DeviceContext) {
    DEVICE_CAPABILITIES.store(
        DeviceCapabilitySnapshot::from_device(ctx).0,
        Ordering::Release,
    );
}

fn clear_device_capabilities() {
    DEVICE_CAPABILITIES.store(DeviceCapabilitySnapshot::CONSERVATIVE.0, Ordering::Release);
}

#[cfg(test)]
mod device_capability_snapshot_tests {
    use super::*;
    use crate::contract::pixel_format::TexelLayout;

    #[test]
    fn the_no_device_snapshot_is_the_conservative_contract() {
        let snapshot = DeviceCapabilitySnapshot::CONSERVATIVE;
        assert_eq!(
            snapshot.max_render_target_dimension(),
            crate::backend::vulkan::caps::device_features::VULKAN_MIN_IMAGE_DIMENSION_2D
        );
        assert!(!snapshot.render_target_layout_supported(TexelLayout::Rgba16Float));
        assert!(!snapshot.deferred_gpu_only_content_allowed());
        assert_eq!(
            snapshot.sampled_layout_linear_filter_if_published(TexelLayout::Rgba16Float),
            None
        );
    }

    #[test]
    fn one_word_preserves_every_published_device_answer() {
        let mut features = crate::backend::vulkan::caps::device_features::DeviceFeatures {
            max_image_dimension_2d: 16_384,
            ..Default::default()
        };
        features.color_attachment_blend[TexelLayout::Rgba16Float.index()] = true;
        features.sampled_linear_filter[TexelLayout::Rgba16Float.index()] = true;

        let snapshot = DeviceCapabilitySnapshot::from_parts(
            &features,
            crate::backend::vulkan::caps::DriverQuirk::default(),
        );
        assert_eq!(snapshot.max_render_target_dimension(), 16_384);
        assert!(snapshot.render_target_layout_supported(TexelLayout::Rgba16Float));
        assert!(!snapshot.render_target_layout_supported(TexelLayout::Rg16Float));
        assert!(snapshot.deferred_gpu_only_content_allowed());
        assert_eq!(
            snapshot.sampled_layout_linear_filter_if_published(TexelLayout::Rgba16Float),
            Some(true)
        );
        assert_eq!(
            snapshot.sampled_layout_linear_filter_if_published(TexelLayout::Rg16Float),
            Some(false)
        );

        let narrowed = DeviceCapabilitySnapshot::from_parts(
            &features,
            crate::backend::vulkan::caps::DriverQuirk {
                guest_pages_stay_authoritative: true,
                ..Default::default()
            },
        );
        assert!(!narrowed.deferred_gpu_only_content_allowed());
    }
}

/// Generation of the resident registry resource handles refer into.
///
/// Reset and device recreation replace the whole registry. A resource object
/// may outlive either operation, so its lease carries this generation and
/// treats a mismatch as an absent handle. Release also checks it under the
/// engine lock, preventing an old lease from unpinning a new resident that
/// happens to reuse the same identity.
static RESIDENT_RESOURCE_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Which thread class is asking for the engine lock.
///
/// The single `ENGINE` mutex serializes the drain worker's guest execution
/// against the host window's present, and only one direction of that
/// contention reaches the screen: a worker delayed by the window loses
/// throughput it can make up, while a window delayed by the worker drops the
/// frame it was about to show. `engine_lock` cannot say which side paid without
/// the two being named apart, so every acquire declares itself.
///
/// # Why there are three and not two
///
/// [`Self::Worker`] used to mean "the drain worker **and** every entry point
/// QEMU reaches that is not the window", which is three populations on one
/// counter and the one reading nobody could take from it. The drain worker owns
/// the lock for a whole tranche — 28-45 ms on a driven x86 boot, 117 ms at the
/// tail — and the threads that queue behind it are not peers of each other:
///
/// * The **drain worker** blocking is throughput it makes up on the next
///   tranche.
/// * A **vCPU** blocking is the guest stopped dead inside an MMIO store, and
///   every other emulated device's timing goes with it. That is the population
///   an audio underrun or a late timer is a symptom of.
/// * QEMU's **main loop** blocking inside the action BH stalls every device in
///   the process, not just this one.
///
/// The last two are both [`Self::Device`]: this crate cannot tell a vCPU thread
/// from the main loop without QEMU telling it, and the actionable split is
/// "the render tranche" against "everything QEMU needed while it ran".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineLockSite {
    /// The drain worker executing guest commands. Recognised by the thread
    /// having entered [`crate::qemu::abi::reims_vgpu_qemu_device_drain`] at
    /// least once, which is a property no other thread has.
    Worker,
    /// Every other entry point QEMU reaches: a vCPU inside an MMIO store, the
    /// main loop inside the action BH, poll, reset, teardown.
    Device,
    /// The host window's event loop: present, attach, resize, detach.
    Window,
}

impl EngineLockSite {
    fn index(self) -> usize {
        match self {
            Self::Worker => 0,
            Self::Device => 1,
            Self::Window => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Worker => "worker",
            Self::Device => "device",
            Self::Window => "window",
        }
    }

    const ALL: [Self; 3] = [Self::Worker, Self::Device, Self::Window];
}

thread_local! {
    /// Whether this thread has ever run a drain.
    ///
    /// Latched rather than scoped: a thread that has drained once is the drain
    /// worker for the process's life on both shims, and a scoped marker would
    /// have to be restored on every early return out of a `?`-heavy call tree.
    /// A test process that drains from its own thread labels that thread the
    /// worker, which is what it is for the duration.
    static IS_DRAIN_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the calling thread as the drain worker. Called by the drain entry point
/// before it takes the lock, so the first tranche is attributed correctly.
pub(crate) fn mark_drain_thread() {
    IS_DRAIN_THREAD.with(|c| c.set(true));
}

/// Which site a `lock_engine()` acquire belongs to, from the calling thread.
fn calling_site() -> EngineLockSite {
    if IS_DRAIN_THREAD.with(std::cell::Cell::get) {
        EngineLockSite::Worker
    } else {
        EngineLockSite::Device
    }
}

/// Wait-to-acquire and hold time on `ENGINE`, split by [`EngineLockSite`].
///
/// Both halves are needed to read either. A window `wait_us` that owns its
/// second says the window is blocked; the worker's `hold_us` beside it says
/// whether the worker is what blocked it, and `hold_max_us` whether that was
/// one long hold or many short ones. Neither is derivable from `drain_duty`,
/// which times the device lock rather than this one and cannot see the window
/// thread at all.
#[derive(Default)]
struct EngineLockCensus {
    /// Acquires that took the mutex with no wait, per site.
    uncontended: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
    /// Acquires that found it held and had to block, per site.
    contended: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
    /// Wall clock blocked on the mutex, summed over `contended`.
    wait_us: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
    wait_max_us: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
    /// Wall clock from acquire to release, over every acquire.
    hold_us: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
    hold_max_us: [std::sync::atomic::AtomicU64; EngineLockSite::ALL.len()],
}

static ENGINE_LOCK: EngineLockCensus = EngineLockCensus::new();

impl EngineLockCensus {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    const fn new() -> Self {
        Self {
            uncontended: [Self::ZERO; EngineLockSite::ALL.len()],
            contended: [Self::ZERO; EngineLockSite::ALL.len()],
            wait_us: [Self::ZERO; EngineLockSite::ALL.len()],
            wait_max_us: [Self::ZERO; EngineLockSite::ALL.len()],
            hold_us: [Self::ZERO; EngineLockSite::ALL.len()],
            hold_max_us: [Self::ZERO; EngineLockSite::ALL.len()],
        }
    }

    fn note_wait(&self, site: EngineLockSite, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = site.index();
        self.contended[i].fetch_add(1, Relaxed);
        self.wait_us[i].fetch_add(us, Relaxed);
        self.wait_max_us[i].fetch_max(us, Relaxed);
    }

    fn note_uncontended(&self, site: EngineLockSite) {
        self.uncontended[site.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn note_hold(&self, site: EngineLockSite, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = site.index();
        self.hold_us[i].fetch_add(us, Relaxed);
        self.hold_max_us[i].fetch_max(us, Relaxed);
    }

    /// Drain the window into one line, or `None` when the lock was never taken
    /// in it (a boot with no engine work at all).
    fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let mut body = String::new();
        let mut any = false;
        for site in EngineLockSite::ALL {
            let i = site.index();
            let free = self.uncontended[i].swap(0, Relaxed);
            let blocked = self.contended[i].swap(0, Relaxed);
            let wait_us = self.wait_us[i].swap(0, Relaxed);
            let wait_max_us = self.wait_max_us[i].swap(0, Relaxed);
            let hold_us = self.hold_us[i].swap(0, Relaxed);
            let hold_max_us = self.hold_max_us[i].swap(0, Relaxed);
            any |= free != 0 || blocked != 0;
            let label = site.label();
            body.push_str(&format!(
                " {label}={} {label}_blocked={blocked} {label}_wait_us={wait_us} \
                 {label}_wait_max_us={wait_max_us} {label}_hold_us={hold_us} \
                 {label}_hold_max_us={hold_max_us}",
                free + blocked
            ));
        }
        any.then(|| format!("engine_lock win_ms={win_ms}{body}"))
    }
}

/// The window `drain_duty` last reported over, drained into one `engine_lock`
/// line. Called from the drain's per-second census block so it shares that
/// denominator rather than deriving a second one.
pub(crate) fn take_engine_lock_census(win_ms: u64) -> Option<String> {
    ENGINE_LOCK.take(win_ms)
}

/// A held engine lock that reports how long it was held.
///
/// Derefs to [`EngineState`], so a call site reads exactly as it did against
/// `parking_lot::MutexGuard`. The hold is timed on release rather than sampled,
/// because the spans that matter here are the long ones — a readback fence
/// inside the lock — and a sampler would miss precisely those.
struct EngineGuard {
    guard: parking_lot::MutexGuard<'static, EngineState>,
    site: EngineLockSite,
    acquired: std::time::Instant,
}

impl std::ops::Deref for EngineGuard {
    type Target = EngineState;
    fn deref(&self) -> &EngineState {
        &self.guard
    }
}

impl std::ops::DerefMut for EngineGuard {
    fn deref_mut(&mut self) -> &mut EngineState {
        &mut self.guard
    }
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        ENGINE_LOCK.note_hold(self.site, self.acquired.elapsed().as_micros() as u64);
    }
}

/// Acquire the global engine lock. The single `ENGINE` mutex serializes all 34
/// engine entry points across the drain worker and the QEMU main/present path,
/// so this is on every one of them.
///
/// The uncontended path reads no clock beyond the one `Instant::now` the hold
/// timer needs: `try_lock` decides whether a wait happened, so an acquire that
/// did not block costs a failed-then-taken atomic and nothing else.
#[inline]
fn lock_engine_at(site: EngineLockSite) -> EngineGuard {
    let guard = match ENGINE.try_lock() {
        Some(guard) => {
            ENGINE_LOCK.note_uncontended(site);
            guard
        }
        None => {
            let blocked_at = std::time::Instant::now();
            let guard = ENGINE.lock();
            ENGINE_LOCK.note_wait(site, blocked_at.elapsed().as_micros() as u64);
            guard
        }
    };
    EngineGuard {
        guard,
        site,
        acquired: std::time::Instant::now(),
    }
}

/// [`lock_engine_at`] for the drain worker and the QEMU entry points, which is
/// every caller but the host window's event loop.
///
/// Which of the two it is comes from the calling thread rather than from the
/// call site: the same functions are reached from a drain tranche and from a
/// vCPU's MMIO store, so no fixed site could name both correctly. See
/// [`EngineLockSite`] for why telling them apart is the whole point.
#[inline]
fn lock_engine() -> EngineGuard {
    lock_engine_at(calling_site())
}

/// Device-reset proxy: guest-derived Vulkan objects evicted at the lifetime boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GuestResetStats {
    pub resident_targets: usize,
    pub pooled_targets: usize,
    pub sampled_images: usize,
    pub storage_images: usize,
    pub had_context: bool,
}

/// Drop guest-identity/resource state while preserving the Vulkan context and
/// immutable content-keyed shader/pipeline caches.
pub fn reset_guest_state() -> GuestResetStats {
    let mut guard = lock_engine();
    LAST_TAIL_BATCH_FLUSH_US.store(0, std::sync::atomic::Ordering::Release);
    RESIDENT_RESOURCE_EPOCH.fetch_add(1, Ordering::Release);
    let (resident_targets, pooled_targets, sampled_images, storage_images) =
        guard.pools.guest_reset_counts();
    let stats = GuestResetStats {
        resident_targets,
        pooled_targets,
        sampled_images,
        storage_images,
        had_context: guard.owner.ctx.is_some(),
    };
    let EngineState {
        ref owner,
        ref mut pools,
        #[cfg(feature = "host-window")]
        ref mut window_presenter,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        if let Err(error) = ctx.queue_wait_idle() {
            let decline = VkCall::new(VkOp::GuestResetDeviceWaitIdle, error);
            crate::observe::Emit::decline("vulkan_guest_reset", &decline).fail_once(0);
        }
        unsafe {
            #[cfg(feature = "host-window")]
            if let Some(presenter) = window_presenter.as_mut() {
                presenter.release_pins_after_idle(pools);
            }
            pools.destroy_all(&ctx.device);
        }
    }
    *pools = ResourcePools::new();
    crate::observe::off(format!(
        "vulkan_guest_reset resident={} pooled_targets={} sampled={} storage={} context={}",
        stats.resident_targets,
        stats.pooled_targets,
        stats.sampled_images,
        stats.storage_images,
        u8::from(stats.had_context)
    ));
    stats
}

/// Ensure the macOS host-window surface and swapchain exist on the engine's
/// Vulkan instance/device.
#[cfg(feature = "host-window")]
pub fn window_present_attach(
    display: raw_window_handle::RawDisplayHandle,
    window: raw_window_handle::RawWindowHandle,
    width: u32,
    height: u32,
) -> Result<(), DrawError> {
    let mut guard = lock_engine_at(EngineLockSite::Window);
    let EngineState {
        ref mut owner,
        ref counters,
        ref mut window_presenter,
        ..
    } = &mut *guard;
    if window_presenter.is_some() {
        return Ok(());
    }
    let ctx = owner.ensure(counters)?;
    *window_presenter = Some(unsafe {
        window_present::WindowPresenter::create(ctx, display, window, width, height)?
    });
    note_window_present_attached(true);
    Ok(())
}

/// Whether the host window is presenting from the engine's own device.
///
/// Read by the present-capture path on the drain worker, which must decide
/// whether to read the finished frame back into host memory *before* it does so
/// — deciding at publish time leaves the readback already paid for. A relaxed
/// atomic rather than [`lock_engine`] because that call site runs once per
/// present on the only thread that executes guest work, and taking the engine
/// lock there to read one bit would serialize it against the window thread's
/// own present.
#[cfg(feature = "host-window")]
static WINDOW_PRESENT_ATTACHED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Publish the window's rail choice.
///
/// Private, and called from exactly the three places that assign or take
/// [`EngineState::window_presenter`]. It used to be `pub`, published by the
/// window thread beside its own `window_present_attach` call — and the third
/// site, [`EngineState::flush_device_derived`], did not know it existed. So a
/// device loss destroyed the presenter and left this flag reading `true` for the
/// rest of the boot: `window_publish` kept routing to a rail that was gone,
/// every present after the loss failed
/// `EngineFacadeDecline::WindowPresenterNotAttached`, and because the window
/// thread latches its error log the whole thing was one line. The flag and the
/// field are one fact now, and no caller outside this module can set one
/// without the other.
#[cfg(feature = "host-window")]
fn note_window_present_attached(attached: bool) {
    WINDOW_PRESENT_ATTACHED.store(attached, Ordering::Release);
}

#[cfg(feature = "host-window")]
pub fn window_present_attached() -> bool {
    WINDOW_PRESENT_ATTACHED.load(Ordering::Acquire)
}

#[cfg(feature = "host-window")]
pub fn window_present_resize(width: u32, height: u32) {
    let mut guard = lock_engine_at(EngineLockSite::Window);
    if let Some(presenter) = guard.window_presenter.as_mut() {
        presenter.resize(width, height);
    }
}

/// Present the current compositor resident through the engine-owned swapchain,
/// falling back to `cpu` for presents no resident carries. Acquire is
/// nonblocking, so a vblank wait never holds `ENGINE`.
#[cfg(feature = "host-window")]
pub fn window_present_frame(
    source: Option<&WindowPresentSource>,
    cpu: Option<WindowCpuFrame<'_>>,
    cursor: Option<&crate::host_window::present::CursorOverlay>,
) -> Result<WindowPresentOutcome, DrawError> {
    let dispatch = {
        let mut guard = lock_engine_at(EngineLockSite::Window);
        let EngineState {
            ref mut owner,
            ref mut pools,
            ref counters,
            ref mut window_presenter,
            ..
        } = &mut *guard;
        let ctx = owner.ensure(counters)?;
        let presenter = window_presenter.as_mut().ok_or(DrawError::Facade(
            EngineFacadeDecline::WindowPresenterNotAttached,
        ))?;
        unsafe { presenter.begin_present(ctx, pools, counters, source, cpu, cursor) }?
    };
    let out = match dispatch {
        window_present::WindowPresentDispatch::Complete(out) => Ok(out),
        window_present::WindowPresentDispatch::Pending(pending) => {
            // The display transaction is already ordered behind every guest
            // batch accepted above and owns its resident pin.  Waiting for the
            // host driver while holding ENGINE would only stop the drain from
            // preparing later work; no resource-state decision remains here.
            let finished = pending.wait();
            let mut guard = lock_engine_at(EngineLockSite::Window);
            match guard.window_presenter.as_mut() {
                None => Err(DrawError::Facade(
                    EngineFacadeDecline::WindowPresenterNotAttached,
                )),
                Some(presenter) => match presenter.finish_present(finished) {
                    Ok(window_present::WindowPresentDispatch::Complete(out)) => Ok(out),
                    Ok(window_present::WindowPresentDispatch::Pending(_)) => unreachable!(
                        "finishing a display transaction cannot enqueue another transaction"
                    ),
                    Err(error) => Err(error),
                },
            }
        }
    };
    // The window thread cannot run the recovery from inside this borrow, so a
    // loss seen here is latched for the drain's end-of-tranche flush. This is
    // the observer that matters most when the guest has already stopped drawing:
    // the presenter is then the only thing still touching the queue.
    if let Err(DrawError::DeviceLost(_)) = out {
        device_lost::note_device_lost_seen();
    }
    out
}

/// Destroy the engine-owned surface while the native AppKit window still
/// exists. Called from winit's `exiting` callback.
#[cfg(feature = "host-window")]
pub fn window_present_detach() {
    let mut guard = lock_engine_at(EngineLockSite::Window);
    let Some(mut presenter) = guard.window_presenter.take() else {
        return;
    };
    note_window_present_attached(false);
    let EngineState {
        ref owner,
        ref mut pools,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        unsafe { presenter.destroy(ctx, Some(pools)) };
    }
}

/// Execute one draw against the persistent engine.
pub fn execute_draw_request(req: &DrawRequest) -> Result<DrawOutput, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let result = unsafe { exec::execute_draw_inner(owner, caches, pools, counters, req) };
    if result.is_err() {
        // A draw can abandon after staging a gathered window and before the copy
        // that fills it is recorded — `SampledResidentNotReady` and its two
        // siblings are recoverable control flow raised in between. The owed copy
        // lives in a local `Vec` and goes with the stack frame; the bind memo it
        // was going to fill does not, and the next draw of the same command
        // buffer would hit that memo, record nothing, and bind the recycled
        // slot's **previous tenant** as its constant buffer or vertex stream.
        //
        // Forget exactly those entries. Counted rather than silent, because a
        // zero here is what says the window was never open on a given workload,
        // and no other counter in this device could see it.
        let n = pools.discard_cb_binds_owed_a_gather();
        if n > 0 {
            crate::runtime::drain::note_store_route_n("cb_bind_dropped_unfilled_gather", n as u64);
        }
    }
    match result {
        Ok(out) => {
            // Guest work reached the GPU, so any recreate that got us here did
            // its job and the storm budget starts over. See
            // `ContextOwner::note_work_completed`.
            guard.owner.note_work_completed();
            Ok(out)
        }
        Err(DrawError::DeviceLost(decline)) => {
            guard.on_device_lost();
            Err(DrawError::DeviceLost(decline))
        }
        Err(e) => Err(e),
    }
}

/// Submit any open deferred draw batch (draw batching increment 1). Called at
/// the end of every drain tranche so batched work never idles unsubmitted
/// while the worker sleeps; every in-engine consumer path (reads, compute,
/// prefetch, next non-joinable draw) already flushes via begin_entry, so this
/// only bounds the idle-tail latency. No-op without a context or open batch.
pub fn flush_batched_draws() {
    if !end_of_tranche_requires_engine(
        BATCH_OPEN.load(std::sync::atomic::Ordering::Acquire),
        device_lost::device_lost_seen(),
    ) {
        return;
    }
    let lock_started = std::time::Instant::now();
    let mut guard = lock_engine();
    guard.counters.batch_tail_lock_us.fetch_add(
        lock_started.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    // This runs at the end of every drain tranche, including hundreds of short
    // tranches per second under load, so it is where a loss
    // observed by a thread that cannot take the engine lock gets acted on. See
    // `device_lost::note_device_lost_seen` for the boot that needed it.
    if device_lost::take_device_lost_seen() {
        guard.on_device_lost();
        return;
    }
    let lost = {
        let EngineState {
            ref mut owner,
            ref mut pools,
            ref counters,
            ..
        } = &mut *guard;
        let Some(ctx) = owner.ctx.as_ref() else {
            return;
        };
        // The lock-free publication can race an in-engine consumer that
        // flushes before this caller acquires the lock. Only a batch observed
        // here belongs in the tail population.
        let had_batch = pools.batch_open_recording().is_some();
        let call_started = std::time::Instant::now();
        let result = unsafe { pools.batch_flush(ctx, counters) };
        counters.batch_tail_call_us.fetch_add(
            call_started.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        match result {
            Ok(()) if had_batch => {
                counters
                    .batch_tail_flushes
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                LAST_TAIL_BATCH_FLUSH_US.store(
                    crate::observe::elapsed_us(),
                    std::sync::atomic::Ordering::Release,
                );
                false
            }
            Ok(()) => false,
            Err(e) => {
                let lost = matches!(e, DrawError::DeviceLost(_));
                crate::observe::Emit::decline("vk_batch_flush", &e).fail_once(0);
                lost
            }
        }
    };
    // "A lost device surfaces again on the next draw" is only true while draws
    // keep coming, and a device loss is one of the things that stops them.
    if lost {
        guard.on_device_lost();
    }
}

#[cfg(test)]
mod end_of_tranche_gate_tests {
    use super::end_of_tranche_requires_engine;

    #[test]
    fn only_owned_batch_or_recovery_work_enters_the_engine() {
        assert!(!end_of_tranche_requires_engine(false, false));
        assert!(end_of_tranche_requires_engine(true, false));
        assert!(end_of_tranche_requires_engine(false, true));
        assert!(end_of_tranche_requires_engine(true, true));
    }
}

/// Wait until nothing this device has recorded will read guest RAM again.
///
/// Called from [`crate::runtime::drain::write_stamp`], immediately before the
/// guest is told a packet finished. The stamp's own contract is that everything
/// the packet named may be freed or repainted from that moment, and a draw that
/// binds guest pages as a copy source reads them when its command buffer
/// *executes* — which, on a device that acks before it runs, is otherwise after
/// the guest was told it was safe.
///
/// A no-op unless a guest-reading command buffer was actually recorded, so a
/// host that cannot import guest RAM never pays for it. See
/// [`pools::ResourcePools::quiesce_guest_reads`] for why the wait retires the
/// whole ring rather than the fences carrying the reads.
/// Whether any guest-page writeback is submitted and not yet settled, readable
/// without the engine lock.
///
/// [`quiesce_guest_writes`] is called from every host read or write of guest
/// mapping bytes, which is a far hotter set of sites than the completion stamp
/// its read-side twin serves. Taking the engine lock at each of them to discover
/// there was nothing to settle would be the cost this change is removing, so the
/// common answer is one relaxed-acquire load and no lock at all.
///
/// Set under the engine lock with `Release` after the copy is parked and before
/// it is submitted; cleared under the same lock once the ring has retired. A
/// thread that reads `false` therefore observed a point at which no writeback
/// was outstanding.
static GUEST_WRITE_DEBT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a recorded command buffer can still read imported guest RAM.
///
/// Kept beside the write-side debt so completion-stamp routing can answer the
/// common no-work case without taking the engine lock. The pool owns the ledger;
/// this atomic is only its lock-free summary.
pub(super) static GUEST_READ_DEBT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Which guest pages the outstanding writeback lands in, when it can be said.
///
/// [`GUEST_WRITE_DEBT`] answers "is anything outstanding", and every caller that
/// reads guest bytes then blocks on the answer. But a writeback lands in one
/// surface's pages and most of the readers asking are reading somewhere else
/// entirely — a glyph atlas, a small linear texture, a uniform staging window —
/// so the wait they take is for a write that will never touch a byte they read.
/// A driven Safari-drag boot spent **11.5 s** in one such reader
/// ([`crate::runtime::render_writeback::SettleSite::LinearMemoRead`]) alone.
///
/// This names the pages so a disjoint reader can be let through. The currency is
/// the guest page address, because it is the only one the two sides share: the
/// writer knows its destination as a page list from a walk, the reader knows its
/// window as a task GVA it can walk to the same list, and
/// [`GuestPageTarget::runs`] carries neither — a [`GuestRef`] deliberately does
/// not expose an absolute position.
///
/// [`GuestRef`]: crate::runtime::guest_ram_map::GuestRef
///
/// # How many writebacks are named, and what happens past that
///
/// One destination's page list per armed-and-unsettled writeback, up to
/// [`RING_DEPTH`] of them. The bound is the submission ring's, because that is
/// what bounds writebacks in flight: a ninth submission blocks in `begin_entry`
/// on the oldest fence rather than being recorded.
///
/// **The two drift, and overflow is routine.** [`GUEST_WRITE_DEBT`] is cleared
/// only by an actual settle, never by the ring retiring a fence on its own, so
/// an entry outlives the copy it names for as long as nothing blocks. A single
/// entry measured `gwdebt_unnamed` 14 125 against 16 626 arms; the ring's worth
/// still measured **3 499** overflows on a driven Safari-drag boot, and each one
/// made every subsequent reader settle globally until the next clear —
/// `settle_linear_memo_read_unnamed` 3 402 of that boot's 4 036 memo waits, at
/// ~200 ms of blocking per second of drag.
///
/// So overflow is a **merge**, not a surrender: the ninth arm folds into the
/// oldest entry and the ledger keeps naming every page it holds. A union of two
/// destination page lists is exactly the set of pages both copies land in, so
/// nothing is lost but the ability to retire one of them individually — and
/// nothing retires individually, because [`clear_guest_write_pages`] clears all
/// of them at once.
///
/// **Never by dropping an entry**: a footprint missing a page it holds would
/// answer "disjoint" for a page a copy is landing in, which is a stale frame
/// served as fresh. Merging is the opposite direction — it can only turn a
/// `Disjoint` into an `Overlap`, never the reverse.
///
/// # Ordering
///
/// Armed under the engine lock immediately before [`GUEST_WRITE_DEBT`] is
/// published, and cleared under the same lock immediately after it is cleared,
/// so a reader that observes the flag set observes a footprint that already
/// names the write, and a reader that observes it clear needs nothing. Readers
/// take only this mutex and never the engine lock — taking that at every guest
/// read is the cost the flag exists to avoid.
static GUEST_WRITE_PAGES: std::sync::Mutex<GuestWriteFootprint> =
    std::sync::Mutex::new(GuestWriteFootprint {
        armed: Vec::new(),
        allocations: Vec::new(),
    });

/// The page lists behind [`GUEST_WRITE_PAGES`].
struct GuestWriteFootprint {
    /// One entry per outstanding writeback, each ascending and deduplicated.
    /// Sorted at arm time — armed thousands of times a boot and asked tens of
    /// thousands, so the ordering is paid on the rarer side and every ask is a
    /// binary search.
    ///
    /// Kept as separate lists rather than merged into one, so a settle could
    /// retire them individually later without re-deriving which page belonged to
    /// which copy. Never longer than [`RING_DEPTH`]; past that, an arm folds
    /// into entry zero, which gives up exactly that future retirement and
    /// nothing else.
    ///
    /// There is no "gave up naming" flag beside this any more. It existed for
    /// the overflow that is now a merge, and the only other way to lose the
    /// ledger — a poisoned mutex — already answers `Unnamed` at every ask
    /// without one.
    armed: Vec<Vec<u64>>,
    /// Resource-owned allocation footprints. A repeated Store into the same
    /// admitted resident retains one immutable identity here rather than
    /// copying and sorting its full page list again. These need no artificial
    /// entry cap: each is a small `Arc`-backed handle, and the same global
    /// settle that clears `armed` clears them all.
    allocations: Vec<crate::runtime::guest_ram::GuestPageFootprint>,
}

/// What the ledger can say about a reader's window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuestWriteReach {
    /// Nothing outstanding lands in any of the pages asked about. The caller may
    /// read them without settling.
    Disjoint,
    /// An outstanding writeback lands in one of them.
    Overlap,
    /// The ledger cannot say, so the caller must settle. Distinguished from
    /// [`Self::Overlap`] because the two want opposite fixes: an overlap is a
    /// wait genuinely owed and this is precision the ledger failed to keep.
    Unnamed,
}

/// Record the guest pages a writeback about to be submitted will land in.
///
/// Called under the engine lock, beside the [`GUEST_WRITE_DEBT`] publish.
fn arm_guest_write_pages(pages: &[u64]) {
    let Ok(mut f) = GUEST_WRITE_PAGES.lock() else {
        // A poisoned lock means nothing is recorded, and it stays poisoned, so
        // `guest_writes_reaching` answers `Unnamed` for the rest of the boot.
        // That is the safe direction and needs no flag here.
        return;
    };
    let mut sorted = pages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    // At the entry cap, fold into the oldest rather than give up naming. The
    // `first_mut` cannot be `None` there — `RING_DEPTH` is non-zero — but the
    // fallthrough is a push rather than an `expect`, because the only thing a
    // panic here would protect is a bound this function does not own.
    if f.armed.len() >= pools::RING_DEPTH {
        if let Some(oldest) = f.armed.first_mut() {
            oldest.extend_from_slice(&sorted);
            oldest.sort_unstable();
            oldest.dedup();
            crate::runtime::drain::note_store_route("gwdebt_merged");
            return;
        }
    }
    f.armed.push(sorted);
}

/// Record one resource-owned allocation that an outstanding GPU Store writes.
/// Repeated writes through the same admitted resource add no ledger work.
fn arm_guest_write_footprint(
    footprint: &crate::runtime::guest_ram::GuestPageFootprint,
) {
    let Ok(mut f) = GUEST_WRITE_PAGES.lock() else {
        return;
    };
    if f.allocations
        .iter()
        .any(|held| held.same_allocation(footprint))
    {
        return;
    }
    f.allocations.push(footprint.clone());
}

/// Forget the outstanding writebacks' pages. Called under the engine lock, after
/// the wait has landed and [`GUEST_WRITE_DEBT`] is cleared.
fn clear_guest_write_pages() {
    if let Ok(mut f) = GUEST_WRITE_PAGES.lock() {
        f.armed.clear();
        f.allocations.clear();
    }
}

/// What the ledger can say about `pages` — see [`GuestWriteReach`].
///
/// [`GuestWriteReach::Unnamed`] is the safe answer and is returned whenever this
/// cannot say otherwise, including a poisoned lock. A [`GuestWriteReach::Disjoint`]
/// licenses the caller to read those guest pages without settling.
///
/// `pages` need not be sorted; it is the reader's window and is usually a
/// handful of entries against a whole frame's worth here.
pub fn guest_writes_reaching(pages: &[u64]) -> GuestWriteReach {
    let Ok(f) = GUEST_WRITE_PAGES.lock() else {
        return GuestWriteReach::Unnamed;
    };
    if f.armed.is_empty() && f.allocations.is_empty() {
        // The flag said something was outstanding and the ledger names nothing:
        // the settle that cleared it raced this ask. Nothing to rule out
        // against, so nothing may be ruled out.
        return GuestWriteReach::Unnamed;
    }
    // Span first, and the loop nesting is inverted for it. Asking "does any
    // ledger entry hold page p" per page cannot short-circuit when the answer is
    // `Disjoint`, so the verdict this test exists to produce is the one that
    // pays the full product of both sides. Asking each entry "do you reach any
    // of these pages" lets an entry whose span misses the reader's leave on two
    // comparisons, and on this workload almost every entry does: a reader names
    // one surface and the ledger holds unrelated ones.
    //
    // The span test is a narrowing of work and never of the verdict. Two spans
    // that do not overlap cannot share a page, so a rejection here is exact; two
    // that do overlap prove nothing and fall through to the per-page search.
    let Some(lo) = pages.iter().min().copied() else {
        // A reader naming no pages reads nothing, so nothing can reach it.
        return GuestWriteReach::Disjoint;
    };
    let hi = pages.iter().max().copied().unwrap_or(lo);
    let disjoint_span = |(a_lo, a_hi): (u64, u64)| a_hi < lo || a_lo > hi;
    let hit = f.armed.iter().any(|a| {
        // Ascending and deduplicated at arm time, so the span is the ends.
        match (a.first(), a.last()) {
            (Some(&a_lo), Some(&a_hi)) if !disjoint_span((a_lo, a_hi)) => {
                pages.iter().any(|p| a.binary_search(p).is_ok())
            }
            _ => false,
        }
    }) || f.allocations.iter().any(|a| {
        match a.page_span() {
            Some(span) if !disjoint_span(span) => pages.iter().any(|p| a.contains_page(*p)),
            _ => false,
        }
    });
    if hit {
        GuestWriteReach::Overlap
    } else {
        GuestWriteReach::Disjoint
    }
}

/// Wait until every guest-page writeback this device has recorded has landed in
/// guest RAM.
///
/// The write-side twin of [`quiesce_guest_reads`], and the settle point for the
/// fence `copy_target_to_guest_pages` no longer takes itself. Call it wherever a
/// reader that is not this device's own command stream is about to observe those
/// bytes: the completion stamp, and the host-side readers that call
/// `runtime::render_writeback::settle_guest_writes` before touching guest
/// mapping bytes.
pub fn quiesce_guest_writes() {
    use std::sync::atomic::Ordering;
    if !GUEST_WRITE_DEBT.load(Ordering::Acquire) {
        return;
    }
    let started = std::time::Instant::now();
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        // No device, so nothing can be in flight and nothing can settle it.
        GUEST_WRITE_DEBT.store(false, Ordering::Release);
        clear_guest_write_pages();
        return;
    };
    if let Err(e) = unsafe { pools.quiesce_guest_writes(ctx, counters) } {
        // The wait failed, so this device cannot say the frame reached the
        // guest's pages. Nothing here can hold the caller back — a stamp that
        // never moves hangs the guest — so report it, and latch a lost device
        // rather than trusting the next draw to surface it, because a loss is
        // one of the things that stops draws coming.
        if matches!(e, DrawError::DeviceLost(_)) {
            device_lost::note_device_lost_seen();
        }
        crate::observe::Emit::decline("vk_guest_write_quiesce", &e).fail_once(0);
    }
    // Cleared whether the wait succeeded or failed, for the reason
    // `ResourcePools::quiesce_guest_writes` takes its own debt before waiting:
    // the slot stays pending either way and the next claimant re-waits, so the
    // ordering survives without every later settle re-running a failing wait.
    GUEST_WRITE_DEBT.store(false, Ordering::Release);
    // Under the same lock as the flag it accompanies, so no reader can see the
    // flag set beside a footprint that has already been forgotten.
    clear_guest_write_pages();
    // Reported as `ReadbackPhase::Fence` because it *is* that phase — the same
    // block on the same fences, moved. Its count is now settles rather than
    // windows, which is the whole of what this change did to the rail, so
    // `fence` no longer tracks `submit` and a reading that assumes it does is
    // reading the old shape.
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Fence,
        started.elapsed().as_micros() as u64,
    );
}

/// Whether any guest-page writeback is submitted and not yet settled.
///
/// The same flag [`quiesce_guest_writes`] short-circuits on. Host-side readers
/// use this to avoid taking the engine lock when no GPU write can race them.
pub fn guest_writes_outstanding() -> bool {
    GUEST_WRITE_DEBT.load(std::sync::atomic::Ordering::Acquire)
}

/// Whether a completion stamp must be queued behind guest-memory access.
pub fn guest_access_outstanding() -> bool {
    GUEST_READ_DEBT.load(std::sync::atomic::Ordering::Acquire) || guest_writes_outstanding()
}

/// Where one FIFO completion stamp obtained its ordering point.
///
/// A batched stamp waits for the open batch's eventual submission; otherwise
/// the stamp reuses the newest successful FIFO submission. Keeping it as a type
/// makes the counter choice exhaustive and gives the rule a device-free
/// behavioural test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampPointSource {
    Batch,
    PressureFlush,
    Reused,
}

impl StampPointSource {
    fn from_route(open: bool, deferred: bool) -> Self {
        match (open, deferred) {
            (true, true) => Self::Batch,
            (true, false) => Self::PressureFlush,
            (false, _) => Self::Reused,
        }
    }

    fn count(self, counters: &EngineCounters) {
        let counter = match self {
            Self::Batch => &counters.gpu_stamp_batch_points,
            Self::PressureFlush => &counters.gpu_stamp_pressure_flushes,
            Self::Reused => &counters.gpu_stamp_reused_points,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod stamp_point_source_tests {
    use super::*;

    #[test]
    fn a_deferred_batch_and_a_reused_point_are_counted_apart() {
        let counters = EngineCounters::default();
        StampPointSource::from_route(true, true).count(&counters);
        StampPointSource::from_route(true, false).count(&counters);
        StampPointSource::from_route(false, false).count(&counters);
        StampPointSource::from_route(false, false).count(&counters);

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.gpu_stamp_batch_points, 1);
        assert_eq!(snapshot.gpu_stamp_pressure_flushes, 1);
        assert_eq!(snapshot.gpu_stamp_reused_points, 2);
        assert_eq!(
            snapshot.gpu_stamp_batch_points
                + snapshot.gpu_stamp_pressure_flushes
                + snapshot.gpu_stamp_reused_points,
            4,
            "every successful GPU stamp has exactly one completion-point source"
        );
    }
}

/// Queue a completion stamp behind the FIFO work this device accepted, without
/// blocking the drain worker.
///
/// # The ordering this buys, and the one it does not
///
/// A completion stamp is a queue fence: the guest may not observe it until the
/// commands preceding it have completed. That includes writes into guest pages
/// and reads from guest pages the guest may reuse after observing the stamp.
/// The blocking rail retires the queue and then stores the word from the CPU.
/// This rail gives every FIFO-owned submission a monotonic timeline point. If a
/// draw batch is still recording, the stamp enters the bounded pending queue
/// without closing it; the batch's eventual successful submit assigns the
/// point. The completion thread waits that point, release-stores the word
/// through its checked [`crate::runtime::guest_ram::GuestRef`], and raises the
/// interrupt. Queue order makes one point cover every earlier guest-memory read
/// and write without manufacturing a submission or a command-buffer boundary
/// for the four-byte stamp.
///
/// **It does not order the interrupt**, and that is not an oversight. The guest
/// reads the stamp word directly and sleeps on it with a one-second deadline, so
/// the interrupt is its wakeup rather than a hint. The FIFO submission signals
/// a timeline value and `stamp_completion`'s thread publishes and announces the
/// stamp the moment it completes.
///
/// # Errors
///
/// Every error is a routing answer: the caller still owes the stamp and settles
/// it the blocking way. Nothing is enqueued unless a successful FIFO submission
/// has published the completion point being waited.
pub fn write_completion_stamp(
    guest_ref: &crate::runtime::guest_ram::GuestRef,
    index: u32,
    value: u32,
) -> Result<(), DrawError> {
    use host_ram::GuestWriteDecline;
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    if !ctx.caps.host_pointer.is_available() {
        return Err(DrawError::GuestPageWrite(GuestWriteDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        }));
    }
    let Some(completion) = ctx.stamp_completion.as_ref() else {
        return Err(DrawError::GuestPageWrite(GuestWriteDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        }));
    };
    let had_batch = pools.batch_open_recording().is_some();
    let deferred = had_batch
        && completion.queue_for_next_submission(index, guest_ref.clone(), value);
    if !deferred {
        // A full pending ring cannot wait while this thread owns the open
        // command buffer: submitting it is what lets completions retire. This
        // is the pressure path only; the ordinary stamp leaves the batch open.
        if had_batch {
            unsafe { pools.batch_flush(ctx, counters)? };
        }
        let Some((_, timeline)) = completion.latest_queued() else {
            let decline = GuestWriteDecline::NoCompletionPoint;
            crate::observe::Emit::decline("gpu_completion_stamp", &decline).fail();
            return Err(DrawError::GuestPageWrite(decline));
        };
        completion.wait_for_stamp(timeline, index, guest_ref.clone(), value);
    }
    // The queued stamp now carries the ordering for every guest read recorded
    // before this submission. A later read records a fresh debt; retaining this
    // one would submit an otherwise empty stamp for every later CPU-only packet.
    pools.take_guest_read_debt();
    counters.gpu_stamps.fetch_add(1, Ordering::Relaxed);
    StampPointSource::from_route(had_batch, deferred).count(counters);
    Ok(())
}

/// Submit the open batch when FIFO `index` is owed a stamp parked in it.
///
/// Called when a guest packet is held on a stamp wait this device has already
/// promised but not put in flight. `queue_for_next_submission` registers a stamp
/// against the open batch's *future* submission point, and the batch has no time
/// bound — it stays open until a draw claims a slot, a readback arrives, a
/// present runs, or the pending ring fills. On a channel that has gone quiet
/// none of those need happen soon, so a timeline can block for tens of
/// milliseconds on a word whose only remaining obstacle is that this device has
/// not submitted the work it belongs to.
///
/// That is a stall this device causes and can end, and ending it changes no
/// ordering: submitting a batch earlier runs exactly the recorded work, in the
/// same order, against the same completion point. It is the honest counterpart
/// of answering the wait from a private record of executed work, which is
/// unsound because a stamp means the settle has run and the memory is
/// reclaimable.
///
/// Returns whether a batch was submitted, so the caller can count how often the
/// stall was real.
pub fn submit_batch_for_waiting_stamp(index: u32) -> bool {
    // Lock-free, and false in the common case, so a held packet does not take
    // the engine lock merely to discover there is nothing parked for it.
    if !stamp_completion::fifo_has_unsubmitted_stamp(index) {
        return false;
    }
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    if pools.batch_open_recording().is_none() {
        return false;
    }
    let Some(ctx) = owner.ctx.as_ref() else {
        return false;
    };
    match unsafe { pools.batch_flush(ctx, counters) } {
        Ok(()) => true,
        Err(e) => {
            // The stamp stays parked and the guest keeps waiting, so this is a
            // real loss of forward progress rather than a missed optimisation.
            crate::observe::fail(format!(
                "stamp_waiter_flush_failed index={index} err={e:?} (a guest \
                 timeline is blocked on a stamp parked in a batch this device \
                 could not submit)"
            ));
            false
        }
    }
}

/// Let every older completion on `index` publish before a CPU fallback writes
/// a newer value into that FIFO's shared stamp slot.
pub fn quiesce_completion_stamps(index: u32) {
    if !stamp_completion::fifo_has_pending_stamp(index) {
        return;
    }
    let guard = lock_engine();
    if let Some(completion) = guard
        .owner
        .ctx
        .as_ref()
        .and_then(|ctx| ctx.stamp_completion.as_ref())
    {
        completion.wait_for_fifo_idle(index);
    }
}

pub fn quiesce_guest_reads() {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    if let Err(e) = unsafe { pools.quiesce_guest_reads(ctx, counters) } {
        // The wait failed, so this device cannot say the guest's pages are done
        // being read. Nothing here can hold the stamp back — the guest would
        // hang — so report it, and latch a lost device rather than trusting the
        // next draw to surface it. Both of these run under the engine lock but
        // inside a borrow of it, so they latch rather than recover; the drain's
        // end-of-tranche `flush_batched_draws` is what acts on it.
        if matches!(e, DrawError::DeviceLost(_)) {
            device_lost::note_device_lost_seen();
        }
        crate::observe::Emit::decline("vk_guest_read_quiesce", &e).fail_once(0);
    }
}

/// Execute one compute dispatch against the persistent engine.
pub fn execute_compute_request(req: &ComputeRequest) -> Result<ComputeOutput, ComputeError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let result =
        unsafe { exec_compute::execute_compute_inner(owner, caches, pools, counters, req) };
    match result {
        Ok(out) => {
            // Same as the draw arm: a dispatch that ran proves the device.
            guard.owner.note_work_completed();
            Ok(out)
        }
        Err(DrawError::DeviceLost(decline)) => {
            guard.on_device_lost();
            Err(DrawError::DeviceLost(decline))
        }
        Err(e) => Err(e),
    }
}

/// Measure-only: does the target registry hold **content_ready** for this identity?
///
/// Used by type-11 sample dig (`sample_src=… resident_ready=`) to detect the
/// resident-vs-guest split without a full readback. Does not create devices or
/// allocate; returns false if the engine is uninit or the key is absent.
/// Whether the window presenter would take this resident for a present at
/// `width`x`height`. Shares [`pools::slot_presentable`] with the presenter's own
/// selection so the two cannot answer differently.
///
/// Not gated on `host-window`, because the question is about the target registry
/// rather than about a window: `runtime::drain`'s `present_unbacked` gate asks it
/// to tell "the guest sent no full frame for this mid AND nothing can carry the
/// present" (a black frame) from "no full frame, but a resident carries it
/// anyway" (a census). That distinction has to be available on every Vulkan
/// build, not only the ones that opened a window.
pub fn resident_presentable(identity: &TargetIdentity, width: u32, height: u32) -> bool {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .is_some_and(|slot| pools::slot_presentable(slot, width, height))
}

fn resident_present_decision(
    pools: &ResourcePools,
    identity: &TargetIdentity,
    width: u32,
    height: u32,
) -> Result<(), &'static str> {
    let Some(slot) = pools.registry_get(identity) else {
        return Err("winpub_no_resident");
    };
    match pools::slot_present_decline(slot, width, height) {
        None => Ok(()),
        Some(pools::ResidentPresentDecline::ContentNotReady) => {
            Err("winpub_content_not_ready")
        }
        Some(pools::ResidentPresentDecline::ScanoutOrder) => Err("winpub_scanout_order"),
        Some(pools::ResidentPresentDecline::Geometry) => Err("winpub_geometry"),
    }
}

/// Maintain the resident working set for one published frame and decide
/// whether that frame can be presented directly, returning the exact fallback
/// route when it cannot.
///
/// The direct present is the point of owning the window on the engine's own
/// device: the fallback copies the whole framebuffer through host memory every
/// frame, so losing it is a throughput cliff. Four conditions collapsed into one
/// `bool` here and the caller fell through without naming any of them, so a boot
/// reading `direct_frac=0.00` in every census window — against a documented
/// expectation of `1.00` — said only that it had stopped, never why.
///
/// The displayed identity is refreshed by the maintenance operation before it
/// selects reclaim victims. Returning the present decision from the same
/// transaction keeps that one resource lifecycle atomic and avoids entering
/// the engine separately to touch, maintain, test, and then diagnose it.
pub fn prepare_window_resident_present(
    identity: &TargetIdentity,
    width: u32,
    height: u32,
) -> Result<(), &'static str> {
    let now_ms = crate::observe::elapsed_ms() as u64;
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        unsafe {
            pools.advance_registry_maintenance(ctx, counters, now_ms);
        }
    }
    resident_present_decision(pools, identity, width, height)
}

/// What storage owns a ready resident's pixels.
///
/// A guest-backed resident is the guest allocation itself. A recyclable
/// resident is a device allocation holding a copy, so callers deciding whether
/// guest CPU writes can make the resident stale must keep the two distinct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContentBacking {
    NotReady,
    GuestAllocation,
    DeviceAllocation,
}

/// One live resource object's ownership of a resident target.
///
/// Construction pins the target for the serialized resource lifetime. Dropping
/// the object ends that lifetime, blocks new retention, and retires the host
/// object once outstanding users release their pins. A device/reset boundary
/// replaces the registry and invalidates the old lease epoch.
#[derive(Debug)]
pub struct ResidentResourceLease {
    identity: TargetIdentity,
    backing: ResidentContentBacking,
    epoch: u64,
}

impl ResidentResourceLease {
    pub fn matches(&self, identity: &TargetIdentity) -> bool {
        self.identity == *identity && self.epoch == RESIDENT_RESOURCE_EPOCH.load(Ordering::Acquire)
    }

    pub fn backing(&self) -> ResidentContentBacking {
        self.backing
    }

    #[cfg(test)]
    pub(crate) fn test_new(identity: TargetIdentity, backing: ResidentContentBacking) -> Self {
        Self {
            identity,
            backing,
            epoch: RESIDENT_RESOURCE_EPOCH.load(Ordering::Acquire),
        }
    }
}

#[cfg(test)]
pub(crate) fn test_advance_resident_resource_epoch() {
    RESIDENT_RESOURCE_EPOCH.fetch_add(1, Ordering::Release);
}

impl Drop for ResidentResourceLease {
    fn drop(&mut self) {
        let mut guard = lock_engine();
        if self.epoch == RESIDENT_RESOURCE_EPOCH.load(Ordering::Acquire) {
            let EngineState {
                ref owner,
                ref mut pools,
                ref counters,
                ..
            } = *guard;
            if let Some(ctx) = owner.ctx.as_ref() {
                unsafe {
                    let _ = pools.release_resident_resource(ctx, &self.identity, counters);
                }
            } else {
                let _ = pools.pin_resident_target(&self.identity, false);
            }
        }
    }
}

/// Resolve a serialized resource object to a retained engine target.
///
/// Allocation ownership is immutable for the returned lease. Mutable content
/// readiness is still validated by draw execution on every use.
pub fn retain_resident_resource(identity: &TargetIdentity) -> Option<ResidentResourceLease> {
    let mut guard = lock_engine();
    let guest_imported = guard.pools.retain_resident_target(identity)?;
    Some(ResidentResourceLease {
        identity: identity.clone(),
        backing: if guest_imported {
            ResidentContentBacking::GuestAllocation
        } else {
            ResidentContentBacking::DeviceAllocation
        },
        epoch: RESIDENT_RESOURCE_EPOCH.load(Ordering::Acquire),
    })
}

fn classify_resident_content(content_ready: bool, guest_imported: bool) -> ResidentContentBacking {
    match (content_ready, guest_imported) {
        (false, _) => ResidentContentBacking::NotReady,
        (true, true) => ResidentContentBacking::GuestAllocation,
        (true, false) => ResidentContentBacking::DeviceAllocation,
    }
}

pub fn resident_content_backing(identity: &TargetIdentity) -> ResidentContentBacking {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .map(|slot| classify_resident_content(slot.content_ready, slot.memory.is_guest_imported()))
        .unwrap_or(ResidentContentBacking::NotReady)
}

pub fn resident_content_ready(identity: &TargetIdentity) -> bool {
    resident_content_backing(identity) != ResidentContentBacking::NotReady
}

#[cfg(test)]
mod resident_content_backing_tests {
    use super::*;

    #[test]
    fn ready_content_preserves_whether_it_is_the_guest_allocation() {
        assert_eq!(
            classify_resident_content(true, true),
            ResidentContentBacking::GuestAllocation
        );
        assert_eq!(
            classify_resident_content(true, false),
            ResidentContentBacking::DeviceAllocation
        );
        assert_eq!(
            classify_resident_content(false, true),
            ResidentContentBacking::NotReady
        );
        assert_eq!(
            classify_resident_content(false, false),
            ResidentContentBacking::NotReady
        );
    }
}

/// Why this identity has no resident, when the reason is that this device
/// destroyed one it was holding.
///
/// `Some` means the registry does not hold this identity **and** a reclaim
/// record for it is still inside `RECLAIM_HISTORY`. `None` means either the
/// resident is present, or this device has no record of ever having held one —
/// which for a surface the guest has simply not been drawn into yet is the
/// ordinary case and not a defect.
///
/// [`resident_content_ready`] cannot answer this: it is
/// `is_some_and(content_ready)`, so it collapses "absent" and "held but not
/// ready yet" into the same `false`. Those are opposite situations for a caller
/// deciding whether falling through to the guest's pages is sound — a resident
/// that is merely not ready still exists and can be merged from, and one this
/// device destroyed cannot.
///
/// The second half of the pair is how many milliseconds ago the reclaim
/// happened. It measures how quickly pressure-reclaimed content was wanted
/// again; it is not an idle-lifetime calculation.
pub fn resident_absent_after_reclaim(
    identity: &TargetIdentity,
) -> Option<(types::ResidentReclaim, u64)> {
    let guard = lock_engine();
    if guard.pools.registry_get(identity).is_some() {
        return None;
    }
    let now = guard.pools.idle_clock_ms();
    guard
        .pools
        .prior_reclaim_at(identity)
        .map(|(why, at)| (why, now.saturating_sub(at)))
}

/// The mapping content epoch this resident's pixels were stamped with, or
/// `None` when the identity is absent, evicted, or has not been vouched for
/// since its last draw.
///
/// Compared by the type-11 LOAD against
/// [`crate::model::MappingEntry::surface_content_epoch`]: equal means the
/// resident already holds exactly the bytes a CPU seed would upload, so the
/// pass may load straight from the resident and skip the upload. Every way the
/// answer can be unknown — no slot, recycled image, a draw since the stamp —
/// resolves to `None` and therefore to the seed.
pub fn resident_content_epoch(identity: &TargetIdentity) -> Option<u32> {
    let guard = lock_engine();
    guard.pools.registry_get(identity)?.content_epoch
}

/// What the registry says about an identity's content stamp, with the two ways
/// [`resident_content_epoch`] can answer `None` told apart.
///
/// The elision path is right to collapse them: a LOAD that cannot prove the
/// resident holds the mapping's bytes takes the CPU seed, and it does not care
/// why. A *deferred window* landing its frame is the opposite case. It already
/// pinned a content-ready slot at this identity and stamped it under the engine
/// lock, so by the time it lands:
///
/// - [`ResidentContent::Unstamped`] is expected traffic. `registry_mark_ready_at`
///   clears the stamp on every draw into a slot, so a later pass over the same
///   surface says the resident no longer holds the frame this window promised.
///   Declining is correct; the newer pass owns the surface now.
/// - [`ResidentContent::Absent`] is not. Allocation-pressure reclaim skips every
///   pinned slot, so an identity that has gone missing between arm and fence means
///   the two spellings of it disagree: the arm pinned one `TargetIdentity` and
///   the flush rebuilt another. That is a lost frame *and* a leaked pin, and it
///   is the same defect shape as `74748d2` and `021e64b`, which is why it must
///   not hide inside the same `None` as the expected case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentContent {
    /// No slot at this identity: evicted, never created, or named differently
    /// by whoever is asking.
    Absent,
    /// Slot present, stamp cleared by a draw since it was written.
    Unstamped,
    /// Slot present and vouched for at this mapping content epoch.
    Epoch(u32),
}

/// [`ResidentContent`] for one identity — the reading a deferred window takes
/// before it believes a resident still holds its frame.
pub fn resident_content_state(identity: &TargetIdentity) -> ResidentContent {
    let guard = lock_engine();
    match guard.pools.registry_get(identity) {
        None => ResidentContent::Absent,
        Some(slot) => match slot.content_epoch {
            None => ResidentContent::Unstamped,
            Some(epoch) => ResidentContent::Epoch(epoch),
        },
    }
}

/// Record that this resident holds the mapping's content as of `epoch`. Returns
/// false when the identity is absent or not content_ready, which the caller
/// must treat as "the elision is off for this surface" rather than ignore.
pub fn stamp_resident_content_epoch(identity: &TargetIdentity, epoch: u32) -> bool {
    let mut guard = lock_engine();
    guard.pools.registry_stamp_content_epoch(identity, epoch)
}

/// Record that this resident's pixels now exist somewhere that outlives the
/// image, so the reclaim paths may take it. Returns whether a slot was found.
///
/// Call this only where the copy has actually landed. Not calling it costs
/// retained VRAM; calling it wrongly costs the frame — see
/// [`crate::backend::vulkan::engine::pools::ResidentTargetSlot::gpu_only_content`].
pub fn note_resident_content_copied_out(identity: &TargetIdentity) -> bool {
    let mut guard = lock_engine();
    guard.pools.registry_note_content_copied_out(identity)
}

/// Whether this backend may leave guest-visible content only in GPU-resident
/// engine state.
///
/// Held back by the `guest_pages_stay_authoritative` driver quirk, because a
/// device recreate drops that registry before guest pages are updated. See
/// [`crate::backend::vulkan::caps::DriverQuirk`] for what the quirk covers and
/// how to retire it.
/// Whether a render target's resident may be created at this texel layout's
/// format, rather than at the engine's neutral eight-bit one.
///
/// The two eight-bit orders answer `true` without asking. They are the engine's
/// own resident colour formats — every render target this device has ever
/// created is one of them — so a query could only ever confirm them, and
/// gating them behind one would make a target's format depend on whether a
/// device had been resolved when the identity was minted. An identity that
/// changes format under the same guest allocation is a registry recreate, so
/// the answer for those two has to be constant.
///
/// Anything wider is a real question and is asked of the device:
/// [`crate::backend::vulkan::caps::device_features::DeviceFeatures::color_attachment_blend`]
/// holds one probe per [`TexelLayout`] for `COLOR_ATTACHMENT` *and*
/// `COLOR_ATTACHMENT_BLEND` under optimal tiling. No device yet resolved
/// answers `false`, which narrows to the format the target would have had
/// anyway — an override or an unresolved device may never widen what the
/// device does.
pub fn render_target_layout_supported(layout: crate::contract::pixel_format::TexelLayout) -> bool {
    use crate::contract::pixel_format::TexelLayout;
    if matches!(layout, TexelLayout::Rgba8 | TexelLayout::Bgra8) {
        return true;
    }
    device_capabilities().render_target_layout_supported(layout)
}

pub fn deferred_gpu_only_content_allowed() -> bool {
    device_capabilities().deferred_gpu_only_content_allowed()
}

/// The largest render-target edge this host can create, from the device's own
/// `maxImageDimension2D`. Before a device is resolved this is the Vulkan 1.2
/// required minimum — the most any implementation is guaranteed to accept.
pub fn max_render_target_dimension() -> u32 {
    device_capabilities().max_render_target_dimension()
}

/// What this Vulkan device can execute, for the GPU-dependent half of the
/// guest's device-info reply.
///
/// Before a device is resolved the answer is the Vulkan 1.2 floor rather than
/// the reply table's own values: the served reply is only ever *reduced* by
/// this, so a boot that answers before the device is up must not be the one
/// that promises the most.
pub fn device_info_limits() -> crate::model::DeviceInfoLimits {
    use crate::backend::vulkan::caps::device_features::{
        VULKAN_MIN_COMPUTE_SHARED_MEMORY_BYTES, VULKAN_MIN_COMPUTE_WORKGROUP_SIZE,
    };
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .map(|ctx| crate::model::DeviceInfoLimits {
            max_sample_count: ctx.features.max_sample_count,
            d24_stencil8: ctx.features.d24_unorm_s8_attachment,
            max_threads_per_threadgroup: ctx.features.max_compute_workgroup_size,
            max_threadgroup_memory_bytes: ctx.features.max_compute_shared_memory_bytes,
            native_fp16: ctx.features.float16,
        })
        .unwrap_or(crate::model::DeviceInfoLimits {
            max_sample_count: 1,
            d24_stencil8: false,
            max_threads_per_threadgroup: VULKAN_MIN_COMPUTE_WORKGROUP_SIZE,
            max_threadgroup_memory_bytes: VULKAN_MIN_COMPUTE_SHARED_MEMORY_BYTES,
            native_fp16: false,
        })
}

/// `(maxTotalThreadsPerThreadgroup, threadExecutionWidth)` for this host, as
/// the guest's `CmdGetComputeInfo` asks for them.
///
/// Both are device limits, so both are queried. Before a device is resolved
/// the answer is the Vulkan 1.2 required minimum and a single lane — the pair
/// no dispatch can be oversized against.
pub fn compute_threadgroup_limits() -> (u32, u32) {
    use crate::backend::vulkan::caps::device_features::VULKAN_MIN_COMPUTE_WORKGROUP_INVOCATIONS;
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .map(|ctx| {
            (
                ctx.features.max_compute_workgroup_invocations,
                ctx.features.subgroup_size,
            )
        })
        .unwrap_or((VULKAN_MIN_COMPUTE_WORKGROUP_INVOCATIONS, 1))
}

/// Pin a content-ready resident render target against LRU eviction (deferred
/// render Store — the GPU image is the only copy until flush-on-access lands
/// it in guest pages). Returns false when the identity is absent or not
/// ready; the caller must then perform the synchronous Store instead.
pub fn pin_resident_target(identity: &TargetIdentity) -> bool {
    let mut guard = lock_engine();
    guard.pools.pin_resident_target(identity, true)
}

/// Drop the deferred render-Store pin (flushed, or the window was dropped at
/// a lifetime boundary). The target stays registered — only LRU protection
/// ends. No-op for an absent identity.
pub fn unpin_resident_target(identity: &TargetIdentity) {
    let mut guard = lock_engine();
    let _ = guard.pools.pin_resident_target(identity, false);
}

/// Which engine entry point's initialization prologue refused, for the
/// `vk_engine_probe` decline's `probe=` field.
///
/// [`EngineProbe::discriminant`] is the `fail_once` dedup key, so it is a
/// stable numbering and not an index: 1 through 6 and 8 are retired holes. 1, 2
/// and 3
/// named the present-proxy GPU stats oracle's context / pool / take prologues
/// (`present_stats_context`, `present_stats_pools`, `take_stats_context`); 4 and
/// 5 named the host-pointer import prologues (`host_import_context`,
/// `host_import_pools`), which went out with the import subsystem; 6 was
/// `compute_writeback_alignment`, which went out with the GPU-direct compute
/// writeback; 8 was `compute_capable`, the public query for a combined
/// GRAPHICS|COMPUTE queue family, which no caller ever asked — the two engine
/// paths that need the capability read `ctx.compute_capable` directly.
/// Do not reuse them — a fail-log line already carrying one of those
/// keys must not be conflated with a new probe's.
#[derive(Clone, Copy, Debug)]
enum EngineProbe {
    StorageWriteWithoutFormat,
    SampledLayoutLinearFilter,
}

impl EngineProbe {
    fn name(self) -> &'static str {
        match self {
            Self::StorageWriteWithoutFormat => "storage_write_without_format",
            Self::SampledLayoutLinearFilter => "sampled_layout_linear_filter",
        }
    }

    /// 1 through 6 are retired (see the type's docs); the rest keep the numbers
    /// they were first logged under.
    fn discriminant(self) -> u64 {
        match self {
            Self::StorageWriteWithoutFormat => 7,
            Self::SampledLayoutLinearFilter => 9,
        }
    }
}

fn engine_probe_decline(probe: EngineProbe, error: &DrawError) -> crate::observe::Emit {
    crate::observe::Emit::decline("vk_engine_probe", error).field("probe", probe.name())
}

/// Generation of a resident compute storage image, if the engine holds one.
///
/// Measure/skip aid for the runtime's stage-time guest-read skip: a skip is
/// taken only when this equals the mapping's current content generation. Does
/// not create devices or allocate; returns `None` when the engine is uninit
/// or the key is absent.
pub fn compute_resident_storage_generation(
    identity: &crate::model::ComputeStorageResidencyKey,
) -> Option<u32> {
    let mut guard = lock_engine();
    guard.pools.compute_resident_generation(identity)
}

/// Generation + engine format of a resident compute storage image, if the
/// engine holds one.
///
/// Skip aid for the runtime's copy-on-sample gate: a sampled guest read is
/// skipped only when the generation matches the runtime's residency mirror
/// AND the resident's vk format equals what the sampled view will bind (the
/// engine's resident-bind path guards format equality and would fail the
/// whole request on mismatch). Does not create devices or allocate; returns
/// `None` when the engine is uninit or the key is absent.
pub fn compute_resident_sample_source(
    identity: &crate::model::ComputeStorageResidencyKey,
) -> Option<(u32, StorageImageFormat)> {
    let mut guard = lock_engine();
    guard.pools.compute_resident_sample_source(identity)
}

/// Drop the deferred-writeback pin of a resident whose guest window can no
/// longer be flushed (ReplacePhysical / unmap drop paths). The resident stays
/// registered — only LRU protection ends. No-op for an absent identity.
pub fn unpin_resident_storage(identity: &crate::model::ComputeStorageResidencyKey) {
    let mut guard = lock_engine();
    guard.pools.pin_resident_storage(identity, false);
}

/// Release a compute-storage resident's claim on being unreclaimable because the
/// guest deleted the object its content belonged to.
///
/// Paired with `unpin_resident_storage` at the teardown sites only. An unpin
/// alone stopped being enough once the reclaim paths learned to refuse a
/// sole-copy resident: `retire_linear_residents` exists to keep a dead cache
/// entry from leaking its pinned VRAM image for the boot, and without this the
/// leak would simply change its name. Never call it on a live object — the whole
/// point of the flag is that content nobody has copied out is not disposable.
pub fn retire_resident_storage_content(identity: &crate::model::ComputeStorageResidencyKey) {
    let mut guard = lock_engine();
    guard.pools.note_compute_storage_content_retired(identity);
}

/// End one guest parent allocation's Vulkan lifetime.
///
/// The runtime emits this only when the mapping incarnation that owns the
/// allocation ends. Existing child images keep the import until their own
/// fence-safe retirement; an allocation with no children enters the same
/// graveyard immediately so already-recorded buffer accesses finish first.
pub fn retire_guest_import(import_id: crate::runtime::guest_ram::ImportId) {
    let mut guard = lock_engine();
    let Some(device) = guard.owner.ctx.as_ref().map(|ctx| ctx.device.clone()) else {
        return;
    };
    unsafe { guard.pools.retire_guest_import(&device, import_id) };
}

/// A synchronous compute writeback landed this resident's output in the guest's
/// own pages, so the image has stopped being the only place that output exists
/// and the reclaim paths may take it.
///
/// The deferred rail already had this edge — `read_resident_storage` clears the
/// flag when its flush lands — but the **synchronous** rail did not, and it is
/// the common one. `mark_resident_storage_image` sets `gpu_only_content` after
/// every executed dispatch, and on this rail nothing ever cleared it again: the
/// bytes were read back, handed to `writeback_texture` and written to guest
/// pages, and the resident stayed flagged as unreproducible for the life of the
/// guest.
///
/// Allocation-pressure recovery refuses a sole copy, so those residents were
/// unreclaimable. The registry only grew, and an allocation failure found
/// nothing to give back at the one moment it needed something. That is
/// the shape of the leak `retire_resident_storage_content` was written to stop
/// for a *dead* cache entry, arriving instead through the live path.
///
/// Called after the writeback rather than after the readback, which is the
/// stricter of the two and deliberately so: the readback only moves the bytes
/// into a host `Vec`, and it is `writeback_texture` returning `Ok` that says a
/// later reader can find them. Clearing at the readback would open a window
/// where a reclaim takes a resident whose output reached nowhere.
pub fn note_resident_storage_copied_out(identity: &crate::model::ComputeStorageResidencyKey) {
    let mut guard = lock_engine();
    guard.pools.note_compute_storage_copied_out(identity);
}

/// True when the device supports format-less storage-image writes
/// (`shaderStorageImageWriteWithoutFormat`). The compute path needs this to
/// composite a guest `BGRA8Unorm` storage surface into a `B8G8R8A8_UNORM` view
/// without an R/B channel swap; when absent it degrades to a `R8G8B8A8_UNORM`
/// view (swapped) and logs the degraded class. Returns `false` if the engine
/// cannot initialize.
pub fn supports_storage_image_write_without_format() -> bool {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.storage_image_write_without_format,
        Err(error) => {
            engine_probe_decline(EngineProbe::StorageWriteWithoutFormat, &error)
                .fail_once(EngineProbe::StorageWriteWithoutFormat.discriminant());
            false
        }
    }
}

/// Whether the bound device can sample this guest texel layout's Vulkan format
/// with **linear** filtering.
///
/// Gates every native sampled rail — those bind the guest's own bytes and let
/// the sampler interpolate them, so a layout the host cannot filter must be
/// declined rather than bound. Asked per layout rather than for the one format
/// once known to be optional, because the mandatory-format table is an
/// API-version assumption and does not cover the set: `R32_SFLOAT` is absent on
/// Apple/MoltenVK and `R16_UNORM`'s linear filtering is optional too.
///
/// Returns `false` — declining the rail, leaving the sample fail-visible — if
/// the engine cannot initialize.
pub fn supports_sampled_layout_linear_filter(
    layout: crate::contract::pixel_format::TexelLayout,
) -> bool {
    if let Some(supported) = device_capabilities().sampled_layout_linear_filter_if_published(layout)
    {
        return supported;
    }

    // The first query is allowed to create the device. Publication happens as
    // part of that creation, so every later query for its lifetime is the
    // lock-free immutable answer above.
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.sampled_linear_filter[layout.index()],
        Err(error) => {
            engine_probe_decline(EngineProbe::SampledLayoutLinearFilter, &error)
                .fail_once(EngineProbe::SampledLayoutLinearFilter.discriminant());
            false
        }
    }
}

/// Read a content-ready **BGRA** resident target as tight BGRA8 for the present
/// capture (the proxy-oracle frame source).
///
/// This is the resident-direct capture source: it performs only the GPU→host
/// readback, with **no** guest-page scatter. `capture_present_frame`'s other
/// source — `flush_intersecting`, the deferred render-flush rail — reads the same
/// resident but additionally scatters it into the fragmented guest pages — work
/// the oracle does not need and which the deferred-writeback rail already
/// performs on a genuine guest read.
///
/// **Converts rather than refusing.** The caller's frame buffer is BGRA8 and
/// this is the only source the present capture has left — a refusal here is not
/// a fallback, it is the host window holding its previous retain until some
/// later frame happens to be readable. That was survivable while every resident
/// this could name was created in guest scanout order; it stopped being so once
/// a type-11 mapping's resident follows the format the mapping declares, because
/// a scanout plane declared at anything else would have frozen the window
/// outright. So the readback's own reported order decides, and a resident that
/// is not already BGRA8 pays one exchange — [`read_target`]'s rail has already
/// narrowed a wide one to four bytes by then, so the exchange is always over
/// RGBA8.
///
/// Returns `None` for every *expected* absence — unknown identity, no ready
/// content, or a short/oversized readback — so the caller can fall back
/// silently. These are speculative conditions on a normal boot (a cold mid has
/// no resident yet), not failures worth a fail-log line.
pub fn read_resident_bgra(identity: &TargetIdentity, need: usize) -> Option<Vec<u8>> {
    {
        let guard = lock_engine();
        let slot = guard.pools.registry_get(identity)?;
        if !slot.content_ready {
            return None;
        }
    }
    let mut px = match read_target_inner(identity) {
        // `into_bgra8` is a no-op for a resident already in scanout order, which
        // is every one this rail sees on a boot measured so far.
        Ok(rb) => rb.into_bgra8(),
        Err(e) => {
            let mut emit = crate::observe::Emit::decline("present_capture", &e);
            for (key, value) in draw_execution::identity_fields(identity) {
                emit = emit.field(key, value);
            }
            emit.off();
            return None;
        }
    };
    if px.len() < need {
        return None;
    }
    px.truncate(need);
    Some(px)
}

/// The six fallible Vulkan calls a whole-image readback makes, named per rail.
///
/// The rails differ in nothing else, but they must not share slugs: a
/// `reason=vk_readback_submit` that could have come from either the present
/// drain or a deferred compute flush names neither, which is the collapse the
/// typed [`VkOp`] vocabulary exists to prevent.
struct ReadbackOps {
    reset_cb: VkOp,
    begin_cb: VkOp,
    end_cb: VkOp,
    submit: VkOp,
    map: VkOp,
    /// `vkInvalidateMappedMemoryRanges`, which the readback owes whenever
    /// `MemoryClass::Readback` landed on a host-cached non-coherent type.
    invalidate: VkOp,
}

/// Copy level 0 of a resident color image to host bytes, tightly packed.
///
/// Shared by the target readback (present / Synchronize / Map / Store boundary)
/// and the pinned-storage deferred flush. `src_access` is the only Vulkan
/// difference: a render target may have a `COLOR_ATTACHMENT_WRITE` to drain, a
/// storage image cannot.
///
/// Async ring advance (retires only the one slot it reuses), NOT a whole-ring
/// quiesce: this reads content that is already ready, not an UNDEFINED-layout
/// seed, so the `ALL_COMMANDS → TRANSFER` barrier below orders the copy after
/// every prior-submitted draw. A whole-ring quiesce would block this guest-drain
/// readback behind an unrelated in-flight heavy draw — the `finish_us` tail. We
/// wait only our own `fence` after submit, and the slot stays pending for the
/// ring to retire later.
///
/// That sentence used to credit "single-queue submission order" alongside the
/// barrier, and it was wrong twice over: submission order is not a memory
/// dependency, and the barrier it named was skipped on the one path where the
/// image already sat in TRANSFER_SRC_OPTIMAL — which is every readback that
/// follows a render pass, because that is the layout a pass resolves its
/// primary to. The barrier is unconditional now.
/// Where a readback's bytes end up: copied out into a `Vec`, or left in the
/// staging buffer for the caller to consume through a lease.
///
/// The copy is what the callers that keep their frame need — a `Vec` outlives
/// everything and belongs to nobody. The lease is for the one caller that
/// consumes the frame once and immediately: it hands the bytes straight on and
/// then has no use for them, so copying 8 MB to own them for a millisecond is
/// pure cost. See [`LeasedFrame`].
#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadbackDelivery {
    Copy,
    Lease,
}

/// The result of a readback in whichever of the two forms was asked for.
enum ReadbackResult {
    Copied(Vec<u8>),
    Leased { token: u64, ptr: usize, len: usize },
}

/// Returns an unhanded-on readback lease when its scope exits.
///
/// The lease is taken before the first of six fallible Vulkan calls, so there
/// are six early returns between checking it out and having anything to give it
/// to. A lease dropped on one of those paths is not a correctness fault on any
/// read — nothing is borrowing it — but it strands the slot for the process
/// lifetime and makes every later teardown wait out its whole quiesce budget,
/// which presents as a hang rather than as a leak. One guard covers all six.
struct ReadbackLeaseGuard(Option<pools::ReadbackLease>);

impl ReadbackLeaseGuard {
    fn new(lease: Option<pools::ReadbackLease>) -> Self {
        Self(lease)
    }

    /// Take the lease out, so the caller owns the obligation from here on.
    fn disarm(&mut self) -> Option<pools::ReadbackLease> {
        self.0.take()
    }
}

impl Drop for ReadbackLeaseGuard {
    fn drop(&mut self) {
        if let Some(lease) = self.0.take() {
            pools::return_readback_lease(lease.token);
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy_image_level0_to_host(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &EngineCounters,
    image: ash::vk::Image,
    old_layout: ash::vk::ImageLayout,
    read_layout: ash::vk::ImageLayout,
    src_access: ash::vk::AccessFlags,
    width: u32,
    height: u32,
    rb_size: u64,
    ops: ReadbackOps,
) -> Result<Vec<u8>, DrawError> {
    match copy_image_level0_to_host_delivered(
        ctx,
        pools,
        counters,
        image,
        old_layout,
        read_layout,
        src_access,
        width,
        height,
        rb_size,
        ops,
        ReadbackDelivery::Copy,
    )? {
        ReadbackResult::Copied(bytes) => Ok(bytes),
        // Unreachable by construction — `Copy` was asked for one line above —
        // and stated as a decline rather than a panic because this rail runs on
        // the drain worker, where a panic takes the device down.
        ReadbackResult::Leased { token, .. } => {
            pools::return_readback_lease(token);
            Err(DrawError::TargetRead(
                reason::TargetReadDecline::NoReadyContent,
            ))
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the image, how to read it, and where to deliver it"
)]
unsafe fn copy_image_level0_to_host_delivered(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &EngineCounters,
    image: ash::vk::Image,
    old_layout: ash::vk::ImageLayout,
    read_layout: ash::vk::ImageLayout,
    src_access: ash::vk::AccessFlags,
    width: u32,
    height: u32,
    rb_size: u64,
    ops: ReadbackOps,
    delivery: ReadbackDelivery,
) -> Result<ReadbackResult, DrawError> {
    let submit_started = std::time::Instant::now();
    // The slot is claimed *after* the entry, and the order is load-bearing.
    //
    // `begin_entry` submits any open draw batch first, and that flush runs
    // `seal_entry`, which moves whatever is in `readback_live` into the
    // **batch's** pending cleanup. A slot acquired before it therefore ends up
    // owned by a ring entry that has nothing to do with the copy about to fill
    // it, and is returned to the free list when the batch's fence signals —
    // which is a fence submitted *earlier* than this copy's. Nothing between
    // this function's submit and its own fence wait can retire a ring slot, so
    // that never actually handed the buffer out from under a live copy, but it
    // was one interleaved `begin_entry` away from doing so.
    //
    // For the leasing path it was not latent at all: the lease found
    // `readback_live` already empty and silently declined on exactly the busy
    // frames the rail exists for. A live boot read `render_flush_copied`
    // outnumbering `render_flush_leased` 5:1 on a host whose readback memory is
    // cached, which is the only symptom that mis-ordering has.
    //
    // # Appending to an open batch instead of flushing it
    //
    // A readback used to be two submissions: `begin_entry` submits the open
    // draw batch to get it out of the way, then this function submits its own
    // command buffer behind it. A driven boot measured
    // `batch_readback_joins` at 58.8 % of all batch flushes and the batches
    // themselves at 1.77 draws against a ceiling of 8 — so nearly every readback
    // was cutting a run of draws short to pay for a second submission.
    //
    // When a batch is recording, the copy is appended to *it* and the batch is
    // submitted once. Queue order is unchanged: the copy is recorded after the
    // draws it reads, which is the same order the two submissions produced. The
    // barrier below is what makes that an actual dependency, and it is recorded
    // either way.
    //
    // This also *removes* the staging-slot hazard the paragraph above describes
    // rather than adding to it. The slot is acquired after the batch's cb is in
    // hand and is sealed into the batch's own pending cleanup by
    // `batch_flush` — so the fence that returns it to the free list is now the
    // very fence this copy was submitted with, instead of one submitted earlier.
    let appended = pools.batch_open_recording();
    let (cb, fence) = match appended {
        Some(pair) => {
            counters
                .batch_readback_joins
                .fetch_add(1, Ordering::Relaxed);
            pair
        }
        None => pools.begin_entry(ctx, counters)?,
    };
    let readback = pools.acquire_readback(ctx, rb_size, counters)?;
    // Acquired here rather than beside the dispatch, for the ordering reason
    // above: every readback slot this submission owns must be claimed after
    // `begin_entry`, or it ends up owned by whatever ring entry the flush
    // inside `begin_entry` happened to seal.
    // Before this function's own `seal_entry`, for the same reason: that call
    // would move the slot into this entry's cleanup and hand it back to
    // `readback_free` when the entry retires, under a borrow still reading it.
    // A leased slot belongs to no ring entry — the copy's fence is waited below
    // and the lease is returned explicitly.
    let mut lease = ReadbackLeaseGuard::new(if delivery == ReadbackDelivery::Lease {
        pools.lease_readback()
    } else {
        None
    });
    // Only for a command buffer this call owns. An appended-to batch is already
    // recording and holds the draws the copy is about to read; resetting it
    // would discard them and beginning it again is invalid.
    if appended.is_none() {
        unsafe {
            pools.begin_slot_recording(
                ctx,
                cb,
                gpu_span::Kind::Readback,
                ops.reset_cb,
                ops.begin_cb,
            )?
        };
    }
    // A deferred draw may have left its render pass standing in this same
    // command buffer. The image barrier and copy below are outside-pass
    // commands; recording either inside that pass is invalid.
    unsafe { pools.close_open_pass(&ctx.device, cb) };
    // Unconditional, and the layout match is exactly why. A barrier is two
    // things — a layout transition and a dependency — and this rail needs the
    // second one whether or not it needs the first. Every render pass resolves
    // its primary attachment to TRANSFER_SRC_OPTIMAL, so the *common* case
    // reaches here already in the layout the copy wants; gating the barrier on
    // a transition being required therefore left the most-taken path of the
    // rail that publishes composited pixels to the guest with no ordering
    // against the draws that produced them.
    //
    // Before the barrier, so slot 0 stamps the GPU's own clock at the instant it
    // begins this command buffer rather than after it has already waited for the
    // draws. The reset must be recorded into the same command buffer: a query
    // pool's results are undefined until reset, and resetting on the host needs
    // `hostQueryReset`, which is a Vulkan 1.2 feature this device does not ask
    // for.
    unsafe { pools.readback_span_arm(ctx, cb) };
    // Nothing else supplies it. Queue submission order starts command buffers
    // in order; it does not finish them in order, and it is not a memory
    // dependency. A render pass's implicit final subpass dependency carries
    // `dstStageMask = BOTTOM_OF_PIPE` and `dstAccessMask = 0`, which makes the
    // colour writes *available* but visible to nothing — a later TRANSFER_READ
    // still has to ask for them. Without that, the copy can read the resident
    // before the draw's writes land and publish the pixels from before the
    // draw: the guest-visible symptom is a composite that is missing what was
    // just drawn into it and comes back on the next redraw.
    //
    // `old_layout` is the caller's *tracked* layout, so this is a real
    // transition whenever the resident is not already a transfer source — which
    // is the common case, because a render pass leaves its attachment in
    // [`caches::color0_pass_exit_layout`]. When the two do match the transition
    // half is a legal no-op and the barrier is doing only its other job, which
    // is the job that was missing. Every caller records the move with
    // `registry_note_access(.., TransferRead)` so the tracked layout follows the
    // image; a caller that did not would leave the next barrier naming an
    // `oldLayout` the image is not in, which is undefined behaviour and not an
    // error.
    let barrier = [ash::vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
        .old_layout(old_layout)
        .new_layout(read_layout)
        .image(image)
        .subresource_range(color_subresource_range())];
    ctx.device.cmd_pipeline_barrier(
        cb,
        ash::vk::PipelineStageFlags::ALL_COMMANDS,
        ash::vk::PipelineStageFlags::TRANSFER,
        feedback_transition_dependency(old_layout),
        &[],
        &[],
        &barrier,
    );
    // Between the barrier and the copy, at the stage the barrier releases into.
    // A queue starts command buffers in order but does not finish them in order,
    // so slot 0 can be stamped while the draw batch is still running and the
    // span from it to the end of the copy contains both. This slot is what
    // separates them: everything before it has reached TRANSFER, which after the
    // barrier means the draws are done.
    unsafe { pools.readback_span_mark(ctx, cb, ash::vk::PipelineStageFlags::TRANSFER, 1) };
    let region = [ash::vk::BufferImageCopy::default()
        .image_subresource(color_subresource_layers())
        .image_extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })];
    // The whole level, every time: nothing upstream tells this call which part
    // of the frame the draws touched. That is what the slots 1-2 span prices —
    // on a discrete part this copy crosses the bus, so `gpu_us`'s share of
    // `fence_us` in `readback_split` is bytes and not latency.
    //
    // **Bounding the region is not an available lever, and this used to say it
    // was.** It needs a damage rect, and the device decodes no source for one.
    // The draw stream's scissors are 100% of the attachment on 99.92% of armed
    // windows, and the guest's own `renderTargetWidth`/`Height` — the remaining
    // candidate, and the one `runtime::exec::note_pass_extent_coverage` was
    // built to score — reads `pass_extent_full` on 11 826 of 11 827 scored
    // passes. The guest states the attachment's geometry, not a sub-rect. Small
    // numbers in the fail log are small *surfaces*.
    //
    // What the same boot does leave open is the other half of the fence. Per
    // readback, 217.6 us divides as `bar_us` 1.16 + `gpu_us` 131.1 + 85.3
    // unaccounted, and `bar_us` near zero proves the draws are already finished
    // when the copy runs — so the 85.3 us is submission and wake latency. A
    // readback is two submissions today, because `begin_entry` above flushes the
    // open draw batch before claiming a slot (`batch_readback_joins` is 58.8%
    // of all batch flushes). Collapsing those two is the lever; see
    // `pools::BATCH_MAX_DRAWS`.
    ctx.device
        .cmd_copy_image_to_buffer(cb, image, read_layout, readback.buffer, &region);
    unsafe { pools.readback_span_mark(ctx, cb, ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE, 2) };
    if appended.is_some() {
        // `batch_flush` ends the command buffer, submits it with the fence
        // `batch_open_recording` handed back, and seals the batch's cleanup —
        // which now carries this copy's staging slot. One submission for the
        // draws and the copy together.
        pools.batch_flush(ctx, counters)?;
    } else {
        unsafe { pools.gpu_span_seal_current(ctx, cb) };
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(ops.end_cb, e)))?;
        let cbs = [cb];
        ctx.submit_queue_work(&cbs, &[], &[], &[], fence)
            .map_err(|e| DrawError::VkCall(VkCall::new(ops.submit, e)))?;
        let sealed = pools.seal_entry(Vec::new(), Vec::new());
        pools.finish_entry_async(&ctx.device, sealed);
    }
    // Split three ways rather than timed as a whole: the submit and the copy
    // scale with the surface, the fence does not scale with anything we control,
    // and the fix for one is not the fix for the others.
    use crate::runtime::drain::{note_readback_phase, ReadbackPhase};
    note_readback_phase(
        ReadbackPhase::Submit,
        submit_started.elapsed().as_micros() as u64,
    );
    let fence_started = std::time::Instant::now();
    pools.wait_entry_fence(ctx, counters, fence)?;
    note_readback_phase(
        ReadbackPhase::Fence,
        fence_started.elapsed().as_micros() as u64,
    );
    // The three queries this command buffer wrote are read when its ring slot
    // retires, by `readback_span_read`, which is the one place a slot's fence is
    // known signalled. They used to be read here, against this call's own fence
    // — correct for this writer and not for the guest-page writeback, which
    // shares the probe and does not wait.
    let map_started = std::time::Instant::now();
    let out = match lease.disarm() {
        // The mapping is already established for the slot's lifetime, so all
        // this owes is the invalidate a non-coherent readback owes any reader.
        // What `ReadbackPhase::Map` measures on this arm is therefore that call
        // alone: the whole-frame memcpy the other arm pays is what the lease
        // exists to delete, and the phase reading near zero is how you tell it
        // happened.
        // The lease's own extent decides whether it can be read in place. It
        // is the third refusal alongside `mapped == 0` and `!cached`, and like
        // both of those the answer is the copying path — which asks
        // `slot_span_fits` again on its own arm — rather than a failure. It
        // cannot fire: `acquire_readback` rounds `rb_size` up to a bucket and
        // records that bucket as the slot's size. It is here because that is a
        // property of a call three frames up, and `bytes()` builds a slice of
        // `len` over this pointer on the strength of it.
        Some(lease) if !pools::slot_span_fits(rb_size, lease.slot_size) => {
            crate::observe::Emit::decline(
                "vk_engine_readback",
                &draw_execution::DrawExecutionDecline::LeaseBeyondSlot {
                    len: rb_size,
                    slot_size: lease.slot_size,
                },
            )
            .fail_once(0);
            pools::return_readback_lease(lease.token);
            pools::read_back_slot(ctx, &readback, rb_size, ops.map, ops.invalidate)
                .map(ReadbackResult::Copied)
        }
        Some(lease) => match pools::invalidate_slot_for_read(ctx, &readback, ops.invalidate) {
            Ok(()) => Ok(ReadbackResult::Leased {
                token: lease.token,
                ptr: lease.ptr,
                len: rb_size as usize,
            }),
            Err(e) => {
                pools::return_readback_lease(lease.token);
                Err(e)
            }
        },
        None => pools::read_back_slot(ctx, &readback, rb_size, ops.map, ops.invalidate)
            .map(ReadbackResult::Copied),
    };
    note_readback_phase(ReadbackPhase::Map, map_started.elapsed().as_micros() as u64);
    out
}

/// A resident target's pixels plus the physical channel order they came out in.
///
/// Reported rather than derivable, and read from the registry slot under the
/// same lock as the copy, so it is the order of the image the bytes were
/// actually copied out of. A caller that re-derived it from the identity would
/// be restating a rule the engine owns; when the two disagree the symptom is an
/// R/B exchange on a whole frame, which is a colour defect no assertion in this
/// crate was watching for.
pub struct TargetReadback {
    pub pixels: Vec<u8>,
    /// BGRA8 when true, semantic RGBA8 otherwise.
    pub bgra: bool,
}

impl TargetReadback {
    /// The frame in semantic RGBA8, exchanging R and B only when it is not
    /// already in that order.
    pub fn into_rgba8(mut self) -> Vec<u8> {
        if self.bgra {
            for px in self.pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        self.pixels
    }

    /// The frame in guest scanout order (BGRA8), exchanging only when needed.
    ///
    /// The mirror of `into_rgba8`, for the guest-page writers that are declared in
    /// scanout order (`mapping_write::write_bgra8`). Both exist so that neither
    /// caller has to know which namespace it is reading: a `Surface` resident is
    /// already BGRA and this is a no-op, and a resident that is not stays correct
    /// instead of landing R and B exchanged in guest memory.
    pub fn into_bgra8(mut self) -> Vec<u8> {
        if !self.bgra {
            for px in self.pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        self.pixels
    }
}

/// A resident target's frame, still in the Vulkan readback buffer it was copied
/// into, borrowed rather than copied out.
///
/// # Why this exists
///
/// The deferred render flush reads a resident back and immediately scatters the
/// result into the guest's pages. Copying the staging buffer into a `Vec` first
/// is a second whole-frame pass over ~8 MB — `readback_split map_us`, about
/// 0.8 ms of a 6.9 ms flush, roughly a hundred times a second on a driven
/// desktop — and the `Vec` is dropped a millisecond later. This hands the
/// scatter the mapped bytes directly and the pass is gone.
///
/// # Why it is sound
///
/// The mapping stays valid until the memory is freed, so the borrow needs
/// exactly one thing held off: the pool giving the slot to something that would
/// write it. `lease_readback` takes the slot out of every list that could —
/// free, live, and the submitted entry's pending cleanup — and only the `Drop`
/// below puts it back. A teardown, which frees the memory and so would unmap
/// the pointer under a live borrow, waits for outstanding leases first.
///
/// # What a holder may not do
///
/// **Take the engine lock.** A teardown waiting out the quiesce budget holds
/// it, so a holder that blocks on it deadlocks until that budget expires and
/// then reads freed memory. The whole point of consuming the frame here is that
/// the engine is *not* locked while an 8 MB scatter runs; a holder that needs
/// the engine should end the lease first. Ending it is device-free and cannot
/// block on anything the holder owns.
pub struct LeasedFrame {
    token: u64,
    ptr: usize,
    len: usize,
    /// BGRA8 when true, semantic RGBA8 otherwise. Reported by the registry slot
    /// the bytes were copied out of, for the same reason
    /// [`TargetReadback::bgra`] is.
    pub bgra: bool,
}

impl LeasedFrame {
    /// The frame, in whatever channel order [`Self::bgra`] reports.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is the host address of a `HOST_VISIBLE` mapping
        // established for the slot's lifetime and covering at least `len`
        // bytes, and the slot is leased — so it is in no pool list, no GPU
        // command references it, and teardown cannot free it while this lease
        // is outstanding.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for LeasedFrame {
    fn drop(&mut self) {
        pools::return_readback_lease(self.token);
    }
}

/// Read a resident target back and keep the bytes in the staging buffer.
///
/// The borrowing form of [`read_target`], for a caller that consumes the frame
/// once and has no use for it afterwards. Answers `None` for a frame that
/// cannot be consumed in place, and every `None` is a reason to take the
/// copying path rather than a failure:
///
/// - **Uncached readback memory.** The mapping then reads at roughly a tenth of
///   memcpy speed, and a row-by-row consumer would pay that rate on every row
///   instead of once on the linear pass the copy makes. `MemoryClass::Readback`
///   asks for `HOST_CACHED` first and usually gets it; where it does not, the
///   copy is the cheaper shape. This is the capability gate, and it is on the
///   property rather than on any driver name.
/// - **No mapping to lend**, which a readback slot always has and so should not
///   occur; the fallback keeps it from mattering if it ever does.
pub fn read_target_leased(identity: &TargetIdentity) -> Result<Option<LeasedFrame>, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    // The leased rail hands the caller a pointer into the mapped readback slot,
    // so unlike `read_target_inner` it has nowhere to narrow a wide resident to
    // — the bytes the caller reads are the slot's. Bounded rather than
    // converted, and named so a firing says which rail could not serve.
    let (snap, layout) = readback_snapshot(pools, identity)?;
    if layout.bytes_per_texel() != RESIDENT_READ_BYTES_PER_TEXEL {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::TexelNotFourBytes {
                format: snap.format,
            },
        ));
    }
    let rb_size =
        (snap.width as u64) * (snap.height as u64) * u64::from(RESIDENT_READ_BYTES_PER_TEXEL);
    let read_access = pools::ResidentAccess::transfer_read(snap.guest_backing.is_some());
    unsafe {
        let delivered = copy_image_level0_to_host_delivered(
            ctx,
            pools,
            counters,
            snap.image,
            snap.layout,
            read_access.layout(),
            RESIDENT_READ_SRC_ACCESS,
            snap.width,
            snap.height,
            rb_size,
            target_readback_ops(),
            ReadbackDelivery::Lease,
        )?;
        pools.registry_note_access(identity, read_access);
        counters.note_target_read(rb_size);
        Ok(match delivered {
            ReadbackResult::Leased { token, ptr, len } => Some(LeasedFrame {
                token,
                ptr,
                len,
                bgra: snap.bgra(),
            }),
            // The slot had no mapping to lend, so the readback fell back to a
            // copy. Drop it and let the caller take its own copying path rather
            // than hand back a `Vec` this signature has no room for; that costs
            // one extra readback on a path that should never run.
            ReadbackResult::Copied(_) => None,
        })
    }
}

/// Where in the guest's own pages a resident's frame lands, as a bounded
/// reference the engine can bind.
///
/// Built by the runtime, which is the only side that knows a mapping's page list
/// and its row pitch; the engine takes it as given and checks only what it can
/// see — that the resident matches the extent and that the range is long enough.
pub struct GuestPageTarget {
    /// The guest bytes the frame lands in, one bindable reference per
    /// contiguous stretch, ascending and tiling the window exactly.
    ///
    /// A `Vec` because the guest backs a surface in 16 KiB granules that are
    /// unrelated to each other, so a 1920x1080 window is ~507 stretches and one
    /// range would name 1/507th of the frame. `references_for_runs` is the only
    /// producer and it guarantees the tiling; see its doc for what that buys.
    pub runs: Vec<crate::runtime::guest_ram_map::GuestWindowRun>,
    /// Guest row pitch in **texels** (`bufferRowLength`). Rows past the first
    /// start this far apart, which is how a padded guest pitch is honoured
    /// without the inter-row bytes ever being written.
    pub row_length_texels: u32,
    pub width: u32,
    pub height: u32,
    /// The format the guest reads these bytes back as, from what it declared
    /// for this destination.
    ///
    /// The copy converts nothing, so this is the format the resident must
    /// already hold; the engine checks the pair and refuses by name rather than
    /// assuming either side. It lives here and not on the identity because it
    /// is a property of the *destination* — the runtime is the only side that
    /// knows what the guest declared, exactly as it is for the row pitch above.
    ///
    /// A whole format and not a channel order, because it also fixes how wide a
    /// texel is, and every byte offset below is computed from that. While every
    /// resident was eight bits per channel that width was a constant `4`
    /// written into each of them; a destination four bytes per texel wider
    /// would have had its rows overlap at half their true pitch.
    pub format: ash::vk::Format,
    /// Exact shared allocation spelling of this destination, when the host can
    /// retain one. A resident may settle synchronization without a copy only
    /// when this equals the allocation it was created over.
    pub shared_backing: Option<GuestTargetBacking>,
}

impl GuestPageTarget {
    /// One past the last byte the copy writes: the last texel of the last row.
    ///
    /// Padding after the final row is deliberately excluded. Those bytes belong
    /// to the surface's plane but are not texels this call was given, and the
    /// copying rail does not write them either — a bound that included them
    /// would make the two rails land different guest memory for one frame.
    fn extent_end(&self) -> u64 {
        let rows_before = u64::from(self.height.saturating_sub(1));
        rows_before * self.pitch_bytes() + u64::from(self.width) * self.bytes_per_texel()
    }

    /// Bytes one texel of the destination occupies.
    ///
    /// `copy_target_to_guest_pages` has already refused a format this cannot
    /// answer for — it compares the destination's format against the resident's,
    /// and a resident exists only at a format these tables know — so the
    /// fallback is unreachable. It is the four this code used to assume rather
    /// than a panic, because being wrong here costs a mis-planned copy and not a
    /// lost boot.
    fn bytes_per_texel(&self) -> u64 {
        u64::from(
            crate::backend::vulkan::translate::pixel::bytes_per_texel(self.format)
                .unwrap_or(RESIDENT_READ_BYTES_PER_TEXEL),
        )
    }

    /// Guest bytes the runs actually name, summed.
    ///
    /// Each run's `requested` and not its `bound_len`: the latter is rounded
    /// out to the import's granularity, so summing it would claim coverage of
    /// bytes past the window and turn a short window into one that passes the
    /// check below.
    fn window_bytes(&self) -> u64 {
        self.runs
            .iter()
            .map(|r| r.guest.requested())
            .fold(0u64, u64::saturating_add)
    }

    /// Guest bytes between the starts of two consecutive rows.
    fn pitch_bytes(&self) -> u64 {
        u64::from(self.row_length_texels.max(self.width)) * self.bytes_per_texel()
    }

    /// The window's byte layout, for planning copy rectangles.
    fn geometry(&self) -> reims_vgpu_paging::regions::WindowGeometry {
        reims_vgpu_paging::regions::WindowGeometry {
            pitch_bytes: self.pitch_bytes(),
            width_texels: self.width,
            height_texels: self.height,
        }
    }

    /// Whether the window's rows carry no padding, so every byte from the first
    /// texel to the last is a texel byte.
    ///
    /// This is the precondition for the linear path, and it is a statement
    /// about the *contract* rather than about a workload: when it holds, window
    /// byte `o` is the frame's byte `o` under a tight packing, so a scratch
    /// buffer detiled at that packing can be scattered by byte range with no
    /// row or format arithmetic left to do. When it does not hold, a run's
    /// bytes include padding that must not be written
    /// (`reims_vgpu_paging::regions` states why), and a
    /// `VkBufferCopy` has no way to skip it — so that window takes the
    /// rectangle path, which does.
    fn rows_are_dense(&self) -> bool {
        self.pitch_bytes() == u64::from(self.width) * self.bytes_per_texel()
    }
}

/// Copy a resident target straight into the guest's pages, with no host copy of
/// the frame at any point.
///
/// # What this replaces
///
/// The copying rail reads the resident into a `HOST_VISIBLE` staging buffer,
/// waits, and then has the CPU scatter it row by row into guest RAM. Both halves
/// move the whole frame — `readback_split` prices them at 0.83 ms of staging
/// memcpy plus 2.68 ms of guest-page write in a 6.9 ms flush — and the GPU is
/// already writing every one of those bytes once. Handing it the guest pages as
/// the copy's destination deletes both, leaving the copy that always had to
/// happen.
///
/// # Why this direction is sound where the read direction is not
///
/// Binding guest pages as a draw's *source* would have the GPU read them when
/// the command buffer executes, and this device acks a command before its work
/// runs — so the guest may repaint the pages first. Nothing here has that shape.
/// The caller runs inside `flush_all_windows_before_fence`, which is ordered
/// before the completion stamp, and [`quiesce_guest_writes`] sits between the
/// two: the pages hold the frame before the guest is told anything.
///
/// # This call does not wait, and takes the pin
///
/// It returns once the copy is submitted. Two obligations follow from that and
/// neither is the caller's to discharge by hand:
///
/// * The bytes are not in guest RAM yet. Anything about to *read* them —
///   including the guest, via the stamp — must call [`quiesce_guest_writes`]
///   first.
/// * On `Ok`, `identity`'s registry pin is now held by the entry carrying the
///   copy and released when its fence retires. **The caller must not unpin it**;
///   unpinning here is what would let the reclaim take an image the GPU has not
///   finished reading. An `Err` is a routing answer taken before anything was
///   recorded, so the pin is untouched and the caller still owns it — which is
///   what lets the copying arms below the decline unpin exactly as they always
///   did.
///
/// # Errors
///
/// Every error is a routing answer — the caller still owes the frame, by the
/// copying rail — except that a `VkCall` failure after the submit means the copy
/// may have partly landed. That is the same exposure the copying rail carries
/// for a partial scatter, and the caller reports it the same way.
/// # `pages` is the ledger's currency, not the copy's
///
/// The copy is driven entirely by `dst`. `pages` is the same destination spelled
/// as guest page addresses, which is the only spelling a later reader of those
/// bytes can compare its own window against — see [`GUEST_WRITE_PAGES`]. Both
/// callers walk it to build `dst.runs` and hand the walk's own output here, so
/// the two cannot describe different memory.
pub fn copy_target_to_guest_pages(
    identity: &TargetIdentity,
    dst: &GuestPageTarget,
    pages: &[u64],
) -> Result<(), DrawError> {
    use host_ram::GuestWriteDecline;
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    if !ctx.caps.host_pointer.is_available() {
        return Err(DrawError::GuestPageWrite(GuestWriteDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        }));
    }
    unsafe { pools.ensure_init(ctx, counters)? };
    let snap = resident_read_snapshot(pools, identity)?;
    // Whole formats, not channel orders. Two formats sharing an order are four
    // bytes per texel apart once a render target may be wider than eight bits,
    // and this copy converts nothing — so an order comparison would admit a
    // half-float destination over an eight-bit resident.
    //
    // **Storage formats, though, and not view formats.** The two sides of this
    // comparison answer two different questions and one of them is not about
    // bytes. `snap.format` is `slot.format.declared()`, the format the
    // *attachment* was created at, and `translate::pixel::color_attachment` deliberately
    // keeps the guest's transfer function there so Vulkan performs the
    // fixed-function linear-to-sRGB encode on write. `dst.format` comes from
    // `vk_texel_layout`, which is the *stored texel* and has no transfer
    // function by construction — as does `TargetIdentity::Gva`'s own format, via
    // `draw::vulkan::gva_resident_format`.
    //
    // So a guest that declares `BGRA8Unorm_sRGB` for a private render target
    // arrives here with `held=B8G8R8A8_SRGB` and `want=B8G8R8A8_UNORM`, forever,
    // and this is the only place the two spellings meet. `storage_format` is
    // this crate's existing name for the fold between them, and
    // `has_bgra_order`'s doc already states the rule it encodes: "the transfer
    // function is deliberately irrelevant: UNORM and sRGB views interpret the
    // same four stored bytes". Vulkan agrees where it matters here —
    // `vkCmdCopyImageToBuffer` performs no format conversion, and `dst.format`
    // is read for nothing but `bytes_per_texel`.
    //
    // Every distinction the paragraph above defends survives the fold, because
    // `storage_format` maps only sRGB onto its own UNORM sibling: `R8G8B8A8` and
    // `B8G8R8A8` stay apart, and so do eight-bit and half-float.
    //
    // Measured: on a driven macos-13 Maps leg this refused **1 001 of 1 001**
    // payments, all on one texture, so the app's whole canvas stayed the zeros
    // its pages were allocated with. The declined fields keep the unfolded
    // spellings so a future firing still says which side held what.
    if !crate::backend::vulkan::translate::pixel::stored_bytes_agree(snap.format, dst.format) {
        return Err(DrawError::GuestPageWrite(
            GuestWriteDecline::ResidentFormatMismatch {
                held: snap.format,
                want: dst.format,
            },
        ));
    }
    if snap.width != dst.width || snap.height != dst.height {
        return Err(DrawError::GuestPageWrite(
            GuestWriteDecline::GeometryMoved {
                resident_width: snap.width,
                resident_height: snap.height,
                want_width: dst.width,
                want_height: dst.height,
            },
        ));
    }
    let need = dst.extent_end();
    let have = dst.window_bytes();
    if need > have {
        return Err(DrawError::GuestPageWrite(
            GuestWriteDecline::WindowTooSmall { need, have },
        ));
    }
    if shared_backing_settles(snap.guest_backing, dst.shared_backing) {
        crate::runtime::drain::note_store_route("target_sync_shared_backing");
        record_guest_write_debt(pools, identity, pages);
        return Ok(());
    }
    unsafe {
        // Dense rows are the common case and the cheap one; a padded pitch
        // falls to the rectangle path, which is the only form that can leave
        // the padding unwritten. Both land the same guest bytes.
        let plan = if dst.rows_are_dense() {
            // The same pool the draw-time gather draws from, and for the same
            // reason: this buffer is device-local, is written and then read by
            // transfer commands in one submission, and must not be reused or
            // freed until that submission's fence retires. A slot from here is
            // held in `gather_live` and returned by the ring, so both of those
            // are properties of the pool rather than of a caller's promise.
            //
            // Sized by `have` and not `need`. The detile writes `need` bytes
            // from offset 0, but the scatter below reads one range per run and
            // those sum to `have` — and the check above only establishes
            // `need <= have`, so `need` is the smaller of the two. They are in
            // fact equal wherever this branch is taken, because dense rows make
            // `extent_end` the same tight frame `references_for_runs` tiled;
            // that is a coincidence of two separately-derived numbers, not a
            // stated relation, and sizing by the one the copies actually read
            // costs nothing and does not depend on it holding.
            let scratch = pools.acquire_guest_gather(
                ctx,
                have,
                ash::vk::BufferUsageFlags::empty(),
                counters,
            )?;
            // The dispatch first, falling back to the regions it replaces —
            // which is the only ordering that keeps the transfer form reachable
            // on every host and for every run shape it still has to serve.
            let scatter = match compute_scatter_enabled() {
                true => plan_guest_scatter_dispatches(ctx, pools, counters, dst, &scratch)?
                    .map(ScatterForm::Dispatches),
                false => None,
            };
            let scatter = match scatter {
                Some(form) => form,
                None => ScatterForm::Regions(plan_guest_linear_copies(ctx, pools, dst)?),
            };
            counters.guest_write_linear.fetch_add(1, Ordering::Relaxed);
            GuestCopyPlan::Linear {
                scratch: scratch.buffer,
                // `buffer_row_length(0)` is Vulkan's "tightly packed", which is
                // exactly what dense means. Passing `row_length_texels` would
                // be the same number whenever it is set and an invalid one
                // (below `width`) if the guest ever understated it.
                detile: ash::vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(color_subresource_layers())
                    .image_offset(ash::vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(ash::vk::Extent3D {
                        width: dst.width,
                        height: dst.height,
                        depth: 1,
                    }),
                scatter,
            }
        } else {
            let plan = GuestCopyPlan::Rectangles(plan_guest_copies(ctx, pools, dst)?);
            counters.guest_write_rects.fetch_add(1, Ordering::Relaxed);
            plan
        };
        counters
            .guest_write_regions
            .fetch_add(plan.regions(), Ordering::Relaxed);
        counters
            .guest_write_dispatches
            .fetch_add(plan.dispatches(), Ordering::Relaxed);
        copy_image_level0_to_buffer(ctx, pools, counters, &snap, &plan)?;
        pools.registry_note_access(
            identity,
            pools::ResidentAccess::transfer_read(snap.guest_backing.is_some()),
        );
        counters.note_target_read(u64::from(dst.width) * u64::from(dst.height) * 4);
    }
    // Past the last fallible step, so this runs exactly when the copy is on the
    // queue. The ledger takes the resident's pin itself here — the caller holds
    // none, and `finish` clears `gpu_only_content` as soon as this returns, so
    // between that and the settle the pin is all that keeps the reclaim off an
    // image the submitted copy still reads. Safe to leave until the end rather
    // than guarding every early return above, because the whole body runs under
    // the engine lock and a reclaim needs the same lock: nothing can take the
    // image while this function is running, only after it returns.
    record_guest_write_debt(pools, identity, pages);
    Ok(())
}

/// Synchronize a resident that is already backed by its guest allocation.
///
/// The resource owns both its imported binding and the physical-page footprint
/// retained when that binding was admitted. Synchronization therefore names
/// only the resource: it publishes the existing GPU write to the completion
/// ledger and performs no destination reconstruction or copy.
pub fn synchronize_guest_backed_target(
    identity: &TargetIdentity,
) -> Result<crate::runtime::guest_ram::GuestPageFootprint, DrawError> {
    use host_ram::GuestWriteDecline;
    let mut guard = lock_engine();
    let pools = &mut guard.pools;
    let snap = resident_read_snapshot(pools, identity)?;
    if snap.guest_backing.is_none() {
        return Err(DrawError::GuestPageWrite(
            GuestWriteDecline::NoSharedBacking,
        ));
    }
    let footprint = snap.guest_footprint.ok_or(DrawError::GuestPageWrite(
        GuestWriteDecline::NoSharedBacking,
    ))?;
    crate::runtime::drain::note_store_route("target_sync_shared_backing");
    record_guest_write_footprint_debt(pools, identity, &footprint);
    Ok(footprint)
}

fn shared_backing_settles(
    resident: Option<GuestTargetBacking>,
    destination: Option<GuestTargetBacking>,
) -> bool {
    resident.is_some() && resident == destination
}

/// Publish one resident write to the fence/dependency ledger.
///
/// Both a queued image-to-buffer copy and rendering into shared backing owe the
/// same completion rule: guest code must not observe its completion stamp until
/// the queue has finished writing these pages.
pub(super) fn record_guest_write_debt(
    pools: &mut pools::ResourcePools,
    identity: &TargetIdentity,
    pages: &[u64],
) {
    pools.note_guest_write_recorded(identity);
    // Before the flag and under the same lock: a reader that observes the flag
    // set must observe a footprint that already names this write, or it would be
    // told "disjoint" about pages this write is landing in.
    arm_guest_write_pages(pages);
    // Published after the ledger entry and while the engine lock is still held,
    // so no thread can observe the flag clear while a write is outstanding.
    GUEST_WRITE_DEBT.store(true, std::sync::atomic::Ordering::Release);
}

/// Publish a write through the immutable footprint retained by the admitted
/// resident. This is the resource-owned form of [`record_guest_write_debt`].
pub(super) fn record_guest_write_footprint_debt(
    pools: &mut pools::ResourcePools,
    identity: &TargetIdentity,
    footprint: &crate::runtime::guest_ram::GuestPageFootprint,
) {
    pools.note_guest_write_recorded(identity);
    arm_guest_write_footprint(footprint);
    GUEST_WRITE_DEBT.store(true, std::sync::atomic::Ordering::Release);
}

/// Report the exact binding equation for one live packed guest surface.
///
/// This remains an observation entry point; resident creation consumes the
/// same planner independently so a probe cannot enable behavior that the live
/// allocation did not itself prove.
#[allow(clippy::too_many_arguments)]
pub fn probe_guest_backed_target(
    host_ptr: usize,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    width: u32,
    height: u32,
    format: ash::vk::Format,
) {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    let Ok(ctx) = owner.ensure(counters) else {
        crate::observe::off(format!(
            "vk_linear_target_window verdict=no_context format={format:?} {width}x{height}"
        ));
        return;
    };
    unsafe {
        linear_target_import::probe_window(
            ctx,
            host_ptr,
            allocation_len,
            plane_offset,
            guest_row_pitch,
            width,
            height,
            format,
            registry_target_usage(format),
        )
    };
}

/// How one frame gets from a resident image into the guest's stretches.
///
/// Two shapes, chosen by whether the guest's rows carry padding
/// ([`GuestPageTarget::rows_are_dense`]), and they land byte-identical guest
/// memory — the choice is a cost decision and never a visible one.
enum GuestCopyPlan {
    /// Straight from the image into guest RAM, as image-copy rectangles.
    ///
    /// The only form that can express "write these texels and not the padding
    /// between their rows", which is why a padded window must take it.
    Rectangles(Vec<(ash::vk::Buffer, Vec<ash::vk::BufferImageCopy>)>),
    /// Detile the whole frame once into a device-local scratch, then scatter
    /// it with plain byte ranges.
    ///
    /// # Why the extra hop is cheaper than the copy it removes
    ///
    /// The rectangle form makes the bus-crossing pass do two jobs at once:
    /// detile an optimal-tiled image *and* write ~1500 part-row rectangles to
    /// system memory, the largest of which is one 16 KiB stretch. Splitting
    /// them lets each run at its own best shape — the detile is one rectangle
    /// against device-local memory, and the crossing becomes ~507 linear reads
    /// with no row or format semantics for the driver to interpret per region.
    ///
    /// The frame is written twice instead of once, but only one of those
    /// writes crosses the bus, which is the one that was ever expensive on a
    /// discrete host.
    Linear {
        scratch: ash::vk::Buffer,
        /// The one rectangle that fills the scratch, tightly packed.
        detile: ash::vk::BufferImageCopy,
        /// How the scratch reaches the guest's stretches.
        scatter: ScatterForm,
    },
}

/// The two ways a detiled frame gets from the scratch into the guest's pages.
///
/// They write the same bytes to the same addresses — the kernel copies `uint`s
/// and carries no format, row or texel semantics — so which one runs is a cost
/// decision and never a visible one, exactly as the choice above it is.
enum ScatterForm {
    /// One `VkBufferCopy` per guest stretch, grouped by the buffer it lands in.
    ///
    /// The only form on a host without the guest-RAM import, the form for a run
    /// the dispatch cannot express, and the A/B baseline. See
    /// [`crate::env::COMPUTE_SCATTER`].
    Regions(Vec<(ash::vk::Buffer, Vec<ash::vk::BufferCopy>)>),
    /// One compute dispatch per destination buffer, over a run table.
    ///
    /// This rail is bound by the number of copy regions it issues rather than by
    /// the bytes in them, which is what makes replacing ~200 regions with one
    /// dispatch the repair — see [`guest_scatter`].
    Dispatches(Vec<ScatterGroup>),
}

/// One dispatch: every run of this writeback that lands in one guest buffer.
struct ScatterGroup {
    set: ash::vk::DescriptorSet,
    run_count: u32,
}

impl GuestCopyPlan {
    /// Copy regions this plan will submit, for the census.
    ///
    /// The detiling rectangle counts: it is a region the driver consumes, and
    /// leaving it out would make the linear path's total read as exactly the
    /// stretch count when it is one more than that.
    /// A dispatch contributes none: it is a grid, not a region list, and
    /// counting one as a region would hide the very thing this counter exists to
    /// show. A linear plan that dispatched therefore reads as exactly 1 — the
    /// detile — which is the reading that says the scatter left the copy engine.
    fn regions(&self) -> u64 {
        match self {
            Self::Rectangles(groups) => groups.iter().map(|(_, r)| r.len() as u64).sum(),
            Self::Linear { scatter, .. } => match scatter {
                ScatterForm::Regions(groups) => {
                    1 + groups.iter().map(|(_, r)| r.len() as u64).sum::<u64>()
                }
                ScatterForm::Dispatches(_) => 1,
            },
        }
    }

    /// Dispatches this plan will submit, for the census. Zero on every other
    /// form, which is what makes the pair a share rather than two counts.
    fn dispatches(&self) -> u64 {
        match self {
            Self::Rectangles(_) => 0,
            Self::Linear { scatter, .. } => match scatter {
                ScatterForm::Regions(_) => 0,
                ScatterForm::Dispatches(groups) => groups.len() as u64,
            },
        }
    }
}

/// How many contiguous sub-ranges the scatter probe cuts each run into.
///
/// Four rather than two so the effect is well clear of boot-to-boot spread, and
/// not more because the sub-ranges have to stay large enough that the answer is
/// about region count rather than about having made every copy tiny.
const SCATTER_SPLIT_PARTS: u64 = 4;

/// Whether the scatter probe is on. See its use site.
fn scatter_split_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            crate::env::read(crate::env::SCATTER_SPLIT).0,
            crate::env::Switch::On
        )
    })
}

/// Bind every run and turn it into one byte range each, grouped by the buffer
/// it lands in.
///
/// Only valid for a dense window: the caller has checked
/// [`GuestPageTarget::rows_are_dense`], which is what makes window byte `o`
/// and scratch byte `o` the same byte. Nothing in a `VkBufferCopy` carries row
/// or format semantics, so that identity is the whole of the arithmetic here —
/// which is why this planner has no geometry and no rectangles.
///
/// Grouped for the reason [`plan_guest_copies`] groups: two runs need not share
/// an import, and one `vkCmdCopyBuffer` names exactly one destination buffer.
unsafe fn plan_guest_linear_copies(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    dst: &GuestPageTarget,
) -> Result<Vec<(ash::vk::Buffer, Vec<ash::vk::BufferCopy>)>, DrawError> {
    use host_ram::GuestWriteDecline;
    let mut grouped: Vec<(ash::vk::Buffer, Vec<ash::vk::BufferCopy>)> = Vec::new();
    for run in &dst.runs {
        let bound = unsafe { pools.bind_guest_ram(ctx, &run.guest) }
            .map_err(|inner| DrawError::GuestPageWrite(GuestWriteDecline::Import { inner }))?;
        let copy = ash::vk::BufferCopy::default()
            // `head` is what the granularity rounding added in front of the
            // byte the caller asked for, so the run's first requested byte
            // sits here — the same re-basing the rectangle path does.
            .dst_offset(bound.offset + bound.head)
            .src_offset(run.window_offset)
            // `requested` and not `bound_len`: the latter is rounded out to the
            // import's granularity, and copying it would write guest bytes
            // either side of the window that this frame was never given.
            .size(run.guest.requested());
        // PROBE — `REIMS_VGPU_SCATTER_SPLIT=on`. Cuts each run into
        // `SCATTER_SPLIT_PARTS` contiguous sub-ranges that tile it exactly, so
        // the guest bytes written are identical and only the region *count*
        // changes. It exists to separate "this rail is bound by the bytes it
        // moves" from "it is bound by the number of copy regions it issues" —
        // the two predict opposite things about a compute scatter, and the host
        // GPU sitting at 86-91 % busy on 3-4 % memory utilization says it is not
        // the bytes. Default off; delete once the question is answered.
        if scatter_split_enabled() {
            let total = copy.size;
            let part = total / SCATTER_SPLIT_PARTS;
            if part != 0 {
                for i in 0..SCATTER_SPLIT_PARTS {
                    // The last part takes the remainder, so the sub-ranges tile
                    // the run exactly rather than losing `total % PARTS` bytes.
                    let len = if i == SCATTER_SPLIT_PARTS - 1 {
                        total - part * (SCATTER_SPLIT_PARTS - 1)
                    } else {
                        part
                    };
                    let off = part * i;
                    group_by_buffer(
                        &mut grouped,
                        bound.buffer,
                        ash::vk::BufferCopy::default()
                            .dst_offset(copy.dst_offset + off)
                            .src_offset(copy.src_offset + off)
                            .size(len),
                    );
                }
                continue;
            }
        }
        group_by_buffer(&mut grouped, bound.buffer, copy);
    }
    Ok(grouped)
}

/// Whether the compute scatter is on. See [`crate::env::COMPUTE_SCATTER`].
fn compute_scatter_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::env::read(crate::env::COMPUTE_SCATTER).0,
            crate::env::Switch::Off
        )
    })
}

/// Bind every run and turn the set into one compute dispatch per destination
/// buffer, or `Ok(None)` when this writeback has to take the transfer regions.
///
/// `Ok(None)` is a routing answer and never a loss: the caller falls back to
/// [`plan_guest_linear_copies`], which lands the identical bytes. Every reason
/// for one is named through [`guest_scatter::ScatterDecline`] so a boot quietly
/// running on the expensive path is visible rather than inferred from a frame
/// rate.
///
/// # Why the run table is host memory the shader reads in place
///
/// It is ~200 `uvec4`s — 3.2 KiB, past every push-constant limit and far below
/// anything worth a staging copy. A staging slot is host-visible, coherent and
/// already carries `STORAGE_BUFFER` usage, so writing it and binding it costs
/// this rail no copy region at all, which is the resource the whole change is
/// about.
unsafe fn plan_guest_scatter_dispatches(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &counters::EngineCounters,
    dst: &GuestPageTarget,
    scratch: &pools::BufferSlot,
) -> Result<Option<Vec<ScatterGroup>>, DrawError> {
    use guest_scatter::{build_run_tables, ScatterRun};
    use host_ram::GuestWriteDecline;
    let Some(pipeline) = (unsafe { pools.scatter_pipeline(ctx) }) else {
        return Ok(None);
    };
    let mut grouped: Vec<(ash::vk::Buffer, Vec<ScatterRun>)> = Vec::new();
    for run in &dst.runs {
        let bound = unsafe { pools.bind_guest_ram(ctx, &run.guest) }
            .map_err(|inner| DrawError::GuestPageWrite(GuestWriteDecline::Import { inner }))?;
        group_by_buffer(
            &mut grouped,
            bound.buffer,
            ScatterRun {
                src: run.window_offset,
                // The same re-basing every planner here does: `head` is what the
                // granularity rounding put in front of the byte asked for.
                dst: bound.offset + bound.head,
                len: run.guest.requested(),
            },
        );
    }
    // Planned for every group before anything is allocated, so a refusal in the
    // last group does not leave the first one's staging slot and descriptor set
    // sitting on the pools for a dispatch that will not be recorded.
    let mut tables = Vec::with_capacity(grouped.len());
    for (buffer, runs) in &grouped {
        match build_run_tables(
            runs,
            ctx.storage_buffer_offset_align,
            ctx.max_storage_buffer_range,
            // The window's own byte count and not the slot's, which is rounded
            // up to a power-of-two bucket. Both bound the memory soundly; this
            // is the tighter, and it is the one that catches a run reaching past
            // what the detile actually wrote rather than merely past the slot.
            dst.window_bytes(),
        ) {
            Ok(built) => tables.extend(built.into_iter().map(|t| (*buffer, t))),
            Err(decline) => {
                counters
                    .guest_write_scatter_declined
                    .fetch_add(1, Ordering::Relaxed);
                crate::observe::Emit::decline("scatter_plan", &decline).fail_once(0);
                return Ok(None);
            }
        }
    }
    // One staging slot for every table this writeback needs; see
    // [`stage_run_tables`]. This rail issues far fewer dispatches than the draw
    // -time gather does, so the saving is small here — it shares the arena
    // because a second copy of the placement arithmetic is a second place to
    // name a descriptor offset the driver will not accept.
    let words: Vec<&[u32]> = tables.iter().map(|(_, t)| &t.words[..]).collect();
    let (runs_slot, places) = unsafe { stage_run_tables(ctx, pools, counters, &words) }?;
    let mut groups = Vec::with_capacity(tables.len());
    for ((buffer, table), place) in tables.iter().zip(&places) {
        let set =
            unsafe { pools.alloc_scatter_descriptor_set(&ctx.device, pipeline.dsl, counters) }?;
        unsafe {
            guest_scatter::ScatterPipeline::write_set(
                &ctx.device,
                set,
                // The scratch is bound whole; the guest import is the windowed
                // side, because a RAMBlock is wider than `maxStorageBufferRange`.
                (scratch.buffer, 0, scratch.size),
                (*buffer, table.bind_offset, table.bind_range),
                (runs_slot.buffer, place.bind_offset, place.bind_range),
            );
        }
        groups.push(ScatterGroup {
            set,
            run_count: table.run_count,
        });
    }
    Ok(Some(groups))
}

/// A run table's `u32`s as the bytes a staging write takes.
///
/// Shared by both directions — the writeback's scatter and the buffer gather —
/// because it is the same table and the same staging write either way.
///
/// A local reinterpret rather than a dependency: `u32` has no padding and no
/// invalid bit patterns, and the destination is a `*mut u8` memcpy either way.
/// The endianness is the host's, which is the guest's, which is what the shader
/// reads — the same reasoning `write_completion_stamp` states for its
/// one word, one layer up.
pub(crate) fn run_table_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: `u32` is `Copy` with no padding, so any `[u32]` is a valid `[u8]`
    // of four times the length, and the borrow keeps the source alive.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

/// Where each of a submission's run tables sits inside the one staging slot they
/// share: byte offset and byte length, in the order they were given.
///
/// `bind_offset` is a multiple of `minStorageBufferOffsetAlignment` by
/// construction, which is what makes it legal as a `VkDescriptorBufferInfo`
/// offset; `bind_range` is the table's own length and not the padded stride, so
/// the shader's bound range never covers a neighbour's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunTablePlace {
    bind_offset: u64,
    bind_range: u64,
}

/// Lay `tables` end to end at `align` and produce the arena's bytes, along with
/// each table's place inside them.
///
/// Split out from [`stage_run_tables`] because it is the whole of the layout —
/// the arithmetic *and* which bytes land where — and it needs no device, so a
/// test can walk both. What is left in the caller is an acquire and a write.
fn pack_run_tables(tables: &[&[u32]], align: u64) -> (Vec<u8>, Vec<RunTablePlace>) {
    let align = align.max(1);
    let mut cursor = 0u64;
    let places: Vec<RunTablePlace> = tables
        .iter()
        .map(|words| {
            // `max(4)` because a zero-length table would give a descriptor a
            // zero range, which Vulkan refuses. It cannot arise — every planner
            // declines an empty table before it gets here — but the bound
            // belongs where the range is chosen rather than in a comment on the
            // planners.
            let len = (std::mem::size_of_val(*words) as u64).max(4);
            let place = RunTablePlace {
                bind_offset: cursor,
                bind_range: len,
            };
            // Pad to the next legal descriptor offset. `align` is a power of two
            // (Vulkan requires it of `minStorageBufferOffsetAlignment`), so this
            // is the round-up and not an approximation of one.
            cursor += len.div_ceil(align) * align;
            place
        })
        .collect();
    let mut packed = vec![0u8; cursor.max(4) as usize];
    for (place, words) in places.iter().zip(tables) {
        let at = place.bind_offset as usize;
        let bytes = run_table_bytes(words);
        packed[at..at + bytes.len()].copy_from_slice(bytes);
    }
    (packed, places)
}

/// Write a whole submission's run tables into **one** staging slot, each at an
/// offset a storage-buffer descriptor may name.
///
/// This is the arena that made the draw-time compute gather affordable. Each
/// dispatch used to take a staging slot of its own for a ~200-byte table, which
/// is an `acquire_staging` and a `write_staging` apiece — ~40 000 of each a
/// second on a driven macos-13 boot, against ~2 200 command buffers. Sharing one
/// slot makes it one of each per command buffer, and the descriptor's own
/// `offset` field is what tells a dispatch which table is its own, so nothing in
/// the kernel changes.
///
/// The writeback's scatter shares it for the same reason it shares
/// [`run_table_bytes`]: it is the same table and the same staging write, and a
/// second copy of the padding arithmetic is a second place to get a descriptor
/// offset wrong.
///
/// # Safety
///
/// `ctx` must be the device `pools` belongs to.
unsafe fn stage_run_tables(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &counters::EngineCounters,
    tables: &[&[u32]],
) -> Result<(pools::BufferSlot, Vec<RunTablePlace>), types::DrawError> {
    // One host-side buffer laid out exactly as the slot wants it, so the slot
    // takes one `write_staging`. A whole command buffer's tables come to a few
    // kilobytes, which is why this is cheaper than the per-table slots it
    // replaces even with the extra copy.
    let (packed, places) = pack_run_tables(tables, ctx.storage_buffer_offset_align);
    let slot = unsafe {
        pools.acquire_staging(
            ctx,
            packed.len() as u64,
            ash::vk::BufferUsageFlags::empty(),
            counters,
        )
    }?;
    unsafe { pools.write_staging(ctx, &slot, &packed) }?;
    Ok((slot, places))
}

/// Add one stretch's copy to the group for `buffer`, opening a group if this is
/// the first stretch to name it.
///
/// # Why every guest-memory planner groups
///
/// Two stretches of one window need not resolve against the same import: a
/// window straddling two RAMBlocks gives two `VkBuffer`s, and one `vkCmdCopy*`
/// names exactly one. Ordinary machines have one RAMBlock, so every group list
/// is a single entry and a planner that ignored this would look correct on every
/// host anyone runs — while landing, on a two-block machine, whichever part of
/// the window happened to come first.
///
/// One implementation because there are now four planners (the writeback's
/// rectangle and linear forms, the draw-time buffer gather, and any that
/// follows) and this is precisely the shape `AGENTS.md` warns about: a rule
/// written by hand N times, where the copies diverge in the arm nobody boots.
/// Generic over the copy type — `VkBufferCopy` and `VkBufferImageCopy` group
/// identically because the grouping is about the buffer, not the copy.
pub(super) fn group_by_buffer<C>(
    groups: &mut Vec<(ash::vk::Buffer, Vec<C>)>,
    buffer: ash::vk::Buffer,
    copy: C,
) {
    match groups.iter_mut().find(|(b, _)| *b == buffer) {
        Some((_, copies)) => copies.push(copy),
        None => groups.push((buffer, vec![copy])),
    }
}

/// Bind every run and turn it into copy rectangles, grouped by the buffer they
/// land in.
///
/// Grouped because two runs need not share an import: a window straddling two
/// RAMBlocks resolves against two `VkBuffer`s, and one `vkCmdCopyImageToBuffer`
/// names exactly one. Ordinary machines have one RAMBlock and this is a
/// single-entry `Vec`, but the grouping is what makes the two-block case land
/// the whole frame instead of the part that happened to be first.
unsafe fn plan_guest_copies(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    dst: &GuestPageTarget,
) -> Result<Vec<(ash::vk::Buffer, Vec<ash::vk::BufferImageCopy>)>, DrawError> {
    use host_ram::GuestWriteDecline;
    let geom = dst.geometry();
    let mut grouped: Vec<(ash::vk::Buffer, Vec<ash::vk::BufferImageCopy>)> = Vec::new();
    for run in &dst.runs {
        let bound = unsafe { pools.bind_guest_ram(ctx, &run.guest) }
            .map_err(|inner| DrawError::GuestPageWrite(GuestWriteDecline::Import { inner }))?;
        // `head` is what the granularity rounding added in front of the byte the
        // caller asked for, so the run's first requested byte sits here.
        let base = bound.offset + bound.head;
        let start = run.window_offset;
        let end = start.saturating_add(run.guest.requested());
        for r in reims_vgpu_paging::regions::plan_regions(&geom, start, end) {
            let region = ash::vk::BufferImageCopy::default()
                // The rectangle's own offset is in window bytes; `- start`
                // re-bases it onto this run, which is what `base` names.
                .buffer_offset(base + (r.window_offset - start))
                // In texels. This is the buffer-side row stride, and it is what
                // makes a merged multi-row rectangle name consecutive guest
                // rows — valid because within one run the guest bytes are
                // contiguous, so consecutive rows really are `pitch` apart.
                .buffer_row_length(dst.row_length_texels)
                .image_subresource(color_subresource_layers())
                .image_offset(ash::vk::Offset3D {
                    x: r.x as i32,
                    y: r.y as i32,
                    z: 0,
                })
                .image_extent(ash::vk::Extent3D {
                    width: r.width,
                    height: r.height,
                    depth: 1,
                });
            group_by_buffer(&mut grouped, bound.buffer, region);
        }
    }
    Ok(grouped)
}

/// Record and submit one image→buffer copy of a resident's level 0, without
/// waiting for it.
///
/// An `Ok` means the copy is on the queue; the caller owns everything that
/// follows from it not having executed yet.
///
/// # Safety
///
/// `buffer` must be bound to memory covering `dst`'s extent, and `snap` must
/// name a live image belonging to `ctx`.
unsafe fn copy_image_level0_to_buffer(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &counters::EngineCounters,
    snap: &ResidentReadSnapshot,
    plan: &GuestCopyPlan,
) -> Result<(), DrawError> {
    use crate::runtime::drain::{note_readback_phase, ReadbackPhase};
    let submit_started = std::time::Instant::now();
    // Appended to a recording batch where there is one, for the reason
    // `copy_image_level0_to_host_delivered` gives: `begin_entry` would submit
    // that batch only to submit this copy behind it, and the copy has to be
    // ordered after those draws either way.
    let appended = pools.batch_open_recording();
    let (cb, fence) = match appended {
        Some(pair) => {
            counters
                .batch_readback_joins
                .fetch_add(1, Ordering::Relaxed);
            pair
        }
        None => pools.begin_entry(ctx, counters)?,
    };
    if appended.is_none() {
        unsafe {
            pools.begin_slot_recording(
                ctx,
                cb,
                gpu_span::Kind::Store,
                VkOp::GuestWriteResetCb,
                VkOp::GuestWriteBeginCb,
            )?
        };
    }
    unsafe { pools.close_open_pass(&ctx.device, cb) };
    // The device's own clock, for the reason the readback rail takes it: `fence_us`
    // is CPU wall clock and cannot tell "the GPU is copying eight megabytes across
    // PCIe" from "the round trip costs more than the work". Those have opposite
    // fixes — a damage rect shrinks the first and does nothing at all to the
    // second — so the rail that is now most of a flush must not be read without
    // this pair. Slot 0 stamps the command buffer's start, slot 1 the point after
    // the barrier where the draws ahead are known done, slot 2 the end of the copy.
    //
    // The reset must be recorded into the same command buffer: a query pool's
    // results are undefined until reset, and resetting on the host needs
    // `hostQueryReset`, a Vulkan 1.2 feature this device does not ask for.
    unsafe { pools.readback_span_arm(ctx, cb) };
    // Unconditional, for the reason `copy_image_level0_to_host_delivered` states
    // at length: the barrier is a layout transition *and* a dependency, and this
    // rail needs the dependency whether or not the layout already matches. A
    // render pass leaves its attachment in [`caches::color0_pass_exit_layout`],
    // so the common case is a real transition too, and it must still order this
    // copy after the draws that produced the pixels.
    let read_access = pools::ResidentAccess::transfer_read(snap.guest_backing.is_some());
    let barrier = [ash::vk::ImageMemoryBarrier::default()
        .src_access_mask(RESIDENT_READ_SRC_ACCESS)
        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
        .old_layout(snap.layout)
        .new_layout(read_access.layout())
        .image(snap.image)
        .subresource_range(color_subresource_range())];
    ctx.device.cmd_pipeline_barrier(
        cb,
        ash::vk::PipelineStageFlags::ALL_COMMANDS,
        ash::vk::PipelineStageFlags::TRANSFER,
        feedback_transition_dependency(snap.layout),
        &[],
        &[],
        &barrier,
    );
    unsafe { pools.readback_span_mark(ctx, cb, ash::vk::PipelineStageFlags::TRANSFER, 1) };
    // One call per buffer, all of them into the same command buffer, so the
    // whole frame is still one submission and one fence however many RAMBlocks
    // it touched — and, on the linear plan, however many hops it takes.
    match plan {
        GuestCopyPlan::Rectangles(groups) => {
            for (buffer, regions) in groups {
                ctx.device.cmd_copy_image_to_buffer(
                    cb,
                    snap.image,
                    read_access.layout(),
                    *buffer,
                    regions,
                );
            }
        }
        GuestCopyPlan::Linear {
            scratch,
            detile,
            scatter,
        } => {
            let one = [*detile];
            ctx.device.cmd_copy_image_to_buffer(
                cb,
                snap.image,
                read_access.layout(),
                *scratch,
                &one,
            );
            // The scatter reads what the detile just wrote, and nothing in one
            // command buffer orders the two by itself. A global memory barrier
            // rather than a buffer one because there is exactly one buffer in
            // flight between them and no other access to exclude.
            match scatter {
                ScatterForm::Regions(groups) => {
                    let detiled = [ash::vk::MemoryBarrier::default()
                        .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)];
                    ctx.device.cmd_pipeline_barrier(
                        cb,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::DependencyFlags::empty(),
                        &detiled,
                        &[],
                        &[],
                    );
                    for (buffer, regions) in groups {
                        ctx.device.cmd_copy_buffer(cb, *scratch, *buffer, regions);
                    }
                }
                ScatterForm::Dispatches(groups) => {
                    // Two dependencies in one barrier because they have the same
                    // destination: the detile's write to the scratch, and the
                    // host's write of the run tables, which happened before this
                    // submission and so needs `HOST` named on the source side.
                    let ready = [ash::vk::MemoryBarrier::default()
                        .src_access_mask(
                            ash::vk::AccessFlags::TRANSFER_WRITE | ash::vk::AccessFlags::HOST_WRITE,
                        )
                        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ)];
                    ctx.device.cmd_pipeline_barrier(
                        cb,
                        ash::vk::PipelineStageFlags::TRANSFER | ash::vk::PipelineStageFlags::HOST,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::DependencyFlags::empty(),
                        &ready,
                        &[],
                        &[],
                    );
                    // Looked up rather than carried in the plan: the plan holds
                    // only what a dispatch needs that is per-writeback, and the
                    // pipeline is a fixture of the device. It is already built —
                    // the plan could not have been made otherwise.
                    if let Some(pipeline) = pools.scatter_pipeline(ctx) {
                        // One bind for the whole run; the handle never changes.
                        pipeline.bind(&ctx.device, cb);
                        for group in groups {
                            pipeline.dispatch(&ctx.device, cb, group.set, group.run_count);
                        }
                    }
                }
            }
        }
    }
    unsafe { pools.readback_span_mark(ctx, cb, ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE, 2) };
    // The reader of these bytes is the guest's vCPU, which is a host reader as
    // far as this device is concerned: the memory is guest RAM the driver
    // imported, not device-local memory that owes a readback. So the write is
    // released to `HOST` with `HOST_READ`, which is what makes it visible to a
    // CPU access after the fence signals.
    //
    // Cache maintenance beyond that is the driver's. A host-pointer import
    // names ordinary system pages this process already has mapped, and a PCIe
    // write to system memory is snooped, so there is no invalidate for this
    // side to issue.
    //
    // The source scope is the stage that actually wrote the guest's pages, which
    // is the dispatch on the compute scatter and the copy on every other form.
    // Naming `TRANSFER` alone against a dispatch would release the detile and
    // leave the writes the guest is about to read unordered — the one place in
    // this rail where the two forms are not interchangeable.
    let (wrote_stage, wrote_access) = match plan {
        GuestCopyPlan::Linear {
            scatter: ScatterForm::Dispatches(_),
            ..
        } => (
            ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            ash::vk::AccessFlags::SHADER_WRITE,
        ),
        _ => (
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::AccessFlags::TRANSFER_WRITE,
        ),
    };
    let host_visible = [ash::vk::MemoryBarrier::default()
        .src_access_mask(wrote_access)
        .dst_access_mask(ash::vk::AccessFlags::HOST_READ)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        wrote_stage,
        ash::vk::PipelineStageFlags::HOST,
        ash::vk::DependencyFlags::empty(),
        &host_visible,
        &[],
        &[],
    );
    // The wait this rail no longer takes here.
    //
    // What the stamp needs is that every copy has landed before the guest is
    // told anything, and that is not the same statement as "this copy has landed
    // before this call returns" — which is what the wait that used to sit here
    // asserted, once per window. The obligation is recorded instead and settled
    // by `quiesce_guest_writes` at the two places where it stops being this
    // device's own business: the completion stamp, and a host reader or writer
    // arriving at the same guest bytes.
    //
    if appended.is_some() {
        // Ends and submits the batch with the fence `batch_open_recording`
        // returned, and seals the batch's cleanup — one submission carrying the
        // draws and this copy together.
        pools.batch_flush(ctx, counters)?;
    } else {
        unsafe { pools.gpu_span_seal_current(ctx, cb) };
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::GuestWriteEndCb, e)))?;
        let cbs = [cb];
        ctx.submit_guest_work(&cbs, fence)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::GuestWriteSubmit, e)))?;
        let sealed = pools.seal_entry(Vec::new(), Vec::new());
        pools.finish_entry_async(&ctx.device, sealed);
    }
    note_readback_phase(
        ReadbackPhase::Submit,
        submit_started.elapsed().as_micros() as u64,
    );
    Ok(())
}

/// Import every RAMBlock in `imports` now, and report how many that took.
///
/// # Why the device does this before the guest asks
///
/// `vkAllocateMemory` with a host pointer chained is where a driver that pins
/// takes its reference on every page of the mapping, and the mapping is the
/// whole of guest RAM. A driven x86 boot on a discrete NVIDIA host measured
/// **2 493 029 µs for a 15 032 385 536-byte RAMBlock and 309 796 µs for a
/// 2 146 435 072-byte one**, with the properties query that precedes both at
/// 0 µs. Left to the first `gather` that references a block, that is ~2.8 s
/// charged to the guest's first frame — and the guest's display pipe abandons a
/// submitted transaction after 1000 ms, so the first frame of every boot missed
/// its own watchdog by a factor of three.
///
/// So the cost is not removed; it is moved to a caller that is not a frame. The
/// per-page work is the extension's, it is proportional to guest RAM, and
/// nothing this device does makes it cheaper — importing sub-ranges instead
/// would be the per-resource import [`host_ram`] exists to avoid.
///
/// # What the move bought, measured
///
/// Both x86 rails boot with all four of their RAMBlocks imported inside the
/// handshake, `guest_ram_warm blocks=4 bytes=17196384256` landing in the same
/// millisecond as the last `host_ram_import`. The first frame's `gather_us`
/// then reads **1 088 µs over 67 gathers and 14 722 144 bytes** on macos-11,
/// against 2 022 259 µs over 6 gathers and 1 176 768 bytes before — three
/// orders of magnitude less time for twelve times the bytes, which is what says
/// the gather itself never cost anything.
///
/// It does not cost the working rail: a macos-13 x86/PCI boot after the move
/// reaches its desktop with Dock and Finder running and the console owned by
/// the login user.
///
/// **It did not fix the macos-11 rail**, whose WindowServer still stops after
/// one composite. That guest does blow its 1000 ms display-transaction watchdog
/// with or without this, and it is now known not to be why it wedges.
///
/// Returns `(warmed, bytes)`: how many blocks this call actually imported and
/// how many bytes they covered. Zero blocks means either they were already
/// imported or the device is not up yet, and neither is a failure — a host with
/// no import capability never reaches this at all, because the resolution that
/// produces `imports` refuses first.
pub fn warm_guest_ram_imports(
    imports: &[std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>],
) -> (usize, u64) {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return (0, 0);
    };
    let mut warmed = 0usize;
    let mut bytes = 0u64;
    for import in imports {
        match unsafe { pools.warm_guest_ram(ctx, import) } {
            Ok(true) => {
                warmed += 1;
                bytes = bytes.saturating_add(import.len());
            }
            Ok(false) => {}
            // The draw path asks again and declines there with the same reason,
            // so this is reported and not propagated: a warm that could not
            // import must not be the thing that decides the rail is off.
            Err(inner) => crate::observe::Emit::decline("vk_guest_ram_warm", &inner).fail_once(0),
        }
    }
    (warmed, bytes)
}

/// Guest memory the device can currently reach through host-pointer imports,
/// and how many RAMBlocks that is.
///
/// # What the count is for
///
/// It is the only thing in the tree that can answer "how much guest RAM can the
/// device reach right now", and the *count* is the reading that says whether the
/// one-import-per-RAMBlock model held: it should be one or two for a whole boot,
/// and a count that tracks the workload is the per-resource import the model
/// exists to avoid.
///
/// Emitted every census window by `runtime::drain::census`'s
/// `guest_import_levels`, which is where the polarity is documented: this is a
/// level, flat is healthy, and a rise is the alarm. It is read every window
/// rather than once at import time precisely because one line at import time
/// cannot tell "imported once" from "imported once per window".
pub fn guest_import_census() -> (u64, usize, usize) {
    let guard = lock_engine();
    let (ramblocks, aliases, bytes) = guard.pools.host_ram_import_census();
    (bytes, ramblocks, aliases)
}

/// Bytes per texel a resident target readback delivers.
///
/// Not a property of the resident — of the *readback*. Its buffer is sized from
/// this and every consumer of the bytes reads them as RGBA8, so it is the one
/// number `resident_read_snapshot` admits a resident's format against. Naming
/// it once is what keeps the check and the buffer size the same number, rather
/// than two `4`s that a later widening could move apart.
const RESIDENT_READ_BYTES_PER_TEXEL: u32 = 4;

/// Bytes an image→buffer copy of one texel of `format` writes.
///
/// The single place a readback slot's size is decided, for both rails that size
/// one. `vkCmdCopyImageToBuffer` names an image extent and reads the *image's*
/// texel width per pixel, so a slot sized by any other number is either short —
/// which is a device-side write past the slot, not a truncated read — or larger
/// than the copy fills. The seed path answers the same question in the other
/// direction and states the same reason.
///
/// The fallback is unreachable for a real resident (an image exists only at a
/// format these tables know) and is the four this code used to assume rather
/// than a panic, on the same grounds as [`GuestPageTarget::bytes_per_texel`].
fn readback_bytes_per_texel(format: ash::vk::Format) -> u32 {
    crate::backend::vulkan::translate::pixel::bytes_per_texel(format)
        .unwrap_or(RESIDENT_READ_BYTES_PER_TEXEL)
}

/// Bring a readback taken at the attachment's own texel width down to the RGBA8
/// every consumer of drawn pixels speaks.
///
/// Both rails that read a colour target back share this: the standalone
/// [`read_target`] and the tail of a draw that could not defer its Store. They
/// used to be one narrowing and one four-byte assumption, and the assumption was
/// the older of the two — which is what made a wide attachment a buffer overrun
/// on one rail and a quantized frame on the other, for the same resident.
///
/// Narrowing rather than refusing, because both callers are the fallback a Store
/// takes when the GPU could not land the frame in guest pages directly. Refusing
/// loses the frame outright, where before a render target could be wide the same
/// frame was merely quantized on its way through an eight-bit resident.
///
/// Returns the bytes and the channel order they are in — `narrow_texel_to_rgba8`
/// produces semantic RGBA8 whatever the resident's order was, so a narrowed
/// frame owes no exchange and says so.
fn narrow_readback_to_rgba8(
    out: Vec<u8>,
    layout: crate::contract::pixel_format::TexelLayout,
    format: ash::vk::Format,
    pixels: u64,
    bgra: bool,
) -> Result<(Vec<u8>, bool), DrawError> {
    if layout.bytes_per_texel() == RESIDENT_READ_BYTES_PER_TEXEL {
        return Ok((out, bgra));
    }
    let count = u32::try_from(pixels).unwrap_or(u32::MAX);
    let mut narrowed = vec![0u8; (pixels * u64::from(RESIDENT_READ_BYTES_PER_TEXEL)) as usize];
    if !crate::contract::pixel_format::narrow_texel_to_rgba8(layout, &out, count, &mut narrowed) {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::TexelNotFourBytes { format },
        ));
    }
    // Visible, because it is a fidelity loss and not just a slow path: the
    // frame this returns carries eight bits of a channel the guest asked for
    // sixteen of. A non-zero reading names the population that would be
    // repaired by teaching this rail's consumers the wider texel.
    crate::runtime::drain::note_store_route("target_read_narrowed");
    Ok((narrowed, false))
}

/// The `srcAccessMask` a resident color target's readback must drain.
const RESIDENT_READ_SRC_ACCESS: ash::vk::AccessFlags = ash::vk::AccessFlags::from_raw(
    ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw()
        | ash::vk::AccessFlags::COLOR_ATTACHMENT_READ.as_raw()
        | ash::vk::AccessFlags::TRANSFER_WRITE.as_raw()
        | ash::vk::AccessFlags::SHADER_WRITE.as_raw()
        | ash::vk::AccessFlags::SHADER_READ.as_raw(),
);

/// Dependency flag required when a barrier transitions an image out of the
/// attachment-feedback-loop layout. Derived from the tracked old layout at the
/// two resident-copy barriers so no caller can forget the extension contract.
fn feedback_transition_dependency(layout: ash::vk::ImageLayout) -> ash::vk::DependencyFlags {
    if layout == ash::vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT {
        ash::vk::DependencyFlags::FEEDBACK_LOOP_EXT
    } else {
        ash::vk::DependencyFlags::empty()
    }
}

#[cfg(test)]
mod feedback_transition_tests {
    use super::*;

    #[test]
    fn only_the_feedback_layout_requires_the_feedback_dependency_flag() {
        assert_eq!(
            feedback_transition_dependency(
                ash::vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
            ),
            ash::vk::DependencyFlags::FEEDBACK_LOOP_EXT
        );
        for layout in [
            ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ] {
            assert_eq!(
                feedback_transition_dependency(layout),
                ash::vk::DependencyFlags::empty(),
                "{layout:?}"
            );
        }
    }
}

/// The per-call `VkOp` names for a resident target readback.
///
/// One set shared by the copying and leasing entry points because they are the
/// same six Vulkan calls on the same rail; splitting the slugs would claim a
/// distinction the failure does not have.
fn target_readback_ops() -> ReadbackOps {
    ReadbackOps {
        reset_cb: VkOp::ReadbackResetCb,
        begin_cb: VkOp::ReadbackBeginCb,
        end_cb: VkOp::ReadbackEndCb,
        submit: VkOp::ReadbackSubmit,
        map: VkOp::ReadbackMap,
        invalidate: VkOp::ReadbackInvalidate,
    }
}

/// What a resident target's copy moves, and the format it is in.
struct ResidentReadSnapshot {
    image: ash::vk::Image,
    width: u32,
    height: u32,
    layout: ash::vk::ImageLayout,
    /// The resident image's own format. A channel order used to be enough here
    /// because every resident was eight bits per channel; it is not once a
    /// render target may be wider, and `copy_target_to_guest_pages` compares
    /// this against the destination's format to decide whether a byte copy
    /// lands the right texel.
    format: ash::vk::Format,
    guest_backing: Option<GuestTargetBacking>,
    guest_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
}

impl ResidentReadSnapshot {
    /// Whether these texels are already in guest scanout order.
    ///
    /// **A channel-order question, and equality does not answer it.** `format`
    /// is `slot.format.declared()`, which `translate::pixel::color_attachment` builds
    /// so that it *keeps* the guest's transfer function — Vulkan then performs
    /// the fixed-function linear-to-sRGB encode on attachment writes. So a guest
    /// render target declared `BGRA8Unorm_sRGB` is resident as
    /// `B8G8R8A8_SRGB`, and comparing that against the single `SCANOUT_FORMAT`
    /// spelling answers "not BGRA" about four bytes stored B, G, R, A.
    ///
    /// This is the same confusion `copy_target_to_guest_pages` carries
    /// `stored_bytes_agree` for, one rail over, and it failed the same way in
    /// both directions: [`TargetReadback::into_rgba8`] then left BGRA bytes
    /// alone and the CPU Store converter exchanged them a second time, so the
    /// copying rails wrote every sRGB-declared target's frame into guest memory
    /// with R and B swapped — measured as a yellow Maps map layer on
    /// `REIMS_VGPU_GUEST_IMPORT=off`, against a correct one on the GPU-direct
    /// rail that never asks this question.
    ///
    /// [`translate::pixel::has_bgra_order`] is the crate's name for the
    /// question, is what [`TargetIdentity::is_bgra`] already asks, and states
    /// the rule in its own doc: the transfer function is irrelevant because
    /// UNORM and sRGB views interpret the same four stored bytes.
    ///
    /// # What the boot pair reads now
    ///
    /// Two undriven macos-13 boots from one snapshot, the gate confirmed at
    /// `host_pointer_import=supported` and `=disabled_by_env` respectively, with
    /// the copying arm confirmed to have run (26 `gva_flush_copied`, 87
    /// `render_flush_copied`, 172 `render_flush_leased`, and 259 declines that
    /// all landed): the restored Maps window's map strip reads mean R,G,B
    /// **182.4, 218.8, 230.4 on both arms** and zero black pixels on both. The
    /// exchange showed as R and B transposed in that mean — the map layer
    /// yellow rather than pale blue — so the three numbers being equal across
    /// the arms *and* ordered R &lt; G &lt; B is the reading, not the equality
    /// alone.
    fn bgra(&self) -> bool {
        crate::backend::vulkan::translate::pixel::has_bgra_order(self.format)
    }
}

#[cfg(test)]
mod resident_read_order_tests {
    use super::*;
    use crate::backend::vulkan::translate::pixel;

    fn snapshot(format: ash::vk::Format) -> ResidentReadSnapshot {
        ResidentReadSnapshot {
            image: ash::vk::Image::null(),
            width: 4,
            height: 2,
            layout: ash::vk::ImageLayout::UNDEFINED,
            format,
            guest_backing: None,
            guest_footprint: None,
        }
    }

    /// The sRGB sibling of the scanout format stores B, G, R, A, and a readback
    /// of it owes the same exchange the UNORM spelling does. Reported as RGBA
    /// instead, both copying Store rails exchange R and B exactly once on their
    /// way into the guest's pages — the only rails a host without
    /// `VK_EXT_external_memory_host` has.
    #[test]
    fn an_srgb_resident_reports_the_channel_order_its_unorm_sibling_does() {
        assert!(snapshot(pixel::SCANOUT_FORMAT).bgra());
        assert!(snapshot(ash::vk::Format::B8G8R8A8_SRGB).bgra());
        assert!(!snapshot(pixel::RESIDENT_RGBA_FORMAT).bgra());
        assert!(!snapshot(ash::vk::Format::R8G8B8A8_SRGB).bgra());
    }

    /// The order the engine reports for a resident and the order the identity
    /// declares for it must be one answer: the copying rails read the first and
    /// the GPU-direct rail is built against the second, and they land the same
    /// guest bytes only while the two agree. Both spellings a guest may declare
    /// for one render target are checked, because it is exactly the pair that
    /// `color_attachment` keeps apart and `vk_texel_layout` folds together.
    #[test]
    fn the_reported_order_agrees_with_the_identity_that_named_the_resident() {
        for declared in [pixel::SCANOUT_FORMAT, ash::vk::Format::B8G8R8A8_SRGB] {
            let identity = TargetIdentity::Gva {
                gva: 0x4000,
                width: 4,
                height: 2,
                generation: 1,
                format: pixel::storage_format(declared),
            };
            assert_eq!(
                snapshot(declared).bgra(),
                identity.is_bgra(),
                "{declared:?}"
            );
        }
    }

    /// The whole invariant the copying Store rails owe, end to end and in the
    /// guest's own vocabulary: a host that cannot write guest pages from the GPU
    /// must land byte-for-byte what `copy_target_to_guest_pages` would have.
    ///
    /// Both rails do it the same way — read the resident, `into_rgba8`, then
    /// `convert_rgba8_to_row` at the format the guest declared — so the pair has
    /// to compose to the identity over the resident's stored bytes. The resident
    /// format is taken from `color_attachment`, which is what the draw actually
    /// creates the attachment with, rather than named here; that is the step
    /// that keeps the guest's transfer function and so the step that made the
    /// two spellings of one declaration behave differently.
    #[test]
    fn a_copied_store_lands_the_bytes_the_gpu_direct_store_would_have() {
        use crate::contract::pixel_format as pf;
        const PIXELS: u32 = 3;
        // Deliberately asymmetric per channel, so an exchange of any two of them
        // is visible and an all-grey frame cannot pass.
        let stored: Vec<u8> = (0..PIXELS * 4).map(|i| (i as u8) * 7 + 1).collect();
        for format in [
            pf::MTL_FORMAT_BGRA8_UNORM,
            pf::MTL_FORMAT_BGRA8_UNORM_SRGB,
            pf::MTL_FORMAT_RGBA8_UNORM,
            pf::MTL_FORMAT_RGBA8_UNORM_SRGB,
        ] {
            let attachment = pixel::color_attachment(format)
                .expect("a renderable eight-bit format")
                .0;
            let readback = TargetReadback {
                pixels: stored.clone(),
                bgra: snapshot(attachment).bgra(),
            };
            let rgba = readback.into_rgba8();
            let mut landed = vec![0u8; stored.len()];
            assert!(
                pf::convert_rgba8_to_row(format, &rgba, PIXELS, &mut landed),
                "{format:#x}"
            );
            assert_eq!(landed, stored, "{format:#x} via {attachment:?}");
        }
    }
}

/// The registry slot's copy geometry, or the typed reason it cannot be read.
///
/// Shared by all three rails that copy out of a resident — the two host
/// readbacks and the GPU-direct guest-page write — so that "is this target
/// copyable" is decided once and answered with one vocabulary. It deliberately
/// does **not** bound the texel width: that is the readbacks' constraint, not
/// the resident's, and `readback_snapshot` below is where it is applied.
fn resident_read_snapshot(
    pools: &pools::ResourcePools,
    identity: &TargetIdentity,
) -> Result<ResidentReadSnapshot, DrawError> {
    let Some(slot) = pools.registry_get(identity) else {
        // The near-miss scan runs only here, on the refusal, and it is what
        // makes the refusal diagnosable: a target that exists under a different
        // generation and one that does not exist at all are opposite defects.
        let (how, held) = pools.registry_key_divergence(identity);
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::UnknownIdentity {
                asked: identity.generation(),
                held,
                how,
                // Asked on the refusal only, and asked of the same history the
                // sampled rail's `prior=` reads: an absent resident this device
                // reclaimed and one the guest never created are opposite
                // defects, and only this record can tell them apart.
                prior: pools.prior_reclaim(identity),
            },
        ));
    };
    if !slot.content_ready {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::NoReadyContent,
        ));
    }
    if slot.sample_count != 1 {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::MultisampleImage {
                sample_count: slot.sample_count,
            },
        ));
    }
    Ok(ResidentReadSnapshot {
        image: slot.image,
        width: slot.width,
        height: slot.height,
        layout: slot.access.layout(),
        format: slot.format.declared(),
        guest_backing: slot.memory.guest_backing(),
        guest_footprint: slot.memory.guest_footprint(),
    })
}

/// [`resident_read_snapshot`] plus what the **host readback** rails need on top
/// of it: how many bytes to ask the GPU for, and how to get RGBA8 out of them.
///
/// Both rails hand their bytes to consumers that only speak RGBA8 —
/// `TargetReadback::into_rgba8` exchanges channels in `chunks_exact_mut(4)`, and
/// the CPU Store rail converts from RGBA8 a row at a time — so a resident wider
/// than four bytes a texel has to be read at its own width and then narrowed.
///
/// Separate from the shared snapshot because the third caller —
/// `copy_target_to_guest_pages` — needs neither: it copies the resident's own
/// bytes into guest pages and never interprets them.
fn readback_snapshot(
    pools: &pools::ResourcePools,
    identity: &TargetIdentity,
) -> Result<
    (
        ResidentReadSnapshot,
        crate::contract::pixel_format::TexelLayout,
    ),
    DrawError,
> {
    let snap = resident_read_snapshot(pools, identity)?;
    let layout = crate::backend::vulkan::translate::pixel::texel_layout_of(snap.format).ok_or(
        DrawError::TargetRead(reason::TargetReadDecline::TexelNotFourBytes {
            format: snap.format,
        }),
    )?;
    Ok((snap, layout))
}

fn read_target_inner(identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let (snap, layout) = readback_snapshot(pools, identity)?;
    // Asked for at the resident's own width — the copy is a raw image→buffer
    // move and reads the image format's texel — and narrowed below if that is
    // not what the caller can read.
    let pixels = (snap.width as u64) * (snap.height as u64);
    let rb_size = pixels * u64::from(layout.bytes_per_texel());
    let read_access = pools::ResidentAccess::transfer_read(snap.guest_backing.is_some());
    unsafe {
        let out = copy_image_level0_to_host(
            ctx,
            pools,
            counters,
            snap.image,
            snap.layout,
            read_access.layout(),
            RESIDENT_READ_SRC_ACCESS,
            snap.width,
            snap.height,
            rb_size,
            target_readback_ops(),
        )?;
        pools.registry_note_access(identity, read_access);
        counters.note_target_read(rb_size);
        // A wide resident is quantized here rather than refused; see
        // `narrow_readback_to_rgba8` for why that direction is the safe one.
        let (pixels, bgra) =
            narrow_readback_to_rgba8(out, layout, snap.format, pixels, snap.bgra())?;
        Ok(TargetReadback { pixels, bgra })
    }
}

/// Full-frame readback of a resident target (present / Synchronize / Map / Store boundary).
pub fn read_target(identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
    read_target_inner(identity)
}

/// Run bounded maintenance for dead resources and already-free pool entries.
/// Live residency is controlled by resource lifetime and allocation pressure,
/// never by this clock.
pub fn maintain_resources(now_ms: u64) {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    unsafe {
        pools.advance_registry_maintenance(ctx, counters, now_ms);
    }
}

/// Snapshot of create/alloc/hit-miss counters (for tests and thrash proxies).
/// Live entries in each immutable-object cache:
/// `(shaders, layouts, passes, pipelines, samplers, compute_pipelines)`.
///
/// See [`caches::ObjectCaches::levels`] for what reading it answers.
pub fn object_cache_levels() -> [usize; 6] {
    lock_engine().caches.levels()
}

/// Active topology policy for one deferred-submit command buffer, for the
/// integration test that drives past it.
///
/// Exported from the live pool rather than restated in the test: unified and
/// discrete devices intentionally carry different capacities, and the
/// environment may narrow either one.
pub fn batch_max_draws() -> u64 {
    lock_engine().pools.batch_capacity()
}

/// Whether the live Vulkan device can execute colour-attachment self-sampling
/// through the attachment-feedback-loop contract. Tests use the answer to
/// distinguish the native rail from its required snapshot fallback.
pub fn attachment_feedback_loop_active() -> bool {
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .is_some_and(|ctx| ctx.features.attachment_feedback_loop_layout)
}

pub fn counter_snapshot() -> CounterSnapshot {
    let eng = lock_engine();
    let mut snap = eng.counters.snapshot();
    if let Some(owner) = eng
        .owner
        .ctx
        .as_ref()
        .and_then(|ctx| ctx.queue_owner.as_ref())
    {
        let (
            submits,
            queue_us,
            driver_us,
            present_transactions,
            present_queue_us,
            present_driver_us,
        ) = owner.stats();
        snap.queue_async_submits = submits;
        snap.queue_async_queue_us = queue_us;
        snap.queue_async_driver_us = driver_us;
        snap.queue_present_transactions = present_transactions;
        snap.queue_present_queue_us = present_queue_us;
        snap.queue_present_driver_us = present_driver_us;
    }
    // Sampled-cache recycle diagnostics live on ResourcePools (single-threaded
    // under this lock), not the atomic counters; merge them in here.
    let (free_hits, free_allocs, recycle_admits, recycle_cap_drops) = eng.pools.recycle_stats();
    snap.sampled_free_hits = free_hits;
    snap.sampled_free_allocs = free_allocs;
    snap.sampled_recycle_admits = recycle_admits;
    snap.sampled_recycle_cap_drops = recycle_cap_drops;
    let (t_hits, t_allocs, t_admits, t_cap_drops) = eng.pools.target_recycle_stats();
    snap.target_free_hits = t_hits;
    snap.target_free_allocs = t_allocs;
    snap.target_recycle_admits = t_admits;
    snap.target_recycle_cap_drops = t_cap_drops;
    let (reg_peak, reg_peak_bytes) = eng.pools.registry_pressure_stats();
    snap.registry_non_pinned_peak = reg_peak;
    snap.registry_non_pinned_peak_bytes = reg_peak_bytes;
    let (sole_peak, sole_peak_bytes) = eng.pools.registry_sole_copy_stats();
    snap.registry_sole_copy_peak = sole_peak;
    snap.registry_sole_copy_peak_bytes = sole_peak_bytes;
    let (cs_sole, cs_sole_bytes) = eng.pools.compute_storage_sole_copy_stats();
    snap.compute_storage_sole_copy_peak = cs_sole;
    snap.compute_storage_sole_copy_peak_bytes = cs_sole_bytes;
    snap.resident_resample_peak_ms = eng.pools.resident_resample_peak_ms();
    let (slab_held, slab_carved) = eng.pools.slab_held_bytes();
    snap.slab_held_bytes = slab_held;
    snap.slab_carved_bytes = slab_carved;
    snap
}

/// Reset create/alloc/hit-miss counters (not device_lost/recreates). For reuse-gate tests.
pub fn reset_draw_counters() {
    lock_engine().counters.reset();
}

/// Test-only: destroy device, clear recreate budget, rebuild on next draw.
pub fn test_reset_engine() {
    let mut g = lock_engine();
    RESIDENT_RESOURCE_EPOCH.fetch_add(1, Ordering::Release);
    // A healthy `DeviceContext` is kept across the reset; only the pools, the
    // caches and the owner's flags are rebuilt.
    //
    // This used to destroy the `VkDevice` and the `VkInstance` too, so a suite
    // creating an instance per test churned one per reset — and on this host's
    // driver the churn has a ceiling: past it `vkEnumeratePhysicalDevices`
    // returns `ERROR_INITIALIZATION_FAILED` and every later init fails
    // permanently. The failure lands wherever the count happens to run out, so
    // it read as one test being fragile and moved to a different test whenever
    // another was added ahead of it. Nothing about the *reset* needed a new
    // instance: a fresh registry, a fresh cache set and a cleared recreate
    // budget are what a test wants, and this is also the shape production runs
    // in — one instance for the life of the process.
    //
    // A poisoned context is the exception. That is what the device-loss tests
    // leave behind, its device really is gone, and reusing it would hand every
    // later test a dead one — so it is torn down and the next `ensure` builds a
    // replacement.
    let poisoned = g.owner.poisoned;
    if let Some(mut ctx) = g.owner.ctx.take() {
        // A healthy test reset reuses the device after destroying every pool.
        // Settle both the owner's FIFO and the GPU before those handles go.
        if !poisoned {
            let _ = ctx.queue_wait_idle();
        } else {
            ctx.queue_barrier();
        }
        unsafe {
            g.caches.destroy_all(&ctx.device);
            g.pools.destroy_all(&ctx.device);
        }
        if poisoned {
            unsafe { ctx.destroy() };
            g.owner = ContextOwner::new();
        } else {
            g.owner = ContextOwner::new();
            g.owner.ctx = Some(ctx);
        }
    } else {
        g.owner = ContextOwner::new();
    }
    g.caches = ObjectCaches::new();
    g.pools = ResourcePools::new();
    g.counters.reset_all();
}

/// Test hook: next execute reports device lost (named path).
pub fn test_force_device_lost_once() {
    lock_engine().owner.force_device_lost = true;
}

/// Test hook: flush any open batch and retire every in-flight ring slot so
/// pending pool cleanup recycles deterministically. Warm-path allocation-free
/// assertions depend on the ring phase without this.
pub fn test_quiesce_ring() {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    let _ = unsafe { pools.retire_all(ctx, counters) };
}

/// Recreate budget remaining / count (for tests).
pub fn device_recreate_count() -> u32 {
    lock_engine().owner.recreate_count
}

/// Mark context poisoned and flush as if device lost (tests that assert recreate cap).
pub fn test_poison_and_flush() {
    let mut g = lock_engine();
    g.counters.device_lost.fetch_add(1, Ordering::Relaxed);
    g.owner.mark_device_lost();
    g.flush_device_derived();
}

#[cfg(all(test, feature = "host-window"))]
mod device_loss_window_rail_tests {
    use super::*;

    /// A device-loss flush must leave the window rail claiming nothing.
    ///
    /// This fails on the tree before the flush published the flag. The flag was
    /// written only by the window thread, so a loss destroyed the presenter and
    /// left [`window_present_attached`] reading `true` with
    /// `window_presenter == None` — after which `device::window_publish` kept
    /// choosing a rail that was gone and every present failed
    /// [`EngineFacadeDecline::WindowPresenterNotAttached`], once logged and then
    /// silent, for the rest of the boot.
    ///
    /// It runs on any host because it needs no device: `test_poison_and_flush`
    /// reaches `flush_device_derived` with `owner.ctx` at `None`, which is
    /// exactly the arm where the take used to be skipped.
    /// A loss seen by a thread that cannot take the engine lock still gets
    /// recovered, because the drain's end-of-tranche flush consumes the latch.
    ///
    /// This is the shape that cost a whole boot: the stamp-completion thread saw
    /// `ERROR_DEVICE_LOST` in `vkWaitSemaphores`, announced the stamp anyway
    /// (correctly — a withheld stamp hangs the guest), and nothing anywhere
    /// poisoned the context. Every path in this backend that is not the draw or
    /// compute arm was written to "let the lost device surface on the next
    /// draw", and the guest stopped drawing *because* its work stopped
    /// completing. `flush_batched_draws` runs about once a second regardless,
    /// which is what makes it the right consumer.
    ///
    /// The assertion is on the window rail rather than on the context, because
    /// the flush is the observable half that needs no GPU: it is
    /// `flush_device_derived` that clears it, and only `on_device_lost` calls
    /// that from here.
    #[test]
    fn a_loss_latched_off_the_engine_lock_is_recovered_by_the_drain_flush() {
        let _ = device_lost::take_device_lost_seen();
        note_window_present_attached(true);
        device_lost::note_device_lost_seen();
        flush_batched_draws();
        assert!(
            !window_present_attached(),
            "the end-of-tranche flush must run the recovery for a latched loss"
        );
        assert!(
            !device_lost::take_device_lost_seen(),
            "and take the latch, so one loss is recovered once"
        );
    }

    #[test]
    fn a_device_loss_flush_leaves_no_window_rail_claimed() {
        note_window_present_attached(true);
        assert!(
            window_present_attached(),
            "precondition: the flag is what a successful attach publishes"
        );
        test_poison_and_flush();
        assert!(
            !window_present_attached(),
            "the presenter died with the device, so nothing may still claim it"
        );
        assert!(
            lock_engine().window_presenter.is_none(),
            "and the field it shadows agrees"
        );
    }
}

#[cfg(test)]
mod engine_lock_site_tests {
    use super::*;

    /// A thread that has not run a drain is not the drain worker, and one that
    /// has is.
    ///
    /// The whole value of the split is that a vCPU stalled inside an MMIO store
    /// is countable apart from the tranche that stalled it, and the only thing
    /// separating those two threads is this latch. Asserted on a fresh thread
    /// because the marker is thread-local: running it on the test's own thread
    /// would pass whatever the latch did, since the test would be both.
    #[test]
    fn a_thread_is_the_worker_only_after_it_has_drained() {
        let seen = std::thread::spawn(|| {
            let before = calling_site();
            mark_drain_thread();
            (before, calling_site())
        })
        .join()
        .expect("probe thread");
        assert_eq!(seen.0, EngineLockSite::Device, "before any drain");
        assert_eq!(seen.1, EngineLockSite::Worker, "after one drain");

        // And the latch does not leak across threads: a second thread that has
        // not drained still reports `Device`, however many have.
        let other = std::thread::spawn(calling_site).join().expect("probe two");
        assert_eq!(other, EngineLockSite::Device);
    }

    /// Every site has its own label and its own census slot. Two sharing either
    /// would put a stalled vCPU and the tranche that stalled it on one counter,
    /// which is the state this split exists to leave.
    #[test]
    fn every_site_has_its_own_slot_and_label() {
        let mut labels: Vec<_> = EngineLockSite::ALL.iter().map(|s| s.label()).collect();
        let count = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), count, "two sites share a label");
        let mut indices: Vec<_> = EngineLockSite::ALL.iter().map(|s| s.index()).collect();
        indices.sort_unstable();
        assert_eq!(
            indices,
            (0..count).collect::<Vec<_>>(),
            "indices must tile the census arrays exactly"
        );
    }
}

#[cfg(test)]
mod group_by_buffer_tests {
    use super::*;
    use ash::vk::Handle;

    fn buffer(raw: u64) -> ash::vk::Buffer {
        ash::vk::Buffer::from_raw(raw)
    }

    /// The case every host anyone develops on hides: a window whose stretches
    /// resolve against two RAMBlocks needs two `vkCmdCopy*` calls, and a planner
    /// that kept one region list would submit whichever import came first and
    /// leave the rest of the window holding stale bytes.
    ///
    /// Order matters as much as the split. Regions inside one group must stay in
    /// the order they were planned, because that order is the window's byte
    /// order and a copy list is what reconstructs it.
    #[test]
    fn stretches_split_by_import_and_keep_their_order_inside_each() {
        let mut groups: Vec<(ash::vk::Buffer, Vec<u32>)> = Vec::new();
        // Interleaved on purpose: a real two-block window alternates as the
        // guest's allocator walks across the boundary and back.
        for (buf, region) in [(1, 10), (2, 20), (1, 11), (2, 21), (1, 12)] {
            group_by_buffer(&mut groups, buffer(buf), region);
        }
        assert_eq!(groups.len(), 2, "two imports, two copy calls");
        assert_eq!(groups[0].0, buffer(1), "groups appear in first-seen order");
        assert_eq!(groups[0].1, vec![10, 11, 12]);
        assert_eq!(groups[1].0, buffer(2));
        assert_eq!(groups[1].1, vec![20, 21]);
    }

    /// The ordinary machine: one RAMBlock, one group, and no per-stretch call.
    #[test]
    fn one_import_stays_one_group() {
        let mut groups: Vec<(ash::vk::Buffer, Vec<u32>)> = Vec::new();
        for region in 0..5 {
            group_by_buffer(&mut groups, buffer(7), region);
        }
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), 5);
    }
}

#[cfg(test)]
mod guest_write_footprint_tests {
    use super::*;

    fn footprint() -> crate::runtime::guest_ram::GuestPageFootprint {
        crate::runtime::guest_ram::GuestPageFootprint::new(
            std::sync::Arc::from([0x1000, 0x2000, 0x9000]),
            0x1000,
        )
        .expect("valid footprint")
    }

    /// The whole point: a reader whose window shares no page with the
    /// outstanding writeback is let through, and one that shares a single page
    /// is not. The wrong answer here is a stale frame, so the overlapping case
    /// is asserted on a window that touches the footprint at exactly one page —
    /// the case an envelope check or an off-by-one bound would wave past.
    ///
    /// Serialized against the rest of the suite by `--test-threads=1`, which the
    /// global these exercise needs as much as the GPU tests do.
    #[test]
    fn a_disjoint_reader_is_let_through_and_a_touching_one_is_not() {
        clear_guest_write_pages();
        arm_guest_write_pages(&[0x4000, 0x9000, 0x2000]);
        let reach = guest_writes_reaching;
        assert_eq!(reach(&[0x1000, 0x3000, 0xa000]), GuestWriteReach::Disjoint);
        assert_eq!(reach(&[0x1000, 0x9000]), GuestWriteReach::Overlap);
        assert_eq!(reach(&[0x2000]), GuestWriteReach::Overlap);
        // Unsorted input on both sides: the arm sorts, the ask does not have to.
        assert_eq!(reach(&[0xf000, 0x4000, 0x1000]), GuestWriteReach::Overlap);
        clear_guest_write_pages();
    }

    #[test]
    fn a_resource_footprint_is_retained_once_and_keeps_scatter_gaps_disjoint() {
        clear_guest_write_pages();
        let allocation = footprint();
        arm_guest_write_footprint(&allocation);
        arm_guest_write_footprint(&allocation.clone());
        {
            let held = GUEST_WRITE_PAGES.lock().expect("ledger lock");
            assert_eq!(held.allocations.len(), 1, "one admitted allocation identity");
            assert!(held.armed.is_empty(), "no copied page-list shadow");
        }
        assert_eq!(guest_writes_reaching(&[0x9000]), GuestWriteReach::Overlap);
        assert_eq!(
            guest_writes_reaching(&[0x5000]),
            GuestWriteReach::Disjoint,
            "the physical gap is not part of the allocation"
        );
        clear_guest_write_pages();
    }

    /// A second writeback is *also* named, up to the ring, and a page belonging
    /// to any of them is an overlap. The single-entry version of this gave up
    /// naming on 85 % of arms.
    #[test]
    fn every_writeback_up_to_the_ring_is_named() {
        clear_guest_write_pages();
        for i in 0..pools::RING_DEPTH as u64 {
            arm_guest_write_pages(&[0x1000 * (i + 1)]);
        }
        assert_eq!(guest_writes_reaching(&[0x1000]), GuestWriteReach::Overlap);
        assert_eq!(
            guest_writes_reaching(&[0x1000 * pools::RING_DEPTH as u64]),
            GuestWriteReach::Overlap
        );
        assert_eq!(
            guest_writes_reaching(&[0x1000 * (pools::RING_DEPTH as u64 + 2)]),
            GuestWriteReach::Disjoint
        );
        clear_guest_write_pages();
    }

    /// Past the entry cap the ledger keeps naming: the overflowing writeback
    /// folds into the oldest entry instead of disabling the rail.
    ///
    /// Three assertions and each catches a different way to get this wrong.
    /// **The overflowing page is named** — dropping the entry instead would
    /// answer `Disjoint` for a page a copy is landing in, which is a stale frame
    /// served as fresh. **The absorbing entry's own page is still named** — a
    /// merge that overwrote rather than unioned would lose it just as quietly.
    /// **A page nobody named is still `Disjoint`** — that is the whole point,
    /// and it is what the old surrender got wrong 3 499 times on a driven boot,
    /// each one making every later reader block globally.
    #[test]
    fn an_arm_past_the_ring_merges_and_still_names_every_page() {
        clear_guest_write_pages();
        for i in 0..pools::RING_DEPTH as u64 {
            arm_guest_write_pages(&[0x1000 * (i + 1)]);
        }
        arm_guest_write_pages(&[0xdead_0000]);
        assert_eq!(
            guest_writes_reaching(&[0xdead_0000]),
            GuestWriteReach::Overlap,
            "the writeback that overflowed the cap must still be named"
        );
        assert_eq!(
            guest_writes_reaching(&[0x1000]),
            GuestWriteReach::Overlap,
            "the entry it merged into must keep its own pages"
        );
        assert_eq!(
            guest_writes_reaching(&[0x8000_0000]),
            GuestWriteReach::Disjoint,
            "a page no outstanding writeback names must still be let through"
        );
        // And it keeps holding past many more, since nothing retires an entry
        // until a settle clears them all.
        for i in 0..4 * pools::RING_DEPTH as u64 {
            arm_guest_write_pages(&[0xbeef_0000 + 0x1000 * i]);
        }
        assert_eq!(
            guest_writes_reaching(&[0xbeef_0000]),
            GuestWriteReach::Overlap
        );
        assert_eq!(guest_writes_reaching(&[0x1000]), GuestWriteReach::Overlap);
        assert_eq!(
            guest_writes_reaching(&[0x8000_0000]),
            GuestWriteReach::Disjoint
        );
        clear_guest_write_pages();
    }

    /// The settle that clears the debt flag clears the ledger with it, so the
    /// next writeback is named against an empty slate rather than against every
    /// page the boot has ever written back.
    #[test]
    fn a_settle_forgets_every_page_the_ledger_was_holding() {
        clear_guest_write_pages();
        for i in 0..=pools::RING_DEPTH as u64 {
            arm_guest_write_pages(&[0x1000 * (i + 1)]);
        }
        assert_eq!(guest_writes_reaching(&[0x1000]), GuestWriteReach::Overlap);
        clear_guest_write_pages();
        arm_guest_write_pages(&[0x4000]);
        assert_eq!(guest_writes_reaching(&[0x8000]), GuestWriteReach::Disjoint);
        assert_eq!(
            guest_writes_reaching(&[0x1000]),
            GuestWriteReach::Disjoint,
            "a page from before the settle is no longer outstanding"
        );
        clear_guest_write_pages();
    }

    /// With nothing armed there is nothing to rule out *against*, and the safe
    /// answer is to settle. Callers only ask when the debt flag is set, so this
    /// state is a race the answer has to be conservative about rather than a
    /// path worth optimising.
    #[test]
    fn an_unarmed_footprint_rules_nothing_out() {
        clear_guest_write_pages();
        assert_eq!(guest_writes_reaching(&[0x1000]), GuestWriteReach::Unnamed);
    }

    /// The span pre-test narrows work, never a verdict, and this is the case
    /// that says so: two interleaved page sets whose spans overlap completely
    /// while sharing no page at all. A span test that concluded anything from
    /// the overlap would answer `Overlap` here and cost a wait nobody owes; one
    /// that short-circuited the other way would answer `Disjoint` for the test
    /// below and serve a stale frame.
    #[test]
    fn overlapping_spans_are_decided_page_by_page() {
        clear_guest_write_pages();
        let armed: Vec<u64> = (0..64).map(|i| 0x1000 + i * 0x2000).collect();
        arm_guest_write_pages(&armed);

        let interleaved: Vec<u64> = (0..64).map(|i| 0x2000 + i * 0x2000).collect();
        assert_eq!(
            guest_writes_reaching(&interleaved),
            GuestWriteReach::Disjoint,
            "same span, no shared page"
        );

        let mut sharing = interleaved.clone();
        sharing.push(armed[40]);
        assert_eq!(
            guest_writes_reaching(&sharing),
            GuestWriteReach::Overlap,
            "one shared page in the middle of the span still overlaps"
        );
        clear_guest_write_pages();
    }

    /// A reader whose pages sit wholly below and wholly above the armed span.
    /// Both are the rejection the span test is for, and the second one is the
    /// direction an off-by-one would break: `hi` is the reader's last page, not
    /// one past it.
    #[test]
    fn a_reader_clear_of_the_armed_span_is_disjoint_from_either_side() {
        clear_guest_write_pages();
        arm_guest_write_pages(&[0x8000, 0x9000, 0xa000]);
        assert_eq!(guest_writes_reaching(&[0x6000, 0x7000]), GuestWriteReach::Disjoint);
        assert_eq!(guest_writes_reaching(&[0xb000, 0xc000]), GuestWriteReach::Disjoint);
        assert_eq!(
            guest_writes_reaching(&[0x7000, 0x8000]),
            GuestWriteReach::Overlap,
            "abutting the low end shares its first page"
        );
        assert_eq!(
            guest_writes_reaching(&[0xa000, 0xb000]),
            GuestWriteReach::Overlap,
            "abutting the high end shares its last page"
        );
        clear_guest_write_pages();
    }
}

#[cfg(test)]
mod engine_lock_census_tests {
    use super::*;

    /// A boot where nothing ever touched the engine must not emit a line of
    /// zeros: `engine_lock` is read as "the window waited this long", and a row
    /// of zeros published every second on an idle device trains a reader to
    /// skip the line on the second where it finally says something.
    #[test]
    fn an_untaken_lock_emits_no_line() {
        let census = EngineLockCensus::new();
        assert!(census.take(1000).is_none());
    }

    /// The two sites are separate ledgers. A worker acquire must not move any
    /// window column, because the whole point of the split is to say which
    /// thread paid.
    #[test]
    fn each_site_keeps_its_own_waits_and_holds() {
        let census = EngineLockCensus::new();
        census.note_uncontended(EngineLockSite::Worker);
        census.note_wait(EngineLockSite::Worker, 40);
        census.note_hold(EngineLockSite::Worker, 900);
        census.note_wait(EngineLockSite::Window, 7000);
        census.note_hold(EngineLockSite::Window, 120);
        let line = census.take(1000).expect("a taken lock emits");
        assert!(
            line.contains(" worker=2 worker_blocked=1 worker_wait_us=40"),
            "{line}"
        );
        assert!(line.contains(" worker_hold_us=900"), "{line}");
        assert!(
            line.contains(" window=1 window_blocked=1 window_wait_us=7000"),
            "{line}"
        );
        assert!(line.contains(" window_hold_us=120"), "{line}");
    }

    /// `wait_max_us` and `hold_max_us` are maxima, not second sums of the
    /// totals beside them. One 30 ms hold and thirty 1 ms holds are the same
    /// `hold_us` and mean opposite things for a window trying to present
    /// between them, and the max is the only column that separates them.
    #[test]
    fn the_max_columns_are_maxima_not_totals() {
        let census = EngineLockCensus::new();
        for us in [3_u64, 30_000, 12] {
            census.note_wait(EngineLockSite::Window, us);
            census.note_hold(EngineLockSite::Worker, us);
        }
        let line = census.take(1000).expect("a taken lock emits");
        assert!(line.contains(" window_wait_us=30015"), "{line}");
        assert!(line.contains(" window_wait_max_us=30000"), "{line}");
        assert!(line.contains(" worker_hold_us=30015"), "{line}");
        assert!(line.contains(" worker_hold_max_us=30000"), "{line}");
    }

    /// Draining is what makes the line a rate. A second window must report the
    /// second window's traffic, not the running total since boot.
    #[test]
    fn taking_the_window_resets_it() {
        let census = EngineLockCensus::new();
        census.note_uncontended(EngineLockSite::Worker);
        census.note_hold(EngineLockSite::Worker, 500);
        assert!(census.take(1000).is_some());
        assert!(census.take(1000).is_none());
        census.note_uncontended(EngineLockSite::Window);
        let line = census.take(1000).expect("the second window emits");
        assert!(line.contains(" worker=0 worker_blocked=0"), "{line}");
        assert!(line.contains(" worker_hold_us=0"), "{line}");
        assert!(line.contains(" window=1"), "{line}");
    }

    /// The real acquire attributes to the site it was asked for, and times a
    /// wait it actually took. Serialized against the rest of the suite by
    /// `--test-threads=1`, which every GPU-touching test here already needs.
    #[test]
    fn a_contended_acquire_charges_the_site_that_blocked() {
        let _ = ENGINE_LOCK.take(0);
        let held = lock_engine_at(EngineLockSite::Worker);
        let waiter = std::thread::spawn(|| {
            drop(lock_engine_at(EngineLockSite::Window));
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(held);
        waiter.join().expect("the waiting thread acquires");
        let line = ENGINE_LOCK.take(1000).expect("both acquires are counted");
        assert!(line.contains(" window=1 window_blocked=1"), "{line}");
        let waited: u64 = line
            .split(" window_wait_us=")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .and_then(|value| value.parse().ok())
            .expect("window_wait_us parses");
        assert!(waited >= 10_000, "waited only {waited} us: {line}");
    }
}

#[cfg(test)]
mod guest_page_target_tests {
    use super::*;

    #[test]
    fn synchronization_settles_only_the_identical_shared_allocation() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x8000,
            plane_offset: 0x2000,
            row_pitch: 256,
        };
        assert!(shared_backing_settles(Some(backing), Some(backing)));
        assert!(!shared_backing_settles(Some(backing), None));
        assert!(!shared_backing_settles(
            Some(backing),
            Some(GuestTargetBacking {
                plane_offset: 0x3000,
                ..backing
            })
        ));
    }

    /// A target over a synthetic import large enough that the bound under test
    /// is the extent arithmetic and not the import's own length.
    fn target(width: u32, height: u32, row_length_texels: u32) -> GuestPageTarget {
        use crate::runtime::guest_ram::{GuestRamImport, GuestRamRegion, GuestRef};
        let import = std::sync::Arc::new(
            GuestRamImport::new(
                GuestRamRegion {
                    gpa_base: 0,
                    host_va: 0x7f00_0000_0000,
                    len: 1 << 30,
                },
                4096,
            )
            .expect("a plausible RAMBlock"),
        );
        let slice = import.slice(0, 1 << 20).expect("inside");
        GuestPageTarget {
            // One run covering the whole window: this fixture is about the
            // extent arithmetic, and a contiguous window is exactly what
            // `references_for_runs` returns a single run for.
            runs: vec![crate::runtime::guest_ram_map::GuestWindowRun {
                window_offset: 0,
                guest: GuestRef::new(import, slice).expect("its own import"),
            }],
            row_length_texels,
            width,
            height,
            // The format is checked against the resident's and no fixture here
            // reaches a resident, so only its texel width matters — these cases
            // are all four-byte extent arithmetic.
            format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
            shared_backing: None,
        }
    }

    /// The bound stops at the last row's last texel, not at a whole row pitch
    /// past it.
    ///
    /// Padding after the final row belongs to the surface's plane but is not a
    /// texel this copy was given, and the copying rail does not write it either.
    /// A bound that included it would make the two rails land different guest
    /// memory for one frame — the same divergence
    /// `write_full_rect_raw_staged_leaves_inter_row_padding_alone` guards on the
    /// CPU side.
    #[test]
    fn the_write_bound_ends_at_the_last_texel_and_not_a_row_pitch_past_it() {
        // 8 rows of 4 texels at a 16-texel pitch: 7 full pitches plus one packed
        // row, not 8 full pitches.
        let padded = target(4, 8, 16);
        assert_eq!(padded.extent_end(), 7 * 16 * 4 + 4 * 4);
        // A tight frame is the same expression with the pitch equal to the
        // width, which reduces to the whole frame.
        let tight = target(4, 8, 4);
        assert_eq!(tight.extent_end(), 4 * 8 * 4);
    }

    /// A zero row length means tight rows, which is what `bufferRowLength`
    /// itself means — so the bound must read it as the frame's own width rather
    /// than as a zero pitch that collapses every row onto the first.
    #[test]
    fn a_zero_row_length_is_a_tight_pitch_and_not_a_zero_one() {
        assert_eq!(
            target(32, 4, 0).extent_end(),
            target(32, 4, 32).extent_end()
        );
    }

    /// The linear path is taken exactly when the window has no inter-row
    /// padding, whichever way the guest spelt the pitch.
    ///
    /// A guest that understates the row length is the case to watch: the pitch
    /// is `max(row_length, width)`, so such a window *is* dense, and the
    /// detiling copy must therefore not pass `row_length_texels` through as
    /// `bufferRowLength` — a value below `width` is invalid there. It passes
    /// zero, which is Vulkan's own spelling of the tight packing this predicate
    /// just established.
    #[test]
    fn a_window_is_dense_exactly_when_its_rows_carry_no_padding() {
        assert!(target(64, 4, 64).rows_are_dense(), "pitch equal to width");
        assert!(target(64, 4, 0).rows_are_dense(), "zero means tight");
        assert!(
            target(64, 4, 32).rows_are_dense(),
            "an understated row length still yields a width pitch"
        );
        assert!(
            !target(64, 4, 65).rows_are_dense(),
            "one texel of padding per row is padding"
        );
    }

    /// A dense window's texels fill it end to end, which is the identity the
    /// linear path's arithmetic rests on: window byte `o` is scratch byte `o`
    /// only if there is no byte in between that belongs to neither.
    ///
    /// Asserted as the relation `extent == pitch * height` rather than against
    /// a literal, so it keeps holding whatever geometry a later reader adds.
    #[test]
    fn a_dense_window_is_exactly_its_rows_end_to_end() {
        for (w, h) in [(64u32, 4u32), (1920, 1080), (1, 1)] {
            let t = target(w, h, 0);
            assert!(t.rows_are_dense());
            assert_eq!(
                t.extent_end(),
                t.pitch_bytes() * u64::from(h),
                "{w}x{h} leaves no gap between the last texel and the extent"
            );
        }
        // The padded case is the contrast: its extent stops short of the last
        // row's padding, so its bytes are *not* contiguous texels and a byte
        // range over one would write bytes this frame was never given.
        let padded = target(64, 4, 65);
        assert!(padded.extent_end() < padded.pitch_bytes() * 4);
    }

    /// The census counts the detiling rectangle, because the driver consumes it
    /// like any other region.
    ///
    /// Leaving it out would make a linear boot's `guest_write_regions` read as
    /// exactly the stretch count, so the census would understate what this rail
    /// submits by one, every frame — and that census is now the only account of
    /// how wide a writeback gets, since nothing caps it.
    #[test]
    fn the_region_census_counts_every_region_a_plan_submits() {
        let null = ash::vk::Buffer::null();
        let rects = GuestCopyPlan::Rectangles(vec![
            (null, vec![ash::vk::BufferImageCopy::default(); 3]),
            (null, vec![ash::vk::BufferImageCopy::default(); 2]),
        ]);
        assert_eq!(rects.regions(), 5, "every group's rectangles, summed");

        let linear = GuestCopyPlan::Linear {
            scratch: null,
            detile: ash::vk::BufferImageCopy::default(),
            scatter: ScatterForm::Regions(vec![(null, vec![ash::vk::BufferCopy::default(); 507])]),
        };
        assert_eq!(linear.regions(), 508, "507 stretches plus the detile");
        assert_eq!(linear.dispatches(), 0, "the transfer form dispatches none");
    }

    /// The same 507 stretches as a dispatch must read as one region and one
    /// dispatch, because the pair is the only account of which form a boot took.
    ///
    /// Counting the dispatch as a region would make the two forms
    /// indistinguishable in the census — 508 either way — which is the reading
    /// the whole change exists to move.
    #[test]
    fn a_dispatched_scatter_reads_as_one_region_and_one_dispatch() {
        let dispatched = GuestCopyPlan::Linear {
            scratch: ash::vk::Buffer::null(),
            detile: ash::vk::BufferImageCopy::default(),
            scatter: ScatterForm::Dispatches(vec![ScatterGroup {
                set: ash::vk::DescriptorSet::null(),
                run_count: 507,
            }]),
        };
        assert_eq!(dispatched.regions(), 1, "the detile, and nothing else");
        assert_eq!(dispatched.dispatches(), 1, "one destination buffer");
    }
}

#[cfg(test)]
mod readback_width_tests {
    use super::*;
    use crate::contract::pixel_format::TexelLayout;

    /// The slot a readback is taken into and the narrowing that consumes it must
    /// derive their texel width from the same place.
    ///
    /// This is the pair that was one function and one literal `4`. A readback
    /// slot is filled by `vkCmdCopyImageToBuffer`, which names an image extent
    /// and reads the *image's* texel per pixel, so a slot sized narrower than
    /// the attachment is a device-side write past it — the failure is a GPU
    /// overrun, not a truncated frame, and no reading of the pixels can see it.
    ///
    /// The property asserted is that **no layout is ever refused for being
    /// short-sized**. Some layouts have no narrowing to RGBA8 at all — a
    /// single-channel or two-channel texel is not a frame a `DrawOutput`
    /// consumer can read — and those refuse whatever they are handed. Telling
    /// the two apart is the whole test: a refusal that a *larger* buffer would
    /// have satisfied is the sizer being wrong, and that is the bug. Driving it
    /// off `TexelLayout::ALL` means a layout added to the contract is swept the
    /// moment it exists.
    #[test]
    fn no_readback_layout_is_refused_for_being_short_sized() {
        const W: u64 = 7;
        const H: u64 = 5;
        const PIXELS: u64 = W * H;
        for &layout in TexelLayout::ALL {
            let format = crate::backend::vulkan::translate::pixel::vk_texel_layout(layout);
            let sized = (PIXELS * u64::from(readback_bytes_per_texel(format))) as usize;
            match narrow_readback_to_rgba8(vec![0u8; sized], layout, format, PIXELS, true) {
                Ok((pixels, bgra)) => {
                    assert_eq!(
                        pixels.len(),
                        (PIXELS * u64::from(RESIDENT_READ_BYTES_PER_TEXEL)) as usize,
                        "{layout:?}: a consumer of drawn pixels reads RGBA8"
                    );
                    // A four-byte layout is handed back untouched, so it keeps
                    // the order it was read in; a narrowed one is semantic RGBA8
                    // and owes no exchange. Both rails pass this straight on.
                    assert_eq!(
                        bgra,
                        layout.bytes_per_texel() == RESIDENT_READ_BYTES_PER_TEXEL,
                        "{layout:?}: reported the wrong channel order for its rail"
                    );
                }
                Err(_) => {
                    // Refused. It must be the layout and not the size, so hand
                    // the same narrowing a buffer no sizing error could make too
                    // small and require the same answer.
                    let mut dst = vec![0u8; (PIXELS * 4) as usize];
                    assert!(
                        !crate::contract::pixel_format::narrow_texel_to_rgba8(
                            layout,
                            &vec![0u8; sized * 8],
                            PIXELS as u32,
                            &mut dst,
                        ),
                        "{layout:?}: a bigger buffer was accepted, so the slot was sized short"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod probe_visibility_tests {
    use super::*;

    #[test]
    fn each_engine_probe_preserves_the_typed_initialization_reason() {
        let error = vk_call::exec_submit_device_lost_fixture();
        for probe in [
            EngineProbe::StorageWriteWithoutFormat,
            EngineProbe::SampledLayoutLinearFilter,
        ] {
            let line = engine_probe_decline(probe, &error).render();
            assert!(line.starts_with("vk_engine_probe reason=vk_exec_submit "));
            assert!(line.ends_with(&format!(" probe={}", probe.name())));
        }
    }
}

#[cfg(test)]
mod run_table_arena_tests {
    use super::*;

    /// `minStorageBufferOffsetAlignment` on the three hosts this rail runs:
    /// 16 on Apple/MoltenVK, 32 on the NVIDIA proprietary driver, 256 on the
    /// several drivers that report the Vulkan maximum-permitted value.
    const ALIGNS: [u64; 3] = [16, 32, 256];

    fn table(runs: usize, seed: u32) -> Vec<u32> {
        (0..runs as u32 * 4).map(|w| w ^ seed).collect()
    }

    /// A descriptor offset the driver refuses is the whole failure mode this
    /// arena introduced, so it is the first thing asserted.
    #[test]
    fn every_place_starts_at_a_legal_descriptor_offset() {
        for align in ALIGNS {
            let words: Vec<Vec<u32>> = (1..=9).map(|n| table(n, n as u32)).collect();
            let borrowed: Vec<&[u32]> = words.iter().map(|w| &w[..]).collect();
            let (_, places) = pack_run_tables(&borrowed, align);
            for place in &places {
                assert_eq!(
                    place.bind_offset % align,
                    0,
                    "align {align}: offset {} is not a multiple",
                    place.bind_offset
                );
            }
        }
    }

    /// A range reaching into the next table would let a dispatch read runs that
    /// belong to another gather — a wrong window, not a slow one.
    #[test]
    fn no_places_bound_range_reaches_its_neighbour() {
        for align in ALIGNS {
            let words: Vec<Vec<u32>> = (1..=9).map(|n| table(n, n as u32)).collect();
            let borrowed: Vec<&[u32]> = words.iter().map(|w| &w[..]).collect();
            let (packed, places) = pack_run_tables(&borrowed, align);
            for pair in places.windows(2) {
                assert!(
                    pair[0].bind_offset + pair[0].bind_range <= pair[1].bind_offset,
                    "align {align}: {:?} overlaps {:?}",
                    pair[0],
                    pair[1]
                );
            }
            let last = places.last().expect("nine tables");
            assert!(
                last.bind_offset + last.bind_range <= packed.len() as u64,
                "align {align}: last place reaches past the arena"
            );
        }
    }

    /// The bytes a dispatch reads at its own place must be its own table's, in
    /// order. This is the property the per-table staging slots gave for free and
    /// the arena has to earn.
    #[test]
    fn each_table_reads_back_its_own_words_at_its_own_place() {
        for align in ALIGNS {
            let words: Vec<Vec<u32>> = (1..=9)
                .map(|n| table(n * 3, 0xa5a5_0000 + n as u32))
                .collect();
            let borrowed: Vec<&[u32]> = words.iter().map(|w| &w[..]).collect();
            let (packed, places) = pack_run_tables(&borrowed, align);
            for (place, want) in places.iter().zip(&words) {
                let at = place.bind_offset as usize;
                let bytes = &packed[at..at + std::mem::size_of_val(&want[..])];
                assert_eq!(bytes, run_table_bytes(want), "align {align}");
                assert_eq!(
                    place.bind_range,
                    std::mem::size_of_val(&want[..]) as u64,
                    "align {align}: a range wider than the table would bind padding"
                );
            }
        }
    }

    /// One table is the writeback's ordinary case and it must not pay for the
    /// arena: its place is the buffer's own start, exactly as the per-table slot
    /// it replaces was.
    #[test]
    fn a_lone_table_sits_at_offset_zero() {
        let words = table(7, 1);
        let (packed, places) = pack_run_tables(&[&words[..]], 256);
        assert_eq!(places.len(), 1);
        assert_eq!(places[0].bind_offset, 0);
        assert_eq!(places[0].bind_range, 7 * 4 * 4);
        assert_eq!(packed.len(), 256, "rounded up to the alignment stride");
    }

    /// `acquire_staging` and `VkDescriptorBufferInfo` both refuse a zero size,
    /// and a caller with nothing to stage would otherwise reach both with one.
    #[test]
    fn no_tables_still_asks_for_a_buffer_a_descriptor_could_name() {
        let (packed, places) = pack_run_tables(&[], 256);
        assert!(places.is_empty());
        assert_eq!(packed.len(), 4);
    }
}
