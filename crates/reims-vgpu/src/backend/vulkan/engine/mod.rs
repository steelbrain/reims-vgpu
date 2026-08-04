//! Persistent Vulkan draw + compute engine for the Linux metal2vulkan product path.
//!
//! Facade: [`execute_draw_request`] / [`execute_compute_request`] /
//! [`read_target`]. Caches L2–L7 + Lc + memory pools so a warm
//! identical static key performs zero `vkCreate*` and zero `vkAllocateMemory` on
//! the product path.

#![allow(unsafe_op_in_unsafe_fn)]

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
mod draw_validation;
mod exec;
mod exec_compute;
mod facade_decline;
mod host_slab;
pub mod init_decline;
mod pools;
pub mod reason;
mod slab;
pub mod stage_phase;
pub mod types;
pub mod vk_call;
#[cfg(feature = "host-window")]
mod window_present;

pub use compute_execution::ComputeExecutionDecline;
pub use compute_validation::ComputeValidationDecline;
pub use context::{FENCE_TIMEOUT_NS, MAX_DEVICE_RECREATES};
pub use counters::{CounterSnapshot, EngineCounters};
pub use device_lost::{DeviceLostDecline, DeviceLostOp};
pub use draw_execution::DrawExecutionDecline;
pub use draw_phase::{take_window as draw_phase_window, DrawPhaseWindow};
pub use draw_preparation::DrawPreparationDecline;
pub use draw_validation::DrawValidationDecline;
pub use facade_decline::EngineFacadeDecline;
pub use init_decline::InitDecline;
pub use reason::DrawReason;
pub use types::{
    BlendFactor, BlendOp, BlendStateResource, BufferContent, ColorWriteMask, ComputeBufferOutput,
    ComputeBufferResource, ComputeOutput, ComputeRequest, ComputeResidentSampleBind,
    ComputeSampledImageResource, ComputeStorageImageResource, ComputeStorageResidency, CullMode,
    DepthState, DrawError, DrawOutput, DrawRequest, GuestRun, GuestRunSource, IndexType,
    IndexedDrawResource, PrimitiveTopology, SampledContentIdentity, SampledImageResource,
    SampledSource, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
    SamplerMipFilter, SamplerResource, ScissorResource, SecondaryColorTarget, SeedOrder,
    StencilFaceOps, StencilOp, StencilState, StorageBufferResource, StorageImageFormat,
    TargetIdentity, VertexAttributeFormat, VertexAttributeResource, VertexStepFunction,
    ViewportResource, WindowPresentSource, COLOR_INPUT_BINDING,
};
pub use vk_call::{VkCall, VkOp};
#[cfg(feature = "host-window")]
pub use window_present::{WindowCpuFrame, WindowPresentOutcome};

use caches::ObjectCaches;
use context::ContextOwner;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pools::ResourcePools;
use std::sync::atomic::Ordering;
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

    fn flush_device_derived(&mut self) {
        if let Some(ctx) = self.owner.ctx.as_ref() {
            unsafe {
                #[cfg(feature = "host-window")]
                if let Some(mut presenter) = self.window_presenter.take() {
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

/// Presents skipped because the drain worker held `ENGINE`.
///
/// A running total rather than a per-event log line: contention is expected
/// under load and logging each one would flood the sink. The count is what makes
/// a wedge distinguishable from ordinary busyness, so it is reported with the
/// cadence figures.
#[cfg(feature = "host-window")]
static WINDOW_PRESENT_LOCK_BUSY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// How many presents have been skipped for engine-lock contention.
#[cfg(feature = "host-window")]
pub fn window_present_lock_busy_count() -> u64 {
    WINDOW_PRESENT_LOCK_BUSY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Latest window size the pump published, packed `width << 32 | height`.
/// Zero means nothing is pending. Applied by the next present.
#[cfg(feature = "host-window")]
static WINDOW_PRESENT_PENDING_EXTENT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Which thread class is asking for the engine lock.
///
/// The single `ENGINE` mutex serializes the drain worker's guest execution
/// against the host window's present, and only one direction of that
/// contention reaches the screen: a worker delayed by the window loses
/// throughput it can make up, while a window delayed by the worker drops the
/// frame it was about to show. `engine_lock` cannot say which side paid without
/// the two being named apart, so every acquire declares itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineLockSite {
    /// The drain worker executing guest commands, and every entry point QEMU
    /// reaches that is not the window's own event loop.
    Worker,
    /// The host window's event loop: present, attach, resize, detach.
    Window,
}

impl EngineLockSite {
    fn index(self) -> usize {
        match self {
            Self::Worker => 0,
            Self::Window => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Worker => "worker",
            Self::Window => "window",
        }
    }

    const ALL: [Self; 2] = [Self::Worker, Self::Window];
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
    uncontended: [std::sync::atomic::AtomicU64; 2],
    /// Acquires that found it held and had to block, per site.
    contended: [std::sync::atomic::AtomicU64; 2],
    /// Wall clock blocked on the mutex, summed over `contended`.
    wait_us: [std::sync::atomic::AtomicU64; 2],
    wait_max_us: [std::sync::atomic::AtomicU64; 2],
    /// Wall clock from acquire to release, over every acquire.
    hold_us: [std::sync::atomic::AtomicU64; 2],
    hold_max_us: [std::sync::atomic::AtomicU64; 2],
}

static ENGINE_LOCK: EngineLockCensus = EngineLockCensus::new();

impl EngineLockCensus {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    const fn new() -> Self {
        Self {
            uncontended: [Self::ZERO; 2],
            contended: [Self::ZERO; 2],
            wait_us: [Self::ZERO; 2],
            wait_max_us: [Self::ZERO; 2],
            hold_us: [Self::ZERO; 2],
            hold_max_us: [Self::ZERO; 2],
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
#[inline]
fn lock_engine() -> EngineGuard {
    lock_engine_at(EngineLockSite::Worker)
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
        if let Err(error) = unsafe { ctx.device.device_wait_idle() } {
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

/// Publish the window's rail choice. Called by the window thread from exactly
/// the two places that create and destroy the presenter.
#[cfg(feature = "host-window")]
pub fn note_window_present_attached(attached: bool) {
    WINDOW_PRESENT_ATTACHED.store(attached, Ordering::Release);
}

#[cfg(feature = "host-window")]
pub fn window_present_attached() -> bool {
    WINDOW_PRESENT_ATTACHED.load(Ordering::Acquire)
}

#[cfg(feature = "host-window")]
pub fn window_present_resize(width: u32, height: u32) {
    let packed = (u64::from(width.max(1)) << 32) | u64::from(height.max(1));
    WINDOW_PRESENT_PENDING_EXTENT.store(packed, std::sync::atomic::Ordering::Release);
}

/// Apply any deferred window size. Caller holds `ENGINE`.
#[cfg(feature = "host-window")]
fn apply_pending_window_extent(presenter: &mut window_present::WindowPresenter) {
    let packed = WINDOW_PRESENT_PENDING_EXTENT.swap(0, std::sync::atomic::Ordering::AcqRel);
    if packed == 0 {
        return;
    }
    presenter.resize((packed >> 32) as u32, packed as u32);
}

/// Present the current compositor resident through the engine-owned swapchain,
/// falling back to `cpu` for presents no resident carries. Acquire is
/// nonblocking, so a vblank wait never holds `ENGINE`.
#[cfg(feature = "host-window")]
pub fn window_present_frame(
    source: Option<&WindowPresentSource>,
    cpu: Option<WindowCpuFrame<'_>>,
) -> Result<WindowPresentOutcome, DrawError> {
    let Some(guard) = ENGINE.try_lock() else {
        WINDOW_PRESENT_LOCK_BUSY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(WindowPresentOutcome::Busy);
    };
    ENGINE_LOCK.note_uncontended(EngineLockSite::Window);
    let mut guard = EngineGuard {
        guard,
        site: EngineLockSite::Window,
        acquired: std::time::Instant::now(),
    };
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
    apply_pending_window_extent(presenter);
    unsafe { presenter.present(ctx, pools, counters, source, cpu) }
}

/// Destroy the engine-owned surface while the native AppKit window still
/// exists. Called from winit's `exiting` callback.
#[cfg(feature = "host-window")]
pub fn window_present_detach() {
    let mut guard = lock_engine_at(EngineLockSite::Window);
    let Some(mut presenter) = guard.window_presenter.take() else {
        return;
    };
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
    match result {
        Ok(out) => Ok(out),
        Err(DrawError::DeviceLost(decline)) => {
            guard.counters.device_lost.fetch_add(1, Ordering::Relaxed);
            guard.owner.mark_device_lost();
            guard.flush_device_derived();
            if let Err(error) = {
                let EngineState {
                    ref mut owner,
                    ref counters,
                    ..
                } = &mut *guard;
                owner.ensure(counters)
            } {
                crate::observe::Emit::decline("vk_device_recreate", &error).fail_once(1);
            }
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
    if let Err(e) = unsafe { pools.batch_flush(ctx, counters) } {
        // A lost device surfaces again on the next draw, which runs the full
        // recreate path; here just make the flush failure visible.
        crate::observe::Emit::decline("vk_batch_flush", &e).fail_once(0);
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
        Ok(out) => Ok(out),
        Err(DrawError::DeviceLost(decline)) => {
            guard.counters.device_lost.fetch_add(1, Ordering::Relaxed);
            guard.owner.mark_device_lost();
            guard.flush_device_derived();
            if let Err(error) = {
                let EngineState {
                    ref mut owner,
                    ref counters,
                    ..
                } = &mut *guard;
                owner.ensure(counters)
            } {
                crate::observe::Emit::decline("vk_device_recreate", &error).fail_once(2);
            }
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

pub fn resident_content_ready(identity: &TargetIdentity) -> bool {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .is_some_and(|s| s.content_ready)
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
/// - [`ResidentContent::Unstamped`] is expected traffic. `registry_mark_ready`
///   clears the stamp on every draw into a slot, so a later pass over the same
///   surface says the resident no longer holds the frame this window promised.
///   Declining is correct; the newer pass owns the surface now.
/// - [`ResidentContent::Absent`] is not. Nothing may evict a pinned slot —
///   `evict_registry_to_cap` and the idle drain both skip pinned slots by design
///   — so an identity that has gone missing between the arm and the fence means
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

/// Whether this backend may leave guest-visible content only in GPU-resident
/// engine state.
///
/// Held back by the `guest_pages_stay_authoritative` driver quirk, because a
/// device recreate drops that registry before guest pages are updated. See
/// [`crate::backend::vulkan::caps::DriverQuirk`] for what the quirk covers and
/// how to retire it.
pub fn deferred_gpu_only_content_allowed() -> bool {
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .is_some_and(|ctx| !ctx.caps.quirks.guest_pages_stay_authoritative)
}

/// The largest render-target edge this host can create, from the device's own
/// `maxImageDimension2D`. Before a device is resolved this is the Vulkan 1.2
/// required minimum — the most any implementation is guaranteed to accept.
pub fn max_render_target_dimension() -> u32 {
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .map(|ctx| ctx.features.max_image_dimension_2d)
        .unwrap_or(crate::backend::vulkan::caps::device_features::VULKAN_MIN_IMAGE_DIMENSION_2D)
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

/// Refresh a resident target's idle-drain timestamp without doing GPU work.
/// The present publish uses this so the displayed resident is not
/// reclaimed underneath the window on a present that does no draw.
pub fn touch_resident_target(identity: Option<&TargetIdentity>, now_ms: u64) {
    let Some(identity) = identity else {
        return;
    };
    let mut guard = lock_engine();
    guard.pools.registry_touch_at(identity, now_ms);
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
    SampledR32fLinearFilter,
}

impl EngineProbe {
    fn name(self) -> &'static str {
        match self {
            Self::StorageWriteWithoutFormat => "storage_write_without_format",
            Self::SampledR32fLinearFilter => "sampled_r32f_linear_filter",
        }
    }

    /// 1 through 6 are retired (see the type's docs); the rest keep the numbers
    /// they were first logged under.
    fn discriminant(self) -> u64 {
        match self {
            Self::StorageWriteWithoutFormat => 7,
            Self::SampledR32fLinearFilter => 9,
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
    let guard = lock_engine();
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
    let guard = lock_engine();
    guard.pools.compute_resident_sample_source(identity)
}

/// Drop the deferred-writeback pin of a resident whose guest window can no
/// longer be flushed (ReplacePhysical / unmap drop paths). The resident stays
/// registered — only LRU protection ends. No-op for an absent identity.
pub fn unpin_resident_storage(identity: &crate::model::ComputeStorageResidencyKey) {
    let mut guard = lock_engine();
    guard.pools.pin_resident_storage(identity, false);
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

/// Whether the bound device can sample an `R32_SFLOAT` image with **linear**
/// filtering. Gates the native single-channel float32 sampled rail (color
/// LUTs): `R16_SFLOAT` linear filtering is spec-mandatory and needs no gate,
/// but `R32_SFLOAT`'s is optional and absent on Apple/MoltenVK. Returns `false`
/// (declining the rail, leaving the sample fail-visible) if the engine cannot
/// initialize.
pub fn supports_sampled_r32f_linear_filter() -> bool {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.sampled_r32f_linear_filter,
        Err(error) => {
            engine_probe_decline(EngineProbe::SampledR32fLinearFilter, &error)
                .fail_once(EngineProbe::SampledR32fLinearFilter.discriminant());
            false
        }
    }
}

/// Read a content-ready **BGRA** resident target as tight BGRA8 for the present
/// capture (the proxy-oracle frame source).
///
/// This is the resident-direct capture source: it performs only the GPU→host
/// readback, with **no** guest-page scatter. `capture_present_frame`'s other
/// source (`flush_intersecting` → `present_into_host_runs`) reads the same
/// resident but additionally scatters it into the fragmented guest pages — work
/// the oracle does not need and which the deferred-writeback rail already
/// performs on a genuine guest read. Errors (rather than swapping channels) on a
/// non-BGRA resident: the caller's frame buffer is BGRA8, and an RGBA resident
/// would hand the proxies channel-swapped pixels.
///
/// Returns `None` for every *expected* absence — unknown identity, no ready
/// content, non-BGRA resident, or a short/oversized readback — so the caller can
/// fall back silently. These are speculative conditions on a normal boot (a cold
/// mid has no resident yet), not failures worth a fail-log line.
pub fn read_resident_bgra(identity: &TargetIdentity, need: usize) -> Option<Vec<u8>> {
    {
        let guard = lock_engine();
        let slot = guard.pools.registry_get(identity)?;
        if !slot.content_ready || !slot.bgra {
            return None;
        }
    }
    let mut px = match read_target_inner(identity) {
        // The `slot.bgra` gate above already established the order, so the
        // reported one cannot disagree and the bytes pass through untouched.
        Ok(rb) => rb.pixels,
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
/// every prior-submitted draw. `begin_entry_sync` would block this guest-drain
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
struct ReadbackLeaseGuard(Option<(u64, usize)>);

impl ReadbackLeaseGuard {
    fn new(lease: Option<(u64, usize)>) -> Self {
        Self(lease)
    }

    /// Take the lease out, so the caller owns the obligation from here on.
    fn disarm(&mut self) -> Option<(u64, usize)> {
        self.0.take()
    }
}

impl Drop for ReadbackLeaseGuard {
    fn drop(&mut self) {
        if let Some((token, _)) = self.0.take() {
            pools::return_readback_lease(token);
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
    let (cb, fence) = pools.begin_entry(ctx, counters)?;
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
    ctx.device
        .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.reset_cb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.begin_cb, e)))?;
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
    if let Some(probe) = ctx.timestamps.as_ref() {
        ctx.device
            .cmd_reset_query_pool(cb, probe.pool, 0, context::TimestampProbe::SLOTS);
        ctx.device
            .cmd_write_timestamp(cb, ash::vk::PipelineStageFlags::TOP_OF_PIPE, probe.pool, 0);
    }
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
    // When `old_layout` already is TRANSFER_SRC_OPTIMAL the transition half is
    // a no-op (`oldLayout == newLayout` is legal) and the barrier is doing only
    // its other job, which is the job that was missing.
    let barrier = [ash::vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
        .old_layout(old_layout)
        .new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image)
        .subresource_range(color_subresource_range())];
    ctx.device.cmd_pipeline_barrier(
        cb,
        ash::vk::PipelineStageFlags::ALL_COMMANDS,
        ash::vk::PipelineStageFlags::TRANSFER,
        ash::vk::DependencyFlags::empty(),
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
    if let Some(probe) = ctx.timestamps.as_ref() {
        ctx.device
            .cmd_write_timestamp(cb, ash::vk::PipelineStageFlags::TRANSFER, probe.pool, 1);
    }
    let region = [ash::vk::BufferImageCopy::default()
        .image_subresource(color_subresource_layers())
        .image_extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })];
    // The whole level, every time: nothing upstream tells this call which part
    // of the frame the draws touched. That is what the slots 1-2 span prices —
    // on a discrete part this copy crosses the bus and is 87-91% of the fence,
    // so `gpu_us` owning `fence_us` in `readback_split` is bytes and not
    // latency, and bounding the region is the lever that would move it.
    ctx.device.cmd_copy_image_to_buffer(
        cb,
        image,
        ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        readback.buffer,
        &region,
    );
    if let Some(probe) = ctx.timestamps.as_ref() {
        ctx.device.cmd_write_timestamp(
            cb,
            ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            probe.pool,
            2,
        );
    }
    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.end_cb, e)))?;
    let queue = ctx.queue();
    let cbs = [cb];
    let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
    ctx.device
        .queue_submit(queue, &[si], fence)
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.submit, e)))?;
    let cleanup = pools.seal_entry(Vec::new(), Vec::new());
    pools.finish_entry_async(cleanup);
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
    // Read against the fence that just signalled, so both queries are available
    // and this cannot block. It divides `fence_us`: what is left after the GPU's
    // own execution of this command buffer is the draw batch ahead of it plus
    // the cost of asking, and only the first of those is work this device could
    // make cheaper.
    if let Some(probe) = ctx.timestamps.as_ref() {
        let mut ticks = [0u64; context::TimestampProbe::SLOTS as usize];
        match ctx.device.get_query_pool_results(
            probe.pool,
            0,
            &mut ticks,
            ash::vk::QueryResultFlags::TYPE_64,
        ) {
            // In f64, not integer ticks-times-period: `timestampPeriod` is a
            // float and drivers do report values below 1 ns (a counter faster
            // than 1 GHz), which an integer multiply would truncate to zero and
            // report as "the GPU did nothing".
            Ok(()) => {
                let us = |from: usize, to: usize| {
                    (ticks[to].saturating_sub(ticks[from]) as f64
                        * probe.ns_per_tick.max(0.0) as f64
                        / 1_000.0) as u64
                };
                crate::runtime::drain::note_readback_gpu_us(us(0, 1), us(1, 2));
            }
            Err(e) => crate::observe::Emit::decline(
                "vk_timestamp_read",
                &VkCall::new(VkOp::ContextGetQueryPoolResults, e),
            )
            .fail_once(0),
        }
    }
    let map_started = std::time::Instant::now();
    let out = match lease.disarm() {
        // The mapping is already established for the slot's lifetime, so all
        // this owes is the invalidate a non-coherent readback owes any reader.
        // What `ReadbackPhase::Map` measures on this arm is therefore that call
        // alone: the whole-frame memcpy the other arm pays is what the lease
        // exists to delete, and the phase reading near zero is how you tell it
        // happened.
        Some((token, ptr)) => {
            match pools::invalidate_slot_for_read(ctx, &readback, ops.invalidate) {
                Ok(()) => Ok(ReadbackResult::Leased {
                    token,
                    ptr,
                    len: rb_size as usize,
                }),
                Err(e) => {
                    pools::return_readback_lease(token);
                    Err(e)
                }
            }
        }
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
    let snap = resident_read_snapshot(pools, identity)?;
    let rb_size = (snap.width as u64) * (snap.height as u64) * 4;
    unsafe {
        let delivered = copy_image_level0_to_host_delivered(
            ctx,
            pools,
            counters,
            snap.image,
            snap.layout,
            RESIDENT_READ_SRC_ACCESS,
            snap.width,
            snap.height,
            rb_size,
            target_readback_ops(),
            ReadbackDelivery::Lease,
        )?;
        pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        counters.note_target_read(rb_size);
        Ok(match delivered {
            ReadbackResult::Leased { token, ptr, len } => Some(LeasedFrame {
                token,
                ptr,
                len,
                bgra: snap.bgra,
            }),
            // The slot had no mapping to lend, so the readback fell back to a
            // copy. Drop it and let the caller take its own copying path rather
            // than hand back a `Vec` this signature has no room for; that costs
            // one extra readback on a path that should never run.
            ReadbackResult::Copied(_) => None,
        })
    }
}

/// The `srcAccessMask` a resident color target's readback must drain.
const RESIDENT_READ_SRC_ACCESS: ash::vk::AccessFlags = ash::vk::AccessFlags::from_raw(
    ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw()
        | ash::vk::AccessFlags::TRANSFER_WRITE.as_raw()
        | ash::vk::AccessFlags::SHADER_WRITE.as_raw(),
);

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

/// What a resident target's readback copies, and the channel order it is in.
struct ResidentReadSnapshot {
    image: ash::vk::Image,
    width: u32,
    height: u32,
    layout: ash::vk::ImageLayout,
    bgra: bool,
}

/// The registry slot's copy geometry, or the typed reason it cannot be read.
///
/// Shared by both readback entry points so that "is this target readable" is
/// decided once and answered with one vocabulary.
fn resident_read_snapshot(
    pools: &pools::ResourcePools,
    identity: &TargetIdentity,
) -> Result<ResidentReadSnapshot, DrawError> {
    let slot = pools.registry_get(identity).ok_or(DrawError::TargetRead(
        reason::TargetReadDecline::UnknownIdentity,
    ))?;
    if !slot.content_ready {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::NoReadyContent,
        ));
    }
    Ok(ResidentReadSnapshot {
        image: slot.image,
        width: slot.width,
        height: slot.height,
        layout: slot.layout,
        bgra: slot.bgra,
    })
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
    let snap = resident_read_snapshot(pools, identity)?;
    let rb_size = (snap.width as u64) * (snap.height as u64) * 4;
    unsafe {
        let out = copy_image_level0_to_host(
            ctx,
            pools,
            counters,
            snap.image,
            snap.layout,
            RESIDENT_READ_SRC_ACCESS,
            snap.width,
            snap.height,
            rb_size,
            target_readback_ops(),
        )?;
        pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        counters.note_target_read(rb_size);
        Ok(TargetReadback {
            pixels: out,
            bgra: snap.bgra,
        })
    }
}

/// Full-frame readback of a resident target (present / Synchronize / Map / Store boundary).
pub fn read_target(identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
    read_target_inner(identity)
}

/// Flush read of a **pinned deferred-writeback resident storage image**: copy
/// the GPU content to the host as tight `width*height*texel` bytes and unpin.
///
/// The caller (runtime deferred-flush) writes these bytes into the guest
/// window and re-establishes its residency mirror. `expected_generation`
/// guards against flushing content from a different chain step than the one
/// the caller deferred — a mismatch (or an absent/evicted resident) is the
/// named error the caller reports as `deferred_flush_lost`. Returns
/// `(bytes, texel_size)`.
pub fn read_resident_storage(
    identity: &crate::model::ComputeStorageResidencyKey,
    expected_generation: u32,
) -> Result<(Vec<u8>, u32), DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let (image, key, generation, old_layout) =
        pools.compute_resident_snapshot(identity).ok_or({
            DrawError::Facade(EngineFacadeDecline::StorageReadResidentAbsent {
                identity: *identity,
            })
        })?;
    if generation != expected_generation {
        return Err(DrawError::Facade(
            EngineFacadeDecline::StorageReadGenerationMismatch {
                identity: *identity,
                actual_generation: generation,
                expected_generation,
            },
        ));
    }
    let texel = key.format.bytes_per_texel() as u32;
    let rb_size = (key.width as u64) * (key.height as u64) * texel as u64;
    unsafe {
        let out = copy_image_level0_to_host(
            ctx,
            pools,
            counters,
            image,
            old_layout,
            // A storage image is never a color attachment, so there is no
            // `COLOR_ATTACHMENT_WRITE` to drain here.
            ash::vk::AccessFlags::TRANSFER_WRITE | ash::vk::AccessFlags::SHADER_WRITE,
            key.width,
            key.height,
            rb_size,
            ReadbackOps {
                reset_cb: VkOp::StorageReadResetCb,
                begin_cb: VkOp::StorageReadBeginCb,
                end_cb: VkOp::StorageReadEndCb,
                submit: VkOp::StorageReadSubmit,
                map: VkOp::StorageReadMap,
                invalidate: VkOp::StorageReadInvalidate,
            },
        )?;
        pools.set_resident_storage_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        pools.pin_resident_storage(identity, false);
        counters.note_compute_deferred_flush(rb_size);
        Ok((out, texel))
    }
}

/// The non-pinned resident-target slot cap. Exposed so a test that must blow
/// past the LRU sweep derives its filler count from the live value instead of
/// hard-coding one — `vk_engine_parity` previously fixed 70 fillers against a
/// cap later retuned to 320, so no eviction fired and its assert could not hold.
pub fn registry_cap() -> usize {
    pools::REGISTRY_CAP
}

/// Advance the wall-clock resident-target idle-drain clock to `now_ms`, keep the
/// currently-presented target (`display`) alive, and reclaim aged non-pinned
/// residents. Called from the poll heartbeat (so the clock keeps ticking when the
/// guest stops publishing) and each present publish. No-op before the device
/// context exists.
pub fn maintain_idle_residents(display: Option<&TargetIdentity>, now_ms: u64) {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    unsafe {
        pools.advance_registry_touch_and_drain(ctx, now_ms, display);
    }
}

/// Snapshot of create/alloc/hit-miss counters (for tests and thrash proxies).
pub fn counter_snapshot() -> CounterSnapshot {
    let eng = lock_engine();
    let mut snap = eng.counters.snapshot();
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
    snap
}

/// Reset create/alloc/hit-miss counters (not device_lost/recreates). For reuse-gate tests.
pub fn reset_draw_counters() {
    lock_engine().counters.reset();
}

/// Test-only: destroy device, clear recreate budget, rebuild on next draw.
pub fn test_reset_engine() {
    let mut g = lock_engine();
    if let Some(mut ctx) = g.owner.ctx.take() {
        unsafe {
            g.caches.destroy_all(&ctx.device);
            g.pools.destroy_all(&ctx.device);
            ctx.destroy();
        }
    }
    g.caches = ObjectCaches::new();
    g.pools = ResourcePools::new();
    g.owner = ContextOwner::new();
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
mod probe_visibility_tests {
    use super::*;

    #[test]
    fn each_engine_probe_preserves_the_typed_initialization_reason() {
        let error = vk_call::exec_submit_device_lost_fixture();
        for probe in [
            EngineProbe::StorageWriteWithoutFormat,
            EngineProbe::SampledR32fLinearFilter,
        ] {
            let line = engine_probe_decline(probe, &error).render();
            assert!(line.starts_with("vk_engine_probe reason=vk_exec_submit "));
            assert!(line.ends_with(&format!(" probe={}", probe.name())));
        }
    }
}

#[cfg(all(test, feature = "host-window"))]
mod window_pump_tests {
    /// A deferred window size survives engine contention and coalesces.
    ///
    /// `window_present_resize` runs on the thread that owns the host window's
    /// message pump, so it must not wait on `ENGINE`. Publishing to an atomic
    /// keeps it lock-free, and this pins the two properties that make deferring
    /// safe rather than lossy: the value is not lost while the engine is busy,
    /// and only the newest one is applied.
    #[test]
    fn a_deferred_window_extent_is_kept_and_coalesced() {
        use std::sync::atomic::Ordering;

        super::WINDOW_PRESENT_PENDING_EXTENT.store(0, Ordering::Release);

        assert_eq!(
            super::WINDOW_PRESENT_PENDING_EXTENT.swap(0, Ordering::AcqRel),
            0
        );

        super::window_present_resize(1280, 720);
        super::window_present_resize(1600, 900);
        super::window_present_resize(1920, 1080);
        let packed = super::WINDOW_PRESENT_PENDING_EXTENT.swap(0, Ordering::AcqRel);
        assert_eq!(((packed >> 32) as u32, packed as u32), (1920, 1080));

        assert_eq!(
            super::WINDOW_PRESENT_PENDING_EXTENT.swap(0, Ordering::AcqRel),
            0
        );

        super::window_present_resize(0, 0);
        let packed = super::WINDOW_PRESENT_PENDING_EXTENT.swap(0, Ordering::AcqRel);
        assert_ne!(packed, 0);
        assert_eq!(((packed >> 32) as u32, packed as u32), (1, 1));
    }
}
